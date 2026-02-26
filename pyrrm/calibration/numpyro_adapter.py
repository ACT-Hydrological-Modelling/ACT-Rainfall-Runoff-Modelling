"""
NumPyro NUTS adapter for pyrrm model calibration.

Provides a Bayesian MCMC interface via the No-U-Turn Sampler (NUTS)
from NumPyro.  Follows the same adapter pattern as pydream_adapter.py.

Key features:
- Automatic prior generation from model parameter bounds
- Multiple likelihood formulations (Gaussian, transformed, AR(1))
- ArviZ InferenceData integration for diagnostics
- Gradient-based sampling via JAX automatic differentiation

References:
    Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler.
    JMLR, 15, 1593-1623.

    Phan, D., Pradhan, N. & Jankowiak, M. (2019). Composable Effects
    for Flexible and Accelerated Probabilistic Programming in NumPyro.
"""

from typing import Dict, Tuple, Optional, Any, Callable
from functools import partial
import time
import warnings
import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    import arviz as az

    jax.config.update("jax_enable_x64", True)
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

_JAX_MODEL_REGISTRY: Dict[str, str] = {
    "gr4j": "pyrrm.models.gr4j_jax",
    "sacramento": "pyrrm.models.sacramento_jax",
}


def _resolve_jax_model_fn(model_name: str) -> Callable:
    """
    Lazily import the JAX forward model matching a pyrrm model name.

    Args:
        model_name: Lowercase model name (e.g. ``"gr4j"``).

    Returns:
        JAX model run function.
    """
    key = model_name.lower()
    if key not in _JAX_MODEL_REGISTRY:
        raise ValueError(
            f"No JAX forward model registered for '{model_name}'. "
            f"Available: {list(_JAX_MODEL_REGISTRY.keys())}"
        )
    module_path = _JAX_MODEL_REGISTRY[key]
    import importlib
    mod = importlib.import_module(module_path)
    if key == "gr4j":
        return mod.gr4j_run_jax
    elif key == "sacramento":
        return mod.sacramento_run_jax
    raise RuntimeError(f"Cannot resolve function for model '{model_name}'")


def _extract_precip_pet(inputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Pull precipitation and PET arrays from a pyrrm-style DataFrame."""
    if "precipitation" in inputs.columns:
        precip = inputs["precipitation"].values.astype(np.float64)
    elif "rainfall" in inputs.columns:
        precip = inputs["rainfall"].values.astype(np.float64)
    else:
        raise ValueError("Input must contain 'precipitation' or 'rainfall' column")

    if "evapotranspiration" in inputs.columns:
        pet = inputs["evapotranspiration"].values.astype(np.float64)
    elif "pet" in inputs.columns:
        pet = inputs["pet"].values.astype(np.float64)
    else:
        raise ValueError("Input must contain 'evapotranspiration' or 'pet' column")

    return precip, pet


def _build_numpyro_model(
    jax_model_fn: Callable,
    param_bounds: Dict[str, Tuple[float, float]],
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    obs_flow: jnp.ndarray,
    warmup_steps: int,
    likelihood_type: str,
    transform: str,
    transform_params: Optional[dict],
    error_model: str,
    prior_config: Optional[dict],
    sigma_prior_scale: float,
    reparameterize: bool = False,
    max_ninc: Optional[int] = None,
    fast_mode: bool = False,
    tvp_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Return a NumPyro model function suitable for ``MCMC``.

    When *reparameterize* is True, Uniform-prior parameters are sampled
    in [0, 1] (``{name}_unit``) and deterministically transformed to
    physical bounds.  This equalises scales across parameters and
    dramatically improves NUTS adaptation for high-dimensional models
    like Sacramento.

    When *max_ninc* is set, it is forwarded to the JAX model via
    ``functools.partial`` to cap the inner sub-daily loop length.

    When *fast_mode* is True, it is forwarded to the JAX model to
    bypass the inner sub-daily ``lax.scan`` entirely (ninc=1).

    When *tvp_config* is provided, the named parameters are sampled as
    time-varying trajectories via their ``TVPPrior.sample_numpyro``
    method instead of as scalars.  The resulting ``params`` dict will
    contain a mix of scalars and ``(T,)`` arrays -- which is exactly
    what the updated ``gr4j_run_jax`` expects.
    """
    from pyrrm.calibration.likelihoods_jax import (
        gaussian_log_likelihood_jax,
        transformed_gaussian_log_likelihood_jax,
        ar1_log_likelihood_jax,
    )

    prior_config = prior_config or {}
    tvp_config = tvp_config or {}
    n_timesteps = int(precip.shape[0])

    extra_kwargs = {}
    if max_ninc is not None:
        extra_kwargs["max_ninc"] = max_ninc
    if fast_mode:
        extra_kwargs["fast_mode"] = True

    if extra_kwargs:
        model_fn = partial(jax_model_fn, **extra_kwargs)
    else:
        model_fn = jax_model_fn

    def numpyro_model():
        params = {}
        for name, (lo, hi) in param_bounds.items():
            if name in tvp_config:
                params[name] = tvp_config[name].sample_numpyro(name, n_timesteps)
            elif name in prior_config:
                params[name] = numpyro.sample(name, prior_config[name])
            elif reparameterize:
                unit = numpyro.sample(f"{name}_unit", dist.Uniform(0.0, 1.0))
                params[name] = numpyro.deterministic(name, lo + (hi - lo) * unit)
            else:
                params[name] = numpyro.sample(name, dist.Uniform(lo, hi))

        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior_scale))

        sim = model_fn(params, precip, pet)["simulated_flow"]

        if error_model == "ar1":
            phi = numpyro.sample("phi", dist.Uniform(-1.0, 1.0))
            ll = ar1_log_likelihood_jax(
                sim, obs_flow, sigma, phi,
                transform=transform,
                transform_params=transform_params,
                warmup_steps=warmup_steps,
            )
        elif likelihood_type == "transformed_gaussian" or transform != "none":
            ll = transformed_gaussian_log_likelihood_jax(
                sim, obs_flow, sigma,
                transform=transform,
                transform_params=transform_params,
                warmup_steps=warmup_steps,
            )
        else:
            ll = gaussian_log_likelihood_jax(
                sim, obs_flow, sigma,
                warmup_steps=warmup_steps,
            )

        numpyro.factor("log_likelihood", ll)

    return numpyro_model


def run_nuts(
    model,
    inputs: pd.DataFrame,
    observed: np.ndarray,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 8,
    likelihood_type: str = "gaussian",
    transform: str = "none",
    transform_params: Optional[dict] = None,
    error_model: str = "iid",
    prior_config: Optional[dict] = None,
    sigma_prior_scale: float = 1.0,
    reparameterize: bool = False,
    use_float64: bool = True,
    max_ninc: Optional[int] = None,
    fast_mode: bool = False,
    tvp_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    progress_bar: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run NumPyro NUTS calibration for a pyrrm model.

    The interface mirrors ``run_pydream()`` so that ``CalibrationRunner``
    can wrap the result in a ``CalibrationResult`` the same way.

    Args:
        model: A pyrrm model instance (must have a ``.name`` attribute
            and ``get_parameter_bounds()``).
        inputs: DataFrame with precipitation and PET columns.
        observed: Observed flow array.
        parameter_bounds: Override for parameter bounds.
        warmup_period: Leading timesteps excluded from the likelihood.
        num_warmup: NUTS warmup (adaptation) iterations.
        num_samples: Post-warmup samples per chain.
        num_chains: Number of independent chains.
        target_accept_prob: Target acceptance probability for NUTS.
        max_tree_depth: Maximum tree depth for NUTS (default 8; max
            leapfrog steps = 2^depth).
        likelihood_type: ``"gaussian"`` or ``"transformed_gaussian"``.
        transform: Flow transformation (``"none"``, ``"sqrt"``, ``"log"``, etc.).
        transform_params: Optional transform parameter overrides.
        error_model: ``"iid"`` or ``"ar1"``.
        prior_config: Dict mapping parameter name to a ``numpyro.distributions``
            object, overriding the default Uniform prior for that parameter.
        sigma_prior_scale: Scale for the ``HalfNormal`` prior on sigma.
        reparameterize: If True, sample Uniform-prior parameters in
            [0, 1] and deterministically transform to physical bounds.
            Recommended for models with many parameters spanning
            different scales (e.g. Sacramento).
        use_float64: If True (default), run in float64 precision.
            Set to False for ~2x speedup on CPU at the cost of
            reduced numerical precision (sufficient for calibration).
        max_ninc: Override the Sacramento inner loop count (default
            ``None`` keeps the model default of 20).  Set to 5 for
            faster calibration; ``ninc`` rarely exceeds 3.
            Ignored when ``fast_mode=True``.
        fast_mode: If True, bypass the Sacramento inner sub-daily
            ``lax.scan`` entirely (ninc=1), using daily drainage
            rates directly.  Eliminates the nested scan, reducing
            the XLA graph by ~10-16× and dramatically speeding up
            JIT compilation and gradient evaluation.  Recommended
            for calibration; validate posterior with ``fast_mode=False``.
        tvp_config: Optional dict mapping parameter names to
            ``TVPPrior`` instances (e.g. ``GaussianRandomWalk``).
            Named parameters are sampled as time-varying trajectories
            instead of scalars.  See :mod:`pyrrm.calibration.tvp_priors`.
        seed: PRNG seed for reproducibility.
        progress_bar: Show NumPyro progress bar.
        verbose: Print summary.

    Returns:
        Dict with keys: ``best_parameters``, ``best_objective``, ``all_samples``,
        ``convergence_diagnostics``, ``runtime_seconds``, ``parameter_names``,
        ``inference_data``, ``tvp_config``. The MCMC object is not stored so that
        the result remains picklable (e.g. for CalibrationReport.save()).
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError(
            "NumPyro is required for NUTS calibration. "
            "Install with: pip install jax jaxlib numpyro arviz"
        )

    start_time = time.time()

    # Resolve JAX forward model
    model_name = getattr(model, "name", type(model).__name__).lower()
    jax_model_fn = _resolve_jax_model_fn(model_name)

    # Parameter bounds
    if parameter_bounds is None:
        parameter_bounds = model.get_parameter_bounds()
    param_names = list(parameter_bounds.keys())

    # Input arrays
    dtype = np.float64 if use_float64 else np.float32
    precip_np, pet_np = _extract_precip_pet(inputs)
    precip_jax = jnp.array(precip_np.astype(dtype))
    pet_jax = jnp.array(pet_np.astype(dtype))
    obs_jax = jnp.array(np.asarray(observed).flatten().astype(dtype))

    tvp_config = tvp_config or {}

    if verbose:
        print(f"Running NumPyro NUTS")
        print(f"  Model:          {model_name}")
        print(f"  Parameters:     {len(param_names)}")
        if tvp_config:
            tvp_names = ", ".join(sorted(tvp_config.keys()))
            print(f"  TVP params:     {tvp_names}")
        print(f"  Warmup iters:   {num_warmup}")
        print(f"  Sample iters:   {num_samples}")
        print(f"  Chains:         {num_chains}")
        print(f"  Reparameterize: {reparameterize}")
        print(f"  Precision:      {'float64' if use_float64 else 'float32'}")
        print(f"  Max tree depth: {max_tree_depth}")
        if fast_mode:
            print(f"  Fast mode:      True (ninc=1, no inner scan)")
        elif max_ninc is not None:
            print(f"  Max ninc:       {max_ninc}")
        print(f"  Likelihood:     {likelihood_type} (transform={transform}, error={error_model})")

    # Build NumPyro model
    numpyro_model_fn = _build_numpyro_model(
        jax_model_fn=jax_model_fn,
        param_bounds=parameter_bounds,
        precip=precip_jax,
        pet=pet_jax,
        obs_flow=obs_jax,
        warmup_steps=warmup_period,
        likelihood_type=likelihood_type,
        transform=transform,
        transform_params=transform_params,
        error_model=error_model,
        prior_config=prior_config,
        sigma_prior_scale=sigma_prior_scale,
        reparameterize=reparameterize,
        max_ninc=max_ninc,
        fast_mode=fast_mode,
        tvp_config=tvp_config,
    )

    # NUTS kernel — init_to_median starts chains from the midpoint
    # of each parameter's prior, avoiding random corners in high
    # dimensions.
    kernel = NUTS(
        numpyro_model_fn,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_median(),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key)

    runtime = time.time() - start_time

    # --- Post-process results ---
    samples = mcmc.get_samples(group_by_chain=True)  # {name: (chains, samples)}

    # ArviZ InferenceData
    inference_data = az.from_numpyro(mcmc)

    # Posterior medians as best parameters
    flat_samples = mcmc.get_samples(group_by_chain=False)

    # Identify which parameters are static (scalar) vs TVP
    static_param_names = [n for n in param_names if n not in tvp_config]

    if reparameterize:
        physical_samples = {}
        for name in static_param_names:
            lo, hi = parameter_bounds[name]
            if name in (prior_config or {}):
                physical_samples[name] = np.array(flat_samples[name])
            else:
                unit_arr = np.array(flat_samples[f"{name}_unit"])
                physical_samples[name] = lo + (hi - lo) * unit_arr
        best_parameters = {
            name: float(np.median(physical_samples[name]))
            for name in static_param_names
        }
    else:
        physical_samples = {
            name: np.array(flat_samples[name]) for name in static_param_names
        }
        best_parameters = {
            name: float(jnp.median(flat_samples[name]))
            for name in static_param_names
        }

    # TVP hyperparameter medians
    tvp_hyperparams = {}
    for tvp_name, tvp_prior in tvp_config.items():
        hp_names = tvp_prior.hyperparameter_names(tvp_name)
        for hp in hp_names:
            if hp in flat_samples:
                arr = np.array(flat_samples[hp])
                if arr.ndim == 1:
                    tvp_hyperparams[hp] = float(np.median(arr))
                else:
                    tvp_hyperparams[hp] = np.median(arr, axis=0).tolist()

    best_parameters.update(tvp_hyperparams)

    # "Best objective" = log-likelihood at posterior median
    # For TVP params we reconstruct the median trajectory
    from pyrrm.calibration.likelihoods_jax import (
        gaussian_log_likelihood_jax,
        transformed_gaussian_log_likelihood_jax,
        ar1_log_likelihood_jax,
    )

    median_params_for_sim = {}
    for name in static_param_names:
        median_params_for_sim[name] = jnp.float64(
            best_parameters[name]
        )
    for tvp_name in tvp_config:
        if tvp_name in flat_samples:
            median_params_for_sim[tvp_name] = jnp.array(
                np.median(np.array(flat_samples[tvp_name]), axis=0)
            )
        else:
            median_params_for_sim[tvp_name] = best_parameters.get(tvp_name, 0.0)

    median_sigma = float(jnp.median(flat_samples["sigma"]))
    median_sim = jax_model_fn(
        median_params_for_sim,
        precip_jax, pet_jax,
    )["simulated_flow"]

    if error_model == "ar1":
        median_phi = float(jnp.median(flat_samples["phi"]))
        best_objective = float(ar1_log_likelihood_jax(
            median_sim, obs_jax, median_sigma, median_phi,
            transform=transform, transform_params=transform_params,
            warmup_steps=warmup_period,
        ))
    elif transform != "none":
        best_objective = float(transformed_gaussian_log_likelihood_jax(
            median_sim, obs_jax, median_sigma,
            transform=transform, transform_params=transform_params,
            warmup_steps=warmup_period,
        ))
    else:
        best_objective = float(gaussian_log_likelihood_jax(
            median_sim, obs_jax, median_sigma,
            warmup_steps=warmup_period,
        ))

    # Build all_samples DataFrame (stacked across chains, physical scale)
    # For static params, include physical samples; for TVP, include hyperparams.
    records = {}
    for name in static_param_names:
        records[name] = physical_samples[name]
    for tvp_name, tvp_prior in tvp_config.items():
        for hp in tvp_prior.hyperparameter_names(tvp_name):
            if hp in flat_samples:
                arr = np.array(flat_samples[hp])
                if arr.ndim == 1:
                    records[hp] = arr
    records["sigma"] = np.array(flat_samples["sigma"])
    if error_model == "ar1":
        records["phi"] = np.array(flat_samples["phi"])

    first_key = next(iter(records))
    n_total = len(records[first_key])
    records["chain"] = np.repeat(np.arange(num_chains), num_samples)[:n_total]
    records["iteration"] = np.tile(np.arange(num_samples), num_chains)[:n_total]
    all_samples_df = pd.DataFrame(records)

    # Convergence diagnostics via ArviZ
    # For static params: diagnose on _unit (if reparameterized) or physical.
    # For TVP params: diagnose on the scalar hyperparameters (intercept, sigma_delta).
    try:
        diag_names = []
        rename_map = {}

        if reparameterize:
            for n in static_param_names:
                if n not in (prior_config or {}):
                    diag_names.append(f"{n}_unit")
                    rename_map[f"{n}_unit"] = n
                else:
                    diag_names.append(n)
        else:
            diag_names.extend(static_param_names)

        for tvp_name, tvp_prior in tvp_config.items():
            for hp in tvp_prior.hyperparameter_names(tvp_name):
                if hp.endswith("_delta"):
                    continue
                if hp in flat_samples:
                    diag_names.append(hp)

        diag_names.append("sigma")

        summary = az.summary(inference_data, var_names=diag_names)
        if rename_map:
            summary = summary.rename(index=rename_map)

        convergence_diagnostics = {
            "rhat": {k: float(summary.loc[k, "r_hat"]) for k in summary.index},
            "ess_bulk": {k: float(summary.loc[k, "ess_bulk"]) for k in summary.index},
            "ess_tail": {k: float(summary.loc[k, "ess_tail"]) for k in summary.index},
            "max_rhat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "divergences": int(mcmc.get_extra_fields().get("diverging", jnp.array([])).sum()),
            "converged": bool(summary["r_hat"].max() < 1.01),
        }
    except Exception as exc:
        warnings.warn(f"Could not compute convergence diagnostics: {exc}")
        convergence_diagnostics = {"error": str(exc)}

    if verbose:
        print(f"\nNUTS completed in {runtime:.1f} seconds")
        if "max_rhat" in convergence_diagnostics:
            print(f"  Max R-hat:       {convergence_diagnostics['max_rhat']:.4f}")
            print(f"  Min ESS (bulk):  {convergence_diagnostics['min_ess_bulk']:.0f}")
            print(f"  Divergences:     {convergence_diagnostics['divergences']}")
        print("  Posterior median parameters:")
        for k, v in best_parameters.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: [array of length {len(v)}]")

    # Do not include "mcmc": it holds a reference to the NumPyro model (a local
    # function from _build_numpyro_model), which is not picklable and would
    # break CalibrationReport.save() and CalibrationResult serialization.
    return {
        "best_parameters": best_parameters,
        "best_objective": best_objective,
        "all_samples": all_samples_df,
        "convergence_diagnostics": convergence_diagnostics,
        "runtime_seconds": runtime,
        "parameter_names": param_names,
        "inference_data": inference_data,
        "tvp_config": tvp_config if tvp_config else None,
    }
