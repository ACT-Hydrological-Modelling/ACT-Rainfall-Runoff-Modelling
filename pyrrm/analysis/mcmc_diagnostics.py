"""
MCMC convergence diagnostics built on ArviZ.

Provides concise helper functions for assessing NumPyro NUTS
calibration quality.
"""

from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False


def _require_arviz():
    if not ARVIZ_AVAILABLE:
        raise ImportError("ArviZ is required. Install with: pip install arviz")


def check_convergence(
    inference_data,
    var_names: Optional[List[str]] = None,
    threshold_rhat: float = 1.01,
    threshold_ess: float = 400,
) -> Dict[str, Any]:
    """
    Check MCMC chain convergence and return a diagnostic summary.

    Args:
        inference_data: ArviZ ``InferenceData`` (from ``az.from_numpyro``).
        var_names: Parameter names to check.  If ``None``, all posterior
            variables are included.
        threshold_rhat: R-hat threshold -- all must be below this.
        threshold_ess: Minimum effective sample size (bulk).

    Returns:
        Dict with keys:
        - ``converged`` (bool): True if all criteria are met.
        - ``rhat`` (dict): Per-variable R-hat.
        - ``ess_bulk`` (dict): Per-variable bulk ESS.
        - ``ess_tail`` (dict): Per-variable tail ESS.
        - ``divergences`` (int): Total divergent transitions.
        - ``warnings`` (list[str]): Human-readable warning messages.
    """
    _require_arviz()

    summary = az.summary(inference_data, var_names=var_names)

    rhat = {k: float(summary.loc[k, "r_hat"]) for k in summary.index}
    ess_bulk = {k: float(summary.loc[k, "ess_bulk"]) for k in summary.index}
    ess_tail = {k: float(summary.loc[k, "ess_tail"]) for k in summary.index}

    try:
        posterior = inference_data.posterior
        if hasattr(inference_data, "sample_stats"):
            div_field = inference_data.sample_stats.get("diverging", None)
            divergences = int(div_field.values.sum()) if div_field is not None else 0
        else:
            divergences = 0
    except Exception:
        divergences = 0

    warn_msgs: List[str] = []
    rhat_ok = all(v < threshold_rhat for v in rhat.values())
    ess_ok = all(v >= threshold_ess for v in ess_bulk.values())

    if not rhat_ok:
        bad = {k: v for k, v in rhat.items() if v >= threshold_rhat}
        warn_msgs.append(f"R-hat above {threshold_rhat} for: {bad}")
    if not ess_ok:
        bad = {k: v for k, v in ess_bulk.items() if v < threshold_ess}
        warn_msgs.append(f"ESS below {threshold_ess} for: {bad}")
    if divergences > 0:
        warn_msgs.append(f"{divergences} divergent transitions detected")

    return {
        "converged": rhat_ok and ess_ok and divergences == 0,
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "divergences": divergences,
        "warnings": warn_msgs,
    }


def posterior_summary(
    inference_data,
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return an ArviZ summary table for selected variables.

    Args:
        inference_data: ArviZ ``InferenceData``.
        var_names: Variables to include (``None`` = all).

    Returns:
        DataFrame with mean, sd, hdi_3%, hdi_97%, r_hat, ess_bulk, ess_tail.
    """
    _require_arviz()
    return az.summary(inference_data, var_names=var_names)


def compute_nse_from_posterior(
    inference_data,
    jax_model_fn,
    precip,
    pet,
    obs_flow,
    warmup_steps: int = 365,
) -> float:
    """
    Compute the Nash-Sutcliffe Efficiency of the posterior-median simulation.

    Args:
        inference_data: ArviZ ``InferenceData``.
        jax_model_fn: JAX forward model function.
        precip: Precipitation array (JAX or NumPy).
        pet: PET array (JAX or NumPy).
        obs_flow: Observed flow array (NumPy).
        warmup_steps: Leading timesteps to discard.

    Returns:
        NSE value.
    """
    _require_arviz()
    import jax.numpy as jnp

    posterior = inference_data.posterior
    param_names = [v for v in posterior.data_vars if v != "sigma" and v != "phi"]

    median_params = {}
    for name in param_names:
        median_params[name] = float(posterior[name].values.flatten().mean())

    sim = np.array(
        jax_model_fn(
            {k: jnp.float64(v) for k, v in median_params.items()},
            jnp.array(precip),
            jnp.array(pet),
        )["simulated_flow"]
    )

    obs = np.asarray(obs_flow).flatten()
    sim_w = sim[warmup_steps:]
    obs_w = obs[warmup_steps:]

    ss_res = np.sum((obs_w - sim_w) ** 2)
    ss_tot = np.sum((obs_w - np.mean(obs_w)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
