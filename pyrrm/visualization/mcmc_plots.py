"""
MCMC diagnostic and posterior predictive visualisation.

Complements the existing calibration_plots.py and model_plots.py
modules with plots tailored to NumPyro NUTS output and ArviZ
InferenceData objects.
"""

from typing import Optional, List, Callable, Union

import numpy as np

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def _require_deps():
    if not ARVIZ_AVAILABLE:
        raise ImportError("ArviZ is required. Install with: pip install arviz")
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")


# -----------------------------------------------------------------------
# Trace / density plots
# -----------------------------------------------------------------------

def plot_mcmc_traces(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
):
    """
    Combined trace + kernel density plot per variable.

    Thin wrapper around ``az.plot_trace`` with sensible defaults.

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables (None = all).
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib axes array.
    """
    _require_deps()
    axes = az.plot_trace(inference_data, var_names=var_names, figsize=figsize, **kwargs)
    plt.tight_layout()
    return axes


# -----------------------------------------------------------------------
# Rank plots (chain mixing diagnostics)
# -----------------------------------------------------------------------

def plot_mcmc_rank(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
):
    """
    Rank plots for assessing chain mixing.

    Thin wrapper around ``az.plot_rank``.

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables.
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib axes array.
    """
    _require_deps()
    axes = az.plot_rank(inference_data, var_names=var_names, figsize=figsize, **kwargs)
    plt.tight_layout()
    return axes


# -----------------------------------------------------------------------
# Posterior pair / corner plot
# -----------------------------------------------------------------------

def plot_posterior_pairs(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
):
    """
    Pair (corner) plot showing marginal posteriors and bivariate scatter.

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables.
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib axes array.
    """
    _require_deps()
    axes = az.plot_pair(
        inference_data,
        var_names=var_names,
        kind="hexbin",
        marginals=True,
        figsize=figsize,
        **kwargs,
    )
    plt.tight_layout()
    return axes


# -----------------------------------------------------------------------
# Hydrograph with posterior predictive uncertainty
# -----------------------------------------------------------------------

def plot_hydrograph_with_uncertainty(
    jax_model_fn: Callable,
    inference_data,
    precip: np.ndarray,
    pet: np.ndarray,
    obs_flow: np.ndarray,
    dates: Optional[np.ndarray] = None,
    warmup_steps: int = 365,
    n_samples: int = 200,
    ci_levels: tuple = (0.05, 0.25, 0.75, 0.95),
    figsize: tuple = (14, 5),
    title: str = "Posterior Predictive Hydrograph",
    seed: int = 0,
):
    """
    Plot observed flow with posterior predictive credible intervals.

    Draws ``n_samples`` parameter sets from the posterior, runs the
    JAX model for each, and plots percentile bands.

    Args:
        jax_model_fn: JAX forward model (e.g. ``gr4j_run_jax``).
        inference_data: ArviZ InferenceData with posterior samples.
        precip: Precipitation array.
        pet: PET array.
        obs_flow: Observed flow array.
        dates: Optional date array for x-axis.
        warmup_steps: Leading timesteps to skip in plot.
        n_samples: Number of posterior draws to simulate.
        ci_levels: Percentile levels for shading (outer, inner pair).
        figsize: Figure size.
        title: Plot title.
        seed: RNG seed for draw selection.

    Returns:
        Matplotlib (fig, ax) tuple.
    """
    _require_deps()
    import jax.numpy as jnp

    skip = {"sigma", "phi"}

    if hasattr(inference_data, "deterministic") and len(inference_data.deterministic.data_vars) > 0:
        src = inference_data.deterministic
        param_names = [v for v in src.data_vars if v not in skip]
    else:
        src = inference_data.posterior
        param_names = [v for v in src.data_vars if v not in skip]

    flat_samples = {
        name: src[name].values.reshape(-1)
        for name in param_names
    }
    n_available = len(flat_samples[param_names[0]])
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_available, size=min(n_samples, n_available), replace=False)

    precip_jax = jnp.array(precip)
    pet_jax = jnp.array(pet)

    sims = []
    for i in idx:
        params = {name: jnp.float64(flat_samples[name][i]) for name in param_names}
        sim = np.array(jax_model_fn(params, precip_jax, pet_jax)["simulated_flow"])
        sims.append(sim)
    sims = np.array(sims)

    sims_w = sims[:, warmup_steps:]
    obs_w = obs_flow[warmup_steps:]
    n_plot = len(obs_w)

    if dates is not None:
        x = dates[warmup_steps:]
    else:
        x = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=figsize)
    lo_outer, lo_inner, hi_inner, hi_outer = ci_levels

    q_outer = np.percentile(sims_w, [lo_outer * 100, hi_outer * 100], axis=0)
    q_inner = np.percentile(sims_w, [lo_inner * 100, hi_inner * 100], axis=0)
    q_median = np.median(sims_w, axis=0)

    ax.fill_between(x, q_outer[0], q_outer[1], alpha=0.2, color="C0",
                     label=f"{lo_outer*100:.0f}-{hi_outer*100:.0f}% CI")
    ax.fill_between(x, q_inner[0], q_inner[1], alpha=0.4, color="C0",
                     label=f"{lo_inner*100:.0f}-{hi_inner*100:.0f}% CI")
    ax.plot(x, q_median, color="C0", linewidth=1.0, label="Median")
    ax.plot(x, obs_w, color="k", linewidth=0.6, alpha=0.7, label="Observed")

    ax.set_xlabel("Time step" if dates is None else "Date")
    ax.set_ylabel("Flow")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    if dates is not None:
        fig.autofmt_xdate()

    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------
# Convenience composite diagnostic figure
# -----------------------------------------------------------------------

def plot_mcmc_diagnostics(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: tuple = (14, 10),
):
    """
    Create a 2x2 diagnostic grid: trace, rank, posterior, energy.

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables.
        figsize: Overall figure size.

    Returns:
        Matplotlib Figure.
    """
    _require_deps()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    ax_trace = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_post = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])

    az.plot_trace(inference_data, var_names=var_names, axes=np.atleast_2d([[ax_trace, ax_trace]]),
                  compact=True)
    az.plot_rank(inference_data, var_names=var_names, ax=ax_rank)
    az.plot_posterior(inference_data, var_names=var_names, ax=ax_post)
    try:
        az.plot_energy(inference_data, ax=ax_energy)
    except Exception:
        ax_energy.text(0.5, 0.5, "Energy plot\nnot available",
                       ha="center", va="center", transform=ax_energy.transAxes)

    fig.suptitle("MCMC Diagnostics", fontsize=14, y=1.01)
    return fig
