"""
MCMC diagnostic and posterior predictive visualisation.

Complements the existing calibration_plots.py and model_plots.py
modules with plots tailored to MCMC output (NumPyro NUTS, PyDREAM)
and ArviZ InferenceData objects.
"""

from typing import Optional, List, Callable, Union, Dict, Any

import numpy as np
import pandas as pd

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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    make_subplots = None


def _require_deps():
    if not ARVIZ_AVAILABLE:
        raise ImportError("ArviZ is required. Install with: pip install arviz")
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")


# -----------------------------------------------------------------------
# DREAM → ArviZ conversion
# -----------------------------------------------------------------------

def dream_result_to_inference_data(
    result,
    burn_fraction: float = 0.3,
    param_names: Optional[List[str]] = None,
    post_convergence_draws: Optional[int] = None,
):
    """
    Convert a PyDREAM ``CalibrationResult`` to an ArviZ ``InferenceData``.

    Extracts per-chain MCMC samples from the raw result, discards an
    initial burn-in fraction, optionally keeps only the last N draws per
    chain (post-convergence), and builds an ``InferenceData`` via
    ``az.from_dict``.  The resulting object can be passed directly to
    every ``plot_mcmc_*`` function in this module.

    Args:
        result: A ``CalibrationResult`` produced by
            ``CalibrationRunner.run_pydream()``.  Must contain
            ``_raw_result['sampled_params_by_chain']`` **or** an
            ``all_samples`` DataFrame with a ``chain`` column.
        burn_fraction: Fraction of each chain to discard as burn-in
            (default 0.3 = first 30 %).
        param_names: Parameter names to include.  If *None*, names are
            inferred from ``_raw_result['parameter_names']`` or from
            the ``all_samples`` DataFrame columns.
        post_convergence_draws: If set, keep only the last N draws per
            chain (post-convergence).  Use for posterior plots so that
            only converged samples are shown.  If *None*, all post-burn-in
            draws are used.

    Returns:
        ``arviz.InferenceData`` with a ``posterior`` group of shape
        *(n_chains, n_draws)* per variable.

    Raises:
        ValueError: If chain-level data cannot be found in *result*.
    """
    _require_deps()

    def _take_post_convergence(chains_list: List[np.ndarray], n: int):
        """Keep only the last n draws per chain."""
        out = []
        for c in chains_list:
            if c.shape[0] <= n:
                out.append(c)
            else:
                out.append(c[-n:])
        return out

    chains_arr = None

    if (
        result._raw_result is not None
        and result._raw_result.get("sampled_params_by_chain") is not None
    ):
        chains_arr = result._raw_result["sampled_params_by_chain"]
        if param_names is None:
            param_names = result._raw_result.get("parameter_names")

    if chains_arr is not None and len(chains_arr) > 0:
        n_chains = len(chains_arr)
        burned = []
        for chain in chains_arr:
            n_samples = chain.shape[0]
            start = int(burn_fraction * n_samples)
            burned.append(chain[start:])

        if post_convergence_draws is not None:
            burned = _take_post_convergence(burned, post_convergence_draws)

        min_draws = min(c.shape[0] for c in burned)
        burned = [c[-min_draws:] for c in burned] if post_convergence_draws is not None else [c[:min_draws] for c in burned]

        stacked = np.stack(burned, axis=0)  # (chains, draws, params)

        if param_names is None:
            param_names = [f"param_{i}" for i in range(stacked.shape[2])]

        posterior_dict: Dict[str, Any] = {
            name: stacked[:, :, i] for i, name in enumerate(param_names)
        }

    elif result.all_samples is not None and "chain" in result.all_samples.columns:
        df = result.all_samples
        skip_cols = {"iteration", "log_likelihood", "chain"}

        if param_names is None:
            param_names = [c for c in df.columns if c not in skip_cols]

        chain_ids = sorted(df["chain"].unique())
        n_chains = len(chain_ids)

        per_chain = []
        for cid in chain_ids:
            cdf = df[df["chain"] == cid]
            start = int(burn_fraction * len(cdf))
            per_chain.append(cdf.iloc[start:][param_names].values)

        if post_convergence_draws is not None:
            per_chain = _take_post_convergence(per_chain, post_convergence_draws)

        min_draws = min(c.shape[0] for c in per_chain)
        per_chain = [c[-min_draws:] for c in per_chain] if post_convergence_draws is not None else [c[:min_draws] for c in per_chain]
        stacked = np.stack(per_chain, axis=0)

        posterior_dict = {
            name: stacked[:, :, i] for i, name in enumerate(param_names)
        }
    else:
        raise ValueError(
            "Cannot build InferenceData: result has no chain-level samples. "
            "Ensure run_pydream() was called with n_chains >= 2."
        )

    return az.from_dict(posterior=posterior_dict)


# -----------------------------------------------------------------------
# Forest plot
# -----------------------------------------------------------------------

def plot_mcmc_forest(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
):
    """
    Forest plot showing credible intervals per variable.

    Thin wrapper around ``az.plot_forest``.  Particularly useful for
    comparing posteriors across multiple MCMC runs when a list of
    ``InferenceData`` objects is passed.

    Args:
        inference_data: ArviZ InferenceData (or a list of them for
            multi-model comparison).
        var_names: Subset of variables (None = all).
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib axes array.
    """
    _require_deps()
    axes = az.plot_forest(
        inference_data,
        var_names=var_names,
        figsize=figsize,
        combined=True,
        **kwargs,
    )
    plt.tight_layout()
    return axes


def plot_forest_grid(
    idatas: Dict[str, Any],
    var_names: List[str],
    param_bounds: Optional[Dict[str, tuple]] = None,
    nrows: int = 6,
    ncols: int = 4,
    figsize: Optional[tuple] = None,
):
    """
    Single figure with a grid of forest plots, one subplot per parameter.

    Each cell shows the posterior of one parameter across all objectives
    (idatas). Axes are set to the parameter's feasible range when
    *param_bounds* is provided.

    Args:
        idatas: Dict of objective_name -> InferenceData.
        var_names: List of parameter names to plot (one per subplot).
        param_bounds: Optional {name: (lo, hi)} for x-axis limits.
        nrows: Number of rows (default 6).
        ncols: Number of columns (default 4).
        figsize: Figure size. If None, (4 * ncols, 1.8 * nrows).

    Returns:
        (fig, axes) with axes shape (nrows, ncols).
    """
    _require_deps()

    n = nrows * ncols
    params = [p for p in var_names if any(p in idata.posterior.data_vars for idata in idatas.values())][:n]

    if figsize is None:
        figsize = (4 * ncols, 1.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, param in enumerate(params):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        idatas_with = {k: idata for k, idata in idatas.items() if param in idata.posterior.data_vars}
        if len(idatas_with) < 2:
            ax.text(0.5, 0.5, param, ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xlim(0, 1)
            continue

        az.plot_forest(
            list(idatas_with.values()),
            model_names=list(idatas_with.keys()),
            var_names=[param],
            combined=True,
            ax=ax,
        )
        if param_bounds and param in param_bounds:
            ax.set_xlim(param_bounds[param])
        ax.set_title(param, fontsize=9, fontweight="bold")

    for idx in range(len(params), n):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_forest_grid_plotly(
    idatas: Dict[str, Any],
    var_names: List[str],
    param_bounds: Optional[Dict[str, tuple]] = None,
    nrows: int = 6,
    ncols: int = 4,
    row_height: int = 320,
    title: Optional[str] = None,
):
    """
    Plotly grid of cross-objective forest plots with a single consolidated legend.

    One subplot per parameter; each subplot shows posterior (mean + 94% HDI) for
    that parameter across all objectives. No per-subplot legends — one shared
    legend at the top. Row height is sized so ~13 objective labels on the y-axis
    fit without clutter (default 320px per row).

    Args:
        idatas: Dict of objective_name -> ArviZ InferenceData.
        var_names: List of parameter names (one per subplot).
        param_bounds: Optional {name: (lo, hi)} for x-axis limits per subplot.
        nrows: Number of rows (default 6).
        ncols: Number of columns (default 4).
        row_height: Height in pixels per row (default 320 for 13 objectives).
        title: Figure title.

    Returns:
        plotly.graph_objects.Figure, or None if Plotly/ArviZ unavailable.
    """
    if not ARVIZ_AVAILABLE or not PLOTLY_AVAILABLE or make_subplots is None:
        return None
    obj_names = list(idatas.keys())
    if len(obj_names) < 2:
        return None

    n = nrows * ncols
    params = [
        p for p in var_names
        if any(p in idatas[o].posterior.data_vars for o in obj_names)
    ][:n]
    if not params:
        return None

    def _get_summary(idata, param):
        if param not in idata.posterior.data_vars:
            return None
        try:
            s = az.summary(idata, var_names=[param])
            row = s.iloc[0]
            mean = float(row["mean"])
            if "hdi_3%" in row and "hdi_97%" in row:
                return (mean, float(row["hdi_3%"]), float(row["hdi_97%"]))
            sd = float(row.get("sd", 0))
            return (mean, mean - 2 * sd, mean + 2 * sd)
        except Exception:
            return None

    # Assign distinct colors to objectives (reusable across subplots)
    import plotly.express as px
    palettes = (
        px.colors.qualitative.Set2
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel1
    )
    obj_colors = dict(zip(obj_names, palettes[: len(obj_names)]))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=params,
        horizontal_spacing=0.08,
        vertical_spacing=0.04,
        row_heights=[1.0] * nrows,
    )

    for idx, param in enumerate(params):
        row, col = idx // ncols + 1, idx % ncols + 1
        # Only the first subplot shows legend entries (single consolidated legend)
        showlegend = idx == 0
        for obj in obj_names:
            summ = _get_summary(idatas[obj], param)
            if summ is None:
                continue
            mean, hdi_lo, hdi_hi = summ
            color = obj_colors.get(obj, "gray")
            fig.add_trace(
                go.Scatter(
                    x=[hdi_lo, mean, hdi_hi],
                    y=[obj, obj, obj],
                    mode="lines+markers",
                    line=dict(width=4, color=color),
                    marker=dict(symbol="diamond", size=8, color=color),
                    name=obj,
                    legendgroup=obj,
                    showlegend=showlegend,
                    hovertemplate=(
                        "<b>%{y}</b><br>Mean: %{customdata[0]:.4g}<br>"
                        "94% HDI: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<extra></extra>"
                    ),
                    customdata=[[mean, hdi_lo, hdi_hi]] * 3,
                ),
                row=row,
                col=col,
            )
        if param_bounds and param in param_bounds:
            lo, hi = param_bounds[param]
            fig.update_xaxes(range=[lo, hi], row=row, col=col)
        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(title_text="", row=row, col=col)

    fig.update_layout(
        title=title or "Cross-objective posterior (all parameters)",
        height=row_height * nrows,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=10),
        ),
        margin=dict(t=100, b=50, l=60, r=50),
        font=dict(size=10),
    )
    # Give y-axes room for ~13 objective labels: fixed range and tick font
    fig.update_yaxes(
        tickfont=dict(size=9),
        fixedrange=True,
    )
    return fig


def plot_forest_interactive(
    idatas: Dict[str, Any],
    var_names: List[str],
    param_bounds: Optional[Dict[str, tuple]] = None,
    height_per_objective: float = 44,
    title: Optional[str] = None,
):
    """
    Interactive Plotly forest plot with a dropdown to select parameter.

    Shows one parameter at a time: posterior (mean + 94% HDI) for that
    parameter across all objectives. Height is sized so all objective
    labels (e.g. 13) are readable. Each objective has a distinct color
    matching the grid plot. Use the dropdown to switch parameters.

    Args:
        idatas: Dict of objective_name -> ArviZ InferenceData.
        var_names: List of parameter names (dropdown options).
        param_bounds: Optional {name: (lo, hi)} for x-axis range.
        height_per_objective: Vertical pixels per objective (default 44 for 13).
        title: Figure title.

    Returns:
        plotly.graph_objects.Figure, or None if Plotly/ArviZ unavailable.
    """
    if not ARVIZ_AVAILABLE or not PLOTLY_AVAILABLE:
        return None
    obj_names = list(idatas.keys())
    if len(obj_names) < 2:
        return None
    # Filter to parameters that appear in at least one idata
    params = [
        p for p in var_names
        if any(p in idatas[idn].posterior.data_vars for idn in obj_names)
    ]
    if not params:
        return None

    # Shared color palette (same as grid) so objectives are recognizable
    try:
        import plotly.express as px
        palettes = (
            px.colors.qualitative.Set2
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Pastel1
        )
        obj_colors = dict(zip(obj_names, palettes[: len(obj_names)]))
    except Exception:
        obj_colors = {}

    # Precompute per-parameter, per-objective: (mean, hdi_lo, hdi_hi)
    def _get_summary(idata, param):
        if param not in idata.posterior.data_vars:
            return None
        try:
            s = az.summary(idata, var_names=[param])
            row = s.iloc[0]
            mean = float(row["mean"])
            if "hdi_3%" in row and "hdi_97%" in row:
                return (mean, float(row["hdi_3%"]), float(row["hdi_97%"]))
            sd = float(row.get("sd", 0))
            return (mean, mean - 2 * sd, mean + 2 * sd)
        except Exception:
            return None

    data_by_param: Dict[str, List[Dict]] = {}
    for param in params:
        traces = []
        for obj in obj_names:
            summ = _get_summary(idatas[obj], param)
            if summ is None:
                continue
            mean, hdi_lo, hdi_hi = summ
            color = obj_colors.get(obj, "gray")
            traces.append(
                go.Scatter(
                    x=[hdi_lo, mean, hdi_hi],
                    y=[obj, obj, obj],
                    mode="lines+markers",
                    line=dict(width=5, color=color),
                    marker=dict(symbol="diamond", size=10, color=color),
                    name=obj,
                    legendgroup=obj,
                    customdata=[[mean, hdi_lo, hdi_hi]] * 3,
                    hovertemplate="<b>%{y}</b><br>Mean: %{customdata[0]:.4g}<br>94% HDI: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<extra></extra>",
                )
            )
        if traces:
            data_by_param[param] = traces

    if not data_by_param:
        return None

    # Initial view: first parameter
    first_param = list(data_by_param.keys())[0]
    param_order = [p for p in params if p in data_by_param]
    x_range = param_bounds.get(first_param) if param_bounds else None

    fig = go.Figure(data=data_by_param[first_param])
    total_height = max(500, int(len(obj_names) * height_per_objective))
    fig.update_layout(
        title=dict(
            text=title or "Cross-objective posterior: select parameter",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=first_param,
        yaxis_title="Objective",
        height=total_height,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
        ),
        margin=dict(l=140, r=60, t=100, b=60),
        font=dict(size=11),
        yaxis=dict(
            tickfont=dict(size=10),
            fixedrange=True,
        ),
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))

    # Dropdown: "Parameter: [param]" with explicit x-axis title and range
    buttons = []
    for param in param_order:
        x_range = param_bounds.get(param) if param_bounds else None
        layout_patch = {"xaxis": {"title": {"text": param}}}
        if x_range is not None:
            layout_patch["xaxis"]["range"] = list(x_range)
        buttons.append(
            dict(
                label=param,
                method="update",
                args=[
                    {"data": data_by_param[param], "layout": layout_patch},
                ],
            )
        )
    active_idx = param_order.index(first_param) if first_param in param_order else 0
    fig.update_layout(
        updatemenus=[
            dict(
                active=active_idx,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.06,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.2)",
                font=dict(size=12),
            )
        ],
    )
    return fig


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


def plot_dream_traces(
    result,
    var_names: Optional[List[str]] = None,
    param_bounds: Optional[Dict[str, tuple]] = None,
    burn_fraction: float = 0.3,
    post_convergence_draws: int = 1000,
    kde_bw: float = 1.5,
    figsize: Optional[tuple] = None,
    chain_colors: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Full-trajectory DREAM trace plot with post-convergence shading.

    Shows all post-burn-in samples for every chain on the right (trace)
    panels, with the last *post_convergence_draws* per chain highlighted
    by a shaded background.  The left (density) panels show a combined
    KDE built only from those post-convergence draws.

    When *param_bounds* is supplied the KDE x-axis spans the full
    feasible range ``[lower, upper]`` so the posterior location within
    the prior is visible.

    Args:
        result: ``CalibrationResult`` from ``run_pydream()``.
        var_names: Parameters to plot.  *None* = all.
        param_bounds: ``{name: (lo, hi)}`` from
            ``model.get_parameter_bounds()``.  Sets the KDE x-axis.
        burn_fraction: Fraction of each chain discarded as burn-in.
        post_convergence_draws: Number of *tail* draws per chain
            treated as post-convergence (highlighted region).
        kde_bw: Bandwidth multiplier for ``scipy.stats.gaussian_kde``.
        figsize: Overall figure size (auto-scaled if *None*).
        chain_colors: One colour per chain.  Defaults to matplotlib
            tab10 cycle.
        title: Optional suptitle override.

    Returns:
        ``(fig, axes)`` — the matplotlib Figure and 2-D axes array
        with shape *(n_params, 2)*.
    """
    _require_deps()
    from scipy.stats import gaussian_kde

    raw = (
        result._raw_result.get("sampled_params_by_chain")
        if result._raw_result is not None
        else None
    )
    if raw is None or len(raw) == 0:
        raise ValueError("No chain-level samples in result._raw_result")

    all_names: List[str] = (
        result._raw_result.get("parameter_names")
        or [f"param_{i}" for i in range(raw[0].shape[1])]
    )

    if var_names is None:
        var_names = all_names
    var_idx = [all_names.index(v) for v in var_names if v in all_names]
    var_names = [all_names[i] for i in var_idx]

    n_chains = len(raw)
    burned: List[np.ndarray] = []
    for chain in raw:
        start = int(burn_fraction * chain.shape[0])
        burned.append(chain[start:])

    min_len = min(c.shape[0] for c in burned)
    burned = [c[:min_len] for c in burned]

    n_draws = min_len
    n_post = min(post_convergence_draws, n_draws)
    conv_start = n_draws - n_post

    n_params = len(var_names)
    if figsize is None:
        figsize = (14, 2.4 * n_params)

    if chain_colors is None:
        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        chain_colors = [cmap[i % len(cmap)] for i in range(n_chains)]

    fig, axes = plt.subplots(n_params, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [1, 2]})
    if n_params == 1:
        axes = axes[np.newaxis, :]

    for row, (pname, pidx) in enumerate(zip(var_names, var_idx)):
        ax_kde = axes[row, 0]
        ax_trace = axes[row, 1]

        post_samples_all: List[np.ndarray] = []

        for ci in range(n_chains):
            vals = burned[ci][:, pidx]
            color = chain_colors[ci]

            ax_trace.plot(
                np.arange(n_draws), vals,
                color=color, alpha=0.35, linewidth=0.4,
            )

            post_vals = vals[conv_start:]
            post_samples_all.append(post_vals)

        ax_trace.axvspan(
            conv_start, n_draws,
            color="0.85", zorder=0,
            label="post-convergence" if row == 0 else None,
        )

        ax_trace.set_ylabel(pname, fontsize=9, fontweight="bold")
        ax_trace.set_xlabel("Draw (post burn-in)" if row == n_params - 1 else "")
        ax_trace.tick_params(labelsize=8)

        pooled = np.concatenate(post_samples_all)

        if param_bounds and pname in param_bounds:
            xlo, xhi = param_bounds[pname]
        else:
            xlo, xhi = pooled.min(), pooled.max()
            pad = 0.05 * (xhi - xlo) if xhi > xlo else 1.0
            xlo, xhi = xlo - pad, xhi + pad

        xs = np.linspace(xlo, xhi, 400)

        try:
            kde = gaussian_kde(pooled, bw_method=kde_bw)
            ax_kde.fill_between(xs, kde(xs), alpha=0.5, color="steelblue")
            ax_kde.plot(xs, kde(xs), color="steelblue", linewidth=1.2)
        except np.linalg.LinAlgError:
            ax_kde.hist(pooled, bins=40, density=True, alpha=0.6,
                        color="steelblue", edgecolor="white", linewidth=0.5)

        ax_kde.set_xlim(xlo, xhi)
        ax_kde.set_ylabel("Density" if row == 0 else "")
        ax_kde.set_xlabel(pname if row == n_params - 1 else "")
        ax_kde.tick_params(labelsize=8)
        ax_kde.set_yticks([])

    if n_params > 0:
        axes[0, 1].legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.suptitle(
        title or (
            f"DREAM Trace & Posterior  "
            f"({n_chains} chains, last {n_post} draws highlighted)"
        ),
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig, axes


# -----------------------------------------------------------------------
# R-hat summary bar chart
# -----------------------------------------------------------------------

def plot_rhat_summary(
    inference_data,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    threshold: float = 1.05,
):
    """
    Horizontal bar chart of per-parameter R-hat values.

    More informative than rank histograms for assessing convergence at
    a glance: each bar is coloured green (converged) or red (not
    converged) relative to *threshold*.

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables.
        figsize: Figure size.
        threshold: R-hat convergence threshold (default 1.05).

    Returns:
        ``(fig, ax)`` tuple.
    """
    _require_deps()

    summary = az.summary(inference_data, var_names=var_names)
    rhats = summary["r_hat"]

    if figsize is None:
        figsize = (8, max(3, 0.35 * len(rhats)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ca02c" if v <= threshold else "#d62728" for v in rhats]
    y_pos = np.arange(len(rhats))

    ax.barh(y_pos, rhats, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rhats.index, fontsize=9)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"threshold = {threshold}")
    ax.axvline(1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("R-hat", fontsize=10)
    ax.set_title("Convergence: R-hat per Parameter", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0.98, max(rhats.max() * 1.05, threshold * 1.1))
    plt.tight_layout()
    return fig, ax


def plot_rhat_from_pydream(
    result,
    var_names: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    threshold: float = 1.05,
    show_both: bool = True,
):
    """
    R-hat bar chart using PyDREAM's Gelman–Rubin values.

    When *show_both* is True (default) and both sources are available,
    draws two bars per parameter: **at stop** (from convergence_history,
    when batch mode stopped) and **end of run** (from gelman_rubin on full
    chains). Otherwise draws a single set of bars.

    Args:
        result: ``CalibrationResult`` from ``run_pydream()``.
        var_names: Subset of parameter names to plot. If None, all
            in the GR dict(s) are used.
        figsize: Figure size.
        threshold: R-hat convergence threshold (default 1.05).
        show_both: If True and both at-stop and end-of-run GR exist,
            show both in grouped bars.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required.")

    cd = getattr(result, "convergence_diagnostics", None) or {}
    raw = getattr(result, "_raw_result", None) or {}

    # GR at the moment we decided to stop (before post-convergence iterations)
    gr_at_stop = None
    history = cd.get("convergence_history") or raw.get("convergence_history")
    conv_iter = cd.get("convergence_iteration") or raw.get("convergence_iteration")
    if history:
        if conv_iter is not None:
            # Use the entry at the iteration when we triggered stop (before extra batches)
            for e in history:
                if e.get("iteration") == conv_iter:
                    gr_at_stop = e.get("gr_values")
                    break
        if gr_at_stop is None:
            converged_entries = [e for e in history if e.get("converged")]
            if converged_entries:
                gr_at_stop = converged_entries[0].get("gr_values")  # first converged, not last

    # GR at end of run (full chains)
    gr_end = cd.get("gelman_rubin") or (raw.get("convergence_diagnostics") or {}).get("gelman_rubin")

    if not gr_at_stop and not gr_end:
        raise ValueError(
            "No convergence_diagnostics['gelman_rubin'] or convergence_history in result. "
            "Use plot_rhat_summary(inference_data) for ArviZ-based R-hat."
        )

    # Unified parameter list
    all_keys = list(gr_at_stop.keys() if gr_at_stop else gr_end.keys())
    if var_names is not None:
        all_keys = [n for n in var_names if n in (gr_at_stop or {}) or n in (gr_end or {})]
    names = all_keys
    n = len(names)

    has_both = show_both and gr_at_stop is not None and gr_end is not None

    if figsize is None:
        figsize = (8, max(3, 0.4 * n))

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)

    if has_both:
        bar_h = 0.36
        vals_stop = np.array([gr_at_stop.get(nm, np.nan) for nm in names])
        vals_end = np.array([gr_end.get(nm, np.nan) for nm in names])
        mask_stop = ~np.isnan(vals_stop)
        mask_end = ~np.isnan(vals_end)
        colors_stop = ["#2ca02c" if v <= threshold else "#d62728" for v in vals_stop]
        colors_end = ["#1f77b4" if v <= threshold else "#ff7f0e" for v in vals_end]
        ax.barh(y_pos - bar_h / 2, vals_stop, height=bar_h, color=colors_stop,
                edgecolor="white", linewidth=0.5, label="At stop")
        ax.barh(y_pos + bar_h / 2, vals_end, height=bar_h, color=colors_end,
                edgecolor="white", linewidth=0.5, label="End of run")
        valid = np.concatenate([vals_stop[np.isfinite(vals_stop)], vals_end[np.isfinite(vals_end)]])
        x_max = (valid.max() * 1.05) if len(valid) else (threshold * 1.1)
    else:
        gr = gr_at_stop if gr_at_stop is not None else gr_end
        values = np.array([gr[nm] for nm in names])
        colors = ["#2ca02c" if v <= threshold else "#d62728" for v in values]
        ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
        x_max = values.max() * 1.05

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"threshold = {threshold}")
    ax.axvline(1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("R-hat (PyDREAM)", fontsize=10)
    ax.set_title("Convergence: R-hat per Parameter", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0.98, max(x_max, threshold * 1.1))
    plt.tight_layout()
    return fig, ax


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
    kde_bw: float = 2.5,
    compact: bool = True,
    use_seaborn: bool = True,
    max_vars: Optional[int] = None,
    max_samples: Optional[int] = 1000,
    **kwargs,
):
    """
    Pair (corner) plot showing marginal and joint posterior KDEs.

    When *use_seaborn* is True (default), uses seaborn's pairplot with
    ``bw_adjust=kde_bw``. Use *max_vars* and *max_samples* to keep runtime
    low for large posteriors (e.g. 22 params × 3000 draws).

    Args:
        inference_data: ArviZ InferenceData.
        var_names: Subset of variables (e.g. all 22 Sacramento params).
        figsize: Figure size. If None and compact, auto-sized.
        kde_bw: KDE bandwidth multiplier (default 2.5 for smooth curves).
        compact: If True, use tight spacing and auto figsize.
        use_seaborn: If True, use seaborn pairplot for consistent bandwidth.
        max_vars: If set, plot at most this many variables (first from
            var_names). Cuts cost for 22-param runs (e.g. 8 or 10).
        max_samples: Max posterior draws to use for KDE (subsample if
            larger). Default 1000 keeps pair plot fast.

    Returns:
        Figure (seaborn) or axes array (ArviZ).
    """
    _require_deps()

    posterior = inference_data.posterior
    names = var_names or list(posterior.data_vars)
    names = [n for n in names if n in posterior.data_vars]

    if max_vars is not None and len(names) > max_vars:
        names = names[:max_vars]

    nvars = len(names)
    if nvars < 2:
        raise ValueError("Need at least 2 variables for pair plot.")

    if figsize is None and compact and nvars > 6:
        # Larger cells for readability when showing all params (e.g. 22)
        side = max(10, min(28, 1.0 * nvars))
        figsize = (side, side)

    if use_seaborn:
        try:
            import seaborn as sns
        except ImportError:
            use_seaborn = False

    if use_seaborn:
        # Flatten posterior to (n_samples, n_vars); subsample for speed
        arrays = [posterior[n].values.ravel() for n in names]
        n_draws = len(arrays[0])
        if max_samples is not None and n_draws > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_draws, size=max_samples, replace=False)
            arrays = [arr[idx] for arr in arrays]
        df = pd.DataFrame(dict(zip(names, arrays)))
        plot_kws = dict(bw_adjust=kde_bw)
        diag_kws = dict(bw_adjust=kde_bw)
        cell_h = (figsize[0] / nvars) if figsize else max(0.9, min(2.5, 22 / nvars))
        fig = sns.pairplot(
            df,
            vars=names,
            kind="kde",
            diag_kind="kde",
            plot_kws=plot_kws,
            diag_kws=diag_kws,
            height=cell_h,
            aspect=1,
        )
        fig.fig.subplots_adjust(wspace=0.06, hspace=0.06)
        plt.figure(fig.fig.number)
        return fig.fig

    # ArviZ path
    kde_kwargs = kwargs.pop("kde_kwargs", None) or {}
    kde_kwargs.setdefault("hdi_probs", [0.5, 0.94])
    kde_kwargs.setdefault("bw", kde_bw)
    marginal_kwargs = kwargs.pop("marginal_kwargs", None) or {}
    marginal_kwargs.setdefault("bw", kde_bw)
    backend_kwargs = kwargs.pop("backend_kwargs", None) or {}
    if compact and nvars > 4:
        backend_kwargs.setdefault("figsize", figsize)
        backend_kwargs.setdefault("gridspec_kw", {"wspace": 0.05, "hspace": 0.05})
    axes = az.plot_pair(
        inference_data,
        var_names=names,
        kind="kde",
        marginals=True,
        figsize=figsize,
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
        backend_kwargs=backend_kwargs,
        **kwargs,
    )
    plt.tight_layout(pad=0.5)
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
