"""
Visualisation functions for BMA ensemble results.

All plots return ``matplotlib.figure.Figure`` objects so they can be
displayed inline (Jupyter) or saved to disk.

Functions
---------
plot_weight_comparison
    Grouped bar chart comparing weights across all 5 levels.
plot_posterior_weights
    Violin / forest plot of posterior weight distributions (Level 4-5).
plot_prediction_bands
    Hydrograph with shaded prediction intervals.
plot_pit_histogram
    Probability Integral Transform calibration diagnostic.
plot_method_comparison
    Side-by-side metric table as a heatmap.
plot_regime_weights
    How BMA weights change across flow regimes (Level 5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import arviz as az
    from pyrrm.bma.config import BMAConfig


# ═════════════════════════════════════════════════════════════════════════
# Weight comparison (all levels)
# ═════════════════════════════════════════════════════════════════════════

def plot_weight_comparison(
    model_names: List[str],
    weights_dict: Dict[str, np.ndarray],
    figsize: Tuple[float, float] = (12, 5),
) -> Figure:
    """Grouped bar chart of weights across combination methods.

    Args:
        model_names: length-K list of model names.
        weights_dict: mapping ``{method_name: weight_vector}``.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    methods = list(weights_dict.keys())
    K = len(model_names)
    x = np.arange(K)
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(methods):
        w = weights_dict[method]
        ax.bar(x + i * width, w, width, label=method, alpha=0.85)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weight")
    ax.set_title("Model Weights by Combination Method")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# Posterior weight distributions
# ═════════════════════════════════════════════════════════════════════════

def plot_posterior_weights(
    idata: "az.InferenceData",
    model_names: List[str],
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """Violin plot of posterior BMA weight distributions.

    Args:
        idata: ArviZ InferenceData containing ``w`` in the posterior.
        model_names: length-K model name list.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    w_samples = idata.posterior["w"].values.reshape(-1, len(model_names))

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        w_samples, positions=np.arange(len(model_names)),
        showmeans=True, showmedians=True,
    )

    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Posterior Weight")
    ax.set_title("BMA Posterior Weight Distributions")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# Prediction bands (hydrograph with intervals)
# ═════════════════════════════════════════════════════════════════════════

def plot_prediction_bands(
    dates: pd.DatetimeIndex,
    y_obs: np.ndarray,
    pred_dict: Dict[str, Any],
    title: str = "BMA Prediction with Uncertainty Bands",
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """Hydrograph with shaded prediction intervals.

    Args:
        dates: DatetimeIndex for the plotted period.
        y_obs: (T,) observed flow.
        pred_dict: output from ``generate_bma_predictions`` or
            ``regime_blend_predict`` (must have ``'mean'`` and
            ``'intervals'`` keys).
        title: plot title.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sorted_levels = sorted(pred_dict["intervals"].keys(), reverse=True)
    alphas = np.linspace(0.15, 0.35, len(sorted_levels))

    for alpha, level in zip(alphas, sorted_levels):
        lo, hi = pred_dict["intervals"][level]
        ax.fill_between(
            dates, lo, hi,
            alpha=alpha, color="steelblue",
            label=f"{int(level * 100)}% PI",
        )

    ax.plot(dates, pred_dict["mean"], color="steelblue", linewidth=0.8,
            label="BMA mean")
    ax.plot(dates, y_obs, color="black", linewidth=0.6, alpha=0.8,
            label="Observed")

    ax.set_xlabel("Date")
    ax.set_ylabel("Flow")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# PIT histogram
# ═════════════════════════════════════════════════════════════════════════

def plot_pit_histogram(
    pit: np.ndarray,
    n_bins: int = 20,
    figsize: Tuple[float, float] = (6, 4),
) -> Figure:
    """PIT histogram — should be approximately uniform if calibrated.

    Args:
        pit: (T,) PIT values from ``evaluation.pit_values``.
        n_bins: number of histogram bins.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(pit, bins=n_bins, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("Probability Integral Transform (PIT) Histogram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# Method comparison heatmap
# ═════════════════════════════════════════════════════════════════════════

def plot_method_comparison(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """Heatmap comparing metrics across methods.

    Args:
        results_df: DataFrame with methods as rows and metrics as
            columns (e.g. from ``pipeline.BMARunner.run_cross_validation``).
        metrics: subset of columns to display.  If None, all numeric
            columns are used.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    if metrics is not None:
        results_df = results_df[metrics]

    data = results_df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Method Comparison")
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# Regime weight heatmap
# ═════════════════════════════════════════════════════════════════════════

def plot_regime_weights(
    model_names: List[str],
    regime_results: Dict[str, Tuple],
    figsize: Tuple[float, float] = (10, 4),
) -> Figure:
    """Heatmap showing mean posterior weights per regime.

    Args:
        model_names: length-K model name list.
        regime_results: dict from ``build_regime_bma``; values are
            ``(model, idata)`` tuples.
        figsize: figure size.

    Returns:
        Matplotlib Figure.
    """
    regimes = list(regime_results.keys())
    K = len(model_names)
    mat = np.zeros((len(regimes), K))

    for i, regime_name in enumerate(regimes):
        _, idata = regime_results[regime_name]
        w = idata.posterior["w"].values.reshape(-1, K)
        mat[i] = w.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_yticklabels([r.capitalize() for r in regimes], fontsize=9)

    for i in range(len(regimes)):
        for j in range(K):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean weight")
    ax.set_title("BMA Weights by Flow Regime")
    fig.tight_layout()
    return fig
