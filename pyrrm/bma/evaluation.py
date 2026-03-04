"""
Evaluation metrics for BMA ensemble predictions.

Deterministic metrics (NSE, KGE, PBIAS, RMSE) are delegated to
``pyrrm.analysis.diagnostics.compute_diagnostics``.  This module adds
the probabilistic metrics unique to BMA:

* **CRPS** (ensemble formulation)
* **PIT** values + uniformity test
* **Coverage** and **interval width** (sharpness) at multiple levels
* **Regime-specific** evaluation wrapper
* **FDC errors** by exceedance-probability segment
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrrm.bma.config import BMAConfig


# ═════════════════════════════════════════════════════════════════════════
# Deterministic metrics (thin wrapper around existing diagnostics)
# ═════════════════════════════════════════════════════════════════════════

def evaluate_deterministic(y_obs: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard deterministic metrics via ``compute_diagnostics``.

    Falls back to a local minimal implementation if the diagnostics
    module is not available (e.g. during isolated testing).
    """
    try:
        from pyrrm.analysis.diagnostics import compute_diagnostics
        return dict(compute_diagnostics(y_pred, y_obs))
    except ImportError:
        pass

    def _nse(o, s):
        ss = np.sum((o - np.mean(o)) ** 2)
        return float(1 - np.sum((o - s) ** 2) / ss) if ss > 0 else np.nan

    def _kge(o, s):
        r = float(np.corrcoef(o, s)[0, 1])
        a = float(np.std(s) / np.std(o)) if np.std(o) > 0 else np.nan
        b = float(np.mean(s) / np.mean(o)) if np.mean(o) != 0 else np.nan
        return float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))

    return {
        "NSE": _nse(y_obs, y_pred),
        "KGE": _kge(y_obs, y_pred),
        "PBIAS": float(100 * np.sum(y_pred - y_obs) / np.sum(y_obs)),
        "RMSE": float(np.sqrt(np.mean((y_obs - y_pred) ** 2))),
    }


# ═════════════════════════════════════════════════════════════════════════
# Probabilistic metrics
# ═════════════════════════════════════════════════════════════════════════

def crps_ensemble(y_obs: np.ndarray, y_samples: np.ndarray) -> float:
    """CRPS using the ensemble formulation: E|Y-x| - 0.5·E|Y-Y'|.

    Args:
        y_obs: (T,) observations.
        y_samples: (S, T) posterior predictive samples.

    Returns:
        Mean CRPS across all timesteps.
    """
    T = len(y_obs)
    crps_vals = np.zeros(T)
    for t in range(T):
        ens = y_samples[:, t]
        crps_vals[t] = (
            np.mean(np.abs(ens - y_obs[t]))
            - 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]))
        )
    return float(crps_vals.mean())


def pit_values(y_obs: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """Probability Integral Transform values.

    For each observation, the fraction of ensemble members below it.
    Should be Uniform(0, 1) if the predictive distribution is calibrated.
    """
    T = len(y_obs)
    pit = np.zeros(T)
    for t in range(T):
        pit[t] = np.mean(y_samples[:, t] <= y_obs[t])
    return pit


def pit_uniformity_pvalue(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov test of PIT uniformity (p-value)."""
    from scipy.stats import kstest

    stat, pval = kstest(pit, "uniform")
    return float(pval)


def coverage(
    y_obs: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> float:
    """Fraction of observations within the prediction interval."""
    return float(np.mean((y_obs >= lower) & (y_obs <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction interval width (sharpness)."""
    return float(np.mean(upper - lower))


def evaluate_probabilistic(
    y_obs: np.ndarray,
    pred_dict: Dict[str, Any],
) -> Dict[str, float]:
    """Compute all probabilistic metrics from a prediction dict."""
    results: Dict[str, float] = {
        "CRPS": crps_ensemble(y_obs, pred_dict["samples"]),
    }
    pit = pit_values(y_obs, pred_dict["samples"])
    results["PIT_KS_pvalue"] = pit_uniformity_pvalue(pit)

    for level, (lo, hi) in pred_dict["intervals"].items():
        pct = int(level * 100)
        results[f"coverage_{pct}"] = coverage(y_obs, lo, hi)
        results[f"width_{pct}"] = interval_width(lo, hi)

    return results


# ═════════════════════════════════════════════════════════════════════════
# Regime-specific evaluation
# ═════════════════════════════════════════════════════════════════════════

def evaluate_by_regime(
    y_obs: np.ndarray,
    y_pred_mean: np.ndarray,
    y_samples: Optional[np.ndarray],
    regime_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics separately per flow regime."""
    results: Dict[str, Dict[str, float]] = {}
    for regime, mask in regime_masks.items():
        if mask.sum() == 0:
            continue
        det = evaluate_deterministic(y_obs[mask], y_pred_mean[mask])
        if y_samples is not None:
            det["CRPS"] = crps_ensemble(y_obs[mask], y_samples[:, mask])
        results[regime] = det
    return results


# ═════════════════════════════════════════════════════════════════════════
# FDC errors
# ═════════════════════════════════════════════════════════════════════════

def fdc_error(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    segments: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Flow Duration Curve RMSE by exceedance-probability segment."""
    if segments is None:
        segments = [(0, 5), (5, 20), (20, 70), (70, 95), (95, 100)]

    obs_sorted = np.sort(y_obs)[::-1]
    pred_sorted = np.sort(y_pred)[::-1]
    n = len(obs_sorted)
    exc = np.arange(1, n + 1) / n * 100

    errors: Dict[str, float] = {}
    for lo, hi in segments:
        mask = (exc >= lo) & (exc < hi)
        if mask.sum() > 0:
            errors[f"FDC_{lo}_{hi}"] = float(
                np.sqrt(np.mean((obs_sorted[mask] - pred_sorted[mask]) ** 2))
            )
    return errors
