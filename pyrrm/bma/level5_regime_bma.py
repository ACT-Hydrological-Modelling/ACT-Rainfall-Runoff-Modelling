"""
Level 5: Flow regime-specific BMA with sigmoid blending.

Fits separate BMA models for high / medium / low flow regimes, then
produces blended predictions using smooth sigmoid transitions at regime
boundaries so the combined prediction has no discontinuities.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    from pyrrm.bma.config import BMAConfig

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Regime-specific fitting
# ═════════════════════════════════════════════════════════════════════════

def build_regime_bma(
    F_train: np.ndarray,
    y_train: np.ndarray,
    regime_masks: Dict[str, np.ndarray],
    config: "BMAConfig",
) -> Dict[str, Tuple]:
    """Build and sample separate BMA models for each flow regime.

    Args:
        F_train: (T, K) predictions.
        y_train: (T,) observations.
        regime_masks: ``{'high': bool_mask, 'medium': ..., 'low': ...}``
        config: BMAConfig.

    Returns:
        Dict mapping regime name to ``(model, idata)`` tuples.  Regimes
        with fewer than 50 timesteps are skipped (logged as a warning).
    """
    from pyrrm.bma.level4_bma import build_bma_model, sample_bma, check_convergence

    results: Dict[str, Tuple] = {}
    for regime_name, mask in regime_masks.items():
        n_points = int(mask.sum())
        if n_points < 50:
            logger.warning(
                "Regime '%s' has only %d points — skipping "
                "(will fall back to global BMA).",
                regime_name, n_points,
            )
            continue

        logger.info(
            "Fitting BMA for regime '%s' (%d timesteps)...",
            regime_name, n_points,
        )
        F_regime = F_train[mask]
        y_regime = y_train[mask]

        model = build_bma_model(F_regime, y_regime, config)
        idata = sample_bma(model, config)

        issues = check_convergence(idata)
        if issues:
            logger.warning("Convergence issues in '%s': %s", regime_name, issues)

        results[regime_name] = (model, idata)

    return results


# ═════════════════════════════════════════════════════════════════════════
# Sigmoid blending weights
# ═════════════════════════════════════════════════════════════════════════

def compute_regime_blend_weights(
    flow_proxy: np.ndarray,
    q_high: float,
    q_low: float,
    blend_width: float,
) -> Dict[str, np.ndarray]:
    """Compute smooth sigmoid blending weights for regime transitions.

    At each timestep the three weights sum to 1.0.

    Args:
        flow_proxy: (T,) estimated flow (e.g. mean of model predictions).
        q_high: threshold separating high from medium.
        q_low: threshold separating medium from low.
        blend_width: width of the sigmoid transition zone (flow units).

    Returns:
        ``{'high': ..., 'medium': ..., 'low': ...}`` — each (T,).
    """
    def _sigmoid(x: np.ndarray, centre: float, width: float) -> np.ndarray:
        z = (x - centre) / max(width, 1e-10)
        return 1.0 / (1.0 + np.exp(-z))

    p_high = _sigmoid(flow_proxy, q_high, blend_width)
    p_low = 1.0 - _sigmoid(flow_proxy, q_low, blend_width)
    p_medium = np.clip(1.0 - p_high - p_low, 0.0, 1.0)

    total = p_high + p_medium + p_low
    total = np.maximum(total, 1e-10)
    return {
        "high": p_high / total,
        "medium": p_medium / total,
        "low": p_low / total,
    }


# ═════════════════════════════════════════════════════════════════════════
# Blended posterior-predictive sampling
# ═════════════════════════════════════════════════════════════════════════

def regime_blend_predict(
    F_val: np.ndarray,
    regime_results: Dict[str, Tuple],
    regime_blend_weights: Dict[str, np.ndarray],
    config: "BMAConfig",
    n_samples: int = 4000,
) -> Dict:
    """Generate blended predictions across regimes.

    For each posterior draw a regime is selected proportional to the
    blend weights and the corresponding regime BMA is used to generate
    the prediction.

    Returns:
        Dict with ``'samples'``, ``'mean'``, ``'median'``, ``'intervals'``.
    """
    rng = np.random.default_rng(config.random_seed)

    T = F_val.shape[0]
    all_samples = np.zeros((n_samples, T))

    regime_names = ["high", "medium", "low"]
    available = {r for r in regime_names if r in regime_results}
    if not available:
        raise ValueError("No regime BMA results available.")

    fallback_regime = next(iter(available))

    _cache: Dict[str, Dict[str, np.ndarray]] = {}
    for rname in available:
        _, idata = regime_results[rname]
        post = idata.posterior
        w_vals = post["w"].values.reshape(-1, post["w"].shape[-1])
        has_bias = "bias" in post
        bias_vals = (
            post["bias"].values.reshape(-1, w_vals.shape[-1])
            if has_bias
            else np.zeros((len(w_vals), w_vals.shape[-1]))
        )
        sigma_vals = post["sigma"].values
        if sigma_vals.ndim == 2:
            sigma_vals = sigma_vals.reshape(-1)[:, None] * np.ones((1, w_vals.shape[-1]))
        else:
            sigma_vals = sigma_vals.reshape(-1, w_vals.shape[-1])
        _cache[rname] = dict(w=w_vals, bias=bias_vals, sigma=sigma_vals)

    for s in range(n_samples):
        for t in range(T):
            probs = np.array([
                regime_blend_weights.get(r, np.zeros(T))[t]
                for r in regime_names
            ])
            probs = np.maximum(probs, 0)
            psum = probs.sum()
            if psum < 1e-15:
                probs = np.ones(3) / 3
            else:
                probs /= psum

            chosen_name = regime_names[rng.choice(3, p=probs)]
            if chosen_name not in available:
                chosen_name = fallback_regime

            c = _cache[chosen_name]
            s_idx = rng.integers(len(c["w"]))
            w_s = c["w"][s_idx]
            k = rng.choice(len(w_s), p=w_s)
            mu_k = F_val[t, k] + c["bias"][s_idx, k]
            all_samples[s, t] = rng.normal(mu_k, c["sigma"][s_idx, k])

    result: Dict = {
        "samples": all_samples,
        "mean": all_samples.mean(axis=0),
        "median": np.median(all_samples, axis=0),
        "intervals": {},
    }
    for level in config.prediction_intervals:
        lo = (1 - level) / 2
        hi = 1 - lo
        result["intervals"][level] = (
            np.quantile(all_samples, lo, axis=0),
            np.quantile(all_samples, hi, axis=0),
        )
    return result
