"""
Posterior predictive generation for global BMA (Level 4).

Draws samples from the Dirichlet-weighted Gaussian mixture and computes
summary statistics (mean, median, prediction intervals).  Also provides
a helper to back-transform all arrays to original flow units.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    from pyrrm.bma.config import BMAConfig


def generate_bma_predictions(
    idata: "az.InferenceData",
    F_val: np.ndarray,
    config: "BMAConfig",
    n_samples: int = 4000,
) -> Dict[str, Any]:
    """Generate posterior predictive samples from a fitted global BMA.

    For each posterior draw:
        1. Get ``w``, ``bias``, ``sigma`` from the posterior.
        2. For each timestep, sample a component ``k ~ Categorical(w)``
           and draw ``y ~ N(mu_k, sigma_k)``.

    Args:
        idata: ArviZ InferenceData from ``sample_bma``.
        F_val: (T_val, K) validation predictions.
        config: BMAConfig.
        n_samples: Number of posterior-predictive draws.

    Returns:
        Dict with ``'samples'`` (S, T), ``'mean'`` (T,), ``'median'``
        (T,), and ``'intervals'`` mapping level to ``(lower, upper)``.
    """
    rng = np.random.default_rng(config.random_seed)

    post = idata.posterior
    w_flat = post["w"].values.reshape(-1, post["w"].shape[-1])

    has_bias = "bias" in post
    if has_bias:
        bias_flat = post["bias"].values.reshape(-1, w_flat.shape[-1])

    sigma_vals = post["sigma"].values
    if sigma_vals.ndim == 2:
        sigma_flat = sigma_vals.reshape(-1)[:, None] * np.ones((1, w_flat.shape[-1]))
    else:
        sigma_flat = sigma_vals.reshape(-1, w_flat.shape[-1])

    T_val = F_val.shape[0]
    K = F_val.shape[1]
    S = min(n_samples, len(w_flat))
    idx = rng.choice(len(w_flat), S, replace=S > len(w_flat))

    y_pred = np.zeros((S, T_val))

    for s_i, s in enumerate(idx):
        w_s = w_flat[s]
        bias_s = bias_flat[s] if has_bias else np.zeros(K)
        sigma_s = sigma_flat[s]

        mu = F_val + bias_s[None, :]  # (T_val, K)
        components = np.array([rng.choice(K, p=w_s) for _ in range(T_val)])

        for t in range(T_val):
            k = components[t]
            y_pred[s_i, t] = rng.normal(mu[t, k], sigma_s[k])

    result: Dict[str, Any] = {
        "samples": y_pred,
        "mean": y_pred.mean(axis=0),
        "median": np.median(y_pred, axis=0),
        "intervals": {},
    }
    for level in config.prediction_intervals:
        lo = (1 - level) / 2
        hi = 1 - lo
        result["intervals"][level] = (
            np.quantile(y_pred, lo, axis=0),
            np.quantile(y_pred, hi, axis=0),
        )
    return result


def back_transform_predictions(
    pred_dict: Dict[str, Any],
    transform_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply inverse transform to all prediction arrays."""
    from pyrrm.bma.data_prep import back_transform

    result: Dict[str, Any] = {
        "samples": back_transform(pred_dict["samples"], transform_params),
        "mean": back_transform(pred_dict["mean"], transform_params),
        "median": back_transform(pred_dict["median"], transform_params),
        "intervals": {},
    }
    for level, (lo, hi) in pred_dict["intervals"].items():
        result["intervals"][level] = (
            back_transform(lo, transform_params),
            back_transform(hi, transform_params),
        )
    return result
