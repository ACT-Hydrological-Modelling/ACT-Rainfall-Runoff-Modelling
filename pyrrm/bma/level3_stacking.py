"""
Level 3: Bayesian stacking via cross-validated log predictive density.

Finds weights that maximise the leave-one-block-out log mixture density.
Unlike classical BMA, stacking does not assume any of the K models is
the "true" data-generating process (M-open view; Yao et al. 2018).

Reference:
    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018).
    Using stacking to average Bayesian predictive distributions.
    Bayesian Analysis, 13(3), 917-1007.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    pass


def stacking_fit(
    F: np.ndarray,
    y_obs: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Fit stacking weights using block temporal CV.

    For each fold the per-model log predictive density is computed on
    the held-out block (using sigma estimated from the training fold).
    Weights are then found by maximising the pooled log mixture density.

    Args:
        F: (T, K) predictions.
        y_obs: (T,) observations.
        cv_splits: list of (train_idx, val_idx) pairs.

    Returns:
        Stacking weight vector of shape (K,).
    """
    K = F.shape[1]
    lpd = np.full((len(y_obs), K), np.nan)

    for train_idx, val_idx in cv_splits:
        for k in range(K):
            sigma_k = float(np.std(y_obs[train_idx] - F[train_idx, k]))
            sigma_k = max(sigma_k, 1e-10)
            residuals = y_obs[val_idx] - F[val_idx, k]
            lpd[val_idx, k] = (
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma_k)
                - 0.5 * (residuals / sigma_k) ** 2
            )

    valid = ~np.any(np.isnan(lpd), axis=1)
    lpd_valid = lpd[valid]

    if len(lpd_valid) == 0:
        return np.ones(K) / K

    def neg_log_score(w: np.ndarray) -> float:
        w_safe = np.clip(w, 1e-15, None)
        w_safe = w_safe / w_safe.sum()
        mix_lpd = np.log(np.sum(w_safe[None, :] * np.exp(lpd_valid), axis=1))
        return -float(np.sum(mix_lpd))

    w0 = np.ones(K) / K
    result = minimize(
        neg_log_score,
        w0,
        method="SLSQP",
        bounds=[(0.0, None)] * K,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def stacking_predict(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Point prediction using stacking weights."""
    return F @ weights
