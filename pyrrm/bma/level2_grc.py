"""
Level 2: Granger-Ramanathan combination (GRC / GRA).

GRC — constrained: weights >= 0, sum to 1, no intercept.
GRA — unconstrained: OLS with intercept, negative weights allowed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def grc_fit(F_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Fit constrained Granger-Ramanathan weights.

    Minimises ``||y_train - F_train @ w||^2`` subject to
    ``w >= 0`` and ``sum(w) == 1``.

    Args:
        F_train: (T, K) training predictions.
        y_train: (T,) training observations.

    Returns:
        Weight vector of shape (K,).
    """
    K = F_train.shape[1]
    w0 = np.ones(K) / K

    def objective(w: np.ndarray) -> float:
        return float(np.sum((y_train - F_train @ w) ** 2))

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, None)] * K,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def gra_fit(
    F_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit unconstrained Granger-Ramanathan with intercept.

    ``y = a + F @ w``  (no positivity or sum-to-one constraint).

    Returns:
        (weights, intercept).
    """
    X = np.column_stack([np.ones(len(y_train)), F_train])
    coeffs, *_ = np.linalg.lstsq(X, y_train, rcond=None)
    return coeffs[1:], float(coeffs[0])


def grc_predict(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply GRC weights."""
    return F @ weights


def gra_predict(
    F: np.ndarray, weights: np.ndarray, intercept: float
) -> np.ndarray:
    """Apply GRA weights + intercept."""
    return intercept + F @ weights
