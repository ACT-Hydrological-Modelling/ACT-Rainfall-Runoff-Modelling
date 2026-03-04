"""
Level 1: Equal-weight ensemble averaging.

The simplest combination — every model contributes equally.  Acts as
the baseline that all more complex methods must beat.
"""

from __future__ import annotations

import numpy as np


def equal_weight_predict(F: np.ndarray) -> np.ndarray:
    """Arithmetic mean across models.

    Args:
        F: (T, K) prediction matrix.

    Returns:
        (T,) point prediction.
    """
    return F.mean(axis=1)


def equal_weights(K: int) -> np.ndarray:
    """Return a uniform weight vector of length *K*."""
    return np.ones(K) / K
