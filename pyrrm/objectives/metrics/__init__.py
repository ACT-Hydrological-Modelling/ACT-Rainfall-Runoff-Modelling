"""
Hydrological objective function metrics.

This module provides traditional metrics, KGE variants, and correlation metrics.
"""

from pyrrm.objectives.metrics.traditional import (
    NSE,
    RMSE,
    MAE,
    PBIAS,
    SDEB,
)
from pyrrm.objectives.metrics.kge import (
    KGE,
    KGENonParametric,
)
from pyrrm.objectives.metrics.correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
)

__all__ = [
    # Traditional metrics
    'NSE',
    'RMSE',
    'MAE',
    'PBIAS',
    'SDEB',
    # KGE family
    'KGE',
    'KGENonParametric',
    # Correlation
    'PearsonCorrelation',
    'SpearmanCorrelation',
]
