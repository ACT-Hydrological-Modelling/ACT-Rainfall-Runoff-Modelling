"""
Hydrological objective function metrics.

This module provides traditional metrics, KGE variants, correlation metrics,
and the APEX objective function.
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
from pyrrm.objectives.metrics.apex import APEX

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
    # APEX (SDEB-enhanced with dynamics/lag multipliers)
    'APEX',
]
