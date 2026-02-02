"""
Composite objective functions for multi-criteria calibration.

Provides WeightedObjective and factory functions for common configurations.
"""

from pyrrm.objectives.composite.weighted import WeightedObjective
from pyrrm.objectives.composite.factories import (
    kge_hilo,
    fdc_multisegment,
    comprehensive_objective,
    nse_multiscale,
    apex_objective,
)
from pyrrm.objectives.composite.adaptive import apex_adaptive

__all__ = [
    'WeightedObjective',
    'kge_hilo',
    'fdc_multisegment',
    'comprehensive_objective',
    'nse_multiscale',
    'apex_objective',
    'apex_adaptive',
]
