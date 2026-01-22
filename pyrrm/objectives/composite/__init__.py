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
)

__all__ = [
    'WeightedObjective',
    'kge_hilo',
    'fdc_multisegment',
    'comprehensive_objective',
    'nse_multiscale',
]
