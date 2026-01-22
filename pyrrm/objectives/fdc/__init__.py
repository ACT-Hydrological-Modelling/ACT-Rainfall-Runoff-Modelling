"""
Flow Duration Curve based metrics and utilities.

Provides FDC computation and segment-based evaluation metrics.
"""

from pyrrm.objectives.fdc.curves import (
    compute_fdc,
    compute_fdc_at_exceedance,
    compute_fdc_slope,
    get_fdc_segment,
)
from pyrrm.objectives.fdc.metrics import FDCMetric

__all__ = [
    # Utilities
    'compute_fdc',
    'compute_fdc_at_exceedance',
    'compute_fdc_slope',
    'get_fdc_segment',
    # Metrics
    'FDCMetric',
]
