"""
Hydrological signature metrics and utilities.

Provides signature-based objective functions and calculation utilities.
"""

from pyrrm.objectives.signatures.flow_indices import SignatureMetric
from pyrrm.objectives.signatures.dynamics import (
    compute_flashiness_index,
    compute_rising_limb_density,
    compute_falling_limb_density,
    extract_recession_segments,
    compute_recession_constant,
)
from pyrrm.objectives.signatures.water_balance import (
    compute_runoff_ratio,
    compute_baseflow_index,
    compute_baseflow_recession_constant,
    compute_streamflow_elasticity,
)

__all__ = [
    # Metric class
    'SignatureMetric',
    # Dynamics functions
    'compute_flashiness_index',
    'compute_rising_limb_density',
    'compute_falling_limb_density',
    'extract_recession_segments',
    'compute_recession_constant',
    # Water balance functions
    'compute_runoff_ratio',
    'compute_baseflow_index',
    'compute_baseflow_recession_constant',
    'compute_streamflow_elasticity',
]
