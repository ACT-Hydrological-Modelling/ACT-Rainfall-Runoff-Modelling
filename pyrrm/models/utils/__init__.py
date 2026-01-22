"""
Utility functions for rainfall-runoff models.

Contains shared functionality used by multiple models:
- S-curve functions for unit hydrograph generation
- Unit hydrograph implementations
"""

from pyrrm.models.utils.s_curves import s_curve1, s_curve2
from pyrrm.models.utils.unit_hydrograph import UnitHydrograph

__all__ = [
    "s_curve1",
    "s_curve2", 
    "UnitHydrograph",
]
