"""
Rainfall-runoff models module.

This module contains implementations of various conceptual rainfall-runoff models:
- Sacramento: Complex conceptual model with multiple zones
- GR4J: 4-parameter daily model (INRAE)
- GR5J: 5-parameter daily model (INRAE)
- GR6J: 6-parameter daily model (INRAE)
"""

from pyrrm.models.base import BaseRainfallRunoffModel, ModelParameter, ModelState

__all__ = [
    "BaseRainfallRunoffModel",
    "ModelParameter", 
    "ModelState",
    "Sacramento",
    "GR4J",
    "GR5J",
    "GR6J",
]

# Lazy imports
def __getattr__(name):
    if name == "Sacramento":
        from pyrrm.models.sacramento import Sacramento
        return Sacramento
    elif name == "GR4J":
        from pyrrm.models.gr4j import GR4J
        return GR4J
    elif name == "GR5J":
        from pyrrm.models.gr5j import GR5J
        return GR5J
    elif name == "GR6J":
        from pyrrm.models.gr6j import GR6J
        return GR6J
    raise AttributeError(f"module 'pyrrm.models' has no attribute '{name}'")
