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
    "gr4j_run_jax",
    "sacramento_run_jax",
    "JAX_AVAILABLE",
    "MLX_AVAILABLE",
    "NUMBA_AVAILABLE",
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
    elif name == "gr4j_run_jax":
        from pyrrm.models.gr4j_jax import gr4j_run_jax
        return gr4j_run_jax
    elif name == "sacramento_run_jax":
        from pyrrm.models.sacramento_jax import sacramento_run_jax
        return sacramento_run_jax
    elif name == "JAX_AVAILABLE":
        try:
            import jax
            return True
        except ImportError:
            return False
    elif name == "MLX_AVAILABLE":
        try:
            import mlx.core  # noqa: F401
            return True
        except ImportError:
            return False
    elif name == "NUMBA_AVAILABLE":
        try:
            from pyrrm.models.numba_kernels import NUMBA_AVAILABLE as _na
            return _na
        except ImportError:
            return False
    raise AttributeError(f"module 'pyrrm.models' has no attribute '{name}'")
