"""
Legacy compatibility layer for old-style objective functions.

This module provides adapters to bridge between the old objective function
interface (from pyrrm.calibration.objective_functions) and the new interface.
"""

from typing import Optional, Dict
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction


class LegacyObjectiveAdapter(ObjectiveFunction):
    """
    Adapter to wrap old-style objective functions for the new interface.
    
    The old interface uses:
    - calculate(simulated, observed) method
    - maximize property
    
    The new interface uses:
    - __call__(obs, sim) method (note: argument order reversed)
    - direction property ('maximize' or 'minimize')
    
    Parameters
    ----------
    legacy_objective : object
        Old-style objective function with calculate() and maximize
    
    Examples
    --------
    >>> # Wrap an old-style objective
    >>> from pyrrm.calibration.objective_functions import NSE as OldNSE
    >>> old_nse = OldNSE()
    >>> new_nse = LegacyObjectiveAdapter(old_nse)
    >>> value = new_nse(obs, sim)  # Uses new interface
    """
    
    def __init__(self, legacy_objective):
        # Determine direction from maximize property
        if hasattr(legacy_objective, 'maximize'):
            direction = 'maximize' if legacy_objective.maximize else 'minimize'
        else:
            direction = 'maximize'  # Default assumption
        
        # Get name
        if hasattr(legacy_objective, 'name'):
            name = legacy_objective.name
        else:
            name = legacy_objective.__class__.__name__
        
        # Determine optimal value
        if direction == 'maximize':
            optimal_value = 1.0
        else:
            optimal_value = 0.0
        
        super().__init__(name=name, direction=direction, optimal_value=optimal_value)
        self._legacy = legacy_objective
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        """
        Calculate using the legacy interface.
        
        Note: Legacy interface is calculate(simulated, observed),
        while new interface is __call__(obs, sim).
        """
        return self._legacy.calculate(sim, obs)
    
    def for_calibration(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """Bridge to legacy calibration value method."""
        if hasattr(self._legacy, 'for_calibration_legacy'):
            return self._legacy.for_calibration_legacy(simulated, observed)
        if hasattr(self._legacy, 'for_spotpy'):
            return self._legacy.for_spotpy(simulated, observed)
        
        value = self._legacy.calculate(simulated, observed)
        return value if self.direction == 'maximize' else -value


def wrap_legacy_objective(legacy_objective) -> ObjectiveFunction:
    """
    Convenience function to wrap a legacy objective.
    
    Parameters
    ----------
    legacy_objective : object
        Old-style objective function
    
    Returns
    -------
    ObjectiveFunction
        Wrapped objective using new interface
    """
    return LegacyObjectiveAdapter(legacy_objective)


def is_legacy_objective(obj) -> bool:
    """
    Check if an object is a legacy-style objective function.
    
    Parameters
    ----------
    obj : object
        Object to check
    
    Returns
    -------
    bool
        True if obj uses the legacy interface
    """
    # New interface has direction attribute
    if hasattr(obj, 'direction'):
        return False
    
    # Legacy interface has maximize property and calculate method
    return hasattr(obj, 'maximize') and hasattr(obj, 'calculate')


def adapt_objective(obj) -> ObjectiveFunction:
    """
    Automatically adapt an objective to the new interface.
    
    If already using new interface, returns as-is.
    If using legacy interface, wraps with adapter.
    
    Parameters
    ----------
    obj : ObjectiveFunction or legacy objective
        Objective function to adapt
    
    Returns
    -------
    ObjectiveFunction
        Objective using new interface
    """
    if is_legacy_objective(obj):
        return LegacyObjectiveAdapter(obj)
    return obj
