"""
Base classes and interfaces for hydrograph routing methods.

This module defines the abstract base class that all routing methods
in pyrrm must inherit from, ensuring a consistent interface across
different routing algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np


class BaseRouter(ABC):
    """
    Abstract base class for all hydrograph routing methods.
    
    This class defines the interface that all routers must implement,
    ensuring consistency across different routing algorithms (Muskingum,
    Muskingum-Cunge, lag-route, etc.).
    
    Routing transforms an inflow hydrograph into a routed outflow hydrograph
    by accounting for:
    - Translation (lag): Delays the hydrograph by travel time
    - Attenuation: Reduces and spreads the peak due to storage effects
    
    All routing parameters use the 'routing_' prefix to distinguish them
    from rainfall-runoff model parameters during calibration.
    
    Subclasses must implement:
        - route(): Core routing algorithm
        - get_parameter_bounds(): Return calibratable parameter bounds
        - set_parameters(): Set parameter values
        - get_parameters(): Get current parameter values
        - reset(): Reset internal state
    """
    
    @abstractmethod
    def route(
        self,
        inflow: np.ndarray,
        dt: float,
        initial_outflow: Optional[float] = None
    ) -> np.ndarray:
        """
        Route an inflow hydrograph through the reach.
        
        This is the core routing method that transforms input flows
        to output flows accounting for storage and travel time effects.
        
        Args:
            inflow: Inflow time series [L³/T, e.g., ML/day or m³/s].
                   Shape: (n_timesteps,)
            dt: Timestep duration [T]. Must be in same units as the
                storage constant K (e.g., days if K is in days).
            initial_outflow: Initial outflow at t=0 [L³/T]. If None,
                           assumes steady-state (outflow = first inflow).
                           
        Returns:
            Routed outflow time series [L³/T]. Shape: (n_timesteps,)
            
        Raises:
            ValueError: If inflow contains negative values or if
                       parameters are invalid for the given inputs.
                       
        Note:
            The routing should conserve mass: total inflow volume should
            equal total outflow volume plus any change in storage.
        """
        pass
    
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Return routing parameter bounds for calibration.
        
        All parameter names should be prefixed with 'routing_' to
        distinguish them from rainfall-runoff model parameters.
        
        Returns:
            Dictionary mapping parameter names to (min, max) bounds.
            
        Example:
            >>> router.get_parameter_bounds()
            {'routing_K': (0.1, 200.0), 'routing_m': (0.3, 1.5)}
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Set routing parameter values.
        
        Accepts parameters with 'routing_' prefix (as used in calibration)
        or without the prefix for convenience.
        
        Args:
            params: Dictionary of parameter names to values.
                   Can use prefixed names ('routing_K') or unprefixed ('K').
                   
        Raises:
            ValueError: If parameter values are outside valid bounds.
            
        Example:
            >>> router.set_parameters({'routing_K': 5.0, 'routing_m': 0.8})
            >>> # Or without prefix:
            >>> router.set_parameters({'K': 5.0, 'm': 0.8})
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current routing parameter values.
        
        Returns parameters with 'routing_' prefix for consistency
        with the calibration framework.
        
        Returns:
            Dictionary mapping parameter names to current values.
            
        Example:
            >>> router.get_parameters()
            {'routing_K': 5.0, 'routing_m': 0.8, 'routing_n_subreaches': 3}
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset router to initial state.
        
        Clears any internal state variables (e.g., stored outflows
        from previous routing). Called before each new simulation
        to ensure independent runs.
        """
        pass
    
    def calculate_storage(self, outflow: float) -> float:
        """
        Calculate storage volume for a given outflow.
        
        The storage-discharge relationship is method-specific.
        Default implementation returns 0 (stateless routing).
        
        Args:
            outflow: Discharge rate [L³/T]
            
        Returns:
            Storage volume [L³]
        """
        return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current router state for saving/restoration.
        
        Returns:
            Dictionary containing internal state variables.
            Default implementation returns empty dict.
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore router state from saved state.
        
        Args:
            state: Dictionary from previous get_state() call.
            Default implementation does nothing.
        """
        pass
    
    def verify_mass_balance(
        self,
        inflow: np.ndarray,
        outflow: np.ndarray,
        dt: float,
        tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Verify mass conservation in routing results.
        
        Checks that total inflow volume equals total outflow volume
        plus any change in storage, within a specified tolerance.
        
        Args:
            inflow: Input hydrograph [L³/T]
            outflow: Routed output hydrograph [L³/T]
            dt: Timestep duration [T]
            tolerance: Acceptable relative error (default 1%)
            
        Returns:
            Tuple of (is_balanced, relative_error):
            - is_balanced: True if error is within tolerance
            - relative_error: Actual relative mass balance error
        """
        total_inflow = np.sum(inflow) * dt
        total_outflow = np.sum(outflow) * dt
        
        # Storage change (using first and last values)
        S_initial = self.calculate_storage(inflow[0])
        S_final = self.calculate_storage(outflow[-1])
        storage_change = S_final - S_initial
        
        # Mass balance check
        if total_inflow == 0:
            return total_outflow == 0, 0.0
        
        mass_error = abs(total_inflow - total_outflow - storage_change)
        relative_error = mass_error / total_inflow
        
        return relative_error < tolerance, relative_error
    
    def __repr__(self) -> str:
        """Return string representation of router."""
        params = self.get_parameters()
        param_str = ', '.join(f"{k.replace('routing_', '')}={v}" 
                              for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
