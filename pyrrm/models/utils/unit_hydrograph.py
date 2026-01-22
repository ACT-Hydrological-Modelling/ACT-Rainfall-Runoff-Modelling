"""
Unit Hydrograph implementations for rainfall-runoff models.

This module provides unit hydrograph classes used for routing
surface runoff in various models.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class UnitHydrograph:
    """
    A discrete convolution unit hydrograph for routing surface runoff.
    
    The unit hydrograph distributes input flows across future time steps
    based on specified proportions (s_curve), then releases accumulated
    flows each time step.
    
    This implementation is used by the Sacramento model.
    
    Attributes:
        s_curve: List of proportions for each lag period
        _stores: Internal storage for delayed flows
    """
    s_curve: List[float] = field(default_factory=list)
    _stores: List[float] = field(default_factory=list)
    
    def initialise_hydrograph(self, proportions: List[float]) -> None:
        """
        Initialize the unit hydrograph with the given proportions.
        
        Args:
            proportions: List of proportions (should sum to 1.0)
        """
        self.s_curve = list(proportions)
        self._stores = [0.0] * len(proportions)
    
    def run_time_step(self, input_value: float) -> float:
        """
        Process one time step of the unit hydrograph.
        
        Args:
            input_value: Input flow to be routed
            
        Returns:
            Output flow for this time step
        """
        if not self.s_curve:
            return input_value
        
        # Add input distributed across stores according to s_curve
        for i in range(len(self._stores)):
            self._stores[i] += input_value * self.s_curve[i]
        
        # Output is the first store
        output = self._stores[0]
        
        # Shift stores (cascade down)
        for i in range(len(self._stores) - 1):
            self._stores[i] = self._stores[i + 1]
        self._stores[-1] = 0.0
        
        return output
    
    def reset(self) -> None:
        """Reset all stores to zero."""
        self._stores = [0.0] * len(self.s_curve) if self.s_curve else []
    
    def proportion_for_item(self, i: int) -> float:
        """
        Get the proportion for a specific lag index.
        
        Args:
            i: Index (0-based)
            
        Returns:
            Proportion value, or 0.0 if index is out of range
        """
        if 0 <= i < len(self.s_curve):
            return self.s_curve[i]
        return 0.0
    
    def get_stores(self) -> List[float]:
        """Get copy of current store values."""
        return list(self._stores)
    
    def set_stores(self, stores: List[float]) -> None:
        """Set store values."""
        if len(stores) == len(self._stores):
            self._stores = list(stores)
        else:
            raise ValueError(
                f"Store length mismatch: expected {len(self._stores)}, got {len(stores)}"
            )


class GRUnitHydrograph:
    """
    Unit hydrograph implementation for GR models.
    
    This class manages the convolution stores for the GR family
    of models (GR4J, GR5J, GR6J).
    
    Attributes:
        ordinates: Array of UH ordinates
        stores: Array of convolution stores
    """
    
    def __init__(self, ordinates: Optional[np.ndarray] = None):
        """
        Initialize the GR unit hydrograph.
        
        Args:
            ordinates: Array of UH ordinates (if None, must call initialize later)
        """
        if ordinates is not None:
            self.initialize(ordinates)
        else:
            self.ordinates = np.array([])
            self.stores = np.array([])
    
    def initialize(self, ordinates: np.ndarray) -> None:
        """
        Initialize the unit hydrograph with given ordinates.
        
        Args:
            ordinates: Array of UH ordinates
        """
        self.ordinates = np.asarray(ordinates, dtype=float)
        self.stores = np.zeros(len(self.ordinates), dtype=float)
    
    def run_timestep(self, input_value: float) -> float:
        """
        Process one time step.
        
        Args:
            input_value: Input flow to route
            
        Returns:
            Output flow for this timestep
        """
        if len(self.ordinates) == 0:
            return input_value
        
        n = len(self.ordinates)
        
        # Convolve: shift stores and add new input
        for i in range(n - 1):
            self.stores[i] = self.stores[i + 1] + self.ordinates[i] * input_value
        self.stores[n - 1] = self.ordinates[n - 1] * input_value
        
        return self.stores[0]
    
    def reset(self) -> None:
        """Reset stores to zero."""
        self.stores = np.zeros(len(self.ordinates), dtype=float)
    
    def get_stores(self) -> np.ndarray:
        """Get copy of current stores."""
        return self.stores.copy()
    
    def set_stores(self, stores: np.ndarray) -> None:
        """Set store values."""
        stores = np.asarray(stores, dtype=float)
        if len(stores) == len(self.stores):
            self.stores = stores.copy()
        else:
            raise ValueError(
                f"Store length mismatch: expected {len(self.stores)}, got {len(stores)}"
            )
