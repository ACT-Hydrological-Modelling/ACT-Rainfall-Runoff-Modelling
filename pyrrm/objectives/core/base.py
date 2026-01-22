"""
Abstract base class for objective functions.

This module provides the ObjectiveFunction abstract base class that all
objective functions must inherit from.

References
----------
Design follows composability and immutability principles for combining
multiple objectives in calibration workflows.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyrrm.objectives.core.result import MetricResult


class ObjectiveFunction(ABC):
    """
    Abstract base class for all objective functions.
    
    All concrete objective functions must inherit from this class
    and implement the __call__ method.
    
    Attributes
    ----------
    name : str
        Human-readable name for the objective function
    direction : str
        Optimization direction: 'maximize' or 'minimize'
    optimal_value : float
        The optimal (best possible) value for this metric
    
    Methods
    -------
    __call__(obs, sim, **kwargs) -> float
        Calculate the objective function value (abstract)
    evaluate(obs, sim, **kwargs) -> MetricResult
        Calculate value and return with components
    get_components(obs, sim, **kwargs) -> Optional[Dict[str, float]]
        Return component breakdown if applicable
    for_calibration(simulated, observed) -> float
        Return value suitable for optimization (always maximize)
    
    Examples
    --------
    >>> class MyMetric(ObjectiveFunction):
    ...     def __init__(self):
    ...         super().__init__(name='MyMetric', direction='maximize', optimal_value=1.0)
    ...     def __call__(self, obs, sim, **kwargs):
    ...         return 1.0 - np.mean(np.abs(obs - sim))
    >>> metric = MyMetric()
    >>> value = metric(observed, simulated)
    """
    
    def __init__(self,
                 name: str,
                 direction: str = 'maximize',
                 optimal_value: float = 1.0):
        """
        Initialize the objective function.
        
        Parameters
        ----------
        name : str
            Descriptive name for the objective function
        direction : str
            'maximize' or 'minimize'
        optimal_value : float
            The optimal (best) value for this metric
            
        Raises
        ------
        ValueError
            If direction is not 'maximize' or 'minimize'
        """
        if direction not in ('maximize', 'minimize'):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{direction}'")
        
        self.name = name
        self.direction = direction
        self.optimal_value = optimal_value
    
    @abstractmethod
    def __call__(self, 
                 obs: np.ndarray, 
                 sim: np.ndarray,
                 **kwargs) -> float:
        """
        Calculate the objective function value.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values (1D array)
        sim : np.ndarray
            Simulated values (1D array, same length as obs)
        **kwargs : dict
            Additional parameters specific to the metric
            
        Returns
        -------
        float
            Objective function value
            
        Raises
        ------
        ValueError
            If obs and sim have different lengths
        """
        pass
    
    def evaluate(self, 
                 obs: np.ndarray, 
                 sim: np.ndarray, 
                 **kwargs) -> 'MetricResult':
        """
        Evaluate and return a MetricResult with components.
        
        This method calls __call__ and get_components, packaging
        the results into a MetricResult object.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result object containing value and component breakdown
        """
        from pyrrm.objectives.core.result import MetricResult
        
        value = self(obs, sim, **kwargs)
        components = self.get_components(obs, sim, **kwargs) or {}
        return MetricResult(value=value, components=components, name=self.name)
    
    def get_components(self, 
                       obs: np.ndarray, 
                       sim: np.ndarray,
                       **kwargs) -> Optional[Dict[str, float]]:
        """
        Return component breakdown (if applicable).
        
        Override in subclasses for multi-component metrics like KGE.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        dict or None
            Dictionary of component names and values, or None
        """
        return None
    
    def _validate_inputs(self, obs: np.ndarray, sim: np.ndarray) -> None:
        """
        Validate input arrays.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
            
        Raises
        ------
        ValueError
            If inputs are invalid
        TypeError
            If inputs are not array-like
        """
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        if not isinstance(sim, np.ndarray):
            sim = np.asarray(sim)
        
        if obs.ndim != 1 or sim.ndim != 1:
            raise ValueError("obs and sim must be 1-dimensional arrays")
        
        if len(obs) != len(sim):
            raise ValueError(f"obs and sim must have same length, got {len(obs)} and {len(sim)}")
        
        if len(obs) == 0:
            raise ValueError("obs and sim cannot be empty")
    
    def _clean_data(self, 
                    obs: np.ndarray, 
                    sim: np.ndarray) -> tuple:
        """
        Remove NaN values using pairwise deletion.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
            
        Returns
        -------
        tuple of np.ndarray
            Cleaned obs and sim arrays with NaN pairs removed
            
        Raises
        ------
        ValueError
            If all values are NaN
        """
        obs = np.asarray(obs).flatten()
        sim = np.asarray(sim).flatten()
        
        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs_clean = obs[mask]
        sim_clean = sim[mask]
        
        if len(obs_clean) == 0:
            raise ValueError("No valid (non-NaN) data pairs")
        
        return obs_clean, sim_clean
    
    def for_calibration(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """
        Return value suitable for optimization (always maximize).
        
        This method bridges the new objective function interface with
        the CalibrationRunner, which expects values to maximize.
        
        Note: The argument order is (simulated, observed) to match
        the existing calibration interface convention.
        
        Parameters
        ----------
        simulated : np.ndarray
            Simulated values
        observed : np.ndarray
            Observed values
            
        Returns
        -------
        float
            Value suitable for maximization
        """
        value = self(observed, simulated)
        return value if self.direction == 'maximize' else -value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self) -> str:
        return self.name
