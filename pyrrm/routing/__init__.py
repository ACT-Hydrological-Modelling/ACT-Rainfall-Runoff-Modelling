"""
Hydrograph routing module for pyrrm.

This module provides channel routing methods to transform rainfall-runoff
model output into routed hydrographs that account for travel time and
attenuation through river reaches.

Routing is useful when:
- The gauge (observation point) is located downstream of the catchment centroid
- The modeled hydrograph arrives too early and has too sharp a peak
- You want to separate runoff generation from channel routing in calibration

Available Classes:
    BaseRouter: Abstract base class for routing methods
    NonlinearMuskingumRouter: Nonlinear Muskingum (S = K*Q^m) routing
    RoutedModel: Wrapper combining RR model with optional routing

Example:
    >>> from pyrrm.models import Sacramento
    >>> from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
    >>> 
    >>> # Create rainfall-runoff model
    >>> rr_model = Sacramento()
    >>> 
    >>> # Create router with initial parameters
    >>> router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    >>> 
    >>> # Combine into routed model
    >>> model = RoutedModel(rr_model, router)
    >>> 
    >>> # Run simulation (routing is applied automatically)
    >>> results = model.run(inputs)
    >>> 
    >>> # For calibration, routing parameters are automatically included
    >>> from pyrrm.calibration import CalibrationRunner
    >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
    >>> result = runner.run_differential_evolution()
"""

from pyrrm.routing.base import BaseRouter
from pyrrm.routing.muskingum import NonlinearMuskingumRouter
from pyrrm.routing.routed_model import RoutedModel

__all__ = [
    'BaseRouter',
    'NonlinearMuskingumRouter',
    'RoutedModel',
    'create_router',
]


def create_router(method: str = 'nonlinear_muskingum', **kwargs) -> BaseRouter:
    """
    Factory function to create router instances.
    
    Args:
        method: Routing method name. Currently supported:
            - 'nonlinear_muskingum': NonlinearMuskingumRouter (default)
            - 'muskingum': Alias for nonlinear_muskingum
        **kwargs: Parameters passed to router constructor:
            - K: Storage constant [days]
            - m: Nonlinear exponent (1.0 = linear)
            - n_subreaches: Number of sub-reaches for numerical routing
            
    Returns:
        BaseRouter instance
        
    Example:
        >>> router = create_router('nonlinear_muskingum', K=5.0, m=0.8)
        >>> router = create_router('muskingum', K=3.0, m=0.7, n_subreaches=4)
        
    Raises:
        ValueError: If unknown routing method is specified
    """
    methods = {
        'nonlinear_muskingum': NonlinearMuskingumRouter,
        'muskingum': NonlinearMuskingumRouter,  # Alias
    }
    
    method_lower = method.lower()
    if method_lower not in methods:
        available = list(methods.keys())
        raise ValueError(
            f"Unknown routing method: '{method}'. "
            f"Available methods: {available}"
        )
    
    return methods[method_lower](**kwargs)
