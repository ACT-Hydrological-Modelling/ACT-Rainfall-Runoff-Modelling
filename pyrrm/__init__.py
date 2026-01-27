"""
pyrrm: Python Rainfall-Runoff Models

A comprehensive Python library for rainfall-runoff modeling, including:
- Multiple conceptual models (Sacramento, GR4J, GR5J, GR6J)
- Channel routing (Nonlinear Muskingum method)
- Flexible calibration framework (SPOTPY DREAM, SCE-UA, scipy)
- Sensitivity analysis (Sobol method)
- Visualization tools

Example:
    >>> from pyrrm.models import Sacramento, GR4J
    >>> from pyrrm.calibration import CalibrationRunner
    >>> from pyrrm.calibration.objective_functions import NSE
    >>> 
    >>> # Create and run a model
    >>> model = GR4J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7})
    >>> results = model.run(input_data)
    >>>
    >>> # Calibrate a model
    >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
    >>> result = runner.run_dream(n_iterations=10000)
    >>>
    >>> # With channel routing
    >>> from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
    >>> router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    >>> routed_model = RoutedModel(model, router)
    >>> results = routed_model.run(input_data)  # Routing applied
"""

__version__ = "0.1.0"
__author__ = "ACT Government"

# Lazy imports to avoid circular dependencies and improve import time
def __getattr__(name):
    # Models
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
    # Calibration
    elif name == "CalibrationRunner":
        from pyrrm.calibration.runner import CalibrationRunner
        return CalibrationRunner
    # Data
    elif name == "InputDataHandler":
        from pyrrm.data.input_handler import InputDataHandler
        return InputDataHandler
    # Routing
    elif name == "BaseRouter":
        from pyrrm.routing.base import BaseRouter
        return BaseRouter
    elif name == "NonlinearMuskingumRouter":
        from pyrrm.routing.muskingum import NonlinearMuskingumRouter
        return NonlinearMuskingumRouter
    elif name == "RoutedModel":
        from pyrrm.routing.routed_model import RoutedModel
        return RoutedModel
    elif name == "create_router":
        from pyrrm.routing import create_router
        return create_router
    raise AttributeError(f"module 'pyrrm' has no attribute '{name}'")

__all__ = [
    # Models
    "Sacramento",
    "GR4J", 
    "GR5J",
    "GR6J",
    # Routing
    "BaseRouter",
    "NonlinearMuskingumRouter",
    "RoutedModel",
    "create_router",
    # Calibration
    "CalibrationRunner",
    # Data
    "InputDataHandler",
    # Version
    "__version__",
]
