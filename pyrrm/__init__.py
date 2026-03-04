"""
pyrrm: Python Rainfall-Runoff Models

A comprehensive Python library for rainfall-runoff modeling, including:
- Multiple conceptual models (Sacramento, GR4J, GR5J, GR6J)
- Channel routing (Nonlinear Muskingum method)
- Flexible calibration framework (PyDREAM, SCE-UA, SciPy)
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
    >>> result = runner.run_sceua_direct(max_evals=20000, seed=42)
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
    # Batch experiment runner
    elif name == "BatchExperimentRunner":
        from pyrrm.calibration.batch import BatchExperimentRunner
        return BatchExperimentRunner
    elif name == "ExperimentGrid":
        from pyrrm.calibration.batch import ExperimentGrid
        return ExperimentGrid
    elif name == "BatchResult":
        from pyrrm.calibration.batch import BatchResult
        return BatchResult
    # Network
    elif name == "CatchmentNetwork":
        from pyrrm.network.topology import CatchmentNetwork
        return CatchmentNetwork
    elif name == "CatchmentNetworkRunner":
        from pyrrm.network.runner import CatchmentNetworkRunner
        return CatchmentNetworkRunner
    elif name == "NetworkCalibrationResult":
        from pyrrm.network.runner import NetworkCalibrationResult
        return NetworkCalibrationResult
    elif name == "NetworkDataLoader":
        from pyrrm.network.data import NetworkDataLoader
        return NetworkDataLoader
    # Parallel backend
    elif name == "create_backend":
        from pyrrm.parallel import create_backend
        return create_backend
    # BMA
    elif name == "BMAConfig":
        from pyrrm.bma.config import BMAConfig
        return BMAConfig
    elif name == "BMARunner":
        from pyrrm.bma.pipeline import BMARunner
        return BMARunner
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
    # Batch experiment runner
    "BatchExperimentRunner",
    "ExperimentGrid",
    "BatchResult",
    # Network
    "CatchmentNetwork",
    "CatchmentNetworkRunner",
    "NetworkCalibrationResult",
    "NetworkDataLoader",
    # Parallel backend
    "create_backend",
    # BMA
    "BMAConfig",
    "BMARunner",
    # Version
    "__version__",
]
