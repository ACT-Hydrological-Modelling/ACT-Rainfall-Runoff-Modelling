"""
Calibration framework for rainfall-runoff models.

Provides:
- CalibrationRunner: Unified interface for model calibration
- SPOTPY adapter: DREAM, SCE-UA, and other MCMC methods via SpotPy
- PyDREAM adapter: MT-DREAM(ZS) with multi-try sampling and snooker updates
- scipy adapter: Optimization methods from scipy.optimize
- Objective functions: NSE, KGE, RMSE, PBIAS, and custom objectives

DREAM Implementation Comparison:
    SpotPy DREAM:
        - Standard DREAM algorithm
        - Built-in convergence checking
        - Database storage options (CSV, RAM, SQL)
        
    PyDREAM (MT-DREAM(ZS)):
        - Multi-try sampling for better mixing
        - Snooker updates for mode jumping
        - Parallel tempering support
        - Better for multi-modal posteriors

Example:
    >>> from pyrrm.calibration import CalibrationRunner, NSE
    >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
    >>> 
    >>> # SpotPy DREAM
    >>> result = runner.run_dream(implementation='spotpy', n_iterations=10000)
    >>> 
    >>> # PyDREAM (MT-DREAM(ZS)) with multi-try
    >>> result = runner.run_dream(
    ...     implementation='pydream', 
    ...     n_iterations=10000,
    ...     multitry=5, 
    ...     snooker=0.1
    ... )
"""

from pyrrm.calibration.runner import CalibrationRunner, CalibrationResult
from pyrrm.calibration.objective_functions import (
    # Legacy objective functions (backward compatible)
    ObjectiveFunction,
    NSE,
    KGE,
    RMSE,
    MAE,
    PBIAS,
    LogNSE,
    GaussianLikelihood,
    FDCError,
    WeightedObjective,
    FlowSignatureError,
    # Utility functions
    calculate_metrics,
    is_new_interface,
    get_calibration_value,
)

# Import new objective functions from pyrrm.objectives if available
try:
    from pyrrm.calibration.objective_functions import (
        NEW_OBJECTIVES_AVAILABLE,
        # New metrics
        SDEB,
        KGENonParametric,
        PearsonCorrelation,
        SpearmanCorrelation,
        FDCMetric,
        SignatureMetric,
        # Utilities
        FlowTransformation,
        MetricResult,
        compute_fdc,
        # Composite factories
        kge_hilo,
        fdc_multisegment,
        comprehensive_objective,
        # Compatibility
        LegacyObjectiveAdapter,
        adapt_objective,
    )
except ImportError:
    NEW_OBJECTIVES_AVAILABLE = False

# Check for optional PyDREAM dependency
try:
    from pyrrm.calibration.pydream_adapter import (
        run_pydream,
        create_pydream_likelihood,
        create_pydream_parameters,
        continue_pydream,
        check_pydream_convergence,
        PYDREAM_AVAILABLE,
    )
except ImportError:
    PYDREAM_AVAILABLE = False
    run_pydream = None
    create_pydream_likelihood = None
    create_pydream_parameters = None
    continue_pydream = None
    check_pydream_convergence = None

# Check for optional SpotPy dependency
try:
    from pyrrm.calibration.spotpy_adapter import (
        SPOTPYModelSetup,
        run_dream as run_spotpy_dream,
        run_sceua,
        run_mcmc,
        continue_spotpy_dream,
        SPOTPY_AVAILABLE,
    )
except ImportError:
    SPOTPY_AVAILABLE = False
    SPOTPYModelSetup = None
    run_spotpy_dream = None
    run_sceua = None
    run_mcmc = None
    continue_spotpy_dream = None

# Import checkpoint manager
from pyrrm.calibration.checkpoint import CheckpointManager, CheckpointInfo

__all__ = [
    # Core classes
    "CalibrationRunner",
    "CalibrationResult",
    # Legacy objective functions
    "ObjectiveFunction",
    "NSE",
    "KGE",
    "RMSE",
    "MAE",
    "PBIAS",
    "LogNSE",
    "FDCError",
    "WeightedObjective",
    "FlowSignatureError",
    # Utility functions
    "calculate_metrics",
    "is_new_interface",
    "get_calibration_value",
    # Availability flags
    "PYDREAM_AVAILABLE",
    "SPOTPY_AVAILABLE",
    "NEW_OBJECTIVES_AVAILABLE",
    # New objective functions (if pyrrm.objectives available)
    "SDEB",
    "KGENonParametric",
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "FDCMetric",
    "SignatureMetric",
    "FlowTransformation",
    "MetricResult",
    "compute_fdc",
    "kge_hilo",
    "fdc_multisegment",
    "comprehensive_objective",
    "LegacyObjectiveAdapter",
    "adapt_objective",
    # PyDREAM functions (if available)
    "run_pydream",
    "create_pydream_likelihood",
    "create_pydream_parameters",
    "continue_pydream",
    "check_pydream_convergence",
    # SpotPy functions (if available)
    "SPOTPYModelSetup",
    "run_spotpy_dream",
    "run_sceua",
    "run_mcmc",
    "continue_spotpy_dream",
    # Checkpointing
    "CheckpointManager",
    "CheckpointInfo",
]
