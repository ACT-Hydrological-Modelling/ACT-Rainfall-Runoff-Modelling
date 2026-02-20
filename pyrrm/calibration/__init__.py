"""
Calibration framework for rainfall-runoff models.

Provides:
- CalibrationRunner: Unified interface for model calibration
- PyDREAM adapter: MT-DREAM(ZS) with multi-try sampling and snooker updates
- Direct SCE-UA: Standalone SCE-UA implementation (no external dependencies)
- SciPy adapter: Optimization methods from scipy.optimize
- Objective functions: NSE, KGE, RMSE, PBIAS, and custom objectives

SCE-UA (Direct):
    - No external dependencies (vendored implementation)
    - ThreadPoolExecutor parallelization
    - PCA recovery for lost dimensions
    - Multiple convergence criteria
    - Initial parameter sets support

PyDREAM (MT-DREAM(ZS)):
    - Multi-try sampling for better mixing
    - Snooker updates for mode jumping
    - Parallel tempering support
    - Better for multi-modal posteriors

Example:
    >>> from pyrrm.calibration import CalibrationRunner, NSE
    >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
    >>>
    >>> # Direct SCE-UA (recommended - no external dependencies)
    >>> result = runner.run_sceua_direct(max_evals=20000, seed=42)
    >>>
    >>> # PyDREAM (MT-DREAM(ZS)) with multi-try
    >>> result = runner.run_dream(
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
    TransformedGaussianLikelihood,
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

# Import checkpoint manager
from pyrrm.calibration.checkpoint import CheckpointManager, CheckpointInfo

# Import CalibrationReport for comprehensive result storage
from pyrrm.calibration.report import CalibrationReport

# Batch experiment runner
from pyrrm.calibration.batch import (
    ExperimentGrid,
    ExperimentSpec,
    BatchExperimentRunner,
    BatchResult,
    get_model_class,
)

# Direct SCE-UA implementation (always available - vendored, no external dependencies)
from pyrrm.calibration.sceua_adapter import (
    run_sceua_direct,
    calibrate_sceua,
    SCEUAModelWrapper,
    SCEUACalibrationResult,
)

__all__ = [
    # Core classes
    "CalibrationRunner",
    "CalibrationResult",
    "CalibrationReport",
    # Legacy objective functions
    "ObjectiveFunction",
    "NSE",
    "KGE",
    "RMSE",
    "MAE",
    "PBIAS",
    "LogNSE",
    "GaussianLikelihood",
    "TransformedGaussianLikelihood",
    "FDCError",
    "WeightedObjective",
    "FlowSignatureError",
    # Utility functions
    "calculate_metrics",
    "is_new_interface",
    "get_calibration_value",
    # Availability flags
    "PYDREAM_AVAILABLE",
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
    # Direct SCE-UA (always available - vendored)
    "run_sceua_direct",
    "calibrate_sceua",
    "SCEUAModelWrapper",
    "SCEUACalibrationResult",
    # Checkpointing
    "CheckpointManager",
    "CheckpointInfo",
    # Batch experiment runner
    "ExperimentGrid",
    "ExperimentSpec",
    "BatchExperimentRunner",
    "BatchResult",
    "get_model_class",
]
