"""
pyrrm.objectives - Hydrological Objective Functions Library

A comprehensive library for constructing and evaluating objective functions
for rainfall-runoff model calibration.

Core Components
---------------
ObjectiveFunction : Abstract base class for all objectives
MetricResult : Container for evaluation results with component breakdown
FlowTransformation : Apply flow transformations to shift high/low flow emphasis

Traditional Metrics
-------------------
NSE : Nash-Sutcliffe Efficiency
RMSE : Root Mean Square Error  
MAE : Mean Absolute Error
PBIAS : Percent Bias
SDEB : Sum of Daily Flows, Daily Exceedance Curve and Bias

KGE Family
----------
KGE : Kling-Gupta Efficiency (2009, 2012, 2021 variants)
KGENonParametric : Non-parametric KGE using Spearman correlation

Correlation Metrics
-------------------
PearsonCorrelation : Linear correlation coefficient
SpearmanCorrelation : Rank-based correlation coefficient

FDC Metrics
-----------
FDCMetric : Flow Duration Curve based metrics
compute_fdc : Compute flow duration curve

Signature Metrics
-----------------
SignatureMetric : Hydrological signature-based evaluation

Composite Objectives
--------------------
WeightedObjective : Combine multiple objectives with weights
kge_hilo : Factory for KGE + KGE(inverse) combination
fdc_multisegment : Factory for multi-segment FDC objectives
comprehensive_objective : Factory for multi-metric combination

Utilities
---------
evaluate_all : Evaluate multiple objectives at once
print_evaluation_report : Generate formatted performance report

Examples
--------
>>> from pyrrm.objectives import NSE, KGE, PBIAS, FlowTransformation
>>> from pyrrm.objectives import evaluate_all, print_evaluation_report

>>> # Single metric evaluation
>>> nse = NSE()
>>> value = nse(observed, simulated)

>>> # Metric with flow transformation for low-flow emphasis
>>> kge_inv = KGE(transform=FlowTransformation('inverse'))
>>> value = kge_inv(observed, simulated)

>>> # Comprehensive evaluation
>>> results = evaluate_all(observed, simulated)
>>> print_evaluation_report(observed, simulated)

>>> # Composite objective for calibration
>>> from pyrrm.objectives import kge_hilo
>>> objective = kge_hilo(kge_weight=0.5)
>>> value = objective(observed, simulated)

References
----------
Nash, J.E., Sutcliffe, J.V. (1970). River flow forecasting through 
conceptual models part I - A discussion of principles.

Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). Decomposition 
of the mean squared error and NSE performance criteria.

Kling, H., Fuchs, M., Paulin, M. (2012). Runoff conditions in the upper 
Danube basin under an ensemble of climate change scenarios.

Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
A robust approach for calibrating continuous hydrological models.
"""

# Core components
from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.result import MetricResult
from pyrrm.objectives.core.constants import (
    FDC_SEGMENTS,
    KGE_BENCHMARK,
    DEFAULT_EPSILON_FRACTION,
    TRANSFORM_EMPHASIS,
)
from pyrrm.objectives.core.utils import (
    evaluate_all,
    print_evaluation_report,
    calculate_metrics_summary,
    compare_simulations,
    rank_simulations,
)

# Flow transformations
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation

# Traditional metrics
from pyrrm.objectives.metrics.traditional import (
    NSE,
    RMSE,
    MAE,
    PBIAS,
    SDEB,
)

# KGE family
from pyrrm.objectives.metrics.kge import (
    KGE,
    KGENonParametric,
)

# Correlation metrics
from pyrrm.objectives.metrics.correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
)

# FDC metrics
from pyrrm.objectives.fdc.curves import compute_fdc
from pyrrm.objectives.fdc.metrics import FDCMetric

# Signature metrics
from pyrrm.objectives.signatures.flow_indices import SignatureMetric

# Composite objectives
from pyrrm.objectives.composite.weighted import WeightedObjective
from pyrrm.objectives.composite.factories import (
    kge_hilo,
    fdc_multisegment,
    comprehensive_objective,
    nse_multiscale,
    apex_objective,
    apex_adaptive,
)

# Compatibility layer
from pyrrm.objectives.compat.legacy import (
    LegacyObjectiveAdapter,
    wrap_legacy_objective,
    adapt_objective,
)

__version__ = '0.1.0'

__all__ = [
    # Core
    'ObjectiveFunction',
    'MetricResult',
    'FDC_SEGMENTS',
    'KGE_BENCHMARK',
    'DEFAULT_EPSILON_FRACTION',
    'TRANSFORM_EMPHASIS',
    # Utilities
    'evaluate_all',
    'print_evaluation_report',
    'calculate_metrics_summary',
    'compare_simulations',
    'rank_simulations',
    # Transformations
    'FlowTransformation',
    # Traditional metrics
    'NSE',
    'RMSE',
    'MAE',
    'PBIAS',
    'SDEB',
    # KGE family
    'KGE',
    'KGENonParametric',
    # Correlation
    'PearsonCorrelation',
    'SpearmanCorrelation',
    # FDC
    'compute_fdc',
    'FDCMetric',
    # Signatures
    'SignatureMetric',
    # Composite
    'WeightedObjective',
    'kge_hilo',
    'fdc_multisegment',
    'comprehensive_objective',
    'nse_multiscale',
    'apex_objective',
    'apex_adaptive',
    # Compatibility
    'LegacyObjectiveAdapter',
    'wrap_legacy_objective',
    'adapt_objective',
]
