"""
Core components for hydrological objective functions.

This module provides the foundational classes and utilities:
- ObjectiveFunction: Abstract base class for all objectives
- MetricResult: Container for evaluation results
- Constants and utility functions
"""

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.result import MetricResult
from pyrrm.objectives.core.constants import (
    FDC_SEGMENTS,
    KGE_BENCHMARK,
    DEFAULT_EPSILON_FRACTION,
    TRANSFORM_EMPHASIS,
    SDEB_DEFAULT_ALPHA,
    SDEB_DEFAULT_LAMBDA,
    HIGH_FLOW_THRESHOLD_MULTIPLIER,
    LOW_FLOW_THRESHOLD_MULTIPLIER,
)
from pyrrm.objectives.core.utils import (
    evaluate_all,
    print_evaluation_report,
    calculate_metrics_summary,
    compare_simulations,
    rank_simulations,
)

__all__ = [
    # Base classes
    'ObjectiveFunction',
    'MetricResult',
    # Constants
    'FDC_SEGMENTS',
    'KGE_BENCHMARK',
    'DEFAULT_EPSILON_FRACTION',
    'TRANSFORM_EMPHASIS',
    'SDEB_DEFAULT_ALPHA',
    'SDEB_DEFAULT_LAMBDA',
    'HIGH_FLOW_THRESHOLD_MULTIPLIER',
    'LOW_FLOW_THRESHOLD_MULTIPLIER',
    # Utilities
    'evaluate_all',
    'print_evaluation_report',
    'calculate_metrics_summary',
    'compare_simulations',
    'rank_simulations',
]
