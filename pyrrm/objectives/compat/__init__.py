"""
Backward compatibility utilities for legacy objective functions.

Provides adapters for bridging old and new objective function interfaces.
"""

from pyrrm.objectives.compat.legacy import (
    LegacyObjectiveAdapter,
    wrap_legacy_objective,
    is_legacy_objective,
    adapt_objective,
)

__all__ = [
    'LegacyObjectiveAdapter',
    'wrap_legacy_objective',
    'is_legacy_objective',
    'adapt_objective',
]
