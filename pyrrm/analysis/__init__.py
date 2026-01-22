"""
Analysis module for rainfall-runoff models.

Provides:
- Sobol sensitivity analysis
- Model diagnostics and performance metrics
"""

from pyrrm.analysis.sensitivity import SobolSensitivityAnalysis, SobolResult
from pyrrm.analysis.diagnostics import ModelDiagnostics

__all__ = [
    "SobolSensitivityAnalysis",
    "SobolResult",
    "ModelDiagnostics",
]
