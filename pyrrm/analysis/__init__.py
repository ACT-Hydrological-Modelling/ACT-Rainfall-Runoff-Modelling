"""
Analysis module for rainfall-runoff models.

Provides:
- Sobol sensitivity analysis
- Model diagnostics and performance metrics
- MCMC convergence diagnostics (when ArviZ available)
"""

from pyrrm.analysis.sensitivity import SobolSensitivityAnalysis, SobolResult
from pyrrm.analysis.diagnostics import (
    ModelDiagnostics,
    compute_diagnostics,
    DIAGNOSTIC_GROUPS,
    print_diagnostics,
    lyne_hollick_baseflow,
)

try:
    from pyrrm.analysis.mcmc_diagnostics import (
        check_convergence,
        posterior_summary,
        compute_nse_from_posterior,
        ARVIZ_AVAILABLE,
    )
except ImportError:
    ARVIZ_AVAILABLE = False
    check_convergence = None
    posterior_summary = None
    compute_nse_from_posterior = None

__all__ = [
    "SobolSensitivityAnalysis",
    "SobolResult",
    "ModelDiagnostics",
    "compute_diagnostics",
    "DIAGNOSTIC_GROUPS",
    "print_diagnostics",
    "lyne_hollick_baseflow",
    "ARVIZ_AVAILABLE",
    "check_convergence",
    "posterior_summary",
    "compute_nse_from_posterior",
]
