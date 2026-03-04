"""
Analysis module for rainfall-runoff models.

Provides:
- Sobol sensitivity analysis
- Model diagnostics and performance metrics
- Hydrological signatures (TOSSH/EflowStats style)
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
from pyrrm.analysis.signatures import (
    SIGNATURE_CATEGORIES,
    SIGNATURE_INFO,
    get_signature_info,
    compute_all_signatures,
    compute_magnitude_signatures,
    compute_variability_signatures,
    compute_timing_signatures,
    compute_fdc_signatures,
    compute_frequency_signatures,
    compute_recession_signatures,
    compute_baseflow_signatures,
    compute_event_signatures,
    compute_seasonality_signatures,
    signature_percent_error,
    compare_signatures,
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
    # Sensitivity analysis
    "SobolSensitivityAnalysis",
    "SobolResult",
    # Diagnostics
    "ModelDiagnostics",
    "compute_diagnostics",
    "DIAGNOSTIC_GROUPS",
    "print_diagnostics",
    "lyne_hollick_baseflow",
    # Hydrological signatures
    "SIGNATURE_CATEGORIES",
    "SIGNATURE_INFO",
    "get_signature_info",
    "compute_all_signatures",
    "compute_magnitude_signatures",
    "compute_variability_signatures",
    "compute_timing_signatures",
    "compute_fdc_signatures",
    "compute_frequency_signatures",
    "compute_recession_signatures",
    "compute_baseflow_signatures",
    "compute_event_signatures",
    "compute_seasonality_signatures",
    "signature_percent_error",
    "compare_signatures",
    # MCMC diagnostics
    "ARVIZ_AVAILABLE",
    "check_convergence",
    "posterior_summary",
    "compute_nse_from_posterior",
]
