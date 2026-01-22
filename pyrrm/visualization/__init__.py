"""
Visualization module for rainfall-runoff models.

Provides plotting functions for:
- Calibration diagnostics (MCMC traces, posteriors, dotty plots)
- Model outputs (hydrographs, flow duration curves, scatter plots)
- Sensitivity analysis (Sobol indices, parameter importance)
"""

from pyrrm.visualization.model_plots import (
    plot_hydrograph_with_precipitation,
    plot_hydrograph_simple,
    plot_flow_duration_curve,
    plot_scatter_with_metrics,
    plot_residuals,
    plot_monthly_boxplot,
    create_calibration_dashboard,
)
from pyrrm.visualization.calibration_plots import (
    plot_mcmc_traces,
    plot_posterior_distributions,
    plot_parameter_correlations,
    plot_objective_evolution,
    plot_dotty_plots,
    # Aliases for alternative naming conventions
    plot_parameter_traces,
    plot_parameter_histograms,
    plot_objective_function_trace,
    plot_dotty,
)
from pyrrm.visualization.sensitivity_plots import (
    plot_sobol_indices,
    plot_parameter_importance_ranking,
    plot_interaction_heatmap,
)

__all__ = [
    # Model plots
    "plot_hydrograph_with_precipitation",
    "plot_hydrograph_simple",
    "plot_flow_duration_curve",
    "plot_scatter_with_metrics",
    "plot_residuals",
    "plot_monthly_boxplot",
    "create_calibration_dashboard",
    # Calibration plots
    "plot_mcmc_traces",
    "plot_posterior_distributions",
    "plot_parameter_correlations",
    "plot_objective_evolution",
    "plot_dotty_plots",
    # Calibration plot aliases
    "plot_parameter_traces",
    "plot_parameter_histograms",
    "plot_objective_function_trace",
    "plot_dotty",
    # Sensitivity plots
    "plot_sobol_indices",
    "plot_parameter_importance_ranking",
    "plot_interaction_heatmap",
]
