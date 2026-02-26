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
    # Integrated precipitation–flow plots
    plot_precip_flow,
    plot_precip_flow_grid,
    plot_precip_flow_plotly,
    plot_precip_flow_grid_plotly,
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

# Report visualization functions
from pyrrm.visualization.report_plots import (
    # Matplotlib functions
    plot_report_card_matplotlib,
    plot_hydrograph_comparison,
    plot_fdc_comparison,
    plot_scatter_comparison,
    plot_parameter_bounds_chart,
    # Plotly functions
    plot_report_card_plotly,
    plot_hydrograph_plotly,
    plot_fdc_plotly,
    plot_scatter_plotly,
    plot_parameter_bounds_plotly,
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
    # Integrated precipitation–flow plots
    "plot_precip_flow",
    "plot_precip_flow_grid",
    "plot_precip_flow_plotly",
    "plot_precip_flow_grid_plotly",
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
    # Report plots (Matplotlib)
    "plot_report_card_matplotlib",
    "plot_hydrograph_comparison",
    "plot_fdc_comparison",
    "plot_scatter_comparison",
    "plot_parameter_bounds_chart",
    # Report plots (Plotly)
    "plot_report_card_plotly",
    "plot_hydrograph_plotly",
    "plot_fdc_plotly",
    "plot_scatter_plotly",
    "plot_parameter_bounds_plotly",
    # MCMC plots (if ArviZ available)
    "plot_mcmc_traces_nuts",
    "plot_mcmc_rank",
    "plot_posterior_pairs",
    "plot_mcmc_forest",
    "plot_hydrograph_with_uncertainty",
    "plot_mcmc_diagnostics_nuts",
    "dream_result_to_inference_data",
    "plot_dream_traces",
    "plot_rhat_summary",
    "plot_rhat_from_pydream",
    "plot_forest_grid",
    "plot_forest_grid_plotly",
    "plot_forest_interactive",
]

try:
    from pyrrm.visualization.mcmc_plots import (
        plot_mcmc_traces as plot_mcmc_traces_nuts,
        plot_mcmc_rank,
        plot_posterior_pairs,
        plot_mcmc_forest,
        plot_hydrograph_with_uncertainty,
        plot_mcmc_diagnostics as plot_mcmc_diagnostics_nuts,
        dream_result_to_inference_data,
        plot_dream_traces,
        plot_rhat_summary,
        plot_rhat_from_pydream,
        plot_forest_grid,
        plot_forest_grid_plotly,
        plot_forest_interactive,
    )
except ImportError:
    pass
