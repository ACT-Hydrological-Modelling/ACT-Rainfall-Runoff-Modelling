"""
Report visualization functions for CalibrationReport.

This module provides plotting functions for generating comprehensive calibration
report cards, including both Matplotlib (static) and Plotly (interactive) versions.

Functions:
    Matplotlib:
    - plot_report_card_matplotlib: Multi-panel summary figure
    - plot_hydrograph_comparison: Observed vs simulated hydrograph
    - plot_fdc_comparison: Flow duration curves
    - plot_scatter_comparison: Scatter plot with 1:1 line
    - plot_parameter_bounds_chart: Parameters as % of bounds
    
    Plotly:
    - plot_report_card_plotly: Interactive HTML dashboard
    - plot_hydrograph_plotly: Interactive hydrograph
    - plot_fdc_plotly: Interactive FDC
    - plot_parameter_bounds_plotly: Interactive parameter bounds chart

Example:
    >>> from pyrrm.calibration import CalibrationReport
    >>> from pyrrm.visualization.report_plots import plot_report_card_matplotlib
    >>> report = CalibrationReport.load('my_calibration.pkl')
    >>> fig = plot_report_card_matplotlib(report)
    >>> fig.savefig('report.png', dpi=300)
"""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyrrm.calibration.report import CalibrationReport
    from matplotlib.figure import Figure

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Styling Constants
# =============================================================================

# Light theme colors
COLORS_LIGHT = {
    'observed': '#e74c3c',      # Red
    'simulated': '#3498db',     # Blue
    'one_to_one': '#2ecc71',    # Green
    'grid': '#bdc3c7',          # Light gray
    'text': '#2c3e50',          # Dark blue-gray
}

# Dark theme colors
COLORS_DARK = {
    'observed': '#ff6b6b',      # Coral red
    'simulated': '#4ecdc4',     # Turquoise
    'one_to_one': '#95e1d3',    # Light green
    'grid': '#34495e',          # Dark gray
    'text': '#ecf0f1',          # Light gray
    'background': '#1a1a2e',
    'panel': '#16213e',
}


def _apply_dark_style():
    """Apply dark style to matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        'figure.facecolor': COLORS_DARK['background'],
        'axes.facecolor': COLORS_DARK['panel'],
        'axes.edgecolor': COLORS_DARK['text'],
        'axes.labelcolor': COLORS_DARK['text'],
        'text.color': COLORS_DARK['text'],
        'xtick.color': COLORS_DARK['text'],
        'ytick.color': COLORS_DARK['text'],
        'grid.color': COLORS_DARK['grid'],
        'legend.facecolor': COLORS_DARK['panel'],
        'legend.edgecolor': COLORS_DARK['text'],
    })


def _reset_style():
    """Reset matplotlib to default style."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update(plt.rcParamsDefault)


# =============================================================================
# Helper Functions
# =============================================================================

def _calculate_basic_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Calculate comprehensive performance metrics directly."""
    # Filter valid data
    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    
    if len(obs_v) == 0:
        return {}
    
    metrics = {}
    
    # ==========================================================================
    # NSE and variants
    # ==========================================================================
    # Standard NSE
    ss_res = np.sum((obs_v - sim_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    metrics['NSE'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # Log NSE (with offset for zeros)
    obs_pos_mask = obs_v > 0
    obs_pos = obs_v[obs_pos_mask]
    sim_pos = sim_v[obs_pos_mask]
    if len(obs_pos) > 0:
        log_obs = np.log(obs_pos + 1)
        log_sim = np.log(np.maximum(sim_pos, 0) + 1)
        ss_res_log = np.sum((log_obs - log_sim) ** 2)
        ss_tot_log = np.sum((log_obs - np.mean(log_obs)) ** 2)
        metrics['LogNSE'] = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else np.nan
    else:
        metrics['LogNSE'] = np.nan
    
    # Inverse NSE (1/Q transformation)
    obs_pos_inv = obs_v[obs_v > 0.01]
    sim_pos_inv = sim_v[obs_v > 0.01]
    if len(obs_pos_inv) > 0:
        inv_obs = 1.0 / obs_pos_inv
        inv_sim = 1.0 / np.maximum(sim_pos_inv, 0.01)
        ss_res_inv = np.sum((inv_obs - inv_sim) ** 2)
        ss_tot_inv = np.sum((inv_obs - np.mean(inv_obs)) ** 2)
        metrics['InvNSE'] = 1 - ss_res_inv / ss_tot_inv if ss_tot_inv > 0 else np.nan
    else:
        metrics['InvNSE'] = np.nan
    
    # Sqrt NSE
    sqrt_obs = np.sqrt(np.maximum(obs_v, 0))
    sqrt_sim = np.sqrt(np.maximum(sim_v, 0))
    ss_res_sqrt = np.sum((sqrt_obs - sqrt_sim) ** 2)
    ss_tot_sqrt = np.sum((sqrt_obs - np.mean(sqrt_obs)) ** 2)
    metrics['SqrtNSE'] = 1 - ss_res_sqrt / ss_tot_sqrt if ss_tot_sqrt > 0 else np.nan
    
    # ==========================================================================
    # KGE (standard) and components
    # ==========================================================================
    r = np.corrcoef(obs_v, sim_v)[0, 1] if len(obs_v) > 1 else np.nan
    alpha = np.std(sim_v) / np.std(obs_v) if np.std(obs_v) > 0 else np.nan
    beta = np.mean(sim_v) / np.mean(obs_v) if np.mean(obs_v) != 0 else np.nan
    metrics['KGE'] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r) else np.nan
    metrics['KGE_r'] = r
    metrics['KGE_alpha'] = alpha
    metrics['KGE_beta'] = beta
    
    # ==========================================================================
    # KGE with transformations (each with components)
    # ==========================================================================
    # KGE(log Q)
    if len(obs_pos) > 0:
        log_obs_kge = np.log(obs_pos + 1)
        log_sim_kge = np.log(np.maximum(sim_pos, 0) + 1)
        r_log = np.corrcoef(log_obs_kge, log_sim_kge)[0, 1] if len(log_obs_kge) > 1 else np.nan
        alpha_log = np.std(log_sim_kge) / np.std(log_obs_kge) if np.std(log_obs_kge) > 0 else np.nan
        beta_log = np.mean(log_sim_kge) / np.mean(log_obs_kge) if np.mean(log_obs_kge) != 0 else np.nan
        metrics['KGE_log'] = 1 - np.sqrt((r_log - 1)**2 + (alpha_log - 1)**2 + (beta_log - 1)**2) if not np.isnan(r_log) else np.nan
        metrics['KGE_log_r'] = r_log
        metrics['KGE_log_alpha'] = alpha_log
        metrics['KGE_log_beta'] = beta_log
    else:
        metrics['KGE_log'] = np.nan
        metrics['KGE_log_r'] = np.nan
        metrics['KGE_log_alpha'] = np.nan
        metrics['KGE_log_beta'] = np.nan
    
    # KGE(sqrt Q)
    r_sqrt = np.corrcoef(sqrt_obs, sqrt_sim)[0, 1] if len(sqrt_obs) > 1 else np.nan
    alpha_sqrt = np.std(sqrt_sim) / np.std(sqrt_obs) if np.std(sqrt_obs) > 0 else np.nan
    beta_sqrt = np.mean(sqrt_sim) / np.mean(sqrt_obs) if np.mean(sqrt_obs) != 0 else np.nan
    metrics['KGE_sqrt'] = 1 - np.sqrt((r_sqrt - 1)**2 + (alpha_sqrt - 1)**2 + (beta_sqrt - 1)**2) if not np.isnan(r_sqrt) else np.nan
    metrics['KGE_sqrt_r'] = r_sqrt
    metrics['KGE_sqrt_alpha'] = alpha_sqrt
    metrics['KGE_sqrt_beta'] = beta_sqrt
    
    # KGE(1/Q)
    if len(obs_pos_inv) > 0:
        inv_obs_kge = 1.0 / obs_pos_inv
        inv_sim_kge = 1.0 / np.maximum(sim_pos_inv, 0.01)
        r_inv = np.corrcoef(inv_obs_kge, inv_sim_kge)[0, 1] if len(inv_obs_kge) > 1 else np.nan
        alpha_inv = np.std(inv_sim_kge) / np.std(inv_obs_kge) if np.std(inv_obs_kge) > 0 else np.nan
        beta_inv = np.mean(inv_sim_kge) / np.mean(inv_obs_kge) if np.mean(inv_obs_kge) != 0 else np.nan
        metrics['KGE_inv'] = 1 - np.sqrt((r_inv - 1)**2 + (alpha_inv - 1)**2 + (beta_inv - 1)**2) if not np.isnan(r_inv) else np.nan
        metrics['KGE_inv_r'] = r_inv
        metrics['KGE_inv_alpha'] = alpha_inv
        metrics['KGE_inv_beta'] = beta_inv
    else:
        metrics['KGE_inv'] = np.nan
        metrics['KGE_inv_r'] = np.nan
        metrics['KGE_inv_alpha'] = np.nan
        metrics['KGE_inv_beta'] = np.nan
    
    # ==========================================================================
    # Traditional error metrics
    # ==========================================================================
    metrics['RMSE'] = np.sqrt(np.mean((sim_v - obs_v) ** 2))
    metrics['MAE'] = np.mean(np.abs(sim_v - obs_v))
    metrics['PBIAS'] = 100 * np.sum(sim_v - obs_v) / np.sum(obs_v) if np.sum(obs_v) != 0 else np.nan
    metrics['R2'] = r ** 2 if not np.isnan(r) else np.nan
    
    # ==========================================================================
    # FDC-based metrics (segmented flow errors)
    # ==========================================================================
    # Sort flows for FDC
    obs_sorted = np.sort(obs_v)[::-1]
    sim_sorted = np.sort(sim_v)[::-1]
    n = len(obs_sorted)
    
    # Flow percentiles
    def get_percentile_value(arr, pct):
        idx = int(pct / 100 * len(arr))
        idx = min(idx, len(arr) - 1)
        return arr[idx]
    
    # High flows (Q5, Q10) - top 5-10% exceedance
    q5_obs = get_percentile_value(obs_sorted, 5)
    q5_sim = get_percentile_value(sim_sorted, 5)
    q10_obs = get_percentile_value(obs_sorted, 10)
    q10_sim = get_percentile_value(sim_sorted, 10)
    
    metrics['Q5_bias'] = 100 * (q5_sim - q5_obs) / q5_obs if q5_obs > 0 else np.nan
    metrics['Q10_bias'] = 100 * (q10_sim - q10_obs) / q10_obs if q10_obs > 0 else np.nan
    
    # Mid flows (Q50)
    q50_obs = get_percentile_value(obs_sorted, 50)
    q50_sim = get_percentile_value(sim_sorted, 50)
    metrics['Q50_bias'] = 100 * (q50_sim - q50_obs) / q50_obs if q50_obs > 0 else np.nan
    
    # Low flows (Q90, Q95) - bottom 90-95% exceedance
    q90_obs = get_percentile_value(obs_sorted, 90)
    q90_sim = get_percentile_value(sim_sorted, 90)
    q95_obs = get_percentile_value(obs_sorted, 95)
    q95_sim = get_percentile_value(sim_sorted, 95)
    
    metrics['Q90_bias'] = 100 * (q90_sim - q90_obs) / q90_obs if q90_obs > 0 else np.nan
    metrics['Q95_bias'] = 100 * (q95_sim - q95_obs) / q95_obs if q95_obs > 0 else np.nan
    
    # FDC slope (mid-section: 33-66% exceedance)
    idx_33 = int(0.33 * n)
    idx_66 = int(0.66 * n)
    if idx_66 > idx_33:
        exc_mid = np.linspace(33, 66, idx_66 - idx_33)
        obs_mid = obs_sorted[idx_33:idx_66]
        sim_mid = sim_sorted[idx_33:idx_66]
        
        # FDC slope in log space
        if np.all(obs_mid > 0) and np.all(sim_mid > 0):
            log_obs_mid = np.log10(obs_mid)
            log_sim_mid = np.log10(sim_mid)
            slope_obs = (log_obs_mid[-1] - log_obs_mid[0]) / (exc_mid[-1] - exc_mid[0]) if len(exc_mid) > 1 else np.nan
            slope_sim = (log_sim_mid[-1] - log_sim_mid[0]) / (exc_mid[-1] - exc_mid[0]) if len(exc_mid) > 1 else np.nan
            metrics['FDC_slope_obs'] = slope_obs
            metrics['FDC_slope_sim'] = slope_sim
            metrics['FDC_slope_error'] = 100 * (slope_sim - slope_obs) / abs(slope_obs) if slope_obs != 0 else np.nan
        else:
            metrics['FDC_slope_obs'] = np.nan
            metrics['FDC_slope_sim'] = np.nan
            metrics['FDC_slope_error'] = np.nan
    
    # ==========================================================================
    # Hydrologic signatures (as percent errors: 100 * (sim - obs) / obs)
    # ==========================================================================
    # Runoff coefficient error
    runoff_obs = np.sum(obs_v)
    runoff_sim = np.sum(sim_v)
    metrics['Runoff_coef_error'] = 100 * (runoff_sim - runoff_obs) / runoff_obs if runoff_obs > 0 else np.nan
    
    # Mean flow error
    mean_obs = np.mean(obs_v)
    mean_sim = np.mean(sim_v)
    metrics['Mean_flow_error'] = 100 * (mean_sim - mean_obs) / mean_obs if mean_obs > 0 else np.nan
    
    # Coefficient of variation error
    cv_obs = np.std(obs_v) / np.mean(obs_v) if np.mean(obs_v) > 0 else np.nan
    cv_sim = np.std(sim_v) / np.mean(sim_v) if np.mean(sim_v) > 0 else np.nan
    metrics['CV_error'] = 100 * (cv_sim - cv_obs) / cv_obs if cv_obs > 0 and not np.isnan(cv_obs) else np.nan
    
    # High flow frequency error (days above mean)
    high_flow_threshold = np.mean(obs_v)
    hf_obs = np.sum(obs_v > high_flow_threshold) / len(obs_v)
    hf_sim = np.sum(sim_v > high_flow_threshold) / len(sim_v)
    metrics['High_flow_freq_error'] = 100 * (hf_sim - hf_obs) / hf_obs if hf_obs > 0 else np.nan
    
    # Low flow frequency error (days below Q90)
    low_flow_threshold = q90_obs
    lf_obs = np.sum(obs_v < low_flow_threshold) / len(obs_v)
    lf_sim = np.sum(sim_v < low_flow_threshold) / len(sim_v)
    metrics['Low_flow_freq_error'] = 100 * (lf_sim - lf_obs) / lf_obs if lf_obs > 0 else np.nan
    
    # Zero flow days error
    zf_obs = np.sum(obs_v <= 0) / len(obs_v) * 100
    zf_sim = np.sum(sim_v <= 0) / len(sim_v) * 100
    # For zero flow: if obs has none, report absolute difference
    if zf_obs > 0:
        metrics['Zero_flow_error'] = 100 * (zf_sim - zf_obs) / zf_obs
    else:
        metrics['Zero_flow_error'] = zf_sim - zf_obs  # Absolute difference in %
    
    return metrics


def _get_metric_color(value: float, metric_name: str) -> str:
    """Get color code for metric value (green=good, red=bad)."""
    if np.isnan(value):
        return '#888888'
    
    # Efficiency metrics (higher is better, 1.0 is perfect)
    # Include all NSE variants with their display names
    efficiency_metrics = [
        'NSE', 'KGE', 'LogNSE', 'SqrtNSE', 'InvNSE', 
        'NSE (high flows)', 'LogNSE (low flows)', 'SqrtNSE (balanced)', 'InvNSE (very low flows)',
        'KGE_log', 'KGE_sqrt', 'KGE_inv', 'KGE(log)', 'KGE(√Q)', 'KGE(1/Q)',
        'KGE_r', 'r', 'R2', 'R²', 'r (corr)', 'R² (variance explained)'
    ]
    
    # Check if metric name contains any efficiency metric substring
    is_efficiency = any(eff in metric_name for eff in ['NSE', 'KGE', 'R²', 'R2'])
    # But not if it's an error metric
    is_error = 'error' in metric_name.lower() or 'Error' in metric_name
    
    if metric_name in efficiency_metrics or (is_efficiency and not is_error and 'α' not in metric_name and 'β' not in metric_name):
        if value >= 0.75:
            return '#27ae60'  # Green - excellent
        elif value >= 0.5:
            return '#f39c12'  # Orange - good
        elif value >= 0.0:
            return '#e67e22'  # Dark orange - acceptable
        else:
            return '#e74c3c'  # Red - poor
    
    # PBIAS and percent errors (closer to 0 is better)
    elif 'PBIAS' in metric_name or 'bias' in metric_name.lower() or 'error' in metric_name.lower():
        if abs(value) <= 10:
            return '#27ae60'  # Green
        elif abs(value) <= 25:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red
    
    # KGE components alpha and beta (closer to 1 is better)
    elif metric_name in ['KGE_alpha', 'KGE_beta', 'α', 'β', 'α (var)', 'β (bias)'] or '  α' in metric_name or '  β' in metric_name:
        if 0.8 <= value <= 1.2:
            return '#27ae60'
        elif 0.5 <= value <= 1.5:
            return '#f39c12'
        else:
            return '#e74c3c'
    
    return '#2c3e50'  # Default


# =============================================================================
# Matplotlib Functions
# =============================================================================

def plot_report_card_matplotlib(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (20, 24),
    dark_theme: bool = False
) -> 'Figure':
    """
    Generate a comprehensive matplotlib report card figure.
    
    Layout (2 columns):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  HEADER: Catchment Name, Method, Objective, Period                          │
    ├─────────────────────────────────┬───────────────────────────────────────────┤
    │                                 │                                           │
    │  Hydrograph (Linear Scale)      │   COMPREHENSIVE METRICS TABLE             │
    │                                 │                                           │
    ├─────────────────────────────────┤   - NSE Variants                          │
    │                                 │   - KGE (Q) + components                  │
    │  Hydrograph (Log Scale)         │   - KGE (log Q) + components              │
    │                                 │   - KGE (√Q) + components                 │
    ├─────────────────────────────────┤   - KGE (1/Q) + components                │
    │                                 │   - Error Metrics                         │
    │  Flow Duration Curve            │   - FDC Segmented Flow Errors             │
    │                                 │   - Hydrologic Signatures                 │
    ├─────────────────────────────────┤                                           │
    │                                 │                                           │
    │  Scatter Plot                   │                                           │
    │                                 │                                           │
    ├─────────────────────────────────┤                                           │
    │                                 │                                           │
    │  Parameter Bounds Chart         │                                           │
    │                                 │                                           │
    └─────────────────────────────────┴───────────────────────────────────────────┘
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size (width, height)
        dark_theme: Use dark background theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    bg_color = COLORS_DARK['background'] if dark_theme else '#ffffff'
    panel_color = COLORS_DARK['panel'] if dark_theme else '#f8f9fa'
    text_color = COLORS_DARK['text'] if dark_theme else '#2c3e50'
    
    # Create figure with 2-column layout
    # Left column: 5 plots stacked vertically
    # Right column: Single metrics table spanning full height
    fig = plt.figure(figsize=figsize)
    
    # Main grid: header row + content row
    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.04, 0.96], hspace=0.02)
    
    # Content grid: 2 columns (plots | table)
    gs_content = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], 
                                                   width_ratios=[1.1, 0.9], wspace=0.08)
    
    # Left column: 5 plots
    gs_plots = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_content[0], 
                                                 height_ratios=[1, 1, 1, 1, 1.2], hspace=0.25)
    
    fig.patch.set_facecolor(bg_color)
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    # Calculate comprehensive metrics
    metrics = _calculate_basic_metrics(obs, sim)
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    ax_header = fig.add_subplot(gs_main[0])
    ax_header.axis('off')
    ax_header.set_facecolor(bg_color)
    
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    area = report.catchment_info.get('area_km2', '')
    
    title_text = f"{catchment_name}"
    if gauge_id:
        title_text += f" ({gauge_id})"
    
    ax_header.text(0.5, 0.65, title_text, transform=ax_header.transAxes,
                   fontsize=18, fontweight='bold', ha='center', va='center',
                   color=text_color)
    
    subtitle = f"Method: {report.result.method} | Objective: {report.result.objective_name}"
    subtitle += f" | Best: {report.result.best_objective:.4f}"
    if area:
        subtitle += f" | Area: {area} km²"
    subtitle += f" | Period: {report.calibration_period[0]} to {report.calibration_period[1]}"
    
    ax_header.text(0.5, 0.15, subtitle, transform=ax_header.transAxes,
                   fontsize=10, ha='center', va='center', color=text_color, alpha=0.8)
    
    # ==========================================================================
    # LEFT COLUMN - PLOTS
    # ==========================================================================
    
    # Plot 1: Hydrograph (Linear Scale)
    ax_hydro = fig.add_subplot(gs_plots[0])
    ax_hydro.set_facecolor(panel_color)
    ax_hydro.plot(dates, obs, color=colors['observed'], linewidth=0.7, 
                  label='Observed', alpha=0.9)
    ax_hydro.plot(dates, sim, color=colors['simulated'], linewidth=0.7, 
                  label='Simulated', alpha=0.9)
    ax_hydro.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_hydro.set_title('Hydrograph (Linear Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_hydro.legend(loc='upper right', fontsize=8)
    ax_hydro.grid(True, alpha=0.3)
    ax_hydro.tick_params(labelsize=8)
    ax_hydro.set_xticklabels([])
    
    # Plot 2: Hydrograph (Log Scale)
    ax_hydro_log = fig.add_subplot(gs_plots[1])
    ax_hydro_log.set_facecolor(panel_color)
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    ax_hydro_log.semilogy(dates, obs_log, color=colors['observed'], linewidth=0.7,
                          label='Observed', alpha=0.9)
    ax_hydro_log.semilogy(dates, sim_log, color=colors['simulated'], linewidth=0.7,
                          label='Simulated', alpha=0.9)
    ax_hydro_log.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_hydro_log.set_title('Hydrograph (Log Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_hydro_log.legend(loc='upper right', fontsize=8)
    ax_hydro_log.grid(True, alpha=0.3, which='both')
    ax_hydro_log.tick_params(labelsize=8)
    ax_hydro_log.set_xticklabels([])
    
    # Plot 3: Flow Duration Curve
    ax_fdc = fig.add_subplot(gs_plots[2])
    ax_fdc.set_facecolor(panel_color)
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    ax_fdc.semilogy(exc_obs, obs_sorted, color=colors['observed'], linewidth=1.5,
                    label='Observed')
    ax_fdc.semilogy(exc_sim, sim_sorted, color=colors['simulated'], linewidth=1.5,
                    linestyle='--', label='Simulated')
    ax_fdc.set_xlabel('Exceedance (%)', fontsize=9)
    ax_fdc.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_fdc.set_title('Flow Duration Curve', fontweight='bold', fontsize=10, pad=3)
    ax_fdc.legend(loc='upper right', fontsize=8)
    ax_fdc.grid(True, alpha=0.3, which='both')
    ax_fdc.set_xlim(0, 100)
    ax_fdc.tick_params(labelsize=8)
    
    # Plot 4: Scatter Plot (Log-Log scale)
    ax_scatter = fig.add_subplot(gs_plots[3])
    ax_scatter.set_facecolor(panel_color)
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    # Filter for positive values only (required for log scale)
    pos_mask = (obs_valid > 0) & (sim_valid > 0)
    obs_pos = obs_valid[pos_mask]
    sim_pos = sim_valid[pos_mask]
    
    ax_scatter.scatter(obs_pos, sim_pos, alpha=0.3, s=8, color=colors['simulated'])
    
    # 1:1 line for log scale
    min_val = max(np.nanmin(obs_pos), np.nanmin(sim_pos), 0.01)
    max_val = max(np.nanmax(obs_pos), np.nanmax(sim_pos))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], '--', color=colors['one_to_one'],
                    linewidth=1.5, label='1:1 Line')
    
    ax_scatter.set_xlabel('Observed (ML/day)', fontsize=9)
    ax_scatter.set_ylabel('Simulated (ML/day)', fontsize=9)
    ax_scatter.set_title('Scatter Plot (Log Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.legend(loc='upper left', fontsize=8)
    ax_scatter.grid(True, alpha=0.3, which='both')
    ax_scatter.tick_params(labelsize=8)
    r_squared = metrics.get('R2', np.nan)
    if not np.isnan(r_squared):
        ax_scatter.text(0.95, 0.05, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes,
                       fontsize=9, ha='right', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 5: Parameter Bounds Chart
    ax_bounds = fig.add_subplot(gs_plots[4])
    ax_bounds.set_facecolor(panel_color)
    
    if report.parameter_bounds:
        param_names = list(report.result.best_parameters.keys())
        norm_values = []
        actual_values = []
        bar_colors = []
        
        for param in param_names:
            value = report.result.best_parameters[param]
            actual_values.append(value)
            if param in report.parameter_bounds:
                low, high = report.parameter_bounds[param]
                norm = (value - low) / (high - low) * 100 if high > low else 50
                norm = np.clip(norm, 0, 100)
                norm_values.append(norm)
                # Red ONLY if within the shaded regions (0-10% or 90-100%)
                if norm <= 10 or norm >= 90:
                    bar_colors.append('#e74c3c')  # Red - inside shaded warning zone
                else:
                    bar_colors.append(colors['simulated'])  # Normal blue
            else:
                norm_values.append(50)
                bar_colors.append(colors['simulated'])
        
        y_pos_bounds = np.arange(len(param_names))
        
        # Add shaded warning regions (0-10% and 90-100%)
        ax_bounds.axvspan(0, 10, alpha=0.15, color='#e74c3c', zorder=0)
        ax_bounds.axvspan(90, 100, alpha=0.15, color='#e74c3c', zorder=0)
        
        bars = ax_bounds.barh(y_pos_bounds, norm_values, color=bar_colors, alpha=0.8, height=0.6, zorder=2)
        ax_bounds.axvline(x=50, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
        ax_bounds.axvline(x=10, linestyle=':', color='#e74c3c', alpha=0.5, linewidth=1, zorder=1)
        ax_bounds.axvline(x=90, linestyle=':', color='#e74c3c', alpha=0.5, linewidth=1, zorder=1)
        
        for bar, val, norm in zip(bars, actual_values, norm_values):
            label_x = norm + 2 if norm < 85 else norm - 3
            ha = 'left' if norm < 85 else 'right'
            ax_bounds.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', ha=ha, fontsize=7, fontweight='bold', zorder=3)
        
        ax_bounds.set_yticks(y_pos_bounds)
        ax_bounds.set_yticklabels(param_names, fontsize=7)
        ax_bounds.set_xlabel('Position within Bounds (%)', fontsize=9)
        ax_bounds.set_xlim(-5, 105)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=10, pad=3)
        ax_bounds.invert_yaxis()
        ax_bounds.tick_params(labelsize=8)
        
        # Add legend for shaded regions
        ax_bounds.text(0.5, -0.08, 'Shaded regions: parameters within 10% of bounds (may be constrained)',
                      transform=ax_bounds.transAxes, fontsize=7, ha='center', va='top', 
                      color='#e74c3c', style='italic')
    else:
        ax_bounds.text(0.5, 0.5, 'No parameter bounds available', ha='center', va='center',
                       transform=ax_bounds.transAxes, fontsize=10)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=10, pad=3)
    
    # ==========================================================================
    # RIGHT COLUMN - METRICS TABLE
    # ==========================================================================
    ax_table = fig.add_subplot(gs_content[1])
    ax_table.axis('off')
    ax_table.set_facecolor(bg_color)
    
    # Build comprehensive metrics data for the table
    # Format: (Metric Name, Value, Type for coloring)
    def fmt_value(val, metric_type):
        """Format value based on type."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        if metric_type == 'bias':
            return f'{val:+.2f}%'
        elif metric_type == 'error':
            return f'{val:.2f}'
        elif metric_type == 'ratio':
            return f'{val:.3f}'
        elif metric_type == 'percent':
            return f'{val:.1f}%'
        else:
            return f'{val:.4f}'
    
    # Build table data: list of (section_header, [(name, value, type), ...])
    # Flow regime clarifications added in parentheses
    table_sections = [
        ('NSE Variants (Flow Regime Focus)', [
            ('NSE (high flows)', metrics.get('NSE', np.nan), 'efficiency'),
            ('LogNSE (low flows)', metrics.get('LogNSE', np.nan), 'efficiency'),
            ('SqrtNSE (balanced)', metrics.get('SqrtNSE', np.nan), 'efficiency'),
            ('InvNSE (very low flows)', metrics.get('InvNSE', np.nan), 'efficiency'),
        ]),
        ('KGE (Q) - High Flows', [
            ('KGE', metrics.get('KGE', np.nan), 'efficiency'),
            ('  r (correlation)', metrics.get('KGE_r', np.nan), 'efficiency'),
            ('  α (variability)', metrics.get('KGE_alpha', np.nan), 'ratio'),
            ('  β (bias)', metrics.get('KGE_beta', np.nan), 'ratio'),
        ]),
        ('KGE (log Q) - Low Flows', [
            ('KGE(log)', metrics.get('KGE_log', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_log_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_log_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_log_beta', np.nan), 'ratio'),
        ]),
        ('KGE (√Q) - Balanced', [
            ('KGE(√Q)', metrics.get('KGE_sqrt', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_sqrt_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_sqrt_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_sqrt_beta', np.nan), 'ratio'),
        ]),
        ('KGE (1/Q) - Very Low Flows', [
            ('KGE(1/Q)', metrics.get('KGE_inv', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_inv_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_inv_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_inv_beta', np.nan), 'ratio'),
        ]),
        ('Error Metrics (Overall)', [
            ('R² (variance explained)', metrics.get('R2', np.nan), 'efficiency'),
            ('RMSE (high flow errors)', metrics.get('RMSE', np.nan), 'error'),
            ('MAE (balanced errors)', metrics.get('MAE', np.nan), 'error'),
            ('PBIAS (volume bias %)', metrics.get('PBIAS', np.nan), 'bias'),
        ]),
        ('FDC Segmented Flow Errors (%)', [
            ('Q5 (peak flows)', metrics.get('Q5_bias', np.nan), 'bias'),
            ('Q10 (high flows)', metrics.get('Q10_bias', np.nan), 'bias'),
            ('Q50 (median flows)', metrics.get('Q50_bias', np.nan), 'bias'),
            ('Q90 (low flows)', metrics.get('Q90_bias', np.nan), 'bias'),
            ('Q95 (very low flows)', metrics.get('Q95_bias', np.nan), 'bias'),
            ('FDC slope (regime shape)', metrics.get('FDC_slope_error', np.nan), 'bias'),
        ]),
        ('Hydrologic Signature Errors (%)', [
            ('Runoff coef (water balance)', metrics.get('Runoff_coef_error', np.nan), 'bias'),
            ('Mean flow (volume bias)', metrics.get('Mean_flow_error', np.nan), 'bias'),
            ('CV (variability)', metrics.get('CV_error', np.nan), 'bias'),
            ('High flow freq (>mean)', metrics.get('High_flow_freq_error', np.nan), 'bias'),
            ('Low flow freq (<Q90)', metrics.get('Low_flow_freq_error', np.nan), 'bias'),
            ('Zero flow days', metrics.get('Zero_flow_error', np.nan), 'bias'),
        ]),
    ]
    
    # Build table rows
    cell_text = []
    row_colors = []
    
    for section_header, section_metrics in table_sections:
        # Add section header row
        cell_text.append([section_header, ''])
        row_colors.append('header')
        
        # Add metric rows
        for name, val, mtype in section_metrics:
            val_str = fmt_value(val, mtype)
            cell_text.append([name, val_str])
            row_colors.append(mtype)
    
    # Create the table
    table = ax_table.table(
        cellText=cell_text,
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        colLoc='left',
        loc='upper center',
        colWidths=[0.65, 0.35]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    
    # Style header row
    for col in range(2):
        cell = table[(0, col)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_height(0.03)
    
    # Style data rows
    for row_idx, (row_type, (text_row)) in enumerate(zip(row_colors, cell_text), start=1):
        for col in range(2):
            cell = table[(row_idx, col)]
            
            if row_type == 'header':
                # Section header
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white', fontweight='bold', fontsize=9)
            else:
                # Data row - alternate colors
                bg = '#f8f9fa' if row_idx % 2 == 0 else '#ffffff'
                cell.set_facecolor(bg)
                
                # Color the value cell based on metric type
                if col == 1:
                    val_text = text_row[1]
                    metric_name = text_row[0].strip()
                    if val_text != 'N/A':
                        try:
                            # Parse value for coloring
                            val_clean = val_text.replace('%', '').replace('+', '')
                            val_num = float(val_clean)
                            color = _get_metric_color(val_num, metric_name)
                            cell.set_text_props(color=color, fontweight='bold')
                        except:
                            pass
    
    # Add title above table
    ax_table.text(0.5, 1.0, 'Comprehensive Performance Metrics', 
                  transform=ax_table.transAxes,
                  fontsize=12, fontweight='bold', ha='center', va='bottom',
                  color=text_color)
    
    # Add color legends at TOP (between title and table)
    # Legend for efficiency metrics (NSE, KGE, R²)
    ax_table.text(0.02, 0.975, 'Efficiency:', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color=text_color, fontweight='bold')
    ax_table.text(0.15, 0.975, '● ≥0.75', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#27ae60', fontweight='bold')
    ax_table.text(0.27, 0.975, '● ≥0.50', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#f39c12', fontweight='bold')
    ax_table.text(0.39, 0.975, '● <0.50', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#e74c3c', fontweight='bold')
    
    # Legend for percent errors (bias, FDC errors, signature errors)
    ax_table.text(0.54, 0.975, '% Errors:', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color=text_color, fontweight='bold')
    ax_table.text(0.67, 0.975, '● ≤±10%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#27ae60', fontweight='bold')
    ax_table.text(0.79, 0.975, '● ≤±25%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#f39c12', fontweight='bold')
    ax_table.text(0.91, 0.975, '● >±25%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#e74c3c', fontweight='bold')
    
    if dark_theme:
        _reset_style()
    
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.06, right=0.98)
    return fig


def plot_hydrograph_comparison(
    report: 'CalibrationReport',
    log_scale: bool = False,
    figsize: Tuple[int, int] = (14, 6),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot observed vs simulated hydrograph.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    if log_scale:
        obs = np.where(obs > 0, obs, np.nan)
        sim = np.where(sim > 0, sim, np.nan)
        ax.semilogy(dates, obs, color=colors['observed'], linewidth=1,
                    label='Observed', alpha=0.9)
        ax.semilogy(dates, sim, color=colors['simulated'], linewidth=1,
                    label='Simulated', alpha=0.9)
    else:
        ax.plot(dates, obs, color=colors['observed'], linewidth=1,
                label='Observed', alpha=0.9)
        ax.plot(dates, sim, color=colors['simulated'], linewidth=1,
                label='Simulated', alpha=0.9)
    
    ax.set_ylabel('Flow (ML/day)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    scale_label = 'Log Scale' if log_scale else 'Linear Scale'
    ax.set_title(f'Hydrograph Comparison ({scale_label})', fontweight='bold')
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_fdc_comparison(
    report: 'CalibrationReport',
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot flow duration curves for observed and simulated.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    if log_scale:
        ax.semilogy(exc_obs, obs_sorted, color=colors['observed'], linewidth=2,
                    label='Observed')
        ax.semilogy(exc_sim, sim_sorted, color=colors['simulated'], linewidth=2,
                    linestyle='--', label='Simulated')
    else:
        ax.plot(exc_obs, obs_sorted, color=colors['observed'], linewidth=2,
                label='Observed')
        ax.plot(exc_sim, sim_sorted, color=colors['simulated'], linewidth=2,
                linestyle='--', label='Simulated')
    
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Flow (ML/day)')
    ax.set_title('Flow Duration Curve', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
    ax.set_xlim(0, 100)
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_scatter_comparison(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (8, 8),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot observed vs simulated scatter plot with 1:1 line.
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    ax.scatter(obs_valid, sim_valid, alpha=0.3, s=10, color=colors['simulated'])
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    min_val = min(np.nanmin(obs_valid), np.nanmin(sim_valid))
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=colors['one_to_one'],
            linewidth=2, label='1:1 Line')
    
    # Add regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], p([min_val, max_val]), ':',
            color=colors['observed'], linewidth=2, label='Regression')
    
    ax.set_xlabel('Observed Flow (ML/day)')
    ax.set_ylabel('Simulated Flow (ML/day)')
    ax.set_title('Observed vs Simulated', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add metrics annotation
    try:
        metrics = report.calculate_metrics()
        metrics_text = f"NSE: {metrics['NSE']:.3f}\nKGE: {metrics['KGE']:.3f}"
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception:
        pass
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_parameter_bounds_chart(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (10, 8),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot calibrated parameters as percentage of their bounds.
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if not report.parameter_bounds:
        raise ValueError("No parameter bounds available in report")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    param_names = list(report.result.best_parameters.keys())
    values = []
    norm_values = []
    
    for param in param_names:
        value = report.result.best_parameters[param]
        values.append(value)
        
        if param in report.parameter_bounds:
            low, high = report.parameter_bounds[param]
            norm = (value - low) / (high - low) * 100 if high > low else 50
            norm_values.append(np.clip(norm, 0, 100))
        else:
            norm_values.append(50)
    
    y_pos = np.arange(len(param_names))
    bars = ax.barh(y_pos, norm_values, color=colors['simulated'], alpha=0.7)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        label_x = width + 2 if width < 90 else width - 5
        ha = 'left' if width < 90 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', ha=ha, fontsize=8)
    
    ax.axvline(x=50, linestyle='--', color=colors['observed'], alpha=0.7,
               label='50% (middle of bounds)')
    ax.axvline(x=0, linestyle='-', color='gray', alpha=0.3)
    ax.axvline(x=100, linestyle='-', color='gray', alpha=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Position within Parameter Bounds (%)')
    ax.set_xlim(-5, 110)
    ax.set_title('Calibrated Parameters vs Bounds', fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


# =============================================================================
# Plotly Functions
# =============================================================================

def plot_report_card_plotly(
    report: 'CalibrationReport',
    height: int = 1400
):
    """
    Generate an interactive Plotly report card with 2-column layout.
    
    Layout (2 columns):
    ┌───────────────────────────────────────┬─────────────────────────────────────┐
    │  HEADER: Catchment Name               │  HEADER (continued)                 │
    ├───────────────────────────────────────┼─────────────────────────────────────┤
    │  Hydrograph (Linear Scale)            │  Comprehensive Performance Metrics  │
    │                                       │  - NSE variants (Q, log, sqrt, inv) │
    ├───────────────────────────────────────┤  - KGE variants (Q, log, sqrt, inv) │
    │  Hydrograph (Log Scale)               │  - KGE components (r, α, β)         │
    │                                       │  - Error metrics (RMSE, MAE, PBIAS) │
    ├───────────────────────────────────────┼─────────────────────────────────────┤
    │  Flow Duration Curve                  │  KGE Components Bar Chart           │
    │                                       │                                     │
    ├───────────────────────────────────────┼─────────────────────────────────────┤
    │  Scatter Plot                         │  Parameter Bounds Chart             │
    │                                       │                                     │
    └───────────────────────────────────────┴─────────────────────────────────────┘
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object (save with fig.write_html('file.html'))
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function. Install with: pip install plotly")
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    # Calculate comprehensive metrics
    metrics = _calculate_basic_metrics(obs, sim)
    
    # Colors
    color_obs = '#e74c3c'
    color_sim = '#3498db'
    color_excellent = '#27ae60'
    color_good = '#f39c12'
    color_poor = '#e74c3c'
    
    # Create subplots with 2-column layout
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=(
            '', '',  # Row 1: Header
            'Hydrograph (Linear Scale)', 'Comprehensive Performance Metrics',
            '', '',  # Row 3: Hydro Log + (merged with metrics table)
            'Flow Duration Curve', 'KGE Components',
            'Scatter Plot', 'Calibrated Parameters vs Bounds'
        ),
        row_heights=[0.05, 0.22, 0.22, 0.22, 0.29],
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scatter", "colspan": 2}, None],  # Header placeholder
            [{"type": "scatter"}, {"type": "table", "rowspan": 2}],  # Hydro + Metrics Table (spans 2 rows)
            [{"type": "scatter"}, None],  # Hydro Log
            [{"type": "scatter"}, {"type": "bar"}],  # FDC + KGE
            [{"type": "scatter"}, {"type": "bar"}]  # Scatter + Params
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.08
    )
    
    # ==========================================================================
    # Header
    # ==========================================================================
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    area = report.catchment_info.get('area_km2', '')
    
    title_text = f"<b>{catchment_name}</b>"
    if gauge_id:
        title_text += f" ({gauge_id})"
    
    subtitle = f"Method: {report.result.method} | Objective: {report.result.objective_name}"
    subtitle += f" | Best: {report.result.best_objective:.4f}"
    if area:
        subtitle += f" | Area: {area} km²"
    subtitle += f" | Period: {report.calibration_period[0]} to {report.calibration_period[1]}"
    
    # ==========================================================================
    # Row 2, Col 1: Hydrograph (Linear Scale)
    # ==========================================================================
    fig.add_trace(
        go.Scatter(x=dates, y=obs, name='Observed', line=dict(color=color_obs, width=1),
                   showlegend=True, legendgroup='main'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sim, name='Simulated', line=dict(color=color_sim, width=1),
                   showlegend=True, legendgroup='main'),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
    
    # ==========================================================================
    # Row 2-3, Col 2: Comprehensive Performance Metrics Table (spans 2 rows)
    # ==========================================================================
    def get_color_for_metric(value, metric_name):
        if np.isnan(value):
            return '#888888'
        # NSE and KGE variants
        if any(x in metric_name for x in ['NSE', 'KGE', 'R²']):
            if 'component' not in metric_name.lower() and metric_name not in ['r', 'α', 'β']:
                if value >= 0.75: return color_excellent
                elif value >= 0.5: return color_good
                else: return color_poor
        # KGE components (close to 1 is good)
        if metric_name in ['r', 'α', 'β']:
            if 0.8 <= value <= 1.2: return color_excellent
            elif 0.5 <= value <= 1.5: return color_good
            else: return color_poor
        # PBIAS (close to 0 is good)
        if 'PBIAS' in metric_name:
            if abs(value) <= 10: return color_excellent
            elif abs(value) <= 25: return color_good
            else: return color_poor
        return '#2c3e50'
    
    # Build comprehensive metric list with grouping
    metric_items = [
        # NSE Variants
        ('<b>NSE Variants</b>', '', '', True),
        ('NSE', f"{metrics.get('NSE', np.nan):.4f}" if not np.isnan(metrics.get('NSE', np.nan)) else "N/A", 'Nash-Sutcliffe Efficiency', False),
        ('LogNSE', f"{metrics.get('LogNSE', np.nan):.4f}" if not np.isnan(metrics.get('LogNSE', np.nan)) else "N/A", 'NSE on log(Q)', False),
        ('SqrtNSE', f"{metrics.get('SqrtNSE', np.nan):.4f}" if not np.isnan(metrics.get('SqrtNSE', np.nan)) else "N/A", 'NSE on √Q', False),
        ('InvNSE', f"{metrics.get('InvNSE', np.nan):.4f}" if not np.isnan(metrics.get('InvNSE', np.nan)) else "N/A", 'NSE on 1/Q', False),
        # KGE Variants
        ('<b>KGE Variants</b>', '', '', True),
        ('KGE', f"{metrics.get('KGE', np.nan):.4f}" if not np.isnan(metrics.get('KGE', np.nan)) else "N/A", 'Kling-Gupta Efficiency', False),
        ('KGE(log)', f"{metrics.get('KGE_log', np.nan):.4f}" if not np.isnan(metrics.get('KGE_log', np.nan)) else "N/A", 'KGE on log(Q)', False),
        ('KGE(√Q)', f"{metrics.get('KGE_sqrt', np.nan):.4f}" if not np.isnan(metrics.get('KGE_sqrt', np.nan)) else "N/A", 'KGE on √Q', False),
        ('KGE(1/Q)', f"{metrics.get('KGE_inv', np.nan):.4f}" if not np.isnan(metrics.get('KGE_inv', np.nan)) else "N/A", 'KGE on 1/Q', False),
        # KGE Components
        ('<b>KGE Components</b>', '', '', True),
        ('r (corr)', f"{metrics.get('KGE_r', np.nan):.4f}" if not np.isnan(metrics.get('KGE_r', np.nan)) else "N/A", 'Pearson correlation', False),
        ('α (var)', f"{metrics.get('KGE_alpha', np.nan):.4f}" if not np.isnan(metrics.get('KGE_alpha', np.nan)) else "N/A", 'Variability ratio σsim/σobs', False),
        ('β (bias)', f"{metrics.get('KGE_beta', np.nan):.4f}" if not np.isnan(metrics.get('KGE_beta', np.nan)) else "N/A", 'Bias ratio μsim/μobs', False),
        # Error Metrics
        ('<b>Error Metrics</b>', '', '', True),
        ('PBIAS', f"{metrics.get('PBIAS', np.nan):+.2f}%" if not np.isnan(metrics.get('PBIAS', np.nan)) else "N/A", 'Percent Bias', False),
        ('RMSE', f"{metrics.get('RMSE', np.nan):.1f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "N/A", 'Root Mean Square Error', False),
        ('MAE', f"{metrics.get('MAE', np.nan):.1f}" if not np.isnan(metrics.get('MAE', np.nan)) else "N/A", 'Mean Absolute Error', False),
        ('R²', f"{metrics.get('R2', np.nan):.4f}" if not np.isnan(metrics.get('R2', np.nan)) else "N/A", 'Coefficient of Determination', False),
    ]
    
    metric_names = []
    metric_values = []
    metric_descs = []
    metric_colors_name = []
    metric_colors_val = []
    metric_colors_desc = []
    
    for name, value, desc, is_header in metric_items:
        metric_names.append(name)
        metric_values.append(value)
        metric_descs.append(desc)
        if is_header:
            metric_colors_name.append('#34495e')
            metric_colors_val.append('#34495e')
            metric_colors_desc.append('#34495e')
        else:
            # Get numeric value for coloring
            try:
                num_val = float(value.replace('%', '').replace('+', ''))
            except:
                num_val = np.nan
            color = get_color_for_metric(num_val, name)
            metric_colors_name.append('#f8f9fa')
            metric_colors_val.append(color)
            metric_colors_desc.append('#f8f9fa')
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>', '<b>Description</b>'],
                fill_color='#34495e',
                font=dict(color='white', size=11),
                align='left',
                height=28
            ),
            cells=dict(
                values=[metric_names, metric_values, metric_descs],
                fill_color=[metric_colors_name, metric_colors_val, metric_colors_desc],
                font=dict(
                    color=[['white' if '#34495e' in c else '#2c3e50' for c in metric_colors_name],
                           ['white' if c != '#f8f9fa' else '#2c3e50' for c in metric_colors_val],
                           ['white' if '#34495e' in c else '#7f8c8d' for c in metric_colors_desc]],
                    size=10
                ),
                align='left',
                height=22
            ),
            columnwidth=[0.2, 0.2, 0.6]
        ),
        row=2, col=2
    )
    
    # ==========================================================================
    # Row 3, Col 1: Hydrograph (Log Scale)
    # ==========================================================================
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    fig.add_trace(
        go.Scatter(x=dates, y=obs_log, name='Observed', line=dict(color=color_obs, width=1),
                   showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sim_log, name='Simulated', line=dict(color=color_sim, width=1),
                   showlegend=False),
        row=3, col=1
    )
    fig.update_yaxes(type="log", title_text="Flow (ML/day)", row=3, col=1)
    # Add subtitle for log scale hydrograph
    fig.add_annotation(
        x=0.5, y=1.0, xref='x3 domain', yref='y3 domain',
        text="<b>Hydrograph (Log Scale)</b>",
        showarrow=False, font=dict(size=12),
        yanchor='bottom'
    )
    
    # ==========================================================================
    # Row 4, Col 1: Flow Duration Curve
    # ==========================================================================
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    fig.add_trace(
        go.Scatter(x=exc_obs, y=obs_sorted, name='Observed',
                   line=dict(color=color_obs, width=2), showlegend=False,
                   hovertemplate='Exceedance: %{x:.1f}%<br>Flow: %{y:.1f} ML/day<extra>Observed</extra>'),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=exc_sim, y=sim_sorted, name='Simulated',
                   line=dict(color=color_sim, width=2, dash='dash'), showlegend=False,
                   hovertemplate='Exceedance: %{x:.1f}%<br>Flow: %{y:.1f} ML/day<extra>Simulated</extra>'),
        row=4, col=1
    )
    fig.update_yaxes(type="log", title_text="Flow (ML/day)", row=4, col=1)
    fig.update_xaxes(range=[0, 100], title_text="Exceedance (%)", row=4, col=1)
    
    # ==========================================================================
    # Row 4, Col 2: KGE Components Breakdown
    # ==========================================================================
    kge_components = ['β (bias)', 'α (variability)', 'r (correlation)']
    kge_values = [
        metrics.get('KGE_beta', np.nan),
        metrics.get('KGE_alpha', np.nan),
        metrics.get('KGE_r', np.nan)
    ]
    kge_colors = []
    for i, v in enumerate(kge_values):
        if np.isnan(v):
            kge_colors.append('#888888')
        elif i == 2:  # correlation (r)
            kge_colors.append(color_excellent if v >= 0.75 else color_good if v >= 0.5 else color_poor)
        else:  # alpha and beta (close to 1 is good)
            kge_colors.append(color_excellent if 0.8 <= v <= 1.2 else color_good if 0.5 <= v <= 1.5 else color_poor)
    
    fig.add_trace(
        go.Bar(
            y=kge_components, 
            x=[v if not np.isnan(v) else 0 for v in kge_values],
            orientation='h',
            marker_color=kge_colors,
            text=[f'{v:.3f}' if not np.isnan(v) else 'N/A' for v in kge_values],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{y}: %{x:.3f}<extra></extra>'
        ),
        row=4, col=2
    )
    kge_max = max(2.0, max([v for v in kge_values if not np.isnan(v)], default=1) * 1.2)
    fig.add_trace(
        go.Scatter(
            x=[1.0, 1.0], y=[-0.5, 2.5],
            mode='lines',
            line=dict(color=color_excellent, dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip',
            name='Optimal (1.0)'
        ),
        row=4, col=2
    )
    fig.update_xaxes(range=[0, kge_max], title_text="Value", row=4, col=2)
    fig.update_yaxes(range=[-0.5, 2.5], row=4, col=2)
    
    # ==========================================================================
    # Row 5, Col 1: Scatter Plot with Statistics
    # ==========================================================================
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    fig.add_trace(
        go.Scatter(x=obs_valid, y=sim_valid, mode='markers',
                   marker=dict(color=color_sim, size=4, opacity=0.3),
                   name='Points', showlegend=False,
                   hovertemplate='Observed: %{x:.1f}<br>Simulated: %{y:.1f}<extra></extra>'),
        row=5, col=1
    )
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                   line=dict(color='green', dash='dash', width=2),
                   name='1:1 Line', showlegend=False),
        row=5, col=1
    )
    
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=p([0, max_val]), mode='lines',
                   line=dict(color=color_obs, dash='dot', width=2),
                   name=f'Fit (y={z[0]:.2f}x+{z[1]:.1f})', showlegend=False),
        row=5, col=1
    )
    
    fig.update_xaxes(title_text="Observed (ML/day)", row=5, col=1)
    fig.update_yaxes(title_text="Simulated (ML/day)", row=5, col=1)
    
    r_squared = metrics.get('R2', np.nan)
    if not np.isnan(r_squared):
        fig.add_annotation(
            x=0.95, y=0.05, xref='x5 domain', yref='y5 domain',
            text=f"<b>R² = {r_squared:.3f}</b>",
            showarrow=False, font=dict(size=11),
            bgcolor='white', bordercolor='gray', borderwidth=1
        )
    
    # ==========================================================================
    # Row 5, Col 2: Parameter Bounds Chart
    # ==========================================================================
    if report.parameter_bounds:
        param_names = list(report.result.best_parameters.keys())
        norm_values = []
        actual_values = []
        bar_colors = []
        
        for param in param_names:
            value = report.result.best_parameters[param]
            actual_values.append(value)
            if param in report.parameter_bounds:
                low, high = report.parameter_bounds[param]
                norm = (value - low) / (high - low) * 100 if high > low else 50
                norm = np.clip(norm, 0, 100)
                norm_values.append(norm)
                if norm < 5 or norm > 95:
                    bar_colors.append(color_poor)
                elif norm < 15 or norm > 85:
                    bar_colors.append(color_good)
                else:
                    bar_colors.append(color_sim)
            else:
                norm_values.append(50)
                bar_colors.append(color_sim)
        
        fig.add_trace(
            go.Bar(
                y=param_names, 
                x=norm_values, 
                orientation='h',
                marker_color=bar_colors,
                text=[f'{v:.2f}' for v in actual_values],
                textposition='outside',
                showlegend=False,
                hovertemplate='%{y}<br>Value: %{text}<br>Position: %{x:.1f}%<extra></extra>'
            ),
            row=5, col=2
        )
        n_params = len(param_names)
        fig.add_trace(
            go.Scatter(
                x=[50, 50], y=[-0.5, n_params - 0.5],
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=5, col=2
        )
        fig.update_xaxes(range=[-5, 115], title_text="Position within Bounds (%)", row=5, col=2)
        fig.update_yaxes(range=[-0.5, n_params - 0.5], row=5, col=2)
    
    # ==========================================================================
    # Update Layout
    # ==========================================================================
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sup>{subtitle}</sup>",
            font=dict(size=18),
            x=0.5,
            xanchor='center'
        ),
        height=height,
        showlegend=True,
        legend=dict(
            orientation='h', 
            y=0.99, 
            x=0.3, 
            xanchor='center',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        font=dict(family='Arial, sans-serif')
    )
    
    # Update all subplot backgrounds
    for i in range(1, 6):
        for j in range(1, 3):
            try:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=i, col=j)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=i, col=j)
            except Exception:
                pass
    
    return fig


def plot_hydrograph_plotly(
    report: 'CalibrationReport',
    log_scale: bool = False,
    height: int = 400
):
    """
    Create an interactive Plotly hydrograph.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    if log_scale:
        obs = np.where(obs > 0, obs, np.nan)
        sim = np.where(sim > 0, sim, np.nan)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=obs, name='Observed',
        line=dict(color='#e74c3c', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=sim, name='Simulated',
        line=dict(color='#3498db', width=1)
    ))
    
    scale_label = 'Log Scale' if log_scale else 'Linear Scale'
    fig.update_layout(
        title=f'Hydrograph Comparison ({scale_label})',
        xaxis_title='Date',
        yaxis_title='Flow (ML/day)',
        height=height,
        yaxis_type='log' if log_scale else 'linear'
    )
    
    return fig


def plot_fdc_plotly(
    report: 'CalibrationReport',
    log_scale: bool = True,
    height: int = 400
):
    """
    Create an interactive Plotly flow duration curve.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=exc_obs, y=obs_sorted, name='Observed',
        line=dict(color='#e74c3c', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=exc_sim, y=sim_sorted, name='Simulated',
        line=dict(color='#3498db', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Flow Duration Curve',
        xaxis_title='Exceedance Probability (%)',
        yaxis_title='Flow (ML/day)',
        height=height,
        yaxis_type='log' if log_scale else 'linear',
        xaxis_range=[0, 100]
    )
    
    return fig


def plot_scatter_plotly(
    report: 'CalibrationReport',
    height: int = 500
):
    """
    Create an interactive Plotly scatter plot of observed vs simulated.
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    # Calculate metrics for annotation
    metrics = _calculate_basic_metrics(obs, sim)
    r_squared = metrics.get('KGE_r', np.nan) ** 2 if not np.isnan(metrics.get('KGE_r', np.nan)) else np.nan
    
    # Regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    min_val = min(np.nanmin(obs_valid), np.nanmin(sim_valid))
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=obs_valid, y=sim_valid, mode='markers',
        marker=dict(color='#3498db', size=5, opacity=0.4),
        name='Data points',
        hovertemplate='Observed: %{x:.1f}<br>Simulated: %{y:.1f}<extra></extra>'
    ))
    
    # 1:1 line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line=dict(color='green', dash='dash', width=2),
        name='1:1 Line'
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=p([min_val, max_val]), mode='lines',
        line=dict(color='#e74c3c', dash='dot', width=2),
        name=f'Regression (y={z[0]:.2f}x+{z[1]:.1f})'
    ))
    
    # Add R² annotation
    annotations = []
    if not np.isnan(r_squared):
        annotations.append(dict(
            x=0.95, y=0.05, xref='paper', yref='paper',
            text=f"<b>R² = {r_squared:.4f}</b><br>NSE = {metrics.get('NSE', np.nan):.4f}",
            showarrow=False, font=dict(size=12),
            bgcolor='white', bordercolor='gray', borderwidth=1
        ))
    
    fig.update_layout(
        title='Observed vs Simulated Flow',
        xaxis_title='Observed Flow (ML/day)',
        yaxis_title='Simulated Flow (ML/day)',
        height=height,
        annotations=annotations,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Make axes equal
    fig.update_xaxes(range=[0, max_val * 1.05])
    fig.update_yaxes(range=[0, max_val * 1.05], scaleanchor='x', scaleratio=1)
    
    return fig


def plot_parameter_bounds_plotly(
    report: 'CalibrationReport',
    height: int = 500
):
    """
    Create an interactive Plotly parameter bounds chart.
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    if not report.parameter_bounds:
        raise ValueError("No parameter bounds available in report")
    
    param_names = list(report.result.best_parameters.keys())
    norm_values = []
    actual_values = []
    hover_texts = []
    
    for param in param_names:
        value = report.result.best_parameters[param]
        actual_values.append(value)
        
        if param in report.parameter_bounds:
            low, high = report.parameter_bounds[param]
            norm = (value - low) / (high - low) * 100 if high > low else 50
            norm_values.append(np.clip(norm, 0, 100))
            hover_texts.append(f"{param}: {value:.4f}<br>Bounds: [{low:.3f}, {high:.3f}]<br>Position: {norm:.1f}%")
        else:
            norm_values.append(50)
            hover_texts.append(f"{param}: {value:.4f}<br>No bounds available")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=param_names,
        x=norm_values,
        orientation='h',
        marker_color='#3498db',
        text=[f'{v:.3f}' for v in actual_values],
        textposition='inside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50%")
    
    fig.update_layout(
        title='Calibrated Parameters vs Bounds',
        xaxis_title='Position within Bounds (%)',
        xaxis_range=[0, 100],
        height=height,
        yaxis_autorange='reversed'
    )
    
    return fig
