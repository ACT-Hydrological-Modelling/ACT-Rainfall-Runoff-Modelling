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
    """Calculate basic performance metrics directly (fallback if objective_functions fails)."""
    # Filter valid data
    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    
    if len(obs_v) == 0:
        return {}
    
    metrics = {}
    
    # NSE
    ss_res = np.sum((obs_v - sim_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    metrics['NSE'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # KGE
    r = np.corrcoef(obs_v, sim_v)[0, 1] if len(obs_v) > 1 else np.nan
    alpha = np.std(sim_v) / np.std(obs_v) if np.std(obs_v) > 0 else np.nan
    beta = np.mean(sim_v) / np.mean(obs_v) if np.mean(obs_v) != 0 else np.nan
    metrics['KGE'] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r) else np.nan
    metrics['KGE_r'] = r
    metrics['KGE_alpha'] = alpha
    metrics['KGE_beta'] = beta
    
    # RMSE
    metrics['RMSE'] = np.sqrt(np.mean((sim_v - obs_v) ** 2))
    
    # MAE
    metrics['MAE'] = np.mean(np.abs(sim_v - obs_v))
    
    # PBIAS
    metrics['PBIAS'] = 100 * np.sum(sim_v - obs_v) / np.sum(obs_v) if np.sum(obs_v) != 0 else np.nan
    
    # Log NSE (with offset for zeros)
    obs_pos = obs_v[obs_v > 0]
    sim_pos = sim_v[obs_v > 0]
    if len(obs_pos) > 0:
        log_obs = np.log(obs_pos + 1)
        log_sim = np.log(sim_pos + 1)
        ss_res_log = np.sum((log_obs - log_sim) ** 2)
        ss_tot_log = np.sum((log_obs - np.mean(log_obs)) ** 2)
        metrics['LogNSE'] = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else np.nan
    
    # Sqrt NSE
    sqrt_obs = np.sqrt(np.maximum(obs_v, 0))
    sqrt_sim = np.sqrt(np.maximum(sim_v, 0))
    ss_res_sqrt = np.sum((sqrt_obs - sqrt_sim) ** 2)
    ss_tot_sqrt = np.sum((sqrt_obs - np.mean(sqrt_obs)) ** 2)
    metrics['SqrtNSE'] = 1 - ss_res_sqrt / ss_tot_sqrt if ss_tot_sqrt > 0 else np.nan
    
    return metrics


def _get_metric_color(value: float, metric_name: str) -> str:
    """Get color code for metric value (green=good, red=bad)."""
    if np.isnan(value):
        return '#888888'
    
    if metric_name in ['NSE', 'KGE', 'LogNSE', 'SqrtNSE', 'KGE_r']:
        if value >= 0.75:
            return '#27ae60'  # Green - excellent
        elif value >= 0.5:
            return '#f39c12'  # Orange - good
        elif value >= 0.0:
            return '#e67e22'  # Dark orange - acceptable
        else:
            return '#e74c3c'  # Red - poor
    elif metric_name == 'PBIAS':
        if abs(value) <= 10:
            return '#27ae60'  # Green
        elif abs(value) <= 25:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red
    elif metric_name in ['KGE_alpha', 'KGE_beta']:
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
    figsize: Tuple[int, int] = (18, 14),
    dark_theme: bool = False
) -> 'Figure':
    """
    Generate a comprehensive matplotlib report card figure.
    
    Layout (4 rows x 4 columns):
    ┌──────────────────────────────────────────────┬────────────────────────┐
    │  HEADER: Catchment Name, Method, Objective   │                        │
    ├──────────────────────────────────────────────┼────────────────────────┤
    │  Hydrograph (Linear Scale)                   │  Performance Metrics   │
    │                                              │  (color-coded table)   │
    ├──────────────────────────────────────────────┼────────────────────────┤
    │  Hydrograph (Log Scale)                      │  KGE Components        │
    │                                              │  (visual breakdown)    │
    ├───────────────────┬──────────────────────────┼────────────────────────┤
    │  Flow Duration    │  Scatter Plot            │  Parameter Bounds      │
    │  Curve            │  (1:1 line + metrics)    │  (horizontal bars)     │
    └───────────────────┴──────────────────────────┴────────────────────────┘
    
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
    
    # Create figure with custom grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                           width_ratios=[1.5, 1.5, 1, 1], 
                           height_ratios=[0.15, 1, 1, 1])
    
    fig.patch.set_facecolor(bg_color)
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    # Calculate metrics using robust fallback
    metrics = _calculate_basic_metrics(obs, sim)
    
    # ==========================================================================
    # Row 0: Header
    # ==========================================================================
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.set_facecolor(bg_color)
    
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    area = report.catchment_info.get('area_km2', '')
    
    # Main title
    title_text = f"{catchment_name}"
    if gauge_id:
        title_text += f" ({gauge_id})"
    
    ax_header.text(0.5, 0.7, title_text, transform=ax_header.transAxes,
                   fontsize=18, fontweight='bold', ha='center', va='center',
                   color=text_color)
    
    # Subtitle with method info
    subtitle = f"Method: {report.result.method} | Objective: {report.result.objective_name}"
    subtitle += f" | Best: {report.result.best_objective:.4f}"
    if area:
        subtitle += f" | Area: {area} km²"
    subtitle += f" | Period: {report.calibration_period[0]} to {report.calibration_period[1]}"
    
    ax_header.text(0.5, 0.2, subtitle, transform=ax_header.transAxes,
                   fontsize=11, ha='center', va='center', color=text_color, alpha=0.8)
    
    # ==========================================================================
    # Row 1, Col 0-1: Hydrograph (Linear Scale)
    # ==========================================================================
    ax_hydro = fig.add_subplot(gs[1, :2])
    ax_hydro.set_facecolor(panel_color)
    
    ax_hydro.plot(dates, obs, color=colors['observed'], linewidth=0.8, 
                  label='Observed', alpha=0.9)
    ax_hydro.plot(dates, sim, color=colors['simulated'], linewidth=0.8, 
                  label='Simulated', alpha=0.9)
    ax_hydro.set_ylabel('Flow (ML/day)', fontsize=10)
    ax_hydro.set_title('Hydrograph Comparison (Linear Scale)', fontweight='bold', fontsize=11)
    ax_hydro.legend(loc='upper right', fontsize=9)
    ax_hydro.grid(True, alpha=0.3)
    ax_hydro.tick_params(labelsize=9)
    
    # ==========================================================================
    # Row 1, Col 2-3: Performance Metrics Table
    # ==========================================================================
    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.axis('off')
    ax_metrics.set_facecolor(panel_color)
    
    # Create metrics display
    metric_items = [
        ('NSE', 'Nash-Sutcliffe Efficiency', metrics.get('NSE', np.nan)),
        ('KGE', 'Kling-Gupta Efficiency', metrics.get('KGE', np.nan)),
        ('PBIAS', 'Percent Bias (%)', metrics.get('PBIAS', np.nan)),
        ('RMSE', 'Root Mean Square Error', metrics.get('RMSE', np.nan)),
        ('MAE', 'Mean Absolute Error', metrics.get('MAE', np.nan)),
        ('LogNSE', 'NSE (log-transformed)', metrics.get('LogNSE', np.nan)),
        ('SqrtNSE', 'NSE (sqrt-transformed)', metrics.get('SqrtNSE', np.nan)),
    ]
    
    ax_metrics.text(0.5, 0.98, 'Performance Metrics', transform=ax_metrics.transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='top', color=text_color)
    
    y_pos = 0.88
    for abbrev, full_name, value in metric_items:
        color = _get_metric_color(value, abbrev)
        
        if abbrev == 'PBIAS':
            val_str = f"{value:+.2f}%" if not np.isnan(value) else "N/A"
        elif abbrev in ['RMSE', 'MAE']:
            val_str = f"{value:.2f}" if not np.isnan(value) else "N/A"
        else:
            val_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
        
        # Metric name
        ax_metrics.text(0.05, y_pos, f"{abbrev}:", transform=ax_metrics.transAxes,
                        fontsize=10, fontweight='bold', ha='left', va='center', color=text_color)
        # Value with color
        ax_metrics.text(0.35, y_pos, val_str, transform=ax_metrics.transAxes,
                        fontsize=10, fontweight='bold', ha='left', va='center', color=color)
        # Description
        ax_metrics.text(0.55, y_pos, full_name, transform=ax_metrics.transAxes,
                        fontsize=8, ha='left', va='center', color=text_color, alpha=0.7)
        
        y_pos -= 0.12
    
    # Add rating legend
    ax_metrics.text(0.05, 0.05, "● Excellent (≥0.75)  ", transform=ax_metrics.transAxes,
                    fontsize=8, ha='left', va='center', color='#27ae60')
    ax_metrics.text(0.40, 0.05, "● Good (≥0.50)  ", transform=ax_metrics.transAxes,
                    fontsize=8, ha='left', va='center', color='#f39c12')
    ax_metrics.text(0.65, 0.05, "● Poor (<0.50)", transform=ax_metrics.transAxes,
                    fontsize=8, ha='left', va='center', color='#e74c3c')
    
    # ==========================================================================
    # Row 2, Col 0-1: Hydrograph (Log Scale)
    # ==========================================================================
    ax_hydro_log = fig.add_subplot(gs[2, :2])
    ax_hydro_log.set_facecolor(panel_color)
    
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    
    ax_hydro_log.semilogy(dates, obs_log, color=colors['observed'], linewidth=0.8,
                          label='Observed', alpha=0.9)
    ax_hydro_log.semilogy(dates, sim_log, color=colors['simulated'], linewidth=0.8,
                          label='Simulated', alpha=0.9)
    ax_hydro_log.set_ylabel('Flow (ML/day)', fontsize=10)
    ax_hydro_log.set_title('Hydrograph Comparison (Log Scale)', fontweight='bold', fontsize=11)
    ax_hydro_log.legend(loc='upper right', fontsize=9)
    ax_hydro_log.grid(True, alpha=0.3, which='both')
    ax_hydro_log.tick_params(labelsize=9)
    
    # ==========================================================================
    # Row 2, Col 2-3: KGE Components Breakdown
    # ==========================================================================
    ax_kge = fig.add_subplot(gs[2, 2:])
    ax_kge.set_facecolor(panel_color)
    
    kge_components = ['r (correlation)', 'α (variability)', 'β (bias)']
    kge_values = [
        metrics.get('KGE_r', np.nan),
        metrics.get('KGE_alpha', np.nan),
        metrics.get('KGE_beta', np.nan)
    ]
    kge_optimal = [1.0, 1.0, 1.0]
    
    y_pos_kge = np.arange(len(kge_components))
    bar_colors = [_get_metric_color(v, 'KGE_r' if i == 0 else 'KGE_alpha') for i, v in enumerate(kge_values)]
    
    bars = ax_kge.barh(y_pos_kge, kge_values, color=bar_colors, alpha=0.7, height=0.6)
    ax_kge.axvline(x=1.0, color='#27ae60', linestyle='--', linewidth=2, label='Optimal (1.0)')
    ax_kge.axvline(x=0.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax_kge.set_yticks(y_pos_kge)
    ax_kge.set_yticklabels(kge_components, fontsize=10)
    ax_kge.set_xlabel('Value', fontsize=10)
    ax_kge.set_title('KGE Components', fontweight='bold', fontsize=11)
    ax_kge.legend(loc='upper right', fontsize=9)
    ax_kge.set_xlim(0, max(2.0, max([v for v in kge_values if not np.isnan(v)]) * 1.1))
    ax_kge.tick_params(labelsize=9)
    
    # Add value labels
    for bar, val in zip(bars, kge_values):
        if not np.isnan(val):
            ax_kge.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # ==========================================================================
    # Row 3, Col 0: Flow Duration Curve
    # ==========================================================================
    ax_fdc = fig.add_subplot(gs[3, 0])
    ax_fdc.set_facecolor(panel_color)
    
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    ax_fdc.semilogy(exc_obs, obs_sorted, color=colors['observed'], linewidth=2,
                    label='Observed')
    ax_fdc.semilogy(exc_sim, sim_sorted, color=colors['simulated'], linewidth=2,
                    linestyle='--', label='Simulated')
    ax_fdc.set_xlabel('Exceedance (%)', fontsize=10)
    ax_fdc.set_ylabel('Flow (ML/day)', fontsize=10)
    ax_fdc.set_title('Flow Duration Curve', fontweight='bold', fontsize=11)
    ax_fdc.legend(loc='upper right', fontsize=9)
    ax_fdc.grid(True, alpha=0.3, which='both')
    ax_fdc.set_xlim(0, 100)
    ax_fdc.tick_params(labelsize=9)
    
    # ==========================================================================
    # Row 3, Col 1: Scatter Plot with Statistics
    # ==========================================================================
    ax_scatter = fig.add_subplot(gs[3, 1])
    ax_scatter.set_facecolor(panel_color)
    
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    ax_scatter.scatter(obs_valid, sim_valid, alpha=0.3, s=8, color=colors['simulated'])
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    ax_scatter.plot([0, max_val], [0, max_val], '--', color=colors['one_to_one'],
                    linewidth=2, label='1:1 Line')
    
    # Add regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    ax_scatter.plot([0, max_val], p([0, max_val]), ':', color=colors['observed'],
                    linewidth=2, label=f'Fit (y={z[0]:.2f}x+{z[1]:.1f})')
    
    ax_scatter.set_xlabel('Observed (ML/day)', fontsize=10)
    ax_scatter.set_ylabel('Simulated (ML/day)', fontsize=10)
    ax_scatter.set_title('Scatter Plot', fontweight='bold', fontsize=11)
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.legend(loc='upper left', fontsize=8)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.tick_params(labelsize=9)
    
    # Add R² annotation
    r_squared = metrics.get('KGE_r', np.nan) ** 2 if not np.isnan(metrics.get('KGE_r', np.nan)) else np.nan
    if not np.isnan(r_squared):
        ax_scatter.text(0.95, 0.05, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes,
                       fontsize=10, ha='right', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ==========================================================================
    # Row 3, Col 2-3: Parameter Bounds Chart
    # ==========================================================================
    ax_bounds = fig.add_subplot(gs[3, 2:])
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
                # Color based on position (red if hitting bounds)
                if norm < 5 or norm > 95:
                    bar_colors.append('#e74c3c')  # Red - hitting bounds
                elif norm < 15 or norm > 85:
                    bar_colors.append('#f39c12')  # Orange - near bounds
                else:
                    bar_colors.append(colors['simulated'])  # Normal
            else:
                norm_values.append(50)
                bar_colors.append(colors['simulated'])
        
        y_pos = np.arange(len(param_names))
        bars = ax_bounds.barh(y_pos, norm_values, color=bar_colors, alpha=0.7, height=0.7)
        ax_bounds.axvline(x=50, linestyle='--', color='gray', alpha=0.5, linewidth=1)
        ax_bounds.axvline(x=0, linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
        ax_bounds.axvline(x=100, linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
        
        # Add value labels
        for bar, val, norm in zip(bars, actual_values, norm_values):
            label_x = norm + 2 if norm < 85 else norm - 3
            ha = 'left' if norm < 85 else 'right'
            ax_bounds.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', ha=ha, fontsize=7, fontweight='bold')
        
        ax_bounds.set_yticks(y_pos)
        ax_bounds.set_yticklabels(param_names, fontsize=8)
        ax_bounds.set_xlabel('Position within Bounds (%)', fontsize=10)
        ax_bounds.set_xlim(-5, 105)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=11)
        ax_bounds.invert_yaxis()
        ax_bounds.tick_params(labelsize=9)
        
        # Add legend for colors
        ax_bounds.text(0.02, -0.08, '● At bounds', transform=ax_bounds.transAxes,
                      fontsize=7, color='#e74c3c')
        ax_bounds.text(0.22, -0.08, '● Near bounds', transform=ax_bounds.transAxes,
                      fontsize=7, color='#f39c12')
        ax_bounds.text(0.45, -0.08, '● Normal', transform=ax_bounds.transAxes,
                      fontsize=7, color=colors['simulated'])
    else:
        ax_bounds.text(0.5, 0.5, 'No parameter bounds\navailable', ha='center', va='center',
                       transform=ax_bounds.transAxes, fontsize=12)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=11)
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
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
    height: int = 1200
):
    """
    Generate an interactive Plotly report card mirroring the matplotlib template.
    
    Layout (4 rows x 4 columns - matching matplotlib):
    ┌──────────────────────────────────────────────┬────────────────────────┐
    │  HEADER: Catchment Name, Method, Objective   │                        │
    ├──────────────────────────────────────────────┼────────────────────────┤
    │  Hydrograph (Linear Scale)                   │  Performance Metrics   │
    │                                              │  (color-coded table)   │
    ├──────────────────────────────────────────────┼────────────────────────┤
    │  Hydrograph (Log Scale)                      │  KGE Components        │
    │                                              │  (visual breakdown)    │
    ├───────────────────┬──────────────────────────┼────────────────────────┤
    │  Flow Duration    │  Scatter Plot            │  Parameter Bounds      │
    │  Curve            │  (1:1 line + metrics)    │  (horizontal bars)     │
    └───────────────────┴──────────────────────────┴────────────────────────┘
    
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
    
    # Calculate metrics using robust fallback
    metrics = _calculate_basic_metrics(obs, sim)
    
    # Colors matching matplotlib template
    color_obs = '#e74c3c'
    color_sim = '#3498db'
    color_excellent = '#27ae60'
    color_good = '#f39c12'
    color_poor = '#e74c3c'
    
    # Create subplots matching matplotlib layout
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=(
            '', '', '', '',  # Row 1: Header (no titles)
            'Hydrograph (Linear Scale)', '', 'Performance Metrics', '',
            'Hydrograph (Log Scale)', '', 'KGE Components', '',
            'Flow Duration Curve', 'Scatter Plot', 'Calibrated Parameters vs Bounds', ''
        ),
        row_heights=[0.08, 0.28, 0.28, 0.36],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        specs=[
            [{"type": "scatter", "colspan": 4}, None, None, None],  # Header placeholder
            [{"type": "scatter", "colspan": 2}, None, {"type": "table", "colspan": 2}, None],  # Hydro + Metrics
            [{"type": "scatter", "colspan": 2}, None, {"type": "bar", "colspan": 2}, None],  # Hydro Log + KGE
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar", "colspan": 2}, None]  # FDC + Scatter + Params
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.05
    )
    
    # ==========================================================================
    # Row 1: Header (using annotations instead of trace)
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
    # Row 2, Col 1-2: Hydrograph (Linear Scale)
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
    # Row 2, Col 3-4: Performance Metrics Table
    # ==========================================================================
    def get_color_for_metric(value, metric_name):
        if np.isnan(value):
            return '#888888'
        if metric_name in ['NSE', 'KGE', 'LogNSE', 'SqrtNSE']:
            if value >= 0.75: return color_excellent
            elif value >= 0.5: return color_good
            else: return color_poor
        elif metric_name == 'PBIAS':
            if abs(value) <= 10: return color_excellent
            elif abs(value) <= 25: return color_good
            else: return color_poor
        return '#2c3e50'
    
    metric_items = [
        ('NSE', 'Nash-Sutcliffe Efficiency', metrics.get('NSE', np.nan)),
        ('KGE', 'Kling-Gupta Efficiency', metrics.get('KGE', np.nan)),
        ('PBIAS', 'Percent Bias (%)', metrics.get('PBIAS', np.nan)),
        ('RMSE', 'Root Mean Square Error', metrics.get('RMSE', np.nan)),
        ('MAE', 'Mean Absolute Error', metrics.get('MAE', np.nan)),
        ('LogNSE', 'NSE (log-transformed)', metrics.get('LogNSE', np.nan)),
        ('SqrtNSE', 'NSE (sqrt-transformed)', metrics.get('SqrtNSE', np.nan)),
    ]
    
    metric_abbrevs = [m[0] for m in metric_items]
    metric_descs = [m[1] for m in metric_items]
    metric_values = []
    metric_colors = []
    for abbrev, desc, value in metric_items:
        if abbrev == 'PBIAS':
            metric_values.append(f"{value:+.2f}%" if not np.isnan(value) else "N/A")
        elif abbrev in ['RMSE', 'MAE']:
            metric_values.append(f"{value:.2f}" if not np.isnan(value) else "N/A")
        else:
            metric_values.append(f"{value:.4f}" if not np.isnan(value) else "N/A")
        metric_colors.append(get_color_for_metric(value, abbrev))
    
    # Create colored cells for values
    cell_colors = [['#f8f9fa'] * len(metric_items),  # Abbrev column
                   [c for c in metric_colors],  # Value column (colored)
                   ['#f8f9fa'] * len(metric_items)]  # Description column
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>', '<b>Description</b>'],
                fill_color='#34495e',
                font=dict(color='white', size=11),
                align='left',
                height=30
            ),
            cells=dict(
                values=[metric_abbrevs, metric_values, metric_descs],
                fill_color=cell_colors,
                font=dict(color=['#2c3e50', 'white', '#7f8c8d'], size=10),
                align='left',
                height=25
            ),
            columnwidth=[0.15, 0.2, 0.65]
        ),
        row=2, col=3
    )
    
    # ==========================================================================
    # Row 3, Col 1-2: Hydrograph (Log Scale)
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
    
    # ==========================================================================
    # Row 3, Col 3-4: KGE Components Breakdown
    # ==========================================================================
    kge_components = ['r (correlation)', 'α (variability)', 'β (bias)']
    kge_values = [
        metrics.get('KGE_r', np.nan),
        metrics.get('KGE_alpha', np.nan),
        metrics.get('KGE_beta', np.nan)
    ]
    kge_colors = []
    for i, v in enumerate(kge_values):
        if np.isnan(v):
            kge_colors.append('#888888')
        elif i == 0:  # correlation
            kge_colors.append(color_excellent if v >= 0.75 else color_good if v >= 0.5 else color_poor)
        else:  # alpha and beta
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
        row=3, col=3
    )
    # Add optimal line at 1.0 using a shape (works better with subplots)
    kge_max = max(2.0, max([v for v in kge_values if not np.isnan(v)], default=1) * 1.2)
    fig.add_trace(
        go.Scatter(
            x=[1.0, 1.0], y=[-0.5, 2.5],
            mode='lines',
            line=dict(color=color_excellent, dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=3, col=3
    )
    fig.update_xaxes(range=[0, kge_max], title_text="Value", row=3, col=3)
    fig.update_yaxes(range=[-0.5, 2.5], row=3, col=3)
    
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
    # Row 4, Col 2: Scatter Plot with Statistics
    # ==========================================================================
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    fig.add_trace(
        go.Scatter(x=obs_valid, y=sim_valid, mode='markers',
                   marker=dict(color=color_sim, size=4, opacity=0.3),
                   name='Points', showlegend=False,
                   hovertemplate='Observed: %{x:.1f}<br>Simulated: %{y:.1f}<extra></extra>'),
        row=4, col=2
    )
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    # 1:1 line
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                   line=dict(color='green', dash='dash', width=2),
                   name='1:1 Line', showlegend=False),
        row=4, col=2
    )
    
    # Regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=p([0, max_val]), mode='lines',
                   line=dict(color=color_obs, dash='dot', width=2),
                   name=f'Fit (y={z[0]:.2f}x+{z[1]:.1f})', showlegend=False),
        row=4, col=2
    )
    
    fig.update_xaxes(title_text="Observed (ML/day)", row=4, col=2)
    fig.update_yaxes(title_text="Simulated (ML/day)", row=4, col=2)
    
    # Add R² annotation
    r_squared = metrics.get('KGE_r', np.nan) ** 2 if not np.isnan(metrics.get('KGE_r', np.nan)) else np.nan
    if not np.isnan(r_squared):
        fig.add_annotation(
            x=0.95, y=0.05, xref='x4 domain', yref='y4 domain',
            text=f"<b>R² = {r_squared:.3f}</b>",
            showarrow=False, font=dict(size=11),
            bgcolor='white', bordercolor='gray', borderwidth=1
        )
    
    # ==========================================================================
    # Row 4, Col 3-4: Parameter Bounds Chart
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
                # Color based on position
                if norm < 5 or norm > 95:
                    bar_colors.append(color_poor)  # At bounds
                elif norm < 15 or norm > 85:
                    bar_colors.append(color_good)  # Near bounds
                else:
                    bar_colors.append(color_sim)  # Normal
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
            row=4, col=3
        )
        # Add center line at 50% using scatter trace (works better with subplots)
        n_params = len(param_names)
        fig.add_trace(
            go.Scatter(
                x=[50, 50], y=[-0.5, n_params - 0.5],
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=4, col=3
        )
        fig.update_xaxes(range=[-5, 105], title_text="Position within Bounds (%)", row=4, col=3)
        fig.update_yaxes(range=[-0.5, n_params - 0.5], row=4, col=3)
    
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
            y=0.98, 
            x=0.25, 
            xanchor='center',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        font=dict(family='Arial, sans-serif')
    )
    
    # Update all subplot backgrounds
    for i in range(1, 5):
        for j in range(1, 5):
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
