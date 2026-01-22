"""
Model output visualization functions.

This module provides plotting functions for rainfall-runoff model outputs,
including hydrographs, flow duration curves, scatter plots, and more.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Dark theme styling
DARK_STYLE = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eaeaea',
    'text.color': '#eaeaea',
    'xtick.color': '#eaeaea',
    'ytick.color': '#eaeaea',
    'grid.color': '#0f3460',
    'legend.facecolor': '#16213e',
    'legend.edgecolor': '#e94560',
}

# Color palette
COLORS = {
    'observed': '#ff6b6b',      # Coral red
    'simulated': '#4ecdc4',     # Turquoise
    'precipitation': '#45b7d1', # Sky blue
    'residual': '#f7dc6f',      # Yellow
    'fill': '#1a1a2e',          # Dark fill
}


def _apply_dark_style():
    """Apply dark style to matplotlib."""
    for key, value in DARK_STYLE.items():
        plt.rcParams[key] = value


def plot_hydrograph_with_precipitation(
    dates: pd.DatetimeIndex,
    observed: np.ndarray,
    simulated: np.ndarray,
    precipitation: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dark_theme: bool = True,
    show_metrics: bool = True
) -> Figure:
    """
    Create dual-panel hydrograph with precipitation.
    
    Layout:
    - TOP PANEL: Precipitation as inverted bar chart
    - BOTTOM PANEL: Observed vs Simulated flow
    
    Args:
        dates: DatetimeIndex for x-axis
        observed: Observed flow values
        simulated: Simulated flow values
        precipitation: Precipitation values
        title: Plot title
        figsize: Figure size (width, height)
        dark_theme: Use dark background theme
        show_metrics: Display performance metrics
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, (ax_precip, ax_flow) = plt.subplots(
        2, 1, 
        figsize=figsize,
        height_ratios=[1, 3],
        sharex=True,
        gridspec_kw={'hspace': 0.05}
    )
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax_precip.set_facecolor('#16213e')
        ax_flow.set_facecolor('#16213e')
    
    # Top panel: Precipitation (inverted)
    ax_precip.bar(
        dates, precipitation, 
        color=COLORS['precipitation'], 
        alpha=0.8,
        width=1.0
    )
    ax_precip.invert_yaxis()
    ax_precip.set_ylabel('Precipitation\n(mm)', fontsize=10)
    ax_precip.set_ylim(bottom=0)
    
    # Remove top spine for precipitation
    ax_precip.spines['bottom'].set_visible(False)
    ax_precip.tick_params(bottom=False)
    
    # Bottom panel: Flow
    ax_flow.plot(
        dates, observed, 
        color=COLORS['observed'], 
        linewidth=1.5, 
        label='Observed',
        alpha=0.9
    )
    ax_flow.plot(
        dates, simulated, 
        color=COLORS['simulated'], 
        linewidth=1.5, 
        label='Simulated',
        alpha=0.9
    )
    
    ax_flow.set_ylabel('Flow (mm/d)', fontsize=11)
    ax_flow.set_xlabel('Date', fontsize=11)
    ax_flow.legend(loc='upper right', framealpha=0.9)
    
    # Format x-axis dates
    ax_flow.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_flow.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add metrics annotation
    if show_metrics:
        from pyrrm.calibration.objective_functions import calculate_metrics
        
        valid = ~(np.isnan(observed) | np.isnan(simulated))
        metrics = calculate_metrics(simulated[valid], observed[valid])
        
        metrics_text = (
            f"NSE: {metrics['NSE']:.3f}\n"
            f"KGE: {metrics['KGE']:.3f}\n"
            f"PBIAS: {metrics['PBIAS']:.1f}%"
        )
        
        ax_flow.text(
            0.02, 0.98, metrics_text,
            transform=ax_flow.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle='round',
                facecolor='#16213e' if dark_theme else 'white',
                edgecolor=COLORS['observed'],
                alpha=0.9
            )
        )
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig


def plot_hydrograph_simple(
    observed: pd.Series,
    simulated: pd.Series,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    dark_theme: bool = True
) -> Figure:
    """
    Simple observed vs simulated hydrograph.
    
    Args:
        observed: Observed flow Series with DatetimeIndex
        simulated: Simulated flow Series with DatetimeIndex
        title: Plot title
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
    
    ax.plot(observed.index, observed.values, color=COLORS['observed'], 
            label='Observed', linewidth=1.2, alpha=0.9)
    ax.plot(simulated.index, simulated.values, color=COLORS['simulated'], 
            label='Simulated', linewidth=1.2, alpha=0.9)
    
    ax.set_ylabel('Flow (mm/d)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='upper right')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_flow_duration_curve(
    observed: np.ndarray,
    simulated: np.ndarray,
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dark_theme: bool = True
) -> Figure:
    """
    Plot flow duration curves.
    
    Args:
        observed: Observed flow values
        simulated: Simulated flow values
        log_scale: Use logarithmic y-axis
        title: Plot title
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
    
    # Sort and calculate exceedance probabilities
    obs_sorted = np.sort(observed[~np.isnan(observed)])[::-1]
    sim_sorted = np.sort(simulated[~np.isnan(simulated)])[::-1]
    
    obs_exceedance = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    sim_exceedance = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    ax.plot(obs_exceedance, obs_sorted, color=COLORS['observed'], 
            label='Observed', linewidth=2)
    ax.plot(sim_exceedance, sim_sorted, color=COLORS['simulated'], 
            label='Simulated', linewidth=2, linestyle='--')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Exceedance Probability (%)', fontsize=11)
    ax.set_ylabel('Flow (mm/d)', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title or 'Flow Duration Curve', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_scatter_with_metrics(
    observed: np.ndarray,
    simulated: np.ndarray,
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    dark_theme: bool = True
) -> Figure:
    """
    Scatter plot with 1:1 line and performance metrics.
    
    Args:
        observed: Observed values
        simulated: Simulated values
        metrics: List of metrics to display
        title: Plot title
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    metrics = metrics or ['NSE', 'KGE', 'RMSE', 'PBIAS']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
    
    # Filter valid data
    valid = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[valid]
    sim = simulated[valid]
    
    # Scatter plot
    ax.scatter(obs, sim, alpha=0.4, s=10, color=COLORS['simulated'])
    
    # 1:1 line
    max_val = max(np.max(obs), np.max(sim))
    min_val = min(np.min(obs), np.min(sim))
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='1:1 Line', alpha=0.8)
    
    # Regression line
    z = np.polyfit(obs, sim, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], p([min_val, max_val]), 
            color=COLORS['simulated'], linewidth=2, label='Regression', linestyle=':')
    
    ax.set_xlabel('Observed Flow (mm/d)', fontsize=11)
    ax.set_ylabel('Simulated Flow (mm/d)', fontsize=11)
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    # Metrics annotation
    from pyrrm.calibration.objective_functions import calculate_metrics
    all_metrics = calculate_metrics(sim, obs)
    
    metrics_lines = [f"{m}: {all_metrics.get(m, np.nan):.3f}" for m in metrics]
    metrics_text = "\n".join(metrics_lines)
    
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='#16213e' if dark_theme else 'white',
            edgecolor=COLORS['observed'],
            alpha=0.9
        )
    )
    
    if title:
        ax.set_title(title or 'Observed vs Simulated', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_residuals(
    observed: np.ndarray,
    simulated: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    figsize: Tuple[int, int] = (14, 10),
    dark_theme: bool = True
) -> Figure:
    """
    Residual analysis plots.
    
    Creates 4-panel figure with:
    - Residuals vs time
    - Residuals vs observed magnitude  
    - Residual histogram
    - Q-Q plot
    
    Args:
        observed: Observed values
        simulated: Simulated values
        dates: Optional DatetimeIndex
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
    
    valid = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[valid]
    sim = simulated[valid]
    residuals = sim - obs
    
    # Residuals vs time
    ax = axes[0, 0]
    x = dates[valid] if dates is not None else np.arange(len(residuals))
    ax.scatter(x, residuals, alpha=0.3, s=5, color=COLORS['residual'])
    ax.axhline(y=0, color=COLORS['observed'], linestyle='--', linewidth=1)
    ax.set_xlabel('Time' if dates is not None else 'Index')
    ax.set_ylabel('Residual (mm/d)')
    ax.set_title('Residuals vs Time', fontweight='bold')
    
    # Residuals vs magnitude
    ax = axes[0, 1]
    ax.scatter(obs, residuals, alpha=0.3, s=5, color=COLORS['residual'])
    ax.axhline(y=0, color=COLORS['observed'], linestyle='--', linewidth=1)
    ax.set_xlabel('Observed Flow (mm/d)')
    ax.set_ylabel('Residual (mm/d)')
    ax.set_title('Residuals vs Magnitude', fontweight='bold')
    
    # Histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, color=COLORS['simulated'], alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color=COLORS['observed'], linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(residuals), color='white', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(residuals):.3f}')
    ax.set_xlabel('Residual (mm/d)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.legend()
    
    # Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, alpha=0.5, s=10, color=COLORS['simulated'])
    ax.plot(osm, slope * osm + intercept, 'r-', linewidth=2)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot (Normal)', fontweight='bold')
    
    fig.suptitle('Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_monthly_boxplot(
    dates: pd.DatetimeIndex,
    observed: np.ndarray,
    simulated: np.ndarray,
    figsize: Tuple[int, int] = (12, 6),
    dark_theme: bool = True
) -> Figure:
    """
    Monthly boxplot comparison.
    
    Args:
        dates: DatetimeIndex
        observed: Observed values
        simulated: Simulated values
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
    
    df = pd.DataFrame({
        'observed': observed,
        'simulated': simulated,
        'month': dates.month
    })
    
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    positions_obs = np.array(months) - 0.2
    positions_sim = np.array(months) + 0.2
    
    obs_data = [df[df['month'] == m]['observed'].dropna().values for m in months]
    sim_data = [df[df['month'] == m]['simulated'].dropna().values for m in months]
    
    bp1 = ax.boxplot(obs_data, positions=positions_obs, widths=0.35,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(sim_data, positions=positions_sim, widths=0.35,
                     patch_artist=True, showfliers=False)
    
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['observed'])
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['simulated'])
        patch.set_alpha(0.7)
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Flow (mm/d)', fontsize=11)
    ax.set_title('Monthly Flow Distribution', fontsize=13, fontweight='bold')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Observed', 'Simulated'])
    
    plt.tight_layout()
    return fig


def create_calibration_dashboard(
    result: Any,  # CalibrationResult
    dates: pd.DatetimeIndex,
    observed: np.ndarray,
    simulated: np.ndarray,
    precipitation: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 12),
    dark_theme: bool = True
) -> Figure:
    """
    Comprehensive calibration summary dashboard.
    
    Args:
        result: CalibrationResult object
        dates: DatetimeIndex
        observed: Observed flow
        simulated: Simulated flow
        precipitation: Precipitation (optional)
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig = plt.figure(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Hydrograph (top row, spans 2 columns)
    ax_hydro = fig.add_subplot(gs[0, :2])
    if dark_theme:
        ax_hydro.set_facecolor('#16213e')
    
    ax_hydro.plot(dates, observed, color=COLORS['observed'], label='Observed', linewidth=1.2)
    ax_hydro.plot(dates, simulated, color=COLORS['simulated'], label='Simulated', linewidth=1.2)
    ax_hydro.set_ylabel('Flow (mm/d)')
    ax_hydro.legend(loc='upper right')
    ax_hydro.set_title('Hydrograph Comparison', fontweight='bold')
    
    # Parameters (top right)
    ax_params = fig.add_subplot(gs[0, 2])
    ax_params.axis('off')
    
    params_text = "Best Parameters:\n" + "─" * 25 + "\n"
    for name, value in result.best_parameters.items():
        params_text += f"{name}: {value:.4f}\n"
    params_text += "─" * 25 + f"\n{result.objective_name}: {result.best_objective:.4f}"
    
    ax_params.text(
        0.1, 0.9, params_text,
        transform=ax_params.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#16213e' if dark_theme else 'white',
                  edgecolor=COLORS['observed'])
    )
    
    # Scatter plot (middle left)
    ax_scatter = fig.add_subplot(gs[1, 0])
    if dark_theme:
        ax_scatter.set_facecolor('#16213e')
    
    valid = ~(np.isnan(observed) | np.isnan(simulated))
    ax_scatter.scatter(observed[valid], simulated[valid], alpha=0.3, s=5, color=COLORS['simulated'])
    max_val = max(np.max(observed[valid]), np.max(simulated[valid]))
    ax_scatter.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
    ax_scatter.set_xlabel('Observed')
    ax_scatter.set_ylabel('Simulated')
    ax_scatter.set_title('Scatter Plot', fontweight='bold')
    ax_scatter.set_aspect('equal')
    
    # FDC (middle center)
    ax_fdc = fig.add_subplot(gs[1, 1])
    if dark_theme:
        ax_fdc.set_facecolor('#16213e')
    
    obs_sorted = np.sort(observed[~np.isnan(observed)])[::-1]
    sim_sorted = np.sort(simulated[~np.isnan(simulated)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    ax_fdc.semilogy(exc_obs, obs_sorted, color=COLORS['observed'], label='Observed')
    ax_fdc.semilogy(exc_sim, sim_sorted, color=COLORS['simulated'], label='Simulated', linestyle='--')
    ax_fdc.set_xlabel('Exceedance (%)')
    ax_fdc.set_ylabel('Flow (mm/d)')
    ax_fdc.set_title('Flow Duration Curve', fontweight='bold')
    ax_fdc.legend()
    ax_fdc.grid(True, alpha=0.3)
    
    # Residual histogram (middle right)
    ax_hist = fig.add_subplot(gs[1, 2])
    if dark_theme:
        ax_hist.set_facecolor('#16213e')
    
    residuals = simulated[valid] - observed[valid]
    ax_hist.hist(residuals, bins=40, color=COLORS['simulated'], alpha=0.7, edgecolor='white')
    ax_hist.axvline(x=0, color=COLORS['observed'], linestyle='--', linewidth=2)
    ax_hist.set_xlabel('Residual (mm/d)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Residual Distribution', fontweight='bold')
    
    # Monthly performance (bottom, spans all columns)
    ax_monthly = fig.add_subplot(gs[2, :])
    if dark_theme:
        ax_monthly.set_facecolor('#16213e')
    
    df = pd.DataFrame({
        'observed': observed,
        'simulated': simulated,
        'month': dates.month
    })
    
    months = list(range(1, 13))
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    obs_monthly = [df[df['month'] == m]['observed'].mean() for m in months]
    sim_monthly = [df[df['month'] == m]['simulated'].mean() for m in months]
    
    x = np.arange(12)
    width = 0.35
    ax_monthly.bar(x - width/2, obs_monthly, width, label='Observed', color=COLORS['observed'], alpha=0.7)
    ax_monthly.bar(x + width/2, sim_monthly, width, label='Simulated', color=COLORS['simulated'], alpha=0.7)
    ax_monthly.set_xticks(x)
    ax_monthly.set_xticklabels(month_names)
    ax_monthly.set_xlabel('Month')
    ax_monthly.set_ylabel('Mean Flow (mm/d)')
    ax_monthly.set_title('Monthly Mean Comparison', fontweight='bold')
    ax_monthly.legend()
    
    fig.suptitle(f'Calibration Results ({result.method})', fontsize=14, fontweight='bold', y=0.98)
    
    return fig
