"""
Calibration visualization functions.

This module provides plotting functions for visualizing calibration results,
including MCMC diagnostics, posterior distributions, and parameter exploration.
"""

from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pyrrm.calibration.runner import CalibrationResult

# Dark theme colors
COLORS = {
    'trace': '#4ecdc4',
    'hist': '#45b7d1',
    'scatter': '#f7dc6f',
    'best': '#ff6b6b',
    'kde': '#4ecdc4',
}


def _apply_dark_style():
    """Apply dark style."""
    plt.rcParams.update({
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
    })


def plot_mcmc_traces(
    result: 'CalibrationResult',
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dark_theme: bool = True
) -> Figure:
    """
    Plot MCMC parameter traces for convergence assessment.
    
    Args:
        result: CalibrationResult from calibration
        params: List of parameters to plot (None for all)
        figsize: Figure size (auto-sized if None)
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    samples = result.all_samples
    
    # Determine parameters to plot
    if params is None:
        params = [col for col in samples.columns 
                  if col not in ['iteration', 'likelihood', 'objective']]
    
    n_params = len(params)
    
    if figsize is None:
        figsize = (12, 2.5 * n_params)
    
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    if n_params == 1:
        axes = [axes]
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
    
    for ax, param in zip(axes, params):
        if dark_theme:
            ax.set_facecolor('#16213e')
        
        if param in samples.columns:
            values = samples[param].values
            ax.plot(values, color=COLORS['trace'], alpha=0.7, linewidth=0.5)
            
            # Add best value line
            if param in result.best_parameters:
                ax.axhline(y=result.best_parameters[param], 
                          color=COLORS['best'], linestyle='--', 
                          linewidth=1.5, label='Best')
            
            ax.set_ylabel(param, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Iteration', fontsize=11)
    fig.suptitle('MCMC Parameter Traces', fontsize=13, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    return fig


def plot_posterior_distributions(
    result: 'CalibrationResult',
    params: Optional[List[str]] = None,
    burnin: float = 0.5,
    figsize: Optional[Tuple[int, int]] = None,
    dark_theme: bool = True
) -> Figure:
    """
    Plot posterior parameter distributions from MCMC.
    
    Args:
        result: CalibrationResult from calibration
        params: List of parameters to plot (None for all)
        burnin: Fraction of samples to discard as burn-in
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    samples = result.all_samples
    
    # Remove burn-in
    n_burnin = int(len(samples) * burnin)
    samples = samples.iloc[n_burnin:]
    
    if params is None:
        params = [col for col in samples.columns 
                  if col not in ['iteration', 'likelihood', 'objective']]
    
    n_params = len(params)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
    
    for i, (ax, param) in enumerate(zip(axes[:n_params], params)):
        if dark_theme:
            ax.set_facecolor('#16213e')
        
        if param in samples.columns:
            values = samples[param].dropna().values
            
            # Histogram
            ax.hist(values, bins=50, density=True, alpha=0.7, 
                   color=COLORS['hist'], edgecolor='white', linewidth=0.5)
            
            # KDE
            try:
                from scipy import stats
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                ax.plot(x_range, kde(x_range), color=COLORS['kde'], linewidth=2)
            except Exception:
                pass
            
            # Best value
            if param in result.best_parameters:
                ax.axvline(x=result.best_parameters[param], 
                          color=COLORS['best'], linestyle='--', linewidth=2)
            
            # Statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.set_title(f'{param}\n{mean_val:.3f} ± {std_val:.3f}', fontsize=10)
    
    # Hide empty axes
    for ax in axes[n_params:]:
        ax.set_visible(False)
    
    fig.suptitle('Posterior Distributions', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_parameter_correlations(
    result: 'CalibrationResult',
    params: Optional[List[str]] = None,
    burnin: float = 0.5,
    figsize: Optional[Tuple[int, int]] = None,
    dark_theme: bool = True
) -> Figure:
    """
    Pairwise parameter correlation plots.
    
    Args:
        result: CalibrationResult
        params: Parameters to include
        burnin: Burn-in fraction
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    samples = result.all_samples
    
    # Remove burn-in
    n_burnin = int(len(samples) * burnin)
    samples = samples.iloc[n_burnin:]
    
    if params is None:
        params = [col for col in samples.columns 
                  if col not in ['iteration', 'likelihood', 'objective']]
    
    n_params = len(params)
    
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if dark_theme:
                ax.set_facecolor('#16213e')
            
            pi = params[i]
            pj = params[j]
            
            if pi in samples.columns and pj in samples.columns:
                if i == j:
                    # Diagonal: histogram
                    ax.hist(samples[pi].values, bins=30, color=COLORS['hist'], 
                           alpha=0.7, edgecolor='white')
                else:
                    # Off-diagonal: scatter
                    ax.scatter(samples[pj].values, samples[pi].values, 
                              alpha=0.1, s=1, color=COLORS['scatter'])
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(pj, fontsize=9)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(pi, fontsize=9)
            else:
                ax.set_yticklabels([])
    
    fig.suptitle('Parameter Correlations', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    return fig


def plot_objective_evolution(
    result: 'CalibrationResult',
    figsize: Tuple[int, int] = (12, 5),
    dark_theme: bool = True
) -> Figure:
    """
    Plot objective function improvement over iterations.
    
    Args:
        result: CalibrationResult
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax1.set_facecolor('#16213e')
        ax2.set_facecolor('#16213e')
    
    samples = result.all_samples
    
    # Find objective column
    obj_col = 'likelihood' if 'likelihood' in samples.columns else 'objective'
    if obj_col not in samples.columns:
        obj_col = [c for c in samples.columns if 'like' in c.lower() or 'obj' in c.lower()]
        obj_col = obj_col[0] if obj_col else samples.columns[-1]
    
    iterations = samples['iteration'].values if 'iteration' in samples.columns else np.arange(len(samples))
    objectives = samples[obj_col].values
    
    # Left: All samples
    ax1.scatter(iterations, objectives, alpha=0.3, s=2, color=COLORS['scatter'])
    ax1.axhline(y=result.best_objective, color=COLORS['best'], 
               linestyle='--', linewidth=2, label=f'Best: {result.best_objective:.4f}')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel(result.objective_name, fontsize=11)
    ax1.set_title('Objective Function Values', fontweight='bold')
    ax1.legend()
    
    # Right: Running best
    running_best = np.zeros_like(objectives)
    running_best[0] = objectives[0]
    for i in range(1, len(objectives)):
        running_best[i] = max(running_best[i-1], objectives[i])
    
    ax2.plot(iterations, running_best, color=COLORS['trace'], linewidth=2)
    ax2.axhline(y=result.best_objective, color=COLORS['best'], 
               linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel(result.objective_name, fontsize=11)
    ax2.set_title('Running Best Objective', fontweight='bold')
    
    fig.suptitle(f'Optimization Progress ({result.method})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_dotty_plots(
    result: 'CalibrationResult',
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dark_theme: bool = True
) -> Figure:
    """
    Dotty plots showing parameter sensitivity.
    
    Shows objective value vs parameter value for all samples.
    
    Args:
        result: CalibrationResult
        params: Parameters to plot
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    samples = result.all_samples
    
    if params is None:
        params = [col for col in samples.columns 
                  if col not in ['iteration', 'likelihood', 'objective']]
    
    # Find objective column
    obj_col = 'likelihood' if 'likelihood' in samples.columns else 'objective'
    if obj_col not in samples.columns:
        obj_col = [c for c in samples.columns if 'like' in c.lower() or 'obj' in c.lower()]
        obj_col = obj_col[0] if obj_col else None
    
    if obj_col is None:
        raise ValueError("Could not find objective column in results")
    
    n_params = len(params)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
    
    objectives = samples[obj_col].values
    
    for i, (ax, param) in enumerate(zip(axes[:n_params], params)):
        if dark_theme:
            ax.set_facecolor('#16213e')
        
        if param in samples.columns:
            values = samples[param].values
            
            # Scatter with color by objective
            scatter = ax.scatter(values, objectives, c=objectives, 
                                cmap='viridis', alpha=0.3, s=5)
            
            # Best value
            if param in result.best_parameters:
                ax.axvline(x=result.best_parameters[param], 
                          color=COLORS['best'], linestyle='--', linewidth=2)
            
            ax.set_xlabel(param, fontsize=10)
            ax.set_ylabel(result.objective_name, fontsize=10)
    
    # Hide empty axes
    for ax in axes[n_params:]:
        ax.set_visible(False)
    
    fig.suptitle('Dotty Plots (Parameter Sensitivity)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


# =============================================================================
# FUNCTION ALIASES
# =============================================================================
# These aliases provide alternative names for backward compatibility
# and consistency with different naming conventions.

# Trace plots
plot_parameter_traces = plot_mcmc_traces

# Histogram/posterior plots
plot_parameter_histograms = plot_posterior_distributions

# Objective function evolution
plot_objective_function_trace = plot_objective_evolution

# Dotty plots (shorter alias)
plot_dotty = plot_dotty_plots
