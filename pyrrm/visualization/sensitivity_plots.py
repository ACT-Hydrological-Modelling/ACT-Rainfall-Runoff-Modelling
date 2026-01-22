"""
Sensitivity analysis visualization functions.

This module provides plotting functions for visualizing Sobol
sensitivity analysis results.
"""

from typing import Optional, Tuple, List, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pyrrm.analysis.sensitivity import SobolResult

# Colors
COLORS = {
    's1': '#4ecdc4',      # First-order
    'st': '#ff6b6b',      # Total-order
    'interaction': '#f7dc6f',
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


def plot_sobol_indices(
    result: 'SobolResult',
    figsize: Tuple[int, int] = (10, 6),
    dark_theme: bool = True
) -> Figure:
    """
    Bar chart of first-order and total-order Sobol indices.
    
    Args:
        result: SobolResult from sensitivity analysis
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
    
    # Sort parameters by total sensitivity
    params = sorted(result.parameter_names, 
                   key=lambda p: result.ST.get(p, 0), reverse=True)
    
    s1_values = [result.S1.get(p, 0) for p in params]
    s1_conf = [result.S1_conf.get(p, 0) for p in params]
    st_values = [result.ST.get(p, 0) for p in params]
    st_conf = [result.ST_conf.get(p, 0) for p in params]
    
    x = np.arange(len(params))
    width = 0.35
    
    # First-order indices
    ax.bar(x - width/2, s1_values, width, yerr=s1_conf, 
           label='First-Order (S1)', color=COLORS['s1'], 
           alpha=0.8, capsize=3)
    
    # Total-order indices
    ax.bar(x + width/2, st_values, width, yerr=st_conf,
           label='Total-Order (ST)', color=COLORS['st'],
           alpha=0.8, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_ylabel('Sensitivity Index', fontsize=11)
    ax.set_xlabel('Parameter', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(st_values) * 1.2, 1.0))
    
    ax.set_title('Sobol Sensitivity Indices', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_parameter_importance_ranking(
    result: 'SobolResult',
    figsize: Tuple[int, int] = (10, 6),
    dark_theme: bool = True
) -> Figure:
    """
    Horizontal bar chart showing ranked parameter importance.
    
    Args:
        result: SobolResult
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
    
    # Sort by total sensitivity
    params = sorted(result.parameter_names, 
                   key=lambda p: result.ST.get(p, 0))
    
    st_values = [result.ST.get(p, 0) for p in params]
    st_conf = [result.ST_conf.get(p, 0) for p in params]
    s1_values = [result.S1.get(p, 0) for p in params]
    
    y = np.arange(len(params))
    
    # Total-order (full bar)
    bars = ax.barh(y, st_values, xerr=st_conf, 
                   color=COLORS['st'], alpha=0.8, 
                   label='Total (ST)', capsize=3)
    
    # First-order (inner bar)
    ax.barh(y, s1_values, color=COLORS['s1'], alpha=0.9, 
            label='First-Order (S1)')
    
    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.set_xlabel('Sensitivity Index', fontsize=11)
    ax.set_ylabel('Parameter', fontsize=11)
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(st_values) * 1.2 if st_values else 1.0)
    
    # Add value labels
    for i, (st, s1) in enumerate(zip(st_values, s1_values)):
        ax.text(st + 0.01, i, f'{st:.3f}', va='center', fontsize=9)
    
    ax.set_title('Parameter Importance Ranking', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_interaction_heatmap(
    result: 'SobolResult',
    figsize: Tuple[int, int] = (10, 8),
    dark_theme: bool = True
) -> Figure:
    """
    Heatmap of second-order interaction indices.
    
    Args:
        result: SobolResult with S2 indices
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if result.S2 is None:
        raise ValueError("Second-order indices not available. "
                        "Run analysis with calc_second_order=True")
    
    if dark_theme:
        _apply_dark_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
    
    # Get S2 matrix
    s2_matrix = result.S2.values
    params = result.parameter_names
    
    # Create mask for diagonal
    mask = np.eye(len(params), dtype=bool)
    
    # Mask diagonal
    s2_display = s2_matrix.copy()
    s2_display[mask] = np.nan
    
    # Plot heatmap
    im = ax.imshow(s2_display, cmap='YlOrRd', aspect='auto')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Second-Order Index (S2)', fontsize=10)
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_yticklabels(params)
    
    # Add value annotations
    for i in range(len(params)):
        for j in range(len(params)):
            if i != j and not np.isnan(s2_matrix[i, j]):
                text_color = 'white' if s2_matrix[i, j] > 0.1 else 'black'
                ax.text(j, i, f'{s2_matrix[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8,
                       color=text_color)
    
    ax.set_title('Parameter Interactions (Second-Order Indices)', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_sensitivity_summary(
    result: 'SobolResult',
    figsize: Tuple[int, int] = (14, 10),
    dark_theme: bool = True
) -> Figure:
    """
    Comprehensive sensitivity analysis summary.
    
    Creates multi-panel figure with:
    - Bar chart of S1 and ST
    - Pie chart of relative importance
    - Interaction heatmap (if available)
    
    Args:
        result: SobolResult
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if dark_theme:
        _apply_dark_style()
    
    has_s2 = result.S2 is not None
    
    if has_s2:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2))
        ax3 = None
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        for ax in [ax1, ax2, ax3] if ax3 else [ax1, ax2]:
            if ax is not None:
                ax.set_facecolor('#16213e')
    
    # Sort parameters
    params = sorted(result.parameter_names, 
                   key=lambda p: result.ST.get(p, 0), reverse=True)
    
    # Panel 1: Bar chart
    s1_values = [result.S1.get(p, 0) for p in params]
    st_values = [result.ST.get(p, 0) for p in params]
    
    x = np.arange(len(params))
    width = 0.35
    
    ax1.bar(x - width/2, s1_values, width, label='S1', color=COLORS['s1'], alpha=0.8)
    ax1.bar(x + width/2, st_values, width, label='ST', color=COLORS['st'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=45, ha='right')
    ax1.set_ylabel('Sensitivity Index')
    ax1.legend()
    ax1.set_title('Sobol Indices', fontweight='bold')
    
    # Panel 2: Pie chart of relative importance
    st_total = sum(st_values)
    if st_total > 0:
        st_normalized = [st / st_total * 100 for st in st_values]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))
        
        wedges, texts, autotexts = ax2.pie(
            st_normalized, labels=params, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        ax2.set_title('Relative Importance (ST)', fontweight='bold')
    
    # Panel 3: Interaction heatmap
    if ax3 is not None and has_s2:
        s2_matrix = result.S2.values
        mask = np.eye(len(params), dtype=bool)
        s2_display = s2_matrix.copy()
        s2_display[mask] = np.nan
        
        im = ax3.imshow(s2_display, cmap='YlOrRd', aspect='auto')
        fig.colorbar(im, ax=ax3, shrink=0.6, label='S2')
        
        ax3.set_xticks(np.arange(len(params)))
        ax3.set_yticks(np.arange(len(params)))
        ax3.set_xticklabels(params, rotation=45, ha='right')
        ax3.set_yticklabels(params)
        ax3.set_title('Parameter Interactions', fontweight='bold')
    
    fig.suptitle(f'Sensitivity Analysis Summary (N={result.n_samples})', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig
