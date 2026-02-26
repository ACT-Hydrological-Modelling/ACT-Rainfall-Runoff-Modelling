"""
Visualization service for generating plots and report cards.

This service integrates with pyrrm's visualization module to generate
plots for the web interface.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd

# Path setup - this module doesn't directly import pyrrm
# but uses Plotly for visualization

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from app.config import get_settings

settings = get_settings()


class VisualizationService:
    """
    Service for generating visualizations.
    
    Provides functionality for:
    - Generating Plotly figures for web display
    - Creating report card visualizations
    - Exporting figures as JSON for frontend rendering
    """
    
    @staticmethod
    def create_hydrograph(
        dates: List[str],
        observed: List[float],
        simulated: List[float],
        precipitation: Optional[List[float]] = None,
        log_scale: bool = False,
        title: str = "Hydrograph Comparison"
    ) -> Dict[str, Any]:
        """
        Create a hydrograph comparison plot.
        
        Returns Plotly figure as JSON-serializable dict.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
        
        if precipitation is not None:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.2, 0.8],
                shared_xaxes=True,
                vertical_spacing=0.02
            )
            
            # Precipitation (inverted)
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=precipitation,
                    name='Precipitation',
                    marker_color='steelblue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Precip (mm)",
                autorange="reversed",
                row=1, col=1
            )
            
            flow_row = 2
        else:
            fig = go.Figure()
            flow_row = None
        
        # Observed flow
        trace_obs = go.Scatter(
            x=dates,
            y=observed,
            name='Observed',
            mode='lines',
            line=dict(color='black', width=1)
        )
        
        # Simulated flow
        trace_sim = go.Scatter(
            x=dates,
            y=simulated,
            name='Simulated',
            mode='lines',
            line=dict(color='red', width=1)
        )
        
        if flow_row:
            fig.add_trace(trace_obs, row=flow_row, col=1)
            fig.add_trace(trace_sim, row=flow_row, col=1)
            fig.update_yaxes(
                title_text="Flow",
                type='log' if log_scale else 'linear',
                row=flow_row, col=1
            )
        else:
            fig.add_trace(trace_obs)
            fig.add_trace(trace_sim)
            fig.update_yaxes(
                title_text="Flow",
                type='log' if log_scale else 'linear'
            )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=500,
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        return json.loads(fig.to_json())
    
    @staticmethod
    def create_fdc(
        observed: List[float],
        simulated: List[float],
        log_scale: bool = True,
        title: str = "Flow Duration Curve"
    ) -> Dict[str, Any]:
        """
        Create a flow duration curve comparison.
        
        Returns Plotly figure as JSON-serializable dict.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
        
        # Compute FDC
        def compute_fdc(data):
            sorted_data = np.sort(data)[::-1]
            exceedance = np.arange(1, len(sorted_data) + 1) / (len(sorted_data) + 1) * 100
            return exceedance, sorted_data
        
        obs_exc, obs_flow = compute_fdc(np.array(observed))
        sim_exc, sim_flow = compute_fdc(np.array(simulated))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=obs_exc, y=obs_flow,
            name='Observed',
            mode='lines',
            line=dict(color='black', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=sim_exc, y=sim_flow,
            name='Simulated',
            mode='lines',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Exceedance Probability (%)",
            yaxis_title="Flow",
            yaxis_type='log' if log_scale else 'linear',
            showlegend=True,
            height=400,
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        return json.loads(fig.to_json())
    
    @staticmethod
    def create_scatter(
        observed: List[float],
        simulated: List[float],
        title: str = "Observed vs Simulated"
    ) -> Dict[str, Any]:
        """
        Create an observed vs simulated scatter plot.
        
        Returns Plotly figure as JSON-serializable dict.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
        
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=observed, y=simulated,
            mode='markers',
            marker=dict(color='steelblue', size=4, opacity=0.5),
            name='Data'
        ))
        
        # 1:1 line
        max_val = max(max(observed), max(simulated))
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='1:1 Line'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Observed",
            yaxis_title="Simulated",
            showlegend=True,
            height=400,
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return json.loads(fig.to_json())
    
    @staticmethod
    def create_parameter_bounds_chart(
        parameters: Dict[str, float],
        bounds: Dict[str, List[float]],
        title: str = "Parameter Values (% of Bounds)"
    ) -> Dict[str, Any]:
        """
        Create a chart showing parameter values as percentage of bounds.
        
        Returns Plotly figure as JSON-serializable dict.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
        
        param_names = []
        percentages = []
        
        for name, value in parameters.items():
            if name in bounds:
                min_val, max_val = bounds[name]
                pct = (value - min_val) / (max_val - min_val) * 100
                param_names.append(name)
                percentages.append(pct)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=param_names,
            x=percentages,
            orientation='h',
            marker_color='steelblue'
        ))
        
        # Add reference lines
        fig.add_vline(x=0, line_dash="solid", line_color="gray")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        fig.add_vline(x=100, line_dash="solid", line_color="gray")
        
        fig.update_layout(
            title=title,
            xaxis_title="% of Parameter Range",
            xaxis_range=[-5, 105],
            height=max(300, len(param_names) * 25),
            margin=dict(l=100, r=20, t=60, b=40)
        )
        
        return json.loads(fig.to_json())
    
    @staticmethod
    def create_objective_evolution(
        iterations: List[int],
        objectives: List[float],
        best_objectives: Optional[List[float]] = None,
        title: str = "Objective Function Evolution"
    ) -> Dict[str, Any]:
        """
        Create a plot showing objective function evolution during calibration.
        
        Returns Plotly figure as JSON-serializable dict.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
        
        fig = go.Figure()
        
        # All evaluations
        fig.add_trace(go.Scatter(
            x=iterations, y=objectives,
            mode='markers',
            marker=dict(color='lightgray', size=2),
            name='All Evaluations'
        ))
        
        # Running best
        if best_objectives:
            fig.add_trace(go.Scatter(
                x=iterations, y=best_objectives,
                mode='lines',
                line=dict(color='red', width=2),
                name='Best So Far'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            showlegend=True,
            height=400,
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        return json.loads(fig.to_json())
    
    @staticmethod
    def create_metrics_table(metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Create a formatted metrics table for display.
        
        Returns metrics with formatting info.
        """
        def get_color(metric_name: str, value: float) -> str:
            """Get color based on metric value."""
            if metric_name in ['NSE', 'KGE', 'NSE_log', 'NSE_inv', 'NSE_sqrt']:
                if value >= 0.75:
                    return 'green'
                elif value >= 0.5:
                    return 'orange'
                else:
                    return 'red'
            elif metric_name == 'PBIAS':
                if abs(value) <= 10:
                    return 'green'
                elif abs(value) <= 25:
                    return 'orange'
                else:
                    return 'red'
            return 'gray'
        
        formatted = {}
        for name, value in metrics.items():
            if value is not None and not np.isnan(value):
                formatted[name] = {
                    'value': round(value, 4),
                    'color': get_color(name, value)
                }
            else:
                formatted[name] = {
                    'value': 'N/A',
                    'color': 'gray'
                }
        
        return formatted
