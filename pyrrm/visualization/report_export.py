"""
Report Card Export Module

Export calibration report cards to PDF and HTML formats suitable for
inclusion in professional hydrological modeling reports.

Dependencies:
- kaleido: For Plotly figure export to static images
- weasyprint (optional): For PDF generation from HTML
"""

import base64
import io
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pyrrm.calibration.report import CalibrationReport

# Check for optional dependencies
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

# WeasyPrint availability is checked lazily to avoid import errors
# when system dependencies (pango, gobject) are not available
WEASYPRINT_AVAILABLE = None  # Will be set on first check

def _check_weasyprint():
    """Lazy check for weasyprint availability."""
    global WEASYPRINT_AVAILABLE
    if WEASYPRINT_AVAILABLE is None:
        try:
            import weasyprint  # noqa: F401
            WEASYPRINT_AVAILABLE = True
        except (ImportError, OSError):
            WEASYPRINT_AVAILABLE = False
    return WEASYPRINT_AVAILABLE

# Export section identifiers
EXPORT_SECTIONS = [
    "summary",
    "metrics",
    "hydrograph_linear",
    "hydrograph_log",
    "fdc",
    "scatter",
    "parameters",
    "signatures",
]

DEFAULT_SECTIONS = EXPORT_SECTIONS.copy()


def _figure_to_base64(fig, width: int = 800, height: int = 400, scale: float = 2.0) -> str:
    """Convert a Plotly figure to a base64-encoded PNG string."""
    if not KALEIDO_AVAILABLE:
        raise ImportError("kaleido is required for figure export. Install with: pip install kaleido")
    
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return base64.b64encode(img_bytes).decode("utf-8")


def _format_value(value: Any, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        elif abs(value) >= 1:
            return f"{value:.{precision}f}"
        else:
            return f"{value:.{precision}f}"
    return str(value)


def _get_metric_color(value: float, metric_name: str) -> str:
    """Get color for metric value based on thresholds."""
    EFFICIENCY_METRICS = {
        "NSE", "NSE_sqrt", "NSE_log", "NSE_inv",
        "KGE", "KGE_sqrt", "KGE_log", "KGE_inv",
        "KGE_np", "KGE_np_sqrt", "KGE_np_log", "KGE_np_inv",
    }
    PERCENT_ERROR_METRICS = {"PBIAS", "FHV", "FMV", "FLV", "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"}
    
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#888888"
    
    if metric_name in EFFICIENCY_METRICS:
        if value >= 0.7:
            return "#27ae60"  # Green
        elif value >= 0.5:
            return "#f39c12"  # Orange
        else:
            return "#e74c3c"  # Red
    
    if metric_name in PERCENT_ERROR_METRICS:
        if abs(value) <= 10:
            return "#27ae60"
        elif abs(value) <= 20:
            return "#f39c12"
        else:
            return "#e74c3c"
    
    return "#2c3e50"  # Neutral


def _get_error_color(pct_err: float) -> str:
    """Get color for percent error value."""
    if pct_err is None or (isinstance(pct_err, float) and np.isnan(pct_err)):
        return "#888888"
    if abs(pct_err) <= 10:
        return "#27ae60"
    elif abs(pct_err) <= 20:
        return "#f39c12"
    else:
        return "#e74c3c"


# CSS styles for portrait multi-page report (legacy)
REPORT_CSS_PORTRAIT = """
@page {
    size: A4;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #2c3e50;
    max-width: 210mm;
    margin: 0 auto;
    padding: 20px;
}

h1 { font-size: 20pt; color: #1a5276; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 5px; }
h2 { font-size: 14pt; color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 25px; margin-bottom: 15px; }
h3 { font-size: 12pt; color: #34495e; margin-top: 15px; margin-bottom: 10px; }
.subtitle { font-size: 10pt; color: #7f8c8d; margin-bottom: 20px; }

.metadata { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
.metadata-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.metadata-item { display: flex; }
.metadata-label { font-weight: 600; color: #495057; min-width: 150px; }
.metadata-value { color: #2c3e50; }

table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 10pt; page-break-inside: avoid; }
th { background: #34495e; color: white; padding: 10px 8px; text-align: left; font-weight: 600; }
td { padding: 8px; border-bottom: 1px solid #e9ecef; }
tr:nth-child(even) { background: #f8f9fa; }

.metric-value { font-weight: 600; padding: 4px 8px; border-radius: 3px; color: white; display: inline-block; min-width: 60px; text-align: center; }
.metric-good { background: #27ae60; }
.metric-ok { background: #f39c12; }
.metric-poor { background: #e74c3c; }
.metric-neutral { background: #2c3e50; }
.metric-na { background: #888888; }

.figure-container { margin: 20px 0; text-align: center; page-break-inside: avoid; }
.figure-container img { max-width: 100%; height: auto; border: 1px solid #e9ecef; border-radius: 5px; }
.figure-caption { font-size: 9pt; color: #7f8c8d; margin-top: 8px; font-style: italic; }

.signatures-table { font-size: 9pt; }
.signatures-table th, .signatures-table td { padding: 5px 6px; }
.category-header { background: #ecf0f1 !important; font-weight: 600; color: #2c3e50; }
.parameter-table td:first-child { font-family: monospace; font-weight: 600; }
.footer { margin-top: 30px; padding-top: 15px; border-top: 1px solid #bdc3c7; font-size: 9pt; color: #7f8c8d; text-align: center; }
"""

# CSS styles for landscape single-page report
REPORT_CSS_LANDSCAPE = """
@page {
    size: A4 landscape;
    margin: 8mm;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 8pt;
    line-height: 1.3;
    color: #2c3e50;
    width: 281mm;
    height: 194mm;
    overflow: hidden;
}

.report-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Header strip */
.header-strip {
    background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
    color: white;
    padding: 6px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}

.header-title {
    font-size: 14pt;
    font-weight: 700;
}

.header-subtitle {
    font-size: 8pt;
    opacity: 0.9;
}

.header-meta {
    display: flex;
    gap: 20px;
    font-size: 8pt;
}

.header-meta-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.header-meta-label {
    font-size: 6pt;
    opacity: 0.8;
    text-transform: uppercase;
}

.header-meta-value {
    font-weight: 600;
    font-size: 10pt;
}

/* Main content grid */
.content-grid {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px;
    padding: 6px;
    min-height: 0;
}

.panel {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.panel-header {
    background: #34495e;
    color: white;
    padding: 4px 8px;
    font-size: 8pt;
    font-weight: 600;
    flex-shrink: 0;
}

.panel-content {
    flex: 1;
    padding: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.panel-content img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

/* Metrics table */
.metrics-table {
    width: 100%;
    font-size: 7pt;
    border-collapse: collapse;
}

.metrics-table th {
    background: #34495e;
    color: white;
    padding: 3px 4px;
    text-align: left;
    font-size: 6pt;
}

.metrics-table td {
    padding: 2px 4px;
    border-bottom: 1px solid #eee;
}

.metrics-table tr:nth-child(even) {
    background: #f8f9fa;
}

.metric-badge {
    display: inline-block;
    padding: 1px 4px;
    border-radius: 2px;
    color: white;
    font-weight: 600;
    font-size: 7pt;
    min-width: 45px;
    text-align: center;
}

.badge-good { background: #27ae60; }
.badge-ok { background: #f39c12; }
.badge-poor { background: #e74c3c; }
.badge-neutral { background: #7f8c8d; }

/* Signatures table - compact 2 columns */
.signatures-compact {
    width: 100%;
    font-size: 6pt;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px;
    overflow-y: auto;
    max-height: 100%;
}

.sig-item {
    display: flex;
    justify-content: space-between;
    padding: 1px 3px;
    border-bottom: 1px solid #f0f0f0;
}

.sig-name {
    color: #555;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 70px;
}

.sig-error {
    font-weight: 600;
    padding: 0 3px;
    border-radius: 2px;
    font-size: 6pt;
}

/* Parameters bar */
.params-compact {
    width: 100%;
    font-size: 6pt;
}

.param-row {
    display: flex;
    align-items: center;
    margin-bottom: 2px;
}

.param-name {
    width: 50px;
    font-family: monospace;
    font-size: 6pt;
    text-align: right;
    padding-right: 4px;
}

.param-bar-container {
    flex: 1;
    height: 10px;
    background: #ecf0f1;
    border-radius: 2px;
    position: relative;
}

.param-bar {
    height: 100%;
    border-radius: 2px;
}

.param-value {
    position: absolute;
    right: 2px;
    top: 0;
    font-size: 5pt;
    line-height: 10px;
    color: #333;
}

/* Footer */
.footer-strip {
    background: #f8f9fa;
    padding: 3px 12px;
    font-size: 6pt;
    color: #7f8c8d;
    display: flex;
    justify-content: space-between;
    flex-shrink: 0;
}
"""

# For backwards compatibility
REPORT_CSS = REPORT_CSS_PORTRAIT


def _render_summary_section(report: 'CalibrationReport') -> str:
    """Render the summary/metadata section."""
    header = report.catchment_info
    result = report.result
    
    catchment_name = header.get('name', 'Unknown Catchment')
    gauge_id = header.get('gauge_id', '')
    area = header.get('area_km2', '')
    
    # Get model name from model_config or experiment_name
    model_name = "Unknown"
    if hasattr(report, 'model_config') and report.model_config:
        model_class = report.model_config.get('class_name', '')
        if model_class:
            model_name = model_class
    if model_name == "Unknown" and hasattr(report, 'experiment_name') and report.experiment_name:
        # Try to extract model from experiment name (e.g., "410734_sacramento_nse_sceua")
        parts = report.experiment_name.split('_')
        if len(parts) >= 2:
            model_name = parts[1].title()  # e.g., "sacramento" -> "Sacramento"
    
    period_start, period_end = report.calibration_period
    
    # Handle runtime_seconds which might be None
    runtime = result.runtime_seconds if result.runtime_seconds is not None else 0.0
    
    # Handle all_samples which might be None
    n_samples = len(result.all_samples) if result.all_samples is not None else 'N/A'
    
    html = f"""
    <div class="metadata">
        <div class="metadata-grid">
            <div class="metadata-item">
                <span class="metadata-label">Model:</span>
                <span class="metadata-value">{model_name}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Calibration Method:</span>
                <span class="metadata-value">{result.method}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Objective Function:</span>
                <span class="metadata-value">{result.objective_name}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Best Objective Value:</span>
                <span class="metadata-value">{result.best_objective:.4f}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Calibration Period:</span>
                <span class="metadata-value">{period_start} to {period_end}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Catchment Area:</span>
                <span class="metadata-value">{area} km²</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Runtime:</span>
                <span class="metadata-value">{runtime:.1f} seconds</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Number of Samples:</span>
                <span class="metadata-value">{n_samples}</span>
            </div>
        </div>
    </div>
    """
    return html


def _render_metrics_section(report: 'CalibrationReport', metrics: Dict[str, float]) -> str:
    """Render the diagnostic metrics table."""
    from pyrrm.visualization.report_plots import _calculate_basic_metrics
    
    if not metrics:
        metrics = _calculate_basic_metrics(report.observed, report.simulated)
    
    HEADLINE_METRICS = [
        ("NSE", "Nash-Sutcliffe Efficiency (high flows)"),
        ("NSE_sqrt", "NSE on √Q (balanced)"),
        ("NSE_log", "NSE on log(Q) (low flows)"),
        ("NSE_inv", "NSE on 1/Q (very low flows)"),
        ("KGE", "Kling-Gupta Efficiency (high flows)"),
        ("KGE_sqrt", "KGE on √Q (balanced)"),
        ("KGE_log", "KGE on log(Q) (low flows)"),
        ("KGE_inv", "KGE on 1/Q (very low flows)"),
        ("KGE_np", "KGE Non-parametric (high flows)"),
        ("KGE_np_sqrt", "KGE_np on √Q (balanced)"),
        ("KGE_np_log", "KGE_np on log(Q) (low flows)"),
        ("KGE_np_inv", "KGE_np on 1/Q (very low flows)"),
        ("RMSE", "Root Mean Square Error"),
        ("MAE", "Mean Absolute Error"),
        ("SDEB", "Spectral Decomposition Error Bias"),
        ("PBIAS", "Percent Bias (%)"),
        ("FHV", "High Flow Volume Error (%)"),
        ("FMV", "Mid Flow Volume Error (%)"),
        ("FLV", "Low Flow Volume Error (%)"),
        ("Sig_BFI", "Baseflow Index Error (%)"),
        ("Sig_Flash", "Flashiness Index Error (%)"),
        ("Sig_Q95", "Q95 (low flow) Error (%)"),
        ("Sig_Q5", "Q5 (high flow) Error (%)"),
    ]
    
    PERCENT_METRICS = {"PBIAS", "FHV", "FMV", "FLV", "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"}
    
    rows = []
    for metric_id, description in HEADLINE_METRICS:
        value = metrics.get(metric_id, np.nan)
        color = _get_metric_color(value, metric_id)
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            formatted = "N/A"
            css_class = "metric-na"
        elif metric_id in PERCENT_METRICS:
            formatted = f"{value:+.2f}%"
            css_class = "metric-good" if abs(value) <= 10 else ("metric-ok" if abs(value) <= 20 else "metric-poor")
        elif metric_id in {"RMSE", "MAE", "SDEB"}:
            formatted = f"{value:.2f}"
            css_class = "metric-neutral"
        else:
            formatted = f"{value:.4f}"
            css_class = "metric-good" if value >= 0.7 else ("metric-ok" if value >= 0.5 else "metric-poor")
        
        rows.append(f"""
            <tr>
                <td><strong>{metric_id}</strong></td>
                <td><span class="metric-value {css_class}">{formatted}</span></td>
                <td>{description}</td>
            </tr>
        """)
    
    return f"""
    <h2>2. Diagnostic Metrics</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 15%">Metric</th>
                <th style="width: 20%">Value</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _render_figure_section(
    title: str,
    section_num: str,
    fig,
    caption: str,
    width: int = 750,
    height: int = 350
) -> str:
    """Render a figure section with title and caption."""
    img_b64 = _figure_to_base64(fig, width=width, height=height)
    return f"""
    <h2>{section_num}. {title}</h2>
    <div class="figure-container">
        <img src="data:image/png;base64,{img_b64}" alt="{title}">
        <div class="figure-caption">{caption}</div>
    </div>
    """


def _render_parameters_section(report: 'CalibrationReport') -> str:
    """Render the calibrated parameters table."""
    params = report.result.best_parameters
    bounds = report.parameter_bounds or {}
    
    rows = []
    for param, value in params.items():
        if param in bounds:
            low, high = bounds[param]
            position = (value - low) / (high - low) * 100 if high > low else 50
            position = np.clip(position, 0, 100)
            bounds_str = f"[{low:.3g}, {high:.3g}]"
            position_str = f"{position:.1f}%"
            
            if position < 5 or position > 95:
                pos_class = "metric-poor"
            elif position < 15 or position > 85:
                pos_class = "metric-ok"
            else:
                pos_class = "metric-good"
        else:
            bounds_str = "—"
            position_str = "—"
            pos_class = "metric-neutral"
        
        rows.append(f"""
            <tr>
                <td>{param}</td>
                <td>{value:.6g}</td>
                <td>{bounds_str}</td>
                <td><span class="metric-value {pos_class}">{position_str}</span></td>
            </tr>
        """)
    
    return f"""
    <h2>6. Calibrated Parameters</h2>
    <table class="parameter-table">
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Calibrated Value</th>
                <th>Bounds</th>
                <th>Position</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    <p style="font-size: 9pt; color: #7f8c8d;">
        Position indicates where the calibrated value falls within the parameter bounds (0% = lower bound, 100% = upper bound).
        Values near the bounds (highlighted) may indicate the bounds are too restrictive.
    </p>
    """


def _render_signatures_section(report: 'CalibrationReport') -> str:
    """Render the hydrologic signatures table."""
    try:
        from pyrrm.analysis.signatures import (
            SIGNATURE_CATEGORIES,
            SIGNATURE_INFO,
            compute_all_signatures,
            signature_percent_error,
        )
    except ImportError:
        return "<h2>7. Hydrologic Signatures</h2><p>Signatures module not available.</p>"
    
    obs_sigs = compute_all_signatures(report.observed, report.dates)
    sim_sigs = compute_all_signatures(report.simulated, report.dates)
    
    rows = []
    for category, sig_ids in SIGNATURE_CATEGORIES.items():
        # Category header row
        rows.append(f'<tr class="category-header"><td colspan="5">{category}</td></tr>')
        
        for sig_id in sig_ids:
            info = SIGNATURE_INFO.get(sig_id, {})
            name = info.get("name", sig_id)
            
            obs_val = obs_sigs.get(sig_id)
            sim_val = sim_sigs.get(sig_id)
            pct_err = signature_percent_error(obs_val, sim_val) if obs_val is not None else None
            
            obs_str = _format_value(obs_val, 3)
            sim_str = _format_value(sim_val, 3)
            
            if pct_err is not None and not np.isnan(pct_err):
                err_str = f"{pct_err:+.1f}%"
                err_class = "metric-good" if abs(pct_err) <= 10 else ("metric-ok" if abs(pct_err) <= 20 else "metric-poor")
            else:
                err_str = "N/A"
                err_class = "metric-na"
            
            rows.append(f"""
                <tr>
                    <td>{name}</td>
                    <td>{obs_str}</td>
                    <td>{sim_str}</td>
                    <td><span class="metric-value {err_class}">{err_str}</span></td>
                </tr>
            """)
    
    return f"""
    <h2>7. Hydrologic Signatures</h2>
    <table class="signatures-table">
        <thead>
            <tr>
                <th>Signature</th>
                <th>Observed</th>
                <th>Simulated</th>
                <th>% Error</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def export_report_card_html(
    report: 'CalibrationReport',
    sections: Optional[List[str]] = None,
    include_css: bool = True,
) -> str:
    """
    Export a calibration report card as self-contained HTML.
    
    Args:
        report: CalibrationReport instance to export
        sections: List of section IDs to include. If None, includes all sections.
                  Valid IDs: summary, metrics, hydrograph_linear, hydrograph_log,
                  fdc, scatter, parameters, signatures
        include_css: Whether to include CSS styles in the HTML
    
    Returns:
        Complete HTML document as string
    """
    from pyrrm.visualization.report_plots import plot_report_card_components
    
    if sections is None:
        sections = DEFAULT_SECTIONS
    
    # Get figure components
    components = plot_report_card_components(report)
    header = components["header"]
    
    # Calculate metrics once
    from pyrrm.visualization.report_plots import _calculate_basic_metrics
    metrics = _calculate_basic_metrics(report.observed, report.simulated)
    
    # Build title
    title = header["catchment_name"]
    if header["gauge_id"]:
        title += f" ({header['gauge_id']})"
    
    # Build content sections
    content_parts = []
    section_num = 1
    
    if "summary" in sections:
        content_parts.append(f"<h2>{section_num}. Summary</h2>")
        content_parts.append(_render_summary_section(report))
        section_num += 1
    
    if "metrics" in sections:
        content_parts.append(_render_metrics_section(report, metrics).replace("2.", f"{section_num}."))
        section_num += 1
    
    if "hydrograph_linear" in sections and components.get("hydrograph_linear"):
        content_parts.append(_render_figure_section(
            "Hydrograph (Linear Scale)", str(section_num),
            components["hydrograph_linear"],
            "Time series comparison of observed and simulated daily streamflow.",
            width=750, height=300
        ))
        section_num += 1
    
    if "hydrograph_log" in sections and components.get("hydrograph_log"):
        content_parts.append(_render_figure_section(
            "Hydrograph (Log Scale)", str(section_num),
            components["hydrograph_log"],
            "Logarithmic scale highlights low flow performance.",
            width=750, height=300
        ))
        section_num += 1
    
    if "fdc" in sections and components.get("fdc"):
        content_parts.append(_render_figure_section(
            "Flow Duration Curve", str(section_num),
            components["fdc"],
            "Exceedance probability distribution of observed and simulated flows.",
            width=700, height=350
        ))
        section_num += 1
    
    if "scatter" in sections and components.get("scatter"):
        content_parts.append(_render_figure_section(
            "Scatter Plot", str(section_num),
            components["scatter"],
            "Observed vs simulated daily flows with 1:1 reference line.",
            width=500, height=450
        ))
        section_num += 1
    
    if "parameters" in sections:
        content_parts.append(_render_parameters_section(report).replace("6.", f"{section_num}."))
        section_num += 1
    
    if "signatures" in sections:
        content_parts.append(_render_signatures_section(report).replace("7.", f"{section_num}."))
        section_num += 1
    
    # Assemble HTML
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calibration Report - {title}</title>
    {'<style>' + REPORT_CSS + '</style>' if include_css else ''}
</head>
<body>
    <h1>Calibration Report Card</h1>
    <div class="subtitle">
        <strong>{title}</strong><br>
        Generated: {generated_date}
    </div>
    
    {''.join(content_parts)}
    
    <div class="footer">
        Generated by pyrrm (Python Rainfall-Runoff Models) • {generated_date}
    </div>
</body>
</html>
"""
    return html


def export_report_card_pdf(
    report: 'CalibrationReport',
    sections: Optional[List[str]] = None,
) -> bytes:
    """
    Export a calibration report card as PDF.
    
    Args:
        report: CalibrationReport instance to export
        sections: List of section IDs to include. If None, includes all sections.
    
    Returns:
        PDF file content as bytes
    
    Raises:
        ImportError: If weasyprint is not available
    """
    if not _check_weasyprint():
        raise ImportError(
            "weasyprint is required for PDF export. "
            "Install with: pip install weasyprint\n"
            "Note: weasyprint requires system dependencies. See: "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        )
    
    html_content = export_report_card_html(report, sections=sections, include_css=True)
    
    from weasyprint import HTML
    pdf_bytes = HTML(string=html_content).write_pdf()
    
    return pdf_bytes


def export_batch_report_html(
    reports: Dict[str, 'CalibrationReport'],
    title: str = "Batch Calibration Report",
    sections: Optional[List[str]] = None,
) -> str:
    """
    Export multiple calibration report cards as a single HTML document.
    
    Args:
        reports: Dictionary mapping experiment keys to CalibrationReport instances
        title: Title for the batch report
        sections: List of section IDs to include for each report
    
    Returns:
        Complete HTML document as string
    """
    if sections is None:
        sections = DEFAULT_SECTIONS
    
    # Build individual report sections
    report_sections = []
    for i, (exp_key, report) in enumerate(reports.items(), 1):
        header = report.catchment_info
        catchment_name = header.get('name', 'Unknown')
        
        # Create experiment header
        report_sections.append(f"""
        <div style="page-break-before: {'always' if i > 1 else 'auto'};">
            <h1 style="font-size: 16pt;">Experiment {i}: {exp_key}</h1>
            <p class="subtitle">{catchment_name}</p>
        """)
        
        # Add content (reusing single report logic but without full HTML wrapper)
        from pyrrm.visualization.report_plots import plot_report_card_components, _calculate_basic_metrics
        
        components = plot_report_card_components(report)
        metrics = _calculate_basic_metrics(report.observed, report.simulated)
        
        section_num = 1
        
        if "summary" in sections:
            report_sections.append(f"<h2>{section_num}. Summary</h2>")
            report_sections.append(_render_summary_section(report))
            section_num += 1
        
        if "metrics" in sections:
            report_sections.append(_render_metrics_section(report, metrics).replace("2.", f"{section_num}."))
            section_num += 1
        
        if "hydrograph_linear" in sections and components.get("hydrograph_linear"):
            report_sections.append(_render_figure_section(
                "Hydrograph (Linear)", str(section_num),
                components["hydrograph_linear"],
                "Observed vs simulated daily streamflow.",
                width=700, height=280
            ))
            section_num += 1
        
        if "hydrograph_log" in sections and components.get("hydrograph_log"):
            report_sections.append(_render_figure_section(
                "Hydrograph (Log)", str(section_num),
                components["hydrograph_log"],
                "Log scale view.",
                width=700, height=280
            ))
            section_num += 1
        
        if "fdc" in sections and components.get("fdc"):
            report_sections.append(_render_figure_section(
                "Flow Duration Curve", str(section_num),
                components["fdc"],
                "Exceedance probability.",
                width=600, height=300
            ))
            section_num += 1
        
        if "scatter" in sections and components.get("scatter"):
            report_sections.append(_render_figure_section(
                "Scatter Plot", str(section_num),
                components["scatter"],
                "Obs vs sim.",
                width=450, height=400
            ))
            section_num += 1
        
        if "parameters" in sections:
            report_sections.append(_render_parameters_section(report).replace("6.", f"{section_num}."))
            section_num += 1
        
        if "signatures" in sections:
            report_sections.append(_render_signatures_section(report).replace("7.", f"{section_num}."))
        
        report_sections.append("</div>")
    
    # Assemble full HTML
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <h1 style="font-size: 24pt;">{title}</h1>
    <div class="subtitle">
        {len(reports)} experiments • Generated: {generated_date}
    </div>
    
    <h2>Table of Contents</h2>
    <ol>
    {''.join(f'<li>{key}</li>' for key in reports.keys())}
    </ol>
    
    {''.join(report_sections)}
    
    <div class="footer">
        Generated by pyrrm (Python Rainfall-Runoff Models) • {generated_date}
    </div>
</body>
</html>
"""
    return html


def export_batch_report_pdf(
    reports: Dict[str, 'CalibrationReport'],
    title: str = "Batch Calibration Report",
    sections: Optional[List[str]] = None,
) -> bytes:
    """
    Export multiple calibration report cards as a single PDF document.
    
    Args:
        reports: Dictionary mapping experiment keys to CalibrationReport instances
        title: Title for the batch report
        sections: List of section IDs to include for each report
    
    Returns:
        PDF file content as bytes
    """
    if not _check_weasyprint():
        raise ImportError(
            "weasyprint is required for PDF export. "
            "Install with: pip install weasyprint"
        )
    
    html_content = export_batch_report_html(reports, title=title, sections=sections)
    
    from weasyprint import HTML
    pdf_bytes = HTML(string=html_content).write_pdf()
    
    return pdf_bytes


# Convenience function for getting available sections
def get_export_sections() -> List[Dict[str, str]]:
    """
    Get list of available export sections with descriptions.
    
    Returns:
        List of dicts with 'id', 'name', and 'description' keys
    """
    return [
        {"id": "summary", "name": "Summary", "description": "Model, method, objective, and calibration metadata"},
        {"id": "metrics", "name": "Diagnostic Metrics", "description": "NSE, KGE, RMSE, and other performance metrics"},
        {"id": "hydrograph_linear", "name": "Hydrograph (Linear)", "description": "Time series on linear scale"},
        {"id": "hydrograph_log", "name": "Hydrograph (Log)", "description": "Time series on logarithmic scale"},
        {"id": "fdc", "name": "Flow Duration Curve", "description": "Exceedance probability distribution"},
        {"id": "scatter", "name": "Scatter Plot", "description": "Observed vs simulated comparison"},
        {"id": "parameters", "name": "Calibrated Parameters", "description": "Parameter values and bounds"},
        {"id": "signatures", "name": "Hydrologic Signatures", "description": "47 hydrologic signature metrics"},
    ]


# =============================================================================
# MATPLOTLIB-BASED REPORT CARD (Single Unified Figure)
# =============================================================================

def generate_matplotlib_report_card(
    report: 'CalibrationReport',
    figsize: tuple = (16, 9),  # 16:9 aspect ratio
    dpi: int = 150,
) -> 'Figure':
    """
    Generate a unified matplotlib report card figure.
    
    Creates a single landscape 16:9 figure with all report card components:
    - Header with metadata
    - Hydrographs (linear and log scale)
    - Diagnostic metrics table
    - Calibrated parameters bar chart
    - Flow duration curve
    - Scatter plot
    - Hydrologic signatures table (all signatures displayed)
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size in inches (default: 16:9 widescreen)
        dpi: Resolution for rendering
    
    Returns:
        matplotlib.figure.Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import pandas as pd
    
    from pyrrm.visualization.report_plots import _calculate_basic_metrics
    
    # Get data from report
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    result = report.result
    bounds = report.parameter_bounds or {}
    catchment_info = report.catchment_info
    
    # Calculate metrics
    metrics = _calculate_basic_metrics(obs, sim)
    
    # Create figure with GridSpec - optimized for 16:9
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Main grid layout:
    # Row 0: Header (thin)
    # Row 1: Hydrographs (2 stacked) | Metrics Table | Parameters Chart
    # Row 2: FDC | Scatter | Signatures Table
    gs = gridspec.GridSpec(
        3, 3, 
        figure=fig,
        height_ratios=[0.05, 1, 1],
        width_ratios=[1.15, 0.75, 1.1],  # Balanced widths, wider signatures
        hspace=0.18,
        wspace=0.12,
        left=0.03, right=0.97, top=0.97, bottom=0.03
    )
    
    # =========================================================================
    # HEADER (spans all 3 columns)
    # =========================================================================
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    # Get header info
    catchment_name = catchment_info.get('name', 'Unknown Catchment')
    gauge_id = catchment_info.get('gauge_id', '')
    period_start, period_end = report.calibration_period
    
    # Model name from config or experiment name
    model_name = "Unknown"
    if hasattr(report, 'model_config') and report.model_config:
        model_name = report.model_config.get('class_name', 'Unknown')
    if model_name == "Unknown" and hasattr(report, 'experiment_name') and report.experiment_name:
        parts = report.experiment_name.split('_')
        if len(parts) >= 2:
            model_name = parts[1].title()
    
    # Header background
    ax_header.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax_header.transAxes,
        facecolor='#1a5276', edgecolor='none'
    ))
    
    # Title
    title = f"{catchment_name}"
    if gauge_id:
        title += f" ({gauge_id})"
    ax_header.text(0.015, 0.5, title, transform=ax_header.transAxes,
                   fontsize=12, fontweight='bold', color='white', va='center')
    
    # Subtitle
    subtitle = f"{model_name} Model  •  {period_start} to {period_end}"
    ax_header.text(0.015, 0.12, subtitle, transform=ax_header.transAxes,
                   fontsize=8, color='#d5dbdb', va='center')
    
    # Right side: Method, Objective, Best Value
    info_x = 0.985
    ax_header.text(info_x, 0.6, f"Method: {result.method}", transform=ax_header.transAxes,
                   fontsize=8, color='white', va='center', ha='right')
    ax_header.text(info_x, 0.2, f"Objective: {result.objective_name} = {result.best_objective:.4f}",
                   transform=ax_header.transAxes, fontsize=8, color='white', va='center', ha='right')
    
    # =========================================================================
    # ROW 1, COL 0: HYDROGRAPHS (stacked linear + log)
    # =========================================================================
    gs_hydro = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.12)
    
    ax_hydro_lin = fig.add_subplot(gs_hydro[0])
    _plot_hydrograph_on_ax(ax_hydro_lin, dates, obs, sim, log_scale=False, show_legend=True)
    ax_hydro_lin.set_title('Hydrograph (Linear)', fontsize=8, fontweight='bold', loc='left')
    
    ax_hydro_log = fig.add_subplot(gs_hydro[1])
    _plot_hydrograph_on_ax(ax_hydro_log, dates, obs, sim, log_scale=True, show_legend=False)
    ax_hydro_log.set_title('Hydrograph (Log)', fontsize=8, fontweight='bold', loc='left')
    
    # =========================================================================
    # ROW 1, COL 1: DIAGNOSTIC METRICS TABLE
    # =========================================================================
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis('off')
    ax_metrics.set_title('Diagnostic Metrics', fontsize=8, fontweight='bold', loc='left', pad=5)
    _render_metrics_table_on_ax(ax_metrics, metrics)
    
    # =========================================================================
    # ROW 1, COL 2: CALIBRATED PARAMETERS
    # =========================================================================
    ax_params = fig.add_subplot(gs[1, 2])
    ax_params.set_title('Calibrated Parameters', fontsize=8, fontweight='bold', loc='left')
    _plot_parameters_on_ax(ax_params, result.best_parameters, bounds)
    
    # =========================================================================
    # ROW 2, COL 0: FLOW DURATION CURVE
    # =========================================================================
    ax_fdc = fig.add_subplot(gs[2, 0])
    ax_fdc.set_title('Flow Duration Curve', fontsize=8, fontweight='bold', loc='left')
    _plot_fdc_on_ax(ax_fdc, obs, sim)
    
    # =========================================================================
    # ROW 2, COL 1: SCATTER PLOT
    # =========================================================================
    ax_scatter = fig.add_subplot(gs[2, 1])
    ax_scatter.set_title('Scatter Plot', fontsize=8, fontweight='bold', loc='left')
    _plot_scatter_on_ax(ax_scatter, obs, sim)
    
    # =========================================================================
    # ROW 2, COL 2: HYDROLOGIC SIGNATURES (all signatures)
    # =========================================================================
    ax_sigs = fig.add_subplot(gs[2, 2])
    ax_sigs.axis('off')
    ax_sigs.set_title('Hydrologic Signatures (% Error)', fontsize=8, fontweight='bold', loc='left', pad=5)
    _render_signatures_table_on_ax(ax_sigs, report)
    
    # Footer
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.5, 0.008, f"Generated by pyrrm (Python Rainfall-Runoff Models)  •  {generated_date}",
             ha='center', fontsize=6, color='#7f8c8d')
    
    return fig


def _plot_hydrograph_on_ax(ax, dates, obs, sim, log_scale=False, show_legend=True):
    """Plot hydrograph on a given axes."""
    import pandas as pd
    
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    
    ax.plot(dates, obs, color='#e74c3c', linewidth=0.6, label='Observed', alpha=0.9)
    ax.plot(dates, sim, color='#3498db', linewidth=0.6, label='Simulated', alpha=0.9)
    
    if log_scale:
        ax.set_yscale('log')
        pos_obs = obs[obs > 0]
        if len(pos_obs) > 0:
            ax.set_ylim(bottom=max(0.1, np.nanmin(pos_obs) * 0.5))
    
    ax.set_ylabel('Flow (ML/day)', fontsize=6)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(axis='x', rotation=0)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=5, framealpha=0.9)


def _render_metrics_table_on_ax(ax, metrics):
    """Render ALL diagnostic metrics as a compact 2-column table on axes."""
    # Organize metrics by category for comprehensive display
    METRIC_GROUPS = [
        # Efficiency metrics (NSE variants)
        ("NSE", "NSE", "eff"),
        ("NSE_log", "NSE(log)", "eff"),
        ("NSE_sqrt", "NSE(√Q)", "eff"),
        ("NSE_inv", "NSE(1/Q)", "eff"),
        # KGE variants
        ("KGE", "KGE", "eff"),
        ("KGE_log", "KGE(log)", "eff"),
        ("KGE_sqrt", "KGE(√Q)", "eff"),
        ("KGE_inv", "KGE(1/Q)", "eff"),
        # KGE non-parametric
        ("KGE_np", "KGE_np", "eff"),
        ("KGE_np_log", "KGE_np(log)", "eff"),
        ("KGE_np_sqrt", "KGE_np(√Q)", "eff"),
        # Error metrics
        ("RMSE", "RMSE", "err"),
        ("MAE", "MAE", "err"),
        ("PBIAS", "PBIAS", "pct"),
        ("R2", "R²", "eff"),
        # FDC segment errors
        ("FHV", "FHV", "pct"),
        ("FMV", "FMV", "pct"),
        ("FLV", "FLV", "pct"),
        # Flow percentile biases
        ("Q5_bias", "Q5 bias", "pct"),
        ("Q50_bias", "Q50 bias", "pct"),
        ("Q95_bias", "Q95 bias", "pct"),
        # Signature errors
        ("Sig_BFI", "BFI err", "pct"),
        ("Sig_Flash", "Flash err", "pct"),
    ]
    
    # Build 2-column layout
    n_metrics = len(METRIC_GROUPS)
    n_rows = (n_metrics + 1) // 2
    
    cell_text = []
    cell_colors = []
    
    def format_metric(metric_id, display_name, metric_type):
        value = metrics.get(metric_id, np.nan)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return display_name, "—", '#f0f0f0'
        
        if metric_type == "pct":
            formatted = f"{value:+.0f}%"
            color = '#d5f4e6' if abs(value) <= 10 else ('#fef9e7' if abs(value) <= 25 else '#fadbd8')
        elif metric_type == "err":
            formatted = f"{value:.1f}"
            color = '#f0f0f0'
        else:  # efficiency
            formatted = f"{value:.2f}"
            color = '#d5f4e6' if value >= 0.7 else ('#fef9e7' if value >= 0.5 else '#fadbd8')
        
        return display_name, formatted, color
    
    for i in range(n_rows):
        row_text = []
        row_colors = []
        
        # Left column
        if i < len(METRIC_GROUPS):
            metric_id, display_name, metric_type = METRIC_GROUPS[i]
            name, val, color = format_metric(metric_id, display_name, metric_type)
            row_text.extend([name, val])
            row_colors.extend(['white', color])
        else:
            row_text.extend(['', ''])
            row_colors.extend(['white', 'white'])
        
        # Right column
        idx_right = i + n_rows
        if idx_right < len(METRIC_GROUPS):
            metric_id, display_name, metric_type = METRIC_GROUPS[idx_right]
            name, val, color = format_metric(metric_id, display_name, metric_type)
            row_text.extend([name, val])
            row_colors.extend(['white', color])
        else:
            row_text.extend(['', ''])
            row_colors.extend(['white', 'white'])
        
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=['Metric', 'Val', 'Metric', 'Val'],
        colColours=['#34495e'] * 4,
        loc='upper center',
        cellLoc='center',
        colWidths=[0.28, 0.18, 0.28, 0.18]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.scale(1.05, 0.95)
    
    for j in range(4):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold', fontsize=5)
        cell.set_height(0.05)


def _plot_parameters_on_ax(ax, params, bounds):
    """Plot calibrated parameters as horizontal bar chart with compact layout."""
    param_names = list(params.keys())
    param_values = list(params.values())
    n_params = len(param_names)
    
    positions = []
    colors = []
    color_palette = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', 
                     '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#d35400']
    
    for i, (name, value) in enumerate(params.items()):
        if name in bounds:
            low, high = bounds[name]
            pct = (value - low) / (high - low) * 100 if high > low else 50
            pct = np.clip(pct, 0, 100)
        else:
            pct = 50
        positions.append(pct)
        colors.append(color_palette[i % len(color_palette)])
    
    # Dynamic bar height based on number of parameters
    bar_height = min(0.7, max(0.3, 12 / n_params))
    y_pos = np.arange(n_params)
    bars = ax.barh(y_pos, positions, color=colors, height=bar_height, alpha=0.85)
    
    # Value labels - always inside if bar is big enough, otherwise to the right of bar but clipped
    for i, (bar, val) in enumerate(zip(bars, param_values)):
        bar_width = bar.get_width()
        # Format value compactly
        if abs(val) >= 100:
            label = f'{val:.0f}'
        elif abs(val) >= 10:
            label = f'{val:.1f}'
        elif abs(val) >= 1:
            label = f'{val:.2f}'
        else:
            label = f'{val:.3f}'
        
        if bar_width > 40:
            # Inside bar
            ax.text(bar_width - 1, bar.get_y() + bar.get_height()/2,
                    label, va='center', ha='right', fontsize=3.5, color='white', fontweight='bold',
                    clip_on=True)
        elif bar_width > 15:
            # Just outside bar but still visible
            ax.text(bar_width + 1, bar.get_y() + bar.get_height()/2,
                    label, va='center', ha='left', fontsize=3.5, clip_on=True)
        else:
            # Small bar - put label a bit further right
            ax.text(max(bar_width + 1, 8), bar.get_y() + bar.get_height()/2,
                    label, va='center', ha='left', fontsize=3.5, clip_on=True)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=4.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Position in bounds (%)', fontsize=5)
    ax.tick_params(axis='x', labelsize=4.5)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.3)
    ax.axvspan(0, 5, alpha=0.1, color='red')
    ax.axvspan(95, 100, alpha=0.1, color='red')


def _plot_fdc_on_ax(ax, obs, sim):
    """Plot flow duration curve on axes."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    
    obs_sorted = np.sort(obs_v)[::-1]
    sim_sorted = np.sort(sim_v)[::-1]
    n = len(obs_sorted)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100
    
    ax.semilogy(exceedance, obs_sorted, color='#e74c3c', linewidth=1, 
                label='Observed', alpha=0.9)
    ax.semilogy(exceedance, sim_sorted, color='#3498db', linewidth=1, 
                label='Simulated', alpha=0.9, linestyle='--')
    
    ax.set_xlabel('Exceedance (%)', fontsize=6)
    ax.set_ylabel('Flow (ML/day)', fontsize=6)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc='upper right', fontsize=5)
    ax.set_xlim(0, 100)


def _plot_scatter_on_ax(ax, obs, sim):
    """Plot scatter plot on axes with log scale."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    
    # Use only positive values for log scale
    pos_mask = (obs_v > 0) & (sim_v > 0)
    obs_pos = obs_v[pos_mask]
    sim_pos = sim_v[pos_mask]
    
    if len(obs_pos) == 0:
        ax.text(0.5, 0.5, 'No positive data', ha='center', va='center', fontsize=8)
        return
    
    ax.scatter(obs_pos, sim_pos, alpha=0.2, s=3, c='#3498db', edgecolors='none')
    
    # Log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 1:1 line
    min_val = max(0.1, min(np.nanmin(obs_pos), np.nanmin(sim_pos)) * 0.8)
    max_val = max(np.nanmax(obs_pos), np.nanmax(sim_pos)) * 1.2
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=0.8, 
            label='1:1 Line', alpha=0.7)
    
    ax.set_xlabel('Observed (ML/day)', fontsize=6)
    ax.set_ylabel('Simulated (ML/day)', fontsize=6)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(True, alpha=0.3, linewidth=0.3, which='both')
    ax.legend(loc='upper left', fontsize=5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')


def _render_signatures_table_on_ax(ax, report):
    """Render ALL hydrologic signatures grouped by category in a compact table."""
    try:
        from pyrrm.analysis.signatures import (
            compute_all_signatures, signature_percent_error, SIGNATURE_CATEGORIES
        )
    except ImportError:
        ax.text(0.5, 0.5, 'Signatures module not available', 
                transform=ax.transAxes, ha='center', va='center', fontsize=7)
        return
    
    obs_sigs = compute_all_signatures(report.observed, report.dates)
    sim_sigs = compute_all_signatures(report.simulated, report.dates)
    
    # Category abbreviations and background colors (light pastels)
    CATEGORY_ABBREV = {
        "Magnitude": ("Mag", "#d6eaf8"),      # Light blue
        "Variability": ("Var", "#e8daef"),    # Light purple
        "Timing": ("Tim", "#fdebd0"),         # Light orange
        "Flow Duration Curve": ("FDC", "#d5f5e3"),  # Light teal
        "Frequency": ("Frq", "#fadbd8"),      # Light red
        "Recession": ("Rec", "#d5d8dc"),      # Light gray
        "Baseflow": ("BF", "#d5f4e6"),        # Light green
        "Event": ("Evt", "#fef9e7"),          # Light yellow
        "Seasonality": ("Sea", "#d4e6f1"),    # Light steel blue
    }
    
    # Build signature to category mapping
    sig_to_category = {}
    for cat_name, sig_list in SIGNATURE_CATEGORIES.items():
        for sig_id in sig_list:
            sig_to_category[sig_id] = cat_name
    
    # Build signature items grouped by category
    sig_items = []
    for cat_name, sig_list in SIGNATURE_CATEGORIES.items():
        abbrev, cat_color = CATEGORY_ABBREV.get(cat_name, ("?", "#95a5a6"))
        for sig_id in sig_list:
            obs_val = obs_sigs.get(sig_id)
            sim_val = sim_sigs.get(sig_id)
            pct_err = signature_percent_error(obs_val, sim_val) if obs_val is not None else None
            
            if pct_err is not None and not np.isnan(pct_err):
                err_str = f"{pct_err:+.0f}%"
                err_color = '#d5f4e6' if abs(pct_err) <= 10 else ('#fef9e7' if abs(pct_err) <= 20 else '#fadbd8')
            else:
                err_str = "—"
                err_color = '#f0f0f0'
            
            # Truncate name
            short_name = sig_id[:7] if len(sig_id) > 7 else sig_id
            sig_items.append((abbrev, short_name, err_str, cat_color, err_color))
    
    # 3-column layout (Cat | Name | Err) x 3
    n_sigs = len(sig_items)
    n_rows = (n_sigs + 2) // 3
    
    cell_text = []
    cell_colors = []
    
    for i in range(n_rows):
        row_text = []
        row_colors = []
        
        for col in range(3):
            idx = i + col * n_rows
            if idx < len(sig_items):
                abbrev, name, err, cat_color, err_color = sig_items[idx]
                row_text.extend([abbrev, name, err])
                row_colors.extend([cat_color, 'white', err_color])
            else:
                row_text.extend(['', '', ''])
                row_colors.extend(['white', 'white', 'white'])
        
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=['Cat', 'Signature', 'Err', 'Cat', 'Signature', 'Err', 'Cat', 'Signature', 'Err'],
        colColours=['#34495e'] * 9,
        loc='upper center',
        cellLoc='center',
        colWidths=[0.06, 0.13, 0.10, 0.06, 0.13, 0.10, 0.06, 0.13, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(4.5)
    table.scale(1.1, 0.92)
    
    # Style header
    for j in range(9):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold', fontsize=4.5)
        cell.set_height(0.045)


# =============================================================================
# LANDSCAPE SINGLE-PAGE EXPORTS (HTML-based - legacy)
# =============================================================================

def _render_landscape_metrics_table(report: 'CalibrationReport') -> str:
    """Render a compact metrics table for landscape layout."""
    from pyrrm.visualization.report_plots import _calculate_basic_metrics
    
    metrics = _calculate_basic_metrics(report.observed, report.simulated)
    
    # Select key metrics for compact display
    KEY_METRICS = [
        ("NSE", "NSE"),
        ("NSE_log", "NSE (log)"),
        ("KGE", "KGE"),
        ("KGE_np", "KGE_np"),
        ("RMSE", "RMSE"),
        ("PBIAS", "PBIAS"),
        ("FHV", "FHV"),
        ("FLV", "FLV"),
        ("Sig_BFI", "BFI Err"),
        ("Sig_Flash", "Flash Err"),
    ]
    
    rows = []
    for metric_id, display_name in KEY_METRICS:
        value = metrics.get(metric_id, np.nan)
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            formatted = "N/A"
            badge_class = "badge-neutral"
        elif metric_id in {"PBIAS", "FHV", "FLV", "FMV", "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"}:
            formatted = f"{value:+.1f}%"
            badge_class = "badge-good" if abs(value) <= 10 else ("badge-ok" if abs(value) <= 20 else "badge-poor")
        elif metric_id in {"RMSE", "MAE", "SDEB"}:
            formatted = f"{value:.2f}"
            badge_class = "badge-neutral"
        else:
            formatted = f"{value:.3f}"
            badge_class = "badge-good" if value >= 0.7 else ("badge-ok" if value >= 0.5 else "badge-poor")
        
        rows.append(f'<tr><td>{display_name}</td><td><span class="metric-badge {badge_class}">{formatted}</span></td></tr>')
    
    return f"""
    <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _render_landscape_parameters(report: 'CalibrationReport') -> str:
    """Render a compact parameter bar chart for landscape layout."""
    params = report.result.best_parameters
    bounds = report.parameter_bounds or {}
    
    rows = []
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    for i, (param, value) in enumerate(params.items()):
        color = colors[i % len(colors)]
        
        if param in bounds:
            low, high = bounds[param]
            pct = (value - low) / (high - low) * 100 if high > low else 50
            pct = np.clip(pct, 0, 100)
        else:
            pct = 50
        
        rows.append(f"""
        <div class="param-row">
            <span class="param-name">{param}</span>
            <div class="param-bar-container">
                <div class="param-bar" style="width: {pct}%; background: {color};"></div>
                <span class="param-value">{value:.3g}</span>
            </div>
        </div>
        """)
    
    return f'<div class="params-compact">{"".join(rows)}</div>'


def _render_landscape_signatures(report: 'CalibrationReport') -> str:
    """Render a compact signatures grid for landscape layout."""
    try:
        from pyrrm.analysis.signatures import compute_all_signatures, signature_percent_error
    except ImportError:
        return '<p style="font-size: 7pt; color: #888;">Signatures unavailable</p>'
    
    obs_sigs = compute_all_signatures(report.observed, report.dates)
    sim_sigs = compute_all_signatures(report.simulated, report.dates)
    
    items = []
    for sig_id in obs_sigs.keys():
        obs_val = obs_sigs.get(sig_id)
        sim_val = sim_sigs.get(sig_id)
        pct_err = signature_percent_error(obs_val, sim_val) if obs_val is not None else None
        
        if pct_err is not None and not np.isnan(pct_err):
            err_str = f"{pct_err:+.0f}%"
            color = "#27ae60" if abs(pct_err) <= 10 else ("#f39c12" if abs(pct_err) <= 20 else "#e74c3c")
        else:
            err_str = "—"
            color = "#888"
        
        # Truncate signature name
        short_name = sig_id[:12] + "…" if len(sig_id) > 12 else sig_id
        
        items.append(f"""
        <div class="sig-item">
            <span class="sig-name" title="{sig_id}">{short_name}</span>
            <span class="sig-error" style="background: {color}; color: white;">{err_str}</span>
        </div>
        """)
    
    return f'<div class="signatures-compact">{"".join(items)}</div>'


def export_report_card_landscape_html(
    report: 'CalibrationReport',
    include_css: bool = True,
) -> str:
    """
    Export a calibration report card as a single-page landscape HTML.
    
    Uses matplotlib to generate a unified figure, then embeds it as a PNG
    in a minimal HTML wrapper for consistent rendering.
    
    Args:
        report: CalibrationReport instance to export
        include_css: Whether to include CSS styles (unused, kept for API compatibility)
    
    Returns:
        Complete HTML document as string with embedded matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig = generate_matplotlib_report_card(report)
    
    # Save to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    
    # Encode as base64
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Get title for HTML
    catchment_info = report.catchment_info
    catchment_name = catchment_info.get('name', 'Unknown Catchment')
    gauge_id = catchment_info.get('gauge_id', '')
    title = f"{catchment_name}"
    if gauge_id:
        title += f" ({gauge_id})"
    
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Card - {title}</title>
    <style>
        @page {{ size: A4 landscape; margin: 5mm; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #f5f6fa;
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .report-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 10px;
            max-width: 100%;
        }}
        .report-image {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .report-container {{ box-shadow: none; border-radius: 0; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <img src="data:image/png;base64,{img_b64}" alt="Calibration Report Card" class="report-image">
    </div>
</body>
</html>
"""
    return html


def export_report_card_landscape_pdf(report: 'CalibrationReport') -> bytes:
    """
    Export a calibration report card as a single-page landscape PDF.
    
    Uses matplotlib to generate a unified figure with all report card components,
    then saves directly to PDF format.
    
    Args:
        report: CalibrationReport instance to export
    
    Returns:
        PDF file content as bytes
    """
    import matplotlib.pyplot as plt
    
    fig = generate_matplotlib_report_card(report)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_report_card_interactive_html(report: 'CalibrationReport') -> str:
    """
    Export a calibration report card as an interactive HTML with Plotly figures.
    
    This version embeds the full Plotly.js library and JSON figure data,
    allowing for interactive zoom, pan, and hover features.
    
    Args:
        report: CalibrationReport instance to export
    
    Returns:
        Complete HTML document as string with embedded interactive figures
    """
    from pyrrm.visualization.report_plots import plot_report_card_components
    import json
    
    components = plot_report_card_components(report)
    header = components["header"]
    result = report.result
    
    # Get title
    title = header["catchment_name"]
    if header["gauge_id"]:
        title += f" ({header['gauge_id']})"
    
    # Get model name
    model_name = "Unknown"
    if hasattr(report, 'model_config') and report.model_config:
        model_class = report.model_config.get('class_name', '')
        if model_class:
            model_name = model_class
    if model_name == "Unknown" and hasattr(report, 'experiment_name') and report.experiment_name:
        parts = report.experiment_name.split('_')
        if len(parts) >= 2:
            model_name = parts[1].title()
    
    # Convert figures to JSON
    def fig_to_json(fig):
        if fig is None:
            return "null"
        return json.dumps(fig.to_dict())
    
    hydro_linear_json = fig_to_json(components.get("hydrograph_linear"))
    hydro_log_json = fig_to_json(components.get("hydrograph_log"))
    fdc_json = fig_to_json(components.get("fdc"))
    scatter_json = fig_to_json(components.get("scatter"))
    
    # Render tables
    metrics_html = _render_landscape_metrics_table(report)
    params_html = _render_landscape_parameters(report)
    signatures_html = _render_landscape_signatures(report)
    
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    period_start, period_end = report.calibration_period
    
    # Interactive CSS (adjusted for browser viewing)
    interactive_css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 12px;
        line-height: 1.4;
        color: #2c3e50;
        background: #f5f6fa;
        padding: 10px;
    }
    
    .report-container {
        max-width: 1400px;
        margin: 0 auto;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    .header-strip {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        color: white;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }
    
    .header-title { font-size: 20px; font-weight: 700; }
    .header-subtitle { font-size: 12px; opacity: 0.9; margin-top: 4px; }
    
    .header-meta {
        display: flex;
        gap: 25px;
    }
    
    .header-meta-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .header-meta-label { font-size: 10px; opacity: 0.8; text-transform: uppercase; }
    .header-meta-value { font-weight: 600; font-size: 14px; }
    
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        grid-template-rows: auto auto;
        gap: 10px;
        padding: 10px;
    }
    
    .panel {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        overflow: hidden;
    }
    
    .panel-header {
        background: #34495e;
        color: white;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .panel-content {
        padding: 8px;
        min-height: 200px;
    }
    
    .plotly-container {
        width: 100%;
        height: 100%;
        min-height: 180px;
    }
    
    .metrics-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .metrics-table th { background: #34495e; color: white; padding: 6px 8px; text-align: left; }
    .metrics-table td { padding: 5px 8px; border-bottom: 1px solid #eee; }
    .metrics-table tr:nth-child(even) { background: #f8f9fa; }
    
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        color: white;
        font-weight: 600;
        font-size: 10px;
        min-width: 50px;
        text-align: center;
    }
    
    .badge-good { background: #27ae60; }
    .badge-ok { background: #f39c12; }
    .badge-poor { background: #e74c3c; }
    .badge-neutral { background: #7f8c8d; }
    
    .params-compact { width: 100%; font-size: 11px; }
    .param-row { display: flex; align-items: center; margin-bottom: 4px; }
    .param-name { width: 60px; font-family: monospace; font-size: 10px; text-align: right; padding-right: 8px; }
    .param-bar-container { flex: 1; height: 14px; background: #ecf0f1; border-radius: 3px; position: relative; }
    .param-bar { height: 100%; border-radius: 3px; }
    .param-value { position: absolute; right: 4px; top: 0; font-size: 9px; line-height: 14px; color: #333; }
    
    .signatures-compact {
        width: 100%;
        font-size: 10px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3px;
        max-height: 220px;
        overflow-y: auto;
    }
    
    .sig-item {
        display: flex;
        justify-content: space-between;
        padding: 2px 4px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .sig-name { color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90px; }
    .sig-error { font-weight: 600; padding: 1px 4px; border-radius: 2px; font-size: 9px; }
    
    .footer-strip {
        background: #f8f9fa;
        padding: 8px 20px;
        font-size: 11px;
        color: #7f8c8d;
        display: flex;
        justify-content: space-between;
        border-top: 1px solid #e0e0e0;
    }
    
    @media (max-width: 1000px) {
        .content-grid { grid-template-columns: 1fr 1fr; }
    }
    
    @media (max-width: 700px) {
        .content-grid { grid-template-columns: 1fr; }
        .header-strip { flex-direction: column; text-align: center; }
        .header-meta { justify-content: center; }
    }
    """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Report Card - {title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{interactive_css}</style>
</head>
<body>
    <div class="report-container">
        <!-- Header Strip -->
        <div class="header-strip">
            <div>
                <div class="header-title">{title}</div>
                <div class="header-subtitle">{model_name} Model • {period_start} to {period_end}</div>
            </div>
            <div class="header-meta">
                <div class="header-meta-item">
                    <span class="header-meta-label">Method</span>
                    <span class="header-meta-value">{result.method}</span>
                </div>
                <div class="header-meta-item">
                    <span class="header-meta-label">Objective</span>
                    <span class="header-meta-value">{result.objective_name}</span>
                </div>
                <div class="header-meta-item">
                    <span class="header-meta-label">Best Value</span>
                    <span class="header-meta-value">{result.best_objective:.4f}</span>
                </div>
            </div>
        </div>
        
        <!-- Content Grid -->
        <div class="content-grid">
            <!-- Row 1 -->
            <div class="panel">
                <div class="panel-header">Hydrograph (Linear)</div>
                <div class="panel-content">
                    <div id="plot-hydro-linear" class="plotly-container"></div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">Diagnostic Metrics</div>
                <div class="panel-content">
                    {metrics_html}
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">Calibrated Parameters</div>
                <div class="panel-content">
                    {params_html}
                </div>
            </div>
            
            <!-- Row 2 -->
            <div class="panel">
                <div class="panel-header">Hydrograph (Log) / FDC</div>
                <div class="panel-content" style="display: flex; flex-direction: column; gap: 5px;">
                    <div id="plot-hydro-log" class="plotly-container" style="flex: 1; min-height: 90px;"></div>
                    <div id="plot-fdc" class="plotly-container" style="flex: 1; min-height: 90px;"></div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">Scatter Plot</div>
                <div class="panel-content">
                    <div id="plot-scatter" class="plotly-container"></div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">Hydrologic Signatures (% Error)</div>
                <div class="panel-content" style="overflow-y: auto;">
                    {signatures_html}
                </div>
            </div>
        </div>
        
        <!-- Footer Strip -->
        <div class="footer-strip">
            <span>Generated by pyrrm (Python Rainfall-Runoff Models)</span>
            <span>{generated_date}</span>
        </div>
    </div>
    
    <script>
        // Configure Plotly defaults for compact display
        const config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false
        }};
        
        // Hydrograph Linear
        const hydroLinear = {hydro_linear_json};
        if (hydroLinear) {{
            hydroLinear.layout = hydroLinear.layout || {{}};
            hydroLinear.layout.margin = {{l: 50, r: 20, t: 30, b: 40}};
            hydroLinear.layout.height = 180;
            hydroLinear.layout.showlegend = true;
            hydroLinear.layout.legend = {{x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {{size: 9}}}};
            Plotly.newPlot('plot-hydro-linear', hydroLinear.data, hydroLinear.layout, config);
        }}
        
        // Hydrograph Log
        const hydroLog = {hydro_log_json};
        if (hydroLog) {{
            hydroLog.layout = hydroLog.layout || {{}};
            hydroLog.layout.margin = {{l: 50, r: 20, t: 10, b: 30}};
            hydroLog.layout.height = 85;
            hydroLog.layout.showlegend = false;
            Plotly.newPlot('plot-hydro-log', hydroLog.data, hydroLog.layout, config);
        }}
        
        // FDC
        const fdc = {fdc_json};
        if (fdc) {{
            fdc.layout = fdc.layout || {{}};
            fdc.layout.margin = {{l: 50, r: 20, t: 10, b: 30}};
            fdc.layout.height = 85;
            fdc.layout.showlegend = false;
            Plotly.newPlot('plot-fdc', fdc.data, fdc.layout, config);
        }}
        
        // Scatter
        const scatter = {scatter_json};
        if (scatter) {{
            scatter.layout = scatter.layout || {{}};
            scatter.layout.margin = {{l: 50, r: 20, t: 20, b: 40}};
            scatter.layout.height = 190;
            scatter.layout.showlegend = false;
            Plotly.newPlot('plot-scatter', scatter.data, scatter.layout, config);
        }}
    </script>
</body>
</html>
"""
    return html


def export_batch_report_landscape_pdf(
    reports: Dict[str, 'CalibrationReport'],
    title: str = "Batch Calibration Report",
) -> bytes:
    """
    Export multiple calibration report cards as a landscape PDF (one page per report).
    
    Uses matplotlib PdfPages to create a multi-page PDF with each report on its
    own page, ensuring consistent layout across all pages.
    
    Args:
        reports: Dictionary mapping experiment keys to CalibrationReport instances
        title: Title for the batch report (unused, kept for API compatibility)
    
    Returns:
        PDF file content as bytes
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    buf = io.BytesIO()
    
    with PdfPages(buf) as pdf:
        for exp_key, report in reports.items():
            fig = generate_matplotlib_report_card(report)
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    
    buf.seek(0)
    return buf.getvalue()


def _export_batch_report_landscape_pdf_legacy(
    reports: Dict[str, 'CalibrationReport'],
    title: str = "Batch Calibration Report",
) -> bytes:
    """
    Legacy: Export using weasyprint (kept for reference).
    """
    if not _check_weasyprint():
        raise ImportError(
            "weasyprint is required for PDF export. "
            "Install with: pip install weasyprint"
        )
    
    # Build combined HTML with page breaks
    all_pages = []
    for i, (exp_key, report) in enumerate(reports.items()):
        page_html = export_report_card_landscape_html(report, include_css=False)
        # Extract just the body content
        import re
        body_match = re.search(r'<body>(.*?)</body>', page_html, re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
            if i > 0:
                all_pages.append('<div style="page-break-before: always;"></div>')
            all_pages.append(body_content)
    
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{REPORT_CSS_LANDSCAPE}</style>
</head>
<body>
    {''.join(all_pages)}
</body>
</html>
"""
    
    from weasyprint import HTML
    pdf_bytes = HTML(string=combined_html).write_pdf()
    
    return pdf_bytes
