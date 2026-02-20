"""
Network visualization: Mermaid diagrams, styled DataFrames, and matplotlib plots.

Provides visualisation at three stages:
  1. Pre-calibration: network topology, per-node config, data quality
  2. Post-calibration: results, hydrographs, flow duration curves

All methods work in Jupyter notebooks producing styled DataFrames
(colour-coded cells) and Mermaid diagrams (via IPython.display.Markdown).

Example:
    >>> network.display()                       # Mermaid topology diagram
    >>> runner.config_summary()                  # styled config table
    >>> result.to_styled_dataframe()             # styled results table
    >>> result.plot_network_hydrographs()        # matplotlib hydrograph grid
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyrrm.network.topology import CatchmentNetwork, CatchmentNode, NetworkLink
    from pyrrm.network.runner import (
        CatchmentNetworkRunner, NetworkCalibrationResult, ResolvedNodeConfig,
    )
    from pyrrm.network.data import DataValidationReport
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# =========================================================================
# A. Network Topology Diagram (Mermaid)
# =========================================================================

def network_to_mermaid(
    network: 'CatchmentNetwork',
    show_areas: bool = True,
    show_routing: bool = True,
    show_wavefronts: bool = False,
    highlight_gauged: bool = True,
) -> str:
    """Generate a Mermaid flowchart string for the network.

    Args:
        network: CatchmentNetwork instance.
        show_areas: Show catchment area on node labels.
        show_routing: Show routing K, m on link edges.
        show_wavefronts: Colour nodes by wavefront level.
        highlight_gauged: Bold border for gauged nodes.

    Returns:
        Mermaid-formatted string.
    """
    lines = ['graph TD']

    wf_map: Dict[str, int] = {}
    if show_wavefronts:
        for wf_idx, wf in enumerate(network.wavefronts()):
            for nid in wf:
                wf_map[nid] = wf_idx

    wf_colors = ['#e3f2fd', '#fff3e0', '#e8f5e9', '#fce4ec', '#f3e5f5', '#fffde7']

    for nid, node in network.nodes.items():
        parts = [nid]
        if node.name:
            parts.append(node.name)
        if show_areas and node.area_km2 > 0:
            parts.append(f"{node.area_km2:.0f} km²")
        if node.calibration_config and node.calibration_config.model_class:
            parts.append(node.calibration_config.model_class)

        label = '<br/>'.join(parts)

        if highlight_gauged and node.is_gauged:
            lines.append(f'    {_safe_id(nid)}["{label}"]')
        else:
            lines.append(f'    {_safe_id(nid)}("{label}")')

    for nid, node in network.nodes.items():
        ds = node.downstream_id
        if ds and ds in network.nodes:
            link = network.get_link(nid, ds)
            if link and show_routing and link.routing_method != 'none':
                params = link.routing_params or {}
                k = params.get('routing_K', '?')
                m = params.get('routing_m', '?')
                label = f"K={k}, m={m}"
                lines.append(f'    {_safe_id(nid)} -->|"{label}"| {_safe_id(ds)}')
            else:
                lines.append(f'    {_safe_id(nid)} --> {_safe_id(ds)}')

    if show_wavefronts and wf_map:
        for wf_idx in sorted(set(wf_map.values())):
            nodes_in_wf = [nid for nid, w in wf_map.items() if w == wf_idx]
            color = wf_colors[wf_idx % len(wf_colors)]
            for nid in nodes_in_wf:
                lines.append(f'    style {_safe_id(nid)} fill:{color}')

    return '\n'.join(lines)


def display_network(network: 'CatchmentNetwork', **kwargs) -> None:
    """Render the Mermaid diagram in a Jupyter notebook."""
    try:
        from IPython.display import display, Markdown
        mermaid = network_to_mermaid(network, **kwargs)
        display(Markdown(f"```mermaid\n{mermaid}\n```"))
    except ImportError:
        print(network_to_mermaid(network, **kwargs))


def plot_network(
    network: 'CatchmentNetwork',
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
) -> 'Figure':
    """Matplotlib static plot of the network DAG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=figsize)

    wavefronts = network.wavefronts()
    n_wf = len(wavefronts)

    pos: Dict[str, Tuple[float, float]] = {}
    for wf_idx, wf_nodes in enumerate(wavefronts):
        y = 1.0 - (wf_idx / max(n_wf - 1, 1))
        n_in_wf = len(wf_nodes)
        for i, nid in enumerate(wf_nodes):
            x = (i + 0.5) / n_in_wf
            pos[nid] = (x, y)

    for nid, node in network.nodes.items():
        x, y = pos[nid]
        color = '#4fc3f7' if node.is_gauged else '#e0e0e0'
        label = f"{nid}\n{node.name}" if node.name else nid
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.06, y - 0.03), 0.12, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='#333333', linewidth=1.5,
        ))
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    for nid, node in network.nodes.items():
        ds = node.downstream_id
        if ds and ds in pos:
            x1, y1 = pos[nid]
            x2, y2 = pos[ds]
            ax.annotate(
                '', xy=(x2, y2 + 0.03), xytext=(x1, y1 - 0.03),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5),
            )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Catchment Network', fontsize=14, fontweight='bold')

    gauged_patch = mpatches.Patch(color='#4fc3f7', label='Gauged')
    ungauged_patch = mpatches.Patch(color='#e0e0e0', label='Ungauged')
    ax.legend(handles=[gauged_patch, ungauged_patch], loc='lower right')

    fig.tight_layout()
    return fig


# =========================================================================
# B. Configuration Summary Tables
# =========================================================================

def config_summary(runner: 'CatchmentNetworkRunner') -> 'pd.io.formats.style.Styler':
    """Styled DataFrame: one row per node showing resolved calibration config."""
    rows = []
    configs = runner.get_resolved_configs()
    wavefronts = runner.network.wavefronts()
    wf_map = {}
    for wf_idx, wf in enumerate(wavefronts):
        for nid in wf:
            wf_map[nid] = wf_idx

    for nid in runner.network.topological_order():
        node = runner.network.get_node(nid)
        cfg = configs[nid]
        nc = node.calibration_config or None

        is_override = {}
        is_override['model'] = nc is not None and nc.model_class is not None
        is_override['objective'] = nc is not None and nc.objective is not None
        is_override['algorithm'] = nc is not None and nc.algorithm is not None
        is_override['warmup'] = nc is not None and nc.warmup_period is not None

        alg_method = cfg.algorithm.get('method', '?')
        alg_evals = cfg.algorithm.get('max_evals', '?')

        obs_period = ''
        if node.is_gauged and node.inputs is not None and len(node.inputs) > 0:
            obs_period = f"{node.inputs.index[0].date()} to {node.inputs.index[-1].date()}"

        row = {
            'node_id': nid,
            'name': node.name,
            'area_km2': node.area_km2,
            'wavefront': wf_map.get(nid, '?'),
            'gauged': node.is_gauged,
            'model': cfg.model_class,
            'objective': getattr(cfg.objective, 'name', str(cfg.objective)),
            'transformation': str(cfg.flow_transformation) if cfg.flow_transformation else 'none',
            'algorithm': alg_method,
            'max_evals': alg_evals,
            'warmup_days': cfg.warmup_period,
            'observed_period': obs_period,
            'n_upstream': len(runner.network.upstream_ids(nid)),
        }
        row['_override_model'] = is_override['model']
        row['_override_obj'] = is_override['objective']
        row['_override_alg'] = is_override['algorithm']
        row['_override_warmup'] = is_override['warmup']
        rows.append(row)

    df = pd.DataFrame(rows)

    def _highlight_overrides(row):
        styles = [''] * len(row)
        col_map = {
            'model': '_override_model',
            'objective': '_override_obj',
            'algorithm': '_override_alg',
            'warmup_days': '_override_warmup',
        }
        for col, flag in col_map.items():
            if flag in row.index and row[flag]:
                idx = list(row.index).index(col)
                styles[idx] = 'background-color: #fff9c4; font-weight: bold'
        if not row.get('gauged', True):
            styles = ['color: #999999' if s == '' else s for s in styles]
        return styles

    display_cols = [c for c in df.columns if not c.startswith('_')]
    styled = df[display_cols].style.apply(
        lambda row: _highlight_overrides(df.iloc[row.name]),
        axis=1,
    )
    styled = styled.set_caption("Network Calibration Configuration")

    return styled


def link_config_summary(runner: 'CatchmentNetworkRunner') -> 'pd.io.formats.style.Styler':
    """Styled DataFrame: one row per link showing routing configuration."""
    rows = []
    for (uid, did), link in runner.network.links.items():
        params = link.routing_params or {}
        bounds = link.routing_bounds or {}
        row = {
            'upstream_id': uid,
            'downstream_id': did,
            'routing_method': link.routing_method,
            'K_init': params.get('routing_K', ''),
            'm_init': params.get('routing_m', ''),
            'n_sub_init': params.get('routing_n_subreaches', ''),
            'calibrate': link.calibrate_routing,
            'K_bounds': str(bounds.get('routing_K', '')) if bounds else '',
            'm_bounds': str(bounds.get('routing_m', '')) if bounds else '',
        }
        rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['upstream_id', 'downstream_id', 'routing_method',
                 'K_init', 'm_init', 'n_sub_init', 'calibrate', 'K_bounds', 'm_bounds']
    )

    def _highlight_calibrate(val):
        if val is True:
            return 'background-color: #e8f5e9'
        elif val is False:
            return 'background-color: #ffebee'
        return ''

    styled = df.style.map(_highlight_calibrate, subset=['calibrate'])
    styled = styled.set_caption("Link Routing Configuration")
    return styled


def data_summary(runner: 'CatchmentNetworkRunner') -> 'pd.io.formats.style.Styler':
    """Styled DataFrame: per-node data availability and quality."""
    rows = []
    for nid in runner.network.topological_order():
        node = runner.network.get_node(nid)
        row = {
            'node_id': nid,
            'name': node.name,
        }
        if node.inputs is not None and len(node.inputs) > 0:
            row['input_start'] = str(node.inputs.index[0].date())
            row['input_end'] = str(node.inputs.index[-1].date())
            row['n_days'] = len(node.inputs)
            p_miss = node.inputs['precipitation'].isna().mean() * 100 if 'precipitation' in node.inputs else 0
            e_miss = node.inputs['pet'].isna().mean() * 100 if 'pet' in node.inputs else 0
            row['precip_gaps_%'] = round(p_miss, 1)
            row['pet_gaps_%'] = round(e_miss, 1)
        else:
            row['input_start'] = 'N/A'
            row['input_end'] = 'N/A'
            row['n_days'] = 0
            row['precip_gaps_%'] = 0
            row['pet_gaps_%'] = 0

        if node.observed is not None:
            q_miss = np.isnan(node.observed.astype(float)).mean() * 100
            row['flow_gaps_%'] = round(q_miss, 1)
        else:
            row['flow_gaps_%'] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    def _colour_gaps(val):
        if val is None:
            return 'color: #999999'
        try:
            val = float(val)
        except (TypeError, ValueError):
            return ''
        if val < 1:
            return 'background-color: #c8e6c9'
        elif val < 5:
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #ffcdd2'

    gap_cols = [c for c in df.columns if 'gaps' in c]
    styled = df.style.map(_colour_gaps, subset=gap_cols)
    styled = styled.set_caption("Network Data Summary")
    return styled


# =========================================================================
# C. Post-Calibration Result Visualizations
# =========================================================================

def result_to_styled_dataframe(
    result: 'NetworkCalibrationResult',
) -> 'pd.io.formats.style.Styler':
    """Styled summary: one row per node, colour-coded by objective performance."""
    df = result.to_dataframe()

    def _colour_objective(val):
        try:
            val = float(val)
        except (TypeError, ValueError):
            return 'background-color: #ffcdd2'
        if val >= 0.7:
            return 'background-color: #c8e6c9'
        elif val >= 0.5:
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #ffcdd2'

    styled = df.style
    if 'best_objective' in df.columns:
        styled = styled.map(_colour_objective, subset=['best_objective'])
    if 'success' in df.columns:
        styled = styled.map(
            lambda v: 'background-color: #ffcdd2' if v is False else '',
            subset=['success'],
        )
    styled = styled.set_caption(f"Network Calibration Results ({result.strategy})")
    return styled


def result_link_styled(
    result: 'NetworkCalibrationResult',
) -> 'pd.io.formats.style.Styler':
    """Styled link summary with calibrated routing parameters."""
    df = result.link_summary()
    styled = df.style.set_caption("Calibrated Link Routing Parameters")
    return styled


def result_to_mermaid(
    result: 'NetworkCalibrationResult',
    colour_by: str = 'objective',
    show_objective_values: bool = True,
    show_routing_params: bool = True,
) -> str:
    """Mermaid diagram annotated with calibration results."""
    if result.network is None:
        return 'graph TD\n    ERROR["No network reference"]'

    lines = ['graph TD']

    for nid, node in result.network.nodes.items():
        parts = [nid]
        if node.name:
            parts.append(node.name)
        if nid in result.node_results and show_objective_values:
            obj_val = result.node_results[nid].result.best_objective
            obj_name = result.node_results[nid].result.objective_name
            if not np.isnan(obj_val):
                parts.append(f"{obj_name}={obj_val:.3f}")
        elif nid in result.failures:
            parts.append("FAILED")

        label = '<br/>'.join(parts)
        lines.append(f'    {_safe_id(nid)}["{label}"]')

    for nid, node in result.network.nodes.items():
        ds = node.downstream_id
        if ds and ds in result.network.nodes:
            link_key = (nid, ds)
            if show_routing_params and link_key in result.link_routing_params:
                params = result.link_routing_params[link_key]
                k = params.get('routing_K', '?')
                m = params.get('routing_m', '?')
                label = f"K={k:.1f}, m={m:.2f}" if isinstance(k, float) else f"K={k}, m={m}"
                lines.append(f'    {_safe_id(nid)} -->|"{label}"| {_safe_id(ds)}')
            else:
                lines.append(f'    {_safe_id(nid)} --> {_safe_id(ds)}')

    if colour_by == 'objective':
        for nid in result.node_results:
            obj_val = result.node_results[nid].result.best_objective
            if np.isnan(obj_val):
                continue
            if obj_val >= 0.7:
                lines.append(f'    style {_safe_id(nid)} fill:#c8e6c9')
            elif obj_val >= 0.5:
                lines.append(f'    style {_safe_id(nid)} fill:#fff9c4')
            else:
                lines.append(f'    style {_safe_id(nid)} fill:#ffcdd2')
        for nid in result.failures:
            lines.append(f'    style {_safe_id(nid)} fill:#ef9a9a')

    return '\n'.join(lines)


def display_result(result: 'NetworkCalibrationResult', **kwargs) -> None:
    """Render the results Mermaid diagram in a Jupyter notebook."""
    try:
        from IPython.display import display, Markdown
        mermaid = result_to_mermaid(result, **kwargs)
        display(Markdown(f"```mermaid\n{mermaid}\n```"))
    except ImportError:
        print(result_to_mermaid(result, **kwargs))


def plot_network_hydrographs(
    result: 'NetworkCalibrationResult',
    nodes: Optional[List[str]] = None,
    figsize: Tuple = (16, 4),
    log_scale: bool = False,
) -> 'Figure':
    """Grid of hydrograph panels, one per gauged node in topological order."""
    import matplotlib.pyplot as plt

    if result.network is None:
        raise ValueError("NetworkCalibrationResult has no network reference")

    if nodes is None:
        nodes = [
            nid for nid in result.network.topological_order()
            if nid in result.node_results
            and result.node_results[nid].observed is not None
            and len(result.node_results[nid].observed) > 0
        ]

    if not nodes:
        raise ValueError("No gauged nodes with results to plot")

    n_panels = len(nodes)
    fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels),
                             squeeze=False, sharex=False)

    for i, nid in enumerate(nodes):
        ax = axes[i, 0]
        report = result.node_results[nid]
        obs = report.observed
        sim = report.simulated
        dates = report.dates if hasattr(report, 'dates') and report.dates is not None else range(len(sim))

        n_plot = min(len(obs), len(sim), len(dates))
        ax.plot(dates[:n_plot], obs[:n_plot], 'b-', alpha=0.7, linewidth=0.8, label='Observed')
        ax.plot(dates[:n_plot], sim[:n_plot], 'r-', alpha=0.7, linewidth=0.8, label='Simulated')

        node = result.network.get_node(nid)
        obj_val = report.result.best_objective
        title = f"{nid} ({node.name})" if node.name else nid
        if not np.isnan(obj_val):
            title += f" | {report.result.objective_name}={obj_val:.3f}"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Flow')

        if log_scale:
            ax.set_yscale('log')

    if n_panels > 0:
        axes[-1, 0].set_xlabel('Date')

    fig.suptitle(f"Network Hydrographs ({result.strategy})", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_network_fdc(
    result: 'NetworkCalibrationResult',
    nodes: Optional[List[str]] = None,
) -> 'Figure':
    """Flow duration curves for all gauged nodes in a single figure."""
    import matplotlib.pyplot as plt

    if result.network is None:
        raise ValueError("NetworkCalibrationResult has no network reference")

    if nodes is None:
        nodes = [
            nid for nid in result.network.topological_order()
            if nid in result.node_results
            and result.node_results[nid].observed is not None
            and len(result.node_results[nid].observed) > 0
        ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, nid in enumerate(nodes):
        report = result.node_results[nid]
        obs = np.sort(report.observed[~np.isnan(report.observed)])[::-1]
        sim = np.sort(report.simulated[~np.isnan(report.simulated)])[::-1]

        exc_obs = np.arange(1, len(obs) + 1) / (len(obs) + 1) * 100
        exc_sim = np.arange(1, len(sim) + 1) / (len(sim) + 1) * 100

        color = colors[i % len(colors)]
        node = result.network.get_node(nid)
        label = f"{nid} ({node.name})" if node.name else nid

        ax.plot(exc_obs, obs, '-', color=color, alpha=0.7, label=f'{label} obs')
        ax.plot(exc_sim, sim, '--', color=color, alpha=0.7, label=f'{label} sim')

    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Flow')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f"Flow Duration Curves ({result.strategy})", fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# D. DataValidationReport Display
# =========================================================================

def validation_to_styled_dataframe(
    report: 'DataValidationReport',
) -> 'pd.io.formats.style.Styler':
    """Per-catchment data quality as a colour-coded styled DataFrame."""
    rows = []
    for nid, s in report.per_catchment.items():
        rows.append({
            'node_id': nid,
            'n_records': s.n_records,
            'period_start': s.period[0],
            'period_end': s.period[1],
            'precip_miss_%': round(s.precip_missing_pct, 1),
            'pet_miss_%': round(s.pet_missing_pct, 1),
            'flow_miss_%': round(s.flow_missing_pct, 1) if s.flow_missing_pct is not None else None,
            'longest_gap': s.longest_gap_days,
        })

    df = pd.DataFrame(rows)

    def _colour_missing(val):
        if val is None:
            return 'color: #999999'
        try:
            val = float(val)
        except (TypeError, ValueError):
            return ''
        if val < 1:
            return 'background-color: #c8e6c9'
        elif val < 5:
            return 'background-color: #fff9c4'
        return 'background-color: #ffcdd2'

    miss_cols = [c for c in df.columns if 'miss' in c]
    styled = df.style.map(_colour_missing, subset=miss_cols)
    styled = styled.set_caption("Data Validation: Per-Catchment Quality")
    return styled


def junction_overlap_styled(
    report: 'DataValidationReport',
) -> 'pd.io.formats.style.Styler':
    """Junction overlap analysis as styled DataFrame."""
    rows = []
    for jid, jo in report.junction_overlaps.items():
        eff = jo.effective_calibration_period
        rows.append({
            'junction_id': jid,
            'obs_start': jo.junction_observed_period[0] if jo.junction_observed_period else 'N/A',
            'obs_end': jo.junction_observed_period[1] if jo.junction_observed_period else 'N/A',
            'eff_start': eff[0] if eff else 'N/A',
            'eff_end': eff[1] if eff else 'N/A',
            'eff_days': jo.effective_days,
            'sufficient': jo.is_sufficient,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['junction_id', 'obs_start', 'obs_end', 'eff_start', 'eff_end', 'eff_days', 'sufficient']
    )

    def _colour_sufficient(val):
        if val is False:
            return 'background-color: #ffcdd2; font-weight: bold'
        elif val is True:
            return 'background-color: #c8e6c9'
        return ''

    styled = df.style.map(_colour_sufficient, subset=['sufficient'])
    styled = styled.set_caption("Junction Overlap Analysis")
    return styled


# =========================================================================
# Helper
# =========================================================================

def _safe_id(node_id: str) -> str:
    """Make node_id safe for Mermaid syntax (no special chars)."""
    return node_id.replace(' ', '_').replace('-', '_').replace('.', '_')


__all__ = [
    'network_to_mermaid',
    'display_network',
    'plot_network',
    'config_summary',
    'link_config_summary',
    'data_summary',
    'result_to_styled_dataframe',
    'result_link_styled',
    'result_to_mermaid',
    'display_result',
    'plot_network_hydrographs',
    'plot_network_fdc',
    'validation_to_styled_dataframe',
    'junction_overlap_styled',
]
