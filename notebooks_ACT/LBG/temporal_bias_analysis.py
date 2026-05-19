# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: pyrrm (Python 3.11.14)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Temporal Bias Analysis — LBG Headwater Calibrations
#
# ## Purpose
#
# This notebook conducts a comprehensive **temporal bias analysis** across all
# headwater calibration experiments in the Lower Burley Griffin (LBG) network.
# For each gauge and each calibration (model × objective × algorithm), it
# produces a 3-panel figure showing:
#
# 1. **Daily flow** — observed vs simulated (log scale)
# 2. **Monthly inflow bias** — (sim − obs) in GL/month
# 3. **Cumulative excess inflow** — running total of bias in GL
#
# Drought periods (Millennium Drought 2002–2009, 2017–20 drought) are
# highlighted to reveal how model bias behaves during extended dry spells.
#
# ## Outputs
#
# - **Static Matplotlib PNGs** (one per experiment, saved to disk)
# - **Interactive Plotly figures** (one per gauge, all experiments overlaid
#   with legend-group toggling)

# %%
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyrrm.calibration import BatchResult, parse_experiment_key

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Configuration

# %%
# -- Path discovery --------------------------------------------------------
_this_file = Path(__file__).resolve() if '__file__' in dir() else Path.cwd()
_candidates = [_this_file.parent, Path.cwd()]
PROJECT_ROOT = None
for _c in _candidates:
    _p = _c
    for _ in range(5):
        if (_p / 'pyrrm').is_dir() and (_p / 'notebooks_ACT').is_dir():
            PROJECT_ROOT = _p
            break
        _p = _p.parent
    if PROJECT_ROOT:
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path.cwd()
    logger.warning("Could not discover PROJECT_ROOT; using cwd: %s", PROJECT_ROOT)

RESULTS_DIR = PROJECT_ROOT / 'notebooks_ACT' / 'LBG' / 'results'

HEADWATER_GAUGES = ['410705', '410734', '410772', '410774', '410790']

SAVE_STATIC = True
STATIC_DPI = 300

# -- Drought periods (start, end, label) -----------------------------------
DROUGHT_PERIODS: List[Tuple[str, str, str]] = [
    ('2002-01-01', '2009-12-31', 'Millennium\nDrought'),
    ('2017-07-01', '2020-03-31', 'Drought\n2017–20'),
]

# -- Colour palette ---------------------------------------------------------
CLR_OBS = '#888888'
CLR_SIM = '#D35F5F'
CLR_EXCESS = '#E8998D'
CLR_DEFICIT = '#8DAAE8'
CLR_DROUGHT_BG = (1.0, 0.8, 0.8, 0.18)

# Plotly qualitative palette for 36+ experiments
_PLOTLY_PALETTE = (
    plotly.colors.qualitative.Dark24
    + plotly.colors.qualitative.Alphabet
)

print(f"Project root : {PROJECT_ROOT}")
print(f"Results dir  : {RESULTS_DIR}")
print(f"Gauges       : {HEADWATER_GAUGES}")

# %% [markdown]
# ## Load Batch Results

# %%
batch_results: Dict[str, BatchResult] = {}

for gauge_id in HEADWATER_GAUGES:
    gauge_dir = RESULTS_DIR / gauge_id
    if not gauge_dir.is_dir():
        logger.warning("No results directory for gauge %s", gauge_id)
        continue

    pkl_files = sorted(gauge_dir.rglob('batch_result.pkl'))
    if not pkl_files:
        logger.warning("No batch_result.pkl found for gauge %s", gauge_id)
        continue

    pkl_path = pkl_files[-1]
    br = BatchResult.load(str(pkl_path))
    batch_results[gauge_id] = br
    print(f"  {gauge_id}: {len(br.results)} experiments loaded from {pkl_path.parent.name}")

print(f"\nLoaded {len(batch_results)} gauges, "
      f"{sum(len(br.results) for br in batch_results.values())} total experiments")

# %% [markdown]
# ## Summary Table

# %%
summary_rows = []
for gauge_id, br in batch_results.items():
    rep = next(iter(br.results.values()))
    info = getattr(rep, 'catchment_info', {}) or {}
    area = info.get('area_km2', '—')
    start_date = rep.dates[0].strftime('%Y-%m-%d') if len(rep.dates) else '—'
    end_date = rep.dates[-1].strftime('%Y-%m-%d') if len(rep.dates) else '—'
    n_years = round(len(rep.dates) / 365.25, 1)
    summary_rows.append({
        'Gauge': gauge_id,
        'Area (km²)': area,
        'Start': start_date,
        'End': end_date,
        'Record (yr)': n_years,
        'Experiments': len(br.results),
        'Failures': len(br.failures),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df

# %% [markdown]
# ---
# ## Plotting Functions

# %%
def prepare_bias_data(report, dates=None) -> Dict[str, Any]:
    """Compute bias timeseries from a CalibrationReport.

    Returns a dict consumed by both Matplotlib and Plotly renderers.
    All volumes are in GL (gigalitres).
    """
    sim = np.asarray(report.simulated, dtype=np.float64)
    obs = np.asarray(report.observed, dtype=np.float64)
    dt = dates if dates is not None else report.dates

    bias_ml = sim - obs

    annual_bias_gl = (
        pd.Series(bias_ml, index=dt)
        .resample('YE').sum() / 1000.0
    )

    cumulative_bias_gl = np.cumsum(bias_ml) / 1000.0

    mean_annual_bias = float(annual_bias_gl.mean())

    ab_valid = annual_bias_gl.dropna()
    if len(ab_valid) > 0:
        peak_pos_idx = ab_valid.idxmax()
        peak_neg_idx = ab_valid.idxmin()
        peak_annual_pos = float(ab_valid[peak_pos_idx])
        peak_annual_neg = float(ab_valid[peak_neg_idx])
        peak_annual_pos_date = peak_pos_idx
        peak_annual_neg_date = peak_neg_idx
    else:
        peak_annual_pos = peak_annual_neg = 0.0
        peak_annual_pos_date = peak_annual_neg_date = dt[0]

    peak_cum_idx = int(np.argmax(np.abs(cumulative_bias_gl)))
    peak_cumulative = float(cumulative_bias_gl[peak_cum_idx])
    peak_cumulative_date = dt[peak_cum_idx]
    final_cumulative = float(cumulative_bias_gl[-1])

    valid = obs > 0
    pbias = float(np.sum(bias_ml[valid]) / np.sum(obs[valid]) * 100.0)

    sqrt_sim = np.sqrt(np.maximum(sim, 0.0))
    sqrt_obs = np.sqrt(np.maximum(obs, 0.0))
    ss_res = np.sum((sqrt_sim - sqrt_obs) ** 2)
    ss_tot = np.sum((sqrt_obs - np.mean(sqrt_obs)) ** 2)
    nse_sqrt = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    r_so = np.corrcoef(sqrt_sim, sqrt_obs)[0, 1] if len(sqrt_sim) > 1 else np.nan
    alpha_so = float(np.std(sqrt_sim) / np.std(sqrt_obs)) if np.std(sqrt_obs) > 0 else np.nan
    beta_so = float(np.mean(sqrt_sim) / np.mean(sqrt_obs)) if np.mean(sqrt_obs) > 0 else np.nan
    kge_sqrt = float(1.0 - np.sqrt((r_so - 1) ** 2 + (alpha_so - 1) ** 2 + (beta_so - 1) ** 2))

    return {
        'dates': dt,
        'obs': obs,
        'sim': sim,
        'annual_bias_gl': annual_bias_gl,
        'cumulative_bias_gl': cumulative_bias_gl,
        'mean_annual_bias': mean_annual_bias,
        'pbias': pbias,
        'nse_sqrt': nse_sqrt,
        'kge_sqrt': kge_sqrt,
        'peak_annual_pos': peak_annual_pos,
        'peak_annual_pos_date': peak_annual_pos_date,
        'peak_annual_neg': peak_annual_neg,
        'peak_annual_neg_date': peak_annual_neg_date,
        'peak_cumulative': peak_cumulative,
        'peak_cumulative_date': peak_cumulative_date,
        'final_cumulative': final_cumulative,
    }


def _make_experiment_label(key: str) -> str:
    """Human-readable short label from an experiment key."""
    p = parse_experiment_key(key)
    model = p.get('model', '?')
    obj = p.get('objective', '?')
    trans = p.get('transformation', '')
    alg = p.get('algorithm', '?')
    label = f"{model} {obj}"
    if trans:
        label += f"({trans})"
    label += f" [{alg}]"
    return label


# %%
def _add_drought_shading_mpl(ax, dates, drought_periods):
    """Add semi-transparent drought bands to a Matplotlib axis."""
    xmin, xmax = dates[0], dates[-1]
    for start_s, end_s, label in drought_periods:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        if end < xmin or start > xmax:
            continue
        start = max(start, xmin)
        end = min(end, xmax)
        ax.axvspan(start, end, color=CLR_DROUGHT_BG, zorder=0)
        mid = start + (end - start) / 2
        ax.text(
            mid, 0.95, label,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='#AA4444', alpha=0.7,
        )


def _compute_gauge_ylims(
    all_bias_data: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    """Compute symmetric y-axis limits across all experiments in a gauge.

    Returns (ylim_annual, ylim_cumulative) — the max absolute value
    across all experiments, padded by 10 %.
    """
    max_annual = max(
        np.nanmax(np.abs(bd['annual_bias_gl'].dropna().values))
        for bd in all_bias_data.values()
    )
    max_cumul = max(
        np.max(np.abs(bd['cumulative_bias_gl']))
        for bd in all_bias_data.values()
    )
    pad = 1.10
    return max_annual * pad, max_cumul * pad


def plot_temporal_bias_mpl(
    bias_data: Dict[str, Any],
    title: str,
    drought_periods: List[Tuple[str, str, str]],
    save_path: Optional[Path] = None,
    ylim_monthly: Optional[float] = None,
    ylim_cumulative: Optional[float] = None,
) -> None:
    """Draw a 3-panel temporal bias figure (Matplotlib) and optionally save.

    When *ylim_monthly* / *ylim_cumulative* are provided, panels (b) and (c)
    use symmetric-log scale with those fixed limits so all experiments in a
    gauge share identical axes.
    """
    dates = bias_data['dates']
    obs = bias_data['obs']
    sim = bias_data['sim']
    monthly = bias_data['monthly_bias_gl']
    cumul = bias_data['cumulative_bias_gl']

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True, layout='constrained',
        gridspec_kw={'height_ratios': [1.2, 1, 1], 'hspace': 0.08},
    )
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # --- Panel (a): daily flow (log scale) ---------------------------------
    obs_plot = np.where(obs > 0, obs, np.nan)
    sim_plot = np.where(sim > 0, sim, np.nan)
    ax_a.plot(dates, obs_plot, color=CLR_OBS, lw=0.4, alpha=0.7,
              label='Observed flow (gauge)')
    ax_a.plot(dates, sim_plot, color=CLR_SIM, lw=0.4, alpha=0.7,
              label='pyrrm simulated')
    ax_a.set_yscale('log')
    ax_a.set_ylabel('Flow\n[log scale]\n(ML/day)', fontsize=9)
    ax_a.legend(loc='lower left', fontsize=7, framealpha=0.8)
    ax_a.text(0.01, 0.93, '(a)  Daily flow at gauge — observed vs simulated',
              transform=ax_a.transAxes, fontsize=9, fontweight='bold', va='top')
    _add_drought_shading_mpl(ax_a, dates, drought_periods)

    # --- Panel (b): annual inflow bias (bars) --------------------------------
    annual = bias_data['annual_bias_gl'].dropna()
    ab_dates = annual.index
    ab_vals = annual.values
    bar_colors = [CLR_EXCESS if v >= 0 else CLR_DEFICIT for v in ab_vals]
    # Width: 300 days so bars are clearly visible and spaced
    bar_width = pd.Timedelta('300D')
    ax_b.bar(ab_dates, ab_vals, width=bar_width, color=bar_colors, alpha=0.8,
             align='center', zorder=2)
    # Invisible proxy patches for the legend
    import matplotlib.patches as mpatches
    ax_b.legend(
        handles=[
            mpatches.Patch(color=CLR_EXCESS, alpha=0.8, label='Excess (sim > obs)'),
            mpatches.Patch(color=CLR_DEFICIT, alpha=0.8, label='Deficit (sim < obs)'),
        ],
        loc='lower left', fontsize=7, framealpha=0.8,
    )
    ax_b.axhline(0, color='black', lw=0.5)
    mean_b = bias_data['mean_annual_bias']
    ax_b.axhline(mean_b, color='black', lw=0.8, ls='--', alpha=0.5)
    if len(ab_dates) >= 2:
        ax_b.text(ab_dates[1], mean_b, f'  Mean: {mean_b:+.2f} GL/yr',
                  fontsize=7, va='bottom' if mean_b >= 0 else 'top', alpha=0.7)
    ax_b.set_yscale('symlog', linthresh=0.5)
    if ylim_monthly is not None:
        ax_b.set_ylim(-ylim_monthly, ylim_monthly)
    ax_b.set_ylabel('Annual inflow\nbias (GL/yr)', fontsize=9)
    ax_b.text(0.01, 0.93,
              '(b)  Annual inflow bias (simulated – observed)',
              transform=ax_b.transAxes, fontsize=9, fontweight='bold', va='top')
    _add_drought_shading_mpl(ax_b, dates, drought_periods)

    if abs(bias_data['peak_annual_pos']) > 0.05:
        ppd = bias_data['peak_annual_pos_date']
        ppv = bias_data['peak_annual_pos']
        ax_b.annotate(
            f'Peak: {ppv:+.1f} GL/yr\n({ppd.year})',
            xy=(ppd, ppv), fontsize=7, color='#AA3333',
            xytext=(0, 8), textcoords='offset points', ha='center',
        )

    # --- Panel (c): cumulative bias ----------------------------------------
    ax_c.fill_between(dates, cumul, 0,
                      where=cumul >= 0, interpolate=True,
                      color=CLR_EXCESS, alpha=0.7, label='Cumulative excess')
    ax_c.fill_between(dates, cumul, 0,
                      where=cumul < 0, interpolate=True,
                      color=CLR_DEFICIT, alpha=0.7, label='Cumulative deficit')
    ax_c.axhline(0, color='black', lw=0.5)
    ax_c.set_yscale('symlog', linthresh=1.0)
    if ylim_cumulative is not None:
        ax_c.set_ylim(-ylim_cumulative, ylim_cumulative)
    ax_c.set_ylabel(f'Cumulative bias\nsince {dates[0].year} (GL)', fontsize=9)
    ax_c.legend(loc='lower left', fontsize=7, framealpha=0.8)
    ax_c.text(0.01, 0.93,
              '(c)  Cumulative excess inflow attributable to model bias',
              transform=ax_c.transAxes, fontsize=9, fontweight='bold', va='top')
    _add_drought_shading_mpl(ax_c, dates, drought_periods)

    pcd = bias_data['peak_cumulative_date']
    pcv = bias_data['peak_cumulative']
    fcv = bias_data['final_cumulative']
    if abs(pcv) > 1.0:
        ax_c.annotate(
            f'Peak: {pcv:+.0f} GL\n({pcd.strftime("%b %Y")})',
            xy=(pcd, pcv), fontsize=7, color='#AA3333',
            xytext=(0, 8 if pcv >= 0 else -14), textcoords='offset points',
            ha='center',
        )
    ax_c.annotate(
        f'End: {fcv:+.0f} GL',
        xy=(dates[-1], fcv), fontsize=7, color='#333333',
        xytext=(-40, 8 if fcv >= 0 else -14), textcoords='offset points',
        ha='right',
    )

    ax_c.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_c.xaxis.set_minor_locator(mdates.YearLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.align_ylabels([ax_a, ax_b, ax_c])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=STATIC_DPI, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


# %%
def _add_drought_shading_plotly(fig, dates, drought_periods, n_rows):
    """Add semi-transparent drought vrects to all rows of a Plotly figure."""
    xmin, xmax = dates[0], dates[-1]
    for start_s, end_s, label in drought_periods:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        if end < xmin or start > xmax:
            continue
        start = max(start, xmin)
        end = min(end, xmax)
        for row in range(1, n_rows + 1):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor='rgba(255,180,180,0.18)', line_width=0,
                row=row, col=1,
            )
        mid = start + (end - start) / 2
        fig.add_annotation(
            x=mid, y=1.0, yref='y domain', text=label.replace('\n', '<br>'),
            showarrow=False, font=dict(size=9, color='#AA4444'),
            opacity=0.6, xanchor='center', yanchor='top', row=1, col=1,
        )


def _split_pos_neg(x, y):
    """Split a series at zero crossings into positive and negative arrays.

    Returns (y_pos, y_neg) where values on the wrong side are set to 0,
    suitable for separate ``fill='tozeroy'`` traces.
    """
    y_pos = np.where(y >= 0, y, 0.0)
    y_neg = np.where(y <= 0, y, 0.0)
    return y_pos, y_neg


def build_gauge_plotly(
    all_bias_data: Dict[str, Dict[str, Any]],
    gauge_title: str,
    drought_periods: List[Tuple[str, str, str]],
    default_visible: int = 3,
    ylim_monthly: Optional[float] = None,
    ylim_cumulative: Optional[float] = None,
) -> go.Figure:
    """Build one interactive Plotly figure for a gauge with all experiments.

    Each experiment gets 5 traces (1 simulated flow line + 2 monthly-bias
    fills + 2 cumulative-bias fills) sharing a ``legendgroup`` so toggling
    one legend entry shows/hides all panels.  Panels (b) and (c) use
    red/blue fill-to-zero matching the Matplotlib static figures.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.35, 0.30, 0.35],
        subplot_titles=[
            '(a) Daily flow — observed vs simulated',
            '(b) Annual inflow bias (GL/yr)',
            '(c) Cumulative bias (GL)',
        ],
    )

    first_data = next(iter(all_bias_data.values()))
    dates = first_data['dates']
    obs = first_data['obs']
    obs_plot = np.where(obs > 0, obs, np.nan)
    fig.add_trace(
        go.Scattergl(
            x=dates, y=obs_plot, mode='lines',
            line=dict(color=CLR_OBS, width=0.5),
            name='Observed', legendgroup='__observed__',
            showlegend=True,
        ),
        row=1, col=1,
    )

    keys_sorted = sorted(all_bias_data.keys())
    for idx, key in enumerate(keys_sorted):
        bd = all_bias_data[key]
        label = _make_experiment_label(key)
        color = _PLOTLY_PALETTE[idx % len(_PLOTLY_PALETTE)]
        visible = True if idx < default_visible else 'legendonly'

        # Panel (a) — simulated flow line (legend entry lives here)
        sim_plot = np.where(bd['sim'] > 0, bd['sim'], np.nan)
        fig.add_trace(
            go.Scattergl(
                x=bd['dates'], y=sim_plot, mode='lines',
                line=dict(color=color, width=0.6),
                name=label, legendgroup=key,
                visible=visible, showlegend=True,
            ),
            row=1, col=1,
        )

        # Panel (b) — annual bias: bar chart coloured by sign
        ab_series = bd['annual_bias_gl'].dropna()
        ab_vals = ab_series.values
        bar_clrs = [
            'rgba(232,153,141,0.75)' if v >= 0 else 'rgba(141,170,232,0.75)'
            for v in ab_vals
        ]
        fig.add_trace(
            go.Bar(
                x=ab_series.index, y=ab_vals,
                marker_color=bar_clrs,
                name=label, legendgroup=key,
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )

        # Panel (c) — cumulative bias: excess (red) and deficit (blue) fills
        cum_vals = bd['cumulative_bias_gl']
        cum_pos, cum_neg = _split_pos_neg(bd['dates'], cum_vals)
        fig.add_trace(
            go.Scattergl(
                x=bd['dates'], y=cum_pos, mode='lines',
                line=dict(width=0, color=CLR_EXCESS),
                fill='tozeroy', fillcolor='rgba(232,153,141,0.45)',
                name=label, legendgroup=key,
                visible=visible, showlegend=False,
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=bd['dates'], y=cum_neg, mode='lines',
                line=dict(width=0, color=CLR_DEFICIT),
                fill='tozeroy', fillcolor='rgba(141,170,232,0.45)',
                name=label, legendgroup=key,
                visible=visible, showlegend=False,
            ),
            row=3, col=1,
        )

    fig.add_hline(y=0, line_dash='solid', line_color='black',
                  line_width=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash='solid', line_color='black',
                  line_width=0.5, row=3, col=1)

    _add_drought_shading_plotly(fig, dates, drought_periods, n_rows=3)

    fig.update_yaxes(type='log', title_text='Flow (ML/day)', row=1, col=1)

    if ylim_monthly is not None:
        fig.update_yaxes(title_text='Annual bias (GL/yr)',
                         range=[-ylim_monthly, ylim_monthly], row=2, col=1)
    else:
        fig.update_yaxes(title_text='Annual bias (GL/yr)', row=2, col=1)

    start_yr = dates[0].year
    if ylim_cumulative is not None:
        fig.update_yaxes(title_text=f'Cumulative bias since {start_yr} (GL)',
                         range=[-ylim_cumulative, ylim_cumulative],
                         row=3, col=1)
    else:
        fig.update_yaxes(title_text=f'Cumulative bias since {start_yr} (GL)',
                         row=3, col=1)

    fig.update_layout(
        title=dict(text=gauge_title, font=dict(size=14)),
        height=1000,
        barmode='overlay',
        legend=dict(
            font=dict(size=9),
            groupclick='togglegroup',
            tracegroupgap=2,
        ),
        hovermode='x unified',
        margin=dict(l=70, r=20, t=80, b=40),
    )

    return fig


# %% [markdown]
# ---
# ## Per-Gauge Analysis
#
# For each gauge we:
# 1. Prepare bias data for every experiment
# 2. Save a static Matplotlib PNG per experiment
# 3. Display one interactive Plotly figure with all experiments overlaid

# %%
for gauge_id, br in batch_results.items():
    print(f"\n{'='*80}")
    print(f"  Gauge {gauge_id}  —  {len(br.results)} experiments")
    print(f"{'='*80}\n")

    rep0 = next(iter(br.results.values()))
    info = getattr(rep0, 'catchment_info', {}) or {}
    area_km2 = info.get('area_km2')
    area_str = f"{area_km2:.0f} km²" if area_km2 else ''

    all_bias_data: Dict[str, Dict[str, Any]] = OrderedDict()

    # --- Phase 1: prepare bias data for all experiments --------------------
    for key in sorted(br.results.keys()):
        report = br.results[key]
        bd = prepare_bias_data(report)
        all_bias_data[key] = bd

    ylim_monthly, ylim_cumulative = _compute_gauge_ylims(all_bias_data)
    print(f"  Y-axis limits — annual: ±{ylim_monthly:.2f} GL/yr, "
          f"cumulative: ±{ylim_cumulative:.0f} GL")

    # --- Phase 2: Matplotlib static PNGs -----------------------------------
    if SAVE_STATIC:
        fig_dir = RESULTS_DIR / gauge_id / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        for key, bd in all_bias_data.items():
            label = _make_experiment_label(key)
            title = (f"HWC_{gauge_id} ({area_str}) — {label}  |  "
                     f"PBIAS = {bd['pbias']:+.1f}%   "
                     f"NSE√Q = {bd['nse_sqrt']:.3f}   "
                     f"KGE√Q = {bd['kge_sqrt']:.3f}")
            save_path = fig_dir / f"{key}_temporal_bias.png"
            plot_temporal_bias_mpl(
                bd, title, DROUGHT_PERIODS, save_path=save_path,
                ylim_monthly=ylim_monthly, ylim_cumulative=ylim_cumulative,
            )
        print(f"  Saved {len(all_bias_data)} PNGs to {fig_dir}")

    # --- Phase 3: single interactive Plotly per gauge ----------------------
    gauge_title = f"HWC_{gauge_id} ({area_str}) — Temporal Bias (all experiments)"
    plotly_fig = build_gauge_plotly(
        all_bias_data, gauge_title, DROUGHT_PERIODS,
        ylim_monthly=ylim_monthly, ylim_cumulative=ylim_cumulative,
    )
    plotly_fig.show()

print("\nDone — all gauges processed.")
