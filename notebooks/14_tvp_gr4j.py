# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pyrrm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time-Varying Parameter (TVP) Calibration — GR4J with Gaussian Random Walk
#
# ## Purpose
#
# This notebook demonstrates **time-varying parameter (TVP)** calibration
# of GR4J on gauge 410734 using a Gaussian Random Walk (GRW) prior on
# the production store capacity **X1**.  We compare TVP-GR4J against
# fixed-parameter calibrations from previous notebooks across a
# comprehensive suite of 27 diagnostic metrics.
#
# ### Why Time-Varying Parameters?
#
# Standard calibration assumes that model parameters are **constant**
# over the entire record.  In reality, catchment properties change:
#
# - **Vegetation dynamics**: seasonal leaf-out/senescence, bushfire
# - **Land-use change**: urbanisation, clearing, revegetation
# - **Soil structure**: compaction, cracking, wetting-drying cycles
# - **Climate non-stationarity**: shifting rainfall patterns
#
# Allowing X1 (production store capacity) to vary captures temporal
# changes in the catchment's ability to absorb rainfall before
# generating runoff.
#
# ### Gaussian Random Walk Prior
#
# The GRW models X1(t) as a random walk:
#
# $$\alpha \sim \mathrm{Uniform}(\text{lower}, \text{upper})$$
# $$\sigma_\delta \sim \mathrm{HalfNormal}(\text{scale})$$
# $$\delta_t \sim \mathcal{N}(0, \sigma_\delta^2)$$
# $$X1(t) = \alpha + \sum_{t' \leq t} \delta_{t'}$$
#
# The `resolution` parameter controls how many timesteps share each
# $\delta$ increment (e.g. `resolution=5` means one increment per 5 days).
#
# ## What You'll Learn
#
# - How to configure a **Gaussian Random Walk prior** on GR4J's X1 parameter
# - How to run **TVP calibration** via NUTS with 4 likelihood transforms
# - How to visualise **X1(t) trajectories** (full-record and yearly decomposition)
# - How to compare TVP-GR4J against **all prior fixed-parameter methods**
#   (SCE-UA, PyDREAM, NUTS-fixed) across 27 diagnostic metrics
# - How to interpret **regime-based analysis** showing where TVP helps most
#
# ## Prerequisites
#
# - Notebooks 02, 06, 07, 13 (for pre-computed calibration reports)
# - `jax`, `numpyro`, `arviz` installed
#
# ## Estimated Time
#
# - ~10-15 minutes for 4 TVP calibrations (500 warmup + 1000 samples × 2 chains)
# - ~5 minutes for comparison analysis and visualisation
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|--------|---------------|
# | 1 | Setup and helpers | Imports, diagnostic metrics, simulation and plotting helpers. |
# | 2 | Load gauge 410734 data | Rainfall, PET, observed flow; prepare calibration arrays. |
# | 3 | TVP-GR4J calibration | GRW on X1 with 4 likelihood transforms (Q, √Q, log Q, 1/Q). |
# | 4 | TVP transform comparison | Side-by-side metrics, hydrographs, and X1(t) trajectories. |
# | 5 | Load prior fixed-parameter results | Reload SCE-UA, PyDREAM, and NUTS-fixed reports from NB02/06/07/13. |
# | 6 | Grand comparison | TVP vs fixed across 27 metrics; regime analysis; heatmap. |
# | 7 | Summary | Key findings, questions addressed, future work. |
#
# ## Key Insight
#
# > A simple 4-parameter model with time-varying X1 can capture catchment
# > non-stationarity that fixed-parameter calibrations miss — the question is
# > whether the improvement justifies the additional computational cost and
# > the risk of overfitting.

# %% [markdown]
# ---
# ## Step 1: Setup and Helpers

# %%
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
from pathlib import Path
from collections import OrderedDict

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpyro
numpyro.set_host_device_count(jax.local_device_count() or 2)

import arviz as az

warnings.filterwarnings("ignore", message=".*where.*used without.*out.*")

from pyrrm.models.gr4j import GR4J
from pyrrm.models.gr4j_jax import gr4j_run_jax
from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration.runner import CalibrationRunner
from pyrrm.calibration.report import CalibrationReport
from pyrrm.calibration.tvp_priors import GaussianRandomWalk
from pyrrm.analysis.mcmc_diagnostics import (
    check_convergence,
    posterior_summary,
)
from pyrrm.visualization.mcmc_plots import (
    plot_mcmc_traces,
    plot_posterior_pairs,
    plot_hydrograph_with_uncertainty,
)

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")

WARMUP = 365

# Report directories from previous notebooks
REPORTS_NB02 = Path('../test_data/reports')             # Notebook 02 SCE-UA
REPORTS_PYDREAM = Path('../test_data/reports/pydream')  # Notebook 06 PyDREAM
REPORTS_MODELS = Path('../test_data/reports/models')    # Notebook 07 Model comparison
REPORTS_NUTS = Path('../test_data/reports/nuts')        # Notebook 13 NUTS fixed

TVP_REPORTS_DIR = Path('../test_data/reports/tvp')
TVP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOAD_FROM_PICKLE = False

CATCHMENT_INFO_410734 = {
    'name': 'Queanbeyan River',
    'gauge_id': '410734',
    'area_km2': 516.62667,
}

# %% [markdown]
# ### Helper Functions
#
# Reusable analysis and plotting helpers.  The `compute_metrics` function
# produces the same 27-metric diagnostic suite used in Notebook 13.

# %%
# =============================================================================
# DIAGNOSTIC HELPERS
# =============================================================================

def compute_metrics(sim, obs):
    """Compute the full suite of 27 diagnostic metrics."""
    sim = np.asarray(sim).flatten()
    obs = np.asarray(obs).flatten()
    mask = ~(np.isnan(sim) | np.isnan(obs) | np.isinf(sim) | np.isinf(obs))
    s, o = sim[mask], obs[mask]
    if len(s) == 0:
        return {}

    m = OrderedDict()
    eps_flow = 0.01

    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    m['NSE'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pos = o > 0
    if pos.sum() > 0:
        lo, ls = np.log(o[pos] + 1), np.log(np.maximum(s[pos], 0) + 1)
        m['LogNSE'] = 1 - np.sum((lo - ls)**2) / np.sum((lo - lo.mean())**2)
    else:
        m['LogNSE'] = np.nan

    sqo, sqs = np.sqrt(np.maximum(o, 0)), np.sqrt(np.maximum(s, 0))
    ss_r_sq = np.sum((sqo - sqs)**2)
    ss_t_sq = np.sum((sqo - sqo.mean())**2)
    m['SqrtNSE'] = 1 - ss_r_sq / ss_t_sq if ss_t_sq > 0 else np.nan

    inv_mask = o > eps_flow
    if inv_mask.sum() > 0:
        io, is_ = 1.0 / o[inv_mask], 1.0 / np.maximum(s[inv_mask], eps_flow)
        m['InvNSE'] = 1 - np.sum((io - is_)**2) / np.sum((io - io.mean())**2)
    else:
        m['InvNSE'] = np.nan

    def _kge(a, b):
        if len(a) < 2:
            return np.nan, np.nan, np.nan, np.nan
        r = np.corrcoef(a, b)[0, 1]
        alpha = np.std(b) / np.std(a) if np.std(a) > 0 else np.nan
        beta = np.mean(b) / np.mean(a) if np.mean(a) != 0 else np.nan
        kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
        return kge, r, alpha, beta

    kge, r, alpha, beta = _kge(o, s)
    m['KGE']       = kge
    m['KGE_r']     = r
    m['KGE_alpha'] = alpha
    m['KGE_beta']  = beta

    if pos.sum() > 0:
        kge_l, r_l, a_l, b_l = _kge(np.log(o[pos]+1), np.log(np.maximum(s[pos],0)+1))
    else:
        kge_l = r_l = a_l = b_l = np.nan
    m['KGE_log']       = kge_l
    m['KGE_log_r']     = r_l
    m['KGE_log_alpha'] = a_l
    m['KGE_log_beta']  = b_l

    kge_s, r_s, a_s, b_s = _kge(sqo, sqs)
    m['KGE_sqrt']       = kge_s
    m['KGE_sqrt_r']     = r_s
    m['KGE_sqrt_alpha'] = a_s
    m['KGE_sqrt_beta']  = b_s

    if inv_mask.sum() > 0:
        kge_i, r_i, a_i, b_i = _kge(1.0/o[inv_mask], 1.0/np.maximum(s[inv_mask], eps_flow))
    else:
        kge_i = r_i = a_i = b_i = np.nan
    m['KGE_inv']       = kge_i
    m['KGE_inv_r']     = r_i
    m['KGE_inv_alpha'] = a_i
    m['KGE_inv_beta']  = b_i

    m['RMSE']  = np.sqrt(np.mean((s - o)**2))
    m['MAE']   = np.mean(np.abs(s - o))
    m['PBIAS'] = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan

    obs_sorted = np.sort(o)[::-1]
    sim_sorted = np.sort(s)[::-1]
    n = len(obs_sorted)

    h = max(int(0.02 * n), 1)
    sum_obs_h = np.sum(obs_sorted[:h])
    m['FHV'] = 100 * (np.sum(sim_sorted[:h]) - sum_obs_h) / sum_obs_h if sum_obs_h > 0 else np.nan

    i20 = int(0.20 * n)
    i70 = min(int(0.70 * n), n)
    sum_obs_m = np.sum(obs_sorted[i20:i70])
    m['FMV'] = 100 * (np.sum(sim_sorted[i20:i70]) - sum_obs_m) / sum_obs_m if sum_obs_m > 0 else np.nan

    i70_start = int(0.70 * n)
    obs_low = obs_sorted[i70_start:]
    sim_low = sim_sorted[i70_start:]
    low_pos = obs_low > eps_flow
    if low_pos.sum() > 0:
        log_obs_low = np.log(obs_low[low_pos])
        log_sim_low = np.log(np.maximum(sim_low[low_pos], eps_flow))
        sum_log_obs = np.sum(log_obs_low - np.min(log_obs_low))
        sum_log_sim = np.sum(log_sim_low - np.min(log_obs_low))
        m['FLV'] = 100 * (sum_log_sim - sum_log_obs) / sum_log_obs if sum_log_obs > 0 else np.nan
    else:
        m['FLV'] = np.nan

    return m


METRIC_GROUPS = OrderedDict([
    ("NSE variants", ["NSE", "LogNSE", "SqrtNSE", "InvNSE"]),
    ("KGE(Q)", ["KGE", "KGE_r", "KGE_alpha", "KGE_beta"]),
    ("KGE(log Q)", ["KGE_log", "KGE_log_r", "KGE_log_alpha", "KGE_log_beta"]),
    ("KGE(√Q)", ["KGE_sqrt", "KGE_sqrt_r", "KGE_sqrt_alpha", "KGE_sqrt_beta"]),
    ("KGE(1/Q)", ["KGE_inv", "KGE_inv_r", "KGE_inv_alpha", "KGE_inv_beta"]),
    ("Error metrics", ["RMSE", "MAE", "PBIAS"]),
    ("FDC volume bias", ["FHV", "FMV", "FLV"]),
])


def print_metrics_table(metrics, label=""):
    """Print a grouped metrics table."""
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTIC METRICS{f'  —  {label}' if label else ''}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Value':>12}")
    for group_name, keys in METRIC_GROUPS.items():
        print(f"  {'-' * 40}")
        print(f"  {group_name}")
        for k in keys:
            v = metrics.get(k, np.nan)
            if np.isnan(v):
                print(f"    {k:<23} {'N/A':>12}")
            else:
                print(f"    {k:<23} {v:>12.4f}")
    print(f"{'=' * 60}")


def print_multi_method_comparison(metrics_dict, label=""):
    """Print a grouped metrics table comparing multiple experiments/methods."""
    methods = list(metrics_dict.keys())
    col_w = max(12, max(len(str(m)) + 2 for m in methods))
    width = 22 + col_w * len(methods)

    header = f"  {'Metric':<20}" + "".join(f"{str(m):>{col_w}}" for m in methods)
    print(f"\n{'=' * width}")
    print(f"  COMPARISON{f'  —  {label}' if label else ''}")
    print(f"{'=' * width}")
    print(header)

    for group_name, keys in METRIC_GROUPS.items():
        print(f"  {'-' * (width - 2)}")
        print(f"  {group_name}")
        for mn in keys:
            vals = [metrics_dict[m].get(mn, float('nan')) for m in methods]
            fmts = []
            for v in vals:
                fmts.append(f"{v:{col_w}.4f}" if not np.isnan(v) else f"{'N/A':>{col_w}}")
            print(f"    {mn:<18}" + "".join(fmts))
    print(f"{'=' * width}")


# =============================================================================
# SIMULATION HELPERS
# =============================================================================

def simulate_jax_from_idata(jax_model_fn, idata, precip, pet, warmup,
                            tvp_config=None):
    """Run JAX model with posterior-median params, return sim after warmup.

    When *tvp_config* is provided, TVP parameters are reconstructed as
    full time-series arrays (posterior-median trajectory) from the
    deterministic group, while static parameters remain scalars.
    """
    tvp_names = set(tvp_config.keys()) if tvp_config else set()
    skip = {"sigma", "phi"}
    median_params = {}

    if hasattr(idata, "deterministic") and len(idata.deterministic.data_vars) > 0:
        for v in idata.deterministic.data_vars:
            if v in skip:
                continue
            vals = idata.deterministic[v].values
            if v in tvp_names:
                median_params[v] = jnp.array(np.median(vals, axis=(0, 1)))
            elif vals.ndim <= 2:
                median_params[v] = float(np.median(vals))

    for v in idata.posterior.data_vars:
        if v in skip or v in median_params:
            continue
        if v.endswith("_unit") or v.endswith("_intercept") or v.endswith("_sigma_delta") or v.endswith("_delta"):
            continue
        vals = idata.posterior[v].values
        if vals.ndim <= 2:
            median_params[v] = float(np.median(vals))

    params_for_sim = {}
    for k, v in median_params.items():
        if isinstance(v, (float, int)):
            params_for_sim[k] = jnp.float64(v)
        else:
            params_for_sim[k] = v

    sim = np.array(
        jax_model_fn(params_for_sim, jnp.array(precip), jnp.array(pet))["simulated_flow"]
    )
    return sim[warmup:], median_params


def simulate_numpy_model(model_class, params, inputs_df, warmup,
                         catchment_area_km2=None):
    """Run a NumPy model and return sim flow after warmup (mm/d)."""
    model = model_class() if catchment_area_km2 is None else model_class(
        catchment_area_km2=catchment_area_km2
    )
    model.set_parameters(params)
    model.reset()
    results = model.run(inputs_df)
    flow = results['flow'].values
    return flow[warmup:]


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def plot_hydrographs(sim, obs, title="", dates=None):
    """Observed vs simulated — linear and log scale."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = dates if dates is not None else np.arange(len(obs))

    for i, (yscale, label) in enumerate([('linear', 'Linear'), ('log', 'Log')]):
        axes[i].plot(x, obs, color='#1f77b4', lw=0.7, alpha=0.8, label='Observed')
        axes[i].plot(x, sim, color='#d62728', lw=0.7, alpha=0.8, label='Simulated')
        if yscale == 'log':
            axes[i].set_yscale('log')
        axes[i].set_ylabel(f'Flow (mm/d){" [log]" if yscale == "log" else ""}')
        axes[i].set_title(f'{title} — {label} Scale', fontweight='bold')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()
    return fig


def plot_fdc(sim, obs, title=""):
    """Flow duration curve — observed vs simulated (log scale)."""
    n = len(obs)
    exc = np.arange(1, n + 1) / (n + 1) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(exc, np.sort(obs)[::-1], color='#1f77b4', lw=1.5, label='Observed')
    ax.semilogy(exc, np.sort(sim)[::-1], color='#d62728', lw=1.5, label='Simulated')
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Flow (mm/d) [log]')
    ax.set_title(f'Flow Duration Curve — {title}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    for pct, lbl in [(5, 'High (Q5)'), (50, 'Median'), (95, 'Low (Q95)')]:
        ax.axvline(pct, color='grey', ls='--', alpha=0.4)
        ax.text(pct + 0.5, ax.get_ylim()[1] * 0.5, lbl, fontsize=8, color='grey')
    plt.tight_layout()
    plt.show()
    return fig


def plot_tvp_trajectory(idata, tvp_name, dates, ci=0.90):
    """Plot full-record TVP trajectory with credible interval.

    Extracts the deterministic TVP array from all posterior samples,
    computes per-timestep percentiles, and plots the median trajectory
    with a shaded credible band.
    """
    tvp_samples = idata.deterministic[tvp_name].values  # (chains, draws, T)
    tvp_flat = tvp_samples.reshape(-1, tvp_samples.shape[-1])  # (total_draws, T)
    median = np.median(tvp_flat, axis=0)
    lo = np.percentile(tvp_flat, (1 - ci) / 2 * 100, axis=0)
    hi = np.percentile(tvp_flat, (1 + ci) / 2 * 100, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, lo, hi, alpha=0.3, color='steelblue',
                     label=f'{int(ci*100)}% CI')
    ax.plot(dates, median, color='steelblue', lw=1.2, label='Posterior median')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{tvp_name} (mm)')
    ax.set_title(f'Time-Varying {tvp_name} — Full Trajectory', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_tvp_yearly_decomposition(idata, tvp_name, dates, ci=0.90,
                                  ncols=4):
    """Santos et al. Figure 8-style yearly decomposition of a TVP.

    Left panel: full-record trajectory with posterior median and CI.
    Right grid: per-calendar-year subplots showing Jan-Dec segments
    of the TVP relative to the January 1st value.
    """
    tvp_samples = idata.deterministic[tvp_name].values  # (chains, draws, T)
    tvp_flat = tvp_samples.reshape(-1, tvp_samples.shape[-1])  # (total_draws, T)
    median_full = np.median(tvp_flat, axis=0)
    lo_full = np.percentile(tvp_flat, (1 - ci) / 2 * 100, axis=0)
    hi_full = np.percentile(tvp_flat, (1 + ci) / 2 * 100, axis=0)

    dates_arr = pd.DatetimeIndex(dates)
    years = sorted(set(dates_arr.year))

    nrows_sub = math.ceil(len(years) / ncols)
    fig = plt.figure(figsize=(20, 4 + 2.5 * nrows_sub))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 0.6], wspace=0.25)

    # --- Left panel: full trajectory ---
    ax_full = fig.add_subplot(gs[0])
    ax_full.fill_between(dates_arr, lo_full, hi_full, alpha=0.25,
                          color='steelblue')
    for year in years:
        yr_mask = dates_arr.year == year
        ax_full.plot(dates_arr[yr_mask], median_full[yr_mask],
                      color='grey', alpha=0.3, lw=0.5)
    ax_full.plot(dates_arr, median_full, color='steelblue', lw=1.0,
                  label='Posterior median')
    ax_full.set_xlabel('Date')
    ax_full.set_ylabel(f'{tvp_name} (mm)')
    ax_full.set_title(f'{tvp_name}(t) — All Years', fontweight='bold')
    ax_full.legend(loc='upper right', fontsize=8)
    ax_full.grid(True, alpha=0.3)

    # --- Right grid: per-year subplots ---
    gs_right = gridspec.GridSpecFromSubplotSpec(
        nrows_sub, ncols, subplot_spec=gs[1], hspace=0.45, wspace=0.3
    )

    for idx, year in enumerate(years):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs_right[row, col])

        yr_mask = dates_arr.year == year
        yr_dates = dates_arr[yr_mask]
        yr_median = median_full[yr_mask]
        yr_lo = lo_full[yr_mask]
        yr_hi = hi_full[yr_mask]

        jan1_val = yr_median[0]
        doy = (yr_dates - pd.Timestamp(f'{year}-01-01')).days

        ax.fill_between(doy, yr_lo - jan1_val, yr_hi - jan1_val,
                          alpha=0.25, color='steelblue')
        ax.plot(doy, yr_median - jan1_val, color='steelblue', lw=0.8)
        ax.axhline(0, color='grey', ls='--', lw=0.5, alpha=0.5)
        ax.set_title(str(year), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f'Δ{tvp_name}', fontsize=8)
        if row == nrows_sub - 1:
            ax.set_xlabel('Day of Year', fontsize=8)

    fig.suptitle(
        f'{tvp_name} Yearly Decomposition — Relative to Jan 1st ({int(ci*100)}% CI)',
        fontweight='bold', fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.show()
    return fig


def _save_report(runner, result, report_name, catchment_info=None):
    """Save a CalibrationReport to TVP_REPORTS_DIR."""
    report = runner.create_report(result, catchment_info=catchment_info)
    report.save(str((TVP_REPORTS_DIR / report_name).with_suffix('')))
    print(f"  Saved: {report_name}.pkl")
    return report


def _load_result(report_name, reports_dir=TVP_REPORTS_DIR):
    """Load a CalibrationResult from a saved CalibrationReport."""
    filepath = reports_dir / f'{report_name}.pkl'
    report = CalibrationReport.load(str(filepath))
    print(f"  Loaded: {report_name}.pkl  "
          f"(Best obj: {report.result.best_objective:.4f})")
    return report.result


# %% [markdown]
# ---
# ## Step 2: Load Gauge 410734 Data

# %%
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

rainfall_df = pd.read_csv(
    DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
    parse_dates=['Date'], index_col='Date',
)
rainfall_df.columns = ['rainfall']

pet_df = pd.read_csv(
    DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
    parse_dates=['Date'], index_col='Date',
)
pet_df.columns = ['pet']

flow_df = pd.read_csv(
    DATA_DIR / '410734_output_SDmodel.csv',
    parse_dates=['Date'], index_col='Date',
)
obs_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[obs_col]].copy()
observed_df.columns = ['observed_flow']
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')
data['observed_mm'] = data['observed_flow'] / CATCHMENT_AREA_KM2

print("GAUGE 410734 — QUEANBEYAN RIVER")
print("=" * 55)
print(f"  Catchment area  : {CATCHMENT_AREA_KM2:.2f} km²")
print(f"  Records         : {len(data):,} days")
print(f"  Period          : {data.index.min().date()} → {data.index.max().date()}")
print(f"  Mean rainfall   : {data['rainfall'].mean():.2f} mm/d")
print(f"  Mean PET        : {data['pet'].mean():.2f} mm/d")
print(f"  Mean flow (mm)  : {data['observed_mm'].mean():.3f} mm/d")

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].bar(data.index, data['rainfall'], color='steelblue', width=1.0)
axes[0].invert_yaxis()
axes[0].set_ylabel('Rainfall (mm/d)')
axes[0].set_title('Input Data — Gauge 410734, Queanbeyan Catchment', fontweight='bold')
axes[1].plot(data.index, data['pet'], color='orange', lw=0.6)
axes[1].set_ylabel('PET (mm/d)')
axes[2].plot(data.index, data['observed_mm'], color='darkblue', lw=0.5)
axes[2].set_ylabel('Flow (mm/d)')
axes[2].set_title('Observed Flow — Linear Scale', fontsize=10)
axes[3].plot(data.index, data['observed_mm'], color='darkblue', lw=0.5)
axes[3].set_yscale('log')
axes[3].set_ylabel('Flow (mm/d) [log]')
axes[3].set_title('Observed Flow — Log Scale', fontsize=10)
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
cal_inputs_410734 = data[['rainfall', 'pet']].copy()
cal_inputs_410734.columns = ['precipitation', 'pet']
cal_obs_mm = data['observed_mm'].values
precip_410734 = cal_inputs_410734['precipitation'].values
pet_410734 = cal_inputs_410734['pet'].values
dates_410734 = data.index

print(f"Calibration period : {len(cal_inputs_410734):,} days")
print(f"Warmup             : {WARMUP} days")
print(f"Effective period   : {len(cal_inputs_410734) - WARMUP:,} days")

# %% [markdown]
# ---
# ## Step 3: TVP-GR4J Calibration
#
# We run four experiments, one per likelihood transform, with X1 as a
# time-varying parameter using a Gaussian Random Walk prior.
# Parameters X2, X3, X4 remain static.
#
# | Experiment | Transform | Description |
# |-----------|-----------|-------------|
# | TVP-GR4J Q | `none` | Raw flow residuals |
# | TVP-GR4J √Q | `sqrt` | Balanced high/low |
# | TVP-GR4J log Q | `log` | Low-flow emphasis |
# | TVP-GR4J 1/Q | `inverse` | Strong low-flow focus |

# %%
tvp_config = {
    "X1": GaussianRandomWalk(lower=1.0, upper=1500.0, resolution=5),
}

NUTS_TVP_COMMON = dict(
    num_warmup=500, num_samples=1000, num_chains=2,
    seed=42, progress_bar=True, verbose=True,
)

GR4J_PARAM_NAMES = ["X1", "X2", "X3", "X4"]

TRANSFORMS = OrderedDict([
    ("none",    "410734_gr4j_gaussian_nuts_tvpx1"),
    ("sqrt",    "410734_gr4j_gaussian_nuts_tvpx1_sqrt"),
    ("log",     "410734_gr4j_gaussian_nuts_tvpx1_log"),
    ("inverse", "410734_gr4j_gaussian_nuts_tvpx1_inverse"),
])

TRANSFORM_DISPLAY = OrderedDict([
    ("none",    "TVP Q (none)"),
    ("sqrt",    "TVP √Q (sqrt)"),
    ("log",     "TVP log Q"),
    ("inverse", "TVP 1/Q (inverse)"),
])

tvp_results = {}
tvp_idatas = {}
tvp_sims = {}
tvp_params = {}
tvp_metrics = {}

# %% [markdown]
# ### Calibration Loop — All 4 Transforms

# %%
for transform, pkl_name in TRANSFORMS.items():
    display_name = TRANSFORM_DISPLAY[transform]
    print(f"\n{'=' * 60}")
    print(f"  {display_name}")
    print(f"{'=' * 60}")

    runner = CalibrationRunner(
        model=GR4J(), inputs=cal_inputs_410734,
        observed=cal_obs_mm, warmup_period=WARMUP,
    )

    if LOAD_FROM_PICKLE and (TVP_REPORTS_DIR / f'{pkl_name}.pkl').exists():
        tvp_results[transform] = _load_result(pkl_name)
    else:
        tvp_results[transform] = runner.run_nuts(
            transform=transform,
            tvp_config=tvp_config,
            **NUTS_TVP_COMMON,
        )
        _save_report(runner, tvp_results[transform], pkl_name,
                     catchment_info=CATCHMENT_INFO_410734)

    idata = tvp_results[transform]._raw_result["inference_data"]
    tvp_idatas[transform] = idata

    sim, params = simulate_jax_from_idata(
        gr4j_run_jax, idata, precip_410734, pet_410734, WARMUP,
        tvp_config=tvp_config,
    )
    tvp_sims[transform] = sim
    tvp_params[transform] = params
    tvp_metrics[transform] = compute_metrics(sim, cal_obs_mm[WARMUP:])

    # Convergence check — on scalar hyperparameters
    scalar_diag_names = ["X2", "X3", "X4", "X1_intercept", "X1_sigma_delta", "sigma"]
    available_vars = list(idata.posterior.data_vars) + (
        list(idata.deterministic.data_vars) if hasattr(idata, "deterministic") else []
    )
    diag_names = [n for n in scalar_diag_names if n in available_vars]
    diag = check_convergence(idata, var_names=diag_names)
    print(f"\nConverged: {diag['converged']}  |  Divergences: {diag['divergences']}")
    print(posterior_summary(idata, var_names=diag_names))

    print_metrics_table(tvp_metrics[transform], display_name)

    plot_hydrographs(sim, cal_obs_mm[WARMUP:],
                     f'{display_name} — GR4J, Gauge 410734',
                     dates_410734[WARMUP:])
    plot_fdc(sim, cal_obs_mm[WARMUP:], f'{display_name} — GR4J, Gauge 410734')

    # TVP trajectory
    plot_tvp_trajectory(idata, "X1", dates_410734, ci=0.90)

    # Santos et al. Figure 8 — yearly decomposition
    plot_tvp_yearly_decomposition(idata, "X1", dates_410734, ci=0.90)

    # MCMC traces for hyperparameters
    trace_vars = [n for n in ["X2", "X3", "X4", "X1_intercept", "X1_sigma_delta"]
                  if n in available_vars]
    if trace_vars:
        plot_mcmc_traces(idata, var_names=trace_vars)
        plt.show()

# %% [markdown]
# ---
# ## Step 4: TVP Transform Comparison
#
# Side-by-side comparison of the 4 TVP transforms across all 27 diagnostic
# metrics, runtimes, hyperparameters, and X1(t) trajectories.

# %%
obs_w = cal_obs_mm[WARMUP:]
dates_w = dates_410734[WARMUP:]

print_multi_method_comparison(
    {TRANSFORM_DISPLAY[t]: tvp_metrics[t] for t in TRANSFORMS},
    label="TVP-GR4J — Gauge 410734 — Transform Comparison",
)

# Runtime comparison
print(f"\n{'=' * 60}")
print(f"  RUNTIME COMPARISON — TVP-GR4J, Gauge 410734")
print(f"{'=' * 60}")
for t in TRANSFORMS:
    rt = tvp_results[t].runtime_seconds
    print(f"  {TRANSFORM_DISPLAY[t]:<25} {rt:10.1f} s")
print(f"{'=' * 60}")

# Hyperparameter comparison
print(f"\n{'=' * 75}")
print(f"  TVP HYPERPARAMETERS — by Transform")
print(f"{'=' * 75}")
hp_keys = ["X1_intercept", "X1_sigma_delta", "X2", "X3", "X4"]
header = f"  {'Parameter':<18}" + "".join(
    f"{TRANSFORM_DISPLAY[t]:>15}" for t in TRANSFORMS
)
print(header)
print(f"  {'-' * 70}")
for hp in hp_keys:
    vals = []
    for t in TRANSFORMS:
        v = tvp_params[t].get(hp, np.nan)
        if isinstance(v, (float, int)):
            vals.append(f"{v:15.3f}")
        else:
            vals.append(f"{'[array]':>15}")
    print(f"  {hp:<18}" + "".join(vals))
print(f"{'=' * 75}")

# %%
TRANSFORM_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for i, yscale in enumerate(['linear', 'log']):
    axes[i].plot(dates_w, obs_w, color='grey', lw=0.5, alpha=0.7, label='Observed')
    for (t, _), col, lbl in zip(TRANSFORMS.items(), TRANSFORM_COLOURS,
                                 [TRANSFORM_DISPLAY[t] for t in TRANSFORMS]):
        axes[i].plot(dates_w, tvp_sims[t], color=col, lw=0.5, alpha=0.7, label=lbl)
    if yscale == 'log':
        axes[i].set_yscale('log')
    scale_tag = 'Log' if yscale == 'log' else 'Linear'
    axes[i].set_ylabel(f'Flow (mm/d){" [log]" if yscale == "log" else ""}')
    axes[i].set_title(f'TVP-GR4J Transform Comparison — {scale_tag} Scale', fontweight='bold')
    axes[i].legend(loc='upper right', fontsize=9)
    axes[i].grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# X1(t) comparison across transforms
fig, ax = plt.subplots(figsize=(14, 5))
for (t, _), col, lbl in zip(TRANSFORMS.items(), TRANSFORM_COLOURS,
                              [TRANSFORM_DISPLAY[t] for t in TRANSFORMS]):
    idata = tvp_idatas[t]
    tvp_vals = idata.deterministic["X1"].values
    median = np.median(tvp_vals.reshape(-1, tvp_vals.shape[-1]), axis=0)
    ax.plot(dates_410734, median, color=col, lw=1.0, alpha=0.8, label=lbl)
ax.set_xlabel('Date')
ax.set_ylabel('X1 (mm)')
ax.set_title('X1(t) Posterior Median — by Likelihood Transform', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 5: Load Prior Fixed-Parameter Results
#
# We load calibration reports from previous notebooks **without re-running**
# the calibrations.  This follows the "single source of truth" pattern:
# each notebook produces its own reports, downstream notebooks load them.
#
# | Source | Method | Model | Reports |
# |--------|--------|-------|---------|
# | Notebook 02 | SCE-UA | Sacramento | 13 objectives |
# | Notebook 06 | PyDREAM | Sacramento | 13 objectives |
# | Notebook 07 | SCE-UA | GR4J | 13 objectives |
# | Notebook 13 | NUTS fixed | GR4J | 4 transforms |

# %%
OBJECTIVES_13 = [
    "nse", "kge", "rmse", "mae", "pbias", "lognse",
    "kge_log", "kge_sqrt", "kge_inv",
    "fdc_high", "fdc_mid", "fdc_low", "flow_sig",
]

TRANSFORMS_4 = ["none", "sqrt", "log", "inverse"]
TRANSFORM_SUFFIX = {"none": "", "sqrt": "_sqrt", "log": "_log", "inverse": "_inverse"}

prior_results = OrderedDict()

# --- Notebook 02: SCE-UA Sacramento on 410734 (13 objectives) ---
print("Loading Notebook 02 SCE-UA Sacramento reports...")
for obj in OBJECTIVES_13:
    name = f"410734_sacramento_{obj}_sceua"
    pkl = REPORTS_NB02 / f'{name}.pkl'
    if pkl.exists():
        try:
            report = CalibrationReport.load(str(pkl))
            prior_results[f"SCE-UA Sac {obj}"] = {
                "result": report.result,
                "sim": report.simulated,
                "obs": report.observed,
                "metrics": compute_metrics(report.simulated, report.observed),
            }
            print(f"  Loaded: {name}.pkl")
        except Exception as e:
            print(f"  Failed: {name}.pkl ({e})")
    else:
        print(f"  Not found: {name}.pkl")

# --- Notebook 06: PyDREAM Sacramento on 410734 (13 objectives) ---
print("\nLoading Notebook 06 PyDREAM Sacramento reports...")
for obj in OBJECTIVES_13:
    name = f"410734_sacramento_{obj}_dream"
    pkl = REPORTS_PYDREAM / f'{name}.pkl'
    if pkl.exists():
        try:
            report = CalibrationReport.load(str(pkl))
            prior_results[f"PyDREAM Sac {obj}"] = {
                "result": report.result,
                "sim": report.simulated,
                "obs": report.observed,
                "metrics": compute_metrics(report.simulated, report.observed),
            }
            print(f"  Loaded: {name}.pkl")
        except Exception as e:
            print(f"  Failed: {name}.pkl ({e})")
    else:
        print(f"  Not found: {name}.pkl")

# --- Notebook 07: SCE-UA GR4J on 410734 (13 objectives) ---
print("\nLoading Notebook 07 SCE-UA GR4J reports...")
for obj in OBJECTIVES_13:
    name = f"410734_gr4j_{obj}_sceua"
    pkl = REPORTS_MODELS / f'{name}.pkl'
    if pkl.exists():
        try:
            report = CalibrationReport.load(str(pkl))
            prior_results[f"SCE-UA GR4J {obj}"] = {
                "result": report.result,
                "sim": report.simulated,
                "obs": report.observed,
                "metrics": compute_metrics(report.simulated, report.observed),
            }
            print(f"  Loaded: {name}.pkl")
        except Exception as e:
            print(f"  Failed: {name}.pkl ({e})")
    else:
        print(f"  Not found: {name}.pkl")

# --- Notebook 13: NUTS fixed GR4J on 410734 (4 transforms) ---
print("\nLoading Notebook 13 NUTS fixed GR4J reports...")
for t in TRANSFORMS_4:
    suffix = TRANSFORM_SUFFIX[t]
    name = f"410734_gr4j_gaussian_nuts{suffix}"
    pkl = REPORTS_NUTS / f'{name}.pkl'
    if pkl.exists():
        try:
            report = CalibrationReport.load(str(pkl))
            prior_results[f"NUTS-fixed {t}"] = {
                "result": report.result,
                "sim": report.simulated,
                "obs": report.observed,
                "metrics": compute_metrics(report.simulated, report.observed),
            }
            print(f"  Loaded: {name}.pkl")
        except Exception as e:
            print(f"  Failed: {name}.pkl ({e})")
    else:
        print(f"  Not found: {name}.pkl")

print(f"\nTotal prior calibrations loaded: {len(prior_results)}")

# %% [markdown]
# ---
# ## Step 6: Grand Comparison — TVP vs Fixed-Parameter
#
# ### 6.1 TVP vs NUTS-fixed (same model, same algorithm, same transforms)
#
# Direct apples-to-apples comparison showing the benefit of allowing X1
# to vary over time.

# %%
print(f"\n{'=' * 80}")
print(f"  TVP vs NUTS-FIXED  —  GR4J, Gauge 410734  (matched transforms)")
print(f"{'=' * 80}")

for t in TRANSFORMS_4:
    fixed_key = f"NUTS-fixed {t}"
    tvp_key = TRANSFORM_DISPLAY[t]

    if fixed_key not in prior_results:
        print(f"\n  {t}: NUTS-fixed not available — skipping")
        continue

    fixed_m = prior_results[fixed_key]["metrics"]
    tvp_m = tvp_metrics[t]

    print(f"\n  --- Transform: {t} ---")
    print(f"  {'Metric':<20} {'NUTS-fixed':>12} {'TVP-GR4J':>12} {'Δ (TVP−fixed)':>14}")
    print(f"  {'-' * 60}")
    for group_name, keys in METRIC_GROUPS.items():
        for mn in keys:
            fv = fixed_m.get(mn, np.nan)
            tv = tvp_m.get(mn, np.nan)
            if np.isnan(fv) or np.isnan(tv):
                continue
            delta = tv - fv
            better = "+" if ((mn not in ("RMSE", "MAE")) and delta > 0) or \
                            ((mn in ("RMSE", "MAE")) and delta < 0) else ""
            print(f"  {mn:<20} {fv:12.4f} {tv:12.4f} {delta:+14.4f} {better}")

# %% [markdown]
# ### 6.2 TVP vs Best-of-All SCE-UA and PyDREAM

# %%
sceua_entries = {k: v for k, v in prior_results.items() if k.startswith("SCE-UA")}
dream_entries = {k: v for k, v in prior_results.items() if k.startswith("PyDREAM")}

if sceua_entries or dream_entries:
    all_metric_names = list(next(iter(tvp_metrics.values())).keys())

    # Compute best-of-N for each family
    def _best_per_metric(entries, metric_names):
        """For each metric, find the best value across all entries."""
        best = {}
        for mn in metric_names:
            vals = [(k, v["metrics"].get(mn, np.nan)) for k, v in entries.items()]
            vals = [(k, v) for k, v in vals if not np.isnan(v)]
            if not vals:
                best[mn] = np.nan
                continue
            if mn in ("RMSE", "MAE", "FHV", "FMV", "FLV", "PBIAS"):
                best[mn] = min(vals, key=lambda x: abs(x[1]))[1]
            else:
                best[mn] = max(vals, key=lambda x: x[1])[1]
        return best

    best_sceua = _best_per_metric(sceua_entries, all_metric_names) if sceua_entries else {}
    best_dream = _best_per_metric(dream_entries, all_metric_names) if dream_entries else {}

    # Find best TVP transform per metric
    best_tvp = {}
    for mn in all_metric_names:
        vals = [(t, tvp_metrics[t].get(mn, np.nan)) for t in TRANSFORMS]
        vals = [(t, v) for t, v in vals if not np.isnan(v)]
        if not vals:
            best_tvp[mn] = np.nan
            continue
        if mn in ("RMSE", "MAE", "FHV", "FMV", "FLV", "PBIAS"):
            best_tvp[mn] = min(vals, key=lambda x: abs(x[1]))[1]
        else:
            best_tvp[mn] = max(vals, key=lambda x: x[1])[1]

    comparison_dict = OrderedDict()
    comparison_dict["Best TVP"] = best_tvp
    if best_sceua:
        comparison_dict["Best SCE-UA"] = best_sceua
    if best_dream:
        comparison_dict["Best PyDREAM"] = best_dream

    for t in TRANSFORMS_4:
        nf_key = f"NUTS-fixed {t}"
        if nf_key in prior_results:
            comparison_dict[f"NUTS-fixed {t}"] = prior_results[nf_key]["metrics"]

    print_multi_method_comparison(
        comparison_dict,
        label="Best-of TVP vs Best-of SCE-UA/PyDREAM/NUTS-fixed"
    )
else:
    print("No prior SCE-UA or PyDREAM reports available for comparison.")
    print("Run notebooks 02, 06, and 07 first to generate reports.")

# %% [markdown]
# ### 6.3 Regime-Based Analysis
#
# Group the 27 metrics by flow regime to see where TVP helps most.

# %%
REGIME_GROUPS = OrderedDict([
    ("Very High",  ["FHV"]),
    ("High",       ["NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "RMSE"]),
    ("Mid",        ["SqrtNSE", "KGE_sqrt", "KGE_sqrt_r", "KGE_sqrt_alpha",
                    "KGE_sqrt_beta", "MAE", "FMV"]),
    ("Low",        ["LogNSE", "KGE_log", "KGE_log_r", "KGE_log_alpha",
                    "KGE_log_beta", "PBIAS"]),
    ("Very Low",   ["InvNSE", "KGE_inv", "KGE_inv_r", "KGE_inv_alpha",
                    "KGE_inv_beta", "FLV"]),
])

# Count wins per method across all regimes
if sceua_entries or dream_entries:
    print(f"\n{'=' * 70}")
    print(f"  REGIME-BASED ANALYSIS — TVP wins vs fixed-parameter methods")
    print(f"{'=' * 70}")

    for regime, metrics in REGIME_GROUPS.items():
        tvp_wins = 0
        total = 0
        for mn in metrics:
            tvp_val = best_tvp.get(mn, np.nan)
            ref_vals = []
            if best_sceua:
                ref_vals.append(best_sceua.get(mn, np.nan))
            if best_dream:
                ref_vals.append(best_dream.get(mn, np.nan))
            for t in TRANSFORMS_4:
                nf_key = f"NUTS-fixed {t}"
                if nf_key in prior_results:
                    ref_vals.append(prior_results[nf_key]["metrics"].get(mn, np.nan))

            ref_vals = [v for v in ref_vals if not np.isnan(v)]
            if np.isnan(tvp_val) or not ref_vals:
                continue
            total += 1

            if mn in ("RMSE", "MAE", "PBIAS", "FHV", "FMV", "FLV"):
                if abs(tvp_val) <= min(abs(v) for v in ref_vals):
                    tvp_wins += 1
            else:
                if tvp_val >= max(ref_vals):
                    tvp_wins += 1

        print(f"  {regime:<12}: TVP wins {tvp_wins}/{total} metrics")
    print(f"{'=' * 70}")

# %% [markdown]
# ### 6.4 Best-Transform Yearly Decomposition
#
# For the best-performing TVP transform (by NSE), produce the full yearly
# decomposition showing seasonal X1 patterns.

# %%
nse_by_transform = {t: tvp_metrics[t].get('NSE', -999) for t in TRANSFORMS}
best_transform = max(nse_by_transform, key=nse_by_transform.get)
print(f"Best TVP transform by NSE: {TRANSFORM_DISPLAY[best_transform]} "
      f"(NSE = {nse_by_transform[best_transform]:.4f})")

plot_tvp_yearly_decomposition(
    tvp_idatas[best_transform], "X1", dates_410734, ci=0.90
)

# %% [markdown]
# ### 6.5 Heatmap — Metrics by Method

# %%
heatmap_methods = OrderedDict()
for t in TRANSFORMS:
    heatmap_methods[TRANSFORM_DISPLAY[t]] = tvp_metrics[t]

for t in TRANSFORMS_4:
    nf_key = f"NUTS-fixed {t}"
    if nf_key in prior_results:
        heatmap_methods[f"NUTS-fixed {t}"] = prior_results[nf_key]["metrics"]

if sceua_entries:
    heatmap_methods["Best SCE-UA"] = best_sceua
if dream_entries:
    heatmap_methods["Best PyDREAM"] = best_dream

display_metrics = ["NSE", "LogNSE", "SqrtNSE", "InvNSE",
                   "KGE", "KGE_log", "KGE_sqrt", "KGE_inv",
                   "RMSE", "MAE", "PBIAS", "FHV", "FMV", "FLV"]

method_names = list(heatmap_methods.keys())
data_matrix = np.full((len(display_metrics), len(method_names)), np.nan)
for j, method in enumerate(method_names):
    for i, mn in enumerate(display_metrics):
        data_matrix[i, j] = heatmap_methods[method].get(mn, np.nan)

fig, ax = plt.subplots(figsize=(max(12, len(method_names) * 1.4), 10))
im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(display_metrics)))
ax.set_yticklabels(display_metrics, fontsize=9)

for i in range(len(display_metrics)):
    for j in range(len(method_names)):
        v = data_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > 0.7 * np.nanmax(np.abs(data_matrix)) else 'black')

ax.set_title('Diagnostic Metrics — TVP vs Fixed-Parameter Methods', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 7: Summary
#
# ### What We Accomplished
#
# 1. **Calibrated TVP-GR4J** with Gaussian Random Walk on X1 using 4
#    likelihood transforms (Q, √Q, log Q, 1/Q) on gauge 410734
# 2. **Visualised X1(t) trajectories** showing how the production store
#    capacity evolves over time — both as full-record traces and Santos
#    et al. Figure 8-style yearly decompositions
# 3. **Compared across transforms** to see how the likelihood space
#    affects the inferred X1 trajectory
# 4. **Loaded all prior fixed-parameter calibrations** from Notebook 02 (SCE-UA),
#    Notebook 06 (PyDREAM), Notebook 07 (model comparison), and Notebook 13 (NUTS fixed)
# 5. **Produced a comprehensive grand comparison** of TVP vs fixed across
#    27 diagnostic metrics, grouped by flow regime
#
# ### Key Questions Addressed
#
# - **Does TVP improve fit quality?** Compare the 27-metric suite across
#   TVP and all fixed-parameter methods
# - **For which flow regimes?** The regime-based analysis shows where TVP
#   gains (or loses) ground
# - **Which likelihood transform works best with TVP?** The transform
#   comparison reveals which residual space best exploits the TVP flexibility
# - **How does TVP computational cost compare?** Runtime comparison shows
#   the overhead of sampling GRW hyperparameters
# - **How does TVP-GR4J (4 params + GRW) compare to Sacramento (22 fixed
#   params)?** Can a simple model with temporal flexibility match a complex
#   model with fixed parameters?
#
# ### Future Work
#
# - **TVP on multiple parameters**: allow X2 and/or X3 to vary
# - **Alternative TVP priors**: Gaussian Processes, changepoint models,
#   seasonal basis functions
# - **Validation period**: test whether TVP improves out-of-sample
#   prediction (critical for operational use)
# - **Physical interpretation**: relate X1(t) temporal patterns to
#   observed land-use change, fire events, or drought indices
