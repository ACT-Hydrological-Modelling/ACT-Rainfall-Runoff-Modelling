# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pyrrm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # APEX: Low Flow Focused Objective Function for Hydrological Calibration
#
# ## Overview
#
# This notebook demonstrates **APEX** (Adaptive Process-Explicit), a composite objective
# function designed to emphasize low flow performance while maintaining overall model quality.
#
# ## What This Notebook Does
#
# 1. **Loads pre-calibrated baselines** from notebook 02:
#    - NSE (traditional high-flow biased)
#    - InvNSE (inverse-transformed for low flows)
#    - SDEB (state-of-the-art composite)
#
# 2. **Calibrates APEX** with low flow focus:
#    - Core: NSE + NSE(inverse) - 40% weight
#    - FDC: Emphasis on low flow segment - 30% weight
#    - Signatures: Baseflow index emphasis - 20% weight
#    - Bias/Timing: Standard controls - 10% weight
#
# 3. **Comprehensive comparison** across multiple metrics
#
# ## Estimated Runtime
#
# - Data loading: < 1 minute
# - APEX calibration: ~20-30 minutes
# - Analysis: ~5 minutes
# - **Total**: ~30-40 minutes

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# pyrrm imports
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner, CalibrationReport
from pyrrm.objectives import (
    NSE, KGE, RMSE, MAE, PBIAS,
    FlowTransformation, PearsonCorrelation,
    FDCMetric, SignatureMetric, SDEB
)
from pyrrm.objectives.composite import WeightedObjective

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("✓ Imports complete")

# %% [markdown]
# ---
# # PART 1: DATA LOADING AND PREPARATION
#
# We follow the exact same approach as notebook 02 to ensure data consistency.

# %%
# Configuration
DATA_DIR = Path('../data/410734')
CATCHMENT_ID = '410734'
CATCHMENT_AREA_KM2 = 516.62667  # km²
WARMUP_DAYS = 365

print("CONFIGURATION")
print("=" * 70)
print(f"Catchment: {CATCHMENT_ID}")
print(f"Area: {CATCHMENT_AREA_KM2:.2f} km²")
print(f"Warmup period: {WARMUP_DAYS} days")

# %%
# Load rainfall data
rainfall_file = DATA_DIR / 'Default Input Set - Rain_QBN01.csv'
rainfall_df = pd.read_csv(rainfall_file, parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

print("\nRAINFALL DATA")
print("=" * 70)
print(f"Records: {len(rainfall_df):,} days")
print(f"Period: {rainfall_df.index.min().date()} to {rainfall_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(rainfall_df['rainfall'].describe().round(2))

# %%
# Load PET data
pet_file = DATA_DIR / 'Default Input Set - Mwet_QBN01.csv'
pet_df = pd.read_csv(pet_file, parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

print("\nPOTENTIAL EVAPOTRANSPIRATION (PET)")
print("=" * 70)
print(f"Records: {len(pet_df):,} days")
print(f"Period: {pet_df.index.min().date()} to {pet_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(pet_df['pet'].describe().round(2))

# %%
# Load observed streamflow
obs_file = DATA_DIR / '410734_recorded_Flow.csv'
observed_df = pd.read_csv(obs_file, parse_dates=['Date'], index_col='Date')
observed_df.columns = ['observed_flow']

# Clean observed flow data (CRITICAL!)
# Handle sentinel values (-9999 = missing) and negative flows
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

print("\nOBSERVED STREAMFLOW")
print("=" * 70)
print(f"Records: {len(observed_df):,} days (after cleaning)")
print(f"Period: {observed_df.index.min().date()} to {observed_df.index.max().date()}")
print(f"\nStatistics (ML/day):")
print(observed_df['observed_flow'].describe().round(2))

# %%
# Merge all datasets
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

print("\nMERGED DATASET")
print("=" * 70)
print(f"Total records: {len(data):,} days")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
print(f"Columns: {list(data.columns)}")
print(f"\nMissing values:")
print(data.isna().sum())

# %%
# Prepare data for calibration
cal_data = data.copy()
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

print("\nCALIBRATION DATA PREPARED")
print("=" * 70)
print(f"Input shape: {cal_inputs.shape}")
print(f"Observed shape: {cal_observed.shape}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_observed) - WARMUP_DAYS:,} days")
print("\n✓ Data loading complete!")

# %% [markdown]
# ---
# # PART 2: LOADING BASELINE CALIBRATIONS
#
# We load NSE, InvNSE, and SDEB calibrations that were created in notebook 02.
# These provide validated baseline performance for comparison.

# %%
print("\n" + "=" * 80)
print("LOADING PRE-CALIBRATED BASELINES")
print("=" * 80)

# Load NSE calibration
print("\n1. Loading NSE calibration...")
report_nse = CalibrationReport.load('../test_data/reports/410734_nse.pkl')
result_nse = report_nse.result
sim_nse = report_nse.simulated
obs_nse = report_nse.observed
print(f"   ✓ NSE: {result_nse.best_objective:.4f}")
print(f"   Data period: {len(obs_nse):,} days")

# Load InvNSE calibration
print("\n2. Loading InvNSE calibration...")
report_invnse = CalibrationReport.load('../test_data/reports/410734_invnse.pkl')
result_invnse = report_invnse.result
sim_invnse = report_invnse.simulated
obs_invnse = report_invnse.observed
print(f"   ✓ InvNSE: {result_invnse.best_objective:.4f}")
print(f"   Data period: {len(obs_invnse):,} days")

# Load SDEB calibration
print("\n3. Loading SDEB calibration...")
report_sdeb = CalibrationReport.load('../test_data/reports/410734_sdeb.pkl')
result_sdeb = report_sdeb.result
sim_sdeb = report_sdeb.simulated
obs_sdeb = report_sdeb.observed
print(f"   ✓ SDEB: {result_sdeb.best_objective:.4f}")
print(f"   Data period: {len(obs_sdeb):,} days")

print("\n" + "=" * 80)
print("✓ All baseline calibrations loaded successfully!")
print("=" * 80)

# %% [markdown]
# ---
# # PART 3: APEX CALIBRATION (Low Flow Focus)
#
# Now we calibrate APEX with emphasis on low flows:
#
# **Weight Distribution:**
# - **Core (40%)**: NSE (20%) + NSE(inverse) (20%) - balanced with strong low flow component
# - **FDC (30%)**: High (5%), Mid (10%), Low (15%) - emphasizes low flows
# - **Signatures (20%)**: Baseflow (15%), Flashiness (5%) - emphasizes baseflow
# - **Bias/Timing (10%)**: PBIAS (5%), Timing Correlation (5%) - standard

# %%
print("\n" + "=" * 80)
print("APEX CALIBRATION (Low Flow Focus)")
print("=" * 80)

# Configure APEX for low flow emphasis - SIMPLIFIED VERSION
# Issue: Complex APEX with many components may have normalization issues
# Solution: Use fewer, more robust components
apex_obj = WeightedObjective(
    objectives=[
        # Core performance (60%)
        (NSE(), 0.30),  # Overall fit
        (NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)), 0.30),  # Low flow fit
        
        # Flow distribution (30%)
        (FDCMetric(segment='low'), 0.20),  # Low flow FDC
        (SignatureMetric('baseflow_index'), 0.10),  # Baseflow
        
        # Bias (10%)
        (PBIAS(), 0.10),  # Volume error
    ],
    aggregation='weighted_sum',
    normalize=True,
    normalize_method='minmax'
)

runner_apex = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=apex_obj,
    warmup_period=WARMUP_DAYS
)

print("\nSimplified APEX Configuration (Low Flow Focus):")
print("  Core (60%): NSE (30%) + NSE(inverse) (30%)")
print("  Flow Distribution (30%): FDC Low (20%), Baseflow Index (10%)")
print("  Bias (10%): PBIAS (10%)")
print("\nNote: Using simplified APEX with fewer components for robustness")

# TEST: Verify APEX works with baseline NSE simulation before calibration
print("\n" + "=" * 70)
print("Pre-Calibration Test: Evaluating NSE-calibrated params with APEX")
print("=" * 70)
# Note: baseline simulations are on their own data period from notebook 02
# We'll evaluate APEX on that data for consistency
test_apex_score = apex_obj(obs_nse, sim_nse)
print(f"APEX score for NSE-calibrated params: {test_apex_score:.4f}")
print("(Expected: reasonable value between 0.3-0.8)")

# Also test individual components to diagnose any issues
print("\nComponent Breakdown:")
test_nse = NSE()(obs_nse, sim_nse)
test_nse_inv = NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))(obs_nse, sim_nse)
test_fdc_low = FDCMetric(segment='low')(obs_nse, sim_nse)
test_bf = SignatureMetric('baseflow_index')(obs_nse, sim_nse)
test_pbias = PBIAS()(obs_nse, sim_nse)
print(f"  NSE: {test_nse:.4f}")
print(f"  NSE(inverse): {test_nse_inv:.4f}")
print(f"  FDC Low: {test_fdc_low:.4f}")
print(f"  Baseflow Index: {test_bf:.4f}")
print(f"  PBIAS: {test_pbias:.2f}")

if test_apex_score < 0 or test_apex_score > 1.0:
    print("\n⚠️  WARNING: APEX score is outside expected range!")
    print("   This suggests a potential issue with the objective function.")
else:
    print("\n✓ APEX objective function is working correctly.")

print("\nRunning SCE-UA calibration...")
print("This will take ~20-30 minutes...")

result_apex = runner_apex.run_sceua_direct(
    max_evals=10000,
    seed=42,
    verbose=True,
    max_tolerant_iter=100,
    tolerance=1e-4
)

print(f"\n✓ APEX Calibration complete!")
print(f"  Best APEX: {result_apex.best_objective:.4f}")
print(f"  Runtime: {result_apex.runtime_seconds:.1f} seconds")

# DIAGNOSTIC: Check APEX calibration quality
print("\n" + "=" * 80)
print("APEX CALIBRATION DIAGNOSTICS")
print("=" * 80)

# Evaluate APEX with individual component metrics
print("\nAPEX Component Evaluation:")
nse_apex = NSE()(cal_observed[WARMUP_DAYS:], runner_apex.model.run(cal_inputs)['flow'].values[WARMUP_DAYS:])
nse_inv_apex = NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))(
    cal_observed[WARMUP_DAYS:], 
    runner_apex.model.run(cal_inputs)['flow'].values[WARMUP_DAYS:]
)
print(f"  NSE: {nse_apex:.4f}")
print(f"  NSE(inverse): {nse_inv_apex:.4f}")

# Check parameter values are reasonable
print("\nAPEX Best Parameters (sample):")
param_names = list(result_apex.best_parameters.keys())[:5]
for name in param_names:
    print(f"  {name}: {result_apex.best_parameters[name]:.4f}")

# Save APEX calibration
report_apex = runner_apex.create_report(result_apex, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
report_apex.save('../test_data/reports/410734_apex_lowflow')
print(f"\n✓ APEX calibration saved to: test_data/reports/410734_apex_lowflow.pkl")

# Extract APEX simulation
sim_apex = report_apex.simulated
obs_apex = report_apex.observed

print(f"\nAPEX Simulation Statistics:")
print(f"  Observed mean: {obs_apex.mean():.2f} ML/day")
print(f"  Simulated mean: {sim_apex.mean():.2f} ML/day")
print(f"  Data length: {len(obs_apex):,} days")

# %% [markdown]
# ---
# # PART 4: PERFORMANCE COMPARISON
#
# Now we evaluate all four calibrations across multiple metrics.

# %%
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

eval_metrics = {
    'NSE': NSE(),
    'NSE(sqrt)': NSE(transform=FlowTransformation('sqrt')),
    'NSE(log)': NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
    'NSE(inverse)': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'KGE': KGE(variant='2012'),
    'KGE(sqrt)': KGE(variant='2012', transform=FlowTransformation('sqrt')),
    'PBIAS': PBIAS(),
    'RMSE': RMSE(),
    'MAE': MAE(),
    'FDC High': FDCMetric(segment='high'),
    'FDC Mid': FDCMetric(segment='mid'),
    'FDC Low': FDCMetric(segment='low'),
    'Baseflow Index': SignatureMetric('baseflow_index'),
    'Flashiness': SignatureMetric('flashiness'),
    'Pearson r': PearsonCorrelation(),
}

# Evaluate each calibration against its own observed data
results_comparison = {}
for name, metric in eval_metrics.items():
    results_comparison[name] = {
        'NSE': metric(obs_nse, sim_nse),
        'InvNSE': metric(obs_invnse, sim_invnse),
        'SDEB': metric(obs_sdeb, sim_sdeb),
        'APEX': metric(obs_apex, sim_apex)
    }

results_df = pd.DataFrame(results_comparison).T

print("\nPerformance Metrics:")
print("=" * 80)
print(results_df.to_string(float_format=lambda x: f'{x:7.4f}'))

# %%
# Identify best performer for each metric
print("\n" + "=" * 80)
print("BEST PERFORMER PER METRIC")
print("=" * 80)

for metric_name in results_df.index:
    row = results_df.loc[metric_name]
    
    # Skip if all NaN
    if row.isna().all():
        print(f"{metric_name:20s}: N/A (all NaN)")
        continue
    
    # Determine if maximize or minimize
    metric = eval_metrics[metric_name]
    if hasattr(metric, 'direction'):
        is_maximize = (metric.direction == 'maximize')
    elif hasattr(metric, 'maximize'):
        is_maximize = metric.maximize
    else:
        is_maximize = True
    
    if is_maximize:
        best_method = row.idxmax(skipna=True)
        best_value = row.max(skipna=True)
    else:
        best_method = row.idxmin(skipna=True)
        best_value = row.min(skipna=True)
    
    print(f"{metric_name:20s}: {best_method:10s} ({best_value:7.4f})")

# %% [markdown]
# ---
# # PART 5: VISUALIZATION
#
# Interactive visual comparison of the four calibrations using Plotly.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("\n" + "=" * 80)
print("INTERACTIVE VISUALIZATIONS (Plotly)")
print("=" * 80)

# %%
# Interactive Hydrograph Comparison (first 2 years)
print("\n1. Interactive Hydrographs (4-panel comparison)...")

n_days = min(730, len(obs_apex))  # First 2 years
dates_plot = cal_data.index[WARMUP_DAYS:WARMUP_DAYS+n_days]

fig_hydro = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        'NSE Calibration (High Flow Biased)',
        'InvNSE Calibration (Low Flow Focused)', 
        'SDEB Calibration (State-of-the-Art)',
        'APEX Calibration (Low Flow Focus)'
    ),
    vertical_spacing=0.06,
    shared_xaxes=True
)

# NSE
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=obs_nse[:n_days], mode='lines',
               name='Observed', line=dict(color='black', width=1.5),
               legendgroup='obs', showlegend=True),
    row=1, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=sim_nse[:n_days], mode='lines',
               name='NSE', line=dict(color='blue', width=1),
               legendgroup='nse', showlegend=True),
    row=1, col=1
)

# InvNSE
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=obs_invnse[:n_days], mode='lines',
               name='Observed', line=dict(color='black', width=1.5),
               legendgroup='obs', showlegend=False),
    row=2, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=sim_invnse[:n_days], mode='lines',
               name='InvNSE', line=dict(color='green', width=1),
               legendgroup='invnse', showlegend=True),
    row=2, col=1
)

# SDEB
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=obs_sdeb[:n_days], mode='lines',
               name='Observed', line=dict(color='black', width=1.5),
               legendgroup='obs', showlegend=False),
    row=3, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=sim_sdeb[:n_days], mode='lines',
               name='SDEB', line=dict(color='orange', width=1),
               legendgroup='sdeb', showlegend=True),
    row=3, col=1
)

# APEX
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=obs_apex[:n_days], mode='lines',
               name='Observed', line=dict(color='black', width=1.5),
               legendgroup='obs', showlegend=False),
    row=4, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates_plot, y=sim_apex[:n_days], mode='lines',
               name='APEX', line=dict(color='red', width=1),
               legendgroup='apex', showlegend=True),
    row=4, col=1
)

# Update layout
fig_hydro.update_xaxes(title_text="Date", row=4, col=1)
fig_hydro.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
fig_hydro.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig_hydro.update_yaxes(title_text="Flow (ML/day)", row=3, col=1)
fig_hydro.update_yaxes(title_text="Flow (ML/day)", row=4, col=1)

fig_hydro.update_layout(
    height=1200,
    title_text="<b>Hydrograph Comparison: NSE vs InvNSE vs SDEB vs APEX</b><br>" +
               "<sub>First 2 years of simulation period</sub>",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig_hydro.show()
print("✓ Interactive hydrograph complete")

# %%
# Interactive Flow Duration Curves
print("\n2. Interactive Flow Duration Curves...")

fig_fdc = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Full Range', 'Low Flows (70-100% exceedance)'),
    horizontal_spacing=0.12
)

# Prepare FDC data
fdc_data = [
    (obs_nse, sim_nse, 'NSE', 'blue'),
    (obs_invnse, sim_invnse, 'InvNSE', 'green'),
    (obs_sdeb, sim_sdeb, 'SDEB', 'orange'),
    (obs_apex, sim_apex, 'APEX', 'red')
]

# Full range FDC
for i, (obs, sim, label, color) in enumerate(fdc_data):
    sorted_obs = np.sort(obs)[::-1]
    sorted_sim = np.sort(sim)[::-1]
    exceedance = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim) * 100
    
    # Observed (only plot once)
    if i == 0:
        fig_fdc.add_trace(
            go.Scatter(x=exceedance, y=sorted_obs, mode='lines',
                       name='Observed', line=dict(color='black', width=2),
                       legendgroup='obs', showlegend=True),
            row=1, col=1
        )
        fig_fdc.add_trace(
            go.Scatter(x=exceedance, y=sorted_obs, mode='lines',
                       name='Observed', line=dict(color='black', width=2),
                       legendgroup='obs', showlegend=False),
            row=1, col=2
        )
    
    # Simulated - full range
    fig_fdc.add_trace(
        go.Scatter(x=exceedance, y=sorted_sim, mode='lines',
                   name=label, line=dict(color=color, width=1.5),
                   legendgroup=label, showlegend=True),
        row=1, col=1
    )
    
    # Simulated - low flows (70-100%)
    mask = exceedance >= 70
    fig_fdc.add_trace(
        go.Scatter(x=exceedance[mask], y=sorted_sim[mask], mode='lines',
                   name=label, line=dict(color=color, width=1.5),
                   legendgroup=label, showlegend=False),
        row=1, col=2
    )

# Update axes
fig_fdc.update_xaxes(title_text="Exceedance Probability (%)", row=1, col=1)
fig_fdc.update_xaxes(title_text="Exceedance Probability (%)", row=1, col=2, range=[70, 100])
fig_fdc.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig_fdc.update_yaxes(title_text="Flow (ML/day)", row=1, col=2)

fig_fdc.update_layout(
    height=500,
    title_text="<b>Flow Duration Curve Comparison</b><br>" +
               "<sub>Left: Full range (log scale), Right: Low flows detail</sub>",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.15)
)

fig_fdc.show()
print("✓ Interactive FDC complete")

# %%
# Performance summary heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Select key metrics for heatmap
key_metrics = ['NSE', 'NSE(sqrt)', 'NSE(inverse)', 'KGE', 'PBIAS', 'RMSE', 
               'FDC Low', 'Baseflow Index', 'Pearson r']
heatmap_data = results_df.loc[key_metrics]

# Normalize for visualization (0-1 scale, higher is better)
heatmap_normalized = heatmap_data.copy()
for metric in key_metrics:
    values = heatmap_data.loc[metric]
    
    # Check if minimize metric
    metric_obj = eval_metrics[metric]
    if hasattr(metric_obj, 'direction'):
        is_maximize = (metric_obj.direction == 'maximize')
    elif hasattr(metric_obj, 'maximize'):
        is_maximize = metric_obj.maximize
    else:
        is_maximize = True
    
    if not is_maximize:
        # For minimize metrics, invert (lower is better)
        values = -values
    
    # Min-max normalization
    vmin, vmax = values.min(), values.max()
    if vmax != vmin:
        heatmap_normalized.loc[metric] = (values - vmin) / (vmax - vmin)
    else:
        heatmap_normalized.loc[metric] = 0.5

sns.heatmap(heatmap_normalized, annot=heatmap_data.values, fmt='.3f', 
            cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
            cbar_kws={'label': 'Normalized Performance (0=worst, 1=best)'},
            ax=ax)
ax.set_title('Calibration Performance Comparison\n(Normalized across methods)', fontsize=14, fontweight='bold')
ax.set_xlabel('Calibration Method', fontsize=12)
ax.set_ylabel('Performance Metric', fontsize=12)

plt.tight_layout()
plt.savefig('../figures/apex_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Performance heatmap complete")

# %% [markdown]
# ---
# # SUMMARY
#
# ## Key Findings
#
# 1. **NSE (Traditional)**:
#    - Optimizes overall fit with high flow bias
#    - Best for flood prediction applications
#
# 2. **InvNSE (Low Flow Focus)**:
#    - Inverse transformation heavily emphasizes low flows
#    - May sacrifice high flow performance
#
# 3. **SDEB (State-of-the-Art)**:
#    - Balanced composite approach
#    - Good all-around performance
#
# 4. **APEX (Low Flow Focus)**:
#    - Configured to emphasize low flows while maintaining overall quality
#    - Combines multiple metrics for robust calibration
#    - Should show improved low flow metrics compared to NSE/SDEB
#
# ## Next Steps
#
# - Compare APEX performance on low flow metrics (NSE(inverse), FDC Low, Baseflow Index)
# - Evaluate if APEX achieves better balance than InvNSE
# - Consider different APEX configurations for other applications
#
# print("\n" + "=" * 80)
# print("NOTEBOOK COMPLETE!")
# print("=" * 80)
# print("\nAll calibrations compared successfully.")
# print("Results saved to: test_data/reports/410734_apex_lowflow.pkl")
# print("Figures saved to: figures/apex_*.png")
