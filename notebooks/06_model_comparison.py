# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python (pyrrm)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Model Comparison: Choosing the Right Rainfall-Runoff Model
#
# ## Purpose
#
# This notebook demonstrates how to configure, calibrate, and compare different
# rainfall-runoff models available in pyrrm. By using the same calibration
# algorithm (SCE-UA) and objective function (KGE), we can isolate the effect
# of model structure on simulation performance.
#
# ## What You'll Learn
#
# - How different rainfall-runoff models are structured conceptually
# - How to configure and calibrate each model in pyrrm
# - How model complexity affects calibration and performance
# - How to choose the right model for your application
# - The concept of parsimony in hydrological modeling
#
# ## Prerequisites
#
# - Completed **Notebook 02: Calibration Quickstart**
# - Basic understanding of calibration concepts
#
# ## Estimated Time
#
# - ~20 minutes for concepts
# - ~1-2 hours for all calibrations (can be shortened)
#
# ## Key Insight
#
# > **More parameters ≠ better performance.** Simple models often generalize
# > better to new data. The "best" model depends on your application, data
# > quality, and computational constraints.

# %% [markdown]
# ---
# ## The Model Selection Problem
#
# ### Why Model Structure Matters
#
# Different rainfall-runoff models represent catchment processes with varying
# levels of complexity:
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    SPECTRUM OF MODEL COMPLEXITY                             │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  SIMPLE                                                          COMPLEX   │
# │    ◄─────────────────────────────────────────────────────────────────►     │
# │                                                                             │
# │    GR4J        GR5J        GR6J                    Sacramento              │
# │    (4 params)  (5 params)  (6 params)              (17+ params)            │
# │                                                                             │
# │    ┌─────────┐ ┌─────────┐ ┌─────────┐             ┌─────────────────┐     │
# │    │Production│ │+Exchange│ │+Exponent│             │ Upper Zone      │     │
# │    │ Store   │ │Threshold│ │ Store   │             │ Lower Zone      │     │
# │    │Routing  │ │         │ │         │             │ Percolation     │     │
# │    │ Store   │ │         │ │         │             │ Baseflow        │     │
# │    └─────────┘ └─────────┘ └─────────┘             │ Interflow       │     │
# │                                                     │ Impervious      │     │
# │                                                     └─────────────────┘     │
# │                                                                             │
# │  PROS:         - Easier to calibrate                - More realistic       │
# │                - Less overfitting risk              - Detailed states      │
# │                - Faster to run                      - Multiple pathways    │
# │                                                                             │
# │  CONS:         - May miss processes                 - Many parameters      │
# │                - Less physically realistic          - Equifinality risk    │
# │                - Limited state tracking             - Slower calibration   │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
# ```
#
# ### The Parsimony Principle
#
# In hydrology, there's a tension between:
# - **Realism**: More parameters can represent more physical processes
# - **Robustness**: Fewer parameters means less overfitting to calibration data
#
# This is related to the **bias-variance tradeoff** in machine learning.
# A model with more parameters has lower bias but higher variance - it fits
# the calibration data better but may perform worse on new data.

# %% [markdown]
# ---
# ## Models Available in pyrrm
#
# ### Overview Table
#
# | Model | Parameters | Origin | Key Features |
# |-------|------------|--------|--------------|
# | **GR4J** | 4 | INRAE, France | Production + routing stores, UH |
# | **GR5J** | 5 | INRAE, France | GR4J + exchange threshold |
# | **GR6J** | 6 | INRAE, France | GR5J + exponential store for low flows |
# | **Sacramento** | 17-22 | US NWS | Multi-zone soil moisture accounting |
#
# ### GR4J (Génie Rural à 4 paramètres Journalier)
#
# The GR4J model is a parsimonious, well-tested model developed by INRAE in France.
# Despite having only 4 parameters, it performs remarkably well across diverse
# catchments worldwide.
#
# **Structure:**
# ```
# Rainfall → Production Store (soil moisture) → Routing Store → Streamflow
#                    ↓                               ↑
#            Percolation ────────────────────────────┘
# ```
#
# **Parameters:**
# - **X1**: Production store capacity (mm) - soil water holding capacity
# - **X2**: Groundwater exchange coefficient (mm/d) - inter-catchment exchange
# - **X3**: Routing store capacity (mm) - quick response storage
# - **X4**: Unit hydrograph time base (days) - routing delay
#
# ### GR5J
#
# GR5J extends GR4J with an additional parameter controlling when groundwater
# exchange occurs. This improves simulation in catchments with significant
# groundwater interactions.
#
# **Additional Parameter:**
# - **X5**: Exchange threshold (-) - storage level at which exchange activates
#
# ### GR6J
#
# GR6J adds an exponential store to improve low-flow simulation. The exponential
# store provides a slow release mechanism that better represents long-term
# baseflow recession.
#
# **Additional Parameter:**
# - **X6**: Exponential store capacity (mm) - slow response storage
#
# ### Sacramento
#
# The Sacramento model is a complex, process-based model with 5 soil zones
# and multiple flow pathways. It's the standard for operational flood forecasting
# in the US and Australia.
#
# **See Notebook 01** for a detailed description of Sacramento's structure.

# %% [markdown]
# ---
# ## Setup

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time

# Interactive visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

print("=" * 70)
print("MODEL COMPARISON: RAINFALL-RUNOFF MODELS")
print("=" * 70)

# %%
# Import pyrrm models
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
from pyrrm.calibration import CalibrationRunner, SPOTPY_AVAILABLE
from pyrrm.calibration.objective_functions import KGE, NSE, calculate_metrics

print("\npyrrm models imported successfully!")
print(f"\nAvailable models:")
print(f"  - Sacramento (17+ parameters)")
print(f"  - GR4J (4 parameters)")
print(f"  - GR5J (5 parameters)")
print(f"  - GR6J (6 parameters)")
print(f"\nCalibration backend (SpotPy): {SPOTPY_AVAILABLE}")

# %% [markdown]
# ---
# ## Load and Prepare Data

# %%
# Load data
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

# Load datasets
rainfall_df = pd.read_csv(DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
                          parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

pet_df = pd.read_csv(DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
                     parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

flow_df = pd.read_csv(DATA_DIR / '410734_output_SDmodel.csv',
                      parse_dates=['Date'], index_col='Date')
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df = observed_df.dropna()

data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

# Define periods
WARMUP_DAYS = 365
CAL_START = pd.Timestamp('1990-01-01')
CAL_END = pd.Timestamp('1999-12-31')  # 10 years calibration
VAL_START = pd.Timestamp('2000-01-01')
VAL_END = pd.Timestamp('2009-12-31')  # 10 years validation

cal_data = data[(data.index >= CAL_START) & (data.index <= CAL_END)].copy()
val_data = data[(data.index >= VAL_START) & (data.index <= VAL_END)].copy()

print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"\nCatchment: Gauge 410734 (Queanbeyan River)")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")
print(f"\nCalibration period: {CAL_START.date()} to {CAL_END.date()}")
print(f"  Records: {len(cal_data)}")
print(f"\nValidation period: {VAL_START.date()} to {VAL_END.date()}")
print(f"  Records: {len(val_data)}")
print(f"\nWarmup: {WARMUP_DAYS} days")

# %% [markdown]
# ---
# ## Configure the Models
#
# Let's examine each model's structure and parameter bounds before calibration.

# %%
# Model configuration display
print("=" * 70)
print("MODEL CONFIGURATIONS")
print("=" * 70)

# Create instances to inspect
models_info = {
    'GR4J': GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
    'GR5J': GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
    'GR6J': GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
    'Sacramento': Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
}

for name, model in models_info.items():
    bounds = model.get_parameter_bounds()
    print(f"\n{name}")
    print("-" * 40)
    print(f"Parameters: {len(bounds)}")
    print(f"Description: {model.description}")
    print(f"\nParameter bounds:")
    for param, (low, high) in list(bounds.items())[:8]:  # Show first 8
        print(f"  {param:8s}: [{low:10.3f}, {high:10.3f}]")
    if len(bounds) > 8:
        print(f"  ... and {len(bounds) - 8} more parameters")

# %% [markdown]
# ### Parameter Count Comparison
#
# Notice the dramatic difference in complexity:
# - **GR4J**: 4 parameters - very parsimonious
# - **GR5J**: 5 parameters - adds exchange threshold
# - **GR6J**: 6 parameters - adds exponential store
# - **Sacramento**: 17+ parameters - detailed process representation
#
# The search space grows exponentially with parameters, making Sacramento
# much harder to calibrate optimally.

# %% [markdown]
# ---
# ## Calibration Configuration
#
# We use the same settings for all models to ensure fair comparison:
# - **Algorithm**: SCE-UA (reliable global optimization)
# - **Objective**: KGE (balanced multi-component metric)
# - **Iterations**: Scaled by parameter count (more params = more iterations)

# %%
# Calibration configuration
OBJECTIVE = KGE()  # Same objective for all

# Scale iterations by model complexity
# Rule of thumb: ~500-1000 evaluations per parameter for SCE-UA
BASE_ITERATIONS = 2000
DEMO_MODE = True  # Set False for production runs

if DEMO_MODE:
    ITERATIONS_BY_MODEL = {
        'GR4J': 2000,
        'GR5J': 2500,
        'GR6J': 3000,
        'Sacramento': 5000,
    }
    print("⚠ DEMO MODE: Using reduced iterations for speed")
else:
    ITERATIONS_BY_MODEL = {
        'GR4J': 5000,
        'GR5J': 7000,
        'GR6J': 10000,
        'Sacramento': 20000,
    }
    print("PRODUCTION MODE: Full iterations")

print(f"\nCalibration settings:")
print(f"  Objective: {OBJECTIVE.name}")
print(f"  Algorithm: SCE-UA")
print(f"\nIterations by model:")
for model_name, iters in ITERATIONS_BY_MODEL.items():
    print(f"  {model_name}: {iters:,}")

# %% [markdown]
# ---
# ## Run Calibrations
#
# Now we calibrate each model using SCE-UA. This allows us to compare
# model structures fairly by eliminating algorithm differences.

# %%
# Prepare calibration inputs
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

# Store results
calibration_results = {}
calibration_times = {}

# Define model colors for plotting
MODEL_COLORS = {
    'GR4J': '#1f77b4',      # Blue
    'GR5J': '#2ca02c',      # Green
    'GR6J': '#ff7f0e',      # Orange
    'Sacramento': '#d62728', # Red
}

# %%
# Calibrate GR4J
print("=" * 70)
print("CALIBRATING GR4J (4 parameters)")
print("=" * 70)

runner_gr4j = CalibrationRunner(
    model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=OBJECTIVE,
    warmup_period=WARMUP_DAYS
)

start_time = time.time()
result_gr4j = runner_gr4j.run_sceua(
    n_iterations=ITERATIONS_BY_MODEL['GR4J'],
    ngs=5,
    kstop=3,
    pcento=0.01,
    dbname='model_cmp_gr4j',
    dbformat='csv'
)
calibration_times['GR4J'] = time.time() - start_time
calibration_results['GR4J'] = result_gr4j

print(f"\n✓ GR4J calibration complete!")
print(f"  Best KGE: {result_gr4j.best_objective:.4f}")
print(f"  Runtime: {calibration_times['GR4J']:.1f} seconds")

# %%
# Calibrate GR5J
print("=" * 70)
print("CALIBRATING GR5J (5 parameters)")
print("=" * 70)

runner_gr5j = CalibrationRunner(
    model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=OBJECTIVE,
    warmup_period=WARMUP_DAYS
)

start_time = time.time()
result_gr5j = runner_gr5j.run_sceua(
    n_iterations=ITERATIONS_BY_MODEL['GR5J'],
    ngs=5,
    kstop=3,
    pcento=0.01,
    dbname='model_cmp_gr5j',
    dbformat='csv'
)
calibration_times['GR5J'] = time.time() - start_time
calibration_results['GR5J'] = result_gr5j

print(f"\n✓ GR5J calibration complete!")
print(f"  Best KGE: {result_gr5j.best_objective:.4f}")
print(f"  Runtime: {calibration_times['GR5J']:.1f} seconds")

# %%
# Calibrate GR6J
print("=" * 70)
print("CALIBRATING GR6J (6 parameters)")
print("=" * 70)

runner_gr6j = CalibrationRunner(
    model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=OBJECTIVE,
    warmup_period=WARMUP_DAYS
)

start_time = time.time()
result_gr6j = runner_gr6j.run_sceua(
    n_iterations=ITERATIONS_BY_MODEL['GR6J'],
    ngs=6,
    kstop=3,
    pcento=0.01,
    dbname='model_cmp_gr6j',
    dbformat='csv'
)
calibration_times['GR6J'] = time.time() - start_time
calibration_results['GR6J'] = result_gr6j

print(f"\n✓ GR6J calibration complete!")
print(f"  Best KGE: {result_gr6j.best_objective:.4f}")
print(f"  Runtime: {calibration_times['GR6J']:.1f} seconds")

# %%
# Calibrate Sacramento
print("=" * 70)
print("CALIBRATING SACRAMENTO (17+ parameters)")
print("=" * 70)

runner_sac = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=OBJECTIVE,
    warmup_period=WARMUP_DAYS
)

start_time = time.time()
result_sac = runner_sac.run_sceua(
    n_iterations=ITERATIONS_BY_MODEL['Sacramento'],
    ngs=8,  # More complexes for more parameters
    kstop=5,
    pcento=0.01,
    dbname='model_cmp_sacramento',
    dbformat='csv'
)
calibration_times['Sacramento'] = time.time() - start_time
calibration_results['Sacramento'] = result_sac

print(f"\n✓ Sacramento calibration complete!")
print(f"  Best KGE: {result_sac.best_objective:.4f}")
print(f"  Runtime: {calibration_times['Sacramento']:.1f} seconds")

# %%
# Summary of calibrations
print("\n" + "=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)
print(f"\n{'Model':<15} {'Parameters':>12} {'Best KGE':>12} {'Runtime (s)':>15}")
print("-" * 58)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    result = calibration_results[model_name]
    n_params = len(result.best_parameters)
    print(f"{model_name:<15} {n_params:>12} {result.best_objective:>12.4f} {calibration_times[model_name]:>15.1f}")

# %% [markdown]
# ---
# ## Generate Simulations
#
# Now we run each calibrated model on both calibration and validation periods
# to assess generalization performance.

# %%
# Generate simulations for calibration and validation periods
simulations_cal = {}
simulations_val = {}

for model_name, result in calibration_results.items():
    print(f"\nRunning {model_name} simulations...")
    
    # Create model instance
    if model_name == 'GR4J':
        model = GR4J(catchment_area_km2=CATCHMENT_AREA_KM2)
    elif model_name == 'GR5J':
        model = GR5J(catchment_area_km2=CATCHMENT_AREA_KM2)
    elif model_name == 'GR6J':
        model = GR6J(catchment_area_km2=CATCHMENT_AREA_KM2)
    else:
        model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    
    model.set_parameters(result.best_parameters)
    
    # Calibration period
    model.reset()
    cal_results = model.run(cal_data)
    simulations_cal[model_name] = cal_results['runoff'].values[WARMUP_DAYS:]
    
    # Validation period (needs warmup too)
    model.reset()
    val_results = model.run(val_data)
    simulations_val[model_name] = val_results['runoff'].values[WARMUP_DAYS:]

# Observed data
obs_cal = cal_observed[WARMUP_DAYS:]
obs_val = val_data['observed_flow'].values[WARMUP_DAYS:]
dates_cal = cal_data.index[WARMUP_DAYS:]
dates_val = val_data.index[WARMUP_DAYS:]

print("\nSimulations complete!")

# %% [markdown]
# ---
# ## Performance Comparison
#
# Let's calculate comprehensive metrics for both calibration and validation periods.

# %%
# Calculate metrics for all models
print("=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

metrics_cal = {}
metrics_val = {}

for model_name in calibration_results.keys():
    metrics_cal[model_name] = calculate_metrics(simulations_cal[model_name], obs_cal)
    metrics_val[model_name] = calculate_metrics(simulations_val[model_name], obs_val)

# Create comparison table
print("\n" + "-" * 80)
print("CALIBRATION PERIOD")
print("-" * 80)
print(f"{'Model':<15} {'NSE':>10} {'KGE':>10} {'RMSE':>12} {'PBIAS (%)':>12}")
print("-" * 60)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    m = metrics_cal[model_name]
    print(f"{model_name:<15} {m['NSE']:>10.4f} {m['KGE']:>10.4f} {m['RMSE']:>12.2f} {m['PBIAS']:>12.2f}")

print("\n" + "-" * 80)
print("VALIDATION PERIOD")
print("-" * 80)
print(f"{'Model':<15} {'NSE':>10} {'KGE':>10} {'RMSE':>12} {'PBIAS (%)':>12}")
print("-" * 60)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    m = metrics_val[model_name]
    print(f"{model_name:<15} {m['NSE']:>10.4f} {m['KGE']:>10.4f} {m['RMSE']:>12.2f} {m['PBIAS']:>12.2f}")

# %%
# Calculate performance drop from calibration to validation
print("\n" + "-" * 80)
print("GENERALIZATION (Cal → Val drop)")
print("-" * 80)
print(f"{'Model':<15} {'NSE drop':>12} {'KGE drop':>12} {'Params':>10}")
print("-" * 50)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    nse_drop = metrics_cal[model_name]['NSE'] - metrics_val[model_name]['NSE']
    kge_drop = metrics_cal[model_name]['KGE'] - metrics_val[model_name]['KGE']
    n_params = len(calibration_results[model_name].best_parameters)
    print(f"{model_name:<15} {nse_drop:>+12.4f} {kge_drop:>+12.4f} {n_params:>10}")

print("\nNote: Larger drops indicate potential overfitting")

# %% [markdown]
# ### Key Observations
#
# Look for these patterns in the results:
#
# 1. **Calibration performance often increases with complexity** - More parameters
#    can fit the calibration data better
#
# 2. **Validation drop may be larger for complex models** - This indicates
#    overfitting to the calibration period
#
# 3. **The "best" model depends on your priority**:
#    - If validation performance is paramount → prefer simpler models
#    - If you need detailed internal states → may need Sacramento
#    - If computational speed matters → prefer GR4J

# %% [markdown]
# ---
# ## Visualization

# %%
# Comprehensive comparison figure
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Calibration Hydrograph (Log Scale)',
        'Validation Hydrograph (Log Scale)',
        'NSE: Calibration vs Validation',
        'KGE: Calibration vs Validation',
        'Flow Duration Curves (Calibration)',
        'Flow Duration Curves (Validation)'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Calibration hydrograph
fig.add_trace(
    go.Scatter(x=dates_cal, y=obs_cal, name='Observed',
               line=dict(color='black', width=1.5)),
    row=1, col=1
)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    fig.add_trace(
        go.Scatter(x=dates_cal, y=simulations_cal[model_name], name=model_name,
                   line=dict(color=MODEL_COLORS[model_name], width=1), opacity=0.8),
        row=1, col=1
    )

# 2. Validation hydrograph
fig.add_trace(
    go.Scatter(x=dates_val, y=obs_val, name='Observed',
               line=dict(color='black', width=1.5), showlegend=False),
    row=1, col=2
)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    fig.add_trace(
        go.Scatter(x=dates_val, y=simulations_val[model_name], name=model_name,
                   line=dict(color=MODEL_COLORS[model_name], width=1), opacity=0.8,
                   showlegend=False),
        row=1, col=2
    )

# 3. NSE comparison
models_list = ['GR4J', 'GR5J', 'GR6J', 'Sacramento']
nse_cal = [metrics_cal[m]['NSE'] for m in models_list]
nse_val = [metrics_val[m]['NSE'] for m in models_list]
x_pos = np.arange(len(models_list))

fig.add_trace(
    go.Bar(x=models_list, y=nse_cal, name='Calibration', 
           marker_color='steelblue', opacity=0.8),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=models_list, y=nse_val, name='Validation',
           marker_color='coral', opacity=0.8),
    row=2, col=1
)

# 4. KGE comparison
kge_cal = [metrics_cal[m]['KGE'] for m in models_list]
kge_val = [metrics_val[m]['KGE'] for m in models_list]

fig.add_trace(
    go.Bar(x=models_list, y=kge_cal, name='Calibration',
           marker_color='steelblue', opacity=0.8, showlegend=False),
    row=2, col=2
)
fig.add_trace(
    go.Bar(x=models_list, y=kge_val, name='Validation',
           marker_color='coral', opacity=0.8, showlegend=False),
    row=2, col=2
)

# 5. FDC Calibration
exc = np.arange(1, len(obs_cal) + 1) / len(obs_cal) * 100
obs_sorted = np.sort(obs_cal)[::-1]
fig.add_trace(
    go.Scatter(x=exc, y=obs_sorted, name='Observed FDC',
               line=dict(color='black', width=2), showlegend=False),
    row=3, col=1
)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    sim_sorted = np.sort(simulations_cal[model_name])[::-1]
    fig.add_trace(
        go.Scatter(x=exc, y=sim_sorted, name=f'{model_name} FDC',
                   line=dict(color=MODEL_COLORS[model_name], width=1.5),
                   showlegend=False),
        row=3, col=1
    )

# 6. FDC Validation
exc_val = np.arange(1, len(obs_val) + 1) / len(obs_val) * 100
obs_sorted_val = np.sort(obs_val)[::-1]
fig.add_trace(
    go.Scatter(x=exc_val, y=obs_sorted_val, name='Observed FDC',
               line=dict(color='black', width=2), showlegend=False),
    row=3, col=2
)
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    sim_sorted = np.sort(simulations_val[model_name])[::-1]
    fig.add_trace(
        go.Scatter(x=exc_val, y=sim_sorted, name=f'{model_name} FDC',
                   line=dict(color=MODEL_COLORS[model_name], width=1.5),
                   showlegend=False),
        row=3, col=2
    )

# Update axes
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
fig.update_yaxes(title_text="NSE", row=2, col=1)
fig.update_yaxes(title_text="KGE", row=2, col=2)
fig.update_xaxes(title_text="Exceedance %", row=3, col=1)
fig.update_xaxes(title_text="Exceedance %", row=3, col=2)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=3, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=3, col=2)

fig.update_layout(
    title="<b>Model Comparison: GR4J vs GR5J vs GR6J vs Sacramento</b>",
    height=1000,
    showlegend=True,
    legend=dict(orientation='h', y=1.02),
    barmode='group'
)
fig.show()

# %%
# Save static comparison figure
fig_static, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Parameter count vs performance
ax = axes[0, 0]
params = [4, 5, 6, len(calibration_results['Sacramento'].best_parameters)]
kge_cal_vals = [metrics_cal[m]['KGE'] for m in models_list]
kge_val_vals = [metrics_val[m]['KGE'] for m in models_list]
ax.scatter(params, kge_cal_vals, s=100, c=[MODEL_COLORS[m] for m in models_list], 
           marker='o', edgecolors='black', label='Calibration')
ax.scatter(params, kge_val_vals, s=100, c=[MODEL_COLORS[m] for m in models_list],
           marker='s', edgecolors='black', label='Validation')
for i, m in enumerate(models_list):
    ax.annotate(m, (params[i], kge_cal_vals[i]), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('KGE')
ax.set_title('Parameter Count vs Performance')
ax.legend()

# 2. Calibration runtime
ax = axes[0, 1]
runtimes = [calibration_times[m] for m in models_list]
bars = ax.bar(models_list, runtimes, color=[MODEL_COLORS[m] for m in models_list])
ax.set_ylabel('Runtime (seconds)')
ax.set_title('Calibration Runtime')
ax.tick_params(axis='x', rotation=45)

# 3. Performance drop (generalization)
ax = axes[0, 2]
nse_drops = [metrics_cal[m]['NSE'] - metrics_val[m]['NSE'] for m in models_list]
kge_drops = [metrics_cal[m]['KGE'] - metrics_val[m]['KGE'] for m in models_list]
x = np.arange(len(models_list))
width = 0.35
ax.bar(x - width/2, nse_drops, width, label='NSE drop', color='steelblue')
ax.bar(x + width/2, kge_drops, width, label='KGE drop', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=45)
ax.set_ylabel('Performance Drop (Cal - Val)')
ax.set_title('Generalization Gap')
ax.legend()
ax.axhline(0, color='black', linestyle='--', alpha=0.5)

# 4. Hydrograph zoom (first year of calibration)
ax = axes[1, 0]
zoom_end = 365
ax.plot(dates_cal[:zoom_end], obs_cal[:zoom_end], 'k-', lw=1.5, label='Observed')
for model_name in models_list:
    ax.plot(dates_cal[:zoom_end], simulations_cal[model_name][:zoom_end],
            color=MODEL_COLORS[model_name], lw=1, label=model_name, alpha=0.8)
ax.set_ylabel('Flow (ML/day)')
ax.set_title('Calibration Period (First Year)')
ax.legend(fontsize=8)
ax.set_yscale('log')

# 5. Validation FDC
ax = axes[1, 1]
ax.plot(exc_val, obs_sorted_val, 'k-', lw=2, label='Observed')
for model_name in models_list:
    sim_sorted = np.sort(simulations_val[model_name])[::-1]
    ax.plot(exc_val, sim_sorted, color=MODEL_COLORS[model_name], lw=1.5, label=model_name)
ax.set_xlabel('Exceedance (%)')
ax.set_ylabel('Flow (ML/day)')
ax.set_title('Flow Duration Curves (Validation)')
ax.set_yscale('log')
ax.legend(fontsize=8)

# 6. Residuals distribution
ax = axes[1, 2]
for model_name in models_list:
    residuals = simulations_val[model_name] - obs_val
    ax.hist(residuals, bins=50, alpha=0.5, label=model_name,
            color=MODEL_COLORS[model_name], density=True)
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel('Residual (Sim - Obs) ML/day')
ax.set_ylabel('Density')
ax.set_title('Validation Residuals Distribution')
ax.legend(fontsize=8)
ax.set_xlim(-500, 500)

plt.suptitle('Model Comparison Summary', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/06_model_comparison.png")

# %% [markdown]
# ---
# ## Best Parameters for Each Model

# %%
print("=" * 70)
print("CALIBRATED PARAMETERS BY MODEL")
print("=" * 70)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    result = calibration_results[model_name]
    print(f"\n{model_name}")
    print("-" * 40)
    for param, value in result.best_parameters.items():
        print(f"  {param:12s}: {value:12.4f}")

# %% [markdown]
# ---
# ## Model Selection Guidance
#
# ### Decision Framework
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    MODEL SELECTION DECISION TREE                            │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# START: What is your primary need?
#   │
#   ├─► Quick regional assessment, limited data
#   │     └─► Use GR4J
#   │         • Fastest calibration
#   │         • Robust with limited data
#   │         • Good generalization
#   │
#   ├─► Need good low-flow simulation
#   │     └─► Use GR6J
#   │         • Exponential store improves recession
#   │         • Better baseflow representation
#   │
#   ├─► Significant groundwater interactions
#   │     └─► Use GR5J or Sacramento
#   │         • Exchange threshold helps
#   │         • Sacramento has explicit baseflow paths
#   │
#   ├─► Operational flood forecasting
#   │     └─► Use Sacramento
#   │         • Industry standard
#   │         • Detailed state tracking
#   │         • Compatible with NWS systems
#   │
#   └─► Research / need detailed soil moisture
#         └─► Use Sacramento
#             • Multi-zone representation
#             • Internal states available
#             • Process-based insights
# ```
#
# ### Summary Table
#
# | Criterion | GR4J | GR5J | GR6J | Sacramento |
# |-----------|------|------|------|------------|
# | **Simplicity** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
# | **Calibration speed** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
# | **Generalization** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
# | **Low flow simulation** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
# | **Internal states** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
# | **Physical realism** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

# %% [markdown]
# ---
# ## Summary
#
# ### Key Takeaways
#
# 1. **Model complexity does not guarantee better performance**
#    - Simple models often generalize better to validation data
#    - More parameters increase overfitting risk
#
# 2. **GR4J is remarkably effective for its simplicity**
#    - Only 4 parameters, yet competitive performance
#    - Excellent choice for data-limited situations
#
# 3. **Sacramento offers detailed process representation**
#    - More parameters but captures complex catchment behavior
#    - Industry standard for operational forecasting
#
# 4. **Choose based on application needs**
#    - General purpose / regional → GR4J or GR5J
#    - Low flows important → GR6J
#    - Operational / detailed states → Sacramento
#
# 5. **Always validate on independent data**
#    - Calibration performance can be misleading
#    - Validation reveals true model capability

# %%
print("=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)
print(f"""
Summary:
  Models compared: 4 (GR4J, GR5J, GR6J, Sacramento)
  Algorithm: SCE-UA
  Objective: KGE

Key results:
  - Calibration period: {CAL_START.date()} to {CAL_END.date()}
  - Validation period: {VAL_START.date()} to {VAL_END.date()}
  
Model complexity vs parsimony is a fundamental trade-off in hydrology.
Choose the simplest model that meets your application needs!
""")

# %% [markdown]
# ---
# ## Export Calibrated Parameters
#
# Save the calibrated parameters for future use.

# %%
# Export all calibrated parameters to CSV
export_data = []
for model_name, result in calibration_results.items():
    for param, value in result.best_parameters.items():
        export_data.append({
            'model': model_name,
            'parameter': param,
            'value': value
        })

params_df = pd.DataFrame(export_data)
params_df.to_csv('../test_data/model_comparison_params.csv', index=False)
print("Calibrated parameters saved to: test_data/model_comparison_params.csv")

# Also save performance summary
perf_data = []
for model_name in models_list:
    perf_data.append({
        'model': model_name,
        'n_params': len(calibration_results[model_name].best_parameters),
        'cal_nse': metrics_cal[model_name]['NSE'],
        'cal_kge': metrics_cal[model_name]['KGE'],
        'val_nse': metrics_val[model_name]['NSE'],
        'val_kge': metrics_val[model_name]['KGE'],
        'runtime_s': calibration_times[model_name],
    })

perf_df = pd.DataFrame(perf_data)
perf_df.to_csv('../test_data/model_comparison_performance.csv', index=False)
print("Performance summary saved to: test_data/model_comparison_performance.csv")
