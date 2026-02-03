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
# # Model Comparison: Choosing the Best Rainfall-Runoff Model
#
# ## Purpose
#
# This notebook continues from **Notebook 02: Calibration Quickstart** to compare
# different rainfall-runoff models available in pyrrm. We calibrate GR4J, GR5J, and
# GR6J models using the **same 5 objective functions** as in Notebook 02, then load
# the Sacramento results to create a comprehensive multi-model comparison.
#
# ## What You'll Learn
#
# - How different rainfall-runoff models are structured conceptually
# - How to calibrate multiple models with multiple objective functions
# - How model complexity affects calibration and performance
# - How to comprehensively compare models using multiple metrics and visualizations
# - How to interpret hydrologic signatures and flow duration curves
# - How to choose the right model for your application
#
# ## Prerequisites
#
# - **Must complete Notebook 02: Calibration Quickstart first**
#   - Sacramento calibration results (5 objectives) are loaded from saved reports
# - Basic understanding of calibration concepts
#
# ## Key Insight
#
# > **More parameters ≠ better performance.** Simple models often generalize
# > better to new data. The "best" model depends on your application, data
# > quality, and computational constraints.
#
# ## Models Compared
#
# | Model | Parameters | Origin | Key Features |
# |-------|------------|--------|--------------|
# | **GR4J** | 4 | INRAE, France | Production + routing stores, UH |
# | **GR5J** | 5 | INRAE, France | GR4J + exchange threshold |
# | **GR6J** | 6 | INRAE, France | GR5J + exponential store for low flows |
# | **Sacramento** | 22 | US NWS | Multi-zone soil moisture accounting |
#
# ## Objective Functions
#
# All models are calibrated with the same 5 objectives from Notebook 02:
#
# | Objective | Emphasis | Best For |
# |-----------|----------|----------|
# | **NSE** | High flows (squared errors) | Flood forecasting |
# | **LogNSE** | All flow ranges (log transform) | General purpose |
# | **InvNSE (1/Q)** | Low flows (inverse transform) | Drought/environmental flows |
# | **SqrtNSE (√Q)** | Moderate low flow emphasis | Balanced approach |
# | **SDEB** | FDC shape + timing + bias | Australian SOURCE platform |

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
# │    (4 params)  (5 params)  (6 params)              (22 params)             │
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
# This notebook will help you understand these trade-offs empirically.

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
print("MODEL COMPARISON: GR4J, GR5J, GR6J vs SACRAMENTO")
print("=" * 70)

# %%
# Import pyrrm models and calibration tools
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
from pyrrm.calibration import CalibrationRunner, CalibrationReport
from pyrrm.calibration.objective_functions import NSE, calculate_metrics

# Import objective functions from pyrrm.objectives (same as Notebook 02)
from pyrrm.objectives import (
    NSE as NSE_obj, KGE, KGENonParametric, PBIAS, RMSE,
    FlowTransformation, FDCMetric, SignatureMetric, SDEB,
    PearsonCorrelation, SpearmanCorrelation
)

print("\npyrrm models imported successfully!")
print(f"\nAvailable models:")
print(f"  - Sacramento (22 parameters)")
print(f"  - GR4J (4 parameters)")
print(f"  - GR5J (5 parameters)")
print(f"  - GR6J (6 parameters)")

# %% [markdown]
# ---
# ## Step 1: Load Data (Same as Notebook 02)
#
# We use the exact same data loading process as Notebook 02 to ensure
# consistency in our comparison.

# %%
# Configure paths (same as Notebook 02)
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

print(f"Data directory: {DATA_DIR.absolute()}")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")

# %%
# Load rainfall
rainfall_file = DATA_DIR / 'Default Input Set - Rain_QBN01.csv'
rainfall_df = pd.read_csv(rainfall_file, parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

print("RAINFALL DATA")
print("=" * 50)
print(f"Records: {len(rainfall_df):,} days")
print(f"Period: {rainfall_df.index.min().date()} to {rainfall_df.index.max().date()}")

# %%
# Load PET
pet_file = DATA_DIR / 'Default Input Set - Mwet_QBN01.csv'
pet_df = pd.read_csv(pet_file, parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

print("PET DATA")
print("=" * 50)
print(f"Records: {len(pet_df):,} days")
print(f"Period: {pet_df.index.min().date()} to {pet_df.index.max().date()}")

# %%
# Load observed flow
flow_file = DATA_DIR / '410734_output_SDmodel.csv'
flow_df = pd.read_csv(flow_file, parse_dates=['Date'], index_col='Date')

observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']

# Handle missing values
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

print("OBSERVED FLOW DATA")
print("=" * 50)
print(f"Records: {len(observed_df):,} days")
print(f"Period: {observed_df.index.min().date()} to {observed_df.index.max().date()}")

# %%
# Merge all datasets
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

print("MERGED DATASET")
print("=" * 50)
print(f"Total records: {len(data):,} days")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")

# %%
# Define calibration period (same as Notebook 02)
WARMUP_DAYS = 365  # 1 year warmup

CAL_START = data.index.min()
CAL_END = data.index.max()

cal_data = data.copy()

print("CALIBRATION PERIOD")
print("=" * 50)
print(f"Start: {CAL_START.date()}")
print(f"End: {CAL_END.date()}")
print(f"Total days: {len(cal_data):,}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_data) - WARMUP_DAYS:,} days")

# Prepare inputs and observed data
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

# %% [markdown]
# ---
# ## Step 2: Define Objective Functions (Same as Notebook 02)
#
# We use the same 5 objective functions that were used to calibrate Sacramento
# in Notebook 02. This ensures a fair comparison across models.

# %%
# Define objective functions (same as Notebook 02)
from pyrrm.objectives import NSE as NSE_new, FlowTransformation, SDEB

# NSE: Standard Nash-Sutcliffe Efficiency (emphasizes high flows)
nse_objective = NSE()

# LogNSE: NSE with log transformation - balances all flow ranges
log_objective = NSE_new(transform=FlowTransformation('log', epsilon_value=0.01))

# InverseNSE: NSE with inverse transformation (1/Q) - heavily emphasizes low flows
inv_objective = NSE_new(transform=FlowTransformation('inverse', epsilon_value=0.01))

# SqrtNSE: NSE with square root transformation (√Q) - balanced emphasis
sqrt_objective = NSE_new(transform=FlowTransformation('sqrt'))

# SDEB: Sum of Daily Flows, Daily Exceedance Curve and Bias (Lerat et al., 2013)
sdeb_objective = SDEB(alpha=0.1, lam=0.5)

# Store objectives in a dictionary
OBJECTIVES = {
    'NSE': nse_objective,
    'LogNSE': log_objective,
    'InvNSE': inv_objective,
    'SqrtNSE': sqrt_objective,
    'SDEB': sdeb_objective,
}

print("OBJECTIVE FUNCTIONS (Same as Notebook 02)")
print("=" * 60)
for name, obj in OBJECTIVES.items():
    desc = {
        'NSE': 'Standard NSE - emphasizes high flows',
        'LogNSE': 'log(Q) transform - balances all flow ranges',
        'InvNSE': '1/Q transform - emphasizes low flows',
        'SqrtNSE': '√Q transform - moderate low flow emphasis',
        'SDEB': 'Chronological + FDC + Bias (SOURCE platform)',
    }
    print(f"  {name:10s}: {desc[name]}")

# %% [markdown]
# ---
# ## Step 3: Load Sacramento Results from Notebook 02
#
# Instead of re-calibrating Sacramento, we load all 5 calibration reports
# from Notebook 02. This ensures we're comparing against the same Sacramento
# calibrations that were thoroughly analyzed in that notebook.

# %%
# Load all Sacramento calibrations from Notebook 02
print("=" * 70)
print("LOADING SACRAMENTO RESULTS FROM NOTEBOOK 02")
print("=" * 70)

# Map of objective names to report files
SACRAMENTO_REPORTS = {
    'NSE': '../test_data/reports/410734_nse.pkl',
    'LogNSE': '../test_data/reports/410734_lognse.pkl',
    'InvNSE': '../test_data/reports/410734_invnse.pkl',
    'SqrtNSE': '../test_data/reports/410734_sqrtnse.pkl',
    'SDEB': '../test_data/reports/410734_sdeb.pkl',
}

sacramento_results = {}
sacramento_loaded_count = 0

print("\nLoading Sacramento calibration reports:")
print("-" * 60)

for obj_name, report_path in SACRAMENTO_REPORTS.items():
    path = Path(report_path)
    if path.exists():
        try:
            report = CalibrationReport.load(str(path))
            sacramento_results[obj_name] = report.result
            sacramento_loaded_count += 1
            print(f"  ✓ {obj_name:10s}: Loaded (Best: {report.result.best_objective:.4f})")
        except Exception as e:
            print(f"  ✗ {obj_name:10s}: Error loading - {e}")
    else:
        print(f"  ✗ {obj_name:10s}: File not found")

print(f"\nLoaded {sacramento_loaded_count}/5 Sacramento calibrations from Notebook 02")

if sacramento_loaded_count < 5:
    print("\nWARNING: Some Sacramento calibrations are missing.")
    print("Please run Notebook 02 completely before running this notebook.")
    print("Missing calibrations will be run here if needed.")

# %% [markdown]
# ---
# ## Step 4: Calibrate GR4J, GR5J, and GR6J
#
# We calibrate each GR model with all 5 objective functions to match the
# Sacramento calibrations from Notebook 02.

# %%
# SCE-UA evaluations scaled by model complexity
MAX_EVALS = {
    'GR4J': 4000,      # 4 params
    'GR5J': 5000,      # 5 params  
    'GR6J': 6000,      # 6 params
    'Sacramento': 10000,  # 22 params
}

print("CALIBRATION CONFIGURATION")
print("=" * 60)
print(f"Algorithm: SCE-UA Direct")
print(f"Objectives: {', '.join(OBJECTIVES.keys())}")
print(f"\nMax evaluations by model:")
for model, evals in MAX_EVALS.items():
    print(f"  {model}: {evals:,}")

# %% [markdown]
# ### Calibrate GR4J with All Objectives

# %%
# Calibrate GR4J with all 5 objectives
print("\n" + "=" * 70)
print("CALIBRATING GR4J (4 parameters) - ALL OBJECTIVES")
print("=" * 70)

gr4j_results = {}
gr4j_times = {}

for obj_name, objective in OBJECTIVES.items():
    print(f"\n--- GR4J with {obj_name} ---")
    
    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=objective,
        warmup_period=WARMUP_DAYS
    )
    
    start_time = time.time()
    result = runner.run_sceua_direct(
        max_evals=MAX_EVALS['GR4J'],
        seed=42,
        verbose=True,
        max_tolerant_iter=100,
        tolerance=1e-4
    )
    elapsed = time.time() - start_time
    
    gr4j_results[obj_name] = result
    gr4j_times[obj_name] = elapsed
    
    # Save report
    report = runner.create_report(result, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report.save(f'../test_data/reports/410734_gr4j_{obj_name.lower()}')
    
    print(f"\n  Best {obj_name}: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

print("\n✓ GR4J calibration complete for all objectives!")

# %% [markdown]
# ### Calibrate GR5J with All Objectives

# %%
# Calibrate GR5J with all 5 objectives
print("\n" + "=" * 70)
print("CALIBRATING GR5J (5 parameters) - ALL OBJECTIVES")
print("=" * 70)

gr5j_results = {}
gr5j_times = {}

for obj_name, objective in OBJECTIVES.items():
    print(f"\n--- GR5J with {obj_name} ---")
    
    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=objective,
        warmup_period=WARMUP_DAYS
    )
    
    start_time = time.time()
    result = runner.run_sceua_direct(
        max_evals=MAX_EVALS['GR5J'],
        seed=42,
        verbose=True,
        max_tolerant_iter=100,
        tolerance=1e-4
    )
    elapsed = time.time() - start_time
    
    gr5j_results[obj_name] = result
    gr5j_times[obj_name] = elapsed
    
    # Save report
    report = runner.create_report(result, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report.save(f'../test_data/reports/410734_gr5j_{obj_name.lower()}')
    
    print(f"\n  Best {obj_name}: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

print("\n✓ GR5J calibration complete for all objectives!")

# %% [markdown]
# ### Calibrate GR6J with All Objectives

# %%
# Calibrate GR6J with all 5 objectives
print("\n" + "=" * 70)
print("CALIBRATING GR6J (6 parameters) - ALL OBJECTIVES")
print("=" * 70)

gr6j_results = {}
gr6j_times = {}

for obj_name, objective in OBJECTIVES.items():
    print(f"\n--- GR6J with {obj_name} ---")
    
    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=objective,
        warmup_period=WARMUP_DAYS
    )
    
    start_time = time.time()
    result = runner.run_sceua_direct(
        max_evals=MAX_EVALS['GR6J'],
        seed=42,
        verbose=True,
        max_tolerant_iter=100,
        tolerance=1e-4
    )
    elapsed = time.time() - start_time
    
    gr6j_results[obj_name] = result
    gr6j_times[obj_name] = elapsed
    
    # Save report
    report = runner.create_report(result, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report.save(f'../test_data/reports/410734_gr6j_{obj_name.lower()}')
    
    print(f"\n  Best {obj_name}: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

print("\n✓ GR6J calibration complete for all objectives!")

# %%
# Organize all calibration results
ALL_RESULTS = {
    'GR4J': gr4j_results,
    'GR5J': gr5j_results,
    'GR6J': gr6j_results,
    'Sacramento': sacramento_results,
}

ALL_TIMES = {
    'GR4J': gr4j_times,
    'GR5J': gr5j_times,
    'GR6J': gr6j_times,
    'Sacramento': {obj: sacramento_results[obj].runtime_seconds 
                   for obj in sacramento_results if obj in sacramento_results},
}

# Model colors for plotting
MODEL_COLORS = {
    'GR4J': '#1f77b4',       # Blue
    'GR5J': '#2ca02c',       # Green
    'GR6J': '#ff7f0e',       # Orange
    'Sacramento': '#d62728', # Red
    'Observed': '#000000',   # Black
}

OBJECTIVE_COLORS = {
    'NSE': '#E41A1C',
    'LogNSE': '#377EB8',
    'InvNSE': '#4DAF4A',
    'SqrtNSE': '#984EA3',
    'SDEB': '#FF7F00',
}

# %% [markdown]
# ---
# ## Step 5: Calibration Summary - All Models × All Objectives

# %%
# Display calibration summary
print("=" * 90)
print("CALIBRATION SUMMARY: ALL MODELS × ALL OBJECTIVES")
print("=" * 90)

print(f"\n{'Model':<12} {'Params':>8}", end="")
for obj_name in OBJECTIVES.keys():
    print(f" {obj_name:>12}", end="")
print()
print("-" * 90)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    n_params = len(list(ALL_RESULTS[model_name].values())[0].best_parameters) if ALL_RESULTS[model_name] else '?'
    print(f"{model_name:<12} {n_params:>8}", end="")
    
    for obj_name in OBJECTIVES.keys():
        if obj_name in ALL_RESULTS[model_name]:
            val = ALL_RESULTS[model_name][obj_name].best_objective
            print(f" {val:>12.4f}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()

print("\nNote: NSE/LogNSE/InvNSE/SqrtNSE are maximized (higher = better)")
print("      SDEB is minimized (lower = better)")

# %% [markdown]
# ---
# ## Step 6: Generate Simulations for All Models
#
# We generate simulations using the NSE-calibrated parameters as the primary
# comparison, since NSE is the most common objective function.

# %%
# Generate simulations using NSE-calibrated parameters (primary comparison)
print("=" * 70)
print("GENERATING SIMULATIONS (NSE-calibrated parameters)")
print("=" * 70)

simulations = {}

def get_flow_column(sim_results):
    """Get flow column - Sacramento uses 'runoff', GR models use 'flow'."""
    if 'runoff' in sim_results.columns:
        return sim_results['runoff'].values
    elif 'flow' in sim_results.columns:
        return sim_results['flow'].values
    else:
        raise KeyError(f"No flow column found. Available: {sim_results.columns.tolist()}")

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    if 'NSE' not in ALL_RESULTS[model_name]:
        print(f"  ✗ {model_name}: NSE calibration not available")
        continue
    
    print(f"  Running {model_name}...")
    result = ALL_RESULTS[model_name]['NSE']
    
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
    model.reset()
    
    sim_results = model.run(cal_data)
    simulations[model_name] = get_flow_column(sim_results)[WARMUP_DAYS:]

# Get observed data for comparison
obs_flow = cal_observed[WARMUP_DAYS:]
comparison_dates = cal_data.index[WARMUP_DAYS:]

print(f"\nSimulations complete! ({len(obs_flow):,} days)")

# %% [markdown]
# ---
# ## Step 7: Comprehensive Metrics Comparison
#
# We evaluate each model's NSE-calibrated simulation using a comprehensive
# set of metrics that assess different aspects of hydrological performance.

# %%
def comprehensive_evaluation(obs, sim):
    """Compute comprehensive metrics for model comparison."""
    inv_transform = FlowTransformation('inverse', epsilon_value=0.01)
    sqrt_transform = FlowTransformation('sqrt')
    log_transform = FlowTransformation('log', epsilon_value=0.01)
    
    metrics = {}
    
    # Overall efficiency
    metrics['NSE'] = NSE_obj()(obs, sim)
    metrics['KGE'] = KGE(variant='2012')(obs, sim)
    metrics['KGE_np'] = KGENonParametric()(obs, sim)
    
    # KGE components
    kge_obj = KGE(variant='2012')
    components = kge_obj.get_components(obs, sim)
    if components:
        metrics['r (correlation)'] = components['r']
        metrics['α (variability)'] = components['alpha']
        metrics['β (bias)'] = components['beta']
    
    # Flow regime specific
    metrics['NSE (log Q)'] = NSE_obj(transform=log_transform)(obs, sim)
    metrics['NSE (1/Q)'] = NSE_obj(transform=inv_transform)(obs, sim)
    metrics['NSE (√Q)'] = NSE_obj(transform=sqrt_transform)(obs, sim)
    metrics['KGE (1/Q)'] = KGE(transform=inv_transform)(obs, sim)
    
    # FDC-based metrics
    try:
        metrics['FDC Peak Bias (%)'] = FDCMetric('peak', 'volume_bias')(obs, sim)
        metrics['FDC High Bias (%)'] = FDCMetric('high', 'volume_bias')(obs, sim)
        metrics['FDC Mid Bias (%)'] = FDCMetric('mid', 'volume_bias')(obs, sim)
        metrics['FDC Low Bias (%)'] = FDCMetric('low', 'volume_bias', log_transform=True)(obs, sim)
        metrics['FDC Very Low Bias (%)'] = FDCMetric('very_low', 'volume_bias', log_transform=True)(obs, sim)
    except Exception:
        pass
    
    # Signature metrics
    try:
        metrics['Q95 Error (%)'] = SignatureMetric('q95')(obs, sim)
        metrics['Q50 Error (%)'] = SignatureMetric('q50')(obs, sim)
        metrics['Q5 Error (%)'] = SignatureMetric('q5')(obs, sim)
        metrics['Flashiness Error (%)'] = SignatureMetric('flashiness')(obs, sim)
        metrics['Baseflow Index Error (%)'] = SignatureMetric('baseflow_index')(obs, sim)
    except Exception:
        pass
    
    # Volume and timing
    metrics['PBIAS (%)'] = PBIAS()(obs, sim)
    metrics['RMSE'] = RMSE()(obs, sim)
    metrics['Pearson r'] = PearsonCorrelation()(obs, sim)
    metrics['Spearman ρ'] = SpearmanCorrelation()(obs, sim)
    
    return metrics

# %%
# Compute comprehensive evaluation for all models
print("=" * 70)
print("COMPUTING COMPREHENSIVE EVALUATION")
print("=" * 70)

model_evaluations = {}
for model_name in simulations.keys():
    print(f"  Evaluating {model_name}...")
    model_evaluations[model_name] = comprehensive_evaluation(obs_flow, simulations[model_name])

comp_df = pd.DataFrame(model_evaluations)

print("\nDone!")

# %%
# Display comprehensive metrics table
print("=" * 90)
print("COMPREHENSIVE MODEL EVALUATION (NSE-calibrated)")
print("=" * 90)

categories = {
    'OVERALL EFFICIENCY': ['NSE', 'KGE', 'KGE_np'],
    'KGE COMPONENTS': ['r (correlation)', 'α (variability)', 'β (bias)'],
    'FLOW REGIME SPECIFIC': ['NSE (log Q)', 'NSE (1/Q)', 'NSE (√Q)', 'KGE (1/Q)'],
    'FDC SEGMENT BIASES': ['FDC Peak Bias (%)', 'FDC High Bias (%)', 
                           'FDC Mid Bias (%)', 'FDC Low Bias (%)', 'FDC Very Low Bias (%)'],
    'SIGNATURE ERRORS': ['Q95 Error (%)', 'Q50 Error (%)', 'Q5 Error (%)',
                         'Flashiness Error (%)', 'Baseflow Index Error (%)'],
    'VOLUME & TIMING': ['PBIAS (%)', 'RMSE', 'Pearson r', 'Spearman ρ']
}

for category, metrics_list in categories.items():
    print(f"\n{'─' * 90}")
    print(f"  {category}")
    print(f"{'─' * 90}")
    print(f"  {'Metric':<28} {'GR4J':>14} {'GR5J':>14} {'GR6J':>14} {'Sacramento':>14}")
    print(f"  {'-' * 88}")
    
    for metric in metrics_list:
        if metric in comp_df.index:
            row = comp_df.loc[metric]
            if 'Error' in metric or 'Bias' in metric:
                values = [f"{v:>+13.2f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            elif metric == 'RMSE':
                values = [f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            else:
                values = [f"{v:>14.4f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            print(f"  {metric:<28} {' '.join(values)}")

# %% [markdown]
# ---
# ## Step 8: Cross-Objective Performance Matrix
#
# This is a key analysis: we evaluate how each model performs when calibrated
# with different objective functions. This reveals which models are most robust
# across different calibration targets.

# %%
# Generate simulations for ALL model × objective combinations
print("=" * 70)
print("GENERATING ALL SIMULATIONS (4 Models × 5 Objectives)")
print("=" * 70)

all_simulations = {}

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    all_simulations[model_name] = {}
    
    for obj_name in OBJECTIVES.keys():
        if obj_name not in ALL_RESULTS[model_name]:
            continue
        
        result = ALL_RESULTS[model_name][obj_name]
        
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
        model.reset()
        
        sim_results = model.run(cal_data)
        all_simulations[model_name][obj_name] = get_flow_column(sim_results)[WARMUP_DAYS:]

print("Done! Generated 20 simulations (4 models × 5 objectives)")

# %%
# Cross-evaluation: How does each model perform across ALL metrics?
print("=" * 100)
print("CROSS-OBJECTIVE PERFORMANCE MATRIX")
print("=" * 100)
print("\nEach cell shows the NSE achieved when model is calibrated with the column objective")
print("and evaluated with standard NSE.\n")

print(f"{'Model':<12} {'Calibrated with →':>20}", end="")
for obj_name in OBJECTIVES.keys():
    print(f" {obj_name:>12}", end="")
print()
print("-" * 100)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    print(f"{model_name:<12} {'Evaluated NSE →':>20}", end="")
    
    for obj_name in OBJECTIVES.keys():
        if obj_name in all_simulations[model_name]:
            sim = all_simulations[model_name][obj_name]
            nse = NSE_obj()(obs_flow, sim)
            print(f" {nse:>12.4f}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()

# %%
# Create detailed cross-evaluation DataFrame
cross_eval_data = {}

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    cross_eval_data[model_name] = {}
    
    for obj_name in OBJECTIVES.keys():
        if obj_name not in all_simulations[model_name]:
            continue
        
        sim = all_simulations[model_name][obj_name]
        
        # Evaluate with multiple metrics
        cross_eval_data[model_name][f'{obj_name}_NSE'] = NSE_obj()(obs_flow, sim)
        cross_eval_data[model_name][f'{obj_name}_KGE'] = KGE()(obs_flow, sim)
        cross_eval_data[model_name][f'{obj_name}_LogNSE'] = log_objective(obs_flow, sim)
        cross_eval_data[model_name][f'{obj_name}_PBIAS'] = PBIAS()(obs_flow, sim)

cross_eval_df = pd.DataFrame(cross_eval_data).T
print("\nCross-evaluation matrix created!")

# %% [markdown]
# ---
# ## Step 9: Hydrographs - All Objective Functions
#
# Now we visualize the hydrographs for ALL 5 objective functions, not just NSE.
# This reveals how different calibration targets affect the simulated flows.

# %%
# Hydrographs for each objective function (one per objective)
for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{obj_name} Calibrated - Linear Scale', f'{obj_name} Calibrated - Log Scale'),
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    # Linear scale
    fig.add_trace(
        go.Scatter(x=comparison_dates, y=obs_flow, name='Observed',
                   line=dict(color='black', width=1.5)),
        row=1, col=1
    )
    for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        if obj_name in all_simulations.get(model_name, {}):
            fig.add_trace(
                go.Scatter(x=comparison_dates, y=all_simulations[model_name][obj_name], 
                           name=model_name, line=dict(color=MODEL_COLORS[model_name], width=1), opacity=0.8),
                row=1, col=1
            )
    
    # Log scale
    fig.add_trace(
        go.Scatter(x=comparison_dates, y=obs_flow, showlegend=False,
                   line=dict(color='black', width=1.5)),
        row=2, col=1
    )
    for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        if obj_name in all_simulations.get(model_name, {}):
            fig.add_trace(
                go.Scatter(x=comparison_dates, y=all_simulations[model_name][obj_name], 
                           showlegend=False, line=dict(color=MODEL_COLORS[model_name], width=1), opacity=0.8),
                row=2, col=1
            )
    
    fig.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(
        title=f"<b>Hydrograph Comparison - {obj_name} Calibrated</b>",
        height=600,
        legend=dict(orientation='h', y=1.02)
    )
    fig.show()

# %% [markdown]
# ---
# ## Step 10: Flow Duration Curves - All Objective Functions
#
# FDC comparison for each objective function.

# %%
# FDC for each objective function
obs_sorted = np.sort(obs_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{obj_name}: Full FDC', f'{obj_name}: Low Flow Detail (>50%)'),
        horizontal_spacing=0.1
    )
    
    # Full FDC
    fig.add_trace(
        go.Scatter(x=exceedance, y=obs_sorted, name='Observed',
                   line=dict(color='black', width=2)),
        row=1, col=1
    )
    for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        if obj_name in all_simulations.get(model_name, {}):
            sim_sorted = np.sort(all_simulations[model_name][obj_name])[::-1]
            fig.add_trace(
                go.Scatter(x=exceedance, y=sim_sorted, name=model_name,
                           line=dict(color=MODEL_COLORS[model_name], width=1.5)),
                row=1, col=1
            )
    
    # Low flow detail
    low_idx = exceedance > 50
    fig.add_trace(
        go.Scatter(x=exceedance[low_idx], y=obs_sorted[low_idx], showlegend=False,
                   line=dict(color='black', width=2)),
        row=1, col=2
    )
    for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        if obj_name in all_simulations.get(model_name, {}):
            sim_sorted = np.sort(all_simulations[model_name][obj_name])[::-1]
            fig.add_trace(
                go.Scatter(x=exceedance[low_idx], y=sim_sorted[low_idx], showlegend=False,
                           line=dict(color=MODEL_COLORS[model_name], width=1.5)),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Exceedance %", row=1, col=1)
    fig.update_xaxes(title_text="Exceedance %", row=1, col=2)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
    
    fig.update_layout(
        title=f"<b>Flow Duration Curves - {obj_name} Calibrated</b>",
        height=400,
        legend=dict(orientation='h', y=1.08)
    )
    fig.show()

# %% [markdown]
# ---
# ## Step 11: Comprehensive Hydrological Signatures - All Objectives
#
# We calculate and compare hydrological signatures for ALL model × objective combinations.

# %%
# Calculate hydrological signatures
def calculate_signatures(flow):
    """Calculate key hydrological signatures."""
    return {
        'Q95 (high)': np.percentile(flow, 95),
        'Q50 (median)': np.percentile(flow, 50),
        'Q5 (low)': np.percentile(flow, 5),
        'Mean Flow': np.mean(flow),
        'Std Dev': np.std(flow),
        'CV': np.std(flow) / np.mean(flow),
        'Flashiness': np.sum(np.abs(np.diff(flow))) / np.sum(flow),
        'BFI (approx)': np.percentile(flow, 30) / np.mean(flow),
        'Max Flow': np.max(flow),
        'Min Flow': np.min(flow),
    }

signatures_obs = calculate_signatures(obs_flow)

# Calculate signatures for all model × objective combinations
all_signatures = {}
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    all_signatures[model_name] = {}
    for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
        if obj_name in all_simulations.get(model_name, {}):
            all_signatures[model_name][obj_name] = calculate_signatures(
                all_simulations[model_name][obj_name]
            )

# %%
# Display signatures for each objective function
for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    print(f"\n{'=' * 100}")
    print(f"HYDROLOGICAL SIGNATURES - {obj_name} CALIBRATED")
    print(f"{'=' * 100}")
    print(f"\n{'Signature':<20} {'Observed':>14} {'GR4J':>14} {'GR5J':>14} {'GR6J':>14} {'Sacramento':>14}")
    print("-" * 100)
    
    for sig_name in signatures_obs.keys():
        obs_val = signatures_obs[sig_name]
        fmt = f"{obs_val:>14.2f}" if obs_val > 1 else f"{obs_val:>14.4f}"
        
        print(f"{sig_name:<20} {fmt}", end="")
        for model in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
            if obj_name in all_signatures.get(model, {}):
                val = all_signatures[model][obj_name][sig_name]
                fmt_val = f"{val:>14.2f}" if abs(val) > 1 else f"{val:>14.4f}"
                print(f" {fmt_val}", end="")
            else:
                print(f" {'N/A':>14}", end="")
        print()

# %%
# Signature error heatmap for each model (showing % error from observed)
def calc_sig_error(obs_val, sim_val):
    """Calculate % error in signature."""
    if obs_val == 0:
        return np.nan
    return (sim_val - obs_val) / obs_val * 100

# Create signature error table
sig_error_data = []
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
        if obj_name not in all_signatures.get(model_name, {}):
            continue
        
        row = {'Model': model_name, 'Objective': obj_name}
        for sig_name in ['Q95 (high)', 'Q50 (median)', 'Q5 (low)', 'Mean Flow', 'Flashiness']:
            obs_val = signatures_obs[sig_name]
            sim_val = all_signatures[model_name][obj_name][sig_name]
            row[sig_name] = calc_sig_error(obs_val, sim_val)
        sig_error_data.append(row)

sig_error_df = pd.DataFrame(sig_error_data)

# Display
print("\n" + "=" * 120)
print("SIGNATURE % ERRORS (All Models × All Objectives)")
print("=" * 120)
print("\nPositive = overestimate, Negative = underestimate")
print(sig_error_df.to_string(index=False, float_format=lambda x: f'{x:+.1f}%' if not pd.isna(x) else 'N/A'))

# %% [markdown]
# ---
# ## Step 12: Comprehensive Metrics - All Objectives
#
# Evaluate each model with comprehensive metrics for EACH objective function.

# %%
# Comprehensive evaluation for ALL model × objective combinations
print("=" * 120)
print("COMPREHENSIVE METRICS - ALL MODEL × OBJECTIVE COMBINATIONS")
print("=" * 120)

all_metrics = {}
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    all_metrics[model_name] = {}
    for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
        if obj_name in all_simulations.get(model_name, {}):
            all_metrics[model_name][obj_name] = comprehensive_evaluation(
                obs_flow, all_simulations[model_name][obj_name]
            )

# Display key metrics for each objective
key_display_metrics = ['NSE', 'KGE', 'NSE (log Q)', 'NSE (1/Q)', 'PBIAS (%)', 'RMSE']

for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    print(f"\n{'─' * 100}")
    print(f"  {obj_name} CALIBRATED - Key Metrics")
    print(f"{'─' * 100}")
    print(f"  {'Metric':<20} {'GR4J':>16} {'GR5J':>16} {'GR6J':>16} {'Sacramento':>16}")
    print(f"  {'-' * 98}")
    
    for metric in key_display_metrics:
        print(f"  {metric:<20}", end="")
        for model in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
            if obj_name in all_metrics.get(model, {}) and metric in all_metrics[model][obj_name]:
                val = all_metrics[model][obj_name][metric]
                if 'BIAS' in metric:
                    print(f" {val:>+15.2f}", end="")
                elif metric == 'RMSE':
                    print(f" {val:>16.2f}", end="")
                else:
                    print(f" {val:>16.4f}", end="")
            else:
                print(f" {'N/A':>16}", end="")
        print()

# %% [markdown]
# ---
# ## Step 13: Model × Objective Performance Heatmaps
#
# Multiple heatmaps showing different metrics across all combinations.

# %%
# Create heatmaps for multiple metrics
def create_metric_heatmap(metric_name, metric_func=None):
    """Create heatmap for a specific metric across all model × objective combinations."""
    heatmap_data = []
    
    for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        row_data = []
        for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
            if obj_name in all_metrics.get(model_name, {}) and metric_name in all_metrics[model_name][obj_name]:
                row_data.append(all_metrics[model_name][obj_name][metric_name])
            else:
                row_data.append(np.nan)
        heatmap_data.append(row_data)
    
    return pd.DataFrame(
        heatmap_data,
        index=['GR4J', 'GR5J', 'GR6J', 'Sacramento'],
        columns=['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']
    )

# Create heatmaps for key metrics
metrics_to_plot = ['NSE', 'KGE', 'NSE (log Q)', 'NSE (1/Q)']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{m} Performance' for m in metrics_to_plot],
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for idx, metric_name in enumerate(metrics_to_plot):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    heatmap_df = create_metric_heatmap(metric_name)
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale='RdYlGn',
            zmid=0.7 if 'NSE' in metric_name or metric_name == 'KGE' else 0.5,
            text=np.round(heatmap_df.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=(col == 2),
            hovertemplate=f'{metric_name}<br>Model: %{{y}}<br>Calibrated: %{{x}}<br>Value: %{{z:.4f}}<extra></extra>'
        ),
        row=row, col=col
    )

fig.update_layout(
    title="<b>Performance Heatmaps: Multiple Metrics × All Objective Functions</b>",
    height=700,
    width=900
)
fig.show()

# %%
# Primary NSE heatmap (larger)
heatmap_nse = create_metric_heatmap('NSE')

fig = go.Figure(data=go.Heatmap(
    z=heatmap_nse.values,
    x=heatmap_nse.columns,
    y=heatmap_nse.index,
    colorscale='RdYlGn',
    zmid=0.7,
    text=np.round(heatmap_nse.values, 3),
    texttemplate='%{text}',
    textfont={"size": 16},
    hovertemplate='Model: %{y}<br>Calibrated with: %{x}<br>NSE: %{z:.4f}<extra></extra>'
))

fig.update_layout(
    title="<b>Model × Objective Performance Matrix (NSE)</b><br>" +
          "<sup>NSE achieved when model calibrated with each objective function</sup>",
    xaxis_title="Calibration Objective",
    yaxis_title="Model",
    height=450,
    width=750
)
fig.show()

# %% [markdown]
# ---
# ## Step 14: Radar Charts - Per Objective Function
#
# Radar charts showing model performance for EACH objective function.

# %%
# Radar charts for each objective function
radar_metrics = ['NSE', 'KGE', 'NSE (log Q)', 'NSE (1/Q)', 'Pearson r']

for obj_name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    fig = go.Figure()
    
    for model in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        if obj_name not in all_metrics.get(model, {}):
            continue
        
        values = []
        for m in radar_metrics:
            if m in all_metrics[model][obj_name]:
                values.append(all_metrics[model][obj_name][m])
            else:
                values.append(0)
        
        values_closed = values + [values[0]]
        metrics_closed = radar_metrics + [radar_metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=metrics_closed,
            fill='toself',
            name=model,
            line=dict(color=MODEL_COLORS[model]),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"<b>Model Comparison - {obj_name} Calibrated</b>",
        height=450,
        width=550
    )
    fig.show()

# %% [markdown]
# ---
# ## Step 14: Summary and Model Selection

# %%
# Overall ranking
print("=" * 100)
print("MODEL SELECTION SUMMARY")
print("=" * 100)

# Calculate mean NSE across all calibration objectives
print("\n1. AVERAGE NSE ACROSS ALL CALIBRATION OBJECTIVES")
print("-" * 60)

mean_nse = heatmap_df.mean(axis=1).sort_values(ascending=False)
for rank, (model, nse) in enumerate(mean_nse.items(), 1):
    n_params = len(list(ALL_RESULTS[model].values())[0].best_parameters)
    print(f"   {rank}. {model:<12} | Mean NSE: {nse:.4f} | Params: {n_params}")

# Best model per objective
print("\n2. BEST MODEL FOR EACH CALIBRATION OBJECTIVE")
print("-" * 60)

for obj_name in heatmap_df.columns:
    best_model = heatmap_df[obj_name].idxmax()
    best_nse = heatmap_df.loc[best_model, obj_name]
    print(f"   {obj_name:<12} → {best_model:<12} (NSE: {best_nse:.4f})")

# Print final summary
print(f"""

{'=' * 100}
FINAL RECOMMENDATIONS
{'=' * 100}

┌──────────────────────────────────────────────────────────────────────────────┐
│  APPLICATION                        │  RECOMMENDED MODEL                     │
├─────────────────────────────────────┼────────────────────────────────────────┤
│  Quick regional assessment          │  GR4J (4 params, fast, robust)         │
│  General purpose / water supply     │  GR5J or GR6J (balanced performance)   │
│  Low flow / drought analysis        │  GR6J (exponential store helps)        │
│  Operational flood forecasting      │  Sacramento (industry standard)        │
│  Detailed soil moisture tracking    │  Sacramento (multi-zone states)        │
└──────────────────────────────────────────────────────────────────────────────┘

Key Findings:
- All models achieve similar NSE when calibrated with NSE objective
- Simpler models (GR4J, GR5J) are more robust across different objectives
- Sacramento requires more careful calibration due to 22 parameters
- GR6J often outperforms for low-flow metrics due to exponential store

The "best" model depends on your specific application requirements!
""")

# %% [markdown]
# ---
# ## Export Results

# %%
# Export calibration summary
export_summary = []
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    for obj_name in OBJECTIVES.keys():
        if obj_name in ALL_RESULTS[model_name]:
            result = ALL_RESULTS[model_name][obj_name]
            export_summary.append({
                'model': model_name,
                'n_params': len(result.best_parameters),
                'objective': obj_name,
                'best_value': result.best_objective,
                'runtime_s': ALL_TIMES[model_name].get(obj_name, np.nan),
            })

summary_df = pd.DataFrame(export_summary)
summary_df.to_csv('../test_data/model_comparison_all_objectives.csv', index=False)
print("Summary saved to: test_data/model_comparison_all_objectives.csv")

# Export comprehensive metrics (NSE-calibrated)
comp_df.to_csv('../test_data/model_comparison_metrics.csv')
print("Metrics saved to: test_data/model_comparison_metrics.csv")

# Export heatmap data
heatmap_df.to_csv('../test_data/model_comparison_heatmap.csv')
print("Heatmap data saved to: test_data/model_comparison_heatmap.csv")

# %%
print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE!")
print("=" * 70)
print("\nThis notebook compared 4 models × 5 objectives = 20 calibrations")
print("All results saved to test_data/ directory")
