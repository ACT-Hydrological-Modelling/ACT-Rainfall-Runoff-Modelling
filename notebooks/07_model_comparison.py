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
#     display_name: Python (pyrrm)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Model Comparison: GR4J, GR5J, GR6J vs Sacramento
#
# ## Purpose
#
# This notebook provides a **comprehensive comparison** of four rainfall-runoff models
# calibrated with **13 different objective functions**. By systematically evaluating
# GR4J, GR5J, GR6J, and Sacramento across multiple performance metrics, we can
# understand which model structures work best for different applications.
#
# ## What You'll Learn
#
# - How model complexity affects calibration performance
# - How different objective functions reveal model strengths and weaknesses
# - How to interpret FDC segment errors and hydrologic signatures
# - How to choose the right model for your application
#
# ## Prerequisites
#
# - **Must complete Notebook 02: Calibration Quickstart first**
#   - Sacramento calibration results (13 objectives) are loaded from saved reports
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
# ## Objective Functions (13 total)
#
# | Category | Objectives | Emphasis |
# |----------|------------|----------|
# | **NSE variants** | NSE, LogNSE, InvNSE, SqrtNSE | Various flow regimes |
# | **KGE variants** | KGE, KGE_inv, KGE_sqrt, KGE_log | Decomposed performance |
# | **KGE_np variants** | KGE_np, KGE_np_inv, KGE_np_sqrt, KGE_np_log | Robust to outliers |
# | **Composite** | SDEB | FDC shape + timing + bias |

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
import plotly.express as px

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

print("=" * 70)
print("MODEL COMPARISON: GR4J, GR5J, GR6J vs SACRAMENTO")
print("13 Objective Functions × 4 Models = 52 Calibrations")
print("=" * 70)

# %%
# Import pyrrm models and calibration tools
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
from pyrrm.calibration import CalibrationRunner, CalibrationReport
from pyrrm.calibration.objective_functions import NSE, calculate_metrics

# Import objective functions from pyrrm.objectives
from pyrrm.objectives import (
    NSE as NSE_obj, KGE, KGENonParametric, PBIAS, RMSE,
    FlowTransformation, FDCMetric, SignatureMetric, SDEB,
    PearsonCorrelation, SpearmanCorrelation, MAE
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

# %%
# Configure paths
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
# Define calibration period
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
# ## Step 2: Define All 13 Objective Functions

# %%
# ==============================================================================
# DEFINE ALL 13 OBJECTIVE FUNCTIONS (Same as Notebook 02)
# ==============================================================================

# --- NSE variants (4) ---
nse_objective = NSE()  # Standard NSE
log_nse_objective = NSE_obj(transform=FlowTransformation('log', epsilon_value=0.01))
inv_nse_objective = NSE_obj(transform=FlowTransformation('inverse', epsilon_value=0.01))
sqrt_nse_objective = NSE_obj(transform=FlowTransformation('sqrt'))

# --- KGE variants (4) ---
kge_objective = KGE(variant='2012')
kge_inv_objective = KGE(variant='2012', transform=FlowTransformation('inverse', epsilon_value=0.01))
kge_sqrt_objective = KGE(variant='2012', transform=FlowTransformation('sqrt'))
kge_log_objective = KGE(variant='2012', transform=FlowTransformation('log', epsilon_value=0.01))

# --- Non-parametric KGE variants (4) ---
kge_np_objective = KGENonParametric()
kge_np_inv_objective = KGENonParametric(transform=FlowTransformation('inverse', epsilon_value=0.01))
kge_np_sqrt_objective = KGENonParametric(transform=FlowTransformation('sqrt'))
kge_np_log_objective = KGENonParametric(transform=FlowTransformation('log', epsilon_value=0.01))

# --- Composite (1) ---
sdeb_objective = SDEB(alpha=0.1, lam=0.5)

# Dictionary of all 13 objectives
OBJECTIVES = {
    # NSE variants (4)
    'nse': nse_objective,
    'lognse': log_nse_objective,
    'invnse': inv_nse_objective,
    'sqrtnse': sqrt_nse_objective,
    # KGE variants (4)
    'kge': kge_objective,
    'kge_inv': kge_inv_objective,
    'kge_sqrt': kge_sqrt_objective,
    'kge_log': kge_log_objective,
    # Non-parametric KGE variants (4)
    'kge_np': kge_np_objective,
    'kge_np_inv': kge_np_inv_objective,
    'kge_np_sqrt': kge_np_sqrt_objective,
    'kge_np_log': kge_np_log_objective,
    # Composite (1)
    'sdeb': sdeb_objective,
}

print("=" * 70)
print("13 OBJECTIVE FUNCTIONS DEFINED")
print("=" * 70)
print("\nNSE variants (4):")
print("  1. nse:      Standard Nash-Sutcliffe Efficiency")
print("  2. lognse:   NSE with log(Q) transform")
print("  3. invnse:   NSE with 1/Q transform (low flow focus)")
print("  4. sqrtnse:  NSE with √Q transform")
print("\nKGE variants (4):")
print("  5. kge:      Standard KGE (2012)")
print("  6. kge_inv:  KGE with 1/Q transform")
print("  7. kge_sqrt: KGE with √Q transform")
print("  8. kge_log:  KGE with log(Q) transform")
print("\nNon-parametric KGE variants (4):")
print("  9.  kge_np:      Non-parametric KGE (Spearman)")
print("  10. kge_np_inv:  KGE_np with 1/Q transform")
print("  11. kge_np_sqrt: KGE_np with √Q transform")
print("  12. kge_np_log:  KGE_np with log(Q) transform")
print("\nComposite (1):")
print("  13. sdeb:    Chronological + FDC + Bias (SOURCE)")

# %% [markdown]
# ---
# ## Step 3: Load Sacramento Results from Notebook 02 (All 13 Objectives)

# %%
# Load ALL 13 Sacramento calibrations from Notebook 02
print("=" * 70)
print("LOADING SACRAMENTO RESULTS FROM NOTEBOOK 02 (ALL 13 OBJECTIVES)")
print("=" * 70)

# Map of ALL 13 objective names to report files
SACRAMENTO_REPORTS = {
    # NSE variants (4)
    'nse': '../test_data/reports/models/410734_sacramento_nse_sceua.pkl',
    'lognse': '../test_data/reports/models/410734_sacramento_nse_sceua_log.pkl',
    'invnse': '../test_data/reports/models/410734_sacramento_nse_sceua_inverse.pkl',
    'sqrtnse': '../test_data/reports/models/410734_sacramento_nse_sceua_sqrt.pkl',
    # KGE variants (4)
    'kge': '../test_data/reports/models/410734_sacramento_kge_sceua.pkl',
    'kge_inv': '../test_data/reports/models/410734_sacramento_kge_sceua_inverse.pkl',
    'kge_sqrt': '../test_data/reports/models/410734_sacramento_kge_sceua_sqrt.pkl',
    'kge_log': '../test_data/reports/models/410734_sacramento_kge_sceua_log.pkl',
    # Non-parametric KGE variants (4)
    'kge_np': '../test_data/reports/models/410734_sacramento_kgenp_sceua.pkl',
    'kge_np_inv': '../test_data/reports/models/410734_sacramento_kgenp_sceua_inverse.pkl',
    'kge_np_sqrt': '../test_data/reports/models/410734_sacramento_kgenp_sceua_sqrt.pkl',
    'kge_np_log': '../test_data/reports/models/410734_sacramento_kgenp_sceua_log.pkl',
    # Composite (1)
    'sdeb': '../test_data/reports/models/410734_sacramento_sdeb_sceua.pkl',
}

sacramento_results = {}
sacramento_loaded = 0

print("\nLoading Sacramento calibration reports (13 total):")
print("-" * 60)

for obj_name, report_path in SACRAMENTO_REPORTS.items():
    path = Path(report_path)
    if path.exists():
        try:
            report = CalibrationReport.load(str(path))
            sacramento_results[obj_name] = report.result
            sacramento_loaded += 1
            print(f"  ✓ {obj_name:12s}: Best = {report.result.best_objective:.4f}")
        except Exception as e:
            print(f"  ✗ {obj_name:12s}: Error - {e}")
    else:
        print(f"  ✗ {obj_name:12s}: File not found")

print(f"\nLoaded {sacramento_loaded}/13 Sacramento calibrations")

if sacramento_loaded < 13:
    print("\n⚠️  WARNING: Some Sacramento calibrations are missing.")
    print("Please run Notebook 02 completely to generate all 13 calibrations.")

# %% [markdown]
# ---
# ## Step 4: Calibration Configuration

# %%
# SCE-UA evaluations scaled by model complexity
MAX_EVALS = {
    'GR4J': 4000,       # 4 params
    'GR5J': 10000,      # 5 params - needs more evals due to UH2-only architecture
    'GR6J': 6000,       # 6 params
    'Sacramento': 10000,  # 22 params (already done)
}

print("CALIBRATION CONFIGURATION")
print("=" * 60)
print(f"Algorithm: SCE-UA Direct")
print(f"Objectives: 13 (NSE variants, KGE variants, KGE_np variants, SDEB)")
print(f"Models: GR4J, GR5J, GR6J")
print(f"\nMax evaluations by model:")
for model, evals in MAX_EVALS.items():
    print(f"  {model}: {evals:,}")
print(f"\nTotal NEW calibrations: 3 GR models × 13 objectives = 39")

# Model colors for plotting
MODEL_COLORS = {
    'GR4J': '#1f77b4',       # Blue
    'GR5J': '#2ca02c',       # Green
    'GR6J': '#ff7f0e',       # Orange
    'Sacramento': '#d62728', # Red
    'Observed': '#000000',   # Black
}

# Models to include in analyses and plots
MODELS_FOR_ANALYSIS = ['GR4J', 'GR5J', 'GR6J', 'Sacramento']
MODELS_GR_ONLY = [m for m in MODELS_FOR_ANALYSIS if m != 'Sacramento']  # GR4J, GR5J, GR6J

# %% [markdown]
# ---
# ## Step 4b: Load from Pickle or Run Calibrations
#
# Set `LOAD_FROM_PICKLE = True` to load GR4J, GR5J, and GR6J results from
# pre-run report files (no need to re-run the 39 calibrations).
# Set to `False` to run all calibrations from scratch.

# %%
# Set to True to load from pre-run pickle files; False to run all calibrations
LOAD_FROM_PICKLE = True

REPORTS_DIR = Path('../test_data/reports/models')

gr4j_results = {}
gr5j_results = {}
gr6j_results = {}
gr4j_times = {}
gr5j_times = {}
gr6j_times = {}

OBJ_TO_CANONICAL = {
    'nse': 'nse_sceua',
    'lognse': 'nse_sceua_log',
    'invnse': 'nse_sceua_inverse',
    'sqrtnse': 'nse_sceua_sqrt',
    'kge': 'kge_sceua',
    'kge_inv': 'kge_sceua_inverse',
    'kge_sqrt': 'kge_sceua_sqrt',
    'kge_log': 'kge_sceua_log',
    'kge_np': 'kgenp_sceua',
    'kge_np_inv': 'kgenp_sceua_inverse',
    'kge_np_sqrt': 'kgenp_sceua_sqrt',
    'kge_np_log': 'kgenp_sceua_log',
    'sdeb': 'sdeb_sceua',
}

if LOAD_FROM_PICKLE:
    print("=" * 70)
    print("LOADING GR4J, GR5J, GR6J FROM PRE-RUN PICKLE FILES")
    print("=" * 70)
    for model_key, results_dict, times_dict in [
        ('gr4j', gr4j_results, gr4j_times),
        ('gr5j', gr5j_results, gr5j_times),
        ('gr6j', gr6j_results, gr6j_times),
    ]:
        loaded = 0
        for obj_name in OBJECTIVES.keys():
            canonical = OBJ_TO_CANONICAL[obj_name]
            pkl_path = REPORTS_DIR / f'410734_{model_key}_{canonical}.pkl'
            if pkl_path.exists():
                try:
                    report = CalibrationReport.load(str(pkl_path))
                    results_dict[obj_name] = report.result
                    times_dict[obj_name] = getattr(report.result, 'runtime_seconds', 0.0)
                    loaded += 1
                except Exception as e:
                    print(f"  ✗ {model_key} {obj_name}: {e}")
        print(f"  {model_key.upper()}: loaded {loaded}/13")
    print("\nDone. Skip the calibration cells below and go to Step 8: Organize All Results.")
else:
    print("LOAD_FROM_PICKLE is False. Run the calibration cells below to perform all 39 calibrations.")

# %% [markdown]
# ---
# ## Step 5: Calibrate GR4J with All 13 Objectives
#
# Each calibration runs in a separate cell for clear progress tracking.

# %% [markdown]
# ### GR4J Calibrations (13 total)

# %%
# GR4J Calibration 1/13: NSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 1/13: NSE")
    print("=" * 70)

    gr4j_results = {}
    gr4j_times = {}

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['nse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['nse'] = result
    gr4j_times['nse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_nse_sceua')
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 2/13: LogNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 2/13: LogNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['lognse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['lognse'] = result
    gr4j_times['lognse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_nse_sceua_log')
    print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 3/13: InvNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 3/13: InvNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['invnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['invnse'] = result
    gr4j_times['invnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_nse_sceua_inverse')
    print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 4/13: SqrtNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 4/13: SqrtNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sqrtnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['sqrtnse'] = result
    gr4j_times['sqrtnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_nse_sceua_sqrt')
    print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 5/13: KGE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 5/13: KGE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge'] = result
    gr4j_times['kge'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kge_sceua')
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 6/13: KGE_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 6/13: KGE_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_inv'] = result
    gr4j_times['kge_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kge_sceua_inverse')
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 7/13: KGE_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 7/13: KGE_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_sqrt'] = result
    gr4j_times['kge_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kge_sceua_sqrt')
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 8/13: KGE_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 8/13: KGE_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_log'] = result
    gr4j_times['kge_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kge_sceua_log')
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 9/13: KGE_np (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 9/13: KGE_np")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_np'] = result
    gr4j_times['kge_np'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kgenp_sceua')
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 10/13: KGE_np_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 10/13: KGE_np_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_np_inv'] = result
    gr4j_times['kge_np_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kgenp_sceua_inverse')
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 11/13: KGE_np_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 11/13: KGE_np_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_np_sqrt'] = result
    gr4j_times['kge_np_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kgenp_sceua_sqrt')
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 12/13: KGE_np_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 12/13: KGE_np_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['kge_np_log'] = result
    gr4j_times['kge_np_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_kgenp_sceua_log')
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 13/13: SDEB (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR4J CALIBRATION 13/13: SDEB")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sdeb'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR4J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr4j_results['sdeb'] = result
    gr4j_times['sdeb'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr4j_sdeb_sceua')
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR4J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 6: Calibrate GR5J with All 13 Objectives

# %% [markdown]
# ### GR5J Calibrations (13 total)

# %%
# GR5J Calibration 1/13: NSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 1/13: NSE")
    print("=" * 70)

    gr5j_results = {}
    gr5j_times = {}

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['nse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['nse'] = result
    gr5j_times['nse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_nse_sceua')
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 2/13: LogNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 2/13: LogNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['lognse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['lognse'] = result
    gr5j_times['lognse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_nse_sceua_log')
    print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 3/13: InvNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 3/13: InvNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['invnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['invnse'] = result
    gr5j_times['invnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_nse_sceua_inverse')
    print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 4/13: SqrtNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 4/13: SqrtNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sqrtnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['sqrtnse'] = result
    gr5j_times['sqrtnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_nse_sceua_sqrt')
    print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 5/13: KGE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 5/13: KGE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge'] = result
    gr5j_times['kge'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kge_sceua')
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 6/13: KGE_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 6/13: KGE_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_inv'] = result
    gr5j_times['kge_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kge_sceua_inverse')
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 7/13: KGE_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 7/13: KGE_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_sqrt'] = result
    gr5j_times['kge_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kge_sceua_sqrt')
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 8/13: KGE_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 8/13: KGE_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_log'] = result
    gr5j_times['kge_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kge_sceua_log')
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 9/13: KGE_np (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 9/13: KGE_np")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_np'] = result
    gr5j_times['kge_np'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kgenp_sceua')
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 10/13: KGE_np_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 10/13: KGE_np_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_np_inv'] = result
    gr5j_times['kge_np_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kgenp_sceua_inverse')
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 11/13: KGE_np_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 11/13: KGE_np_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_np_sqrt'] = result
    gr5j_times['kge_np_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kgenp_sceua_sqrt')
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 12/13: KGE_np_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 12/13: KGE_np_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['kge_np_log'] = result
    gr5j_times['kge_np_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_kgenp_sceua_log')
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 13/13: SDEB (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR5J CALIBRATION 13/13: SDEB")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sdeb'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], n_complexes=11, seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr5j_results['sdeb'] = result
    gr5j_times['sdeb'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr5j_sdeb_sceua')
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR5J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 7: Calibrate GR6J with All 13 Objectives

# %% [markdown]
# ### GR6J Calibrations (13 total)

# %%
# GR6J Calibration 1/13: NSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 1/13: NSE")
    print("=" * 70)

    gr6j_results = {}
    gr6j_times = {}

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['nse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['nse'] = result
    gr6j_times['nse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_nse_sceua')
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 2/13: LogNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 2/13: LogNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['lognse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['lognse'] = result
    gr6j_times['lognse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_nse_sceua_log')
    print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 3/13: InvNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 3/13: InvNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['invnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['invnse'] = result
    gr6j_times['invnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_nse_sceua_inverse')
    print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 4/13: SqrtNSE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 4/13: SqrtNSE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sqrtnse'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['sqrtnse'] = result
    gr6j_times['sqrtnse'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_nse_sceua_sqrt')
    print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 5/13: KGE (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 5/13: KGE")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge'] = result
    gr6j_times['kge'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kge_sceua')
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 6/13: KGE_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 6/13: KGE_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_inv'] = result
    gr6j_times['kge_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kge_sceua_inverse')
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 7/13: KGE_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 7/13: KGE_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_sqrt'] = result
    gr6j_times['kge_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kge_sceua_sqrt')
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 8/13: KGE_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 8/13: KGE_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_log'] = result
    gr6j_times['kge_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kge_sceua_log')
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 9/13: KGE_np (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 9/13: KGE_np")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_np'] = result
    gr6j_times['kge_np'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kgenp_sceua')
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 10/13: KGE_np_inv (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 10/13: KGE_np_inv")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_inv'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_np_inv'] = result
    gr6j_times['kge_np_inv'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kgenp_sceua_inverse')
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 11/13: KGE_np_sqrt (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 11/13: KGE_np_sqrt")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_sqrt'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_np_sqrt'] = result
    gr6j_times['kge_np_sqrt'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kgenp_sceua_sqrt')
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 12/13: KGE_np_log (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 12/13: KGE_np_log")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['kge_np_log'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['kge_np_log'] = result
    gr6j_times['kge_np_log'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_kgenp_sceua_log')
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 13/13: SDEB (skipped if LOAD_FROM_PICKLE)
if not LOAD_FROM_PICKLE:
    print("=" * 70)
    print("GR6J CALIBRATION 13/13: SDEB")
    print("=" * 70)

    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=OBJECTIVES['sdeb'],
        warmup_period=WARMUP_DAYS
    )

    start_time = time.time()
    result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR6J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
    elapsed = time.time() - start_time

    gr6j_results['sdeb'] = result
    gr6j_times['sdeb'] = elapsed

    report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
    report.save('../test_data/reports/models/410734_gr6j_sdeb_sceua')
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR6J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 8: Organize All Results

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

# Display calibration summary
print("=" * 140)
print("CALIBRATION SUMMARY: ALL MODELS × 13 OBJECTIVES")
print("=" * 140)

obj_names = list(OBJECTIVES.keys())
print(f"\n{'Model':<12} {'Params':>6}", end="")
for obj_name in obj_names:
    print(f" {obj_name:>10}", end="")
print()
print("-" * 140)

for model_name in MODELS_FOR_ANALYSIS:
    if ALL_RESULTS[model_name]:
        n_params = len(list(ALL_RESULTS[model_name].values())[0].best_parameters)
    else:
        n_params = '?'
    print(f"{model_name:<12} {n_params:>6}", end="")
    
    for obj_name in obj_names:
        if obj_name in ALL_RESULTS[model_name]:
            val = ALL_RESULTS[model_name][obj_name].best_objective
            print(f" {val:>10.4f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

# %% [markdown]
# ---
# ## Step 9: Generate Simulations for All Combinations

# %%
# Generate simulations for ALL model × objective combinations
print("=" * 70)
print("GENERATING ALL SIMULATIONS (4 Models × 13 Objectives = 52 total)")
print("=" * 70)

def get_flow_column(sim_results):
    """Get flow column - Sacramento uses 'runoff', GR models use 'flow'."""
    if 'runoff' in sim_results.columns:
        return sim_results['runoff'].values
    elif 'flow' in sim_results.columns:
        return sim_results['flow'].values
    else:
        raise KeyError(f"No flow column found. Available: {sim_results.columns.tolist()}")

all_simulations = {}

for model_name in MODELS_FOR_ANALYSIS:
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
    
    print(f"  ✓ {model_name}: {len(all_simulations[model_name])} simulations")

# Get observed data for comparison
obs_flow = cal_observed[WARMUP_DAYS:]
comparison_dates = cal_data.index[WARMUP_DAYS:]

print(f"\nSimulations complete! ({len(obs_flow):,} days each)")

# %% [markdown]
# ---
# ## Step 10: Comprehensive Model Comparison Visualizations
#
# Now we create intelligent visualizations comparing all models across:
# 1. Flow Duration Curves by objective function family
# 2. FDC segment errors (Peak, High, Mid, Low, Very Low)
# 3. Hydrologic signature errors
# 4. Traditional error metrics (RMSE, MAE, PBIAS)

# %% [markdown]
# ### 10.1 Comprehensive Metrics Calculation

# %%
# Calculate comprehensive metrics for ALL model × objective combinations
def comprehensive_evaluation(obs, sim):
    """Compute comprehensive metrics for model comparison."""
    inv_transform = FlowTransformation('inverse', epsilon_value=0.01)
    sqrt_transform = FlowTransformation('sqrt')
    log_transform = FlowTransformation('log', epsilon_value=0.01)
    
    metrics = {}
    
    # Overall efficiency metrics
    metrics['NSE'] = NSE_obj()(obs, sim)
    metrics['KGE'] = KGE(variant='2012')(obs, sim)
    metrics['KGE_np'] = KGENonParametric()(obs, sim)
    
    # Transformed metrics
    metrics['LogNSE'] = NSE_obj(transform=log_transform)(obs, sim)
    metrics['InvNSE'] = NSE_obj(transform=inv_transform)(obs, sim)
    metrics['SqrtNSE'] = NSE_obj(transform=sqrt_transform)(obs, sim)
    metrics['KGE_inv'] = KGE(transform=inv_transform)(obs, sim)
    metrics['KGE_sqrt'] = KGE(transform=sqrt_transform)(obs, sim)
    metrics['KGE_log'] = KGE(transform=log_transform)(obs, sim)
    
    # Traditional errors
    metrics['RMSE'] = RMSE()(obs, sim)
    metrics['MAE'] = MAE()(obs, sim)
    metrics['PBIAS'] = PBIAS()(obs, sim)
    metrics['Pearson_r'] = PearsonCorrelation()(obs, sim)
    metrics['Spearman_r'] = SpearmanCorrelation()(obs, sim)
    
    # FDC segment errors
    try:
        metrics['FDC_Peak'] = FDCMetric('peak', 'volume_bias')(obs, sim)
        metrics['FDC_High'] = FDCMetric('high', 'volume_bias')(obs, sim)
        metrics['FDC_Mid'] = FDCMetric('mid', 'volume_bias')(obs, sim)
        metrics['FDC_Low'] = FDCMetric('low', 'volume_bias', log_transform=True)(obs, sim)
        metrics['FDC_VeryLow'] = FDCMetric('very_low', 'volume_bias', log_transform=True)(obs, sim)
    except Exception:
        pass
    
    # Hydrologic signatures
    try:
        metrics['Q95_error'] = SignatureMetric('q95')(obs, sim)
        metrics['Q50_error'] = SignatureMetric('q50')(obs, sim)
        metrics['Q5_error'] = SignatureMetric('q5')(obs, sim)
        metrics['Flashiness_error'] = SignatureMetric('flashiness')(obs, sim)
        metrics['BFI_error'] = SignatureMetric('baseflow_index')(obs, sim)
    except Exception:
        pass
    
    # SDEB
    try:
        metrics['SDEB'] = SDEB(alpha=0.1, lam=0.5)(obs, sim)
    except Exception:
        pass
    
    return metrics

# Calculate metrics for all combinations
print("=" * 70)
print("COMPUTING COMPREHENSIVE EVALUATION FOR ALL COMBINATIONS")
print("=" * 70)

all_metrics = {}
for model_name in MODELS_FOR_ANALYSIS:
    all_metrics[model_name] = {}
    for obj_name in OBJECTIVES.keys():
        if obj_name in all_simulations.get(model_name, {}):
            all_metrics[model_name][obj_name] = comprehensive_evaluation(
                obs_flow, all_simulations[model_name][obj_name]
            )
    print(f"  ✓ {model_name}: {len(all_metrics[model_name])} evaluations")

print("\nDone!")

# %% [markdown]
# ### 10.2 Flow Duration Curves by Objective Function Family

# %%
# FDC comparison grouped by objective function family
obs_sorted = np.sort(obs_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

# Define objective families ordered from high-flow to low-flow emphasis
# Non-transformed → sqrt → log → inverse
obj_families = {
    'NSE variants': [
        ('nse',     'NSE — High-flow emphasis'),
        ('sqrtnse', 'Sqrt-NSE — Mid-flow emphasis'),
        ('lognse',  'Log-NSE — Low-flow emphasis'),
        ('invnse',  'Inv-NSE — Lowest-flow emphasis'),
    ],
    'KGE variants': [
        ('kge',      'KGE — High-flow emphasis'),
        ('kge_sqrt', 'KGE-Sqrt — Mid-flow emphasis'),
        ('kge_log',  'KGE-Log — Low-flow emphasis'),
        ('kge_inv',  'KGE-Inv — Lowest-flow emphasis'),
    ],
    'KGE_np variants': [
        ('kge_np',      'KGE\' — High-flow emphasis'),
        ('kge_np_sqrt', 'KGE\'-Sqrt — Mid-flow emphasis'),
        ('kge_np_log',  'KGE\'-Log — Low-flow emphasis'),
        ('kge_np_inv',  'KGE\'-Inv — Lowest-flow emphasis'),
    ],
}

for family_name, family_items in obj_families.items():
    subplot_titles = [label for _, label in family_items]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, (obj_name, _) in enumerate(family_items):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Observed
        fig.add_trace(
            go.Scatter(x=exceedance, y=obs_sorted, name='Observed' if idx == 0 else None,
                      showlegend=(idx == 0), line=dict(color='black', width=2)),
            row=row, col=col
        )
        
        # All models
        for model_name in MODELS_FOR_ANALYSIS:
            if obj_name in all_simulations.get(model_name, {}):
                sim_sorted = np.sort(all_simulations[model_name][obj_name])[::-1]
                fig.add_trace(
                    go.Scatter(x=exceedance, y=sim_sorted, 
                              name=model_name if idx == 0 else None,
                              showlegend=(idx == 0),
                              line=dict(color=MODEL_COLORS[model_name], width=1.5)),
                    row=row, col=col
                )
        
        fig.update_yaxes(type="log", row=row, col=col)
        fig.update_xaxes(title_text="Exceedance %" if row == 2 else None, row=row, col=col)
        fig.update_yaxes(title_text="Flow (ML/day)" if col == 1 else None, row=row, col=col)
    
    fig.update_layout(
        title=f"<b>How does flow-emphasis shift the FDC fit across models? — {family_name}</b>",
        height=700,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.06,
            xanchor='right',
            x=1.0
        )
    )
    fig.show()

# %% [markdown]
# ### 10.3 FDC Segment Error Analysis

# %%
# FDC segment errors heatmap - one per model showing all objectives
fdc_segments = ['FDC_Peak', 'FDC_High', 'FDC_Mid', 'FDC_Low', 'FDC_VeryLow']
fdc_labels = ['Peak (0-2%)', 'High (2-10%)', 'Mid (20-70%)', 'Low (70-90%)', 'Very Low (90-100%)']

# Objectives ordered by flow emphasis (high → very low) with descriptive labels
OBJ_ORDERED = [
    # --- High-flow focus (non-transformed Q) ---
    ('nse',        'NSE'),
    ('kge',        'KGE'),
    ('kge_np',     "KGE'"),
    # --- Balanced focus (√Q transform) ---
    ('sqrtnse',    '√NSE'),
    ('kge_sqrt',   'KGE-√Q'),
    ('kge_np_sqrt',"KGE'-√Q"),
    ('sdeb',       'SDEB'),
    # --- Low-flow focus (log Q transform) ---
    ('lognse',     'log-NSE'),
    ('kge_log',    'KGE-log'),
    ('kge_np_log', "KGE'-log"),
    # --- Very-low-flow focus (1/Q transform) ---
    ('invnse',     '1/Q NSE'),
    ('kge_inv',    'KGE-1/Q'),
    ('kge_np_inv', "KGE'-1/Q"),
]
OBJ_KEYS_ORDERED = [k for k, _ in OBJ_ORDERED]
OBJ_LABELS_ORDERED = [lbl for _, lbl in OBJ_ORDERED]

# Group boundaries for annotation (index of first objective in each group)
OBJ_GROUPS = [
    (0,  'High-flow focus'),
    (3,  'Balanced (√Q)'),
    (7,  'Low-flow focus (log Q)'),
    (10, 'Very-low-flow focus (1/Q)'),
]

fig = make_subplots(
    rows=1, cols=len(MODELS_FOR_ANALYSIS),
    subplot_titles=MODELS_FOR_ANALYSIS,
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

for idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
    row = 1
    col = idx + 1
    
    data = []
    for obj_key in OBJ_KEYS_ORDERED:
        row_data = []
        for seg in fdc_segments:
            if obj_key in all_metrics.get(model_name, {}) and seg in all_metrics[model_name][obj_key]:
                row_data.append(all_metrics[model_name][obj_key][seg])
            else:
                row_data.append(np.nan)
        data.append(row_data)
    
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=['Peak', 'High', 'Mid', 'Low', 'V.Low'],
            y=OBJ_LABELS_ORDERED,
            colorscale='RdBu',
            zmid=0,
            zmin=-50,
            zmax=50,
            showscale=(idx == len(MODELS_FOR_ANALYSIS) - 1),
            text=np.round(data, 1),
            texttemplate='%{text}',
            textfont={"size": 8},
        ),
        row=row, col=col
    )

# Add horizontal lines between emphasis groups and group labels on the left
for g_idx, (start_row_idx, group_label) in enumerate(OBJ_GROUPS):
    if start_row_idx > 0:
        y_line = start_row_idx - 0.5
        fig.add_hline(y=y_line, line_width=2, line_color='black', opacity=0.6)
    mid_idx = (start_row_idx + (OBJ_GROUPS[g_idx + 1][0] if g_idx + 1 < len(OBJ_GROUPS) else len(OBJ_ORDERED))) / 2
    fig.add_annotation(
        x=-0.08, y=OBJ_LABELS_ORDERED[int(mid_idx)],
        xref='paper', yref='y',
        text=f"<b>{group_label}</b>",
        showarrow=False,
        xanchor='right',
        font=dict(size=12, color='#444'),
    )

fig.update_layout(
    title="<b>Where does each model over- or underestimate volume across FDC segments and objectives?</b><br>" +
          "<sup>Red = overestimate, Blue = underestimate (% volume bias)</sup>",
    height=800,
    width=1600,
    margin=dict(l=350),
)
fig.show()

# %% [markdown]
# #### Alternative FDC visualizations: GR vs Sacramento
#
# The following figures make it easier to compare GR family models against Sacramento
# and to see which model "wins" for each objective and flow segment.

# %%
# 1) Winner heatmap: which model has the smallest absolute bias per (objective × segment)
segment_labels_short = ['Peak', 'High', 'Mid', 'Low', 'V.Low']
winner_to_num = {m: i for i, m in enumerate(MODELS_FOR_ANALYSIS)}
winner_z = []
winner_text = []
for obj_key in OBJ_KEYS_ORDERED:
    row_z = []
    row_txt = []
    for seg in fdc_segments:
        vals = [all_metrics.get(m, {}).get(obj_key, {}).get(seg, np.nan) for m in MODELS_FOR_ANALYSIS]
        abs_vals = [abs(x) if np.isfinite(x) else np.inf for x in vals]
        if not abs_vals or all(np.isinf(x) for x in abs_vals):
            row_z.append(np.nan)
            row_txt.append('—')
        else:
            idx_best = int(np.argmin(abs_vals))
            best_model = MODELS_FOR_ANALYSIS[idx_best]
            row_z.append(winner_to_num[best_model])
            row_txt.append(f"{best_model}<br>{vals[idx_best]:.0f}")
    winner_z.append(row_z)
    winner_text.append(row_txt)

fig_win = go.Figure(data=go.Heatmap(
    z=winner_z,
    x=segment_labels_short,
    y=OBJ_LABELS_ORDERED,
    text=winner_text,
    texttemplate='%{text}',
    textfont={"size": 9},
    zmin=0,
    zmax=3,
    colorscale=[[0, MODEL_COLORS['GR4J']], [0.33, MODEL_COLORS['GR5J']], [0.67, MODEL_COLORS['GR6J']], [1, MODEL_COLORS['Sacramento']]],
    showscale=True,
    colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=MODELS_FOR_ANALYSIS, len=0.5),
))
for g_idx, (start_row_idx, group_label) in enumerate(OBJ_GROUPS):
    if start_row_idx > 0:
        fig_win.add_hline(y=start_row_idx - 0.5, line_width=2, line_color='black', opacity=0.6)
    mid_idx = (start_row_idx + (OBJ_GROUPS[g_idx + 1][0] if g_idx + 1 < len(OBJ_GROUPS) else len(OBJ_ORDERED))) / 2
    fig_win.add_annotation(
        x=-0.15, y=OBJ_LABELS_ORDERED[int(mid_idx)],
        xref='paper', yref='y',
        text=f"<b>{group_label}</b>",
        showarrow=False, xanchor='right',
        font=dict(size=12, color='#444'),
    )
fig_win.update_layout(
    title="<b>Which model has the lowest absolute FDC bias for each objective and segment?</b><br><sup>Cell color = winning model; number = bias (%)</sup>",
    height=700,
    width=1300,
    xaxis_title="FDC segment",
    yaxis_title="Calibration objective",
    margin=dict(l=380),
)
fig_win.show()

# %%
# 2) Difference heatmaps: |GR bias| − |Sacramento bias| with symmetric-log color scale
# sign(x) * log10(1 + |x|) compresses outliers so detail near zero is visible
sac_bias = {}
for obj_key in OBJ_KEYS_ORDERED:
    sac_bias[obj_key] = {seg: all_metrics.get('Sacramento', {}).get(obj_key, {}).get(seg, np.nan) for seg in fdc_segments}

def symlog(x):
    """Signed log transform: sign(x) * log10(1 + |x|)."""
    if np.isfinite(x):
        return np.sign(x) * np.log10(1 + abs(x))
    return np.nan

# Colorbar tick values in original space → transformed positions
cbar_orig = [-500, -100, -50, -10, 0, 10, 50, 100, 500]
cbar_transformed = [symlog(v) for v in cbar_orig]

fig_diff = make_subplots(
    rows=1, cols=len(MODELS_GR_ONLY),
    subplot_titles=[f'|{m}| − |Sacramento|' for m in MODELS_GR_ONLY],
    horizontal_spacing=0.06,
)
for idx, gr_model in enumerate(MODELS_GR_ONLY):
    raw_data = []
    log_data = []
    for obj_key in OBJ_KEYS_ORDERED:
        raw_row = []
        log_row = []
        for seg in fdc_segments:
            gr_val = all_metrics.get(gr_model, {}).get(obj_key, {}).get(seg, np.nan)
            sac_val = sac_bias[obj_key].get(seg, np.nan)
            if np.isfinite(gr_val) and np.isfinite(sac_val):
                diff = abs(gr_val) - abs(sac_val)
                raw_row.append(diff)
                log_row.append(symlog(diff))
            else:
                raw_row.append(np.nan)
                log_row.append(np.nan)
        raw_data.append(raw_row)
        log_data.append(log_row)
    
    is_last = (idx == len(MODELS_GR_ONLY) - 1)
    fig_diff.add_trace(
        go.Heatmap(
            z=log_data,
            x=segment_labels_short,
            y=OBJ_LABELS_ORDERED,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(raw_data, 1),
            texttemplate='%{text}',
            textfont={"size": 8},
            showscale=is_last,
            colorbar=dict(
                title='|bias| diff (%)',
                tickvals=cbar_transformed,
                ticktext=[str(v) for v in cbar_orig],
            ) if is_last else None,
        ),
        row=1, col=idx + 1
    )

# Add group separators and labels
for g_idx, (start_row_idx, group_label) in enumerate(OBJ_GROUPS):
    if start_row_idx > 0:
        fig_diff.add_hline(y=start_row_idx - 0.5, line_width=2, line_color='black', opacity=0.6)
    mid_idx = (start_row_idx + (OBJ_GROUPS[g_idx + 1][0] if g_idx + 1 < len(OBJ_GROUPS) else len(OBJ_ORDERED))) / 2
    fig_diff.add_annotation(
        x=-0.06, y=OBJ_LABELS_ORDERED[int(mid_idx)],
        xref='paper', yref='y',
        text=f"<b>{group_label}</b>",
        showarrow=False, xanchor='right',
        font=dict(size=12, color='#444'),
    )

fig_diff.update_layout(
    title="<b>Which model has lower absolute FDC bias — each GR or Sacramento?</b><br>" +
          "<sup>Negative (blue) = GR closer to zero bias; Positive (red) = Sacramento closer; symmetric-log color scale</sup>",
    height=750,
    width=1700,
    margin=dict(l=320),
)
fig_diff.show()

# %%
# 3) Box + strip plots: distribution of bias across 13 objectives, per model and segment
#    Reveals which model is most *consistently* close to zero bias
seg_labels_full = ['Peak (0–2%)', 'High (2–10%)', 'Mid (20–70%)', 'Low (70–90%)', 'Very Low (90–100%)']
fig_box = make_subplots(
    rows=1, cols=5,
    subplot_titles=seg_labels_full,
    horizontal_spacing=0.05,
)
for seg_idx, seg in enumerate(fdc_segments):
    for model_name in MODELS_FOR_ANALYSIS:
        y_vals = []
        hover_labels = []
        for obj_key, obj_lbl in OBJ_ORDERED:
            v = all_metrics.get(model_name, {}).get(obj_key, {}).get(seg, np.nan)
            if np.isfinite(v):
                y_vals.append(v)
                hover_labels.append(obj_lbl)
        fig_box.add_trace(
            go.Box(
                y=y_vals,
                name=model_name,
                marker_color=MODEL_COLORS[model_name],
                line_color=MODEL_COLORS[model_name],
                boxmean=True,
                showlegend=(seg_idx == 0),
                legendgroup=model_name,
            ),
            row=1, col=seg_idx + 1
        )
        fig_box.add_trace(
            go.Scatter(
                y=y_vals,
                x=[model_name] * len(y_vals),
                mode='markers',
                marker=dict(size=5, color=MODEL_COLORS[model_name], opacity=0.5),
                text=hover_labels,
                hovertemplate='%{text}<br>Bias: %{y:.1f}%<extra></extra>',
                showlegend=False,
                legendgroup=model_name,
            ),
            row=1, col=seg_idx + 1
        )
    fig_box.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.6, row=1, col=seg_idx + 1)
    fig_box.update_yaxes(title_text='% volume bias' if seg_idx == 0 else None, row=1, col=seg_idx + 1)

fig_box.update_layout(
    title="<b>Which model most consistently achieves low FDC bias across all calibration objectives?</b><br>" +
          "<sup>Each box = distribution over 13 objectives; dashed line = zero (perfect); diamond = mean</sup>",
    height=550,
    width=1400,
    legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1.0),
)
fig_box.show()

# %%
# 4) Summary breakdown by FDC segment: where does each model excel?
seg_colors = {'FDC_Peak': '#264653', 'FDC_High': '#2a9d8f', 'FDC_Mid': '#e9c46a', 'FDC_Low': '#f4a261', 'FDC_VeryLow': '#e76f51'}

# Compute mean |bias| per model per segment, and wins per model per segment
mean_abs_by_seg = {m: {} for m in MODELS_FOR_ANALYSIS}
wins_by_seg = {m: {seg: 0 for seg in fdc_segments} for m in MODELS_FOR_ANALYSIS}

for model_name in MODELS_FOR_ANALYSIS:
    for seg in fdc_segments:
        vals = []
        for obj_key in OBJ_KEYS_ORDERED:
            v = all_metrics.get(model_name, {}).get(obj_key, {}).get(seg, np.nan)
            if np.isfinite(v):
                vals.append(abs(v))
        mean_abs_by_seg[model_name][seg] = np.mean(vals) if vals else np.nan

for obj_key in OBJ_KEYS_ORDERED:
    for seg in fdc_segments:
        abs_vals = [abs(all_metrics.get(m, {}).get(obj_key, {}).get(seg, np.nan)) for m in MODELS_FOR_ANALYSIS]
        valid = [i for i, v in enumerate(abs_vals) if np.isfinite(v)]
        if valid:
            idx_best = valid[int(np.argmin([abs_vals[i] for i in valid]))]
            wins_by_seg[MODELS_FOR_ANALYSIS[idx_best]][seg] += 1

fig_sum = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'Mean |bias| by FDC segment (lower is better)',
        'Win count by FDC segment (higher is better)'
    ],
    horizontal_spacing=0.12,
)

# Left panel: grouped bars — mean |bias| per model per segment
# Use manual x-positions with gaps between model groups
n_segs = len(fdc_segments)
bar_width = 0.14
group_width = n_segs * bar_width
group_gap = 0.35
x_group_centres = np.arange(len(MODELS_FOR_ANALYSIS)) * (group_width + group_gap)

for s_idx, (seg, seg_lbl) in enumerate(zip(fdc_segments, segment_labels_short)):
    x_bars = x_group_centres + (s_idx - (n_segs - 1) / 2) * bar_width
    fig_sum.add_trace(
        go.Bar(
            name=seg_lbl,
            x=x_bars,
            y=[mean_abs_by_seg[m][seg] for m in MODELS_FOR_ANALYSIS],
            width=bar_width,
            marker_color=seg_colors[seg],
            showlegend=True,
            legendgroup=seg,
        ),
        row=1, col=1
    )
fig_sum.update_xaxes(
    tickvals=x_group_centres,
    ticktext=MODELS_FOR_ANALYSIS,
    row=1, col=1
)

# Right panel: stacked bars — wins per model per segment
for s_idx, (seg, seg_lbl) in enumerate(zip(fdc_segments, segment_labels_short)):
    fig_sum.add_trace(
        go.Bar(
            name=seg_lbl,
            x=MODELS_FOR_ANALYSIS,
            y=[wins_by_seg[m][seg] for m in MODELS_FOR_ANALYSIS],
            marker_color=seg_colors[seg],
            showlegend=False,
            legendgroup=seg,
        ),
        row=1, col=2
    )

fig_sum.update_layout(
    title="<b>Where does each model excel across the flow duration curve?</b><br>" +
          "<sup>Left: mean absolute bias (lower = better); Right: number of wins out of 13 objectives per segment</sup>",
    barmode='stack',
    height=500,
    width=1300,
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1.0,
        xanchor='left',
        x=1.02,
        title_text='FDC segment',
    ),
)
fig_sum.update_yaxes(title_text="Mean |bias| (%)", row=1, col=1)
fig_sum.update_yaxes(title_text="Wins (out of 13)", row=1, col=2)
fig_sum.show()

# %%
# 5) Scatter: |GR bias| vs |Sacramento bias| on log axes, colored by FDC segment
obj_key_to_label = dict(OBJ_ORDERED)
seg_scatter_colors = {
    'FDC_Peak': '#264653', 'FDC_High': '#2a9d8f', 'FDC_Mid': '#e9c46a',
    'FDC_Low': '#f4a261', 'FDC_VeryLow': '#e76f51',
}

fig_scatter = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f'{m} vs Sacramento' for m in MODELS_GR_ONLY],
    horizontal_spacing=0.08,
)
for idx, gr_model in enumerate(MODELS_GR_ONLY):
    n_below = 0
    n_total = 0
    for seg_i, seg in enumerate(fdc_segments):
        x_vals, y_vals, hover = [], [], []
        for obj_key in OBJ_KEYS_ORDERED:
            sac_val = sac_bias[obj_key].get(seg, np.nan)
            gr_val = all_metrics.get(gr_model, {}).get(obj_key, {}).get(seg, np.nan)
            if np.isfinite(sac_val) and np.isfinite(gr_val):
                abs_sac = abs(sac_val) + 0.1
                abs_gr = abs(gr_val) + 0.1
                x_vals.append(abs_sac)
                y_vals.append(abs_gr)
                hover.append(f"{obj_key_to_label[obj_key]} / {segment_labels_short[seg_i]}")
                n_total += 1
                if abs_gr < abs_sac:
                    n_below += 1
        fig_scatter.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                name=segment_labels_short[seg_i],
                marker=dict(size=8, color=seg_scatter_colors[seg], opacity=0.75),
                text=hover,
                hovertemplate='%{text}<br>|Sac|: %{x:.1f}%<br>|GR|: %{y:.1f}%<extra></extra>',
                showlegend=(idx == 0),
                legendgroup=seg,
            ),
            row=1, col=idx + 1
        )
    # 1:1 line on log scale
    fig_scatter.add_trace(
        go.Scatter(
            x=[0.1, 1000], y=[0.1, 1000], mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name='1:1', showlegend=(idx == 0),
        ),
        row=1, col=idx + 1
    )
    # Win rate annotation
    pct = n_below / n_total * 100 if n_total > 0 else 0
    fig_scatter.add_annotation(
        text=f"<b>{gr_model} wins {n_below}/{n_total} ({pct:.0f}%)</b>",
        x=0.05, y=0.95,
        xanchor='left', yanchor='top',
        xref=f'x{idx + 1} domain' if idx > 0 else 'x domain',
        yref=f'y{idx + 1} domain' if idx > 0 else 'y domain',
        showarrow=False,
        font=dict(size=12, color=MODEL_COLORS[gr_model]),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=MODEL_COLORS[gr_model],
        borderwidth=1,
    )
    fig_scatter.update_xaxes(type='log', title_text='|Sacramento bias| (%)', row=1, col=idx + 1)
    fig_scatter.update_yaxes(type='log', title_text=f'|{gr_model} bias| (%)' if idx == 0 else None, row=1, col=idx + 1)

fig_scatter.update_layout(
    title="<b>Does any GR model consistently outperform Sacramento on FDC bias?</b><br>" +
          "<sup>Points below the 1:1 line = GR has lower absolute bias; log scale; colored by FDC segment</sup>",
    height=500,
    width=1400,
    legend=dict(
        orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02,
        title_text='FDC segment',
    ),
)
fig_scatter.show()

# %% [markdown]
# ### 10.4 Hydrologic Signature Errors

# %%
# Hydrologic signature errors — heatmap with saturating colour scale
# Colour resolution concentrated near zero; large errors all saturate to deep colour
signature_metrics = ['Q95_error', 'Q50_error', 'Q5_error', 'Flashiness_error', 'BFI_error']
signature_labels = ['Q95 (High Flow)', 'Q50 (Median)', 'Q5 (Low Flow)', 'Flashiness', 'Baseflow Index']

OBJ_EMPHASIS_GROUPS = {
    'High-flow (Q)': [k for k, _ in OBJ_ORDERED[:3]],
    'Balanced (√Q)': [k for k, _ in OBJ_ORDERED[3:7]],
    'Low-flow (log Q)': [k for k, _ in OBJ_ORDERED[7:10]],
    'Very-low-flow (1/Q)': [k for k, _ in OBJ_ORDERED[10:]],
}

KNEE = 15  # ±15% maps to ±0.5 on the colour scale
def soft_clamp(x, k=KNEE):
    """Saturating transform: maps (-inf,+inf) → (-1,+1), concentrating resolution near zero."""
    if not np.isfinite(x):
        return np.nan
    return np.sign(x) * abs(x) / (abs(x) + k)

emphasis_names = list(OBJ_EMPHASIS_GROUPS.keys())

fig_sig = make_subplots(
    rows=2, cols=2,
    subplot_titles=emphasis_names,
    vertical_spacing=0.16,
    horizontal_spacing=0.12,
)

grid_data = {}
for g_idx, (group_name, group_keys) in enumerate(OBJ_EMPHASIS_GROUPS.items()):
    z_raw = np.full((len(signature_metrics), len(MODELS_FOR_ANALYSIS)), np.nan)
    for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
        for s_idx, sig in enumerate(signature_metrics):
            errors = []
            for obj_key in group_keys:
                if obj_key in all_metrics.get(model_name, {}) and sig in all_metrics[model_name][obj_key]:
                    errors.append(all_metrics[model_name][obj_key][sig])
            if errors:
                z_raw[s_idx, m_idx] = np.mean(errors)
    grid_data[group_name] = z_raw

# Colorbar ticks: show original % values at their transformed positions
cb_originals = [0, 2, 5, 10, 15, 25, 50]
cb_ticks = []
cb_labels = []
for v in reversed(cb_originals[1:]):
    cb_ticks.append(soft_clamp(-v))
    cb_labels.append(f'−{v}%')
cb_ticks.append(0.0)
cb_labels.append('0%')
for v in cb_originals[1:]:
    cb_ticks.append(soft_clamp(v))
    cb_labels.append(f'+{v}%')
# Add "beyond" indicators at the edges
cb_ticks.insert(0, -0.98)
cb_labels.insert(0, '< −50%')
cb_ticks.append(0.98)
cb_labels.append('> +50%')

for g_idx, (group_name, z_raw) in enumerate(grid_data.items()):
    r = g_idx // 2 + 1
    c = g_idx % 2 + 1
    
    z_sc = np.vectorize(soft_clamp)(z_raw)
    
    # Annotation text: actual values
    ann_text = []
    for s_idx in range(z_raw.shape[0]):
        row_text = []
        for m_idx in range(z_raw.shape[1]):
            v = z_raw[s_idx, m_idx]
            if np.isfinite(v):
                if abs(v) >= 100:
                    row_text.append(f'{v:+.0f}%')
                elif abs(v) >= 10:
                    row_text.append(f'{v:+.1f}%')
                else:
                    row_text.append(f'{v:+.2f}%')
            else:
                row_text.append('')
        ann_text.append(row_text)
    
    show_colorbar = (g_idx == 1)
    
    fig_sig.add_trace(
        go.Heatmap(
            z=z_sc,
            x=MODELS_FOR_ANALYSIS,
            y=signature_labels,
            text=ann_text,
            texttemplate='%{text}',
            textfont=dict(size=11),
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=show_colorbar,
            colorbar=dict(
                title='Avg error',
                tickvals=cb_ticks,
                ticktext=cb_labels,
                len=0.9,
                y=0.5,
            ) if show_colorbar else None,
        ),
        row=r, col=c
    )

fig_sig.update_layout(
    title="<b>How well does each model reproduce hydrologic signatures across objective emphasis groups?</b><br>" +
          "<sup>Avg % error from observed — blue = underestimate, red = overestimate, white = unbiased; "
          "colour detail concentrated near zero</sup>",
    height=700,
    width=1100,
    margin=dict(t=100),
)
fig_sig.show()

# %% [markdown]
# ### 10.5 Traditional Error Metrics Comparison

# %%
# RMSE, MAE, PBIAS comparison across all objectives (ordered by flow emphasis)
traditional_metrics = ['RMSE', 'MAE', 'PBIAS']

# Question-framed titles per metric
metric_questions = {
    'RMSE': 'Which model–objective combination minimises RMSE?',
    'MAE': 'Which model–objective combination minimises MAE?',
    'PBIAS': 'Which model–objective combination is closest to zero percent bias?',
}
metric_subtitles = {
    'RMSE': 'Lower is better (ML/day)',
    'MAE': 'Lower is better (ML/day)',
    'PBIAS': 'Closer to 0 is better (%)',
}

for metric in traditional_metrics:
    data = []
    for obj_key in OBJ_KEYS_ORDERED:
        row_data = []
        for model_name in MODELS_FOR_ANALYSIS:
            if obj_key in all_metrics.get(model_name, {}) and metric in all_metrics[model_name][obj_key]:
                row_data.append(all_metrics[model_name][obj_key][metric])
            else:
                row_data.append(np.nan)
        data.append(row_data)
    
    if metric == 'PBIAS':
        colorscale = 'RdBu'
        zmid = 0
        zmin, zmax = -100, 100
    else:
        colorscale = 'Viridis'
        zmid = None
        zmin, zmax = None, None
    
    heatmap_kw = dict(
        z=data,
        x=MODELS_FOR_ANALYSIS,
        y=OBJ_LABELS_ORDERED,
        colorscale=colorscale,
        zmid=zmid,
        text=np.round(data, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    )
    if zmin is not None and zmax is not None:
        heatmap_kw['zmin'] = zmin
        heatmap_kw['zmax'] = zmax
    fig = go.Figure(data=go.Heatmap(**heatmap_kw))
    
    for g_idx, (start_row_idx, group_label) in enumerate(OBJ_GROUPS):
        if start_row_idx > 0:
            fig.add_hline(y=start_row_idx - 0.5, line_width=2, line_color='black', opacity=0.6)
        mid_idx = (start_row_idx + (OBJ_GROUPS[g_idx + 1][0] if g_idx + 1 < len(OBJ_GROUPS) else len(OBJ_ORDERED))) / 2
        fig.add_annotation(
            x=-0.20, y=OBJ_LABELS_ORDERED[int(mid_idx)],
            xref='paper', yref='y',
            text=f"<b>{group_label}</b>",
            showarrow=False, xanchor='right',
            font=dict(size=12, color='#444'),
        )
    
    fig.update_layout(
        title=f"<b>{metric_questions[metric]}</b><br><sup>{metric_subtitles[metric]}</sup>",
        xaxis_title="Model",
        yaxis_title="Calibration Objective",
        height=600,
        width=1000,
        margin=dict(l=380),
    )
    fig.show()

# %% [markdown]
# ### 10.6 Model Improvement Over Sacramento

# %%
# GR improvement over Sacramento — one panel per GR model
# Each panel: y-axis = calibration objectives, x-axis = diagnostic metrics
# Color = sign-normalised improvement (blue = GR better, red = Sacramento better)

# Skill / efficiency diagnostics (comparable 0–1 scale)
# SDEB is a minimize metric — flip sign so positive diff = GR better (consistent with all others)
DIAG_SKILL = [
    ('NSE',        'NSE'),
    ('KGE',        'KGE'),
    ("KGE_np",     "KGE'"),
    ('Pearson_r',  'Pearson r'),
    ('Spearman_r', 'Spearman ρ'),
    ('SqrtNSE',    '√NSE'),
    ('KGE_sqrt',   'KGE-√Q'),
    ('SDEB',       'SDEB'),
    ('LogNSE',     'log-NSE'),
    ('KGE_log',    'KGE-log'),
    ('InvNSE',     '1/Q NSE'),
    ('KGE_inv',    'KGE-1/Q'),
]
DIAG_MINIMIZE = {'SDEB'}  # metrics where lower = better

DIAG_SKILL_GROUPS = [
    (0,  'High-flow'),
    (5,  'Balanced (√Q)'),
    (8,  'Low-flow (log Q)'),
    (10, 'Very-low-flow (1/Q)'),
]

diag_labels = [lbl for _, lbl in DIAG_SKILL]
n_diag = len(DIAG_SKILL)
n_obj = len(OBJ_KEYS_ORDERED)

fig_imp = make_subplots(
    rows=3, cols=1,
    subplot_titles=[f'<b>{m}</b> vs Sacramento' for m in MODELS_GR_ONLY],
    vertical_spacing=0.04,
    shared_xaxes=True,
)

# Compute improvement matrices: shape (n_obj, n_diag) per model
KNEE_IMP = 0.15
all_raw = {}
for m_idx, gr_model in enumerate(MODELS_GR_ONLY):
    z_raw = np.full((n_obj, n_diag), np.nan)
    for d_idx, (dkey, dlbl) in enumerate(DIAG_SKILL):
        for o_idx, obj_key in enumerate(OBJ_KEYS_ORDERED):
            sac_val = all_metrics.get('Sacramento', {}).get(obj_key, {}).get(dkey, np.nan)
            gr_val  = all_metrics.get(gr_model, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(sac_val) and np.isfinite(gr_val):
                diff = gr_val - sac_val
                z_raw[o_idx, d_idx] = -diff if dkey in DIAG_MINIMIZE else diff
    all_raw[gr_model] = z_raw

# Colorbar ticks
cb_vals = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
cb_ticks_imp, cb_labels_imp = [], []
for v in reversed(cb_vals[1:]):
    cb_ticks_imp.append(soft_clamp(-v, k=KNEE_IMP))
    cb_labels_imp.append(f'−{v}')
cb_ticks_imp.append(0.0)
cb_labels_imp.append('0')
for v in cb_vals[1:]:
    cb_ticks_imp.append(soft_clamp(v, k=KNEE_IMP))
    cb_labels_imp.append(f'+{v}')

for m_idx, gr_model in enumerate(MODELS_GR_ONLY):
    r = m_idx + 1
    z_raw = all_raw[gr_model]
    z_sc = np.vectorize(lambda x: soft_clamp(x, k=KNEE_IMP))(z_raw)

    # Cell annotations: show actual improvement values
    ann_text = []
    for o_idx in range(n_obj):
        row_text = []
        for d_idx in range(n_diag):
            v = z_raw[o_idx, d_idx]
            if np.isfinite(v):
                row_text.append(f'{v:+.3f}')
            else:
                row_text.append('')
        ann_text.append(row_text)

    fig_imp.add_trace(
        go.Heatmap(
            z=z_sc,
            x=diag_labels,
            y=OBJ_LABELS_ORDERED,
            text=ann_text,
            texttemplate='%{text}',
            textfont=dict(size=9),
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(m_idx == 0),
            colorbar=dict(
                title=dict(text='GR − Sac', side='right'),
                tickvals=cb_ticks_imp,
                ticktext=cb_labels_imp,
                len=0.85,
                y=0.5,
                x=1.02,
                xpad=10,
            ) if m_idx == 0 else None,
        ),
        row=r, col=1,
    )

    # Objective-group horizontal separators (solid white lines for clear separation)
    for start_idx, _ in OBJ_GROUPS:
        if start_idx > 0:
            fig_imp.add_hline(
                y=start_idx - 0.5,
                line_color='white', line_width=3,
                row=r, col=1,
            )

    # Objective-group labels (left side — HORIZONTAL text, tucked between y-labels and heatmap edge)
    for g_i, (start_idx, grp_label) in enumerate(OBJ_GROUPS):
        next_starts = [s for s, _ in OBJ_GROUPS if s > start_idx]
        end_idx = next_starts[0] if next_starts else n_obj
        mid_frac = (start_idx + end_idx - 1) / 2 / (n_obj - 1) if n_obj > 1 else 0.5
        fig_imp.add_annotation(
            text=f'<b>{grp_label}</b>',
            x=-0.08, y=mid_frac,
            xref='x domain', yref='y domain',
            showarrow=False,
            font=dict(size=9, color='#555'),
            textangle=0,
            xanchor='right',
            row=r, col=1,
        )

# Diagnostic-group labels (top of figure, above x-axis ticks)
for dg_start, dg_label in DIAG_SKILL_GROUPS:
    next_dg = [s for s, _ in DIAG_SKILL_GROUPS if s > dg_start]
    dg_end = next_dg[0] if next_dg else n_diag
    mid_idx = (dg_start + dg_end - 1) / 2
    x_frac = (mid_idx + 0.5) / n_diag
    fig_imp.add_annotation(
        text=f'<b>{dg_label}</b>',
        x=x_frac, y=1.03,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color='#444'),
    )

# Diagnostic-group vertical separators (solid white lines for clear separation)
for dg_start, _ in DIAG_SKILL_GROUPS:
    if dg_start > 0:
        for r in range(1, 4):
            fig_imp.add_vline(
                x=dg_start - 0.5,
                line_color='white', line_width=3,
                row=r, col=1,
            )

fig_imp.update_layout(
    title="<b>Do GR models improve over Sacramento, and for which diagnostics?</b><br>"
          "<sup>Rows (↕) = calibration objective used (grouped by flow emphasis) · "
          "Columns (↔) = diagnostic metric evaluated (grouped by flow emphasis)<br>"
          "Each cell = GR value − Sacramento value for that metric · "
          "Blue = GR outperforms Sacramento · Red = Sacramento outperforms GR · White ≈ no difference<br>"
          "Read across a row to see how one calibration objective affects all diagnostics · "
          "Read down a column to see how one diagnostic responds to different objectives</sup>",
    height=1800,
    width=1700,
    margin=dict(l=250, t=220, r=160, b=100),
)

# Only show x-axis tick labels on the bottom panel
fig_imp.update_xaxes(showticklabels=False, row=1, col=1)
fig_imp.update_xaxes(showticklabels=False, row=2, col=1)
fig_imp.update_xaxes(tickangle=-45, tickfont=dict(size=11), row=3, col=1)

fig_imp.show()

# %% [markdown]
# ### 10.7 Radar Charts: Model Profiles by Objective Emphasis Group

# %%
# Radar charts: comprehensive diagnostic profile per model, grouped by objective emphasis
# All 12 skill metrics on each radar, one panel per emphasis group, combined into a single figure

# Full set of diagnostic metrics for radar axes (ordered by flow emphasis group)
RADAR_METRICS = [
    # High-flow
    ('NSE',        'NSE'),
    ('KGE',        'KGE'),
    ("KGE_np",     "KGE'"),
    ('Pearson_r',  'Pearson r'),
    ('Spearman_r', 'Spearman ρ'),
    # Balanced
    ('SqrtNSE',    '√NSE'),
    ('KGE_sqrt',   'KGE-√Q'),
    ('SDEB',       'SDEB*'),
    # Low-flow
    ('LogNSE',     'log-NSE'),
    ('KGE_log',    'KGE-log'),
    # Very-low-flow
    ('InvNSE',     '1/Q NSE'),
    ('KGE_inv',    'KGE-1/Q'),
]
radar_keys = [k for k, _ in RADAR_METRICS]
radar_labels = [lbl for _, lbl in RADAR_METRICS]
RADAR_MINIMIZE = {'SDEB'}  # metrics where lower = better

emphasis_items = list(OBJ_EMPHASIS_GROUPS.items())

# Pre-compute SDEB range across ALL models and objectives for log + reversed min-max normalisation
# log1p transform stretches differences among good (low) SDEB values, compresses the bad tail
all_sdeb_vals = []
for model in MODELS_FOR_ANALYSIS:
    for obj_key in OBJ_KEYS_ORDERED:
        v = all_metrics.get(model, {}).get(obj_key, {}).get('SDEB', np.nan)
        if np.isfinite(v):
            all_sdeb_vals.append(v)
sdeb_log = np.log1p(all_sdeb_vals) if all_sdeb_vals else [0.0]
sdeb_log_min = float(np.min(sdeb_log))
sdeb_log_max = float(np.max(sdeb_log))
sdeb_log_range = sdeb_log_max - sdeb_log_min if sdeb_log_max > sdeb_log_min else 1.0

# Build 2×2 grid using four separate polar subplots
fig_radar = go.Figure()

# Domain positions for 2×2 grid (x0, x1, y0, y1) with gaps
domains = [
    {'x': [0.02, 0.46], 'y': [0.53, 0.97]},   # top-left
    {'x': [0.54, 0.98], 'y': [0.53, 0.97]},   # top-right
    {'x': [0.02, 0.46], 'y': [0.03, 0.47]},   # bottom-left
    {'x': [0.54, 0.98], 'y': [0.03, 0.47]},   # bottom-right
]
polar_names = ['polar', 'polar2', 'polar3', 'polar4']

for g_idx, (group_name, group_keys) in enumerate(emphasis_items):
    polar_key = polar_names[g_idx]

    for model in MODELS_FOR_ANALYSIS:
        values = []
        for mkey in radar_keys:
            metric_vals = []
            for obj_key in group_keys:
                if obj_key in all_metrics.get(model, {}) and mkey in all_metrics[model][obj_key]:
                    val = all_metrics[model][obj_key][mkey]
                    if np.isfinite(val):
                        if mkey in RADAR_MINIMIZE:
                            log_val = np.log1p(val)
                            metric_vals.append(max(0, (sdeb_log_max - log_val) / sdeb_log_range))
                        else:
                            metric_vals.append(max(val, 0))
            values.append(np.mean(metric_vals) if metric_vals else 0)

        values_closed = values + [values[0]]
        labels_closed = radar_labels + [radar_labels[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill='toself',
            name=model,
            line=dict(color=MODEL_COLORS[model], width=2),
            opacity=0.5,
            legendgroup=model,
            showlegend=(g_idx == 0),
            subplot=polar_key,
        ))

# Configure each polar subplot
for g_idx, (group_name, _) in enumerate(emphasis_items):
    polar_key = polar_names[g_idx]
    dom = domains[g_idx]

    polar_config = dict(
        domain=dom,
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0.2', '0.4', '0.6', '0.8', '1.0'],
            tickfont=dict(size=8, color='#999'),
            gridcolor='rgba(200,200,200,0.3)',
        ),
        angularaxis=dict(
            tickfont=dict(size=10),
            rotation=90,
            direction='clockwise',
            gridcolor='rgba(200,200,200,0.3)',
        ),
        bgcolor='rgba(245,245,250,0.3)',
    )
    fig_radar.update_layout(**{polar_key: polar_config})

    # Panel title annotation (positioned above each radar with clearance)
    cx = (dom['x'][0] + dom['x'][1]) / 2
    ty = dom['y'][1] + 0.025
    fig_radar.add_annotation(
        text=f'<b>{group_name}</b>',
        x=cx, y=ty,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=13, color='#333'),
        xanchor='center',
    )

fig_radar.update_layout(
    title="<b>Which model has the strongest overall diagnostic profile under each objective emphasis?</b><br>"
          "<sup>Each axis = one of 12 skill diagnostics (all normalised to 0–1, higher = better) · Axes grouped clockwise: "
          "High-flow → Balanced → Low-flow → Very-low-flow<br>"
          "Larger polygon = stronger model · A perfectly round polygon = balanced performance across all diagnostics<br>"
          "SDEB* = log-scaled reversed normalisation (best SDEB → 1, worst → 0; log stretches differences among good models) · "
          "Negative scores clipped to 0 · Values are averages across objectives in each group</sup>",
    height=1400,
    width=1500,
    margin=dict(t=160, b=60, l=100, r=100),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.02,
        xanchor='center',
        x=0.5,
        font=dict(size=11),
    ),
)
fig_radar.show()

# %% [markdown]
# ### 10.8 Comprehensive Performance Summary
#
# **What is the mean diagnostic score for each model across all 13 calibration objectives?**

# %%
# Comprehensive performance summary — heatmap of mean diagnostic scores
# Rows = models, Columns = all diagnostics (grouped by type), values = mean across 13 objectives

SUMMARY_DIAG = [
    # (key, label, higher_is_better)
    ('NSE',        'NSE',         True),
    ('KGE',        'KGE',         True),
    ("KGE_np",     "KGE'",        True),
    ('Pearson_r',  'Pearson r',   True),
    ('Spearman_r', 'Spearman ρ',  True),
    ('SqrtNSE',    '√NSE',        True),
    ('KGE_sqrt',   'KGE-√Q',      True),
    ('LogNSE',     'log-NSE',     True),
    ('KGE_log',    'KGE-log',     True),
    ('InvNSE',     '1/Q NSE',     True),
    ('KGE_inv',    'KGE-1/Q',     True),
    ('SDEB',       'SDEB',        False),
    ('RMSE',       'RMSE',        False),
    ('MAE',        'MAE',         False),
    ('PBIAS',      '|PBIAS|',     False),  # use absolute value
]

SUMMARY_DIAG_GROUPS = [
    (0,  'High-flow'),
    (5,  'Balanced'),
    (7,  'Low-flow'),
    (9,  'Very-low-flow'),
    (11, 'Error metrics'),
]

summary_labels = [lbl for _, lbl, _ in SUMMARY_DIAG]
n_sd = len(SUMMARY_DIAG)

# Compute mean values across all 13 objectives for each model × diagnostic
z_summary = np.full((len(MODELS_FOR_ANALYSIS), n_sd), np.nan)
ann_summary = []

for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
    row_text = []
    for d_idx, (dkey, dlbl, higher_better) in enumerate(SUMMARY_DIAG):
        vals = []
        for obj_key in OBJ_KEYS_ORDERED:
            v = all_metrics.get(model_name, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(v):
                if dkey == 'PBIAS':
                    vals.append(abs(v))
                else:
                    vals.append(v)
        if vals:
            mean_val = np.mean(vals)
            z_summary[m_idx, d_idx] = mean_val
            if dkey in ['RMSE', 'MAE']:
                row_text.append(f'{mean_val:.1f}')
            elif dkey == 'SDEB':
                row_text.append(f'{mean_val:.2f}')
            elif dkey == 'PBIAS':
                row_text.append(f'{mean_val:.1f}%')
            else:
                row_text.append(f'{mean_val:.3f}')
        else:
            row_text.append('')
    ann_summary.append(row_text)

# Colour ranking: 0 = best model for that diagnostic, 1 = worst
# Normalised per column so green = best, red = worst (one clear direction)
z_rank = np.full_like(z_summary, np.nan)
for d_idx, (_, _, higher_better) in enumerate(SUMMARY_DIAG):
    col_vals = z_summary[:, d_idx]
    valid = np.isfinite(col_vals)
    if valid.any():
        if higher_better:
            best = np.nanmax(col_vals)
            worst = np.nanmin(col_vals)
        else:
            best = np.nanmin(col_vals)
            worst = np.nanmax(col_vals)
        rng = abs(best - worst) if abs(best - worst) > 1e-12 else 1.0
        for m_idx in range(len(MODELS_FOR_ANALYSIS)):
            if np.isfinite(col_vals[m_idx]):
                if higher_better:
                    z_rank[m_idx, d_idx] = (best - col_vals[m_idx]) / rng  # 0 = best (highest), 1 = worst
                else:
                    z_rank[m_idx, d_idx] = (col_vals[m_idx] - best) / rng  # 0 = best (lowest), 1 = worst

fig_summary = go.Figure()
fig_summary.add_trace(go.Heatmap(
    z=z_rank,
    x=summary_labels,
    y=MODELS_FOR_ANALYSIS,
    text=ann_summary,
    texttemplate='%{text}',
    textfont=dict(size=13, color='black'),
    colorscale=[
        [0.0, '#1a9641'],   # dark green — best
        [0.25, '#a6d96a'],  # light green
        [0.5, '#ffffbf'],   # pale yellow — middle
        [0.75, '#fdae61'],  # orange
        [1.0, '#d7191c'],   # red — worst
    ],
    zmin=0,
    zmax=1,
    showscale=True,
    colorbar=dict(
        title='Relative<br>performance',
        tickvals=[0, 0.5, 1],
        ticktext=['Best', 'Middle', 'Worst'],
        len=0.7,
        y=0.5,
    ),
))

# Diagnostic-group vertical separators and labels
for dg_start, dg_label in SUMMARY_DIAG_GROUPS:
    if dg_start > 0:
        fig_summary.add_vline(x=dg_start - 0.5, line_color='white', line_width=3)
    next_dg = [s for s, _ in SUMMARY_DIAG_GROUPS if s > dg_start]
    dg_end = next_dg[0] if next_dg else n_sd
    mid_x = (dg_start + dg_end - 1) / 2
    fig_summary.add_annotation(
        text=f'<b>{dg_label}</b>',
        x=summary_labels[int(round(mid_x))], y=1.12,
        xref='x', yref='paper',
        showarrow=False,
        font=dict(size=12, color='#444'),
    )

fig_summary.update_layout(
    title="<b>What is the mean diagnostic score for each model across all 13 calibration objectives?</b><br>"
          "<sup>Colour shows relative rank per column: green = best among the 4 models, red = worst · "
          "Cell values = actual mean score<br>"
          "Skill metrics (NSE, KGE, etc.): higher is better · "
          "Error metrics (SDEB, RMSE, MAE, |PBIAS|): lower is better</sup>",
    height=450,
    width=1500,
    margin=dict(t=140, b=100, l=120, r=100),
    xaxis=dict(side='bottom', tickangle=-45, tickfont=dict(size=11)),
    yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
)
fig_summary.show()

# %% [markdown]
# ---
# ## Step 11: Final Summary and Recommendations
#
# **Which model wins most often, and what does the data say about model selection for different applications?**

# %%
# Data-driven rankings — which model is best for each flow-emphasis group?
# Two panels: (1) Normalised ranking heatmap, (2) Win count bars

RANKING_DIAG = {
    'High-flow': [('NSE', True), ('KGE', True), ("KGE_np", True), ('Pearson_r', True), ('Spearman_r', True)],
    'Balanced (√Q)': [('SqrtNSE', True), ('KGE_sqrt', True), ('SDEB', False)],
    'Low-flow (log Q)': [('LogNSE', True), ('KGE_log', True)],
    'Very-low-flow (1/Q)': [('InvNSE', True), ('KGE_inv', True)],
    'Volume accuracy': [('PBIAS', False), ('RMSE', False), ('MAE', False)],
}

# Compute aggregate score per model per group (mean across objectives × diagnostics)
group_scores = {}
for grp_name, diag_list in RANKING_DIAG.items():
    model_scores = {}
    for model_name in MODELS_FOR_ANALYSIS:
        vals = []
        for dkey, higher_better in diag_list:
            for obj_key in OBJ_KEYS_ORDERED:
                v = all_metrics.get(model_name, {}).get(obj_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    if dkey == 'PBIAS':
                        vals.append(-abs(v))
                    elif not higher_better:
                        vals.append(-v)
                    else:
                        vals.append(v)
        model_scores[model_name] = np.mean(vals) if vals else np.nan
    group_scores[grp_name] = model_scores

# Normalise scores within each group to [0, 1] (1 = best, 0 = worst among 4 models)
grp_names = list(RANKING_DIAG.keys())
z_norm = np.full((len(grp_names), len(MODELS_FOR_ANALYSIS)), np.nan)
ann_rank = []

for g_idx, grp_name in enumerate(grp_names):
    scores = group_scores[grp_name]
    vals = np.array([scores[m] for m in MODELS_FOR_ANALYSIS])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    rng = vmax - vmin if abs(vmax - vmin) > 1e-12 else 1.0

    # Rank models (1 = best)
    ranked = sorted(MODELS_FOR_ANALYSIS, key=lambda m: scores[m], reverse=True)
    rank_map = {m: r + 1 for r, m in enumerate(ranked)}

    row_text = []
    for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
        norm_val = (scores[model_name] - vmin) / rng  # 0 = worst, 1 = best
        z_norm[g_idx, m_idx] = norm_val
        row_text.append(f'#{rank_map[model_name]}')
    ann_rank.append(row_text)

# Count how many diagnostic-objective cells each model wins (across all 12 skill metrics)
win_counts = {m: 0 for m in MODELS_FOR_ANALYSIS}
total_cells = 0
for dkey, dlbl in DIAG_SKILL:
    for obj_key in OBJ_KEYS_ORDERED:
        cell_vals = {}
        for model_name in MODELS_FOR_ANALYSIS:
            v = all_metrics.get(model_name, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(v):
                if dkey in DIAG_MINIMIZE:
                    cell_vals[model_name] = -v
                else:
                    cell_vals[model_name] = v
        if cell_vals:
            winner = max(cell_vals, key=cell_vals.get)
            win_counts[winner] += 1
            total_cells += 1

# Also count wins per emphasis group
win_by_group = {g: {m: 0 for m in MODELS_FOR_ANALYSIS} for g in grp_names}
group_total = {g: 0 for g in grp_names}
for grp_name, diag_list in RANKING_DIAG.items():
    for dkey, higher_better in diag_list:
        for obj_key in OBJ_KEYS_ORDERED:
            cell_vals = {}
            for model_name in MODELS_FOR_ANALYSIS:
                v = all_metrics.get(model_name, {}).get(obj_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    if dkey == 'PBIAS':
                        cell_vals[model_name] = -abs(v)
                    elif not higher_better:
                        cell_vals[model_name] = -v
                    else:
                        cell_vals[model_name] = v
            if cell_vals:
                winner = max(cell_vals, key=cell_vals.get)
                win_by_group[grp_name][winner] += 1
                group_total[grp_name] += 1

# Build figure: ranking heatmap (left) + win count bars (right)
fig_rank = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        '<b>Which model ranks best for each flow emphasis?</b>',
        '<b>How often does each model win?</b>',
    ],
    horizontal_spacing=0.15,
    column_widths=[0.55, 0.45],
    specs=[[{'type': 'heatmap'}, {'type': 'bar'}]],
)

# LEFT: ranking heatmap — rows = emphasis groups, cols = models
fig_rank.add_trace(
    go.Heatmap(
        z=z_norm,
        x=MODELS_FOR_ANALYSIS,
        y=grp_names,
        text=ann_rank,
        texttemplate='%{text}',
        textfont=dict(size=16, color='black'),
        colorscale=[
            [0.0, '#d7191c'],
            [0.25, '#fdae61'],
            [0.5, '#ffffbf'],
            [0.75, '#a6d96a'],
            [1.0, '#1a9641'],
        ],
        zmin=0,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title='Score',
            tickvals=[0, 0.5, 1],
            ticktext=['Worst', 'Middle', 'Best'],
            len=0.8,
            y=0.5,
            x=0.48,
        ),
    ),
    row=1, col=1,
)

# RIGHT: stacked bar chart — wins per model, broken down by emphasis group
grp_colors = {
    'High-flow': '#1f77b4',
    'Balanced (√Q)': '#2ca02c',
    'Low-flow (log Q)': '#ff7f0e',
    'Very-low-flow (1/Q)': '#d62728',
    'Volume accuracy': '#9467bd',
}

sorted_models = sorted(MODELS_FOR_ANALYSIS, key=lambda m: win_counts[m])

for grp_name in grp_names:
    fig_rank.add_trace(
        go.Bar(
            name=grp_name,
            y=sorted_models,
            x=[win_by_group[grp_name][m] for m in sorted_models],
            orientation='h',
            marker_color=grp_colors[grp_name],
            showlegend=True,
        ),
        row=1, col=2,
    )

# Add total annotation on each bar
for m in sorted_models:
    total_w = win_counts[m]
    fig_rank.add_annotation(
        text=f'<b>{total_w}/{total_cells} ({100*total_w/total_cells:.0f}%)</b>',
        x=total_w + 2,
        y=m,
        xref='x2', yref='y2',
        showarrow=False,
        font=dict(size=10),
        xanchor='left',
    )

fig_rank.update_xaxes(title_text='Wins', row=1, col=2)
fig_rank.update_layout(
    barmode='stack',
)

fig_rank.update_layout(
    title="<b>Which model should you choose, and for what application?</b><br>"
          "<sup>Left: normalised ranking per flow-emphasis group (1 = best among 4 models, 0 = worst) — "
          "scores normalised within each row so all groups are comparable<br>"
          "Right: win counts across all diagnostic × objective combinations, broken down by emphasis group — "
          "a model 'wins' when it has the best value for that cell</sup>",
    height=500,
    width=1500,
    margin=dict(t=140, b=60, l=120, r=140),
    legend=dict(
        orientation='h', yanchor='bottom', y=-0.18,
        xanchor='center', x=0.72,
        font=dict(size=10),
        title_text='Emphasis group (win breakdown)',
    ),
)
fig_rank.show()

# %%
# Data-driven best-model table per emphasis group
print()
print("=" * 100)
print("DATA-DRIVEN MODEL RECOMMENDATIONS")
print("=" * 100)
print()
print(f"{'Flow emphasis':<30} {'Best model':<14} {'Score':>8}  Interpretation")
print("-" * 100)

for grp_name, model_scores in group_scores.items():
    best_model = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]
    runner_up = sorted(model_scores, key=model_scores.get, reverse=True)[1]
    ru_score = model_scores[runner_up]
    margin = best_score - ru_score

    if grp_name == 'High-flow focus':
        interp = "Best for flood forecasting and peak flow estimation"
    elif grp_name == 'Balanced focus (√Q)':
        interp = "Best for general-purpose water supply modelling"
    elif grp_name == 'Low-flow focus (log Q)':
        interp = "Best for baseflow and dry-season flow estimation"
    elif grp_name == 'Very-low-flow focus (1/Q)':
        interp = "Best for drought analysis and environmental flow"
    else:
        interp = "Best for overall volume accuracy and error minimisation"

    print(f"  {grp_name:<28} {best_model:<14} {best_score:>+8.4f}  {interp}")
    print(f"  {'':28} (runner-up: {runner_up}, margin: {margin:+.4f})")
    print()

overall_winner = max(win_counts, key=win_counts.get)
print(f"  OVERALL WINNER (most wins):  {overall_winner} "
      f"with {win_counts[overall_winner]}/{total_cells} wins "
      f"({100*win_counts[overall_winner]/total_cells:.0f}%)")
print()

# Parameter count
print("\n  Model complexity:")
for model_name in MODELS_FOR_ANALYSIS:
    if ALL_RESULTS.get(model_name):
        n_params = len(list(ALL_RESULTS[model_name].values())[0].best_parameters)
    else:
        n_params = '?'
    print(f"    {model_name:<12}  {n_params} parameters")

print(f"""
{'=' * 100}
KEY FINDINGS (data-driven)
{'=' * 100}

  1. The best model varies by application — no single model dominates all diagnostics
  2. Win counts reveal which model is most consistently strong across the full
     diagnostic × objective matrix ({total_cells} combinations)
  3. Models with fewer parameters offer parsimony advantages for ungauged catchments
  4. The margin between models is often small — model structural uncertainty may be
     less important than objective function choice for this catchment
""")

# %% [markdown]
# ---
# ## Step 12: Report Cards by Objective (Sacramento vs GR4J vs GR5J vs GR6J)
#
# For each calibration objective, one 4-panel report card: linear hydrograph,
# log hydrograph, scatter (log axes), and flow duration curve. Each panel
# compares Observed with Sacramento, GR4J, GR5J, and GR6J (4 models).

# %%
# Report card for each objective: 4 panels (linear hydrograph, log hydrograph, scatter log, FDC)
# Use ordered objectives with descriptive labels
obj_key_to_label_rc = dict(OBJ_ORDERED)

for obj_key in OBJ_KEYS_ORDERED:
    if not all(obj_key in all_simulations.get(m, {}) for m in MODELS_FOR_ANALYSIS):
        continue

    obj_label = obj_key_to_label_rc[obj_key]
    obs = obs_flow
    dates = comparison_dates
    series = {'Observed': obs}
    for model_name in MODELS_FOR_ANALYSIS:
        series[model_name] = all_simulations[model_name][obj_key]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Linear hydrograph',
            'Log hydrograph',
            'Observed vs simulated (log)',
            'Flow duration curve'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # 1. Linear hydrograph (row=1, col=1)
    for name, y in series.items():
        fig.add_trace(
            go.Scatter(
                x=dates, y=y, name=name,
                line=dict(color=MODEL_COLORS.get(name, '#333'), width=1.5 if name == 'Observed' else 1),
                legendgroup=name
            ),
            row=1, col=1
        )

    # 2. Log hydrograph (row=1, col=2)
    for name, y in series.items():
        y_safe = np.where(np.asarray(y) > 0.01, np.asarray(y), np.nan)
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_safe, name=name,
                line=dict(color=MODEL_COLORS.get(name, '#333'), width=1.5 if name == 'Observed' else 1),
                legendgroup=name, showlegend=False
            ),
            row=1, col=2
        )

    # 3. Scatter observed vs simulated, log axes (row=2, col=1)
    obs_arr = np.asarray(obs)
    valid = ~(np.isnan(obs_arr) | (obs_arr <= 0))
    obs_valid = obs_arr[valid]
    for model_name in MODELS_FOR_ANALYSIS:
        sim_arr = np.asarray(series[model_name])
        sim_valid = np.where(sim_arr[valid] > 0.01, sim_arr[valid], np.nan)
        fig.add_trace(
            go.Scatter(
                x=obs_valid, y=sim_valid, name=model_name, mode='markers',
                marker=dict(size=2, color=MODEL_COLORS[model_name], opacity=0.6),
                legendgroup=model_name, showlegend=False
            ),
            row=2, col=1
        )
    o_min, o_max = np.nanmin(obs_valid), np.nanmax(obs_valid)
    fig.add_trace(
        go.Scatter(
            x=[o_min, o_max], y=[o_min, o_max], name='1:1',
            line=dict(color='gray', dash='dash', width=1)
        ),
        row=2, col=1
    )

    # 4. Flow duration curve (row=2, col=2)
    for name, y in series.items():
        y_arr = np.asarray(y)
        y_clean = y_arr[~(np.isnan(y_arr) | (y_arr <= 0))]
        if len(y_clean) == 0:
            continue
        sorted_flow = np.sort(y_clean)[::-1]
        exceedance_rc = np.arange(1, len(sorted_flow) + 1) / (len(sorted_flow) + 1) * 100
        fig.add_trace(
            go.Scatter(
                x=exceedance_rc, y=sorted_flow, name=name,
                line=dict(color=MODEL_COLORS.get(name, '#333'), width=2 if name == 'Observed' else 1.5),
                legendgroup=name, showlegend=False
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Flow (ML/day)', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=2)
    fig.update_yaxes(title_text='Flow (ML/day)', type='log', row=1, col=2)
    fig.update_xaxes(title_text='Observed (ML/day)', type='log', row=2, col=1)
    fig.update_yaxes(title_text='Simulated (ML/day)', type='log', row=2, col=1)
    fig.update_xaxes(title_text='Exceedance (%)', row=2, col=2)
    fig.update_yaxes(title_text='Flow (ML/day)', type='log', row=2, col=2)

    fig.update_layout(
        title=f"<b>How do the four models compare when calibrated with {obj_label}?</b><br>" +
              "<sup>Sacramento vs GR4J vs GR5J vs GR6J</sup>",
        height=700,
        width=1000,
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1.0)
    )
    fig.show()

# %% [markdown]
# ---
# ## Export Results

# %%
# Export calibration summary
export_summary = []
for model_name in MODELS_FOR_ANALYSIS:
    for obj_name in OBJECTIVES.keys():
        if obj_name in ALL_RESULTS.get(model_name, {}):
            result = ALL_RESULTS[model_name][obj_name]
            export_summary.append({
                'model': model_name,
                'n_params': len(result.best_parameters),
                'objective': obj_name,
                'best_value': result.best_objective,
                'runtime_s': ALL_TIMES[model_name].get(obj_name, np.nan),
            })

summary_df = pd.DataFrame(export_summary)
summary_df.to_csv('../test_data/model_comparison_all_13_objectives.csv', index=False)
print("Calibration summary saved to: test_data/model_comparison_all_13_objectives.csv")

# Export comprehensive metrics
metrics_export = []
for model_name in MODELS_FOR_ANALYSIS:
    for obj_name in OBJECTIVES.keys():
        if obj_name in all_metrics.get(model_name, {}):
            row = {'model': model_name, 'objective': obj_name}
            row.update(all_metrics[model_name][obj_name])
            metrics_export.append(row)

metrics_df = pd.DataFrame(metrics_export)
metrics_df.to_csv('../test_data/model_comparison_comprehensive_metrics.csv', index=False)
print("Comprehensive metrics saved to: test_data/model_comparison_comprehensive_metrics.csv")

# %%
print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE!")
print("=" * 70)
print(f"\nThis notebook compared:")
print(f"  • 4 models: GR4J, GR5J, GR6J, Sacramento")
print(f"  • 13 objective functions each")
print(f"  • 52 calibrations in analysis (39 GR + 13 Sacramento)")
print("\nAll results saved to test_data/ directory")
