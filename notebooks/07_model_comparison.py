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
    'nse': '../test_data/reports/410734_nse.pkl',
    'lognse': '../test_data/reports/410734_lognse.pkl',
    'invnse': '../test_data/reports/410734_invnse.pkl',
    'sqrtnse': '../test_data/reports/410734_sqrtnse.pkl',
    # KGE variants (4)
    'kge': '../test_data/reports/410734_kge.pkl',
    'kge_inv': '../test_data/reports/410734_kge_inv.pkl',
    'kge_sqrt': '../test_data/reports/410734_kge_sqrt.pkl',
    'kge_log': '../test_data/reports/410734_kge_log.pkl',
    # Non-parametric KGE variants (4)
    'kge_np': '../test_data/reports/410734_kge_np.pkl',
    'kge_np_inv': '../test_data/reports/410734_kge_np_inv.pkl',
    'kge_np_sqrt': '../test_data/reports/410734_kge_np_sqrt.pkl',
    'kge_np_log': '../test_data/reports/410734_kge_np_log.pkl',
    # Composite (1)
    'sdeb': '../test_data/reports/410734_sdeb.pkl',
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
    'GR5J': 5000,       # 5 params  
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
print(f"\nTotal NEW calibrations: 3 models × 13 objectives = 39")

# Model colors for plotting
MODEL_COLORS = {
    'GR4J': '#1f77b4',       # Blue
    'GR5J': '#2ca02c',       # Green
    'GR6J': '#ff7f0e',       # Orange
    'Sacramento': '#d62728', # Red
    'Observed': '#000000',   # Black
}

# %% [markdown]
# ---
# ## Step 5: Calibrate GR4J with All 13 Objectives
#
# Each calibration runs in a separate cell for clear progress tracking.

# %% [markdown]
# ### GR4J Calibrations (13 total)

# %%
# GR4J Calibration 1/13: NSE
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
report.save('../test_data/reports/410734_gr4j_nse')
print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 2/13: LogNSE
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
report.save('../test_data/reports/410734_gr4j_lognse')
print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 3/13: InvNSE
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
report.save('../test_data/reports/410734_gr4j_invnse')
print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 4/13: SqrtNSE
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
report.save('../test_data/reports/410734_gr4j_sqrtnse')
print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 5/13: KGE
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
report.save('../test_data/reports/410734_gr4j_kge')
print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 6/13: KGE_inv
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
report.save('../test_data/reports/410734_gr4j_kge_inv')
print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 7/13: KGE_sqrt
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
report.save('../test_data/reports/410734_gr4j_kge_sqrt')
print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 8/13: KGE_log
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
report.save('../test_data/reports/410734_gr4j_kge_log')
print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 9/13: KGE_np
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
report.save('../test_data/reports/410734_gr4j_kge_np')
print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 10/13: KGE_np_inv
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
report.save('../test_data/reports/410734_gr4j_kge_np_inv')
print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 11/13: KGE_np_sqrt
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
report.save('../test_data/reports/410734_gr4j_kge_np_sqrt')
print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 12/13: KGE_np_log
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
report.save('../test_data/reports/410734_gr4j_kge_np_log')
print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 13/13: SDEB
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
report.save('../test_data/reports/410734_gr4j_sdeb')
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
# GR5J Calibration 1/13: NSE
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['nse'] = result
gr5j_times['nse'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_nse')
print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 2/13: LogNSE
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['lognse'] = result
gr5j_times['lognse'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_lognse')
print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 3/13: InvNSE
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['invnse'] = result
gr5j_times['invnse'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_invnse')
print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 4/13: SqrtNSE
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['sqrtnse'] = result
gr5j_times['sqrtnse'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_sqrtnse')
print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 5/13: KGE
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge'] = result
gr5j_times['kge'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge')
print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 6/13: KGE_inv
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_inv'] = result
gr5j_times['kge_inv'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_inv')
print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 7/13: KGE_sqrt
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_sqrt'] = result
gr5j_times['kge_sqrt'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_sqrt')
print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 8/13: KGE_log
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_log'] = result
gr5j_times['kge_log'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_log')
print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 9/13: KGE_np
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_np'] = result
gr5j_times['kge_np'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_np')
print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 10/13: KGE_np_inv
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_np_inv'] = result
gr5j_times['kge_np_inv'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_np_inv')
print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 11/13: KGE_np_sqrt
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_np_sqrt'] = result
gr5j_times['kge_np_sqrt'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_np_sqrt')
print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 12/13: KGE_np_log
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['kge_np_log'] = result
gr5j_times['kge_np_log'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_kge_np_log')
print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 13/13: SDEB
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
result = runner.run_sceua_direct(max_evals=MAX_EVALS['GR5J'], seed=42, verbose=True, max_tolerant_iter=100, tolerance=1e-4)
elapsed = time.time() - start_time

gr5j_results['sdeb'] = result
gr5j_times['sdeb'] = elapsed

report = runner.create_report(result, catchment_info={'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2})
report.save('../test_data/reports/410734_gr5j_sdeb')
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
# GR6J Calibration 1/13: NSE
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
report.save('../test_data/reports/410734_gr6j_nse')
print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 2/13: LogNSE
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
report.save('../test_data/reports/410734_gr6j_lognse')
print(f"\n✓ Best LogNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 3/13: InvNSE
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
report.save('../test_data/reports/410734_gr6j_invnse')
print(f"\n✓ Best InvNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 4/13: SqrtNSE
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
report.save('../test_data/reports/410734_gr6j_sqrtnse')
print(f"\n✓ Best SqrtNSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 5/13: KGE
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
report.save('../test_data/reports/410734_gr6j_kge')
print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 6/13: KGE_inv
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
report.save('../test_data/reports/410734_gr6j_kge_inv')
print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 7/13: KGE_sqrt
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
report.save('../test_data/reports/410734_gr6j_kge_sqrt')
print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 8/13: KGE_log
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
report.save('../test_data/reports/410734_gr6j_kge_log')
print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 9/13: KGE_np
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
report.save('../test_data/reports/410734_gr6j_kge_np')
print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 10/13: KGE_np_inv
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
report.save('../test_data/reports/410734_gr6j_kge_np_inv')
print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 11/13: KGE_np_sqrt
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
report.save('../test_data/reports/410734_gr6j_kge_np_sqrt')
print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 12/13: KGE_np_log
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
report.save('../test_data/reports/410734_gr6j_kge_np_log')
print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 13/13: SDEB
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
report.save('../test_data/reports/410734_gr6j_sdeb')
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
print("CALIBRATION SUMMARY: ALL 4 MODELS × 13 OBJECTIVES")
print("=" * 140)

obj_names = list(OBJECTIVES.keys())
print(f"\n{'Model':<12} {'Params':>6}", end="")
for obj_name in obj_names:
    print(f" {obj_name:>10}", end="")
print()
print("-" * 140)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
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
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
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

# Define objective families
obj_families = {
    'NSE variants': ['nse', 'lognse', 'invnse', 'sqrtnse'],
    'KGE variants': ['kge', 'kge_inv', 'kge_sqrt', 'kge_log'],
    'KGE_np variants': ['kge_np', 'kge_np_inv', 'kge_np_sqrt', 'kge_np_log'],
}

for family_name, family_objs in obj_families.items():
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{obj.upper()} Calibrated' for obj in family_objs],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, obj_name in enumerate(family_objs):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Observed
        fig.add_trace(
            go.Scatter(x=exceedance, y=obs_sorted, name='Observed' if idx == 0 else None,
                      showlegend=(idx == 0), line=dict(color='black', width=2)),
            row=row, col=col
        )
        
        # All models
        for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
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
        title=f"<b>Flow Duration Curves - {family_name}</b>",
        height=700,
        legend=dict(orientation='h', y=1.05)
    )
    fig.show()

# %% [markdown]
# ### 10.3 FDC Segment Error Analysis

# %%
# FDC segment errors heatmap - one per model showing all objectives
fdc_segments = ['FDC_Peak', 'FDC_High', 'FDC_Mid', 'FDC_Low', 'FDC_VeryLow']
fdc_labels = ['Peak (0-2%)', 'High (2-10%)', 'Mid (20-70%)', 'Low (70-90%)', 'Very Low (90-100%)']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['GR4J', 'GR5J', 'GR6J', 'Sacramento'],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

for idx, model_name in enumerate(['GR4J', 'GR5J', 'GR6J', 'Sacramento']):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    # Build data matrix: objectives × segments
    data = []
    for obj_name in OBJECTIVES.keys():
        row_data = []
        for seg in fdc_segments:
            if obj_name in all_metrics.get(model_name, {}) and seg in all_metrics[model_name][obj_name]:
                row_data.append(all_metrics[model_name][obj_name][seg])
            else:
                row_data.append(np.nan)
        data.append(row_data)
    
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=['Peak', 'High', 'Mid', 'Low', 'V.Low'],
            y=[obj.upper() for obj in OBJECTIVES.keys()],
            colorscale='RdBu',
            zmid=0,
            zmin=-50,
            zmax=50,
            showscale=(idx == 3),
            text=np.round(data, 1),
            texttemplate='%{text}',
            textfont={"size": 8},
        ),
        row=row, col=col
    )

fig.update_layout(
    title="<b>FDC Segment Errors (% Volume Bias) by Model</b><br>" +
          "<sup>Red = overestimate, Blue = underestimate</sup>",
    height=800,
    width=1000
)
fig.show()

# %% [markdown]
# ### 10.4 Hydrologic Signature Errors

# %%
# Hydrologic signature errors comparison
signature_metrics = ['Q95_error', 'Q50_error', 'Q5_error', 'Flashiness_error', 'BFI_error']
signature_labels = ['Q95 (High Flow)', 'Q50 (Median)', 'Q5 (Low Flow)', 'Flashiness', 'Baseflow Index']

# Create grouped bar chart for each objective family
for family_name, family_objs in obj_families.items():
    fig = go.Figure()
    
    x_positions = np.arange(len(signature_metrics))
    bar_width = 0.18
    
    for i, model_name in enumerate(['GR4J', 'GR5J', 'GR6J', 'Sacramento']):
        # Average signature errors across objectives in this family
        avg_errors = []
        for sig in signature_metrics:
            errors = []
            for obj_name in family_objs:
                if obj_name in all_metrics.get(model_name, {}) and sig in all_metrics[model_name][obj_name]:
                    errors.append(all_metrics[model_name][obj_name][sig])
            avg_errors.append(np.mean(errors) if errors else np.nan)
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=x_positions + i * bar_width,
            y=avg_errors,
            width=bar_width,
            marker_color=MODEL_COLORS[model_name]
        ))
    
    fig.update_layout(
        title=f"<b>Hydrologic Signature Errors - {family_name} (Average)</b><br>" +
              "<sup>% error from observed signatures</sup>",
        xaxis=dict(tickvals=x_positions + 1.5 * bar_width, ticktext=signature_labels),
        yaxis_title="Error (%)",
        barmode='group',
        height=450
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.show()

# %% [markdown]
# ### 10.5 Traditional Error Metrics Comparison

# %%
# RMSE, MAE, PBIAS comparison across all objectives
traditional_metrics = ['RMSE', 'MAE', 'PBIAS']

# Create heatmaps for each metric
for metric in traditional_metrics:
    # Build data matrix: objectives × models
    data = []
    for obj_name in OBJECTIVES.keys():
        row_data = []
        for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
            if obj_name in all_metrics.get(model_name, {}) and metric in all_metrics[model_name][obj_name]:
                row_data.append(all_metrics[model_name][obj_name][metric])
            else:
                row_data.append(np.nan)
        data.append(row_data)
    
    # Determine colorscale based on metric
    if metric == 'PBIAS':
        colorscale = 'RdBu'
        zmid = 0
    else:
        colorscale = 'Viridis'
        zmid = None
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=['GR4J', 'GR5J', 'GR6J', 'Sacramento'],
        y=[obj.upper() for obj in OBJECTIVES.keys()],
        colorscale=colorscale,
        zmid=zmid,
        text=np.round(data, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title=f"<b>{metric} by Model × Objective</b><br>" +
              f"<sup>{'Lower is better' if metric != 'PBIAS' else 'Closer to 0 is better'}</sup>",
        xaxis_title="Model",
        yaxis_title="Calibration Objective",
        height=550,
        width=700
    )
    fig.show()

# %% [markdown]
# ### 10.6 Model Improvement Over Sacramento

# %%
# Calculate improvement metrics: (GR_model - Sacramento) for each metric
# Positive = GR model is better (for NSE/KGE), Negative = Sacramento is better

improvement_metrics = ['NSE', 'KGE', 'LogNSE', 'InvNSE']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{m} Improvement vs Sacramento' for m in improvement_metrics],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

for idx, metric in enumerate(improvement_metrics):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    # Build improvement data: objectives × models
    data = []
    for obj_name in OBJECTIVES.keys():
        row_data = []
        sac_val = all_metrics.get('Sacramento', {}).get(obj_name, {}).get(metric, np.nan)
        
        for model_name in ['GR4J', 'GR5J', 'GR6J']:
            model_val = all_metrics.get(model_name, {}).get(obj_name, {}).get(metric, np.nan)
            if not np.isnan(sac_val) and not np.isnan(model_val):
                # For efficiency metrics, higher is better
                improvement = model_val - sac_val
                row_data.append(improvement)
            else:
                row_data.append(np.nan)
        data.append(row_data)
    
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=['GR4J', 'GR5J', 'GR6J'],
            y=[obj.upper() for obj in OBJECTIVES.keys()],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(data, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            showscale=(idx == 3),
        ),
        row=row, col=col
    )

fig.update_layout(
    title="<b>GR Model Performance vs Sacramento</b><br>" +
          "<sup>Green = GR model better, Red = Sacramento better</sup>",
    height=800,
    width=900
)
fig.show()

# %% [markdown]
# ### 10.7 Radar Charts: Model Profiles by Objective Family

# %%
# Radar charts showing model performance profile
radar_metrics = ['NSE', 'KGE', 'LogNSE', 'InvNSE', 'Pearson_r']

for family_name, family_objs in obj_families.items():
    fig = go.Figure()
    
    for model in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
        # Average metrics across objectives in this family
        values = []
        for m in radar_metrics:
            metric_values = []
            for obj_name in family_objs:
                if obj_name in all_metrics.get(model, {}) and m in all_metrics[model][obj_name]:
                    metric_values.append(all_metrics[model][obj_name][m])
            values.append(np.mean(metric_values) if metric_values else 0)
        
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
        title=f"<b>Model Performance Profile - {family_name} (Average)</b>",
        height=500,
        width=600
    )
    fig.show()

# %% [markdown]
# ### 10.8 Comprehensive Performance Summary Table

# %%
# Create summary table with mean performance across all 13 objectives
print("=" * 130)
print("COMPREHENSIVE PERFORMANCE SUMMARY (Mean Across All 13 Objectives)")
print("=" * 130)

summary_metrics = ['NSE', 'KGE', 'LogNSE', 'InvNSE', 'RMSE', 'MAE', 'PBIAS']

print(f"\n{'Model':<12} {'Params':>6}", end="")
for m in summary_metrics:
    print(f" {m:>10}", end="")
print()
print("-" * 130)

for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
    if ALL_RESULTS.get(model_name):
        n_params = len(list(ALL_RESULTS[model_name].values())[0].best_parameters)
    else:
        n_params = '?'
    
    print(f"{model_name:<12} {n_params:>6}", end="")
    
    for metric in summary_metrics:
        values = []
        for obj_name in OBJECTIVES.keys():
            if obj_name in all_metrics.get(model_name, {}) and metric in all_metrics[model_name][obj_name]:
                values.append(all_metrics[model_name][obj_name][metric])
        
        if values:
            mean_val = np.mean(values)
            if metric in ['RMSE', 'MAE']:
                print(f" {mean_val:>10.2f}", end="")
            elif metric == 'PBIAS':
                print(f" {mean_val:>+9.2f}%", end="")
            else:
                print(f" {mean_val:>10.4f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

# %% [markdown]
# ---
# ## Step 11: Final Summary and Recommendations

# %%
# Model ranking by different criteria
print("=" * 100)
print("FINAL MODEL RANKING AND RECOMMENDATIONS")
print("=" * 100)

# Calculate mean metrics for ranking
def calc_mean_metric(model_name, metric_name):
    values = []
    for obj_name in OBJECTIVES.keys():
        if obj_name in all_metrics.get(model_name, {}) and metric_name in all_metrics[model_name][obj_name]:
            values.append(all_metrics[model_name][obj_name][metric_name])
    return np.mean(values) if values else np.nan

print("\n1. OVERALL EFFICIENCY RANKING (Mean NSE across all 13 objectives)")
print("-" * 60)
nse_means = {m: calc_mean_metric(m, 'NSE') for m in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']}
for rank, (model, nse) in enumerate(sorted(nse_means.items(), key=lambda x: -x[1]), 1):
    print(f"   {rank}. {model:<12} | Mean NSE: {nse:.4f}")

print("\n2. LOW FLOW PERFORMANCE RANKING (Mean InvNSE)")
print("-" * 60)
inv_means = {m: calc_mean_metric(m, 'InvNSE') for m in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']}
for rank, (model, val) in enumerate(sorted(inv_means.items(), key=lambda x: -x[1]), 1):
    print(f"   {rank}. {model:<12} | Mean InvNSE: {val:.4f}")

print("\n3. VOLUME BALANCE RANKING (Mean |PBIAS|)")
print("-" * 60)
pbias_means = {m: abs(calc_mean_metric(m, 'PBIAS')) for m in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']}
for rank, (model, val) in enumerate(sorted(pbias_means.items(), key=lambda x: x[1]), 1):
    print(f"   {rank}. {model:<12} | Mean |PBIAS|: {val:.2f}%")

print(f"""

{'=' * 100}
RECOMMENDATIONS BY APPLICATION
{'=' * 100}

┌────────────────────────────────────────────────────────────────────────────────┐
│  APPLICATION                        │  RECOMMENDED MODEL                       │
├─────────────────────────────────────┼──────────────────────────────────────────┤
│  Quick regional assessment          │  GR4J (4 params, fast, robust)           │
│  General purpose / water supply     │  GR5J or GR6J (balanced performance)     │
│  Low flow / drought analysis        │  GR6J (exponential store for low flows)  │
│  Operational flood forecasting      │  Sacramento (industry standard)          │
│  Detailed soil moisture tracking    │  Sacramento (multi-zone states)          │
│  Limited data / ungauged basins     │  GR4J (fewer parameters to estimate)     │
└────────────────────────────────────────────────────────────────────────────────┘

KEY FINDINGS:
─────────────
• Simpler models (GR4J, GR5J, GR6J) often perform comparably to Sacramento
• GR6J typically excels at low-flow simulation due to its exponential store
• Sacramento's 22 parameters provide flexibility but increase equifinality risk
• The "best" model depends on your specific application and data quality

""")

# %% [markdown]
# ---
# ## Export Results

# %%
# Export calibration summary
export_summary = []
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
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
for model_name in ['GR4J', 'GR5J', 'GR6J', 'Sacramento']:
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
print(f"  • 52 total calibrations (39 new + 13 loaded from Notebook 02)")
print("\nAll results saved to test_data/ directory")
