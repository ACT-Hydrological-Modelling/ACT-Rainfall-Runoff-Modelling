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
# calibrated with **13 different objective functions** using two calibration algorithms
# (SCE-UA and PyDREAM). By systematically evaluating GR4J, GR5J, GR6J, and Sacramento
# across multiple performance metrics, we can understand which model structures work
# best for different applications — from flood forecasting to low-flow estimation.
#
# ## What You'll Learn
#
# - How model complexity (4 to 22 parameters) affects calibration performance
# - How different objective functions reveal model strengths and weaknesses
# - How to interpret FDC segment errors and hydrologic signatures
# - How to use data-driven ranking to choose the right model for your application
# - How SCE-UA (point estimate) and PyDREAM (Bayesian posterior) give complementary views
#
# ## Prerequisites
#
# - **Notebook 02: Calibration Quickstart** — Sacramento SCE-UA results (13 objectives)
# - **Notebook 06: Algorithm Comparison** — Sacramento PyDREAM results (4 transforms)
# - Basic understanding of NSE, KGE, and flow transformations (covered in Notebook 04)
#
# ## Estimated Time
#
# - **With existing reports** in `test_data/07_model_comparison/reports/`: ~2 minutes (loads and skips calibrations)
# - **Running all calibrations from scratch**: ~2-3 hours (52 SCE-UA + 16 PyDREAM)
# - **Analysis and visualisation**: ~5 minutes
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|-------|-------------|
# | 1 | Load data | Import Queanbeyan River forcing and observed flow |
# | 2 | Define objectives | Set up all 13 objective functions (NSE, KGE, KGE', SDEB) |
# | 3 | Load Sacramento results | Import pre-calibrated Sacramento results from Notebook 02 |
# | 4 | Calibration config | Configure SCE-UA settings and decide load-vs-run |
# | 5-7 | Calibrate GR models | Run (or load) GR4J, GR5J, GR6J across 13 objectives |
# | 8 | Organise results | Combine all 4 models × 13 objectives into unified tables |
# | 9 | Generate simulations | Produce simulated hydrographs for all 52 combinations |
# | 10 | Visualise (SCE-UA) | FDCs, heatmaps, signatures, metric comparisons |
# | 11 | Summary & rankings | Data-driven model recommendations by flow emphasis |
# | 12 | Report cards | Side-by-side 4-model visual report per objective |
# | 13-15 | PyDREAM calibration | Bayesian comparison: 4 models × 4 flow transforms |
# | 16-17 | PyDREAM analysis | Metrics, heatmaps, and report cards for MCMC results |
# | 18 | Export & wrap-up | Save all results to CSV and summarise findings |
#
# ## Key Insight
#
# > **No single model dominates all flow regimes.** Sacramento excels at high flows
# > thanks to its detailed soil moisture accounting, while GR6J's exponential store
# > makes it competitive for low-flow simulation with only 6 parameters. The best
# > model depends on your application — this notebook gives you the data to decide.
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
# | **NSE variants** | NSE, NSE_log, NSE_inv, NSE_sqrt | Various flow regimes |
# | **KGE variants** | KGE, KGE_inv, KGE_sqrt, KGE_log | Decomposed performance |
# | **KGE_np variants** | KGE_np, KGE_np_inv, KGE_np_sqrt, KGE_np_log | Robust to outliers |
# | **Composite** | SDEB | FDC shape + timing + bias |

# %% [markdown]
# ---
# ## Setup
#
# We begin by importing the standard scientific Python stack (NumPy, Pandas,
# Matplotlib) together with Plotly for interactive figures. The `pyrrm` library
# provides all four model implementations, the `CalibrationRunner` / `CalibrationReport`
# classes for managing calibration workflows, and the objective function toolkit.

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
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J, NUMBA_AVAILABLE
from pyrrm.calibration import CalibrationRunner, CalibrationReport
from pyrrm.calibration.objective_functions import NSE
from pyrrm.analysis.diagnostics import compute_diagnostics, print_diagnostics, DIAGNOSTIC_GROUPS

# Import objective functions from pyrrm.objectives
from pyrrm.objectives import (
    NSE as NSE_obj, KGE, KGENonParametric, PBIAS, RMSE,
    FlowTransformation, FDCMetric, SignatureMetric, SDEB,
    PearsonCorrelation, SpearmanCorrelation, MAE
)

print("\npyrrm models imported successfully!")
print(f"\nNumba JIT acceleration: {'ACTIVE' if NUMBA_AVAILABLE else 'not available (pip install numba)'}")
print(f"\nAvailable models:")
print(f"  - Sacramento (22 parameters)")
print(f"  - GR4J (4 parameters)")
print(f"  - GR5J (5 parameters)")
print(f"  - GR6J (6 parameters)")

# %% [markdown]
# ---
# ## Step 1: Load Data (Same as Notebook 02)
#
# We use the same Queanbeyan River dataset (gauge 410734) as Notebook 02: daily
# rainfall, PET, and observed streamflow. A 365-day warmup period allows model
# stores to initialise before the calibration window begins.
#
# The catchment area (516.63 km²) is required to convert depth-based model
# outputs (mm/day) to volumetric flow (ML/day) for comparison with the gauge.

# %%
from pyrrm.data import load_catchment_data

DATA_DIR = Path('../data/410734')
OUTPUT_DIR = Path('../test_data/07_model_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_REPORTS_DIR = OUTPUT_DIR / 'reports'
MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
NB02_REPORTS_DIR = Path('../test_data/02_calibration_quickstart/reports')
CATCHMENT_AREA_KM2 = 516.62667
WARMUP_DAYS = 365

cal_inputs, cal_observed = load_catchment_data(
    precipitation_file=DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
    pet_file=DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
    observed_file=DATA_DIR / '410734_output_SDmodel.csv',
    observed_value_column='Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)',
)

print("MERGED DATASET")
print("=" * 50)
print(f"Total records: {len(cal_inputs):,} days")
print(f"Period: {cal_inputs.index.min().date()} to {cal_inputs.index.max().date()}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_inputs) - WARMUP_DAYS:,} days")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")

# %% [markdown]
# ---
# ## Step 2: Define All 13 Objective Functions
#
# A key question in model comparison is: *"best at what?"*  Different objective
# functions reward different aspects of the hydrograph:
#
# - **Untransformed** (NSE, KGE, KGE') — dominated by high-flow peaks because
#   squared errors scale with magnitude.
# - **√Q transform** — compresses the range, giving a balanced view across the
#   entire flow regime.
# - **log(Q) transform** — emphasises low flows and recession behaviour.
# - **1/Q transform** — extreme emphasis on very low flows (near-zero).
# - **SDEB (composite)** — combines chronological fit, FDC shape, and volume bias.
#
# By calibrating every model with all 13 objectives, we can see whether a model's
# advantage holds across emphasis levels or only under certain conditions.

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
#
# Sacramento was already calibrated with all 13 objective functions in Notebook 02.
# Rather than re-running those expensive 22-parameter optimisations, we load the
# saved `CalibrationReport` pickle files. Each report contains the best parameter
# set, convergence diagnostics, and metadata needed for downstream comparison.
#
# If any reports are missing, you'll need to run Notebook 02 first.

# %%
# Load ALL 13 Sacramento calibrations from Notebook 02
print("=" * 70)
print("LOADING SACRAMENTO RESULTS FROM NOTEBOOK 02 (ALL 13 OBJECTIVES)")
print("=" * 70)

# Map of ALL 13 objective names to report files
SACRAMENTO_REPORTS = {
    # NSE variants (4)
    'nse': str(NB02_REPORTS_DIR / '410734_sacramento_nse_sceua.pkl'),
    'lognse': str(NB02_REPORTS_DIR / '410734_sacramento_nse_sceua_log.pkl'),
    'invnse': str(NB02_REPORTS_DIR / '410734_sacramento_nse_sceua_inverse.pkl'),
    'sqrtnse': str(NB02_REPORTS_DIR / '410734_sacramento_nse_sceua_sqrt.pkl'),
    # KGE variants (4)
    'kge': str(NB02_REPORTS_DIR / '410734_sacramento_kge_sceua.pkl'),
    'kge_inv': str(NB02_REPORTS_DIR / '410734_sacramento_kge_sceua_inverse.pkl'),
    'kge_sqrt': str(NB02_REPORTS_DIR / '410734_sacramento_kge_sceua_sqrt.pkl'),
    'kge_log': str(NB02_REPORTS_DIR / '410734_sacramento_kge_sceua_log.pkl'),
    # Non-parametric KGE variants (4)
    'kge_np': str(NB02_REPORTS_DIR / '410734_sacramento_kgenp_sceua.pkl'),
    'kge_np_inv': str(NB02_REPORTS_DIR / '410734_sacramento_kgenp_sceua_inverse.pkl'),
    'kge_np_sqrt': str(NB02_REPORTS_DIR / '410734_sacramento_kgenp_sceua_sqrt.pkl'),
    'kge_np_log': str(NB02_REPORTS_DIR / '410734_sacramento_kgenp_sceua_log.pkl'),
    # Composite (1)
    'sdeb': str(NB02_REPORTS_DIR / '410734_sacramento_sdeb_sceua.pkl'),
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
#
# SCE-UA's computational budget (`max_evals`) is scaled by model complexity.
# More parameters require more function evaluations to explore the search space
# adequately. Following Duan et al. (1994), we use `n_complexes = 2n + 1` where
# *n* is the number of parameters — this is handled automatically by the
# `CalibrationRunner`.
#
# We also define consistent colour coding for the four models, which will be
# used throughout all figures in this notebook.

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
# ## Step 4b: Calibration Results (Load or Run)
#
# Running 39 SCE-UA calibrations from scratch takes 1-2 hours. Each calibration
# cell below **checks whether a report already exists** in
# `test_data/07_model_comparison/reports/`. If the `.pkl` file exists, the
# result is loaded and the calibration is skipped; otherwise the calibration
# runs and the report is saved. This avoids re-running calibrations when
# reports are already present.

# %%
REPORTS_DIR = MODELS_REPORTS_DIR

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

print("Calibration cells below will load from reports when present, else run and save.")
print(f"Reports directory: {MODELS_REPORTS_DIR}")

# %% [markdown]
# ---
# ## Step 5: Calibrate GR4J with All 13 Objectives
#
# GR4J (Perrin et al., 2003) is the simplest model in our comparison with only
# **4 free parameters**: production store capacity (x1), groundwater exchange
# coefficient (x2), routing store capacity (x3), and unit hydrograph time base
# (x4). Its parsimony makes it a strong benchmark — if a more complex model
# cannot beat GR4J, the added complexity may not be justified.
#
# Each of the 13 calibrations runs SCE-UA independently. The cells below load from
# the reports directory when a report file exists; otherwise they run and save.

# %% [markdown]
# ### GR4J Calibrations (13 total)

# %%
# GR4J Calibration 1/13: NSE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['nse'] = report.result
    gr4j_times['nse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J nse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR4J CALIBRATION 1/13: NSE")
    print("=" * 70)

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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua'))
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 2/13: NSE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['lognse'] = report.result
    gr4j_times['lognse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J lognse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR4J CALIBRATION 2/13: NSE_log")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_log'))
    print(f"\n✓ Best NSE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 3/13: NSE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['invnse'] = report.result
    gr4j_times['invnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J invnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR4J CALIBRATION 3/13: NSE_inv")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_inverse'))
    print(f"\n✓ Best NSE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 4/13: NSE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['sqrtnse'] = report.result
    gr4j_times['sqrtnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J sqrtnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR4J CALIBRATION 4/13: NSE_sqrt")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_nse_sceua_sqrt'))
    print(f"\n✓ Best NSE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 5/13: KGE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge'] = report.result
    gr4j_times['kge'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua'))
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 6/13: KGE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_inv'] = report.result
    gr4j_times['kge_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_inverse'))
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 7/13: KGE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_sqrt'] = report.result
    gr4j_times['kge_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_sqrt'))
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 8/13: KGE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_log'] = report.result
    gr4j_times['kge_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kge_sceua_log'))
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 9/13: KGE_np (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_np'] = report.result
    gr4j_times['kge_np'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_np: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua'))
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 10/13: KGE_np_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_np_inv'] = report.result
    gr4j_times['kge_np_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_np_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_inverse'))
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 11/13: KGE_np_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_np_sqrt'] = report.result
    gr4j_times['kge_np_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_np_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_sqrt'))
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 12/13: KGE_np_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['kge_np_log'] = report.result
    gr4j_times['kge_np_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J kge_np_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_kgenp_sceua_log'))
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR4J Calibration 13/13: SDEB (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr4j_sdeb_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr4j_results['sdeb'] = report.result
    gr4j_times['sdeb'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR4J sdeb: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr4j_sdeb_sceua'))
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR4J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 6: Calibrate GR5J with All 13 Objectives
#
# GR5J (Le Moine, 2008) extends GR4J by adding a **fifth parameter** (x5) that
# sets a groundwater exchange threshold. This allows the model to better represent
# inter-catchment groundwater flows — water lost to (or gained from) neighbouring
# catchments through deep aquifer pathways. The extra flexibility may improve
# performance in catchments where baseflow is influenced by regional hydrogeology.

# %% [markdown]
# ### GR5J Calibrations (13 total)

# %%
# GR5J Calibration 1/13: NSE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['nse'] = report.result
    gr5j_times['nse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J nse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR5J CALIBRATION 1/13: NSE")
    print("=" * 70)

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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua'))
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 2/13: NSE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['lognse'] = report.result
    gr5j_times['lognse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J lognse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR5J CALIBRATION 2/13: NSE_log")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_log'))
    print(f"\n✓ Best NSE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 3/13: NSE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['invnse'] = report.result
    gr5j_times['invnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J invnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR5J CALIBRATION 3/13: NSE_inv")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_inverse'))
    print(f"\n✓ Best NSE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 4/13: NSE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['sqrtnse'] = report.result
    gr5j_times['sqrtnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J sqrtnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR5J CALIBRATION 4/13: NSE_sqrt")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_nse_sceua_sqrt'))
    print(f"\n✓ Best NSE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 5/13: KGE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge'] = report.result
    gr5j_times['kge'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua'))
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 6/13: KGE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_inv'] = report.result
    gr5j_times['kge_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_inverse'))
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 7/13: KGE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_sqrt'] = report.result
    gr5j_times['kge_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_sqrt'))
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 8/13: KGE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_log'] = report.result
    gr5j_times['kge_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kge_sceua_log'))
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 9/13: KGE_np (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_np'] = report.result
    gr5j_times['kge_np'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_np: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua'))
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 10/13: KGE_np_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_np_inv'] = report.result
    gr5j_times['kge_np_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_np_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_inverse'))
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 11/13: KGE_np_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_np_sqrt'] = report.result
    gr5j_times['kge_np_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_np_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_sqrt'))
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 12/13: KGE_np_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['kge_np_log'] = report.result
    gr5j_times['kge_np_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J kge_np_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_kgenp_sceua_log'))
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR5J Calibration 13/13: SDEB (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr5j_sdeb_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr5j_results['sdeb'] = report.result
    gr5j_times['sdeb'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR5J sdeb: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr5j_sdeb_sceua'))
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR5J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 7: Calibrate GR6J with All 13 Objectives
#
# GR6J (Pushpalatha et al., 2011) adds a **sixth parameter** controlling an
# exponential store specifically designed to improve **low-flow simulation**.
# This makes it particularly interesting for our comparison: does the extra
# store meaningfully improve performance on inverse- and log-transformed
# objectives that emphasise low flows?

# %% [markdown]
# ### GR6J Calibrations (13 total)

# %%
# GR6J Calibration 1/13: NSE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['nse'] = report.result
    gr6j_times['nse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J nse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR6J CALIBRATION 1/13: NSE")
    print("=" * 70)

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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua'))
    print(f"\n✓ Best NSE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 2/13: NSE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['lognse'] = report.result
    gr6j_times['lognse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J lognse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR6J CALIBRATION 2/13: NSE_log")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_log'))
    print(f"\n✓ Best NSE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 3/13: NSE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['invnse'] = report.result
    gr6j_times['invnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J invnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR6J CALIBRATION 3/13: NSE_inv")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_inverse'))
    print(f"\n✓ Best NSE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 4/13: NSE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['sqrtnse'] = report.result
    gr6j_times['sqrtnse'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J sqrtnse: loaded from {pkl_path.name}")
else:
    print("=" * 70)
    print("GR6J CALIBRATION 4/13: NSE_sqrt")
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_nse_sceua_sqrt'))
    print(f"\n✓ Best NSE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 5/13: KGE (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge'] = report.result
    gr6j_times['kge'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua'))
    print(f"\n✓ Best KGE: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 6/13: KGE_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_inv'] = report.result
    gr6j_times['kge_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_inverse'))
    print(f"\n✓ Best KGE_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 7/13: KGE_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_sqrt'] = report.result
    gr6j_times['kge_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_sqrt'))
    print(f"\n✓ Best KGE_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 8/13: KGE_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_log'] = report.result
    gr6j_times['kge_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kge_sceua_log'))
    print(f"\n✓ Best KGE_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 9/13: KGE_np (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_np'] = report.result
    gr6j_times['kge_np'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_np: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua'))
    print(f"\n✓ Best KGE_np: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 10/13: KGE_np_inv (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_inverse.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_np_inv'] = report.result
    gr6j_times['kge_np_inv'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_np_inv: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_inverse'))
    print(f"\n✓ Best KGE_np_inv: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 11/13: KGE_np_sqrt (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_sqrt.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_np_sqrt'] = report.result
    gr6j_times['kge_np_sqrt'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_np_sqrt: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_sqrt'))
    print(f"\n✓ Best KGE_np_sqrt: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 12/13: KGE_np_log (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_log.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['kge_np_log'] = report.result
    gr6j_times['kge_np_log'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J kge_np_log: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_kgenp_sceua_log'))
    print(f"\n✓ Best KGE_np_log: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

# %%
# GR6J Calibration 13/13: SDEB (skipped if report exists)
pkl_path = MODELS_REPORTS_DIR / '410734_gr6j_sdeb_sceua.pkl'
if pkl_path.exists():
    report = CalibrationReport.load(str(pkl_path))
    gr6j_results['sdeb'] = report.result
    gr6j_times['sdeb'] = getattr(report.result, 'runtime_seconds', 0.0)
    print(f"  ✓ GR6J sdeb: loaded from {pkl_path.name}")
else:
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
    report.save(str(MODELS_REPORTS_DIR / '410734_gr6j_sdeb_sceua'))
    print(f"\n✓ Best SDEB: {result.best_objective:.4f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("GR6J CALIBRATION COMPLETE - ALL 13 OBJECTIVES")
    print("=" * 70)

# %% [markdown]
# ---
# ## Step 8: Organize All Results
#
# With all four models calibrated (or loaded) for 13 objectives each, we now
# consolidate the 52 `CalibrationResult` objects into unified dictionaries keyed
# by model name and objective. This makes downstream analysis — simulation,
# metric computation, and plotting — straightforward.
#
# The summary table printed below gives a quick overview: each cell shows the
# best objective value achieved by that model/objective combination.

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
#
# Before we can compute performance metrics or plot hydrographs, we need the
# simulated flow timeseries for every model/objective pair. Each simulation
# re-runs the model with its calibrated parameter set over the full input period,
# then extracts the post-warmup segment for comparison with observed flow.
#
# Note that Sacramento outputs a column named `'runoff'` while the GR models
# use `'flow'` — the helper function `get_flow_column()` handles this
# transparently.

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
        
        sim_results = model.run(cal_inputs)
        all_simulations[model_name][obj_name] = get_flow_column(sim_results)[WARMUP_DAYS:]
    
    print(f"  ✓ {model_name}: {len(all_simulations[model_name])} simulations")

# Get observed data for comparison
obs_flow = cal_observed[WARMUP_DAYS:]
comparison_dates = cal_inputs.index[WARMUP_DAYS:]

print(f"\nSimulations complete! ({len(obs_flow):,} days each)")

# %% [markdown]
# ---
# ## Step 10: Comprehensive Model Comparison Visualizations
#
# This is the analytical core of the notebook. We evaluate every simulated
# hydrograph against observed flow using a **comprehensive diagnostic suite**
# that goes well beyond the calibration objective:
#
# 1. **Flow Duration Curves (FDCs)** grouped by objective family — do the models
#    reproduce the full flow distribution, not just what they were calibrated on?
# 2. **FDC segment errors** (Peak / High / Mid / Low / Very Low) — where on the
#    flow spectrum does each model succeed or fail?
# 3. **Hydrologic signature errors** (Q5, Q50, Q95, BFI, flashiness) — do the
#    models capture physically meaningful catchment characteristics?
# 4. **Traditional error metrics** (RMSE, MAE, PBIAS) — simple, interpretable
#    measures of overall accuracy.
#
# Together, these provide a multi-dimensional view that a single metric cannot.

# %% [markdown]
# ### 10.1 Comprehensive Metrics Calculation
#
# We start by computing a rich set of diagnostics for all 52 combinations.
# The `comprehensive_evaluation()` function extends the canonical 48-metric
# suite from `compute_diagnostics()` with additional FDC segments, signature
# metrics, and the SDEB composite objective.

# %%
# Calculate comprehensive metrics for ALL model × objective combinations.
# Start from the canonical 48-metric suite, then add notebook-specific
# extended FDC segments, extra signature metrics, and SDEB for the multi-model analysis.
def comprehensive_evaluation(obs, sim):
    """Canonical diagnostics + extended FDC/signature metrics for model comparison."""
    metrics = dict(compute_diagnostics(sim, obs))

    inv_transform = FlowTransformation('inverse', epsilon_value=0.01)
    sqrt_transform = FlowTransformation('sqrt')
    log_transform = FlowTransformation('log', epsilon_value=0.01)

    metrics['Pearson_r'] = PearsonCorrelation()(obs, sim)
    metrics['Spearman_r'] = SpearmanCorrelation()(obs, sim)

    try:
        metrics['FDC_Peak'] = FDCMetric('peak', 'volume_bias')(obs, sim)
        metrics['FDC_High'] = FDCMetric('high', 'volume_bias')(obs, sim)
        metrics['FDC_Mid'] = FDCMetric('mid', 'volume_bias')(obs, sim)
        metrics['FDC_Low'] = FDCMetric('low', 'volume_bias', log_transform=True)(obs, sim)
        metrics['FDC_VeryLow'] = FDCMetric('very_low', 'volume_bias', log_transform=True)(obs, sim)
    except Exception:
        pass

    try:
        metrics['Q95_error'] = SignatureMetric('q95')(obs, sim)
        metrics['Q50_error'] = SignatureMetric('q50')(obs, sim)
        metrics['Q5_error'] = SignatureMetric('q5')(obs, sim)
        metrics['Flashiness_error'] = SignatureMetric('flashiness')(obs, sim)
        metrics['BFI_error'] = SignatureMetric('baseflow_index')(obs, sim)
    except Exception:
        pass

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
#
# The **Flow Duration Curve (FDC)** ranks all daily flows from highest to lowest
# and plots them against exceedance probability. It is one of the most informative
# diagnostic tools in hydrology because it shows model performance across the
# entire flow spectrum in a single plot.
#
# We group the FDCs by objective family (NSE, KGE, KGE') so you can see how
# the flow transformation (none → sqrt → log → inverse) shifts each model's
# fit from high flows toward low flows. Within each panel, all four models
# are overlaid against the observed FDC.

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
#
# The FDC can be split into five segments representing different hydrological
# regimes:
#
# | Segment | Exceedance Range | Physical Meaning |
# |---------|-----------------|------------------|
# | **Peak** | 0-2% | Flood events |
# | **High** | 2-20% | Above-average flows |
# | **Mid** | 20-70% | Normal/median flows |
# | **Low** | 70-95% | Below-average / dry season |
# | **Very Low** | 95-100% | Drought / minimum flows |
#
# The heatmaps below show the volume bias for each segment, model, and objective.
# Positive values mean the model over-predicts; negative means under-prediction.
# This reveals, for example, whether Sacramento's 22-parameter structure gives
# it an advantage specifically in low-flow segments.

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
#
# **Hydrologic signatures** are summary statistics of the flow regime that
# describe physically meaningful catchment behaviour:
#
# - **Q5, Q50, Q95** — flow quantiles capturing high, median, and low flows
# - **BFI** (Baseflow Index) — fraction of total flow that comes from baseflow
# - **Flashiness** — day-to-day variability in flow (Richards-Baker index)
#
# A good model should reproduce these signatures even when calibrated with a
# metric that doesn't explicitly target them. The heatmaps below show the
# relative error for each signature, revealing which models best capture
# catchment-level behaviour beyond the calibration objective.
#
# **What is actually plotted?** We have 13 objective functions but only four
# subplots. The 13 objectives are **grouped by flow emphasis** (from
# `OBJ_ORDERED`), and each subplot corresponds to one group:
#
# | Subplot | Emphasis group | Objectives in the group |
# |---------|----------------|-------------------------|
# | Top-left | **High-flow (Q)** | NSE, KGE, KGE′ (3) — untransformed Q |
# | Top-right | **Balanced (√Q)** | √NSE, KGE-√Q, KGE′-√Q, SDEB (4) |
# | Bottom-left | **Low-flow (log Q)** | log-NSE, KGE-log, KGE′-log (3) |
# | Bottom-right | **Very-low-flow (1/Q)** | 1/Q NSE, KGE-1/Q, KGE′-1/Q (3) |
#
# Within each subplot, each **(signature × model)** cell shows the **average**
# percentage error across all objectives in that group — not a single run.
# So you see how each model performs on each signature when calibrated with
# objectives that share the same flow emphasis; comparing across subplots
# shows how that picture changes with calibration emphasis (e.g. high-flow
# vs very-low-flow).

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
#
# While NSE, KGE, and their variants are widely used in hydrology, traditional
# error metrics provide complementary information:
#
# - **RMSE** — root mean squared error in original units (ML/day), dominated by
#   large errors at high flows
# - **MAE** — mean absolute error, less sensitive to outliers than RMSE
# - **PBIAS** — percent bias (positive = over-prediction, negative = under-prediction),
#   indicating systematic volume errors
#
# The grouped bar charts below compare these metrics across all models and
# objectives, ordered from high-flow to low-flow emphasis.

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
    'RMSE': 'Lower is better (ML/day); log color scale',
    'MAE': 'Lower is better (ML/day); log color scale',
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
        z_plot = data
    else:
        # RMSE and MAE: use log10 scale for color so outliers don't dominate
        colorscale = 'Viridis'
        zmid = None
        zmin, zmax = None, None
        arr = np.array(data, dtype=float)
        z_plot = np.log10(arr + 1).tolist()  # log10(value+1); text still shows raw values
    
    heatmap_kw = dict(
        z=z_plot,
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
#
# Sacramento is the most complex model in our comparison (22 parameters). The
# key question for practitioners is: **can a simpler GR model match or beat it?**
#
# The figure below uses the **canonical 48-metric diagnostic suite** from
# `compute_diagnostics()` (plus SDEB from the composite objective). Each GR
# model gets two side-by-side panels:
#
# | Panel | Columns | Interpretation |
# |-------|---------|----------------|
# | **Left — Skill metrics** | NSE, KGE, KGE_np (× 4 transforms) | Higher = better; blue = GR wins |
# | **Right — Error & bias** | RMSE, MAE, PBIAS, FHV/FMV/FLV, Sig_BFI/Flash/Q95/Q5, SDEB | Closer to 0 = better; blue = GR wins |
#
# All cells show the **sign-normalised difference** (GR − Sacramento) so that
# **blue always means the GR model outperforms Sacramento** and **red means
# Sacramento wins**. For error/bias metrics where lower absolute value is
# better, differences are computed on absolute values and flipped so the
# colour convention is consistent.

# %%
# GR improvement over Sacramento — two panels per GR model
# Left:  skill metrics (higher = better, 0–1 scale)
# Right: error & bias metrics (closer to 0 = better)
# Blue = GR outperforms Sacramento · Red = Sacramento outperforms GR

# --- Left panel: 12 skill metrics (canonical names) ---
DIAG_SKILL_COLS = [
    # High-flow (untransformed Q)
    ('NSE',        'NSE'),
    ('KGE',        'KGE'),
    ('KGE_np',     'KGE_np'),
    # Balanced (sqrt Q)
    ('NSE_sqrt',   'NSE_sqrt'),
    ('KGE_sqrt',   'KGE_sqrt'),
    ('KGE_np_sqrt','KGE_np_sqrt'),
    # Low-flow (log Q)
    ('NSE_log',    'NSE_log'),
    ('KGE_log',    'KGE_log'),
    ('KGE_np_log', 'KGE_np_log'),
    # Very-low-flow (1/Q)
    ('NSE_inv',    'NSE_inv'),
    ('KGE_inv',    'KGE_inv'),
    ('KGE_np_inv', 'KGE_np_inv'),
]
SKILL_GROUPS = [
    (0,  'High-flow'),
    (3,  'Balanced (√Q)'),
    (6,  'Low-flow (log Q)'),
    (9,  'Very-low-flow (1/Q)'),
]

# --- Right panel: 11 error & bias metrics ---
DIAG_ERR_COLS = [
    # Error (ML/day) — lower = better
    ('RMSE',       'RMSE'),
    ('MAE',        'MAE'),
    # Bias (%) — closer to 0 = better
    ('PBIAS',      'PBIAS'),
    # FDC volume bias (%) — closer to 0 = better
    ('FHV',        'FHV'),
    ('FMV',        'FMV'),
    ('FLV',        'FLV'),
    # Signature errors (%) — closer to 0 = better
    ('Sig_BFI',    'Sig_BFI'),
    ('Sig_Flash',  'Sig_Flash'),
    ('Sig_Q95',    'Sig_Q95'),
    ('Sig_Q5',     'Sig_Q5'),
    # Composite — lower = better
    ('SDEB',       'SDEB'),
]
ERR_GROUPS = [
    (0,  'Error (ML/d)'),
    (2,  'Bias (%)'),
    (3,  'FDC bias (%)'),
    (6,  'Signatures (%)'),
    (10, 'Composite'),
]

# All error/bias metrics: closer to 0 = better.
# Difference = |Sac| − |GR|, so positive = GR closer to zero = GR better = blue.
ERR_ALL_KEYS = {k for k, _ in DIAG_ERR_COLS}

skill_labels = [lbl for _, lbl in DIAG_SKILL_COLS]
err_labels   = [lbl for _, lbl in DIAG_ERR_COLS]
n_skill = len(DIAG_SKILL_COLS)
n_err   = len(DIAG_ERR_COLS)
n_obj   = len(OBJ_KEYS_ORDERED)

fig_imp = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        f'<b>{m}</b> — Skill metrics' if c == 0
        else f'<b>{m}</b> — Error & bias metrics'
        for m in MODELS_GR_ONLY for c in (0, 1)
    ],
    vertical_spacing=0.05,
    horizontal_spacing=0.08,
    shared_yaxes=True,
)

# --- Compute improvement matrices ---
KNEE_SKILL = 0.15   # for 0–1 skill metrics
KNEE_ERR   = 5.0    # for %, ML/day error metrics

all_skill_raw = {}
all_err_raw   = {}

for gr_model in MODELS_GR_ONLY:
    z_skill = np.full((n_obj, n_skill), np.nan)
    for d_idx, (dkey, _) in enumerate(DIAG_SKILL_COLS):
        for o_idx, obj_key in enumerate(OBJ_KEYS_ORDERED):
            sac = all_metrics.get('Sacramento', {}).get(obj_key, {}).get(dkey, np.nan)
            gr  = all_metrics.get(gr_model, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(sac) and np.isfinite(gr):
                z_skill[o_idx, d_idx] = gr - sac  # higher = GR better
    all_skill_raw[gr_model] = z_skill

    z_err = np.full((n_obj, n_err), np.nan)
    for d_idx, (dkey, _) in enumerate(DIAG_ERR_COLS):
        for o_idx, obj_key in enumerate(OBJ_KEYS_ORDERED):
            sac = all_metrics.get('Sacramento', {}).get(obj_key, {}).get(dkey, np.nan)
            gr  = all_metrics.get(gr_model, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(sac) and np.isfinite(gr):
                z_err[o_idx, d_idx] = abs(sac) - abs(gr)  # positive = GR closer to 0
    all_err_raw[gr_model] = z_err

# --- Colorbar ticks for skill panel ---
cb_skill_vals = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
cb_skill_ticks, cb_skill_labels = [], []
for v in reversed(cb_skill_vals[1:]):
    cb_skill_ticks.append(soft_clamp(-v, k=KNEE_SKILL))
    cb_skill_labels.append(f'−{v}')
cb_skill_ticks.append(0.0)
cb_skill_labels.append('0')
for v in cb_skill_vals[1:]:
    cb_skill_ticks.append(soft_clamp(v, k=KNEE_SKILL))
    cb_skill_labels.append(f'+{v}')

# --- Colorbar ticks for error/bias panel ---
cb_err_vals = [0, 1, 2, 5, 10, 25, 50]
cb_err_ticks, cb_err_labels = [], []
for v in reversed(cb_err_vals[1:]):
    cb_err_ticks.append(soft_clamp(-v, k=KNEE_ERR))
    cb_err_labels.append(f'−{v}')
cb_err_ticks.append(0.0)
cb_err_labels.append('0')
for v in cb_err_vals[1:]:
    cb_err_ticks.append(soft_clamp(v, k=KNEE_ERR))
    cb_err_labels.append(f'+{v}')

# --- Helper to build annotation text ---
def _ann_text(z_raw, fmt='+.3f'):
    text = []
    for row in z_raw:
        text.append([f'{v:{fmt}}' if np.isfinite(v) else '' for v in row])
    return text

# --- Add traces for each GR model ---
for m_idx, gr_model in enumerate(MODELS_GR_ONLY):
    r = m_idx + 1

    # Left panel: skill metrics
    z_sk = all_skill_raw[gr_model]
    z_sk_sc = np.vectorize(lambda x: soft_clamp(x, k=KNEE_SKILL))(z_sk)
    fig_imp.add_trace(
        go.Heatmap(
            z=z_sk_sc, x=skill_labels, y=OBJ_LABELS_ORDERED,
            text=_ann_text(z_sk), texttemplate='%{text}', textfont=dict(size=8),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            showscale=(m_idx == 0),
            colorbar=dict(
                title=dict(text='GR − Sac<br>(skill)', side='right'),
                tickvals=cb_skill_ticks, ticktext=cb_skill_labels,
                len=0.85, y=0.5, x=0.46, xpad=5,
            ) if m_idx == 0 else None,
        ),
        row=r, col=1,
    )

    # Right panel: error & bias metrics
    z_er = all_err_raw[gr_model]
    z_er_sc = np.vectorize(lambda x: soft_clamp(x, k=KNEE_ERR))(z_er)
    fig_imp.add_trace(
        go.Heatmap(
            z=z_er_sc, x=err_labels, y=OBJ_LABELS_ORDERED,
            text=_ann_text(z_er, '+.1f'), texttemplate='%{text}', textfont=dict(size=8),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            showscale=(m_idx == 0),
            colorbar=dict(
                title=dict(text='|Sac| − |GR|<br>(error/bias)', side='right'),
                tickvals=cb_err_ticks, ticktext=cb_err_labels,
                len=0.85, y=0.5, x=1.02, xpad=5,
            ) if m_idx == 0 else None,
        ),
        row=r, col=2,
    )

    # Objective-group horizontal separators (both panels)
    for start_idx, _ in OBJ_GROUPS:
        if start_idx > 0:
            for c in (1, 2):
                fig_imp.add_hline(
                    y=start_idx - 0.5, line_color='white', line_width=3,
                    row=r, col=c,
                )

    # Objective-group labels (left side of left panel only)
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
            textangle=0, xanchor='right',
            row=r, col=1,
        )

# --- Skill column-group labels and vertical separators ---
for dg_start, dg_label in SKILL_GROUPS:
    next_dg = [s for s, _ in SKILL_GROUPS if s > dg_start]
    dg_end = next_dg[0] if next_dg else n_skill
    mid_idx = (dg_start + dg_end - 1) / 2
    x_frac = (mid_idx + 0.5) / n_skill * 0.44
    fig_imp.add_annotation(
        text=f'<b>{dg_label}</b>', x=x_frac, y=1.025,
        xref='paper', yref='paper', showarrow=False,
        font=dict(size=10, color='#444'),
    )
    if dg_start > 0:
        for row_r in range(1, 4):
            fig_imp.add_vline(
                x=dg_start - 0.5, line_color='white', line_width=3,
                row=row_r, col=1,
            )

# --- Error column-group labels and vertical separators ---
for dg_start, dg_label in ERR_GROUPS:
    next_dg = [s for s, _ in ERR_GROUPS if s > dg_start]
    dg_end = next_dg[0] if next_dg else n_err
    mid_idx = (dg_start + dg_end - 1) / 2
    x_frac = 0.56 + (mid_idx + 0.5) / n_err * 0.44
    fig_imp.add_annotation(
        text=f'<b>{dg_label}</b>', x=x_frac, y=1.025,
        xref='paper', yref='paper', showarrow=False,
        font=dict(size=10, color='#444'),
    )
    if dg_start > 0:
        for row_r in range(1, 4):
            fig_imp.add_vline(
                x=dg_start - 0.5, line_color='white', line_width=3,
                row=row_r, col=2,
            )

fig_imp.update_layout(
    title="<b>Do GR models improve over Sacramento, and for which diagnostics?</b><br>"
          "<sup>Left = skill metrics (GR − Sac; higher = better) · Right = error & bias metrics (|Sac| − |GR|; closer to 0 = better)<br>"
          "Blue = GR outperforms Sacramento · Red = Sacramento outperforms GR · White ≈ no difference</sup>",
    height=1800,
    width=1900,
    margin=dict(l=250, t=200, r=160, b=100),
)

# Only show x-axis tick labels on the bottom panels
for c in (1, 2):
    fig_imp.update_xaxes(showticklabels=False, row=1, col=c)
    fig_imp.update_xaxes(showticklabels=False, row=2, col=c)
    fig_imp.update_xaxes(tickangle=-45, tickfont=dict(size=10), row=3, col=c)

fig_imp.show()

# Backward-compatible aliases used by later sections (win counting, PyDREAM heatmap)
DIAG_SKILL = DIAG_SKILL_COLS
DIAG_MINIMIZE = set()
diag_labels = skill_labels
KNEE_IMP = KNEE_SKILL
cb_ticks_imp = cb_skill_ticks
cb_labels_imp = cb_skill_labels

# %% [markdown]
# ### 10.7 Radar Charts: Model Profiles by Objective Emphasis Group
#
# Radar (spider) charts provide an intuitive visual summary of how each model
# performs across multiple diagnostics simultaneously. Each axis represents a
# different metric; the polygon area roughly indicates overall performance.
#
# We group the radar charts by objective emphasis (high-flow, balanced, low-flow,
# very-low-flow) so you can see how each model's diagnostic profile changes
# depending on what it was calibrated for.

# %%
# Radar charts: full canonical diagnostic profile per model, grouped by objective emphasis
# 23 metrics: 12 skill + RMSE/MAE + PBIAS + FHV/FMV/FLV + 4 signatures + SDEB

RADAR_METRICS = (
    list(DIAG_SKILL_COLS)
    + [
        # Error (ML/d) — lower = better
        ('RMSE',      'RMSE*'),
        ('MAE',       'MAE*'),
        # Bias (%) — closer to 0 = better
        ('PBIAS',     'PBIAS*'),
        # FDC volume bias (%) — closer to 0 = better
        ('FHV',       'FHV*'),
        ('FMV',       'FMV*'),
        ('FLV',       'FLV*'),
        # Signature errors (%) — closer to 0 = better
        ('Sig_BFI',   'Sig_BFI*'),
        ('Sig_Flash', 'Sig_Flash*'),
        ('Sig_Q95',   'Sig_Q95*'),
        ('Sig_Q5',    'Sig_Q5*'),
        # Composite — lower = better
        ('SDEB',      'SDEB*'),
    ]
)
radar_keys   = [k for k, _ in RADAR_METRICS]
radar_labels = [lbl for _, lbl in RADAR_METRICS]

# Normalisation categories — all mapped to "1 = best, 0 = worst"
RADAR_HIGHER_BETTER = {k for k, _ in DIAG_SKILL_COLS}
RADAR_LOWER_BETTER  = {'RMSE', 'MAE', 'SDEB'}       # non-negative magnitude; log1p + reversed
RADAR_ZERO_BETTER   = {'PBIAS', 'FHV', 'FMV', 'FLV',
                        'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}  # abs() + reversed

emphasis_items = list(OBJ_EMPHASIS_GROUPS.items())

# --- Global min-max per metric across ALL models × ALL objectives ---
_raw_vals = {k: [] for k in radar_keys}
for model in MODELS_FOR_ANALYSIS:
    for obj_key in OBJ_KEYS_ORDERED:
        mdict = all_metrics.get(model, {}).get(obj_key, {})
        for mkey in radar_keys:
            v = mdict.get(mkey, np.nan)
            if np.isfinite(v):
                _raw_vals[mkey].append(v)

_metric_range = {}
for mkey in radar_keys:
    vals = np.array(_raw_vals[mkey]) if _raw_vals[mkey] else np.array([0.0])
    if mkey in RADAR_LOWER_BETTER:
        vals = np.log1p(vals)
    elif mkey in RADAR_ZERO_BETTER:
        vals = np.abs(vals)
    vmin, vmax = float(vals.min()), float(vals.max())
    _metric_range[mkey] = (vmin, vmax)

def _normalise(mkey: str, raw_val: float) -> float:
    """Map raw metric value to [0, 1] where 1 = best across all models."""
    vmin, vmax = _metric_range[mkey]
    rng = vmax - vmin
    if rng == 0:
        return 0.5
    if mkey in RADAR_LOWER_BETTER:
        return (vmax - np.log1p(raw_val)) / rng
    if mkey in RADAR_ZERO_BETTER:
        return (vmax - abs(raw_val)) / rng
    return (raw_val - vmin) / rng

# Build 2×2 grid using four separate polar subplots
fig_radar = go.Figure()

domains = [
    {'x': [0.02, 0.46], 'y': [0.53, 0.97]},
    {'x': [0.54, 0.98], 'y': [0.53, 0.97]},
    {'x': [0.02, 0.46], 'y': [0.03, 0.47]},
    {'x': [0.54, 0.98], 'y': [0.03, 0.47]},
]
polar_names = ['polar', 'polar2', 'polar3', 'polar4']

for g_idx, (group_name, group_keys) in enumerate(emphasis_items):
    polar_key = polar_names[g_idx]

    for model in MODELS_FOR_ANALYSIS:
        values = []
        for mkey in radar_keys:
            metric_vals = []
            for obj_key in group_keys:
                mdict = all_metrics.get(model, {}).get(obj_key, {})
                val = mdict.get(mkey, np.nan)
                if np.isfinite(val):
                    metric_vals.append(_normalise(mkey, val))
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
            tickfont=dict(size=9),
            rotation=90,
            direction='clockwise',
            gridcolor='rgba(200,200,200,0.3)',
        ),
        bgcolor='rgba(245,245,250,0.3)',
    )
    fig_radar.update_layout(**{polar_key: polar_config})

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
          "<sup>23 canonical diagnostics (12 skill, 2 error, 1 bias, 3 FDC, 4 signature, 1 composite), "
          "all min-max normalised to 0–1 (1 = best across all models)<br>"
          "Skill metrics: higher = better · Error metrics (*): reversed so lower error → higher score · "
          "Bias & signature metrics (*): reversed so closer to 0 → higher score<br>"
          "Axes grouped clockwise: High-flow skill → Very-low-flow skill → Error → Bias → FDC → Signatures → SDEB<br>"
          "Values are averages across objectives in each group</sup>",
    height=1500,
    width=1600,
    margin=dict(t=200, b=60, l=120, r=120),
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
# The final summary heatmap distils all 52 calibrations into a single view:
# **what is the mean diagnostic score for each model across all 13 calibration
# objectives?** This answers the question *"which model is best on average?"*
# — though the answer may differ by metric. Skill metrics (NSE, KGE) use
# higher-is-better colouring; error metrics (SDEB, RMSE, MAE, |PBIAS|) use
# lower-is-better.

# %%
# Comprehensive performance summary — heatmap of mean diagnostic scores
# Same canonical diagnostics as radar plots: DIAG_SKILL_COLS + DIAG_ERR_COLS (23 total)

SUMMARY_DIAG = []  # (key, label, higher_is_better, closer_to_zero) — 12 skill + 11 error/bias
for _k, _lbl in DIAG_SKILL_COLS:
    SUMMARY_DIAG.append((_k, _lbl, True, False))
for _k, _lbl in DIAG_ERR_COLS:
    _closer = _k in {'PBIAS', 'FHV', 'FMV', 'FLV', 'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}
    SUMMARY_DIAG.append((_k, _lbl, False, _closer))

SUMMARY_DIAG_GROUPS = [
    (0,  'High-flow'),
    (3,  'Balanced (√Q)'),
    (6,  'Low-flow (log Q)'),
    (9,  'Very-low-flow (1/Q)'),
    (12, 'Error (ML/d)'),
    (14, 'Bias & FDC'),
    (17, 'Signatures (%)'),
    (21, 'Composite'),
]

summary_labels = [lbl for _, lbl, _, _ in SUMMARY_DIAG]
n_sd = len(SUMMARY_DIAG)

# Compute mean values across all 13 objectives for each model × diagnostic
z_summary = np.full((len(MODELS_FOR_ANALYSIS), n_sd), np.nan)
ann_summary = []

for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
    row_text = []
    for d_idx, (dkey, dlbl, higher_better, closer_zero) in enumerate(SUMMARY_DIAG):
        vals = []
        for obj_key in OBJ_KEYS_ORDERED:
            v = all_metrics.get(model_name, {}).get(obj_key, {}).get(dkey, np.nan)
            if np.isfinite(v):
                if closer_zero:
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
            elif closer_zero:
                row_text.append(f'{mean_val:.1f}%')
            else:
                row_text.append(f'{mean_val:.3f}')
        else:
            row_text.append('')
    ann_summary.append(row_text)

# Colour ranking: 0 = best model for that diagnostic, 1 = worst
# Normalised per column so green = best, red = worst (one clear direction)
z_rank = np.full_like(z_summary, np.nan)
for d_idx, (_, _, higher_better, _) in enumerate(SUMMARY_DIAG):
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
    width=2200,
    margin=dict(t=140, b=120, l=120, r=100),
    xaxis=dict(side='bottom', tickangle=-45, tickfont=dict(size=11)),
    yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
)
fig_summary.show()

# %% [markdown]
# ---
# ## Step 11: Final Summary and Recommendations
#
# So far we've looked at individual metrics and visualisations. Now we ask the
# summary question: **which model wins most often, and for which applications?**
#
# We approach this data-driven, not by opinion:
#
# 1. **Normalised ranking heatmap** — for each flow-emphasis group (high, balanced,
#    low, very-low, volume), we compute an aggregate score across all relevant
#    diagnostics and objectives, then rank the four models from best (#1) to
#    worst (#4). Scores are normalised to [0, 1] so all groups are comparable.
#
# 2. **Win count bar chart** — across all diagnostic × objective cells, how many
#    times does each model achieve the best value? This gives an overall
#    "consistency" measure.
#
# 3. **Data-driven recommendation table** — the best model for each application
#    type, with margin of victory over the runner-up.

# %%
# Data-driven rankings — which model is best for each flow-emphasis group?
# Two panels: (1) Normalised ranking heatmap, (2) Win count bars

RANKING_DIAG = {
    'High-flow': [('NSE', True), ('KGE', True), ("KGE_np", True), ('Pearson_r', True), ('Spearman_r', True)],
    'Balanced (√Q)': [('NSE_sqrt', True), ('KGE_sqrt', True), ('SDEB', False)],
    'Low-flow (log Q)': [('NSE_log', True), ('KGE_log', True)],
    'Very-low-flow (1/Q)': [('NSE_inv', True), ('KGE_inv', True)],
    'Volume accuracy': [('PBIAS', False), ('RMSE', False), ('MAE', False), ('FHV', False), ('FMV', False), ('FLV', False)],
    'Signatures': [('BFI_obs', False), ('BFI_sim', False), ('Sig_BFI', False), ('Sig_Flash', False), ('Sig_Q95', False), ('Sig_Q5', False)],
}

_CLOSER_TO_ZERO_KEYS = {'PBIAS', 'FHV', 'FMV', 'FLV', 'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}

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
                    if dkey in _CLOSER_TO_ZERO_KEYS:
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
    'Signatures': '#8c564b',
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
# Summary statistics and heatmaps are powerful, but sometimes you need to
# **see the hydrographs**. This step generates one interactive 4-panel report
# card per objective function, each comparing all four models side-by-side:
#
# 1. **Linear hydrograph** — peaks and timing at a glance
# 2. **Log-scale hydrograph** — low-flow and recession behaviour
# 3. **Scatter plot (log axes)** — simulated vs. observed with 1:1 line
# 4. **Flow Duration Curve** — full-spectrum comparison
#
# Scroll through the 13 report cards to build intuition about how objective
# function choice shapes model behaviour.

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
# ## Export SCE-UA Results
#
# We save two CSV files for external analysis or reporting:
#
# 1. **Calibration summary** — best objective value and runtime for each model/objective
# 2. **Comprehensive metrics** — the full diagnostic suite (NSE, KGE, FDC segments,
#    signatures, RMSE, etc.) for all 52 combinations
#
# These files enable further analysis in Excel, R, or any tool of your choice.

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
summary_df.to_csv(OUTPUT_DIR / 'model_comparison_all_13_objectives.csv', index=False)
print(f"Calibration summary saved to: {OUTPUT_DIR / 'model_comparison_all_13_objectives.csv'}")

# Export comprehensive metrics
metrics_export = []
for model_name in MODELS_FOR_ANALYSIS:
    for obj_name in OBJECTIVES.keys():
        if obj_name in all_metrics.get(model_name, {}):
            row = {'model': model_name, 'objective': obj_name}
            row.update(all_metrics[model_name][obj_name])
            metrics_export.append(row)

metrics_df = pd.DataFrame(metrics_export)
metrics_df.to_csv(OUTPUT_DIR / 'model_comparison_comprehensive_metrics.csv', index=False)
print(f"Comprehensive metrics saved to: {OUTPUT_DIR / 'model_comparison_comprehensive_metrics.csv'}")

# %% [markdown]
# ---
# ---
# # Part 2: PyDREAM Model Comparison (4 Models x 4 Flow Transforms)
#
# Part 1 used **SCE-UA** — a deterministic optimiser that returns a single best
# parameter set. Part 2 repeats the model comparison using **PyDREAM**, a
# Bayesian MCMC sampler that maps the full posterior distribution of parameters.
#
# This gives us two additional perspectives:
#
# 1. **Parameter uncertainty** — are some models better constrained by the data?
# 2. **Robustness of ranking** — does the "best model" conclusion change when we
#    account for parameter uncertainty rather than relying on a point estimate?
#
# ## Step 13: PyDREAM Configuration
#
# ### Central Question
#
# **Is any GR model better than Sacramento when evaluated through a Bayesian lens?**
#
# PyDREAM uses a `TransformedGaussianLikelihood` — one per **flow transform** —
# so the comparison axis changes from 13 named objectives to 4 transforms:
#
# | Transform | Flow Emphasis | SCE-UA Equivalent Objectives |
# |-----------|---------------|------------------------------|
# | `none` | High flows | NSE, KGE, KGE' |
# | `sqrt` | Balanced | √NSE, KGE-√Q, KGE'-√Q, SDEB |
# | `log` | Low flows | log-NSE, KGE-log, KGE'-log |
# | `inverse` | Very low flows | 1/Q NSE, KGE-1/Q, KGE'-1/Q |
#
# **16 calibrations**: 4 models × 4 transforms (12 new GR runs + 4 Sacramento
# loaded from Notebook 06).
#
# MCMC chain length and tuning parameters are scaled per model — GR4J's
# 4-dimensional posterior converges quickly, while GR6J's 6 dimensions need
# longer chains.
#
# ### Sub-question
#
# Does PyDREAM provide different insight compared to SCE-UA on the question of
# which model is best?

# %%
from pyrrm.calibration import PYDREAM_AVAILABLE
from pyrrm.calibration.objective_functions import TransformedGaussianLikelihood

if not PYDREAM_AVAILABLE:
    raise RuntimeError(
        "This section requires PyDREAM. Install with: pip install pydream"
    )

PYDREAM_TRANSFORMS = ['none', 'sqrt', 'log', 'inverse']

TRANSFORM_ORDERED = [
    ('none',    'No transform (high-flow)'),
    ('sqrt',    'Sqrt (balanced)'),
    ('log',     'Log (low-flow)'),
    ('inverse', 'Inverse (very-low-flow)'),
]
TRANSFORM_KEYS = [k for k, _ in TRANSFORM_ORDERED]
TRANSFORM_LABELS = [lbl for _, lbl in TRANSFORM_ORDERED]

LOAD_PYDREAM_FROM_PICKLE = False  # Set True to skip running when pickle missing (load-only mode)

PYDREAM_CONFIG = {
    'GR4J': dict(
        n_iterations=5000, n_chains=3, DEpairs=1, multitry=1,
        batch_size=500, min_iterations=1500,
        post_convergence_iterations=500,
    ),
    'GR5J': dict(
        n_iterations=7000, n_chains=3, DEpairs=1, multitry=3,  # multitry=2 hits PyDREAM bug in mt_evaluate_logps
        batch_size=500, min_iterations=2000,
        post_convergence_iterations=750,
    ),
    'GR6J': dict(
        n_iterations=8000, n_chains=5, DEpairs=1, multitry=3,  # multitry=2 hits PyDREAM bug in mt_evaluate_logps
        batch_size=500, min_iterations=2500,
        post_convergence_iterations=1000,
    ),
}
PYDREAM_COMMON = dict(
    snooker=0.15, convergence_threshold=1.05, patience=2,
    adapt_gamma=False, parallel=False, verbose=True,
)

NB06_REPORTS_DIR = Path('../test_data/06_algorithm_comparison/reports')

MODEL_CLASSES = {
    'GR4J': GR4J, 'GR5J': GR5J, 'GR6J': GR6J, 'Sacramento': Sacramento,
}

print("=" * 70)
print("PYDREAM MODEL COMPARISON CONFIGURATION")
print("=" * 70)
print(f"\nTransforms: {PYDREAM_TRANSFORMS}")
print(f"Models: {list(PYDREAM_CONFIG.keys())} + Sacramento (loaded from NB06)")
print(f"Total calibrations: 4 models × 4 transforms = 16")
print(f"Load from pickle: {LOAD_PYDREAM_FROM_PICKLE}")
print(f"\nAdaptive settings per model:")
for m, cfg in PYDREAM_CONFIG.items():
    print(f"  {m:12s}: {cfg['n_iterations']} iter, {cfg['n_chains']} chains, "
          f"DEpairs={cfg['DEpairs']}, multitry={cfg['multitry']}")
print(f"  {'Sacramento':12s}: loaded from NB06 (15k iter, 5 chains)")

# %% [markdown]
# ---
# ## Step 14: Load Sacramento PyDREAM Results from Notebook 06
#
# Sacramento's PyDREAM posteriors were computed in Notebook 06 (Algorithm
# Comparison). We load those results here so Sacramento can be included in the
# 4-model comparison without re-running 60+ hours of MCMC.

# %%
pydream_results = {m: {} for m in MODELS_FOR_ANALYSIS}

print("=" * 70)
print("LOADING SACRAMENTO PYDREAM RESULTS FROM NOTEBOOK 06")
print("=" * 70)

for transform in PYDREAM_TRANSFORMS:
    pkl_path = NB06_REPORTS_DIR / f'410734_sacramento_dream_{transform}.pkl'
    if pkl_path.exists():
        report = CalibrationReport.load(str(pkl_path))
        pydream_results['Sacramento'][transform] = report.result
        cd = report.result.convergence_diagnostics or {}
        converged = cd.get('converged', False)
        max_gr = cd.get('max_gr', 'N/A')
        print(f"  ✓ {transform:<10}: Loaded (log-lik: {report.result.best_objective:.4f}, "
              f"converged: {converged}, max GR: {max_gr})")
    else:
        print(f"  ✗ {transform:<10}: Not found at {pkl_path}")
        print(f"    Run Notebook 06 first to generate Sacramento PyDREAM results.")

print(f"\nSacramento PyDREAM results loaded: {len(pydream_results['Sacramento'])}/4")

# %% [markdown]
# ---
# ## Step 15: Run PyDREAM Calibrations for GR4J, GR5J, GR6J
#
# For each GR model we run PyDREAM with all 4 flow transforms. Like the SCE-UA
# steps, these cells check for existing pickle files first and only run the MCMC
# if results haven't been saved yet.
#
# Each calibration uses the batch-mode early stopping criterion: chains are
# checked for convergence (Gelman-Rubin R-hat < 1.05) every `batch_size`
# iterations. Once all parameters converge for `patience` consecutive batches,
# additional `post_convergence_iterations` are drawn for reliable posterior
# summaries.

# %% [markdown]
# ### 15.1 GR4J PyDREAM Calibrations

# %%
print("=" * 70)
print("GR4J PYDREAM CALIBRATIONS (4 transforms)")
print("=" * 70)

for transform in PYDREAM_TRANSFORMS:
    pkl_name = f'410734_gr4j_dream_{transform}'
    pkl_path = MODELS_REPORTS_DIR / f'{pkl_name}.pkl'

    if pkl_path.exists():
        report = CalibrationReport.load(str(pkl_path))
        pydream_results['GR4J'][transform] = report.result
        print(f"  ✓ {transform:<10}: Loaded from pickle")
        continue

    if LOAD_PYDREAM_FROM_PICKLE:
        print(f"  - {transform:<10}: Pickle not found, skipping (LOAD_PYDREAM_FROM_PICKLE=True)")
        continue

    print(f"\n  Running GR4J / {transform}...")
    likelihood = TransformedGaussianLikelihood(transform)
    runner = CalibrationRunner(
        model=GR4J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs, observed=cal_observed,
        objective=likelihood, warmup_period=WARMUP_DAYS,
    )
    progress_file = MODELS_REPORTS_DIR / f'progress_{pkl_name}'
    start_time = time.time()
    try:
        result = runner.run_pydream(
            **PYDREAM_CONFIG['GR4J'], **PYDREAM_COMMON,
            dbname=str(progress_file),
        )
        elapsed = time.time() - start_time
        print(f"  ✓ {transform:<10}: Done in {elapsed:.1f}s (log-lik: {result.best_objective:.4f})")
        report = runner.create_report(result, catchment_info={
            'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2,
        })
        report.save(str(pkl_path.with_suffix('')))
        pydream_results['GR4J'][transform] = result
    except Exception as e:
        print(f"  ✗ {transform:<10}: Failed — {e}")

print(f"\nGR4J PyDREAM: {len(pydream_results['GR4J'])}/4 transforms")

# %% [markdown]
# ### 15.2 GR5J PyDREAM Calibrations

# %%
print("=" * 70)
print("GR5J PYDREAM CALIBRATIONS (4 transforms)")
print("=" * 70)

for transform in PYDREAM_TRANSFORMS:
    pkl_name = f'410734_gr5j_dream_{transform}'
    pkl_path = MODELS_REPORTS_DIR / f'{pkl_name}.pkl'

    if pkl_path.exists():
        report = CalibrationReport.load(str(pkl_path))
        pydream_results['GR5J'][transform] = report.result
        print(f"  ✓ {transform:<10}: Loaded from pickle")
        continue

    if LOAD_PYDREAM_FROM_PICKLE:
        print(f"  - {transform:<10}: Pickle not found, skipping (LOAD_PYDREAM_FROM_PICKLE=True)")
        continue

    print(f"\n  Running GR5J / {transform}...")
    likelihood = TransformedGaussianLikelihood(transform)
    runner = CalibrationRunner(
        model=GR5J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs, observed=cal_observed,
        objective=likelihood, warmup_period=WARMUP_DAYS,
    )
    progress_file = MODELS_REPORTS_DIR / f'progress_{pkl_name}'
    start_time = time.time()
    try:
        result = runner.run_pydream(
            **PYDREAM_CONFIG['GR5J'], **PYDREAM_COMMON,
            dbname=str(progress_file),
        )
        elapsed = time.time() - start_time
        print(f"  ✓ {transform:<10}: Done in {elapsed:.1f}s (log-lik: {result.best_objective:.4f})")
        report = runner.create_report(result, catchment_info={
            'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2,
        })
        report.save(str(pkl_path.with_suffix('')))
        pydream_results['GR5J'][transform] = result
    except Exception as e:
        print(f"  ✗ {transform:<10}: Failed — {e}")

print(f"\nGR5J PyDREAM: {len(pydream_results['GR5J'])}/4 transforms")

# %% [markdown]
# ### 15.3 GR6J PyDREAM Calibrations

# %%
print("=" * 70)
print("GR6J PYDREAM CALIBRATIONS (4 transforms)")
print("=" * 70)

for transform in PYDREAM_TRANSFORMS:
    pkl_name = f'410734_gr6j_dream_{transform}'
    pkl_path = MODELS_REPORTS_DIR / f'{pkl_name}.pkl'

    if pkl_path.exists():
        report = CalibrationReport.load(str(pkl_path))
        pydream_results['GR6J'][transform] = report.result
        print(f"  ✓ {transform:<10}: Loaded from pickle")
        continue

    if LOAD_PYDREAM_FROM_PICKLE:
        print(f"  - {transform:<10}: Pickle not found, skipping (LOAD_PYDREAM_FROM_PICKLE=True)")
        continue

    print(f"\n  Running GR6J / {transform}...")
    likelihood = TransformedGaussianLikelihood(transform)
    runner = CalibrationRunner(
        model=GR6J(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs, observed=cal_observed,
        objective=likelihood, warmup_period=WARMUP_DAYS,
    )
    progress_file = MODELS_REPORTS_DIR / f'progress_{pkl_name}'
    start_time = time.time()
    try:
        result = runner.run_pydream(
            **PYDREAM_CONFIG['GR6J'], **PYDREAM_COMMON,
            dbname=str(progress_file),
        )
        elapsed = time.time() - start_time
        print(f"  ✓ {transform:<10}: Done in {elapsed:.1f}s (log-lik: {result.best_objective:.4f})")
        report = runner.create_report(result, catchment_info={
            'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2,
        })
        report.save(str(pkl_path.with_suffix('')))
        pydream_results['GR6J'][transform] = result
    except Exception as e:
        print(f"  ✗ {transform:<10}: Failed — {e}")

print(f"\nGR6J PyDREAM: {len(pydream_results['GR6J'])}/4 transforms")

# %% [markdown]
# ---
# ## Step 16: Organise PyDREAM Results, Generate Simulations, Compute Metrics
#
# This step mirrors Steps 8-9 from Part 1 but for the PyDREAM results. We:
#
# 1. Summarise how many transforms were successfully calibrated per model.
# 2. Re-run each model with its MAP (maximum a posteriori) parameters to
#    produce simulated hydrographs.
# 3. Compute the same comprehensive diagnostic suite used in Part 1, enabling
#    direct comparison between SCE-UA and PyDREAM outcomes.

# %%
print("=" * 70)
print("ORGANISING PYDREAM RESULTS (4 Models × 4 Transforms)")
print("=" * 70)

# Summary table
print(f"\n{'Model':<12} {'Transforms':>10}")
print("-" * 24)
for model_name in MODELS_FOR_ANALYSIS:
    n = len(pydream_results.get(model_name, {}))
    print(f"{model_name:<12} {n:>10}/4")

# %%
# Generate simulations for all PyDREAM calibrations
print("=" * 70)
print("GENERATING PYDREAM SIMULATIONS")
print("=" * 70)

pydream_simulations = {}

for model_name in MODELS_FOR_ANALYSIS:
    pydream_simulations[model_name] = {}
    for transform in PYDREAM_TRANSFORMS:
        if transform not in pydream_results.get(model_name, {}):
            continue
        result = pydream_results[model_name][transform]
        model = MODEL_CLASSES[model_name](catchment_area_km2=CATCHMENT_AREA_KM2)
        model.set_parameters(result.best_parameters)
        model.reset()
        sim_out = model.run(cal_inputs)
        pydream_simulations[model_name][transform] = get_flow_column(sim_out)[WARMUP_DAYS:]
    print(f"  ✓ {model_name}: {len(pydream_simulations[model_name])} simulations")

pd_obs_flow = cal_observed[WARMUP_DAYS:]
pd_comparison_dates = cal_inputs.index[WARMUP_DAYS:]
print(f"\nSimulations complete! ({len(pd_obs_flow):,} days each)")

# %%
# Compute comprehensive metrics for all PyDREAM calibrations
print("=" * 70)
print("COMPUTING COMPREHENSIVE EVALUATION (PYDREAM)")
print("=" * 70)

pydream_metrics = {}
for model_name in MODELS_FOR_ANALYSIS:
    pydream_metrics[model_name] = {}
    for transform in PYDREAM_TRANSFORMS:
        if transform in pydream_simulations.get(model_name, {}):
            pydream_metrics[model_name][transform] = comprehensive_evaluation(
                pd_obs_flow, pydream_simulations[model_name][transform]
            )
    print(f"  ✓ {model_name}: {len(pydream_metrics[model_name])} evaluations")

# Print summary table
print(f"\n{'Model':<12}", end="")
for t in TRANSFORM_KEYS:
    print(f"  {'NSE('+t+')':>16}", end="")
print()
print("-" * 80)
for model_name in MODELS_FOR_ANALYSIS:
    print(f"{model_name:<12}", end="")
    for t in TRANSFORM_KEYS:
        nse_val = pydream_metrics.get(model_name, {}).get(t, {}).get('NSE', np.nan)
        print(f"  {nse_val:>16.4f}", end="")
    print()

# %% [markdown]
# ---
# ## Step 17: PyDREAM Model Comparison Visualisations
#
# The following plots mirror the SCE-UA visualisations from Step 10, but with
# **4 flow transforms** on the y-axis instead of 13 objectives. This lets us
# compare how the models perform under a Bayesian framework and whether the
# ranking conclusions from Part 1 hold up.

# %% [markdown]
# ### 17.1 Flow Duration Curves by Transform (PyDREAM)
#
# One FDC panel per flow transform — do the Bayesian MAP parameters reproduce
# the observed flow distribution as well as the SCE-UA point estimates?

# %%
pd_obs_sorted = np.sort(pd_obs_flow)[::-1]
pd_exceedance = np.arange(1, len(pd_obs_sorted) + 1) / len(pd_obs_sorted) * 100

fig_pd_fdc = make_subplots(
    rows=2, cols=2,
    subplot_titles=TRANSFORM_LABELS,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

for idx, (t_key, t_label) in enumerate(TRANSFORM_ORDERED):
    row = idx // 2 + 1
    col = idx % 2 + 1

    fig_pd_fdc.add_trace(
        go.Scatter(x=pd_exceedance, y=pd_obs_sorted, name='Observed' if idx == 0 else None,
                   showlegend=(idx == 0), line=dict(color='black', width=2)),
        row=row, col=col,
    )

    for model_name in MODELS_FOR_ANALYSIS:
        if t_key in pydream_simulations.get(model_name, {}):
            sim_sorted = np.sort(pydream_simulations[model_name][t_key])[::-1]
            fig_pd_fdc.add_trace(
                go.Scatter(x=pd_exceedance, y=sim_sorted,
                           name=model_name if idx == 0 else None,
                           showlegend=(idx == 0),
                           line=dict(color=MODEL_COLORS[model_name], width=1.5)),
                row=row, col=col,
            )

    fig_pd_fdc.update_yaxes(type='log', row=row, col=col)
    fig_pd_fdc.update_xaxes(title_text='Exceedance %' if row == 2 else None, row=row, col=col)
    fig_pd_fdc.update_yaxes(title_text='Flow (ML/day)' if col == 1 else None, row=row, col=col)

fig_pd_fdc.update_layout(
    title="<b>How does flow-emphasis shift the FDC fit across models? — PyDREAM calibrations</b>",
    height=700,
    legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='right', x=1.0),
)
fig_pd_fdc.show()

# %% [markdown]
# ### 17.2 FDC Segment Error Heatmap (PyDREAM)
#
# Volume bias for each FDC segment (Peak through Very Low) across all models
# and transforms — the same diagnostic as Section 10.3 but for PyDREAM MAP
# parameters.

# %%
pd_fdc_segments = ['FDC_Peak', 'FDC_High', 'FDC_Mid', 'FDC_Low', 'FDC_VeryLow']

fig_pd_seg = make_subplots(
    rows=1, cols=len(MODELS_FOR_ANALYSIS),
    subplot_titles=MODELS_FOR_ANALYSIS,
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

for idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
    data = []
    for t_key in TRANSFORM_KEYS:
        row_data = []
        for seg in pd_fdc_segments:
            val = pydream_metrics.get(model_name, {}).get(t_key, {}).get(seg, np.nan)
            row_data.append(val)
        data.append(row_data)

    fig_pd_seg.add_trace(
        go.Heatmap(
            z=data,
            x=['Peak', 'High', 'Mid', 'Low', 'V.Low'],
            y=TRANSFORM_LABELS,
            colorscale='RdBu', zmid=0, zmin=-50, zmax=50,
            showscale=(idx == len(MODELS_FOR_ANALYSIS) - 1),
            text=np.round(data, 1),
            texttemplate='%{text}',
            textfont={"size": 9},
        ),
        row=1, col=idx + 1,
    )

fig_pd_seg.update_layout(
    title="<b>Where does each model over- or underestimate volume across FDC segments? — PyDREAM</b><br>"
          "<sup>Red = overestimate, Blue = underestimate (% volume bias)</sup>",
    height=500, width=1600, margin=dict(l=250),
)
fig_pd_seg.show()

# %% [markdown]
# ### 17.3 FDC Alternative Visualisations: GR vs Sacramento (PyDREAM)
#
# Additional views including a winner heatmap (which model has the smallest
# segment error for each transform?) and scatter plots comparing GR model
# segment errors against Sacramento.

# %%
# 1) Winner heatmap
pd_seg_labels_short = ['Peak', 'High', 'Mid', 'Low', 'V.Low']
pd_winner_z = []
pd_winner_text = []
for t_key in TRANSFORM_KEYS:
    row_z, row_txt = [], []
    for seg in pd_fdc_segments:
        vals = [pydream_metrics.get(m, {}).get(t_key, {}).get(seg, np.nan) for m in MODELS_FOR_ANALYSIS]
        abs_vals = [abs(x) if np.isfinite(x) else np.inf for x in vals]
        if all(np.isinf(x) for x in abs_vals):
            row_z.append(np.nan); row_txt.append('—')
        else:
            idx_best = int(np.argmin(abs_vals))
            best_model = MODELS_FOR_ANALYSIS[idx_best]
            row_z.append({m: i for i, m in enumerate(MODELS_FOR_ANALYSIS)}[best_model])
            row_txt.append(f"{best_model}<br>{vals[idx_best]:.0f}")
    pd_winner_z.append(row_z)
    pd_winner_text.append(row_txt)

fig_pd_win = go.Figure(data=go.Heatmap(
    z=pd_winner_z, x=pd_seg_labels_short, y=TRANSFORM_LABELS,
    text=pd_winner_text, texttemplate='%{text}', textfont={"size": 10},
    zmin=0, zmax=3,
    colorscale=[[0, MODEL_COLORS['GR4J']], [0.33, MODEL_COLORS['GR5J']],
                [0.67, MODEL_COLORS['GR6J']], [1, MODEL_COLORS['Sacramento']]],
    showscale=True,
    colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=MODELS_FOR_ANALYSIS, len=0.5),
))
fig_pd_win.update_layout(
    title="<b>Which model has the lowest absolute FDC bias? — PyDREAM</b><br>"
          "<sup>Cell color = winning model; number = bias (%)</sup>",
    height=450, width=900, xaxis_title='FDC segment', margin=dict(l=250),
)
fig_pd_win.show()

# %%
# 2) Difference heatmaps: |GR bias| − |Sacramento bias|
pd_sac_bias = {}
for t_key in TRANSFORM_KEYS:
    pd_sac_bias[t_key] = {seg: pydream_metrics.get('Sacramento', {}).get(t_key, {}).get(seg, np.nan)
                           for seg in pd_fdc_segments}

fig_pd_diff = make_subplots(
    rows=1, cols=len(MODELS_GR_ONLY),
    subplot_titles=[f'|{m}| − |Sacramento|' for m in MODELS_GR_ONLY],
    horizontal_spacing=0.06,
)
for idx, gr_model in enumerate(MODELS_GR_ONLY):
    raw_data, log_data = [], []
    for t_key in TRANSFORM_KEYS:
        raw_row, log_row = [], []
        for seg in pd_fdc_segments:
            gr_val = pydream_metrics.get(gr_model, {}).get(t_key, {}).get(seg, np.nan)
            sac_val = pd_sac_bias[t_key].get(seg, np.nan)
            if np.isfinite(gr_val) and np.isfinite(sac_val):
                diff = abs(gr_val) - abs(sac_val)
                raw_row.append(diff)
                log_row.append(symlog(diff))
            else:
                raw_row.append(np.nan); log_row.append(np.nan)
        raw_data.append(raw_row); log_data.append(log_row)

    is_last = (idx == len(MODELS_GR_ONLY) - 1)
    fig_pd_diff.add_trace(
        go.Heatmap(
            z=log_data, x=pd_seg_labels_short, y=TRANSFORM_LABELS,
            colorscale='RdBu_r', zmid=0,
            text=np.round(raw_data, 1), texttemplate='%{text}', textfont={"size": 9},
            showscale=is_last,
            colorbar=dict(
                title='|bias| diff (%)',
                tickvals=cbar_transformed, ticktext=[str(v) for v in cbar_orig],
            ) if is_last else None,
        ),
        row=1, col=idx + 1,
    )

fig_pd_diff.update_layout(
    title="<b>Which model has lower absolute FDC bias — each GR or Sacramento? — PyDREAM</b><br>"
          "<sup>Negative (blue) = GR closer to zero bias; Positive (red) = Sacramento closer</sup>",
    height=450, width=1400, margin=dict(l=250),
)
fig_pd_diff.show()

# %%
# 3) Box + strip plots: distribution of bias across 4 transforms per segment
fig_pd_box = make_subplots(
    rows=1, cols=5,
    subplot_titles=[f.replace('FDC_', '') for f in pd_fdc_segments],
    horizontal_spacing=0.05,
)
for seg_idx, seg in enumerate(pd_fdc_segments):
    for model_name in MODELS_FOR_ANALYSIS:
        y_vals, hover_labels = [], []
        for t_key, t_lbl in TRANSFORM_ORDERED:
            v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(seg, np.nan)
            if np.isfinite(v):
                y_vals.append(v); hover_labels.append(t_lbl)
        fig_pd_box.add_trace(
            go.Box(
                y=y_vals, name=model_name,
                marker_color=MODEL_COLORS[model_name], line_color=MODEL_COLORS[model_name],
                boxmean=True, showlegend=(seg_idx == 0), legendgroup=model_name,
            ),
            row=1, col=seg_idx + 1,
        )
        fig_pd_box.add_trace(
            go.Scatter(
                y=y_vals, x=[model_name] * len(y_vals), mode='markers',
                marker=dict(size=6, color=MODEL_COLORS[model_name], opacity=0.6),
                text=hover_labels,
                hovertemplate='%{text}<br>Bias: %{y:.1f}%<extra></extra>',
                showlegend=False, legendgroup=model_name,
            ),
            row=1, col=seg_idx + 1,
        )
    fig_pd_box.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.6, row=1, col=seg_idx + 1)
    fig_pd_box.update_yaxes(title_text='% volume bias' if seg_idx == 0 else None, row=1, col=seg_idx + 1)

fig_pd_box.update_layout(
    title="<b>Which model most consistently achieves low FDC bias across all transforms? — PyDREAM</b><br>"
          "<sup>Each box = distribution over 4 transforms; dashed line = zero; diamond = mean</sup>",
    height=550, width=1400,
    legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1.0),
)
fig_pd_box.show()

# %%
# 4) Summary breakdown: mean |bias| + win count by FDC segment
pd_mean_abs = {m: {} for m in MODELS_FOR_ANALYSIS}
pd_wins_seg = {m: {seg: 0 for seg in pd_fdc_segments} for m in MODELS_FOR_ANALYSIS}

for model_name in MODELS_FOR_ANALYSIS:
    for seg in pd_fdc_segments:
        vals = []
        for t_key in TRANSFORM_KEYS:
            v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(seg, np.nan)
            if np.isfinite(v):
                vals.append(abs(v))
        pd_mean_abs[model_name][seg] = np.mean(vals) if vals else np.nan

for t_key in TRANSFORM_KEYS:
    for seg in pd_fdc_segments:
        abs_vals = [abs(pydream_metrics.get(m, {}).get(t_key, {}).get(seg, np.nan)) for m in MODELS_FOR_ANALYSIS]
        valid = [i for i, v in enumerate(abs_vals) if np.isfinite(v)]
        if valid:
            idx_best = valid[int(np.argmin([abs_vals[i] for i in valid]))]
            pd_wins_seg[MODELS_FOR_ANALYSIS[idx_best]][seg] += 1

fig_pd_sum = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Mean |bias| by FDC segment (lower is better)',
                    'Win count by FDC segment (higher is better)'],
    horizontal_spacing=0.12,
)
n_segs_pd = len(pd_fdc_segments)
bar_w = 0.14
grp_w = n_segs_pd * bar_w
grp_gap = 0.35
x_gc = np.arange(len(MODELS_FOR_ANALYSIS)) * (grp_w + grp_gap)

for s_idx, (seg, seg_lbl) in enumerate(zip(pd_fdc_segments, pd_seg_labels_short)):
    x_bars = x_gc + (s_idx - (n_segs_pd - 1) / 2) * bar_w
    fig_pd_sum.add_trace(
        go.Bar(name=seg_lbl, x=x_bars,
               y=[pd_mean_abs[m][seg] for m in MODELS_FOR_ANALYSIS],
               width=bar_w, marker_color=seg_colors[seg], showlegend=True, legendgroup=seg),
        row=1, col=1,
    )

fig_pd_sum.update_xaxes(tickvals=x_gc, ticktext=MODELS_FOR_ANALYSIS, row=1, col=1)

for s_idx, (seg, seg_lbl) in enumerate(zip(pd_fdc_segments, pd_seg_labels_short)):
    fig_pd_sum.add_trace(
        go.Bar(name=seg_lbl, x=MODELS_FOR_ANALYSIS,
               y=[pd_wins_seg[m][seg] for m in MODELS_FOR_ANALYSIS],
               marker_color=seg_colors[seg], showlegend=False, legendgroup=seg),
        row=1, col=2,
    )

fig_pd_sum.update_layout(
    title="<b>Where does each model excel across the flow duration curve? — PyDREAM</b><br>"
          "<sup>Left: mean absolute bias; Right: wins out of 4 transforms per segment</sup>",
    barmode='stack', height=500, width=1300,
    legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02, title_text='FDC segment'),
)
fig_pd_sum.update_yaxes(title_text='Mean |bias| (%)', row=1, col=1)
fig_pd_sum.update_yaxes(title_text='Wins (out of 4)', row=1, col=2)
fig_pd_sum.show()

# %%
# 5) Scatter: |GR bias| vs |Sacramento bias| (PyDREAM)
fig_pd_scat = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f'{m} vs Sacramento' for m in MODELS_GR_ONLY],
    horizontal_spacing=0.08,
)
for idx, gr_model in enumerate(MODELS_GR_ONLY):
    n_below, n_total = 0, 0
    for seg_i, seg in enumerate(pd_fdc_segments):
        x_vals, y_vals, hover = [], [], []
        for t_key in TRANSFORM_KEYS:
            sac_val = pd_sac_bias[t_key].get(seg, np.nan)
            gr_val = pydream_metrics.get(gr_model, {}).get(t_key, {}).get(seg, np.nan)
            if np.isfinite(sac_val) and np.isfinite(gr_val):
                abs_sac = abs(sac_val) + 0.1
                abs_gr = abs(gr_val) + 0.1
                x_vals.append(abs_sac); y_vals.append(abs_gr)
                hover.append(f"{dict(TRANSFORM_ORDERED).get(t_key, t_key)} / {pd_seg_labels_short[seg_i]}")
                n_total += 1
                if abs_gr < abs_sac:
                    n_below += 1
        fig_pd_scat.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                name=pd_seg_labels_short[seg_i],
                marker=dict(size=9, color=seg_scatter_colors[seg], opacity=0.75),
                text=hover, hovertemplate='%{text}<br>|Sac|: %{x:.1f}%<br>|GR|: %{y:.1f}%<extra></extra>',
                showlegend=(idx == 0), legendgroup=seg,
            ),
            row=1, col=idx + 1,
        )
    fig_pd_scat.add_trace(
        go.Scatter(x=[0.1, 1000], y=[0.1, 1000], mode='lines',
                   line=dict(dash='dash', color='gray', width=1),
                   name='1:1', showlegend=(idx == 0)),
        row=1, col=idx + 1,
    )
    pct = n_below / n_total * 100 if n_total > 0 else 0
    fig_pd_scat.add_annotation(
        text=f"<b>{gr_model} wins {n_below}/{n_total} ({pct:.0f}%)</b>",
        x=0.05, y=0.95, xanchor='left', yanchor='top',
        xref=f'x{idx + 1} domain' if idx > 0 else 'x domain',
        yref=f'y{idx + 1} domain' if idx > 0 else 'y domain',
        showarrow=False, font=dict(size=12, color=MODEL_COLORS[gr_model]),
        bgcolor='rgba(255,255,255,0.8)', bordercolor=MODEL_COLORS[gr_model], borderwidth=1,
    )
    fig_pd_scat.update_xaxes(type='log', title_text='|Sacramento bias| (%)', row=1, col=idx + 1)
    fig_pd_scat.update_yaxes(type='log', title_text=f'|{gr_model} bias| (%)' if idx == 0 else None, row=1, col=idx + 1)

fig_pd_scat.update_layout(
    title="<b>Does any GR model consistently outperform Sacramento on FDC bias? — PyDREAM</b><br>"
          "<sup>Points below 1:1 line = GR has lower absolute bias; coloured by FDC segment</sup>",
    height=500, width=1400,
    legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02, title_text='FDC segment'),
)
fig_pd_scat.show()

# %% [markdown]
# ### 17.4 Hydrologic Signature Errors (PyDREAM)
#
# Relative errors in key hydrologic signatures (Q5, Q50, Q95, BFI, flashiness)
# for the PyDREAM MAP parameters. Compare with Section 10.4 (SCE-UA) to check
# whether Bayesian calibration produces different signature behaviour.

# %%
pd_sig_metrics = ['Q95_error', 'Q50_error', 'Q5_error', 'Flashiness_error', 'BFI_error']
pd_sig_labels = ['Q95 (High Flow)', 'Q50 (Median)', 'Q5 (Low Flow)', 'Flashiness', 'Baseflow Index']

fig_pd_sig = make_subplots(
    rows=2, cols=2,
    subplot_titles=TRANSFORM_LABELS,
    vertical_spacing=0.16, horizontal_spacing=0.12,
)

for g_idx, (t_key, t_label) in enumerate(TRANSFORM_ORDERED):
    r = g_idx // 2 + 1
    c = g_idx % 2 + 1
    z_raw = np.full((len(pd_sig_metrics), len(MODELS_FOR_ANALYSIS)), np.nan)
    for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
        for s_idx, sig in enumerate(pd_sig_metrics):
            v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(sig, np.nan)
            if np.isfinite(v):
                z_raw[s_idx, m_idx] = v

    z_sc = np.vectorize(soft_clamp)(z_raw)
    ann_text = []
    for s_idx in range(z_raw.shape[0]):
        row_text = []
        for m_idx in range(z_raw.shape[1]):
            v = z_raw[s_idx, m_idx]
            if np.isfinite(v):
                if abs(v) >= 100: row_text.append(f'{v:+.0f}%')
                elif abs(v) >= 10: row_text.append(f'{v:+.1f}%')
                else: row_text.append(f'{v:+.2f}%')
            else:
                row_text.append('')
        ann_text.append(row_text)

    show_colorbar = (g_idx == 1)
    fig_pd_sig.add_trace(
        go.Heatmap(
            z=z_sc, x=MODELS_FOR_ANALYSIS, y=pd_sig_labels,
            text=ann_text, texttemplate='%{text}', textfont=dict(size=11),
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            showscale=show_colorbar,
            colorbar=dict(
                title='Avg error', tickvals=cb_ticks, ticktext=cb_labels, len=0.9, y=0.5,
            ) if show_colorbar else None,
        ),
        row=r, col=c,
    )

fig_pd_sig.update_layout(
    title="<b>How well does each model reproduce hydrologic signatures? — PyDREAM</b><br>"
          "<sup>% error from observed — blue = underestimate, red = overestimate, white = unbiased</sup>",
    height=700, width=1100, margin=dict(t=100),
)
fig_pd_sig.show()

# %% [markdown]
# ### 17.5 Traditional Error Metrics (RMSE, MAE, PBIAS) — PyDREAM
#
# RMSE, MAE, and PBIAS for the PyDREAM MAP parameters, grouped by flow
# transform. These complement the skill metrics and provide a simple,
# interpretable view of absolute accuracy.

# %%
for metric in ['RMSE', 'MAE', 'PBIAS']:
    data = []
    for t_key in TRANSFORM_KEYS:
        row_data = []
        for model_name in MODELS_FOR_ANALYSIS:
            val = pydream_metrics.get(model_name, {}).get(t_key, {}).get(metric, np.nan)
            row_data.append(val)
        data.append(row_data)

    if metric == 'PBIAS':
        cs, zm, zn, zx = 'RdBu', 0, -100, 100
    else:
        cs, zm, zn, zx = 'Viridis', None, None, None

    heatmap_kw = dict(z=data, x=MODELS_FOR_ANALYSIS, y=TRANSFORM_LABELS,
                      colorscale=cs, zmid=zm,
                      text=np.round(data, 2), texttemplate='%{text}', textfont={"size": 11})
    if zn is not None:
        heatmap_kw['zmin'] = zn; heatmap_kw['zmax'] = zx

    fig_m = go.Figure(data=go.Heatmap(**heatmap_kw))
    fig_m.update_layout(
        title=f"<b>Which model-transform combination minimises {metric}? — PyDREAM</b>",
        xaxis_title='Model', yaxis_title='Flow Transform',
        height=400, width=800, margin=dict(l=250),
    )
    fig_m.show()

# %% [markdown]
# ### 17.6 Model Improvement Over Sacramento (PyDREAM)
#
# Difference heatmaps (GR − Sacramento) for the PyDREAM results. Green cells
# indicate a GR model outperforms Sacramento for that metric/transform. This
# directly tests whether the parsimony advantage of GR models extends to the
# Bayesian setting.

# %%
pd_n_transform = len(TRANSFORM_KEYS)

fig_pd_imp = make_subplots(
    rows=3, cols=1,
    subplot_titles=[f'<b>{m}</b> vs Sacramento' for m in MODELS_GR_ONLY],
    vertical_spacing=0.06,
    shared_xaxes=True,
)

pd_all_raw = {}
for gr_model in MODELS_GR_ONLY:
    z_raw = np.full((pd_n_transform, len(DIAG_SKILL)), np.nan)
    for d_idx, (dkey, dlbl) in enumerate(DIAG_SKILL):
        for t_idx, t_key in enumerate(TRANSFORM_KEYS):
            sac_val = pydream_metrics.get('Sacramento', {}).get(t_key, {}).get(dkey, np.nan)
            gr_val = pydream_metrics.get(gr_model, {}).get(t_key, {}).get(dkey, np.nan)
            if np.isfinite(sac_val) and np.isfinite(gr_val):
                diff = gr_val - sac_val
                z_raw[t_idx, d_idx] = -diff if dkey in DIAG_MINIMIZE else diff
    pd_all_raw[gr_model] = z_raw

for m_idx, gr_model in enumerate(MODELS_GR_ONLY):
    r = m_idx + 1
    z_raw = pd_all_raw[gr_model]
    z_sc = np.vectorize(lambda x: soft_clamp(x, k=KNEE_IMP))(z_raw)

    ann_text = []
    for t_idx in range(pd_n_transform):
        row_text = []
        for d_idx in range(len(DIAG_SKILL)):
            v = z_raw[t_idx, d_idx]
            row_text.append(f'{v:+.3f}' if np.isfinite(v) else '')
        ann_text.append(row_text)

    fig_pd_imp.add_trace(
        go.Heatmap(
            z=z_sc, x=diag_labels, y=TRANSFORM_LABELS,
            text=ann_text, texttemplate='%{text}', textfont=dict(size=9),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            showscale=(m_idx == 0),
            colorbar=dict(
                title=dict(text='GR − Sac', side='right'),
                tickvals=cb_ticks_imp, ticktext=cb_labels_imp,
                len=0.85, y=0.5, x=1.02, xpad=10,
            ) if m_idx == 0 else None,
        ),
        row=r, col=1,
    )

fig_pd_imp.update_xaxes(showticklabels=False, row=1, col=1)
fig_pd_imp.update_xaxes(showticklabels=False, row=2, col=1)
fig_pd_imp.update_xaxes(tickangle=-45, tickfont=dict(size=11), row=3, col=1)

fig_pd_imp.update_layout(
    title="<b>Do GR models improve over Sacramento, and for which diagnostics? — PyDREAM</b><br>"
          "<sup>Blue = GR outperforms Sacramento · Red = Sacramento outperforms GR · White ≈ no difference</sup>",
    height=900, width=1700, margin=dict(l=250, t=140, r=160, b=100),
)
fig_pd_imp.show()

# %% [markdown]
# ### 17.7 Radar Charts: Model Profiles by Transform (PyDREAM)
#
# One radar chart per flow transform showing each model's diagnostic profile.
# Compare with Section 10.7 to see if the Bayesian approach changes the
# relative model strengths.

# %%
pd_radar_keys = [k for k, _ in RADAR_METRICS]
pd_radar_labels = [lbl for _, lbl in RADAR_METRICS]

# --- Global min-max per metric across ALL models × ALL transforms (PyDREAM data) ---
_pd_raw = {k: [] for k in pd_radar_keys}
for model in MODELS_FOR_ANALYSIS:
    for t_key in TRANSFORM_KEYS:
        mdict = pydream_metrics.get(model, {}).get(t_key, {})
        for mkey in pd_radar_keys:
            v = mdict.get(mkey, np.nan)
            if np.isfinite(v):
                _pd_raw[mkey].append(v)

_pd_range = {}
for mkey in pd_radar_keys:
    vals = np.array(_pd_raw[mkey]) if _pd_raw[mkey] else np.array([0.0])
    if mkey in RADAR_LOWER_BETTER:
        vals = np.log1p(vals)
    elif mkey in RADAR_ZERO_BETTER:
        vals = np.abs(vals)
    _pd_range[mkey] = (float(vals.min()), float(vals.max()))

def _pd_normalise(mkey: str, raw_val: float) -> float:
    """Map raw metric to [0, 1] where 1 = best (PyDREAM data)."""
    vmin, vmax = _pd_range[mkey]
    rng = vmax - vmin
    if rng == 0:
        return 0.5
    if mkey in RADAR_LOWER_BETTER:
        return (vmax - np.log1p(raw_val)) / rng
    if mkey in RADAR_ZERO_BETTER:
        return (vmax - abs(raw_val)) / rng
    return (raw_val - vmin) / rng

fig_pd_radar = go.Figure()

pd_domains = [
    {'x': [0.02, 0.46], 'y': [0.53, 0.97]},
    {'x': [0.54, 0.98], 'y': [0.53, 0.97]},
    {'x': [0.02, 0.46], 'y': [0.03, 0.47]},
    {'x': [0.54, 0.98], 'y': [0.03, 0.47]},
]
pd_polar_names = ['polar', 'polar2', 'polar3', 'polar4']

for g_idx, (t_key, t_label) in enumerate(TRANSFORM_ORDERED):
    polar_key = pd_polar_names[g_idx]

    for model in MODELS_FOR_ANALYSIS:
        values = []
        for mkey in pd_radar_keys:
            v = pydream_metrics.get(model, {}).get(t_key, {}).get(mkey, np.nan)
            if np.isfinite(v):
                values.append(_pd_normalise(mkey, v))
            else:
                values.append(0)

        values_closed = values + [values[0]]
        labels_closed = pd_radar_labels + [pd_radar_labels[0]]

        fig_pd_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=labels_closed, fill='toself',
            name=model, line=dict(color=MODEL_COLORS[model], width=2),
            opacity=0.5, legendgroup=model, showlegend=(g_idx == 0),
            subplot=polar_key,
        ))

for g_idx, (t_key, t_label) in enumerate(TRANSFORM_ORDERED):
    polar_key = pd_polar_names[g_idx]
    dom = pd_domains[g_idx]
    polar_config = dict(
        domain=dom,
        radialaxis=dict(visible=True, range=[0, 1],
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['0.2', '0.4', '0.6', '0.8', '1.0'],
                        tickfont=dict(size=8, color='#999'),
                        gridcolor='rgba(200,200,200,0.3)'),
        angularaxis=dict(tickfont=dict(size=9), rotation=90, direction='clockwise',
                         gridcolor='rgba(200,200,200,0.3)'),
        bgcolor='rgba(245,245,250,0.3)',
    )
    fig_pd_radar.update_layout(**{polar_key: polar_config})

    cx = (dom['x'][0] + dom['x'][1]) / 2
    ty = dom['y'][1] + 0.025
    fig_pd_radar.add_annotation(text=f'<b>{t_label}</b>', x=cx, y=ty,
                                xref='paper', yref='paper', showarrow=False,
                                font=dict(size=13, color='#333'), xanchor='center')

fig_pd_radar.update_layout(
    title="<b>Which model has the strongest overall diagnostic profile under each transform? — PyDREAM</b><br>"
          "<sup>23 canonical diagnostics, all min-max normalised to 0–1 (1 = best across all models)<br>"
          "Skill metrics: higher = better · Error metrics (*): reversed so lower → higher score · "
          "Bias & signature metrics (*): reversed so closer to 0 → higher score</sup>",
    height=1500, width=1600, margin=dict(t=200, b=60, l=120, r=120),
    legend=dict(orientation='h', yanchor='bottom', y=-0.02, xanchor='center', x=0.5, font=dict(size=11)),
)
fig_pd_radar.show()

# %% [markdown]
# ### 17.8 Comprehensive Performance Summary (PyDREAM)
#
# Mean diagnostic scores across all 4 transforms — the PyDREAM equivalent
# of the summary heatmap in Section 10.8.

# %%
pd_summary_labels = [lbl for _, lbl, _, _ in SUMMARY_DIAG]
pd_n_sd = len(SUMMARY_DIAG)

pd_z_summary = np.full((len(MODELS_FOR_ANALYSIS), pd_n_sd), np.nan)
pd_ann_summary = []

for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
    row_text = []
    for d_idx, (dkey, dlbl, higher_better, closer_zero) in enumerate(SUMMARY_DIAG):
        vals = []
        for t_key in TRANSFORM_KEYS:
            v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(dkey, np.nan)
            if np.isfinite(v):
                vals.append(abs(v) if closer_zero else v)
        if vals:
            mean_val = np.mean(vals)
            pd_z_summary[m_idx, d_idx] = mean_val
            if dkey in ['RMSE', 'MAE']: row_text.append(f'{mean_val:.1f}')
            elif dkey == 'SDEB': row_text.append(f'{mean_val:.2f}')
            elif closer_zero: row_text.append(f'{mean_val:.1f}%')
            else: row_text.append(f'{mean_val:.3f}')
        else:
            row_text.append('')
    pd_ann_summary.append(row_text)

pd_z_rank = np.full_like(pd_z_summary, np.nan)
for d_idx, (_, _, higher_better, _) in enumerate(SUMMARY_DIAG):
    col_vals = pd_z_summary[:, d_idx]
    valid = np.isfinite(col_vals)
    if valid.any():
        best = np.nanmax(col_vals) if higher_better else np.nanmin(col_vals)
        worst = np.nanmin(col_vals) if higher_better else np.nanmax(col_vals)
        rng = abs(best - worst) if abs(best - worst) > 1e-12 else 1.0
        for m_idx in range(len(MODELS_FOR_ANALYSIS)):
            if np.isfinite(col_vals[m_idx]):
                if higher_better:
                    pd_z_rank[m_idx, d_idx] = (best - col_vals[m_idx]) / rng
                else:
                    pd_z_rank[m_idx, d_idx] = (col_vals[m_idx] - best) / rng

fig_pd_summary = go.Figure()
fig_pd_summary.add_trace(go.Heatmap(
    z=pd_z_rank, x=pd_summary_labels, y=MODELS_FOR_ANALYSIS,
    text=pd_ann_summary, texttemplate='%{text}', textfont=dict(size=13, color='black'),
    colorscale=[[0.0, '#1a9641'], [0.25, '#a6d96a'], [0.5, '#ffffbf'], [0.75, '#fdae61'], [1.0, '#d7191c']],
    zmin=0, zmax=1, showscale=True,
    colorbar=dict(title='Relative<br>performance', tickvals=[0, 0.5, 1],
                  ticktext=['Best', 'Middle', 'Worst'], len=0.7, y=0.5),
))

for dg_start, dg_label in SUMMARY_DIAG_GROUPS:
    if dg_start > 0:
        fig_pd_summary.add_vline(x=dg_start - 0.5, line_color='white', line_width=3)
    next_dg = [s for s, _ in SUMMARY_DIAG_GROUPS if s > dg_start]
    dg_end = next_dg[0] if next_dg else pd_n_sd
    mid_x = (dg_start + dg_end - 1) / 2
    fig_pd_summary.add_annotation(
        text=f'<b>{dg_label}</b>', x=pd_summary_labels[int(round(mid_x))], y=1.12,
        xref='x', yref='paper', showarrow=False, font=dict(size=12, color='#444'),
    )

fig_pd_summary.update_layout(
    title="<b>What is the mean diagnostic score for each model across 4 flow transforms? — PyDREAM</b><br>"
          "<sup>Colour = relative rank per column: green = best, red = worst · Cell values = actual mean</sup>",
    height=450, width=2200, margin=dict(t=140, b=120, l=120, r=100),
    xaxis=dict(side='bottom', tickangle=-45, tickfont=dict(size=11)),
    yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
)
fig_pd_summary.show()

# %% [markdown]
# ### 17.9 Final Rankings and Recommendations (PyDREAM)
#
# Data-driven ranking and win-count analysis for the PyDREAM results, mirroring
# Step 11's methodology. This answers: **does the model ranking change when
# using Bayesian MAP estimates instead of SCE-UA point estimates?**

# %%
PD_RANKING_DIAG = {
    'High-flow': [('NSE', True), ('KGE', True), ("KGE_np", True), ('Pearson_r', True), ('Spearman_r', True)],
    'Balanced (sqrt)': [('NSE_sqrt', True), ('KGE_sqrt', True), ('SDEB', False)],
    'Low-flow (log)': [('NSE_log', True), ('KGE_log', True)],
    'Very-low-flow (1/Q)': [('NSE_inv', True), ('KGE_inv', True)],
}

_PD_CLOSER_ZERO = {'PBIAS', 'FHV', 'FMV', 'FLV', 'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}

pd_group_scores = {}
for grp_name, diag_list in PD_RANKING_DIAG.items():
    model_scores = {}
    for model_name in MODELS_FOR_ANALYSIS:
        vals = []
        for dkey, higher_better in diag_list:
            for t_key in TRANSFORM_KEYS:
                v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    if dkey in _PD_CLOSER_ZERO: vals.append(-abs(v))
                    elif not higher_better: vals.append(-v)
                    else: vals.append(v)
        model_scores[model_name] = np.mean(vals) if vals else np.nan
    pd_group_scores[grp_name] = model_scores

pd_grp_names = list(PD_RANKING_DIAG.keys())
pd_z_norm = np.full((len(pd_grp_names), len(MODELS_FOR_ANALYSIS)), np.nan)
pd_ann_rank = []

for g_idx, grp_name in enumerate(pd_grp_names):
    scores = pd_group_scores[grp_name]
    vals_arr = np.array([scores[m] for m in MODELS_FOR_ANALYSIS])
    vmin, vmax = np.nanmin(vals_arr), np.nanmax(vals_arr)
    rng = vmax - vmin if abs(vmax - vmin) > 1e-12 else 1.0
    ranked = sorted(MODELS_FOR_ANALYSIS, key=lambda m: scores[m], reverse=True)
    rank_map = {m: r + 1 for r, m in enumerate(ranked)}
    row_text = []
    for m_idx, model_name in enumerate(MODELS_FOR_ANALYSIS):
        pd_z_norm[g_idx, m_idx] = (scores[model_name] - vmin) / rng
        row_text.append(f'#{rank_map[model_name]}')
    pd_ann_rank.append(row_text)

pd_win_counts = {m: 0 for m in MODELS_FOR_ANALYSIS}
pd_total_cells = 0
for dkey, dlbl in DIAG_SKILL:
    for t_key in TRANSFORM_KEYS:
        cell_vals = {}
        for model_name in MODELS_FOR_ANALYSIS:
            v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(dkey, np.nan)
            if np.isfinite(v):
                cell_vals[model_name] = -v if dkey in DIAG_MINIMIZE else v
        if cell_vals:
            winner = max(cell_vals, key=cell_vals.get)
            pd_win_counts[winner] += 1
            pd_total_cells += 1

pd_win_by_group = {g: {m: 0 for m in MODELS_FOR_ANALYSIS} for g in pd_grp_names}
for grp_name, diag_list in PD_RANKING_DIAG.items():
    for dkey, higher_better in diag_list:
        for t_key in TRANSFORM_KEYS:
            cell_vals = {}
            for model_name in MODELS_FOR_ANALYSIS:
                v = pydream_metrics.get(model_name, {}).get(t_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    if dkey == 'PBIAS': cell_vals[model_name] = -abs(v)
                    elif not higher_better: cell_vals[model_name] = -v
                    else: cell_vals[model_name] = v
            if cell_vals:
                winner = max(cell_vals, key=cell_vals.get)
                pd_win_by_group[grp_name][winner] += 1

fig_pd_rank = make_subplots(
    rows=1, cols=2,
    subplot_titles=['<b>Which model ranks best? (PyDREAM)</b>',
                    '<b>How often does each model win? (PyDREAM)</b>'],
    horizontal_spacing=0.15, column_widths=[0.55, 0.45],
    specs=[[{'type': 'heatmap'}, {'type': 'bar'}]],
)

fig_pd_rank.add_trace(
    go.Heatmap(
        z=pd_z_norm, x=MODELS_FOR_ANALYSIS, y=pd_grp_names,
        text=pd_ann_rank, texttemplate='%{text}', textfont=dict(size=16, color='black'),
        colorscale=[[0.0, '#d7191c'], [0.25, '#fdae61'], [0.5, '#ffffbf'],
                     [0.75, '#a6d96a'], [1.0, '#1a9641']],
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(title='Score', tickvals=[0, 0.5, 1],
                      ticktext=['Worst', 'Middle', 'Best'], len=0.8, y=0.5, x=0.48),
    ),
    row=1, col=1,
)

pd_grp_colors = {
    'High-flow': '#1f77b4',
    'Balanced (sqrt)': '#2ca02c',
    'Low-flow (log)': '#ff7f0e',
    'Very-low-flow (1/Q)': '#d62728',
}
pd_sorted_models = sorted(MODELS_FOR_ANALYSIS, key=lambda m: pd_win_counts[m])

for grp_name in pd_grp_names:
    fig_pd_rank.add_trace(
        go.Bar(name=grp_name, y=pd_sorted_models,
               x=[pd_win_by_group[grp_name][m] for m in pd_sorted_models],
               orientation='h', marker_color=pd_grp_colors[grp_name], showlegend=True),
        row=1, col=2,
    )

for m in pd_sorted_models:
    total_w = pd_win_counts[m]
    fig_pd_rank.add_annotation(
        text=f'<b>{total_w}/{pd_total_cells} ({100*total_w/pd_total_cells:.0f}%)</b>',
        x=total_w + 1, y=m, xref='x2', yref='y2', showarrow=False,
        font=dict(size=10), xanchor='left',
    )

fig_pd_rank.update_xaxes(title_text='Wins', row=1, col=2)
fig_pd_rank.update_layout(
    barmode='stack',
    title="<b>Which model should you choose? — PyDREAM calibrations</b><br>"
          "<sup>Left: normalised ranking per emphasis group; Right: win counts across diagnostic × transform cells</sup>",
    height=500, width=1500, margin=dict(t=140, b=60, l=120, r=140),
    legend=dict(orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.72,
                font=dict(size=10), title_text='Emphasis group'),
)
fig_pd_rank.show()

# %%
print()
print("=" * 100)
print("DATA-DRIVEN MODEL RECOMMENDATIONS (PyDREAM)")
print("=" * 100)
print()
print(f"{'Flow emphasis':<30} {'Best model':<14} {'Score':>8}  Interpretation")
print("-" * 100)

for grp_name, model_scores in pd_group_scores.items():
    best_model = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]
    runner_up = sorted(model_scores, key=model_scores.get, reverse=True)[1]
    ru_score = model_scores[runner_up]
    margin = best_score - ru_score

    interp_map = {
        'High-flow': 'Best for flood forecasting and peak flow estimation',
        'Balanced (sqrt)': 'Best for general-purpose water supply modelling',
        'Low-flow (log)': 'Best for baseflow and dry-season flow estimation',
        'Very-low-flow (1/Q)': 'Best for drought analysis and environmental flow',
    }
    interp = interp_map.get(grp_name, '')
    print(f"  {grp_name:<28} {best_model:<14} {best_score:>+8.4f}  {interp}")
    print(f"  {'':28} (runner-up: {runner_up}, margin: {margin:+.4f})")
    print()

pd_overall_winner = max(pd_win_counts, key=pd_win_counts.get)
print(f"  OVERALL WINNER (most wins):  {pd_overall_winner} "
      f"with {pd_win_counts[pd_overall_winner]}/{pd_total_cells} wins "
      f"({100*pd_win_counts[pd_overall_winner]/pd_total_cells:.0f}%)")

# %% [markdown]
# ### 17.10 Report Cards by Transform (PyDREAM)
#
# Interactive 4-panel report cards for each flow transform — the same format
# as Step 12 (linear hydrograph, log hydrograph, scatter, FDC) but using
# PyDREAM MAP parameter sets. Scroll through all 4 to compare model behaviour
# visually under Bayesian calibration.

# %%
for t_key, t_label in TRANSFORM_ORDERED:
    if not all(t_key in pydream_simulations.get(m, {}) for m in MODELS_FOR_ANALYSIS):
        continue

    obs = pd_obs_flow
    dates = pd_comparison_dates
    series = {'Observed': obs}
    for model_name in MODELS_FOR_ANALYSIS:
        series[model_name] = pydream_simulations[model_name][t_key]

    fig_rc = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Linear hydrograph', 'Log hydrograph',
                        'Observed vs simulated (log)', 'Flow duration curve'],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    for name, y in series.items():
        fig_rc.add_trace(
            go.Scatter(x=dates, y=y, name=name,
                       line=dict(color=MODEL_COLORS.get(name, '#333'),
                                 width=1.5 if name == 'Observed' else 1),
                       legendgroup=name),
            row=1, col=1,
        )

    for name, y in series.items():
        y_safe = np.where(np.asarray(y) > 0.01, np.asarray(y), np.nan)
        fig_rc.add_trace(
            go.Scatter(x=dates, y=y_safe, name=name,
                       line=dict(color=MODEL_COLORS.get(name, '#333'),
                                 width=1.5 if name == 'Observed' else 1),
                       legendgroup=name, showlegend=False),
            row=1, col=2,
        )

    obs_arr = np.asarray(obs)
    valid = ~(np.isnan(obs_arr) | (obs_arr <= 0))
    obs_valid = obs_arr[valid]
    for model_name in MODELS_FOR_ANALYSIS:
        sim_arr = np.asarray(series[model_name])
        sim_valid = np.where(sim_arr[valid] > 0.01, sim_arr[valid], np.nan)
        fig_rc.add_trace(
            go.Scatter(x=obs_valid, y=sim_valid, name=model_name, mode='markers',
                       marker=dict(size=2, color=MODEL_COLORS[model_name], opacity=0.6),
                       legendgroup=model_name, showlegend=False),
            row=2, col=1,
        )
    o_min, o_max = np.nanmin(obs_valid), np.nanmax(obs_valid)
    fig_rc.add_trace(
        go.Scatter(x=[o_min, o_max], y=[o_min, o_max], name='1:1',
                   line=dict(color='gray', dash='dash', width=1)),
        row=2, col=1,
    )

    for name, y in series.items():
        y_arr = np.asarray(y)
        y_clean = y_arr[~(np.isnan(y_arr) | (y_arr <= 0))]
        if len(y_clean) == 0:
            continue
        sorted_flow = np.sort(y_clean)[::-1]
        exc = np.arange(1, len(sorted_flow) + 1) / (len(sorted_flow) + 1) * 100
        fig_rc.add_trace(
            go.Scatter(x=exc, y=sorted_flow, name=name,
                       line=dict(color=MODEL_COLORS.get(name, '#333'),
                                 width=2 if name == 'Observed' else 1.5),
                       legendgroup=name, showlegend=False),
            row=2, col=2,
        )

    fig_rc.update_xaxes(title_text='Date', row=1, col=1)
    fig_rc.update_yaxes(title_text='Flow (ML/day)', row=1, col=1)
    fig_rc.update_xaxes(title_text='Date', row=1, col=2)
    fig_rc.update_yaxes(title_text='Flow (ML/day)', type='log', row=1, col=2)
    fig_rc.update_xaxes(title_text='Observed (ML/day)', type='log', row=2, col=1)
    fig_rc.update_yaxes(title_text='Simulated (ML/day)', type='log', row=2, col=1)
    fig_rc.update_xaxes(title_text='Exceedance (%)', row=2, col=2)
    fig_rc.update_yaxes(title_text='Flow (ML/day)', type='log', row=2, col=2)

    fig_rc.update_layout(
        title=f"<b>How do the four models compare with {t_label}? — PyDREAM</b><br>"
              "<sup>Sacramento vs GR4J vs GR5J vs GR6J</sup>",
        height=700, width=1000,
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1.0),
    )
    fig_rc.show()

# %% [markdown]
# ---
# ## Step 18: Export PyDREAM Results and Final Summary
#
# We save the PyDREAM calibration summary and comprehensive metrics to CSV,
# then print a final recap of everything this notebook has covered.

# %%
# Export PyDREAM calibration summary
pd_export = []
for model_name in MODELS_FOR_ANALYSIS:
    for t_key in TRANSFORM_KEYS:
        if t_key in pydream_results.get(model_name, {}):
            result = pydream_results[model_name][t_key]
            pd_export.append({
                'model': model_name,
                'n_params': len(result.best_parameters),
                'transform': t_key,
                'best_loglik': result.best_objective,
                'runtime_s': result.runtime_seconds,
            })

pd_export_df = pd.DataFrame(pd_export)
pd_export_df.to_csv(OUTPUT_DIR / 'model_comparison_pydream_4transforms.csv', index=False)
print(f"PyDREAM summary saved to: {OUTPUT_DIR / 'model_comparison_pydream_4transforms.csv'}")

# Export PyDREAM comprehensive metrics
pd_metrics_export = []
for model_name in MODELS_FOR_ANALYSIS:
    for t_key in TRANSFORM_KEYS:
        if t_key in pydream_metrics.get(model_name, {}):
            row = {'model': model_name, 'transform': t_key}
            row.update(pydream_metrics[model_name][t_key])
            pd_metrics_export.append(row)

pd_metrics_df = pd.DataFrame(pd_metrics_export)
pd_metrics_df.to_csv(OUTPUT_DIR / 'model_comparison_pydream_metrics.csv', index=False)
print(f"PyDREAM metrics saved to: {OUTPUT_DIR / 'model_comparison_pydream_metrics.csv'}")

# %% [markdown]
# ---
# ## Step 19: SCE-UA vs PyDREAM — Algorithm Comparison per GR Model
#
# The previous sections compared models within a single algorithm. Here we flip
# the question: **for each GR model, does the calibration algorithm matter?**
#
# We map the 4 PyDREAM transforms to the 13 SCE-UA objectives using the
# same flow-emphasis groups used elsewhere in this notebook:
#
# | DREAM transform | SCE-UA objectives | Emphasis |
# |---|---|---|
# | none | NSE, KGE, KGE′ | High-flow (Q) |
# | sqrt | √NSE, KGE-√Q, KGE′-√Q, SDEB | Balanced (√Q) |
# | log  | log-NSE, KGE-log, KGE′-log | Low-flow (log Q) |
# | inverse | 1/Q NSE, KGE-1/Q, KGE′-1/Q | Very-low-flow (1/Q) |
#
# For each GR model and emphasis group we compute the mean of each canonical
# diagnostic across the SCE-UA objectives in that group, and the single PyDREAM
# value for the matching transform. The resulting heatmap shows whether the
# Bayesian approach (PyDREAM) systematically improves or degrades performance
# relative to the point-estimate optimiser (SCE-UA).

# %%
# SCE-UA vs PyDREAM comparison — per GR model, grouped by flow emphasis

DREAM_TO_SCEUA = {
    'none':    [k for k, _ in OBJ_ORDERED[:3]],
    'sqrt':    [k for k, _ in OBJ_ORDERED[3:7]],
    'log':     [k for k, _ in OBJ_ORDERED[7:10]],
    'inverse': [k for k, _ in OBJ_ORDERED[10:]],
}
EMPHASIS_ORDER = [
    ('none',    'High-flow (Q)'),
    ('sqrt',    'Balanced (√Q)'),
    ('log',     'Low-flow (log Q)'),
    ('inverse', 'Very-low-flow (1/Q)'),
]

COMP_DIAG = list(DIAG_SKILL_COLS) + list(DIAG_ERR_COLS)
comp_keys   = [k for k, _ in COMP_DIAG]
comp_labels = [lbl for _, lbl in COMP_DIAG]
n_comp = len(COMP_DIAG)

EMPHASIS_SHORT = {
    'High-flow (Q)': 'High-flow',
    'Balanced (√Q)': 'Balanced',
    'Low-flow (log Q)': 'Low-flow',
    'Very-low-flow (1/Q)': 'V-low-flow',
}
fig_alg = make_subplots(
    rows=3, cols=4,
    subplot_titles=[
        f'<b>{m}</b> — {EMPHASIS_SHORT[elbl]}'
        for m in MODELS_GR_ONLY
        for _, elbl in EMPHASIS_ORDER
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.08,
)

for m_idx, gr_model in enumerate(MODELS_GR_ONLY):
    r = m_idx + 1
    for e_idx, (t_key, e_label) in enumerate(EMPHASIS_ORDER):
        c = e_idx + 1
        sceua_keys = DREAM_TO_SCEUA[t_key]

        z_diff = np.full(n_comp, np.nan)
        sceua_vals_arr = np.full(n_comp, np.nan)
        dream_vals_arr = np.full(n_comp, np.nan)

        for d_idx, (dkey, _) in enumerate(COMP_DIAG):
            sceua_v = []
            for obj_key in sceua_keys:
                v = all_metrics.get(gr_model, {}).get(obj_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    sceua_v.append(v)
            sceua_mean = np.mean(sceua_v) if sceua_v else np.nan
            dream_v = pydream_metrics.get(gr_model, {}).get(t_key, {}).get(dkey, np.nan)

            sceua_vals_arr[d_idx] = sceua_mean
            dream_vals_arr[d_idx] = dream_v

            if np.isfinite(sceua_mean) and np.isfinite(dream_v):
                if dkey in RADAR_HIGHER_BETTER:
                    z_diff[d_idx] = dream_v - sceua_mean
                else:
                    z_diff[d_idx] = abs(sceua_mean) - abs(dream_v)

        z_clamped = np.vectorize(lambda x: soft_clamp(x, k=KNEE_SKILL))(
            np.where(np.isfinite(z_diff), z_diff, 0)
        )
        z_clamped = np.where(np.isfinite(z_diff), z_clamped, np.nan)

        ann_text = []
        for i in range(n_comp):
            if np.isfinite(z_diff[i]):
                ann_text.append(f'{z_diff[i]:+.3f}')
            else:
                ann_text.append('')

        # Vertical heatmap: diagnostics on y-axis (readable), single column "DREAM − SCE-UA" on x
        z_vert = np.reshape(z_clamped, (-1, 1))
        text_vert = [[t] for t in ann_text]
        fig_alg.add_trace(
            go.Heatmap(
                z=z_vert,
                x=['Δ'],
                y=comp_labels,
                text=text_vert,
                texttemplate='%{text}',
                textfont=dict(size=9),
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=(m_idx == 2 and e_idx == 3),
                colorbar=dict(
                    title=dict(text='DREAM advantage', side='right'),
                    tickvals=cb_skill_ticks,
                    ticktext=cb_skill_labels,
                    len=0.6,
                    y=0.5,
                ) if (m_idx == 2 and e_idx == 3) else None,
            ),
            row=r,
            col=c,
        )

# Y-axis: show diagnostic labels only on left column
for r in range(1, 4):
    for c in range(1, 5):
        fig_alg.update_yaxes(
            showticklabels=(c == 1),
            tickfont=dict(size=9),
            tickangle=0,
            row=r,
            col=c,
        )
# X-axis: hide everywhere (titles convey the info)
for r in range(1, 4):
    for c in range(1, 5):
        fig_alg.update_xaxes(showticklabels=False, row=r, col=c)

fig_alg.update_layout(
    title="<b>SCE-UA vs PyDREAM: does the algorithm change the diagnostic profile?</b><br>"
          "<sup>Blue = PyDREAM outperforms SCE-UA · Red = SCE-UA outperforms PyDREAM · "
          "White ≈ no difference<br>"
          "Skill metrics: DREAM − SCE-UA (higher = better) · "
          "Error/bias metrics: |SCE-UA| − |DREAM| (closer to 0 = better)</sup>",
    height=1300,
    width=1200,
    margin=dict(l=160, t=140, r=140, b=60),
)
fig_alg.show()

# %% [markdown]
# The heatmap above shows one row per GR model and four columns corresponding
# to the four flow-emphasis groups. Within each panel every canonical diagnostic
# is compared: **blue** cells indicate that PyDREAM yielded a better value for
# that diagnostic than the mean of the corresponding SCE-UA objectives, while
# **red** cells show the opposite.
#
# Because PyDREAM explores the full posterior while SCE-UA finds a point optimum,
# we generally expect SCE-UA to be slightly better on the metric the objective
# directly targets. The interesting question is whether the Bayesian posterior
# mode provides a better *overall* diagnostic profile — i.e. does it trade a
# small loss on the calibration objective for gains on independent diagnostics?

# %%
# Summary: count how many of the 23 diagnostics DREAM wins, per model × emphasis
# Build rows (D, S, T per emphasis, then totals)
win_rows = []
for gr_model in MODELS_GR_ONLY:
    total_dream, total_sceua, total_tie = 0, 0, 0
    row = {"Model": gr_model}
    for t_key, e_label in EMPHASIS_ORDER:
        sceua_keys = DREAM_TO_SCEUA[t_key]
        d_wins, s_wins, ties = 0, 0, 0
        for dkey, _ in COMP_DIAG:
            sceua_v = []
            for obj_key in sceua_keys:
                v = all_metrics.get(gr_model, {}).get(obj_key, {}).get(dkey, np.nan)
                if np.isfinite(v):
                    sceua_v.append(v)
            sceua_mean = np.mean(sceua_v) if sceua_v else np.nan
            dream_v = pydream_metrics.get(gr_model, {}).get(t_key, {}).get(dkey, np.nan)
            if np.isfinite(sceua_mean) and np.isfinite(dream_v):
                if dkey in RADAR_HIGHER_BETTER:
                    diff = dream_v - sceua_mean
                else:
                    diff = abs(sceua_mean) - abs(dream_v)
                if diff > 1e-6:
                    d_wins += 1
                elif diff < -1e-6:
                    s_wins += 1
                else:
                    ties += 1
        row[f"{e_label} (D)"] = d_wins
        row[f"{e_label} (S)"] = s_wins
        row[f"{e_label} (T)"] = ties
        total_dream += d_wins
        total_sceua += s_wins
        total_tie += ties
    row["Total (D)"] = total_dream
    row["Total (S)"] = total_sceua
    row["Total (T)"] = total_tie
    win_rows.append(row)

win_counts_df = pd.DataFrame(win_rows)
emphasis_labels = [elbl for _, elbl in EMPHASIS_ORDER]

# Net wins matrix: D − S (positive = PyDREAM wins more); rows = models, cols = emphasis
z_net = np.array([
    [row[f"{elbl} (D)"] - row[f"{elbl} (S)"] for elbl in emphasis_labels]
    for row in win_rows
])
text_cells = [
    [f"{row[f'{elbl} (D)']} / {row[f'{elbl} (S)']}" for elbl in emphasis_labels]
    for row in win_rows
]

fig_wins = go.Figure(data=go.Heatmap(
    z=z_net,
    x=emphasis_labels,
    y=MODELS_GR_ONLY,
    text=text_cells,
    texttemplate="%{text}",
    textfont=dict(size=14),
    colorscale="RdBu",
    zmid=0,
    zmin=-23,
    zmax=23,
    showscale=True,
    colorbar=dict(
        title=dict(text="Net wins<br>D − S", side="right"),
        tickvals=[-20, -10, 0, 10, 20],
        len=0.6,
        y=0.5,
    ),
))
fig_wins.update_layout(
    title="<b>SCE-UA vs PyDREAM: who wins more of the 23 diagnostics?</b><br>"
          "<sup>Cell = D / S (PyDREAM wins / SCE-UA wins). "
          "Blue = PyDREAM wins more; red = SCE-UA wins more.</sup>",
    xaxis_title="Flow emphasis",
    yaxis_title="Model",
    height=380,
    width=700,
    margin=dict(t=100, l=80, r=80, b=80),
)
fig_wins.show()

# One-line takeaway
sceua_wins_cells = [
    (row["Model"], elbl)
    for row in win_rows
    for _, elbl in EMPHASIS_ORDER
    if row[f"{elbl} (S)"] > row[f"{elbl} (D)"]
]
if sceua_wins_cells:
    takeaway = "PyDREAM wins more diagnostics in most cells; SCE-UA wins more only for: " + ", ".join(
        f"{m} {e}" for m, e in sceua_wins_cells
    ) + "."
else:
    takeaway = "PyDREAM wins more diagnostics than SCE-UA in every model × emphasis cell."
print("\n" + takeaway)

# Compact totals table (model, total D, total S, winner)
totals_df = win_counts_df[["Model", "Total (D)", "Total (S)"]].copy()
totals_df["Winner"] = totals_df.apply(
    lambda r: "PyDREAM" if r["Total (D)"] > r["Total (S)"] else "SCE-UA",
    axis=1,
)
totals_df = totals_df.rename(columns={"Total (D)": "PyDREAM wins", "Total (S)": "SCE-UA wins"})
try:
    display(totals_df)  # noqa: F821
except NameError:
    print(totals_df.to_string(index=False))

# %% [markdown]
# ---
# ## Summary
#
# ### What We Did
#
# | Part | Algorithm | Models | Objectives / Transforms | Total Calibrations |
# |------|-----------|--------|------------------------|--------------------|
# | 1 | SCE-UA | GR4J, GR5J, GR6J, Sacramento | 13 objective functions | 52 (39 GR + 13 Sacramento) |
# | 2 | PyDREAM | GR4J, GR5J, GR6J, Sacramento | 4 flow transforms | 16 (12 GR + 4 Sacramento) |
# | **Total** | | | | **68 calibrations** |
#
# ### Key Takeaways
#
# 1. **No single model dominates all flow regimes.** The best model depends on
#    the application — high-flow forecasting vs. low-flow estimation vs. general
#    purpose water balance modelling.
#
# 2. **Model complexity is not always rewarded.** Sacramento's 22 parameters give
#    it flexibility, but GR6J (6 parameters) is often competitive, especially for
#    low flows thanks to its exponential store.
#
# 3. **Objective function choice matters as much as model choice.** Switching from
#    NSE to NSE_log can change model performance more than switching from GR4J to
#    Sacramento.
#
# 4. **SCE-UA and PyDREAM give consistent rankings.** The Bayesian approach largely
#    confirms the SCE-UA model ranking, adding uncertainty information.
#
# 5. **GR4J is a strong baseline.** With only 4 parameters, it performs
#    remarkably well for high-flow metrics — a valuable benchmark before
#    adopting more complex models.
#
# ### Next Steps
#
# - **Notebook 08**: Calibration Monitor — real-time visualisation of convergence
# - **Notebook 09**: Calibration Reports — automated report generation
# - Use the exported CSVs for further analysis or publication figures
# - Consider running validation on an independent period to test generalisability

# %%
print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE!")
print("=" * 70)
print(f"\nThis notebook compared:")
print(f"  Part 1 — SCE-UA:")
print(f"    • 4 models: GR4J, GR5J, GR6J, Sacramento")
print(f"    • 13 objective functions each")
print(f"    • 52 calibrations (39 GR + 13 Sacramento)")
print(f"  Part 2 — PyDREAM:")
print(f"    • 4 models: GR4J, GR5J, GR6J, Sacramento")
print(f"    • 4 flow transforms (none, sqrt, log, inverse)")
print(f"    • 16 calibrations (12 GR + 4 Sacramento from NB06)")
print(f"  Total: 68 calibrations")
print(f"\nAll results saved to {OUTPUT_DIR}")
print("""
Key findings:
  ✓ No single model dominates all flow regimes
  ✓ GR6J's exponential store improves low-flow simulation
  ✓ Objective function choice matters as much as model choice
  ✓ SCE-UA and PyDREAM give consistent model rankings
  ✓ GR4J is a strong 4-parameter baseline for high flows
""")
