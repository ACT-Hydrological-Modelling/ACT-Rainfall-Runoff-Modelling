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
#     display_name: pyrrm (Python 3.11.14)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # LBG Headwater DREAM Batch Calibrations (Untransformed + Sqrt)
#
# ## Purpose
#
# This notebook runs DREAM (MT-DREAM(ZS)) Bayesian MCMC calibrations for all
# headwater gauges in the Lower Burley Griffin (LBG) catchment network using
# **two Gaussian likelihoods**:
#
# | Likelihood | Flow emphasis |
# |------------|---------------|
# | `GaussianLikelihood` | High flows (untransformed) |
# | `TransformedGaussianLikelihood('sqrt')` | Balanced mid-range flows |
#
# Each run is stored in a **timestamped folder** under
# `results/batch_runs/dream_batch_YYYY-MM-DD_HHMMSS/` so that repeated
# executions never overwrite previous calibration data. Set `RESUME_FROM` to
# the label of a previous run to resume it instead of starting a new one.
#
# ## Steps
#
# | Step | Topic |
# |------|-------|
# | 1 | Setup and imports |
# | 2 | Data discovery |
# | 3 | Data cleaning and preparation |
# | 4 | Prep diagnostics |
# | 5 | Run configuration (DREAM, 2 likelihoods × 2 models) |
# | 6 | Batch execution (timestamped output folder) |
# | 7 | Results aggregation |
# | 8 | Diagnostic clustermap |

# %% [markdown]
# ---
# ## Step 1: Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
import struct
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

from pyrrm.models import GR4J, Sacramento, NUMBA_AVAILABLE
from pyrrm.calibration.batch import (
    ExperimentList,
    ExperimentSpec,
    BatchExperimentRunner,
    BatchResult,
)
from pyrrm.calibration.objective_functions import (
    GaussianLikelihood,
    TransformedGaussianLikelihood,
)

logger = logging.getLogger(__name__)

print("=" * 70)
print("LBG HEADWATER DREAM BATCH CALIBRATIONS (UNTRANSFORMED + SQRT)")
print("=" * 70)
print(f"\nNumba JIT acceleration: {'ACTIVE' if NUMBA_AVAILABLE else 'not available'}")

# %% [markdown]
# ---
# ### Path configuration

# %%
def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find a directory containing 'data' and 'pyproject.toml'."""
    current = start.resolve()
    for _ in range(10):
        if (current / "data").is_dir() and (current / "pyproject.toml").is_file():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return start


try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = _find_project_root(Path.cwd())

DATA_DIR = PROJECT_ROOT / 'data'
SILO_DIR = Path('/Users/jcastilla/Desktop/ACTGOV/pywr-ACT/data/SILO/timeseries')
CATCHMENT_SHP_DBF = Path('/Users/jcastilla/Desktop/ACTGOV/pywr-ACT/data/ACT_catchments/LBG_subcatchments.dbf')
HEADWATER_GAUGES = ['410705', '410734', '410772', '410774', '410790']

# Calibration window — applied when loading observed flow data
START_DATE: Optional[str] = '1995-01-01'
END_DATE: Optional[str] = None

print(f"Project root     : {PROJECT_ROOT}")
print(f"Data directory   : {DATA_DIR}")
print(f"SILO timeseries  : {SILO_DIR}")
print(f"Catchment areas  : {CATCHMENT_SHP_DBF}")
print(f"Headwater gauges : {HEADWATER_GAUGES}")

# %% [markdown]
# ---
# ### Timestamped run folder
#
# Each execution creates a new `dream_batch_YYYY-MM-DD_HHMMSS` folder so
# existing results are never overwritten.  To **resume** a previous run, set
# `RESUME_FROM` to its label (e.g. `"dream_batch_2026-05-19_110234"`); the
# `BatchExperimentRunner` will detect completed experiments and skip them.

# %%
# ─────────────────────────────────────────────────────────────────
#  RESUME CONTROL
#  Set to None  → start a fresh timestamped run (recommended)
#  Set to label → resume that exact run  (e.g. after a crash)
# ─────────────────────────────────────────────────────────────────
RESUME_FROM: Optional[str] = None  # e.g. "dream_batch_2026-05-19_110234"

_BATCH_RUNS_ROOT = PROJECT_ROOT / 'notebooks_ACT' / 'LBG' / 'results' / 'batch_runs'

if RESUME_FROM is not None:
    RUN_LABEL = RESUME_FROM
    print(f"Resuming existing run : {RUN_LABEL}")
else:
    RUN_LABEL = f"dream_batch_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    print(f"New run               : {RUN_LABEL}")

RUN_OUTPUT_DIR = _BATCH_RUNS_ROOT / RUN_LABEL
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory      : {RUN_OUTPUT_DIR}")

# %% [markdown]
# ---
# ## Step 2: Data Discovery

# %%
SENTINEL_VALUES = [-9999, -9999.0, -99.99]


def _find_flow_file(gauge_dir: Path) -> Optional[Path]:
    """Locate observed flow CSV in a gauge folder.

    Priority (first match wins):
      1. ``*_observed_flow.csv``
      2. ``*_output_SDmodel.csv`` (not 'copy')
      3. ``*_recorded_Flow.csv``
    """
    observed_flow_file: Optional[Path] = None
    sdmodel_flow_file: Optional[Path] = None
    recorded_flow_file: Optional[Path] = None

    for f in sorted(gauge_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.csv':
            continue
        name_lower = f.name.lower()
        if 'observed_flow' in name_lower:
            observed_flow_file = f
        elif 'output_sdmodel' in name_lower and 'copy' not in name_lower:
            sdmodel_flow_file = f
        elif 'recorded_flow' in name_lower:
            recorded_flow_file = f

    return observed_flow_file or sdmodel_flow_file or recorded_flow_file


def discover_gauge_files(
    gauge_ids: List[str],
    silo_dir: Path,
    data_dir: Path,
) -> List[Dict[str, Any]]:
    """Build gauge inventory from SILO CSVs and existing flow files."""
    gauges = []
    for gauge_id in gauge_ids:
        silo_file = silo_dir / f'subcatch_{gauge_id}.0.csv'
        gauge_dir = data_dir / gauge_id
        flow_file = _find_flow_file(gauge_dir) if gauge_dir.is_dir() else None
        gauges.append({
            'gauge_id': gauge_id,
            'silo_file': silo_file,
            'flow_file': flow_file,
        })
    return gauges


gauge_inventory = discover_gauge_files(HEADWATER_GAUGES, SILO_DIR, DATA_DIR)

print(f"Found {len(gauge_inventory)} gauges:\n")
for g in gauge_inventory:
    silo_ok = g['silo_file'].exists()
    flow = g['flow_file']
    print(f"  {g['gauge_id']}:")
    print(f"    SILO climate  : {g['silo_file'].name}  ({'OK' if silo_ok else 'MISSING'})")
    print(f"    Observed flow : {flow.name if flow else 'NOT FOUND'}")

# %% [markdown]
# ---
# ## Step 3: Data Cleaning and Preparation

# %%
def load_catchment_areas_from_dbf(dbf_path: Path) -> Dict[str, float]:
    """Read catchment areas from LBG_subcatchments.dbf.

    Returns ``{gauge_id: area_km2}``.
    """
    areas: Dict[str, float] = {}
    with open(dbf_path, 'rb') as f:
        header = f.read(32)
        nrec = struct.unpack('<I', header[4:8])[0]
        hlen = struct.unpack('<H', header[8:10])[0]
        rlen = struct.unpack('<H', header[10:12])[0]

        fields = []
        while True:
            field_data = f.read(32)
            if field_data[0] == 0x0D:
                break
            name = field_data[:11].split(b'\x00')[0].decode('ascii')
            flen = field_data[16]
            fields.append((name, flen))

        f.seek(hlen)
        for _ in range(nrec):
            record = f.read(rlen)
            pos = 1
            row: Dict[str, str] = {}
            for name, flen in fields:
                val = record[pos:pos + flen].decode('ascii', errors='replace').strip()
                row[name] = val
                pos += flen
            fid = row.get('fid', '').split('.')[0]
            try:
                area = float(row.get('area_km2', '0'))
            except ValueError:
                area = 0.0
            if fid:
                areas[fid] = area
    return areas


CATCHMENT_AREAS_ALL = load_catchment_areas_from_dbf(CATCHMENT_SHP_DBF)
print(f"Loaded {len(CATCHMENT_AREAS_ALL)} catchment areas from shapefile DBF")
for gid, area in sorted(CATCHMENT_AREAS_ALL.items()):
    marker = " <-- headwater" if gid in HEADWATER_GAUGES else ""
    print(f"  {gid}: {area:>6.0f} km2{marker}")


def load_silo_climate(filepath: Path) -> pd.DataFrame:
    """Load a SILO area-weighted daily CSV and return precipitation + PET."""
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    df.index.name = None
    return df[['precip', 'pet_mwet']].rename(columns={
        'precip': 'precipitation',
        'pet_mwet': 'pet',
    })


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Find and parse the date column, set as DatetimeIndex."""
    date_candidates = ['Date', 'date', 'datetime', 'Datetime', 'time', 'timestamp']
    dcol = None
    for c in date_candidates:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        dcol = df.columns[0]

    raw = df[dcol].astype(str).str.strip()
    df[dcol] = pd.to_datetime(raw, dayfirst=True, format='mixed', errors='coerce')
    if df[dcol].isna().all():
        raise ValueError(f"Date column '{dcol}' could not be parsed.")
    df = df.dropna(subset=[dcol]).set_index(dcol)
    df.index.name = None
    return df


def _load_observed_flow(filepath: Path) -> pd.Series:
    """Load observed flow CSV, auto-detect value column, clean sentinels."""
    df = pd.read_csv(filepath)
    df = _parse_dates(df)

    flow_aliases = [
        'Recorded Gauging Station Flow',
        'observed_flow', 'flow', 'Flow', 'discharge', 'Q', 'streamflow',
        'Flow (ML/d)', 'Flow(ML/d)', 'Flow (Ml/d)', 'Flow (ML.day^-1)',
    ]
    fcol = None
    for alias in flow_aliases:
        if alias in df.columns:
            fcol = alias
            break
    if fcol is None:
        for alias in flow_aliases:
            for c in df.columns:
                if alias.lower() in c.lower():
                    fcol = c
                    break
            if fcol:
                break
    if fcol is None:
        numeric = df.select_dtypes(include='number').columns
        if len(numeric) == 0:
            raise ValueError(f"No numeric column found in {filepath.name}")
        fcol = numeric[0]

    flow = df[fcol].copy()
    for sv in SENTINEL_VALUES:
        flow = flow.replace(sv, np.nan)
    flow[flow < 0] = np.nan
    flow.name = 'observed_flow'
    return flow


def prepare_gauge_data(
    gauge_info: Dict[str, Any],
    catchment_areas: Dict[str, float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Full data preparation pipeline for a single gauge."""
    gauge_id = gauge_info['gauge_id']

    if not gauge_info['silo_file'].exists():
        raise FileNotFoundError(
            f"Gauge {gauge_id}: SILO file not found: {gauge_info['silo_file']}"
        )
    inputs_df = load_silo_climate(gauge_info['silo_file'])

    if gauge_info['flow_file'] is None:
        raise FileNotFoundError(f"Gauge {gauge_id}: no observed flow file found")
    flow = _load_observed_flow(gauge_info['flow_file'])

    merged = inputs_df.join(flow, how='inner').dropna(subset=['observed_flow'])

    if start_date is not None:
        merged = merged.loc[start_date:]
    if end_date is not None:
        merged = merged.loc[:end_date]

    if len(merged) == 0:
        raise ValueError(f"Gauge {gauge_id}: no data remaining after merge and filtering")

    inputs_out = merged[['precipitation', 'pet']]
    observed_out = merged['observed_flow'].values
    area_km2 = catchment_areas.get(gauge_id, 0.0)

    meta = {
        'gauge_id': gauge_id,
        'data_source': 'SILO',
        'silo_file': str(gauge_info['silo_file'].name),
        'start_date': str(merged.index[0].date()),
        'end_date': str(merged.index[-1].date()),
        'n_days': len(merged),
        'precip_na_count': int(inputs_out['precipitation'].isna().sum()),
        'pet_na_count': int(inputs_out['pet'].isna().sum()),
        'mean_precip': float(inputs_out['precipitation'].mean()),
        'mean_pet': float(inputs_out['pet'].mean()),
        'mean_flow': float(observed_out.mean()),
        'area_km2': area_km2,
    }
    return inputs_out, observed_out, meta

# %% [markdown]
# ### Load all gauges

# %%
gauge_data: Dict[str, Tuple[pd.DataFrame, np.ndarray, Dict]] = {}
prep_errors: Dict[str, str] = {}

for g in gauge_inventory:
    gauge_id = g['gauge_id']
    try:
        inputs_df, observed, meta = prepare_gauge_data(
            g, CATCHMENT_AREAS_ALL, start_date=START_DATE, end_date=END_DATE
        )
        gauge_data[gauge_id] = (inputs_df, observed, meta)
        area_str = f"{meta['area_km2']:.0f} km2" if meta['area_km2'] > 0 else "UNKNOWN"
        print(f"  {gauge_id}: {meta['n_days']} days  "
              f"({meta['start_date']} to {meta['end_date']})  "
              f"Area={area_str}")
    except Exception as e:
        prep_errors[gauge_id] = str(e)
        print(f"  {gauge_id}: FAILED - {e}")

print(f"\nSuccessfully loaded: {len(gauge_data)}/{len(gauge_inventory)} gauges")

# %% [markdown]
# ---
# ## Step 4: Prep Diagnostics

# %%
diag_rows = []
for gauge_id, (_, _, meta) in gauge_data.items():
    diag_rows.append({
        'gauge_id': meta['gauge_id'],
        'start_date': meta['start_date'],
        'end_date': meta['end_date'],
        'n_days': meta['n_days'],
        'data_source': meta['data_source'],
        'silo_file': meta['silo_file'],
        'area_km2': round(meta['area_km2'], 0),
        'precip_na': meta['precip_na_count'],
        'pet_na': meta['pet_na_count'],
        'mean_precip_mm': round(meta['mean_precip'], 2),
        'mean_pet_mm': round(meta['mean_pet'], 2),
        'mean_flow_ML': round(meta['mean_flow'], 2),
        'status': 'OK',
    })
for gauge_id, err in prep_errors.items():
    diag_rows.append({'gauge_id': gauge_id, 'status': f'FAILED: {err}'})

diag_df = pd.DataFrame(diag_rows)
diag_df

# %% [markdown]
# ### Quick visual check: precipitation and flow for each gauge

# %%
from pyrrm.visualization import plot_precip_flow_grid_plotly

gauge_plot_data = {
    gauge_id: {
        'dates': inputs_df.index,
        'precipitation': inputs_df['precipitation'].values,
        'observed_flow': observed,
        'title': f'{gauge_id} — P & Observed Flow'
                 f'  (area={meta["area_km2"]:.1f} km², '
                 f'{meta["n_days"]:,d} days)',
    }
    for gauge_id, (inputs_df, observed, meta) in gauge_data.items()
}

fig_overview = plot_precip_flow_grid_plotly(
    gauge_plot_data,
    per_row_height=350,
    flow_units='ML/d',
)
fig_overview.show()

# %% [markdown]
# ---
# ## Step 5: Run Configuration
#
# **DREAM only — 2 likelihoods × 2 models = 4 experiments per gauge (20 total).**
#
# | Likelihood | Short name | Flow emphasis |
# |------------|------------|---------------|
# | `GaussianLikelihood` | `likelihood` | High flows (untransformed) |
# | `TransformedGaussianLikelihood('sqrt')` | `likelihood_sqrt` | Balanced mid-range |
#
# The Gaussian log-likelihood (Vrugt, 2016) is:
# `log_lik = −n/2 · log(Σ (T(obs) − T(sim))²)` where `T()` is the
# flow transformation.

# %%
# =====================================================================
# CONFIGURATION — adjust these before running
# =====================================================================

MODELS: List[str] = ['Sacramento']

WARMUP_DAYS: int = 365                  # warmup period (days)

BACKEND: str = 'sequential'             # 'sequential' | 'multiprocessing' | 'ray'
MAX_WORKERS: Optional[int] = None       # None = auto

CATCHMENT_AREAS: Dict[str, float] = {
    gid: CATCHMENT_AREAS_ALL[gid]
    for gid in gauge_data
    if gid in CATCHMENT_AREAS_ALL and CATCHMENT_AREAS_ALL[gid] > 0
}

EXPORT_FORMAT: str = 'both'             # 'excel' | 'csv' | 'both'

# -- DREAM algorithm settings ------------------------------------------
# Previous run (10k iter / 3 chains / mt=3) showed max GR 1.13–1.56 across
# all gauges (18/22 params failing for untransformed, 9/22 for sqrt).
# Root cause: 22-parameter Sacramento needs far more sampling. Persistently
# struggling params: lzfsm/lzsk/lzpk (correlated lower zone), zperc/rexp
# (percolation), adimp/pctim/side (impervious cluster), uh1–uh5.
# Fix: 3× more iterations, 5 chains for better 22-D exploration, multitry=5
# to improve proposals for the correlated lower-zone cluster.
# Est. runtime: ~17 min/experiment × 10 experiments ≈ 3 hours total.
DREAM_N_ITERATIONS: int = 20_000        # was 10_000 — need ≥ 2× for 22-D Sacramento
DREAM_N_CHAINS: int = 5                 # was 3 — more chains = better 22-D exploration
DREAM_MULTITRY: int = 5                 # was 3 — helps correlated lzfsm/lzsk/lzpk cluster
DREAM_SNOOKER: float = 0.2              # keep — effective for correlated posteriors

print(f"Configuration:")
print(f"  Models           : {MODELS}")
print(f"  Warmup           : {WARMUP_DAYS} days")
print(f"  Backend          : {BACKEND}")
print(f"  DREAM iterations : {DREAM_N_ITERATIONS}")
print(f"  DREAM chains     : {DREAM_N_CHAINS}")
print()
print(f"Catchment areas (from LBG_subcatchments shapefile):")
for gid, area in CATCHMENT_AREAS.items():
    print(f"  {gid}: {area:>10.0f} km2")

# %% [markdown]
# ### Build ExperimentList per gauge
#
# Two `ExperimentSpec` objects per model:
# 1. `likelihood` — `GaussianLikelihood()` (untransformed, high-flow emphasis)
# 2. `likelihood_sqrt` — `TransformedGaussianLikelihood('sqrt')` (balanced)

# %%
import copy

_MODEL_CLASSES: Dict[str, Any] = {
    'Sacramento': Sacramento,
    'GR4J': GR4J,
}

# (short_key_suffix, transformation_name, likelihood_object)
_DREAM_LIKELIHOODS: List[Tuple[str, str, Any]] = [
    ('likelihood',      'none', GaussianLikelihood()),
    ('likelihood_sqrt', 'sqrt', TransformedGaussianLikelihood('sqrt')),
]


def build_gauge_experiments(gauge_id: str) -> ExperimentList:
    """Build DREAM ExperimentList for a single gauge.

    Creates 2 DREAM experiments per model (untransformed + sqrt):
      - {gauge}_{model}_likelihood_dream
      - {gauge}_{model}_likelihood_sqrt_dream
    """
    area_km2 = CATCHMENT_AREAS.get(gauge_id)
    model_params: Dict[str, Any] = {}
    if area_km2 and area_km2 > 0:
        model_params['catchment_area_km2'] = area_km2

    specs: List[ExperimentSpec] = []
    for model_name in MODELS:
        m = model_name.lower()
        model_cls = _MODEL_CLASSES[model_name]
        for key_suffix, t_name, likelihood_obj in _DREAM_LIKELIHOODS:
            exp_key = f"{gauge_id}_{m}_{key_suffix}_dream"
            # dbname writes a per-evaluation CSV to logs/<key>_progress.csv,
            # enabling live objective monitoring during sampling.
            progress_csv = str(
                RUN_OUTPUT_DIR / gauge_id / 'logs' / f'{exp_key}_progress'
            )
            dream_alg_kwargs = {
                'method': 'dream',
                'n_iterations': DREAM_N_ITERATIONS,
                'n_chains': DREAM_N_CHAINS,
                'multitry': DREAM_MULTITRY,
                'snooker': DREAM_SNOOKER,
                'dbname': progress_csv,
            }
            specs.append(ExperimentSpec(
                key=exp_key,
                model_name=model_name,
                model=model_cls(**model_params),
                objective_name=key_suffix,
                objective=copy.deepcopy(likelihood_obj),
                algorithm_name='dream',
                algorithm_kwargs=dream_alg_kwargs,
                transformation_name=t_name if t_name != 'none' else None,
                transformation=None,
            ))

    return ExperimentList(specs)


gauge_experiments: Dict[str, ExperimentList] = {}

for gauge_id in gauge_data:
    exp_list = build_gauge_experiments(gauge_id)
    gauge_experiments[gauge_id] = exp_list
    print(f"  {gauge_id}: {len(exp_list)} DREAM experiments")

print(f"\nTotal experiments across all gauges: "
      f"{sum(len(el) for el in gauge_experiments.values())}")

# %% [markdown]
# ### Experiment inventory

# %%
for gauge_id, exp_list in gauge_experiments.items():
    print(f"\n{gauge_id}:")
    for spec in exp_list.combinations():
        t = spec.transformation_name or 'none'
        print(f"  {spec.key:<55s}  model={spec.model_name:<12s}  "
              f"obj={spec.objective_name:<20s}  transform={t}")

# %% [markdown]
# ---
# ## Step 6: Batch Execution
#
# Results are stored under:
# ```
# results/batch_runs/{RUN_LABEL}/{gauge_id}/
# ```
#
# Each gauge run is fully isolated — a failure in one gauge does not halt the
# loop. Set `RESUME_FROM` (Step 1) to the run label to skip already-completed
# experiments on re-execution.

# %%
batch_results: Dict[str, BatchResult] = {}
execution_errors: Dict[str, str] = {}

for gauge_id, exp_list in gauge_experiments.items():
    inputs_df, observed, meta = gauge_data[gauge_id]

    catchment_info: Dict[str, Any] = {'gauge_id': gauge_id}
    if gauge_id in CATCHMENT_AREAS:
        catchment_info['area_km2'] = CATCHMENT_AREAS[gauge_id]

    # Pre-create a fixed gauge folder so the runner uses it directly via
    # resume_from, bypassing its internal timestamp/hash naming.
    gauge_run_dir = RUN_OUTPUT_DIR / gauge_id
    gauge_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gauge {gauge_id}  ({len(exp_list)} experiments)")
    print(f"  Run label : {RUN_LABEL}")
    print(f"  Run dir   : {gauge_run_dir}")
    print(f"{'='*60}")

    try:
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=exp_list,
            output_dir=str(RUN_OUTPUT_DIR),
            warmup_period=WARMUP_DAYS,
            catchment_info=catchment_info,
            backend=BACKEND,
            max_workers=MAX_WORKERS,
            progress_bar=True,
            log_level='INFO',
        )

        result = runner.run(resume_from=str(gauge_run_dir))
        batch_results[gauge_id] = result

        print(f"\n  Completed: {len(result.results)} | "
              f"Failed: {len(result.failures)} | "
              f"Runtime: {result.runtime_seconds:.1f}s")

        if result.failures:
            for fkey, ferr in result.failures.items():
                print(f"    FAILED: {fkey} -- {ferr}")

    except Exception as e:
        execution_errors[gauge_id] = str(e)
        print(f"\n  GAUGE-LEVEL FAILURE: {e}")

print(f"\n{'='*60}")
print(f"BATCH EXECUTION SUMMARY")
print(f"  Run label        : {RUN_LABEL}")
print(f"  Output directory : {RUN_OUTPUT_DIR}")
print(f"{'='*60}")
print(f"  Gauges completed : {len(batch_results)}")
print(f"  Gauges failed    : {len(execution_errors)}")
if execution_errors:
    for gid, err in execution_errors.items():
        print(f"    {gid}: {err}")

# %% [markdown]
# ---
# ## Step 7: Results Aggregation

# %%
all_summary_dfs = []

for gauge_id, result in batch_results.items():
    df = result.to_dataframe()
    df.insert(0, 'gauge_id', gauge_id)
    all_summary_dfs.append(df)

if all_summary_dfs:
    combined_df = pd.concat(all_summary_dfs, ignore_index=True)
    display_cols = [
        'gauge_id', 'key', 'model', 'objective',
        'best_objective', 'runtime_seconds', 'success',
    ]
    available = [c for c in display_cols if c in combined_df.columns]
    print("Combined results:\n")
    print(combined_df[available].sort_values(
        ['gauge_id', 'best_objective'], ascending=[True, False]
    ).to_string(index=False))
else:
    combined_df = pd.DataFrame()
    print("No results to display.")

# %% [markdown]
# ### Best experiment per gauge

# %%
if not combined_df.empty:
    print("Best experiment per gauge / objective:\n")
    for gauge_id, result in batch_results.items():
        best = result.best_by_objective()
        for obj_name, (key, val) in best.items():
            print(f"  {gauge_id}  {obj_name:>20s}:  {key:<50s}  ({val:.4f})")

# %% [markdown]
# ### Run folder inventory

# %%
for gauge_id, result in batch_results.items():
    run_dir = Path(result.run_dir)
    print(f"\n{gauge_id}: {RUN_OUTPUT_DIR / gauge_id}")
    for root, dirs, files in os.walk(run_dir):
        level = len(Path(root).relative_to(run_dir).parts)
        indent = '  ' * level
        print(f"  {indent}{Path(root).name}/")
        for fname in sorted(files):
            fsize = (Path(root) / fname).stat().st_size
            size_str = f"{fsize / 1024:.1f} KB" if fsize > 1024 else f"{fsize} B"
            print(f"  {indent}  {fname:<40s}  ({size_str})")

# %% [markdown]
# ### Export results

# %%
if batch_results:
    export_dir = RUN_OUTPUT_DIR / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    for gauge_id, result in batch_results.items():
        gauge_export_dir = export_dir / gauge_id
        try:
            files = result.export(str(gauge_export_dir), format=EXPORT_FORMAT)
            n_files = sum(len(v) for v in files.values())
            print(f"  {gauge_id}: exported {n_files} files to {gauge_export_dir}")
        except Exception as e:
            print(f"  {gauge_id}: export failed -- {e}")

# %% [markdown]
# ---
# ## Live Progress Monitor
#
# While Step 6 is running, re-execute this cell to see the current best
# log-likelihood per chain and total evaluations so far.
# Each experiment writes a `logs/<key>_progress.csv` with columns:
# `like1, par0..parN, chain, simulation`.

# %%
def show_dream_progress(run_output_dir: Path) -> None:
    """Print a live summary of DREAM progress from progress CSVs."""
    csvs = sorted(run_output_dir.rglob('*_progress.csv'))
    if not csvs:
        print("No progress CSVs yet — experiment may still be in the first iteration block.")
        return
    for csv_path in csvs:
        exp_key = csv_path.stem.replace('_progress', '')
        try:
            df = pd.read_csv(csv_path, header=0)
            if df.empty:
                print(f"  {exp_key}: no rows yet")
                continue
            like_col = df.columns[0]
            n_evals = len(df)
            best = df[like_col].max()
            recent_best = df[like_col].tail(500).max()
            chains = df['chain'].nunique() if 'chain' in df.columns else '?'
            print(f"  {exp_key}")
            print(f"    Evaluations : {n_evals:,d}  |  Chains: {chains}")
            print(f"    Best so far : {best:.4f}")
            print(f"    Last 500    : {recent_best:.4f}  "
                  f"({'improving' if recent_best > best - 0.5 * abs(best) * 0.01 else 'plateauing'})")
        except Exception as e:
            print(f"  {exp_key}: error reading CSV — {e}")
    print()


show_dream_progress(RUN_OUTPUT_DIR)

# %% [markdown]
# ---
# ## Step 8: Diagnostic Clustermap
#
# Canonical diagnostic suite (23 metrics) computed for every experiment:
#
# | Group | Metrics | Ideal |
# |-------|---------|-------|
# | Skill | NSE × 4, KGE × 4, KGE_np × 4 | 1 |
# | Error | RMSE, MAE, SDEB | 0 |
# | Volume bias | PBIAS, FHV, FMV, FLV | 0 |
# | Signature errors | Sig_BFI, Sig_Flash, Sig_Q95, Sig_Q5 | 0 |
#
# Metrics are normalised to a "higher = better" [0, 1] scale for clustering.

# %%
try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

from pyrrm.analysis.diagnostics import compute_diagnostics
from pyrrm.objectives import SDEB as _SDEB_cls

_sdeb_func = _SDEB_cls(alpha=0.1, lam=0.5)

HEADLINE_METRICS = [
    'NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
    'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
    'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv',
    'RMSE', 'MAE', 'SDEB',
    'PBIAS', 'FHV', 'FMV', 'FLV',
    'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5',
]

NEGATE_METRICS     = {'RMSE', 'MAE', 'SDEB'}
ABS_NEGATE_METRICS = {'PBIAS', 'FHV', 'FMV', 'FLV',
                      'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}


def _build_diagnostics_df(result: BatchResult, gauge_id: str) -> pd.DataFrame:
    """Build a DataFrame of canonical diagnostics for every experiment."""
    rows = {}
    for key, report in result.results.items():
        metrics = dict(compute_diagnostics(report.simulated, report.observed))
        metrics['SDEB'] = float(_sdeb_func(report.observed, report.simulated))
        short_key = key.replace(f'{gauge_id}_', '', 1)
        rows[short_key] = {m: metrics.get(m, np.nan) for m in HEADLINE_METRICS}
    return pd.DataFrame.from_dict(rows, orient='index')


def _normalise_higher_is_better(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to 'higher = better' [0, 1] for each column."""
    df_norm = df.copy()
    df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df_norm.columns:
        if col in NEGATE_METRICS:
            df_norm[col] = -df_norm[col]
        elif col in ABS_NEGATE_METRICS:
            df_norm[col] = -df_norm[col].abs()

    for col in df_norm.columns:
        finite_min = df_norm[col].min()
        if pd.isna(finite_min):
            finite_min = 0.0
        df_norm[col] = df_norm[col].fillna(finite_min)

    col_min = df_norm.min()
    col_max = df_norm.max()
    span = (col_max - col_min).replace(0.0, 1.0)
    df_norm = (df_norm - col_min) / span
    df_norm = df_norm.fillna(0.0)
    return df_norm


if not batch_results:
    print("No batch results available — run Step 6 first.")
elif not _SNS_AVAILABLE:
    print("seaborn is not installed — install with:  pip install seaborn")
else:
    for gauge_id, result in batch_results.items():
        if not result.results:
            print(f"  {gauge_id}: no successful experiments, skipping.")
            continue

        df_raw  = _build_diagnostics_df(result, gauge_id)
        df_norm = _normalise_higher_is_better(df_raw)

        n_rows = len(df_norm)
        fig_h  = max(6, 0.55 * n_rows)

        g = sns.clustermap(
            df_norm,
            method='ward',
            metric='euclidean',
            cmap='RdYlGn',
            figsize=(16, fig_h),
            row_cluster=True,
            col_cluster=True,
            linewidths=0.5,
            linecolor='white',
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8},
            dendrogram_ratio=(0.12, 0.08),
            cbar=False,
        )
        if g.ax_cbar is not None:
            g.ax_cbar.set_visible(False)
        g.figure.suptitle(
            f'Gauge {gauge_id} — DREAM Calibration Diagnostic Clustermap\n'
            f'Run: {RUN_LABEL}',
            y=1.02, fontsize=13, fontweight='bold',
        )
        plt.show()

        mean_scores = df_norm.mean(axis=1).sort_values(ascending=False)
        print(f"\n  Gauge {gauge_id} — Top-4 experiments (mean normalised score):")
        for rank, (exp, score) in enumerate(mean_scores.head(4).items(), 1):
            print(f"    {rank}. {exp:<50s}  {score:.3f}")
        print()

# %% [markdown]
# ### Stylized table — raw values with per-category colour scales

# %%
_METRIC_SKILL  = ['NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
                  'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
                  'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv']
_METRIC_ERROR  = ['RMSE', 'MAE', 'SDEB']
_METRIC_VOLUME = ['PBIAS', 'FHV', 'FMV', 'FLV']
_METRIC_SIG    = ['Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5']


def _style_diagnostics_table(df_raw: pd.DataFrame):
    """Style raw diagnostics with per-column colour scales (green = best)."""
    df = df_raw.replace([np.inf, -np.inf], np.nan).copy()
    skill_cols  = [c for c in _METRIC_SKILL  if c in df.columns]
    error_cols  = [c for c in _METRIC_ERROR  if c in df.columns]
    volume_cols = [c for c in _METRIC_VOLUME if c in df.columns]
    sig_cols    = [c for c in _METRIC_SIG    if c in df.columns]

    sty = df.style.format('{:.3g}', na_rep='—')
    if skill_cols:
        sty = sty.background_gradient(subset=skill_cols, cmap='RdYlGn', axis=0)
    for col in error_cols:
        sty = sty.background_gradient(subset=[col], cmap='RdYlGn', gmap=-df[col], axis=0)
    for col in volume_cols:
        sty = sty.background_gradient(subset=[col], cmap='RdYlGn', gmap=-np.abs(df[col]), axis=0)
    for col in sig_cols:
        sty = sty.background_gradient(subset=[col], cmap='RdYlGn', gmap=-np.abs(df[col]), axis=0)
    return sty


if batch_results:
    for gauge_id, result in batch_results.items():
        if not result.results:
            continue
        df_raw = _build_diagnostics_df(result, gauge_id)
        styled = _style_diagnostics_table(df_raw)
        display(styled.set_caption(
            f'Gauge {gauge_id} — Raw diagnostics | Run: {RUN_LABEL}'
        ))

# %% [markdown]
# ---
# ## Resuming a Previous Run
#
# To resume a run that was interrupted or to add new experiments to an
# existing run, set `RESUME_FROM` at the top of this notebook (Step 1):
#
# ```python
# RESUME_FROM = "dream_batch_2026-05-19_110234"
# ```
#
# Then re-execute all cells. The `BatchExperimentRunner` detects completed
# experiments (via saved `.pkl` files in the run folder) and skips them,
# running only the remaining or new experiments.
#
# ### All run folders
#
# ```python
# import os
# from pathlib import Path
# runs_root = PROJECT_ROOT / 'notebooks_ACT' / 'LBG' / 'results' / 'batch_runs'
# for run_dir in sorted(runs_root.iterdir()):
#     pkl_count = sum(1 for _ in run_dir.rglob('*.pkl'))
#     print(f"  {run_dir.name}  ({pkl_count} .pkl files)")
# ```
