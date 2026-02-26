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
#     display_name: pyrrm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LBG Headwater Batch Calibrations
#
# ## Purpose
#
# This notebook runs systematic batch calibration experiments for **all headwater
# gauges** in the `data/` folder. It automates gauge discovery, data preparation
# (including averaging multi-column rainfall/PET inputs), experiment configuration,
# and execution using pyrrm's `ExperimentList` and `BatchExperimentRunner`.
#
# ## What You'll Learn
#
# - How to discover and load gauge data with inconsistent file/column naming
# - How to average multiple `Rain_*` / `Mwet_*` columns into single P and PET series
# - How to compute total catchment area by matching subcatchment IDs from column
#   names to the `LBG_Subcatcment_areas.csv` reference table
# - How to build per-gauge `ExperimentList` configurations with placeholders
# - How to execute batch runs across many gauges with resume support
# - How to aggregate and compare results across the full gauge set
#
# ## Prerequisites
#
# - Familiarity with `CalibrationRunner` (Notebook 02: Calibration Quickstart)
# - Familiarity with `BatchExperimentRunner` (Notebook 11: Batch Runners)
# - Gauge data folders under `data/` with rainfall/PET and observed flow CSVs
#
# ## Estimated Time
#
# - Data prep: ~1 minute
# - Calibration: depends on model complexity, algorithm budget, and number of gauges
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|-------|-------------|
# | 1 | Setup and imports | Libraries, pyrrm APIs, path configuration |
# | 2 | Data discovery | Scan gauge folders, locate rainfall/PET/flow files |
# | 3 | Data cleaning and preparation | Load, average multi-column inputs, compute areas, clean, align |
# | 4 | Prep diagnostics | Summary table of loaded data quality per gauge |
# | 5 | Batch configuration | 13 SCE-UA objectives + 4 DREAM likelihoods x 2 models per gauge |
# | 6 | Batch execution | Per-gauge `BatchExperimentRunner` with resume |
# | 7 | Results aggregation | Cross-gauge summary, best experiments, export |
# | 8 | Resume and reproducibility | Re-run behaviour and YAML config template |
#
# ## Key Insight
#
# > By wrapping `BatchExperimentRunner` in a per-gauge loop with isolated error
# > handling, we get structured, resumable calibration across an entire gauge
# > network without manual intervention. Each gauge's results live in their own
# > timestamped run folder with full logs and config snapshots.

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
import json
import os
import logging

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

from pyrrm.models import GR4J, Sacramento, NUMBA_AVAILABLE
from pyrrm.objectives import NSE, KGE
from pyrrm.calibration.batch import (
    ExperimentList,
    ExperimentSpec,
    BatchExperimentRunner,
    BatchResult,
    make_experiment_key,
)
from pyrrm.calibration.objective_functions import (
    GaussianLikelihood,
    TransformedGaussianLikelihood,
)
from pyrrm.objectives import APEX

logger = logging.getLogger(__name__)

print("=" * 70)
print("LBG HEADWATER BATCH CALIBRATIONS")
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
SUBCATCHMENT_AREAS_FILE = DATA_DIR / 'LBG_other' / 'LBG_Subcatcment_areas.csv'
OUTPUT_ROOT = PROJECT_ROOT / 'notebooks_ACT' / 'LBG' / 'results'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Project root            : {PROJECT_ROOT}")
print(f"Data directory          : {DATA_DIR}")
print(f"Subcatchment areas file : {SUBCATCHMENT_AREAS_FILE}")
print(f"Output root             : {OUTPUT_ROOT}")

# %% [markdown]
# ---
# ## Step 2: Data Discovery
#
# Gauge data lives in numeric subdirectories under `data/`. File naming is
# inconsistent across gauges:
#
# | Pattern | Example |
# |---------|---------|
# | Combined rain + PET | `410772_rain_mwet.csv` (columns: `Rain_*`, `Mwet_*`) |
# | Separate rain and PET | `Default Input Set - Rain_QBN01.csv`, `Default Input Set - Mwet_QBN01.csv` |
# | Observed flow | `410772_observed_flow.csv`, `410734_output_SDmodel.csv`, or `410734_recorded_Flow.csv` |
#
# The folder name does not always match the file prefix (e.g. folder `410745`
# contains `410705_*` files).

# %%
def discover_gauge_files(data_dir: Path) -> List[Dict[str, Any]]:
    """Scan data_dir for numeric gauge folders and locate input/flow files.

    Returns a list of dicts with keys: gauge_id, gauge_dir, rain_files,
    pet_files, flow_file, combined_rain_pet_file.

    Flow file priority (first match wins):
      1. ``*_observed_flow.csv``  – standard naming for most gauges
      2. ``*_output_SDmodel.csv`` – SOURCE model output containing recorded
         gauging station flow (used for 410734)
      3. ``*_recorded_Flow.csv``  – single-column recorded flow (may be a
         sub-catchment only)
    """
    gauges = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue

        gauge_id = d.name
        info: Dict[str, Any] = {
            'gauge_id': gauge_id,
            'gauge_dir': d,
            'rain_files': [],
            'pet_files': [],
            'flow_file': None,
            'combined_rain_pet_file': None,
        }

        observed_flow_file: Optional[Path] = None
        sdmodel_flow_file: Optional[Path] = None
        recorded_flow_file: Optional[Path] = None

        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() != '.csv':
                continue
            name_lower = f.name.lower()

            if 'rain_mwet' in name_lower:
                info['combined_rain_pet_file'] = f
            elif 'rain' in name_lower and 'mwet' not in name_lower:
                info['rain_files'].append(f)
            elif 'mwet' in name_lower and 'rain' not in name_lower:
                info['pet_files'].append(f)

            if 'observed_flow' in name_lower:
                observed_flow_file = f
            elif 'output_sdmodel' in name_lower and 'copy' not in name_lower:
                sdmodel_flow_file = f
            elif 'recorded_flow' in name_lower:
                recorded_flow_file = f

        info['flow_file'] = observed_flow_file or sdmodel_flow_file or recorded_flow_file
        gauges.append(info)

    return gauges


gauge_inventory = discover_gauge_files(DATA_DIR)

print(f"Found {len(gauge_inventory)} gauge folders:\n")
for g in gauge_inventory:
    combined = g['combined_rain_pet_file']
    rain_count = len(g['rain_files'])
    pet_count = len(g['pet_files'])
    flow = g['flow_file']
    print(f"  {g['gauge_id']}:")
    if combined:
        print(f"    Combined rain+PET : {combined.name}")
    if rain_count:
        print(f"    Separate rain     : {rain_count} file(s)")
    if pet_count:
        print(f"    Separate PET      : {pet_count} file(s)")
    print(f"    Observed flow     : {flow.name if flow else 'NOT FOUND'}")

# %% [markdown]
# ---
# ## Step 3: Data Cleaning and Preparation
#
# For each gauge we:
#
# 1. Read the rainfall/PET CSV(s) and parse the date column.
# 2. Identify all `Rain_*` columns and **average** them row-wise to get a single
#    `precipitation` series. Do the same for all `Mwet_*` columns to get `pet`.
# 3. **Compute total catchment area**: extract subcatchment IDs from the `Rain_*`
#    column names (e.g. `Rain_SUL01` -> `SUL01`), look them up in
#    `LBG_Subcatcment_areas.csv`, and sum the areas (hectares -> km2).
# 4. Read the observed flow CSV, auto-detect the value column, replace sentinels
#    (`-9999`) with NaN, drop negative flows.
# 5. Inner-join on the date index so all three series are aligned.
# 6. Return `(inputs_df, observed_array, metadata_dict)`.

# %%
SENTINEL_VALUES = [-9999, -9999.0, -99.99]


def _load_subcatchment_areas(filepath: Path) -> Dict[str, float]:
    """Load LBG_Subcatcment_areas.csv into a dict mapping ID -> area in hectares."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Subcatchment_ID'])
    return dict(zip(df['Subcatchment_ID'].str.strip(), df['Area_ha']))


def _extract_subcatchment_ids(column_names: List[str], prefix: str = 'Rain_') -> List[str]:
    """Extract subcatchment IDs from column names like Rain_SUL01 -> SUL01.

    Also handles the separate-file pattern where the column is e.g.
    'Default Input Set - Rain_QBN01 (mm.day^-1)' -> QBN01.
    """
    ids = []
    for col in column_names:
        if col.startswith(prefix):
            ids.append(col[len(prefix):])
        elif f'- {prefix}' in col:
            after = col.split(f'- {prefix}', 1)[1]
            sub_id = after.split()[0].rstrip('(')
            if sub_id:
                ids.append(sub_id)
    return ids


def compute_catchment_area_km2(
    rain_columns: List[str],
    subcatchment_areas: Dict[str, float],
) -> Tuple[float, List[str], List[str]]:
    """Sum subcatchment areas (ha -> km2) for the IDs found in rain column names.

    Returns (total_area_km2, matched_ids, unmatched_ids).
    """
    sub_ids = _extract_subcatchment_ids(rain_columns, prefix='Rain_')
    matched = []
    unmatched = []
    total_ha = 0.0
    for sid in sub_ids:
        if sid in subcatchment_areas:
            total_ha += subcatchment_areas[sid]
            matched.append(sid)
        else:
            unmatched.append(sid)
    total_km2 = total_ha / 100.0
    return total_km2, matched, unmatched


SUBCATCHMENT_AREAS = _load_subcatchment_areas(SUBCATCHMENT_AREAS_FILE)
print(f"Loaded {len(SUBCATCHMENT_AREAS)} subcatchment area records")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Find and parse the date column, set as DatetimeIndex.

    Handles mixed formats (YYYY-MM-DD, DD/MM/YYYY) and strips whitespace.
    """
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
        raise ValueError(
            f"Date column '{dcol}' could not be parsed. "
            "Check for mixed formats (e.g. YYYY-MM-DD vs DD/MM/YYYY) or invalid values."
        )
    df = df.dropna(subset=[dcol])
    df = df.set_index(dcol)
    df.index.name = None
    return df


def _load_rain_pet_combined(filepath: Path) -> Tuple[pd.Series, pd.Series, Dict]:
    """Load a combined rain_mwet CSV, average Rain_* and Mwet_* columns."""
    df = pd.read_csv(filepath)
    df = _parse_dates(df)

    rain_cols = [c for c in df.columns if c.startswith('Rain_')]
    mwet_cols = [c for c in df.columns if c.startswith('Mwet_')]

    if not rain_cols:
        raise ValueError(f"No Rain_* columns found in {filepath.name}")
    if not mwet_cols:
        raise ValueError(f"No Mwet_* columns found in {filepath.name}")

    precipitation = df[rain_cols].mean(axis=1, skipna=True)
    pet = df[mwet_cols].mean(axis=1, skipna=True)

    meta = {
        'rain_columns': rain_cols,
        'pet_columns': mwet_cols,
        'n_rain_cols': len(rain_cols),
        'n_pet_cols': len(mwet_cols),
        'source': 'combined',
    }
    return precipitation, pet, meta


def _load_rain_pet_separate(
    rain_files: List[Path],
    pet_files: List[Path],
) -> Tuple[pd.Series, pd.Series, Dict]:
    """Load separate rain and PET CSVs, average numeric columns within each."""
    rain_dfs = []
    rain_col_names = []
    for rf in rain_files:
        df = pd.read_csv(rf)
        df = _parse_dates(df)
        numeric = df.select_dtypes(include='number')
        rain_dfs.append(numeric)
        rain_col_names.extend(numeric.columns.tolist())

    pet_dfs = []
    pet_col_names = []
    for pf in pet_files:
        df = pd.read_csv(pf)
        df = _parse_dates(df)
        numeric = df.select_dtypes(include='number')
        pet_dfs.append(numeric)
        pet_col_names.extend(numeric.columns.tolist())

    if not rain_dfs:
        raise ValueError("No separate rain files provided")
    if not pet_dfs:
        raise ValueError("No separate PET files provided")

    rain_combined = pd.concat(rain_dfs, axis=1)
    pet_combined = pd.concat(pet_dfs, axis=1)

    precipitation = rain_combined.mean(axis=1, skipna=True)
    pet = pet_combined.mean(axis=1, skipna=True)

    meta = {
        'rain_columns': rain_col_names,
        'pet_columns': pet_col_names,
        'n_rain_cols': len(rain_col_names),
        'n_pet_cols': len(pet_col_names),
        'source': 'separate',
    }
    return precipitation, pet, meta


def _load_observed_flow(filepath: Path) -> pd.Series:
    """Load observed flow CSV, auto-detect value column, clean sentinels.

    Handles three file layouts:
      - Single-column flow files (``*_observed_flow.csv``, ``*_recorded_Flow.csv``)
      - Multi-column SOURCE output (``*_output_SDmodel.csv``) where the
        ``Recorded Gauging Station Flow`` column is extracted.
    """
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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Full data preparation pipeline for a single gauge.

    Returns (inputs_df, observed_array, metadata).
    inputs_df has columns 'precipitation' and 'pet' with a DatetimeIndex.
    """
    gauge_id = gauge_info['gauge_id']

    if gauge_info['combined_rain_pet_file'] is not None:
        precip, pet, meta = _load_rain_pet_combined(
            gauge_info['combined_rain_pet_file']
        )
    elif gauge_info['rain_files'] and gauge_info['pet_files']:
        precip, pet, meta = _load_rain_pet_separate(
            gauge_info['rain_files'], gauge_info['pet_files']
        )
    else:
        raise FileNotFoundError(
            f"Gauge {gauge_id}: no rainfall/PET data files found"
        )

    if gauge_info['flow_file'] is None:
        raise FileNotFoundError(
            f"Gauge {gauge_id}: no observed flow file found"
        )
    flow = _load_observed_flow(gauge_info['flow_file'])

    inputs_df = pd.DataFrame({
        'precipitation': precip,
        'pet': pet,
    })

    merged = inputs_df.join(flow, how='inner')
    merged = merged.dropna(subset=['observed_flow'])

    if start_date is not None:
        merged = merged.loc[start_date:]
    if end_date is not None:
        merged = merged.loc[:end_date]

    if len(merged) == 0:
        raise ValueError(f"Gauge {gauge_id}: no data remaining after merge and filtering")

    inputs_out = merged[['precipitation', 'pet']]
    observed_out = merged['observed_flow'].values

    area_km2, matched_ids, unmatched_ids = compute_catchment_area_km2(
        meta['rain_columns'], SUBCATCHMENT_AREAS,
    )

    meta.update({
        'gauge_id': gauge_id,
        'start_date': str(merged.index[0].date()),
        'end_date': str(merged.index[-1].date()),
        'n_days': len(merged),
        'precip_na_count': int(inputs_out['precipitation'].isna().sum()),
        'pet_na_count': int(inputs_out['pet'].isna().sum()),
        'mean_precip': float(inputs_out['precipitation'].mean()),
        'mean_pet': float(inputs_out['pet'].mean()),
        'mean_flow': float(observed_out.mean()),
        'area_km2': area_km2,
        'subcatchment_ids': matched_ids,
        'unmatched_subcatchment_ids': unmatched_ids,
    })
    return inputs_out, observed_out, meta

# %% [markdown]
# ### Load all gauges

# %%
gauge_data: Dict[str, Tuple[pd.DataFrame, np.ndarray, Dict]] = {}
prep_errors: Dict[str, str] = {}

for g in gauge_inventory:
    gauge_id = g['gauge_id']
    try:
        inputs_df, observed, meta = prepare_gauge_data(g)
        gauge_data[gauge_id] = (inputs_df, observed, meta)
        area_str = f"{meta['area_km2']:.1f} km2" if meta['area_km2'] > 0 else "UNKNOWN"
        warn = ""
        if meta['unmatched_subcatchment_ids']:
            warn = f"  (unmatched IDs: {meta['unmatched_subcatchment_ids']})"
        print(f"  {gauge_id}: {meta['n_days']} days  "
              f"({meta['start_date']} to {meta['end_date']})  "
              f"Rain cols={meta['n_rain_cols']}  PET cols={meta['n_pet_cols']}  "
              f"Area={area_str}  [{meta['source']}]{warn}")
    except Exception as e:
        prep_errors[gauge_id] = str(e)
        print(f"  {gauge_id}: FAILED - {e}")

print(f"\nSuccessfully loaded: {len(gauge_data)}/{len(gauge_inventory)} gauges")
if prep_errors:
    print(f"Failed: {len(prep_errors)} gauges")

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
        'source': meta['source'],
        'rain_cols_averaged': meta['n_rain_cols'],
        'pet_cols_averaged': meta['n_pet_cols'],
        'area_km2': round(meta['area_km2'], 2),
        'subcatchments': len(meta['subcatchment_ids']),
        'unmatched_ids': len(meta['unmatched_subcatchment_ids']),
        'precip_na': meta['precip_na_count'],
        'pet_na': meta['pet_na_count'],
        'mean_precip_mm': round(meta['mean_precip'], 2),
        'mean_pet_mm': round(meta['mean_pet'], 2),
        'mean_flow_ML': round(meta['mean_flow'], 2),
        'status': 'OK',
    })
for gauge_id, err in prep_errors.items():
    diag_rows.append({
        'gauge_id': gauge_id,
        'status': f'FAILED: {err}',
    })

diag_df = pd.DataFrame(diag_rows)
diag_df

# %% [markdown]
# ### Quick visual check: precipitation and flow for each gauge
#
# Uses `pyrrm.visualization.plot_precip_flow_grid_plotly` which overlays
# inverted precipitation bars and the observed flow hydrograph on a single
# panel per gauge. Hover for exact values.

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
# ## Step 5: Batch Configuration
#
# Two experiment batches per gauge, for both Sacramento and GR4J:
#
# **Batch A -- SCE-UA (14 objective/transform combos per model, 28 total):**
#
# | Objective | Transformations | Count |
# |-----------|-----------------|-------|
# | NSE | none, sqrt, log, inverse | 4 |
# | KGE | none, sqrt, log, inverse | 4 |
# | KGE Non-Parametric | none, sqrt, log, inverse | 4 |
# | SDEB | none (built-in power transform) | 1 |
# | APEX sqrt (κ=0.7, uniform) | sqrt (built-in) | 1 |
#
# **Batch B -- DREAM with Gaussian likelihood (4 per model, 8 total):**
#
# DREAM is a Bayesian MCMC sampler that requires a **likelihood function**,
# not a standard objective like NSE. We use:
#
# | Likelihood | Flow emphasis | Count |
# |------------|---------------|-------|
# | `GaussianLikelihood` | High flows (untransformed) | 1 |
# | `TransformedGaussianLikelihood('sqrt')` | Balanced | 1 |
# | `TransformedGaussianLikelihood('log')` | Low flows | 1 |
# | `TransformedGaussianLikelihood('inverse')` | Very low flows | 1 |
#
# The Gaussian log-likelihood (Vrugt, 2016) is:
# `log_lik = -n/2 * log(sum((T(obs) - T(sim))^2))`
# where `T()` is the flow transformation.
#
# **Total: (14 + 4) x 2 models = 36 experiments per gauge.**

# %%
# =====================================================================
# CONFIGURATION
# =====================================================================

MODELS: List[str] = ['Sacramento', 'GR4J']

TRANSFORMATIONS: List[str] = ['none', 'sqrt', 'log', 'inverse']

WARMUP_DAYS: int = 365              # TODO: adjust warmup period (days)

BACKEND: str = 'sequential'         # Options: 'sequential', 'multiprocessing', 'ray'
MAX_WORKERS: Optional[int] = None   # None = auto for multiprocessing/ray

CATCHMENT_AREAS: Dict[str, float] = {
    gid: meta['area_km2']
    for gid, (_, _, meta) in gauge_data.items()
    if meta['area_km2'] > 0
}

print("Catchment areas (auto-computed from subcatchment IDs):")
for gid, area in CATCHMENT_AREAS.items():
    ids = gauge_data[gid][2]['subcatchment_ids']
    print(f"  {gid}: {area:>10.2f} km2  ({len(ids)} subcatchments: {', '.join(ids)})")

START_DATE: Optional[str] = None    # TODO: set if you want to trim the period, e.g. '2000-01-01'
END_DATE: Optional[str] = None      # TODO: set if you want to trim the period, e.g. '2024-12-31'

EXPORT_FORMAT: str = 'both'         # Options: 'excel', 'csv', 'both'

# -- SCE-UA algorithm settings -------------------------------------
SCEUA_MAX_EVALS: int = 10000        # TODO: adjust eval budget
SCEUA_SEED: int = 42

# -- DREAM algorithm settings ---------------------------------------
DREAM_N_ITERATIONS: int = 10_000
DREAM_N_CHAINS: int = 3             # min for Gelman-Rubin; sufficient up to ~25-d
DREAM_MULTITRY: int = 3             # balances mixing vs cost; avoids multitry=2 bug
DREAM_SNOOKER: float = 0.2          # helps explore correlated posteriors in 22-d Sacramento

print(f"\nConfiguration:")
print(f"  Models          : {MODELS}")
print(f"  Transformations : {TRANSFORMATIONS}")
print(f"  Warmup          : {WARMUP_DAYS} days")
print(f"  Backend         : {BACKEND}")
print(f"  SCE-UA evals    : {SCEUA_MAX_EVALS}")
print(f"  DREAM iterations: {DREAM_N_ITERATIONS}")

# %% [markdown]
# ### Build ExperimentList per gauge
#
# We construct a curated `ExperimentList` for each gauge containing:
#
# - **14 SCE-UA experiments** per model: NSE x4, KGE x4, KGE-NP x4, SDEB x1, APEX-sqrt x1
#   (NSE/KGE/KGENP/SDEB via `from_dicts`; APEX via explicit `ExperimentSpec`)
# - **4 DREAM experiments** per model: Gaussian likelihood x4 transformations
#   (built as explicit `ExperimentSpec` objects because `GaussianLikelihood` /
#   `TransformedGaussianLikelihood` are not in the dict-based registry)
#
# Experiment keys encode gauge, model, objective, algorithm, and transformation.

# %%
# Model class lookup for ExperimentSpec construction
_MODEL_CLASSES: Dict[str, Any] = {
    'Sacramento': Sacramento,
    'GR4J': GR4J,
}

# DREAM likelihood definitions: (short_name, transform_name, objective_instance)
# short_name is always 'likelihood'; the transform goes into t_name so the
# key reads  {gauge}_{model}_likelihood[_{transform}]_dream
_DREAM_LIKELIHOODS: List[Tuple[str, str, Any]] = [
    ('likelihood', 'none',    GaussianLikelihood()),
    ('likelihood', 'sqrt',    TransformedGaussianLikelihood('sqrt')),
    ('likelihood', 'log',     TransformedGaussianLikelihood('log')),
    ('likelihood', 'inverse', TransformedGaussianLikelihood('inverse')),
]


def build_gauge_experiments(gauge_id: str) -> ExperimentList:
    """Build the full ExperimentList for a single gauge.

    Batch A (SCE-UA): 14 objective/transform combos x len(MODELS) models
        (NSE x4, KGE x4, KGENP x4, SDEB x1, APEX-sqrt x1)
    Batch B (DREAM):   4 likelihood/transform combos x len(MODELS) models

    Models are created with catchment_area_km2 so simulated flow is in ML/day
    (matching the observed flow units in the gauge CSVs).
    """
    import copy
    area_km2 = CATCHMENT_AREAS.get(gauge_id)
    model_params: Dict[str, Any] = {}
    if area_km2 and area_km2 > 0:
        model_params['catchment_area_km2'] = area_km2

    # -- Batch A: SCE-UA via from_dicts (13 per model) ----------------------
    sceua_dicts: List[Dict[str, Any]] = []
    sceua_objectives = [
        ('nse', 'NSE'),
        ('kge', 'KGE'),
        ('kgenp', 'KGENP'),
    ]

    sceua_alg_base = {
        'method': 'sceua_direct',
        'max_evals': SCEUA_MAX_EVALS,
        'seed': SCEUA_SEED,
        'max_tolerant_iter': 50,
        'tolerance': 1e-4,
    }

    for model_name in MODELS:
        m = model_name.lower()
        for obj_short, obj_type in sceua_objectives:
            for t in TRANSFORMATIONS:
                if t == 'none':
                    key = f"{gauge_id}_{m}_{obj_short}_sceua"
                else:
                    key = f"{gauge_id}_{m}_{obj_short}_{t}_sceua"
                exp: Dict[str, Any] = {
                    'key': key,
                    'model': model_name,
                    'model_params': model_params,
                    'objective': {'type': obj_type},
                    'algorithm': dict(sceua_alg_base),
                }
                if t != 'none':
                    exp['transformation'] = t
                sceua_dicts.append(exp)

        sceua_dicts.append({
            'key': f"{gauge_id}_{m}_sdeb_sceua",
            'model': model_name,
            'model_params': model_params,
            'objective': {'type': 'SDEB'},
            'algorithm': dict(sceua_alg_base),
        })

    sceua_list = ExperimentList.from_dicts(sceua_dicts, catchment=gauge_id)

    # -- APEX-sqrt (κ=0.7, uniform) via explicit ExperimentSpec ----
    # APEX has its own built-in transform so no external transformation is needed.
    # Config matches notebook 05 Q1: SDEB + dynamics multiplier comparison.
    apex_obj = APEX(
        alpha=0.1,
        transform='sqrt',
        dynamics_strength=0.7,
        regime_emphasis='uniform',
    )
    apex_specs: List[ExperimentSpec] = []
    for model_name in MODELS:
        m = model_name.lower()
        model_cls = _MODEL_CLASSES[model_name]
        apex_specs.append(ExperimentSpec(
            key=f"{gauge_id}_{m}_apex_sqrt_sceua",
            model_name=model_name,
            model=model_cls(**model_params),
            objective_name='apex_sqrt',
            objective=copy.deepcopy(apex_obj),
            algorithm_name='sceua',
            algorithm_kwargs=dict(sceua_alg_base),
        ))

    # -- Batch B: DREAM via explicit ExperimentSpec (4 per model) -----------
    dream_specs: List[ExperimentSpec] = []
    dream_alg_kwargs = {
        'method': 'dream',
        'n_iterations': DREAM_N_ITERATIONS,
        'n_chains': DREAM_N_CHAINS,
        'multitry': DREAM_MULTITRY,
        'snooker': DREAM_SNOOKER,
    }

    for model_name in MODELS:
        m = model_name.lower()
        model_cls = _MODEL_CLASSES[model_name]
        for obj_short, t_name, likelihood_obj in _DREAM_LIKELIHOODS:
            if t_name == 'none':
                key = f"{gauge_id}_{m}_{obj_short}_dream"
            else:
                key = f"{gauge_id}_{m}_{obj_short}_{t_name}_dream"
            dream_specs.append(ExperimentSpec(
                key=key,
                model_name=model_name,
                model=model_cls(**model_params),
                objective_name=obj_short,
                objective=copy.deepcopy(likelihood_obj),
                algorithm_name='dream',
                algorithm_kwargs=dict(dream_alg_kwargs),
                transformation_name=t_name if t_name != 'none' else None,
                transformation=None,
            ))

    # Combine all batches
    all_specs = sceua_list.specs + apex_specs + dream_specs
    return ExperimentList(all_specs)


gauge_experiments: Dict[str, ExperimentList] = {}

for gauge_id in gauge_data:
    exp_list = build_gauge_experiments(gauge_id)
    gauge_experiments[gauge_id] = exp_list

    sceua_count = sum(1 for s in exp_list.specs if s.algorithm_name != 'dream')
    dream_count = sum(1 for s in exp_list.specs if s.algorithm_name == 'dream')
    print(f"  {gauge_id}: {len(exp_list)} experiments  "
          f"(SCE-UA: {sceua_count}, DREAM: {dream_count})")

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
              f"obj={spec.objective_name:<25s}  alg={spec.algorithm_name:<15s}  "
              f"transform={t}")

# %% [markdown]
# ---
# ## Step 6: Batch Execution
#
# For each gauge we instantiate a `BatchExperimentRunner`, run with resume
# support, and collect results. Failures at the gauge level are caught and
# logged without stopping the entire loop.

# %%
batch_results: Dict[str, BatchResult] = {}
execution_errors: Dict[str, str] = {}

for gauge_id, exp_list in gauge_experiments.items():
    inputs_df, observed, meta = gauge_data[gauge_id]

    gauge_output_dir = OUTPUT_ROOT / gauge_id
    catchment_info = {
        'gauge_id': gauge_id,
    }
    if gauge_id in CATCHMENT_AREAS:
        catchment_info['area_km2'] = CATCHMENT_AREAS[gauge_id]

    print(f"\n{'='*60}")
    print(f"  Gauge {gauge_id}  ({len(exp_list)} experiments)")
    print(f"{'='*60}")

    try:
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=exp_list,
            output_dir=str(gauge_output_dir),
            warmup_period=WARMUP_DAYS,
            catchment_info=catchment_info,
            backend=BACKEND,
            max_workers=MAX_WORKERS,
            progress_bar=True,
            log_level='INFO',
        )

        result = runner.run(resume=True, run_name=f'{gauge_id}_headwater')
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
print(f"{'='*60}")
print(f"  Gauges completed : {len(batch_results)}")
print(f"  Gauges failed    : {len(execution_errors)}")
if execution_errors:
    for gid, err in execution_errors.items():
        print(f"    {gid}: {err}")

# %% [markdown]
# ---
# ## Step 7: Results Aggregation
#
# Combine per-gauge `BatchResult` summary DataFrames into a single cross-gauge
# view. Identify the best experiment for each gauge and each objective.

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
# ### Best experiment per gauge and objective

# %%
if not combined_df.empty:
    print("Best experiment per gauge / objective:\n")
    for gauge_id, result in batch_results.items():
        best = result.best_by_objective()
        for obj_name, (key, val) in best.items():
            print(f"  {gauge_id}  {obj_name:>5s}:  {key:<45s}  ({val:.4f})")

# %% [markdown]
# ### Run folder inventory

# %%
for gauge_id, result in batch_results.items():
    run_dir = Path(result.run_dir)
    print(f"\n{gauge_id}: {run_dir.name}")
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
#
# Export calibration reports to Excel and/or CSV for sharing. Toggle
# `EXPORT_FORMAT` in the configuration section above.

# %%
if batch_results:
    export_dir = OUTPUT_ROOT / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    for gauge_id, result in batch_results.items():
        gauge_export_dir = export_dir / gauge_id
        try:
            files = result.export(
                str(gauge_export_dir),
                format=EXPORT_FORMAT,
            )
            n_files = sum(len(v) for v in files.values())
            print(f"  {gauge_id}: exported {n_files} files to {gauge_export_dir}")
        except Exception as e:
            print(f"  {gauge_id}: export failed -- {e}")

# %% [markdown]
# ---
# ## Step 8: Resume and Reproducibility
#
# Re-running the execution cell with `resume=True` (the default) will detect
# previously completed experiments and skip them. Only new or failed experiments
# are re-executed.
#
# ### Reload a previous BatchResult

# %%
# Example: reload results from disk without re-running calibration
#
# reloaded = BatchResult.load(str(OUTPUT_ROOT / '410734' / '<run_folder>' / 'batch_result.pkl'))
# print(reloaded)
# print(reloaded.to_dataframe())

# %% [markdown]
# ### YAML configuration template
#
# For fully externalized, version-controlled experiment definitions, save the
# configuration as a YAML file and load with `BatchExperimentRunner.from_config()`.
#
# ```yaml
# # headwater_config.yaml
# experiments:
#   - key: "410734_sacramento_nse_sceua"
#     model: Sacramento
#     objective: {type: NSE}
#     algorithm: {method: sceua_direct, max_evals: 10000, seed: 42}
#   - key: "410734_gr4j_kge_sceua"
#     model: GR4J
#     objective: {type: KGE}
#     algorithm: {method: sceua_direct, max_evals: 5000, seed: 42}
#
# catchment:
#   name: "Queanbeyan River"
#   gauge_id: "410734"
# warmup_days: 365
# output_dir: ./results/410734
# backend: sequential
# progress_bar: true
# ```
#
# ```python
# runner = BatchExperimentRunner.from_config(
#     'headwater_config.yaml', inputs_df, observed,
# )
# result = runner.run()
# ```

# %% [markdown]
# ---
# ## Step 9: Diagnostic Clustermap — Which Calibration is Best?
#
# For each catchment we compute the **canonical diagnostic suite** on every
# calibration experiment — 23 metrics organised into four groups:
#
# | Group | Metrics | Ideal | Count |
# |-------|---------|-------|-------|
# | **Skill** | NSE (×4 transforms), KGE (×4), KGE_np (×4) | 1 | 12 |
# | **Error** | RMSE, MAE, SDEB | 0 | 3 |
# | **Volume bias** | PBIAS, FHV, FMV, FLV | 0 | 4 |
# | **Signature errors** | Sig_BFI, Sig_Flash, Sig_Q95, Sig_Q5 | 0 | 4 |
#
# Metrics are **normalised** so that every column follows a *"higher is better"*
# convention:
# - **Skill** (NSE, KGE, KGE_np): used as-is (ideal = 1).
# - **Error** (RMSE, MAE, SDEB): negated (ideal → 0 becomes max).
# - **Bias / signatures** (PBIAS, FHV, FMV, FLV, Sig_*): negated absolute
#   value (ideal = 0 becomes max).
#
# Each column is then **min-max scaled to [0, 1]** within the catchment so that
# the colour scale is comparable across metrics.
#
# A **seaborn `clustermap`** with Ward hierarchical clustering groups
# experiments (rows) and metrics (columns) that behave similarly, making it easy
# to spot which calibrations dominate and which metrics cluster together.

# %%
try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

from pyrrm.analysis.diagnostics import compute_diagnostics
from pyrrm.objectives import SDEB as _SDEB_cls

_sdeb_func = _SDEB_cls(alpha=0.1, lam=0.5)

# ── Canonical diagnostic metrics for the clustermap ────────────────────
#
# 12 skill metrics  : NSE × 4 transforms + KGE × 4 + KGE_np × 4
# 3  error metrics  : RMSE, MAE, SDEB
# 4  volume metrics : PBIAS, FHV (high), FMV (mid), FLV (low)
# 4  signature errs : Sig_BFI, Sig_Flash, Sig_Q95, Sig_Q5
# -----------------------------------------------------------------------
HEADLINE_METRICS = [
    # Skill (higher = better, ideal = 1)
    'NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
    'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
    'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv',
    # Error (lower = better, ideal = 0)
    'RMSE', 'MAE', 'SDEB',
    # Volume bias (closer to 0 = better)
    'PBIAS', 'FHV', 'FMV', 'FLV',
    # Signature % errors (closer to 0 = better)
    'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5',
]

NEGATE_METRICS     = {'RMSE', 'MAE', 'SDEB'}
ABS_NEGATE_METRICS = {'PBIAS', 'FHV', 'FMV', 'FLV',
                      'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5'}


def _build_diagnostics_df(
    result: BatchResult,
    gauge_id: str,
) -> pd.DataFrame:
    """Build a DataFrame of canonical diagnostics for every experiment."""
    rows = {}
    for key, report in result.results.items():
        metrics = dict(compute_diagnostics(report.simulated, report.observed))
        metrics['SDEB'] = float(_sdeb_func(report.observed, report.simulated))
        short_key = key.replace(f'{gauge_id}_', '', 1)
        rows[short_key] = {m: metrics.get(m, np.nan) for m in HEADLINE_METRICS}
    return pd.DataFrame.from_dict(rows, orient='index')


def _normalise_higher_is_better(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise so every column follows 'higher = better' in [0, 1].

    Handles NaN, +/-inf, and constant columns gracefully so that
    scipy's linkage never encounters non-finite values.

    Note: A column can be constant (e.g. all 0) when the metric is undefined
    (NaN) for every experiment. For ephemeral/low-flow gauges, Sig_Q95 and FLV
    now use fallbacks (0 or 100) instead of NaN; see compute_diagnostics Notes.
    """
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
    span = col_max - col_min
    span = span.replace(0.0, 1.0)
    df_norm = (df_norm - col_min) / span

    df_norm = df_norm.fillna(0.0)
    return df_norm


# ── Generate one clustermap per catchment ──────────────────────────────
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
        fig_h  = max(10, 0.45 * n_rows)

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
            annot_kws={'size': 7},
            dendrogram_ratio=(0.12, 0.08),
            cbar=False,
        )
        if g.ax_cbar is not None:
            g.ax_cbar.set_visible(False)
        g.figure.suptitle(
            f'Gauge {gauge_id} — Calibration Diagnostic Clustermap',
            y=1.02, fontsize=14, fontweight='bold',
        )
        plt.show()

        # ── Top-3 experiments by mean normalised score ─────────────
        mean_scores = df_norm.mean(axis=1).sort_values(ascending=False)
        print(f"\n  Gauge {gauge_id} — Top-5 experiments (mean normalised score):")
        for rank, (exp, score) in enumerate(mean_scores.head(5).items(), 1):
            print(f"    {rank}. {exp:<45s}  {score:.3f}")
        print()


# %% [markdown]
# ### Stylized table — raw values with per-category color scales
#
# Raw diagnostic values in a table with **per-category** color scaling so that
# each metric type is interpreted correctly: skill (green = 1), error (green =
# low), volume/signature (green = near 0). No single global scale; each
# column or group uses a scale appropriate to that metric.

# %%
# Column groups for per-category styling (same order as HEADLINE_METRICS)
_METRIC_SKILL = [
    'NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
    'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
    'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv',
]
_METRIC_ERROR = ['RMSE', 'MAE', 'SDEB']
_METRIC_VOLUME = ['PBIAS', 'FHV', 'FMV', 'FLV']
_METRIC_SIG = ['Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5']

def _style_diagnostics_table(df_raw: pd.DataFrame):
    """Style raw diagnostics with per-column color scales (green = best in that column)."""
    df = df_raw.replace([np.inf, -np.inf], np.nan).copy()
    skill_cols = [c for c in _METRIC_SKILL if c in df.columns]
    error_cols = [c for c in _METRIC_ERROR if c in df.columns]
    volume_cols = [c for c in _METRIC_VOLUME if c in df.columns]
    sig_cols = [c for c in _METRIC_SIG if c in df.columns]

    sty = df.style.format('{:.3g}', na_rep='—')
    # Skill: best = max (1). Per-column scale from column min to max; high = green.
    if skill_cols:
        sty = sty.background_gradient(subset=skill_cols, cmap='RdYlGn', axis=0)
    # Error: best = lowest. Per-column scale; invert so low = green.
    for col in error_cols:
        sty = sty.background_gradient(subset=[col], cmap='RdYlGn', gmap=-df[col], axis=0)
    # Volume / Signature: best = closest to zero. Per-column scale; |value| small = green.
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
        display(styled.set_caption(f'Gauge {gauge_id} — Raw diagnostics (green = best per column)'))

# %% [markdown]
# ---
# ## Summary
#
# | Feature | Component | Description |
# |---------|-----------|-------------|
# | **Data discovery** | `discover_gauge_files()` | Auto-scan gauge folders with inconsistent naming |
# | **Multi-column averaging** | `_load_rain_pet_combined()` | Average `Rain_*` / `Mwet_*` columns row-wise |
# | **Auto catchment area** | `compute_catchment_area_km2()` | Sum subcatchment areas from `Rain_*` column IDs |
# | **Per-gauge experiments** | `ExperimentList.from_dicts()` | Config-driven, non-combinatorial experiment specs |
# | **Batch execution** | `BatchExperimentRunner` | Resumable, parallel-friendly per-gauge calibration |
# | **Result export** | `BatchResult.export()` | Excel / CSV export with diagnostics and FDC |
# | **Resume** | `run(resume=True)` | Skip previously completed experiments on re-run |
# | **Diagnostic clustermap** | `sns.clustermap` + `compute_diagnostics` | Hierarchical heatmap ranking calibrations per catchment |
#
# ### Key Takeaways
#
# 1. Multi-column rainfall/PET files are averaged to single series before calibration.
# 2. Catchment areas are computed automatically by matching subcatchment IDs in the
#    column names to the reference area table -- no manual entry needed.
# 3. Each gauge gets its own `ExperimentList` and `BatchExperimentRunner` instance.
# 4. Gauge-level failures are isolated -- one broken gauge does not stop the loop.
# 5. Every run creates a timestamped folder with full logs, config snapshot, and
#    per-experiment `.pkl` reports.
# 6. Update the **placeholder configuration** (Step 5) before executing calibrations.
