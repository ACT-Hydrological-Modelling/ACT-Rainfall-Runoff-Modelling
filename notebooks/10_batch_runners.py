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
# # Batch Experiment Runners
#
# ## Purpose
#
# This notebook focuses on **BatchExperimentRunner**: running many calibrations on a
# **single catchment** across combinations of models, objectives, and algorithms.
# It replaces manual `for` loops with a structured, resumable, parallel-friendly
# framework and organised timestamped output.
#
# ## What You'll Learn
#
# - How to define a **combinatorial experiment grid** (`ExperimentGrid`) and run
#   every combination in a single call
# - How to define a **curated experiment list** (`ExperimentList`) when you want
#   fine-grained control over which experiments to run
# - How the batch runner creates **organised, timestamped run folders** with
#   per-experiment logs, summary files, and config snapshots
# - How to **resume** interrupted batch runs and inspect batch results
#
# ## Prerequisites
#
# - Familiarity with `CalibrationRunner` (see **Notebook 02: Calibration Quickstart**)
# - Understanding of `GR4J` and `Sacramento` models
#
# ## Estimated Time
#
# - ~5 minutes for grid and list experiments (synthetic data, small `max_evals`)
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|--------|---------------|
# | 1 | Setup and synthetic data | Imports; generate single-catchment synthetic P/PET/flow. |
# | 2 | Experiment grid | Define `ExperimentGrid` (models × objectives × algorithms). |
# | 3 | Run grid and inspect results | `BatchExperimentRunner.run()`, `BatchResult`, summary table, best-by-objective. |
# | 4 | Hydrographs and run folder | Plot hydrographs; describe timestamped run folder (logs, summaries, config). |
# | 5 | Save, reload, and resume | `BatchResult.load()`, `run(resume=True)`. |
# | 6 | Explicit experiment list | `ExperimentList` and `ExperimentSpec` for non-combinatorial runs. |
# | 7 | List from dicts and YAML | `ExperimentList.from_dicts()`; YAML config examples for grid vs list. |
#
# ## Key Insight
#
# > The batch runner eliminates manual `for` loops and provides structured,
# > resumable, parallel-friendly experiment management.  Every run is stored
# > in a dated folder with full logs, so you can always reproduce and compare
# > past experiments.

# %% [markdown]
# ---
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import json
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# pyrrm models and objectives
from pyrrm.models import GR4J, Sacramento, NUMBA_AVAILABLE
from pyrrm.objectives import NSE, KGE

# Batch experiment runner
from pyrrm.calibration.batch import (
    ExperimentGrid,
    ExperimentList,
    ExperimentSpec,
    BatchExperimentRunner,
    BatchResult,
)

OUTPUT_DIR = Path('../test_data/11_batch_runners')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BATCH EXPERIMENT RUNNERS")
print("=" * 70)
print("\nAll imports loaded successfully!")
print(f"Numba JIT acceleration: {'ACTIVE' if NUMBA_AVAILABLE else 'not available (pip install numba)'}")

# %% [markdown]
# ---
# ## Step 1: Setup and Synthetic Data
#
# We create synthetic rainfall, PET, and observed flow data so the notebook runs
# quickly.  A GR4J model with known parameters generates the "truth" signal,
# and we add 5% multiplicative noise to simulate observation error.

# %%
np.random.seed(42)
n_days = 3000

dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
precip = np.maximum(0, np.random.exponential(3, n_days))
pet = 3.0 + 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 0.3, n_days)
pet = np.maximum(0, pet)

truth_model = GR4J({'X1': 350, 'X2': 0.5, 'X3': 90, 'X4': 1.7})
inputs_df = pd.DataFrame({'precipitation': precip, 'pet': pet}, index=dates)
result_df = truth_model.run(inputs_df)
observed = result_df['flow'].values + np.random.normal(0, 0.05, n_days) * result_df['flow'].values
observed = np.maximum(0, observed)

print(f"Synthetic data: {n_days} days, {dates[0].date()} to {dates[-1].date()}")
print(f"Mean observed flow: {observed.mean():.2f} mm/day")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax1.bar(dates, precip, width=1, color='steelblue', alpha=0.7)
ax1.invert_yaxis()
ax1.set_ylabel('Precipitation (mm/d)')
ax1.set_title('Synthetic Catchment Data')
ax2.plot(dates, observed, 'b-', linewidth=0.5, alpha=0.7, label='Observed flow')
ax2.set_ylabel('Flow (mm/d)')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 2: Experiment Grid
#
# The `BatchExperimentRunner` supports two modes:
#
# | Mode | Class | Description |
# |------|-------|-------------|
# | **Combinatorial grid** | `ExperimentGrid` | Cartesian product of models × objectives × algorithms |
# | **Explicit list** | `ExperimentList` | User-curated list of specific experiments |
#
# Here we define an **ExperimentGrid**: we cross two models (GR4J, Sacramento)
# with two objectives (NSE, KGE) and one algorithm (SCE-UA), producing 4 experiments.

# %%
grid = ExperimentGrid(
    models={
        'GR4J': GR4J(),
        'Sacramento': Sacramento(),
    },
    objectives={
        'nse': NSE(),
        'kge': KGE(),
    },
    algorithms={
        'sceua': {'method': 'sceua_direct', 'max_evals': 2000, 'seed': 42},
    },
    catchment='synthetic',
)

print(f"Experiment grid: {len(grid)} experiments\n")
for s in grid.combinations():
    print(f"  {s.key}")

# %% [markdown]
# ---
# ## Step 3: Run Grid and Inspect Results
#
# The runner delegates each experiment to `CalibrationRunner`, saves results
# incrementally to disk, and supports resuming if interrupted.  The
# `progress_bar=True` option shows a tqdm bar (works in both notebook and terminal).

# %%
runner = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=grid,
    output_dir=str(OUTPUT_DIR / 'batch_demo'),
    warmup_period=365,
    catchment_info={'name': 'Synthetic Catchment', 'gauge_id': 'DEMO'},
    backend='sequential',
    progress_bar=True,
    log_level='INFO',
)

batch_result = runner.run(resume=False, run_name='grid_demo')
print(batch_result)

# %% [markdown]
# The `BatchResult` provides a summary DataFrame and a method to find the best
# experiment for each objective function.

# %%
df = batch_result.to_dataframe()
display_cols = ['key', 'model', 'objective', 'best_objective', 'runtime_seconds', 'success']
df[display_cols].sort_values('best_objective', ascending=False)

# %%
best = batch_result.best_by_objective()
print("Best experiment per objective:\n")
for obj_name, (key, val) in best.items():
    print(f"  {obj_name:>5s}:  {key:<35s}  ({val:.4f})")

# %% [markdown]
# ---
# ## Step 4: Hydrographs and Run Folder
#
# Each experiment stores a `CalibrationReport` with observed and simulated
# time series.  We plot all four hydrographs stacked vertically, then show
# scatter and FDC plots for the overall best result.
#
# Every `run()` call creates a **timestamped folder** under `output_dir`:
#
# | File / Folder | Content |
# |---|---|
# | `batch.log` | Full log of the batch run |
# | `batch_summary.json` | Machine-readable summary (completed, failed, best per objective) |
# | `batch_summary.csv` | Results DataFrame as CSV |
# | `batch_result.pkl` | Serialised `BatchResult` object |
# | `config.yaml` | Snapshot of the experiment grid / list configuration |
# | `results/` | Per-experiment `.pkl` files (one `CalibrationReport` each) |
# | `logs/` | Per-experiment `.log` files (timing, parameters, result) |

# %%
n_exps = len(batch_result.results)
fig, axes = plt.subplots(n_exps, 1, figsize=(14, 3.5 * n_exps), squeeze=False, sharex=True)

for i, (key, report) in enumerate(batch_result.results.items()):
    ax = axes[i, 0]
    obs = report.observed
    sim = report.simulated
    plot_dates = report.dates if report.dates is not None else np.arange(len(obs))
    n = min(len(obs), len(sim), len(plot_dates))

    ax.plot(plot_dates[:n], obs[:n], 'b-', alpha=0.6, linewidth=0.8, label='Observed')
    ax.plot(plot_dates[:n], sim[:n], 'r-', alpha=0.6, linewidth=0.8, label='Simulated')

    obj_val = report.result.best_objective
    ax.set_title(f"{key}  |  {report.result.objective_name} = {obj_val:.4f}", fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('Flow (mm/d)')

axes[-1, 0].set_xlabel('Date')
fig.suptitle('Batch Experiment – Hydrographs', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.show()

# %%
overall_best_key = max(
    batch_result.results,
    key=lambda k: batch_result.results[k].result.best_objective,
)
best_report = batch_result.results[overall_best_key]
print(f"Best overall: {overall_best_key}")
print(f"  {best_report.result.objective_name} = {best_report.result.best_objective:.4f}")

fig_scatter = best_report.plot_scatter()
plt.show()

fig_fdc = best_report.plot_fdc()
plt.show()

# %%
run_dir = Path(batch_result.run_dir)
print(f"Run folder: {run_dir.name}")
print(f"Full path:  {run_dir}\n")

for root, dirs, files in os.walk(run_dir):
    level = len(Path(root).relative_to(run_dir).parts)
    indent = '  ' * level
    print(f"{indent}{Path(root).name}/")
    for fname in sorted(files):
        fsize = (Path(root) / fname).stat().st_size
        if fsize > 1024:
            size_str = f"{fsize / 1024:.1f} KB"
        else:
            size_str = f"{fsize} B"
        print(f"{indent}  {fname:<40s}  ({size_str})")

# %%
with open(run_dir / 'batch_summary.json') as f:
    summary = json.load(f)
print(json.dumps(summary, indent=2))

# %% [markdown]
# ---
# ## Step 5: Save, Reload, and Resume
#
# The `BatchResult` is automatically saved inside the run folder as
# `batch_result.pkl`.  You can load it later without re-running anything.
# If a batch run is interrupted (or you later add more experiments to the grid),
# the runner detects previously completed results in the most recent run folder
# and skips them.  Pass `resume=True` (the default) to enable this.

# %%
reloaded = BatchResult.load(str(run_dir / 'batch_result.pkl'))
print(f"Reloaded: {reloaded}")
print(f"  Experiments: {len(reloaded.results)}")
print(f"  Failures:    {len(reloaded.failures)}")

# %%
runner_resume = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=grid,
    output_dir=str(OUTPUT_DIR / 'batch_demo'),
    warmup_period=365,
    catchment_info={'name': 'Synthetic Catchment', 'gauge_id': 'DEMO'},
    backend='sequential',
    progress_bar=False,
)

resumed_result = runner_resume.run(resume=True)
print(f"Resumed result: {resumed_result}")
print(f"\nRuntime = {resumed_result.runtime_seconds:.1f}s  (should be ~0s since all experiments were cached)")

# %% [markdown]
# ---
# ## Step 6: Explicit Experiment List
#
# Sometimes the full Cartesian product is not what you want.  For example,
# you may wish to:
#
# - Run different `max_evals` for different models (more budget for Sacramento)
# - Run only specific model / objective combinations
# - Reproduce a predefined set of experiments from a config file
#
# `ExperimentList` lets you define the exact set of experiments to run, with
# no combinatorial expansion.  Each `ExperimentSpec` defines one experiment:
# model, objective, algorithm, and a unique key.

# %%
experiment_specs = [
    ExperimentSpec(
        key='synthetic_gr4j_nse_fast',
        model_name='GR4J',
        model=GR4J(),
        objective_name='nse',
        objective=NSE(),
        algorithm_name='sceua',
        algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 1000, 'seed': 42},
    ),
    ExperimentSpec(
        key='synthetic_gr4j_kge_thorough',
        model_name='GR4J',
        model=GR4J(),
        objective_name='kge',
        objective=KGE(),
        algorithm_name='sceua',
        algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 3000, 'seed': 42},
    ),
    ExperimentSpec(
        key='synthetic_sacramento_nse_thorough',
        model_name='Sacramento',
        model=Sacramento(),
        objective_name='nse',
        objective=NSE(),
        algorithm_name='sceua',
        algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 3000, 'seed': 42},
    ),
]

exp_list = ExperimentList(experiment_specs)
print(f"Experiment list: {len(exp_list)} experiments\n")
for s in exp_list.combinations():
    print(f"  {s.key:<30s}  model={s.model_name:<12s}  max_evals={s.algorithm_kwargs.get('max_evals')}")

# %%
list_runner = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=exp_list,
    output_dir=str(OUTPUT_DIR / 'batch_demo_list'),
    warmup_period=365,
    catchment_info={'name': 'Synthetic Catchment', 'gauge_id': 'DEMO'},
    backend='sequential',
    progress_bar=True,
)

list_result = list_runner.run(resume=False, run_name='list_demo')
print(list_result)

# %%
list_df = list_result.to_dataframe()
list_df[['key', 'model', 'objective', 'best_objective', 'runtime_seconds', 'success']]

# %% [markdown]
# ---
# ## Step 7: List from Dicts and YAML
#
# For YAML/JSON-driven workflows or programmatic generation, use the
# `ExperimentList.from_dicts()` factory.  Each dict specifies the model by
# name (resolved from the registry), an objective spec, and algorithm kwargs.

# %%
experiment_dicts = [
    {
        'key': 'synthetic_gr4j_nse_from_dict',
        'model': 'GR4J',
        'objective': {'type': 'NSE'},
        'algorithm': {'method': 'sceua_direct', 'max_evals': 1000, 'seed': 99},
    },
    {
        'key': 'synthetic_sacramento_kge_from_dict',
        'model': 'Sacramento',
        'objective': {'type': 'KGE'},
        'algorithm': {'method': 'sceua_direct', 'max_evals': 2000, 'seed': 99},
    },
]

exp_list_from_dicts = ExperimentList.from_dicts(experiment_dicts)
print(f"From dicts: {len(exp_list_from_dicts)} experiments\n")
for s in exp_list_from_dicts.combinations():
    print(f"  {s.key}: model={s.model_name}, objective={s.objective_name}")

# %% [markdown]
# Both modes are supported in YAML/JSON config files loaded with
# `BatchExperimentRunner.from_config()`.
#
# **Combinatorial grid** (all combinations):
#
# ```yaml
# models:
#   GR4J: {}
#   Sacramento: {}
# objectives:
#   nse: {type: NSE}
#   kge: {type: KGE}
# algorithms:
#   sceua:
#     method: sceua_direct
#     max_evals: 5000
#     seed: 42
# warmup_days: 365
# output_dir: ./results
# ```
#
# **Explicit experiment list** (curated):
#
# ```yaml
# experiments:
#   - key: synthetic_gr4j_nse
#     model: GR4J
#     objective: {type: NSE}
#     algorithm: {method: sceua_direct, max_evals: 5000}
#   - key: synthetic_sacramento_kge
#     model: Sacramento
#     objective: {type: KGE}
#     algorithm: {method: sceua_direct, max_evals: 10000}
# warmup_days: 365
# output_dir: ./results
# ```
#
# The two modes are **mutually exclusive** -- a config cannot contain both
# `experiments` and `models`/`objectives`/`algorithms`.

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated the **batch experiment runner** in `pyrrm`:
#
# | Feature | Component | Description |
# |---------|-----------|-------------|
# | **Combinatorial grid** | `ExperimentGrid` | Full Cartesian product of models × objectives × algorithms |
# | **Explicit list** | `ExperimentList` | User-curated list of specific experiments |
# | | `ExperimentList.from_dicts()` | Build experiment lists from dictionaries / YAML |
# | **Batch runner** | `BatchExperimentRunner` | Run experiments with resume, progress bar, logging |
# | | `BatchResult` | Structured results with summary tables |
# | **Organised output** | Timestamped run folders | `results/`, `logs/`, `batch.log`, `batch_summary.json/csv` |
# | **Resume** | `run(resume=True)` | Skip previously completed experiments |
#
# ### Key Takeaways
#
# 1. **Use `ExperimentGrid`** when you want to systematically test all combinations
#    of models, objectives, and algorithms.
# 2. **Use `ExperimentList`** when you want fine-grained control -- different
#    settings per experiment, a subset of combinations, or configs from a file.
# 3. **Every run creates a timestamped folder** with full logs, summaries, and
#    config snapshots for reproducibility.
# 4. **Resume is built-in** -- interrupted runs pick up where they left off.
