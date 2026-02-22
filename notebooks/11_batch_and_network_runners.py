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
# # Batch Experiments and Catchment Network Calibration
#
# ## Purpose
#
# This notebook demonstrates two experiment runner frameworks in `pyrrm` that
# automate multi-configuration calibration workflows:
#
# 1. **BatchExperimentRunner** -- runs multiple calibrations across combinations
#    of models, objectives, algorithms, and flow transformations on a
#    **single catchment**.
# 2. **CatchmentNetworkRunner** -- calibrates a **network of catchments** from
#    upstream to downstream, handling flow aggregation and optional link routing.
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
# - How to build a **catchment network** topology and calibrate it in
#   topological (upstream-to-downstream) order
# - How **layered configuration** lets you override model/objective/algorithm on
#   a per-node basis
#
# ## Prerequisites
#
# - Familiarity with `CalibrationRunner` (see **Notebook 02: Calibration Quickstart**)
# - Understanding of `GR4J` and `Sacramento` models
#
# ## Estimated Time
#
# - ~5 minutes for the batch experiments (synthetic data, small `max_evals`)
# - ~5 minutes for the network calibration
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
from pyrrm.models import GR4J, Sacramento
from pyrrm.objectives import NSE, KGE, FlowTransformation

# Batch experiment runner
from pyrrm.calibration.batch import (
    ExperimentGrid,
    ExperimentList,
    ExperimentSpec,
    BatchExperimentRunner,
    BatchResult,
)

# Network runner
from pyrrm.network import (
    CatchmentNode,
    NetworkLink,
    CatchmentNetwork,
    CatchmentNetworkRunner,
    NetworkCalibrationResult,
    NodeCalibrationConfig,
)

print("=" * 70)
print("BATCH EXPERIMENTS AND NETWORK CALIBRATION")
print("=" * 70)
print("\nAll imports loaded successfully!")

# %% [markdown]
# ---
# ## 1. Generate Synthetic Catchment Data
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
# ## Part A: Batch Experiment Runner
#
# The `BatchExperimentRunner` replaces manual `for` loops with a structured,
# resumable, parallelisable experiment framework.  It supports two modes:
#
# | Mode | Class | Description |
# |------|-------|-------------|
# | **Combinatorial grid** | `ExperimentGrid` | Cartesian product of models × objectives × algorithms |
# | **Explicit list** | `ExperimentList` | User-curated list of specific experiments |
#
# Each call to `run()` creates a **unique timestamped folder** containing all
# results, per-experiment logs, a batch log, and machine-readable summaries.

# %% [markdown]
# ### A.1 Define an Experiment Grid (Combinatorial)
#
# We cross two models (GR4J, Sacramento) with two objectives (NSE, KGE)
# and one algorithm (SCE-UA), producing 2 × 2 × 1 = 4 experiments.

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
# ### A.2 Run the Grid Experiment
#
# The runner delegates each experiment to `CalibrationRunner`, saves results
# incrementally to disk, and supports resuming if interrupted.  The
# `progress_bar=True` option shows a tqdm bar (works in both notebook and terminal).

# %%
runner = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=grid,
    output_dir='../test_data/batch_demo',
    warmup_period=365,
    catchment_info={'name': 'Synthetic Catchment', 'gauge_id': 'DEMO'},
    backend='sequential',
    progress_bar=True,
    log_level='INFO',
)

batch_result = runner.run(resume=False, run_name='grid_demo')
print(batch_result)

# %% [markdown]
# ### A.3 Inspect Batch Results
#
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
# ### A.4 Hydrographs and Calibration Fits
#
# Each experiment stores a `CalibrationReport` with observed and simulated
# time series.  We plot all four hydrographs stacked vertically, then show
# scatter and FDC plots for the overall best result.

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

# %% [markdown]
# ### A.5 Run Folder Structure
#
# Every `run()` call creates a timestamped folder under `output_dir`.
# Inside you'll find:
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
# ### A.6 Save and Reload
#
# The `BatchResult` is automatically saved inside the run folder as
# `batch_result.pkl`.  You can load it later without re-running anything.

# %%
reloaded = BatchResult.load(str(run_dir / 'batch_result.pkl'))
print(f"Reloaded: {reloaded}")
print(f"  Experiments: {len(reloaded.results)}")
print(f"  Failures:    {len(reloaded.failures)}")

# %% [markdown]
# ### A.7 Resume Capability
#
# If a batch run is interrupted (or you later add more experiments to the grid),
# the runner detects previously completed results in the most recent run folder
# and skips them.  Pass `resume=True` (the default) to enable this.

# %%
runner_resume = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=grid,
    output_dir='../test_data/batch_demo',
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
# ## Part A-bis: Explicit Experiment List (Non-Combinatorial)
#
# Sometimes the full Cartesian product is not what you want.  For example,
# you may wish to:
#
# - Run different `max_evals` for different models (more budget for Sacramento)
# - Run only specific model / objective combinations
# - Reproduce a predefined set of experiments from a config file
#
# `ExperimentList` lets you define the exact set of experiments to run, with
# no combinatorial expansion.

# %% [markdown]
# ### AL.1 Build an Experiment List (Programmatic)
#
# Each `ExperimentSpec` defines one experiment: model, objective, algorithm,
# and a unique key.

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

# %% [markdown]
# ### AL.2 Run the Experiment List
#
# The `BatchExperimentRunner` accepts either `ExperimentGrid` or
# `ExperimentList` -- the interface is identical.

# %%
list_runner = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=exp_list,
    output_dir='../test_data/batch_demo_list',
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
# ### AL.3 Build Experiment List from Dictionaries
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
# ### AL.4 YAML Config Example
#
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
# ## Part B: Catchment Network Runner
#
# The `CatchmentNetworkRunner` calibrates a river network from upstream to
# downstream, handling flow aggregation and optional Muskingum routing
# between nodes.
#
# We build a simple 4-node network:
#
# ```
# hw1 (Upper Creek, 50 km²) ──┐
#                               ├── j1 (Mid River, 80 km²) ──── outlet (Lower River, 120 km²)
# hw2 (Side Creek, 30 km²) ──┘
# ```
#
# The network is processed in **wavefronts**: headwaters (hw1, hw2) can be
# calibrated in parallel, followed by the junction (j1), then the outlet.

# %% [markdown]
# ### B.1 Generate Synthetic Network Data

# %%
np.random.seed(123)

def make_synthetic_data(n_days, scale=3.0, pet_offset=0.0):
    """Create synthetic P, PET, and 'true' flow using GR4J."""
    dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
    precip = np.maximum(0, np.random.exponential(scale, n_days))
    pet = 3.0 + pet_offset + 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    pet = np.maximum(0.1, pet + np.random.normal(0, 0.2, n_days))
    inputs = pd.DataFrame({'precipitation': precip, 'pet': pet}, index=dates)

    m = GR4J({'X1': 300, 'X2': 0.3, 'X3': 80, 'X4': 1.5})
    result = m.run(inputs)
    flow = result['flow'].values
    flow = np.maximum(0, flow + np.random.normal(0, 0.03, n_days) * flow)
    return inputs, flow

hw1_inputs, hw1_obs = make_synthetic_data(2500, scale=3.5)
hw2_inputs, hw2_obs = make_synthetic_data(2500, scale=2.5, pet_offset=0.5)
j1_inputs, _ = make_synthetic_data(2500, scale=3.0)
outlet_inputs, _ = make_synthetic_data(2500, scale=3.0, pet_offset=-0.3)

j1_obs = hw1_obs + hw2_obs * 0.8 + np.random.normal(0, 0.1, 2500)
j1_obs = np.maximum(0, j1_obs)
outlet_obs = j1_obs * 0.9 + np.random.normal(0, 0.2, 2500)
outlet_obs = np.maximum(0, outlet_obs)

print(f"Created synthetic data for 4-node network ({len(hw1_obs)} days each)")

# %% [markdown]
# ### B.2 Build Network Topology
#
# Each node has inputs, observed flow, and an optional downstream connection.
# Links carry optional Muskingum routing parameters.

# %%
nodes = [
    CatchmentNode(
        id='hw1', name='Upper Creek', area_km2=50,
        inputs=hw1_inputs, observed=hw1_obs,
        downstream_id='j1',
    ),
    CatchmentNode(
        id='hw2', name='Side Creek', area_km2=30,
        inputs=hw2_inputs, observed=hw2_obs,
        downstream_id='j1',
    ),
    CatchmentNode(
        id='j1', name='Mid River', area_km2=80,
        inputs=j1_inputs, observed=j1_obs,
        downstream_id='outlet',
    ),
    CatchmentNode(
        id='outlet', name='Lower River', area_km2=120,
        inputs=outlet_inputs, observed=outlet_obs,
    ),
]

links = [
    NetworkLink(
        upstream_id='hw1', downstream_id='j1',
        routing_method='muskingum',
        routing_params={'routing_K': 5.0, 'routing_m': 0.8, 'routing_n_subreaches': 3},
        calibrate_routing=True,
    ),
    NetworkLink(
        upstream_id='hw2', downstream_id='j1',
        routing_method='muskingum',
        routing_params={'routing_K': 3.0, 'routing_m': 0.7, 'routing_n_subreaches': 2},
        calibrate_routing=True,
    ),
    NetworkLink(
        upstream_id='j1', downstream_id='outlet',
        routing_method='muskingum',
        routing_params={'routing_K': 8.0, 'routing_m': 0.8, 'routing_n_subreaches': 5},
        calibrate_routing=True,
    ),
]

network = CatchmentNetwork(nodes, links)
print(network)
print()
print(network.summary())

# %% [markdown]
# ### B.3 Visualise the Network Topology

# %%
from pyrrm.network.visualization import (
    network_to_mermaid, plot_network,
    config_summary, link_config_summary, data_summary,
)

mermaid_str = network_to_mermaid(network, show_routing=True, show_wavefronts=True)
print(mermaid_str)

# %%
fig = plot_network(network)
fig.savefig('../figures/network_topology_demo.png', dpi=150, bbox_inches='tight')
print("Network topology plot saved")

# %% [markdown]
# ### B.4 Configure and Run the Network Calibration
#
# We use the **incremental** strategy with link routing calibration.
# The runner processes wavefronts in topological order: headwaters first,
# then junctions, then the outlet.
#
# We use a small `max_evals` for this demo.  In practice, use 10 000+.

# %%
net_runner = CatchmentNetworkRunner(
    network=network,
    default_model_class='GR4J',
    default_objective=NSE(),
    default_algorithm={'method': 'sceua_direct', 'max_evals': 2000, 'seed': 42},
    default_warmup_period=365,
    output_dir='../test_data/network_demo',
    strategy='incremental',
    link_routing=True,
    backend='sequential',
)

# Preview resolved config per node
configs = net_runner.get_resolved_configs()
print("Resolved configuration per node:\n")
for nid, cfg in configs.items():
    obj_name = getattr(cfg.objective, 'name', str(cfg.objective))
    print(f"  {nid:<10s}  model={cfg.model_class:<12s}  obj={obj_name:<6s}  warmup={cfg.warmup_period}")

# %%
net_result = net_runner.run(resume=False)
print(net_result)

# %% [markdown]
# ### B.5 Examine Network Results

# %%
results_df = net_result.to_dataframe()
results_df[['node_id', 'best_objective', 'runtime_seconds', 'method', 'success']]

# %%
link_df = net_result.link_summary()
if not link_df.empty:
    print("Calibrated link routing parameters:\n")
    display(link_df)
else:
    print("No link routing parameters calibrated")

# %% [markdown]
# ### B.6 Visualise Calibration Results
#
# The network hydrograph grid shows observed vs simulated for every gauged
# node in topological order.  The FDC overlay allows quick comparison of
# flow distributions.

# %%
from pyrrm.network.visualization import (
    result_to_mermaid, result_to_styled_dataframe,
    plot_network_hydrographs, plot_network_fdc,
)

result_mermaid = result_to_mermaid(net_result)
print(result_mermaid)

# %%
fig = plot_network_hydrographs(net_result)
fig.savefig('../figures/network_hydrographs_demo.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
fig = plot_network_fdc(net_result)
fig.savefig('../figures/network_fdc_demo.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### B.7 Per-Node Scatter and FDC Plots
#
# Scatter plots and flow duration curves for each node give a closer look
# at calibration quality at individual gauges.

# %%
for nid, report in net_result.node_results.items():
    if report.observed is None or len(report.observed) == 0:
        continue
    node = net_result.network.get_node(nid)
    label = f"{nid} ({node.name})" if node.name else nid
    obj_val = report.result.best_objective

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    obs = report.observed
    sim = report.simulated
    max_val = max(np.nanmax(obs), np.nanmax(sim)) * 1.05

    ax1.scatter(obs, sim, s=6, alpha=0.3, edgecolors='none')
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Observed')
    ax1.set_ylabel('Simulated')
    ax1.set_title(f'{label} – Scatter')
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_aspect('equal')

    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100

    ax2.plot(exc_obs, obs_sorted, 'b-', alpha=0.7, label='Observed')
    ax2.plot(exc_sim, sim_sorted, 'r--', alpha=0.7, label='Simulated')
    ax2.set_yscale('log')
    ax2.set_xlabel('Exceedance (%)')
    ax2.set_ylabel('Flow')
    ax2.set_title(f'{label} – FDC')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{label}  |  {report.result.objective_name} = {obj_val:.4f}",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### B.8 Save and Reload Network Results

# %%
net_result.save('../test_data/network_demo/network_result.pkl')
reloaded_net = NetworkCalibrationResult.load('../test_data/network_demo/network_result.pkl')
print(f"Reloaded: {reloaded_net}")

# %% [markdown]
# ### B.9 Per-Node Configuration Overrides (Layered Config)
#
# The network runner supports **layered configuration**: set network-wide
# defaults, then override specific nodes.  Here we configure the junction
# node (`j1`) to use Sacramento instead of GR4J, and `hw2` to use KGE
# instead of NSE.

# %%
nodes_v2 = [
    CatchmentNode(
        id='hw1', name='Upper Creek', area_km2=50,
        inputs=hw1_inputs, observed=hw1_obs,
        downstream_id='j1',
    ),
    CatchmentNode(
        id='hw2', name='Side Creek', area_km2=30,
        inputs=hw2_inputs, observed=hw2_obs,
        downstream_id='j1',
        calibration_config=NodeCalibrationConfig(
            objective=KGE(),
        ),
    ),
    CatchmentNode(
        id='j1', name='Mid River', area_km2=80,
        inputs=j1_inputs, observed=j1_obs,
        downstream_id='outlet',
        calibration_config=NodeCalibrationConfig(
            model_class='Sacramento',
            warmup_period=730,
        ),
    ),
    CatchmentNode(
        id='outlet', name='Lower River', area_km2=120,
        inputs=outlet_inputs, observed=outlet_obs,
    ),
]

network_v2 = CatchmentNetwork(nodes_v2, links)

net_runner_v2 = CatchmentNetworkRunner(
    network=network_v2,
    default_model_class='GR4J',
    default_objective=NSE(),
    default_algorithm={'method': 'sceua_direct', 'max_evals': 2000, 'seed': 42},
    default_warmup_period=365,
    output_dir='../test_data/network_demo_v2',
    strategy='incremental',
    link_routing=False,
    backend='sequential',
)

configs_v2 = net_runner_v2.get_resolved_configs()
print("Mixed-model configuration:\n")
for nid, cfg in configs_v2.items():
    obj_name = getattr(cfg.objective, 'name', str(cfg.objective))
    print(f"  {nid:<10s}  model={cfg.model_class:<12s}  obj={obj_name:<6s}  warmup={cfg.warmup_period}")

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated two experiment runner frameworks in `pyrrm`:
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
# | **Network calibration** | `CatchmentNetwork` | DAG topology with wavefronts |
# | | `CatchmentNetworkRunner` | Upstream-to-downstream calibration |
# | | `NetworkLink` | Muskingum routing between nodes |
# | | `NetworkCalibrationResult` | Per-node and per-link results |
# | **Layered config** | `NodeCalibrationConfig` | Per-node overrides for model/objective/algorithm |
# | **Visualisation** | Mermaid + matplotlib | Network topology and results diagrams |
# | **Parallelism** | `ParallelBackend` | Sequential, multiprocessing, or Ray |
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
# 5. **The network runner** handles multi-catchment topologies with wavefront
#    scheduling and optional routing calibration.
