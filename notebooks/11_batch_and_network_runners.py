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
# # Batch Experiments and Catchment Network Calibration
#
# This notebook demonstrates the two new experiment runner capabilities
# added to `pyrrm`:
#
# 1. **BatchExperimentRunner** -- automates running multiple calibrations
#    across combinations of models, objectives, algorithms, and flow
#    transformations on a **single catchment**.
#
# 2. **CatchmentNetworkRunner** -- calibrates a **network of catchments**
#    from upstream to downstream, handling flow aggregation and optional
#    link routing.
#
# Both runners support parallelism (sequential, multiprocessing, or Ray)
# and provide resume capability, structured results, and visualisations.
#
# ## Contents
#
# - Part A: Batch Experiment Runner (single catchment, multiple configurations)
# - Part B: Catchment Network Runner (multi-catchment upstream-to-downstream)

# %% [markdown]
# ---
# ## Part A: Batch Experiment Runner
#
# The `BatchExperimentRunner` replaces manual `for` loops with a
# structured, resumable, parallelisable grid experiment framework.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

from pyrrm.models import GR4J, Sacramento
from pyrrm.objectives import NSE, KGE, FlowTransformation
from pyrrm.calibration.batch import (
    ExperimentGrid, ExperimentSpec, BatchExperimentRunner, BatchResult,
)

# %% [markdown]
# ### A.1 Generate synthetic data for a quick demo
#
# We create synthetic rainfall, PET, and flow data so the notebook runs
# fast. In practice you would load real gauge data.

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
print(f"Mean flow: {observed.mean():.2f} mm/day")

# %% [markdown]
# ### A.2 Define the experiment grid
#
# We cross two models (GR4J, Sacramento) with two objectives (NSE, KGE)
# and one algorithm (SCE-UA). This creates 2 × 2 × 1 = 4 experiments.

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
)

specs = grid.combinations()
print(f"Experiment grid: {len(grid)} experiments")
for s in specs:
    print(f"  {s.key}")

# %% [markdown]
# ### A.3 Run the batch experiment
#
# The runner delegates each experiment to `CalibrationRunner`, saves
# results incrementally, and supports resuming if interrupted.

# %%
runner = BatchExperimentRunner(
    inputs=inputs_df,
    observed=observed,
    grid=grid,
    output_dir='../test_data/batch_demo',
    warmup_period=365,
    catchment_info={'name': 'Synthetic Catchment', 'gauge_id': 'DEMO'},
    backend='sequential',
)

batch_result = runner.run(resume=True)
print(batch_result)

# %%
df = batch_result.to_dataframe()
display_cols = ['key', 'model', 'objective', 'best_objective', 'runtime_seconds', 'success']
df[display_cols].sort_values('best_objective', ascending=False)

# %%
best = batch_result.best_by_objective()
for obj_name, (key, val) in best.items():
    print(f"Best {obj_name}: {key} (value={val:.4f})")

# %% [markdown]
# ### A.4 Save and reload results
#
# The `BatchResult` can be saved and loaded for later analysis.

# %%
batch_result.save('../test_data/batch_demo/batch_result.pkl')
reloaded = BatchResult.load('../test_data/batch_demo/batch_result.pkl')
print(f"Reloaded: {reloaded}")

# %% [markdown]
# ---
# ## Part B: Catchment Network Runner
#
# The `CatchmentNetworkRunner` calibrates a river network from upstream
# to downstream, handling flow aggregation and optional Muskingum routing.
#
# We build a simple 4-node network:
#
# ```
# hw1 (Upper Creek, 50 km²) ──┐
#                               ├── j1 (Mid River, 80 km²) ──── outlet (Lower River, 120 km²)
# hw2 (Side Creek, 30 km²) ──┘
# ```

# %% [markdown]
# ### B.1 Build network topology

# %%
from pyrrm.network import (
    CatchmentNode, NetworkLink, CatchmentNetwork,
    CatchmentNetworkRunner, NetworkCalibrationResult,
    NodeCalibrationConfig,
)

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
# ### B.2 Visualise the network topology

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
print("Network plot saved")

# %% [markdown]
# ### B.3 Configure and run the network calibration
#
# We use the **incremental** strategy with link routing calibration.
# The runner processes wavefronts in order: headwaters first, then
# junctions, then the outlet.
#
# We use a small `max_evals` for this demo. In practice, use 10000+.

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

# %% [markdown]
# Preview the resolved configuration for each node:

# %%
configs = net_runner.get_resolved_configs()
for nid, cfg in configs.items():
    print(f"  {nid}: model={cfg.model_class}, obj={getattr(cfg.objective, 'name', str(cfg.objective))}, warmup={cfg.warmup_period}")

# %%
net_result = net_runner.run(resume=False)
print(net_result)

# %% [markdown]
# ### B.4 Examine network results

# %%
results_df = net_result.to_dataframe()
results_df[['node_id', 'best_objective', 'runtime_seconds', 'method', 'success']]

# %%
link_df = net_result.link_summary()
if not link_df.empty:
    print("Calibrated link routing parameters:")
    display(link_df)
else:
    print("No link routing parameters calibrated")

# %% [markdown]
# ### B.5 Visualise calibration results

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
print("Hydrographs saved")

# %%
fig = plot_network_fdc(net_result)
fig.savefig('../figures/network_fdc_demo.png', dpi=150, bbox_inches='tight')
print("FDCs saved")

# %% [markdown]
# ### B.6 Save and reload network results

# %%
net_result.save('../test_data/network_demo/network_result.pkl')
reloaded_net = NetworkCalibrationResult.load('../test_data/network_demo/network_result.pkl')
print(f"Reloaded: {reloaded_net}")

# %% [markdown]
# ### B.7 Per-node calibration with different configurations
#
# The network runner supports **layered configuration**: set network-wide
# defaults, then override specific nodes. Here we configure the junction
# node to use Sacramento instead of GR4J.

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
print("Mixed-model configuration:")
for nid, cfg in configs_v2.items():
    print(f"  {nid}: model={cfg.model_class}, obj={getattr(cfg.objective, 'name', str(cfg.objective))}, warmup={cfg.warmup_period}")

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated:
#
# | Feature | Component | Description |
# |---------|-----------|-------------|
# | **Batch experiments** | `ExperimentGrid` | Define model × objective × algorithm combinations |
# | | `BatchExperimentRunner` | Run all combinations with resume, progress, parallelism |
# | | `BatchResult` | Structured results with summary tables |
# | **Network calibration** | `CatchmentNetwork` | DAG topology with wavefronts |
# | | `CatchmentNetworkRunner` | Upstream-to-downstream calibration |
# | | `NetworkLink` | Muskingum routing between nodes |
# | | `NetworkCalibrationResult` | Per-node and per-link results |
# | **Layered config** | `NodeCalibrationConfig` | Per-node overrides for model/objective/algorithm |
# | **Visualization** | Mermaid + matplotlib | Network topology and results diagrams |
# | **Parallelism** | `ParallelBackend` | Sequential, multiprocessing, or Ray |
