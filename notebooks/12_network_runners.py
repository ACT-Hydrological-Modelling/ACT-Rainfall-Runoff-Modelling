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
# # Catchment Network Runners
#
# ## Purpose
#
# This notebook focuses on **CatchmentNetworkRunner**: calibrating a **network of
# catchments** in topological order (upstream to downstream), with flow
# aggregation and optional Muskingum routing between nodes. It demonstrates how
# to build a catchment DAG, run wavefront-based calibration, and use layered
# per-node configuration.
#
# ## What You'll Learn
#
# - How to build a **catchment network** topology with `CatchmentNode` and
#   `NetworkLink`
# - How the network runner processes nodes in **wavefronts** (headwaters first,
#   then junctions, then outlet)
# - How **layered configuration** lets you override model, objective, or
#   algorithm on a per-node basis
# - How to visualise topology and calibration results (Mermaid, hydrographs, FDC)
#
# ## Prerequisites
#
# - Familiarity with `CalibrationRunner` (see **Notebook 02: Calibration Quickstart**)
# - Understanding of GR4J (used for synthetic data and default model)
#
# ## Estimated Time
#
# - ~5 minutes for network calibration (synthetic 4-node network, small `max_evals`)
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|--------|---------------|
# | 1 | Setup and synthetic network data | Imports; build synthetic P/PET/flow for a 4-node network. |
# | 2 | Build network topology | `CatchmentNode`, `NetworkLink`, `CatchmentNetwork`; optional Muskingum. |
# | 3 | Visualise topology | Mermaid and `plot_network`. |
# | 4 | Configure and run network calibration | `CatchmentNetworkRunner`, strategy, link routing, resolved configs. |
# | 5 | Examine results | `NetworkCalibrationResult`, node table, link summary. |
# | 6 | Visualise hydrographs and FDC | `plot_network_hydrographs`, `plot_network_fdc`. |
# | 7 | Per-node scatter and FDC | Per-gauge diagnostics. |
# | 8 | Save and reload | `NetworkCalibrationResult.save/load`. |
# | 9 | Per-node configuration overrides | `NodeCalibrationConfig`, layered config. |
#
# ## Key Insight
#
# > The network runner calibrates from upstream to downstream in wavefronts, so
# > each node has upstream flows available when needed. Optional link routing
# > (e.g. Muskingum) can be calibrated between nodes for realistic flow propagation.

# %% [markdown]
# ---
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# pyrrm models and objectives
from pyrrm.models import GR4J
from pyrrm.objectives import NSE, KGE

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
print("CATCHMENT NETWORK RUNNERS")
print("=" * 70)
print("\nAll imports loaded successfully!")

# %% [markdown]
# ---
# ## Step 1: Setup and Synthetic Network Data
#
# We create synthetic rainfall, PET, and observed flow for a simple 4-node
# network so the notebook runs quickly without external data. A GR4J model
# generates local flow at each node; we aggregate flows at junctions and add
# noise to simulate observation error.
#
# Network layout:
#
# ```
# hw1 (Upper Creek, 50 km²) ──┐
#                               ├── j1 (Mid River, 80 km²) ──── outlet (Lower River, 120 km²)
# hw2 (Side Creek, 30 km²) ──┘
# ```

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
# ---
# ## Step 2: Build Network Topology
#
# Each node has inputs, observed flow, and an optional downstream connection.
# Links carry optional Muskingum routing parameters. The network is processed
# in **wavefronts**: headwaters (hw1, hw2) can be calibrated in parallel,
# followed by the junction (j1), then the outlet.

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
# ---
# ## Step 3: Visualise Topology
#
# Mermaid output can be pasted into documentation or rendered in tools that
# support it. `plot_network` draws the DAG with matplotlib.

# %%
from pyrrm.network.visualization import (
    network_to_mermaid, plot_network,
)

mermaid_str = network_to_mermaid(network, show_routing=True, show_wavefronts=True)
print(mermaid_str)

# %%
fig = plot_network(network)
fig.savefig('../figures/network_topology_demo.png', dpi=150, bbox_inches='tight')
print("Network topology plot saved")
plt.show()

# %% [markdown]
# ---
# ## Step 4: Configure and Run Network Calibration
#
# We use the **incremental** strategy with link routing calibration. The runner
# processes wavefronts in topological order: headwaters first, then junctions,
# then the outlet. We use a small `max_evals` for this demo; in practice use
# 10 000+.

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
# ---
# ## Step 5: Examine Results
#
# `NetworkCalibrationResult` provides a node-level summary DataFrame and an
# optional link summary for calibrated routing parameters.

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
# ---
# ## Step 6: Visualise Hydrographs and FDC
#
# The network hydrograph grid shows observed vs simulated for every gauged
# node in topological order. The FDC overlay allows quick comparison of
# flow distributions.

# %%
from pyrrm.network.visualization import (
    result_to_mermaid, plot_network_hydrographs, plot_network_fdc,
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
# ---
# ## Step 7: Per-Node Scatter and FDC
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
# ---
# ## Step 8: Save and Reload
#
# Save the full `NetworkCalibrationResult` to disk and reload it later without
# re-running calibration.

# %%
net_result.save('../test_data/network_demo/network_result.pkl')
reloaded_net = NetworkCalibrationResult.load('../test_data/network_demo/network_result.pkl')
print(f"Reloaded: {reloaded_net}")

# %% [markdown]
# ---
# ## Step 9: Per-Node Configuration Overrides
#
# The network runner supports **layered configuration**: set network-wide
# defaults, then override specific nodes. Here we configure the junction
# node (`j1`) to use Sacramento instead of GR4J, and `hw2` to use KGE
# instead of NSE.

# %%
from pyrrm.models import Sacramento

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
# This notebook demonstrated the **catchment network runner** in `pyrrm`:
#
# | Feature | Component | Description |
# |---------|-----------|-------------|
# | **Network topology** | `CatchmentNetwork` | DAG of nodes and links with wavefront ordering |
# | **Nodes** | `CatchmentNode` | Inputs, observed flow, optional downstream_id |
# | **Links** | `NetworkLink` | Optional Muskingum routing between nodes |
# | **Runner** | `CatchmentNetworkRunner` | Upstream-to-downstream calibration |
# | **Results** | `NetworkCalibrationResult` | Per-node and per-link results |
# | **Layered config** | `NodeCalibrationConfig` | Per-node overrides for model/objective/algorithm |
# | **Visualisation** | Mermaid + matplotlib | Network topology and results diagrams |
#
# ### Key Takeaways
#
# 1. **Build the network** with `CatchmentNode` and `NetworkLink`; the runner
#    infers topological order and wavefronts.
# 2. **Use layered configuration** when different nodes need different models,
#    objectives, or warmup periods.
# 3. **Link routing** (e.g. Muskingum) can be calibrated between nodes for
#    realistic flow propagation along reaches.
# 4. **Save and reload** `NetworkCalibrationResult` for later analysis without
#    re-running calibration.
