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
# # Objective Functions: The Heart of Model Calibration
#
# > **⚠️ WORK IN PROGRESS (WIP)** - This notebook is currently under development.
#
# ## Purpose
#
# This notebook provides a deep dive into objective function selection and its
# profound impact on calibration results. Different objective functions lead to
# different "optimal" parameter sets - understanding this is crucial for
# effective hydrological modeling.
#
# ## What You'll Learn
#
# - Why objective function choice matters (it's not just NSE!)
# - The full catalog of metrics available in `pyrrm.objectives`
# - How flow transformations shift emphasis between high and low flows
# - How to build composite objectives for balanced calibration
# - Practical guidance: which objective for which use case?
#
# ## Prerequisites
#
# - Completed **Notebook 02: Calibration Quickstart**
# - Basic understanding of calibration concepts
#
# ## Estimated Time
#
# - ~20 minutes for concepts and examples
# - ~2-4 hours for full calibration experiment (can be shortened)
#
# ## Key Insight
#
# > **Different objective functions lead to different "optimal" parameter sets.**
# > There is no single "best" objective - the choice depends on your application.

# %% [markdown]
# ---
# ## The pyrrm.objectives Module Architecture
#
# The `pyrrm.objectives` module provides a comprehensive toolkit for constructing
# and evaluating objective functions. It follows a composable design where
# metrics, transformations, and aggregation methods can be combined flexibly.
#
# ```
# pyrrm.objectives/
# ├── core/                    ← Foundation classes
# │   ├── base.py              ← ObjectiveFunction (abstract base class)
# │   ├── result.py            ← MetricResult (evaluation results container)
# │   └── constants.py         ← Default values, FDC segments
# │
# ├── metrics/                 ← Individual objective functions
# │   ├── traditional.py       ← NSE, RMSE, MAE, PBIAS, SDEB
# │   ├── kge.py               ← KGE family (2009, 2012, 2021, non-parametric)
# │   └── correlation.py       ← Pearson, Spearman correlation
# │
# ├── transformations/         ← Flow transformations
# │   └── flow_transforms.py   ← sqrt, log, inverse, power, Box-Cox
# │
# ├── fdc/                     ← Flow Duration Curve metrics
# │   ├── curves.py            ← compute_fdc()
# │   └── metrics.py           ← FDCMetric (segment-based evaluation)
# │
# ├── signatures/              ← Hydrological signatures
# │   ├── flow_indices.py      ← Q5, Q50, Q95, etc.
# │   └── water_balance.py     ← Runoff ratio, baseflow index
# │
# └── composite/               ← Combined objectives
#     ├── weighted.py          ← WeightedObjective
#     └── factories.py         ← kge_hilo(), fdc_multisegment(), etc.
# ```
#
# ### Design Philosophy
#
# 1. **Composable**: Combine metrics, transforms, and aggregation methods
# 2. **Consistent Interface**: All objectives have `__call__(obs, sim)` → `float`
# 3. **Direction-aware**: Knows whether to maximize or minimize
# 4. **Transform-aware**: Built-in support for flow transformations

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

# Interactive visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("OBJECTIVE FUNCTIONS: DEEP DIVE")
print("=" * 70)

# %%
# Import pyrrm objectives
from pyrrm.objectives import (
    # Traditional metrics
    NSE, RMSE, MAE, PBIAS,
    # KGE family
    KGE, KGENonParametric,
    # Flow transformations
    FlowTransformation,
    # Composite objectives
    WeightedObjective,
    kge_hilo, fdc_multisegment, comprehensive_objective,
    # Utilities
    evaluate_all, print_evaluation_report,
)

# Import calibration tools
from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration import CalibrationRunner
from pyrrm.calibration.objective_functions import calculate_metrics

print("\npyrrm.objectives imported successfully!")
print("\nAvailable metrics:")
print("  Traditional: NSE, RMSE, MAE, PBIAS")
print("  KGE family:  KGE (2009, 2012, 2021), KGENonParametric")
print("  Transforms:  FlowTransformation (sqrt, log, inverse, power, boxcox)")
print("  Composite:   WeightedObjective, kge_hilo, fdc_multisegment")

# %% [markdown]
# ---
# ## Part 1: Traditional Metrics
#
# ### Nash-Sutcliffe Efficiency (NSE)
#
# The most widely used metric in hydrology, introduced by Nash & Sutcliffe (1970).
#
# **Formula:**
# $$\text{NSE} = 1 - \frac{\sum_{i=1}^{n}(Q_{obs,i} - Q_{sim,i})^2}{\sum_{i=1}^{n}(Q_{obs,i} - \bar{Q}_{obs})^2}$$
#
# **Properties:**
# - Range: (-∞, 1]
# - Optimal: 1 (perfect fit)
# - NSE = 0: Model equals the mean
# - NSE < 0: Model worse than mean
#
# **Strengths:**
# - Intuitive interpretation
# - Widely used, comparable across studies
# - Accounts for variance
#
# **Weaknesses:**
# - Dominated by high flows (squared errors)
# - Sensitive to systematic bias
# - Not normalized for different catchments

# %%
# Load example data for demonstrations
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

# Load and prepare data
rainfall_df = pd.read_csv(DATA_DIR / 'Default Input Set - Rain_QBN01.csv', 
                          parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']
pet_df = pd.read_csv(DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
                     parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']
flow_df = pd.read_csv(DATA_DIR / '410734_output_SDmodel.csv',
                      parse_dates=['Date'], index_col='Date')
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df = observed_df.dropna()

data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')
demo_data = data['1995':'1999'].copy()

# Run a simulation with default parameters for demonstration
demo_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
demo_results = demo_model.run(demo_data)
demo_data['sim_default'] = demo_results['runoff'].values

obs = demo_data['observed_flow'].values[365:]  # After warmup
sim_default = demo_data['sim_default'].values[365:]

print("Demo data prepared: 5 years from gauge 410734")
print(f"Comparison period: {len(obs)} days (after 365-day warmup)")

# %%
# Demonstrate NSE
nse = NSE()
nse_value = nse(obs, sim_default)

print("=" * 50)
print("NASH-SUTCLIFFE EFFICIENCY (NSE)")
print("=" * 50)
print(f"\nNSE = {nse_value:.4f}")
print(f"\nInterpretation:")
if nse_value > 0.7:
    print("  → Good model performance")
elif nse_value > 0.5:
    print("  → Acceptable performance")
elif nse_value > 0:
    print("  → Better than mean, but needs improvement")
else:
    print("  → Worse than predicting the mean!")

# %% [markdown]
# ### Kling-Gupta Efficiency (KGE)
#
# A more diagnostic metric that decomposes performance into three components
# (Gupta et al., 2009; Kling et al., 2012).
#
# **Formula:**
# $$\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$
#
# Where:
# - **r** = Pearson correlation (timing)
# - **α** = variability ratio: σ_sim / σ_obs (2009) or CV_sim / CV_obs (2012)
# - **β** = bias ratio: μ_sim / μ_obs
#
# **Properties:**
# - Range: (-∞, 1]
# - Optimal: 1 (all components = 1)
# - KGE > -0.41 → Better than mean benchmark
#
# **Variants:**
# - **2009**: Original formulation
# - **2012**: CV-based variability (recommended)
# - **2021**: Better for near-zero means

# %%
# Demonstrate KGE variants
kge_2009 = KGE(variant='2009')
kge_2012 = KGE(variant='2012')

value_2009 = kge_2009(obs, sim_default)
value_2012 = kge_2012(obs, sim_default)

# Get components for 2012 variant
kge_2012_detailed = KGE(variant='2012')
_, components = kge_2012_detailed.compute_components(obs, sim_default)

print("=" * 50)
print("KLING-GUPTA EFFICIENCY (KGE)")
print("=" * 50)
print(f"\nKGE 2009: {value_2009:.4f}")
print(f"KGE 2012: {value_2012:.4f} (recommended)")
print(f"\nKGE 2012 Components:")
print(f"  Correlation (r): {components['correlation']:.4f}")
print(f"  Variability (α): {components['variability_ratio']:.4f}")
print(f"  Bias (β):        {components['bias_ratio']:.4f}")
print(f"\nInterpretation:")
print(f"  r = 1 → Perfect timing")
print(f"  α = 1 → Same variability")
print(f"  β = 1 → No bias")

# %% [markdown]
# ### Other Traditional Metrics
#
# | Metric | Formula | Range | Optimal | Use Case |
# |--------|---------|-------|---------|----------|
# | **RMSE** | √[Σ(obs-sim)²/n] | [0, ∞) | 0 | Error magnitude |
# | **MAE** | Σ|obs-sim|/n | [0, ∞) | 0 | Robust error |
# | **PBIAS** | 100×Σ(sim-obs)/Σobs | (-∞, ∞) | 0 | Volume bias |

# %%
# Demonstrate other metrics
rmse = RMSE()
mae = MAE()
pbias = PBIAS()

print("=" * 50)
print("OTHER TRADITIONAL METRICS")
print("=" * 50)
print(f"\nRMSE:  {rmse(obs, sim_default):.2f} ML/day  (lower = better)")
print(f"MAE:   {mae(obs, sim_default):.2f} ML/day  (lower = better)")
print(f"PBIAS: {pbias(obs, sim_default):.2f}%       (closer to 0 = better)")

pbias_val = pbias(obs, sim_default)
if pbias_val > 0:
    print(f"\n  → Model OVERESTIMATES by {abs(pbias_val):.1f}%")
else:
    print(f"\n  → Model UNDERESTIMATES by {abs(pbias_val):.1f}%")

# %% [markdown]
# ---
# ## Part 2: Flow Transformations
#
# ### The High-Flow Bias Problem
#
# Most metrics (NSE, RMSE, KGE) are dominated by high flows because:
# - Squared errors give high flows more weight
# - High flows have larger absolute values
#
# **Example:** A 100 ML/day error on a 1000 ML/day flood (10% error) contributes
# more to NSE than a 10 ML/day error on a 20 ML/day baseflow (50% error).
#
# ### Solution: Flow Transformations
#
# By transforming flows before calculating the metric, we can shift emphasis:
#
# | Transform | Formula | Emphasis | Use Case |
# |-----------|---------|----------|----------|
# | **none** | Q | High flows | Flood forecasting |
# | **sqrt** | √Q | Balanced | General purpose |
# | **log** | ln(Q) | Low flows | Baseflow |
# | **inverse** | 1/Q | Very low flows | Low-flow indices |

# %%
# Demonstrate flow transformations
transforms = {
    'none': FlowTransformation('none'),
    'sqrt': FlowTransformation('sqrt'),
    'log': FlowTransformation('log'),
    'inverse': FlowTransformation('inverse'),
}

# Calculate NSE with each transformation
print("=" * 50)
print("NSE WITH DIFFERENT FLOW TRANSFORMATIONS")
print("=" * 50)
print(f"\n{'Transform':<12} {'NSE':>8}  {'Emphasis':<15}")
print("-" * 45)

for name, transform in transforms.items():
    nse_t = NSE(transform=transform)
    value = nse_t(obs, sim_default)
    emphasis = transform.flow_emphasis if hasattr(transform, 'flow_emphasis') else 'N/A'
    print(f"{name:<12} {value:>8.4f}  {emphasis:<15}")

# %%
# Visualize the effect of transformations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original flows
ax = axes[0, 0]
ax.scatter(obs, sim_default, alpha=0.3, s=5)
ax.plot([0, obs.max()], [0, obs.max()], 'r--', lw=2)
ax.set_xlabel('Observed (ML/day)')
ax.set_ylabel('Simulated (ML/day)')
ax.set_title(f'Original (none)\nNSE = {NSE()(obs, sim_default):.3f}')

# Sqrt transform
ax = axes[0, 1]
t = FlowTransformation('sqrt')
obs_t = t.apply(obs, obs)
sim_t = t.apply(sim_default, obs)
ax.scatter(obs_t, sim_t, alpha=0.3, s=5)
ax.plot([0, obs_t.max()], [0, obs_t.max()], 'r--', lw=2)
ax.set_xlabel('Observed (√ML/day)')
ax.set_ylabel('Simulated (√ML/day)')
ax.set_title(f'Square Root (sqrt)\nNSE = {NSE(transform=t)(obs, sim_default):.3f}')

# Log transform
ax = axes[1, 0]
t = FlowTransformation('log')
obs_t = t.apply(obs, obs)
sim_t = t.apply(sim_default, obs)
ax.scatter(obs_t, sim_t, alpha=0.3, s=5)
ax.plot([obs_t.min(), obs_t.max()], [obs_t.min(), obs_t.max()], 'r--', lw=2)
ax.set_xlabel('Observed (ln ML/day)')
ax.set_ylabel('Simulated (ln ML/day)')
ax.set_title(f'Log (log)\nNSE = {NSE(transform=t)(obs, sim_default):.3f}')

# Inverse transform
ax = axes[1, 1]
t = FlowTransformation('inverse')
obs_t = t.apply(obs, obs)
sim_t = t.apply(sim_default, obs)
# Clip for visualization
valid = (obs_t < np.percentile(obs_t, 99)) & (sim_t < np.percentile(sim_t, 99))
ax.scatter(obs_t[valid], sim_t[valid], alpha=0.3, s=5)
ax.plot([0, obs_t[valid].max()], [0, obs_t[valid].max()], 'r--', lw=2)
ax.set_xlabel('Observed (1/ML/day)')
ax.set_ylabel('Simulated (1/ML/day)')
ax.set_title(f'Inverse (inverse)\nNSE = {NSE(transform=t)(obs, sim_default):.3f}')

plt.suptitle('Effect of Flow Transformations on Model Evaluation', y=1.02)
plt.tight_layout()
plt.savefig('figures/03_flow_transformations.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: figures/03_flow_transformations.png")

# %% [markdown]
# #### Key Observation
#
# Notice how the NSE values differ dramatically across transformations!
# - High NSE (none) → Model captures high flows well
# - Lower NSE (log/inverse) → Model struggles with low flows
#
# This is exactly why objective function choice matters for calibration.

# %% [markdown]
# ---
# ## Part 3: Composite Objectives
#
# ### Why Combine Metrics?
#
# Single metrics often have blind spots:
# - NSE ignores low flows
# - LogNSE may over-emphasize low flows
# - KGE components may not all improve together
#
# **Solution:** Combine multiple metrics into a weighted objective.
#
# ### WeightedObjective
#
# The `WeightedObjective` class combines multiple metrics with weights:
#
# ```python
# WeightedObjective([
#     (NSE(), 0.5),                    # 50% weight on NSE
#     (KGE(transform=inverse), 0.5),   # 50% weight on KGE-inverse
# ])
# ```

# %%
# Demonstrate composite objectives
print("=" * 50)
print("COMPOSITE OBJECTIVES")
print("=" * 50)

# Manual construction
nse_obj = NSE()
kge_inv = KGE(transform=FlowTransformation('inverse'))

composite = WeightedObjective([
    (nse_obj, 0.5),
    (kge_inv, 0.5),
])

print("\n1. Manual WeightedObjective:")
print(f"   Components: NSE (50%) + KGE-inverse (50%)")
print(f"   Value: {composite(obs, sim_default):.4f}")

# %% [markdown]
# ### Factory Functions
#
# `pyrrm.objectives` provides convenient factory functions for common combinations:
#
# | Factory | Description |
# |---------|-------------|
# | `kge_hilo()` | KGE + KGE(inverse) for balanced high/low flow |
# | `fdc_multisegment()` | Multi-segment FDC evaluation |
# | `comprehensive_objective()` | Full multi-metric combination |

# %%
# Demonstrate factory functions
print("\n2. Factory Functions:")

# kge_hilo - balances high and low flow performance
hilo = kge_hilo(kge_weight=0.5)
print(f"\n   kge_hilo(kge_weight=0.5):")
print(f"   Combines: KGE (high flows) + KGE-inverse (low flows)")
print(f"   Value: {hilo(obs, sim_default):.4f}")

# comprehensive_objective - full multi-metric
comprehensive = comprehensive_objective()
print(f"\n   comprehensive_objective():")
print(f"   Combines: KGE + KGE(sqrt) + KGE(inverse) + PBIAS penalty")
print(f"   Value: {comprehensive(obs, sim_default):.4f}")

# %% [markdown]
# ---
# ## Part 4: Calibration Experiment
#
# Now let's see how different objective functions lead to different calibration
# results. We'll calibrate the Sacramento model using four different objectives
# and compare the results.
#
# **Objectives to compare:**
# 1. **NSE** - Traditional, emphasizes high flows
# 2. **KGE** - Modern standard, balanced components
# 3. **KGE(inverse)** - Low-flow emphasis
# 4. **kge_hilo** - Balanced high and low flow
#
# **Algorithm:** We use SCE-UA for all to ensure differences come from the
# objective function, not the algorithm.

# %%
# Prepare calibration data
CAL_START = pd.Timestamp('1990-01-01')
CAL_END = pd.Timestamp('1994-12-31')  # 5 years for faster demo
WARMUP_DAYS = 365

cal_data = data[(data.index >= CAL_START) & (data.index <= CAL_END)].copy()
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

print("=" * 70)
print("CALIBRATION EXPERIMENT SETUP")
print("=" * 70)
print(f"\nCalibration period: {CAL_START.date()} to {CAL_END.date()}")
print(f"Records: {len(cal_data)}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_data) - WARMUP_DAYS} days")
print(f"\nObjectives to compare:")
print("  1. NSE (traditional)")
print("  2. KGE (modern standard)")
print("  3. KGE with inverse transform (low-flow emphasis)")
print("  4. kge_hilo (balanced)")

# %%
# Define objectives for calibration
from pyrrm.calibration.objective_functions import NSE as CalibNSE, KGE as CalibKGE

calibration_objectives = {
    'NSE': CalibNSE(),
    'KGE': CalibKGE(),
    'KGE_inverse': CalibKGE(transform='inverse'),
}

# Note: kge_hilo requires wrapping for calibration compatibility
# For simplicity, we'll compare the first three

print("\nObjectives configured for calibration:")
for name, obj in calibration_objectives.items():
    print(f"  {name}: maximize={obj.maximize}")

# %%
# Run calibrations (this takes some time)
print("\n" + "=" * 70)
print("RUNNING CALIBRATIONS")
print("=" * 70)
print("\nThis will take several minutes. Each objective runs separately.")
print("For faster results, reduce n_iterations.\n")

calibration_results = {}

# SCE-UA configuration (Duan et al., 1994)
# Rule: ngs = 2*n + 1 where n = number of parameters
# Sacramento has 22 parameters
N_PARAMS = 22
NGS = 2 * N_PARAMS + 1  # = 45
N_ITERATIONS = 10000

print(f"SCE-UA Configuration: ngs={NGS} (2×{N_PARAMS}+1), iterations={N_ITERATIONS}")

for obj_name, objective in calibration_objectives.items():
    print(f"\n{'='*60}")
    print(f"Calibrating with: {obj_name}")
    print(f"{'='*60}")
    
    runner = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=objective,
        warmup_period=WARMUP_DAYS
    )
    
    result = runner.run_sceua_direct(
        max_evals=N_ITERATIONS,
        n_complexes=NGS,
        seed=42,
    )
    
    calibration_results[obj_name] = result
    print(f"\n✓ {obj_name} complete: best = {result.best_objective:.4f}")

print("\n" + "=" * 70)
print("ALL CALIBRATIONS COMPLETE")
print("=" * 70)

# %% [markdown]
# ### Comparing Calibration Results
#
# Now let's analyze how the different objectives affected the calibrated parameters
# and model performance.

# %%
# Compare best parameters
print("=" * 70)
print("PARAMETER COMPARISON")
print("=" * 70)

param_comparison = {}
for obj_name, result in calibration_results.items():
    param_comparison[obj_name] = result.best_parameters

param_df = pd.DataFrame(param_comparison)
print("\nBest parameters by objective:")
print(param_df.round(4).to_string())

# Highlight key differences
print("\nKey parameter differences (selected parameters):")
key_params = ['uztwm', 'lztwm', 'uzk', 'lzpk']
print(param_df.loc[key_params].round(4).to_string())

# %%
# Generate simulations with each calibrated parameter set
simulations = {}
for obj_name, result in calibration_results.items():
    model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model.set_parameters(result.best_parameters)
    model.reset()
    sim_results = model.run(cal_data)
    simulations[obj_name] = sim_results['runoff'].values[WARMUP_DAYS:]

obs_compare = cal_observed[WARMUP_DAYS:]
dates_compare = cal_data.index[WARMUP_DAYS:]

print("Simulations generated for all calibrated parameter sets.")

# %%
# Calculate comprehensive metrics for each
print("=" * 70)
print("PERFORMANCE METRICS COMPARISON")
print("=" * 70)

metrics_comparison = {}
for obj_name, sim in simulations.items():
    metrics = calculate_metrics(sim, obs_compare)
    metrics_comparison[obj_name] = metrics

metrics_df = pd.DataFrame(metrics_comparison).T
print("\nPerformance metrics by calibration objective:")
print(metrics_df.round(4).to_string())

# %%
# Visualize the comparison
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Hydrograph Comparison (Log Scale)',
        'NSE by Objective',
        'Flow Duration Curves',
        'Low Flow Performance (NSE-inverse)'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

colors = {'NSE': 'blue', 'KGE': 'green', 'KGE_inverse': 'red'}

# 1. Hydrograph comparison
fig.add_trace(
    go.Scatter(x=dates_compare, y=obs_compare, name='Observed',
               line=dict(color='black', width=1)),
    row=1, col=1
)
for obj_name, sim in simulations.items():
    fig.add_trace(
        go.Scatter(x=dates_compare, y=sim, name=obj_name,
                   line=dict(color=colors[obj_name], width=1), opacity=0.7),
        row=1, col=1
    )

# 2. NSE comparison bar chart
nse_values = [metrics_comparison[obj]['NSE'] for obj in calibration_results.keys()]
fig.add_trace(
    go.Bar(x=list(calibration_results.keys()), y=nse_values,
           marker_color=[colors[obj] for obj in calibration_results.keys()],
           showlegend=False),
    row=1, col=2
)

# 3. Flow Duration Curves
exc = np.arange(1, len(obs_compare) + 1) / len(obs_compare) * 100
obs_sorted = np.sort(obs_compare)[::-1]
fig.add_trace(
    go.Scatter(x=exc, y=obs_sorted, name='Observed FDC',
               line=dict(color='black', width=2), showlegend=False),
    row=2, col=1
)
for obj_name, sim in simulations.items():
    sim_sorted = np.sort(sim)[::-1]
    fig.add_trace(
        go.Scatter(x=exc, y=sim_sorted, name=f'{obj_name} FDC',
                   line=dict(color=colors[obj_name], width=1.5), showlegend=False),
        row=2, col=1
    )

# 4. Low flow performance (NSE on inverse-transformed flows)
nse_inv = NSE(transform=FlowTransformation('inverse'))
inv_values = [nse_inv(obs_compare, sim) for sim in simulations.values()]
fig.add_trace(
    go.Bar(x=list(calibration_results.keys()), y=inv_values,
           marker_color=[colors[obj] for obj in calibration_results.keys()],
           showlegend=False),
    row=2, col=2
)

# Update layout
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig.update_yaxes(title_text="NSE", row=1, col=2)
fig.update_xaxes(title_text="Exceedance %", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=1)
fig.update_yaxes(title_text="NSE (inverse)", row=2, col=2)

fig.update_layout(
    title="<b>Calibration Results by Objective Function</b>",
    height=800,
    showlegend=True,
    legend=dict(orientation='h', y=1.02)
)
fig.show()

# %%
# Save comparison figure
plt.figure(figsize=(14, 10))

# Recreate key plots with matplotlib for saving
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Hydrograph (zoomed to show differences)
ax = axes[0, 0]
zoom_start = 365
zoom_end = 500
ax.plot(dates_compare[zoom_start:zoom_end], obs_compare[zoom_start:zoom_end], 
        'k-', lw=1.5, label='Observed')
for obj_name, sim in simulations.items():
    ax.plot(dates_compare[zoom_start:zoom_end], sim[zoom_start:zoom_end],
            color=colors[obj_name], lw=1, alpha=0.8, label=obj_name)
ax.set_ylabel('Flow (ML/day)')
ax.set_title('Hydrograph Comparison (Zoomed)')
ax.legend()
ax.set_yscale('log')

# 2. NSE comparison
ax = axes[0, 1]
ax.bar(list(calibration_results.keys()), nse_values,
       color=[colors[obj] for obj in calibration_results.keys()])
ax.set_ylabel('NSE')
ax.set_title('NSE by Calibration Objective')
ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax.legend()

# 3. FDC
ax = axes[1, 0]
ax.plot(exc, obs_sorted, 'k-', lw=2, label='Observed')
for obj_name, sim in simulations.items():
    sim_sorted = np.sort(sim)[::-1]
    ax.plot(exc, sim_sorted, color=colors[obj_name], lw=1.5, label=obj_name)
ax.set_xlabel('Exceedance (%)')
ax.set_ylabel('Flow (ML/day)')
ax.set_title('Flow Duration Curves')
ax.set_yscale('log')
ax.legend()

# 4. Low flow NSE
ax = axes[1, 1]
ax.bar(list(calibration_results.keys()), inv_values,
       color=[colors[obj] for obj in calibration_results.keys()])
ax.set_ylabel('NSE (inverse-transformed)')
ax.set_title('Low Flow Performance')

plt.suptitle('Impact of Objective Function on Calibration Results', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('figures/03_objective_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/03_objective_comparison.png")

# %% [markdown]
# ### Key Observations
#
# 1. **NSE calibration** → Best overall NSE, but may sacrifice low flows
# 2. **KGE calibration** → Balanced performance, good for general use
# 3. **KGE_inverse calibration** → Best low-flow performance, may sacrifice peaks
#
# **The objective you choose determines what the model learns to do well!**

# %% [markdown]
# ---
# ## Part 5: Practical Guidance
#
# ### Decision Tree: Which Objective Should I Use?
#
# ```
# START
#   │
#   ▼
# What's your primary application?
#   │
#   ├─► Flood forecasting
#   │     └─► Use NSE or KGE (emphasize peaks)
#   │
#   ├─► Environmental flows / Low flow analysis
#   │     └─► Use KGE(inverse) or NSE(log) (emphasize baseflow)
#   │
#   ├─► Water supply planning
#   │     └─► Use kge_hilo() or comprehensive_objective()
#   │           (balanced across flow regimes)
#   │
#   └─► General purpose / Uncertain application
#         └─► Use KGE (2012) - modern standard with
#               interpretable components
# ```
#
# ### Common Pitfalls
#
# 1. **Using only NSE** → Ignores low-flow performance
# 2. **Log transform with KGE** → Numerical issues (use inverse instead)
# 3. **Overfitting** → Using too complex objective can lead to overfitting
# 4. **Not validating** → Always check performance on independent period

# %%
# Summary table
print("=" * 70)
print("SUMMARY: OBJECTIVE FUNCTION SELECTION GUIDE")
print("=" * 70)

summary_data = {
    'Objective': ['NSE', 'KGE', 'NSE(sqrt)', 'KGE(inverse)', 'kge_hilo()', 'comprehensive()'],
    'Emphasis': ['High flows', 'Balanced (components)', 'Balanced', 'Low flows', 'High + Low', 'Multi-aspect'],
    'Use Case': ['Floods', 'General', 'General', 'Env. flows', 'Water supply', 'Research'],
    'Pros': ['Simple, standard', 'Diagnostic', 'Balanced', 'Low flow fit', 'Balanced', 'Complete'],
    'Cons': ['Ignores low flows', 'Complex', 'Less common', 'May miss peaks', 'Slower', 'Complex'],
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# %% [markdown]
# ---
# ## Summary
#
# ### Key Takeaways
#
# 1. **Objective function choice profoundly affects calibration results**
#    - Different objectives lead to different "optimal" parameters
#    - No single "best" objective - depends on your application
#
# 2. **Flow transformations shift emphasis between high and low flows**
#    - sqrt: Balanced
#    - log/inverse: Low-flow emphasis
#    - none: High-flow emphasis
#
# 3. **Composite objectives balance multiple aspects**
#    - `kge_hilo()` for high and low flow balance
#    - `comprehensive_objective()` for full evaluation
#
# 4. **Always validate on independent data**
#    - Different objectives may generalize differently
#
# ### Next Steps
#
# - **Notebook 06**: Compare calibration algorithms (DREAM, PyDREAM, SCE-UA)
# - **Notebook 08**: Monitor long-running calibrations

# %%
print("=" * 70)
print("OBJECTIVE FUNCTIONS TUTORIAL COMPLETE")
print("=" * 70)
print("""
You now understand:
  ✓ Traditional metrics (NSE, KGE, RMSE, PBIAS)
  ✓ Flow transformations and their effects
  ✓ Composite objective construction
  ✓ How objective choice affects calibration
  
Next: Explore algorithm comparison in Notebook 06!
""")
