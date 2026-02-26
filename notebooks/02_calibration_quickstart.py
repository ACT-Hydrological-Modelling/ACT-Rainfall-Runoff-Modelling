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
# # Getting Started with pyrrm: Your First Model Calibration
#
# ## Purpose
#
# This notebook provides a beginner-friendly introduction to rainfall-runoff
# modeling and calibration using the `pyrrm` library. By the end, you'll have
# successfully calibrated a hydrological model to match observed streamflow data.
#
# ## What You'll Learn
#
# - What rainfall-runoff modeling is and why it's useful
# - How to load and explore hydrological data
# - How to set up and configure the Sacramento model
# - How to calibrate model parameters to match observed streamflow
# - How to evaluate and interpret model performance
#
# ## Prerequisites
#
# - Basic Python/pandas knowledge (DataFrames, plotting)
# - Conceptual understanding of "what goes in, what comes out" for a catchment
# - No prior hydrology experience required!
#
# ## Estimated Time
#
# - ~15 minutes to work through the concepts
# - ~30-60 minutes for calibration to run (can be shortened)
#
# ## Saving Calibration Results
#
# All calibrations in this notebook are automatically saved using `CalibrationReport` objects.
# These reports are stored as pickle files in `test_data/02_calibration_quickstart/reports/` and include:
#
# - **Complete calibration results** (best parameters, all samples, diagnostics)
# - **Input and observed data** for visualization and re-simulation
# - **Model configuration** for reproducing results
#
# After running this notebook, you can load any saved calibration:
#
# ```python
# from pyrrm.calibration import CalibrationReport
# report = CalibrationReport.load('test_data/02_calibration_quickstart/reports/410734_sacramento_nse_sceua.pkl')
# report.plot_report_card()  # Generate comprehensive visualization
# ```
#
# Saved reports can also be used to generate interactive HTML dashboards:
#
# ```python
# fig = report.plot_report_card_interactive()
# fig.write_html('my_report.html')
# ```

# %% [markdown]
# ---
# ## What is Rainfall-Runoff Modeling?
#
# ### The Big Picture
#
# When rain falls on a catchment (the land area draining to a river), some of it:
# - **Evaporates** back to the atmosphere
# - **Infiltrates** into the soil
# - **Runs off** over the surface
# - **Percolates** deep into groundwater
#
# Eventually, some of this water reaches the river as **streamflow**.
#
# A **rainfall-runoff model** is a mathematical representation of these processes.
# Given rainfall and evaporation inputs, it predicts how much water will flow
# in the river.
#
# ```
# ┌─────────────────────────────────────────────────────────────────────┐
# │                        THE CATCHMENT                                │
# │                                                                     │
# │    ☁️ Rainfall (mm/day)      ☀️ Evaporation (mm/day)                 │
# │         │                         ↑                                 │
# │         ▼                         │                                 │
# │    ┌─────────────────────────────────────────────────┐              │
# │    │                                                 │              │
# │    │          RAINFALL-RUNOFF MODEL                  │              │
# │    │                                                 │              │
# │    │    • Soil moisture accounting                   │              │
# │    │    • Percolation to groundwater                 │              │
# │    │    • Surface runoff generation                  │              │
# │    │    • Baseflow from aquifers                     │              │
# │    │                                                 │              │
# │    └─────────────────────────────────────────────────┘              │
# │                        │                                            │
# │                        ▼                                            │
# │               🌊 Streamflow (ML/day)                                │
# │                                                                     │
# └─────────────────────────────────────────────────────────────────────┘
# ```
#
# ### Why Model Streamflow?
#
# - **Water supply planning**: How much water will be available?
# - **Flood forecasting**: When will the river peak?
# - **Environmental flows**: What releases maintain ecosystem health?
# - **Climate change**: How will future rainfall patterns affect water availability?

# %% [markdown]
# ### The Calibration Problem
#
# Models have **parameters** - numbers that control their behavior. For example:
# - How much water can the soil hold?
# - How fast does groundwater drain?
# - What fraction of rain becomes runoff?
#
# **Calibration** is the process of finding parameter values that make the model
# output match observed reality as closely as possible.
#
# ```
# ┌──────────────────────────────────────────────────────────────────────────┐
# │                                                                          │
# │   INPUTS                  MODEL                        OUTPUTS           │
# │   ──────                  ─────                        ───────           │
# │                                                                          │
# │   Rainfall ───────┐                                                      │
# │                   │      ┌────────────────┐                              │
# │                   ├─────►│                │                              │
# │                   │      │   Sacramento   │        Simulated             │
# │   Evaporation ────┤      │     Model      ├─────► Streamflow             │
# │                   │      │                │                              │
# │                   │      │  Parameters:   │                              │
# │                   │      │  (20+ values)  │                              │
# │                   │      └────────────────┘                              │
# │                                 ▲                                        │
# │                                 │                                        │
# │                          CALIBRATION                                     │
# │                          ───────────                                     │
# │                          Adjust parameters                               │
# │                          to minimize error         Observed              │
# │                                 │                  Streamflow            │
# │                                 │                      │                 │
# │                                 └──────────────────────┘                 │
# │                                       Compare                            │
# │                                                                          │
# └──────────────────────────────────────────────────────────────────────────┘
# ```
#
# With 20+ parameters, manually adjusting them would take forever. **Automated
# calibration algorithms** search the parameter space efficiently to find good
# values.

# %% [markdown]
# ---
# ## The pyrrm Library Overview
#
# `pyrrm` (Python Rainfall-Runoff Models) provides everything you need for
# rainfall-runoff modeling and calibration:
#
# ```
# pyrrm/
# ├── models/              ← Rainfall-runoff models
# │   ├── sacramento.py    ← Sacramento model (we'll use this)
# │   ├── gr4j.py          ← GR4J model (French, 4 parameters)
# │   └── ...
# │
# ├── calibration/         ← Optimization algorithms
# │   ├── runner.py        ← CalibrationRunner (unified interface)
# │   └── ...              ← SCE-UA, PyDREAM, SciPy, NumPyro adapters
# │
# ├── objectives/          ← How to measure "goodness of fit"
# │   ├── metrics/         ← NSE, KGE, RMSE, etc.
# │   └── ...
# │
# └── visualization/       ← Plotting utilities
# ```
#
# ### Today's Workflow
#
# 1. **Load data** (rainfall, evaporation, observed flow)
# 2. **Create model** (Sacramento)
# 3. **Set up calibration** (CalibrationRunner + objective function)
# 4. **Run calibration** (algorithm searches for best parameters)
# 5. **Evaluate results** (compare simulated vs observed)

# %% [markdown]
# ---
# ## Setup
#
# Let's load the necessary libraries and set up our environment.

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

OUTPUT_DIR = Path('../test_data/02_calibration_quickstart')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PYRRM CALIBRATION QUICKSTART")
print("=" * 70)
print("\nLibraries loaded successfully!")

# %%
# Import pyrrm components
from pyrrm.models.sacramento import Sacramento
from pyrrm.models import NUMBA_AVAILABLE
from pyrrm.calibration import (
    CalibrationRunner,
    PYDREAM_AVAILABLE,
    NUMPYRO_AVAILABLE,
)

# Reload modules to pick up any code changes (useful during development)
import importlib
import pyrrm.visualization.report_plots
import pyrrm.calibration.report
importlib.reload(pyrrm.visualization.report_plots)
importlib.reload(pyrrm.calibration.report)
from pyrrm.calibration.objective_functions import NSE, KGE
from pyrrm.analysis.diagnostics import compute_diagnostics, print_diagnostics, DIAGNOSTIC_GROUPS

print("\npyrrm components imported:")
print(f"  - Sacramento model")
print(f"  - CalibrationRunner")
print(f"  - Objective functions (NSE, KGE)")
print(f"\nModel acceleration:")
print(f"  - Numba JIT: {'ACTIVE' if NUMBA_AVAILABLE else 'not available (pip install numba)'}")
print(f"\nAvailable calibration backends:")
print(f"  - SCE-UA (vendored): always available")
print(f"  - SciPy (DE, Dual Annealing): always available")
print(f"  - PyDREAM (MT-DREAM(ZS)): {PYDREAM_AVAILABLE}")
print(f"  - NumPyro NUTS: {NUMPYRO_AVAILABLE}")

# %% [markdown]
# ---
# ## Step 1: Load and Explore the Data
#
# We'll use data from **Gauge 410734** on the Queanbeyan River
# (NSW/ACT border, Australia). This catchment has:
#
# - ~40 years of daily data (1985-2024)
# - Area: 517 km²
# - Mixed land use: forests, agriculture, urban
#
# ### What Data Do We Need?
#
# | Data | Units | Description |
# |------|-------|-------------|
# | **Rainfall** | mm/day | Gridded rainfall over the catchment |
# | **PET** | mm/day | Potential evapotranspiration |
# | **Observed Flow** | ML/day | Recorded streamflow at the gauge |
#
# **Note**: PET = Potential Evapotranspiration, representing the maximum
# evaporation if water were unlimited. It's driven by temperature, humidity,
# wind, and solar radiation.

# %%
from pyrrm.data import load_catchment_data

DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

inputs, observed = load_catchment_data(
    precipitation_file=DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
    pet_file=DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
    observed_file=DATA_DIR / '410734_output_SDmodel.csv',
    observed_value_column='Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)',
)

# Build a combined DataFrame for visualization
data = inputs.copy()
data['observed_flow'] = observed

print("MERGED DATASET")
print("=" * 50)
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")
print(f"Total records: {len(data):,} days")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
print(f"Columns: {list(data.columns)}")
print(f"\nMissing values:")
print(data.isna().sum())

# %% [markdown]
# ### Visualizing the Data
#
# Before modeling, it's always good to look at your data!
# Let's create an interactive plot to explore the time series.

# %%
# Interactive visualization of input data
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,  # Links x-axes for synchronized zooming
    vertical_spacing=0.04,
    subplot_titles=(
        'Rainfall (mm/day)', 
        'PET (mm/day)', 
        'Observed Flow - Linear Scale (ML/day)',
        'Observed Flow - Log Scale (ML/day)'
    )
)

# Rainfall
fig.add_trace(
    go.Scatter(x=data.index, y=data['precipitation'], name='Rainfall',
               fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.6)',
               line=dict(color='steelblue', width=0.5)),
    row=1, col=1
)

# PET
fig.add_trace(
    go.Scatter(x=data.index, y=data['pet'], name='PET',
               line=dict(color='orange', width=0.8)),
    row=2, col=1
)

# Observed flow - Linear scale (shows peaks)
fig.add_trace(
    go.Scatter(x=data.index, y=data['observed_flow'], name='Flow (linear)',
               line=dict(color='darkblue', width=0.8)),
    row=3, col=1
)

# Observed flow - Log scale (shows low flows)
fig.add_trace(
    go.Scatter(x=data.index, y=data['observed_flow'], name='Flow (log)',
               line=dict(color='darkblue', width=0.8), showlegend=False),
    row=4, col=1
)

# Update axes
fig.update_yaxes(title_text="Rain (mm)", autorange="reversed", row=1, col=1)
fig.update_yaxes(title_text="PET (mm)", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/d)", row=3, col=1)
fig.update_yaxes(title_text="Flow (ML/d)", type="log", row=4, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)

fig.update_layout(
    title="<b>Input Data: Gauge 410734 - Queanbeyan Catchment</b><br>" +
          "<sup>Zoom on any subplot to synchronize all views</sup>",
    height=850,
    showlegend=True,
    legend=dict(orientation='h', y=1.02)
)

fig.show()
print("Interactive plot displayed - zoom on any subplot and all will follow!")

# %% [markdown]
# #### What to Look For
#
# - **Rainfall-runoff response**: Do flow peaks follow rainfall events?
# - **Seasonality**: Higher flows in winter/spring, lower in summer/autumn?
# - **Baseflow**: The persistent flow between rainfall events (groundwater contribution)
# - **Data quality**: Any suspicious spikes or flat periods?

# %% [markdown]
# ---
# ## Step 2: Set Up the Model
#
# We'll use the **Sacramento model**, a well-established rainfall-runoff model
# developed by the US National Weather Service. It has 22 parameters that
# control soil moisture storage, drainage rates, and runoff generation.
#
# ### Creating the Model Instance
#
# We pass the `catchment_area_km2` so the model can automatically convert
# outputs from mm/day to ML/day (to match our observed flow units).

# %%
# Create the Sacramento model
model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)

print("SACRAMENTO MODEL")
print("=" * 50)
print(f"Catchment area: {model.catchment_area_km2} km²")
print(f"Output units: {model.output_units}")

# Show the parameters and their bounds
param_bounds = model.get_parameter_bounds()
print(f"\nCalibration parameters: {len(param_bounds)}")
print("\nParameter bounds (search space for calibration):")
print("-" * 50)
for name, (low, high) in param_bounds.items():
    print(f"  {name:8s}: [{low:8.3f}, {high:8.3f}]")

# %% [markdown]
# #### Understanding the Parameter Bounds
#
# These bounds define the **search space** for calibration. The algorithm
# will try different combinations of values within these ranges to find
# the best fit.
#
# Key parameter groups:
# - **Storage capacities** (UZTWM, UZFWM, LZTWM, etc.): How much water the soil can hold
# - **Drainage rates** (UZK, LZPK, LZSK): How fast water drains from each zone
# - **Percolation** (ZPERC, REXP): How water moves from upper to lower zones
# - **Unit hydrograph** (UH1-UH5): How surface runoff is delayed reaching the outlet

# %% [markdown]
# ---
# ## Step 3: Choose an Objective Function
#
# An **objective function** measures how well the simulated flow matches the
# observed flow. The calibration algorithm will try to maximize (or minimize)
# this value.
#
# ### Common Objective Functions
#
# | Metric | Range | Perfect | Description |
# |--------|-------|---------|-------------|
# | **NSE** | -∞ to 1 | 1 | Nash-Sutcliffe Efficiency |
# | **KGE** | -∞ to 1 | 1 | Kling-Gupta Efficiency |
# | **RMSE** | 0 to ∞ | 0 | Root Mean Square Error |
#
# We'll use **NSE (Nash-Sutcliffe Efficiency)** - the most widely used metric
# in hydrology.
#
# ### What Does NSE Mean?
#
# - **NSE = 1**: Perfect match (impossible in practice)
# - **NSE > 0.7**: Good model performance
# - **NSE > 0.5**: Acceptable performance
# - **NSE = 0**: Model is as good as using the mean observed flow
# - **NSE < 0**: Model is worse than just predicting the mean!

# %%
# Create the objective function
objective = NSE()

print("OBJECTIVE FUNCTION")
print("=" * 50)
print(f"Name: {objective.name}")
print(f"Maximize: {objective.maximize}")
print("\nInterpretation:")
print("  NSE = 1   → Perfect (impossible)")
print("  NSE > 0.7 → Good")
print("  NSE > 0.5 → Acceptable")
print("  NSE = 0   → Same as mean")
print("  NSE < 0   → Worse than mean")

# %% [markdown]
# **Note**: There are many other objective functions available in `pyrrm.objectives`.
# See **Notebook 04: Objective Functions** for a detailed exploration of
# different metrics and how they affect calibration results.
#
# ### The KGE Family
#
# The **Kling-Gupta Efficiency (KGE)** is an increasingly popular alternative to NSE.
# KGE decomposes model performance into three interpretable components:
#
# - **r** (correlation): Timing accuracy
# - **α** (variability): How well the model reproduces flow variability
# - **β** (bias): Volume balance
#
# | Variant | Description | Best For |
# |---------|-------------|----------|
# | **KGE (2009)** | Original formulation | General use |
# | **KGE (2012)** | Modified variability ratio (CV-based) | Default recommendation |
# | **KGE (2021)** | Alternative bias formulation | Near-zero means |
# | **KGE Non-parametric** | Uses Spearman correlation + FDC variability | Skewed distributions |
#
# Like NSE, KGE can be combined with **flow transformations** to emphasize different flow regimes.

# %% [markdown]
# ---
# ## Step 4: Configure the Calibration
#
# Now we bring everything together using `CalibrationRunner`. This class
# provides a unified interface to different optimization algorithms.
#
# ### Defining Calibration and Warmup Periods
#
# - **Warmup period**: Initial days to "spin up" the model (storage values stabilize)
# - **Calibration period**: Days used to optimize parameters
#
# We use 1 year of warmup to ensure the model's internal states are reasonable
# before we start comparing against observed data.

# %%
# Define periods
WARMUP_DAYS = 365  # 1 year warmup

# Use the full observed data period for calibration
# This gives the most robust parameter estimates
CAL_START = data.index.min()
CAL_END = data.index.max()

# Use full dataset for calibration
cal_data = data.copy()

print("CALIBRATION PERIOD")
print("=" * 50)
print(f"Start: {CAL_START.date()}")
print(f"End: {CAL_END.date()}")
print(f"Total days: {len(cal_data):,}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_data) - WARMUP_DAYS:,} days")

# Prepare inputs and observed data
cal_inputs = cal_data[['precipitation', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

# %%
# Create the CalibrationRunner
runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=objective,
    warmup_period=WARMUP_DAYS
)

print("\nCALIBRATION RUNNER CONFIGURED")
print("=" * 50)
print(f"Model: Sacramento")
print(f"Objective: {objective.name}")
print(f"Input shape: {cal_inputs.shape}")
print(f"Observed shape: {cal_observed.shape}")
print(f"Warmup period: {WARMUP_DAYS} days")

# %% [markdown]
# ---
# ## Step 5: Run the Calibration!
#
# Now we run the optimization algorithm to find the best parameters.
#
# ### Choosing an Algorithm
#
# pyrrm supports several calibration algorithms:
#
# | Algorithm | Type | Speed | Uncertainty? | Best For |
# |-----------|------|-------|--------------|----------|
# | **SCE-UA Direct** | Optimization | Fast | No | Quick calibration (recommended) |
# | **SciPy DE** | Optimization | Fast | No | Simple cases |
# | **PyDREAM / NumPyro NUTS** | MCMC | Slow | Yes | Full uncertainty |
# | **PyDREAM** | MCMC | Slow | Yes | Complex problems |
#
# For this quickstart, we'll use **SCE-UA Direct** - a vendored implementation
# of the Shuffled Complex Evolution algorithm. This is a classic hydrology
# algorithm that's fast, reliable, and requires no external dependencies.
#
# **Note**: See **Notebook 06: Algorithm Comparison** for a detailed
# comparison of all available algorithms.

# %%
# Run SCE-UA calibration
print("=" * 70)
print("RUNNING SCE-UA CALIBRATION")
print("=" * 70)
print(f"\nObjective: {objective.name}")
print(f"Algorithm: SCE-UA Direct (Shuffled Complex Evolution)")
print("\nThis may take a few minutes...")
print("Progress will be shown below.\n")

# SCE-UA settings
# - n_complexes: Number of complexes (auto-determined if None)
# - max_evals: Maximum function evaluations
n_params = len(runner._param_bounds)
max_evals = 10000

print(f"SCE-UA Configuration: max_evals={max_evals}, n_params={n_params}")

result = runner.run_sceua_direct(
    max_evals=max_evals,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,  # Maximum iterations without improvement
    tolerance=1e-4          # More lenient improvement threshold
)

print("\n" + result.summary())

# Save calibration report
from pyrrm.calibration import CalibrationReport
nse_report = runner.create_report(result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
nse_report.save(REPORTS_DIR / '410734_sacramento_nse_sceua')
print(f"\nCalibration saved to: {REPORTS_DIR / '410734_sacramento_nse_sceua.pkl'}")

# %% [markdown]
# ### Understanding the Output
#
# The calibration returns a `CalibrationResult` object containing:
#
# - **best_parameters**: The optimal parameter values found
# - **best_objective**: The best NSE achieved
# - **runtime_seconds**: How long calibration took
# - **all_samples**: Complete history of all tried parameter sets

# %%
# Display best parameters
print("=" * 50)
print("BEST PARAMETERS FOUND")
print("=" * 50)
print(f"\nBest NSE: {result.best_objective:.4f}")
print(f"Runtime: {result.runtime_seconds:.1f} seconds")
print("\nOptimal parameter values:")
print("-" * 50)
for name, value in result.best_parameters.items():
    bounds = param_bounds[name]
    # Show where the parameter is relative to its bounds
    pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
    bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
    print(f"  {name:8s}: {value:10.4f}  [{bar}]")

# %% [markdown]
# ---
# ## Step 6: Evaluate the Results
#
# Now let's see how well our calibrated model performs!
#
# ### Running the Calibrated Model

# %%
# Create model with calibrated parameters
calibrated_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
calibrated_model.set_parameters(result.best_parameters)
calibrated_model.reset()

# Run simulation
sim_results = calibrated_model.run(cal_data)
cal_data['simulated_flow'] = sim_results['runoff'].values

# Extract comparison period (after warmup)
comparison = cal_data.iloc[WARMUP_DAYS:].copy()
sim_flow = comparison['simulated_flow'].values
obs_flow = comparison['observed_flow'].values

print("Simulation complete!")
print(f"Comparison period: {len(comparison):,} days (after warmup)")

# %% [markdown]
# ### Performance Metrics
#
# Let's calculate multiple metrics to get a complete picture of model performance.

# %%
# Compute the canonical diagnostic suite (48 metrics)
from pyrrm.analysis.diagnostics import compute_diagnostics, print_diagnostics

metrics = compute_diagnostics(sim_flow, obs_flow)
print_diagnostics(metrics, label="Sacramento -- NSE calibration")

# %% [markdown]
# ### Visualizing the Results
#
# The best way to evaluate a model is to look at the hydrograph!

# %%
# Create comprehensive diagnostic plots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Hydrograph (Log Scale)',
        'Simulated vs Observed (Scatter)',
        'Hydrograph (Linear Scale)',
        'Flow Duration Curves',
        'Residuals Over Time',
        'Residual Distribution'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "histogram"}]
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# 1. Hydrograph - Log scale (good for seeing low flows)
fig.add_trace(
    go.Scatter(x=comparison.index, y=obs_flow, name='Observed',
               line=dict(color='blue', width=1)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=comparison.index, y=sim_flow, name='Simulated',
               line=dict(color='red', width=1)),
    row=1, col=1
)

# 2. Scatter plot (1:1 comparison)
max_flow = max(obs_flow.max(), sim_flow.max())
fig.add_trace(
    go.Scatter(x=obs_flow, y=sim_flow, mode='markers', name='Data',
               marker=dict(color='steelblue', size=3, opacity=0.5),
               showlegend=False),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines', name='1:1 Line',
               line=dict(color='black', dash='dash'), showlegend=False),
    row=1, col=2
)

# 3. Hydrograph - Linear scale (good for seeing peaks)
fig.add_trace(
    go.Scatter(x=comparison.index, y=obs_flow, name='Observed',
               line=dict(color='blue', width=1), showlegend=False),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=comparison.index, y=sim_flow, name='Simulated',
               line=dict(color='red', width=1), showlegend=False),
    row=2, col=1
)

# 4. Flow Duration Curves
obs_sorted = np.sort(obs_flow)[::-1]
sim_sorted = np.sort(sim_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

fig.add_trace(
    go.Scatter(x=exceedance, y=obs_sorted, name='Observed FDC',
               line=dict(color='blue', width=2), showlegend=False),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=exceedance, y=sim_sorted, name='Simulated FDC',
               line=dict(color='red', width=2), showlegend=False),
    row=2, col=2
)

# 5. Residuals over time
residuals = sim_flow - obs_flow
fig.add_trace(
    go.Scatter(x=comparison.index, y=residuals, name='Residuals',
               line=dict(color='purple', width=0.5), showlegend=False),
    row=3, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)

# 6. Residual histogram
fig.add_trace(
    go.Histogram(x=residuals, nbinsx=50, name='Residuals',
                 marker_color='purple', opacity=0.7, showlegend=False),
    row=3, col=2
)
fig.add_vline(x=0, line_dash="dash", line_color="red", row=3, col=2)

# Update axes
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig.update_yaxes(title_text="Simulated (ML/day)", row=1, col=2)
fig.update_xaxes(title_text="Observed (ML/day)", row=1, col=2)
fig.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=2)
fig.update_xaxes(title_text="Exceedance (%)", row=2, col=2)
fig.update_yaxes(title_text="Residual (ML/day)", row=3, col=1)
fig.update_xaxes(title_text="Residual (ML/day)", row=3, col=2)

fig.update_layout(
    title=f"<b>Calibration Results: NSE = {metrics['NSE']:.3f}</b>",
    height=900,
    showlegend=True,
    legend=dict(orientation='h', y=1.02),
    # Link x-axes of all time-series plots (first column)
    xaxis=dict(matches='x'),
    xaxis3=dict(matches='x'),
    xaxis5=dict(matches='x')
)

fig.show()

# %% [markdown]
# ### Interpreting the Diagnostics
#
# **Hydrograph (Log Scale)**: Shows how well the model captures flow across all
# magnitudes. Look for:
# - Do the peaks align?
# - Is baseflow (the floor) captured correctly?
# - Are recession curves (falling limb) the right shape?
#
# **Scatter Plot**: Points should cluster around the 1:1 line. Deviations indicate:
# - Above line: Model over-predicts
# - Below line: Model under-predicts
#
# **Flow Duration Curve**: Shows the distribution of flows. Check:
# - High flows (left side): Peak flow performance
# - Low flows (right side): Baseflow performance
#
# **Residuals**: Should be random noise around zero. Patterns indicate:
# - Trends: Model bias changing over time
# - Seasonality: Missing processes
# - Heteroscedasticity: Errors scaling with flow

# %% [markdown]
# ---
# ## Step 7: The Effect of Flow Transformations
#
# ### Why Flow Transformations Matter
#
# The standard **NSE is biased towards high flows** because it uses squared errors
# in absolute units. Consider two scenarios:
#
# - **High flow**: Observed = 10,000 ML/day, Simulated = 10,100 ML/day
#   - Error = 100 ML/day (only 1% relative error)
#   - Squared error = 100² = **10,000**
#
# - **Low flow**: Observed = 50 ML/day, Simulated = 60 ML/day
#   - Error = 10 ML/day (20% relative error!)
#   - Squared error = 10² = **100**
#
# Even though the low flow prediction is proportionally much worse (20% vs 1% error),
# the high flow error contributes 100× more to the NSE calculation. This means the
# calibration algorithm prioritizes getting high flows right at the expense of low flows.
#
# **Flow transformations** change how errors are weighted:
#
# | Transformation | Effect | Best For |
# |----------------|--------|----------|
# | **None (Q)** | Emphasizes high flows | Flood forecasting |
# | **log(Q)** | Balances all flow ranges | General purpose |
# | **1/Q** | Heavily emphasizes low flows | Drought/low flow |
# | **√Q** | Moderate emphasis on low flows | Balanced approach |
#
# Let's visualize how these transformations change the flow distribution.

# %%
# Visualize flow transformations
epsilon = 0.01  # Small value to avoid log(0) or 1/0

# Calculate transformations
obs_q = obs_flow
obs_log = np.log(obs_flow + epsilon)
obs_inv = 1.0 / (obs_flow + epsilon)
obs_sqrt = np.sqrt(obs_flow)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Original Flow (Q)',
        'Log Transform: log(Q)',
        'Inverse Transform: 1/Q',
        'Square Root Transform: √Q'
    )
)

# Original
fig.add_trace(go.Histogram(x=obs_q, nbinsx=50, name='Q',
              marker_color='steelblue', opacity=0.7), row=1, col=1)
# Log
fig.add_trace(go.Histogram(x=obs_log, nbinsx=50, name='log(Q)',
              marker_color='forestgreen', opacity=0.7), row=1, col=2)
# Inverse
fig.add_trace(go.Histogram(x=obs_inv[obs_inv < 1], nbinsx=50, name='1/Q',
              marker_color='darkorange', opacity=0.7), row=2, col=1)
# Sqrt
fig.add_trace(go.Histogram(x=obs_sqrt, nbinsx=50, name='√Q',
              marker_color='purple', opacity=0.7), row=2, col=2)

fig.update_xaxes(title_text="Flow (ML/day)", row=1, col=1)
fig.update_xaxes(title_text="log(Flow + ε)", row=1, col=2)
fig.update_xaxes(title_text="1/(Flow + ε)", row=2, col=1)
fig.update_xaxes(title_text="√Flow", row=2, col=2)

fig.update_layout(
    title="<b>Effect of Flow Transformations on Distribution</b><br>" +
          "<sup>Transformations compress high flows, expanding the influence of low flows</sup>",
    height=600, showlegend=False
)
fig.show()

print("Notice how transformations change the distribution:")
print(f"  Original Q:  range [{obs_q.min():.1f}, {obs_q.max():.1f}] ML/day")
print(f"  log(Q):      range [{obs_log.min():.2f}, {obs_log.max():.2f}]")
print(f"  √Q:          range [{obs_sqrt.min():.2f}, {obs_sqrt.max():.2f}]")

# %% [markdown]
# ### Using Flow Transformations with Objective Functions
#
# The pyrrm library provides built-in support for flow transformations through
# the `FlowTransformation` class. This allows you to easily create NSE variants
# that emphasize different parts of the flow regime:
#
# - **NSE_log**: Log-transformed - balances all flow ranges
# - **NSE_inv**: 1/Q transform - heavily emphasizes low flows
# - **NSE_sqrt**: √Q transform - moderate, balanced emphasis
# - **SDEB**: Combines chronological timing + FDC shape + bias penalty
#
# This eliminates the need to define custom classes!

# %%
# Import objective functions from pyrrm.objectives (new unified interface)
from pyrrm.objectives import NSE, KGE, KGENonParametric, FlowTransformation, SDEB

# ==============================================================================
# NSE-BASED OBJECTIVE FUNCTIONS (with flow transformations)
# ==============================================================================

# NSE_log: NSE with log transformation - balances all flow ranges
log_nse_objective = NSE(transform=FlowTransformation('log', epsilon_value=0.01))

# NSE_inv: NSE with inverse transformation (1/Q) - heavily emphasizes low flows
inv_nse_objective = NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))

# NSE_sqrt: NSE with square root transformation (√Q) - balanced emphasis
sqrt_nse_objective = NSE(transform=FlowTransformation('sqrt'))

# ==============================================================================
# COMPOSITE OBJECTIVE FUNCTIONS
# ==============================================================================

# SDEB: Sum of Daily Flows, Daily Exceedance Curve and Bias (Lerat et al., 2013)
# Composite objective combining chronological timing, FDC shape, and bias penalty
# Parameters: alpha=0.1 (low chronological weight), lam=0.5 (sqrt transform)
sdeb_objective = SDEB(alpha=0.1, lam=0.5)

# ==============================================================================
# KGE-BASED OBJECTIVE FUNCTIONS (with flow transformations)
# ==============================================================================

# Standard KGE (2012 variant, recommended) - on original flows
kge_objective = KGE(variant='2012')

# KGE with inverse transformation (1/Q) - heavily emphasizes low flows
# Uses 'inverse' transform to weight errors towards low flow periods
kge_inv_objective = KGE(variant='2012', transform=FlowTransformation('inverse', epsilon_value=0.01))

# KGE with log transformation - balances all flow ranges
# NOTE: Santos et al. (2018) warn about log+KGE numerical instabilities
# Consider sqrt or inverse for production use
kge_log_objective = KGE(variant='2012', transform=FlowTransformation('log', epsilon_value=0.01))

# KGE with square root transformation - balanced, recommended for general use
kge_sqrt_objective = KGE(variant='2012', transform=FlowTransformation('sqrt'))

# Non-parametric KGE (Pool et al., 2018) - robust to outliers and skewed distributions
# Uses Spearman correlation instead of Pearson, better for extreme events
kge_np_objective = KGENonParametric()

# Non-parametric KGE with inverse transformation (1/Q) - emphasizes low flows with robustness to outliers
kge_np_inv_objective = KGENonParametric(transform=FlowTransformation('inverse', epsilon_value=0.01))

# Non-parametric KGE with square root transformation - balanced with robustness to outliers
kge_np_sqrt_objective = KGENonParametric(transform=FlowTransformation('sqrt'))

# Non-parametric KGE with log transformation - balances all flow ranges
# NOTE: Same caveats as KGE(log) apply - use with caution
kge_np_log_objective = KGENonParametric(transform=FlowTransformation('log', epsilon_value=0.01))

print("=" * 70)
print("OBJECTIVE FUNCTIONS DEFINED")
print("=" * 70)
print("\nNSE-based objectives (with flow transformations):")
print(f"  - NSE(log):     log(Q) transform - balances all flow ranges")
print(f"  - NSE(inverse): 1/Q transform - heavily emphasizes low flows")
print(f"  - NSE(sqrt):    √Q transform - moderate emphasis on low flows")
print("\nComposite objectives:")
print(f"  - SDEB:         Combines chronological + FDC errors with bias penalty")
print("\nKGE-based objectives (with flow transformations):")
print(f"  - KGE:          Standard KGE (2012) on original flows")
print(f"  - KGE(inverse): 1/Q transform - heavily emphasizes low flows")
print(f"  - KGE(log):     log(Q) transform (use with caution, see Santos 2018)")
print(f"  - KGE(sqrt):    √Q transform - balanced, recommended")
print("\nNon-parametric KGE objectives (robust to outliers and skewed distributions):")
print(f"  - KGE_np:           Standard non-parametric KGE on original flows")
print(f"  - KGE_np(inverse):  1/Q transform - low flow emphasis")
print(f"  - KGE_np(sqrt):     √Q transform - balanced")
print(f"  - KGE_np(log):      log(Q) transform (use with caution)")

# %% [markdown]
# ### Calibrating with Multiple Objective Functions
#
# Now let's run calibrations with each transformation and compare the results.

# %%
# SCE-UA configuration for Sacramento model
# Sacramento has 22 parameters
N_PARAMS_SAC = 22
MAX_EVALS_SAC = 10000

print(f"SCE-UA Direct Configuration for Sacramento ({N_PARAMS_SAC} parameters):")
print(f"  max_evals = {MAX_EVALS_SAC}")

# %%
# Run calibration with NSE_log
print("=" * 70)
print("CALIBRATION 1: NSE_log (log-transformed)")
print("=" * 70)
print("\nRunning calibration...\n")

log_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=log_nse_objective,
    warmup_period=WARMUP_DAYS
)

log_result = log_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + log_result.summary())

# Save calibration report
log_report = log_runner.create_report(log_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
log_report.save(REPORTS_DIR / '410734_sacramento_nse_sceua_log')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_nse_sceua_log.pkl'}")

# %%
# Run calibration with InverseNSE
print("=" * 70)
print("CALIBRATION 2: Inverse-Transformed NSE (1/Q)")
print("=" * 70)
print("\nRunning calibration...\n")

inv_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=inv_nse_objective,
    warmup_period=WARMUP_DAYS
)

inv_result = inv_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + inv_result.summary())

# Save calibration report
inv_report = inv_runner.create_report(inv_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
inv_report.save(REPORTS_DIR / '410734_sacramento_nse_sceua_inverse')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_nse_sceua_inverse.pkl'}")

# %%
# Run calibration with NSE_sqrt
print("=" * 70)
print("CALIBRATION 3: NSE_sqrt (square-root transformed)")
print("=" * 70)
print("\nRunning calibration...\n")

sqrt_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sqrt_nse_objective,
    warmup_period=WARMUP_DAYS
)

sqrt_result = sqrt_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + sqrt_result.summary())

# Save calibration report
sqrt_report = sqrt_runner.create_report(sqrt_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
sqrt_report.save(REPORTS_DIR / '410734_sacramento_nse_sceua_sqrt')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_nse_sceua_sqrt.pkl'}")

# %%
# Run calibration with SDEB
print("=" * 70)
print("CALIBRATION 4: SDEB (Chronological + FDC + Bias)")
print("=" * 70)
print("\nSDEB combines chronological errors, FDC errors, and bias penalty.")
print("Used in Australian SOURCE modeling platform (Lerat et al., 2013).")
print("\nRunning calibration...\n")

sdeb_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sdeb_objective,
    warmup_period=WARMUP_DAYS
)

sdeb_result = sdeb_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + sdeb_result.summary())

# Save calibration report
sdeb_report = sdeb_runner.create_report(sdeb_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
sdeb_report.save(REPORTS_DIR / '410734_sacramento_sdeb_sceua')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_sdeb_sceua.pkl'}")

# %% [markdown]
# ---
# ## Step 7b: KGE-Based Calibrations
#
# Now let's run calibrations using the **Kling-Gupta Efficiency (KGE)** family.
# KGE decomposes performance into correlation, variability, and bias components,
# making it easier to diagnose where models succeed or fail.
#
# We'll calibrate with:
# - **KGE (standard)**: On original flows, emphasizes high flows
# - **KGE(inverse)**: 1/Q transformation, heavily emphasizes low flows
# - **KGE(sqrt)**: √Q transformation, balanced emphasis (recommended)
# - **KGE(log)**: log(Q) transformation (use with caution, see Santos 2018)
# - **KGE_np**: Non-parametric variant, robust to outliers

# %%
# Run calibration with standard KGE
print("=" * 70)
print("CALIBRATION 5: Standard KGE (2012)")
print("=" * 70)
print("\nKGE decomposes into correlation, variability, and bias components.")
print("Variant 2012 uses coefficient of variation ratio (recommended).")
print("\nRunning calibration...\n")

kge_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_objective,
    warmup_period=WARMUP_DAYS
)

kge_result = kge_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_result.summary())

# Save calibration report
kge_report = kge_runner.create_report(kge_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_report.save(REPORTS_DIR / '410734_sacramento_kge_sceua')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kge_sceua.pkl'}")

# %%
# Run calibration with KGE(inverse)
print("=" * 70)
print("CALIBRATION 6: KGE with Inverse Transform (1/Q)")
print("=" * 70)
print("\nInverse transformation heavily emphasizes low flows.")
print("Useful for drought analysis and environmental flow assessments.")
print("\nRunning calibration...\n")

kge_inv_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_inv_objective,
    warmup_period=WARMUP_DAYS
)

kge_inv_result = kge_inv_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_inv_result.summary())

# Save calibration report
kge_inv_report = kge_inv_runner.create_report(kge_inv_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_inv_report.save(REPORTS_DIR / '410734_sacramento_kge_sceua_inverse')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kge_sceua_inverse.pkl'}")

# %%
# Run calibration with KGE(sqrt)
print("=" * 70)
print("CALIBRATION 7: KGE with Square Root Transform (√Q)")
print("=" * 70)
print("\nSquare root transformation provides balanced emphasis.")
print("Recommended for general water resources applications.")
print("\nRunning calibration...\n")

kge_sqrt_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_sqrt_objective,
    warmup_period=WARMUP_DAYS
)

kge_sqrt_result = kge_sqrt_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_sqrt_result.summary())

# Save calibration report
kge_sqrt_report = kge_sqrt_runner.create_report(kge_sqrt_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_sqrt_report.save(REPORTS_DIR / '410734_sacramento_kge_sceua_sqrt')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kge_sceua_sqrt.pkl'}")

# %%
# Run calibration with KGE(log) - use with caution
import warnings
print("=" * 70)
print("CALIBRATION 8: KGE with Log Transform (log Q)")
print("=" * 70)
print("\nWARNING: Santos et al. (2018) showed that log transform with KGE")
print("can cause numerical instabilities and unit-dependence issues.")
print("Consider sqrt or inverse transforms for production use.")
print("\nRunning calibration...\n")

# Suppress the warning during calibration (we've already warned)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kge_log_runner = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=kge_log_objective,
        warmup_period=WARMUP_DAYS
    )

    kge_log_result = kge_log_runner.run_sceua_direct(
        max_evals=MAX_EVALS_SAC,
        seed=42,
        verbose=True,
        max_tolerant_iter=50,
        tolerance=1e-4
    )
print("\n" + kge_log_result.summary())

# Save calibration report
kge_log_report = kge_log_runner.create_report(kge_log_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_log_report.save(REPORTS_DIR / '410734_sacramento_kge_sceua_log')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kge_sceua_log.pkl'}")

# %%
# Run calibration with Non-parametric KGE
print("=" * 70)
print("CALIBRATION 9: Non-parametric KGE (Pool et al., 2018)")
print("=" * 70)
print("\nNon-parametric KGE uses Spearman rank correlation instead of Pearson.")
print("More robust to outliers and skewed flow distributions.")
print("Better suited for flashy catchments with extreme events.")
print("\nRunning calibration...\n")

kge_np_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_np_objective,
    warmup_period=WARMUP_DAYS
)

kge_np_result = kge_np_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_np_result.summary())

# Save calibration report
kge_np_report = kge_np_runner.create_report(kge_np_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_np_report.save(REPORTS_DIR / '410734_sacramento_kgenp_sceua')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kgenp_sceua.pkl'}")

# %%
# Run calibration with Non-parametric KGE with inverse transformation
print("=" * 70)
print("CALIBRATION 10: Non-parametric KGE with Inverse Transform (1/Q)")
print("=" * 70)
print("\nNon-parametric KGE uses Spearman correlation (robust to outliers).")
print("Combined with inverse transform to emphasize low flows.")
print("\nRunning calibration...\n")

kge_np_inv_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_np_inv_objective,
    warmup_period=WARMUP_DAYS
)

kge_np_inv_result = kge_np_inv_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_np_inv_result.summary())

# Save calibration report
kge_np_inv_report = kge_np_inv_runner.create_report(kge_np_inv_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_np_inv_report.save(REPORTS_DIR / '410734_sacramento_kgenp_sceua_inverse')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kgenp_sceua_inverse.pkl'}")

# %%
# Run calibration with Non-parametric KGE with sqrt transformation
print("=" * 70)
print("CALIBRATION 11: Non-parametric KGE with Square Root Transform (√Q)")
print("=" * 70)
print("\nNon-parametric KGE uses Spearman correlation (robust to outliers).")
print("Combined with sqrt transform for balanced calibration.")
print("\nRunning calibration...\n")

kge_np_sqrt_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=kge_np_sqrt_objective,
    warmup_period=WARMUP_DAYS
)

kge_np_sqrt_result = kge_np_sqrt_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=50,
    tolerance=1e-4
)
print("\n" + kge_np_sqrt_result.summary())

# Save calibration report
kge_np_sqrt_report = kge_np_sqrt_runner.create_report(kge_np_sqrt_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_np_sqrt_report.save(REPORTS_DIR / '410734_sacramento_kgenp_sceua_sqrt')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kgenp_sceua_sqrt.pkl'}")

# %%
# Run calibration with Non-parametric KGE with log transformation
import warnings
print("=" * 70)
print("CALIBRATION 12: Non-parametric KGE with Log Transform (log Q)")
print("=" * 70)
print("\nWARNING: Santos et al. (2018) showed that log transform with KGE")
print("can cause numerical instabilities. This also applies to KGE_np.")
print("Consider sqrt or inverse transforms for production use.")
print("\nRunning calibration...\n")

# Suppress the warning during calibration (we've already warned)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kge_np_log_runner = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=kge_np_log_objective,
        warmup_period=WARMUP_DAYS
    )

    kge_np_log_result = kge_np_log_runner.run_sceua_direct(
        max_evals=MAX_EVALS_SAC,
        seed=42,
        verbose=True,
        max_tolerant_iter=50,
        tolerance=1e-4
    )
print("\n" + kge_np_log_result.summary())

# Save calibration report
kge_np_log_report = kge_np_log_runner.create_report(kge_np_log_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
kge_np_log_report.save(REPORTS_DIR / '410734_sacramento_kgenp_sceua_log')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_kgenp_sceua_log.pkl'}")

# %%
# Run simulations with all calibrated parameter sets (NSE-based objectives)
log_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
log_model.set_parameters(log_result.best_parameters)
log_model.reset()
log_sim_flow = log_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

inv_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
inv_model.set_parameters(inv_result.best_parameters)
inv_model.reset()
inv_sim_flow = inv_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

sqrt_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
sqrt_model.set_parameters(sqrt_result.best_parameters)
sqrt_model.reset()
sqrt_sim_flow = sqrt_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

sdeb_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
sdeb_model.set_parameters(sdeb_result.best_parameters)
sdeb_model.reset()
sdeb_sim_flow = sdeb_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

# Run simulations with KGE-based calibrated parameter sets
kge_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_model.set_parameters(kge_result.best_parameters)
kge_model.reset()
kge_sim_flow = kge_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_inv_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_inv_model.set_parameters(kge_inv_result.best_parameters)
kge_inv_model.reset()
kge_inv_sim_flow = kge_inv_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_sqrt_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_sqrt_model.set_parameters(kge_sqrt_result.best_parameters)
kge_sqrt_model.reset()
kge_sqrt_sim_flow = kge_sqrt_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_log_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_log_model.set_parameters(kge_log_result.best_parameters)
kge_log_model.reset()
kge_log_sim_flow = kge_log_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_np_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_np_model.set_parameters(kge_np_result.best_parameters)
kge_np_model.reset()
kge_np_sim_flow = kge_np_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

# Run simulations with KGE_np transformed calibrated parameter sets
kge_np_inv_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_np_inv_model.set_parameters(kge_np_inv_result.best_parameters)
kge_np_inv_model.reset()
kge_np_inv_sim_flow = kge_np_inv_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_np_sqrt_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_np_sqrt_model.set_parameters(kge_np_sqrt_result.best_parameters)
kge_np_sqrt_model.reset()
kge_np_sqrt_sim_flow = kge_np_sqrt_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

kge_np_log_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
kge_np_log_model.set_parameters(kge_np_log_result.best_parameters)
kge_np_log_model.reset()
kge_np_log_sim_flow = kge_np_log_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

# Compute canonical diagnostics for all calibrations
nse_metrics = compute_diagnostics(sim_flow, obs_flow)
log_metrics = compute_diagnostics(log_sim_flow, obs_flow)
inv_metrics = compute_diagnostics(inv_sim_flow, obs_flow)
sqrt_metrics = compute_diagnostics(sqrt_sim_flow, obs_flow)
sdeb_metrics = compute_diagnostics(sdeb_sim_flow, obs_flow)

kge_metrics = compute_diagnostics(kge_sim_flow, obs_flow)
kge_inv_metrics = compute_diagnostics(kge_inv_sim_flow, obs_flow)
kge_sqrt_metrics = compute_diagnostics(kge_sqrt_sim_flow, obs_flow)
kge_log_metrics = compute_diagnostics(kge_log_sim_flow, obs_flow)
kge_np_metrics = compute_diagnostics(kge_np_sim_flow, obs_flow)
kge_np_inv_metrics = compute_diagnostics(kge_np_inv_sim_flow, obs_flow)
kge_np_sqrt_metrics = compute_diagnostics(kge_np_sqrt_sim_flow, obs_flow)
kge_np_log_metrics = compute_diagnostics(kge_np_log_sim_flow, obs_flow)

# Calculate transformed NSE/KGE values for each calibration using the objective functions
# All objectives from pyrrm.objectives use __call__(obs, sim) interface
nse_lognse = log_nse_objective(obs_flow, sim_flow)
nse_invnse = inv_nse_objective(obs_flow, sim_flow)
nse_sqrtnse = sqrt_nse_objective(obs_flow, sim_flow)
nse_sdeb = sdeb_objective(obs_flow, sim_flow)

log_lognse = log_nse_objective(obs_flow, log_sim_flow)
log_invnse = inv_nse_objective(obs_flow, log_sim_flow)
log_sqrtnse = sqrt_nse_objective(obs_flow, log_sim_flow)
log_sdeb = sdeb_objective(obs_flow, log_sim_flow)

inv_lognse = log_nse_objective(obs_flow, inv_sim_flow)
inv_invnse = inv_nse_objective(obs_flow, inv_sim_flow)
inv_sqrtnse = sqrt_nse_objective(obs_flow, inv_sim_flow)
inv_sdeb = sdeb_objective(obs_flow, inv_sim_flow)

sqrt_lognse = log_nse_objective(obs_flow, sqrt_sim_flow)
sqrt_invnse = inv_nse_objective(obs_flow, sqrt_sim_flow)
sqrt_sqrtnse = sqrt_nse_objective(obs_flow, sqrt_sim_flow)
sqrt_sdeb = sdeb_objective(obs_flow, sqrt_sim_flow)

sdeb_lognse = log_nse_objective(obs_flow, sdeb_sim_flow)
sdeb_invnse = inv_nse_objective(obs_flow, sdeb_sim_flow)
sdeb_sqrtnse = sqrt_nse_objective(obs_flow, sdeb_sim_flow)
sdeb_sdeb = sdeb_objective(obs_flow, sdeb_sim_flow)

# KGE objective values for all calibrations
kge_kge = kge_objective(obs_flow, kge_sim_flow)
kge_inv_kge = kge_objective(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge = kge_objective(obs_flow, kge_sqrt_sim_flow)
kge_log_kge = kge_objective(obs_flow, kge_log_sim_flow)
kge_np_kge = kge_np_objective(obs_flow, kge_np_sim_flow)

# Calculate KGE components for all calibrations (NSE-based)
nse_kge_comp = kge_objective.get_components(obs_flow, sim_flow)
log_kge_comp = kge_objective.get_components(obs_flow, log_sim_flow)
inv_kge_comp = kge_objective.get_components(obs_flow, inv_sim_flow)
sqrt_kge_comp = kge_objective.get_components(obs_flow, sqrt_sim_flow)
sdeb_kge_comp = kge_objective.get_components(obs_flow, sdeb_sim_flow)

# Calculate KGE components for all calibrations (KGE-based)
kge_kge_comp = kge_objective.get_components(obs_flow, kge_sim_flow)
kge_inv_kge_comp = kge_objective.get_components(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_comp = kge_objective.get_components(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_comp = kge_objective.get_components(obs_flow, kge_log_sim_flow)
kge_np_kge_comp = kge_np_objective.get_components(obs_flow, kge_np_sim_flow)

# Calculate KGE(1/Q) for all calibrations
nse_kge_inv = kge_inv_objective(obs_flow, sim_flow)
log_kge_inv = kge_inv_objective(obs_flow, log_sim_flow)
inv_kge_inv = kge_inv_objective(obs_flow, inv_sim_flow)
sqrt_kge_inv = kge_inv_objective(obs_flow, sqrt_sim_flow)
sdeb_kge_inv = kge_inv_objective(obs_flow, sdeb_sim_flow)
kge_kge_inv = kge_inv_objective(obs_flow, kge_sim_flow)
kge_inv_kge_inv = kge_inv_objective(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_inv = kge_inv_objective(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_inv = kge_inv_objective(obs_flow, kge_log_sim_flow)
kge_np_kge_inv = kge_inv_objective(obs_flow, kge_np_sim_flow)

# Calculate KGE(1/Q) components for all calibrations
nse_kge_inv_comp = kge_inv_objective.get_components(obs_flow, sim_flow)
log_kge_inv_comp = kge_inv_objective.get_components(obs_flow, log_sim_flow)
inv_kge_inv_comp = kge_inv_objective.get_components(obs_flow, inv_sim_flow)
sqrt_kge_inv_comp = kge_inv_objective.get_components(obs_flow, sqrt_sim_flow)
sdeb_kge_inv_comp = kge_inv_objective.get_components(obs_flow, sdeb_sim_flow)
kge_kge_inv_comp = kge_inv_objective.get_components(obs_flow, kge_sim_flow)
kge_inv_kge_inv_comp = kge_inv_objective.get_components(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_inv_comp = kge_inv_objective.get_components(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_inv_comp = kge_inv_objective.get_components(obs_flow, kge_log_sim_flow)
kge_np_kge_inv_comp = kge_inv_objective.get_components(obs_flow, kge_np_sim_flow)

# Calculate KGE(√Q) for all calibrations
nse_kge_sqrt = kge_sqrt_objective(obs_flow, sim_flow)
log_kge_sqrt = kge_sqrt_objective(obs_flow, log_sim_flow)
inv_kge_sqrt = kge_sqrt_objective(obs_flow, inv_sim_flow)
sqrt_kge_sqrt = kge_sqrt_objective(obs_flow, sqrt_sim_flow)
sdeb_kge_sqrt = kge_sqrt_objective(obs_flow, sdeb_sim_flow)
kge_kge_sqrt = kge_sqrt_objective(obs_flow, kge_sim_flow)
kge_inv_kge_sqrt = kge_sqrt_objective(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_sqrt = kge_sqrt_objective(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_sqrt = kge_sqrt_objective(obs_flow, kge_log_sim_flow)
kge_np_kge_sqrt = kge_sqrt_objective(obs_flow, kge_np_sim_flow)

# Calculate KGE(√Q) components for all calibrations
nse_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, sim_flow)
log_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, log_sim_flow)
inv_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, inv_sim_flow)
sqrt_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, sqrt_sim_flow)
sdeb_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, sdeb_sim_flow)
kge_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, kge_sim_flow)
kge_inv_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, kge_log_sim_flow)
kge_np_kge_sqrt_comp = kge_sqrt_objective.get_components(obs_flow, kge_np_sim_flow)

# Calculate KGE(log) for all calibrations
nse_kge_log = kge_log_objective(obs_flow, sim_flow)
log_kge_log = kge_log_objective(obs_flow, log_sim_flow)
inv_kge_log = kge_log_objective(obs_flow, inv_sim_flow)
sqrt_kge_log = kge_log_objective(obs_flow, sqrt_sim_flow)
sdeb_kge_log = kge_log_objective(obs_flow, sdeb_sim_flow)
kge_kge_log = kge_log_objective(obs_flow, kge_sim_flow)
kge_inv_kge_log = kge_log_objective(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_log = kge_log_objective(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_log = kge_log_objective(obs_flow, kge_log_sim_flow)
kge_np_kge_log = kge_log_objective(obs_flow, kge_np_sim_flow)

# Calculate KGE(log) components for all calibrations
nse_kge_log_comp = kge_log_objective.get_components(obs_flow, sim_flow)
log_kge_log_comp = kge_log_objective.get_components(obs_flow, log_sim_flow)
inv_kge_log_comp = kge_log_objective.get_components(obs_flow, inv_sim_flow)
sqrt_kge_log_comp = kge_log_objective.get_components(obs_flow, sqrt_sim_flow)
sdeb_kge_log_comp = kge_log_objective.get_components(obs_flow, sdeb_sim_flow)
kge_kge_log_comp = kge_log_objective.get_components(obs_flow, kge_sim_flow)
kge_inv_kge_log_comp = kge_log_objective.get_components(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_log_comp = kge_log_objective.get_components(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_log_comp = kge_log_objective.get_components(obs_flow, kge_log_sim_flow)
kge_np_kge_log_comp = kge_log_objective.get_components(obs_flow, kge_np_sim_flow)

# Calculate KGE_np for all calibrations
nse_kge_np = kge_np_objective(obs_flow, sim_flow)
log_kge_np = kge_np_objective(obs_flow, log_sim_flow)
inv_kge_np = kge_np_objective(obs_flow, inv_sim_flow)
sqrt_kge_np = kge_np_objective(obs_flow, sqrt_sim_flow)
sdeb_kge_np = kge_np_objective(obs_flow, sdeb_sim_flow)
kge_kge_np = kge_np_objective(obs_flow, kge_sim_flow)
kge_inv_kge_np = kge_np_objective(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_np = kge_np_objective(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_np = kge_np_objective(obs_flow, kge_log_sim_flow)
kge_np_kge_np = kge_np_objective(obs_flow, kge_np_sim_flow)

# Calculate KGE_np components for all calibrations
nse_kge_np_comp = kge_np_objective.get_components(obs_flow, sim_flow)
log_kge_np_comp = kge_np_objective.get_components(obs_flow, log_sim_flow)
inv_kge_np_comp = kge_np_objective.get_components(obs_flow, inv_sim_flow)
sqrt_kge_np_comp = kge_np_objective.get_components(obs_flow, sqrt_sim_flow)
sdeb_kge_np_comp = kge_np_objective.get_components(obs_flow, sdeb_sim_flow)
kge_kge_np_comp = kge_np_objective.get_components(obs_flow, kge_sim_flow)
kge_inv_kge_np_comp = kge_np_objective.get_components(obs_flow, kge_inv_sim_flow)
kge_sqrt_kge_np_comp = kge_np_objective.get_components(obs_flow, kge_sqrt_sim_flow)
kge_log_kge_np_comp = kge_np_objective.get_components(obs_flow, kge_log_sim_flow)
kge_np_kge_np_comp = kge_np_objective.get_components(obs_flow, kge_np_sim_flow)

# Calculate additional transformed NSE metrics for KGE-based calibrations
kge_invnse = inv_nse_objective(obs_flow, kge_sim_flow)
kge_sqrtnse = sqrt_nse_objective(obs_flow, kge_sim_flow)
kge_sdeb = sdeb_objective(obs_flow, kge_sim_flow)

kge_inv_invnse = inv_nse_objective(obs_flow, kge_inv_sim_flow)
kge_inv_sqrtnse = sqrt_nse_objective(obs_flow, kge_inv_sim_flow)
kge_inv_sdeb = sdeb_objective(obs_flow, kge_inv_sim_flow)

kge_sqrt_invnse = inv_nse_objective(obs_flow, kge_sqrt_sim_flow)
kge_sqrt_sqrtnse = sqrt_nse_objective(obs_flow, kge_sqrt_sim_flow)
kge_sqrt_sdeb = sdeb_objective(obs_flow, kge_sqrt_sim_flow)

kge_log_invnse = inv_nse_objective(obs_flow, kge_log_sim_flow)
kge_log_sqrtnse = sqrt_nse_objective(obs_flow, kge_log_sim_flow)
kge_log_sdeb = sdeb_objective(obs_flow, kge_log_sim_flow)

kge_np_invnse = inv_nse_objective(obs_flow, kge_np_sim_flow)
kge_np_sqrtnse = sqrt_nse_objective(obs_flow, kge_np_sim_flow)
kge_np_sdeb = sdeb_objective(obs_flow, kge_np_sim_flow)

# Additional transformed NSE metrics for KGE_np transformed calibrations
kge_np_inv_invnse = inv_nse_objective(obs_flow, kge_np_inv_sim_flow)
kge_np_inv_sqrtnse = sqrt_nse_objective(obs_flow, kge_np_inv_sim_flow)
kge_np_inv_sdeb = sdeb_objective(obs_flow, kge_np_inv_sim_flow)

kge_np_sqrt_invnse = inv_nse_objective(obs_flow, kge_np_sqrt_sim_flow)
kge_np_sqrt_sqrtnse = sqrt_nse_objective(obs_flow, kge_np_sqrt_sim_flow)
kge_np_sqrt_sdeb = sdeb_objective(obs_flow, kge_np_sqrt_sim_flow)

kge_np_log_invnse = inv_nse_objective(obs_flow, kge_np_log_sim_flow)
kge_np_log_sqrtnse = sqrt_nse_objective(obs_flow, kge_np_log_sim_flow)
kge_np_log_sdeb = sdeb_objective(obs_flow, kge_np_log_sim_flow)

# Calculate KGE_np transformed objective values for all calibrations
kge_np_inv_kge = kge_np_inv_objective(obs_flow, kge_np_inv_sim_flow)
kge_np_sqrt_kge = kge_np_sqrt_objective(obs_flow, kge_np_sqrt_sim_flow)
kge_np_log_kge = kge_np_log_objective(obs_flow, kge_np_log_sim_flow)

# %% [markdown]
# ### Comprehensive Comparison of All Calibrations
#
# Now let's compare all objective functions side by side in a single comprehensive table.
# Objective functions are shown as columns, with diagnostic metrics as rows for easy comparison.

# %%
print("=" * 150)
print("COMPREHENSIVE COMPARISON: ALL OBJECTIVE FUNCTIONS")
print("=" * 150)
print(f"\n{'Metric':<14} {'NSE':>11} {'NSE_log':>11} {'NSE_inv':>11} {'NSE_sqrt':>11} {'SDEB':>11} {'KGE':>11} {'KGE_inv':>11} {'KGE_sqrt':>11} {'KGE_log':>11} {'KGE_np':>11}")
print("-" * 150)

# NSE variants (canonical names: NSE, NSE_log, NSE_inv, NSE_sqrt)
print(f"{'NSE':<14} {nse_metrics['NSE']:>11.4f} {log_metrics['NSE']:>11.4f} {inv_metrics['NSE']:>11.4f} {sqrt_metrics['NSE']:>11.4f} {sdeb_metrics['NSE']:>11.4f} {kge_metrics['NSE']:>11.4f} {kge_inv_metrics['NSE']:>11.4f} {kge_sqrt_metrics['NSE']:>11.4f} {kge_log_metrics['NSE']:>11.4f} {kge_np_metrics['NSE']:>11.4f}")
print(f"{'NSE_log':<14} {nse_metrics['NSE_log']:>11.4f} {log_metrics['NSE_log']:>11.4f} {inv_metrics['NSE_log']:>11.4f} {sqrt_metrics['NSE_log']:>11.4f} {sdeb_metrics['NSE_log']:>11.4f} {kge_metrics['NSE_log']:>11.4f} {kge_inv_metrics['NSE_log']:>11.4f} {kge_sqrt_metrics['NSE_log']:>11.4f} {kge_log_metrics['NSE_log']:>11.4f} {kge_np_metrics['NSE_log']:>11.4f}")
print(f"{'NSE_inv':<14} {nse_metrics['NSE_inv']:>11.4f} {log_metrics['NSE_inv']:>11.4f} {inv_metrics['NSE_inv']:>11.4f} {sqrt_metrics['NSE_inv']:>11.4f} {sdeb_metrics['NSE_inv']:>11.4f} {kge_metrics['NSE_inv']:>11.4f} {kge_inv_metrics['NSE_inv']:>11.4f} {kge_sqrt_metrics['NSE_inv']:>11.4f} {kge_log_metrics['NSE_inv']:>11.4f} {kge_np_metrics['NSE_inv']:>11.4f}")
print(f"{'NSE_sqrt':<14} {nse_metrics['NSE_sqrt']:>11.4f} {log_metrics['NSE_sqrt']:>11.4f} {inv_metrics['NSE_sqrt']:>11.4f} {sqrt_metrics['NSE_sqrt']:>11.4f} {sdeb_metrics['NSE_sqrt']:>11.4f} {kge_metrics['NSE_sqrt']:>11.4f} {kge_inv_metrics['NSE_sqrt']:>11.4f} {kge_sqrt_metrics['NSE_sqrt']:>11.4f} {kge_log_metrics['NSE_sqrt']:>11.4f} {kge_np_metrics['NSE_sqrt']:>11.4f}")
print(f"{'SDEB':<14} {nse_sdeb:>11.2f} {log_sdeb:>11.2f} {inv_sdeb:>11.2f} {sqrt_sdeb:>11.2f} {sdeb_sdeb:>11.2f} {kge_sdeb:>11.2f} {kge_inv_sdeb:>11.2f} {kge_sqrt_sdeb:>11.2f} {kge_log_sdeb:>11.2f} {kge_np_sdeb:>11.2f}")

# KGE(Q) + components (canonical: KGE, KGE_r, KGE_alpha, KGE_beta)
print(f"{'KGE':<14} {nse_metrics['KGE']:>11.4f} {log_metrics['KGE']:>11.4f} {inv_metrics['KGE']:>11.4f} {sqrt_metrics['KGE']:>11.4f} {sdeb_metrics['KGE']:>11.4f} {kge_metrics['KGE']:>11.4f} {kge_inv_metrics['KGE']:>11.4f} {kge_sqrt_metrics['KGE']:>11.4f} {kge_log_metrics['KGE']:>11.4f} {kge_np_metrics['KGE']:>11.4f}")
print(f"{'  KGE_r':<14} {nse_metrics['KGE_r']:>11.4f} {log_metrics['KGE_r']:>11.4f} {inv_metrics['KGE_r']:>11.4f} {sqrt_metrics['KGE_r']:>11.4f} {sdeb_metrics['KGE_r']:>11.4f} {kge_metrics['KGE_r']:>11.4f} {kge_inv_metrics['KGE_r']:>11.4f} {kge_sqrt_metrics['KGE_r']:>11.4f} {kge_log_metrics['KGE_r']:>11.4f} {kge_np_metrics['KGE_r']:>11.4f}")
print(f"{'  KGE_alpha':<14} {nse_metrics['KGE_alpha']:>11.4f} {log_metrics['KGE_alpha']:>11.4f} {inv_metrics['KGE_alpha']:>11.4f} {sqrt_metrics['KGE_alpha']:>11.4f} {sdeb_metrics['KGE_alpha']:>11.4f} {kge_metrics['KGE_alpha']:>11.4f} {kge_inv_metrics['KGE_alpha']:>11.4f} {kge_sqrt_metrics['KGE_alpha']:>11.4f} {kge_log_metrics['KGE_alpha']:>11.4f} {kge_np_metrics['KGE_alpha']:>11.4f}")
print(f"{'  KGE_beta':<14} {nse_metrics['KGE_beta']:>11.4f} {log_metrics['KGE_beta']:>11.4f} {inv_metrics['KGE_beta']:>11.4f} {sqrt_metrics['KGE_beta']:>11.4f} {sdeb_metrics['KGE_beta']:>11.4f} {kge_metrics['KGE_beta']:>11.4f} {kge_inv_metrics['KGE_beta']:>11.4f} {kge_sqrt_metrics['KGE_beta']:>11.4f} {kge_log_metrics['KGE_beta']:>11.4f} {kge_np_metrics['KGE_beta']:>11.4f}")

# KGE(1/Q) + components (canonical: KGE_inv, KGE_inv_r, KGE_inv_alpha, KGE_inv_beta)
print(f"{'KGE_inv':<14} {nse_metrics['KGE_inv']:>11.4f} {log_metrics['KGE_inv']:>11.4f} {inv_metrics['KGE_inv']:>11.4f} {sqrt_metrics['KGE_inv']:>11.4f} {sdeb_metrics['KGE_inv']:>11.4f} {kge_metrics['KGE_inv']:>11.4f} {kge_inv_metrics['KGE_inv']:>11.4f} {kge_sqrt_metrics['KGE_inv']:>11.4f} {kge_log_metrics['KGE_inv']:>11.4f} {kge_np_metrics['KGE_inv']:>11.4f}")
print(f"{'  KGE_inv_r':<14} {nse_metrics['KGE_inv_r']:>11.4f} {log_metrics['KGE_inv_r']:>11.4f} {inv_metrics['KGE_inv_r']:>11.4f} {sqrt_metrics['KGE_inv_r']:>11.4f} {sdeb_metrics['KGE_inv_r']:>11.4f} {kge_metrics['KGE_inv_r']:>11.4f} {kge_inv_metrics['KGE_inv_r']:>11.4f} {kge_sqrt_metrics['KGE_inv_r']:>11.4f} {kge_log_metrics['KGE_inv_r']:>11.4f} {kge_np_metrics['KGE_inv_r']:>11.4f}")
print(f"{'  KGE_inv_a':<14} {nse_metrics['KGE_inv_alpha']:>11.4f} {log_metrics['KGE_inv_alpha']:>11.4f} {inv_metrics['KGE_inv_alpha']:>11.4f} {sqrt_metrics['KGE_inv_alpha']:>11.4f} {sdeb_metrics['KGE_inv_alpha']:>11.4f} {kge_metrics['KGE_inv_alpha']:>11.4f} {kge_inv_metrics['KGE_inv_alpha']:>11.4f} {kge_sqrt_metrics['KGE_inv_alpha']:>11.4f} {kge_log_metrics['KGE_inv_alpha']:>11.4f} {kge_np_metrics['KGE_inv_alpha']:>11.4f}")
print(f"{'  KGE_inv_b':<14} {nse_metrics['KGE_inv_beta']:>11.4f} {log_metrics['KGE_inv_beta']:>11.4f} {inv_metrics['KGE_inv_beta']:>11.4f} {sqrt_metrics['KGE_inv_beta']:>11.4f} {sdeb_metrics['KGE_inv_beta']:>11.4f} {kge_metrics['KGE_inv_beta']:>11.4f} {kge_inv_metrics['KGE_inv_beta']:>11.4f} {kge_sqrt_metrics['KGE_inv_beta']:>11.4f} {kge_log_metrics['KGE_inv_beta']:>11.4f} {kge_np_metrics['KGE_inv_beta']:>11.4f}")

# KGE(sqrt Q) + components (canonical: KGE_sqrt, KGE_sqrt_r, KGE_sqrt_alpha, KGE_sqrt_beta)
print(f"{'KGE_sqrt':<14} {nse_metrics['KGE_sqrt']:>11.4f} {log_metrics['KGE_sqrt']:>11.4f} {inv_metrics['KGE_sqrt']:>11.4f} {sqrt_metrics['KGE_sqrt']:>11.4f} {sdeb_metrics['KGE_sqrt']:>11.4f} {kge_metrics['KGE_sqrt']:>11.4f} {kge_inv_metrics['KGE_sqrt']:>11.4f} {kge_sqrt_metrics['KGE_sqrt']:>11.4f} {kge_log_metrics['KGE_sqrt']:>11.4f} {kge_np_metrics['KGE_sqrt']:>11.4f}")
print(f"{'  KGE_sqrt_r':<14} {nse_metrics['KGE_sqrt_r']:>11.4f} {log_metrics['KGE_sqrt_r']:>11.4f} {inv_metrics['KGE_sqrt_r']:>11.4f} {sqrt_metrics['KGE_sqrt_r']:>11.4f} {sdeb_metrics['KGE_sqrt_r']:>11.4f} {kge_metrics['KGE_sqrt_r']:>11.4f} {kge_inv_metrics['KGE_sqrt_r']:>11.4f} {kge_sqrt_metrics['KGE_sqrt_r']:>11.4f} {kge_log_metrics['KGE_sqrt_r']:>11.4f} {kge_np_metrics['KGE_sqrt_r']:>11.4f}")
print(f"{'  KGE_sqrt_a':<14} {nse_metrics['KGE_sqrt_alpha']:>11.4f} {log_metrics['KGE_sqrt_alpha']:>11.4f} {inv_metrics['KGE_sqrt_alpha']:>11.4f} {sqrt_metrics['KGE_sqrt_alpha']:>11.4f} {sdeb_metrics['KGE_sqrt_alpha']:>11.4f} {kge_metrics['KGE_sqrt_alpha']:>11.4f} {kge_inv_metrics['KGE_sqrt_alpha']:>11.4f} {kge_sqrt_metrics['KGE_sqrt_alpha']:>11.4f} {kge_log_metrics['KGE_sqrt_alpha']:>11.4f} {kge_np_metrics['KGE_sqrt_alpha']:>11.4f}")
print(f"{'  KGE_sqrt_b':<14} {nse_metrics['KGE_sqrt_beta']:>11.4f} {log_metrics['KGE_sqrt_beta']:>11.4f} {inv_metrics['KGE_sqrt_beta']:>11.4f} {sqrt_metrics['KGE_sqrt_beta']:>11.4f} {sdeb_metrics['KGE_sqrt_beta']:>11.4f} {kge_metrics['KGE_sqrt_beta']:>11.4f} {kge_inv_metrics['KGE_sqrt_beta']:>11.4f} {kge_sqrt_metrics['KGE_sqrt_beta']:>11.4f} {kge_log_metrics['KGE_sqrt_beta']:>11.4f} {kge_np_metrics['KGE_sqrt_beta']:>11.4f}")

# KGE(log Q) + components (canonical: KGE_log, KGE_log_r, KGE_log_alpha, KGE_log_beta)
print(f"{'KGE_log':<14} {nse_metrics['KGE_log']:>11.4f} {log_metrics['KGE_log']:>11.4f} {inv_metrics['KGE_log']:>11.4f} {sqrt_metrics['KGE_log']:>11.4f} {sdeb_metrics['KGE_log']:>11.4f} {kge_metrics['KGE_log']:>11.4f} {kge_inv_metrics['KGE_log']:>11.4f} {kge_sqrt_metrics['KGE_log']:>11.4f} {kge_log_metrics['KGE_log']:>11.4f} {kge_np_metrics['KGE_log']:>11.4f}")
print(f"{'  KGE_log_r':<14} {nse_metrics['KGE_log_r']:>11.4f} {log_metrics['KGE_log_r']:>11.4f} {inv_metrics['KGE_log_r']:>11.4f} {sqrt_metrics['KGE_log_r']:>11.4f} {sdeb_metrics['KGE_log_r']:>11.4f} {kge_metrics['KGE_log_r']:>11.4f} {kge_inv_metrics['KGE_log_r']:>11.4f} {kge_sqrt_metrics['KGE_log_r']:>11.4f} {kge_log_metrics['KGE_log_r']:>11.4f} {kge_np_metrics['KGE_log_r']:>11.4f}")
print(f"{'  KGE_log_a':<14} {nse_metrics['KGE_log_alpha']:>11.4f} {log_metrics['KGE_log_alpha']:>11.4f} {inv_metrics['KGE_log_alpha']:>11.4f} {sqrt_metrics['KGE_log_alpha']:>11.4f} {sdeb_metrics['KGE_log_alpha']:>11.4f} {kge_metrics['KGE_log_alpha']:>11.4f} {kge_inv_metrics['KGE_log_alpha']:>11.4f} {kge_sqrt_metrics['KGE_log_alpha']:>11.4f} {kge_log_metrics['KGE_log_alpha']:>11.4f} {kge_np_metrics['KGE_log_alpha']:>11.4f}")
print(f"{'  KGE_log_b':<14} {nse_metrics['KGE_log_beta']:>11.4f} {log_metrics['KGE_log_beta']:>11.4f} {inv_metrics['KGE_log_beta']:>11.4f} {sqrt_metrics['KGE_log_beta']:>11.4f} {sdeb_metrics['KGE_log_beta']:>11.4f} {kge_metrics['KGE_log_beta']:>11.4f} {kge_inv_metrics['KGE_log_beta']:>11.4f} {kge_sqrt_metrics['KGE_log_beta']:>11.4f} {kge_log_metrics['KGE_log_beta']:>11.4f} {kge_np_metrics['KGE_log_beta']:>11.4f}")

# Error metrics (canonical: RMSE, MAE, PBIAS)
print(f"{'RMSE':<14} {nse_metrics['RMSE']:>11.2f} {log_metrics['RMSE']:>11.2f} {inv_metrics['RMSE']:>11.2f} {sqrt_metrics['RMSE']:>11.2f} {sdeb_metrics['RMSE']:>11.2f} {kge_metrics['RMSE']:>11.2f} {kge_inv_metrics['RMSE']:>11.2f} {kge_sqrt_metrics['RMSE']:>11.2f} {kge_log_metrics['RMSE']:>11.2f} {kge_np_metrics['RMSE']:>11.2f}")
print(f"{'MAE':<14} {nse_metrics['MAE']:>11.2f} {log_metrics['MAE']:>11.2f} {inv_metrics['MAE']:>11.2f} {sqrt_metrics['MAE']:>11.2f} {sdeb_metrics['MAE']:>11.2f} {kge_metrics['MAE']:>11.2f} {kge_inv_metrics['MAE']:>11.2f} {kge_sqrt_metrics['MAE']:>11.2f} {kge_log_metrics['MAE']:>11.2f} {kge_np_metrics['MAE']:>11.2f}")
print(f"{'PBIAS (%)':<14} {nse_metrics['PBIAS']:>+11.2f} {log_metrics['PBIAS']:>+11.2f} {inv_metrics['PBIAS']:>+11.2f} {sqrt_metrics['PBIAS']:>+11.2f} {sdeb_metrics['PBIAS']:>+11.2f} {kge_metrics['PBIAS']:>+11.2f} {kge_inv_metrics['PBIAS']:>+11.2f} {kge_sqrt_metrics['PBIAS']:>+11.2f} {kge_log_metrics['PBIAS']:>+11.2f} {kge_np_metrics['PBIAS']:>+11.2f}")

# FDC volume biases (canonical: FHV, FMV, FLV)
print(f"{'FHV (%)':<14} {nse_metrics['FHV']:>+11.2f} {log_metrics['FHV']:>+11.2f} {inv_metrics['FHV']:>+11.2f} {sqrt_metrics['FHV']:>+11.2f} {sdeb_metrics['FHV']:>+11.2f} {kge_metrics['FHV']:>+11.2f} {kge_inv_metrics['FHV']:>+11.2f} {kge_sqrt_metrics['FHV']:>+11.2f} {kge_log_metrics['FHV']:>+11.2f} {kge_np_metrics['FHV']:>+11.2f}")
print(f"{'FMV (%)':<14} {nse_metrics['FMV']:>+11.2f} {log_metrics['FMV']:>+11.2f} {inv_metrics['FMV']:>+11.2f} {sqrt_metrics['FMV']:>+11.2f} {sdeb_metrics['FMV']:>+11.2f} {kge_metrics['FMV']:>+11.2f} {kge_inv_metrics['FMV']:>+11.2f} {kge_sqrt_metrics['FMV']:>+11.2f} {kge_log_metrics['FMV']:>+11.2f} {kge_np_metrics['FMV']:>+11.2f}")
print(f"{'FLV (%)':<14} {nse_metrics.get('FLV', np.nan):>+11.2f} {log_metrics.get('FLV', np.nan):>+11.2f} {inv_metrics.get('FLV', np.nan):>+11.2f} {sqrt_metrics.get('FLV', np.nan):>+11.2f} {sdeb_metrics.get('FLV', np.nan):>+11.2f} {kge_metrics.get('FLV', np.nan):>+11.2f} {kge_inv_metrics.get('FLV', np.nan):>+11.2f} {kge_sqrt_metrics.get('FLV', np.nan):>+11.2f} {kge_log_metrics.get('FLV', np.nan):>+11.2f} {kge_np_metrics.get('FLV', np.nan):>+11.2f}")

print("\n" + "-" * 150)
print("Notes:")
print("  • NSE/NSE_log/NSE_inv/NSE_sqrt/KGE are maximized (higher=better, optimal=1.0)")
print("  • SDEB/RMSE/MAE are minimized (lower=better)")
print("  • r = correlation, α = variability ratio, β = bias ratio (all optimal = 1.0)")
print("  • KGE > -0.41 indicates improvement over mean benchmark (Knoben et al., 2019)")
print("  • SDEB combines chronological timing + FDC shape + bias penalty")

# %% [markdown]
# ### NSE vs KGE vs SDEB: Which to Use?
#
# **Key differences:**
#
# | Aspect | NSE | KGE | SDEB |
# |--------|-----|-----|------|
# | **Decomposition** | None (single score) | r, α, β components | Chronological + FDC + bias |
# | **Benchmark** | NSE=0 means model = mean | KGE=-0.41 means model = mean | Lower is better |
# | **Bias sensitivity** | Squared errors | Linear bias term | Explicit bias penalty |
# | **Timing sensitivity** | High (chronological order matters) | High (chronological order matters) | **Reduced** (FDC component) |
# | **Interpretability** | "Variance explained" | "Correlation + variability + bias" | "Timing + distribution + bias" |
#
# **When to use KGE:**
# - When you want to understand *why* a model performs well/poorly (diagnose components)
# - When you need to balance timing, variability, and bias explicitly
# - For comparing models across different catchments (more interpretable decomposition)
#
# **When to use NSE:**
# - When benchmarking against existing literature (NSE is more widely used)
# - When you primarily care about variance explained
# - For consistency with regulatory/reporting requirements
#
# **When to use SDEB:**
# - When timing errors are uncertain or less critical (FDC component reduces timing sensitivity)
# - When you need to match both chronological patterns AND flow distribution
# - For catchments where phase shifts are common (e.g., due to routing delays)
# - When you want explicit control over bias through the bias penalty term

# %% [markdown]
# ### Visual Comparison: All Objective Functions
#
# This figure combines all NSE-based, composite, and KGE-based calibrations for direct comparison.
# Use the legend to toggle individual datasets on/off across all subplots.

# %%
# Define colors for each calibration
colors = {
    'observed': 'black',
    # NSE-based
    'nse': '#E41A1C',      # Red
    'log': '#377EB8',      # Blue  
    'inv': '#4DAF4A',      # Green
    'sqrt': '#984EA3',     # Purple
    'sdeb': '#FF7F00',     # Orange
    # KGE-based
    'kge': '#A65628',      # Brown
    'kge_inv': '#F781BF',  # Pink
    'kge_sqrt': '#999999', # Gray
    'kge_log': '#66C2A5',  # Teal
    'kge_np': '#FC8D62',   # Salmon
    # KGE_np transformed
    'kge_np_inv': '#B2DF8A',   # Light green
    'kge_np_sqrt': '#CAB2D6',  # Light purple
    'kge_np_log': '#FDBF6F'    # Light orange
}

# Prepare sorted flows for FDC
obs_sorted = np.sort(obs_flow)[::-1]
nse_sorted = np.sort(sim_flow)[::-1]
log_sorted = np.sort(log_sim_flow)[::-1]
inv_sorted = np.sort(inv_sim_flow)[::-1]
sqrt_sorted = np.sort(sqrt_sim_flow)[::-1]
sdeb_sorted = np.sort(sdeb_sim_flow)[::-1]
kge_sorted = np.sort(kge_sim_flow)[::-1]
kge_inv_sorted = np.sort(kge_inv_sim_flow)[::-1]
kge_sqrt_sorted = np.sort(kge_sqrt_sim_flow)[::-1]
kge_log_sorted = np.sort(kge_log_sim_flow)[::-1]
kge_np_sorted = np.sort(kge_np_sim_flow)[::-1]
kge_np_inv_sorted = np.sort(kge_np_inv_sim_flow)[::-1]
kge_np_sqrt_sorted = np.sort(kge_np_sqrt_sim_flow)[::-1]
kge_np_log_sorted = np.sort(kge_np_log_sim_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
max_flow = max(obs_flow.max(), sim_flow.max())

# %%
# Combined comparison: All NSE-based and KGE-based calibrations (single column layout)
fig_all = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        'Full Hydrograph (Log Scale)',
        'Full Hydrograph (Linear Scale)',
        'Flow Duration Curves',
        'Scatter: All Calibrations vs Observed'
    ),
    specs=[
        [{"type": "scatter"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}]
    ],
    vertical_spacing=0.06
)

# 1. Full hydrograph (log scale) - All calibrations with legendgroup for linked legend selection
# Observed
fig_all.add_trace(go.Scatter(x=comparison.index, y=obs_flow, name='Observed',
              legendgroup='observed', line=dict(color=colors['observed'], width=2)), row=1, col=1)

# NSE-based objectives
fig_all.add_trace(go.Scatter(x=comparison.index, y=sim_flow, name='NSE',
              legendgroup='nse', line=dict(color=colors['nse'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=log_sim_flow, name='NSE_log',
              legendgroup='log', line=dict(color=colors['log'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=inv_sim_flow, name='NSE_inv',
              legendgroup='inv', line=dict(color=colors['inv'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=sqrt_sim_flow, name='NSE_sqrt',
              legendgroup='sqrt', line=dict(color=colors['sqrt'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=sdeb_sim_flow, name='SDEB',
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=1)), row=1, col=1)

# KGE-based objectives
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_sim_flow, name='KGE',
              legendgroup='kge', line=dict(color=colors['kge'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_inv_sim_flow, name='KGE(1/Q)',
              legendgroup='kge_inv', line=dict(color=colors['kge_inv'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_sqrt_sim_flow, name='KGE(√Q)',
              legendgroup='kge_sqrt', line=dict(color=colors['kge_sqrt'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_log_sim_flow, name='KGE(log)',
              legendgroup='kge_log', line=dict(color=colors['kge_log'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_sim_flow, name='KGE_np',
              legendgroup='kge_np', line=dict(color=colors['kge_np'], width=1)), row=1, col=1)

# KGE_np transformed variants
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_inv_sim_flow, name='KGE_np(1/Q)',
              legendgroup='kge_np_inv', line=dict(color=colors['kge_np_inv'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_sqrt_sim_flow, name='KGE_np(√Q)',
              legendgroup='kge_np_sqrt', line=dict(color=colors['kge_np_sqrt'], width=1)), row=1, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_log_sim_flow, name='KGE_np(log)',
              legendgroup='kge_np_log', line=dict(color=colors['kge_np_log'], width=1)), row=1, col=1)

# 2. Full hydrograph (linear scale) - All calibrations
# Observed
fig_all.add_trace(go.Scatter(x=comparison.index, y=obs_flow,
              legendgroup='observed', line=dict(color=colors['observed'], width=2), showlegend=False), row=2, col=1)

# NSE-based objectives
fig_all.add_trace(go.Scatter(x=comparison.index, y=sim_flow,
              legendgroup='nse', line=dict(color=colors['nse'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=log_sim_flow,
              legendgroup='log', line=dict(color=colors['log'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=inv_sim_flow,
              legendgroup='inv', line=dict(color=colors['inv'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=sqrt_sim_flow,
              legendgroup='sqrt', line=dict(color=colors['sqrt'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=sdeb_sim_flow,
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=1), showlegend=False), row=2, col=1)

# KGE-based objectives
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_sim_flow,
              legendgroup='kge', line=dict(color=colors['kge'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_inv_sim_flow,
              legendgroup='kge_inv', line=dict(color=colors['kge_inv'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_sqrt_sim_flow,
              legendgroup='kge_sqrt', line=dict(color=colors['kge_sqrt'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_log_sim_flow,
              legendgroup='kge_log', line=dict(color=colors['kge_log'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_sim_flow,
              legendgroup='kge_np', line=dict(color=colors['kge_np'], width=1), showlegend=False), row=2, col=1)

# KGE_np transformed variants
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_inv_sim_flow,
              legendgroup='kge_np_inv', line=dict(color=colors['kge_np_inv'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_sqrt_sim_flow,
              legendgroup='kge_np_sqrt', line=dict(color=colors['kge_np_sqrt'], width=1), showlegend=False), row=2, col=1)
fig_all.add_trace(go.Scatter(x=comparison.index, y=kge_np_log_sim_flow,
              legendgroup='kge_np_log', line=dict(color=colors['kge_np_log'], width=1), showlegend=False), row=2, col=1)

# 3. Flow Duration Curves - All calibrations
# Observed
fig_all.add_trace(go.Scatter(x=exceedance, y=obs_sorted, name='Observed',
              legendgroup='observed', line=dict(color=colors['observed'], width=2), showlegend=False), row=3, col=1)

# NSE-based objectives
fig_all.add_trace(go.Scatter(x=exceedance, y=nse_sorted, name='NSE',
              legendgroup='nse', line=dict(color=colors['nse'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=log_sorted, name='NSE_log',
              legendgroup='log', line=dict(color=colors['log'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=inv_sorted, name='NSE_inv',
              legendgroup='inv', line=dict(color=colors['inv'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=sqrt_sorted, name='NSE_sqrt',
              legendgroup='sqrt', line=dict(color=colors['sqrt'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=sdeb_sorted, name='SDEB',
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=1.5), showlegend=False), row=3, col=1)

# KGE-based objectives
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_sorted, name='KGE',
              legendgroup='kge', line=dict(color=colors['kge'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_inv_sorted, name='KGE(1/Q)',
              legendgroup='kge_inv', line=dict(color=colors['kge_inv'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_sqrt_sorted, name='KGE(√Q)',
              legendgroup='kge_sqrt', line=dict(color=colors['kge_sqrt'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_log_sorted, name='KGE(log)',
              legendgroup='kge_log', line=dict(color=colors['kge_log'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_np_sorted, name='KGE_np',
              legendgroup='kge_np', line=dict(color=colors['kge_np'], width=1.5), showlegend=False), row=3, col=1)

# KGE_np transformed variants
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_np_inv_sorted, name='KGE_np(1/Q)',
              legendgroup='kge_np_inv', line=dict(color=colors['kge_np_inv'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_np_sqrt_sorted, name='KGE_np(√Q)',
              legendgroup='kge_np_sqrt', line=dict(color=colors['kge_np_sqrt'], width=1.5), showlegend=False), row=3, col=1)
fig_all.add_trace(go.Scatter(x=exceedance, y=kge_np_log_sorted, name='KGE_np(log)',
              legendgroup='kge_np_log', line=dict(color=colors['kge_np_log'], width=1.5), showlegend=False), row=3, col=1)

# 4. Scatter plot - All calibrations
# NSE-based objectives
fig_all.add_trace(go.Scatter(x=obs_flow, y=sim_flow, mode='markers', name='NSE',
              legendgroup='nse', marker=dict(color=colors['nse'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=log_sim_flow, mode='markers', name='NSE_log',
              legendgroup='log', marker=dict(color=colors['log'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=inv_sim_flow, mode='markers', name='NSE_inv',
              legendgroup='inv', marker=dict(color=colors['inv'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=sqrt_sim_flow, mode='markers', name='NSE_sqrt',
              legendgroup='sqrt', marker=dict(color=colors['sqrt'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=sdeb_sim_flow, mode='markers', name='SDEB',
              legendgroup='sdeb', marker=dict(color=colors['sdeb'], size=2, opacity=0.3), showlegend=False), row=4, col=1)

# KGE-based objectives
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_sim_flow, mode='markers', name='KGE',
              legendgroup='kge', marker=dict(color=colors['kge'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_inv_sim_flow, mode='markers', name='KGE(1/Q)',
              legendgroup='kge_inv', marker=dict(color=colors['kge_inv'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_sqrt_sim_flow, mode='markers', name='KGE(√Q)',
              legendgroup='kge_sqrt', marker=dict(color=colors['kge_sqrt'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_log_sim_flow, mode='markers', name='KGE(log)',
              legendgroup='kge_log', marker=dict(color=colors['kge_log'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_np_sim_flow, mode='markers', name='KGE_np',
              legendgroup='kge_np', marker=dict(color=colors['kge_np'], size=2, opacity=0.3), showlegend=False), row=4, col=1)

# KGE_np transformed variants
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_np_inv_sim_flow, mode='markers', name='KGE_np(1/Q)',
              legendgroup='kge_np_inv', marker=dict(color=colors['kge_np_inv'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_np_sqrt_sim_flow, mode='markers', name='KGE_np(√Q)',
              legendgroup='kge_np_sqrt', marker=dict(color=colors['kge_np_sqrt'], size=2, opacity=0.3), showlegend=False), row=4, col=1)
fig_all.add_trace(go.Scatter(x=obs_flow, y=kge_np_log_sim_flow, mode='markers', name='KGE_np(log)',
              legendgroup='kge_np_log', marker=dict(color=colors['kge_np_log'], size=2, opacity=0.3), showlegend=False), row=4, col=1)

# 1:1 line
fig_all.add_trace(go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines',
              line=dict(color='black', dash='dash'), showlegend=False), row=4, col=1)

# Update axes
fig_all.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig_all.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig_all.update_yaxes(title_text="Flow (ML/day)", type="log", row=3, col=1)
fig_all.update_xaxes(title_text="Exceedance (%)", row=3, col=1)
fig_all.update_yaxes(title_text="Simulated (ML/day)", type="log", row=4, col=1)
fig_all.update_xaxes(title_text="Observed (ML/day)", type="log", row=4, col=1)

fig_all.update_layout(
    title="<b>Comparison: All Objective Functions (NSE + Composite + KGE)</b><br>" +
          "<sup>Click legend items to toggle visibility across all subplots | NSE-based: solid warm colors | Composite: orange | KGE-based: muted colors</sup>",
    height=2200,
    width=1600,
    legend=dict(
        orientation='h', 
        y=1.02, 
        x=0.5, 
        xanchor='center',
        font=dict(size=10),
        itemwidth=40
    )
)
fig_all.show()

print("\nLegend guide:")
print("  NSE-based: NSE (red), NSE_log (blue), NSE_inv (green), NSE_sqrt (purple)")
print("  Composite: SDEB (orange)")
print("  KGE-based: KGE (brown), KGE(1/Q) (pink), KGE(√Q) (gray), KGE(log) (teal), KGE_np (salmon)")
print("  KGE_np transformed: KGE_np(1/Q) (light green), KGE_np(√Q) (light purple), KGE_np(log) (light orange)")

# %% [markdown]
# ### Key Observations: NSE-Based Objectives
#
# **NSE (Q) Calibration** (red):
# - Best at matching peak flows
# - May miss low flow dynamics
# - Highest standard NSE
#
# **NSE_log Calibration** (blue):
# - Balances high and low flow performance
# - Good all-round choice for general applications
#
# **NSE_inv Calibration** (green):
# - Heavily emphasizes low flows
# - Best for drought analysis and environmental flows
# - May sacrifice peak accuracy
#
# **NSE_sqrt Calibration** (purple):
# - Moderate emphasis on low flows
# - Compromise between NSE and NSE_log
#
# **SDEB Calibration** (orange):
# - Combines chronological timing + FDC shape + bias penalty
# - Used in Australian SOURCE platform
# - Reduces sensitivity to peak timing errors
#
# ### Key Observations: KGE-Based Objectives
#
# **KGE Calibration** (brown):
# - Standard KGE on original flows (2012 variant)
# - Emphasizes high flows (like NSE) but with interpretable components
# - Check r, α, β to diagnose model behavior
#
# **KGE(1/Q) Calibration** (pink):
# - Inverse transformation, heavily emphasizes low flows
# - Useful for drought indices and environmental flow studies
# - May sacrifice peak flow accuracy
#
# **KGE(√Q) Calibration** (gray):
# - Balanced emphasis between high and low flows
# - **Recommended for general water resources applications**
# - Good compromise without the numerical issues of log transform
#
# **KGE(log) Calibration** (teal):
# - Log transformation (use with caution!)
# - Santos et al. (2018) showed potential numerical instabilities
# - Consider sqrt or inverse transforms instead
#
# **KGE Non-parametric** (salmon):
# - Uses Spearman rank correlation (robust to outliers)
# - Better for catchments with extreme events
# - More robust to non-Gaussian error distributions
#
# ### Which Objective to Choose?
#
# | Application | Recommended Objectives | Reason |
# |-------------|------------------------|--------|
# | Flood forecasting | NSE, KGE | Peaks matter most |
# | Water supply | NSE_log, KGE_sqrt, NSE_sqrt | Balance matters |
# | Drought/low flow | NSE_inv, KGE_inv | Low flows critical |
# | Environmental flows | NSE_log, KGE_inv | Baseflow important |
# | General purpose | NSE_log, KGE_sqrt, SDEB | Best balance |
# | Uncertain timing | SDEB | Reduces timing sensitivity |
# | Flashy catchments | KGE_np | Robust to outliers |
# | Model diagnostics | KGE family | Interpretable components |

# %% [markdown]
# ### Parameter Comparison Across All Objective Functions

# %%
# Compare parameters across all calibrations
print("=" * 180)
print("PARAMETER COMPARISON ACROSS ALL OBJECTIVE FUNCTIONS")
print("=" * 180)

# All objective functions in one table (NSE-based | KGE-based | KGE_np-based)
print(f"\n{'':^10} |{'--- NSE-Based ---':^62}|{'--- KGE-Based ---':^62}|{'--- KGE_np Transformed ---':^40}")
print(f"{'Parameter':<10} {'NSE':>10} {'NSE_log':>10} {'NSE_inv':>10} {'NSE_sqrt':>10} {'SDEB':>10} | {'KGE':>10} {'KGE_inv':>10} {'KGE_sqrt':>10} {'KGE_log':>10} {'KGE_np':>10} | {'KGE_np_inv':>12} {'KGE_np_sqrt':>12} {'KGE_np_log':>12}")
print("-" * 180)

for param in result.best_parameters.keys():
    nse_val = result.best_parameters[param]
    log_val = log_result.best_parameters[param]
    inv_val = inv_result.best_parameters[param]
    sqrt_val = sqrt_result.best_parameters[param]
    sdeb_val = sdeb_result.best_parameters[param]
    kge_val = kge_result.best_parameters[param]
    kge_inv_val = kge_inv_result.best_parameters[param]
    kge_sqrt_val = kge_sqrt_result.best_parameters[param]
    kge_log_val = kge_log_result.best_parameters[param]
    kge_np_val = kge_np_result.best_parameters[param]
    kge_np_inv_val = kge_np_inv_result.best_parameters[param]
    kge_np_sqrt_val = kge_np_sqrt_result.best_parameters[param]
    kge_np_log_val = kge_np_log_result.best_parameters[param]
    print(f"{param:<10} {nse_val:>10.4f} {log_val:>10.4f} {inv_val:>10.4f} {sqrt_val:>10.4f} {sdeb_val:>10.4f} | {kge_val:>10.4f} {kge_inv_val:>10.4f} {kge_sqrt_val:>10.4f} {kge_log_val:>10.4f} {kge_np_val:>10.4f} | {kge_np_inv_val:>12.4f} {kge_np_sqrt_val:>12.4f} {kge_np_log_val:>12.4f}")

# %%
# Visual comparison of calibrated parameters across all objective functions
# Each parameter shown as position within bounds (0% = lower bound, 100% = upper bound)

param_names = list(result.best_parameters.keys())
n_params = len(param_names)

# Calculate normalized positions (0-100%) for each parameter
def normalize_param(value, param_name):
    low, high = param_bounds[param_name]
    return (value - low) / (high - low) * 100

# Format text to show actual value (smart formatting based on magnitude)
def format_value(v):
    if abs(v) >= 100:
        return f'{v:.1f}'
    elif abs(v) >= 10:
        return f'{v:.2f}'
    elif abs(v) >= 1:
        return f'{v:.3f}'
    else:
        return f'{v:.4f}'

# Prepare data for all methods (NSE-based + KGE-based + KGE_np-based)
# Row 1: NSE-based (5), Row 2: KGE-based (4 + 1 empty), Row 3: KGE_np family (4 + 1 empty)
methods = ['NSE', 'NSE_log', 'NSE_inv', 'NSE_sqrt', 'SDEB',
           'KGE', 'KGE_inv', 'KGE_sqrt', 'KGE_log', '',
           'KGE_np', 'KGE_np_inv', 'KGE_np_sqrt', 'KGE_np_log', '']
results_list = [result, log_result, inv_result, sqrt_result, sdeb_result,
                kge_result, kge_inv_result, kge_sqrt_result, kge_log_result, None,
                kge_np_result, kge_np_inv_result, kge_np_sqrt_result, kge_np_log_result, None]
method_colors = [colors['nse'], colors['log'], colors['inv'], colors['sqrt'], colors['sdeb'],
                 colors['kge'], colors['kge_inv'], colors['kge_sqrt'], colors['kge_log'], None,
                 colors['kge_np'], colors['kge_np_inv'], colors['kge_np_sqrt'], colors['kge_np_log'], None]

# Create figure with subplots - 3 rows x 5 columns
# Row 1: NSE-based (5), Row 2: KGE-based (4), Row 3: KGE_np family (4)
fig = make_subplots(
    rows=3, cols=5,
    subplot_titles=methods,
    horizontal_spacing=0.02,
    vertical_spacing=0.10,
    row_titles=['NSE-Based', 'KGE-Based', 'KGE_np (Non-Parametric)']
)

for idx, (method_result, color) in enumerate(zip(results_list, method_colors)):
    # Skip empty slots (None entries)
    if method_result is None:
        continue
    
    # Determine row and column (0-4 -> row 1, 5-9 -> row 2, 10-14 -> row 3)
    row = (idx // 5) + 1
    col = (idx % 5) + 1
    
    # Get actual values and normalized values for this method
    actual_values = [method_result.best_parameters[p] for p in param_names]
    norm_values = [normalize_param(v, p) for v, p in zip(actual_values, param_names)]
    
    bar_text = [format_value(v) for v in actual_values]
    
    # Create hover text with full details
    hover_text = [
        f'{p}<br>Value: {format_value(v)}<br>Position: {n:.1f}%<br>Bounds: [{param_bounds[p][0]:.3f}, {param_bounds[p][1]:.3f}]'
        for p, v, n in zip(param_names, actual_values, norm_values)
    ]
    
    # Create horizontal bar for each parameter
    fig.add_trace(
        go.Bar(
            y=param_names,
            x=norm_values,
            orientation='h',
            marker_color=color,
            text=bar_text,
            textposition='inside',
            textfont=dict(color='white', size=8),
            showlegend=False,
            hovertext=hover_text,
            hoverinfo='text'
        ),
        row=row, col=col
    )
    
    # Add vertical line at 50%
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

# Update layout
fig.update_xaxes(range=[0, 100], tickvals=[0, 50, 100], ticktext=['0%', '50%', '100%'])
fig.update_yaxes(categoryorder='array', categoryarray=param_names[::-1])  # Reverse for top-to-bottom

# Only show y-axis labels on first column
for row in [1, 2, 3]:
    for col in range(2, 6):
        fig.update_yaxes(showticklabels=False, row=row, col=col)

fig.update_layout(
    title="<b>Visual Comparison: Calibrated Parameters Across All Objective Functions</b><br>" +
          "<sup>Bar position = % within bounds | Bar text = actual parameter value | Hover for details</sup>",
    height=1600,
    width=1600,
    bargap=0.15
)

fig.show()

print("\nInterpretation:")
print("  - Bar length shows position within parameter bounds (0% = lower, 100% = upper)")
print("  - Numbers on bars show the actual calibrated parameter values")
print("  - Hover over bars for full details (value, position %, and bounds)")
print("  - Parameters at extremes (near 0% or 100%) may indicate bounds should be reviewed")
print("  - Row 1: NSE-based | Row 2: KGE-based | Row 3: KGE_np Transformed")

# %% [markdown]
# ---
# ## Summary
#
# Congratulations! You've successfully:
#
# 1. **Loaded hydrological data** (rainfall, PET, observed flow)
# 2. **Set up the Sacramento model** with automatic unit conversion
# 3. **Calibrated with standard NSE** (emphasizes high flows)
# 4. **Explored flow transformations** (log, inverse, sqrt)
# 5. **Calibrated with multiple NSE objectives** (NSE_log, NSE_inv, NSE_sqrt)
# 6. **Calibrated with composite objective** (SDEB)
# 7. **Calibrated with KGE family** (KGE, KGE_inv, KGE_sqrt, KGE_log, KGE_np)
# 8. **Compared results** to understand the trade-offs
#
# ### Key Takeaways
#
# #### NSE vs KGE
# - **NSE** is the traditional metric (variance explained), biased towards high flows
# - **KGE** decomposes into interpretable components (r, α, β) - useful for diagnostics
# - **KGE > -0.41** indicates improvement over mean benchmark
#
# #### Flow Transformations (work with both NSE and KGE)
# - **log(Q)**: Balances all flow ranges (use cautiously with KGE)
# - **1/Q**: Heavily emphasizes low flows
# - **√Q**: Moderate emphasis on low flows (recommended for KGE)
#
# #### Composite Objectives
# - **SDEB**: Combines chronological timing + FDC shape + bias penalty (reduces timing sensitivity)
#
# #### Special KGE Variants
# - **KGE Non-parametric**: Robust to outliers, uses Spearman correlation
#
# ### Recommended Objectives by Application
#
# | Application | NSE-based | Composite | KGE-based |
# |-------------|-----------|-----------|-----------|
# | Flood forecasting | NSE (Q) | - | KGE |
# | Water supply | NSE_log, NSE_sqrt | - | KGE_sqrt |
# | Drought/low flow | NSE_inv | - | KGE_inv |
# | Environmental flows | NSE_log | - | KGE_inv, KGE_sqrt |
# | General purpose | NSE_log | SDEB | KGE_sqrt, KGE_np |
# | Uncertain timing | - | SDEB | - |
# | Flashy catchments | - | - | KGE_np |
#
# The detailed calibration results for all objective functions are shown in the comprehensive comparison table in the section above.

# %%
# Save calibrated parameters
params_file = OUTPUT_DIR / 'quickstart_calibrated_params.csv'
pd.DataFrame([result.best_parameters]).to_csv(params_file, index=False)
print(f"Calibrated parameters saved to: {params_file}")

# %% [markdown]
# ---
# ## Step 8: Comprehensive Model Evaluation
#
# To fairly compare models calibrated with different objective functions, we evaluate
# **multiple aspects** of model performance across six categories:
#
# | Category | Metrics Evaluated |
# |----------|-------------------|
# | **Overall Efficiency** | NSE, KGE, KGE_np |
# | **KGE Components** | r (correlation), α (variability), β (bias) |
# | **Flow Regime Specific** | NSE(log Q), NSE(1/Q), NSE(√Q), KGE(1/Q) |
# | **FDC-Based** | Peak, High, Mid, Low, Very Low segment biases |
# | **Hydrological Signatures** | Q95, Q50, Q5, Flashiness, Baseflow Index, Flow frequencies |
# | **Volume & Timing** | PBIAS, RMSE, Pearson r, Spearman ρ |

# %%
# The canonical diagnostic suite is imported from pyrrm.analysis.diagnostics
# (compute_diagnostics already imported above)
print("Using canonical compute_diagnostics() from pyrrm.analysis.diagnostics")
print(f"  {len(DIAGNOSTIC_GROUPS)} metric groups, {sum(len(v) for v in DIAGNOSTIC_GROUPS.values())} metrics total")

# %%
# Compute comprehensive evaluation for all calibrations
print("Computing comprehensive evaluation for all calibrations...")
print("\nNSE-based objectives:")

eval_nse = compute_diagnostics(sim_flow, obs_flow)
eval_log = compute_diagnostics(log_sim_flow, obs_flow)
eval_inv = compute_diagnostics(inv_sim_flow, obs_flow)
eval_sqrt = compute_diagnostics(sqrt_sim_flow, obs_flow)
eval_sdeb = compute_diagnostics(sdeb_sim_flow, obs_flow)

print("KGE-based objectives:")
eval_kge = compute_diagnostics(kge_sim_flow, obs_flow)
eval_kge_inv = compute_diagnostics(kge_inv_sim_flow, obs_flow)
eval_kge_sqrt = compute_diagnostics(kge_sqrt_sim_flow, obs_flow)
eval_kge_log = compute_diagnostics(kge_log_sim_flow, obs_flow)
eval_kge_np = compute_diagnostics(kge_np_sim_flow, obs_flow)

print("Non-parametric KGE objectives with transformations:")
eval_kge_np_inv = compute_diagnostics(kge_np_inv_sim_flow, obs_flow)
eval_kge_np_sqrt = compute_diagnostics(kge_np_sqrt_sim_flow, obs_flow)
eval_kge_np_log = compute_diagnostics(kge_np_log_sim_flow, obs_flow)

print("Done! Creating comparison tables...")

# %%
# Create comprehensive comparison DataFrame for NSE-based objectives (canonical column names)
comparison_data_nse = {
    'NSE': eval_nse,
    'NSE_log': eval_log,
    'NSE_inv': eval_inv,
    'NSE_sqrt': eval_sqrt,
    'SDEB': eval_sdeb
}

# Create comprehensive comparison DataFrame for KGE-based objectives (canonical column names)
comparison_data_kge = {
    'KGE': eval_kge,
    'KGE_inv': eval_kge_inv,
    'KGE_sqrt': eval_kge_sqrt,
    'KGE_log': eval_kge_log,
    'KGE_np': eval_kge_np,
    'KGE_np_inv': eval_kge_np_inv,
    'KGE_np_sqrt': eval_kge_np_sqrt,
    'KGE_np_log': eval_kge_np_log
}

# Combine all comparisons
comparison_data = {**comparison_data_nse, **comparison_data_kge}

# Convert to DataFrame
comp_df = pd.DataFrame(comparison_data)
comp_df_nse = pd.DataFrame(comparison_data_nse)
comp_df_kge = pd.DataFrame(comparison_data_kge)

# %% [markdown]
# ### Comprehensive Metric Comparison Table
#
# The table below shows all diagnostic metrics with color coding based on performance:
# - **Green** = best performance for that metric
# - **Red** = worst performance for that metric
# - Color intensity indicates relative performance within each metric row

# %%
# Create color-coded table using Plotly
# Define metric categories and their optimal values
def get_cell_color(value, metric_name, all_values):
    """Get color for a cell based on metric type and performance."""
    if pd.isna(value):
        return 'rgb(240, 240, 240)'

    higher_better = [
        'NSE', 'NSE_log', 'NSE_sqrt', 'NSE_inv',
        'KGE', 'KGE_log', 'KGE_sqrt', 'KGE_inv',
        'KGE_np', 'KGE_np_log', 'KGE_np_sqrt', 'KGE_np_inv',
    ]

    ideal_one = [
        'KGE_r', 'KGE_alpha', 'KGE_beta',
        'KGE_log_r', 'KGE_log_alpha', 'KGE_log_beta',
        'KGE_sqrt_r', 'KGE_sqrt_alpha', 'KGE_sqrt_beta',
        'KGE_inv_r', 'KGE_inv_alpha', 'KGE_inv_beta',
        'KGE_np_r', 'KGE_np_alpha', 'KGE_np_beta',
        'KGE_np_log_r', 'KGE_np_log_alpha', 'KGE_np_log_beta',
        'KGE_np_sqrt_r', 'KGE_np_sqrt_alpha', 'KGE_np_sqrt_beta',
        'KGE_np_inv_r', 'KGE_np_inv_alpha', 'KGE_np_inv_beta',
    ]

    lower_better = ['RMSE', 'MAE']

    zero_better = ['PBIAS', 'FHV', 'FMV', 'FLV',
                    'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5']

    if metric_name in higher_better:
        min_val = all_values.min()
        max_val = all_values.max()
        if max_val > min_val:
            score = (value - min_val) / (max_val - min_val)
        else:
            score = 0.5
    elif metric_name in ideal_one:
        # Score based on distance from 1.0
        score = 1.0 - min(abs(value - 1.0), 1.0)
    elif metric_name in lower_better:
        max_val = all_values.max()
        if max_val > 0:
            score = 1.0 - (value / max_val)
        else:
            score = 1.0
    elif metric_name in zero_better:
        # For zero-centered metrics: 0 is ideal, deviations are bad
        abs_values = np.abs(all_values)
        max_abs = abs_values.max()
        abs_value = abs(value)
        
        if max_abs > 0:
            # Score decreases as we move away from 0
            normalized_distance = abs_value / max_abs
            score = 1.0 - (normalized_distance ** 0.7)
        else:
            score = 1.0
            normalized_distance = 0.0
    else:
        # Default: higher is better
        min_val = all_values.min()
        max_val = all_values.max()
        if max_val > min_val:
            score = (value - min_val) / (max_val - min_val)
        else:
            score = 0.5
        normalized_distance = None
    
    # Color from red (0) to green (1) via yellow (0.5)
    # For zero-centered metrics, use a more intuitive color scale
    if metric_name in zero_better and normalized_distance is not None:
        # Color scale that emphasizes closeness to 0
        if normalized_distance < 0.1:
            # Very close to 0 (< 10% of max deviation) - bright green
            r, g, b = 150, 255, 150
        elif normalized_distance < 0.3:
            # Close to 0 (10-30% of max) - light green
            r, g, b = 200, 255, 200
        elif normalized_distance < 0.5:
            # Moderate deviation (30-50% of max) - yellow
            r, g, b = 255, 255, 200
        elif normalized_distance < 0.7:
            # Large deviation (50-70% of max) - orange
            r, g, b = 255, 200, 150
        else:
            # Very large deviation (> 70% of max) - red
            r, g, b = 255, 150, 150
    else:
        # Standard color scale for other metrics
        if score < 0.5:
            # Red to yellow
            r = 255
            g = int(255 * (score * 2))
            b = 0
        else:
            # Yellow to green
            r = int(255 * (2 - score * 2))
            g = 255
            b = 0
        
        # Lighten the background
        r = int(200 + (r * 0.2))
        g = int(200 + (g * 0.2))
        b = int(200 + (b * 0.2))
    
    return f'rgb({r},{g},{b})'

# Prepare data for Plotly table
table_data = []
cell_colors = []

for metric in comp_df.index:
    row_data = [metric]
    row_colors = ['white']  # Header column
    
    for col in comp_df.columns:
        value = comp_df.loc[metric, col]
        color = get_cell_color(value, metric, comp_df.loc[metric])
        row_colors.append(color)
        
        # Format value
        if pd.isna(value):
            row_data.append('N/A')
        elif abs(value) >= 1000:
            row_data.append(f'{value:.2f}')
        elif 'Error' in metric or 'Bias' in metric:
            row_data.append(f'{value:+.2f}')
        else:
            row_data.append(f'{value:.4f}')
    
    table_data.append(row_data)
    cell_colors.append(row_colors)

# Create header
header = ['Metric'] + list(comp_df.columns)

# Transpose for Plotly (columns become rows)
table_values = list(zip(*table_data))
color_values = list(zip(*cell_colors))

# Create Plotly table
fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=header,
        fill_color='lightblue',
        align='left',
        font=dict(size=11, color='black')
    ),
    cells=dict(
        values=table_values,
        fill_color=color_values,
        align='left',
        font=dict(size=10),
        height=25
    )
)])

fig_table.update_layout(
    title="<b>Comprehensive Metric Comparison (Color-Coded by Performance)</b><br>" +
          "<sup>Green = best performance, Red = worst performance for each metric</sup>",
    height=min(800, len(comp_df) * 30 + 100),
    width=1400
)

fig_table.show()

# Also print a text version
print("\n" + "=" * 120)
print("Text version of comparison table:")
print("=" * 120)
print(comp_df.round(4).to_string())

# %% [markdown]
# ### HiPlot Parallel Coordinate Visualization
#
# HiPlot provides an interactive parallel coordinate plot that allows you to:
# - **Filter**: Click and drag on any axis to filter data points
# - **Reorder**: Drag column headers to reorder axes
# - **Explore**: Hover over lines to see individual calibration details
#
# This visualization helps identify trade-offs between different objective functions
# and understand which metrics are correlated or in conflict.

# %%
# HiPlot Parallel Coordinate Plot for Comprehensive Metrics
# Data is transformed so that HIGHER = BETTER for all metrics

try:
    import hiplot as hip
    
    higher_better = [
        'NSE', 'NSE_log', 'NSE_sqrt', 'NSE_inv',
        'KGE', 'KGE_log', 'KGE_sqrt', 'KGE_inv',
        'KGE_np', 'KGE_np_log', 'KGE_np_sqrt', 'KGE_np_inv',
    ]
    ideal_one = [
        'KGE_r', 'KGE_alpha', 'KGE_beta',
        'KGE_log_r', 'KGE_log_alpha', 'KGE_log_beta',
        'KGE_sqrt_r', 'KGE_sqrt_alpha', 'KGE_sqrt_beta',
        'KGE_inv_r', 'KGE_inv_alpha', 'KGE_inv_beta',
        'KGE_np_r', 'KGE_np_alpha', 'KGE_np_beta',
        'KGE_np_log_r', 'KGE_np_log_alpha', 'KGE_np_log_beta',
        'KGE_np_sqrt_r', 'KGE_np_sqrt_alpha', 'KGE_np_sqrt_beta',
        'KGE_np_inv_r', 'KGE_np_inv_alpha', 'KGE_np_inv_beta',
    ]
    lower_better = ['RMSE', 'MAE']
    zero_better = ['PBIAS', 'FHV', 'FMV', 'FLV',
                    'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5']
    
    def transform_metric(values, metric_name):
        """Transform metric values so higher = better, scaled 0-1."""
        values = values.copy()
        
        # Handle NaN
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return values
        
        valid_values = values[valid_mask]
        
        if metric_name in higher_better:
            # Normalize: (x - min) / (max - min)
            min_val, max_val = valid_values.min(), valid_values.max()
            if max_val > min_val:
                values[valid_mask] = (valid_values - min_val) / (max_val - min_val)
            else:
                values[valid_mask] = 1.0
                
        elif metric_name in ideal_one:
            # Score = 1 - |value - 1|, then normalize
            deviations = np.abs(valid_values - 1.0)
            max_dev = deviations.max()
            if max_dev > 0:
                # Transform so 1.0 gives score 1, max deviation gives score 0
                values[valid_mask] = 1.0 - (deviations / max_dev)
            else:
                values[valid_mask] = 1.0
                
        elif metric_name in lower_better:
            # Invert and normalize: lower original = higher score
            min_val, max_val = valid_values.min(), valid_values.max()
            if max_val > min_val:
                values[valid_mask] = 1.0 - (valid_values - min_val) / (max_val - min_val)
            else:
                values[valid_mask] = 1.0
                
        elif metric_name in zero_better:
            # Score based on distance from 0: closer to 0 = higher score
            abs_values = np.abs(valid_values)
            max_abs = abs_values.max()
            if max_abs > 0:
                values[valid_mask] = 1.0 - (abs_values / max_abs)
            else:
                values[valid_mask] = 1.0
        else:
            # Default: assume higher is better
            min_val, max_val = valid_values.min(), valid_values.max()
            if max_val > min_val:
                values[valid_mask] = (valid_values - min_val) / (max_val - min_val)
            else:
                values[valid_mask] = 0.5
        
        return values
    
    # Transform the comparison DataFrame
    # Transpose so each row is a calibration method
    transformed_df = comp_df.T.copy()
    
    # Apply transformations to each metric (column in transposed df)
    for metric in transformed_df.columns:
        transformed_df[metric] = transform_metric(transformed_df[metric].values, metric)
    
    # Reset index and rename
    hiplot_df = transformed_df.reset_index()
    hiplot_df.columns = ['Calibration'] + list(comp_df.index)
    
    # Convert to list of dictionaries for HiPlot
    hiplot_data = hiplot_df.to_dict('records')
    
    # Create HiPlot experiment
    exp = hip.Experiment.from_iterable(hiplot_data)
    
    # Configure the display
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({
        'categoricalMaximumValues': 20,
        'hide': ['uid']
    })
    
    # Configure table to show all rows
    exp.display_data(hip.Displays.TABLE).update({
        'hide': ['uid', 'from_uid'],
        'order': ['Calibration'] + list(comp_df.index),
        'defaultRowsPerPage': 50  # Show more rows by default
    })
    
    # Display the parallel coordinate plot
    print("HiPlot Parallel Coordinate Plot (Normalized: Higher = Better)")
    print("=" * 60)
    print("All metrics transformed to 0-1 scale where:")
    print("  - 1.0 (top) = Best performance")
    print("  - 0.0 (bottom) = Worst performance")
    print("")
    print("Transformations applied:")
    print("  - NSE, KGE, correlations: normalized (higher original = higher score)")
    print("  - KGE components (r, α, β): distance from 1.0 (closer to 1 = higher score)")
    print("  - RMSE, MAE: inverted (lower original = higher score)")
    print("  - Biases, % errors: distance from 0 (closer to 0 = higher score)")
    print("")
    print("Interaction tips:")
    print("  - Drag on axes to filter calibrations")
    print("  - Drag column headers to reorder axes")
    print("  - Click lines to highlight individual calibrations")
    exp.display()
    
except ImportError:
    print("HiPlot not installed. Install with: pip install hiplot")
    print("Skipping HiPlot visualization...")

# %% [markdown]
# ### Key Findings
#
# The parallel coordinate plot reveals clear tradeoffs:
#
# - **NSE** excels at overall NSE and high-flow metrics but may underperform on low flows
# - **NSE_log** provides balanced performance across all flow regimes
# - **NSE_inv** and **KGE_inv** excel at low-flow metrics but may sacrifice peak accuracy
# - **NSE_sqrt** and **KGE_sqrt** offer moderate low-flow emphasis with better balance
# - **SDEB** reduces timing sensitivity through FDC matching
# - **KGE_np** is robust to outliers and skewed distributions
#
# Use the parallel coordinate plots to identify which objective best matches your application's priorities.

# %%
# Summary: Which calibration is "best" depends on your goals
print("=" * 90)
print("SUMMARY: CHOOSING THE RIGHT OBJECTIVE FUNCTION")
print("=" * 90)
print("\nUse the radar plots, parallel coordinate plots, and styled comparison table")
print("above to identify which objective function best matches your application's priorities.")
print("\nKey considerations:")
print("  • Flood forecasting → Prioritize NSE, Q95, FDC Peak metrics")
print("  • Water supply → Prioritize balanced metrics (KGE, PBIAS, FDC Mid)")
print("  • Drought/low flow → Prioritize NSE(1/Q), Q5, FDC Low metrics")
print("  • Environmental flows → Prioritize Baseflow Index, FDC Low metrics")
print("  • General purpose → Look for balanced performance across all metrics")

# %% [markdown]
# ---
# ## Step 9: Manual Parameter Bounds Adjustment
#
# ### Why Adjust Parameter Bounds?
#
# Based on analysis of calibration results, we may observe that certain parameters
# consistently approach their lower or upper bounds. This suggests the optimizer
# may be constrained and unable to find the true optimum.
#
# In this example, we manually extend the lower bounds of selected parameters
# that showed potential limitation in our calibrations:
#
# - **lzfpm**: Lower zone free water primary maximum (mm)
# - **lzpk**: Lower zone primary depletion rate (1/day)  
# - **uztwm**: Upper zone tension water maximum (mm)
# - **uzk**: Upper zone depletion rate (1/day)

# %%
# Define custom bounds with extended lower limits
print("=" * 80)
print("MANUAL PARAMETER BOUNDS ADJUSTMENT")
print("=" * 80)

# Parameters to adjust (lower bound × 0.5)
params_to_adjust = ['lzfpm', 'lzpk', 'uztwm', 'uzk']

print("\nOriginal vs Custom Bounds:")
print("-" * 60)
print(f"{'Parameter':<10} {'Original Low':>15} {'Custom Low':>15} {'Upper':>15}")
print("-" * 60)

custom_bounds = dict(param_bounds)  # Copy original bounds
for param in params_to_adjust:
    orig_low, high = param_bounds[param]
    new_low = orig_low * 0.7
    custom_bounds[param] = (new_low, high)
    print(f"{param:<10} {orig_low:>15.4f} {new_low:>15.4f} {high:>15.4f}")

# Show all bounds for reference
print("\n" + "-" * 60)
print("Complete Custom Bounds:")
print("-" * 60)
for param, (low, high) in custom_bounds.items():
    adjusted = " *" if param in params_to_adjust else ""
    print(f"{param:<10} [{low:>10.4f}, {high:>10.4f}]{adjusted}")

# %%
# Save custom bounds to file
custom_bounds_file = Path('../data/410734/sacramento_bounds_custom.txt')

print("\n" + "=" * 80)
print("SAVING CUSTOM BOUNDS FILE")
print("=" * 80)

with open(custom_bounds_file, 'w') as f:
    f.write("# Sacramento Model Parameter Bounds - Custom Adjusted\n")
    f.write("# Manual adjustment: lower bounds of lzfpm, lzpk, uztwm, uzk × 0.5\n")
    f.write(f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("#\n")
    f.write("# Format: parameter_name lower_bound upper_bound\n")
    f.write("#\n")
    for param in sorted(custom_bounds.keys()):
        low, high = custom_bounds[param]
        adjusted = "  # adjusted" if param in params_to_adjust else ""
        f.write(f"{param} {low} {high}{adjusted}\n")

print(f"\n✓ Custom bounds saved to: {custom_bounds_file}")

# %% [markdown]
# ### Re-calibrate with Custom Bounds
#
# We'll re-run a selection of calibrations using the custom bounds to see if 
# the extended lower limits allow the optimizer to find better solutions.

# %%
# Re-calibrate SDEB with custom bounds
print("=" * 80)
print("RE-CALIBRATION WITH CUSTOM BOUNDS")
print("=" * 80)
print("\nRunning calibration with SDEB objective and custom bounds...")

custom_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sdeb_objective,
    parameter_bounds=custom_bounds,
    warmup_period=WARMUP_DAYS
)

custom_result = custom_runner.run_sceua_direct(
    max_evals=MAX_EVALS_SAC,
    seed=42,
    verbose=True,
    max_tolerant_iter=100
)

print("\n" + custom_result.summary())

# Run simulation with custom-calibrated parameters
custom_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
custom_model.set_parameters(custom_result.best_parameters)
custom_model.reset()
custom_sim_flow = custom_model.run(cal_data)['runoff'].values[WARMUP_DAYS:]

print(f"\n✓ Calibration complete!")
print(f"  Best SDEB: {-custom_result.best_objective:.4f}")

# %%
# Save the custom calibration report
custom_report = custom_runner.create_report(custom_result, catchment_info={
    'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
})
custom_report.save(REPORTS_DIR / '410734_sacramento_sdeb_sceua_custom')
print(f"Calibration saved to: {REPORTS_DIR / '410734_sacramento_sdeb_sceua_custom.pkl'}")

# %%
# Compare default bounds vs custom bounds results (SDEB)
print("\n" + "=" * 80)
print("COMPARISON: DEFAULT BOUNDS vs CUSTOM BOUNDS (SDEB)")
print("=" * 80)

# Get metrics for both calibrations
default_metrics = compute_diagnostics(sdeb_sim_flow, obs_flow)
custom_metrics = compute_diagnostics(custom_sim_flow, obs_flow)

print("\n" + "-" * 60)
print(f"{'Metric':<20} {'Default Bounds':>18} {'Custom Bounds':>18}")
print("-" * 60)

for metric in default_metrics.keys():
    default_val = default_metrics[metric]
    custom_val = custom_metrics[metric]
    print(f"{metric:<20} {default_val:>18.4f} {custom_val:>18.4f}")

print("-" * 60)

# %%
# Compare calibrated parameters: default vs custom bounds (SDEB)
print("\n" + "=" * 80)
print("CALIBRATED PARAMETERS COMPARISON (SDEB)")
print("=" * 80)

print(f"\n{'Parameter':<10} {'Default Bounds':>18} {'Custom Bounds':>18} {'Difference':>15}")
print("-" * 65)

for param in sorted(sdeb_result.best_parameters.keys()):
    default_val = sdeb_result.best_parameters[param]
    custom_val = custom_result.best_parameters[param]
    diff = custom_val - default_val
    diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "~0"
    print(f"{param:<10} {default_val:>18.4f} {custom_val:>18.4f} {diff_str:>15}")

# %%
# Visual comparison: Default vs Custom bounds hydrographs (SDEB)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Hydrograph (Linear)', 'Hydrograph (Log)', 
                    'Flow Duration Curve', 'Scatter Plot'],
    specs=[[{}, {}], [{}, {}]]
)

# Hydrograph - Linear
fig.add_trace(go.Scatter(x=comparison.index, y=obs_flow, name='Observed', 
                         line=dict(color='black', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=sdeb_sim_flow, name='Default Bounds', 
                         line=dict(color='blue', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=custom_sim_flow, name='Custom Bounds', 
                         line=dict(color='red', width=1)), row=1, col=1)

# Hydrograph - Log
fig.add_trace(go.Scatter(x=comparison.index, y=obs_flow, name='Observed', 
                         line=dict(color='black', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=comparison.index, y=sdeb_sim_flow, name='Default Bounds', 
                         line=dict(color='blue', width=1, dash='dash'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=comparison.index, y=custom_sim_flow, name='Custom Bounds', 
                         line=dict(color='red', width=1), showlegend=False), row=1, col=2)
fig.update_yaxes(type='log', row=1, col=2)

# FDC
obs_sorted = np.sort(obs_flow)[::-1]
sdeb_sorted = np.sort(sdeb_sim_flow)[::-1]
custom_sorted = np.sort(custom_sim_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

fig.add_trace(go.Scatter(x=exceedance, y=obs_sorted, name='Observed', 
                         line=dict(color='black', width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=exceedance, y=sdeb_sorted, name='Default Bounds', 
                         line=dict(color='blue', width=1, dash='dash'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=exceedance, y=custom_sorted, name='Custom Bounds', 
                         line=dict(color='red', width=1), showlegend=False), row=2, col=1)
fig.update_yaxes(type='log', row=2, col=1)
fig.update_xaxes(title_text='Exceedance %', row=2, col=1)

# Scatter
fig.add_trace(go.Scatter(x=obs_flow, y=sdeb_sim_flow, mode='markers', name='Default Bounds',
                         marker=dict(color='blue', size=3, opacity=0.5), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs_flow, y=custom_sim_flow, mode='markers', name='Custom Bounds',
                         marker=dict(color='red', size=3, opacity=0.5), showlegend=False), row=2, col=2)
max_flow = max(obs_flow.max(), sdeb_sim_flow.max(), custom_sim_flow.max())
fig.add_trace(go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines', 
                         line=dict(color='gray', dash='dash'), showlegend=False), row=2, col=2)
fig.update_xaxes(title_text='Observed', row=2, col=2)
fig.update_yaxes(title_text='Simulated', row=2, col=2)

fig.update_layout(
    title='<b>Comparison: Default Bounds vs Custom Bounds (SDEB Calibration)</b>',
    height=700,
    width=1200,
    legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
)
fig.show()

# %% [markdown]
# ### Key Takeaways on Parameter Bounds Adjustment
#
# - **Manual adjustment**: Based on analysis, we extended lower bounds of `lzfpm`, `lzpk`, `uztwm`, `uzk` by ×0.5
# - **Custom bounds file**: Saved to `data/410734/sacramento_bounds_custom.txt` for future use
# - **Re-calibration**: NSE calibration re-run with custom bounds to test sensitivity
# - **Comparison**: Metrics and hydrographs compared to evaluate impact of bound changes


# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Loading data** and setting up the Sacramento model
# 2. **Configuring calibration** with the CalibrationRunner
# 3. **Running multiple calibrations** with different objective functions:
#    - NSE variants (standard, log, sqrt, inverse)
#    - KGE variants (standard, log, sqrt, inverse, non-parametric)
#    - Composite objectives (SDEB)
# 4. **Comparing results** to understand trade-offs between metrics
# 5. **Manual parameter bounds adjustment** based on observed limitations
#
# ### Key Recommendations
#
# | Flow Regime | Recommended Objective |
# |-------------|----------------------|
# | **Balanced performance** | `KGE_sqrt` or `NSE_sqrt` |
# | **High flows / floods** | Standard `NSE` or `KGE` |
# | **Low flows / drought** | `NSE_log` or `KGE_log` |
# | **Very low flows** | `NSE_inv` or `KGE_inv` |
# | **Flow duration curve** | `SDEB` |
#
# All calibration reports are saved to `test_data/02_calibration_quickstart/reports/` for future analysis.

# %% [markdown]
# ---
# ## Next Steps
#
# ### Working with Saved Reports
#
# All calibrations have been saved as `CalibrationReport` objects. See **Notebook 08: Calibration Reports**
# for detailed instructions on:
# - Loading and inspecting saved reports
# - Generating comprehensive report cards
# - Comparing multiple calibrations
# - Exporting data for external analysis
#
# ### Improve Your Calibration
#
# - **More evaluations**: Increase `max_evals` for better convergence
# - **Longer period**: Use more years of data for robust parameters
# - **Different objective**: Try composite objectives (see Notebook 04)
# - **Different algorithm**: Try DREAM for uncertainty estimation (see Notebook 05)
#
# ### Validate Your Model
#
# **Important**: Always test your calibrated model on an independent period!
# Use split-sample validation:
# 1. Calibrate on years A
# 2. Validate on years B
# 3. If performance is similar, parameters are robust
#
# ### Explore Further
#
# | Notebook | Topic |
# |----------|-------|
# | **01_sacramento_verification** | How pyrrm was verified against reference implementations |
# | **03_routing_quickstart** | Add channel routing to improve downstream predictions |
# | **04_objective_functions** | Different metrics and how they affect calibration |
# | **05_algorithm_comparison** | Compare DREAM, PyDREAM, SCE-UA Direct, and SciPy DE |
# | **06_calibration_monitor** | Monitor long-running calibrations in real-time |
# | **08_calibration_reports** | Working with saved calibration reports |
#
# ### Troubleshooting
#
# **Low NSE?** Try:
# - Check data quality (missing values, outliers)
# - Increase calibration evaluations (`max_evals`)
# - Try a different objective function
# - Verify catchment area is correct
#
# **Calibration too slow?** Try:
# - Use fewer evaluations (for exploration)
# - Use SCE-UA Direct instead of DREAM (faster)
# - Reduce the calibration period for initial exploration

# %% [markdown]
# ---
# ## Exporting Calibration Results (Excel and CSV)
#
# All calibration results saved earlier in this notebook are stored as pickle
# files that require Python to open.  For sharing with colleagues (or opening
# in Excel), `pyrrm` provides a dedicated export API that writes each report
# to a multi-sheet Excel workbook **or** a set of CSV files.
#
# Each export contains four views of the calibration:
#
# | Sheet / File           | Contents |
# |------------------------|----------|
# | **TimeSeries**         | Date, precipitation, PET, observed flow, simulated flow, baseflow & quickflow (Lyne-Hollick) |
# | **Best_Calibration**   | Best parameters and run metadata (method, objective, runtime, ...) |
# | **Diagnostics**        | Canonical 48-metric suite (NSE, KGE, KGE_np variants, RMSE, PBIAS, FHV/FMV/FLV, raw BFI, signatures) |
# | **FDC**                | Flow duration curve on a 1 % exceedance grid (observed and simulated) |
#
# ### Single report export
#
# ```python
# report = CalibrationReport.load('test_data/02_calibration_quickstart/reports/410734_sacramento_nse_sceua.pkl')
# report.export('exports/410734_nse.xlsx', format='excel')   # one Excel file
# report.export('exports/410734_nse', format='csv')           # four CSV files
# report.export('exports/410734_nse', format='both')          # both at once
# ```
#
# ### Batch export (all experiments at once)
#
# If you have many reports (as in this notebook), load them into a
# dictionary and use `export_batch`.  Each experiment is placed in its
# own subdirectory to keep things tidy:
#
# ```
# exports/
# ├── 410734_sacramento_nse_sceua/
# │   └── 410734_sacramento_nse_sceua.xlsx
# ├── 410734_sacramento_kge_sceua/
# │   └── 410734_sacramento_kge_sceua.xlsx
# └── ...
# ```
#
# ```python
# from pyrrm.calibration import export_batch
# export_batch(batch_result, 'exports/', format='excel')
# ```
#
# Or loop over saved pickle files on disk and export each one.
# Below we demonstrate both approaches.

# %%
# --- Export: load all saved reports from this notebook --------------------
from pyrrm.calibration import CalibrationReport, export_report, export_batch

EXPORT_DIR = OUTPUT_DIR / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

pkl_files = sorted(REPORTS_DIR.glob('410734_sacramento_*.pkl'))
print(f"Found {len(pkl_files)} saved calibration reports in {REPORTS_DIR}\n")
for p in pkl_files:
    print(f"  {p.name}")

# %%
# --- Single report export (Excel + CSV) -----------------------------------
# Pick one report to demonstrate the single-report API.
# We put it in its own subdirectory so it mirrors the batch layout.
demo_pkl = REPORTS_DIR / '410734_sacramento_nse_sceua.pkl'
demo_report = CalibrationReport.load(str(demo_pkl))

SINGLE_DIR = EXPORT_DIR / 'single_demo' / '410734_sacramento_nse_sceua'
SINGLE_DIR.mkdir(parents=True, exist_ok=True)
single_files = demo_report.export(
    str(SINGLE_DIR / '410734_sacramento_nse_sceua'),
    format='both',
)
print("Single report export (Excel + CSV):")
for f in single_files:
    print(f"  {Path(f).relative_to(EXPORT_DIR)}")

# %%
# --- Batch export (all reports to Excel) ----------------------------------
# Build a pseudo-BatchResult dict from the on-disk pkl files so we can
# use export_batch.  In a real batch workflow you would pass the
# BatchResult object directly.
from types import SimpleNamespace

reports_dict = {}
for pkl_path in pkl_files:
    try:
        r = CalibrationReport.load(str(pkl_path))
        key = pkl_path.stem
        reports_dict[key] = r
    except Exception as e:
        print(f"  Skipped {pkl_path.name}: {e}")

batch_like = SimpleNamespace(results=reports_dict)

BATCH_DIR = EXPORT_DIR / 'batch'
batch_files = export_batch(batch_like, str(BATCH_DIR), format='excel')

print(f"\nBatch export: {len(batch_files)} experiments exported to {BATCH_DIR}/\n")
for key, paths in sorted(batch_files.items()):
    for p in paths:
        print(f"  {Path(p).relative_to(BATCH_DIR)}")

# %%
# --- Quick sanity check: read back one exported Excel file ----------------
import pandas as pd

sample_xlsx = BATCH_DIR / '410734_sacramento_nse_sceua' / '410734_sacramento_nse_sceua.xlsx'
if sample_xlsx.exists():
    xl = pd.ExcelFile(sample_xlsx)
    print(f"Sheets in {sample_xlsx.name}: {xl.sheet_names}\n")
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        print(f"  {sheet}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"    columns: {list(df.columns)}\n")

# %%
print("=" * 70)
print("QUICKSTART COMPLETE!")
print("=" * 70)
print("\nYou've successfully calibrated your first rainfall-runoff model.")
print("Explore the other notebooks to deepen your understanding.")
print("\nNext: See Notebook 08 for working with saved calibration reports.")
