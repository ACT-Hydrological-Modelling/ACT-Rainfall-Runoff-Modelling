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
# │   └── ...              ← SpotPy, PyDREAM, SciPy adapters
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

print("=" * 70)
print("PYRRM CALIBRATION QUICKSTART")
print("=" * 70)
print("\nLibraries loaded successfully!")

# %%
# Import pyrrm components
from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration import CalibrationRunner, SPOTPY_AVAILABLE, PYDREAM_AVAILABLE
from pyrrm.calibration.objective_functions import NSE, KGE, calculate_metrics

print("\npyrrm components imported:")
print(f"  - Sacramento model")
print(f"  - CalibrationRunner")
print(f"  - Objective functions (NSE, KGE)")
print(f"\nAvailable calibration backends:")
print(f"  - SpotPy (DREAM, SCE-UA): {SPOTPY_AVAILABLE}")
print(f"  - PyDREAM: {PYDREAM_AVAILABLE}")

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
# Configure paths
DATA_DIR = Path('../data/410734')

# Catchment area - used to convert model output from mm to ML/day
# Formula: Flow (ML/day) = Depth (mm/day) × Area (km²)
CATCHMENT_AREA_KM2 = 516.62667

print(f"Data directory: {DATA_DIR.absolute()}")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")

# %% [markdown]
# ### Loading Rainfall Data
#
# Rainfall is the primary driver of streamflow. We load daily gridded
# rainfall data (already averaged over the catchment area).

# %%
# Load rainfall
rainfall_file = DATA_DIR / 'Default Input Set - Rain_QBN01.csv'
rainfall_df = pd.read_csv(rainfall_file, parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

print("RAINFALL DATA")
print("=" * 50)
print(f"File: {rainfall_file.name}")
print(f"Records: {len(rainfall_df):,} days")
print(f"Period: {rainfall_df.index.min().date()} to {rainfall_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(rainfall_df['rainfall'].describe().round(2))

# %% [markdown]
# ### Loading PET Data
#
# PET (Potential Evapotranspiration) represents how much water *could* evaporate
# given the atmospheric conditions. In pyrrm, we use Morton's Wet Environment
# evapotranspiration.

# %%
# Load PET
pet_file = DATA_DIR / 'Default Input Set - Mwet_QBN01.csv'
pet_df = pd.read_csv(pet_file, parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

print("PET DATA")
print("=" * 50)
print(f"File: {pet_file.name}")
print(f"Records: {len(pet_df):,} days")
print(f"Period: {pet_df.index.min().date()} to {pet_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(pet_df['pet'].describe().round(2))

# %% [markdown]
# ### Loading Observed Flow Data
#
# This is what we're trying to match! Observed streamflow is measured at the
# gauging station (Gauge 410734).

# %%
# Load observed flow
flow_file = DATA_DIR / '410734_output_SDmodel.csv'
flow_df = pd.read_csv(flow_file, parse_dates=['Date'], index_col='Date')

# Extract the recorded gauging station flow column
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']

# Handle missing values (-9999 = missing)
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

print("OBSERVED FLOW DATA")
print("=" * 50)
print(f"File: {flow_file.name}")
print(f"Records: {len(observed_df):,} days")
print(f"Period: {observed_df.index.min().date()} to {observed_df.index.max().date()}")
print(f"\nStatistics (ML/day):")
print(observed_df['observed_flow'].describe().round(2))

# %% [markdown]
# ### Merging the Datasets
#
# We need all three datasets aligned on the same dates. We use an "inner join"
# to keep only dates where all data is available.

# %%
# Merge all datasets
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

print("MERGED DATASET")
print("=" * 50)
print(f"Total records: {len(data):,} days")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
print(f"Columns: {list(data.columns)}")

# Quick check for missing values
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
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Rainfall (mm/day)', 'PET (mm/day)', 'Observed Flow (ML/day)')
)

# Rainfall
fig.add_trace(
    go.Scatter(x=data.index, y=data['rainfall'], name='Rainfall',
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

# Observed flow (log scale better for flow data)
fig.add_trace(
    go.Scatter(x=data.index, y=data['observed_flow'], name='Observed Flow',
               line=dict(color='darkblue', width=0.8)),
    row=3, col=1
)

# Update axes
fig.update_yaxes(title_text="Rain (mm)", autorange="reversed", row=1, col=1)
fig.update_yaxes(title_text="PET (mm)", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/d)", type="log", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)

fig.update_layout(
    title="<b>Input Data: Gauge 410734 - Queanbeyan Catchment</b><br>" +
          "<sup>Use toolbar to zoom and pan</sup>",
    height=700,
    showlegend=True,
    legend=dict(orientation='h', y=1.02)
)

fig.show()
print("Interactive plot displayed - zoom in on rainfall events to see the flow response!")

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
# See **Notebook 03: Objective Functions** for a detailed exploration of
# different metrics and how they affect calibration results.

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

# Use 10 years for calibration (faster than full period)
# For production use, you might use more data
CAL_START = pd.Timestamp('1990-01-01')
CAL_END = pd.Timestamp('1999-12-31')

# Extract calibration data
cal_data = data[(data.index >= CAL_START) & (data.index <= CAL_END)].copy()

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
# | **SCE-UA** | Optimization | Fast | No | Quick calibration |
# | **SciPy DE** | Optimization | Fast | No | Simple cases |
# | **SpotPy DREAM** | MCMC | Slow | Yes | Full uncertainty |
# | **PyDREAM** | MCMC | Slow | Yes | Complex problems |
#
# For this quickstart, we'll use **SCE-UA** (Shuffled Complex Evolution) -
# a classic hydrology algorithm that's fast and reliable.
#
# **Note**: See **Notebook 04: Algorithm Comparison** for a detailed
# comparison of all available algorithms.

# %%
# Run SCE-UA calibration
print("=" * 70)
print("RUNNING SCE-UA CALIBRATION")
print("=" * 70)
print(f"\nObjective: {objective.name}")
print(f"Algorithm: SCE-UA (Shuffled Complex Evolution)")
print("\nThis may take a few minutes...")
print("Progress will be shown below.\n")

# SCE-UA settings
# - n_iterations: Maximum function evaluations
# - ngs: Number of complexes (more = better search, but slower)
# For quick demo, we use fewer iterations. Increase for production!
result = runner.run_sceua(
    n_iterations=5000,   # Increase to 10000+ for production
    ngs=7,               # Number of complexes
    kstop=3,             # Stop if no improvement in 3 loops
    pcento=0.01,         # Convergence criterion
    dbname='quickstart_sceua',
    dbformat='csv'
)

print("\n" + result.summary())

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
# Calculate all metrics
metrics = calculate_metrics(sim_flow, obs_flow)

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"\n{'Metric':<15} {'Value':>12}  Interpretation")
print("-" * 55)
for name, value in metrics.items():
    # Add interpretation
    if name == 'NSE':
        interp = "Good" if value > 0.7 else ("Acceptable" if value > 0.5 else "Poor")
    elif name == 'KGE':
        interp = "Good" if value > 0.7 else ("Acceptable" if value > 0.5 else "Poor")
    elif name == 'PBIAS':
        interp = "Good" if abs(value) < 10 else ("Acceptable" if abs(value) < 25 else "High bias")
    else:
        interp = ""
    print(f"  {name:<13} {value:>12.4f}  {interp}")

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
    legend=dict(orientation='h', y=1.02)
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
# ## Summary
#
# Congratulations! You've successfully:
#
# 1. **Loaded hydrological data** (rainfall, PET, observed flow)
# 2. **Set up the Sacramento model** with automatic unit conversion
# 3. **Configured calibration** with NSE objective function
# 4. **Run SCE-UA optimization** to find optimal parameters
# 5. **Evaluated performance** with multiple metrics and visualizations
#
# ### Your Results

# %%
print("=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)
print(f"""
Catchment: Gauge 410734 (Queanbeyan River)
Calibration period: {CAL_START.date()} to {CAL_END.date()}
Model: Sacramento
Algorithm: SCE-UA
Objective: NSE

Performance Metrics:
  NSE:   {metrics['NSE']:.4f}  {'(Good!)' if metrics['NSE'] > 0.7 else '(Acceptable)' if metrics['NSE'] > 0.5 else '(Needs improvement)'}
  KGE:   {metrics['KGE']:.4f}
  PBIAS: {metrics['PBIAS']:.2f}%
""")

# Save calibrated parameters
params_file = Path('../test_data/quickstart_calibrated_params.csv')
pd.DataFrame([result.best_parameters]).to_csv(params_file, index=False)
print(f"Calibrated parameters saved to: {params_file}")

# %% [markdown]
# ---
# ## Next Steps
#
# ### Improve Your Calibration
#
# - **More iterations**: Increase `n_iterations` for better convergence
# - **Longer period**: Use more years of data for robust parameters
# - **Different objective**: Try KGE or composite objectives (see Notebook 03)
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
# | **03_objective_functions** | Different metrics and how they affect calibration |
# | **04_algorithm_comparison** | Compare DREAM, PyDREAM, SCE-UA, and SciPy DE |
# | **05_calibration_monitor** | Monitor long-running calibrations in real-time |
#
# ### Troubleshooting
#
# **Low NSE?** Try:
# - Check data quality (missing values, outliers)
# - Increase calibration iterations
# - Try a different objective function
# - Verify catchment area is correct
#
# **Calibration too slow?** Try:
# - Reduce calibration period
# - Use fewer iterations (for exploration)
# - Use SCE-UA instead of DREAM (faster)

# %%
print("=" * 70)
print("QUICKSTART COMPLETE!")
print("=" * 70)
print("\nYou've successfully calibrated your first rainfall-runoff model.")
print("Explore the other notebooks to deepen your understanding.")
