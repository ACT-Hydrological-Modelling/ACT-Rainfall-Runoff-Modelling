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
    n_iterations=5000,  # Full dataset calibration
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
# ## Step 7: The Effect of Flow Transformations
#
# ### Why Flow Transformations Matter
#
# The standard **NSE is biased towards high flows**. This happens because NSE
# uses squared errors, and high flows have much larger absolute errors than
# low flows. A 100 ML/day error during a flood (10,000 ML/day) is relatively
# small (1%), but dominates the objective function compared to a 10 ML/day
# error during baseflow (50 ML/day = 20% error).
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
# - **LogNSE**: Log-transformed - balances all flow ranges
# - **NSE(inverse)**: 1/Q transform - heavily emphasizes low flows
# - **NSE(sqrt)**: √Q transform - moderate, balanced emphasis
# - **SDEB**: Combines chronological timing + FDC shape + bias penalty
#
# This eliminates the need to define custom classes!

# %%
# Import objective functions from pyrrm.objectives (new unified interface)
from pyrrm.objectives import NSE, FlowTransformation, SDEB

# Create objective function instances using the new interface with FlowTransformation
# All objectives use the same interface: objective(obs, sim) returns a metric value

# LogNSE: NSE with log transformation - balances all flow ranges
log_objective = NSE(transform=FlowTransformation('log', epsilon_value=0.01))

# InverseNSE: NSE with inverse transformation (1/Q) - heavily emphasizes low flows
inv_objective = NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))

# SqrtNSE: NSE with square root transformation (√Q) - balanced emphasis
sqrt_objective = NSE(transform=FlowTransformation('sqrt'))

# SDEB: Sum of Daily Flows, Daily Exceedance Curve and Bias (Lerat et al., 2013)
# Parameters: alpha=0.1 (low chronological weight), lam=0.5 (sqrt transform)
sdeb_objective = SDEB(alpha=0.1, lam=0.5)

print("Objective functions defined (all using new pyrrm.objectives interface):")
print(f"  - NSE(log):     log(Q) transform - balances all flow ranges")
print(f"  - NSE(inverse): 1/Q transform - heavily emphasizes low flows")
print(f"  - NSE(sqrt):    √Q transform - moderate emphasis on low flows")
print(f"  - SDEB:         Combines chronological + FDC errors with bias penalty")

# %% [markdown]
# ### Calibrating with Multiple Objective Functions
#
# Now let's run calibrations with each transformation and compare the results.

# %%
# Run calibration with LogNSE
print("=" * 70)
print("CALIBRATION 1: Log-Transformed NSE (log Q)")
print("=" * 70)
print("\nRunning calibration...\n")

log_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=log_objective,
    warmup_period=WARMUP_DAYS
)

log_result = log_runner.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_lognse',
    dbformat='csv'
)
print("\n" + log_result.summary())

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
    objective=inv_objective,
    warmup_period=WARMUP_DAYS
)

inv_result = inv_runner.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_invnse',
    dbformat='csv'
)
print("\n" + inv_result.summary())

# %%
# Run calibration with SqrtNSE
print("=" * 70)
print("CALIBRATION 3: Square Root-Transformed NSE (√Q)")
print("=" * 70)
print("\nRunning calibration...\n")

sqrt_runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sqrt_objective,
    warmup_period=WARMUP_DAYS
)

sqrt_result = sqrt_runner.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_sqrtnse',
    dbformat='csv'
)
print("\n" + sqrt_result.summary())

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

sdeb_result = sdeb_runner.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_sdeb',
    dbformat='csv'
)
print("\n" + sdeb_result.summary())

# %%
# Run simulations with all calibrated parameter sets
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

# Calculate all metrics for comparison
nse_metrics = calculate_metrics(sim_flow, obs_flow)
log_metrics = calculate_metrics(log_sim_flow, obs_flow)
inv_metrics = calculate_metrics(inv_sim_flow, obs_flow)
sqrt_metrics = calculate_metrics(sqrt_sim_flow, obs_flow)
sdeb_metrics = calculate_metrics(sdeb_sim_flow, obs_flow)

# Calculate transformed NSE values for each calibration
# All objectives from pyrrm.objectives use __call__(obs, sim) interface
nse_lognse = log_objective(obs_flow, sim_flow)
nse_invnse = inv_objective(obs_flow, sim_flow)
nse_sqrtnse = sqrt_objective(obs_flow, sim_flow)
nse_sdeb = sdeb_objective(obs_flow, sim_flow)

log_lognse = log_objective(obs_flow, log_sim_flow)
log_invnse = inv_objective(obs_flow, log_sim_flow)
log_sqrtnse = sqrt_objective(obs_flow, log_sim_flow)
log_sdeb = sdeb_objective(obs_flow, log_sim_flow)

inv_lognse = log_objective(obs_flow, inv_sim_flow)
inv_invnse = inv_objective(obs_flow, inv_sim_flow)
inv_sqrtnse = sqrt_objective(obs_flow, inv_sim_flow)
inv_sdeb = sdeb_objective(obs_flow, inv_sim_flow)

sqrt_lognse = log_objective(obs_flow, sqrt_sim_flow)
sqrt_invnse = inv_objective(obs_flow, sqrt_sim_flow)
sqrt_sqrtnse = sqrt_objective(obs_flow, sqrt_sim_flow)
sqrt_sdeb = sdeb_objective(obs_flow, sqrt_sim_flow)

sdeb_lognse = log_objective(obs_flow, sdeb_sim_flow)
sdeb_invnse = inv_objective(obs_flow, sdeb_sim_flow)
sdeb_sqrtnse = sqrt_objective(obs_flow, sdeb_sim_flow)
sdeb_sdeb = sdeb_objective(obs_flow, sdeb_sim_flow)

# %% [markdown]
# ### Comprehensive Comparison of All Calibrations

# %%
print("=" * 95)
print("COMPARISON: ALL OBJECTIVE FUNCTION CALIBRATIONS")
print("=" * 95)
print(f"\n{'Metric':<12} {'NSE (Q)':>12} {'LogNSE':>12} {'InvNSE':>12} {'SqrtNSE':>12} {'SDEB':>12}")
print("-" * 76)
print(f"{'NSE':<12} {nse_metrics['NSE']:>12.4f} {log_metrics['NSE']:>12.4f} {inv_metrics['NSE']:>12.4f} {sqrt_metrics['NSE']:>12.4f} {sdeb_metrics['NSE']:>12.4f}")
print(f"{'LogNSE':<12} {nse_lognse:>12.4f} {log_lognse:>12.4f} {inv_lognse:>12.4f} {sqrt_lognse:>12.4f} {sdeb_lognse:>12.4f}")
print(f"{'InvNSE':<12} {nse_invnse:>12.4f} {log_invnse:>12.4f} {inv_invnse:>12.4f} {sqrt_invnse:>12.4f} {sdeb_invnse:>12.4f}")
print(f"{'SqrtNSE':<12} {nse_sqrtnse:>12.4f} {log_sqrtnse:>12.4f} {inv_sqrtnse:>12.4f} {sqrt_sqrtnse:>12.4f} {sdeb_sqrtnse:>12.4f}")
print(f"{'SDEB':<12} {nse_sdeb:>12.2f} {log_sdeb:>12.2f} {inv_sdeb:>12.2f} {sqrt_sdeb:>12.2f} {sdeb_sdeb:>12.2f}")
print(f"{'KGE':<12} {nse_metrics['KGE']:>12.4f} {log_metrics['KGE']:>12.4f} {inv_metrics['KGE']:>12.4f} {sqrt_metrics['KGE']:>12.4f} {sdeb_metrics['KGE']:>12.4f}")
print(f"{'PBIAS (%)':<12} {nse_metrics['PBIAS']:>12.2f} {log_metrics['PBIAS']:>12.2f} {inv_metrics['PBIAS']:>12.2f} {sqrt_metrics['PBIAS']:>12.2f} {sdeb_metrics['PBIAS']:>12.2f}")

print("\n" + "-" * 76)
print("Note: NSE/LogNSE/InvNSE/SqrtNSE are maximized (higher=better).")
print("      SDEB is minimized (lower=better).")

# %% [markdown]
# ### Visual Comparison: All Objective Functions
#
# Let's compare how all five calibrations perform across different flow ranges.

# %%
# Define colors for each calibration
colors = {
    'observed': 'black',
    'nse': '#E41A1C',      # Red
    'log': '#377EB8',      # Blue  
    'inv': '#4DAF4A',      # Green
    'sqrt': '#984EA3',     # Purple
    'sdeb': '#FF7F00'      # Orange
}

# Compare hydrographs
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Full Hydrograph (Log Scale)',
        'Flow Duration Curves',
        'Low Flow Detail (< 200 ML/day)',
        'Scatter: All Calibrations vs Observed'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}]
    ]
)

# 1. Full hydrograph (log scale)
fig.add_trace(go.Scatter(x=comparison.index, y=obs_flow, name='Observed',
              line=dict(color=colors['observed'], width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=sim_flow, name='NSE (Q)',
              line=dict(color=colors['nse'], width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=log_sim_flow, name='LogNSE',
              line=dict(color=colors['log'], width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=inv_sim_flow, name='InvNSE (1/Q)',
              line=dict(color=colors['inv'], width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=sqrt_sim_flow, name='SqrtNSE (√Q)',
              line=dict(color=colors['sqrt'], width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison.index, y=sdeb_sim_flow, name='SDEB',
              line=dict(color=colors['sdeb'], width=1)), row=1, col=1)

# 2. Flow Duration Curves
obs_sorted = np.sort(obs_flow)[::-1]
nse_sorted = np.sort(sim_flow)[::-1]
log_sorted = np.sort(log_sim_flow)[::-1]
inv_sorted = np.sort(inv_sim_flow)[::-1]
sqrt_sorted = np.sort(sqrt_sim_flow)[::-1]
sdeb_sorted = np.sort(sdeb_sim_flow)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

fig.add_trace(go.Scatter(x=exceedance, y=obs_sorted, name='Observed',
              line=dict(color=colors['observed'], width=2), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=exceedance, y=nse_sorted, name='NSE',
              line=dict(color=colors['nse'], width=1.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=exceedance, y=log_sorted, name='LogNSE',
              line=dict(color=colors['log'], width=1.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=exceedance, y=inv_sorted, name='InvNSE',
              line=dict(color=colors['inv'], width=1.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=exceedance, y=sqrt_sorted, name='SqrtNSE',
              line=dict(color=colors['sqrt'], width=1.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=exceedance, y=sdeb_sorted, name='SDEB',
              line=dict(color=colors['sdeb'], width=1.5), showlegend=False), row=1, col=2)

# 3. Low flow detail
low_mask = obs_flow < 200
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=obs_flow[low_mask],
              line=dict(color=colors['observed'], width=1.5), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=sim_flow[low_mask],
              line=dict(color=colors['nse'], width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=log_sim_flow[low_mask],
              line=dict(color=colors['log'], width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=inv_sim_flow[low_mask],
              line=dict(color=colors['inv'], width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=sqrt_sim_flow[low_mask],
              line=dict(color=colors['sqrt'], width=1), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=comparison.index[low_mask], y=sdeb_sim_flow[low_mask],
              line=dict(color=colors['sdeb'], width=1), showlegend=False), row=2, col=1)

# 4. Scatter plot - all calibrations
max_flow = max(obs_flow.max(), sim_flow.max())
fig.add_trace(go.Scatter(x=obs_flow, y=sim_flow, mode='markers', name='NSE',
              marker=dict(color=colors['nse'], size=2, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs_flow, y=log_sim_flow, mode='markers', name='LogNSE',
              marker=dict(color=colors['log'], size=2, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs_flow, y=inv_sim_flow, mode='markers', name='InvNSE',
              marker=dict(color=colors['inv'], size=2, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs_flow, y=sqrt_sim_flow, mode='markers', name='SqrtNSE',
              marker=dict(color=colors['sqrt'], size=2, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs_flow, y=sdeb_sim_flow, mode='markers', name='SDEB',
              marker=dict(color=colors['sdeb'], size=2, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines',
              line=dict(color='black', dash='dash'), showlegend=False), row=2, col=2)

# Update axes
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
fig.update_xaxes(title_text="Exceedance (%)", row=1, col=2)
fig.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig.update_yaxes(title_text="Simulated (ML/day)", type="log", row=2, col=2)
fig.update_xaxes(title_text="Observed (ML/day)", type="log", row=2, col=2)

fig.update_layout(
    title="<b>Comparison: Effect of Objective Functions on Calibration</b><br>" +
          "<sup>NSE=peaks | LogNSE=balanced | InvNSE=low flows | SqrtNSE=moderate | SDEB=FDC+timing</sup>",
    height=750,
    legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
)
fig.show()

# %% [markdown]
# ### Key Observations
#
# **NSE (Q) Calibration** (red):
# - Best at matching peak flows
# - May miss low flow dynamics
# - Highest standard NSE
#
# **LogNSE Calibration** (blue):
# - Balances high and low flow performance
# - Good all-round choice for general applications
#
# **InvNSE (1/Q) Calibration** (green):
# - Heavily emphasizes low flows
# - Best for drought analysis and environmental flows
# - May sacrifice peak accuracy
#
# **SqrtNSE (√Q) Calibration** (purple):
# - Moderate emphasis on low flows
# - Compromise between NSE and LogNSE
#
# **SDEB Calibration** (orange):
# - Combines chronological timing + FDC shape + bias penalty
# - Used in Australian SOURCE platform
# - Reduces sensitivity to peak timing errors
#
# ### Which Objective to Choose?
#
# | Application | Recommended | Reason |
# |-------------|-------------|--------|
# | Flood forecasting | NSE (Q) | Peaks matter most |
# | Water supply | LogNSE or SqrtNSE | Balance matters |
# | Drought/low flow | InvNSE (1/Q) | Low flows critical |
# | Environmental flows | LogNSE or InvNSE | Baseflow important |
# | General purpose | LogNSE or SDEB | Best balance |
# | Uncertain timing | SDEB | Reduces timing sensitivity |

# %% [markdown]
# ### Parameter Comparison Across All Objective Functions

# %%
# Compare parameters across all calibrations
print("=" * 105)
print("PARAMETER COMPARISON ACROSS ALL OBJECTIVE FUNCTIONS")
print("=" * 105)
print(f"\n{'Parameter':<10} {'NSE (Q)':>12} {'LogNSE':>12} {'InvNSE':>12} {'SqrtNSE':>12} {'SDEB':>12}")
print("-" * 74)

for param in result.best_parameters.keys():
    nse_val = result.best_parameters[param]
    log_val = log_result.best_parameters[param]
    inv_val = inv_result.best_parameters[param]
    sqrt_val = sqrt_result.best_parameters[param]
    sdeb_val = sdeb_result.best_parameters[param]
    print(f"{param:<10} {nse_val:>12.4f} {log_val:>12.4f} {inv_val:>12.4f} {sqrt_val:>12.4f} {sdeb_val:>12.4f}")

# %%
# Visual comparison of calibrated parameters across all objective functions
# Each parameter shown as position within bounds (0% = lower bound, 100% = upper bound)

param_names = list(result.best_parameters.keys())
n_params = len(param_names)

# Calculate normalized positions (0-100%) for each parameter
def normalize_param(value, param_name):
    low, high = param_bounds[param_name]
    return (value - low) / (high - low) * 100

# Prepare data for each method
methods = ['NSE (Q)', 'LogNSE', 'InvNSE (1/Q)', 'SqrtNSE (√Q)', 'SDEB']
results_list = [result, log_result, inv_result, sqrt_result, sdeb_result]
method_colors = [colors['nse'], colors['log'], colors['inv'], colors['sqrt'], colors['sdeb']]

# Create figure with subplots - one column per method
fig = make_subplots(
    rows=1, cols=5,
    subplot_titles=methods,
    horizontal_spacing=0.02
)

for col, (method_result, color) in enumerate(zip(results_list, method_colors), 1):
    # Get actual values and normalized values for this method
    actual_values = [method_result.best_parameters[p] for p in param_names]
    norm_values = [normalize_param(v, p) for v, p in zip(actual_values, param_names)]
    
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
            textfont=dict(color='white', size=9),
            showlegend=False,
            hovertext=hover_text,
            hoverinfo='text'
        ),
        row=1, col=col
    )
    
    # Add vertical line at 50%
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col)

# Update layout
fig.update_xaxes(range=[0, 100], tickvals=[0, 50, 100], ticktext=['0%', '50%', '100%'])
fig.update_yaxes(categoryorder='array', categoryarray=param_names[::-1])  # Reverse for top-to-bottom

# Only show y-axis labels on first subplot
for col in range(2, 6):
    fig.update_yaxes(showticklabels=False, row=1, col=col)

fig.update_layout(
    title="<b>Visual Comparison: Calibrated Parameters Across Objective Functions</b><br>" +
          "<sup>Bar position = % within bounds | Bar text = actual parameter value | Hover for details</sup>",
    height=650,
    width=1200,
    bargap=0.3
)

fig.show()

print("\nInterpretation:")
print("  - Bar length shows position within parameter bounds (0% = lower, 100% = upper)")
print("  - Numbers on bars show the actual calibrated parameter values")
print("  - Hover over bars for full details (value, position %, and bounds)")
print("  - Parameters at extremes (near 0% or 100%) may indicate bounds should be reviewed")

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
# 5. **Calibrated with multiple objectives** (LogNSE, InvNSE, SqrtNSE, SDEB)
# 6. **Compared results** to understand the trade-offs
#
# ### Key Takeaways
#
# - **NSE** is biased towards high flows due to squared errors
# - **Flow transformations** redistribute error weights across flow magnitudes:
#   - **log(Q)**: Balances all flow ranges
#   - **1/Q**: Heavily emphasizes low flows
#   - **√Q**: Moderate emphasis on low flows
# - **SDEB** combines chronological + FDC + bias (reduces timing sensitivity)
# - Choose your objective based on your application (floods vs drought vs general)
#
# ### Your Results

# %%
print("=" * 105)
print("CALIBRATION SUMMARY - ALL OBJECTIVE FUNCTIONS")
print("=" * 105)
print(f"""
Catchment: Gauge 410734 (Queanbeyan River)
Calibration period: {CAL_START.date()} to {CAL_END.date()}
Model: Sacramento | Algorithm: SCE-UA (5,000 iterations)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  OBJECTIVE FUNCTION COMPARISON                                                                │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│  Metric      │   NSE (Q)    │   LogNSE     │  InvNSE(1/Q) │  SqrtNSE(√Q) │      SDEB        │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
│  NSE         │ {nse_metrics['NSE']:>10.4f}   │ {log_metrics['NSE']:>10.4f}   │ {inv_metrics['NSE']:>10.4f}   │ {sqrt_metrics['NSE']:>10.4f}   │ {sdeb_metrics['NSE']:>10.4f}       │
│  KGE         │ {nse_metrics['KGE']:>10.4f}   │ {log_metrics['KGE']:>10.4f}   │ {inv_metrics['KGE']:>10.4f}   │ {sqrt_metrics['KGE']:>10.4f}   │ {sdeb_metrics['KGE']:>10.4f}       │
│  PBIAS (%)   │ {nse_metrics['PBIAS']:>+10.2f}   │ {log_metrics['PBIAS']:>+10.2f}   │ {inv_metrics['PBIAS']:>+10.2f}   │ {sqrt_metrics['PBIAS']:>+10.2f}   │ {sdeb_metrics['PBIAS']:>+10.2f}       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
│  Best for    │ Flood peaks  │ General use  │ Low flows    │ Balanced     │ FDC + timing     │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
""")

# Save calibrated parameters
params_file = Path('../test_data/quickstart_calibrated_params.csv')
pd.DataFrame([result.best_parameters]).to_csv(params_file, index=False)
print(f"Calibrated parameters saved to: {params_file}")

# %% [markdown]
# ---
# ## Step 8: Comprehensive Model Evaluation
#
# ### The Problem with Simple Metrics
#
# The summary table above uses only NSE, KGE, and PBIAS. However, **this comparison
# is inherently biased** because:
#
# - **NSE** is weighted towards high flows (squared errors)
# - **KGE** is more balanced but still volume-focused
# - **PBIAS** only captures overall volume error
#
# To fairly compare models calibrated with different objective functions, we need
# metrics that evaluate **multiple aspects** of model performance:
#
# | Category | What it Tests | Example Metrics |
# |----------|---------------|-----------------|
# | **Overall Efficiency** | General fit | NSE, KGE |
# | **Flow Regime Specific** | High vs Low flows | NSE(log), NSE(1/Q), NSE(√Q) |
# | **FDC-Based** | Flow distribution | FDC segment biases |
# | **Hydrological Signatures** | Catchment behavior | Q95, Q5, Flashiness |
# | **Timing/Dynamics** | Event response | Correlation, Rising limb |
# | **Volume Balance** | Water budget | PBIAS, Runoff ratio |

# %%
# Import comprehensive evaluation metrics from pyrrm.objectives
from pyrrm.objectives import (
    NSE as NSE_new, KGE, KGENonParametric, PBIAS, RMSE,
    FlowTransformation, FDCMetric, SignatureMetric,
    PearsonCorrelation, SpearmanCorrelation
)

def comprehensive_evaluation(obs, sim, label="Model"):
    """
    Compute comprehensive metrics for fair model comparison.
    
    Returns a dictionary with metrics organized by category:
    - Overall Efficiency
    - Flow Regime Specific  
    - FDC-Based
    - Hydrological Signatures
    - Timing/Dynamics
    - Volume Balance
    """
    
    # Initialize transformations
    inv_transform = FlowTransformation('inverse', epsilon_value=0.01)
    sqrt_transform = FlowTransformation('sqrt')
    log_transform = FlowTransformation('log', epsilon_value=0.01)
    
    metrics = {}
    
    # === 1. OVERALL EFFICIENCY ===
    metrics['NSE'] = NSE_new()(obs, sim)
    metrics['KGE'] = KGE(variant='2012')(obs, sim)
    metrics['KGE_np'] = KGENonParametric()(obs, sim)
    
    # === 2. KGE COMPONENTS (diagnostic) ===
    kge_obj = KGE(variant='2012')
    components = kge_obj.get_components(obs, sim)
    if components:
        metrics['r (correlation)'] = components['r']
        metrics['α (variability)'] = components['alpha']
        metrics['β (bias)'] = components['beta']
    
    # === 3. FLOW REGIME SPECIFIC ===
    # Different transformations emphasize different flow ranges
    metrics['NSE (log Q)'] = NSE_new(transform=log_transform)(obs, sim)
    metrics['NSE (1/Q)'] = NSE_new(transform=inv_transform)(obs, sim)
    metrics['NSE (√Q)'] = NSE_new(transform=sqrt_transform)(obs, sim)
    
    # KGE on transformed flows
    metrics['KGE (1/Q)'] = KGE(transform=inv_transform)(obs, sim)
    
    # === 4. FDC-BASED METRICS ===
    # These evaluate performance across different exceedance probabilities
    try:
        metrics['FDC Peak Bias (%)'] = FDCMetric('peak', 'volume_bias')(obs, sim)
        metrics['FDC High Bias (%)'] = FDCMetric('high', 'volume_bias')(obs, sim)
        metrics['FDC Mid Bias (%)'] = FDCMetric('mid', 'volume_bias')(obs, sim)
        metrics['FDC Low Bias (%)'] = FDCMetric('low', 'volume_bias', log_transform=True)(obs, sim)
        metrics['FDC Very Low Bias (%)'] = FDCMetric('very_low', 'volume_bias', log_transform=True)(obs, sim)
    except Exception:
        # FDC metrics may fail with edge cases
        pass
    
    # === 5. SIGNATURE METRICS ===
    # Flow percentiles (percent error between observed and simulated)
    try:
        metrics['Q95 Error (%)'] = SignatureMetric('q95')(obs, sim)  # High flow
        metrics['Q50 Error (%)'] = SignatureMetric('q50')(obs, sim)  # Median
        metrics['Q5 Error (%)'] = SignatureMetric('q5')(obs, sim)    # Low flow
        
        # Dynamics
        metrics['Flashiness Error (%)'] = SignatureMetric('flashiness')(obs, sim)
        metrics['Baseflow Index Error (%)'] = SignatureMetric('baseflow_index')(obs, sim)
        metrics['High Flow Freq Error (%)'] = SignatureMetric('high_flow_freq')(obs, sim)
        metrics['Low Flow Freq Error (%)'] = SignatureMetric('low_flow_freq')(obs, sim)
    except Exception:
        pass
    
    # === 6. VOLUME/ERROR ===
    metrics['PBIAS (%)'] = PBIAS()(obs, sim)
    metrics['RMSE'] = RMSE()(obs, sim)
    
    # Correlations
    metrics['Pearson r'] = PearsonCorrelation()(obs, sim)
    metrics['Spearman ρ'] = SpearmanCorrelation()(obs, sim)
    
    return metrics

print("Comprehensive evaluation function defined!")
print("\nMetric categories:")
print("  1. Overall Efficiency: NSE, KGE, KGE_np")
print("  2. KGE Components: r, α, β")
print("  3. Flow Regime: NSE(log), NSE(1/Q), NSE(√Q), KGE(1/Q)")
print("  4. FDC-Based: Peak, High, Mid, Low, Very Low biases")
print("  5. Signatures: Q95, Q50, Q5, Flashiness, Baseflow Index")
print("  6. Volume/Error: PBIAS, RMSE, Pearson r, Spearman ρ")

# %%
# Compute comprehensive evaluation for all calibrations
print("Computing comprehensive evaluation for all calibrations...")

eval_nse = comprehensive_evaluation(obs_flow, sim_flow, "NSE(Q)")
eval_log = comprehensive_evaluation(obs_flow, log_sim_flow, "LogNSE")
eval_inv = comprehensive_evaluation(obs_flow, inv_sim_flow, "InvNSE(1/Q)")
eval_sqrt = comprehensive_evaluation(obs_flow, sqrt_sim_flow, "SqrtNSE(√Q)")
eval_sdeb = comprehensive_evaluation(obs_flow, sdeb_sim_flow, "SDEB")

print("Done! Creating comparison table...")

# %%
# Create comprehensive comparison DataFrame
comparison_data = {
    'NSE(Q)': eval_nse,
    'LogNSE': eval_log,
    'InvNSE(1/Q)': eval_inv,
    'SqrtNSE(√Q)': eval_sqrt,
    'SDEB': eval_sdeb
}

# Convert to DataFrame
comp_df = pd.DataFrame(comparison_data)

# %% [markdown]
# ### Comprehensive Evaluation Results
#
# The table below shows how each calibration performs across **all metric categories**.
# This provides a much fairer comparison than using only NSE/KGE/PBIAS.

# %%
# Display comprehensive comparison with formatting
print("=" * 115)
print("COMPREHENSIVE MODEL EVALUATION - ALL METRICS")
print("=" * 115)
print(f"\nCalibration objectives: NSE(Q), LogNSE, InvNSE(1/Q), SqrtNSE(√Q), SDEB")
print(f"Evaluation period: {len(obs_flow):,} days (after warmup)")
print()

# Group metrics by category for clearer display
categories = {
    'OVERALL EFFICIENCY (higher = better)': ['NSE', 'KGE', 'KGE_np'],
    'KGE COMPONENTS (ideal = 1.0)': ['r (correlation)', 'α (variability)', 'β (bias)'],
    'FLOW REGIME SPECIFIC (higher = better)': ['NSE (log Q)', 'NSE (1/Q)', 'NSE (√Q)', 'KGE (1/Q)'],
    'FDC SEGMENT BIASES (closer to 0 = better)': ['FDC Peak Bias (%)', 'FDC High Bias (%)', 
                                                   'FDC Mid Bias (%)', 'FDC Low Bias (%)', 
                                                   'FDC Very Low Bias (%)'],
    'SIGNATURE ERRORS (closer to 0 = better)': ['Q95 Error (%)', 'Q50 Error (%)', 'Q5 Error (%)',
                                                 'Flashiness Error (%)', 'Baseflow Index Error (%)',
                                                 'High Flow Freq Error (%)', 'Low Flow Freq Error (%)'],
    'VOLUME & TIMING': ['PBIAS (%)', 'RMSE', 'Pearson r', 'Spearman ρ']
}

for category, metrics_list in categories.items():
    print(f"\n{'─' * 115}")
    print(f"  {category}")
    print(f"{'─' * 115}")
    print(f"  {'Metric':<28} {'NSE(Q)':>14} {'LogNSE':>14} {'InvNSE(1/Q)':>14} {'SqrtNSE(√Q)':>14} {'SDEB':>14}")
    print(f"  {'-' * 113}")
    
    for metric in metrics_list:
        if metric in comp_df.index:
            row = comp_df.loc[metric]
            # Format based on metric type
            if 'Error' in metric or 'Bias' in metric:
                # Error/bias metrics: show sign, smaller magnitude is better
                values = [f"{v:>+13.2f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            elif metric == 'RMSE':
                # RMSE: no sign, smaller is better
                values = [f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            else:
                # Efficiency metrics: higher is better
                values = [f"{v:>14.4f}" if not pd.isna(v) else f"{'N/A':>14}" for v in row]
            print(f"  {metric:<28} {values[0]} {values[1]} {values[2]} {values[3]} {values[4]}")

print(f"\n{'=' * 115}")

# %% [markdown]
# ### Interpretation Guide
#
# #### Metric Categories Explained
#
# **Overall Efficiency** (NSE, KGE, KGE_np):
# - Higher values = better overall fit
# - NSE > 0.7 is generally considered "good"
# - KGE > 0.7 is also "good"; KGE > -0.41 beats the mean benchmark
#
# **KGE Components** (r, α, β):
# - **r** = Correlation (timing accuracy)
# - **α** = Variability ratio (σ_sim/σ_obs for 2012 variant = CV ratio)
# - **β** = Bias ratio (μ_sim/μ_obs)
# - All should be close to 1.0 for a perfect model
#
# **Flow Regime Specific** (NSE variants):
# - **NSE(log Q)**: Emphasizes all flow ranges equally
# - **NSE(1/Q)**: Heavily emphasizes low flows
# - **NSE(√Q)**: Moderate emphasis on low flows
# - A model calibrated with NSE(Q) should "win" on NSE but may lose on NSE(1/Q)
#
# **FDC Segment Biases** (% bias in each flow regime):
# - **Peak** (0-2% exceedance): Flash events
# - **High** (2-20%): Quick runoff, major events  
# - **Mid** (20-70%): Intermediate flows
# - **Low** (70-95%): Baseflow-dominated periods
# - **Very Low** (95-100%): Drought conditions
# - Values closer to 0% indicate better reproduction of that flow regime
#
# **Signature Errors** (% error in hydrological indices):
# - **Q95/Q50/Q5**: Errors in flow percentiles (high/median/low flow magnitudes)
# - **Flashiness**: Error in Richards-Baker flashiness index (flow variability)
# - **Baseflow Index**: Error in baseflow contribution
# - **High/Low Flow Freq**: Error in frequency of extreme flows
# - Values closer to 0% indicate better signature reproduction
#
# **Volume & Timing**:
# - **PBIAS**: Percent bias in total volume (negative = underestimate)
# - **RMSE**: Root mean square error (in flow units)
# - **Pearson r**: Linear correlation (timing)
# - **Spearman ρ**: Rank correlation (robust to outliers)

# %%
# Visual comparison: Radar/polar plot of key metrics (normalized)
# Select representative metrics from each category for visualization

key_metrics = ['NSE', 'KGE', 'NSE (log Q)', 'NSE (1/Q)', 'Pearson r']
key_metrics_present = [m for m in key_metrics if m in comp_df.index]

# Create radar chart data
fig_radar = go.Figure()

colors_radar = {
    'NSE(Q)': '#E41A1C',
    'LogNSE': '#377EB8',
    'InvNSE(1/Q)': '#4DAF4A',
    'SqrtNSE(√Q)': '#984EA3',
    'SDEB': '#FF7F00'
}

for method in comp_df.columns:
    values = [comp_df.loc[m, method] for m in key_metrics_present]
    # Close the radar chart
    values_closed = values + [values[0]]
    metrics_closed = key_metrics_present + [key_metrics_present[0]]
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=metrics_closed,
        fill='toself',
        name=method,
        line=dict(color=colors_radar[method]),
        opacity=0.6
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    title="<b>Radar Chart: Key Efficiency Metrics Comparison</b><br>" +
          "<sup>Outer = better performance | Each calibration's strengths are visible</sup>",
    height=500,
    showlegend=True
)
fig_radar.show()

# %%
# Heatmap of all metrics (normalized for visual comparison)
# Normalize metrics so they're comparable on same scale

# For efficiency metrics (NSE, KGE, correlations): higher is better
# For error/bias metrics: closer to 0 is better

# Create a normalized version for the heatmap
heatmap_df = comp_df.copy()

# Identify which metrics to normalize and how
efficiency_metrics = ['NSE', 'KGE', 'KGE_np', 'NSE (log Q)', 'NSE (1/Q)', 'NSE (√Q)', 
                      'KGE (1/Q)', 'Pearson r', 'Spearman ρ', 'r (correlation)']
ideal_one_metrics = ['α (variability)', 'β (bias)']  # Ideal value is 1.0
minimize_metrics = ['RMSE']  # Lower is better
zero_centered_metrics = ['FDC Peak Bias (%)', 'FDC High Bias (%)', 'FDC Mid Bias (%)',
                        'FDC Low Bias (%)', 'FDC Very Low Bias (%)', 'Q95 Error (%)',
                        'Q50 Error (%)', 'Q5 Error (%)', 'Flashiness Error (%)',
                        'Baseflow Index Error (%)', 'High Flow Freq Error (%)',
                        'Low Flow Freq Error (%)', 'PBIAS (%)']

# Create performance score: higher = better for all
score_df = pd.DataFrame(index=heatmap_df.index, columns=heatmap_df.columns, dtype=float)

for metric in heatmap_df.index:
    row = heatmap_df.loc[metric]
    if metric in efficiency_metrics:
        # Higher is better, scale 0-1 (assuming NSE-like range -inf to 1)
        # Use actual range for normalization
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:
            score_df.loc[metric] = (row - min_val) / (max_val - min_val)
        else:
            score_df.loc[metric] = 0.5
    elif metric in ideal_one_metrics:
        # Ideal is 1.0, so score = 1 - |value - 1|
        score_df.loc[metric] = 1 - np.abs(row - 1)
    elif metric in minimize_metrics:
        # Lower is better
        max_val = row.max()
        if max_val > 0:
            score_df.loc[metric] = 1 - (row / max_val)
        else:
            score_df.loc[metric] = 1.0
    elif metric in zero_centered_metrics:
        # Closer to 0 is better
        max_abs = np.abs(row).max()
        if max_abs > 0:
            score_df.loc[metric] = 1 - (np.abs(row) / max_abs)
        else:
            score_df.loc[metric] = 1.0
    else:
        # Default: treat as efficiency metric
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:
            score_df.loc[metric] = (row - min_val) / (max_val - min_val)
        else:
            score_df.loc[metric] = 0.5

# Create heatmap
fig_heat = go.Figure(data=go.Heatmap(
    z=score_df.values,
    x=score_df.columns,
    y=score_df.index,
    colorscale='RdYlGn',  # Red = poor, Yellow = medium, Green = good
    zmid=0.5,
    text=heatmap_df.round(3).values,
    texttemplate='%{text}',
    textfont={"size": 9},
    hovertemplate='%{y}<br>%{x}<br>Value: %{text}<br>Score: %{z:.2f}<extra></extra>'
))

fig_heat.update_layout(
    title="<b>Comprehensive Metric Heatmap</b><br>" +
          "<sup>Colors show relative performance (green=best, red=worst) | Numbers are actual values</sup>",
    height=800,
    width=900,
    xaxis_title="Calibration Objective",
    yaxis_title="Evaluation Metric",
    yaxis=dict(tickmode='linear')
)
fig_heat.show()

# %% [markdown]
# ### Key Findings from Comprehensive Evaluation
#
# Looking at the comprehensive evaluation results, we can see clear patterns:
#
# 1. **Each objective function excels in its target domain**:
#    - NSE(Q) → Best standard NSE and high flow metrics
#    - LogNSE → Good balance across all metrics
#    - InvNSE(1/Q) → Best low flow metrics (Q5, FDC Low)
#    - SqrtNSE(√Q) → Good compromise
#    - SDEB → Good FDC reproduction and timing
#
# 2. **Trade-offs are clearly visible**:
#    - High NSE often comes with poor low-flow performance
#    - Low-flow focused calibrations may sacrifice peak accuracy
#
# 3. **Recommendations by application**:
#
# | Application | Best Objective | Key Metrics to Check |
# |-------------|----------------|----------------------|
# | Flood forecasting | NSE(Q) | NSE, Q95 Error, FDC Peak |
# | Water supply | LogNSE/SqrtNSE | KGE, PBIAS, FDC Mid |
# | Drought/low flow | InvNSE(1/Q) | NSE(1/Q), Q5 Error, FDC Low |
# | Environmental flows | LogNSE/InvNSE | Baseflow Index, FDC Low |
# | General purpose | SDEB/LogNSE | All metrics balanced |

# %%
# Summary: Which calibration is "best" depends on your goals
print("=" * 90)
print("SUMMARY: CHOOSING THE RIGHT OBJECTIVE FUNCTION")
print("=" * 90)

# Calculate category averages for each calibration
category_scores = {}
for category, metrics_list in categories.items():
    present_metrics = [m for m in metrics_list if m in score_df.index]
    if present_metrics:
        for method in score_df.columns:
            if method not in category_scores:
                category_scores[method] = {}
            category_scores[method][category] = score_df.loc[present_metrics, method].mean()

print("\nAverage Performance Score by Category (0-1, higher = better):")
print("-" * 90)
cat_summary = pd.DataFrame(category_scores).T
print(cat_summary.round(3).to_string())

# Find winners by category
print("\n" + "-" * 90)
print("Best Objective by Category:")
print("-" * 90)
for category in cat_summary.columns:
    winner = cat_summary[category].idxmax()
    score = cat_summary.loc[winner, category]
    print(f"  {category[:50]:<52} → {winner:<15} (score: {score:.3f})")

# %% [markdown]
# ---
# ## Step 9: Customizing Parameter Bounds
#
# ### Why Adjust Parameter Bounds?
#
# The calibration results above may show parameters hitting their bounds (at 0% or 100%).
# This suggests the optimal values might lie outside our initial search space. Adjusting
# parameter bounds can:
#
# - **Improve calibration**: Allow the optimizer to find better solutions
# - **Incorporate prior knowledge**: Use catchment-specific information
# - **Ensure physical realism**: Constrain parameters to sensible ranges
#
# ### Loading Custom Bounds from a File
#
# pyrrm supports loading parameter bounds from external configuration files. This makes
# it easy to:
# - Share bounds between projects
# - Version control your calibration settings
# - Adjust bounds without modifying code
#
# The file format is simple:
# ```
# # Comment lines start with #
# parameter_name = min_value, max_value  # optional description
# ```

# %%
# Display the custom bounds file
print("=" * 70)
print("CUSTOM PARAMETER BOUNDS FILE")
print("=" * 70)

bounds_file = Path('../data/410734/sacramento_bounds_custom.txt')
print(f"\nFile: {bounds_file}")
print("\nContents:")
print("-" * 70)

with open(bounds_file, 'r') as f:
    print(f.read())

# %%
# Load custom bounds and compare with default bounds
from pyrrm.data import load_parameter_bounds

custom_bounds = load_parameter_bounds(bounds_file)
default_bounds = param_bounds  # From the model's default

print("=" * 70)
print("COMPARISON: DEFAULT vs CUSTOM PARAMETER BOUNDS")
print("=" * 70)
print(f"\n{'Parameter':<10} {'Default Min':>12} {'Default Max':>12} {'Custom Min':>12} {'Custom Max':>12} {'Changed?':>10}")
print("-" * 82)

for param in default_bounds.keys():
    d_min, d_max = default_bounds[param]
    c_min, c_max = custom_bounds.get(param, (d_min, d_max))
    changed = "Yes" if (d_min != c_min or d_max != c_max) else ""
    print(f"{param:<10} {d_min:>12.4f} {d_max:>12.4f} {c_min:>12.4f} {c_max:>12.4f} {changed:>10}")

# %% [markdown]
# ### Re-calibrating with Custom Bounds
#
# Now let's re-run all four transformed objective function calibrations 
# (LogNSE, InvNSE, SqrtNSE, SDEB) using the custom parameter bounds and 
# compare the results.

# %%
# Create model with custom bounds
model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
model_custom.load_parameter_bounds(bounds_file)

print("=" * 70)
print("RE-CALIBRATION WITH CUSTOM BOUNDS")
print("=" * 70)
print(f"\nLoaded {len(custom_bounds)} parameter bounds from: {bounds_file}")
print("\nRunning calibrations with LogNSE, InvNSE, SqrtNSE, and SDEB objectives...")

# %%
# Re-calibrate with LogNSE using custom bounds
print("\n" + "=" * 70)
print("CALIBRATION WITH CUSTOM BOUNDS: LogNSE")
print("=" * 70)

log_runner_custom = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=log_objective,
    parameter_bounds=custom_bounds,
    warmup_period=WARMUP_DAYS
)

log_result_custom = log_runner_custom.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_lognse_custom',
    dbformat='csv'
)
print("\n" + log_result_custom.summary())

# %%
# Re-calibrate with InvNSE using custom bounds
print("\n" + "=" * 70)
print("CALIBRATION WITH CUSTOM BOUNDS: InvNSE (1/Q)")
print("=" * 70)

inv_runner_custom = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=inv_objective,
    parameter_bounds=custom_bounds,
    warmup_period=WARMUP_DAYS
)

inv_result_custom = inv_runner_custom.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_invnse_custom',
    dbformat='csv'
)
print("\n" + inv_result_custom.summary())

# %%
# Re-calibrate with SqrtNSE using custom bounds
print("\n" + "=" * 70)
print("CALIBRATION WITH CUSTOM BOUNDS: SqrtNSE")
print("=" * 70)

sqrt_runner_custom = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sqrt_objective,
    parameter_bounds=custom_bounds,
    warmup_period=WARMUP_DAYS
)

sqrt_result_custom = sqrt_runner_custom.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_sqrtnse_custom',
    dbformat='csv'
)
print("\n" + sqrt_result_custom.summary())

# %%
# Re-calibrate with SDEB using custom bounds
print("\n" + "=" * 70)
print("CALIBRATION WITH CUSTOM BOUNDS: SDEB")
print("=" * 70)

sdeb_runner_custom = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=sdeb_objective,
    parameter_bounds=custom_bounds,
    warmup_period=WARMUP_DAYS
)

sdeb_result_custom = sdeb_runner_custom.run_sceua(
    n_iterations=5000,
    ngs=7,
    kstop=3,
    pcento=0.01,
    dbname='quickstart_sdeb_custom',
    dbformat='csv'
)
print("\n" + sdeb_result_custom.summary())

# %%
# Run simulations with new calibrated parameters
log_model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
log_model_custom.set_parameters(log_result_custom.best_parameters)
log_model_custom.reset()
log_sim_custom = log_model_custom.run(cal_data)['runoff'].values[WARMUP_DAYS:]

inv_model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
inv_model_custom.set_parameters(inv_result_custom.best_parameters)
inv_model_custom.reset()
inv_sim_custom = inv_model_custom.run(cal_data)['runoff'].values[WARMUP_DAYS:]

sqrt_model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
sqrt_model_custom.set_parameters(sqrt_result_custom.best_parameters)
sqrt_model_custom.reset()
sqrt_sim_custom = sqrt_model_custom.run(cal_data)['runoff'].values[WARMUP_DAYS:]

sdeb_model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
sdeb_model_custom.set_parameters(sdeb_result_custom.best_parameters)
sdeb_model_custom.reset()
sdeb_sim_custom = sdeb_model_custom.run(cal_data)['runoff'].values[WARMUP_DAYS:]

# Calculate metrics for custom bounds calibrations
log_metrics_custom = calculate_metrics(log_sim_custom, obs_flow)
inv_metrics_custom = calculate_metrics(inv_sim_custom, obs_flow)
sqrt_metrics_custom = calculate_metrics(sqrt_sim_custom, obs_flow)
sdeb_metrics_custom = calculate_metrics(sdeb_sim_custom, obs_flow)

# %% [markdown]
# ### Comparing Default vs Custom Bounds Results

# %%
# Comprehensive evaluation: Default bounds vs Custom bounds
print("=" * 100)
print("COMPREHENSIVE COMPARISON: DEFAULT BOUNDS vs CUSTOM BOUNDS")
print("=" * 100)

# Apply comprehensive evaluation to all calibrations (default and custom)
bounds_comparison = {
    'LogNSE (Default)': comprehensive_evaluation(obs_flow, log_sim_flow),
    'LogNSE (Custom)': comprehensive_evaluation(obs_flow, log_sim_custom),
    'InvNSE (Default)': comprehensive_evaluation(obs_flow, inv_sim_flow),
    'InvNSE (Custom)': comprehensive_evaluation(obs_flow, inv_sim_custom),
    'SqrtNSE (Default)': comprehensive_evaluation(obs_flow, sqrt_sim_flow),
    'SqrtNSE (Custom)': comprehensive_evaluation(obs_flow, sqrt_sim_custom),
    'SDEB (Default)': comprehensive_evaluation(obs_flow, sdeb_sim_flow),
    'SDEB (Custom)': comprehensive_evaluation(obs_flow, sdeb_sim_custom),
}

# Create DataFrame with all metrics
bounds_df = pd.DataFrame(bounds_comparison).T

# Display comprehensive results
print("\n" + "=" * 100)
print("FULL METRICS COMPARISON")
print("=" * 100)
print(bounds_df.round(4).to_string())

# %%
# Calculate improvements (Custom - Default) for each objective
print("\n" + "=" * 100)
print("IMPROVEMENT FROM CUSTOM BOUNDS (Custom - Default)")
print("=" * 100)
print("Positive values = improvement, Negative values = degradation")
print("(For error metrics like PBIAS, RMSE, closer to 0 is better)")
print("-" * 100)

improvements = {}
for obj in ['LogNSE', 'InvNSE', 'SqrtNSE', 'SDEB']:
    default_metrics = bounds_comparison[f'{obj} (Default)']
    custom_metrics = bounds_comparison[f'{obj} (Custom)']
    improvements[obj] = {k: custom_metrics[k] - default_metrics[k] for k in default_metrics.keys()}

improve_df = pd.DataFrame(improvements).T
print(improve_df.round(4).to_string())

# %%
# Summary: Which calibrations improved most with custom bounds?
print("\n" + "=" * 100)
print("SUMMARY: IMPACT OF CUSTOM BOUNDS BY OBJECTIVE FUNCTION")
print("=" * 100)

# Key metrics to highlight
key_metrics = ['NSE', 'KGE', 'KGE_np', 'NSE (log Q)', 'NSE (1/Q)', 'NSE (√Q)', 'PBIAS (%)']
available_key = [m for m in key_metrics if m in improve_df.columns]

print(f"\n{'Objective':<12} | ", end="")
for metric in available_key:
    print(f"{metric:>12} | ", end="")
print()
print("-" * (15 + len(available_key) * 16))

for obj in improve_df.index:
    print(f"{obj:<12} | ", end="")
    for metric in available_key:
        val = improve_df.loc[obj, metric]
        # For PBIAS, negative improvement means getting closer to 0 (good if it was positive)
        sign = "+" if val > 0 else ""
        print(f"{sign}{val:>11.4f} | ", end="")
    print()

print("\n" + "-" * 100)
print("Interpretation:")
print("  - NSE/KGE metrics: Positive = better (higher efficiency)")
print("  - PBIAS: Change direction depends on original bias direction")
print("  - Flow regime metrics (log Q, 1/Q, √Q): Positive = better for that regime")

# %%
# Visual comparison: Default vs Custom bounds calibration
fig = make_subplots(
    rows=3, cols=4,
    subplot_titles=(
        'LogNSE - Linear',
        'InvNSE (1/Q) - Linear',
        'SqrtNSE - Linear',
        'SDEB - Linear',
        'LogNSE - Log Scale',
        'InvNSE (1/Q) - Log Scale',
        'SqrtNSE - Log Scale',
        'SDEB - Log Scale',
        'LogNSE - FDC',
        'InvNSE (1/Q) - FDC',
        'SqrtNSE - FDC',
        'SDEB - FDC'
    ),
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
    row_heights=[0.35, 0.35, 0.30]
)

# Colors
color_default = '#1f77b4'  # Blue
color_custom = '#ff7f0e'   # Orange

# Data for each objective
sim_data = [
    (log_sim_flow, log_sim_custom),
    (inv_sim_flow, inv_sim_custom),
    (sqrt_sim_flow, sqrt_sim_custom),
    (sdeb_sim_flow, sdeb_sim_custom)
]

# Row 1: Linear scale hydrographs
for col, (sim_default, sim_custom) in enumerate(sim_data, 1):
    show_legend = (col == 1)
    fig.add_trace(go.Scatter(x=comparison.index, y=obs_flow, 
                  name='Observed' if show_legend else None, showlegend=show_legend,
                  line=dict(color='black', width=1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=comparison.index, y=sim_default, 
                  name='Default bounds' if show_legend else None, showlegend=show_legend,
                  line=dict(color=color_default, width=1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=comparison.index, y=sim_custom, 
                  name='Custom bounds' if show_legend else None, showlegend=show_legend,
                  line=dict(color=color_custom, width=1)), row=1, col=col)

# Row 2: Log scale hydrographs
for col, (sim_default, sim_custom) in enumerate(sim_data, 1):
    fig.add_trace(go.Scatter(x=comparison.index, y=obs_flow, showlegend=False,
                  line=dict(color='black', width=1)), row=2, col=col)
    fig.add_trace(go.Scatter(x=comparison.index, y=sim_default, showlegend=False,
                  line=dict(color=color_default, width=1)), row=2, col=col)
    fig.add_trace(go.Scatter(x=comparison.index, y=sim_custom, showlegend=False,
                  line=dict(color=color_custom, width=1)), row=2, col=col)

# Row 3: FDC comparisons
obs_fdc = np.sort(obs_flow)[::-1]
exceedance = np.arange(1, len(obs_fdc) + 1) / len(obs_fdc) * 100

for col, (sim_default, sim_custom) in enumerate(sim_data, 1):
    fdc_default = np.sort(sim_default)[::-1]
    fdc_custom = np.sort(sim_custom)[::-1]
    
    fig.add_trace(go.Scatter(x=exceedance, y=obs_fdc, showlegend=False,
                  line=dict(color='black', width=2)), row=3, col=col)
    fig.add_trace(go.Scatter(x=exceedance, y=fdc_default, showlegend=False,
                  line=dict(color=color_default, width=1.5)), row=3, col=col)
    fig.add_trace(go.Scatter(x=exceedance, y=fdc_custom, showlegend=False,
                  line=dict(color=color_custom, width=1.5)), row=3, col=col)

# Update axes
for col in range(1, 5):
    # Row 1: Linear scale
    fig.update_yaxes(title_text="Flow (ML/day)" if col == 1 else None, row=1, col=col)
    # Row 2: Log scale
    fig.update_yaxes(title_text="Flow (ML/day)" if col == 1 else None, type="log", row=2, col=col)
    # Row 3: FDC (log scale)
    fig.update_yaxes(title_text="Flow (ML/day)" if col == 1 else None, type="log", row=3, col=col)
    fig.update_xaxes(title_text="Exceedance (%)", row=3, col=col)

fig.update_layout(
    title="<b>Impact of Custom Parameter Bounds on Calibration Results</b><br>" +
          "<sup>Black=Observed | Blue=Default bounds | Orange=Custom bounds</sup>",
    height=1000,
    width=1600,
    legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
)
fig.show()

# %% [markdown]
# ### Saving Custom Bounds
#
# You can also save parameter bounds from a calibrated model to share or modify later:
#
# ```python
# # Save bounds to a text file
# model.save_parameter_bounds('my_custom_bounds.txt')
#
# # Save bounds to CSV format
# model.save_parameter_bounds('my_custom_bounds.csv')
# ```
#
# ### Key Takeaways on Parameter Bounds
#
# - **Check for bound-hitting parameters**: If parameters consistently hit bounds, consider expanding them
# - **Use catchment knowledge**: Adjust bounds based on catchment characteristics
# - **Document your choices**: Use comments in the bounds file to explain adjustments
# - **Validate changes**: Always compare before/after to ensure improvements are genuine

# %% [markdown]
# ---
# ## Next Steps
#
# ### Improve Your Calibration
#
# - **More iterations**: Increase `n_iterations` for better convergence
# - **Longer period**: Use more years of data for robust parameters
# - **Different objective**: Try KGE or composite objectives (see Notebook 03)
# - **Different algorithm**: Try DREAM for uncertainty estimation (see Notebook 04)
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
# - Use fewer iterations (for exploration)
# - Use SCE-UA instead of DREAM (faster)
# - Reduce the calibration period for initial exploration

# %%
print("=" * 70)
print("QUICKSTART COMPLETE!")
print("=" * 70)
print("\nYou've successfully calibrated your first rainfall-runoff model.")
print("Explore the other notebooks to deepen your understanding.")
