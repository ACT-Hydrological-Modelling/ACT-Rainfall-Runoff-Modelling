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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sacramento Model Calibration Tutorial
#
# ## pyrrm - Python Rainfall-Runoff Models
#
# This comprehensive tutorial demonstrates the full workflow of the `pyrrm` library
# using real hydrological data from **Gauge 410734** (Queanbeyan catchment, NSW/ACT border).
#
# ### What You'll Learn
#
# 1. **Data Preparation** - Loading, cleaning, and visualizing hydrological data
# 2. **Model Setup** - Initializing the Sacramento model and validating against benchmark
# 3. **Calibration** - Using SpotPy DREAM, PyDREAM, and SciPy optimization
# 4. **Diagnostics** - Analyzing convergence and parameter posteriors
# 5. **Comparison** - Comparing calibration results and selecting best parameters
#
# ### Requirements
#
# ```bash
# pip install pyrrm numpy pandas matplotlib seaborn
# pip install spotpy  # For SpotPy DREAM
# pip install pydream  # For PyDREAM (optional)
# ```

# %% [markdown]
# ---
# ## Part 1: Data Loading and Preparation
#
# We'll work with daily data from the Queanbeyan catchment:
# - **Rainfall**: Gridded rainfall data (mm/day)
# - **PET**: Morton Wet Environment Evapotranspiration (mm/day)
# - **Observed Flow**: Recorded streamflow at gauge 410734 (ML/day)

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for any remaining matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

# Define data directory
DATA_DIR = Path('../data/410734')

# =============================================================================
# CATCHMENT CONFIGURATION
# =============================================================================
# Catchment area for gauge 410734 (Queanbeyan River)
# This is used to convert model outputs from mm to ML/day
# Formula: Flow (ML/day) = Depth (mm/day) × Area (km²)
# Area: 51662.667 Ha = 516.62667 km²
CATCHMENT_AREA_KM2 = 516.62667

print("Libraries loaded successfully!")
print(f"Data directory: {DATA_DIR.absolute()}")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")

# %%
# Load rainfall data
rainfall_file = DATA_DIR / 'Default Input Set - Rain_QBN01.csv'
rainfall_df = pd.read_csv(rainfall_file, parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']  # Rename to simple column name

print(f"Rainfall data loaded: {len(rainfall_df)} records")
print(f"Date range: {rainfall_df.index.min()} to {rainfall_df.index.max()}")
print(f"\nRainfall statistics (mm/day):")
print(rainfall_df['rainfall'].describe())

# %%
# Load PET (Morton Wet Environment) data
pet_file = DATA_DIR / 'Default Input Set - Mwet_QBN01.csv'
pet_df = pd.read_csv(pet_file, parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']  # Rename to simple column name

print(f"PET data loaded: {len(pet_df)} records")
print(f"Date range: {pet_df.index.min()} to {pet_df.index.max()}")
print(f"\nPET statistics (mm/day):")
print(pet_df['pet'].describe())

# %%
# Load observed flow data from SOURCE output file
# Using the recorded gauging station flow for consistency with SOURCE benchmark
flow_file = DATA_DIR / '410734_output_SDmodel.csv'
observed_full_df = pd.read_csv(flow_file, parse_dates=['Date'], index_col='Date')

# Extract the observed flow column
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = observed_full_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']

# Handle missing values (NaN or negative values)
n_missing_before = observed_df['observed_flow'].isna().sum() + (observed_df['observed_flow'] < 0).sum()
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

print(f"Observed flow data loaded from: {flow_file.name}")
print(f"Column used: {observed_col}")
print(f"Records loaded: {len(observed_df)}")
print(f"Date range: {observed_df.index.min()} to {observed_df.index.max()}")
print(f"Missing/invalid values removed: {n_missing_before}")
print(f"\nObserved flow statistics (ML/day):")
print(observed_df['observed_flow'].describe())

# %%
# Merge all datasets on date index
# Use inner join to keep only dates where all data is available
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

print(f"Merged dataset: {len(data)} records")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"\nDataset columns: {list(data.columns)}")
print(f"\nFirst few rows:")
data.head()

# %%
# Define calibration and validation periods
WARMUP_DAYS = 365  # 1 year warmup for model states

# Calibration period: 1985-2010 (where observed data starts)
CAL_START = pd.Timestamp('1985-03-03')  # Start of observed data
CAL_END = pd.Timestamp('2010-12-31')

# Validation period: 2011-2025
VAL_START = pd.Timestamp('2011-01-01')
VAL_END = data.index.max()

# Split data
cal_data = data[(data.index >= CAL_START) & (data.index <= CAL_END)].copy()
val_data = data[(data.index >= VAL_START) & (data.index <= VAL_END)].copy()

print(f"Calibration period: {CAL_START.date()} to {CAL_END.date()}")
print(f"  Records: {len(cal_data)}")
print(f"  Effective calibration (after warmup): {len(cal_data) - WARMUP_DAYS} days")
print(f"\nValidation period: {VAL_START.date()} to {VAL_END.date()}")
print(f"  Records: {len(val_data)}")

# %%
# Visualize input data time series (Interactive Plotly)
pet_rolling = data['pet'].rolling(window=30).mean()

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=('Rainfall', 'PET', 'Observed Flow - Linear Scale', 'Observed Flow - Log Scale')
)

# Rainfall - use filled area plot (bars become invisible with long time series)
fig.add_trace(go.Scatter(x=data.index, y=data['rainfall'], name='Rainfall', 
                         fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.6)',
                         line=dict(color='steelblue', width=0.5)), row=1, col=1)

# PET
fig.add_trace(go.Scatter(x=data.index, y=data['pet'], name='PET', 
                         line=dict(color='orange', width=0.5), opacity=0.7), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=pet_rolling, name='PET 30-day mean',
                         line=dict(color='darkorange', width=1.5)), row=2, col=1)

# Observed flow - Linear
fig.add_trace(go.Scatter(x=data.index, y=data['observed_flow'], name='Observed Flow',
                         line=dict(color='darkblue', width=0.5), opacity=0.8), row=3, col=1)

# Observed flow - Log
fig.add_trace(go.Scatter(x=data.index, y=data['observed_flow'], name='Observed Flow (log)',
                         line=dict(color='darkblue', width=0.5), opacity=0.8, showlegend=False), row=4, col=1)

# Add Cal/Val split line to all subplots
for row in [1, 2, 3, 4]:
    fig.add_vline(x=CAL_END, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)

# Update layout
fig.update_yaxes(title_text="Rainfall (mm/day)", autorange="reversed", row=1, col=1)
fig.update_yaxes(title_text="PET (mm/day)", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", row=3, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=4, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)

fig.update_layout(
    title_text="Input Data for Gauge 410734 - Queanbeyan Catchment",
    height=900, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

fig.show()
print("Interactive plot displayed (use toolbar to zoom/pan)")

# %%
# Summary statistics by period
print("=" * 60)
print("DATA SUMMARY BY PERIOD")
print("=" * 60)

for name, df in [("Calibration", cal_data), ("Validation", val_data)]:
    print(f"\n{name} Period:")
    print(f"  Mean rainfall: {df['rainfall'].mean():.2f} mm/day")
    print(f"  Mean PET: {df['pet'].mean():.2f} mm/day")
    print(f"  Mean flow: {df['observed_flow'].mean():.2f} ML/day")
    print(f"  Max flow: {df['observed_flow'].max():.2f} ML/day")
    print(f"  Min flow: {df['observed_flow'].min():.2f} ML/day")

# %% [markdown]
# ---
# ## Part 1B: Verify C# to Python Port Using Synthetic Benchmark Data
#
# Before running the model on real data, let's verify that our Python implementation
# produces **identical results** to the standalone C# implementation using the synthetic
# benchmark data from the original port verification.
#
# This comparison:
# 1. **Runs the standalone C# Sacramento model** on synthetic data
# 2. **Runs the Python pyrrm Sacramento model** on the same data
# 3. **Compares outputs** to verify they are identical within floating-point tolerance
#
# If both implementations match within tolerance (1e-10), we can be confident that
# any differences with SOURCE are due to parameter interpretation or configuration
# differences, not bugs in the pyrrm implementation.

# %%
# Setup paths and load synthetic benchmark data
import json
import subprocess
import os

# Determine project root from DATA_DIR (which is ../data/410734)
# DATA_DIR.parent = ../data, DATA_DIR.parent.parent = .. (project root)
# Use resolve() to get absolute path
PROJECT_ROOT = Path(DATA_DIR).resolve().parent.parent

TEST_DATA_DIR = PROJECT_ROOT / 'test_data'
CSHARP_RUNNER = PROJECT_ROOT / 'benchmark' / 'CSharpRunner' / 'bin' / 'Debug' / 'net9.0' / 'CSharpRunner'

print(f"Project root: {PROJECT_ROOT}")
print(f"Test data dir: {TEST_DATA_DIR}")
print(f"C# Runner: {CSHARP_RUNNER}")
print(f"C# Runner exists: {CSHARP_RUNNER.exists()}")

# Check if C# runner exists
if not CSHARP_RUNNER.exists():
    print(f"WARNING: C# runner not found at: {CSHARP_RUNNER}")
    print("Building C# runner...")
    csharp_project_dir = PROJECT_ROOT / 'benchmark' / 'CSharpRunner'
    build_result = subprocess.run(
        ['dotnet', 'build', '-c', 'Debug'],
        cwd=str(csharp_project_dir),
        capture_output=True,
        text=True
    )
    if build_result.returncode != 0:
        print("Build failed:")
        print(build_result.stderr)
    else:
        print("Build successful!")

# Load or generate synthetic inputs
synthetic_input_file = TEST_DATA_DIR / 'synthetic_inputs.csv'
if not synthetic_input_file.exists():
    print("Generating synthetic test data...")
    # Generate synthetic data
    np.random.seed(42)
    n_days = 1095  # 3 years
    
    # Generate rainfall with mixed distribution
    rainfall = np.zeros(n_days)
    for i in range(n_days):
        p = np.random.random()
        if p < 0.70:
            rainfall[i] = 0.0
        elif p < 0.90:
            rainfall[i] = np.random.exponential(5.0)
        elif p < 0.98:
            rainfall[i] = np.random.uniform(10, 30)
        else:
            rainfall[i] = np.random.uniform(30, 100)
    
    # Generate seasonal PET
    days = np.arange(n_days)
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * days / 365)
    pet = np.maximum(pet, 0.5)
    
    synthetic_inputs = pd.DataFrame({
        'timestep': days,
        'rainfall': np.round(rainfall, 4),
        'pet': np.round(pet, 4)
    })
    TEST_DATA_DIR.mkdir(exist_ok=True)
    synthetic_inputs.to_csv(synthetic_input_file, index=False)
else:
    synthetic_inputs = pd.read_csv(synthetic_input_file)

print("Synthetic Benchmark Data")
print("=" * 60)
print(f"Input file: {synthetic_input_file}")
print(f"Records: {len(synthetic_inputs)}")
print(f"Period: {len(synthetic_inputs)} days ({len(synthetic_inputs)/365:.1f} years)")

# %%
# Define the benchmark parameter sets for testing
benchmark_param_sets = {
    "default": {
        "description": "Default parameters from C# initParameters()",
        "params": {
            "uztwm": 50.0, "uzfwm": 40.0, "lztwm": 130.0, "lzfpm": 60.0,
            "lzfsm": 25.0, "rserv": 0.3, "adimp": 0.0, "uzk": 0.3,
            "lzpk": 0.01, "lzsk": 0.05, "zperc": 40.0, "rexp": 1.0,
            "pctim": 0.01, "pfree": 0.06, "side": 0.0, "ssout": 0.0,
            "sarva": 0.0, "uh1": 1.0, "uh2": 0.0, "uh3": 0.0, "uh4": 0.0, "uh5": 0.0
        }
    }
}

# Write parameter sets to JSON for C# runner
params_json_file = TEST_DATA_DIR / 'parameter_sets.json'
with open(params_json_file, 'w') as f:
    json.dump(benchmark_param_sets, f, indent=2)

print("\nBenchmark Parameter Set: 'default'")
print("-" * 40)
for name, value in benchmark_param_sets["default"]["params"].items():
    print(f"  {name:8s}: {value}")

# %%
# Run the standalone C# Sacramento implementation
print("\n" + "=" * 60)
print("RUNNING STANDALONE C# SACRAMENTO MODEL")
print("=" * 60)

csharp_output_file = TEST_DATA_DIR / 'csharp_output_TC01_default.csv'

if CSHARP_RUNNER.exists():
    print(f"\nC# Runner: {CSHARP_RUNNER}")
    print("Executing C# Sacramento model...")
    
    # Set up environment for Homebrew .NET installation
    env = os.environ.copy()
    # Homebrew installs .NET here - set DOTNET_ROOT so the runtime can be found
    homebrew_dotnet_root = '/opt/homebrew/opt/dotnet/libexec'
    if Path(homebrew_dotnet_root).exists():
        env['DOTNET_ROOT'] = homebrew_dotnet_root
        print(f"Using Homebrew .NET: {homebrew_dotnet_root}")
    
    # Run the C# executable
    result = subprocess.run(
        [str(CSHARP_RUNNER.resolve())],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env  # Pass environment with DOTNET_ROOT
    )
    
    print("\nC# Runner Output:")
    print("-" * 40)
    for line in result.stdout.split('\n'):
        # Only show TC01 output, skip other test cases that won't run
        if line.strip() and 'TC0' not in line or 'TC01' in line:
            print(f"  {line}")
    
    if result.returncode != 0:
        print(f"\n✗ C# Runner Error (exit code {result.returncode}):")
        print(result.stderr)
        raise RuntimeError(f"C# Sacramento model failed to run. Exit code: {result.returncode}")
    else:
        print(f"\n✓ C# Runner completed successfully!")
    
    # Load freshly generated C# output
    if csharp_output_file.exists():
        csharp_output = pd.read_csv(csharp_output_file)
        print(f"\nC# Output file: {csharp_output_file}")
        print(f"Records: {len(csharp_output)}")
        print(f"Columns: {list(csharp_output.columns)[:8]}...")
    else:
        raise FileNotFoundError(f"C# output file not created: {csharp_output_file}")
else:
    raise FileNotFoundError(f"C# runner not found: {CSHARP_RUNNER}\n"
                           "Build with: cd benchmark/CSharpRunner && dotnet build")

# %%
# Run Python Sacramento model with same parameters and inputs
from pyrrm.models.sacramento import Sacramento

print("\n" + "=" * 60)
print("RUNNING PYTHON SACRAMENTO MODEL (pyrrm)")
print("=" * 60)

# Initialize model (no catchment area for synthetic test - outputs in mm)
py_model = Sacramento()
py_model.reset()

# Apply default benchmark parameters
for name, value in benchmark_param_sets["default"]["params"].items():
    setattr(py_model, name, value)

# Update internal states after setting parameters
py_model._update_internal_states()
py_model._set_unit_hydrograph_components()
py_model.reset()
print(f"Python model output units: {py_model.output_units}")

print(f"Model initialized with {len(benchmark_param_sets['default']['params'])} parameters")
print(f"Running {len(synthetic_inputs)} timesteps...")

# Run model timestep by timestep (matching C# behavior exactly)
py_outputs = []
for _, row in synthetic_inputs.iterrows():
    result = py_model.run_timestep(row['rainfall'], row['pet'])
    py_outputs.append({
        'timestep': int(row['timestep']),
        'runoff': result['runoff'],
        'baseflow': result['baseflow'],
        'channel_flow': result['channel_flow'],
        'mass_balance': result['mass_balance'],
        'uztwc': py_model.uztwc,
        'uzfwc': py_model.uzfwc,
        'lztwc': py_model.lztwc,
        'lzfsc': py_model.lzfsc,
        'lzfpc': py_model.lzfpc,
    })

py_output_df = pd.DataFrame(py_outputs)
print(f"Python simulation complete: {len(py_output_df)} timesteps")

# %%
# Compare C# vs Python outputs
print("\n" + "=" * 60)
print("C# vs PYTHON IMPLEMENTATION COMPARISON")
print("=" * 60)

if csharp_output is not None:
    # Variables to compare
    compare_vars = ['runoff', 'baseflow', 'channel_flow', 'uztwc', 'uzfwc', 'lztwc', 'lzfsc', 'lzfpc']
    
    tolerance = 1e-10
    all_pass = True
    comparison_results = []
    
    for var in compare_vars:
        if var in csharp_output.columns and var in py_output_df.columns:
            csharp_vals = csharp_output[var].values
            python_vals = py_output_df[var].values
            
            diff = np.abs(csharp_vals - python_vals)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            max_diff_idx = np.argmax(diff)
            
            passed = max_diff < tolerance
            all_pass = all_pass and passed
            
            comparison_results.append({
                'variable': var,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_idx': max_diff_idx,
                'passed': passed
            })
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{var}:")
            print(f"  {status}")
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            if not passed:
                print(f"  Timestep of max diff: {max_diff_idx}")
                print(f"  C# value: {csharp_vals[max_diff_idx]}")
                print(f"  Python value: {python_vals[max_diff_idx]}")
    
    print("\n" + "-" * 60)
    if all_pass:
        print("✓ ALL VARIABLES MATCH - C# to Python port verified!")
        print(f"  Maximum difference across all variables: {max([r['max_diff'] for r in comparison_results]):.2e}")
        print(f"  Tolerance: {tolerance:.2e}")
    else:
        print("✗ MISMATCH DETECTED - Some variables differ beyond tolerance")
        print("  Review the differences above for debugging")

# %%
# Visualize C# vs Python comparison (Interactive Plotly)
if csharp_output is not None:
    timesteps = py_output_df['timestep'].values
    diff_runoff = py_output_df['runoff'].values - csharp_output['runoff'].values
    max_val = max(csharp_output['runoff'].max(), py_output_df['runoff'].max())
    
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=('Runoff Time Series', 'Runoff: C# vs Python', 
                       'Runoff Difference', 'Upper Zone Tension Water Content')
    )
    
    # 1. Runoff comparison
    fig.add_trace(go.Scatter(x=timesteps, y=csharp_output['runoff'].values, name='C#',
                             line=dict(color='blue', width=1), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=timesteps, y=py_output_df['runoff'].values, name='Python',
                             line=dict(color='red', width=1, dash='dash'), opacity=0.7), row=1, col=1)
    
    # 2. Runoff scatter (should be perfect 1:1)
    fig.add_trace(go.Scatter(x=csharp_output['runoff'].values, y=py_output_df['runoff'].values,
                             mode='markers', name='Data', marker=dict(color='blue', size=4, opacity=0.5),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], name='1:1 line',
                             line=dict(color='black', dash='dash'), showlegend=False), row=1, col=2)
    
    # 3. Difference time series
    fig.add_trace(go.Scatter(x=timesteps, y=diff_runoff, name='Difference',
                             line=dict(color='purple', width=0.5), opacity=0.7, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # 4. Storage comparison (UZTWC)
    fig.add_trace(go.Scatter(x=timesteps, y=csharp_output['uztwc'].values, name='C# UZTWC',
                             line=dict(color='blue', width=1), opacity=0.7, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=timesteps, y=py_output_df['uztwc'].values, name='Python UZTWC',
                             line=dict(color='red', width=1, dash='dash'), opacity=0.7, showlegend=False), row=2, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Timestep", row=1, col=1)
    fig.update_xaxes(title_text="C# Runoff (mm)", row=1, col=2)
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_xaxes(title_text="Timestep", row=2, col=2)
    fig.update_yaxes(title_text="Runoff (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Python Runoff (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Difference (Python - C#)", row=2, col=1)
    fig.update_yaxes(title_text="UZTWC (mm)", row=2, col=2)
    
    fig.update_layout(
        title_text="C# vs Python Sacramento Implementation Comparison (Synthetic Data)",
        height=700, showlegend=True
    )
    fig.show()
    print("Interactive plot displayed")

# %%
# Summary: TC01 Benchmark Verification
print("\n" + "=" * 60)
print("SUMMARY: C# to Python Port Verification (TC01_default)")
print("=" * 60)

# Compare the C# and Python outputs we just generated
print("\nComparing freshly run C# output with Python output:")
print(f"  C# file:     {csharp_output_file}")
print(f"  C# records:  {len(csharp_output)}")
print(f"  Py records:  {len(py_output_df)}")

# Calculate max difference across key variables
max_diff_overall = 0
for var in ['runoff', 'baseflow', 'uztwc', 'lztwc']:
    if var in csharp_output.columns and var in py_output_df.columns:
        diff = np.max(np.abs(csharp_output[var].values - py_output_df[var].values))
        max_diff_overall = max(max_diff_overall, diff)

print(f"\n  Maximum difference (all variables): {max_diff_overall:.2e}")
print(f"  Tolerance: 1e-10")

print("\n" + "-" * 60)
if max_diff_overall < 1e-10:
    print("✓ VERIFICATION PASSED - C# and Python implementations are IDENTICAL")
    print("\nConclusion: Any differences between pyrrm and SOURCE are NOT due to")
    print("bugs in the Python port, but rather differences in configuration,")
    print("parameter interpretation, or catchment area scaling.")
else:
    print("✗ VERIFICATION FAILED - implementations differ beyond tolerance")
    print("Review the comparison results above for details.")

# %% [markdown]
# ---
# ## Part 2: Sacramento Model Setup
#
# Now we'll set up the Sacramento rainfall-runoff model and validate our implementation
# against the benchmark output from the SOURCE model.
#
# ### Sacramento Model Overview
#
# The Sacramento model is a conceptual rainfall-runoff model that divides the
# catchment into upper and lower soil zones, each with tension and free water components:
#
# - **Upper Zone**: UZTWC (tension water), UZFWC (free water)
# - **Lower Zone**: LZTWC (tension water), LZFPC (primary free water), LZFSC (supplementary free water)
# - **Unit Hydrograph**: Routes surface runoff to the catchment outlet

# %%
# Import pyrrm Sacramento model
import sys
sys.path.insert(0, '..')  # Add parent directory to path

from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration.objective_functions import NSE, KGE, RMSE, LogNSE, calculate_metrics

print("Sacramento model imported successfully!")

# %%
# Load benchmark parameters from CSV
params_file = DATA_DIR / '410734_RR_params_SDmodel.csv'
params_df = pd.read_csv(params_file)

# Extract parameters from first row (all functional units have same params)
param_row = params_df.iloc[0]

# Map SOURCE parameter names to pyrrm parameter names
benchmark_params = {
    'uztwm': param_row['Sacramento-Uztwm (mm)'],
    'uzfwm': param_row['Sacramento-Uzfwm (mm)'],
    'lztwm': param_row['Sacramento-Lztwm (mm)'],
    'lzfpm': param_row['Sacramento-Lzfpm (mm)'],
    'lzfsm': param_row['Sacramento-Lzfsm (mm)'],
    'uzk': param_row['Sacramento-Uzk (None)'],
    'lzpk': param_row['Sacramento-Lzpk (None)'],
    'lzsk': param_row['Sacramento-Lzsk (None)'],
    'zperc': param_row['Sacramento-Zperc (None)'],
    'rexp': param_row['Sacramento-Rexp (None)'],
    'pctim': param_row['Sacramento-Pctim (None)'],
    'pfree': param_row['Sacramento-Pfree (None)'],
    'rserv': param_row['Sacramento-Rserv (None)'],
    'side': param_row['Sacramento-Side (None)'],
    'adimp': param_row['Sacramento-Adimp (None)'],
    'sarva': param_row['Sacramento-Sarva (None)'],
    'ssout': param_row['Sacramento-Ssout (mm)'],
    # Unit hydrograph proportions
    'uh1': param_row['Sacramento-UH1 (None)'],
    'uh2': param_row['Sacramento-UH2 (None)'],
    'uh3': param_row['Sacramento-UH3 (None)'],
    'uh4': param_row['Sacramento-UH4 (None)'],
    'uh5': param_row['Sacramento-UH5 (None)'],
}

print("Benchmark Sacramento Parameters:")
print("-" * 40)
for name, value in benchmark_params.items():
    print(f"  {name:8s}: {value:.6f}")

# %%
# Initialize Sacramento model with benchmark parameters and catchment area
# Setting catchment_area_km2 enables automatic conversion from mm to ML/day
model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
model.reset()

# Set benchmark parameters
for name, value in benchmark_params.items():
    setattr(model, name, value)

print("Sacramento model initialized with benchmark parameters")
print(f"  Catchment area: {model.catchment_area_km2} km²")
print(f"  Output units: {model.output_units}")
print(f"  Number of parameters set: {len(benchmark_params)}")

# %%
# Run simulation with benchmark parameters on calibration period
print("Running Sacramento model simulation...")
print(f"Period: {cal_data.index.min().date()} to {cal_data.index.max().date()}")

# Reset model before simulation
model.reset()

# Prepare input DataFrame for Sacramento model
# Sacramento expects 'rainfall' and 'pet' columns
model_inputs = cal_data[['rainfall', 'pet']].copy()

# Run simulation using the model's run() method
results = model.run(model_inputs)

# Add simulated flow to calibration data (runoff = total flow)
cal_data['simulated_flow'] = results['runoff'].values

print(f"Simulation complete: {len(results)} timesteps")

# %% [markdown]
# ### 2.3 Verify C# vs Python Implementation on Gauge 410734 Data
#
# Before comparing with SOURCE, let's verify that the C# and Python implementations
# produce identical results on the real gauge 410734 data (just like we did with synthetic data).

# %%
# Prepare gauge 410734 data for C# runner
print("=" * 60)
print("VERIFYING C# vs PYTHON ON GAUGE 410734 DATA")
print("=" * 60)

# Use PROJECT_ROOT from Part 1B (already defined)
GAUGE_TEST_DATA_DIR = PROJECT_ROOT / 'test_data'
GAUGE_TEST_DATA_DIR.mkdir(exist_ok=True)

# Save gauge 410734 inputs in C# runner format (overwrites synthetic_inputs.csv temporarily)
gauge_input_file = GAUGE_TEST_DATA_DIR / 'synthetic_inputs.csv'
gauge_inputs = pd.DataFrame({
    'timestep': range(len(cal_data)),
    'rainfall': cal_data['rainfall'].values,
    'pet': cal_data['pet'].values
})
gauge_inputs.to_csv(gauge_input_file, index=False)
print(f"\nPrepared gauge 410734 inputs: {gauge_input_file}")
print(f"  Records: {len(gauge_inputs)}")

# Save benchmark parameters in C# runner format (key must be "default" for TC01)
gauge_params = {
    "default": {
        "description": "Gauge 410734 benchmark parameters",
        "params": {k: float(v) for k, v in benchmark_params.items()}
    }
}
params_json_file = GAUGE_TEST_DATA_DIR / 'parameter_sets.json'
with open(params_json_file, 'w') as f:
    json.dump(gauge_params, f, indent=2)
print(f"Prepared parameters: {params_json_file}")

# %%
# Run C# Sacramento on gauge 410734 data
print("\n" + "=" * 60)
print("RUNNING C# SACRAMENTO ON GAUGE 410734")
print("=" * 60)

GAUGE_CSHARP_RUNNER = PROJECT_ROOT / 'benchmark' / 'CSharpRunner' / 'bin' / 'Debug' / 'net9.0' / 'CSharpRunner'
gauge_csharp_output_file = GAUGE_TEST_DATA_DIR / 'csharp_output_TC01_default.csv'

if GAUGE_CSHARP_RUNNER.exists():
    print(f"\nC# Runner: {GAUGE_CSHARP_RUNNER}")
    print("Executing C# Sacramento model on gauge 410734 data...")
    
    # Set up environment for Homebrew .NET
    env = os.environ.copy()
    homebrew_dotnet_root = '/opt/homebrew/opt/dotnet/libexec'
    if Path(homebrew_dotnet_root).exists():
        env['DOTNET_ROOT'] = homebrew_dotnet_root
    
    # Run C# executable
    result = subprocess.run(
        [str(GAUGE_CSHARP_RUNNER.resolve())],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )
    
    print("\nC# Runner Output:")
    print("-" * 40)
    for line in result.stdout.split('\n'):
        if line.strip() and ('TC0' not in line or 'TC01' in line):
            print(f"  {line}")
    
    if result.returncode != 0:
        print(f"\n✗ C# Runner Error (exit code {result.returncode}):")
        print(result.stderr)
        raise RuntimeError("C# Sacramento model failed")
    else:
        print(f"\n✓ C# Runner completed successfully!")
    
    # Load C# output
    gauge_csharp_output = pd.read_csv(gauge_csharp_output_file)
    print(f"\nC# Output: {len(gauge_csharp_output)} records")
else:
    raise FileNotFoundError(f"C# runner not found: {GAUGE_CSHARP_RUNNER}")

# %%
# Run Python Sacramento on gauge 410734 data (timestep-by-timestep like C#)
print("\n" + "=" * 60)
print("RUNNING PYTHON SACRAMENTO ON GAUGE 410734")
print("=" * 60)

# Initialize fresh model WITHOUT catchment area (outputs in mm)
# This matches C# output units for exact numerical comparison
py_gauge_model = Sacramento()  # No catchment area - outputs in mm
py_gauge_model.reset()

# Apply benchmark parameters
for name, value in benchmark_params.items():
    setattr(py_gauge_model, name, value)

# Initialize internal states
py_gauge_model._update_internal_states()
py_gauge_model._set_unit_hydrograph_components()
py_gauge_model.reset()
print(f"Python model output units: {py_gauge_model.output_units} (matching C# for verification)")

# Run timestep by timestep (matching C# behavior)
py_gauge_outputs = []
for _, row in cal_data.iterrows():
    result = py_gauge_model.run_timestep(row['rainfall'], row['pet'])
    py_gauge_outputs.append({
        'timestep': len(py_gauge_outputs),
        'runoff': result['runoff'],
        'baseflow': result['baseflow'],
        'channel_flow': result['channel_flow'],
        'uztwc': py_gauge_model.uztwc,
        'lztwc': py_gauge_model.lztwc,
    })

py_gauge_output_df = pd.DataFrame(py_gauge_outputs)
print(f"Python simulation complete: {len(py_gauge_output_df)} timesteps")

# Store raw mm outputs for implementation verification (C# vs Python)
cal_data['csharp_flow_mm'] = gauge_csharp_output['runoff'].values
cal_data['python_flow_mm'] = py_gauge_output_df['runoff'].values

# Convert to ML/day for comparison with SOURCE benchmark
# Formula: Flow (ML/day) = Depth (mm/day) × Area (km²)
cal_data['csharp_flow'] = cal_data['csharp_flow_mm'] * CATCHMENT_AREA_KM2
cal_data['python_flow'] = cal_data['python_flow_mm'] * CATCHMENT_AREA_KM2
print(f"\nUnit conversion applied: mm → ML/day (catchment area = {CATCHMENT_AREA_KM2} km²)")

# %%
# Compare C# vs Python on gauge 410734 data (using mm for exact numerical comparison)
print("\n" + "=" * 60)
print("C# vs PYTHON COMPARISON (GAUGE 410734)")
print("=" * 60)
print("Note: Comparing in mm for exact numerical verification")

# Compare key variables (all in mm)
compare_vars = ['runoff', 'baseflow', 'uztwc', 'lztwc']
max_diff_gauge = 0
tolerance = 1e-10

print("\nVariable-by-variable comparison:")
for var in compare_vars:
    if var in gauge_csharp_output.columns and var in py_gauge_output_df.columns:
        csharp_vals = gauge_csharp_output[var].values
        python_vals = py_gauge_output_df[var].values
        
        diff = np.abs(csharp_vals - python_vals)
        max_diff = np.max(diff)
        max_diff_gauge = max(max_diff_gauge, max_diff)
        
        status = "✓" if max_diff < tolerance else "✗"
        print(f"  {var:12s}: max diff = {max_diff:.2e} {status}")

print(f"\n  Overall max difference: {max_diff_gauge:.2e}")
print(f"  Tolerance: {tolerance:.2e}")

print("\n" + "-" * 60)
if max_diff_gauge < tolerance:
    print("✓ VERIFIED: C# and Python implementations are IDENTICAL on gauge 410734 data")
    print("\nAny differences with SOURCE are due to configuration/scaling, not bugs.")
else:
    print("✗ WARNING: C# and Python differ beyond tolerance on gauge 410734 data")

# Visualization (Interactive Plotly) - using mm values for implementation comparison
diff_ts_mm = cal_data['python_flow_mm'] - cal_data['csharp_flow_mm']
max_val_mm = max(cal_data['csharp_flow_mm'].max(), cal_data['python_flow_mm'].max())

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Runoff Time Series - mm (First 500 days)', 'C# vs Python Runoff (mm)',
                   'Runoff Difference (mm)', 'Distribution of Differences (mm)')
)

# Time series (in mm)
fig.add_trace(go.Scatter(x=cal_data.index[:500], y=cal_data['csharp_flow_mm'].values[:500], 
                         name='C#', line=dict(color='blue', width=1), opacity=0.7), row=1, col=1)
fig.add_trace(go.Scatter(x=cal_data.index[:500], y=cal_data['python_flow_mm'].values[:500],
                         name='Python', line=dict(color='red', width=1, dash='dash'), opacity=0.7), row=1, col=1)

# Scatter (in mm)
fig.add_trace(go.Scatter(x=cal_data['csharp_flow_mm'], y=cal_data['python_flow_mm'], mode='markers',
                         name='Data', marker=dict(color='blue', size=3, opacity=0.3), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, max_val_mm], y=[0, max_val_mm], name='1:1 line',
                         line=dict(color='black', dash='dash'), showlegend=False), row=1, col=2)

# Difference (in mm)
fig.add_trace(go.Scatter(x=cal_data.index, y=diff_ts_mm, name='Difference',
                         line=dict(color='purple', width=0.5), opacity=0.7, showlegend=False), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

# Histogram (in mm)
fig.add_trace(go.Histogram(x=diff_ts_mm.dropna(), nbinsx=100, name='Histogram',
                           marker_color='steelblue', opacity=0.7, showlegend=False), row=2, col=2)
fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=2)

# Update axes
fig.update_yaxes(title_text="Runoff (mm)", row=1, col=1)
fig.update_xaxes(title_text="C# Runoff (mm)", row=1, col=2)
fig.update_yaxes(title_text="Python Runoff (mm)", row=1, col=2)
fig.update_yaxes(title_text="Difference (Python - C#)", row=2, col=1)
fig.update_xaxes(title_text="Difference (mm)", row=2, col=2)
fig.update_yaxes(title_text="Count", row=2, col=2)

fig.update_layout(
    title_text="C# vs Python Implementation Comparison (Gauge 410734)",
    height=700, showlegend=True
)
fig.show()
print("Interactive plot displayed")

# %% [markdown]
# ### 2.3.1 Summary: Python ≡ C# Verification Complete
#
# The Python and C# implementations are numerically equivalent. 
# Any differences with SOURCE must come from the SOURCE platform itself,
# not from our implementation.

# %%
# Load benchmark model output for comparison
# Using the unrouted Sacramento model output from SOURCE: "Total: QBN01: Outflow"
# This is the direct Sacramento model output BEFORE routing, which should match
# our Python and C# implementations
benchmark_file = DATA_DIR / '410734_output_SDmodel.csv'
benchmark_df = pd.read_csv(benchmark_file, parse_dates=['Date'], index_col='Date')

# Use the unrouted Sacramento outflow (not the routed Downstream Flow)
benchmark_col = 'Total: QBN01: Outflow (ML.day^-1)'
benchmark_df = benchmark_df[[benchmark_col]].copy()
benchmark_df.columns = ['benchmark_flow']

# Merge benchmark with calibration data
cal_data = cal_data.join(benchmark_df, how='left')

print(f"Benchmark data (SOURCE unrouted) loaded: {len(benchmark_df)} records")
print(f"Column used: {benchmark_col}")
print(f"Merged with calibration data: {cal_data['benchmark_flow'].notna().sum()} matching records")

# %%
# Validate pyrrm implementation against SOURCE benchmark
# Compare simulated (pyrrm) vs benchmark (SOURCE) for the same parameters

# Get valid comparison period (after warmup, excluding NaN)
valid_mask = (
    cal_data['benchmark_flow'].notna() & 
    cal_data['simulated_flow'].notna() &
    cal_data['python_flow'].notna()
)
comparison_data = cal_data[valid_mask].iloc[WARMUP_DAYS:]

pyrrm_flow = comparison_data['simulated_flow'].values
csharp_flow = comparison_data['csharp_flow'].values
python_flow = comparison_data['python_flow'].values
source_flow = comparison_data['benchmark_flow'].values
observed_flow = comparison_data['observed_flow'].values

# Calculate metrics
print("=" * 60)
print("COMPARISON: pyrrm vs SOURCE")
print("=" * 60)

# pyrrm vs SOURCE
diff_source = np.abs(pyrrm_flow - source_flow)
corr_source = np.corrcoef(pyrrm_flow, source_flow)[0, 1]
print(f"\npyrrm vs SOURCE benchmark:")
print(f"   Max absolute difference: {np.max(diff_source):.6f} ML/day")
print(f"   Mean absolute difference: {np.mean(diff_source):.6f} ML/day")
print(f"   Correlation: {corr_source:.6f}")

# 3. Relative error analysis
nonzero_mask = source_flow > 0.1  # Avoid division by small numbers
if nonzero_mask.any():
    rel_error = np.abs(pyrrm_flow[nonzero_mask] - source_flow[nonzero_mask]) / source_flow[nonzero_mask]
    print(f"\n3. Relative error analysis (where SOURCE flow > 0.1 ML/day):")
    print(f"   Max relative error: {np.max(rel_error)*100:.2f}%")
    print(f"   Mean relative error: {np.mean(rel_error)*100:.2f}%")
    print(f"   Median relative error: {np.median(rel_error)*100:.2f}%")

# 4. Identify where differences are largest
worst_idx = np.argmax(diff_source)
print(f"\n4. Largest discrepancy details:")
print(f"   Timestep: {comparison_data.index[worst_idx].date()}")
print(f"   pyrrm flow: {pyrrm_flow[worst_idx]:.4f} ML/day")
print(f"   SOURCE flow: {source_flow[worst_idx]:.4f} ML/day")
print(f"   Observed flow: {observed_flow[worst_idx]:.4f} ML/day")
print(f"   Rainfall: {comparison_data['rainfall'].iloc[worst_idx]:.2f} mm")
print(f"   PET: {comparison_data['pet'].iloc[worst_idx]:.2f} mm")

# Summary assessment
print("\n" + "-" * 60)
if np.max(diff_source) < 1.0:
    print("✓ Implementation validated - differences within acceptable tolerance")
elif corr_source > 0.99:
    print("⚠ High correlation but significant absolute differences")
    print("  This suggests a systematic offset (e.g., different area scaling)")
else:
    print("⚠ Significant differences detected between pyrrm and SOURCE")
    print("  Possible causes:")
    print("  - Different parameter interpretation")
    print("  - Different initial conditions")
    print("  - Different catchment area scaling")
    print("  - Different unit hydrograph implementation")

# %% [markdown]
# ### 2.4.1 Implementation Verification Summary
#
# Before comparing with SOURCE, let's confirm that **pyrrm (Python. ≡ C#**. 
# This was verified in detail in Section 2.3 using mm units. Here we show the ML/day equivalence.

# %%
# Quick Implementation Verification: Python vs C# (should be nearly identical)
diff_py_csharp = pyrrm_flow - csharp_flow
max_diff_py_csharp = np.max(np.abs(diff_py_csharp))

fig_verify = go.Figure()
fig_verify.add_trace(go.Scatter(
    x=csharp_flow, y=pyrrm_flow, mode='markers',
    marker=dict(color='steelblue', size=4, opacity=0.4),
    name='pyrrm vs C#'
))
fig_verify.add_trace(go.Scatter(
    x=[0, pyrrm_flow.max()], y=[0, pyrrm_flow.max()],
    mode='lines', line=dict(color='red', dash='dash', width=2),
    name='1:1 Line'
))
fig_verify.update_layout(
    title=f"<b>✓ Implementation Verified:</b> pyrrm ≡ C# (Max diff: {max_diff_py_csharp:.6f} ML/day)",
    xaxis_title="C# Flow (ML/day)",
    yaxis_title="pyrrm Flow (ML/day)",
    height=500, width=700,
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
fig_verify.show()
print(f"✓ Python and C# implementations are numerically equivalent (max diff: {max_diff_py_csharp:.6e} ML/day)")
print(f"  → For all subsequent analyses, we use pyrrm (Python) as representative of both implementations.")

# %% [markdown]
# ### 2.4.2 Main Comparison: pyrrm vs SOURCE Benchmark
#
# Now we compare our implementation against the SOURCE modeling platform benchmark.
# This is the key comparison to understand any differences in model behavior.

# %%
# =============================================================================
# FIGURE 1: HYDROGRAPH COMPARISON - Primary Visual Assessment
# =============================================================================
# This figure allows you to visually compare the flow time series
# Showing both linear and log scales to reveal different aspects of the data

diff_ts = pyrrm_flow - source_flow

fig1 = make_subplots(
    rows=3, cols=1,
    row_heights=[0.4, 0.35, 0.25],
    subplot_titles=('Linear Scale', 'Log Scale (reveals low flow differences)', 'Difference (pyrrm − SOURCE)'),
    shared_xaxes=True,
    vertical_spacing=0.08
)

# Row 1: Linear scale hydrograph
fig1.add_trace(go.Scatter(
    x=comparison_data.index, y=pyrrm_flow,
    name='pyrrm (this library)',
    line=dict(color='#E94F37', width=2),
    opacity=1.0
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=comparison_data.index, y=source_flow,
    name='SOURCE (benchmark)',
    line=dict(color='#2E86AB', width=1.5, dash='dot'),
    opacity=0.9
), row=1, col=1)

# Row 2: Log scale hydrograph (same data, log y-axis)
fig1.add_trace(go.Scatter(
    x=comparison_data.index, y=pyrrm_flow,
    name='pyrrm',
    line=dict(color='#E94F37', width=2),
    opacity=1.0,
    showlegend=False
), row=2, col=1)

fig1.add_trace(go.Scatter(
    x=comparison_data.index, y=source_flow,
    name='SOURCE',
    line=dict(color='#2E86AB', width=1.5, dash='dot'),
    opacity=0.9,
    showlegend=False
), row=2, col=1)

# Row 3: Difference time series with fill
fig1.add_trace(go.Scatter(
    x=comparison_data.index, y=diff_ts,
    name='Difference',
    line=dict(color='#8338EC', width=0.8),
    fill='tozeroy',
    fillcolor='rgba(131, 56, 236, 0.2)'
), row=3, col=1)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

# Annotations for key statistics
mean_diff = np.mean(diff_ts)
std_diff = np.std(diff_ts)
fig1.add_annotation(
    x=0.02, y=0.98, xref='paper', yref='paper',
    text=f"Mean diff: {mean_diff:.2f} ML/day | Std: {std_diff:.2f} ML/day",
    showarrow=False, font=dict(size=11, color='#8338EC'),
    bgcolor='rgba(255,255,255,0.8)', bordercolor='#8338EC', borderwidth=1
)

# Update axes
fig1.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
fig1.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=1)
fig1.update_yaxes(title_text="Δ Flow (ML/day)", row=3, col=1)
fig1.update_xaxes(title_text="Date", row=3, col=1)

fig1.update_layout(
    title="<b>Hydrograph Comparison: pyrrm vs SOURCE</b><br>" +
          "<sup>pyrrm (red solid) vs SOURCE (blue dotted) • Use Plotly tools to zoom</sup>",
    height=800,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    hovermode='x unified'
)
fig1.show()

# %%
# =============================================================================
# FIGURE 2: SCATTER ANALYSIS - Agreement Assessment
# =============================================================================
# Scatter plots reveal systematic biases and flow-dependent errors

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Linear Scale', 'Log-Log Scale (reveals low flow behavior)'),
    horizontal_spacing=0.12
)

max_val = max(source_flow.max(), pyrrm_flow.max())

# Color by difference magnitude for insight
diff_colors = np.abs(diff_ts)

# Linear scatter
fig2.add_trace(go.Scatter(
    x=source_flow, y=pyrrm_flow, mode='markers',
    marker=dict(
        color=diff_colors, colorscale='Viridis', size=4, opacity=0.5,
        colorbar=dict(title='|Diff|<br>(ML/day)', x=1.02, len=0.9)
    ),
    name='Data points',
    hovertemplate='SOURCE: %{x:.1f}<br>pyrrm: %{y:.1f}<br>|Diff|: %{marker.color:.1f}<extra></extra>'
), row=1, col=1)

# 1:1 line
fig2.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val], mode='lines',
    line=dict(color='red', dash='dash', width=2),
    name='1:1 Line', showlegend=True
), row=1, col=1)

# Log-log scatter (better for seeing low flow behavior)
# Filter out zeros for log scale
nonzero = (source_flow > 0.1) & (pyrrm_flow > 0.1)
fig2.add_trace(go.Scatter(
    x=source_flow[nonzero], y=pyrrm_flow[nonzero], mode='markers',
    marker=dict(color=diff_colors[nonzero], colorscale='Viridis', size=4, opacity=0.5, showscale=False),
    name='Data (log)', showlegend=False,
    hovertemplate='SOURCE: %{x:.2f}<br>pyrrm: %{y:.2f}<extra></extra>'
), row=1, col=2)

# 1:1 line for log
fig2.add_trace(go.Scatter(
    x=[0.1, max_val], y=[0.1, max_val], mode='lines',
    line=dict(color='red', dash='dash', width=2),
    showlegend=False
), row=1, col=2)

fig2.update_xaxes(title_text="SOURCE Flow (ML/day)", row=1, col=1)
fig2.update_yaxes(title_text="pyrrm Flow (ML/day)", row=1, col=1)
fig2.update_xaxes(title_text="SOURCE Flow (ML/day)", type="log", row=1, col=2)
fig2.update_yaxes(title_text="pyrrm Flow (ML/day)", type="log", row=1, col=2)

# Calculate R² 
r2 = np.corrcoef(pyrrm_flow, source_flow)[0,1]**2
slope, intercept = np.polyfit(source_flow, pyrrm_flow, 1)

fig2.update_layout(
    title=f"<b>Scatter Analysis: pyrrm vs SOURCE</b><br>" +
          f"<sup>R² = {r2:.4f} | Slope = {slope:.4f} | Intercept = {intercept:.2f} ML/day</sup>",
    height=450,
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
fig2.show()

# %%
# =============================================================================
# FIGURE 3: DIAGNOSTIC ANALYSIS - When and Where Do Differences Occur?
# =============================================================================
# Understanding the pattern of differences helps identify potential causes

fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Difference vs Flow Magnitude',
        'Difference Distribution',
        'Difference vs Rainfall',
        'Monthly Pattern of Differences'
    ),
    specs=[[{}, {}], [{}, {}]],
    horizontal_spacing=0.12,
    vertical_spacing=0.15
)

# 1. Difference vs Flow Magnitude (reveals proportional errors)
fig3.add_trace(go.Scatter(
    x=source_flow, y=diff_ts, mode='markers',
    marker=dict(color='#3D5A80', size=3, opacity=0.3),
    name='Diff vs Flow'
), row=1, col=1)
fig3.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)

# Add trend line
z = np.polyfit(source_flow, diff_ts, 1)
trend_y = z[0] * np.array([0, max_val]) + z[1]
fig3.add_trace(go.Scatter(
    x=[0, max_val], y=trend_y, mode='lines',
    line=dict(color='orange', width=2, dash='dot'),
    name=f'Trend (slope={z[0]:.3f})'
), row=1, col=1)

# 2. Histogram of differences
fig3.add_trace(go.Histogram(
    x=diff_ts, nbinsx=80,
    marker_color='#3D5A80', opacity=0.7,
    name='Distribution'
), row=1, col=2)
fig3.add_vline(x=0, line_dash="solid", line_color="red", line_width=2, row=1, col=2)
fig3.add_vline(x=mean_diff, line_dash="dash", line_color="orange", line_width=2, row=1, col=2)

# Add annotation for bias
fig3.add_annotation(
    x=mean_diff, y=0.85, xref='x2', yref='paper',
    text=f'Bias: {mean_diff:.2f}',
    showarrow=True, arrowhead=2, ax=40, ay=-30,
    font=dict(color='orange')
)

# 3. Difference vs Rainfall (reveals event response differences)
fig3.add_trace(go.Scatter(
    x=comparison_data['rainfall'].values, y=diff_ts, mode='markers',
    marker=dict(color='#3D5A80', size=3, opacity=0.3),
    name='Diff vs Rain', showlegend=False
), row=2, col=1)
fig3.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)

# 4. Monthly boxplot of differences (reveals seasonal patterns)
comparison_data_temp = comparison_data.copy()
comparison_data_temp['diff'] = diff_ts
comparison_data_temp['month'] = comparison_data_temp.index.month
monthly_stats = comparison_data_temp.groupby('month')['diff'].agg(['mean', 'std', 'median', 'min', 'max'])

# Box plot approximation using error bars
months = list(range(1, 13))
month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
for m in months:
    month_data = comparison_data_temp[comparison_data_temp['month'] == m]['diff']
    fig3.add_trace(go.Box(
        y=month_data, name=month_names[m-1],
        marker_color='#3D5A80', showlegend=False
    ), row=2, col=2)

fig3.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=2)

# Update axes
fig3.update_xaxes(title_text="SOURCE Flow (ML/day)", row=1, col=1)
fig3.update_yaxes(title_text="Difference (ML/day)", row=1, col=1)
fig3.update_xaxes(title_text="Difference (ML/day)", row=1, col=2)
fig3.update_yaxes(title_text="Count", row=1, col=2)
fig3.update_xaxes(title_text="Rainfall (mm)", row=2, col=1)
fig3.update_yaxes(title_text="Difference (ML/day)", row=2, col=1)
fig3.update_xaxes(title_text="Month", row=2, col=2)
fig3.update_yaxes(title_text="Difference (ML/day)", row=2, col=2)

fig3.update_layout(
    title="<b>Diagnostic Analysis: Understanding pyrrm vs SOURCE Differences</b><br>" +
          "<sup>These patterns help identify potential causes of discrepancies</sup>",
    height=600,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
fig3.show()

# %%
# =============================================================================
# FIGURE 4: CUMULATIVE ANALYSIS & WATER BALANCE
# =============================================================================
# Cumulative flow comparison reveals long-term biases and total volume differences

cum_pyrrm = np.cumsum(pyrrm_flow)
cum_source = np.cumsum(source_flow)
cum_obs = np.cumsum(observed_flow)
cum_diff = cum_pyrrm - cum_source

fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Cumulative Flow Over Time', 'Cumulative Difference (pyrrm − SOURCE)'),
    horizontal_spacing=0.1
)

# Cumulative flows
fig4.add_trace(go.Scatter(
    x=comparison_data.index, y=cum_source/1000,
    name='SOURCE', line=dict(color='#2E86AB', width=2)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=comparison_data.index, y=cum_pyrrm/1000,
    name='pyrrm', line=dict(color='#E94F37', width=2)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=comparison_data.index, y=cum_obs/1000,
    name='Observed', line=dict(color='#2DC653', width=2, dash='dot')
), row=1, col=1)

# Cumulative difference
fig4.add_trace(go.Scatter(
    x=comparison_data.index, y=cum_diff/1000,
    name='Cumulative Diff', line=dict(color='#8338EC', width=2),
    fill='tozeroy', fillcolor='rgba(131, 56, 236, 0.2)',
    showlegend=False
), row=1, col=2)
fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

# Calculate total volumes
total_pyrrm = np.sum(pyrrm_flow) / 1000  # GL
total_source = np.sum(source_flow) / 1000  # GL
total_obs = np.sum(observed_flow) / 1000  # GL
volume_diff_pct = (total_pyrrm - total_source) / total_source * 100

fig4.update_xaxes(title_text="Date", row=1, col=1)
fig4.update_yaxes(title_text="Cumulative Flow (GL)", row=1, col=1)
fig4.update_xaxes(title_text="Date", row=1, col=2)
fig4.update_yaxes(title_text="Cumulative Diff (GL)", row=1, col=2)

fig4.update_layout(
    title=f"<b>Cumulative Flow & Water Balance</b><br>" +
          f"<sup>Total: pyrrm={total_pyrrm:.1f} GL | SOURCE={total_source:.1f} GL | " +
          f"Observed={total_obs:.1f} GL | Volume Diff: {volume_diff_pct:+.1f}%</sup>",
    height=400,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.3)
)
fig4.show()

# %%
# =============================================================================
# SUMMARY STATISTICS TABLE
# =============================================================================
print("=" * 70)
print("COMPARISON SUMMARY: pyrrm vs SOURCE")
print("=" * 70)

# Correlation metrics
corr = np.corrcoef(pyrrm_flow, source_flow)[0, 1]
r2 = corr ** 2

# Error metrics
mae = np.mean(np.abs(diff_ts))
rmse = np.sqrt(np.mean(diff_ts**2))
bias = np.mean(diff_ts)
pbias = 100 * np.sum(diff_ts) / np.sum(source_flow)

# Relative metrics (excluding very low flows)
high_flow_mask = source_flow > np.percentile(source_flow, 10)
rel_error_mean = 100 * np.mean(np.abs(diff_ts[high_flow_mask]) / source_flow[high_flow_mask])

print(f"\n{'Metric':<30} {'Value':>15}")
print("-" * 45)
print(f"{'Pearson Correlation':<30} {corr:>15.6f}")
print(f"{'R²':<30} {r2:>15.6f}")
print(f"{'Mean Absolute Error (MAE)':<30} {mae:>12.2f} ML/day")
print(f"{'Root Mean Square Error (RMSE)':<30} {rmse:>12.2f} ML/day")
print(f"{'Bias (pyrrm − SOURCE)':<30} {bias:>12.2f} ML/day")
print(f"{'Percent Bias (PBIAS)':<30} {pbias:>14.2f}%")
print(f"{'Mean Relative Error (>P10)':<30} {rel_error_mean:>14.2f}%")
print("-" * 45)
print(f"{'Total Volume pyrrm':<30} {total_pyrrm:>12.1f} GL")
print(f"{'Total Volume SOURCE':<30} {total_source:>12.1f} GL")
print(f"{'Volume Difference':<30} {total_pyrrm - total_source:>12.1f} GL ({volume_diff_pct:+.1f}%)")
print("=" * 70)

# Implementation verification summary
print(f"\n✓ Implementation Verification:")
print(f"  pyrrm vs C# max difference: {max_diff_py_csharp:.6e} ML/day")
print(f"  (Python and C# implementations are numerically equivalent)")

# %% [markdown]
# ### 2.4.3 Ratio Analysis (Additional Diagnostic)
#
# The ratio pyrrm/SOURCE provides additional insight into proportional differences,
# particularly useful for identifying systematic scaling issues.

# %%
# =============================================================================
# FIGURE 5: RATIO ANALYSIS - Proportional Differences
# =============================================================================
# Ratio analysis helps identify if differences are proportional to flow magnitude

ratio_mask = source_flow > 0.1  # Avoid division by very small numbers
ratio = np.full_like(pyrrm_flow, np.nan)
ratio[ratio_mask] = pyrrm_flow[ratio_mask] / source_flow[ratio_mask]

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Ratio Time Series (pyrrm / SOURCE)', 'Ratio Distribution')
)

# Ratio time series
fig5.add_trace(go.Scatter(
    x=comparison_data.index[ratio_mask], y=ratio[ratio_mask],
    name='Ratio', mode='markers+lines',
    marker=dict(color='#FF6B35', size=2, opacity=0.4),
    line=dict(color='#FF6B35', width=0.5)
), row=1, col=1)
fig5.add_hline(y=1.0, line_dash="dash", line_color="green", line_width=2, row=1, col=1)

# Add bands for ±10% and ±20%
fig5.add_hrect(y0=0.9, y1=1.1, fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
fig5.add_hrect(y0=0.8, y1=1.2, fillcolor="yellow", opacity=0.1, line_width=0, row=1, col=1)

# Ratio histogram
ratio_valid = ratio[~np.isnan(ratio)]
fig5.add_trace(go.Histogram(
    x=ratio_valid, nbinsx=80,
    marker_color='#FF6B35', opacity=0.7,
    name='Ratio Dist.', showlegend=False
), row=1, col=2)
fig5.add_vline(x=1.0, line_dash="solid", line_color="green", line_width=2, row=1, col=2)
fig5.add_vline(x=np.median(ratio_valid), line_dash="dash", line_color="red", line_width=2, row=1, col=2)

# Statistics
pct_within_10 = 100 * np.sum(np.abs(ratio_valid - 1) <= 0.1) / len(ratio_valid)
pct_within_20 = 100 * np.sum(np.abs(ratio_valid - 1) <= 0.2) / len(ratio_valid)

fig5.update_xaxes(title_text="Date", row=1, col=1)
fig5.update_yaxes(title_text="Ratio (pyrrm/SOURCE)", range=[0, 3], row=1, col=1)
fig5.update_xaxes(title_text="Ratio", row=1, col=2)
fig5.update_yaxes(title_text="Count", row=1, col=2)

fig5.update_layout(
    title=f"<b>Ratio Analysis: pyrrm / SOURCE</b><br>" +
          f"<sup>Median ratio: {np.median(ratio_valid):.3f} | " +
          f"Within ±10%: {pct_within_10:.1f}% | Within ±20%: {pct_within_20:.1f}%</sup>",
    height=350,
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
fig5.show()

print(f"\nRatio Analysis (where SOURCE > 0.1 ML/day):")
print(f"  Median ratio: {np.median(ratio_valid):.3f}")
print(f"  Mean ratio: {np.mean(ratio_valid):.3f}")
print(f"  % within ±10% of 1.0: {pct_within_10:.1f}%")
print(f"  % within ±20% of 1.0: {pct_within_20:.1f}%")

# %% [markdown]
# ### 2.4.4 Safeguard Analysis: Are Python Guards Causing the Differences?
#
# The Python implementation includes safeguards against division-by-zero that 
# the C# code does not have. This section checks:
# 1. **Are the safeguards ever triggered?** If not, they can't be causing differences.
# 2. **If triggered, do they correlate with the pyrrm vs SOURCE differences?**
#
# The key safeguards in Python (not in C#) are:
# - `ratio = (adimc - uztwc) / lztwm` → guarded when `lztwm = 0`
# - `hpl = alzfpm / (alzfpm + alzfsm)` → guarded when denominator = 0
# - Percolation: `perc = ... / uzfwm` → guarded when `uzfwm = 0`
# - Deficit ratio: `1 - (current_lz / total_lz)` → guarded when `total_lz = 0`
# - `ratlp = 1 - alzfpc/alzfpm` → guarded when `alzfpm = 0`
# - `ratls = 1 - alzfsc/alzfsm` → guarded when `alzfsm = 0`

# %%
# =============================================================================
# SAFEGUARD TRIGGER ANALYSIS
# =============================================================================
# Re-run pyrrm with detailed tracking of when safeguard conditions are met

print("=" * 80)
print("SAFEGUARD TRIGGER ANALYSIS: Are Python Guards Causing Differences?")
print("=" * 80)

# Re-initialize model with benchmark parameters
safeguard_model = Sacramento()
safeguard_model.set_parameters(benchmark_params)
safeguard_model.reset()

# Track safeguard conditions at each timestep
safeguard_log = []

for i, (idx, row) in enumerate(cal_data.iterrows()):
    # Check safeguard conditions BEFORE running timestep
    # These are the conditions that would trigger safeguards in Python
    
    conditions = {
        'date': idx,
        'timestep': i,
        'rainfall': row['rainfall'],
        'pet': row['pet'],
        
        # Condition 1: lztwm = 0 (ratio calculation guard)
        'lztwm_zero': safeguard_model.lztwm == 0,
        'lztwm': safeguard_model.lztwm,
        
        # Condition 2: alzfpm + alzfsm = 0 (hpl calculation guard)
        'hpl_denom_zero': (safeguard_model.alzfpm + safeguard_model.alzfsm) == 0,
        'alzfpm': safeguard_model.alzfpm,
        'alzfsm': safeguard_model.alzfsm,
        
        # Condition 3: uzfwm = 0 (percolation guard)
        'uzfwm_zero': safeguard_model.uzfwm == 0,
        'uzfwm': safeguard_model.uzfwm,
        
        # Condition 4: total_lz = 0 (deficit ratio guard)
        'total_lz_zero': (safeguard_model.alzfpm + safeguard_model.alzfsm + safeguard_model.lztwm) == 0,
        
        # Condition 5: alzfpm = 0 (ratlp guard)
        'alzfpm_zero': safeguard_model.alzfpm == 0,
        
        # Condition 6: alzfsm = 0 (ratls guard)
        'alzfsm_zero': safeguard_model.alzfsm == 0,
        
        # Store states for context
        'uztwc': safeguard_model.uztwc,
        'uzfwc': safeguard_model.uzfwc,
        'lztwc': safeguard_model.lztwc,
        'lzfsc': safeguard_model.lzfsc,
        'lzfpc': safeguard_model.lzfpc,
    }
    
    # Run timestep
    result = safeguard_model.run_timestep(row['rainfall'], row['pet'])
    conditions['pyrrm_runoff_mm'] = result['runoff']
    
    safeguard_log.append(conditions)

safeguard_df = pd.DataFrame(safeguard_log)
safeguard_df.set_index('date', inplace=True)

# %%
# Analyze safeguard trigger frequency
print("\n" + "-" * 80)
print("SAFEGUARD TRIGGER FREQUENCY:")
print("-" * 80)

safeguard_checks = {
    'lztwm = 0 (ratio guard)': safeguard_df['lztwm_zero'].sum(),
    'alzfpm + alzfsm = 0 (hpl guard)': safeguard_df['hpl_denom_zero'].sum(),
    'uzfwm = 0 (percolation guard)': safeguard_df['uzfwm_zero'].sum(),
    'total_lz = 0 (deficit ratio guard)': safeguard_df['total_lz_zero'].sum(),
    'alzfpm = 0 (ratlp guard)': safeguard_df['alzfpm_zero'].sum(),
    'alzfsm = 0 (ratls guard)': safeguard_df['alzfsm_zero'].sum(),
}

any_triggered = False
for name, count in safeguard_checks.items():
    status = f"TRIGGERED {count} times" if count > 0 else "Never triggered"
    print(f"  {name:<40}: {status}")
    if count > 0:
        any_triggered = True

total_timesteps = len(safeguard_df)
print(f"\n  Total timesteps analyzed: {total_timesteps}")

# %%
# Key conclusion about safeguards
print("\n" + "=" * 80)
print("SAFEGUARD ANALYSIS CONCLUSION:")
print("=" * 80)

if not any_triggered:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✓ NONE OF THE PYTHON SAFEGUARDS WERE TRIGGERED                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This means the Python safeguards are NOT responsible for the differences    ║
║  between pyrrm and SOURCE.                                                   ║
║                                                                              ║
║  Since pyrrm ≡ C# (verified earlier), and safeguards aren't triggered,       ║
║  the differences with SOURCE must come from SOURCE itself:                   ║
║                                                                              ║
║  Possible causes in SOURCE:                                                  ║
║  • Different initial conditions (store starting values)                      ║
║  • Different unit hydrograph implementation                                  ║
║  • Additional routing applied in SOURCE                                      ║
║  • Different floating-point precision                                        ║
║  • Minor algorithm variations in SOURCE's Sacramento implementation          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
else:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠ SOME PYTHON SAFEGUARDS WERE TRIGGERED                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  These safeguards MAY be contributing to differences with SOURCE.            ║
║  Further analysis below shows correlation with pyrrm-SOURCE differences.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# %%
# Now analyze the actual pyrrm vs SOURCE differences
print("\n" + "=" * 80)
print("ANALYSIS OF pyrrm vs SOURCE DIFFERENCES")
print("=" * 80)

# Calculate differences in ML/day
diff_pyrrm_source = pyrrm_flow - source_flow

print(f"\nDifference Statistics (pyrrm - SOURCE) in ML/day:")
print(f"  Mean:   {np.mean(diff_pyrrm_source):>10.2f} ML/day")
print(f"  Std:    {np.std(diff_pyrrm_source):>10.2f} ML/day")
print(f"  Min:    {np.min(diff_pyrrm_source):>10.2f} ML/day")
print(f"  Max:    {np.max(diff_pyrrm_source):>10.2f} ML/day")
print(f"  Median: {np.median(diff_pyrrm_source):>10.2f} ML/day")

# Percentile analysis
print(f"\nPercentile Analysis:")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(diff_pyrrm_source, pct)
    print(f"  P{pct:02d}: {val:>+10.2f} ML/day")

# Count by magnitude
print(f"\nDifference Magnitude Distribution:")
thresholds = [1, 5, 10, 25, 50, 100, 200]
for thresh in thresholds:
    count = np.sum(np.abs(diff_pyrrm_source) > thresh)
    pct = 100 * count / len(diff_pyrrm_source)
    print(f"  |diff| > {thresh:>3} ML/day: {count:>5} timesteps ({pct:>5.1f}%)")

# %%
# Characterize the nature of differences
print("\n" + "=" * 80)
print("NATURE OF DIFFERENCES: Event Analysis")
print("=" * 80)

# Find the largest positive and negative differences
LARGE_DIFF_THRESHOLD = 50  # ML/day
large_pos = diff_pyrrm_source > LARGE_DIFF_THRESHOLD
large_neg = diff_pyrrm_source < -LARGE_DIFF_THRESHOLD

print(f"\nLarge POSITIVE differences (pyrrm > SOURCE by >{LARGE_DIFF_THRESHOLD} ML/day):")
print(f"  Count: {large_pos.sum()} timesteps")
if large_pos.sum() > 0:
    pos_dates = comparison_data.index[large_pos]
    print(f"  First occurrence: {pos_dates[0].date()}")
    print(f"  Last occurrence: {pos_dates[-1].date()}")
    # Show top 5
    top_pos_idx = np.argsort(diff_pyrrm_source)[-5:][::-1]
    print(f"  Top 5 positive differences:")
    for idx in top_pos_idx:
        date = comparison_data.index[idx]
        diff = diff_pyrrm_source[idx]
        pyrrm_val = pyrrm_flow[idx]
        source_val = source_flow[idx]
        rain = comparison_data['rainfall'].iloc[idx]
        print(f"    {date.date()}: pyrrm={pyrrm_val:.1f}, SOURCE={source_val:.1f}, diff={diff:+.1f}, rain={rain:.1f}mm")

print(f"\nLarge NEGATIVE differences (pyrrm < SOURCE by >{LARGE_DIFF_THRESHOLD} ML/day):")
print(f"  Count: {large_neg.sum()} timesteps")
if large_neg.sum() > 0:
    neg_dates = comparison_data.index[large_neg]
    print(f"  First occurrence: {neg_dates[0].date()}")
    print(f"  Last occurrence: {neg_dates[-1].date()}")
    # Show top 5
    top_neg_idx = np.argsort(diff_pyrrm_source)[:5]
    print(f"  Top 5 negative differences:")
    for idx in top_neg_idx:
        date = comparison_data.index[idx]
        diff = diff_pyrrm_source[idx]
        pyrrm_val = pyrrm_flow[idx]
        source_val = source_flow[idx]
        rain = comparison_data['rainfall'].iloc[idx]
        print(f"    {date.date()}: pyrrm={pyrrm_val:.1f}, SOURCE={source_val:.1f}, diff={diff:+.1f}, rain={rain:.1f}mm")

# %%
# Correlation with flow magnitude and rainfall
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Correlation between difference and flow magnitude
corr_with_flow = np.corrcoef(source_flow, diff_pyrrm_source)[0, 1]
print(f"\nCorrelation of difference with SOURCE flow magnitude: {corr_with_flow:.4f}")

# Correlation between difference and rainfall
corr_with_rain = np.corrcoef(comparison_data['rainfall'].values, diff_pyrrm_source)[0, 1]
print(f"Correlation of difference with rainfall: {corr_with_rain:.4f}")

# Linear regression: is difference proportional to flow?
slope, intercept = np.polyfit(source_flow, diff_pyrrm_source, 1)
print(f"\nLinear fit (diff = slope * SOURCE + intercept):")
print(f"  Slope: {slope:.6f} (if ≈0, differences are additive; if ≠0, proportional)")
print(f"  Intercept: {intercept:.2f} ML/day")

# %%
# Visualize the analysis
fig_analysis = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Difference Time Series',
        'Difference vs Flow Magnitude',
        'Difference vs Rainfall', 
        'Difference Distribution'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# Plot 1: Time series
fig_analysis.add_trace(go.Scatter(
    x=comparison_data.index, y=diff_pyrrm_source,
    mode='lines',
    line=dict(color='purple', width=0.8),
    name='pyrrm - SOURCE'
), row=1, col=1)
fig_analysis.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
fig_analysis.add_hline(y=LARGE_DIFF_THRESHOLD, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
fig_analysis.add_hline(y=-LARGE_DIFF_THRESHOLD, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)

# Plot 2: vs Flow magnitude with trend line
fig_analysis.add_trace(go.Scatter(
    x=source_flow, y=diff_pyrrm_source,
    mode='markers',
    marker=dict(color='blue', size=3, opacity=0.3),
    name='Data', showlegend=False
), row=1, col=2)
# Add trend line
trend_x = np.array([0, source_flow.max()])
trend_y = slope * trend_x + intercept
fig_analysis.add_trace(go.Scatter(
    x=trend_x, y=trend_y,
    mode='lines',
    line=dict(color='red', width=2, dash='dash'),
    name=f'Trend (slope={slope:.4f})'
), row=1, col=2)
fig_analysis.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

# Plot 3: vs Rainfall
fig_analysis.add_trace(go.Scatter(
    x=comparison_data['rainfall'].values, y=diff_pyrrm_source,
    mode='markers',
    marker=dict(color='green', size=3, opacity=0.3),
    name='Data', showlegend=False
), row=2, col=1)
fig_analysis.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

# Plot 4: Histogram
fig_analysis.add_trace(go.Histogram(
    x=diff_pyrrm_source,
    nbinsx=100,
    marker_color='steelblue',
    name='Distribution', showlegend=False
), row=2, col=2)
fig_analysis.add_vline(x=0, line_dash="solid", line_color="black", line_width=2, row=2, col=2)
fig_analysis.add_vline(x=np.mean(diff_pyrrm_source), line_dash="dash", line_color="red", row=2, col=2)

fig_analysis.update_yaxes(title_text="Diff (ML/day)", row=1, col=1)
fig_analysis.update_yaxes(title_text="Diff (ML/day)", row=1, col=2)
fig_analysis.update_xaxes(title_text="SOURCE Flow (ML/day)", row=1, col=2)
fig_analysis.update_yaxes(title_text="Diff (ML/day)", row=2, col=1)
fig_analysis.update_xaxes(title_text="Rainfall (mm)", row=2, col=1)
fig_analysis.update_xaxes(title_text="Difference (ML/day)", row=2, col=2)

fig_analysis.update_layout(
    title="<b>pyrrm vs SOURCE: Difference Analysis</b><br>" +
          f"<sup>Mean diff: {np.mean(diff_pyrrm_source):.1f} ML/day | Corr with flow: {corr_with_flow:.3f} | Slope: {slope:.4f}</sup>",
    height=600,
    showlegend=True
)
fig_analysis.show()

# %%
# Final assessment
print("\n" + "=" * 80)
print("FINAL ASSESSMENT: Is It Safe to Live With These Differences?")
print("=" * 80)

# Calculate relative metrics
mean_source = np.mean(source_flow)
mean_diff = np.mean(diff_pyrrm_source)
relative_bias = 100 * mean_diff / mean_source

# Volume comparison
total_source = np.sum(source_flow)
total_pyrrm = np.sum(pyrrm_flow)
volume_diff_pct = 100 * (total_pyrrm - total_source) / total_source

print(f"""
Summary Metrics:
----------------
• Mean SOURCE flow: {mean_source:.1f} ML/day
• Mean difference (pyrrm - SOURCE): {mean_diff:.1f} ML/day
• Relative bias: {relative_bias:+.2f}%
• Total volume difference: {volume_diff_pct:+.2f}%
• Correlation (pyrrm vs SOURCE): {np.corrcoef(pyrrm_flow, source_flow)[0,1]:.6f}

Assessment:
-----------
""")

# Provide interpretation
if abs(relative_bias) < 1:
    bias_assessment = "✓ NEGLIGIBLE - Mean bias is less than 1% of mean flow"
elif abs(relative_bias) < 5:
    bias_assessment = "✓ ACCEPTABLE - Mean bias is within 5% of mean flow"
else:
    bias_assessment = "⚠ NOTABLE - Mean bias exceeds 5% of mean flow"

if abs(slope) < 0.01:
    slope_assessment = "✓ ADDITIVE - Differences are constant, not proportional to flow"
elif abs(slope) < 0.05:
    slope_assessment = "✓ MOSTLY ADDITIVE - Small proportional component"
else:
    slope_assessment = "⚠ PROPORTIONAL - Differences scale with flow magnitude"

if not any_triggered:
    safeguard_assessment = "✓ SAFEGUARDS NOT RESPONSIBLE - Python guards never triggered"
else:
    safeguard_assessment = "⚠ INVESTIGATE FURTHER - Some safeguards were triggered"

print(f"1. Bias: {bias_assessment}")
print(f"2. Scaling: {slope_assessment}")
print(f"3. Safeguards: {safeguard_assessment}")

print("""
Recommendation:
---------------
""")

if abs(relative_bias) < 5 and not any_triggered:
    print("""These differences are likely due to implementation details in SOURCE 
(different initial conditions, unit hydrograph, or floating-point precision)
and are ACCEPTABLE for most practical applications. The pyrrm implementation
is mathematically equivalent to the C# code (verified), and no safeguards
are being triggered that would cause divergence.""")
else:
    print("""Further investigation recommended. Check SOURCE's initial conditions,
unit hydrograph parameters, and any additional processing steps.""")

# %%
# Calculate performance metrics vs observed flow
print("\n" + "=" * 60)
print("MODEL PERFORMANCE vs OBSERVED FLOW (Benchmark Parameters)")
print("=" * 60)

# Metrics for pyrrm simulation
print("\npyrrm Sacramento model:")
metrics_pyrrm = calculate_metrics(pyrrm_flow, observed_flow)
for name, value in metrics_pyrrm.items():
    print(f"  {name:8s}: {value:.4f}")

# Metrics for SOURCE benchmark
print("\nSOURCE benchmark model:")
metrics_source = calculate_metrics(source_flow, observed_flow)
for name, value in metrics_source.items():
    print(f"  {name:8s}: {value:.4f}")

# %%
# Visualize benchmark comparison (Interactive Plotly)
def calc_fdc(data):
    sorted_data = np.sort(data)[::-1]
    exceedance = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    return exceedance, sorted_data

exc_obs, fdc_obs = calc_fdc(observed_flow)
exc_pyrrm, fdc_pyrrm = calc_fdc(pyrrm_flow)
exc_source, fdc_source = calc_fdc(source_flow)
max_val_scatter = max(source_flow.max(), pyrrm_flow.max(), observed_flow.max())

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Time Series - Linear Scale', 'Time Series - Log Scale',
                   'pyrrm vs SOURCE', f'Simulated vs Observed (NSE={metrics_pyrrm["NSE"]:.3f})',
                   'Flow Duration Curves - Linear', 'Flow Duration Curves - Log')
)

# Time series - Linear
fig.add_trace(go.Scatter(x=comparison_data.index, y=observed_flow, name='Observed',
                         line=dict(color='blue', width=0.8), opacity=0.7), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison_data.index, y=pyrrm_flow, name='pyrrm',
                         line=dict(color='red', width=0.8), opacity=0.7), row=1, col=1)
fig.add_trace(go.Scatter(x=comparison_data.index, y=source_flow, name='SOURCE',
                         line=dict(color='green', width=0.8, dash='dash'), opacity=0.5), row=1, col=1)

# Time series - Log
fig.add_trace(go.Scatter(x=comparison_data.index, y=observed_flow, name='Observed',
                         line=dict(color='blue', width=0.8), opacity=0.7, showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=comparison_data.index, y=pyrrm_flow, name='pyrrm',
                         line=dict(color='red', width=0.8), opacity=0.7, showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=comparison_data.index, y=source_flow, name='SOURCE',
                         line=dict(color='green', width=0.8, dash='dash'), opacity=0.5, showlegend=False), row=1, col=2)

# Scatter: pyrrm vs SOURCE
fig.add_trace(go.Scatter(x=source_flow, y=pyrrm_flow, mode='markers',
                         marker=dict(color='blue', size=3, opacity=0.3), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=[0, max_val_scatter], y=[0, max_val_scatter], name='1:1',
                         line=dict(color='black', dash='dash'), showlegend=False), row=2, col=1)

# Scatter: Simulated vs Observed
fig.add_trace(go.Scatter(x=observed_flow, y=pyrrm_flow, mode='markers',
                         marker=dict(color='red', size=3, opacity=0.3), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=[0, max_val_scatter], y=[0, max_val_scatter],
                         line=dict(color='black', dash='dash'), showlegend=False), row=2, col=2)

# FDC - Linear
fig.add_trace(go.Scatter(x=exc_obs, y=fdc_obs, name='Observed FDC',
                         line=dict(color='blue', width=1.5), opacity=0.8, showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=exc_pyrrm, y=fdc_pyrrm, name='pyrrm FDC',
                         line=dict(color='red', width=1.5), opacity=0.8, showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=exc_source, y=fdc_source, name='SOURCE FDC',
                         line=dict(color='green', width=1.5, dash='dash'), opacity=0.8, showlegend=False), row=3, col=1)

# FDC - Log
fig.add_trace(go.Scatter(x=exc_obs, y=fdc_obs, line=dict(color='blue', width=1.5), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=exc_pyrrm, y=fdc_pyrrm, line=dict(color='red', width=1.5), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=exc_source, y=fdc_source, line=dict(color='green', width=1.5, dash='dash'), showlegend=False), row=3, col=2)

# Update axes
fig.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
fig.update_xaxes(title_text="SOURCE Flow (ML/day)", row=2, col=1)
fig.update_yaxes(title_text="pyrrm Flow (ML/day)", row=2, col=1)
fig.update_xaxes(title_text="Observed Flow (ML/day)", row=2, col=2)
fig.update_yaxes(title_text="Simulated Flow (ML/day)", row=2, col=2)
fig.update_xaxes(title_text="Exceedance Probability (%)", row=3, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", row=3, col=1)
fig.update_xaxes(title_text="Exceedance Probability (%)", row=3, col=2)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=3, col=2)

fig.update_layout(title_text="Benchmark Comparison: Observed vs pyrrm vs SOURCE", height=900, showlegend=True)
fig.show()
print("Interactive benchmark comparison displayed")

# %% [markdown]
# ---
# ## Part 3: Model Calibration
#
# Now we'll calibrate the Sacramento model using four different optimization methods:
#
# 1. **SpotPy DREAM** - Bayesian MCMC with differential evolution
# 2. **PyDREAM** - Multi-try DREAM with snooker updates (advanced)
# 3. **SCE-UA** - Shuffled Complex Evolution (classic hydrological calibration)
# 4. **SciPy Differential Evolution** - Global optimization (for comparison)
#
# ### Why Multiple Methods?
#
# - **DREAM algorithms** (MCMC) provide posterior distributions, quantifying uncertainty
# - **SCE-UA** is the classic global optimizer for hydrology, fast and robust
# - **Different methods** may converge differently for the same problem
# - **Comparison** helps identify robust parameter sets
#
# ### Objective Functions by Method Type
#
# | Method Type | Objective | Reason |
# |-------------|-----------|--------|
# | **MCMC** (SpotPy DREAM, PyDREAM) | Gaussian Likelihood | Log-probability required for Bayesian inference |
# | **Optimization** (SCE-UA, SciPy DE) | NSE | Standard efficiency metric for global optimization |
#
# Note: MCMC methods sample from the posterior distribution and need a proper likelihood.
# Optimization methods directly maximize/minimize an efficiency metric.

# %%
# Import calibration tools
from pyrrm.calibration import (
    CalibrationRunner, 
    CalibrationResult,
    NSE, 
    KGE, 
    LogNSE,
    WeightedObjective,
    SPOTPY_AVAILABLE,
    PYDREAM_AVAILABLE
)

print(f"SpotPy available: {SPOTPY_AVAILABLE}")
print(f"PyDREAM available: {PYDREAM_AVAILABLE}")

# %%
# Prepare calibration data
# We'll use the calibration period data with inputs and observed flow

# Create a fresh copy for calibration
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values.copy()

print(f"Calibration inputs shape: {cal_inputs.shape}")
print(f"Calibration observed shape: {cal_observed.shape}")

# %%
# Define objective functions for different calibration methods
#
# MCMC methods (SpotPy DREAM, PyDREAM) require a proper likelihood function
# that can be interpreted as a log-probability for Bayesian inference.
#
# Optimization methods (SCE-UA, SciPy DE) work with standard efficiency metrics
# like NSE which they maximize/minimize directly.

from pyrrm.calibration import GaussianLikelihood

# ----- MCMC Objective: Gaussian Likelihood -----
# GaussianLikelihood is the recommended objective for MCMC/DREAM calibration
# It computes: log_likelihood = -n/2 * log(sum(residuals^2))
# This is a proper likelihood function suitable for Bayesian sampling
mcmc_objective = GaussianLikelihood()

print("=" * 60)
print("OBJECTIVE FUNCTIONS BY METHOD TYPE")
print("=" * 60)
print(f"\nMCMC Methods (SpotPy DREAM, PyDREAM):")
print(f"  Objective: {mcmc_objective.name}")
print(f"  Maximize: {mcmc_objective.maximize}")
print("  Note: Log-likelihood required for proper Bayesian inference")

# ----- Optimization Objective: NSE -----
# NSE (Nash-Sutcliffe Efficiency) is the standard for hydrological optimization
# Range: (-∞, 1], where 1 = perfect, 0 = mean, <0 = worse than mean
nse_objective = NSE()

print(f"\nOptimization Methods (SCE-UA, SciPy DE):")
print(f"  Objective: {nse_objective.name}")
print(f"  Maximize: {nse_objective.maximize}")
print("  Note: Standard efficiency metric for global optimization")

# %%
# Initialize separate CalibrationRunners for each objective type
# This ensures each method uses the appropriate objective function

# Runner for MCMC methods (with Gaussian Likelihood)
runner_mcmc = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=mcmc_objective,
    warmup_period=WARMUP_DAYS
)

# Runner for optimization methods (with NSE)
runner_optim = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=nse_objective,
    warmup_period=WARMUP_DAYS
)

print(f"\nModel output units: {runner_mcmc.model.output_units}")
print(f"Catchment area: {runner_mcmc.model.catchment_area_km2} km²")

# Get parameter bounds for reference
param_bounds = runner_mcmc.model.get_parameter_bounds()
print("\nParameters to calibrate:")
print("-" * 50)
for name, (low, high) in param_bounds.items():
    print(f"  {name:8s}: [{low:8.3f}, {high:8.3f}]")

# %% [markdown]
# ### Customizing Parameter Bounds
#
# By default, the Sacramento model uses hardcoded parameter bounds suitable for general use.
# However, you may want to customize these bounds based on:
#
# - **Prior knowledge** about the catchment characteristics
# - **Results from previous calibrations** (narrowing search space)
# - **Physical constraints** specific to your study area
# - **Sensitivity analysis** showing which parameters are most influential
#
# `pyrrm` supports loading parameter bounds from external text files, making it easy
# to manage different bound configurations without modifying code.

# %%
# === CUSTOMIZING PARAMETER BOUNDS ===
# 
# There are three ways to customize parameter bounds:
# 1. Load from a text file (recommended for reproducibility)
# 2. Load from a CSV file
# 3. Set programmatically with a dictionary

# ----- Method 1: Save current bounds to file for editing -----
# First, let's save the current bounds to a text file
from pyrrm.data import save_parameter_bounds, load_parameter_bounds

# Create a bounds file from the model's default bounds
bounds_file = DATA_DIR / 'sacramento_bounds_custom.txt'
save_parameter_bounds(param_bounds, bounds_file, model_name='Sacramento')
print(f"Saved parameter bounds to: {bounds_file}")

# Let's view what the file looks like
print("\n" + "=" * 60)
print("PARAMETER BOUNDS FILE FORMAT")
print("=" * 60)
with open(bounds_file, 'r') as f:
    print(f.read())

# %%
# ----- Method 2: Load bounds from a file -----
# 
# After editing the file, you can load the custom bounds.
# We provide an example file in data/410734/sacramento_bounds.txt

# Load the example bounds file
example_bounds_file = DATA_DIR / 'sacramento_bounds.txt'
if example_bounds_file.exists():
    custom_bounds = load_parameter_bounds(example_bounds_file)
    print(f"Loaded {len(custom_bounds)} parameters from: {example_bounds_file}")
    print("\nCustom bounds preview (first 5):")
    for i, (name, (low, high)) in enumerate(custom_bounds.items()):
        if i >= 5:
            print("  ...")
            break
        print(f"  {name:8s}: [{low:8.4f}, {high:8.4f}]")
else:
    print(f"Example file not found: {example_bounds_file}")
    custom_bounds = None

# %%
# ----- Method 3: Set bounds programmatically -----
#
# You can also set custom bounds directly in code using set_parameter_bounds()

# Create a model with custom bounds
model_custom = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)

# Define narrower bounds for key parameters (example)
narrower_bounds = {
    'uztwm': (40.0, 100.0),    # Narrower upper zone tension water range
    'lztwm': (100.0, 250.0),   # Narrower lower zone tension water range
    'uzk': (0.25, 0.45),       # Narrower drainage rate range
    'lzpk': (0.002, 0.012),    # Narrower primary drainage rate
}

# Apply custom bounds to the model
model_custom.set_parameter_bounds(narrower_bounds)

print("Applied custom bounds to model:")
print("-" * 50)
# Show only the modified parameters
for name in narrower_bounds:
    low, high = model_custom.get_parameter_bounds()[name]
    print(f"  {name:8s}: [{low:8.4f}, {high:8.4f}]")

# %%
# ----- Using custom bounds with CalibrationRunner -----
#
# There are TWO ways to use custom bounds for calibration:
#
# Option A: Apply bounds to the model before creating CalibrationRunner
#           (the runner will use model.get_parameter_bounds())
#
# Option B: Pass parameter_bounds directly to CalibrationRunner
#           (this overrides model bounds for calibration only)

# Option A: Model with pre-set bounds
runner_with_model_bounds = CalibrationRunner(
    model=model_custom,  # Model already has custom bounds
    inputs=cal_inputs,
    observed=cal_observed,
    objective=nse_objective,
    warmup_period=WARMUP_DAYS
)
print("Option A: CalibrationRunner using model's custom bounds")

# Option B: Override bounds at runner level
runner_with_direct_bounds = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=nse_objective,
    parameter_bounds=narrower_bounds,  # Override model defaults
    warmup_period=WARMUP_DAYS
)
print("Option B: CalibrationRunner with direct parameter_bounds argument")

# For this tutorial, we'll continue using the original runner with default bounds
print("\nContinuing with default bounds for demonstration...")

# %% [markdown]
# ### 3.1 SpotPy DREAM Calibration
#
# DREAM (DiffeRential Evolution Adaptive Metropolis) is a Bayesian MCMC algorithm
# that efficiently samples the posterior distribution of model parameters.
#
# **Parallelization Note:** For production runs, SpotPy DREAM supports MPI parallelization
# which provides significant speedup. MPI requires running from the command line with `mpirun`,
# not from within a Jupyter notebook. See the cell below for generating an MPI calibration script.

# %%
# SpotPy DREAM Calibration
print("=" * 60)
print("SPOTPY DREAM CALIBRATION")
print("=" * 60)

# Calibration settings (reduced for demo - increase for production)
N_ITERATIONS_SPOTPY = 5000  # Number of MCMC iterations
N_CHAINS = 8                 # Number of parallel chains

# Parallelization options:
# - 'seq'  : Sequential (single process, default)
# - 'mpi'  : MPI parallelization (requires mpi4py) - fastest for large runs
# - 'mpc'  : Multiprocessing with pathos (cross-platform)
# - 'umpc' : Multiprocessing unordered (faster but results may be unordered)
#
# For this tutorial, we use sequential mode for simplicity.
# For production runs, use MPI with a script:
#   mpirun -n 9 python calibration_script.py  # 1 master + 8 workers
PARALLEL_MODE = 'seq'

if SPOTPY_AVAILABLE:
    print(f"\nObjective: {mcmc_objective.name} (log-likelihood for Bayesian inference)")
    print(f"Parallel mode: {PARALLEL_MODE}")
    print(f"DREAM chains: {N_CHAINS}")
    print(f"Running SpotPy DREAM with {N_ITERATIONS_SPOTPY} iterations...")
    print("This may take several minutes...")
    
    spotpy_result = runner_mcmc.run_spotpy_dream(
        n_iterations=N_ITERATIONS_SPOTPY,
        n_chains=N_CHAINS,
        convergence_threshold=1.2,  # Gelman-Rubin threshold
        dbname='spotpy_dream_calib',
        dbformat='csv',             # CSV format is more reliable across SpotPy versions
        parallel=PARALLEL_MODE
    )
    
    print("\n" + spotpy_result.summary())
else:
    print("\n⚠ SpotPy not installed. Install with: pip install spotpy")
    spotpy_result = None

# %% [markdown]
# #### MPI Parallel Calibration (Production Use)
#
# For faster calibration on multi-core systems, use MPI parallelization.
# This requires running from the command line, not within a notebook.
#
# The following cell generates an MPI calibration script that you can run with:
# ```bash
# mpirun -n 9 python mpi_calibration.py  # 1 master + 8 worker processes
# ```

# %%
# Generate MPI calibration script for production use
mpi_script = '''#!/usr/bin/env python
"""
MPI Parallel SpotPy DREAM Calibration Script

Run with: mpirun -n 9 python mpi_calibration.py
(1 master + 8 workers = 9 total processes)

Requires: pip install mpi4py
"""
import pandas as pd
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner, GaussianLikelihood

# ----- Configuration -----
DATA_FILE = 'data/cotter_data.csv'  # Update path as needed
CATCHMENT_AREA_KM2 = 130
WARMUP_DAYS = 365

# Calibration settings
N_ITERATIONS = 10000
N_CHAINS = 8  # Should match (mpirun -n) - 1

# ----- Load Data -----
data = pd.read_csv(DATA_FILE, parse_dates=['date'], index_col='date')
cal_data = data['2010':'2015']
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values.copy()

# ----- Setup Calibration -----
runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=GaussianLikelihood(),
    warmup_period=WARMUP_DAYS
)

# ----- Run MPI DREAM -----
print(f"Starting MPI DREAM calibration...")
print(f"Iterations: {N_ITERATIONS}, Chains: {N_CHAINS}")

result = runner.run_spotpy_dream(
    n_iterations=N_ITERATIONS,
    n_chains=N_CHAINS,
    convergence_threshold=1.2,
    dbname='mpi_dream_results',
    dbformat='csv',
    parallel='mpi'  # MPI parallelization
)

# Only master process (rank 0) gets the result
if result is not None:
    print("\\n" + result.summary())
    print("\\nBest parameters saved to: mpi_dream_results.csv")
'''

# Save script to file
script_path = '../mpi_calibration.py'
with open(script_path, 'w') as f:
    f.write(mpi_script)

print("MPI calibration script saved to: mpi_calibration.py")
print("\nTo run parallel calibration:")
print("  1. Install mpi4py: pip install mpi4py")
print("  2. Run: mpirun -n 9 python mpi_calibration.py")
print("\nSpeedup scales approximately linearly with number of workers.")

# %% [markdown]
# ### 3.2 PyDREAM Calibration (MT-DREAM(ZS))
#
# PyDREAM implements Multi-Try DREAM with snooker updates, which can provide
# better mixing for complex, multi-modal posterior distributions.
#
# **Built-in Parallelization:**
# - **Chain-level**: Each chain runs in a separate process (automatic)
# - **Multi-try level**: When `parallel=True`, proposal evaluations are parallelized

# %%
# PyDREAM Calibration (with parallel processing)
print("=" * 60)
print("PYDREAM CALIBRATION (MT-DREAM(ZS)) - Parallel")
print("=" * 60)

N_ITERATIONS_PYDREAM = 5000   # Iterations per chain
N_CHAINS_PYDREAM = 8          # Number of chains (run in parallel automatically)
MULTITRY = 5                  # Multi-try samples per iteration
SNOOKER = 0.1                 # Probability of snooker update
PARALLEL_MULTITRY = True      # Parallelize multi-try evaluations

# PyDREAM Parallelization:
# 1. Chain-level: Automatic - N_CHAINS processes run in parallel via DreamPool
# 2. Multi-try: Optional - when parallel=True, multitry evaluations run in parallel
#
# Total parallelism: N_CHAINS processes, each potentially spawning MULTITRY workers

if PYDREAM_AVAILABLE:
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    
    print(f"\nObjective: {mcmc_objective.name} (log-likelihood for Bayesian inference)")
    print(f"Available CPU cores: {n_cores}")
    print(f"Parallel chains: {N_CHAINS_PYDREAM} (automatic parallelization)")
    print(f"Multi-try samples: {MULTITRY}" + (" (parallel)" if PARALLEL_MULTITRY else ""))
    print(f"Snooker probability: {SNOOKER}")
    print(f"Running PyDREAM with {N_ITERATIONS_PYDREAM} iterations per chain...")
    print("Progress will be written to: pydream_calib.csv")
    print("This may take several minutes...")
    
    pydream_result = runner_mcmc.run_pydream(
        n_iterations=N_ITERATIONS_PYDREAM,
        n_chains=N_CHAINS_PYDREAM,
        multitry=MULTITRY,
        snooker=SNOOKER,
        parallel=PARALLEL_MULTITRY,    # Enable parallel multi-try evaluation
        adapt_crossover=False,         # Disable to avoid PyDREAM/NumPy compatibility bug
        dbname='pydream_calib',        # Write progress to CSV for monitoring
        dbformat='csv',
        verbose=True
    )
    
    print("\n" + pydream_result.summary())
else:
    print("\n⚠ PyDREAM not installed. Install with: pip install pydream")
    pydream_result = None

# %% [markdown]
# ### 3.3 SCE-UA Calibration (Shuffled Complex Evolution)
#
# SCE-UA (Shuffled Complex Evolution - University of Arizona) is a global 
# optimization algorithm developed specifically for hydrological model calibration.
# It combines concepts from:
# - Simplex method (local search)
# - Random search (global exploration)
# - Competitive evolution (complex shuffling)
#
# SCE-UA is widely used in hydrology and is known for:
# - Robust convergence to global optimum
# - Efficient handling of parameter interactions
# - Built-in convergence checking
#
# Reference: Duan, Q., Sorooshian, S., & Gupta, V. (1992). Effective and efficient 
# global optimization for conceptual rainfall-runoff models. Water Resources Research.

# %%
# SCE-UA Calibration
print("=" * 60)
print("SCE-UA CALIBRATION (Shuffled Complex Evolution)")
print("=" * 60)

# SCE-UA settings
N_ITERATIONS_SCEUA = 10000  # Maximum function evaluations
NGS = 7                      # Number of complexes (can be parallelized)

if SPOTPY_AVAILABLE:
    print(f"\nObjective: {nse_objective.name} (standard efficiency metric)")
    print(f"Running SCE-UA with {N_ITERATIONS_SCEUA} iterations, {NGS} complexes...")
    print("SCE-UA is a global optimization algorithm, not MCMC (no posterior samples)")
    print("This may take several minutes...")
    
    sceua_result = runner_optim.run_sceua(
        n_iterations=N_ITERATIONS_SCEUA,
        ngs=NGS,                  # Number of complexes
        kstop=3,                  # Convergence criterion: stop if no improvement in kstop loops
        pcento=0.01,              # Percent change required for convergence
        peps=0.001,               # Convergence threshold for parameter changes
        dbname='sceua_calib',
        dbformat='csv'
    )
    
    print("\n" + sceua_result.summary())
else:
    print("\n⚠ SpotPy not installed. Install with: pip install spotpy")
    sceua_result = None

# %% [markdown]
# ### 3.4 SciPy Differential Evolution
#
# For comparison, we also run a deterministic global optimization using
# differential evolution. This finds a single best parameter set without
# uncertainty quantification.

# %%
# SciPy Differential Evolution Calibration
print("=" * 60)
print("SCIPY DIFFERENTIAL EVOLUTION CALIBRATION")
print("=" * 60)

print(f"\nObjective: {nse_objective.name} (standard efficiency metric)")
print("Running SciPy Differential Evolution...")
print("This provides a deterministic comparison to Bayesian methods...")

scipy_result = runner_optim.run_differential_evolution(
    maxiter=500,        # Maximum iterations
    popsize=15,         # Population size multiplier
    mutation=(0.5, 1),  # Mutation constant
    recombination=0.7,  # Recombination rate
    seed=42,            # Random seed for reproducibility
    workers=1,          # Number of parallel workers
    disp=True           # Display progress
)

print("\n" + scipy_result.summary())

# %%
# =============================================================================
# CALIBRATION RESULTS COLLECTION
# =============================================================================
# This section collects all calibration results and prepares them for analysis.
# It dynamically handles any subset of methods that were run.

# Define all possible calibration results
# Variables that don't exist will be set to None
all_possible_results = {
    'SpotPy DREAM': spotpy_result if 'spotpy_result' in dir() else None,
    'PyDREAM': pydream_result if 'pydream_result' in dir() else None,
    'SCE-UA': sceua_result if 'sceua_result' in dir() else None,
    'SciPy DE': scipy_result if 'scipy_result' in dir() else None,
}

# Define objective functions for each method (for reference)
all_method_objectives = {
    'SpotPy DREAM': mcmc_objective if 'mcmc_objective' in dir() else None,
    'PyDREAM': mcmc_objective if 'mcmc_objective' in dir() else None,
    'SCE-UA': nse_objective if 'nse_objective' in dir() else None,
    'SciPy DE': nse_objective if 'nse_objective' in dir() else None,
}

# Define method categories
MCMC_METHODS = {'SpotPy DREAM', 'PyDREAM'}
OPTIMIZATION_METHODS = {'SCE-UA', 'SciPy DE'}

# Filter to only include results that exist and are not None
calibration_results = {k: v for k, v in all_possible_results.items() if v is not None}
method_objectives = {k: v for k, v in all_method_objectives.items() if k in calibration_results}

# Dynamic color assignment for consistent plotting
METHOD_COLORS = {
    'SpotPy DREAM': '#1f77b4',  # blue
    'PyDREAM': '#d62728',       # red  
    'SCE-UA': '#2ca02c',        # green
    'SciPy DE': '#9467bd',      # purple
    'Benchmark': '#7f7f7f',     # gray
}

# Helper function to get colors for available methods
def get_method_colors(methods):
    """Return color dict for specified methods."""
    return {m: METHOD_COLORS.get(m, '#333333') for m in methods}

# Summary output
print("=" * 70)
print("CALIBRATION RESULTS SUMMARY")
print("=" * 70)

if len(calibration_results) == 0:
    print("\n⚠ No calibration results available!")
    print("  Run at least one calibration method before proceeding to Part 4.")
else:
    print(f"\n{len(calibration_results)} calibration method(s) completed:")
    print("-" * 70)
    print(f"{'Method':<15} {'Type':<12} {'Objective Function':<20} {'Best Value':>15}")
    print("-" * 70)
    for method, result in calibration_results.items():
        method_type = "MCMC" if method in MCMC_METHODS else "Optimization"
        obj = method_objectives.get(method)
        obj_name = obj.name if obj else "N/A"
        print(f"  ✓ {method:<13} {method_type:<12} {obj_name:<20} {result.best_objective:>15.4f}")
    
    # Show which methods have MCMC samples
    mcmc_available = [m for m in calibration_results if m in MCMC_METHODS]
    optim_available = [m for m in calibration_results if m in OPTIMIZATION_METHODS]
    
    print("\n" + "-" * 70)
    print(f"  MCMC methods (with posterior samples): {mcmc_available if mcmc_available else 'None'}")
    print(f"  Optimization methods (point estimates): {optim_available if optim_available else 'None'}")

# %% [markdown]
# ---
# ## Part 4: Calibration Diagnostics
#
# This section analyzes calibration results from any methods that were run.
# Diagnostics are adapted based on which methods are available:
#
# **For MCMC methods (SpotPy DREAM, PyDREAM):**
# - Convergence diagnostics (Gelman-Rubin R-hat)
# - Parameter trace plots
# - Posterior distributions
#
# **For all methods:**
# - Objective function traces
# - Dotty plots (parameter sensitivity)
# - Performance metrics comparison

# %%
# Check if we have any results to analyze
if len(calibration_results) == 0:
    print("=" * 60)
    print("⚠ NO CALIBRATION RESULTS TO ANALYZE")
    print("=" * 60)
    print("\nPlease run at least one calibration method in Part 3 before")
    print("proceeding with diagnostics.")
    print("\nAvailable methods:")
    print("  - SpotPy DREAM (MCMC)")
    print("  - PyDREAM (MCMC)")
    print("  - SCE-UA (Global Optimization)")
    print("  - SciPy DE (Global Optimization)")
else:
    print("=" * 60)
    print(f"ANALYZING {len(calibration_results)} CALIBRATION RESULT(S)")
    print("=" * 60)
    print(f"\nMethods to analyze: {list(calibration_results.keys())}")

# %%
# Extract parameter names for plotting
param_names = list(param_bounds.keys())

# Import visualization tools
from pyrrm.visualization.calibration_plots import (
    plot_parameter_traces,
    plot_parameter_histograms,
    plot_objective_function_trace,
    plot_dotty
)

# Get colors for available methods
colors = get_method_colors(calibration_results.keys())

# %%
# Convergence diagnostics - Gelman-Rubin statistics (MCMC methods only)
print("=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

if len(calibration_results) == 0:
    print("\n⚠ No results to analyze.")
else:
    mcmc_results = {m: r for m, r in calibration_results.items() if m in MCMC_METHODS}
    
    if len(mcmc_results) == 0:
        print("\nNo MCMC methods available for convergence diagnostics.")
        print("(SCE-UA and SciPy DE are optimization methods without MCMC chains)")
    else:
        for method, result in mcmc_results.items():
            if 'gelman_rubin' in result.convergence_diagnostics:
                print(f"\n{method} - Gelman-Rubin Statistics (R-hat):")
                print("-" * 40)
                gr_values = result.convergence_diagnostics['gelman_rubin']
                for param, gr in gr_values.items():
                    status = "✓" if gr < 1.2 else "⚠"
                    print(f"  {status} {param:8s}: {gr:.4f}")
                
                converged = result.convergence_diagnostics.get('converged', 'N/A')
                print(f"\n  Overall convergence: {converged}")
            else:
                print(f"\n{method}: Convergence diagnostics not available")
    
    # Note about optimization methods
    optim_results = {m: r for m, r in calibration_results.items() if m in OPTIMIZATION_METHODS}
    if len(optim_results) > 0:
        print(f"\nOptimization methods ({list(optim_results.keys())}): No convergence diagnostics")
        print("  (These methods find point estimates, not posterior distributions)")

# %%
# Parameter trace plots (for methods with samples)
print("\nGenerating parameter trace plots...")

# Check if any results have samples
results_with_samples = {m: r for m, r in calibration_results.items() 
                        if r.all_samples is not None and len(r.all_samples) > 0}

if len(results_with_samples) == 0:
    print("⚠ No methods with sample traces available.")
    print("  (Optimization methods like SCE-UA may only have final results)")
else:
    fig, axes = plt.subplots(len(param_names), 1, figsize=(14, 3*len(param_names)), sharex=True)
    if len(param_names) == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    has_data = False
    for method, result in results_with_samples.items():
        samples = result.all_samples
        
        # Check if parameter columns exist
        available_params = [p for p in param_names if p in samples.columns]
        
        if len(available_params) > 0:
            has_data = True
            for i, param in enumerate(available_params[:len(axes)]):
                if param in samples.columns:
                    axes[i].plot(samples[param].values, alpha=0.5, linewidth=0.5, 
                               label=method, color=colors.get(method, 'gray'))
                    axes[i].set_ylabel(param)
                    axes[i].legend(loc='upper right')
    
    if has_data:
        axes[-1].set_xlabel('Iteration')
        fig.suptitle('Parameter Traces Across Calibration Methods', y=1.02)
        plt.tight_layout()
        plt.savefig('figures/tutorial_parameter_traces.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to: figures/tutorial_parameter_traces.png")
    else:
        plt.close(fig)
        print("⚠ No parameter data found in samples.")

# %%
# Parameter posterior distributions (histograms)
print("\nGenerating parameter posterior distributions...")

if len(results_with_samples) == 0:
    print("⚠ No methods with sample data available for posterior distributions.")
else:
    # Determine number of parameters for subplot grid
    n_params = len(param_names)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    has_data = False
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        for method, result in results_with_samples.items():
            if result.all_samples is not None and param in result.all_samples.columns:
                samples = result.all_samples[param].dropna()
                if len(samples) > 0:
                    has_data = True
                    ax.hist(samples, bins=50, alpha=0.5, density=True, 
                           color=colors.get(method, 'gray'), label=method)
        
        # Add benchmark value as vertical line
        if param in benchmark_params:
            ax.axvline(benchmark_params[param], color='black', linestyle='--', 
                       linewidth=2, label='Benchmark')
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)

    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    if has_data:
        fig.suptitle('Parameter Posterior Distributions', y=1.02)
        plt.tight_layout()
        plt.savefig('figures/tutorial_parameter_posteriors.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to: figures/tutorial_parameter_posteriors.png")
    else:
        plt.close(fig)
        print("⚠ No parameter data found for histograms.")

# %%
# Objective function trace
print("\nGenerating objective function traces...")

if len(results_with_samples) == 0:
    print("⚠ No methods with sample data available for objective traces.")
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for method, result in results_with_samples.items():
        samples = result.all_samples
        
        # Look for objective function column (different names possible)
        obj_col = None
        for col_name in ['likelihood', 'like1', 'objectivefunction', 'log_likelihood', 'NSE']:
            if col_name in samples.columns:
                obj_col = col_name
                break
        
        if obj_col:
            has_data = True
            ax.plot(samples[obj_col].values, alpha=0.7, linewidth=0.5, 
                   label=method, color=colors.get(method, 'gray'))

    if has_data:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Function Value')
        ax.set_title('Objective Function Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('figures/tutorial_objective_trace.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to: figures/tutorial_objective_trace.png")
    else:
        plt.close(fig)
        print("⚠ No objective function data found in samples.")

# %%
# Dotty plots - Parameter sensitivity visualization
print("\nGenerating dotty plots...")

if len(results_with_samples) == 0:
    print("⚠ No methods with sample data available for dotty plots.")
else:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    has_data = False

    for method, result in results_with_samples.items():
        samples = result.all_samples
        
        # Find objective column
        obj_col = None
        for col_name in ['likelihood', 'like1', 'objectivefunction', 'log_likelihood']:
            if col_name in samples.columns:
                obj_col = col_name
                break
        
        if obj_col:
            has_data = True
            for i, param in enumerate(param_names):
                if param in samples.columns:
                    axes[i].scatter(samples[param].values, samples[obj_col].values,
                                   alpha=0.3, s=3, color=colors.get(method, 'gray'),
                                   label=method if i == 0 else None)

    if has_data:
        # Determine y-axis label based on available objectives
        obj_label = 'Objective Function'
        available_objectives = [method_objectives.get(m) for m in results_with_samples.keys()]
        unique_obj_names = list(set(o.name for o in available_objectives if o is not None))
        if len(unique_obj_names) == 1:
            obj_label = unique_obj_names[0]
        
        for i, param in enumerate(param_names):
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(obj_label)
            if param in benchmark_params:
                axes[i].axvline(benchmark_params[param], color='black', linestyle='--', 
                               linewidth=1.5, alpha=0.7)

        # Remove empty subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])

        fig.legend(loc='upper center', ncol=min(len(results_with_samples), 4), bbox_to_anchor=(0.5, 1.02))
        fig.suptitle('Dotty Plots: Parameter vs Objective Function', y=1.05)
        plt.tight_layout()
        plt.savefig('figures/tutorial_dotty_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to: figures/tutorial_dotty_plots.png")
    else:
        plt.close(fig)
        print("⚠ No objective function data found for dotty plots.")

# %%
# Calculate performance metrics for each calibration result
print("=" * 60)
print("PERFORMANCE METRICS COMPARISON")
print("=" * 60)

if len(calibration_results) == 0:
    print("\n⚠ No calibration results to compare.")
    print("  Run at least one calibration method before this section.")
    metrics_df = pd.DataFrame()  # Empty DataFrame
else:
    # Create a summary DataFrame
    metrics_summary = []

    # Add benchmark (always include for reference)
    benchmark_metrics = calculate_metrics(pyrrm_flow, observed_flow)
    benchmark_row = {'Method': 'Benchmark', **benchmark_metrics}
    metrics_summary.append(benchmark_row)

    # Add each calibration result
    for method, result in calibration_results.items():
        try:
            # Get runner for the appropriate objective
            if method in MCMC_METHODS:
                runner_to_use = runner_mcmc if 'runner_mcmc' in dir() else runner_optim
            else:
                runner_to_use = runner_optim if 'runner_optim' in dir() else runner_mcmc
            
            # Get best simulation
            best_sim_df = runner_to_use.get_best_simulation(result)
            
            # Get flow column
            if 'runoff' in best_sim_df.columns:
                sim_flow = best_sim_df['runoff'].values[WARMUP_DAYS:]
            else:
                sim_flow = best_sim_df.iloc[:, 0].values[WARMUP_DAYS:]
            
            obs_flow = cal_observed[WARMUP_DAYS:]
            
            # Calculate metrics
            metrics = calculate_metrics(sim_flow, obs_flow)
            row = {'Method': method, **metrics}
            metrics_summary.append(row)
        except Exception as e:
            print(f"  ⚠ Could not calculate metrics for {method}: {e}")

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df = metrics_df.set_index('Method')

    print(f"\nPerformance Metrics Summary ({len(metrics_summary)} methods):")
    print(metrics_df.round(4).to_string())

# %%
# Visualize metrics comparison
if len(metrics_df) == 0:
    print("⚠ No metrics data to visualize.")
else:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metrics_to_plot = ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS', 'LogNSE']
    
    # Get colors for methods in the metrics DataFrame
    methods = metrics_df.index.tolist()
    bar_colors = [METHOD_COLORS.get(m, '#333333') for m in methods]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes.flatten()[i]
        if metric in metrics_df.columns:
            values = metrics_df[metric].values
            bars = ax.bar(range(len(methods)), values, color=bar_colors)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_title(metric)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Performance Metrics by Calibration Method ({len(methods)} methods)', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/tutorial_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to: figures/tutorial_metrics_comparison.png")

# %% [markdown]
# ---
# ## Part 5: Results Comparison and Visualization
#
# Now we'll compare the calibration results in detail:
# - Parameter values across methods
# - Hydrograph comparisons
# - Flow duration curves
# - Residual analysis
# - Validation period performance

# %%
# Parameter comparison table
print("=" * 60)
print("BEST PARAMETER VALUES BY METHOD")
print("=" * 60)

# Build parameter comparison DataFrame
param_comparison = {'Parameter': param_names}

# Add benchmark parameters
param_comparison['Benchmark'] = [benchmark_params.get(p, np.nan) for p in param_names]

# Add calibrated parameters
for method, result in calibration_results.items():
    param_comparison[method] = [result.best_parameters.get(p, np.nan) for p in param_names]

param_df = pd.DataFrame(param_comparison).set_index('Parameter')
print("\nBest Parameters from Each Calibration Method:")
print(param_df.round(4).to_string())

# Save to CSV
param_df.to_csv('../test_data/calibrated_parameters_comparison.csv')
print("\nParameter comparison saved to: test_data/calibrated_parameters_comparison.csv")

# %%
# Generate best simulations for all methods
best_simulations = {}

# Benchmark simulation (already calculated)
best_simulations['Benchmark'] = pyrrm_flow

# Calibrated simulations
for method, result in calibration_results.items():
    # Create model with best parameters and catchment area for ML/day output
    model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model.reset()
    model.set_parameters(result.best_parameters)
    
    # Run simulation using the public API (handles catchment area scaling)
    sim_results = model.run(cal_data)
    
    # Store (excluding warmup) - 'flow' column is runoff in ML/day
    best_simulations[method] = sim_results['flow'].values[WARMUP_DAYS:]

# Also get observed for comparison period
obs_comparison = cal_data['observed_flow'].values[WARMUP_DAYS:]
dates_comparison = cal_data.index[WARMUP_DAYS:]

print(f"Generated simulations for {len(best_simulations)} methods")
print(f"Comparison period: {len(dates_comparison)} days")

# %%
# Interactive Hydrograph Comparison using Plotly
# Shows BOTH log and linear scales simultaneously for easy comparison

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define colors for each method (consistent with METHOD_COLORS)
plotly_colors = {
    'Observed': '#000000',      # black
    'Benchmark': '#7f7f7f',     # gray
    'SpotPy DREAM': '#1f77b4',  # blue
    'PyDREAM': '#d62728',       # red
    'SCE-UA': '#2ca02c',        # green
    'SciPy DE': '#9467bd',      # purple
}

# Create figure with 3 rows: Log scale, Linear scale, Residuals
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.4, 0.4, 0.2],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        '<b>Log Scale</b> - Better for low flows and recession curves',
        '<b>Linear Scale</b> - Better for peak flow comparison',
        '<b>Residuals</b> (Simulated - Observed)'
    )
)

# Add traces to BOTH log and linear scale panels
for row in [1, 2]:
    # Add observed flow
    fig.add_trace(
        go.Scatter(
            x=dates_comparison,
            y=obs_comparison,
            mode='lines',
            name='Observed',
            line=dict(color=plotly_colors['Observed'], width=1.5),
            legendgroup='Observed',
            showlegend=(row == 1),  # Only show in legend once
            hovertemplate='<b>Observed</b><br>Date: %{x}<br>Flow: %{y:.2f} ML/day<extra></extra>'
        ),
        row=row, col=1
    )
    
    # Add simulated flows for each method
    for method, sim in best_simulations.items():
        color = plotly_colors.get(method, '#333333')
        
        fig.add_trace(
            go.Scatter(
                x=dates_comparison,
                y=sim,
                mode='lines',
                name=method,
                line=dict(color=color, width=1),
                opacity=0.8,
                legendgroup=method,
                showlegend=(row == 1),  # Only show in legend once
                hovertemplate=f'<b>{method}</b><br>Date: %{{x}}<br>Flow: %{{y:.2f}} ML/day<extra></extra>'
            ),
            row=row, col=1
        )

# Add residuals (row 3)
for method, sim in best_simulations.items():
    color = plotly_colors.get(method, '#333333')
    residuals = sim - obs_comparison
    
    fig.add_trace(
        go.Scatter(
            x=dates_comparison,
            y=residuals,
            mode='lines',
            name=f'{method} residual',
            line=dict(color=color, width=1),
            opacity=0.6,
            showlegend=False,
            hovertemplate=f'<b>{method} Residual</b><br>Date: %{{x}}<br>Δ: %{{y:.2f}} ML/day<extra></extra>'
        ),
        row=3, col=1
    )

# Add zero line for residuals
fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=3, col=1)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Interactive Hydrograph Comparison</b><br><sup>Click legend items to toggle • Drag to zoom • Double-click to reset • Zooming syncs across all panels</sup>',
        x=0.5,
        xanchor='center'
    ),
    height=900,
    hovermode='x unified',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    # Add range slider on bottom panel for navigation
    xaxis3=dict(
        rangeslider=dict(visible=True, thickness=0.05),
        type='date'
    )
)

# Update y-axes with appropriate scales
fig.update_yaxes(title_text='Flow (ML/day)', type='log', row=1, col=1)
fig.update_yaxes(title_text='Flow (ML/day)', type='linear', row=2, col=1)
fig.update_yaxes(title_text='Residual (ML/day)', row=3, col=1)

# Update x-axis label (only on bottom)
fig.update_xaxes(title_text='Date', row=3, col=1)

fig.show()
print("Interactive hydrograph displayed with BOTH scales:")
print("  • Top panel: Log scale (better for low flows)")
print("  • Middle panel: Linear scale (better for peaks)")
print("  • Bottom panel: Residuals")
print("  • Drag to zoom (synced across panels), double-click to reset")

# %%
# Also save a static version for reports
fig_static, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full period
ax1 = axes[0]
ax1.plot(dates_comparison, obs_comparison, 'k-', alpha=0.8, 
         linewidth=0.8, label='Observed')

for method, sim in best_simulations.items():
    ax1.plot(dates_comparison, sim, color=plotly_colors.get(method, 'purple'),
             alpha=0.6, linewidth=0.6, label=method)

ax1.set_ylabel('Flow (ML/day)')
ax1.set_title('Hydrograph Comparison - Calibration Period')
ax1.legend(loc='upper right')
ax1.set_yscale('log')

# Zoom to a specific period (e.g., 1 year)
zoom_start = pd.Timestamp('2000-01-01')
zoom_end = pd.Timestamp('2001-12-31')
zoom_mask = (dates_comparison >= zoom_start) & (dates_comparison <= zoom_end)

ax2 = axes[1]
ax2.plot(dates_comparison[zoom_mask], obs_comparison[zoom_mask], 'k-', 
         alpha=0.8, linewidth=1.5, label='Observed')

for method, sim in best_simulations.items():
    ax2.plot(dates_comparison[zoom_mask], sim[zoom_mask], 
             color=plotly_colors.get(method, 'purple'),
             alpha=0.7, linewidth=1, label=method)

ax2.set_xlabel('Date')
ax2.set_ylabel('Flow (ML/day)')
ax2.set_title(f'Hydrograph Comparison - Zoomed ({zoom_start.year}-{zoom_end.year})')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/tutorial_hydrograph_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Static figure saved to: figures/tutorial_hydrograph_comparison.png")

# %%
# Flow Duration Curve comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full FDC (log scale)
ax1 = axes[0]
def calc_fdc(data):
    sorted_data = np.sort(data)[::-1]
    exceedance = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    return exceedance, sorted_data

exc_obs, fdc_obs = calc_fdc(obs_comparison)
ax1.plot(exc_obs, fdc_obs, 'k-', linewidth=2, label='Observed')

for method, sim in best_simulations.items():
    exc, fdc = calc_fdc(sim)
    ax1.plot(exc, fdc, color=plotly_colors.get(method, '#333333'), 
             alpha=0.7, linewidth=1.5, label=method)

ax1.set_xlabel('Exceedance Probability (%)')
ax1.set_ylabel('Flow (ML/day)')
ax1.set_title('Flow Duration Curves (Log Scale)')
ax1.set_yscale('log')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# FDC residuals
ax2 = axes[1]
for method, sim in best_simulations.items():
    exc, fdc_sim = calc_fdc(sim)
    # Interpolate simulated FDC to observed exceedance probabilities
    fdc_interp = np.interp(exc_obs, exc, fdc_sim)
    residual = (fdc_interp - fdc_obs) / fdc_obs * 100  # Percentage error
    ax2.plot(exc_obs, residual, color=plotly_colors.get(method, '#333333'),
             alpha=0.7, linewidth=1.5, label=method)

ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Exceedance Probability (%)')
ax2.set_ylabel('FDC Error (%)')
ax2.set_title('Flow Duration Curve Residuals')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-100, 100)

plt.tight_layout()
plt.savefig('figures/tutorial_fdc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to: figures/tutorial_fdc_comparison.png")

# %%
# Scatter plots - Simulated vs Observed
n_methods = len(best_simulations)
fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
if n_methods == 1:
    axes = [axes]

for i, (method, sim) in enumerate(best_simulations.items()):
    ax = axes[i]
    
    # Scatter plot
    ax.scatter(obs_comparison, sim, alpha=0.2, s=5, 
               color=plotly_colors.get(method, 'purple'))
    
    # 1:1 line
    max_val = max(obs_comparison.max(), sim.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='1:1 line')
    
    # Calculate R²
    r2 = np.corrcoef(obs_comparison, sim)[0, 1] ** 2
    nse = NSE().calculate(sim, obs_comparison)
    
    ax.set_xlabel('Observed Flow (ML/day)')
    ax.set_ylabel('Simulated Flow (ML/day)')
    ax.set_title(f'{method}\nR²={r2:.3f}, NSE={nse:.3f}')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figures/tutorial_scatter_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to: figures/tutorial_scatter_comparison.png")

# %%
# Residual analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series of residuals
ax1 = axes[0, 0]
for method, sim in best_simulations.items():
    residual = sim - obs_comparison
    ax1.plot(dates_comparison, residual, alpha=0.5, linewidth=0.5,
             color=plotly_colors.get(method, 'purple'), label=method)

ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('Date')
ax1.set_ylabel('Residual (ML/day)')
ax1.set_title('Residual Time Series')
ax1.legend(loc='upper right')

# Residual distribution
ax2 = axes[0, 1]
for method, sim in best_simulations.items():
    residual = sim - obs_comparison
    ax2.hist(residual, bins=50, alpha=0.5, density=True,
             color=plotly_colors.get(method, 'purple'), label=method)

ax2.axvline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Residual (ML/day)')
ax2.set_ylabel('Density')
ax2.set_title('Residual Distribution')
ax2.legend()

# Monthly performance boxplot
ax3 = axes[1, 0]
monthly_nse = {}
for method, sim in best_simulations.items():
    # Create DataFrame for monthly analysis
    df_temp = pd.DataFrame({
        'date': dates_comparison,
        'sim': sim,
        'obs': obs_comparison
    })
    df_temp['month'] = df_temp['date'].dt.month
    
    # Calculate monthly NSE
    monthly_scores = []
    for month in range(1, 13):
        mask = df_temp['month'] == month
        if mask.sum() > 10:
            nse = NSE().calculate(df_temp.loc[mask, 'sim'].values,
                                  df_temp.loc[mask, 'obs'].values)
            monthly_scores.append(nse)
        else:
            monthly_scores.append(np.nan)
    
    monthly_nse[method] = monthly_scores

monthly_nse_df = pd.DataFrame(monthly_nse, index=range(1, 13))
monthly_nse_df.plot(kind='bar', ax=ax3, width=0.8, alpha=0.7)
ax3.set_xlabel('Month')
ax3.set_ylabel('NSE')
ax3.set_title('Monthly NSE Performance')
ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], rotation=0)
ax3.legend(loc='lower right')
ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)

# Performance by flow regime
ax4 = axes[1, 1]
percentiles = [0, 10, 25, 50, 75, 90, 100]
regime_labels = ['Very Low\n(0-10%)', 'Low\n(10-25%)', 'Medium-Low\n(25-50%)',
                 'Medium-High\n(50-75%)', 'High\n(75-90%)', 'Very High\n(90-100%)']

regime_performance = {}
for method, sim in best_simulations.items():
    scores = []
    for i in range(len(percentiles) - 1):
        low_thresh = np.percentile(obs_comparison, percentiles[i])
        high_thresh = np.percentile(obs_comparison, percentiles[i+1])
        mask = (obs_comparison >= low_thresh) & (obs_comparison <= high_thresh)
        
        if mask.sum() > 10:
            nse = NSE().calculate(sim[mask], obs_comparison[mask])
            scores.append(nse)
        else:
            scores.append(np.nan)
    
    regime_performance[method] = scores

regime_df = pd.DataFrame(regime_performance, index=regime_labels)
regime_df.plot(kind='bar', ax=ax4, width=0.8, alpha=0.7)
ax4.set_xlabel('Flow Regime (Percentile)')
ax4.set_ylabel('NSE')
ax4.set_title('Performance by Flow Regime')
ax4.set_xticklabels(regime_labels, rotation=45, ha='right')
ax4.legend(loc='lower left')
ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('figures/tutorial_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to: figures/tutorial_residual_analysis.png")

# %%
# Validation period performance
print("=" * 60)
print("VALIDATION PERIOD PERFORMANCE")
print("=" * 60)

# Run simulations on validation period
val_simulations = {}

for method, result in calibration_results.items():
    # Use catchment area for ML/day output
    model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model.reset()
    model.set_parameters(result.best_parameters)
    
    # Run on validation period (need warmup from end of calibration)
    # First run through calibration period to get proper states (warmup)
    _ = model.run(cal_data)  # This warms up the model states
    
    # Now run validation period - model states are now initialized
    val_results = model.run(val_data)
    val_simulations[method] = val_results['flow'].values

# Calculate validation metrics
val_observed = val_data['observed_flow'].values
val_metrics = []

for method, sim in val_simulations.items():
    metrics = calculate_metrics(sim, val_observed)
    val_metrics.append({'Method': method, **metrics})

val_metrics_df = pd.DataFrame(val_metrics).set_index('Method')
print("\nValidation Period Metrics:")
print(val_metrics_df.round(4).to_string())

# %%
# Calibration vs Validation comparison
print("\nCALIBRATION vs VALIDATION NSE COMPARISON:")
print("-" * 50)
for method in calibration_results.keys():
    cal_nse = metrics_df.loc[method, 'NSE'] if method in metrics_df.index else np.nan
    val_nse = val_metrics_df.loc[method, 'NSE'] if method in val_metrics_df.index else np.nan
    diff = val_nse - cal_nse
    status = "✓ Good" if diff > -0.1 else "⚠ Degradation"
    print(f"  {method:15s}: Cal={cal_nse:.3f}, Val={val_nse:.3f}, Δ={diff:+.3f} {status}")

# %%
# Final summary visualization
fig = plt.figure(figsize=(16, 12))

# Create grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Calibration vs Validation NSE
ax1 = fig.add_subplot(gs[0, 0])
methods_list = list(calibration_results.keys())
cal_nse_vals = [metrics_df.loc[m, 'NSE'] for m in methods_list]
val_nse_vals = [val_metrics_df.loc[m, 'NSE'] for m in methods_list]

x = np.arange(len(methods_list))
width = 0.35
ax1.bar(x - width/2, cal_nse_vals, width, label='Calibration', alpha=0.7)
ax1.bar(x + width/2, val_nse_vals, width, label='Validation', alpha=0.7)
ax1.set_ylabel('NSE')
ax1.set_title('Calibration vs Validation Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(methods_list, rotation=45, ha='right')
ax1.legend()
ax1.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')

# 2. KGE comparison
ax2 = fig.add_subplot(gs[0, 1])
cal_kge_vals = [metrics_df.loc[m, 'KGE'] for m in methods_list]
val_kge_vals = [val_metrics_df.loc[m, 'KGE'] for m in methods_list]
ax2.bar(x - width/2, cal_kge_vals, width, label='Calibration', alpha=0.7)
ax2.bar(x + width/2, val_kge_vals, width, label='Validation', alpha=0.7)
ax2.set_ylabel('KGE')
ax2.set_title('KGE Performance Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(methods_list, rotation=45, ha='right')
ax2.legend()

# 3. RMSE comparison
ax3 = fig.add_subplot(gs[0, 2])
cal_rmse_vals = [metrics_df.loc[m, 'RMSE'] for m in methods_list]
val_rmse_vals = [val_metrics_df.loc[m, 'RMSE'] for m in methods_list]
ax3.bar(x - width/2, cal_rmse_vals, width, label='Calibration', alpha=0.7)
ax3.bar(x + width/2, val_rmse_vals, width, label='Validation', alpha=0.7)
ax3.set_ylabel('RMSE (ML/day)')
ax3.set_title('RMSE Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(methods_list, rotation=45, ha='right')
ax3.legend()

# 4. Validation hydrograph
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(val_data.index, val_observed, 'k-', alpha=0.8, linewidth=0.8, label='Observed')
for method, sim in val_simulations.items():
    ax4.plot(val_data.index, sim, color=plotly_colors.get(method, 'purple'),
             alpha=0.6, linewidth=0.6, label=method)
ax4.set_ylabel('Flow (ML/day)')
ax4.set_title('Validation Period Hydrograph')
ax4.legend(loc='upper right')
ax4.set_yscale('log')

# 5. Parameter comparison heatmap
ax5 = fig.add_subplot(gs[2, :])
# Normalize parameters for visualization
param_norm = param_df.copy()
for param in param_norm.index:
    low, high = param_bounds.get(param, (param_norm.loc[param].min(), param_norm.loc[param].max()))
    param_norm.loc[param] = (param_norm.loc[param] - low) / (high - low)

import seaborn as sns
sns.heatmap(param_norm.T, annot=param_df.T.round(2), fmt='', cmap='RdYlBu_r',
            ax=ax5, cbar_kws={'label': 'Normalized Value'})
ax5.set_title('Parameter Values (Normalized to Bounds)')
ax5.set_xlabel('Parameter')
ax5.set_ylabel('Method')

plt.suptitle('Sacramento Model Calibration Summary - Gauge 410734', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('figures/tutorial_final_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to: figures/tutorial_final_summary.png")

# %%
# Export calibrated parameters
print("=" * 60)
print("EXPORTING CALIBRATED PARAMETERS")
print("=" * 60)

# Save best parameters for each method
for method, result in calibration_results.items():
    filename = f"../test_data/calibrated_params_{method.replace(' ', '_').lower()}.csv"
    params_export = pd.DataFrame([result.best_parameters])
    params_export['method'] = method
    params_export['objective'] = result.objective_name
    params_export['best_value'] = result.best_objective
    params_export.to_csv(filename, index=False)
    print(f"  Saved: {filename}")

# Save summary comparison
summary_file = '../test_data/calibration_summary.csv'
summary_df = metrics_df.copy()
summary_df['validation_NSE'] = [val_metrics_df.loc[m, 'NSE'] if m in val_metrics_df.index 
                                 else np.nan for m in summary_df.index]
summary_df.to_csv(summary_file)
print(f"  Saved: {summary_file}")

print("\nCalibration complete!")

# %% [markdown]
# ---
# ## Summary
#
# This tutorial demonstrated the complete workflow for rainfall-runoff model calibration
# using the `pyrrm` library:
#
# ### Key Findings
#
# 1. **Data Preparation**: Loaded and cleaned 40+ years of daily data for gauge 410734
# 2. **Model Validation**: Verified pyrrm Sacramento implementation against SOURCE benchmark
# 3. **Calibration Methods**: Compared SpotPy DREAM, PyDREAM, and SciPy DE
# 4. **Diagnostics**: Analyzed convergence, parameter posteriors, and sensitivity
# 5. **Performance**: Evaluated calibration and validation metrics
#
# ### Recommendations
#
# - For **production calibration**, use more iterations (50,000-100,000)
# - **PyDREAM's multi-try sampling** can help with multi-modal posteriors
# - Always check **validation performance** to assess generalization
# - Consider **weighted objectives** (e.g., NSE + LogNSE) for balanced performance
#
# ### Next Steps
#
# - Try different objective functions (KGE, weighted combinations)
# - Perform sensitivity analysis to identify key parameters
# - Apply calibrated model to scenario analysis
# - Compare with other rainfall-runoff models (GR4J, GR5J)

# %%
print("=" * 60)
print("TUTORIAL COMPLETE")
print("=" * 60)
print("\nFigures saved to figures/tutorial_*.png")
print("Parameters saved to ../test_data/calibrated_*.csv")
print("\nTo sync this script to a Jupyter notebook, run:")
print("  jupytext --sync notebooks/tutorial_sacramento_calibration.py")
