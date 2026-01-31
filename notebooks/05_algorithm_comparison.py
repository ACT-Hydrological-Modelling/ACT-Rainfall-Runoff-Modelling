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
# # Calibration Algorithms: Finding Optimal Parameters
#
# ## Purpose
#
# This notebook compares different optimization algorithms for rainfall-runoff
# model calibration. Understanding their trade-offs helps you choose the right
# tool for your application.
#
# ## What You'll Learn
#
# - The landscape of calibration algorithms available in pyrrm
# - Fundamental difference: MCMC (Bayesian) vs Optimization (frequentist)
# - When to use DREAM, PyDREAM, SCE-UA, or Differential Evolution
# - How to interpret convergence diagnostics
# - Practical considerations: speed, parallelization, uncertainty quantification
#
# ## Prerequisites
#
# - Completed **Notebook 02: Calibration Quickstart**
# - Completed **Notebook 03: Objective Functions** (recommended)
#
# ## Estimated Time
#
# - ~20 minutes for concepts
# - ~2-4 hours for full algorithm comparison (can be shortened)
#
# ## Key Insight
#
# > **MCMC methods (DREAM, PyDREAM) give you parameter uncertainty estimates.**
# > **Optimization methods (SCE-UA, SciPy DE) give you a single "best" answer faster.**

# %% [markdown]
# ---
# ## Algorithm Overview
#
# ### Two Paradigms for Calibration
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        CALIBRATION ALGORITHMS                               │
# ├─────────────────────────────────┬───────────────────────────────────────────┤
# │          MCMC METHODS           │        OPTIMIZATION METHODS               │
# │       (Sample posterior)        │         (Find optimum)                    │
# ├─────────────────────────────────┼───────────────────────────────────────────┤
# │  SpotPy DREAM                   │  SCE-UA                                   │
# │  PyDREAM (MT-DREAM(ZS))         │  SciPy Differential Evolution             │
# ├─────────────────────────────────┼───────────────────────────────────────────┤
# │  OUTPUT: Posterior distribution │  OUTPUT: Single best point                │
# │  - Parameter samples            │  - Optimal parameter values               │
# │  - Credible intervals           │  - Best objective value                   │
# │  - Uncertainty estimates        │                                           │
# ├─────────────────────────────────┼───────────────────────────────────────────┤
# │  SPEED: Slower (10K+ samples)   │  SPEED: Faster (1-5K evaluations)         │
# │  USE: Uncertainty, Bayesian     │  USE: Point estimates, quick calibration  │
# └─────────────────────────────────┴───────────────────────────────────────────┘
# ```
#
# ### Available Algorithms in pyrrm
#
# | Algorithm | Type | Package | Uncertainty? | Speed | Complexity |
# |-----------|------|---------|--------------|-------|------------|
# | **SpotPy DREAM** | MCMC | SpotPy | Yes | Slow | Medium |
# | **PyDREAM** | MCMC | PyDREAM | Yes | Slow | High |
# | **SCE-UA** | Optimization | SpotPy | No | Fast | Low |
# | **SciPy DE** | Optimization | SciPy | No | Fast | Low |

# %% [markdown]
# ---
# ## Understanding the Algorithms
#
# ### MCMC Methods: Sampling the Posterior
#
# **MCMC (Markov Chain Monte Carlo)** methods don't just find *one* good parameter
# set - they sample the entire posterior distribution of parameters consistent
# with the data.
#
# #### SpotPy DREAM
#
# DREAM (DiffeRential Evolution Adaptive Metropolis) combines:
# - **Differential Evolution**: Uses multiple chains that learn from each other
# - **Adaptive Metropolis**: Automatically tunes proposal distribution
#
# **Key settings:**
# - `n_iterations`: Total samples to draw
# - `n_chains`: Number of parallel chains (typically 3-8)
# - `convergence_threshold`: Gelman-Rubin R-hat threshold (default 1.2)
#
# #### PyDREAM (MT-DREAM(ZS))
#
# An advanced variant with:
# - **Multi-Try**: Proposes multiple candidates, picks best
# - **Snooker Updates**: Better exploration of multi-modal posteriors
# - **Z-matrix**: Uses past history for better proposals
#
# **When to use PyDREAM:**
# - Complex, multi-modal posteriors
# - When SpotPy DREAM struggles to converge
# - Need better mixing between modes
#
# ### Optimization Methods: Finding the Peak
#
# #### SCE-UA (Shuffled Complex Evolution)
#
# Developed specifically for hydrology (Duan et al., 1992):
# - Divides population into "complexes"
# - Each complex evolves independently
# - Periodically shuffles members between complexes
#
# **Key settings:**
# - `n_iterations`: Maximum function evaluations
# - `ngs`: Number of complexes (more = better search)
#
# #### SciPy Differential Evolution
#
# Modern evolutionary algorithm:
# - Creates new candidates by combining existing solutions
# - More general-purpose than SCE-UA
#
# **Key settings:**
# - `maxiter`: Maximum generations
# - `popsize`: Population size multiplier

# %% [markdown]
# ---
# ## The pyrrm.calibration Architecture
#
# ```
# pyrrm.calibration/
# ├── runner.py              ← CalibrationRunner (unified interface)
# │                             • run_spotpy_dream()
# │                             • run_pydream()
# │                             • run_sceua()
# │                             • run_differential_evolution()
# │
# ├── spotpy_adapter.py      ← SpotPy interface
# │                             • DREAM algorithm
# │                             • SCE-UA algorithm
# │
# ├── pydream_adapter.py     ← PyDREAM interface
# │                             • MT-DREAM(ZS) algorithm
# │
# └── scipy_adapter.py       ← SciPy interface
#                               • Differential Evolution
# ```
#
# ### The Adapter Pattern
#
# Each algorithm has a different API. The adapters normalize them:
#
# ```python
# # Same interface, different algorithms
# result = runner.run_spotpy_dream(n_iterations=10000)
# result = runner.run_pydream(n_iterations=5000)
# result = runner.run_sceua(n_iterations=5000)
# result = runner.run_differential_evolution(maxiter=500)
# ```
#
# All return a `CalibrationResult` with:
# - `best_parameters`: Dictionary of optimal values
# - `best_objective`: Best metric achieved
# - `all_samples`: Full sampling history (if available)
# - `runtime_seconds`: Execution time

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
import time

# Interactive visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("CALIBRATION ALGORITHM COMPARISON")
print("=" * 70)

# %%
# Import pyrrm components
from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration import (
    CalibrationRunner, 
    CalibrationResult,
    SPOTPY_AVAILABLE,
    PYDREAM_AVAILABLE
)
from pyrrm.calibration.objective_functions import (
    NSE, KGE, GaussianLikelihood, calculate_metrics
)

print("\npyrrm.calibration imported successfully!")
print(f"\nAvailable backends:")
print(f"  SpotPy (DREAM, SCE-UA): {SPOTPY_AVAILABLE}")
print(f"  PyDREAM: {PYDREAM_AVAILABLE}")
print(f"  SciPy: Always available")

# %% [markdown]
# ---
# ## Prepare Calibration Data

# %%
# Load data
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667

# Load datasets
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

# Define calibration period
WARMUP_DAYS = 365
CAL_START = pd.Timestamp('1990-01-01')
CAL_END = pd.Timestamp('1994-12-31')  # 5 years for demo

cal_data = data[(data.index >= CAL_START) & (data.index <= CAL_END)].copy()
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

print("=" * 50)
print("CALIBRATION DATA")
print("=" * 50)
print(f"\nPeriod: {CAL_START.date()} to {CAL_END.date()}")
print(f"Records: {len(cal_data)}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_data) - WARMUP_DAYS} days")

# %% [markdown]
# ---
# ## Objective Functions for Different Algorithm Types
#
# **Important:** MCMC and optimization methods require different objective functions:
#
# | Method Type | Objective | Reason |
# |-------------|-----------|--------|
# | **MCMC** | GaussianLikelihood | Log-probability for Bayesian inference |
# | **Optimization** | NSE | Standard efficiency metric |

# %%
# Define objective functions
mcmc_objective = GaussianLikelihood()  # For DREAM, PyDREAM
optim_objective = NSE()                 # For SCE-UA, SciPy DE

print("=" * 50)
print("OBJECTIVE FUNCTIONS BY METHOD TYPE")
print("=" * 50)
print(f"\nMCMC Methods (SpotPy DREAM, PyDREAM):")
print(f"  Objective: {mcmc_objective.name}")
print(f"  Maximize: {mcmc_objective.maximize}")
print(f"  Note: Log-likelihood required for Bayesian inference")

print(f"\nOptimization Methods (SCE-UA, SciPy DE):")
print(f"  Objective: {optim_objective.name}")
print(f"  Maximize: {optim_objective.maximize}")
print(f"  Note: Standard efficiency metric for optimization")

# %% [markdown]
# ---
# ## Running the Calibrations
#
# We'll run all four algorithms and compare their results.
#
# **Note:** For demonstration, we use reduced iterations. Increase for production!

# %%
# Configuration for demo (reduce for faster execution)
DEMO_MODE = True  # Set False for production runs

if DEMO_MODE:
    SPOTPY_DREAM_ITERATIONS = 3000
    PYDREAM_ITERATIONS = 2000
    SCEUA_ITERATIONS = 3000
    SCIPY_MAXITER = 200
    N_CHAINS = 5
    print("⚠ DEMO MODE: Using reduced iterations for speed")
else:
    SPOTPY_DREAM_ITERATIONS = 10000
    PYDREAM_ITERATIONS = 5000
    SCEUA_ITERATIONS = 10000
    SCIPY_MAXITER = 500
    N_CHAINS = 8
    print("PRODUCTION MODE: Full iterations")

print(f"\nConfiguration:")
print(f"  SpotPy DREAM: {SPOTPY_DREAM_ITERATIONS} iterations, {N_CHAINS} chains")
print(f"  PyDREAM: {PYDREAM_ITERATIONS} iterations, {N_CHAINS} chains")
print(f"  SCE-UA: {SCEUA_ITERATIONS} iterations")
print(f"  SciPy DE: {SCIPY_MAXITER} generations")

# %% [markdown]
# ### 1. SpotPy DREAM
#
# DREAM combines differential evolution with adaptive Metropolis sampling.
# Multiple chains explore the parameter space, sharing information.

# %%
print("=" * 70)
print("1. SPOTPY DREAM CALIBRATION")
print("=" * 70)

if SPOTPY_AVAILABLE:
    runner_mcmc = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=mcmc_objective,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"\nObjective: {mcmc_objective.name}")
    print(f"Chains: {N_CHAINS}")
    print(f"Iterations: {SPOTPY_DREAM_ITERATIONS}")
    print("\nRunning... (this may take several minutes)")
    
    spotpy_result = runner_mcmc.run_spotpy_dream(
        n_iterations=SPOTPY_DREAM_ITERATIONS,
        n_chains=N_CHAINS,
        convergence_threshold=1.2,
        dbname='algo_spotpy_dream',
        dbformat='csv',
        parallel='seq'
    )
    
    print("\n" + spotpy_result.summary())
else:
    print("\n⚠ SpotPy not installed. Install with: pip install spotpy")
    spotpy_result = None

# %% [markdown]
# ### 2. PyDREAM (MT-DREAM(ZS))
#
# Multi-Try DREAM with snooker updates. Better mixing for complex posteriors.

# %%
print("=" * 70)
print("2. PYDREAM CALIBRATION (MT-DREAM(ZS))")
print("=" * 70)

if PYDREAM_AVAILABLE:
    runner_pydream = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=mcmc_objective,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"\nObjective: {mcmc_objective.name}")
    print(f"Chains: {N_CHAINS}")
    print(f"Iterations per chain: {PYDREAM_ITERATIONS}")
    print("\nRunning... (this may take several minutes)")
    
    pydream_result = runner_pydream.run_pydream(
        n_iterations=PYDREAM_ITERATIONS,
        n_chains=N_CHAINS,
        multitry=5,
        snooker=0.1,
        parallel=False,
        adapt_crossover=False,
        dbname='algo_pydream',
        dbformat='csv',
        verbose=False
    )
    
    print("\n" + pydream_result.summary())
else:
    print("\n⚠ PyDREAM not installed. Install with: pip install pydream")
    pydream_result = None

# %% [markdown]
# ### 3. SCE-UA (Shuffled Complex Evolution)
#
# The classic hydrology algorithm. Fast, reliable, no uncertainty estimates.

# %%
print("=" * 70)
print("3. SCE-UA CALIBRATION")
print("=" * 70)

if SPOTPY_AVAILABLE:
    runner_sceua = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=optim_objective,
        warmup_period=WARMUP_DAYS
    )
    
    # SCE-UA configuration (Duan et al., 1994)
    # Rule: ngs = 2*n + 1 where n = number of parameters
    n_params_sceua = len(runner_sceua._param_bounds)
    ngs_sceua = 2 * n_params_sceua + 1
    
    print(f"\nObjective: {optim_objective.name}")
    print(f"Parameters: {n_params_sceua}")
    print(f"Complexes (ngs): {ngs_sceua} (2×{n_params_sceua}+1)")
    print(f"Max iterations: {SCEUA_ITERATIONS}")
    print("\nRunning...")
    
    sceua_result = runner_sceua.run_sceua(
        n_iterations=SCEUA_ITERATIONS,
        ngs=ngs_sceua,
        kstop=5,
        pcento=0.01,
        dbname='algo_sceua',
        dbformat='csv'
    )
    
    print("\n" + sceua_result.summary())
else:
    print("\n⚠ SpotPy not installed.")
    sceua_result = None

# %% [markdown]
# ### 4. SciPy Differential Evolution
#
# Modern evolutionary optimization. Always available, no extra dependencies.

# %%
print("=" * 70)
print("4. SCIPY DIFFERENTIAL EVOLUTION")
print("=" * 70)

runner_scipy = CalibrationRunner(
    model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
    inputs=cal_inputs,
    observed=cal_observed,
    objective=optim_objective,
    warmup_period=WARMUP_DAYS
)

print(f"\nObjective: {optim_objective.name}")
print(f"Max generations: {SCIPY_MAXITER}")
print("\nRunning...")

scipy_result = runner_scipy.run_differential_evolution(
    maxiter=SCIPY_MAXITER,
    popsize=15,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    workers=1,
    disp=True
)

print("\n" + scipy_result.summary())

# %% [markdown]
# ---
# ## Results Comparison

# %%
# Collect all results
all_results = {}
if spotpy_result is not None:
    all_results['SpotPy DREAM'] = spotpy_result
if pydream_result is not None:
    all_results['PyDREAM'] = pydream_result
if sceua_result is not None:
    all_results['SCE-UA'] = sceua_result
all_results['SciPy DE'] = scipy_result

# Define colors
METHOD_COLORS = {
    'SpotPy DREAM': '#1f77b4',
    'PyDREAM': '#d62728',
    'SCE-UA': '#2ca02c',
    'SciPy DE': '#9467bd',
}

print("=" * 70)
print("CALIBRATION RESULTS SUMMARY")
print("=" * 70)
print(f"\n{len(all_results)} algorithm(s) completed")

# %%
# Performance comparison
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

# Generate simulations with each result
simulations = {}
for method, result in all_results.items():
    model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model.set_parameters(result.best_parameters)
    model.reset()
    sim_results = model.run(cal_data)
    simulations[method] = sim_results['runoff'].values[WARMUP_DAYS:]

obs_compare = cal_observed[WARMUP_DAYS:]
dates_compare = cal_data.index[WARMUP_DAYS:]

# Calculate metrics for all
metrics_comparison = {}
for method, sim in simulations.items():
    metrics = calculate_metrics(sim, obs_compare)
    metrics_comparison[method] = metrics

metrics_df = pd.DataFrame(metrics_comparison).T

print("\nPerformance Metrics:")
print(metrics_df.round(4).to_string())

# %%
# Runtime comparison
print("\n" + "=" * 70)
print("RUNTIME COMPARISON")
print("=" * 70)
print(f"\n{'Method':<20} {'Runtime (s)':>15} {'Best Objective':>15}")
print("-" * 55)
for method, result in all_results.items():
    print(f"{method:<20} {result.runtime_seconds:>15.1f} {result.best_objective:>15.4f}")

# %%
# Parameter comparison
print("\n" + "=" * 70)
print("PARAMETER COMPARISON")
print("=" * 70)

param_comparison = {method: result.best_parameters for method, result in all_results.items()}
param_df = pd.DataFrame(param_comparison)

# Show key parameters
key_params = ['uztwm', 'lztwm', 'uzk', 'lzpk', 'lzsk']
print("\nKey parameters by algorithm:")
print(param_df.loc[key_params].round(4).to_string())

# %% [markdown]
# ---
# ## Convergence Diagnostics
#
# ### Gelman-Rubin Statistic (R-hat)
#
# For MCMC methods, we check convergence using the Gelman-Rubin statistic:
# - **R-hat ≈ 1.0**: Chains have converged
# - **R-hat > 1.2**: Chains haven't converged, need more iterations
#
# The idea: if multiple chains are exploring the same distribution, their
# within-chain variance should match their between-chain variance.

# %%
print("=" * 70)
print("CONVERGENCE DIAGNOSTICS (MCMC Methods)")
print("=" * 70)

mcmc_methods = ['SpotPy DREAM', 'PyDREAM']
for method in mcmc_methods:
    if method in all_results:
        result = all_results[method]
        if 'gelman_rubin' in result.convergence_diagnostics:
            print(f"\n{method} - Gelman-Rubin Statistics (R-hat):")
            print("-" * 50)
            gr_values = result.convergence_diagnostics['gelman_rubin']
            for param, gr in gr_values.items():
                status = "✓" if gr < 1.2 else "⚠"
                print(f"  {status} {param:8s}: {gr:.4f}")
            
            converged = result.convergence_diagnostics.get('converged', 'N/A')
            print(f"\n  Overall: {'Converged' if converged else 'Not converged'}")
        else:
            print(f"\n{method}: Convergence diagnostics not available")

print("\nNote: Optimization methods (SCE-UA, SciPy DE) don't have R-hat diagnostics")

# %% [markdown]
# ---
# ## Posterior Visualization (MCMC Methods)
#
# One key advantage of MCMC methods: we can visualize the uncertainty in parameters.

# %%
# Check if we have MCMC samples
mcmc_with_samples = {}
for method in ['SpotPy DREAM', 'PyDREAM']:
    if method in all_results:
        result = all_results[method]
        if result.all_samples is not None and len(result.all_samples) > 0:
            mcmc_with_samples[method] = result

if len(mcmc_with_samples) > 0:
    print("=" * 70)
    print("POSTERIOR DISTRIBUTIONS (MCMC Methods)")
    print("=" * 70)
    
    # Get parameter names
    param_names = list(all_results[list(all_results.keys())[0]].best_parameters.keys())
    
    # Create histogram plots for key parameters
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names[:12]):  # First 12 params
        ax = axes[i]
        
        for method, result in mcmc_with_samples.items():
            samples = result.all_samples
            if param in samples.columns:
                ax.hist(samples[param].values, bins=50, alpha=0.5, density=True,
                       color=METHOD_COLORS.get(method, 'gray'), label=method)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Parameter Posterior Distributions (MCMC Methods)', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/04_posterior_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved: figures/04_posterior_distributions.png")
else:
    print("No MCMC samples available for posterior visualization")

# %% [markdown]
# ---
# ## Results Visualization

# %%
# Comprehensive comparison figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Hydrograph Comparison (Log Scale)',
        'NSE by Algorithm',
        'Flow Duration Curves',
        'Runtime Comparison'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

# 1. Hydrograph
fig.add_trace(
    go.Scatter(x=dates_compare, y=obs_compare, name='Observed',
               line=dict(color='black', width=1.5)),
    row=1, col=1
)
for method, sim in simulations.items():
    fig.add_trace(
        go.Scatter(x=dates_compare, y=sim, name=method,
                   line=dict(color=METHOD_COLORS[method], width=1), opacity=0.8),
        row=1, col=1
    )

# 2. NSE comparison
nse_values = [metrics_comparison[m]['NSE'] for m in all_results.keys()]
fig.add_trace(
    go.Bar(x=list(all_results.keys()), y=nse_values,
           marker_color=[METHOD_COLORS[m] for m in all_results.keys()],
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
for method, sim in simulations.items():
    sim_sorted = np.sort(sim)[::-1]
    fig.add_trace(
        go.Scatter(x=exc, y=sim_sorted, name=f'{method} FDC',
                   line=dict(color=METHOD_COLORS[method], width=1.5), showlegend=False),
        row=2, col=1
    )

# 4. Runtime comparison
runtime_values = [all_results[m].runtime_seconds for m in all_results.keys()]
fig.add_trace(
    go.Bar(x=list(all_results.keys()), y=runtime_values,
           marker_color=[METHOD_COLORS[m] for m in all_results.keys()],
           showlegend=False),
    row=2, col=2
)

# Update axes
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig.update_yaxes(title_text="NSE", row=1, col=2)
fig.update_xaxes(title_text="Exceedance %", row=2, col=1)
fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=1)
fig.update_yaxes(title_text="Runtime (seconds)", row=2, col=2)

fig.update_layout(
    title="<b>Algorithm Comparison Results</b>",
    height=800,
    showlegend=True,
    legend=dict(orientation='h', y=1.02)
)
fig.show()

# %%
# Save static figure
fig_static, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. NSE comparison
ax = axes[0, 0]
ax.bar(list(all_results.keys()), nse_values,
       color=[METHOD_COLORS[m] for m in all_results.keys()])
ax.set_ylabel('NSE')
ax.set_title('NSE by Algorithm')
ax.axhline(0.7, color='green', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=45)

# 2. Runtime comparison
ax = axes[0, 1]
ax.bar(list(all_results.keys()), runtime_values,
       color=[METHOD_COLORS[m] for m in all_results.keys()])
ax.set_ylabel('Runtime (seconds)')
ax.set_title('Runtime Comparison')
ax.tick_params(axis='x', rotation=45)

# 3. Parameter spread (boxplot proxy using bar + error)
ax = axes[1, 0]
param_to_show = 'uztwm'
values = [all_results[m].best_parameters[param_to_show] for m in all_results.keys()]
ax.bar(list(all_results.keys()), values,
       color=[METHOD_COLORS[m] for m in all_results.keys()])
ax.set_ylabel(f'{param_to_show} value')
ax.set_title(f'Best {param_to_show} by Algorithm')
ax.tick_params(axis='x', rotation=45)

# 4. FDC comparison
ax = axes[1, 1]
ax.plot(exc, obs_sorted, 'k-', lw=2, label='Observed')
for method, sim in simulations.items():
    sim_sorted = np.sort(sim)[::-1]
    ax.plot(exc, sim_sorted, color=METHOD_COLORS[method], lw=1.5, label=method)
ax.set_xlabel('Exceedance (%)')
ax.set_ylabel('Flow (ML/day)')
ax.set_title('Flow Duration Curves')
ax.set_yscale('log')
ax.legend(fontsize=8)

plt.suptitle('Algorithm Comparison Summary', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('figures/04_algorithm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/04_algorithm_comparison.png")

# %% [markdown]
# ---
# ## Practical Guidance
#
# ### Decision Flowchart: Which Algorithm Should I Use?
#
# ```
# START
#   │
#   ▼
# Do you need uncertainty estimates?
#   │
#   ├─► YES
#   │     │
#   │     ▼
#   │   Is the posterior complex/multi-modal?
#   │     │
#   │     ├─► YES → Use PyDREAM (MT-DREAM(ZS))
#   │     │           Better mixing, snooker updates
#   │     │
#   │     └─► NO  → Use SpotPy DREAM
#   │                 Simpler, well-tested
#   │
#   └─► NO
#         │
#         ▼
#       Need fast calibration?
#         │
#         ├─► YES → Use SCE-UA
#         │           Classic, reliable, fast
#         │
#         └─► NO  → Either SCE-UA or SciPy DE
#                     Both give similar results
# ```
#
# ### When to Use Each Algorithm
#
# | Algorithm | Use When | Avoid When |
# |-----------|----------|------------|
# | **SpotPy DREAM** | Need uncertainty; standard problems | Very limited time |
# | **PyDREAM** | Complex posteriors; multi-modal | Simple problems (overkill) |
# | **SCE-UA** | Quick calibration; point estimate | Need uncertainty |
# | **SciPy DE** | No extra dependencies; simple setup | Need uncertainty |
#
# ### Parallelization for Production
#
# For production calibrations, use parallelization:
#
# **SpotPy DREAM with MPI:**
# ```bash
# # Create script: mpi_calibration.py
# mpirun -n 9 python mpi_calibration.py  # 1 master + 8 workers
# ```
#
# **PyDREAM (automatic):**
# - Chains run in parallel automatically
# - Multi-try evaluations can be parallelized with `parallel=True`

# %% [markdown]
# ---
# ## Summary
#
# ### Key Takeaways
#
# 1. **MCMC vs Optimization**: Choose based on whether you need uncertainty
#    - MCMC (DREAM, PyDREAM): Slower but gives credible intervals
#    - Optimization (SCE-UA, SciPy DE): Faster point estimates
#
# 2. **All algorithms found similar optima**: This is expected for well-behaved
#    problems. Differences are larger for complex, multi-modal posteriors.
#
# 3. **Convergence matters for MCMC**: Check Gelman-Rubin R-hat < 1.2
#
# 4. **Runtime varies significantly**: Optimization is 5-10x faster than MCMC

# %%
print("=" * 70)
print("ALGORITHM COMPARISON SUMMARY")
print("=" * 70)

# Create summary table
summary_data = []
for method, result in all_results.items():
    method_type = "MCMC" if method in ['SpotPy DREAM', 'PyDREAM'] else "Optimization"
    nse = metrics_comparison[method]['NSE']
    summary_data.append({
        'Algorithm': method,
        'Type': method_type,
        'NSE': f"{nse:.4f}",
        'Runtime (s)': f"{result.runtime_seconds:.1f}",
        'Uncertainty?': 'Yes' if method_type == 'MCMC' else 'No'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("""
\nRecommendations:
  • For quick calibration     → SCE-UA
  • For uncertainty analysis  → SpotPy DREAM
  • For complex posteriors    → PyDREAM
  • For minimal dependencies  → SciPy DE
""")

# %% [markdown]
# ---
# ## Next Steps
#
# - **Notebook 05**: Monitor long-running calibrations in real-time
# - For production: Increase iterations significantly (10,000+ for MCMC)
# - Consider MPI parallelization for large-scale calibrations

# %%
print("=" * 70)
print("ALGORITHM COMPARISON COMPLETE")
print("=" * 70)
print("""
You now understand:
  ✓ MCMC vs Optimization paradigms
  ✓ When to use each algorithm
  ✓ How to interpret convergence diagnostics
  ✓ Runtime/uncertainty trade-offs
  
Happy calibrating!
""")
