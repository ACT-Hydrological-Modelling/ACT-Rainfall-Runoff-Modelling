# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # APEX: Adaptive Process-Explicit Objective Function
#
# ## Overview
#
# This notebook demonstrates **APEX** (Adaptive Process-Explicit), an objective function
# that extends SDEB (Lerat et al., 2013) with multiplicative dynamics and lag penalty
# terms for comprehensive hydrological model calibration.
#
# ## What Makes APEX Different
#
# APEX follows SDEB's proven multiplicative structure while adding **two key contributions**:
#
# 1. **Dynamics Multiplier**: Penalizes mismatch in gradient/rate-of-change patterns
#    - `DynamicsMultiplier = 1 + κ × (1 - ρ_gradient)`
#    - Ensures the model captures recession behavior, not just levels
#
# 2. **Lag Multiplier** (optional): Penalizes systematic timing offsets
#    - `LagMultiplier = 1 + λ × |optimal_lag| / τ`
#    - Catches models that consistently lead or lag observations
#
# ## APEX Formula
#
# ```
# APEX = [α × E_chron + (1-α) × E_ranked] × BiasMultiplier × DynamicsMultiplier × [LagMultiplier]
# ```
#
# Where:
# - `E_chron` = Chronological error term (timing-sensitive)
# - `E_ranked` = Ranked/FDC error term (timing-insensitive)
# - `α` = Weight between timing and distribution (default: 0.1)
#
# ## What This Notebook Does
#
# 1. **Loads pre-calibrated baselines** from notebook 02:
#    - NSE (traditional high-flow biased)
#    - InvNSE (inverse-transformed for low flows)
#    - SDEB (state-of-the-art composite)
#
# 2. **Calibrates APEX** with the dynamics multiplier enabled
#
# 3. **Comprehensive comparison** including dynamics diagnostics
#
# ## Estimated Runtime
#
# - Data loading: < 1 minute
# - APEX calibration: ~20-30 minutes
# - Analysis: ~5 minutes
# - **Total**: ~30-40 minutes

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# pyrrm imports
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner, CalibrationReport
from pyrrm.objectives import (
    NSE, KGE, RMSE, MAE, PBIAS,
    FlowTransformation, PearsonCorrelation,
    FDCMetric, SignatureMetric, SDEB,
    APEX  # APEX: SDEB-enhanced with dynamics multiplier
)

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("✓ Imports complete")

# %% [markdown]
# ---
# # PART 1: DATA LOADING AND PREPARATION
#
# We follow the exact same approach as notebook 02 to ensure data consistency.

# %%
# Configuration
DATA_DIR = Path('../data/410734')
CATCHMENT_ID = '410734'
CATCHMENT_AREA_KM2 = 516.62667  # km²
WARMUP_DAYS = 365

print("CONFIGURATION")
print("=" * 70)
print(f"Catchment: {CATCHMENT_ID}")
print(f"Area: {CATCHMENT_AREA_KM2:.2f} km²")
print(f"Warmup period: {WARMUP_DAYS} days")

# %%
# Load rainfall data
rainfall_file = DATA_DIR / 'Default Input Set - Rain_QBN01.csv'
rainfall_df = pd.read_csv(rainfall_file, parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

print("\nRAINFALL DATA")
print("=" * 70)
print(f"Records: {len(rainfall_df):,} days")
print(f"Period: {rainfall_df.index.min().date()} to {rainfall_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(rainfall_df['rainfall'].describe().round(2))

# %%
# Load PET data
pet_file = DATA_DIR / 'Default Input Set - Mwet_QBN01.csv'
pet_df = pd.read_csv(pet_file, parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

print("\nPOTENTIAL EVAPOTRANSPIRATION (PET)")
print("=" * 70)
print(f"Records: {len(pet_df):,} days")
print(f"Period: {pet_df.index.min().date()} to {pet_df.index.max().date()}")
print(f"\nStatistics (mm/day):")
print(pet_df['pet'].describe().round(2))

# %%
# Load observed streamflow (MUST use same file as notebook 02 for consistency)
flow_file = DATA_DIR / '410734_output_SDmodel.csv'
flow_df = pd.read_csv(flow_file, parse_dates=['Date'], index_col='Date')

# Extract the recorded gauging station flow column (ML/day)
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']

# Clean observed flow data (CRITICAL!)
# Handle missing values and negative flows
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

print("\nOBSERVED STREAMFLOW")
print("=" * 70)
print(f"Records: {len(observed_df):,} days (after cleaning)")
print(f"Period: {observed_df.index.min().date()} to {observed_df.index.max().date()}")
print(f"\nStatistics (ML/day):")
print(observed_df['observed_flow'].describe().round(2))

# %%
# Merge all datasets
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')

print("\nMERGED DATASET")
print("=" * 70)
print(f"Total records: {len(data):,} days")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
print(f"Columns: {list(data.columns)}")
print(f"\nMissing values:")
print(data.isna().sum())

# %%
# Prepare data for calibration
cal_data = data.copy()
cal_inputs = cal_data[['rainfall', 'pet']].copy()
cal_observed = cal_data['observed_flow'].values

print("\nCALIBRATION DATA PREPARED")
print("=" * 70)
print(f"Input shape: {cal_inputs.shape}")
print(f"Observed shape: {cal_observed.shape}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_observed) - WARMUP_DAYS:,} days")
print("\n✓ Data loading complete!")

# %% [markdown]
# ---
# # PART 2: LOADING BASELINE CALIBRATIONS
#
# We load NSE, InvNSE, and SDEB calibrations that were created in notebook 02.
# We extract the best parameters and re-run simulations using our loaded data
# to ensure all comparisons use identical observed data.

# %%
print("\n" + "=" * 80)
print("LOADING PRE-CALIBRATED BASELINES")
print("=" * 80)

# Create comparison DataFrame (after warmup) - same approach as notebook 02
comparison = cal_data.iloc[WARMUP_DAYS:].copy()
obs_flow = comparison['observed_flow'].values
dates = comparison.index

print(f"\nComparison period: {len(comparison):,} days (after warmup)")
print(f"Date range: {dates.min().date()} to {dates.max().date()}")

# Load NSE calibration and re-simulate
print("\n1. Loading NSE calibration...")
report_nse = CalibrationReport.load('../test_data/reports/410734_nse.pkl')
result_nse = report_nse.result

nse_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
nse_model.set_parameters(result_nse.best_parameters)
nse_model.reset()
sim_nse = nse_model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]

print(f"   ✓ NSE: {result_nse.best_objective:.4f}")
print(f"   Simulation length: {len(sim_nse):,} days")

# Load InvNSE calibration and re-simulate
print("\n2. Loading InvNSE calibration...")
report_invnse = CalibrationReport.load('../test_data/reports/410734_invnse.pkl')
result_invnse = report_invnse.result

invnse_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
invnse_model.set_parameters(result_invnse.best_parameters)
invnse_model.reset()
sim_invnse = invnse_model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]

print(f"   ✓ InvNSE: {result_invnse.best_objective:.4f}")
print(f"   Simulation length: {len(sim_invnse):,} days")

# Load SDEB calibration and re-simulate
print("\n3. Loading SDEB calibration...")
report_sdeb = CalibrationReport.load('../test_data/reports/410734_sdeb.pkl')
result_sdeb = report_sdeb.result

sdeb_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
sdeb_model.set_parameters(result_sdeb.best_parameters)
sdeb_model.reset()
sim_sdeb = sdeb_model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]

print(f"   ✓ SDEB: {result_sdeb.best_objective:.4f}")
print(f"   Simulation length: {len(sim_sdeb):,} days")

# Verify all simulations have same length as observed
print("\n" + "=" * 80)
print("DATA CONSISTENCY CHECK")
print("=" * 80)
print(f"Observed flow: {len(obs_flow):,} days")
print(f"NSE simulated: {len(sim_nse):,} days")
print(f"InvNSE simulated: {len(sim_invnse):,} days")
print(f"SDEB simulated: {len(sim_sdeb):,} days")
assert len(obs_flow) == len(sim_nse) == len(sim_invnse) == len(sim_sdeb), "Data length mismatch!"
print("✓ All data series have matching lengths!")

print("\n" + "=" * 80)
print("✓ All baseline calibrations loaded and re-simulated successfully!")
print("=" * 80)

# %% [markdown]
# ---
# # PART 3: APEX CALIBRATION
#
# Now we calibrate using the **APEX** objective function, which extends SDEB with:
#
# - **Dynamics Multiplier**: Penalizes gradient mismatch (ensures recession behavior is captured)
# - **Optional Lag Multiplier**: Penalizes systematic timing offsets
# - **Regime Emphasis**: Continuous flow-regime weighting in ranked term
#
# ## APEX Formula
#
# ```
# APEX = [α × E_chron + (1-α) × E_ranked] × BiasMultiplier × DynamicsMultiplier × [LagMultiplier]
# ```

# %%
print("\n" + "=" * 80)
print("APEX CALIBRATIONS - SENSITIVITY ANALYSIS")
print("=" * 80)

# =============================================================================
# COMPREHENSIVE APEX CONFIGURATION EXPERIMENTS
# =============================================================================
# We explore sensitivity to:
# 1. Transform type: sqrt, log, inverse
# 2. Alpha (α): 0.1, 0.5, 0.9, 1.0 (chronological vs ranked weight)
# 3. Regime emphasis: uniform, low_flow, balanced
# 4. Dynamics strength: 0.5, 0.8

apex_configs = {
    # ---------------------------------------------------------------------
    # EXPERIMENT 1: Transform Type Sensitivity (α=0.1, like SDEB)
    # ---------------------------------------------------------------------
    'sqrt_a01': {
        'description': 'Sqrt transform, α=0.1 (SDEB-like)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'red',
        'group': 'transform'
    },
    'log_a01': {
        'description': 'Log transform, α=0.1',
        'params': {
            'alpha': 0.1,
            'transform': 'log',
            'transform_param': 0.01,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'darkred',
        'group': 'transform'
    },
    'inv_a01': {
        'description': 'Inverse (1/Q) transform, α=0.1',
        'params': {
            'alpha': 0.1,
            'transform': 'inverse',
            'transform_param': 0.01,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'purple',
        'group': 'transform'
    },
    
    # ---------------------------------------------------------------------
    # EXPERIMENT 2: Alpha (α) Sensitivity (sqrt transform)
    # α controls weight between chronological (timing) and ranked (FDC) terms
    # α=0: 100% ranked (FDC), α=1: 100% chronological (timing)
    # ---------------------------------------------------------------------
    'sqrt_a05': {
        'description': 'Sqrt, α=0.5 (equal timing/FDC weight)',
        'params': {
            'alpha': 0.5,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'orangered',
        'group': 'alpha'
    },
    'sqrt_a09': {
        'description': 'Sqrt, α=0.9 (heavy timing weight)',
        'params': {
            'alpha': 0.9,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'coral',
        'group': 'alpha'
    },
    'sqrt_a10': {
        'description': 'Sqrt, α=1.0 (100% chronological)',
        'params': {
            'alpha': 1.0,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'uniform'
        },
        'color': 'salmon',
        'group': 'alpha'
    },
    
    # ---------------------------------------------------------------------
    # EXPERIMENT 3: Regime Emphasis Sensitivity (sqrt, α=0.1, κ=0.8)
    # Controls weighting in ranked term by exceedance probability p
    # p=0 → highest flows, p=1 → lowest flows
    # ---------------------------------------------------------------------
    'sqrt_lowflow': {
        'description': 'Sqrt, low_flow regime: w(p)=p',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'low_flow'  # w(p) = p (emphasize baseflow)
        },
        'color': 'darkgreen',
        'group': 'regime'
    },
    'sqrt_highflow': {
        'description': 'Sqrt, high_flow regime: w(p)=1-p',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'high_flow'  # w(p) = 1-p (emphasize peaks)
        },
        'color': 'darkblue',
        'group': 'regime'
    },
    'sqrt_balanced': {
        'description': 'Sqrt, balanced regime: w(p)=4p(1-p)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'balanced'  # w(p) = 4p(1-p), peaks at median
        },
        'color': 'teal',
        'group': 'regime'
    },
    'sqrt_extremes': {
        'description': 'Sqrt, extremes regime: w(p)=1-4p(1-p)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.8,
            'regime_emphasis': 'extremes'  # w(p) = 1-4p(1-p), emphasize both tails
        },
        'color': 'darkcyan',
        'group': 'regime'
    },
    
    # ---------------------------------------------------------------------
    # EXPERIMENT 4: Dynamics Multiplier Strength Sensitivity (sqrt, α=0.1)
    # The dynamics multiplier penalizes poor gradient correlation
    # κ=0: No dynamics penalty (equivalent to SDEB)
    # κ=0.5: Moderate dynamics penalty
    # κ=0.8: Strong dynamics penalty (default)
    # κ=1.0: Very strong dynamics penalty
    # ---------------------------------------------------------------------
    'sqrt_dyn00': {
        'description': 'Sqrt, NO dynamics (κ=0, SDEB equivalent)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.0,  # NO dynamics penalty
            'regime_emphasis': 'uniform'
        },
        'color': 'gray',
        'group': 'dynamics'
    },
    'sqrt_dyn05': {
        'description': 'Sqrt, moderate dynamics (κ=0.5)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 0.5,  # Moderate dynamics penalty
            'regime_emphasis': 'uniform'
        },
        'color': 'indianred',
        'group': 'dynamics'
    },
    'sqrt_dyn10': {
        'description': 'Sqrt, very strong dynamics (κ=1.0)',
        'params': {
            'alpha': 0.1,
            'transform': 'power',
            'transform_param': 0.5,
            'dynamics_strength': 1.0,  # Very strong dynamics penalty
            'regime_emphasis': 'uniform'
        },
        'color': 'darkmagenta',
        'group': 'dynamics'
    },
}

print(f"\nTotal APEX configurations to calibrate: {len(apex_configs)}")
print("\n" + "-" * 80)
print("EXPERIMENT GROUPS:")
print("-" * 80)

# Group by experiment type
groups = {}
for name, config in apex_configs.items():
    group = config.get('group', 'other')
    if group not in groups:
        groups[group] = []
    groups[group].append(name)

for group, configs in groups.items():
    print(f"\n{group.upper()} SENSITIVITY ({len(configs)} configs):")
    for name in configs:
        config = apex_configs[name]
        p = config['params']
        print(f"  {name}: α={p['alpha']}, {p['transform']}, regime={p['regime_emphasis']}")

# Store results
apex_results = {}
apex_simulations = {}

# %%
# Run calibrations for each APEX configuration
for config_name, config in apex_configs.items():
    print("\n" + "=" * 80)
    print(f"CALIBRATING: APEX_{config_name}")
    print(f"  {config['description']}")
    print("=" * 80)
    
    # Create APEX objective with this configuration
    apex_obj = APEX(
        alpha=config['params']['alpha'],
        transform=config['params']['transform'],
        transform_param=config['params']['transform_param'],
        bias_strength=1.0,
        bias_power=1.0,
        dynamics_strength=config['params']['dynamics_strength'],
        lag_penalty=False,
        regime_emphasis=config['params']['regime_emphasis']
    )
    
    # Create calibration runner
    runner = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=apex_obj,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"\nRunning SCE-UA calibration for APEX_{config_name}...")
    
    # Run calibration
    result = runner.run_sceua_direct(
        max_evals=10000,
        seed=42,
        verbose=True,
        max_tolerant_iter=100,
        tolerance=1e-4
    )
    
    print(f"\n✓ APEX_{config_name} Calibration complete!")
    print(f"  Best objective: {result.best_objective:.4f}")
    print(f"  Runtime: {result.runtime_seconds:.1f} seconds")
    
    # Re-simulate with calibrated parameters
    model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model.set_parameters(result.best_parameters)
    model.reset()
    sim = model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
    
    # Store results
    apex_results[config_name] = {
        'result': result,
        'objective': apex_obj,
        'config': config
    }
    apex_simulations[config_name] = sim
    
    # Save calibration report with unique name
    report = runner.create_report(result, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    save_name = f"410734_APEX_{config_name.replace('(', '_').replace(')', '')}"
    report.save(f'../test_data/reports/{save_name}')
    print(f"  Saved to: test_data/reports/{save_name}.pkl")

# Use the first APEX configuration as the "main" sim_apex for backward compatibility
sim_apex = list(apex_simulations.values())[0]

print("\n" + "=" * 80)
print(f"ALL {len(apex_configs)} APEX CALIBRATIONS COMPLETE")
print("=" * 80)

# %%
# Summary comparison of all configurations
print("\n" + "=" * 80)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("=" * 80)

# Create a DataFrame for easier analysis
comparison_data = []

# Add baseline calibrations
baselines = {
    'NSE_cal': sim_nse,
    'InvNSE_cal': sim_invnse,
    'SDEB_cal': sim_sdeb,
}

for name, sim in baselines.items():
    comparison_data.append({
        'Config': name,
        'Group': 'Baseline',
        'NSE': NSE()(obs_flow, sim),
        'NSE(inv)': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))(obs_flow, sim),
        'NSE(log)': NSE(transform=FlowTransformation('log', epsilon_value=0.01))(obs_flow, sim),
        'NSE(sqrt)': NSE(transform=FlowTransformation('sqrt'))(obs_flow, sim),
        'KGE': KGE(variant='2012')(obs_flow, sim),
        'PBIAS': PBIAS()(obs_flow, sim),
    })

# Add all APEX configurations (prefixed with APEX_)
for config_name, sim in apex_simulations.items():
    config = apex_configs[config_name]
    comparison_data.append({
        'Config': f'APEX_{config_name}',
        'Group': config.get('group', 'other').capitalize(),
        'NSE': NSE()(obs_flow, sim),
        'NSE(inv)': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01))(obs_flow, sim),
        'NSE(log)': NSE(transform=FlowTransformation('log', epsilon_value=0.01))(obs_flow, sim),
        'NSE(sqrt)': NSE(transform=FlowTransformation('sqrt'))(obs_flow, sim),
        'KGE': KGE(variant='2012')(obs_flow, sim),
        'PBIAS': PBIAS()(obs_flow, sim),
    })

# Create DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Display grouped by experiment
print("\n" + "=" * 100)
print("RESULTS BY EXPERIMENT GROUP")
print("=" * 100)

for group in comparison_df['Group'].unique():
    group_df = comparison_df[comparison_df['Group'] == group].copy()
    print(f"\n{group.upper()} ({len(group_df)} configs)")
    print("-" * 100)
    print(group_df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x))

# Find best performers
print("\n" + "=" * 100)
print("BEST PERFORMERS BY METRIC")
print("=" * 100)

metrics_to_maximize = ['NSE', 'NSE(inv)', 'NSE(log)', 'NSE(sqrt)', 'KGE']
metrics_to_minimize = ['PBIAS']

for metric in metrics_to_maximize:
    best_idx = comparison_df[metric].idxmax()
    best_row = comparison_df.loc[best_idx]
    print(f"  Best {metric:12s}: {best_row['Config']:15s} = {best_row[metric]:.4f}")

for metric in metrics_to_minimize:
    best_idx = comparison_df[metric].abs().idxmin()
    best_row = comparison_df.loc[best_idx]
    print(f"  Best {metric:12s}: {best_row['Config']:15s} = {best_row[metric]:.2f}%")

# Check if any APEX config beats SDEB on multiple metrics
print("\n" + "=" * 100)
print("APEX vs SDEB COMPARISON")
print("=" * 100)

sdeb_row = comparison_df[comparison_df['Config'] == 'SDEB_cal'].iloc[0]
apex_df = comparison_df[comparison_df['Group'] != 'Baseline']

print("\nAPEX configs that beat SDEB:")
for metric in metrics_to_maximize:
    sdeb_val = sdeb_row[metric]
    better_configs = apex_df[apex_df[metric] > sdeb_val]['Config'].tolist()
    if better_configs:
        print(f"  {metric}: {', '.join(better_configs)}")
    else:
        print(f"  {metric}: None")

print("\n✓ All configurations calibrated and compared!")

# %% [markdown]
# ---
# # PART 4: PERFORMANCE COMPARISON
#
# Now we evaluate all four calibrations across multiple metrics.
# All calibrations are evaluated against the same observed data for fair comparison.

# %%
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

eval_metrics = {
    'NSE': NSE(),
    'NSE(sqrt)': NSE(transform=FlowTransformation('sqrt')),
    'NSE(log)': NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
    'NSE(inverse)': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'KGE': KGE(variant='2012'),
    'KGE(sqrt)': KGE(variant='2012', transform=FlowTransformation('sqrt')),
    'PBIAS': PBIAS(),
    'RMSE': RMSE(),
    'MAE': MAE(),
    'FDC High': FDCMetric(segment='high'),
    'FDC Mid': FDCMetric(segment='mid'),
    'FDC Low': FDCMetric(segment='low'),
    'Baseflow Index': SignatureMetric('baseflow_index'),
    'Flashiness': SignatureMetric('flashiness'),
    'Pearson r': PearsonCorrelation(),
}

# Evaluate each calibration against the SAME observed data
# This ensures fair comparison across all methods
results_comparison = {}
for name, metric in eval_metrics.items():
    results_comparison[name] = {
        'NSE': metric(obs_flow, sim_nse),
        'InvNSE': metric(obs_flow, sim_invnse),
        'SDEB': metric(obs_flow, sim_sdeb),
    }
    # Add all APEX configurations (with APEX_ prefix)
    for apex_name, apex_sim in apex_simulations.items():
        results_comparison[name][f'APEX_{apex_name}'] = metric(obs_flow, apex_sim)

results_df = pd.DataFrame(results_comparison).T

print("\nPerformance Metrics:")
print("=" * 80)
print(results_df.to_string(float_format=lambda x: f'{x:7.4f}'))

# %%
# Identify best performer for each metric
print("\n" + "=" * 80)
print("BEST PERFORMER PER METRIC")
print("=" * 80)

for metric_name in results_df.index:
    row = results_df.loc[metric_name]
    
    # Skip if all NaN
    if row.isna().all():
        print(f"{metric_name:20s}: N/A (all NaN)")
        continue
    
    # Determine if maximize or minimize
    metric = eval_metrics[metric_name]
    if hasattr(metric, 'direction'):
        is_maximize = (metric.direction == 'maximize')
    elif hasattr(metric, 'maximize'):
        is_maximize = metric.maximize
    else:
        is_maximize = True
    
    if is_maximize:
        best_method = row.idxmax(skipna=True)
        best_value = row.max(skipna=True)
    else:
        best_method = row.idxmin(skipna=True)
        best_value = row.min(skipna=True)
    
    print(f"{metric_name:20s}: {best_method:10s} ({best_value:7.4f})")

# %% [markdown]
# ---
# # PART 5: VISUALIZATION
#
# Interactive visual comparison of the four calibrations using Plotly.
# All plots use the full time series with identical observed data.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("\n" + "=" * 80)
print("INTERACTIVE VISUALIZATIONS (Plotly)")
print("=" * 80)

# %%
# Interactive Hydrograph Comparison with LINKED Log and Linear scales
print("\n1. Interactive Hydrographs (Log + Linear scale, linked zoom)...")

# Define colors for all calibrations
colors = {
    'observed': 'black',
    'nse': 'blue',
    'invnse': 'green', 
    'sdeb': 'orange',
    'apex': 'red',  # For backward compatibility with sim_apex
}
# Add colors from APEX configs
for name, config in apex_configs.items():
    colors[name] = config.get('color', 'red')

# Create 2-row layout: Log scale (top) and Linear scale (bottom)
fig_hydro = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        'Full Hydrograph (Log Scale) - Good for seeing low flows and recession behavior',
        'Full Hydrograph (Linear Scale) - Good for seeing peaks'
    ),
    vertical_spacing=0.18,  # Increased spacing for legend between subplots
    shared_xaxes=True  # CRITICAL: Links zoom across panels
)

# Row 1: LOG SCALE - All calibrations overlaid
# Observed
fig_hydro.add_trace(
    go.Scatter(x=dates, y=obs_flow, mode='lines',
               name='Observed', line=dict(color=colors['observed'], width=2),
               legendgroup='observed', showlegend=True),
    row=1, col=1
)
# Baseline calibrations
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_nse, mode='lines',
               name='NSE', line=dict(color=colors['nse'], width=1),
               legendgroup='nse', showlegend=True),
    row=1, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_invnse, mode='lines',
               name='InvNSE', line=dict(color=colors['invnse'], width=1),
               legendgroup='invnse', showlegend=True),
    row=1, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_sdeb, mode='lines',
               name='SDEB', line=dict(color=colors['sdeb'], width=1),
               legendgroup='sdeb', showlegend=True),
    row=1, col=1
)
# All APEX configurations
for apex_name, apex_sim in apex_simulations.items():
    fig_hydro.add_trace(
        go.Scatter(x=dates, y=apex_sim, mode='lines',
                   name=f'APEX_{apex_name}', line=dict(color=colors.get(apex_name, 'red'), width=1),
                   legendgroup=apex_name, showlegend=True),
        row=1, col=1
    )

# Row 2: LINEAR SCALE - All calibrations overlaid (linked legend)
# Observed
fig_hydro.add_trace(
    go.Scatter(x=dates, y=obs_flow, mode='lines',
               name='Observed', line=dict(color=colors['observed'], width=2),
               legendgroup='observed', showlegend=False),
    row=2, col=1
)
# Baseline calibrations
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_nse, mode='lines',
               name='NSE', line=dict(color=colors['nse'], width=1),
               legendgroup='nse', showlegend=False),
    row=2, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_invnse, mode='lines',
               name='InvNSE', line=dict(color=colors['invnse'], width=1),
               legendgroup='invnse', showlegend=False),
    row=2, col=1
)
fig_hydro.add_trace(
    go.Scatter(x=dates, y=sim_sdeb, mode='lines',
               name='SDEB', line=dict(color=colors['sdeb'], width=1),
               legendgroup='sdeb', showlegend=False),
    row=2, col=1
)
# All APEX configurations
for apex_name, apex_sim in apex_simulations.items():
    fig_hydro.add_trace(
        go.Scatter(x=dates, y=apex_sim, mode='lines',
                   name=f'APEX_{apex_name}', line=dict(color=colors.get(apex_name, 'red'), width=1),
                   legendgroup=apex_name, showlegend=False),
        row=2, col=1
    )

# Update axes
fig_hydro.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig_hydro.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig_hydro.update_xaxes(title_text="Date", row=2, col=1)

fig_hydro.update_layout(
    height=1600,
    title_text="<b>Hydrograph Comparison: Baselines vs APEX Configurations</b><br>" +
               f"<sub>Full simulation period: {dates.min().date()} to {dates.max().date()} ({len(dates):,} days) | Click legend to toggle | Zoom is linked</sub>",
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation='h', 
        yanchor='middle', 
        y=0.52,  # Position between the two subplots
        xanchor='center', 
        x=0.5,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='lightgray',
        borderwidth=1
    )
)

fig_hydro.show()
print("✓ Interactive hydrograph complete (log + linear scales linked)")

# %%
# Interactive Flow Duration Curves
print("\n2. Interactive Flow Duration Curves...")

fig_fdc = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Full Range (Log Scale)', 'Low Flows Detail (70-100% exceedance)'),
    horizontal_spacing=0.12
)

# Prepare FDC data - all simulations compared to same observed
sorted_obs = np.sort(obs_flow)[::-1]
exceedance = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs) * 100

# Sort baseline simulations
nse_sorted = np.sort(sim_nse)[::-1]
invnse_sorted = np.sort(sim_invnse)[::-1]
sdeb_sorted = np.sort(sim_sdeb)[::-1]

# Add observed to both panels
fig_fdc.add_trace(
    go.Scatter(x=exceedance, y=sorted_obs, mode='lines',
               name='Observed', line=dict(color=colors['observed'], width=2),
               legendgroup='obs', showlegend=True),
    row=1, col=1
)
fig_fdc.add_trace(
    go.Scatter(x=exceedance, y=sorted_obs, mode='lines',
               name='Observed', line=dict(color=colors['observed'], width=2),
               legendgroup='obs', showlegend=False),
    row=1, col=2
)

# Add baseline FDCs
baseline_fdc = [
    (nse_sorted, 'NSE', colors['nse']),
    (invnse_sorted, 'InvNSE', colors['invnse']),
    (sdeb_sorted, 'SDEB', colors['sdeb']),
]

for sorted_sim, label, color in baseline_fdc:
    fig_fdc.add_trace(
        go.Scatter(x=exceedance, y=sorted_sim, mode='lines',
                   name=label, line=dict(color=color, width=1.5),
                   legendgroup=label, showlegend=True),
        row=1, col=1
    )
    mask = exceedance >= 70
    fig_fdc.add_trace(
        go.Scatter(x=exceedance[mask], y=sorted_sim[mask], mode='lines',
                   name=label, line=dict(color=color, width=1.5),
                   legendgroup=label, showlegend=False),
        row=1, col=2
    )

# Add all APEX configurations
for apex_name, apex_sim in apex_simulations.items():
    apex_sorted = np.sort(apex_sim)[::-1]
    color = colors.get(apex_name, 'red')
    
    fig_fdc.add_trace(
        go.Scatter(x=exceedance, y=apex_sorted, mode='lines',
                   name=f'APEX_{apex_name}', line=dict(color=color, width=1.5),
                   legendgroup=apex_name, showlegend=True),
        row=1, col=1
    )
    mask = exceedance >= 70
    fig_fdc.add_trace(
        go.Scatter(x=exceedance[mask], y=apex_sorted[mask], mode='lines',
                   name=f'APEX_{apex_name}', line=dict(color=color, width=1.5),
                   legendgroup=apex_name, showlegend=False),
        row=1, col=2
    )

# Update axes
fig_fdc.update_xaxes(title_text="Exceedance Probability (%)", row=1, col=1)
fig_fdc.update_xaxes(title_text="Exceedance Probability (%)", row=1, col=2, range=[70, 100])
fig_fdc.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig_fdc.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)

fig_fdc.update_layout(
    height=800,
    title_text="<b>Flow Duration Curve Comparison: Baselines vs APEX Configurations</b><br>" +
               "<sub>Left: Full range, Right: Low flows detail (both log scale)</sub>",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.18)
)

fig_fdc.show()
print("✓ Interactive FDC complete")

# %%
# Combined comparison plot (like notebook 02) - ALL APEX CONFIGURATIONS
print("\n3. Combined comparison plot (all APEX configs)...")

# Ensure colors dict has all required keys (in case cells ran out of order)
if 'colors' not in dir() or 'observed' not in colors:
    colors = {
        'observed': 'black',
        'nse': 'blue',
        'invnse': 'green', 
        'sdeb': 'orange',
    }

# Distinct red shades for APEX configs (from dark to light)
red_shades = [
    '#8B0000',  # dark red
    '#B22222',  # firebrick
    '#CD5C5C',  # indian red
    '#DC143C',  # crimson
    '#FF0000',  # red
    '#FF4500',  # orange red
    '#FF6347',  # tomato
    '#FA8072',  # salmon
    '#E9967A',  # dark salmon
    '#F08080',  # light coral
    '#FF7F7F',  # light red
    '#FFA07A',  # light salmon
    '#FF69B4',  # hot pink (for visibility)
    '#DB7093',  # pale violet red
    '#C71585',  # medium violet red
]

# Line dash patterns for additional distinction
dash_patterns = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

# Build APEX styles: cycle through colors and dashes
apex_styles = {}
apex_names = list(apex_simulations.keys())
for i, name in enumerate(apex_names):
    apex_styles[name] = {
        'color': red_shades[i % len(red_shades)],
        'dash': dash_patterns[i % len(dash_patterns)],
    }

fig_combined = make_subplots(
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
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 1. Full hydrograph (log scale) - Baseline calibrations
fig_combined.add_trace(go.Scatter(x=dates, y=obs_flow, name='Observed',
              legendgroup='observed', line=dict(color=colors['observed'], width=2)), row=1, col=1)
fig_combined.add_trace(go.Scatter(x=dates, y=sim_nse, name='NSE',
              legendgroup='nse', line=dict(color=colors['nse'], width=1.5)), row=1, col=1)
fig_combined.add_trace(go.Scatter(x=dates, y=sim_invnse, name='InvNSE',
              legendgroup='invnse', line=dict(color=colors['invnse'], width=1.5)), row=1, col=1)
fig_combined.add_trace(go.Scatter(x=dates, y=sim_sdeb, name='SDEB',
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=1.5)), row=1, col=1)

# All APEX configurations with distinct red styles
for apex_name, apex_sim in apex_simulations.items():
    style = apex_styles[apex_name]
    fig_combined.add_trace(go.Scatter(x=dates, y=apex_sim, name=f'APEX_{apex_name}',
                  legendgroup=apex_name, line=dict(color=style['color'], width=1.5, dash=style['dash'])), row=1, col=1)

# 2. Flow Duration Curves - Baseline
fig_combined.add_trace(go.Scatter(x=exceedance, y=sorted_obs, name='Observed',
              legendgroup='observed', line=dict(color=colors['observed'], width=2), showlegend=False), row=1, col=2)
fig_combined.add_trace(go.Scatter(x=exceedance, y=nse_sorted, name='NSE',
              legendgroup='nse', line=dict(color=colors['nse'], width=2), showlegend=False), row=1, col=2)
fig_combined.add_trace(go.Scatter(x=exceedance, y=invnse_sorted, name='InvNSE',
              legendgroup='invnse', line=dict(color=colors['invnse'], width=2), showlegend=False), row=1, col=2)
fig_combined.add_trace(go.Scatter(x=exceedance, y=sdeb_sorted, name='SDEB',
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=2), showlegend=False), row=1, col=2)

# All APEX configurations - FDC
for apex_name, apex_sim in apex_simulations.items():
    style = apex_styles[apex_name]
    apex_sorted = np.sort(apex_sim)[::-1]
    fig_combined.add_trace(go.Scatter(x=exceedance, y=apex_sorted, name=f'APEX_{apex_name}',
                  legendgroup=apex_name, line=dict(color=style['color'], width=2, dash=style['dash']), showlegend=False), row=1, col=2)

# 3. Low flow detail (< 200 ML/day) - Baseline
low_mask = obs_flow < 200
fig_combined.add_trace(go.Scatter(x=dates[low_mask], y=obs_flow[low_mask],
              legendgroup='observed', line=dict(color=colors['observed'], width=2), showlegend=False), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=dates[low_mask], y=sim_nse[low_mask],
              legendgroup='nse', line=dict(color=colors['nse'], width=1.5), showlegend=False), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=dates[low_mask], y=sim_invnse[low_mask],
              legendgroup='invnse', line=dict(color=colors['invnse'], width=1.5), showlegend=False), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=dates[low_mask], y=sim_sdeb[low_mask],
              legendgroup='sdeb', line=dict(color=colors['sdeb'], width=1.5), showlegend=False), row=2, col=1)

# All APEX configurations - Low flow
for apex_name, apex_sim in apex_simulations.items():
    style = apex_styles[apex_name]
    fig_combined.add_trace(go.Scatter(x=dates[low_mask], y=apex_sim[low_mask],
                  legendgroup=apex_name, line=dict(color=style['color'], width=1.5, dash=style['dash']), showlegend=False), row=2, col=1)

# 4. Scatter plot - Baseline
max_flow = max(obs_flow.max(), sim_nse.max(), sim_invnse.max(), sim_sdeb.max(), 
               max(apex_sim.max() for apex_sim in apex_simulations.values()))
fig_combined.add_trace(go.Scatter(x=obs_flow, y=sim_nse, mode='markers', name='NSE',
              legendgroup='nse', marker=dict(color=colors['nse'], size=3, opacity=0.4), showlegend=False), row=2, col=2)
fig_combined.add_trace(go.Scatter(x=obs_flow, y=sim_invnse, mode='markers', name='InvNSE',
              legendgroup='invnse', marker=dict(color=colors['invnse'], size=3, opacity=0.4), showlegend=False), row=2, col=2)
fig_combined.add_trace(go.Scatter(x=obs_flow, y=sim_sdeb, mode='markers', name='SDEB',
              legendgroup='sdeb', marker=dict(color=colors['sdeb'], size=3, opacity=0.4), showlegend=False), row=2, col=2)

# All APEX configurations - Scatter (use different marker symbols)
marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                  'pentagon', 'hexagon', 'star', 'hexagram', 'star-triangle-up', 'star-square', 'circle-open', 'square-open']
for i, (apex_name, apex_sim) in enumerate(apex_simulations.items()):
    style = apex_styles[apex_name]
    fig_combined.add_trace(go.Scatter(x=obs_flow, y=apex_sim, mode='markers', name=f'APEX_{apex_name}',
                  legendgroup=apex_name, marker=dict(color=style['color'], size=3, opacity=0.5, 
                  symbol=marker_symbols[i % len(marker_symbols)]), showlegend=False), row=2, col=2)

# 1:1 line
fig_combined.add_trace(go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines',
              line=dict(color='gray', dash='dash', width=2), showlegend=False), row=2, col=2)

# Update axes
fig_combined.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=1)
fig_combined.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
fig_combined.update_xaxes(title_text="Exceedance (%)", row=1, col=2)
fig_combined.update_yaxes(title_text="Flow (ML/day)", row=2, col=1)
fig_combined.update_yaxes(title_text="Simulated (ML/day)", type="log", row=2, col=2)
fig_combined.update_xaxes(title_text="Observed (ML/day)", type="log", row=2, col=2)

fig_combined.update_layout(
    title="<b>Comparison: NSE vs InvNSE vs SDEB vs All APEX Configurations</b><br>" +
          "<sup>Click legend items to toggle visibility across all subplots | APEX configs in red shades with distinct line styles</sup>",
    height=1100,
    width=1800,
    legend=dict(
        orientation='h', 
        y=1.02, 
        x=0.5, 
        xanchor='center',
        font=dict(size=11),
        itemwidth=50,
        traceorder='normal'
    ),
    xaxis=dict(matches='x'),
    xaxis3=dict(matches='x'),
    font=dict(size=12)
)
fig_combined.show()
print(f"✓ Combined comparison complete ({len(apex_simulations)} APEX configs plotted)")

# %%
# Performance summary table (color-coded)
def get_cell_color(value, metric_name, all_values, eval_metrics):
    """Get color for a cell based on metric type and performance."""
    if pd.isna(value):
        return 'rgb(240, 240, 240)'  # Light gray for NaN
    
    # Determine metric direction from eval_metrics if available
    if metric_name in eval_metrics:
        metric_obj = eval_metrics[metric_name]
        if hasattr(metric_obj, 'direction'):
            is_maximize = (metric_obj.direction == 'maximize')
        elif hasattr(metric_obj, 'maximize'):
            is_maximize = metric_obj.maximize
        else:
            is_maximize = True
    else:
        # Default categorization
        # Higher is better metrics
        higher_better = ['NSE', 'KGE', 'NSE(sqrt)', 'NSE(log)', 'NSE(inverse)', 
                         'KGE(sqrt)', 'Pearson r', 'FDC High', 'FDC Mid', 'FDC Low',
                         'Baseflow Index', 'Flashiness']
        # Lower is better metrics
        lower_better = ['RMSE', 'MAE', 'PBIAS']
        
        is_maximize = metric_name in higher_better or metric_name not in lower_better
    
    # Calculate score based on direction
    min_val = all_values.min()
    max_val = all_values.max()
    
    if max_val != min_val:
        if is_maximize:
            score = (value - min_val) / (max_val - min_val)
        else:
            # For minimize metrics, invert the score
            score = (max_val - value) / (max_val - min_val)
    else:
        score = 0.5
    
    # Color from red (0) to green (1) via yellow (0.5)
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

for metric in results_df.index:
    row_data = [metric]
    row_colors = ['white']  # Header column
    
    for col in results_df.columns:
        value = results_df.loc[metric, col]
        color = get_cell_color(value, metric, results_df.loc[metric], eval_metrics)
        row_colors.append(color)
        
        # Format value
        if pd.isna(value):
            row_data.append('N/A')
        elif abs(value) >= 1000:
            row_data.append(f'{value:.2f}')
        elif 'PBIAS' in metric or 'Error' in metric or 'Bias' in metric:
            row_data.append(f'{value:+.3f}')
        else:
            row_data.append(f'{value:.4f}')
    
    table_data.append(row_data)
    cell_colors.append(row_colors)

# Create header
header = ['Metric'] + list(results_df.columns)

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
    title="<b>APEX Calibration Performance Comparison (Color-Coded)</b><br>" +
          "<sup>Green = best performance, Red = worst performance for each metric</sup>",
    height=600,
    width=2000
)

fig_table.show()

print("✓ Performance comparison table complete")

# %% [markdown]
# ---
# # SUMMARY
#
# ## Key Findings
#
# 1. **NSE (Traditional)**:
#    - Optimizes overall fit with high flow bias
#    - Best for flood prediction applications
#
# 2. **InvNSE (Low Flow Focus)**:
#    - Inverse transformation heavily emphasizes low flows
#    - May sacrifice high flow performance
#
# 3. **SDEB (Lerat et al., 2013)**:
#    - Balanced composite approach
#    - Combines chronological errors, ranked errors, and bias penalty
#    - Good all-around performance
#
# 4. **APEX (SDEB-Enhanced with Dynamics Multiplier)**:
#    - Builds on SDEB's proven multiplicative structure
#    - **Dynamics Multiplier** - penalizes gradient/rate-of-change mismatch
#    - **Optional Lag Multiplier** - penalizes systematic timing offsets
#    - No normalization issues (all terms in same error space)
#    - Check `gradient_correlation` in components for dynamics diagnostic
#
# ## APEX Key Features
#
# | Component | Formula | Purpose |
# |-----------|---------|---------|
# | Dynamics Multiplier | `1 + κ × (1 - ρ_gradient)` | Penalizes recession/dynamics mismatch |
# | Lag Multiplier | `1 + λ × \|lag\| / τ` | Penalizes systematic timing offsets |
# | Regime Weighting | `w(p)` in ranked term | Continuous emphasis on flow regimes |
#
# ## Example APEX Configurations
#
# ```python
# # Default: SDEB-equivalent with dynamics
# apex = APEX()
#
# # Low flow emphasis
# apex_low = APEX(transform='log', regime_emphasis='low_flow')
#
# # High flow / flood focus
# apex_flood = APEX(transform_param=0.7, regime_emphasis='high_flow', dynamics_strength=0.7)
#
# # With lag penalty for timing-critical applications
# apex_timing = APEX(lag_penalty=True, lag_strength=0.5)
#
# # SDEB-equivalent (disable dynamics)
# apex_sdeb = APEX(dynamics_strength=0.0)
# ```
#
# ## Next Steps
#
# - Compare APEX performance across different configurations
# - Evaluate dynamics multiplier impact (`gradient_correlation` diagnostic)
# - Test lag multiplier for flood forecasting applications
# - Compare APEX vs SDEB on different catchment types
#
# ## References
#
# - Lerat et al. (2013). A robust approach for calibrating continuous hydrological models.
#   Journal of Hydrology, 494, 80-91.
#
# %%
print("\n" + "=" * 80)
print("NOTEBOOK COMPLETE!")
print("=" * 80)
print("\nAll calibrations compared successfully.")
print(f"Results saved to: test_data/reports/410734_apex.pkl")
print("Figures saved to: figures/apex_*.png")
print("\nAPEX key features demonstrated:")
print("  - Dynamics multiplier (gradient correlation)")
print("  - Component breakdown diagnostics")
print("  - Fair comparison with NSE, InvNSE, and SDEB baselines")
print(f"\nData consistency: All {len(obs_flow):,} days compared using identical observed data")
