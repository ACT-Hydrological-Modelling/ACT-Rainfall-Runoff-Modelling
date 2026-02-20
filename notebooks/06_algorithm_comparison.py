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
# # Algorithm Comparison: PyDREAM vs SCE-UA
#
# ## Purpose
#
# This notebook provides a comprehensive comparison between two fundamentally different
# calibration paradigms:
#
# - **SCE-UA (Shuffled Complex Evolution)**: Optimization-based, provides point estimates
# - **PyDREAM (MT-DREAM(ZS))**: MCMC-based, provides full posterior distributions
#
# We run calibrations across **13 different objective functions** and compare:
# - Parameter estimates (point vs posterior)
# - Model performance metrics
# - Computational requirements
# - Uncertainty quantification
#
# ## What You'll Learn
#
# - How to run PyDREAM calibrations for multiple objective functions
# - How to compare MCMC posteriors with optimization point estimates
# - How objective function choice affects both algorithms
# - Practical insights on when to use each approach
#
# ## Prerequisites
#
# - Completed **Notebook 02: Calibration Quickstart** (provides SCE-UA results)
# - Understanding of MCMC concepts (recommended)
#
# ## Estimated Time
#
# - ~1-2 hours for PyDREAM calibrations (can be run overnight)
# - ~10-20 minutes for analysis and comparison
#
# ## Key Insight
#
# > **SCE-UA finds the peak of the likelihood surface.**
# > **PyDREAM maps the entire posterior landscape.**

# %% [markdown]
# ---
# ## The 13 Objective Functions
#
# We compare calibrations using the following objective functions:
#
# | # | Objective | Type | Flow Emphasis |
# |---|-----------|------|---------------|
# | 1 | NSE | Standard | High flows |
# | 2 | LogNSE | Transformed | All flows |
# | 3 | InvNSE (1/Q) | Transformed | Low flows |
# | 4 | SqrtNSE (√Q) | Transformed | Balanced |
# | 5 | SDEB | Composite | Balanced + FDC |
# | 6 | KGE | Standard | High flows |
# | 7 | KGE(1/Q) | Transformed | Low flows |
# | 8 | KGE(√Q) | Transformed | Balanced |
# | 9 | KGE(log) | Transformed | All flows |
# | 10 | KGE_np | Non-parametric | Robust |
# | 11 | KGE_np(1/Q) | Non-parametric | Low flows |
# | 12 | KGE_np(√Q) | Non-parametric | Balanced |
# | 13 | KGE_np(log) | Non-parametric | All flows |

# %% [markdown]
# ---
# ## Setup

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
import pickle

# Interactive visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("ALGORITHM COMPARISON: PyDREAM vs SCE-UA")
print("=" * 70)
print("\nLibraries loaded successfully!")

# %%
# Import pyrrm components
from pyrrm.models.sacramento import Sacramento
from pyrrm.calibration import (
    CalibrationRunner, 
    CalibrationResult,
    CalibrationReport,
    PYDREAM_AVAILABLE
)
from pyrrm.calibration.objective_functions import calculate_metrics, GaussianLikelihood, TransformedGaussianLikelihood
from pyrrm.objectives import (
    NSE, KGE, KGENonParametric, FlowTransformation, SDEB
)

print("\npyrrm components imported!")
print(f"\nAvailable calibration backends:")
print(f"  SCE-UA (direct, vendored): always available")
print(f"  PyDREAM (MT-DREAM(ZS)): {PYDREAM_AVAILABLE}")

if not PYDREAM_AVAILABLE:
    print("\nWARNING: PyDREAM not installed!")
    print("Install with: pip install pydream")
    print("This notebook requires PyDREAM for the algorithm comparison.")

# Create figures directory for saving plots
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)
print(f"\nFigures will be saved to: {figures_dir.absolute()}")

# %% [markdown]
# ---
# ## Load SCE-UA Results from Notebook 02
#
# First, we load all the calibration results from the SCE-UA runs in Notebook 02.
# These are stored as `CalibrationReport` pickle files.

# %%
# Define report file paths
REPORTS_DIR = Path('../test_data/reports')

# Map of objective function names to report files
SCEUA_REPORTS = {
    'NSE': '410734_nse.pkl',
    'LogNSE': '410734_lognse.pkl',
    'InvNSE': '410734_invnse.pkl',
    'SqrtNSE': '410734_sqrtnse.pkl',
    'SDEB': '410734_sdeb.pkl',
    'KGE': '410734_kge.pkl',
    'KGE_inv': '410734_kge_inv.pkl',
    'KGE_sqrt': '410734_kge_sqrt.pkl',
    'KGE_log': '410734_kge_log.pkl',
    'KGE_np': '410734_kge_np.pkl',
    'KGE_np_inv': '410734_kge_np_inv.pkl',
    'KGE_np_sqrt': '410734_kge_np_sqrt.pkl',
    'KGE_np_log': '410734_kge_np_log.pkl',
}

# Load all SCE-UA reports
sceua_results = {}
sceua_reports = {}

print("=" * 70)
print("LOADING SCE-UA RESULTS FROM NOTEBOOK 02")
print("=" * 70)

for name, filename in SCEUA_REPORTS.items():
    filepath = REPORTS_DIR / filename
    if filepath.exists():
        report = CalibrationReport.load(str(filepath))
        sceua_reports[name] = report
        sceua_results[name] = report.result
        print(f"  ✓ {name:<12}: Loaded ({filename})")
    else:
        print(f"  ✗ {name:<12}: File not found ({filename})")
        print(f"    Run Notebook 02 first to generate this calibration.")

print(f"\nLoaded {len(sceua_results)}/13 SCE-UA calibrations")

# %% [markdown]
# ---
# ## Prepare Calibration Data
#
# We need the same data used in Notebook 02 for the PyDREAM calibrations.

# %%
# Load calibration data from files
# Note: We need full data (including warmup) for running PyDREAM calibrations
DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667
WARMUP_DAYS = 365

# Load rainfall
rainfall_df = pd.read_csv(DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
                          parse_dates=['Date'], index_col='Date')
rainfall_df.columns = ['rainfall']

# Load PET
pet_df = pd.read_csv(DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
                     parse_dates=['Date'], index_col='Date')
pet_df.columns = ['pet']

# Load observed flow
flow_df = pd.read_csv(DATA_DIR / '410734_output_SDmodel.csv',
                      parse_dates=['Date'], index_col='Date')
observed_col = 'Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)'
observed_df = flow_df[[observed_col]].copy()
observed_df.columns = ['observed_flow']
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df = observed_df.dropna()

# Merge datasets
data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')
cal_inputs = data[['rainfall', 'pet']].copy()
cal_observed = data['observed_flow'].values

print("=" * 50)
print("CALIBRATION DATA")
print("=" * 50)
print(f"\nRecords: {len(cal_inputs):,} days")
print(f"Period: {cal_inputs.index.min().date()} to {cal_inputs.index.max().date()}")
print(f"Warmup: {WARMUP_DAYS} days")
print(f"Effective calibration: {len(cal_inputs) - WARMUP_DAYS:,} days")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")

# %% [markdown]
# ---
# ## Define Objective Functions
#
# We define all 13 objective functions matching those used in Notebook 02.

# %%
# Define all objective functions
objectives = {
    # NSE-based
    'NSE': NSE(),
    'LogNSE': NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
    'InvNSE': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'SqrtNSE': NSE(transform=FlowTransformation('sqrt')),
    
    # Composite
    'SDEB': SDEB(alpha=0.1, lam=0.5),
    
    # KGE-based
    'KGE': KGE(variant='2012'),
    'KGE_inv': KGE(variant='2012', transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'KGE_sqrt': KGE(variant='2012', transform=FlowTransformation('sqrt')),
    'KGE_log': KGE(variant='2012', transform=FlowTransformation('log', epsilon_value=0.01)),
    
    # Non-parametric KGE
    'KGE_np': KGENonParametric(),
    'KGE_np_inv': KGENonParametric(transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'KGE_np_sqrt': KGENonParametric(transform=FlowTransformation('sqrt')),
    'KGE_np_log': KGENonParametric(transform=FlowTransformation('log', epsilon_value=0.01)),
}

print("=" * 70)
print("OBJECTIVE FUNCTIONS DEFINED")
print("=" * 70)
print(f"\nTotal: {len(objectives)} objective functions")
print("\nNSE-based (4):")
for name in ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE']:
    print(f"  - {name}")
print("\nComposite (1):")
print(f"  - SDEB")
print("\nKGE-based (4):")
for name in ['KGE', 'KGE_inv', 'KGE_sqrt', 'KGE_log']:
    print(f"  - {name}")
print("\nNon-parametric KGE (4):")
for name in ['KGE_np', 'KGE_np_inv', 'KGE_np_sqrt', 'KGE_np_log']:
    print(f"  - {name}")

# %% [markdown]
# ### Likelihood Transform Mapping for PyDREAM
#
# PyDREAM requires a proper log-likelihood function, not an efficiency metric.
# To ensure fair comparison with SCE-UA results, we use `TransformedGaussianLikelihood`
# with transforms that match the flow emphasis of each objective function:
#
# | Objective | Transform | Flow Emphasis |
# |-----------|-----------|---------------|
# | NSE, KGE, KGE_np | 'none' | High flows |
# | SqrtNSE, KGE_sqrt, KGE_np_sqrt | 'sqrt' | Balanced |
# | LogNSE, KGE_log, KGE_np_log | 'log' | Low flows |
# | InvNSE, KGE_inv, KGE_np_inv | 'inverse' | Very low flows |
# | SDEB | 'sqrt' | Balanced (default) |

# %%
# Mapping from objective function names to equivalent likelihood transforms
# This ensures PyDREAM calibration uses the same flow emphasis as the objective
LIKELIHOOD_TRANSFORM_MAPPING = {
    # NSE-based - transform matches the NSE variant
    'NSE': 'none',           # Standard NSE → no transform (high-flow emphasis)
    'LogNSE': 'log',         # Log-transformed NSE → log transform (low-flow emphasis)
    'InvNSE': 'inverse',     # Inverse NSE → inverse transform (very low-flow emphasis)
    'SqrtNSE': 'sqrt',       # Sqrt NSE → sqrt transform (balanced)
    
    # Composite objectives - use balanced transform
    'SDEB': 'sqrt',          # SDEB is balanced → sqrt transform
    
    # Standard KGE - same mapping as NSE
    'KGE': 'none',           # Standard KGE → no transform
    'KGE_inv': 'inverse',    # Inverse KGE → inverse transform
    'KGE_sqrt': 'sqrt',      # Sqrt KGE → sqrt transform
    'KGE_log': 'log',        # Log KGE → log transform
    
    # Non-parametric KGE - same mapping
    'KGE_np': 'none',
    'KGE_np_inv': 'inverse',
    'KGE_np_sqrt': 'sqrt',
    'KGE_np_log': 'log',
}

print("\n" + "=" * 70)
print("LIKELIHOOD TRANSFORM MAPPING FOR PYDREAM")
print("=" * 70)
print("\nEach objective function will use its equivalent log-likelihood:")
for obj_name, transform in LIKELIHOOD_TRANSFORM_MAPPING.items():
    emphasis = {
        'none': 'high flows',
        'sqrt': 'balanced',
        'log': 'low flows',
        'inverse': 'very low flows'
    }.get(transform, transform)
    print(f"  {obj_name:<15} → TransformedGaussianLikelihood('{transform}') [{emphasis}]")

# %% [markdown]
# ---
# ## PyDREAM Calibration Configuration
#
# PyDREAM uses MCMC (Markov Chain Monte Carlo) to sample from the posterior
# distribution of parameters. Key settings:
#
# - **niterations**: Number of iterations per chain (more = better convergence)
# - **nchains**: Number of parallel MCMC chains (minimum: 3, recommended: 5-8)
# - **multitry**: Multi-try proposals for better mixing
# - **snooker**: Snooker updates for escaping local modes
#
# For production use, increase iterations significantly (5000-10000+).

# %%
# PyDREAM configuration - MATCHING SYNTHETIC TEST SETTINGS
# Settings chosen to avoid PyDREAM NumPy bugs:
# - adapt_crossover=False (causes NumPy errors)
# - adapt_gamma=False (causes NumPy errors)
# - multitry=1 (standard DREAM, no multi-try)
# Same hyperparameters as synthetic test for consistency

PYDREAM_ITERATIONS = 1500    # Iterations per chain (increased for better convergence)
PYDREAM_CHAINS = 3           # 3 chains (minimum for DEpairs=1: 2*1+1=3)
PYDREAM_MULTITRY = 1         # Standard DREAM (no multi-try)
PYDREAM_DEPAIRS = 1          # 1 DE pair
PYDREAM_SNOOKER = 0.10       # 10% snooker probability
PYDREAM_ADAPT_GAMMA = False  # Disabled - causes PyDREAM NumPy bugs
PYDREAM_ADAPT_CROSSOVER = False  # Disabled - causes PyDREAM NumPy bugs
PYDREAM_CONVERGENCE_THRESHOLD = 1.05  # Stricter R-hat threshold

print("PyDREAM Configuration (Matching Synthetic Test):")
print(f"  Iterations per chain: {PYDREAM_ITERATIONS}")
print(f"  Number of chains: {PYDREAM_CHAINS} (min: {2*PYDREAM_DEPAIRS+1} for DEpairs={PYDREAM_DEPAIRS})")
print(f"  Multi-try samples: {PYDREAM_MULTITRY}")
print(f"  DE pairs: {PYDREAM_DEPAIRS}")
print(f"  Snooker probability: {PYDREAM_SNOOKER:.0%}")
print(f"  Adaptive gamma: {PYDREAM_ADAPT_GAMMA}")
print(f"  Adaptive crossover: {PYDREAM_ADAPT_CROSSOVER}")
print(f"  Convergence threshold: GR < {PYDREAM_CONVERGENCE_THRESHOLD}")
print(f"  Total samples: ~{PYDREAM_ITERATIONS * PYDREAM_CHAINS:,}")

# %% [markdown]
# ---
# ## Synthetic Hydrograph Validation
#
# Before running full calibrations on real data, let's validate that both
# SCE-UA and PyDREAM work correctly using a **synthetic hydrograph**.
#
# We'll:
# 1. Define "true" Sacramento model parameters
# 2. Generate synthetic rainfall/PET forcing data
# 3. Run the model to create a synthetic "observed" hydrograph (with noise)
# 4. Calibrate with both algorithms to recover the true parameters
#
# This tests the full calibration pipeline with realistic hydrological data.

# %%
print("=" * 70)
print("SYNTHETIC HYDROGRAPH VALIDATION")
print("=" * 70)
print("\nGenerating synthetic hydrograph with known Sacramento parameters...")

# Sacramento parameters tuned for realistic sustained baseflow
# Key changes from default:
# - Very large lower zone free water stores for sustained baseflow
# - Very slow drainage rates to maintain flow during extended dry periods
# - High percolation to groundwater to keep stores recharged
TRUE_SAC_PARAMS = {
    'uztwm': 50.0,    # Upper zone tension water max (mm) - moderate
    'uzfwm': 40.0,    # Upper zone free water max (mm) - for interflow
    'lztwm': 200.0,   # Lower zone tension water max (mm) - large
    'lzfpm': 400.0,   # Lower zone free water PRIMARY max (mm) - VERY LARGE for deep baseflow
    'lzfsm': 150.0,   # Lower zone free water SUPPLEMENTAL max (mm) - large for shallow baseflow
    'uzk': 0.25,      # Upper zone lateral drainage rate (1/day)
    'lzpk': 0.004,    # Lower zone PRIMARY drainage rate (1/day) - VERY SLOW (~250 day recession)
    'lzsk': 0.025,    # Lower zone SUPPLEMENTAL drainage rate (1/day) - slow (~40 day recession)
    'zperc': 250.0,   # Maximum percolation rate - HIGH to recharge groundwater
    'rexp': 1.5,      # Percolation equation exponent - lower for more consistent percolation
    'pfree': 0.50,    # Fraction of percolation to free water - HIGH (50% goes to groundwater)
    'pctim': 0.01,    # Impervious fraction
    'apts': 0.0,      # Active area adjustment
    'riva': 0.0,      # Riparian vegetation area
    'side': 0.0,      # Ratio of deep recharge to streamflow
    'ssout': 0.0,     # Groundwater outflow (none - all goes to stream)
    'uh1': 0.5,       # Unit hydrograph ordinates - slightly smoothed
    'uh2': 0.3,
    'uh3': 0.15,
    'uh4': 0.05,
    'uh5': 0.0,
    'rserv': 0.3,     # Fraction of lower zone free water not transferable
}

# Generate realistic synthetic forcing data (8 years: 2 warmup + 6 calibration)
# Using stochastic weather generator with wet/dry spell persistence
np.random.seed(42)
n_years = 8
n_days = n_years * 365
dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
day_of_year = np.arange(n_days) % 365

print("\nGenerating realistic stochastic forcing data...")

# ============================================================================
# RAINFALL: Markov chain with extended dry spells for realistic recessions
# ============================================================================
# Key features:
# - Wet/dry spell persistence via Markov chain
# - Extended dry spells (2-6 weeks) to create long recession events
# - Seasonal variation (ACT climate: wetter in winter/spring)
# - Multi-day storm events

rainfall = np.zeros(n_days)
is_wet = False  # Start dry
dry_spell_counter = 0  # Track days in current dry spell
forced_dry_days = 0  # Remaining days of forced dry spell

for i in range(n_days):
    doy = day_of_year[i]
    
    # If in a forced dry spell, skip rainfall
    if forced_dry_days > 0:
        forced_dry_days -= 1
        is_wet = False
        continue
    
    # Seasonal transition probabilities (ACT: cool-season dominant rainfall)
    # Winter/Spring (Jun-Nov, doy 150-330): wetter
    # Summer/Autumn (Dec-May): drier with occasional storms
    if 150 <= doy <= 330:  # Cool season
        p_wet_given_wet = 0.60  # Wet spells persist
        p_wet_given_dry = 0.25  # Lower to allow dry spells
        mean_rain = 8.0
        shape = 0.7
    else:  # Warm season  
        p_wet_given_wet = 0.40  # Less persistent
        p_wet_given_dry = 0.12  # Much lower - longer dry spells
        mean_rain = 15.0  # But when it does, can be intense
        shape = 0.5
    
    # Markov chain transition
    if is_wet:
        is_wet = np.random.random() < p_wet_given_wet
    else:
        is_wet = np.random.random() < p_wet_given_dry
    
    # Generate rainfall amount if wet day
    if is_wet:
        scale = mean_rain / shape
        rainfall[i] = np.random.gamma(shape, scale)
        dry_spell_counter = 0
    else:
        dry_spell_counter += 1

# ============================================================================
# Insert extended dry spells (2-6 weeks) for long recession events
# ============================================================================
# Add 3-5 extended dry spells per year to create pronounced recessions
n_dry_spells = np.random.poisson(n_years * 4)
print(f"  Inserting {n_dry_spells} extended dry spells...")

for _ in range(n_dry_spells):
    # Random start day (avoid first year warmup for plotting purposes)
    start_day = np.random.randint(100, n_days - 60)
    
    # Dry spell duration: 14-45 days (2-6 weeks)
    duration = np.random.randint(14, 46)
    
    # Set rainfall to zero for the dry spell
    end_day = min(start_day + duration, n_days)
    rainfall[start_day:end_day] = 0

# Add occasional extreme events (1-2 per year on average)
n_extremes = np.random.poisson(n_years * 1.5)
extreme_days = np.random.choice(n_days, n_extremes, replace=False)
for day in extreme_days:
    rainfall[day] += np.random.exponential(35)  # Add 35mm+ on top

# Add multi-day storm events (rainfall persistence for 2-4 days)
for i in range(1, n_days):
    if rainfall[i-1] > 20 and rainfall[i] == 0 and np.random.random() < 0.5:
        # Continuing storm event
        rainfall[i] = rainfall[i-1] * np.random.uniform(0.3, 0.7)

# ============================================================================
# PET: Seasonal pattern with daily variability (cloud cover, weather)
# ============================================================================
# ACT: Summer PET ~6-7 mm/day, Winter PET ~1-2 mm/day
pet_seasonal = 3.5 + 2.5 * np.cos(2 * np.pi * (day_of_year - 355) / 365)  # Peak in early Jan

# Add weather-related variability (cloudy days have lower PET)
pet_noise = np.random.normal(0, 0.5, n_days)
# Cloudy/rainy days have reduced PET
cloud_reduction = np.where(rainfall > 5, -1.0, np.where(rainfall > 0, -0.3, 0))
pet = np.maximum(pet_seasonal + pet_noise + cloud_reduction, 0.3)

# Add some autocorrelation (weather persistence)
for i in range(1, n_days):
    pet[i] = 0.7 * pet[i] + 0.3 * pet[i-1]

# ============================================================================
# Add inter-annual variability (wet years, dry years)
# ============================================================================
# Create annual multipliers
annual_factors = np.random.lognormal(0, 0.25, n_years)  # ~±25% variability
for year in range(n_years):
    start_idx = year * 365
    end_idx = (year + 1) * 365
    rainfall[start_idx:end_idx] *= annual_factors[year]

# Create input DataFrame
synthetic_inputs = pd.DataFrame({
    'rainfall': rainfall,
    'pet': pet
}, index=dates)

# Summary statistics
wet_days = np.sum(rainfall > 0)
mean_wet_day = np.mean(rainfall[rainfall > 0])
max_rain = np.max(rainfall)

print(f"\nSynthetic forcing data generated:")
print(f"  Period: {dates[0].date()} to {dates[-1].date()} ({n_days} days, {n_years} years)")
print(f"  Total rainfall: {rainfall.sum():.0f} mm")
print(f"  Mean annual rainfall: {rainfall.sum()/n_years:.0f} mm/year")
print(f"  Wet days: {wet_days} ({100*wet_days/n_days:.1f}%)")
print(f"  Mean wet-day rainfall: {mean_wet_day:.1f} mm")
print(f"  Max daily rainfall: {max_rain:.1f} mm")
print(f"  Mean daily PET: {pet.mean():.2f} mm/day")
print(f"  Annual variability: {annual_factors.round(2)}")

# %%
# Run Sacramento model with true parameters to generate synthetic "observed" flow
print("\nRunning Sacramento model with true parameters...")

# Create model with true parameters
SYNTHETIC_CATCHMENT_AREA = 100.0  # km²
true_model = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
true_model.set_parameters(TRUE_SAC_PARAMS)
true_model.reset()

# Run model
true_output = true_model.run(synthetic_inputs)
true_flow = true_output['runoff'].values

print(f"  Flow range: {true_flow.min():.1f} - {true_flow.max():.1f} ML/day")
print(f"  Mean flow: {true_flow.mean():.1f} ML/day")

# ============================================================================
# Add realistic observation noise (rating curve uncertainty)
# ============================================================================
# Real streamflow measurements have:
# 1. Heteroscedastic errors (larger absolute errors at high flows)
# 2. Rating curve uncertainty (~5-15% at low flows, ~10-20% at high flows)
# 3. Some temporal autocorrelation in errors
# 4. Occasional measurement outliers

print("\nAdding realistic observation noise...")

# Base noise: proportional to flow magnitude (rating curve uncertainty)
# Higher relative uncertainty at low flows, lower at high flows
relative_error = 0.08 + 0.05 * np.exp(-true_flow / 50)  # 8-13% depending on flow
noise_std = relative_error * true_flow + 0.5  # Plus small constant for very low flows

# Generate noise with some autocorrelation (measurement errors persist slightly)
raw_noise = np.random.normal(0, 1, n_days)
autocorr_noise = np.zeros(n_days)
autocorr_noise[0] = raw_noise[0]
for i in range(1, n_days):
    autocorr_noise[i] = 0.3 * autocorr_noise[i-1] + 0.7 * raw_noise[i]

noise = autocorr_noise * noise_std

# Add occasional outliers (measurement errors, ~1% of days)
outlier_mask = np.random.random(n_days) < 0.01
outlier_magnitude = np.random.choice([-1, 1], n_days) * np.random.exponential(true_flow * 0.3, n_days)
noise[outlier_mask] += outlier_magnitude[outlier_mask]

# Apply noise and ensure positive flow
synthetic_observed = np.maximum(true_flow + noise, 0.1)

# Calculate noise statistics
nse_true_vs_obs = 1 - np.sum((synthetic_observed - true_flow)**2) / np.sum((true_flow - true_flow.mean())**2)
print(f"  Mean true flow: {true_flow.mean():.2f} ML/day")
print(f"  Mean observed (with noise): {synthetic_observed.mean():.2f} ML/day")
print(f"  NSE(true vs observed): {nse_true_vs_obs:.4f}")
print(f"  (This is the 'ceiling' - best achievable NSE due to observation noise)")

# Plot synthetic hydrograph (showing post-warmup period only for calibration view)
# Use 2-year warmup for plotting as well to show clean data
PLOT_WARMUP = 730  # 2 year warmup for visualization

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# Get post-warmup indices
plot_dates = dates[PLOT_WARMUP:]
plot_rainfall = rainfall[PLOT_WARMUP:]
plot_pet = pet[PLOT_WARMUP:]
plot_observed = synthetic_observed[PLOT_WARMUP:]
plot_true = true_flow[PLOT_WARMUP:]

# Rainfall
ax = axes[0]
ax.bar(plot_dates, plot_rainfall, color='steelblue', alpha=0.7, width=1)
ax.set_ylabel('Rainfall (mm)')
ax.set_title(f'Synthetic Forcing Data and Hydrograph (Post-warmup: {n_years-1} years)')
ax.invert_yaxis()

# PET
ax = axes[1]
ax.plot(plot_dates, plot_pet, color='orange', linewidth=0.8)
ax.set_ylabel('PET (mm/day)')
ax.fill_between(plot_dates, 0, plot_pet, alpha=0.3, color='orange')

# Flow
ax = axes[2]
ax.plot(plot_dates, plot_observed, 'b-', linewidth=0.8, alpha=0.7, label='Observed (noisy)')
ax.plot(plot_dates, plot_true, 'r-', linewidth=1.5, label='True flow')
ax.set_ylabel('Flow (ML/day)')
ax.set_xlabel('Date')
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(figures_dir / '05_synthetic_hydrograph.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved: {figures_dir / '05_synthetic_hydrograph.png'}")
print(f"Note: Showing post-warmup period only ({n_years-1} years)")
print(f"      First year used as warmup to stabilize model states")

# %% [markdown]
# ### Calibration Setup for Synthetic Test
#
# For faster testing, we'll only calibrate a subset of the most sensitive
# Sacramento parameters. The full model has 22 parameters, but we'll focus
# on the 6 most important ones for this validation.

# %%
# Define subset of parameters to calibrate (for faster testing)
# Includes key baseflow parameters: lzfpm (storage), lzpk (recession rate), pfree (recharge)
CALIBRATION_PARAMS = ['uztwm', 'uzfwm', 'lzfpm', 'uzk', 'lzpk', 'pfree']

# Get parameter bounds for calibration subset
full_bounds = Sacramento().get_parameter_bounds()
subset_bounds = {p: full_bounds[p] for p in CALIBRATION_PARAMS}

print("=" * 70)
print("CALIBRATION SETUP (SYNTHETIC DATA)")
print("=" * 70)
print(f"\nCalibrating {len(CALIBRATION_PARAMS)} parameters (subset for speed):")
for p in CALIBRATION_PARAMS:
    bounds = subset_bounds[p]
    true_val = TRUE_SAC_PARAMS[p]
    print(f"  {p:8s}: bounds [{bounds[0]:>8.3f}, {bounds[1]:>8.3f}], true = {true_val:.3f}")

# Fixed parameters (not calibrated) - these will be set as initial values
fixed_params = {k: v for k, v in TRUE_SAC_PARAMS.items() if k not in CALIBRATION_PARAMS}

# Warmup period - 2 full years to allow groundwater stores to fill
SYNTHETIC_WARMUP = 730  # 2 years warmup for deep groundwater equilibrium

print(f"\nWarmup period: {SYNTHETIC_WARMUP} days (2 years)")
print(f"Calibration period: {n_days - SYNTHETIC_WARMUP} days ({(n_days - SYNTHETIC_WARMUP)/365:.1f} years)")

# Create a CalibrationRunner for synthetic data
# First, set up a model with fixed parameters as starting point
synthetic_model = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
synthetic_model.set_parameters(fixed_params)  # Set fixed params as base

# Note: objective and parameter_bounds are passed to CalibrationRunner constructor
# For SCE-UA (optimizer) - use NSE with sqrt transform for BALANCED flow emphasis
# This matches the TransformedGaussianLikelihood('sqrt') used for PyDREAM below
synthetic_runner_sceua = CalibrationRunner(
    model=synthetic_model,
    inputs=synthetic_inputs,
    observed=synthetic_observed,
    objective=NSE(transform=FlowTransformation('sqrt')),  # Balanced: NSE(sqrt)
    parameter_bounds=subset_bounds,  # Only calibrate the subset of parameters
    warmup_period=SYNTHETIC_WARMUP
)

# For PyDREAM (MCMC sampler) - use TransformedGaussianLikelihood (proper log-likelihood)
# This is required because DREAM is a Bayesian MCMC algorithm that samples from the
# posterior distribution and needs a proper log-likelihood, not an efficiency metric.
#
# TransformedGaussianLikelihood implements: log_lik = -n/2 * log(Σ(T(obs) - T(sim))²)
# where T() is a flow transformation function. This formulation (from Vrugt 2016)
# integrates out the measurement error variance.
#
# Flow transformation options:
#   - 'none': High flow emphasis (equivalent to GaussianLikelihood)
#   - 'sqrt': BALANCED emphasis (recommended default) - equivalent to NSE(sqrt)
#   - 'log':  Low flow emphasis - equivalent to LogNSE
#   - 'inverse': Very low flow emphasis - equivalent to InvNSE
#
# Using 'sqrt' for balanced calibration that gives appropriate weight to
# both high flows (peaks) and low flows (baseflow/recession).
synthetic_model_pydream = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
synthetic_model_pydream.set_parameters(fixed_params)

synthetic_runner_pydream = CalibrationRunner(
    model=synthetic_model_pydream,
    inputs=synthetic_inputs,
    observed=synthetic_observed,
    objective=TransformedGaussianLikelihood('sqrt'),  # Balanced log-likelihood for MCMC
    parameter_bounds=subset_bounds,
    warmup_period=SYNTHETIC_WARMUP
)

print(f"\nCalibrationRunners created:")
print(f"  - SCE-UA runner: uses NSE(sqrt) (optimizer objective, BALANCED flow emphasis)")
print(f"  - PyDREAM runner: uses TransformedGaussianLikelihood('sqrt') (BALANCED Bayesian log-likelihood)")
print(f"\nBoth algorithms use sqrt transformation for equivalent flow emphasis!")

# Helper function to calculate NSE for any parameter set
# Helper function to calculate NSE(sqrt) for any parameter set - MATCHES the objective used
nse_sqrt_objective = NSE(transform=FlowTransformation('sqrt'))

def calc_synthetic_nse(params_dict):
    """Calculate NSE(sqrt) for a parameter set on synthetic data.
    
    Uses sqrt transformation to match the objective function used by both
    SCE-UA and PyDREAM (TransformedGaussianLikelihood('sqrt')).
    """
    full_params = {**fixed_params, **params_dict}
    model = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
    model.set_parameters(full_params)
    model.reset()
    output = model.run(synthetic_inputs)
    simulated = output['runoff'].values[SYNTHETIC_WARMUP:]
    obs = synthetic_observed[SYNTHETIC_WARMUP:]
    
    # Use NSE(sqrt) for balanced flow evaluation
    return nse_sqrt_objective(obs, simulated)

# %% [markdown]
# ### Test 1: SCE-UA on Synthetic Hydrograph

# %%
print("=" * 70)
print("TEST 1: SCE-UA ON SYNTHETIC HYDROGRAPH")
print("=" * 70)

# Use CalibrationRunner's run_sceua_direct method
print(f"\nRunning SCE-UA with {len(CALIBRATION_PARAMS)} parameters...")
print("  Max evaluations: 5000")
print("  Using CalibrationRunner.run_sceua_direct()")

start_time = time.time()

sceua_result = synthetic_runner_sceua.run_sceua_direct(
    max_evals=10000,  # Increased for better convergence
    max_tolerant_iter=100,  # Stop after 100 iterations without improvement
    n_complexes=7,  # More complexes for better exploration
    seed=42,
    verbose=True,
    progress_bar=True
)

sceua_time = time.time() - start_time
sceua_params = sceua_result.best_parameters
sceua_nse = sceua_result.best_objective

print(f"\nSCE-UA Results (took {sceua_time:.1f}s):")
print(f"  Best NSE(sqrt): {sceua_nse:.6f}")
print(f"  Function evaluations: {sceua_result.convergence_diagnostics.get('nfev', 'N/A')}")
print(f"\n  {'Parameter':<10} {'True':>10} {'SCE-UA':>10} {'Error':>10}")
print("  " + "-" * 42)
for p in CALIBRATION_PARAMS:
    true_val = TRUE_SAC_PARAMS[p]
    est_val = sceua_params.get(p, true_val)  # Use true if not in calibrated set
    error = abs(est_val - true_val) / true_val * 100
    print(f"  {p:<10} {true_val:>10.3f} {est_val:>10.3f} {error:>9.1f}%")

# %% [markdown]
# ### Test 2: PyDREAM on Synthetic Hydrograph

# %%
print("=" * 70)
print("TEST 2: PYDREAM ON SYNTHETIC HYDROGRAPH")
print("=" * 70)

if PYDREAM_AVAILABLE:
    # PyDREAM hyperparameters - MATCHING REAL DATA SETTINGS
    # Same settings used for both synthetic and real data for consistency
    SYNTHETIC_ITERATIONS = 500       # Iterations per chain
    SYNTHETIC_CHAINS = 3             # 3 chains (minimum for DEpairs=1: 2*1+1=3)
    SYNTHETIC_MULTITRY = 1           # Standard DREAM (no multi-try)
    SYNTHETIC_DEPAIRS = 1            # 1 DE pair
    SYNTHETIC_SNOOKER = 0.10         # 10% snooker probability
    SYNTHETIC_CONVERGENCE = 1.05     # Stricter R-hat threshold
    SYNTHETIC_ADAPT_GAMMA = False    # Disabled - PyDREAM NumPy bug
    SYNTHETIC_ADAPT_CROSSOVER = False  # Disabled - PyDREAM NumPy bug
    
    print(f"\nRunning PyDREAM (STANDARD MODE) with {len(CALIBRATION_PARAMS)} parameters...")
    print(f"  Iterations per chain: {SYNTHETIC_ITERATIONS}")
    print(f"  Number of chains: {SYNTHETIC_CHAINS} (parallel)")
    print(f"  Multi-try samples: {SYNTHETIC_MULTITRY}")
    print(f"  DE pairs: {SYNTHETIC_DEPAIRS}")
    print(f"  Snooker probability: {SYNTHETIC_SNOOKER:.0%}")
    print(f"  Adaptive gamma: {SYNTHETIC_ADAPT_GAMMA}")
    print(f"  Adaptive crossover: {SYNTHETIC_ADAPT_CROSSOVER}")
    print(f"  Convergence threshold: GR < {SYNTHETIC_CONVERGENCE}")
    print(f"  Total samples: ~{SYNTHETIC_ITERATIONS * SYNTHETIC_CHAINS:,}")
    
    start_time = time.time()
    
    # Force stdout flush to ensure progress is visible
    import sys
    sys.stdout.flush()
    
    # Ensure figures directory exists (fallback if setup cell wasn't run)
    if 'figures_dir' not in dir():
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)
    
    # Use dbname for CSV-based progress tracking - monitor externally with:
    # tail -f figures/pydream_synthetic_progress.csv
    pydream_progress_file = figures_dir / 'pydream_synthetic_progress'
    
    print(f"\n  Progress file: {pydream_progress_file}.csv")
    print(f"  Monitor externally with: tail -f {pydream_progress_file}.csv")
    print("-" * 60)
    sys.stdout.flush()
    
    # Run PyDREAM in STANDARD MODE (no batches, no adaptive features)
    pydream_result = synthetic_runner_pydream.run_pydream(
        n_iterations=SYNTHETIC_ITERATIONS,
        n_chains=SYNTHETIC_CHAINS,
        multitry=SYNTHETIC_MULTITRY,
        snooker=SYNTHETIC_SNOOKER,
        DEpairs=SYNTHETIC_DEPAIRS,
        adapt_crossover=SYNTHETIC_ADAPT_CROSSOVER,  # Disabled - PyDREAM bug
        adapt_gamma=SYNTHETIC_ADAPT_GAMMA,          # Disabled - PyDREAM bug
        convergence_threshold=SYNTHETIC_CONVERGENCE,
        parallel=True,
        dbname=str(pydream_progress_file),
        hardboundaries=True,
        verbose=True,
    )
    
    pydream_time = time.time() - start_time
    
    # Extract results (CalibrationResult is a dataclass, use dot notation)
    pydream_best = pydream_result.best_parameters
    pydream_loglik = pydream_result.best_objective  # This is log-likelihood, not NSE!
    
    # Calculate NSE for the best PyDREAM parameters (for fair comparison with SCE-UA)
    pydream_nse = calc_synthetic_nse(pydream_best)
    
    # Get posterior statistics from all_samples
    all_samples_df = pydream_result.all_samples
    if all_samples_df is not None and len(all_samples_df) > 0:
        # Burn-in: discard first 30%
        burn_in = int(0.3 * len(all_samples_df))
        samples_burned = all_samples_df.iloc[burn_in:]
        
        posterior_means = {p: samples_burned[p].mean() for p in CALIBRATION_PARAMS if p in samples_burned.columns}
        posterior_stds = {p: samples_burned[p].std() for p in CALIBRATION_PARAMS if p in samples_burned.columns}
    else:
        posterior_means = pydream_best.copy()
        posterior_stds = {p: 0.0 for p in CALIBRATION_PARAMS}
    
    # Get convergence diagnostics
    conv_diag = pydream_result.convergence_diagnostics
    converged = conv_diag.get('converged', False) if conv_diag else False
    max_gr = conv_diag.get('max_gr', 'N/A') if conv_diag else 'N/A'
    
    # Check for early stopping info
    early_stopped = pydream_result.early_stopped if hasattr(pydream_result, 'early_stopped') else False
    total_iter = pydream_result.total_iterations if hasattr(pydream_result, 'total_iterations') else SYNTHETIC_ITERATIONS
    
    print(f"\nPyDREAM Results (took {pydream_time:.1f}s):")
    if early_stopped:
        print(f"  ✓ EARLY STOPPED at {total_iter} iterations (max was {SYNTHETIC_ITERATIONS})")
    else:
        print(f"  Total iterations: {total_iter} per chain")
    if converged:
        print(f"  ✓ CONVERGED (max GR: {max_gr:.4f} < {SYNTHETIC_CONVERGENCE})")
    else:
        print(f"  ○ Not converged (max GR: {max_gr if isinstance(max_gr, str) else f'{max_gr:.4f}'} >= {SYNTHETIC_CONVERGENCE})")
    print(f"  Best log-likelihood: {pydream_loglik:.4f}")
    print(f"  Best NSE(sqrt) (calculated): {pydream_nse:.6f}")
    print(f"\n  {'Parameter':<10} {'True':>10} {'MAP':>10} {'Mean±Std':>20}")
    print("  " + "-" * 52)
    for p in CALIBRATION_PARAMS:
        true_val = TRUE_SAC_PARAMS[p]
        map_val = pydream_best.get(p, true_val)
        mean_val = posterior_means.get(p, map_val)
        std_val = posterior_stds.get(p, 0.0)
        print(f"  {p:<10} {true_val:>10.3f} {map_val:>10.3f} {mean_val:>10.3f}±{std_val:<8.3f}")
    
else:
    print("\n⚠ PyDREAM not available - skipping test")
    pydream_best = None
    posterior_means = None
    posterior_stds = None
    pydream_time = 0
    pydream_nse = np.nan

# %% [markdown]
# ### Comparison: SCE-UA vs PyDREAM on Synthetic Hydrograph

# %%
print("=" * 70)
print("COMPARISON: SCE-UA vs PyDREAM (SYNTHETIC HYDROGRAPH)")
print("=" * 70)

# Run both calibrated models
sceua_full_params = {**fixed_params, **sceua_params}
sceua_model = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
sceua_model.set_parameters(sceua_full_params)
sceua_model.reset()
sceua_sim = sceua_model.run(synthetic_inputs)['runoff'].values

if PYDREAM_AVAILABLE and pydream_best is not None:
    pydream_full_params = {**fixed_params, **pydream_best}
    pydream_model = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
    pydream_model.set_parameters(pydream_full_params)
    pydream_model.reset()
    pydream_sim = pydream_model.run(synthetic_inputs)['runoff'].values
else:
    pydream_sim = None

# Create comparison figure
fig = plt.figure(figsize=(14, 10))

# 1. Hydrograph comparison
ax1 = plt.subplot(2, 2, 1)
ax1.plot(dates[SYNTHETIC_WARMUP:], synthetic_observed[SYNTHETIC_WARMUP:], 
         'gray', alpha=0.7, linewidth=0.8, label='Observed (noisy)')
ax1.plot(dates[SYNTHETIC_WARMUP:], true_flow[SYNTHETIC_WARMUP:], 
         'k-', linewidth=1.5, label='True')
ax1.plot(dates[SYNTHETIC_WARMUP:], sceua_sim[SYNTHETIC_WARMUP:], 
         'b--', linewidth=1.2, label=f'SCE-UA (NSE√={sceua_nse:.4f})')
if pydream_sim is not None:
    ax1.plot(dates[SYNTHETIC_WARMUP:], pydream_sim[SYNTHETIC_WARMUP:], 
             'r:', linewidth=1.2, label=f'PyDREAM (NSE√={pydream_nse:.4f})')
ax1.set_ylabel('Flow (ML/day)')
ax1.set_xlabel('Date')
ax1.set_title('Hydrograph Comparison')
ax1.legend(fontsize=9)
ax1.set_yscale('log')

# 2. Parameter recovery
ax2 = plt.subplot(2, 2, 2)
x_pos = np.arange(len(CALIBRATION_PARAMS))
width = 0.25

true_vals = [TRUE_SAC_PARAMS[p] for p in CALIBRATION_PARAMS]
sceua_vals = [sceua_params.get(p, TRUE_SAC_PARAMS[p]) for p in CALIBRATION_PARAMS]

# Normalize by true values for comparison
true_norm = [1.0] * len(CALIBRATION_PARAMS)
sceua_norm = [sceua_params.get(p, TRUE_SAC_PARAMS[p]) / TRUE_SAC_PARAMS[p] for p in CALIBRATION_PARAMS]

ax2.bar(x_pos - width/2, true_norm, width, label='True', color='black', alpha=0.7)
ax2.bar(x_pos + width/2, sceua_norm, width, label='SCE-UA', color='blue', alpha=0.7)

if PYDREAM_AVAILABLE and posterior_means is not None:
    pydream_norm = [posterior_means.get(p, TRUE_SAC_PARAMS[p]) / TRUE_SAC_PARAMS[p] for p in CALIBRATION_PARAMS]
    pydream_err = [posterior_stds.get(p, 0.0) / TRUE_SAC_PARAMS[p] for p in CALIBRATION_PARAMS]
    ax2.bar(x_pos + width*1.5, pydream_norm, width, label='PyDREAM', color='red', alpha=0.7,
            yerr=pydream_err, capsize=3)

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(CALIBRATION_PARAMS, rotation=45, ha='right')
ax2.set_ylabel('Ratio to True Value')
ax2.set_title('Parameter Recovery (normalized)')
ax2.legend()
ax2.set_ylim(0, 2)

# 3. Scatter plot
ax3 = plt.subplot(2, 2, 3)
obs_compare = synthetic_observed[SYNTHETIC_WARMUP:]
ax3.scatter(obs_compare, sceua_sim[SYNTHETIC_WARMUP:], alpha=0.3, s=10, 
            color='blue', label='SCE-UA')
if pydream_sim is not None:
    ax3.scatter(obs_compare, pydream_sim[SYNTHETIC_WARMUP:], alpha=0.3, s=10, 
                color='red', label='PyDREAM')
max_val = max(obs_compare.max(), sceua_sim[SYNTHETIC_WARMUP:].max())
ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
ax3.set_xlabel('Observed (ML/day)')
ax3.set_ylabel('Simulated (ML/day)')
ax3.set_title('Simulated vs Observed')
ax3.legend()

# 4. Runtime comparison
ax4 = plt.subplot(2, 2, 4)
methods = ['SCE-UA']
times = [sceua_time]
nses = [sceua_nse]
colors = ['blue']

if PYDREAM_AVAILABLE and pydream_time > 0:
    methods.append('PyDREAM')
    times.append(pydream_time)
    nses.append(pydream_nse)
    colors.append('red')

bars = ax4.bar(methods, times, color=colors, alpha=0.7)
ax4.set_ylabel('Runtime (seconds)')
ax4.set_title('Computational Time')
for i, (bar, t, nse) in enumerate(zip(bars, times, nses)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{t:.1f}s\nNSE={nse:.4f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Algorithm Validation: Synthetic Hydrograph', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / '05_synthetic_validation.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print("\nBoth algorithms optimized NSE(sqrt) for balanced flow emphasis")
print(f"\n{'Metric':<25} {'SCE-UA':>15} {'PyDREAM':>15}")
print("-" * 55)
print(f"{'NSE(sqrt)':<25} {sceua_nse:>15.4f} {pydream_nse:>15.4f}")
print(f"{'Runtime (seconds)':<25} {sceua_time:>15.1f} {pydream_time:>15.1f}")

# Check if both algorithms recovered reasonable parameters
sceua_ok = sceua_nse > 0.9
pydream_ok = pydream_nse > 0.9 if PYDREAM_AVAILABLE else True

print(f"\n{'SCE-UA validation:':<25} {'✓ PASSED' if sceua_ok else '✗ FAILED'}")
print(f"{'PyDREAM validation:':<25} {'✓ PASSED' if pydream_ok else '✗ FAILED'}")

if sceua_ok and pydream_ok:
    print("\n✓ Both algorithms successfully validated!")
    print("  Ready to proceed with real data calibrations.")
else:
    print("\n⚠ Validation issues detected - review results before proceeding.")

print(f"\nFigure saved: {figures_dir / '05_synthetic_validation.png'}")

# %% [markdown]
# ---
# ## Run PyDREAM Calibrations (Real Data)
#
# Now that we've validated both algorithms work correctly, we run PyDREAM
# for all 13 objective functions on the real hydrological data.
#
# **Note**: Results are saved to disk, so subsequent runs can skip calibration.

# %%
# Directory for PyDREAM results
PYDREAM_RESULTS_DIR = REPORTS_DIR / 'pydream'
PYDREAM_RESULTS_DIR.mkdir(exist_ok=True)

# Storage for PyDREAM results
pydream_results = {}
pydream_reports = {}

print(f"PyDREAM results directory: {PYDREAM_RESULTS_DIR}")

# %%
# =============================================================================
# GELMAN-RUBIN R-HAT IMPLEMENTATION
# =============================================================================
# Same implementation as Notebook 08 (Calibration Monitor) for consistency.
# Uses PyDREAM's Gelman_Rubin function when available, with proper chain handling.

try:
    from pydream.convergence import Gelman_Rubin as pydream_gelman_rubin
    PYDREAM_GR_AVAILABLE = True
except ImportError:
    PYDREAM_GR_AVAILABLE = False

# Minimum samples per chain for reliable R-hat (after 50% burn-in removal)
MIN_SAMPLES_PER_CHAIN_RELIABLE = 100
MIN_SAMPLES_PER_CHAIN_MARGINAL = 50
MIN_SAMPLES_PER_CHAIN_MINIMUM = 20


def _compute_gelman_rubin_fallback(chains: list, param_names: list) -> dict:
    """
    Fallback Gelman-Rubin implementation matching PyDREAM's formula.
    
    Uses 50% burn-in removal and standard Gelman-Rubin formula.
    Reference: Gelman & Rubin (1992). Statistical Science, 7(4), 457-472.
    """
    nchains = len(chains)
    nsamples = len(chains[0])
    nburnin = nsamples // 2
    
    r_hat_values = {}
    
    for i, param in enumerate(param_names):
        # Extract parameter values for all chains, after burn-in
        param_chains = [chain[nburnin:, i] for chain in chains]
        
        # Within-chain variance
        chain_vars = [np.var(c, ddof=1) for c in param_chains]
        W = np.mean(chain_vars)
        
        # Between-chain variance
        chain_means = [np.mean(c) for c in param_chains]
        B = np.var(chain_means, ddof=1)
        
        # Variance estimate
        n_post_burnin = nsamples - nburnin
        var_est = W * (1 - 1./n_post_burnin) + B
        
        # R-hat
        if W > 1e-10:
            r_hat = np.sqrt(var_est / W)
        else:
            r_hat = np.nan
        
        r_hat_values[param] = r_hat
    
    return r_hat_values


def compute_gelman_rubin_from_progress(progress_file: Path, param_names: list, n_chains: int = 3) -> dict:
    """
    Compute Gelman-Rubin R-hat from PyDREAM progress CSV file.
    
    Same implementation as Notebook 08 for consistency.
    
    Args:
        progress_file: Path to progress CSV (without .csv extension)
        param_names: List of parameter names
        n_chains: Number of chains (default 3)
        
    Returns:
        Dictionary with r_hat values, reliability info, etc.
    """
    csv_path = Path(str(progress_file) + '.csv')
    
    if not csv_path.exists():
        return {'r_hat': {p: np.nan for p in param_names}, 'reliability': 'no_file'}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {'r_hat': {p: np.nan for p in param_names}, 'reliability': 'read_error', 'error': str(e)}
    
    # Map parameter names to actual column names in the CSV
    # PyDREAM progress files use 'par' prefix (e.g., 'paruztwm' instead of 'uztwm')
    csv_columns = df.columns.tolist()
    col_mapping = {}
    for param in param_names:
        if param in csv_columns:
            col_mapping[param] = param
        elif f'par{param}' in csv_columns:
            col_mapping[param] = f'par{param}'
        else:
            # Parameter not found in CSV - skip R-hat calculation
            return {
                'r_hat': {p: np.nan for p in param_names},
                'reliability': 'column_mismatch',
                'error': f"Parameter '{param}' not found in CSV columns: {csv_columns[:10]}..."
            }
    
    # Get the actual column names to use
    actual_columns = [col_mapping[p] for p in param_names]
    
    # Check for chain column (PyDREAM adapter adds this)
    if 'chain' in df.columns:
        chain_groups = df.groupby('chain')
        n_chains = len(chain_groups)
        chains = []
        for chain_id in sorted(df['chain'].unique()):
            chain_df = df[df['chain'] == chain_id]
            chain_data = chain_df[actual_columns].values
            chains.append(chain_data)
        min_len = min(len(c) for c in chains)
        chains = [c[:min_len] for c in chains]
        samples_per_chain = min_len
    else:
        # Interleaved format
        values = df[actual_columns].values
        n_total = len(values)
        samples_per_chain = n_total // n_chains
        chains = []
        for chain_idx in range(n_chains):
            chain_data = values[chain_idx::n_chains][:samples_per_chain]
            chains.append(chain_data)
    
    # PyDREAM uses 50% burn-in
    effective_samples = samples_per_chain // 2
    
    # Determine reliability
    if effective_samples < MIN_SAMPLES_PER_CHAIN_MINIMUM:
        reliability = 'insufficient'
    elif effective_samples < MIN_SAMPLES_PER_CHAIN_MARGINAL:
        reliability = 'unreliable'
    elif effective_samples < MIN_SAMPLES_PER_CHAIN_RELIABLE:
        reliability = 'marginal'
    else:
        reliability = 'reliable'
    
    # Compute R-hat
    r_hat_values = {}
    method = 'unknown'
    
    if PYDREAM_GR_AVAILABLE and reliability != 'insufficient':
        try:
            gr_array = pydream_gelman_rubin(chains)
            for i, param in enumerate(param_names):
                r_hat_values[param] = float(gr_array[i])
            method = 'pydream'
        except Exception as e:
            method = 'fallback'
            r_hat_values = _compute_gelman_rubin_fallback(chains, param_names)
    elif reliability == 'insufficient':
        for param in param_names:
            r_hat_values[param] = np.nan
        method = 'insufficient_samples'
    else:
        method = 'fallback'
        r_hat_values = _compute_gelman_rubin_fallback(chains, param_names)
    
    return {
        'r_hat': r_hat_values,
        'n_chains': n_chains,
        'samples_per_chain': samples_per_chain,
        'effective_samples': effective_samples,
        'reliability': reliability,
        'method': method
    }


def print_rhat_summary(gr_result: dict, threshold: float = 1.05):
    """Print R-hat summary with interpretation."""
    r_hat = gr_result['r_hat']
    reliability = gr_result.get('reliability', 'unknown')
    method = gr_result.get('method', 'unknown')
    
    if reliability in ['no_file', 'read_error']:
        print(f"  R-hat: Cannot compute ({reliability})")
        return False
    
    if reliability == 'insufficient':
        effective = gr_result.get('effective_samples', 0)
        print(f"  R-hat: Insufficient samples ({effective} < {MIN_SAMPLES_PER_CHAIN_MINIMUM})")
        return False
    
    # Compute summary statistics
    valid_rhats = [v for v in r_hat.values() if not np.isnan(v)]
    if not valid_rhats:
        print(f"  R-hat: No valid values")
        return False
    
    max_rhat = max(valid_rhats)
    n_converged = sum(1 for v in valid_rhats if v < threshold)
    n_total = len(valid_rhats)
    
    # Determine status
    if max_rhat < 1.01:
        status = "✓✓ Excellent"
    elif max_rhat < 1.05:
        status = "✓ Good"
    elif max_rhat < 1.1:
        status = "~ Acceptable"
    elif max_rhat < 1.2:
        status = "⚠ Borderline"
    else:
        status = "✗ Not converged"
    
    print(f"  R-hat ({method}): max={max_rhat:.3f} {status}")
    print(f"  Converged: {n_converged}/{n_total} parameters (R-hat < {threshold})")
    
    if reliability == 'marginal':
        print(f"  ⚠ Marginal sample size - R-hat estimates may be noisy")
    
    return max_rhat < threshold

# %%
# Helper function to run a single PyDREAM calibration
def run_pydream_calibration(obj_name, objective_func):
    """
    Run PyDREAM calibration for a single objective function.
    
    Returns the result and report, or loads from disk if already exists.
    """
    result_file = PYDREAM_RESULTS_DIR / f'410734_pydream_{obj_name.lower()}.pkl'
    progress_file = PYDREAM_RESULTS_DIR / f'progress_{obj_name.lower()}'
    
    print("=" * 60)
    print(f"PyDREAM CALIBRATION: {obj_name}")
    print("=" * 60)
    
    # Check if already exists
    if result_file.exists():
        print(f"Loading existing result: {result_file.name}")
        report = CalibrationReport.load(str(result_file))
        result = report.result
        print(f"✓ Loaded (Best objective: {result.best_objective:.4f})")
        if hasattr(result, 'actual_objective_value'):
            print(f"  Actual {obj_name}: {result.actual_objective_value:.4f}")
        
        # Compute R-hat from progress file if it exists
        param_names = list(result.best_parameters.keys())
        gr_result = compute_gelman_rubin_from_progress(progress_file, param_names, PYDREAM_CHAINS)
        print_rhat_summary(gr_result, PYDREAM_CONVERGENCE_THRESHOLD)
        
        return result, report
    
    if not PYDREAM_AVAILABLE:
        print("✗ PyDREAM not available!")
        return None, None
    
    # Get likelihood transform
    likelihood_transform = LIKELIHOOD_TRANSFORM_MAPPING.get(obj_name, 'sqrt')
    likelihood = TransformedGaussianLikelihood(likelihood_transform)
    
    print(f"Objective: {obj_name}")
    print(f"Likelihood: TransformedGaussianLikelihood('{likelihood_transform}')")
    print(f"Flow emphasis: {likelihood.flow_emphasis}")
    print(f"Config: {PYDREAM_ITERATIONS} iter × {PYDREAM_CHAINS} chains")
    print(f"Monitor: tail -f {progress_file}.csv")
    print("-" * 60)
    
    # Create runner
    runner = CalibrationRunner(
        model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
        inputs=cal_inputs,
        observed=cal_observed,
        objective=likelihood,
        warmup_period=WARMUP_DAYS
    )
    
    start_time = time.time()
    
    try:
        result = runner.run_pydream(
            n_iterations=PYDREAM_ITERATIONS,
            n_chains=PYDREAM_CHAINS,
            multitry=PYDREAM_MULTITRY,
            snooker=PYDREAM_SNOOKER,
            DEpairs=PYDREAM_DEPAIRS,
            parallel=True,
            adapt_crossover=PYDREAM_ADAPT_CROSSOVER,
            adapt_gamma=PYDREAM_ADAPT_GAMMA,
            verbose=True,
            nverbose=100,
            dbname=str(progress_file),
            hardboundaries=True,
            convergence_check=True,
            convergence_threshold=PYDREAM_CONVERGENCE_THRESHOLD
        )
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"✓ Completed in {elapsed:.1f}s")
        print(f"Best log-likelihood: {result.best_objective:.4f}")
        
        # Calculate actual objective value
        eval_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        eval_model.set_parameters(result.best_parameters)
        eval_model.reset()
        sim_output = eval_model.run(cal_inputs)
        sim_values = sim_output['runoff'].values[WARMUP_DAYS:]
        obs_values = cal_observed[WARMUP_DAYS:]
        
        actual_obj_value = objective_func(obs_values, sim_values)
        print(f"Actual {obj_name}: {actual_obj_value:.4f}")
        
        result.actual_objective_value = actual_obj_value
        result.objective_name = obj_name
        
        # Compute R-hat using same implementation as Notebook 08
        param_names = list(result.best_parameters.keys())
        gr_result = compute_gelman_rubin_from_progress(progress_file, param_names, PYDREAM_CHAINS)
        print_rhat_summary(gr_result, PYDREAM_CONVERGENCE_THRESHOLD)
        
        # Store R-hat in result for later analysis
        result.gelman_rubin_results = gr_result
        
        # Save
        report = runner.create_report(result, catchment_info={
            'name': 'Queanbeyan River', 
            'gauge_id': '410734', 
            'area_km2': CATCHMENT_AREA_KM2
        })
        report.save(str(result_file.with_suffix('')))
        print(f"✓ Saved to: {result_file}")
        
        return result, report
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# %% [markdown]
# ### Calibration 1/13: NSE (High-flow emphasis)

# %%
result, report = run_pydream_calibration('NSE', objectives['NSE'])
if result: pydream_results['NSE'] = result; pydream_reports['NSE'] = report

# %% [markdown]
# ### Calibration 2/13: LogNSE (Low-flow emphasis)

# %%
result, report = run_pydream_calibration('LogNSE', objectives['LogNSE'])
if result: pydream_results['LogNSE'] = result; pydream_reports['LogNSE'] = report

# %% [markdown]
# ### Calibration 3/13: InvNSE (Very low-flow emphasis)

# %%
result, report = run_pydream_calibration('InvNSE', objectives['InvNSE'])
if result: pydream_results['InvNSE'] = result; pydream_reports['InvNSE'] = report

# %% [markdown]
# ### Calibration 4/13: SqrtNSE (Balanced emphasis)

# %%
result, report = run_pydream_calibration('SqrtNSE', objectives['SqrtNSE'])
if result: pydream_results['SqrtNSE'] = result; pydream_reports['SqrtNSE'] = report

# %% [markdown]
# ### Calibration 5/13: SDEB (Balanced + FDC)

# %%
result, report = run_pydream_calibration('SDEB', objectives['SDEB'])
if result: pydream_results['SDEB'] = result; pydream_reports['SDEB'] = report

# %% [markdown]
# ### Calibration 6/13: KGE (High-flow emphasis)

# %%
result, report = run_pydream_calibration('KGE', objectives['KGE'])
if result: pydream_results['KGE'] = result; pydream_reports['KGE'] = report

# %% [markdown]
# ### Calibration 7/13: KGE_inv (Very low-flow emphasis)

# %%
result, report = run_pydream_calibration('KGE_inv', objectives['KGE_inv'])
if result: pydream_results['KGE_inv'] = result; pydream_reports['KGE_inv'] = report

# %% [markdown]
# ### Calibration 8/13: KGE_sqrt (Balanced emphasis)

# %%
result, report = run_pydream_calibration('KGE_sqrt', objectives['KGE_sqrt'])
if result: pydream_results['KGE_sqrt'] = result; pydream_reports['KGE_sqrt'] = report

# %% [markdown]
# ### Calibration 9/13: KGE_log (Low-flow emphasis)

# %%
result, report = run_pydream_calibration('KGE_log', objectives['KGE_log'])
if result: pydream_results['KGE_log'] = result; pydream_reports['KGE_log'] = report

# %% [markdown]
# ### Calibration 10/13: KGE_np (Robust high-flow)

# %%
result, report = run_pydream_calibration('KGE_np', objectives['KGE_np'])
if result: pydream_results['KGE_np'] = result; pydream_reports['KGE_np'] = report

# %% [markdown]
# ### Calibration 11/13: KGE_np_inv (Robust very low-flow)

# %%
result, report = run_pydream_calibration('KGE_np_inv', objectives['KGE_np_inv'])
if result: pydream_results['KGE_np_inv'] = result; pydream_reports['KGE_np_inv'] = report

# %% [markdown]
# ### Calibration 12/13: KGE_np_sqrt (Robust balanced)

# %%
result, report = run_pydream_calibration('KGE_np_sqrt', objectives['KGE_np_sqrt'])
if result: pydream_results['KGE_np_sqrt'] = result; pydream_reports['KGE_np_sqrt'] = report

# %% [markdown]
# ### Calibration 13/13: KGE_np_log (Robust low-flow)

# %%
result, report = run_pydream_calibration('KGE_np_log', objectives['KGE_np_log'])
if result: pydream_results['KGE_np_log'] = result; pydream_reports['KGE_np_log'] = report

# %% [markdown]
# ### Calibration Summary

# %%
print("=" * 60)
print("PYDREAM CALIBRATION SUMMARY")
print("=" * 60)
print(f"\nCompleted: {len(pydream_results)}/13 calibrations")
print("\nResults:")
for name in objectives.keys():
    if name in pydream_results:
        result = pydream_results[name]
        actual = getattr(result, 'actual_objective_value', result.best_objective)
        print(f"  ✓ {name:<12}: {actual:.4f}")
    else:
        print(f"  - {name:<12}: Not available")

# %% [markdown]
# ---
# ## Load Existing PyDREAM Results
#
# If calibrations were run previously, load them from disk.

# %%
# Load any existing PyDREAM results not already in memory
print("=" * 70)
print("LOADING PYDREAM RESULTS")
print("=" * 70)

for name in objectives.keys():
    if name in pydream_results:
        continue  # Already loaded
    
    result_file = PYDREAM_RESULTS_DIR / f'410734_pydream_{name.lower()}.pkl'
    if result_file.exists():
        try:
            report = CalibrationReport.load(str(result_file))
            pydream_reports[name] = report
            pydream_results[name] = report.result
            print(f"  ✓ {name:<12}: Loaded")
        except Exception as e:
            print(f"  ✗ {name:<12}: Failed to load ({e})")
    else:
        print(f"  - {name:<12}: Not found")

print(f"\nTotal PyDREAM results available: {len(pydream_results)}/{len(objectives)}")

# %% [markdown]
# ---
# ## SCE-UA vs PyDREAM: Performance Comparison
#
# Let's compare the best objective values and performance metrics achieved
# by each algorithm across all objective functions.

# %%
# Build comprehensive comparison table with all NSE and KGE variants
import plotly.graph_objects as go

# Define all NSE and KGE variants to calculate
# Each tuple: (display_name, objective_class, transform_type)
nse_variants = [
    ('NSE', NSE, None),
    ('NSE(√Q)', NSE, 'sqrt'),
    ('NSE(log Q)', NSE, 'log'),
    ('NSE(1/Q)', NSE, 'inverse'),
]

kge_variants = [
    ('KGE', KGE, None),
    ('KGE(√Q)', KGE, 'sqrt'),
    ('KGE(log Q)', KGE, 'log'),
    ('KGE(1/Q)', KGE, 'inverse'),
]

def calculate_all_variants(sim, obs):
    """Calculate all NSE and KGE variants for a simulation.
    
    Note: pyrrm.objectives classes use __call__(obs, sim) interface,
    not compute(sim, obs).
    """
    results = {}
    
    # Calculate NSE variants
    # NSE classes are callable: obj(obs, sim) - note argument order!
    for name, obj_class, transform in nse_variants:
        try:
            if transform is None:
                obj = obj_class()
            else:
                obj = obj_class(transform=FlowTransformation(transform, epsilon_value=0.01))
            # Use callable interface with correct argument order (obs, sim)
            results[name] = float(obj(obs, sim))
        except Exception as e:
            print(f"  Warning: Failed to calculate {name}: {e}")
            results[name] = np.nan
    
    # Calculate KGE variants
    for name, obj_class, transform in kge_variants:
        try:
            if transform is None:
                obj = obj_class(variant='2012')
            else:
                obj = obj_class(variant='2012', transform=FlowTransformation(transform, epsilon_value=0.01))
            # Use callable interface with correct argument order (obs, sim)
            results[name] = float(obj(obs, sim))
        except Exception as e:
            print(f"  Warning: Failed to calculate {name}: {e}")
            results[name] = np.nan
    
    # Also calculate RMSE and PBIAS using calculate_metrics
    basic_metrics = calculate_metrics(sim, obs)
    results['RMSE'] = basic_metrics.get('RMSE', np.nan)
    results['PBIAS'] = basic_metrics.get('PBIAS', np.nan)
    
    return results

# Build comparison data
comparison_data = []

for name in objectives.keys():
    row = {'Objective': name}
    
    # SCE-UA results
    if name in sceua_results:
        row['SCE-UA Runtime (s)'] = sceua_results[name].runtime_seconds
        
        # Calculate performance metrics
        model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        model.set_parameters(sceua_results[name].best_parameters)
        model.reset()
        sim = model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        obs = cal_observed[WARMUP_DAYS:]
        
        # Calculate all variants
        all_metrics = calculate_all_variants(sim, obs)
        for metric_name, value in all_metrics.items():
            row[f'SCE-UA {metric_name}'] = value
    else:
        row['SCE-UA Runtime (s)'] = np.nan
        for metric_name, _, _ in nse_variants + kge_variants:
            row[f'SCE-UA {metric_name}'] = np.nan
        row['SCE-UA RMSE'] = np.nan
        row['SCE-UA PBIAS'] = np.nan
    
    # PyDREAM results
    if name in pydream_results:
        row['PyDREAM Runtime (s)'] = pydream_results[name].runtime_seconds
        
        # Calculate performance metrics
        model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        model.set_parameters(pydream_results[name].best_parameters)
        model.reset()
        sim = model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        obs = cal_observed[WARMUP_DAYS:]
        
        # Calculate all variants
        all_metrics = calculate_all_variants(sim, obs)
        for metric_name, value in all_metrics.items():
            row[f'PyDREAM {metric_name}'] = value
    else:
        row['PyDREAM Runtime (s)'] = np.nan
        for metric_name, _, _ in nse_variants + kge_variants:
            row[f'PyDREAM {metric_name}'] = np.nan
        row['PyDREAM RMSE'] = np.nan
        row['PyDREAM PBIAS'] = np.nan
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Reorder columns: SCE-UA on left half, PyDREAM on right half
ordered_cols = ['Objective']

# LEFT HALF: All SCE-UA metrics
# SCE-UA NSE variants
for metric_name, _, _ in nse_variants:
    ordered_cols.append(f'SCE-UA {metric_name}')
# SCE-UA KGE variants
for metric_name, _, _ in kge_variants:
    ordered_cols.append(f'SCE-UA {metric_name}')
# SCE-UA error metrics and runtime
ordered_cols.extend(['SCE-UA RMSE', 'SCE-UA PBIAS', 'SCE-UA Runtime (s)'])

# RIGHT HALF: All PyDREAM metrics
# PyDREAM NSE variants
for metric_name, _, _ in nse_variants:
    ordered_cols.append(f'PyDREAM {metric_name}')
# PyDREAM KGE variants
for metric_name, _, _ in kge_variants:
    ordered_cols.append(f'PyDREAM {metric_name}')
# PyDREAM error metrics and runtime
ordered_cols.extend(['PyDREAM RMSE', 'PyDREAM PBIAS', 'PyDREAM Runtime (s)'])

# Filter to only include columns that exist
ordered_cols = [c for c in ordered_cols if c in comparison_df.columns]
comparison_df = comparison_df[ordered_cols]

print(f"Comparison table built with {len(comparison_df)} objectives and {len(comparison_df.columns)} metrics")
print(f"  Left half (SCE-UA): {sum(1 for c in comparison_df.columns if 'SCE-UA' in c)} columns")
print(f"  Right half (PyDREAM): {sum(1 for c in comparison_df.columns if 'PyDREAM' in c)} columns")

# %%
# Create color-coded Plotly table showing winners (blue) vs losers (red)

def create_comparison_table_plotly(df):
    """
    Create a Plotly table with color coding:
    - Blue (#4169E1): Winner (better performance)
    - Red (#DC143C): Loser (worse performance)
    - White: Neutral (objective name, or no comparison possible)
    
    For NSE, KGE (all variants): Higher is better
    For RMSE, PBIAS (absolute): Lower is better
    For Runtime: Lower is better
    """
    
    # Define metric pairs and which direction is better
    # Include all NSE and KGE variants
    metric_pairs = [
        # NSE variants (higher is better)
        ('SCE-UA NSE', 'PyDREAM NSE', 'higher'),
        ('SCE-UA NSE(√Q)', 'PyDREAM NSE(√Q)', 'higher'),
        ('SCE-UA NSE(log Q)', 'PyDREAM NSE(log Q)', 'higher'),
        ('SCE-UA NSE(1/Q)', 'PyDREAM NSE(1/Q)', 'higher'),
        # KGE variants (higher is better)
        ('SCE-UA KGE', 'PyDREAM KGE', 'higher'),
        ('SCE-UA KGE(√Q)', 'PyDREAM KGE(√Q)', 'higher'),
        ('SCE-UA KGE(log Q)', 'PyDREAM KGE(log Q)', 'higher'),
        ('SCE-UA KGE(1/Q)', 'PyDREAM KGE(1/Q)', 'higher'),
        # Error metrics (lower is better)
        ('SCE-UA RMSE', 'PyDREAM RMSE', 'lower'),
        ('SCE-UA PBIAS', 'PyDREAM PBIAS', 'lower_abs'),  # Lower absolute value is better
        # Runtime (lower is better)
        ('SCE-UA Runtime (s)', 'PyDREAM Runtime (s)', 'lower'),
    ]
    
    # Prepare cell colors
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Initialize all cells as white
    cell_colors = [['white'] * n_rows for _ in range(n_cols)]
    
    # Color code the comparison columns
    for sceua_col, pydream_col, direction in metric_pairs:
        if sceua_col in df.columns and pydream_col in df.columns:
            sceua_idx = df.columns.get_loc(sceua_col)
            pydream_idx = df.columns.get_loc(pydream_col)
            
            for row_idx in range(n_rows):
                sceua_val = df.iloc[row_idx][sceua_col]
                pydream_val = df.iloc[row_idx][pydream_col]
                
                # Skip if either value is NaN
                if pd.isna(sceua_val) or pd.isna(pydream_val):
                    continue
                
                # Determine winner based on direction
                if direction == 'higher':
                    sceua_wins = sceua_val > pydream_val
                    pydream_wins = pydream_val > sceua_val
                elif direction == 'lower':
                    sceua_wins = sceua_val < pydream_val
                    pydream_wins = pydream_val < sceua_val
                elif direction == 'lower_abs':
                    sceua_wins = abs(sceua_val) < abs(pydream_val)
                    pydream_wins = abs(pydream_val) < abs(sceua_val)
                
                # Apply colors: Blue for winner, Red for loser
                if sceua_wins:
                    cell_colors[sceua_idx][row_idx] = '#4169E1'  # Royal Blue
                    cell_colors[pydream_idx][row_idx] = '#DC143C'  # Crimson
                elif pydream_wins:
                    cell_colors[sceua_idx][row_idx] = '#DC143C'  # Crimson
                    cell_colors[pydream_idx][row_idx] = '#4169E1'  # Royal Blue
                # If equal, both stay white
    
    # Format values for display
    formatted_values = []
    for col in df.columns:
        if col == 'Objective':
            formatted_values.append(df[col].tolist())
        elif 'Runtime' in col:
            formatted_values.append([f'{v:.1f}' if pd.notna(v) else '-' for v in df[col]])
        elif 'PBIAS' in col:
            formatted_values.append([f'{v:.2f}%' if pd.notna(v) else '-' for v in df[col]])
        elif 'RMSE' in col:
            formatted_values.append([f'{v:.3f}' if pd.notna(v) else '-' for v in df[col]])
        else:
            formatted_values.append([f'{v:.4f}' if pd.notna(v) else '-' for v in df[col]])
    
    # Determine text colors (white text on colored cells, black on white)
    font_colors = []
    for col_colors in cell_colors:
        col_font = ['white' if c in ['#4169E1', '#DC143C'] else 'black' for c in col_colors]
        font_colors.append(col_font)
    
    # Create header colors: GREY SHADES to avoid conflict with winner/loser colors
    header_colors = []
    for col in df.columns:
        if col == 'Objective':
            header_colors.append('#2F4F4F')  # Dark slate gray for objective
        elif 'SCE-UA' in col:
            header_colors.append('#607D8B')  # LIGHT GREY (Blue Grey 500) for SCE-UA
        elif 'PyDREAM' in col:
            header_colors.append('#37474F')  # DARK GREY (Blue Grey 800) for PyDREAM
        else:
            header_colors.append('#2F4F4F')  # Default
    
    # Display headers with algorithm label underneath
    display_headers = []
    for col in df.columns:
        if col == 'Objective':
            display_headers.append(f'<b>{col}</b>')
        elif 'SCE-UA' in col:
            short_name = col.replace('SCE-UA ', '')
            display_headers.append(f'<b>{short_name}</b><br><sub>(SCE-UA)</sub>')
        elif 'PyDREAM' in col:
            short_name = col.replace('PyDREAM ', '')
            display_headers.append(f'<b>{short_name}</b><br><sub>(PyDREAM)</sub>')
        else:
            display_headers.append(f'<b>{col}</b>')
    
    # Create Plotly table - CRITICAL: fill_color WITHOUT brackets for per-column colors
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=display_headers,
            fill_color=header_colors,  # NO BRACKETS - applies one color per column
            font=dict(color='white', size=11),
            align='center',
            height=50
        ),
        cells=dict(
            values=formatted_values,
            fill_color=cell_colors,
            font=dict(color=font_colors, size=10),
            align=['left'] + ['center'] * (n_cols - 1),
            height=26
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='<b>SCE-UA vs PyDREAM: Comprehensive Performance Comparison</b><br>' +
                 '<span style="font-size:12px; color:gray">Cell colors: 🔵 Blue = Winner | 🔴 Red = Loser</span><br>' +
                 '<span style="font-size:12px"><b style="background-color:#607D8B; color:white; padding:2px 6px;">LIGHT GREY = SCE-UA</b> | ' +
                 '<b style="background-color:#37474F; color:white; padding:2px 6px;">DARK GREY = PyDREAM</b></span>',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        width=1800,  # Wider to accommodate more columns
        height=600,  # Taller for larger headers
        margin=dict(l=20, r=20, t=100, b=20)  # More top margin for title
    )
    
    return fig

# Create separate focused tables for better readability
def create_focused_table(df, metric_group, title_suffix):
    """Create a focused comparison table for a specific metric group."""
    # Select columns for this group
    cols = ['Objective']
    for col in df.columns:
        if metric_group in col or 'Runtime' in col:
            cols.append(col)
    
    subset_df = df[[c for c in cols if c in df.columns]]
    return create_comparison_table_plotly(subset_df)

# Display the full comparison table
print("=" * 80)
print("FULL COMPARISON TABLE: All Metrics")
print("=" * 80)
fig_comparison = create_comparison_table_plotly(comparison_df)
fig_comparison.show()

# %%
# Create focused tables for easier analysis
print("\n" + "=" * 80)
print("NSE VARIANTS COMPARISON")
print("=" * 80)

# NSE variants table
nse_cols = ['Objective'] + [c for c in comparison_df.columns if 'NSE' in c]
nse_df = comparison_df[nse_cols]
fig_nse = create_comparison_table_plotly(nse_df)
fig_nse.update_layout(
    title=dict(
        text='<b>NSE Variants: SCE-UA vs PyDREAM</b><br>' +
             '<span style="font-size:12px; color:gray">' +
             '🔵 Blue = Winner | 🔴 Red = Loser | Higher NSE is better</span>',
        font=dict(size=16),
        x=0.5,
        xanchor='center'
    ),
    width=1100,
    height=500
)
fig_nse.show()

# %%
print("\n" + "=" * 80)
print("KGE VARIANTS COMPARISON")
print("=" * 80)

# KGE variants table
kge_cols = ['Objective'] + [c for c in comparison_df.columns if 'KGE' in c]
kge_df = comparison_df[kge_cols]
fig_kge = create_comparison_table_plotly(kge_df)
fig_kge.update_layout(
    title=dict(
        text='<b>KGE Variants: SCE-UA vs PyDREAM</b><br>' +
             '<span style="font-size:12px; color:gray">' +
             '🔵 Blue = Winner | 🔴 Red = Loser | Higher KGE is better</span>',
        font=dict(size=16),
        x=0.5,
        xanchor='center'
    ),
    width=1100,
    height=500
)
fig_kge.show()

# %%
print("\n" + "=" * 80)
print("ERROR METRICS & RUNTIME COMPARISON")
print("=" * 80)

# Other metrics table
other_cols = ['Objective'] + [c for c in comparison_df.columns 
              if 'RMSE' in c or 'PBIAS' in c or 'Runtime' in c]
other_df = comparison_df[other_cols]
fig_other = create_comparison_table_plotly(other_df)
fig_other.update_layout(
    title=dict(
        text='<b>Error Metrics & Runtime: SCE-UA vs PyDREAM</b><br>' +
             '<span style="font-size:12px; color:gray">' +
             '🔵 Blue = Winner | 🔴 Red = Loser | Lower RMSE/|PBIAS|/Runtime is better</span>',
        font=dict(size=16),
        x=0.5,
        xanchor='center'
    ),
    width=900,
    height=500
)
fig_other.show()

# %%
# Summary statistics: Count wins for each algorithm
print("\n" + "=" * 70)
print("SUMMARY: ALGORITHM WIN COUNTS")
print("=" * 70)

win_counts = {'SCE-UA': 0, 'PyDREAM': 0, 'Tie': 0}
metric_wins = {}

metrics_to_compare = [
    # NSE variants (higher is better)
    ('NSE', 'higher'),
    ('NSE(√Q)', 'higher'),
    ('NSE(log Q)', 'higher'),
    ('NSE(1/Q)', 'higher'),
    # KGE variants (higher is better)
    ('KGE', 'higher'),
    ('KGE(√Q)', 'higher'),
    ('KGE(log Q)', 'higher'),
    ('KGE(1/Q)', 'higher'),
    # Error metrics (lower is better)
    ('RMSE', 'lower'),
    ('PBIAS', 'lower_abs'),
    # Runtime (lower is better)
    ('Runtime (s)', 'lower')
]

for metric, direction in metrics_to_compare:
    sceua_col = f'SCE-UA {metric}'
    pydream_col = f'PyDREAM {metric}'
    
    if sceua_col not in comparison_df.columns or pydream_col not in comparison_df.columns:
        continue
    
    sceua_wins = 0
    pydream_wins = 0
    ties = 0
    
    for idx, row in comparison_df.iterrows():
        sceua_val = row[sceua_col]
        pydream_val = row[pydream_col]
        
        if pd.isna(sceua_val) or pd.isna(pydream_val):
            continue
        
        if direction == 'higher':
            if sceua_val > pydream_val:
                sceua_wins += 1
            elif pydream_val > sceua_val:
                pydream_wins += 1
            else:
                ties += 1
        elif direction == 'lower':
            if sceua_val < pydream_val:
                sceua_wins += 1
            elif pydream_val < sceua_val:
                pydream_wins += 1
            else:
                ties += 1
        elif direction == 'lower_abs':
            if abs(sceua_val) < abs(pydream_val):
                sceua_wins += 1
            elif abs(pydream_val) < abs(sceua_val):
                pydream_wins += 1
            else:
                ties += 1
    
    metric_wins[metric] = {'SCE-UA': sceua_wins, 'PyDREAM': pydream_wins, 'Tie': ties}
    win_counts['SCE-UA'] += sceua_wins
    win_counts['PyDREAM'] += pydream_wins
    win_counts['Tie'] += ties
    
    winner = 'SCE-UA' if sceua_wins > pydream_wins else ('PyDREAM' if pydream_wins > sceua_wins else 'Tie')
    print(f"\n{metric}:")
    print(f"  SCE-UA wins:  {sceua_wins:2d} / {sceua_wins + pydream_wins + ties}")
    print(f"  PyDREAM wins: {pydream_wins:2d} / {sceua_wins + pydream_wins + ties}")
    print(f"  Ties:         {ties:2d} / {sceua_wins + pydream_wins + ties}")
    print(f"  → Overall winner for {metric}: {winner}")

print("\n" + "-" * 70)
print("OVERALL TOTALS (across all metrics and objectives):")
print("-" * 70)
total = win_counts['SCE-UA'] + win_counts['PyDREAM'] + win_counts['Tie']
print(f"  SCE-UA total wins:  {win_counts['SCE-UA']:3d} / {total} ({100*win_counts['SCE-UA']/total:.1f}%)")
print(f"  PyDREAM total wins: {win_counts['PyDREAM']:3d} / {total} ({100*win_counts['PyDREAM']/total:.1f}%)")
print(f"  Ties:               {win_counts['Tie']:3d} / {total} ({100*win_counts['Tie']/total:.1f}%)")

overall_winner = 'SCE-UA' if win_counts['SCE-UA'] > win_counts['PyDREAM'] else (
    'PyDREAM' if win_counts['PyDREAM'] > win_counts['SCE-UA'] else 'Tie'
)
print(f"\n  🏆 OVERALL WINNER: {overall_winner}")

# %%
# Create visual summary: Win counts by metric category
import plotly.express as px

# Group metrics by category
category_wins = {
    'NSE (all variants)': {'SCE-UA': 0, 'PyDREAM': 0},
    'KGE (all variants)': {'SCE-UA': 0, 'PyDREAM': 0},
    'RMSE': {'SCE-UA': 0, 'PyDREAM': 0},
    'PBIAS': {'SCE-UA': 0, 'PyDREAM': 0},
    'Runtime': {'SCE-UA': 0, 'PyDREAM': 0},
}

for metric, wins in metric_wins.items():
    if 'NSE' in metric:
        category_wins['NSE (all variants)']['SCE-UA'] += wins['SCE-UA']
        category_wins['NSE (all variants)']['PyDREAM'] += wins['PyDREAM']
    elif 'KGE' in metric:
        category_wins['KGE (all variants)']['SCE-UA'] += wins['SCE-UA']
        category_wins['KGE (all variants)']['PyDREAM'] += wins['PyDREAM']
    elif metric == 'RMSE':
        category_wins['RMSE']['SCE-UA'] += wins['SCE-UA']
        category_wins['RMSE']['PyDREAM'] += wins['PyDREAM']
    elif metric == 'PBIAS':
        category_wins['PBIAS']['SCE-UA'] += wins['SCE-UA']
        category_wins['PBIAS']['PyDREAM'] += wins['PyDREAM']
    elif metric == 'Runtime (s)':
        category_wins['Runtime']['SCE-UA'] += wins['SCE-UA']
        category_wins['Runtime']['PyDREAM'] += wins['PyDREAM']

# Create summary visualization
summary_data = []
for category, wins in category_wins.items():
    summary_data.append({'Category': category, 'Algorithm': 'SCE-UA', 'Wins': wins['SCE-UA']})
    summary_data.append({'Category': category, 'Algorithm': 'PyDREAM', 'Wins': wins['PyDREAM']})

summary_df = pd.DataFrame(summary_data)

fig_wins = px.bar(
    summary_df, 
    x='Category', 
    y='Wins', 
    color='Algorithm',
    barmode='group',
    color_discrete_map={'SCE-UA': '#4169E1', 'PyDREAM': '#DC143C'},
    title='<b>Algorithm Win Counts by Metric Category</b><br>' +
          '<span style="font-size:12px; color:gray">Across all 13 objective functions</span>'
)
fig_wins.update_layout(
    xaxis_title='Metric Category',
    yaxis_title='Number of Wins (out of 13 objectives)',
    legend_title='Algorithm',
    width=900,
    height=450
)
fig_wins.show()

# Print category summary
print("\n" + "=" * 70)
print("SUMMARY BY METRIC CATEGORY")
print("=" * 70)
for category, wins in category_wins.items():
    total_comparisons = wins['SCE-UA'] + wins['PyDREAM']
    if total_comparisons > 0:
        winner = 'SCE-UA' if wins['SCE-UA'] > wins['PyDREAM'] else (
            'PyDREAM' if wins['PyDREAM'] > wins['SCE-UA'] else 'Tie')
        print(f"\n{category}:")
        print(f"  SCE-UA:  {wins['SCE-UA']:2d} wins")
        print(f"  PyDREAM: {wins['PyDREAM']:2d} wins")
        print(f"  → Winner: {winner}")

# %%
from scipy.stats import gaussian_kde

# Get parameter bounds for all 22 Sacramento parameters
model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
param_bounds = model.get_parameter_bounds()
print(f"Sacramento model has {len(param_bounds)} parameters")

# %% [markdown]
# ---
# ## Compact Visualizations
#
# Here we present 2 compact visualization approaches that efficiently summarize 
# the comparison between SCE-UA point estimates and PyDREAM posterior distributions
# for all 22 Sacramento model parameters:
#
# 1. **Ridge Plot (Joy Plot)** - Beautiful overlapping KDEs using Seaborn FacetGrid
#    ([inspired by Python Graph Gallery](https://python-graph-gallery.com/ridgeline-graph-seaborn/))
#    with vertical markers for SCE-UA (red solid) and PyDREAM (green dashed) best estimates
# 2. **Summary Statistics Table** - Color-coded table with agreement indicators showing
#    whether SCE-UA estimates fall within PyDREAM's 95% credible intervals

# %%
# =============================================================================
# OPTION 1: RIDGE PLOT (JOY PLOT) - SEABORN FACETGRID VERSION
# =============================================================================
# Beautiful overlapping KDEs using Seaborn FacetGrid
# Inspired by: https://python-graph-gallery.com/ridgeline-graph-seaborn/

def plot_all_ridge_posteriors(pydream_results, sceua_results, param_bounds, burn_in_fraction=0.5):
    """
    Create a combined figure with ridge plots for ALL 13 objective functions.
    
    Organized by objective function family:
    - Row 1: NSE variants (NSE, LogNSE, InvNSE, SqrtNSE)
    - Row 2: KGE variants (KGE, KGE_inv, KGE_sqrt, KGE_log)
    - Row 3: KGE nonparametric variants (KGE_np, KGE_np_inv, KGE_np_sqrt, KGE_np_log)
    - Row 4: SDEB + legend
    
    Features:
    - Logical grouping by objective function family
    - Agreement-based coloring:
      * Dark Green: SCE-UA within 50% CI (strong agreement)
      * Light Green: SCE-UA within 95% CI (moderate agreement)
      * Orange: SCE-UA outside 95% CI but close (borderline)
      * Red: SCE-UA outside 95% CI and far (disagreement)
    - Dot markers for best parameter estimates
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Agreement color scheme
    COLOR_STRONG_AGREE = '#2E7D32'    # Dark green - within 50% CI
    COLOR_MODERATE_AGREE = '#8BC34A'  # Light green - within 95% CI
    COLOR_BORDERLINE = '#FF9800'      # Orange - outside CI but close
    COLOR_DISAGREE = '#E53935'        # Red - outside CI and far
    
    def get_agreement_color(sceua_value, posterior, norm_sceua, norm_pydream):
        """Determine ridge color based on SCE-UA agreement with PyDREAM posterior."""
        # Calculate credible intervals
        ci_50_low, ci_50_high = np.percentile(posterior, [25, 75])
        ci_95_low, ci_95_high = np.percentile(posterior, [2.5, 97.5])
        
        # Calculate normalized distance between point estimates
        distance = abs(norm_sceua - norm_pydream)
        
        # Determine agreement level
        if ci_50_low <= sceua_value <= ci_50_high:
            return COLOR_STRONG_AGREE  # Strong agreement
        elif ci_95_low <= sceua_value <= ci_95_high:
            return COLOR_MODERATE_AGREE  # Moderate agreement
        else:
            # Outside 95% CI - check distance
            if distance < 0.2:
                return COLOR_BORDERLINE  # Borderline - close but outside CI
            else:
                return COLOR_DISAGREE  # Clear disagreement
    
    # Organize objectives by family (row-wise)
    # Column order: Base, Sqrt, Log, Inverse
    objective_rows = [
        # Row 1: NSE family
        ['NSE', 'SqrtNSE', 'LogNSE', 'InvNSE'],
        # Row 2: KGE family
        ['KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv'],
        # Row 3: KGE nonparametric family
        ['KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv'],
        # Row 4: SDEB in sqrt column (col 2), rest empty
        [None, 'SDEB', None, None]
    ]
    
    row_labels = ['NSE Family', 'KGE Family', 'KGE Nonparametric', 'Other']
    
    # Layout: 4 rows x 4 columns
    n_cols = 4
    n_rows = 4
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(14, 14))
    
    # Get parameter list from first available result
    first_obj = None
    for row in objective_rows:
        for obj in row:
            if obj in pydream_results and pydream_results[obj].all_samples is not None:
                first_obj = obj
                break
        if first_obj:
            break
    
    if first_obj is None:
        print("No objectives available for plotting")
        return None
    
    all_params = list(sceua_results[first_obj].best_parameters.keys())
    n_params = len(all_params)
    
    # Create subplots
    axes = []
    for row_idx in range(n_rows):
        row_axes = []
        for col_idx in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Plot each objective in its designated position
    for row_idx, (obj_row, row_label) in enumerate(zip(objective_rows, row_labels)):
        for col_idx, obj_name in enumerate(obj_row):
            ax = axes[row_idx][col_idx]
            
            # Handle empty cells (None)
            if obj_name is None:
                ax.axis('off')
                continue
            
            # Check if objective is available
            if (obj_name not in pydream_results or obj_name not in sceua_results or
                pydream_results[obj_name].all_samples is None):
                ax.text(0.5, 0.5, f'{obj_name}\n(No data)', ha='center', va='center',
                       fontsize=9, transform=ax.transAxes, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            pydream_result = pydream_results[obj_name]
            sceua_result = sceua_results[obj_name]
            
            # Get samples with burn-in
            all_samples = pydream_result.all_samples
            n_burnin = int(len(all_samples) * burn_in_fraction)
            samples = all_samples.iloc[n_burnin:].copy()
            
            params_to_plot = [p for p in all_params if p in samples.columns]
            
            # Spacing for ridge effect
            spacing = 1.0
            
            for i, param in enumerate(params_to_plot):
                posterior = samples[param].values
                bounds = param_bounds.get(param, (posterior.min(), posterior.max()))
                
                # Get actual SCE-UA value (not normalized) for CI comparison
                sceua_value = sceua_result.best_parameters[param]
                
                # Normalize to [0, 1]
                norm_posterior = (posterior - bounds[0]) / (bounds[1] - bounds[0])
                norm_sceua = (sceua_value - bounds[0]) / (bounds[1] - bounds[0])
                norm_pydream = (pydream_result.best_parameters[param] - bounds[0]) / (bounds[1] - bounds[0])
                
                # Determine agreement-based color
                ridge_color = get_agreement_color(sceua_value, posterior, norm_sceua, norm_pydream)
                
                y_offset = i * spacing
                
                try:
                    kde = gaussian_kde(norm_posterior)
                    x_grid = np.linspace(0, 1, 100)
                    y_kde = kde(x_grid)
                    # Scale KDE height
                    y_kde = y_kde / y_kde.max() * 0.85
                    
                    # Plot filled KDE with agreement-based color
                    ax.fill_between(x_grid, y_offset, y_offset + y_kde, 
                                   alpha=0.85, color=ridge_color, linewidth=0)
                    # White contour
                    ax.plot(x_grid, y_offset + y_kde, color='white', linewidth=0.8)
                except Exception:
                    pass
                
                # Baseline with same color
                ax.axhline(y=y_offset, color=ridge_color, linewidth=0.3, alpha=0.3)
                
                # Point estimates as small markers with high contrast colors
                # Blue for SCE-UA, Magenta for PyDREAM
                ax.scatter([norm_sceua], [y_offset], color='#1E88E5', s=20, 
                          marker='|', zorder=10, linewidths=2)
                ax.scatter([norm_pydream], [y_offset], color='#D81B60', s=20,
                          marker='|', zorder=10, linewidths=2)
            
            # Subplot styling
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.5, n_params * spacing + 0.5)
            ax.set_title(obj_name, fontsize=10, fontweight='bold', pad=2)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'], fontsize=7)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Show x-label on bottom row only
            if row_idx == n_rows - 1:
                ax.set_xlabel('Norm. Value', fontsize=8)
        
        # Add row label on leftmost plot
        axes[row_idx][0].set_ylabel(row_label, fontsize=10, fontweight='bold', 
                                    rotation=90, labelpad=10)
        
        # Hide unused columns in this row
        for col_idx in range(len(obj_row), n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
    
    # Use the empty cells in the last row for legend and info
    # Row 4 layout: [Point Estimates, SDEB, Color Legend, Info]
    
    # Point estimates legend in column 1 (index 0)
    legend_ax = axes[3][0]
    legend_ax.axis('off')
    marker_elements = [
        Line2D([0], [0], marker='|', color='#1E88E5', markersize=14, 
               markeredgewidth=3, label='SCE-UA', linestyle='None'),
        Line2D([0], [0], marker='|', color='#D81B60', markersize=14,
               markeredgewidth=3, label='PyDREAM', linestyle='None')
    ]
    legend_ax.legend(handles=marker_elements, loc='center', fontsize=10, 
                     frameon=True, fancybox=True, title='Point Estimates',
                     title_fontsize=11)
    
    # SDEB is in column 2 (index 1) - plotted automatically
    
    # Agreement color legend in column 3 (index 2)
    color_ax = axes[3][2]
    color_ax.axis('off')
    color_elements = [
        Patch(facecolor=COLOR_STRONG_AGREE, edgecolor='white', label='Within 50% CI'),
        Patch(facecolor=COLOR_MODERATE_AGREE, edgecolor='white', label='Within 95% CI'),
        Patch(facecolor=COLOR_BORDERLINE, edgecolor='white', label='Outside CI (close)'),
        Patch(facecolor=COLOR_DISAGREE, edgecolor='white', label='Outside CI (far)')
    ]
    color_ax.legend(handles=color_elements, loc='center', fontsize=9, 
                    frameon=True, fancybox=True, title='SCE-UA Agreement',
                    title_fontsize=10)
    
    # Parameter info in column 4 (index 3)
    info_ax = axes[3][3]
    info_ax.axis('off')
    info_ax.text(0.5, 0.7, f'{n_params} Parameters', ha='center', va='center', 
                fontsize=11, fontweight='bold', transform=info_ax.transAxes)
    info_ax.text(0.5, 0.45, 'Stacked top → bottom\n(uztwm ... uh5)', 
                ha='center', va='center', fontsize=9, transform=info_ax.transAxes)
    info_ax.text(0.5, 0.2, 'X-axis: Normalized [0,1]', ha='center', va='center',
                fontsize=8, style='italic', transform=info_ax.transAxes)
    
    # Main title
    fig.suptitle('Parameter Posterior Distributions by Objective Function Family', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Add explanatory text box at the bottom
    explanation_text = (
        "How to read this figure: Each subplot shows the posterior distributions (ridges) for all 22 Sacramento model parameters "
        "calibrated using a specific objective function. The ridge color indicates agreement between SCE-UA and PyDREAM: "
        "dark green = SCE-UA within 50% credible interval (strong agreement), light green = within 95% CI (moderate), "
        "orange = outside CI but estimates are close, red = outside CI and far apart (disagreement). "
        "Blue vertical bars (|) mark SCE-UA point estimates; magenta bars mark PyDREAM best estimates. "
        "X-axis shows normalized parameter values [0=lower bound, 1=upper bound]."
    )
    fig.text(0.5, 0.02, explanation_text, ha='center', va='bottom', fontsize=8,
             wrap=True, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', 
                                  edgecolor='gray', alpha=0.9),
             transform=fig.transFigure)
    
    return fig

# %%
# =============================================================================
# OPTION 3: SUMMARY STATISTICS TABLE WITH COLOR CODING
# =============================================================================
# Plotly table with agreement indicators

def create_summary_table(obj_name, pydream_result, sceua_result, param_bounds, burn_in_fraction=0.5):
    """
    Create a color-coded summary table showing SCE-UA vs PyDREAM comparison.
    """
    if pydream_result.all_samples is None or len(pydream_result.all_samples) == 0:
        print(f"No samples available for {obj_name}")
        return None
    
    all_samples = pydream_result.all_samples
    n_burnin = int(len(all_samples) * burn_in_fraction)
    samples = all_samples.iloc[n_burnin:].copy()
    
    all_params = list(sceua_result.best_parameters.keys())
    params_to_plot = [p for p in all_params if p in samples.columns]
    
    # Build table data
    table_data = []
    for param in params_to_plot:
        posterior = samples[param].values
        sceua_val = sceua_result.best_parameters[param]
        pydream_val = pydream_result.best_parameters[param]
        ci_low, ci_high = np.percentile(posterior, [2.5, 97.5])
        
        # Calculate percentage difference
        if sceua_val != 0:
            pct_diff = (pydream_val - sceua_val) / abs(sceua_val) * 100
        else:
            pct_diff = 0 if pydream_val == 0 else 100
        
        # Check if SCE-UA is within 95% CI
        in_ci = ci_low <= sceua_val <= ci_high
        
        table_data.append({
            'Parameter': param,
            'SCE-UA': sceua_val,
            'PyDREAM': pydream_val,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'Diff_Pct': pct_diff,
            'In_CI': in_ci
        })
    
    df = pd.DataFrame(table_data)
    
    # Color coding based on agreement
    colors = []
    for _, row in df.iterrows():
        if row['In_CI']:
            if abs(row['Diff_Pct']) < 10:
                colors.append('#C8E6C9')  # Light green - excellent agreement
            else:
                colors.append('#FFF9C4')  # Light yellow - good agreement
        else:
            colors.append('#FFCDD2')  # Light red - poor agreement
    
    # Format values
    formatted = {
        'Parameter': df['Parameter'].tolist(),
        'SCE-UA': [f'{v:.4g}' for v in df['SCE-UA']],
        'PyDREAM': [f'{v:.4g}' for v in df['PyDREAM']],
        '95% CI': [f'[{l:.4g}, {h:.4g}]' for l, h in zip(df['CI_Low'], df['CI_High'])],
        'Δ%': [f'{v:+.1f}%' for v in df['Diff_Pct']],
        'Agreement': ['✓ In CI' if x else '✗ Outside' for x in df['In_CI']]
    }
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Parameter</b>', '<b>SCE-UA</b>', '<b>PyDREAM</b>', 
                    '<b>95% CI</b>', '<b>Δ%</b>', '<b>Agreement</b>'],
            fill_color='#455A64',
            font=dict(color='white', size=11),
            align='center',
            height=30
        ),
        cells=dict(
            values=list(formatted.values()),
            fill_color=[['white'] * len(df), ['white'] * len(df), ['white'] * len(df),
                       ['white'] * len(df), colors, colors],
            font=dict(size=10),
            align=['left', 'right', 'right', 'center', 'right', 'center'],
            height=25
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f'<b>{obj_name}: Parameter Comparison Summary</b><br>' +
                 '<span style="font-size:11px; color:gray">Green = SCE-UA within PyDREAM 95% CI | ' +
                 'Yellow = In CI but >10% diff | Red = Outside CI</span>',
            font=dict(size=14),
            x=0.5
        ),
        width=900,
        height=min(800, 100 + len(df) * 28),
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

# %%
# =============================================================================
# COMBINED SUMMARY TABLES - ALL 13 OBJECTIVES IN ONE FIGURE
# =============================================================================

def create_combined_summary_tables(pydream_results, sceua_results, param_bounds, burn_in_fraction=0.5):
    """
    Create a single Plotly figure with all 13 summary tables as subplots.
    
    Layout: 4 rows x 4 columns matching the ridge plot organization
    - Row 1: NSE family (NSE, SqrtNSE, LogNSE, InvNSE)
    - Row 2: KGE family (KGE, KGE_sqrt, KGE_log, KGE_inv)
    - Row 3: KGE nonparametric (KGE_np, KGE_np_sqrt, KGE_np_log, KGE_np_inv)
    - Row 4: Legend, SDEB (sqrt column), Info, Organization
    """
    from plotly.subplots import make_subplots
    
    # Organize objectives by family (matching ridge plot)
    # Column order: Base, Sqrt, Log, Inverse
    objective_rows = [
        ['NSE', 'SqrtNSE', 'LogNSE', 'InvNSE'],
        ['KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv'],
        ['KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv'],
        [None, 'SDEB', None, None]  # SDEB in sqrt column
    ]
    
    row_titles = ['NSE Family', 'KGE Family', 'KGE Nonparametric', 'Other']
    
    # Create subplot specs (4x4 grid, all tables)
    specs = [[{"type": "table"} for _ in range(4)] for _ in range(4)]
    
    # Create subplot titles
    subplot_titles = []
    for row in objective_rows:
        for obj in row:
            if obj is not None:
                subplot_titles.append(f"<b>{obj}</b>")
            else:
                subplot_titles.append("")
    
    fig = make_subplots(
        rows=4, cols=4,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,  # Reduced to minimize row gaps
        horizontal_spacing=0.005  # Minimal horizontal spacing
    )
    
    def get_table_data(obj_name):
        """Generate table data for a single objective."""
        if (obj_name not in pydream_results or obj_name not in sceua_results or
            pydream_results[obj_name].all_samples is None):
            return None
        
        pydream_result = pydream_results[obj_name]
        sceua_result = sceua_results[obj_name]
        
        all_samples = pydream_result.all_samples
        n_burnin = int(len(all_samples) * burn_in_fraction)
        samples = all_samples.iloc[n_burnin:].copy()
        
        all_params = list(sceua_result.best_parameters.keys())
        params_to_plot = [p for p in all_params if p in samples.columns]
        
        # Build compact table data
        params = []
        sceua_vals = []
        pydream_vals = []
        agreements = []
        colors = []
        
        for param in params_to_plot:
            posterior = samples[param].values
            sceua_val = sceua_result.best_parameters[param]
            pydream_val = pydream_result.best_parameters[param]
            ci_low, ci_high = np.percentile(posterior, [2.5, 97.5])
            
            # Check if SCE-UA is within 95% CI
            in_ci = ci_low <= sceua_val <= ci_high
            
            # Calculate percentage difference
            if sceua_val != 0:
                pct_diff = abs((pydream_val - sceua_val) / sceua_val * 100)
            else:
                pct_diff = 0 if pydream_val == 0 else 100
            
            params.append(param)
            sceua_vals.append(f'{sceua_val:.3g}')
            pydream_vals.append(f'{pydream_val:.3g}')
            
            if in_ci:
                if pct_diff < 10:
                    agreements.append('✓')
                    colors.append('#C8E6C9')  # Light green
                else:
                    agreements.append('~')
                    colors.append('#FFF9C4')  # Light yellow
            else:
                agreements.append('✗')
                colors.append('#FFCDD2')  # Light red
        
        return {
            'params': params,
            'sceua': sceua_vals,
            'pydream': pydream_vals,
            'agreements': agreements,
            'colors': colors
        }
    
    # Add tables to subplots
    for row_idx, obj_row in enumerate(objective_rows):
        for col_idx, obj_name in enumerate(obj_row):
            # Skip None cells (will be filled with legend/info later)
            if obj_name is None:
                continue
                
            data = get_table_data(obj_name)
            
            if data is None:
                # Add empty table for missing data
                fig.add_trace(
                    go.Table(
                        header=dict(values=['No Data'], fill_color='#f0f0f0', height=25),
                        cells=dict(values=[['N/A']], fill_color='white', height=20)
                    ),
                    row=row_idx + 1, col=col_idx + 1
                )
                continue
            
            # Create compact table
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>Param</b>', '<b>SCE</b>', '<b>DRM</b>', '<b>Agr</b>'],
                        fill_color='#455A64',
                        font=dict(color='white', size=9),
                        align='center',
                        height=22
                    ),
                    cells=dict(
                        values=[data['params'], data['sceua'], data['pydream'], data['agreements']],
                        fill_color=[['white'] * len(data['params']), 
                                   ['white'] * len(data['params']),
                                   ['white'] * len(data['params']), 
                                   data['colors']],
                        font=dict(size=8),
                        align=['left', 'right', 'right', 'center'],
                        height=18
                    )
                ),
                row=row_idx + 1, col=col_idx + 1
            )
    
    # Add legend table in empty cell (row 4, col 1)
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Legend</b>'],
                fill_color='#455A64',
                font=dict(color='white', size=10),
                align='center',
                height=25
            ),
            cells=dict(
                values=[['✓ = In 95% CI, <10% diff',
                        '~ = In 95% CI, >10% diff', 
                        '✗ = Outside 95% CI',
                        '',
                        'SCE = SCE-UA',
                        'DRM = PyDREAM',
                        'Agr = Agreement']],
                fill_color=[['#C8E6C9', '#FFF9C4', '#FFCDD2', 'white', 'white', 'white', 'white']],
                font=dict(size=9),
                align='left',
                height=20
            )
        ),
        row=4, col=1
    )
    
    # SDEB table is in row 4, col 2 (placed by the loop above)
    
    # Add info in empty cell (row 4, col 3)
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>About This Figure</b>'],
                fill_color='#455A64',
                font=dict(color='white', size=10),
                align='center',
                height=25
            ),
            cells=dict(
                values=[['Compares SCE-UA point estimates',
                        'with PyDREAM posterior distributions',
                        'for all 22 Sacramento parameters',
                        '',
                        'Color indicates whether SCE-UA',
                        'falls within PyDREAM 95% CI']],
                fill_color='white',
                font=dict(size=9),
                align='left',
                height=20
            )
        ),
        row=4, col=3
    )
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Organization</b>'],
                fill_color='#455A64',
                font=dict(color='white', size=10),
                align='center',
                height=25
            ),
            cells=dict(
                values=[['Cols: Base, Sqrt, Log, Inv',
                        'Row 1: NSE variants',
                        'Row 2: KGE variants',
                        'Row 3: KGE nonparametric',
                        'Row 4: SDEB (sqrt col)',
                        '22 params per table']],
                fill_color='white',
                font=dict(size=9),
                align='left',
                height=20
            )
        ),
        row=4, col=4
    )
    
    fig.update_layout(
        title=dict(
            text='<b>Parameter Comparison Summary: All Objective Functions</b><br>' +
                 '<span style="font-size:11px; color:gray">' +
                 'Comparing SCE-UA point estimates with PyDREAM 95% credible intervals</span>',
            font=dict(size=16),
            x=0.5
        ),
        height=1800,  # Compact height to reduce row gaps
        width=1400,
        margin=dict(l=5, r=5, t=80, b=10),  # Minimal left/right margins
        showlegend=False
    )
    
    return fig

# %% [markdown]
# ---
# ## Generate Compact Visualizations
#
# We generate two types of compact visualizations:
# 1. **Combined Ridge Plot** - All 13 objectives in a single figure with subplots
# 2. **Combined Summary Tables** - All 13 tables in a single Plotly figure

# %%
# Generate COMBINED ridge plot for ALL 13 objective functions
print("=" * 70)
print("COMBINED RIDGE PLOT: ALL 13 OBJECTIVES")
print("=" * 70)

fig_combined = plot_all_ridge_posteriors(pydream_results, sceua_results, param_bounds)
if fig_combined:
    fig_combined.savefig(figures_dir / '06_ridge_all_objectives.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig_combined)

print("\nCombined ridge plot saved: figures/06_ridge_all_objectives.png")

# %%
# Generate COMBINED summary tables for ALL 13 objective functions
print("\n" + "=" * 70)
print("COMBINED SUMMARY TABLES: ALL 13 OBJECTIVES")
print("=" * 70)

fig_tables = create_combined_summary_tables(pydream_results, sceua_results, param_bounds)
if fig_tables:
    fig_tables.show()
    # Save as HTML for interactivity
    fig_tables.write_html(str(figures_dir / '06_summary_tables_all_objectives.html'))
    print(f"\nCombined tables saved: figures/06_summary_tables_all_objectives.html")

print("\n" + "=" * 70)
print("COMPACT VISUALIZATIONS COMPLETE")
print("=" * 70)
print("""
Generated visualizations:
  1. Combined Ridge Plot   - All 13 objectives in one figure (4x4 grid)
                            Blue bar = SCE-UA | Magenta bar = PyDREAM
                            Ridge color indicates agreement level
  2. Combined Summary Tables - All 13 tables in one Plotly figure (4x4 grid)
                              ✓ = Good agreement | ~ = Moderate | ✗ = Poor
""")


# %% [markdown]
# ---
# ## Visual Model Fit Comparison: SCE-UA vs PyDREAM
#
# A crucial aspect of comparing calibration algorithms is visually inspecting how well each
# algorithm's best parameters reproduce the observed hydrograph. This section provides
# interactive Plotly visualizations comparing:
#
# - **Hydrograph (Linear Scale)**: For inspecting peak flows
# - **Hydrograph (Log Scale)**: For inspecting low flows and recession behavior
# - **One-to-One Scatter (Log-Log)**: For assessing bias across the flow range
# - **Flow Duration Curves**: For overall flow distribution comparison
#
# Each objective function gets its own comparison figure, allowing you to see how the
# choice of objective function and algorithm affects the model fit.

# %%
# Generate simulations for all objective functions and both algorithms
print("=" * 70)
print("GENERATING MODEL SIMULATIONS FOR VISUAL COMPARISON")
print("=" * 70)

# Store simulated flows for both algorithms
sceua_simulations = {}
pydream_simulations = {}

# Get observed flow (after warmup)
obs_flow = cal_observed[WARMUP_DAYS:]
comparison_dates = cal_inputs.index[WARMUP_DAYS:]

print(f"\nSimulation period: {comparison_dates[0].date()} to {comparison_dates[-1].date()}")
print(f"Number of days: {len(obs_flow):,}")
print()

for obj_name in objectives.keys():
    # SCE-UA simulation
    if obj_name in sceua_results:
        sceua_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        sceua_model.set_parameters(sceua_results[obj_name].best_parameters)
        sceua_model.reset()
        sceua_sim = sceua_model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        sceua_simulations[obj_name] = sceua_sim
        sceua_status = "✓"
    else:
        sceua_status = "✗"
    
    # PyDREAM simulation
    if obj_name in pydream_results:
        pydream_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        pydream_model.set_parameters(pydream_results[obj_name].best_parameters)
        pydream_model.reset()
        pydream_sim = pydream_model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        pydream_simulations[obj_name] = pydream_sim
        pydream_status = "✓"
    else:
        pydream_status = "✗"
    
    print(f"  {obj_name:<15} SCE-UA: {sceua_status}  PyDREAM: {pydream_status}")

print(f"\nGenerated {len(sceua_simulations)} SCE-UA simulations")
print(f"Generated {len(pydream_simulations)} PyDREAM simulations")

# %%
def create_algorithm_comparison_plot(obj_name, obs, sceua_sim, pydream_sim, dates):
    """
    Create a 2x2 interactive Plotly comparison figure for SCE-UA vs PyDREAM.
    
    Includes:
    - Hydrograph (Linear Scale)
    - Hydrograph (Log Scale)  
    - One-to-One Scatter (Log-Log axes)
    - Flow Duration Curves
    
    Args:
        obj_name: Name of the objective function
        obs: Observed flow array
        sceua_sim: SCE-UA simulated flow array
        pydream_sim: PyDREAM simulated flow array
        dates: DatetimeIndex for x-axis
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Hydrograph (Linear Scale)',
            'Hydrograph (Log Scale)',
            'One-to-One Scatter (Log-Log)',
            'Flow Duration Curves'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Color scheme
    obs_color = 'black'
    sceua_color = '#1f77b4'  # Blue
    pydream_color = '#d62728'  # Red
    
    # Calculate metrics for title
    from pyrrm.calibration.objective_functions import calculate_metrics
    sceua_metrics = calculate_metrics(sceua_sim, obs)
    pydream_metrics = calculate_metrics(pydream_sim, obs)
    
    # =========================================================================
    # 1. Hydrograph - Linear Scale (top-left)
    # =========================================================================
    fig.add_trace(
        go.Scatter(
            x=dates, y=obs, name='Observed',
            line=dict(color=obs_color, width=1),
            legendgroup='obs'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=sceua_sim, 
            name=f'SCE-UA (NSE={sceua_metrics["NSE"]:.3f})',
            line=dict(color=sceua_color, width=1, dash='solid'),
            legendgroup='sceua'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=pydream_sim,
            name=f'PyDREAM (NSE={pydream_metrics["NSE"]:.3f})',
            line=dict(color=pydream_color, width=1, dash='solid'),
            legendgroup='pydream'
        ),
        row=1, col=1
    )
    
    # =========================================================================
    # 2. Hydrograph - Log Scale (top-right)
    # =========================================================================
    fig.add_trace(
        go.Scatter(
            x=dates, y=obs, name='Observed',
            line=dict(color=obs_color, width=1),
            legendgroup='obs', showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=sceua_sim, name='SCE-UA',
            line=dict(color=sceua_color, width=1),
            legendgroup='sceua', showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=pydream_sim, name='PyDREAM',
            line=dict(color=pydream_color, width=1),
            legendgroup='pydream', showlegend=False
        ),
        row=1, col=2
    )
    
    # =========================================================================
    # 3. One-to-One Scatter - Log-Log axes (bottom-left)
    # =========================================================================
    # Add small epsilon to avoid log(0)
    epsilon = 0.01
    obs_log = np.maximum(obs, epsilon)
    sceua_log = np.maximum(sceua_sim, epsilon)
    pydream_log = np.maximum(pydream_sim, epsilon)
    
    fig.add_trace(
        go.Scatter(
            x=obs_log, y=sceua_log, mode='markers',
            name='SCE-UA',
            marker=dict(color=sceua_color, size=3, opacity=0.4),
            legendgroup='sceua', showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=obs_log, y=pydream_log, mode='markers',
            name='PyDREAM',
            marker=dict(color=pydream_color, size=3, opacity=0.4),
            legendgroup='pydream', showlegend=False
        ),
        row=2, col=1
    )
    
    # 1:1 line
    max_flow = max(obs.max(), sceua_sim.max(), pydream_sim.max())
    min_flow = max(epsilon, min(obs.min(), sceua_sim.min(), pydream_sim.min()))
    fig.add_trace(
        go.Scatter(
            x=[min_flow, max_flow], y=[min_flow, max_flow],
            mode='lines', name='1:1 Line',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # =========================================================================
    # 4. Flow Duration Curves (bottom-right)
    # =========================================================================
    obs_sorted = np.sort(obs)[::-1]
    sceua_sorted = np.sort(sceua_sim)[::-1]
    pydream_sorted = np.sort(pydream_sim)[::-1]
    exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    
    fig.add_trace(
        go.Scatter(
            x=exceedance, y=obs_sorted, name='Observed FDC',
            line=dict(color=obs_color, width=2),
            legendgroup='obs', showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=exceedance, y=sceua_sorted, name='SCE-UA FDC',
            line=dict(color=sceua_color, width=2),
            legendgroup='sceua', showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=exceedance, y=pydream_sorted, name='PyDREAM FDC',
            line=dict(color=pydream_color, width=2),
            legendgroup='pydream', showlegend=False
        ),
        row=2, col=2
    )
    
    # =========================================================================
    # Update axes
    # =========================================================================
    # Row 1 - Hydrographs
    fig.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
    
    # Row 2 - Scatter and FDC
    fig.update_xaxes(title_text="Observed (ML/day)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Simulated (ML/day)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Exceedance (%)", row=2, col=2)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=2)
    
    # =========================================================================
    # Layout
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=f'<b>Model Fit Comparison: {obj_name}</b><br>' +
                 f'<span style="font-size:12px; color:gray">' +
                 f'SCE-UA NSE={sceua_metrics["NSE"]:.3f}, KGE={sceua_metrics["KGE"]:.3f} | ' +
                 f'PyDREAM NSE={pydream_metrics["NSE"]:.3f}, KGE={pydream_metrics["KGE"]:.3f}</span>',
            font=dict(size=16),
            x=0.5
        ),
        height=700,
        width=1200,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified'
    )
    
    return fig


def create_combined_fit_comparison(objectives_list, obs, sceua_sims, pydream_sims, dates):
    """
    Create a combined figure showing all objective functions in a grid.
    Each row is one objective function with 4 panels.
    
    Args:
        objectives_list: List of objective function names
        obs: Observed flow array
        sceua_sims: Dict of SCE-UA simulations by objective name
        pydream_sims: Dict of PyDREAM simulations by objective name
        dates: DatetimeIndex for x-axis
    
    Returns:
        Plotly figure
    """
    n_objectives = len(objectives_list)
    
    # Create subplot titles
    subplot_titles = []
    for obj in objectives_list:
        subplot_titles.extend([
            f'{obj} - Linear',
            f'{obj} - Log',
            f'{obj} - Scatter',
            f'{obj} - FDC'
        ])
    
    fig = make_subplots(
        rows=n_objectives, cols=4,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter"} for _ in range(4)] for _ in range(n_objectives)],
        vertical_spacing=0.03,
        horizontal_spacing=0.04,
        row_heights=[1/n_objectives] * n_objectives
    )
    
    # Colors
    obs_color = 'black'
    sceua_color = '#1f77b4'
    pydream_color = '#d62728'
    epsilon = 0.01
    
    # Calculate FDC data once
    obs_sorted = np.sort(obs)[::-1]
    exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    max_obs = obs.max()
    min_obs = max(epsilon, obs.min())
    
    for row_idx, obj_name in enumerate(objectives_list, 1):
        if obj_name not in sceua_sims or obj_name not in pydream_sims:
            continue
        
        sceua_sim = sceua_sims[obj_name]
        pydream_sim = pydream_sims[obj_name]
        
        # Show legend only for first row
        show_legend = (row_idx == 1)
        
        # ----- Column 1: Hydrograph Linear -----
        fig.add_trace(
            go.Scatter(x=dates, y=obs, name='Observed',
                      line=dict(color=obs_color, width=0.8),
                      legendgroup='obs', showlegend=show_legend),
            row=row_idx, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=sceua_sim, name='SCE-UA',
                      line=dict(color=sceua_color, width=0.8),
                      legendgroup='sceua', showlegend=show_legend),
            row=row_idx, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=pydream_sim, name='PyDREAM',
                      line=dict(color=pydream_color, width=0.8),
                      legendgroup='pydream', showlegend=show_legend),
            row=row_idx, col=1
        )
        
        # ----- Column 2: Hydrograph Log -----
        fig.add_trace(
            go.Scatter(x=dates, y=obs, name='Observed',
                      line=dict(color=obs_color, width=0.8),
                      legendgroup='obs', showlegend=False),
            row=row_idx, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=sceua_sim, name='SCE-UA',
                      line=dict(color=sceua_color, width=0.8),
                      legendgroup='sceua', showlegend=False),
            row=row_idx, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=pydream_sim, name='PyDREAM',
                      line=dict(color=pydream_color, width=0.8),
                      legendgroup='pydream', showlegend=False),
            row=row_idx, col=2
        )
        fig.update_yaxes(type="log", row=row_idx, col=2)
        
        # ----- Column 3: Scatter Log-Log -----
        obs_log = np.maximum(obs, epsilon)
        sceua_log = np.maximum(sceua_sim, epsilon)
        pydream_log = np.maximum(pydream_sim, epsilon)
        
        fig.add_trace(
            go.Scatter(x=obs_log, y=sceua_log, mode='markers', name='SCE-UA',
                      marker=dict(color=sceua_color, size=2, opacity=0.3),
                      legendgroup='sceua', showlegend=False),
            row=row_idx, col=3
        )
        fig.add_trace(
            go.Scatter(x=obs_log, y=pydream_log, mode='markers', name='PyDREAM',
                      marker=dict(color=pydream_color, size=2, opacity=0.3),
                      legendgroup='pydream', showlegend=False),
            row=row_idx, col=3
        )
        # 1:1 line
        max_flow = max(max_obs, sceua_sim.max(), pydream_sim.max())
        fig.add_trace(
            go.Scatter(x=[min_obs, max_flow], y=[min_obs, max_flow],
                      mode='lines', line=dict(color='gray', dash='dash', width=1),
                      showlegend=False),
            row=row_idx, col=3
        )
        fig.update_xaxes(type="log", row=row_idx, col=3)
        fig.update_yaxes(type="log", row=row_idx, col=3)
        
        # ----- Column 4: FDC -----
        sceua_sorted = np.sort(sceua_sim)[::-1]
        pydream_sorted = np.sort(pydream_sim)[::-1]
        
        fig.add_trace(
            go.Scatter(x=exceedance, y=obs_sorted, name='Observed FDC',
                      line=dict(color=obs_color, width=1.5),
                      legendgroup='obs', showlegend=False),
            row=row_idx, col=4
        )
        fig.add_trace(
            go.Scatter(x=exceedance, y=sceua_sorted, name='SCE-UA FDC',
                      line=dict(color=sceua_color, width=1.5),
                      legendgroup='sceua', showlegend=False),
            row=row_idx, col=4
        )
        fig.add_trace(
            go.Scatter(x=exceedance, y=pydream_sorted, name='PyDREAM FDC',
                      line=dict(color=pydream_color, width=1.5),
                      legendgroup='pydream', showlegend=False),
            row=row_idx, col=4
        )
        fig.update_yaxes(type="log", row=row_idx, col=4)
    
    fig.update_layout(
        title=dict(
            text='<b>Model Fit Comparison: All 13 Objective Functions</b><br>' +
                 '<span style="font-size:12px; color:gray">' +
                 'SCE-UA (blue) vs PyDREAM (red) | Columns: Linear, Log, Scatter, FDC</span>',
            font=dict(size=18),
            x=0.5
        ),
        height=250 * n_objectives,
        width=1400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.005,
            xanchor='center',
            x=0.5
        ),
        showlegend=True
    )
    
    return fig

print("Comparison plotting functions defined!")

# %% [markdown]
# ### Generate Comparison Figures
#
# First, we generate all comparison plots and calculate metrics for each objective function.

# %%
# Generate individual comparison plots for all objective functions
print("=" * 70)
print("GENERATING MODEL FIT COMPARISON FIGURES")
print("=" * 70)

comparison_figs = {}
objectives_with_both = [obj for obj in objectives.keys() 
                       if obj in sceua_simulations and obj in pydream_simulations]

print(f"\nGenerating plots for {len(objectives_with_both)} objective functions...")
print()

for obj_name in objectives_with_both:
    fig = create_algorithm_comparison_plot(
        obj_name=obj_name,
        obs=obs_flow,
        sceua_sim=sceua_simulations[obj_name],
        pydream_sim=pydream_simulations[obj_name],
        dates=comparison_dates
    )
    comparison_figs[obj_name] = fig
    
    # Calculate metrics for display
    sceua_metrics = calculate_metrics(sceua_simulations[obj_name], obs_flow)
    pydream_metrics = calculate_metrics(pydream_simulations[obj_name], obs_flow)
    
    print(f"{obj_name}:")
    print(f"  SCE-UA  - NSE: {sceua_metrics['NSE']:.3f}, KGE: {sceua_metrics['KGE']:.3f}, PBIAS: {sceua_metrics['PBIAS']:+.1f}%")
    print(f"  PyDREAM - NSE: {pydream_metrics['NSE']:.3f}, KGE: {pydream_metrics['KGE']:.3f}, PBIAS: {pydream_metrics['PBIAS']:+.1f}%")
    
    # Determine winner for this objective
    nse_diff = pydream_metrics['NSE'] - sceua_metrics['NSE']
    winner = "PyDREAM" if nse_diff > 0.001 else ("SCE-UA" if nse_diff < -0.001 else "Tie")
    print(f"  → NSE Winner: {winner} (diff: {nse_diff:+.4f})")
    print()

print(f"\n✓ Generated {len(comparison_figs)} comparison figures")

# %% [markdown]
# ### Combined Overview: All 13 Objectives
#
# We start with the big picture - a mega-figure showing all 13 objective functions in a 
# single scrollable view. This allows for quick visual comparison of how different 
# objectives affect model fit.
#
# Each row shows one objective function with 4 diagnostic panels:
# - **Linear hydrograph**: Good for peak flow comparison
# - **Log hydrograph**: Good for low flow and recession comparison  
# - **Log-log scatter**: Shows bias across all flow magnitudes
# - **Flow duration curves**: Shows overall flow distribution fit

# %%
# Generate combined figure with all objectives
print("=" * 70)
print("COMBINED MODEL FIT COMPARISON: ALL 13 OBJECTIVES")
print("=" * 70)

fig_all = create_combined_fit_comparison(
    objectives_list=objectives_with_both,
    obs=obs_flow,
    sceua_sims=sceua_simulations,
    pydream_sims=pydream_simulations,
    dates=comparison_dates
)

fig_all.show()

# Save as HTML for interactivity
fig_all.write_html(str(figures_dir / '06_model_fits_all_objectives.html'))
print(f"\n✓ Combined figure saved to: figures/06_model_fits_all_objectives.html")

# %% [markdown]
# ### Individual Objective Function Comparisons
#
# Below we show larger, more detailed interactive plots for each family of objective 
# functions. These are easier to inspect individually than the combined overview above.

# %%
# Display NSE-family comparisons (NSE, LogNSE, InvNSE, SqrtNSE)
print("=" * 70)
print("NSE-FAMILY OBJECTIVE FUNCTIONS")
print("=" * 70)

nse_family = ['NSE', 'LogNSE', 'InvNSE', 'SqrtNSE']
for obj_name in nse_family:
    if obj_name in comparison_figs:
        print(f"\n{obj_name}:")
        comparison_figs[obj_name].show()

# %%
# Display KGE-family comparisons (KGE, KGE_inv, KGE_sqrt, KGE_log)
print("=" * 70)
print("KGE-FAMILY OBJECTIVE FUNCTIONS")
print("=" * 70)

kge_family = ['KGE', 'KGE_inv', 'KGE_sqrt', 'KGE_log']
for obj_name in kge_family:
    if obj_name in comparison_figs:
        print(f"\n{obj_name}:")
        comparison_figs[obj_name].show()

# %%
# Display KGE-nonparametric family comparisons
print("=" * 70)
print("KGE NON-PARAMETRIC FAMILY OBJECTIVE FUNCTIONS")
print("=" * 70)

kge_np_family = ['KGE_np', 'KGE_np_inv', 'KGE_np_sqrt', 'KGE_np_log']
for obj_name in kge_np_family:
    if obj_name in comparison_figs:
        print(f"\n{obj_name}:")
        comparison_figs[obj_name].show()

# %%
# Display SDEB (composite objective)
print("=" * 70)
print("COMPOSITE OBJECTIVE FUNCTION (SDEB)")
print("=" * 70)

if 'SDEB' in comparison_figs:
    print("\nSDEB (Spectral Decomposition-based Efficiency):")
    comparison_figs['SDEB'].show()

# %% [markdown]
# ### Key Observations from Visual Comparison
#
# Looking at the hydrographs and diagnostics above, we can observe:
#
# **High-Flow Emphasis Objectives (NSE, KGE, KGE_np):**
# - Both algorithms capture peak flows well
# - May underestimate some low flow periods (visible in log-scale hydrographs)
# - Flow duration curves typically match well at high exceedance percentiles
#
# **Low-Flow Emphasis Objectives (LogNSE, InvNSE, KGE_inv, KGE_log):**
# - Better performance during recession and baseflow periods
# - May not capture the highest peaks as accurately
# - Flow duration curves match better at low exceedance percentiles
#
# **Balanced Objectives (SqrtNSE, KGE_sqrt, SDEB):**
# - Compromise between high and low flow performance
# - Often provide the most visually satisfying overall fit
# - Flow duration curves typically match across the full range
#
# **Algorithm Differences:**
# - SCE-UA and PyDREAM often produce very similar fits when NSE is high
# - Differences are more apparent when the objective function is challenging
# - Check the scatter plots for systematic biases between algorithms

# %% [markdown]
# ---
# ## Summary
#
# This notebook compared SCE-UA (optimization) and PyDREAM (MCMC) calibration algorithms
# across 13 objective functions for the Sacramento model.
#
# **Key outputs:**
# - Combined Ridge Plot showing posterior distributions with agreement coloring
# - Combined Summary Tables showing parameter comparisons across all objectives
# - Interactive model fit comparisons for all 13 objective functions
# - Combined mega-figure saved as HTML for external viewing

# %%
print("=" * 70)
print("ALGORITHM COMPARISON COMPLETE")
print("=" * 70)
print("""
Generated outputs:
  1. Combined Ridge Plot     - figures/06_ridge_posteriors_all_objectives.png
  2. Combined Summary Tables - figures/06_summary_tables_all_objectives.html
  3. Model Fit Comparisons   - figures/06_model_fits_all_objectives.html

Key findings:
  • SCE-UA provides fast point estimates
  • PyDREAM provides full posterior distributions  
  • Agreement coloring shows parameter identifiability
  • Green ridges = SCE-UA within posterior (good agreement)
  • Red ridges = SCE-UA outside posterior (algorithms disagree)
  
Visual comparison highlights:
  • Linear hydrographs show peak flow performance
  • Log hydrographs reveal low flow and recession behavior
  • Log-log scatter plots expose systematic biases
  • Flow duration curves summarize overall distribution fit
""")
