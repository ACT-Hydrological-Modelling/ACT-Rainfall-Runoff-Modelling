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
# | 2 | NSE_log | Transformed | All flows |
# | 3 | NSE_inv (1/Q) | Transformed | Low flows |
# | 4 | NSE_sqrt (√Q) | Transformed | Balanced |
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
#
# Calibrations in this notebook use the **NUMBA-accelerated Sacramento** backend.
# Numba must be installed (`pip install numba`); the notebook will raise an error
# at run time if it is not available, to avoid long runs on the slow Python path.

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import pickle

# Interactive visualizations
import plotly.graph_objects as go

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
from pyrrm.models import NUMBA_AVAILABLE
from pyrrm.calibration import (
    CalibrationRunner,
    CalibrationResult,
    CalibrationReport,
    export_report,
    PYDREAM_AVAILABLE,
    NUMPYRO_AVAILABLE,
)
from pyrrm.calibration.objective_functions import GaussianLikelihood, TransformedGaussianLikelihood
from pyrrm.analysis.diagnostics import compute_diagnostics, print_diagnostics
from pyrrm.objectives import (
    NSE, KGE, KGENonParametric, FlowTransformation, SDEB
)

# ArviZ-based MCMC visualisation (shared with NUTS notebooks 13/14)
try:
    import arviz as az
    from pyrrm.visualization.mcmc_plots import (
        dream_result_to_inference_data,
        plot_mcmc_traces,
        plot_dream_traces,
        plot_posterior_pairs,
        plot_rhat_summary,
        plot_rhat_from_pydream,
    )
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False

print("\npyrrm components imported!")
print(f"\nModel acceleration:")
print(f"  Numba JIT: {'ACTIVE' if NUMBA_AVAILABLE else 'not available (pip install numba)'}")

# Require NUMBA for Sacramento so calibrations use the accelerated backend
if not NUMBA_AVAILABLE:
    raise RuntimeError(
        "This notebook requires the NUMBA-accelerated Sacramento backend. "
        "Install numba: pip install numba. Then restart the kernel."
    )
print("  Sacramento: using NUMBA-accelerated backend (required for this notebook)")

print(f"\nAvailable calibration backends:")
print(f"  SCE-UA (vendored): always available")
print(f"  SciPy (DE, Dual Annealing): always available")
print(f"  PyDREAM (MT-DREAM(ZS)): {PYDREAM_AVAILABLE}")
print(f"  NumPyro NUTS: {NUMPYRO_AVAILABLE}")
print(f"\nVisualization:")
print(f"  ArviZ MCMC diagnostics: {'ACTIVE' if ARVIZ_AVAILABLE else 'not available (pip install arviz)'}")

if not PYDREAM_AVAILABLE:
    print("\nWARNING: PyDREAM not installed!")
    print("Install with: pip install pydream")
    print("This notebook requires PyDREAM for the algorithm comparison.")

OUTPUT_DIR = Path('../test_data/06_algorithm_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figures_dir = OUTPUT_DIR / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
NB02_REPORTS_DIR = Path('../test_data/02_calibration_quickstart/reports')
print(f"\nFigures will be saved to: {figures_dir.absolute()}")

# %% [markdown]
# ---
# ## Load SCE-UA Results from Notebook 02
#
# First, we load all the calibration results from the SCE-UA runs in Notebook 02.
# These are stored as `CalibrationReport` pickle files.

# %%
# Define report file paths
REPORTS_DIR = NB02_REPORTS_DIR

# Map of objective function names to report files
SCEUA_REPORTS = {
    'NSE': '410734_sacramento_nse_sceua.pkl',
    'NSE_log': '410734_sacramento_nse_sceua_log.pkl',
    'NSE_inv': '410734_sacramento_nse_sceua_inverse.pkl',
    'NSE_sqrt': '410734_sacramento_nse_sceua_sqrt.pkl',
    'SDEB': '410734_sacramento_sdeb_sceua.pkl',
    'KGE': '410734_sacramento_kge_sceua.pkl',
    'KGE_inv': '410734_sacramento_kge_sceua_inverse.pkl',
    'KGE_sqrt': '410734_sacramento_kge_sceua_sqrt.pkl',
    'KGE_log': '410734_sacramento_kge_sceua_log.pkl',
    'KGE_np': '410734_sacramento_kgenp_sceua.pkl',
    'KGE_np_inv': '410734_sacramento_kgenp_sceua_inverse.pkl',
    'KGE_np_sqrt': '410734_sacramento_kgenp_sceua_sqrt.pkl',
    'KGE_np_log': '410734_sacramento_kgenp_sceua_log.pkl',
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
from pyrrm.data import load_catchment_data

DATA_DIR = Path('../data/410734')
CATCHMENT_AREA_KM2 = 516.62667
WARMUP_DAYS = 365

cal_inputs, cal_observed = load_catchment_data(
    precipitation_file=DATA_DIR / 'Default Input Set - Rain_QBN01.csv',
    pet_file=DATA_DIR / 'Default Input Set - Mwet_QBN01.csv',
    observed_file=DATA_DIR / '410734_output_SDmodel.csv',
    observed_value_column='Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)',
)

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
    'NSE_log': NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
    'NSE_inv': NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
    'NSE_sqrt': NSE(transform=FlowTransformation('sqrt')),
    
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
for name in ['NSE', 'NSE_log', 'NSE_inv', 'NSE_sqrt']:
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
# | NSE_sqrt, KGE_sqrt, KGE_np_sqrt | 'sqrt' | Balanced |
# | NSE_log, KGE_log, KGE_np_log | 'log' | Low flows |
# | NSE_inv, KGE_inv, KGE_np_inv | 'inverse' | Very low flows |
# | SDEB | 'sqrt' | Balanced (default) |

# %%
# Mapping from objective function names to equivalent likelihood transforms
# This ensures PyDREAM calibration uses the same flow emphasis as the objective
LIKELIHOOD_TRANSFORM_MAPPING = {
    # NSE-based - transform matches the NSE variant
    'NSE': 'none',           # Standard NSE → no transform (high-flow emphasis)
    'NSE_log': 'log',        # NSE_log → log transform (low-flow emphasis)
    'NSE_inv': 'inverse',    # NSE_inv → inverse transform (very low-flow emphasis)
    'NSE_sqrt': 'sqrt',      # NSE_sqrt → sqrt transform (balanced)
    
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
# PyDREAM configuration - OPTIMIZED FOR FAST CONVERGENCE
# Key optimizations:
# - Fewer chains (3-4) for faster convergence while maintaining GR diagnostics
# - DEpairs=2 for better exploration with fewer chains
# - Adaptive gamma for better step size adaptation
# - Reduced iterations for synthetic test (converges faster on clean data)
# - Convergence threshold: GR < 1.2 (standard MCMC criterion)

DEMO_MODE = True  # Set to False for full production runs

if DEMO_MODE:
    # Demo: enough iterations for 22-parameter Sacramento to converge on real data
    PYDREAM_ITERATIONS = 15000  # Max per chain (early stopping when GR < 1.05)
    PYDREAM_CHAINS = 5          # 5 chains (minimum for DEpairs=2)
    PYDREAM_MULTITRY = 3        # Multi-try for better mixing
    PYDREAM_DEPAIRS = 2         # DE pairs for better exploration
    PYDREAM_SNOOKER = 0.15      # Snooker probability for mode jumping
    PYDREAM_ADAPT_GAMMA = False # Disabled due to PyDREAM bug
    PYDREAM_BATCH_SIZE = 1000   # Check GR convergence every 1000 iterations
    PYDREAM_MIN_ITER = 4000     # Allow burn-in before checking convergence
    PYDREAM_PATIENCE = 2        # 2 consecutive converged batches to stop
    PYDREAM_POST_CONV = 1500    # Extra samples after convergence (better behaved chains)
    print("DEMO MODE: 15k max iterations with early stopping (GR < 1.05)")
else:
    # Production settings for publication-quality posteriors
    PYDREAM_ITERATIONS = 25000  # Max per chain
    PYDREAM_CHAINS = 5          # 5 chains for robust Gelman-Rubin
    PYDREAM_MULTITRY = 4        # Multi-try for better mixing
    PYDREAM_DEPAIRS = 2         # DE pairs for exploration
    PYDREAM_SNOOKER = 0.15      # Snooker probability
    PYDREAM_ADAPT_GAMMA = False # Disabled due to PyDREAM bug
    PYDREAM_BATCH_SIZE = 2000   # Check GR convergence every 2000 iterations
    PYDREAM_MIN_ITER = 6000     # Allow burn-in before checking convergence
    PYDREAM_PATIENCE = 2        # 2 consecutive converged batches to stop
    PYDREAM_POST_CONV = 2500    # Extra samples after convergence (better behaved chains)
    print("PRODUCTION MODE: 25k max iterations with early stopping (GR < 1.05)")

# Convergence: strict threshold so "at stop" R-hats are all ≤ 1.05
PYDREAM_CONVERGENCE_THRESHOLD = 1.05

print(f"\nPyDREAM Configuration (Real Data — 22 Parameters):")
print(f"  Max iterations per chain: {PYDREAM_ITERATIONS}")
print(f"  Number of chains: {PYDREAM_CHAINS} (min: {2*PYDREAM_DEPAIRS+1} for DEpairs={PYDREAM_DEPAIRS})")
print(f"  Multi-try samples: {PYDREAM_MULTITRY}")
print(f"  DE pairs: {PYDREAM_DEPAIRS} (for better exploration)")
print(f"  Snooker probability: {PYDREAM_SNOOKER:.1%}")
print(f"  Convergence threshold: GR < {PYDREAM_CONVERGENCE_THRESHOLD}")
print(f"  Batch early stopping: every {PYDREAM_BATCH_SIZE} iter, "
      f"min {PYDREAM_MIN_ITER}, patience {PYDREAM_PATIENCE}")
print(f"  Post-convergence samples: {PYDREAM_POST_CONV}")
print(f"  Max total samples: ~{PYDREAM_ITERATIONS * PYDREAM_CHAINS:,}")

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
    'adimp': 0.0,     # Additional impervious area
    'sarva': 0.0,     # Riparian vegetation area fraction
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
# We calibrate the **6 most important Sacramento parameters** (the water
# balance stores and upper zone drainage rate) while holding the remaining
# 16 at their true values. This keeps the synthetic test fast and focused
# on demonstrating algorithm behaviour.  The real-data section below
# calibrates all 22 parameters.

# %%
# Calibrate only the 6 most important Sacramento parameters (dominant water
# balance stores and drainage).  The remaining 16 are fixed at their true
# values — this keeps the synthetic test fast and focused on demonstrating
# that both algorithms can recover the key parameters.
_sac_synth = Sacramento(catchment_area_km2=SYNTHETIC_CATCHMENT_AREA)
full_bounds = _sac_synth.get_parameter_bounds()

CALIBRATION_PARAMS = ['uztwm', 'uzfwm', 'lztwm', 'lzfpm', 'lzfsm', 'uzk']
subset_bounds = {p: full_bounds[p] for p in CALIBRATION_PARAMS}

print("=" * 70)
print("CALIBRATION SETUP (SYNTHETIC DATA)")
print("=" * 70)
print(f"\nCalibrating {len(CALIBRATION_PARAMS)} key Sacramento parameters:")
for p in CALIBRATION_PARAMS:
    bounds = subset_bounds[p]
    true_val = TRUE_SAC_PARAMS.get(p)
    true_str = f"{true_val:.3f}" if true_val is not None else "N/A"
    print(f"  {p:8s}: bounds [{bounds[0]:>8.3f}, {bounds[1]:>8.3f}], true = {true_str}")

# Fixed parameters (not calibrated) — held at true values
fixed_params = {k: v for k, v in TRUE_SAC_PARAMS.items() if k not in CALIBRATION_PARAMS}
print(f"\nFixed at true values ({len(fixed_params)} params): {', '.join(fixed_params.keys())}")

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
    parameter_bounds=subset_bounds,  # 6 key parameters
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
#   - 'log':  Low flow emphasis - equivalent to NSE_log
#   - 'inverse': Very low flow emphasis - equivalent to NSE_inv
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

# Use CalibrationRunner's run_sceua_direct method (n_complexes = 2n+1 for n params)
n_params_synth = len(CALIBRATION_PARAMS)
n_complexes_synth = 2 * n_params_synth + 1
print(f"\nRunning SCE-UA with {n_params_synth} key parameters...")
print(f"  Max evaluations: 20000")
print(f"  n_complexes: {n_complexes_synth} (2n+1 for good exploration)")
print("  Using CalibrationRunner.run_sceua_direct()")

start_time = time.time()

sceua_result = synthetic_runner_sceua.run_sceua_direct(
    max_evals=20000,  # Generous budget for tight convergence
    max_tolerant_iter=200,  # Allow more patience before early stop
    n_complexes=n_complexes_synth,  # 2n+1 per Duan et al. (1994)
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
    # PyDREAM hyperparameters - BATCH MODE (with early stopping), 6 key params
    SYNTHETIC_ITERATIONS = 8000      # Generous budget for 6 parameters
    SYNTHETIC_CHAINS = 3              # 3 chains (minimum for DEpairs=1: 2*1+1=3)
    SYNTHETIC_MULTITRY = 1            # Standard DREAM (no multi-try for speed)
    SYNTHETIC_DEPAIRS = 1             # DE pairs
    SYNTHETIC_SNOOKER = 0.10          # Snooker probability (10% default)
    SYNTHETIC_CONVERGENCE = 1.05      # Stricter R-hat threshold
    SYNTHETIC_BATCH_SIZE = 500        # Check convergence every 500 iterations
    SYNTHETIC_MIN_ITER = 1500         # Allow burn-in before checking convergence
    SYNTHETIC_PATIENCE = 3            # Stop after 3 consecutive converged batches

    print(f"\nRunning PyDREAM (BATCH MODE with early stopping) with {len(CALIBRATION_PARAMS)} key parameters...")
    print(f"  Max iterations per chain: {SYNTHETIC_ITERATIONS}")
    print(f"  Number of chains: {SYNTHETIC_CHAINS}")
    print(f"  Multi-try samples: {SYNTHETIC_MULTITRY}")
    print(f"  DE pairs: {SYNTHETIC_DEPAIRS}")
    print(f"  Snooker probability: {SYNTHETIC_SNOOKER:.1%}")
    print(f"  Convergence threshold: GR < {SYNTHETIC_CONVERGENCE}")
    print(f"  Batch size: {SYNTHETIC_BATCH_SIZE} | Min iter: {SYNTHETIC_MIN_ITER} | Patience: {SYNTHETIC_PATIENCE}")
    print("  Using CalibrationRunner.run_pydream() with early stopping")
    
    start_time = time.time()
    
    # Force stdout flush to ensure progress is visible
    import sys
    sys.stdout.flush()
    
    # Ensure figures directory exists (fallback if setup cell wasn't run)
    # figures_dir already defined at top of notebook
    
    # Use dbname for CSV-based progress tracking - monitor externally with:
    # tail -f figures/pydream_synthetic_progress.csv
    pydream_progress_file = figures_dir / 'pydream_synthetic_progress'
    
    print(f"\n  Progress file: {pydream_progress_file}.csv")
    print(f"  Monitor externally with: tail -f {pydream_progress_file}.csv")
    print("-" * 60)
    sys.stdout.flush()
    
    # Run PyDREAM in BATCH MODE (with early stopping)
    pydream_result = synthetic_runner_pydream.run_pydream(
        n_iterations=SYNTHETIC_ITERATIONS,
        n_chains=SYNTHETIC_CHAINS,
        multitry=SYNTHETIC_MULTITRY,
        snooker=SYNTHETIC_SNOOKER,
        DEpairs=SYNTHETIC_DEPAIRS,
        convergence_threshold=SYNTHETIC_CONVERGENCE,
        # Batch mode settings for early stopping
        batch_size=SYNTHETIC_BATCH_SIZE,
        min_iterations=SYNTHETIC_MIN_ITER,
        patience=SYNTHETIC_PATIENCE,
        post_convergence_iterations=500,  # Extra samples after convergence
        parallel=False,           # Multi-try parallelism off; Numba evals too fast to benefit
        dbname=str(pydream_progress_file),  # CSV progress tracking
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
# ### ArviZ Posterior Diagnostics (Synthetic Test)
#
# Professional MCMC diagnostics using ArviZ — trace plots show chain
# mixing and marginal densities, pair plots reveal parameter correlations.
# **Only the last 1000 samples per chain (post-convergence)** are used for
# these posterior plots.

# %%
if PYDREAM_AVAILABLE and ARVIZ_AVAILABLE and pydream_best is not None:
    print("=" * 70)
    print("ARVIZ POSTERIOR DIAGNOSTICS — SYNTHETIC DREAM")
    print("=" * 70)

    idata_synth = dream_result_to_inference_data(
        pydream_result, burn_fraction=0.3, post_convergence_draws=1000
    )
    all_params_synth = list(idata_synth.posterior.data_vars)

    print(f"\nTrace + posterior (all {len(all_params_synth)} params, full trajectory, post-convergence highlighted):")
    plot_dream_traces(
        pydream_result, var_names=all_params_synth,
        param_bounds=subset_bounds,
        burn_fraction=0.3, post_convergence_draws=1000, kde_bw=1.5,
    )
    plt.savefig(figures_dir / '05_synth_dream_traces.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPair plot (all {len(all_params_synth)} params — posterior correlations, smooth KDE):")
    plot_posterior_pairs(
        idata_synth, var_names=all_params_synth, kde_bw=2.5, compact=True, use_seaborn=True,
        max_samples=1000,
    )
    plt.savefig(figures_dir / '05_synth_dream_pairs.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nConvergence: R-hat per parameter (all {len(all_params_synth)} params, from PyDREAM full chains):")
    try:
        plot_rhat_from_pydream(pydream_result, var_names=all_params_synth)
    except ValueError:
        plot_rhat_summary(idata_synth, var_names=all_params_synth)
    plt.savefig(figures_dir / '05_synth_dream_rhat.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nArviZ summary (R-hat, ESS, posterior stats — all parameters):")
    synth_summary = az.summary(idata_synth, var_names=all_params_synth)
    display(synth_summary) if hasattr(__builtins__, '__IPYTHON__') else print(synth_summary.to_string())
else:
    if not ARVIZ_AVAILABLE:
        print("ArviZ not available — install with: pip install arviz")

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

# 2. Parameter recovery: True bar always at 1.0; recovery = ratio when true!=0, else 1+estimate when true=0
ax2 = plt.subplot(2, 2, 2)
x_pos = np.arange(len(CALIBRATION_PARAMS))
width = 0.25
eps = 1e-10  # treat as zero for display

# Single rule: display value = 1 means "perfect recovery" for every parameter.
# When true != 0: recovery = est/true (so 1 = perfect).
# When true == 0: recovery = 1 + est (so 1 = estimated 0 = perfect). No division by zero.
true_norm = [1.0] * len(CALIBRATION_PARAMS)  # True reference always at 1.0
sceua_norm = []
for p in CALIBRATION_PARAMS:
    t = TRUE_SAC_PARAMS.get(p, 0)
    s = sceua_params.get(p, t)
    if abs(t) > eps:
        sceua_norm.append(s / t)
    else:
        sceua_norm.append(1.0 + s)

ax2.bar(x_pos - width/2, true_norm, width, label='True', color='black', alpha=0.7)
ax2.bar(x_pos + width/2, sceua_norm, width, label='SCE-UA', color='blue', alpha=0.7)

if PYDREAM_AVAILABLE and posterior_means is not None:
    pydream_norm = []
    pydream_err = []
    for p in CALIBRATION_PARAMS:
        t = TRUE_SAC_PARAMS.get(p, 0)
        m = posterior_means.get(p, t)
        std = posterior_stds.get(p, 0.0)
        if abs(t) > eps:
            pydream_norm.append(m / t)
            pydream_err.append(std / t)
        else:
            pydream_norm.append(1.0 + m)
            pydream_err.append(std)
    ax2.bar(x_pos + width*1.5, pydream_norm, width, label='PyDREAM', color='red', alpha=0.7,
            yerr=pydream_err, capsize=3)

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(CALIBRATION_PARAMS, rotation=90, ha='center', fontsize=8)
ax2.set_ylabel('Recovery (1 = perfect: ratio when true≠0, 1+est when true=0)')
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
PYDREAM_RESULTS_DIR = OUTPUT_DIR / 'reports'
PYDREAM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PyDREAM runs once per flow transform (4 runs), not per objective (13 runs).
# Mapping: objective name -> likelihood transform used for that run
TRANSFORMS = ['none', 'sqrt', 'log', 'inverse']
PYDREAM_FILENAMES_BY_TRANSFORM = {
    t: f'410734_sacramento_dream_{t}' for t in TRANSFORMS
}

# Storage: one result per transform, then pydream_results[name] = by_transform[transform]
pydream_results_by_transform = {}
pydream_reports_by_transform = {}
pydream_results = {}   # name -> result (same result shared by objectives with same transform)
pydream_reports = {}   # name -> report (for compatibility; one report per transform)

# Check if we should run calibrations or load existing results
RUN_CALIBRATIONS = False  # Set to False to skip running and only load existing results

if not PYDREAM_AVAILABLE:
    print("=" * 70)
    print("PYDREAM NOT AVAILABLE - SKIPPING CALIBRATIONS")
    print("=" * 70)
    print("\nInstall PyDREAM with: pip install pydream")
    RUN_CALIBRATIONS = False

# %%
if RUN_CALIBRATIONS and PYDREAM_AVAILABLE:
    print("=" * 70)
    print("RUNNING PYDREAM CALIBRATIONS (4 RUNS — ONE PER FLOW TRANSFORM)")
    print("=" * 70)
    print(f"\nTransforms: {TRANSFORMS} (one PyDREAM run per transform)")
    print(f"  Early stopping enabled — may finish sooner if chains converge")
    print("\n" + "-" * 70)

    total_start = time.time()

    for i, transform in enumerate(TRANSFORMS, 1):
        print(f"\n[{i}/{len(TRANSFORMS)}] Transform: {transform}")
        print("=" * 50)

        _pkl_name = PYDREAM_FILENAMES_BY_TRANSFORM[transform]
        result_file = PYDREAM_RESULTS_DIR / f'{_pkl_name}.pkl'

        if result_file.exists():
            print(f"  Loading existing result: {result_file.name}")
            report = CalibrationReport.load(str(result_file))
            pydream_reports_by_transform[transform] = report
            pydream_results_by_transform[transform] = report.result
            print(f"  ✓ Loaded (Best log-lik: {report.result.best_objective:.4f})")
            continue

        likelihood = TransformedGaussianLikelihood(transform)
        print(f"  Likelihood: {likelihood.flow_emphasis} emphasis")

        runner = CalibrationRunner(
            model=Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2),
            inputs=cal_inputs,
            observed=cal_observed,
            objective=likelihood,
            warmup_period=WARMUP_DAYS
        )

        progress_file = PYDREAM_RESULTS_DIR / f'progress_{_pkl_name}'
        print(f"  Config: {PYDREAM_ITERATIONS} max iter × {PYDREAM_CHAINS} chains")
        print(f"  Progress: tail -f {progress_file}.csv")
        start_time = time.time()

        try:
            result = runner.run_pydream(
                n_iterations=PYDREAM_ITERATIONS,
                n_chains=PYDREAM_CHAINS,
                multitry=PYDREAM_MULTITRY,
                snooker=PYDREAM_SNOOKER,
                parallel=False,
                adapt_crossover=False,
                adapt_gamma=PYDREAM_ADAPT_GAMMA,
                DEpairs=PYDREAM_DEPAIRS,
                verbose=True,
                nverbose=100,
                dbname=str(progress_file),
                hardboundaries=True,
                convergence_check=True,
                convergence_threshold=PYDREAM_CONVERGENCE_THRESHOLD,
                batch_size=PYDREAM_BATCH_SIZE,
                min_iterations=PYDREAM_MIN_ITER,
                patience=PYDREAM_PATIENCE,
                post_convergence_iterations=PYDREAM_POST_CONV,
            )

            elapsed = time.time() - start_time
            print(f"  ✓ Completed in {elapsed:.1f}s (Best log-lik: {result.best_objective:.4f})")

            if 'gelman_rubin' in result.convergence_diagnostics:
                gr_vals = result.convergence_diagnostics['gelman_rubin']
                max_gr = max(gr_vals.values())
                converged = result.convergence_diagnostics.get('converged', False)
                status = "✓ Converged" if converged else f"⚠ Max R-hat: {max_gr:.3f}"
                print(f"  Convergence: {status}")

            report = runner.create_report(result, catchment_info={
                'name': 'Queanbeyan River',
                'gauge_id': '410734',
                'area_km2': CATCHMENT_AREA_KM2
            })
            report.save(str(result_file.with_suffix('')))

            pydream_results_by_transform[transform] = result
            pydream_reports_by_transform[transform] = report

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    # Per-objective view: each objective uses its transform's result
    for name in objectives.keys():
        t = LIKELIHOOD_TRANSFORM_MAPPING.get(name, 'sqrt')
        if t in pydream_results_by_transform:
            pydream_results[name] = pydream_results_by_transform[t]
            pydream_reports[name] = pydream_reports_by_transform.get(t)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PYDREAM CALIBRATIONS COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Runs completed: {len(pydream_results_by_transform)}/{len(TRANSFORMS)}")
    print(f"Objectives mapped: {len(pydream_results)}/{len(objectives)}")
    print("=" * 70)

# %% [markdown]
# ---
# ## Load Existing PyDREAM Results
#
# If calibrations were run previously, load them from disk.

# %%
# Load existing PyDREAM results by transform (4 files), then map to objectives
print("=" * 70)
print("LOADING PYDREAM RESULTS")
print("=" * 70)

for transform in TRANSFORMS:
    if transform in pydream_results_by_transform:
        continue  # Already loaded from run section
    _pkl_name = PYDREAM_FILENAMES_BY_TRANSFORM[transform]
    result_file = PYDREAM_RESULTS_DIR / f'{_pkl_name}.pkl'
    if result_file.exists():
        try:
            report = CalibrationReport.load(str(result_file))
            pydream_reports_by_transform[transform] = report
            pydream_results_by_transform[transform] = report.result
            print(f"  ✓ {transform:<10}: Loaded")
        except Exception as e:
            print(f"  ✗ {transform:<10}: Failed to load ({e})")
    else:
        print(f"  - {transform:<10}: Not found")

# Per-objective view
for name in objectives.keys():
    t = LIKELIHOOD_TRANSFORM_MAPPING.get(name, 'sqrt')
    if t in pydream_results_by_transform:
        pydream_results[name] = pydream_results_by_transform[t]
        pydream_reports[name] = pydream_reports_by_transform.get(t)

print(f"\nPyDREAM runs loaded: {len(pydream_results_by_transform)}/{len(TRANSFORMS)}")
print(f"Objectives mapped: {len(pydream_results)}/{len(objectives)}")

# %% [markdown]
# ---
# ## Compact Algorithm Comparison
#
# One table combining objective values, cross-evaluated NSE/KGE, and
# runtime for **both** algorithms across all 13 objective functions.
# Objectives are grouped by the PyDREAM flow transform they map to.

# %%
print("=" * 70)
print("COMPACT ALGORITHM COMPARISON (13 objectives)")
print("=" * 70)

compact_rows = []
for name in objectives.keys():
    obj_func = objectives[name]
    transform = LIKELIHOOD_TRANSFORM_MAPPING.get(name, 'sqrt')
    row = {"Objective": name, "Transform": transform}

    if name in sceua_results:
        res_s = sceua_results[name]
        row["SCE-UA_best"] = res_s.best_objective
        row["SCE-UA_time_s"] = res_s.runtime_seconds
        model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        model.set_parameters(res_s.best_parameters)
        model.reset()
        sim = model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        obs = cal_observed[WARMUP_DAYS:]
        m = compute_diagnostics(sim, obs)
        row["SCE-UA_NSE"] = m["NSE"]
        row["SCE-UA_KGE"] = m["KGE"]
    else:
        row.update({"SCE-UA_best": np.nan, "SCE-UA_time_s": np.nan, "SCE-UA_NSE": np.nan, "SCE-UA_KGE": np.nan})

    if name in pydream_results:
        res_p = pydream_results[name]
        row["PyDREAM_time_s"] = res_p.runtime_seconds
        model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
        model.set_parameters(res_p.best_parameters)
        model.reset()
        sim = model.run(cal_inputs)['runoff'].values[WARMUP_DAYS:]
        obs = cal_observed[WARMUP_DAYS:]
        row["PyDREAM_best"] = obj_func(obs, sim)
        m = compute_diagnostics(sim, obs)
        row["PyDREAM_NSE"] = m["NSE"]
        row["PyDREAM_KGE"] = m["KGE"]
    else:
        row.update({"PyDREAM_best": np.nan, "PyDREAM_time_s": np.nan, "PyDREAM_NSE": np.nan, "PyDREAM_KGE": np.nan})

    if not np.isnan(row.get("SCE-UA_time_s", np.nan)) and not np.isnan(row.get("PyDREAM_time_s", np.nan)):
        row["Speedup"] = f"{row['PyDREAM_time_s'] / row['SCE-UA_time_s']:.0f}x"
    else:
        row["Speedup"] = ""
    compact_rows.append(row)

compact_df = pd.DataFrame(compact_rows).sort_values(["Transform", "Objective"]).reset_index(drop=True)
display_cols = ["Objective", "Transform", "SCE-UA_best", "PyDREAM_best",
                "SCE-UA_NSE", "PyDREAM_NSE", "SCE-UA_KGE", "PyDREAM_KGE",
                "SCE-UA_time_s", "PyDREAM_time_s", "Speedup"]
pd.set_option("display.float_format", "{:.4f}".format)
display(compact_df[display_cols]) if hasattr(__builtins__, '__IPYTHON__') else print(compact_df[display_cols].to_string(index=False))
pd.reset_option("display.float_format")
compact_df.to_csv(figures_dir / "06_compact_comparison.csv", index=False)
print(f"\nSaved: {figures_dir / '06_compact_comparison.csv'}")

# %% [markdown]
# ### Runtime: SCE-UA vs PyDREAM by Flow Transform

# %%
if len(sceua_results) > 0 and len(pydream_results) > 0:
    runtime_data = []
    for t in TRANSFORMS:
        objs_t = [n for n, tr in LIKELIHOOD_TRANSFORM_MAPPING.items() if tr == t]
        s_times = [sceua_results[n].runtime_seconds for n in objs_t if n in sceua_results]
        p_time = pydream_results_by_transform[t].runtime_seconds if t in pydream_results_by_transform else np.nan
        runtime_data.append({"Transform": t, "SCE-UA_mean_s": np.mean(s_times) if s_times else np.nan, "PyDREAM_s": p_time})
    rt_df = pd.DataFrame(runtime_data)

    fig_rt = go.Figure()
    fig_rt.add_trace(go.Bar(name="SCE-UA (mean)", x=rt_df["Transform"], y=rt_df["SCE-UA_mean_s"], marker_color="#1f77b4"))
    fig_rt.add_trace(go.Bar(name="PyDREAM", x=rt_df["Transform"], y=rt_df["PyDREAM_s"], marker_color="#d62728"))
    fig_rt.update_layout(
        title="Runtime: SCE-UA (mean over mapped objectives) vs PyDREAM",
        yaxis_title="Runtime (seconds)", yaxis_type="log",
        barmode="group", template="plotly_white", height=350,
        legend=dict(orientation="h", y=1.02, xanchor="center", x=0.5),
    )
    fig_rt.show()
    fig_rt.write_html(str(figures_dir / "06_runtime_comparison.html"))
    print(f"Saved: {figures_dir / '06_runtime_comparison.html'}")

# %% [markdown]
# ---
# ## Posterior Diagnostics: All Parameters, All Objectives
#
# Professional MCMC diagnostics for **every** PyDREAM calibration using
# **all 22 Sacramento parameters**.  For each objective function we show:
#
# 1. **Trace + posterior** — full chain trajectory with the post-convergence
#    window highlighted; KDE spans the full feasible parameter range.
# 2. **Pair plot** — KDE contours for all parameters.
# 3. **R-hat bar chart** — PyDREAM’s Gelman–Rubin (full chains), so it matches the convergence criterion.
# 4. **Numerical summary** — R-hat, ESS, mean, sd, HDI.
#
# Only the **last 1000 post-convergence samples** per chain feed the KDEs
# and summary statistics.

# %%
# Full Sacramento parameter list (canonical order for pair/forest plots)
SAC_PARAM_ORDER = [
    'uztwm', 'uzfwm', 'lztwm', 'lzfpm', 'lzfsm', 'uzk', 'lzpk', 'lzsk',
    'zperc', 'rexp', 'pctim', 'adimp', 'pfree', 'rserv', 'side', 'ssout', 'sarva',
    'uh1', 'uh2', 'uh3', 'uh4', 'uh5',
]
_sac_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
SAC_BOUNDS = _sac_model.get_parameter_bounds()

# Convert DREAM results to ArviZ InferenceData (one per transform, then map by objective)
pydream_idatas = {}
idata_by_transform = {}

if ARVIZ_AVAILABLE and len(pydream_results_by_transform) > 0:
    for transform in TRANSFORMS:
        if transform not in pydream_results_by_transform:
            continue
        try:
            result = pydream_results_by_transform[transform]
            idata = dream_result_to_inference_data(
                result, burn_fraction=0.3, post_convergence_draws=1000
            )
            idata_by_transform[transform] = idata
        except Exception as e:
            print(f"  Could not convert transform {transform}: {e}")
    for name in objectives.keys():
        t = LIKELIHOOD_TRANSFORM_MAPPING.get(name, 'sqrt')
        if t in idata_by_transform:
            pydream_idatas[name] = idata_by_transform[t]
    print(f"Converted {len(idata_by_transform)} DREAM runs to ArviZ InferenceData; {len(pydream_idatas)} objectives mapped")
else:
    print("ArviZ not available or no DREAM results — skipping conversion")

# %%
# Posterior diagnostics: one block per flow transform (4 blocks; objectives share by transform)
print("=" * 70)
print("POSTERIOR DIAGNOSTICS — BY FLOW TRANSFORM (4 RUNS)")
print("=" * 70)

for transform in TRANSFORMS:
    if transform not in idata_by_transform or transform not in pydream_results_by_transform:
        print(f"\nTransform {transform}: No InferenceData — skipping")
        continue

    idata = idata_by_transform[transform]
    result = pydream_results_by_transform[transform]
    obj_names_using = [n for n, t in LIKELIHOOD_TRANSFORM_MAPPING.items() if t == transform]
    # Use canonical order; include only params present in this idata
    all_params = [p for p in SAC_PARAM_ORDER if p in idata.posterior.data_vars]
    if not all_params:
        all_params = list(idata.posterior.data_vars)

    print(f"\n{'='*70}")
    print(f"  Transform: {transform}  — objectives: {', '.join(obj_names_using)}  ({len(all_params)} params)")
    print(f"{'='*70}")

    # 1) Trace + posterior (full parameter range)
    print(f"\n  Trace + posterior (full range, last 1000 highlighted):")
    plot_dream_traces(
        result, var_names=all_params,
        param_bounds=SAC_BOUNDS,
        burn_fraction=0.3, post_convergence_draws=1000, kde_bw=1.5,
        title=f"DREAM Posterior — transform '{transform}'",
    )
    plt.savefig(figures_dir / f'06_dream_traces_{transform}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 2) Pair plot — all parameters, larger figure; subsampled draws for speed
    if len(all_params) >= 2:
        print(f"\n  Pair plot (all {len(all_params)} params, smooth KDE, subsampled):")
        plot_posterior_pairs(
            idata, var_names=all_params, kde_bw=2.5, compact=True, use_seaborn=True,
            max_samples=1000,
        )
        plt.savefig(figures_dir / f'06_dream_pairs_{transform}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    # 3) R-hat bar chart — uses PyDREAM’s GR at stop (when batch mode), so all bars ≤ threshold
    print(f"\n  R-hat convergence summary:")
    try:
        plot_rhat_from_pydream(result, var_names=all_params)
    except ValueError:
        plot_rhat_summary(idata, var_names=all_params)
    plt.savefig(figures_dir / f'06_dream_rhat_{transform}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 4) PyDREAM R-hat table (matches the bar chart above)
    _cd = getattr(result, 'convergence_diagnostics', None) or {}
    _raw = getattr(result, '_raw_result', None) or {}
    _hist = _cd.get('convergence_history') or _raw.get('convergence_history')
    _conv_iter = _cd.get('convergence_iteration') or _raw.get('convergence_iteration')
    _gr_end = _cd.get('gelman_rubin') or (_raw.get('convergence_diagnostics') or {}).get('gelman_rubin')
    _gr_stop = None
    if _hist and _conv_iter is not None:
        for _e in _hist:
            if _e.get('iteration') == _conv_iter:
                _gr_stop = _e.get('gr_values')
                break
    if _gr_stop is None and _hist:
        _conv_entries = [_e for _e in _hist if _e.get('converged')]
        if _conv_entries:
            _gr_stop = _conv_entries[0].get('gr_values')
    if _gr_end is not None:
        _names = [p for p in all_params if p in _gr_end]
        if _names:
            _rhat_df = pd.DataFrame({
                'param': _names,
                'Rhat_at_stop': [_gr_stop.get(p, np.nan) if _gr_stop else np.nan for p in _names],
                'Rhat_end_of_run': [_gr_end.get(p, np.nan) for p in _names],
            }).set_index('param')
            print(f"\n  PyDREAM R-hat (values in bar chart above):")
            display(_rhat_df) if hasattr(__builtins__, '__IPYTHON__') else print(_rhat_df.to_string())
            print("  (ArviZ summary below is on last-1000 draws per chain; its r_hat can differ.)")

    # 5) ArviZ summary (ESS, HDI, etc.; r_hat from ArviZ on truncated posterior)
    print(f"\n  ArviZ summary (on last 1000 post-convergence draws; r_hat ≠ bar chart):")
    summary_df = az.summary(idata, var_names=all_params)
    display(summary_df) if hasattr(__builtins__, '__IPYTHON__') else print(summary_df.to_string())
    print()

# %% [markdown]
# ---
# ## Section A — Parameter Identifiability Heatmap
#
# One heatmap answering: **which parameters are well-identified and which show
# equifinality?** Rows = 22 parameters, columns = 4 PyDREAM transforms,
# cell = normalised posterior width (94 % HDI width / parameter range).
# Green = narrow (well-identified), red = wide (poorly identified).

# %%
forest_params = ['uztwm', 'uzfwm', 'lztwm', 'lzfpm', 'lzfsm', 'uzk',
                 'lzpk', 'lzsk', 'zperc', 'rexp', 'pctim', 'adimp',
                 'pfree', 'rserv', 'side', 'ssout', 'sarva',
                 'uh1', 'uh2', 'uh3', 'uh4', 'uh5']

def _posterior_summary(idata, param):
    """Return (mean, hdi_lo, hdi_hi) for *param* from ArviZ InferenceData."""
    if param not in idata.posterior.data_vars:
        return None
    try:
        s = az.summary(idata, var_names=[param])
        row = s.iloc[0]
        mean = float(row["mean"])
        if "hdi_3%" in row and "hdi_97%" in row:
            return mean, float(row["hdi_3%"]), float(row["hdi_97%"])
        sd = float(row.get("sd", 0))
        return mean, mean - 2 * sd, mean + 2 * sd
    except Exception:
        return None

if ARVIZ_AVAILABLE and len(idata_by_transform) > 0:
    norm_width = np.full((len(forest_params), len(TRANSFORMS)), np.nan)
    hdi_text = np.empty_like(norm_width, dtype=object)
    hover_text = np.empty_like(norm_width, dtype=object)

    for i, param in enumerate(forest_params):
        lo_bound, hi_bound = SAC_BOUNDS.get(param, (0, 1))
        param_range = hi_bound - lo_bound
        for j, t in enumerate(TRANSFORMS):
            if t not in idata_by_transform:
                hdi_text[i, j] = ""
                hover_text[i, j] = ""
                continue
            summ = _posterior_summary(idata_by_transform[t], param)
            if summ is None:
                hdi_text[i, j] = ""
                hover_text[i, j] = ""
                continue
            mean, hdi_lo, hdi_hi = summ
            width = hdi_hi - hdi_lo
            nw = width / param_range if param_range > 0 else np.nan
            norm_width[i, j] = nw
            hdi_text[i, j] = f"{nw:.0%}"
            hover_text[i, j] = (
                f"<b>{param}</b> ({t})<br>"
                f"Mean: {mean:.4g}<br>"
                f"94% HDI: [{hdi_lo:.4g}, {hdi_hi:.4g}]<br>"
                f"Width: {width:.4g}<br>"
                f"Normalised: {nw:.1%}"
            )

    fig_ident = go.Figure(data=go.Heatmap(
        z=norm_width,
        x=TRANSFORMS,
        y=forest_params,
        text=hdi_text,
        texttemplate="%{text}",
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=[[0, "#2ca02c"], [0.5, "#ffffbf"], [1, "#d62728"]],
        zmin=0, zmax=1,
        colorbar=dict(title="HDI / range", tickformat=".0%"),
    ))
    fig_ident.update_layout(
        title="Parameter Identifiability (94% HDI width / parameter range)",
        yaxis=dict(autorange="reversed", dtick=1),
        xaxis_title="PyDREAM flow transform",
        template="plotly_white", height=700, width=550,
        margin=dict(l=100, r=40, t=60, b=50),
    )
    fig_ident.show()
    fig_ident.write_html(str(figures_dir / "06_identifiability_heatmap.html"))
    print(f"Saved: {figures_dir / '06_identifiability_heatmap.html'}")
else:
    print("ArviZ or PyDREAM data unavailable — skipping identifiability heatmap.")

# %% [markdown]
# ---
# ## Section B — SCE-UA Point Estimates vs PyDREAM Posteriors
#
# Heatmap answering: **where do the 13 SCE-UA point estimates fall within the
# posterior distributions?** Rows = 22 parameters, columns = 13 objectives
# (grouped by transform). Cell = position of the SCE-UA point within the
# PyDREAM 94 % HDI as a percentage (0 % = lower bound, 100 % = upper bound).
# Blue = inside HDI, red = outside.

# %%
if ARVIZ_AVAILABLE and len(idata_by_transform) > 0 and len(sceua_results) > 0:
    obj_names_ordered = sorted(objectives.keys(), key=lambda n: (LIKELIHOOD_TRANSFORM_MAPPING.get(n, "sqrt"), n))
    n_params = len(forest_params)
    n_objs = len(obj_names_ordered)

    pos_matrix = np.full((n_params, n_objs), np.nan)
    annot_text = np.empty_like(pos_matrix, dtype=object)
    hover_matrix = np.empty_like(pos_matrix, dtype=object)

    for i, param in enumerate(forest_params):
        for j, obj_name in enumerate(obj_names_ordered):
            transform = LIKELIHOOD_TRANSFORM_MAPPING.get(obj_name, "sqrt")
            annot_text[i, j] = ""
            hover_matrix[i, j] = ""
            if transform not in idata_by_transform:
                continue
            summ = _posterior_summary(idata_by_transform[transform], param)
            if summ is None:
                continue
            if obj_name not in sceua_results or param not in sceua_results[obj_name].best_parameters:
                continue
            _, hdi_lo, hdi_hi = summ
            pt = sceua_results[obj_name].best_parameters[param]
            width = hdi_hi - hdi_lo
            if width > 0:
                pct = (pt - hdi_lo) / width * 100
            else:
                pct = 50.0
            inside = 0 <= pct <= 100
            pos_matrix[i, j] = pct if inside else -1
            annot_text[i, j] = "In" if inside else "Out"
            hover_matrix[i, j] = (
                f"<b>{param}</b> — {obj_name} ({transform})<br>"
                f"SCE-UA: {pt:.4g}<br>"
                f"HDI: [{hdi_lo:.4g}, {hdi_hi:.4g}]<br>"
                f"Position: {pct:.0f}%<br>"
                f"{'Inside' if inside else 'Outside'} 94% HDI"
            )

    colorscale = [
        [0.0, "#d62728"],   # -1 (Out) → red
        [0.009, "#d62728"],
        [0.01, "#6baed6"],  # 0% (In, at lower HDI bound) → light blue
        [0.5, "#08519c"],   # 50% (In, at mean) → dark blue
        [0.99, "#6baed6"],  # 100% (In, at upper HDI bound) → light blue
        [1.0, "#6baed6"],
    ]

    fig_pip = go.Figure(data=go.Heatmap(
        z=pos_matrix,
        x=[f"{n}<br>({LIKELIHOOD_TRANSFORM_MAPPING.get(n, '?')})" for n in obj_names_ordered],
        y=forest_params,
        text=annot_text,
        texttemplate="%{text}",
        hovertext=hover_matrix,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=colorscale,
        zmin=-1, zmax=100,
        colorbar=dict(title="Position in HDI (%)", tickvals=[0, 25, 50, 75, 100]),
    ))
    fig_pip.update_layout(
        title="SCE-UA Point Estimates: Position Within PyDREAM 94% HDI",
        yaxis=dict(autorange="reversed", dtick=1),
        xaxis_title="Objective (transform)",
        template="plotly_white", height=750, width=1000,
        margin=dict(l=100, r=40, t=60, b=120),
        xaxis=dict(tickangle=-45),
    )
    fig_pip.show()
    fig_pip.write_html(str(figures_dir / "06_sceua_in_posterior_heatmap.html"))
    print(f"Saved: {figures_dir / '06_sceua_in_posterior_heatmap.html'}")

    # Print summary: fraction of SCE-UA points inside HDI, overall and per transform
    total_inside = np.nansum((pos_matrix >= 0) & (pos_matrix <= 100))
    total_valid = np.sum(~np.isnan(pos_matrix))
    print(f"\nOverall: {int(total_inside)}/{int(total_valid)} SCE-UA point estimates inside PyDREAM 94% HDI "
          f"({total_inside / total_valid * 100:.0f}%)")
    for t in TRANSFORMS:
        cols_t = [k for k, n in enumerate(obj_names_ordered) if LIKELIHOOD_TRANSFORM_MAPPING.get(n, "sqrt") == t]
        if not cols_t:
            continue
        sub = pos_matrix[:, cols_t]
        ins = np.nansum((sub >= 0) & (sub <= 100))
        val = np.sum(~np.isnan(sub))
        if val > 0:
            print(f"  {t}: {int(ins)}/{int(val)} ({ins / val * 100:.0f}%)")
else:
    print("Skipping SCE-UA-in-posterior heatmap (missing data).")

# %% [markdown]
# ---
# ## Section E — MCMC Convergence Summary
#
# One compact table summarising convergence across the four PyDREAM transforms:
# max/mean R-hat, min ESS (bulk and tail), and overall convergence status.

# %%
if ARVIZ_AVAILABLE and len(idata_by_transform) > 0:
    conv_rows = []
    for t in TRANSFORMS:
        if t not in idata_by_transform:
            continue
        idata = idata_by_transform[t]
        all_vars = list(idata.posterior.data_vars)
        if not all_vars:
            continue
        s = az.summary(idata, var_names=all_vars)
        rhat_col = "r_hat" if "r_hat" in s.columns else None
        ess_bulk_col = "ess_bulk" if "ess_bulk" in s.columns else None
        ess_tail_col = "ess_tail" if "ess_tail" in s.columns else None
        row = {"Transform": t}
        if rhat_col:
            row["Max R-hat"] = s[rhat_col].max()
            row["Mean R-hat"] = s[rhat_col].mean()
            row["Converged"] = "Yes" if s[rhat_col].max() < 1.2 else "No"
        if ess_bulk_col:
            row["Min ESS bulk"] = int(s[ess_bulk_col].min())
        if ess_tail_col:
            row["Min ESS tail"] = int(s[ess_tail_col].min())
        conv_rows.append(row)

    if conv_rows:
        conv_summary_df = pd.DataFrame(conv_rows)
        print("=" * 70)
        print("MCMC CONVERGENCE SUMMARY (4 PyDREAM transforms)")
        print("=" * 70)
        print("R-hat < 1.2 indicates convergence; ESS > 100 is a practical minimum.\n")
        display(conv_summary_df.round(3)) if hasattr(__builtins__, '__IPYTHON__') else print(conv_summary_df.round(3).to_string(index=False))
    else:
        print("No convergence data available.")
else:
    print("ArviZ or PyDREAM data unavailable — skipping convergence summary.")

# %% [markdown]
# ---
# ## Summary and Recommendations
#
# | Aspect | SCE-UA | PyDREAM |
# |--------|--------|---------|
# | **Output** | Point estimate | Full posterior |
# | **Uncertainty** | None | Credible intervals |
# | **Speed** | Fast (minutes) | Slow (hours) |
# | **Convergence** | Easy | Requires checking |
# | **Best for** | Quick calibration | Uncertainty analysis |
#
# ### When to Use Each Algorithm
#
# - **SCE-UA**: quick calibration, point estimates sufficient, many catchments, limited compute.
# - **PyDREAM**: parameter uncertainty matters, credible intervals for predictions, research papers, structural uncertainty.

# %%
print("=" * 70)
print("ALGORITHM COMPARISON SUMMARY")
print("=" * 70)

if len(sceua_results) > 0 and len(pydream_results) > 0:
    common = set(sceua_results.keys()) & set(pydream_results.keys())
    if common:
        sceua_times = [sceua_results[k].runtime_seconds for k in common]
        pydream_times = [pydream_results[k].runtime_seconds for k in common]
        print(f"\nCompared {len(common)} objective functions")
        print(f"\nRuntime comparison:")
        print(f"  SCE-UA average:  {np.mean(sceua_times):.1f} seconds")
        print(f"  PyDREAM average: {np.mean(pydream_times):.1f} seconds")
        print(f"  Speed ratio:     {np.mean(pydream_times)/np.mean(sceua_times):.1f}x slower")

print("""
Recommendations:
  - For quick calibration     -> SCE-UA (fast, reliable)
  - For uncertainty analysis  -> PyDREAM (full posteriors)
  - For research papers       -> PyDREAM (publishable uncertainty)
  - For operational use       -> SCE-UA (efficient at scale)
  - For best of both          -> SCE-UA first, PyDREAM for final model
""")

# %% [markdown]
# ---
# ## Export PyDREAM Calibrations to Excel
#
# Export the four PyDREAM calibration reports (one per flow transform) to Excel
# for sharing or further analysis. Each file contains timeseries, best parameters,
# diagnostics, and FDC sheets.

# %%
PYDREAM_EXPORT_DIR = OUTPUT_DIR / 'exports'
PYDREAM_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

exported_pydream = []
for transform in TRANSFORMS:
    if transform not in pydream_reports_by_transform:
        continue
    report = pydream_reports_by_transform[transform]
    base_name = PYDREAM_FILENAMES_BY_TRANSFORM[transform]
    xlsx_path = PYDREAM_EXPORT_DIR / f'{base_name}.xlsx'
    try:
        export_report(report, str(xlsx_path), format='excel')
        exported_pydream.append(xlsx_path)
    except Exception as e:
        print(f"  ✗ {transform}: {e}")

if exported_pydream:
    print("=" * 70)
    print("PYDREAM REPORTS EXPORTED TO EXCEL")
    print("=" * 70)
    print(f"\nDirectory: {PYDREAM_EXPORT_DIR.absolute()}\n")
    for p in exported_pydream:
        print(f"  {p.name}")
    print(f"\nTotal: {len(exported_pydream)}/{len(TRANSFORMS)} PyDREAM calibrations exported.")
else:
    print("No PyDREAM reports available to export (run or load calibrations first).")

# %% [markdown]
# ---
# ## Next Steps
#
# - **Notebook 06**: Sensitivity Analysis (Sobol indices)
# - For production: Increase PyDREAM iterations (5000-10000+)
# - Consider running PyDREAM overnight for comprehensive posteriors
# - Use posteriors for prediction uncertainty bounds

# %%
print("=" * 70)
print("ALGORITHM COMPARISON COMPLETE")
print("=" * 70)
print("""
You now understand:
  ✓ How SCE-UA and PyDREAM differ fundamentally
  ✓ How to run calibrations with both algorithms
  ✓ How to compare posteriors with point estimates
  ✓ When to use each approach
  ✓ How to check MCMC convergence
  
The posterior distributions provide valuable insight into parameter
uncertainty that point estimates alone cannot capture!
""")
