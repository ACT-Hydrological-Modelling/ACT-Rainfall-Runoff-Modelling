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
#     display_name: pyrrm (Python 3.11.14)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Flow Range Effects on Routing Calibration
#
# ## Purpose
#
# This notebook explores how **limiting the calibration flow range** affects
# the multi-objective routing experiments from Notebook 03. We trim the
# observed flow data to values between **1 ML/day and 1000 ML/day**, excluding:
# - Very low flows (< 1 ML/day) - often dominated by measurement error
# - Very high flows (> 1000 ML/day) - extreme events that may bias calibration
#
# ## Research Questions
#
# 1. How do calibrated parameters change when extreme flows are excluded?
# 2. Does trimming improve or degrade performance on the trimmed range?
# 3. How do trimmed-calibration models perform on the full flow range?
# 4. Does the equifinality between UH and Muskingum routing change?
#
# ## Prerequisites
#
# - Completed Notebook 02 (calibration quickstart) - for Stage A baselines
# - Completed Notebook 03 (routing quickstart) - for full-range comparison
#
# ## Methodology
#
# We run the same **four-stage calibration** as Notebook 03:
# - **A**: Baseline without routing (loaded from Notebook 02)
# - **B**: Add routing (fix RR params from A, calibrate routing only)
# - **C**: Joint calibration (all RR + routing params)
# - **D**: Fixed UH (uh1=1) + routing (equifinality test)
#
# But this time, objective functions are calculated **only on timesteps where
# observed flow is between 1 and 1000 ML/day**.

# %% [markdown]
# ---
# ## Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# pyrrm imports
from pyrrm.models import Sacramento
from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
from pyrrm.calibration import CalibrationRunner
from pyrrm.calibration.report import CalibrationReport
from pyrrm.objectives import NSE, KGE, RMSE, PBIAS, SDEB, FlowTransformation
from pyrrm.data import load_parameter_bounds

# Reload modules to pick up any code changes
import importlib
import pyrrm.visualization.report_plots
import pyrrm.calibration.report
import pyrrm.calibration.runner
importlib.reload(pyrrm.visualization.report_plots)
importlib.reload(pyrrm.calibration.report)
importlib.reload(pyrrm.calibration.runner)

print("=" * 70)
print("NOTEBOOK 04: FLOW RANGE EFFECTS ON ROUTING CALIBRATION")
print("=" * 70)
print("\nThis notebook explores calibration with trimmed flow range (1-1000 ML/day)")

# %% [markdown]
# ---
# ## Define Flow Range Limits
#
# We define the flow range to include in calibration. Values outside this range
# will be masked (set to NaN) and excluded from objective function calculations.

# %%
# Flow range limits (in ML/day)
FLOW_MIN = 1.0      # Minimum flow to include
FLOW_MAX = 1000.0   # Maximum flow to include

print(f"Flow Range for Calibration: {FLOW_MIN} - {FLOW_MAX} ML/day")
print("\nFlows outside this range will be masked during calibration.")

# %% [markdown]
# ---
# ## Load Data from Calibration Reports
#
# **IMPORTANT**: We load the input data from the stored calibration reports from
# Notebook 02. This ensures we use *exactly* the same data that was used for
# calibration, avoiding any data mismatch issues.

# %%
# Data paths
DATA_DIR = Path('../data/410734')
REPORTS_DIR = Path('../test_data/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Catchment parameters (from Notebook 03)
CATCHMENT_AREA_KM2 = 516.62667
WARMUP_DAYS = 365

# Load a reference report to get the exact data used for calibration
reference_report = CalibrationReport.load(REPORTS_DIR / '410734_nse.pkl')

# Use the inputs and observed data from the report
# This ensures EXACT match with what was used during calibration
inputs_df = reference_report.inputs.copy()
observed_post_warmup = reference_report.observed.copy()

print("=" * 70)
print("DATA LOADED FROM CALIBRATION REPORT")
print("=" * 70)
print(f"\nInputs shape: {inputs_df.shape}")
print(f"Observed (post-warmup) shape: {observed_post_warmup.shape}")
print(f"Inputs columns: {list(inputs_df.columns)}")

# The report stores observed AFTER warmup, but CalibrationRunner expects FULL observed
# (including warmup period) because it applies warmup internally
# Calculate warmup from the difference in lengths
WARMUP_DAYS = len(inputs_df) - len(observed_post_warmup)
print(f"\nImplied warmup: {WARMUP_DAYS} days")

# Create full observed by prepending NaN for warmup period
# This matches what CalibrationRunner expects
observed_full = np.concatenate([
    np.full(WARMUP_DAYS, np.nan),  # Warmup period (will be ignored)
    observed_post_warmup            # Actual data
])
print(f"Full observed shape (with warmup NaN): {observed_full.shape}")

# Verify alignment
assert len(observed_full) == len(inputs_df), \
    f"Length mismatch: observed={len(observed_full)}, inputs={len(inputs_df)}"

print(f"\nCatchment area: {CATCHMENT_AREA_KM2} km²")
print(f"Warmup period: {WARMUP_DAYS} days")
print(f"Date range: {inputs_df.index[0].date()} to {inputs_df.index[-1].date()}")

# Handle missing data flags (-9999)
# Replace -9999 with NaN for proper handling
MISSING_FLAG = -9999.0
n_missing_before = (observed_full == MISSING_FLAG).sum()
observed_full = np.where(observed_full == MISSING_FLAG, np.nan, observed_full)
print(f"Missing data (-9999) replaced with NaN: {n_missing_before} values")

# %%
# Analyze flow distribution
# Use observed_post_warmup for statistics (this is the actual data without NaN padding)
print("\n" + "=" * 70)
print("FLOW DISTRIBUTION ANALYSIS")
print("=" * 70)

obs_valid = observed_post_warmup[~np.isnan(observed_post_warmup)]
print(f"\nCalibration period statistics (post-warmup):")
print(f"  Total timesteps: {len(observed_post_warmup)}")
print(f"  Valid observations: {len(obs_valid)}")
print(f"  Min flow: {obs_valid.min():.2f} ML/day")
print(f"  Max flow: {obs_valid.max():.2f} ML/day")
print(f"  Mean flow: {obs_valid.mean():.2f} ML/day")
print(f"  Median flow: {np.median(obs_valid):.2f} ML/day")

# Calculate how many observations fall within range
in_range = (obs_valid >= FLOW_MIN) & (obs_valid <= FLOW_MAX)
n_in_range = in_range.sum()
pct_in_range = 100 * n_in_range / len(obs_valid)

below_range = obs_valid < FLOW_MIN
above_range = obs_valid > FLOW_MAX

print(f"\nFlow range analysis ({FLOW_MIN}-{FLOW_MAX} ML/day):")
print(f"  Within range: {n_in_range} ({pct_in_range:.1f}%)")
print(f"  Below {FLOW_MIN} ML/day: {below_range.sum()} ({100*below_range.sum()/len(obs_valid):.1f}%)")
print(f"  Above {FLOW_MAX} ML/day: {above_range.sum()} ({100*above_range.sum()/len(obs_valid):.1f}%)")

# %%
# Analyze trimmed flow range
# Use observed_post_warmup for analysis (actual data without NaN padding)
observed_trimmed_pw = observed_post_warmup.copy()
mask_low = observed_trimmed_pw < FLOW_MIN
mask_high = observed_trimmed_pw > FLOW_MAX
observed_trimmed_pw[mask_low | mask_high] = np.nan

# Count valid observations in calibration period
valid_full = ~np.isnan(observed_post_warmup)
valid_trimmed = ~np.isnan(observed_trimmed_pw)

print(f"\nObservation counts (calibration period):")
print(f"  Full range valid: {valid_full.sum()} timesteps")
print(f"  Trimmed range valid: {valid_trimmed.sum()} timesteps")
print(f"  Reduction: {valid_full.sum() - valid_trimmed.sum()} timesteps ({100*(1-valid_trimmed.sum()/valid_full.sum()):.1f}%)")

# %% [markdown]
# ---
# ## Visualize Flow Range

# %%
# Plot flow distribution with range limits
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
ax1 = axes[0]
ax1.hist(obs_valid, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(FLOW_MIN, color='red', linestyle='--', linewidth=2, label=f'Min: {FLOW_MIN} ML/day')
ax1.axvline(FLOW_MAX, color='red', linestyle='--', linewidth=2, label=f'Max: {FLOW_MAX} ML/day')
ax1.set_xlabel('Flow (ML/day)')
ax1.set_ylabel('Frequency')
ax1.set_title('Flow Distribution with Range Limits')
ax1.legend()

# Log histogram
ax2 = axes[1]
ax2.hist(np.log10(obs_valid + 0.1), bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(np.log10(FLOW_MIN), color='red', linestyle='--', linewidth=2)
ax2.axvline(np.log10(FLOW_MAX), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('log10(Flow + 0.1)')
ax2.set_ylabel('Frequency')
ax2.set_title('Log-Transformed Flow Distribution')

# Time series with range
ax3 = axes[2]
time_idx = np.arange(len(observed_full))
ax3.plot(time_idx, observed_full, 'b-', alpha=0.5, linewidth=0.5, label='Full')
ax3.axhline(FLOW_MIN, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(FLOW_MAX, color='red', linestyle='--', linewidth=1.5)
ax3.fill_between(time_idx, FLOW_MIN, FLOW_MAX, alpha=0.2, color='green', label='Calibration range')
ax3.set_xlabel('Day')
ax3.set_ylabel('Flow (ML/day)')
ax3.set_title('Time Series with Calibration Range')
ax3.set_yscale('log')
ax3.legend()

plt.tight_layout()
plt.show()

print("Observations outside the green band will be excluded from calibration.")

# %% [markdown]
# ---
# ## Load Baseline Calibrations from Notebook 02
#
# We load the Stage A (no routing) calibrations from Notebook 02 as our baselines.

# %%
# Load saved reports from Notebook 02
print("=" * 70)
print("LOADING BASELINE CALIBRATIONS FROM NOTEBOOK 02")
print("=" * 70)

# Report name mapping
REPORT_NAMES = {
    'NSE (default)': '410734_nse.pkl',
    'LogNSE (custom)': '410734_lognse_custom.pkl',
    'InvNSE (custom)': '410734_invnse_custom.pkl',
    'SqrtNSE (custom)': '410734_sqrtnse_custom.pkl',
    'SDEB (custom)': '410734_sdeb_custom.pkl',
}

loaded_reports = {}
for name, filename in REPORT_NAMES.items():
    filepath = REPORTS_DIR / filename
    if filepath.exists():
        try:
            report = CalibrationReport.load(filepath)
            loaded_reports[name] = report
            print(f"  ✓ Loaded: {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    else:
        print(f"  ✗ Not found: {filename}")

print(f"\nLoaded {len(loaded_reports)} baseline calibrations")

# %%
# Load custom parameter bounds
bounds_file = DATA_DIR / 'sacramento_bounds_custom.txt'
custom_bounds = load_parameter_bounds(bounds_file)

print(f"Loaded {len(custom_bounds)} parameter bounds from {bounds_file.name}")

# %% [markdown]
# ---
# ## Define Objective Functions and Configurations

# %%
# Define objective functions for analysis
OBJECTIVE_CONFIGS = {
    'NSE': {
        'report_key': 'NSE (default)',
        'objective_fn': lambda: NSE(),
        'description': 'Standard NSE (high flow emphasis, maximize)'
    },
    'LogNSE': {
        'report_key': 'LogNSE (custom)',
        # epsilon_value=0.01 matches Notebook 02 to handle zero/near-zero flows
        'objective_fn': lambda: NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
        'description': 'Log-transformed (low flow emphasis, maximize)'
    },
    'InvNSE': {
        'report_key': 'InvNSE (custom)',
        # epsilon_value=0.01 matches Notebook 02 to prevent division by zero
        'objective_fn': lambda: NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
        'description': 'Inverse-transformed (recession emphasis, maximize)'
    },
    'SqrtNSE': {
        'report_key': 'SqrtNSE (custom)',
        'objective_fn': lambda: NSE(transform=FlowTransformation('sqrt')),
        'description': 'Sqrt-transformed (balanced emphasis, maximize)'
    },
    'SDEB': {
        'report_key': 'SDEB (custom)',
        # alpha=0.1, lam=0.5 matches Notebook 02 (Lerat et al., 2013)
        'objective_fn': lambda: SDEB(alpha=0.1, lam=0.5),
        'description': 'Sum Daily/Exceedance/Bias (multi-scale, minimize)'
    },
}

# Filter to available objectives
available_objectives = {}
for name, config in OBJECTIVE_CONFIGS.items():
    if config['report_key'] in loaded_reports:
        available_objectives[name] = config
        print(f"  ✓ {name}: {config['description']}")
    else:
        print(f"  ✗ {name}: Baseline not found")

print(f"\nAnalyzing {len(available_objectives)} objective functions with trimmed flow range")

# %% [markdown]
# ---
# ## Custom Objective Function for Trimmed Data
#
# We create a wrapper that calculates metrics only on timesteps where
# observed flow is within the specified range.

# %%
class TrimmedFlowObjective:
    """
    Wrapper that calculates objective function only on flows within a specified range.
    
    This wrapper properly exposes the direction/maximize attributes from the base
    objective so that calibration runners handle maximization/minimization correctly.
    
    Parameters
    ----------
    base_objective : callable
        The underlying objective function (e.g., NSE)
    flow_min : float
        Minimum flow to include in calibration
    flow_max : float
        Maximum flow to include in calibration
    """
    
    def __init__(self, base_objective, flow_min=FLOW_MIN, flow_max=FLOW_MAX):
        self.base_objective = base_objective
        self.flow_min = flow_min
        self.flow_max = flow_max
        
        # CRITICAL: Expose direction/maximize from base objective for calibration runners
        # The SCE-UA adapter uses these to determine whether to negate the objective
        # We MUST set BOTH attributes with consistent values
        
        # Get direction from base objective, default to 'maximize' (NSE, KGE)
        self.direction = getattr(base_objective, 'direction', 'maximize')
        
        # Get maximize from base objective, or infer from direction
        if hasattr(base_objective, 'maximize'):
            self.maximize = base_objective.maximize
        else:
            # Infer from direction
            self.maximize = (self.direction == 'maximize')
        
        # Copy other relevant attributes
        if hasattr(base_objective, 'name'):
            self.name = f"Trimmed({base_objective.name})"
        else:
            self.name = "TrimmedObjective"
        if hasattr(base_objective, 'optimal_value'):
            self.optimal_value = base_objective.optimal_value
    
    def __call__(self, observed, simulated):
        """Calculate objective only on flows within range."""
        # Create mask for valid observations
        obs = np.asarray(observed)
        sim = np.asarray(simulated)
        
        # Mask: within range AND not NaN
        valid_mask = (obs >= self.flow_min) & (obs <= self.flow_max) & ~np.isnan(obs) & ~np.isnan(sim)
        
        if valid_mask.sum() < 10:
            # Not enough valid points - return worst possible value
            if hasattr(self, 'direction'):
                return -1e10 if self.direction == 'maximize' else 1e10
            return -1e10  # Default assumes maximize
        
        # Apply mask
        obs_masked = obs[valid_mask]
        sim_masked = sim[valid_mask]
        
        # Calculate base objective on masked data
        return self.base_objective(obs_masked, sim_masked)
    
    def calculate(self, simulated, observed):
        """Legacy interface for compatibility with old calibration code."""
        return self.__call__(observed, simulated)
    
    def for_calibration(self, simulated, observed):
        """For compatibility with new objective interface."""
        return self.__call__(observed, simulated)
    
    def __repr__(self):
        return f"TrimmedFlowObjective({self.base_objective}, range=[{self.flow_min}, {self.flow_max}])"


def create_trimmed_objective(objective_fn, flow_min=FLOW_MIN, flow_max=FLOW_MAX):
    """Create a trimmed version of an objective function."""
    return TrimmedFlowObjective(objective_fn, flow_min, flow_max)


# Test the trimmed objective with detailed diagnostics
print("Testing TrimmedFlowObjective...")
print(f"\n{'='*60}")
print("DIAGNOSTIC: TrimmedFlowObjective Configuration")
print(f"{'='*60}")

base_nse = NSE()
trimmed_nse = TrimmedFlowObjective(base_nse, FLOW_MIN, FLOW_MAX)

print(f"Base objective: {base_nse}")
print(f"Base direction: {getattr(base_nse, 'direction', 'NOT SET')}")
print(f"Trimmed objective: {trimmed_nse}")
print(f"Trimmed direction: {getattr(trimmed_nse, 'direction', 'NOT SET')}")
print(f"Trimmed maximize: {getattr(trimmed_nse, 'maximize', 'NOT SET')}")

# Test with actual data from Stage A
# observed_full has NaN padding for warmup, observed_post_warmup is the actual data
print(f"\n--- Testing with Stage A model on actual data ---")
if 'NSE (default)' in loaded_reports:
    test_report = loaded_reports['NSE (default)']
    test_params = test_report.result.best_parameters.copy()
    
    print(f"\n  Report stored objective: {test_report.result.best_objective:.4f}")
    print(f"  Report method: {test_report.result.method}")
    
    # Check if report has stored simulation we can compare against
    if hasattr(test_report, 'simulated') and test_report.simulated is not None:
        print(f"  Report has stored simulation: {len(test_report.simulated)} values")
        report_sim = test_report.simulated
        report_obs = test_report.observed
        stored_nse = base_nse(report_obs, report_sim)
        print(f"  NSE from stored sim/obs: {stored_nse:.4f}")
    
    # Print some key parameter values
    print(f"\n  Sample parameters from report:")
    for i, (k, v) in enumerate(list(test_params.items())[:5]):
        print(f"    {k}: {v:.4f}")
    print(f"    ... and {len(test_params)-5} more")
    
    # Create model and run with our inputs (which are from the report)
    test_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    test_model.set_parameters(test_params)
    test_model.reset()
    test_sim = test_model.run(inputs_df)['runoff'].values
    
    # Get simulation after warmup (to match observed_post_warmup)
    sim_post_warmup = test_sim[WARMUP_DAYS:]
    
    # Compare full simulation stats (use observed_post_warmup for stats)
    print(f"\n  Full simulation stats (post-warmup):")
    print(f"    Observed: min={np.nanmin(observed_post_warmup):.2f}, max={np.nanmax(observed_post_warmup):.2f}, mean={np.nanmean(observed_post_warmup):.2f}")
    print(f"    Simulated: min={sim_post_warmup.min():.2f}, max={sim_post_warmup.max():.2f}, mean={sim_post_warmup.mean():.2f}")
    print(f"    Observed length: {len(observed_post_warmup)}, Simulated length: {len(sim_post_warmup)}")
    
    # Calculate metrics on FULL range
    full_nse = base_nse(observed_post_warmup, sim_post_warmup)
    print(f"\n  Full-range NSE (verification): {full_nse:.4f}")
    
    # Calculate metrics on TRIMMED range
    valid_mask = (observed_post_warmup >= FLOW_MIN) & (observed_post_warmup <= FLOW_MAX) & ~np.isnan(observed_post_warmup) & ~np.isnan(sim_post_warmup)
    n_valid = valid_mask.sum()
    print(f"\n  Valid points in [{FLOW_MIN}, {FLOW_MAX}] ML/day: {n_valid}")
    
    if n_valid > 10:
        obs_masked = observed_post_warmup[valid_mask]
        sim_masked = sim_post_warmup[valid_mask]
        
        print(f"\n  Observed (trimmed range):")
        print(f"    Min: {obs_masked.min():.2f}, Max: {obs_masked.max():.2f}, Mean: {obs_masked.mean():.2f}")
        print(f"  Simulated (trimmed range):")
        print(f"    Min: {sim_masked.min():.2f}, Max: {sim_masked.max():.2f}, Mean: {sim_masked.mean():.2f}")
        
        # Manual NSE calculation on trimmed
        numerator = np.sum((obs_masked - sim_masked) ** 2)
        denominator = np.sum((obs_masked - np.mean(obs_masked)) ** 2)
        manual_nse = 1.0 - numerator / denominator
        print(f"\n  Manual NSE (trimmed): {manual_nse:.4f}")
        
        # Using TrimmedFlowObjective
        trimmed_result = trimmed_nse(observed_post_warmup, sim_post_warmup)
        print(f"  TrimmedFlowObjective result: {trimmed_result:.4f}")
        
        # Show what SCE-UA would see (negated for minimization)
        print(f"\n  What SCE-UA sees (negated for minimization): {-trimmed_result:.4f}")
else:
    print("  NSE (default) report not found - skipping diagnostic")

print(f"{'='*60}")

# Create test data
test_obs = np.array([0.5, 10, 100, 500, 1500, 50, 200])  # Some outside range
test_sim = np.array([0.6, 12, 95, 480, 1400, 55, 210])

full_nse = base_nse(test_obs, test_sim)
trimmed_result = trimmed_nse(test_obs, test_sim)

print(f"  Test observed: {test_obs}")
print(f"  Test simulated: {test_sim}")
print(f"  Full NSE (all 7 points): {full_nse:.4f}")
print(f"  Trimmed NSE (only flows in range): {trimmed_result:.4f}")
print(f"  Points used: {((test_obs >= FLOW_MIN) & (test_obs <= FLOW_MAX)).sum()} / {len(test_obs)}")

# %% [markdown]
# ---
# ## Run Four-Stage Calibration with Trimmed Flow Range

# %%
# Define parameter bounds for each stage

# Routing-only bounds (Stage B)
routing_only_bounds = {
    'routing_K': (0.5, 15.0),
    'routing_m': (0.1, 1.2),
    'routing_n_subreaches': (1, 10)
}

# Joint bounds - RR + routing (Stage C)
joint_bounds = custom_bounds.copy()
joint_bounds['routing_K'] = (0.5, 15.0)
joint_bounds['routing_m'] = (0.1, 1.2)
joint_bounds['routing_n_subreaches'] = (1, 10)

# Fixed UH bounds - Sacramento without UH + routing (Stage D)
fixed_uh_bounds = {k: v for k, v in custom_bounds.items() if not k.startswith('uh')}
fixed_uh_bounds['routing_K'] = (0.5, 15.0)
fixed_uh_bounds['routing_m'] = (0.1, 1.2)
fixed_uh_bounds['routing_n_subreaches'] = (1, 10)

print("=" * 70)
print("FOUR-STAGE CALIBRATION WITH TRIMMED FLOW RANGE")
print("=" * 70)
print(f"\nFlow range: {FLOW_MIN} - {FLOW_MAX} ML/day")
print(f"Calibrating B, C, D stages for {len(available_objectives)} objective functions")

# %%
# Storage for results
all_results_trimmed = {}

# Loop through each objective function
for obj_name, config in available_objectives.items():
    print("\n" + "=" * 70)
    print(f"OBJECTIVE: {obj_name} - {config['description']}")
    print(f"(Trimmed flow range: {FLOW_MIN}-{FLOW_MAX} ML/day)")
    print("=" * 70)
    
    # Create trimmed objective function
    base_objective = config['objective_fn']()
    trimmed_objective = TrimmedFlowObjective(base_objective, FLOW_MIN, FLOW_MAX)
    
    # -------------------------------------------------------------------------
    # Calibration A: Load from saved report (same as full-range)
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration A: Load from Notebook 02 ---")
    report_A = loaded_reports[config['report_key']]
    result_A = report_A.result
    params_A = result_A.best_parameters.copy()
    
    print(f"  Loaded: {config['report_key']}")
    print(f"  Original objective: {result_A.best_objective:.4f}")
    
    # -------------------------------------------------------------------------
    # Calibration B: Add routing (fix RR params) - TRIMMED
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration B (Trimmed): Add Routing (fix RR from A) ---")
    
    rr_model_B = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    rr_model_B.set_parameters(params_A)
    router_B = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_B = RoutedModel(rr_model_B, router_B, routing_enabled=True)
    
    runner_B = CalibrationRunner(
        model=model_B,
        inputs=inputs_df,
        observed=observed_full,  # Use full observed, trimming happens in objective
        objective=trimmed_objective,
        parameter_bounds=routing_only_bounds,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"  Running SCE-UA (3 routing params, max_evals=3000)...")
    print(f"  Objective calculated only on flows in [{FLOW_MIN}, {FLOW_MAX}] ML/day")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result_B = runner_B.run_sceua_direct(
            max_evals=3000, seed=42, verbose=True,
            max_tolerant_iter=100, tolerance=1e-4
        )
    
    routing_params_B = result_B.best_parameters.copy()
    K_B = routing_params_B.get('routing_K', 5.0)
    m_B = routing_params_B.get('routing_m', 0.8)
    n_B = int(round(routing_params_B.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective (trimmed): {result_B.best_objective:.4f}")
    print(f"  Routing: K={K_B:.2f}, m={m_B:.3f}, n={n_B}")
    
    # Save report
    report_B = runner_B.create_report(result_B, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_B_filename = f"410734_{obj_name.lower()}_routing_B_trimmed"
    report_B.save(REPORTS_DIR / report_B_filename)
    print(f"  Saved: {report_B_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Calibration C: Joint (RR + routing) - TRIMMED
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration C (Trimmed): Joint Calibration (RR + Routing) ---")
    
    rr_model_C = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    router_C = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_C = RoutedModel(rr_model_C, router_C, routing_enabled=True)
    
    runner_C = CalibrationRunner(
        model=model_C,
        inputs=inputs_df,
        observed=observed_full,
        objective=trimmed_objective,
        parameter_bounds=joint_bounds,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"  Running SCE-UA (25 params, max_evals=15000)...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result_C = runner_C.run_sceua_direct(
            max_evals=15000, seed=42, verbose=True,
            max_tolerant_iter=100, tolerance=1e-4
        )
    
    params_C = result_C.best_parameters.copy()
    rr_params_C = {k: v for k, v in params_C.items() if not k.startswith('routing_')}
    K_C = params_C.get('routing_K', 5.0)
    m_C = params_C.get('routing_m', 0.8)
    n_C = int(round(params_C.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective (trimmed): {result_C.best_objective:.4f}")
    print(f"  Routing: K={K_C:.2f}, m={m_C:.3f}, n={n_C}")
    
    # Save report
    report_C = runner_C.create_report(result_C, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_C_filename = f"410734_{obj_name.lower()}_routing_C_trimmed"
    report_C.save(REPORTS_DIR / report_C_filename)
    print(f"  Saved: {report_C_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Calibration D: Fixed UH + Routing - TRIMMED
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration D (Trimmed): Fixed UH + Routing ---")
    
    rr_model_D = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    fixed_uh_params = {'uh1': 1.0, 'uh2': 0.0, 'uh3': 0.0, 'uh4': 0.0, 'uh5': 0.0}
    router_D = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_D = RoutedModel(rr_model_D, router_D, routing_enabled=True)
    
    runner_D = CalibrationRunner(
        model=model_D,
        inputs=inputs_df,
        observed=observed_full,
        objective=trimmed_objective,
        parameter_bounds=fixed_uh_bounds,
        warmup_period=WARMUP_DAYS
    )
    
    print(f"  Running SCE-UA ({len(fixed_uh_bounds)} params, max_evals=15000)...")
    print(f"  Note: UH fixed at uh1=1.0, uh2-5=0.0 (no internal delay)")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result_D = runner_D.run_sceua_direct(
            max_evals=15000, seed=42, verbose=True,
            max_tolerant_iter=100, tolerance=1e-4
        )
    
    params_D = result_D.best_parameters.copy()
    params_D.update(fixed_uh_params)
    rr_params_D = {k: v for k, v in params_D.items() if not k.startswith('routing_')}
    K_D = params_D.get('routing_K', 5.0)
    m_D = params_D.get('routing_m', 0.8)
    n_D = int(round(params_D.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective (trimmed): {result_D.best_objective:.4f}")
    print(f"  Routing: K={K_D:.2f}, m={m_D:.3f}, n={n_D}")
    
    # Save report
    report_D = runner_D.create_report(result_D, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_D_filename = f"410734_{obj_name.lower()}_routing_D_trimmed"
    report_D.save(REPORTS_DIR / report_D_filename)
    print(f"  Saved: {report_D_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    all_results_trimmed[obj_name] = {
        'A': {'result': result_A, 'params': params_A},
        'B': {'result': result_B, 'params_rr': params_A, 'routing': {'K': K_B, 'm': m_B, 'n': n_B}},
        'C': {'result': result_C, 'params_rr': rr_params_C, 'routing': {'K': K_C, 'm': m_C, 'n': n_C}},
        'D': {'result': result_D, 'params_rr': rr_params_D, 'routing': {'K': K_D, 'm': m_D, 'n': n_D}},
    }
    
    print(f"\n✓ {obj_name} complete!")

print("\n" + "=" * 70)
print("ALL TRIMMED CALIBRATIONS COMPLETE")
print("=" * 70)

# %% [markdown]
# ---
# ## Load Full-Range Results from Notebook 03 for Comparison

# %%
print("=" * 70)
print("LOADING FULL-RANGE RESULTS FROM NOTEBOOK 03")
print("=" * 70)

# Load full-range routing results
all_results_full = {}

for obj_name in available_objectives.keys():
    obj_lower = obj_name.lower()
    
    # Load B, C, D from Notebook 03
    results_obj = {'A': all_results_trimmed[obj_name]['A']}  # A is same
    
    for stage in ['B', 'C', 'D']:
        filename = f"410734_{obj_lower}_routing_{stage}.pkl"
        filepath = REPORTS_DIR / filename
        
        if filepath.exists():
            try:
                report = CalibrationReport.load(filepath)
                params = report.result.best_parameters.copy()
                rr_params = {k: v for k, v in params.items() if not k.startswith('routing_')}
                routing = {
                    'K': params.get('routing_K', 5.0),
                    'm': params.get('routing_m', 0.8),
                    'n': int(round(params.get('routing_n_subreaches', 3)))
                }
                results_obj[stage] = {
                    'result': report.result,
                    'params_rr': rr_params if stage != 'A' else params,
                    'routing': routing
                }
                print(f"  ✓ Loaded: {filename}")
            except Exception as e:
                print(f"  ✗ Failed: {filename} - {e}")
        else:
            print(f"  ✗ Not found: {filename}")
    
    if len(results_obj) == 4:  # A, B, C, D
        all_results_full[obj_name] = results_obj

print(f"\nLoaded full-range results for {len(all_results_full)} objectives")

# %% [markdown]
# ---
# ## Generate Simulations and Calculate Metrics

# %%
print("=" * 70)
print("GENERATING SIMULATIONS AND CALCULATING METRICS")
print("=" * 70)

# Metrics calculators
nse_metric = NSE()
kge_metric = KGE()
rmse_metric = RMSE()
pbias_metric = PBIAS()
lognse_metric = NSE(transform=FlowTransformation('log'))
sqrtnse_metric = NSE(transform=FlowTransformation('sqrt'))
invnse_metric = NSE(transform=FlowTransformation('inverse'))

# Trimmed metrics (only calculate on flows in range)
nse_trimmed = TrimmedFlowObjective(nse_metric, FLOW_MIN, FLOW_MAX)
kge_trimmed = TrimmedFlowObjective(kge_metric, FLOW_MIN, FLOW_MAX)

obs_full = observed_full[WARMUP_DAYS:]
valid_mask = (obs_full >= FLOW_MIN) & (obs_full <= FLOW_MAX) & ~np.isnan(obs_full)

# Storage for comparison results
comparison_results = []
simulations_trimmed = {}
simulations_full = {}

for obj_name in available_objectives.keys():
    print(f"\nProcessing {obj_name}...")
    
    # Get trimmed calibration results
    results_t = all_results_trimmed[obj_name]
    
    # Generate simulations for trimmed calibration
    sims_t = {}
    for stage in ['A', 'B', 'C', 'D']:
        if stage == 'A':
            model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
            model.set_parameters(results_t['A']['params'])
            model.reset()
            sim = model.run(inputs_df)['runoff'].values[WARMUP_DAYS:]
        else:
            params_rr = results_t[stage].get('params_rr', results_t['A']['params'])
            routing = results_t[stage]['routing']
            
            model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
            model.set_parameters(params_rr)
            model.reset()
            sim_direct = model.run(inputs_df)['runoff'].values
            router = NonlinearMuskingumRouter(K=routing['K'], m=routing['m'], n_subreaches=routing['n'])
            sim = router.route(sim_direct, dt=1.0)[WARMUP_DAYS:]
        
        sims_t[stage] = sim
    
    simulations_trimmed[obj_name] = sims_t
    
    # Generate simulations for full-range calibration (if available)
    if obj_name in all_results_full:
        results_f = all_results_full[obj_name]
        sims_f = {}
        
        for stage in ['A', 'B', 'C', 'D']:
            if stage == 'A':
                sims_f[stage] = sims_t['A']  # Same
            else:
                params_rr = results_f[stage].get('params_rr', results_f['A']['params'])
                routing = results_f[stage]['routing']
                
                model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
                model.set_parameters(params_rr)
                model.reset()
                sim_direct = model.run(inputs_df)['runoff'].values
                router = NonlinearMuskingumRouter(K=routing['K'], m=routing['m'], n_subreaches=routing['n'])
                sims_f[stage] = router.route(sim_direct, dt=1.0)[WARMUP_DAYS:]
        
        simulations_full[obj_name] = sims_f
    
    # Calculate metrics for both calibration approaches
    for cal_type, sims, results in [('Trimmed', sims_t, results_t), ('Full', simulations_full.get(obj_name, {}), all_results_full.get(obj_name, {}))]:
        if not sims:
            continue
        
        for stage in ['A', 'B', 'C', 'D']:
            sim = sims[stage]
            routing = results.get(stage, {}).get('routing', {})
            
            # Full-range metrics
            comparison_results.append({
                'Objective': obj_name,
                'Calibration': cal_type,
                'Stage': stage,
                'NSE_full': nse_metric(obs_full, sim),
                'KGE_full': kge_metric(obs_full, sim),
                'PBIAS_full': pbias_metric(obs_full, sim),
                'LogNSE_full': lognse_metric(obs_full, sim),
                'SqrtNSE_full': sqrtnse_metric(obs_full, sim),
                # Trimmed-range metrics
                'NSE_trimmed': nse_trimmed(obs_full, sim),
                'KGE_trimmed': kge_trimmed(obs_full, sim),
                # Routing parameters
                'K': routing.get('K'),
                'm': routing.get('m'),
                'n': routing.get('n'),
            })

print("\n✓ All simulations complete!")

# %%
# Create results DataFrame
results_df = pd.DataFrame(comparison_results)

print("\n" + "=" * 90)
print("COMPARISON: TRIMMED vs FULL-RANGE CALIBRATION")
print("=" * 90)

# Pivot table comparing calibration approaches
for metric in ['NSE_full', 'NSE_trimmed', 'KGE_full']:
    print(f"\n=== {metric} ===")
    pivot = results_df.pivot_table(
        index=['Objective', 'Stage'],
        columns='Calibration',
        values=metric
    )
    if 'Trimmed' in pivot.columns and 'Full' in pivot.columns:
        pivot['Δ (Trimmed-Full)'] = pivot['Trimmed'] - pivot['Full']
    print(pivot.round(4).to_string())

# %% [markdown]
# ---
# ## Visualization: Trimmed vs Full-Range Comparison

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 70)
print("VISUALIZATION: TRIMMED vs FULL-RANGE COMPARISON")
print("=" * 70)

# Create comparison figure
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        '<b>NSE on Full Range</b>',
        '<b>NSE on Trimmed Range</b>',
        '<b>K Parameter Comparison</b>',
        '<b>KGE on Full Range</b>',
        '<b>Trimmed-Full Difference (NSE)</b>',
        '<b>Routing Parameter m</b>'
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.08
)

colors_cal = {'Full': '#3498db', 'Trimmed': '#e74c3c'}
stages = ['A', 'B', 'C', 'D']

# Get unique objectives
objectives = results_df['Objective'].unique()

# Row 1: NSE comparisons
for cal_type in ['Full', 'Trimmed']:
    data = results_df[results_df['Calibration'] == cal_type]
    for stage in stages:
        stage_data = data[data['Stage'] == stage]
        if len(stage_data) > 0:
            fig.add_trace(go.Bar(
                name=f'{cal_type}-{stage}' if stage == 'A' else None,
                x=[f"{row['Objective']}-{stage}" for _, row in stage_data.iterrows()],
                y=stage_data['NSE_full'],
                marker_color=colors_cal[cal_type],
                opacity=0.7 if cal_type == 'Trimmed' else 1.0,
                showlegend=(stage == 'A')
            ), row=1, col=1)

# Similar for other panels...
fig.update_layout(
    height=700,
    width=1400,
    title_text=f"<b>Calibration Comparison: Trimmed ({FLOW_MIN}-{FLOW_MAX} ML/day) vs Full Range</b>",
    barmode='group'
)

fig.show()

# %% [markdown]
# ---
# ## Summary: Key Findings

# %%
print("\n" + "=" * 90)
print("SUMMARY: TRIMMED vs FULL-RANGE CALIBRATION")
print("=" * 90)

# Calculate summary statistics
trimmed_data = results_df[results_df['Calibration'] == 'Trimmed']
full_data = results_df[results_df['Calibration'] == 'Full']

if len(full_data) > 0:
    print("\n=== Performance on FULL Flow Range ===")
    for stage in ['B', 'C', 'D']:
        t_nse = trimmed_data[trimmed_data['Stage'] == stage]['NSE_full'].mean()
        f_nse = full_data[full_data['Stage'] == stage]['NSE_full'].mean()
        diff = t_nse - f_nse
        print(f"  Stage {stage}: Trimmed={t_nse:.4f}, Full={f_nse:.4f}, Δ={diff:+.4f}")
    
    print("\n=== Performance on TRIMMED Flow Range ===")
    for stage in ['B', 'C', 'D']:
        t_nse = trimmed_data[trimmed_data['Stage'] == stage]['NSE_trimmed'].mean()
        f_nse = full_data[full_data['Stage'] == stage]['NSE_trimmed'].mean()
        diff = t_nse - f_nse
        print(f"  Stage {stage}: Trimmed={t_nse:.4f}, Full={f_nse:.4f}, Δ={diff:+.4f}")
    
    print("\n=== Routing Parameter K Comparison ===")
    for stage in ['B', 'C', 'D']:
        t_k = trimmed_data[(trimmed_data['Stage'] == stage) & (trimmed_data['K'].notna())]['K'].mean()
        f_k = full_data[(full_data['Stage'] == stage) & (full_data['K'].notna())]['K'].mean()
        diff = t_k - f_k
        print(f"  Stage {stage}: Trimmed K={t_k:.2f}, Full K={f_k:.2f}, Δ={diff:+.2f} days")

print("\n" + "=" * 70)
print("NOTEBOOK COMPLETE")
print("=" * 70)
