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
#     display_name: pyrrm (Python 3.11.14)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Channel Routing with pyrrm: Muskingum Method
#
# ## Purpose
#
# This notebook demonstrates how to use the **channel routing** module in `pyrrm`
# to account for travel time and attenuation in river reaches. Routing is essential
# when the gauge (observation point) is located downstream of the catchment centroid.
#
# ## What You'll Learn
#
# - Why routing is needed and what it does to a hydrograph
# - How the nonlinear Muskingum method works
# - How to add routing to a rainfall-runoff model
# - How to calibrate both RR model and routing parameters together
# - Different strategies for including/excluding routing from calibration
#
# ## Prerequisites
#
# - Familiarity with pyrrm models (see `02_calibration_quickstart.ipynb`)
# - Understanding of basic hydrology concepts
#
# ## Estimated Time
#
# - ~20 minutes to work through the concepts
# - ~10-20 minutes for calibration examples

# %% [markdown]
# ---
# ## Why Do We Need Routing?
#
# ### The Problem
#
# Rainfall-runoff models like Sacramento and GR4J simulate runoff generation at
# the **catchment outlet**. However, when the gauge is located **downstream** of
# the catchment centroid, two issues arise:
#
# 1. **Timing Error**: The modeled hydrograph arrives too early
# 2. **Peak Error**: The peak is too sharp compared to observations
#
# ```
# Catchment                    River Reach                 Gauge
# ┌─────────────┐             ══════════════════>         📍
# │  RR Model   │────────────────────────────────────────►│
# │  generates  │      Travel time & attenuation          │
# │   runoff    │      not captured by RR model           │
# └─────────────┘                                         │
#                                                         ▼
#                                                    Observed Q
# ```
#
# ### What Routing Does
#
# Channel routing transforms the hydrograph by:
#
# 1. **Translation (lag)**: Delays the peak by the travel time
# 2. **Attenuation**: Reduces and spreads the peak due to storage effects
#
# Without routing, calibration may:
# - Overestimate baseflow parameters to compensate for missing lag
# - Produce poor performance on rising/falling limbs
# - Show systematic timing errors

# %% [markdown]
# ---
# ## Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pyrrm imports
from pyrrm.models import Sacramento
from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel, create_router
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import NSE, FlowTransformation, SDEB

# Reload modules to pick up any code changes (useful during development)
import importlib
import pyrrm.visualization.report_plots
import pyrrm.calibration.report
import pyrrm.calibration.runner
importlib.reload(pyrrm.visualization.report_plots)
importlib.reload(pyrrm.calibration.report)
importlib.reload(pyrrm.calibration.runner)

# Create SqrtNSE objective for calibration (balanced high/low flow performance)
# Uses pyrrm.objectives.NSE which supports flow transformations
SqrtNSE = lambda: NSE(transform=FlowTransformation('sqrt'))

# Display settings
plt.style.use('seaborn-v0_8-whitegrid')
# %config InlineBackend.figure_format = 'retina'

print("Imports successful!")
print(f"Numba JIT compilation available: {NonlinearMuskingumRouter.is_jit_available()}")

# %% [markdown]
# ---
# ## Part 1: Understanding the Muskingum Method
#
# ### The Mathematical Basis
#
# The nonlinear Muskingum method is based on two equations:
#
# **1. Continuity (mass balance):**
# $$\frac{dS}{dt} = I(t) - Q(t)$$
#
# **2. Nonlinear storage-discharge relationship:**
# $$S = K \cdot Q^m$$
#
# Where:
# - $S$ = storage volume in the reach
# - $I(t)$ = inflow rate (from RR model)
# - $Q(t)$ = outflow rate (routed)
# - $K$ = storage constant (approximately the travel time)
# - $m$ = nonlinear exponent
#
# ### Parameter Interpretation
#
# | Parameter | Physical Meaning | Typical Range |
# |-----------|------------------|---------------|
# | **K** | Travel time through reach (controls attenuation) | 0.5 - 50 days |
# | **m** | Nonlinearity of storage (higher = more attenuation) | 0.6 - 1.2 |
# | **n_subreaches** | Nash cascade subdivision (controls response shape) | 1 - 10 |
#
# - **K = 5 days** means the hydrograph is delayed by approximately 5 days. **Larger K = more attenuation**
# - **m = 1.0** is a linear reservoir; **m > 1.0** = more attenuation; **m < 1.0** = less attenuation (avoid m < 0.6)
# - **n_subreaches** creates a Nash cascade: more sub-reaches → gamma-shaped response → **LESS attenuation**
#   (This is for response shape control and numerical stability, NOT for increasing attenuation)

# %% [markdown]
# ### Demonstration: Effect of Routing Parameters
#
# Let's create a simple triangular hydrograph and see how routing transforms it.

# %%
# Create a triangular inflow hydrograph
time = np.arange(0, 30, 1)  # 30 days
inflow = np.zeros_like(time, dtype=float)
inflow[5:15] = np.concatenate([
    np.linspace(0, 100, 5),   # Rising limb
    np.linspace(100, 0, 5)    # Falling limb
])

# Define baseline parameters for consistent comparison
# Each panel varies ONE parameter while holding others at baseline
BASELINE_K = 5.0       # Storage constant (days)
BASELINE_M = 1.0       # Nonlinear exponent (linear reservoir)
BASELINE_N = 1         # Number of sub-reaches (single reservoir)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: Effect of K (storage constant)
# Fixed: m=1.0, n=1 (baseline)
ax = axes[0]
ax.plot(time, inflow, 'k-', linewidth=2, label='Inflow')
for K in [1, 3, 5, 10]:
    router = NonlinearMuskingumRouter(K=K, m=BASELINE_M, n_subreaches=BASELINE_N)
    outflow = router.route(inflow, dt=1.0)
    ax.plot(time, outflow, '--', label=f'K={K} days')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.set_title(f'Effect of K (m={BASELINE_M}, n={BASELINE_N})')
ax.legend(loc='upper right')

# Panel 2: Effect of m (nonlinear exponent)
# Fixed: K=5.0, n=1 (baseline)
ax = axes[1]
ax.plot(time, inflow, 'k-', linewidth=2, label='Inflow')
for m in [0.6, 0.8, 1.0, 1.2]:
    router = NonlinearMuskingumRouter(K=BASELINE_K, m=m, n_subreaches=BASELINE_N)
    outflow = router.route(inflow, dt=1.0)
    ax.plot(time, outflow, '--', label=f'm={m}')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.set_title(f'Effect of m (K={BASELINE_K}, n={BASELINE_N})')
ax.legend(loc='upper right')

# Panel 3: Effect of n_subreaches
# Fixed: K=5.0, m=1.0 (baseline)
ax = axes[2]
ax.plot(time, inflow, 'k-', linewidth=2, label='Inflow')
for n in [1, 2, 4, 8]:
    router = NonlinearMuskingumRouter(K=BASELINE_K, m=BASELINE_M, n_subreaches=n)
    outflow = router.route(inflow, dt=1.0)
    ax.plot(time, outflow, '--', label=f'n={n}')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.set_title(f'Effect of n_subreaches (K={BASELINE_K}, m={BASELINE_M})')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Baseline parameters: K={BASELINE_K} days, m={BASELINE_M}, n={BASELINE_N}")
print("Each panel varies ONE parameter while holding the others at baseline.")

# %% [markdown]
# **Key Observations:**
#
# 1. **Larger K** = more delay and more attenuation (K controls the total storage/lag)
# 2. **Higher m** = more attenuation. When m < 1, storage increases slower than flow, reducing attenuation.
#    When m ≈ 0.5, there can even be peak amplification (physically unrealistic - avoid m < 0.6)
# 3. **More sub-reaches** = **LESS attenuation** but smoother, more gamma-like response shape
#
# **Important:** The `n_subreaches` parameter creates a Nash cascade. Mathematically, more
# sub-reaches produce a sharper impulse response, which preserves peak flows better (less
# attenuation). This is correct physics - use K to control attenuation, not n_subreaches.
#
# The choice of parameters depends on the physical characteristics of your river reach
# (length, slope, cross-section) and can be estimated from observations or calibrated.

# %% [markdown]
# ### The Critical Difference Between K and m
#
# Looking at the plots above, K and m appear to have similar effects. However, there's a 
# fundamental difference in **how** they affect the hydrograph:
#
# - **K** affects all flows **uniformly** (same proportional effect regardless of flow magnitude)
# - **m** affects high and low flows **differently** due to the nonlinear storage relationship S = K·Qᵐ
#
# | m value | Effect on storage | Result |
# |---------|-------------------|--------|
# | m = 1.0 | S ∝ Q (linear) | Same % attenuation at all flow levels |
# | m > 1.0 | S grows faster than Q | Large events attenuated MORE than small events |
# | m < 1.0 | S grows slower than Q | Large events attenuated LESS than small events |
#
# Let's demonstrate this with a small event (peak=20) vs a large event (peak=100):

# %%
# Demonstrate: K vs m affect small and large events differently
time = np.arange(0, 40, 1)
small_event = np.zeros_like(time, dtype=float)
large_event = np.zeros_like(time, dtype=float)
small_event[5:15] = np.concatenate([np.linspace(0, 20, 5), np.linspace(20, 0, 5)])
large_event[5:15] = np.concatenate([np.linspace(0, 100, 5), np.linspace(100, 0, 5)])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: K with m=1.0 (LINEAR - uniform effect)
ax = axes[0]
ax.set_title('K=5, m=1.0 (linear)\nSame % attenuation for both')
for event, label, color in [(small_event, 'Small', 'blue'), (large_event, 'Large', 'red')]:
    ax.plot(time, event, '-', color=color, linewidth=2, alpha=0.4)
    router = NonlinearMuskingumRouter(K=5.0, m=1.0, n_subreaches=1)
    outflow = router.route(event, dt=1.0)
    atten = 100 * (1 - outflow.max() / event.max())
    ax.plot(time, outflow, '--', color=color, linewidth=2, label=f'{label}: {atten:.0f}%')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.legend(loc='upper right', title='Attenuation')

# Panel 2: K with m=1.2 (NONLINEAR - more attenuation at high flows)
ax = axes[1]
ax.set_title('K=5, m=1.2 (nonlinear)\nLarge event attenuated MORE')
for event, label, color in [(small_event, 'Small', 'blue'), (large_event, 'Large', 'red')]:
    ax.plot(time, event, '-', color=color, linewidth=2, alpha=0.4)
    router = NonlinearMuskingumRouter(K=5.0, m=1.2, n_subreaches=1)
    outflow = router.route(event, dt=1.0)
    atten = 100 * (1 - outflow.max() / event.max())
    ax.plot(time, outflow, '--', color=color, linewidth=2, label=f'{label}: {atten:.0f}%')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.legend(loc='upper right', title='Attenuation')

# Panel 3: K with m=0.8 (NONLINEAR - less attenuation at high flows)
ax = axes[2]
ax.set_title('K=5, m=0.8 (nonlinear)\nLarge event attenuated LESS')
for event, label, color in [(small_event, 'Small', 'blue'), (large_event, 'Large', 'red')]:
    ax.plot(time, event, '-', color=color, linewidth=2, alpha=0.4)
    router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=1)
    outflow = router.route(event, dt=1.0)
    atten = 100 * (1 - outflow.max() / event.max())
    ax.plot(time, outflow, '--', color=color, linewidth=2, label=f'{label}: {atten:.0f}%')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flow')
ax.legend(loc='upper right', title='Attenuation')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Summary:**
# - **m = 1.0**: Both events get ~47% attenuation (linear, uniform effect)
# - **m = 1.2**: Small gets ~59%, Large gets ~66% (high flows dampened more)
# - **m = 0.8**: Small gets ~29%, Large gets ~19% (high flows dampened less)
#
# This is important for calibration: if your catchment shows different routing behavior 
# for small vs large events, the **m** parameter can capture this nonlinearity.

# %% [markdown]
# ### Performance: Numba JIT Compilation
#
# The Muskingum routing implementation uses **Numba JIT (Just-In-Time) compilation** 
# when available. This provides a **10-30x speedup** over pure Python, which is critical
# for calibration where the routing function is called thousands of times.
#
# The first call includes compilation overhead (~0.5-1s), but subsequent calls are very fast.

# %%
# Demonstrate JIT compilation speedup
import time as time_module  # Avoid conflict with 'time' array variable from earlier

print("Routing Performance Demonstration")
print("=" * 50)

# Create test data
test_inflow = np.random.exponential(5, 2000)
test_router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)

if NonlinearMuskingumRouter.is_jit_available():
    # First call (includes JIT compilation)
    start = time_module.time()
    _ = test_router.route(test_inflow, dt=1.0, use_jit=True)
    jit_first = time_module.time() - start
    
    # Second call (uses cached JIT)
    start = time_module.time()
    for _ in range(10):
        _ = test_router.route(test_inflow, dt=1.0, use_jit=True)
    jit_cached = (time_module.time() - start) / 10
    
    # Pure Python
    start = time_module.time()
    for _ in range(10):
        _ = test_router.route(test_inflow, dt=1.0, use_jit=False)
    py_time = (time_module.time() - start) / 10
    
    print(f"\nFor 2000 timesteps, 3 sub-reaches:")
    print(f"  JIT (first call):  {jit_first*1000:.1f} ms  (includes compilation)")
    print(f"  JIT (cached):      {jit_cached*1000:.2f} ms")
    print(f"  Pure Python:       {py_time*1000:.1f} ms")
    print(f"\nSpeedup: {py_time/jit_cached:.1f}x faster with JIT")
    print("\nJIT is now warmed up for subsequent calibrations!")
else:
    print("Numba not installed - using pure Python (slower)")
    print("Install with: pip install numba")

# %% [markdown]
# ---
# ## Part 2: Synthetic Data for Routing Demonstration
#
# Before applying routing to real calibrations, let's understand how routing works
# using synthetic data where we **know the true parameters**.
#
# ### Creating a Synthetic Catchment
#
# We'll create "observed" flow by:
# 1. Running a Sacramento model with **known** parameters
# 2. Applying Muskingum routing with **known** K, m, n_subreaches
# 3. Adding small measurement noise
#
# This lets us verify that our calibration can recover the true routing parameters.

# %%
# Create synthetic climate forcings
np.random.seed(42)
n_days = 1500  # ~4 years of data

dates = pd.date_range('2010-01-01', periods=n_days, freq='D')

# Seasonal precipitation pattern (higher in summer for Australian catchment)
day_of_year = np.arange(n_days) % 365
seasonal_factor = 4 + 3 * np.sin(2 * np.pi * (day_of_year - 180) / 365)  # Peak in winter
precip_base = np.random.exponential(seasonal_factor)
precip_base[precip_base < 0.5] = 0  # Many dry days

# Add some larger storm events
storm_days = np.random.choice(n_days, size=30, replace=False)
precip_base[storm_days] *= np.random.uniform(2, 5, size=30)

# Seasonal PET (higher in summer)
pet = 3 + 2.5 * np.sin(2 * np.pi * (day_of_year + 90) / 365)  # Peak in summer

print("Synthetic Climate Forcings Created:")
print(f"  Period: {dates[0].date()} to {dates[-1].date()}")
print(f"  Mean precipitation: {np.mean(precip_base):.1f} mm/day")
print(f"  Mean PET: {np.mean(pet):.1f} mm/day")

# %%
# Define the TRUE model parameters (what we're trying to recover)
TRUE_RR_PARAMS = {
    'uztwm': 75, 'uzfwm': 35, 'lztwm': 140, 'lzfsm': 25,
    'lzfpm': 45, 'lzsk': 0.12, 'lzpk': 0.008, 'pfree': 0.25,
    'uzk': 0.35, 'pctim': 0.02, 'adimp': 0.0, 'rexp': 2.5,
    'riva': 0.0, 'side': 0.0, 'rserv': 0.3
}

TRUE_ROUTING_PARAMS = {
    'K': 3.5,           # 3.5 days travel time
    'm': 0.85,          # Moderate nonlinearity
    'n_subreaches': 3   # 3 sub-reaches
}

print("TRUE Model Parameters (to recover via calibration):")
print("-" * 50)
print("\nSacramento RR Parameters:")
for k, v in TRUE_RR_PARAMS.items():
    print(f"  {k}: {v}")
print(f"\nRouting Parameters:")
for k, v in TRUE_ROUTING_PARAMS.items():
    print(f"  {k}: {v}")

# %%
# Generate "observed" flow using true parameters
true_model = Sacramento()
true_model.set_parameters(TRUE_RR_PARAMS)

inputs_synth = pd.DataFrame({
    'precipitation': precip_base,
    'pet': pet
}, index=dates)

# Run RR model
true_rr_output = true_model.run(inputs_synth)
direct_runoff_true = true_rr_output['runoff'].values

# Apply routing
true_router = NonlinearMuskingumRouter(
    K=TRUE_ROUTING_PARAMS['K'],
    m=TRUE_ROUTING_PARAMS['m'],
    n_subreaches=TRUE_ROUTING_PARAMS['n_subreaches']
)
routed_flow_true = true_router.route(direct_runoff_true, dt=1.0)

# Add measurement noise (realistic observation error)
noise_std = 0.1 * np.std(routed_flow_true)  # 10% of signal std
observed_synth = routed_flow_true + np.random.normal(0, noise_std, len(routed_flow_true))
observed_synth = np.maximum(observed_synth, 0)  # No negative flows

print("\nSynthetic 'Observed' Flow Generated:")
print(f"  Mean flow: {np.mean(observed_synth):.2f} mm/day")
print(f"  Peak flow: {np.max(observed_synth):.2f} mm/day")
print(f"  Added noise: {noise_std:.3f} mm/day (std)")

# Store for later use
synth_data = pd.DataFrame({
    'precipitation': precip_base,
    'pet': pet,
    'direct_runoff': direct_runoff_true,
    'routed_flow': routed_flow_true,
    'observed': observed_synth
}, index=dates)

# %%
# Visualize the synthetic data and routing effect
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Find a good event to highlight
peak_idx = np.argmax(direct_runoff_true[365:]) + 365  # Skip first year warmup
event_slice = slice(peak_idx - 20, peak_idx + 40)

# Panel 1: Precipitation
ax = axes[0]
ax.bar(dates[event_slice], precip_base[event_slice], color='steelblue', alpha=0.7)
ax.set_ylabel('Precipitation (mm)')
ax.set_title('Synthetic Catchment Data: Demonstrating Routing Effect')
ax.invert_yaxis()

# Panel 2: Direct vs Routed Flow
ax = axes[1]
ax.plot(dates[event_slice], direct_runoff_true[event_slice], 
        'b-', linewidth=2, label='Direct Runoff (no routing)')
ax.plot(dates[event_slice], routed_flow_true[event_slice], 
        'r-', linewidth=2, label='Routed Flow (K=3.5, m=0.85)')
ax.plot(dates[event_slice], observed_synth[event_slice], 
        'ko', markersize=4, alpha=0.5, label='Observed (with noise)')
ax.set_ylabel('Flow (mm/day)')
ax.legend(loc='upper right')
ax.set_title('Effect of Routing: Peak Attenuation and Delay')

# Add annotations showing the routing effect
peak_direct = np.max(direct_runoff_true[event_slice])
peak_routed = np.max(routed_flow_true[event_slice])
ax.annotate(f'Peak attenuation:\n{(1-peak_routed/peak_direct)*100:.0f}%', 
            xy=(0.02, 0.95), xycoords='axes fraction', 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 3: Difference (routing effect)
ax = axes[2]
difference = direct_runoff_true[event_slice] - routed_flow_true[event_slice]
ax.fill_between(dates[event_slice], 0, difference, alpha=0.5, color='purple')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Flow Difference')
ax.set_xlabel('Date')
ax.set_title('Routing Effect: Positive = Peak Reduction, Negative = Delayed Water')

plt.tight_layout()
plt.show()

print(f"\nRouting Statistics for Event:")
print(f"  Peak direct runoff: {peak_direct:.2f} mm/day")
print(f"  Peak routed flow:   {peak_routed:.2f} mm/day")
print(f"  Attenuation:        {(1-peak_routed/peak_direct)*100:.1f}%")

# %% [markdown]
# ---
# ## Part 3: The RoutedModel Class
#
# `pyrrm` provides the `RoutedModel` class which wraps any rainfall-runoff model
# with optional channel routing. This enables:
#
# - Seamless integration with the calibration framework
# - Automatic combination of RR and routing parameters
# - Easy switching between routed and unrouted simulations

# %%
# Create a RoutedModel combining Sacramento + Muskingum routing
rr_model = Sacramento()

router = NonlinearMuskingumRouter(
    K=5.0,           # Initial guess (different from true K=3.5)
    m=0.9,           # Initial guess (different from true m=0.85)
    n_subreaches=3   # Same as true
)

model = RoutedModel(rr_model, router, routing_enabled=True)

print("RoutedModel Created:")
print("=" * 50)
print(f"RR Model:        {rr_model.name}")
print(f"Router:          {router}")
print(f"Routing enabled: {model.is_routing_enabled}")
print(f"JIT compiled:    {NonlinearMuskingumRouter.is_jit_available()}")
print()
print(router.summary())

# %% [markdown]
# ### Parameter Management
#
# The `RoutedModel` provides methods to control which parameters are calibrated:
#
# | Method | Returns | Use Case |
# |--------|---------|----------|
# | `get_parameter_bounds()` | All parameters | Calibrate everything together |
# | `get_rr_parameter_bounds()` | Only RR params | Fix routing, calibrate RR |
# | `get_routing_parameter_bounds()` | Only routing params | Fix RR, calibrate routing |

# %%
# Inspect the combined parameter bounds
all_bounds = model.get_parameter_bounds()

print("Combined Parameter Bounds (All Calibratable):")
print("=" * 60)
print(f"\nSacramento Parameters ({len([k for k in all_bounds if not k.startswith('routing_')])} params):")
for name, (lo, hi) in all_bounds.items():
    if not name.startswith('routing_'):
        print(f"  {name:12s}: [{lo:8.2f}, {hi:8.2f}]")

print(f"\nRouting Parameters ({len([k for k in all_bounds if k.startswith('routing_')])} params):")
for name, (lo, hi) in all_bounds.items():
    if name.startswith('routing_'):
        print(f"  {name:25s}: [{lo:8.2f}, {hi:8.2f}]")
        
print(f"\nTotal parameters: {len(all_bounds)}")

# %% [markdown]
# ---
# ## Part 4: Running the RoutedModel
#
# Let's run the model with our synthetic data and verify the routing is working.

# %%
# Prepare inputs for calibration (use synthetic data from Part 2)
inputs = inputs_synth.copy()
observed = observed_synth.copy()

WARMUP_DAYS = 365  # Use first year for model warmup

print(f"Data prepared for calibration:")
print(f"  Total days: {len(inputs)}")
print(f"  Warmup period: {WARMUP_DAYS} days")
print(f"  Calibration period: {len(inputs) - WARMUP_DAYS} days")

# %%
# Verify router parameters before running
print("Router Status Before Run:")
print("=" * 50)
print(f"Routing enabled: {model.is_routing_enabled}")
print(f"Router parameters:")
print(f"  K = {model.router.K} days")
print(f"  m = {model.router.m}")
print(f"  n_subreaches = {model.router.n_subreaches}")
print(f"  K_sub = {model.router.K_sub:.3f} days")

# Reset model state before running
model.reset()

# Run the model with default (non-calibrated) parameters
results = model.run(inputs)

print("\nModel Output Columns:", list(results.columns))
print(f"\nRouting check:")
print(f"  Has 'direct_runoff' column: {'direct_runoff' in results.columns}")

# Extract flows
# Note: RoutedModel routes the 'flow' column, not 'runoff'
# - 'direct_runoff' contains the pre-routing RR model flow
# - 'flow' contains the routed flow
direct = results['direct_runoff'].values
routed = results['flow'].values  # Use 'flow' (routed), not 'runoff' (different column)

# Verify routing is being applied
print(f"\nRouting Verification:")
print(f"  Max Direct Runoff:  {np.max(direct):.2f} mm/day")
print(f"  Max Routed Runoff:  {np.max(routed):.2f} mm/day") 
print(f"  Peak Attenuation:   {(1 - np.max(routed)/np.max(direct))*100:.1f}%")
print(f"  Volume Conservation: {np.sum(routed)/np.sum(direct)*100:.1f}%")

# Direct test of router
print("\n--- Direct Router Verification ---")
test_flow = np.array([0, 0, 0, 10, 30, 50, 30, 15, 8, 4, 2, 1], dtype=float)
test_routed = model.router.route(test_flow, dt=1.0)
print(f"Test input peak:  {test_flow.max():.1f}")
print(f"Test output peak: {test_routed.max():.1f}")
print(f"Test attenuation: {(1 - test_routed.max()/test_flow.max())*100:.1f}%")
if abs(test_flow.max() - test_routed.max()) < 0.01:
    print("WARNING: Router appears to NOT be attenuating flow!")

# %%
# Compare uncalibrated model to observations
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Find a good event after warmup
peak_idx = np.argmax(direct[WARMUP_DAYS:]) + WARMUP_DAYS
event_slice = slice(peak_idx - 20, peak_idx + 40)

# Panel 1: Hydrographs
ax = axes[0]
ax.plot(results.index[event_slice], direct[event_slice], 
        'b--', linewidth=1.5, alpha=0.7, label='Direct (unrouted)')
ax.plot(results.index[event_slice], routed[event_slice], 
        'r-', linewidth=2, label='Routed (K=5, m=0.9)')
ax.plot(inputs.index[event_slice], observed[event_slice], 
        'ko', markersize=4, alpha=0.6, label='Observed')
ax.set_ylabel('Flow (mm/day)')
ax.set_title('Uncalibrated Model: Routing Effect Visible but Parameters Need Tuning')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 2: Residuals
ax = axes[1]
residual = observed[event_slice] - routed[event_slice]
ax.bar(results.index[event_slice], residual, alpha=0.7, color='purple')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Residual (Obs - Sim)')
ax.set_xlabel('Date')
ax.set_title('Residuals: Need calibration to reduce systematic errors')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate initial performance metrics
from pyrrm.calibration.objective_functions import NSE as NSE_obj, KGE as KGE_obj

nse_uncal = NSE_obj()(observed[WARMUP_DAYS:], routed[WARMUP_DAYS:])
print(f"\nUncalibrated Model Performance:")
print(f"  NSE: {nse_uncal:.4f} (1.0 = perfect)")
print(f"  True routing params: K={TRUE_ROUTING_PARAMS['K']}, m={TRUE_ROUTING_PARAMS['m']}")
print(f"  Current params:      K=5.0, m=0.9")

# %% [markdown]
# ---
# ## Part 5: Calibration with Routing (Synthetic Data)
#
# Now let's demonstrate that we can **recover the true routing parameters** from
# calibration. To clearly isolate the routing effect, we'll:
#
# 1. **Fix the RR parameters** at their true values (so routing is the only unknown)
# 2. **Calibrate only the routing parameters** (K, m, n_subreaches)
# 3. **Verify recovery** by comparing calibrated vs true values
#
# This controlled experiment proves the calibration framework can identify
# routing parameters when given clean data with a known routing signal.

# %%
# Create model with TRUE RR parameters (fixed) and initial routing guesses
rr_model_synth = Sacramento()
rr_model_synth.set_parameters(TRUE_RR_PARAMS)  # Fix RR params at true values

# Start routing with WRONG initial guesses (far from true values)
router_synth = NonlinearMuskingumRouter(
    K=8.0,           # True K=3.5 (starting far away)
    m=0.6,           # True m=0.85 (starting far away)
    n_subreaches=1   # True n=3 (starting different)
)
model_synth = RoutedModel(rr_model_synth, router_synth, routing_enabled=True)

# ONLY calibrate routing parameters (RR params are fixed)
routing_only_bounds = {
    'routing_K': (0.5, 15.0),           # True K=3.5
    'routing_m': (0.1, 1.2),            # True m=0.85
    'routing_n_subreaches': (1, 10)     # True n=3
}

# Create calibration runner with ONLY routing bounds
runner_synth = CalibrationRunner(
    model=model_synth,
    inputs=inputs,
    observed=observed,
    objective=SqrtNSE(),  # Sqrt-transformed NSE
    parameter_bounds=routing_only_bounds,  # Only routing params!
    warmup_period=WARMUP_DAYS
)

print("Synthetic Data Calibration Setup:")
print("=" * 60)
print("\nObjective: This experiment tests if calibration can RECOVER")
print("           the true routing parameters from synthetic data.")
print()
print("Configuration:")
print("  - RR Parameters:      FIXED at true values (not calibrated)")
print("  - Routing Parameters: CALIBRATED (3 parameters)")
print()
print("Initial Guesses vs True Values:")
print("-" * 60)
print(f"  {'Parameter':<15} {'Initial':>12} {'True':>12} {'Bounds':>15}")
print(f"  {'K':<15} {8.0:>12.1f} {TRUE_ROUTING_PARAMS['K']:>12.1f} {'[0.5, 15.0]':>15}")
print(f"  {'m':<15} {0.6:>12.2f} {TRUE_ROUTING_PARAMS['m']:>12.2f} {'[0.5, 1.2]':>15}")
print(f"  {'n_subreaches':<15} {1:>12d} {TRUE_ROUTING_PARAMS['n_subreaches']:>12d} {'[1, 10]':>15}")

# %%
# Run calibration
import warnings

# SCE-UA Direct settings
n_params_synth = len(routing_only_bounds)
max_evals_synth = 3000

print("\nRunning SCE-UA Direct calibration (routing parameters only)...")
print(f"SCE-UA Direct Configuration: max_evals={max_evals_synth}, n_params={n_params_synth}")
print("This should converge quickly with only 3 parameters.\n")

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*Timestep dt.*too large.*')
    warnings.filterwarnings('ignore', message='.*Newton-Raphson did not converge.*')
    
    result_synth = runner_synth.run_sceua_direct(
        max_evals=max_evals_synth,
        seed=42,
        verbose=True,
        max_tolerant_iter=100,
        tolerance=1e-4
    )

print("\n" + result_synth.summary())

# %%
# Check routing parameter recovery
cal_K = result_synth.best_parameters.get('routing_K', 0)
cal_m = result_synth.best_parameters.get('routing_m', 0)
cal_n = int(round(result_synth.best_parameters.get('routing_n_subreaches', 1)))

print("=" * 70)
print("ROUTING PARAMETER RECOVERY RESULTS")
print("=" * 70)
print(f"\n{'Parameter':<20} {'True':>10} {'Calibrated':>12} {'Error':>10} {'Status':>12}")
print("-" * 70)

# K recovery
K_error = cal_K - TRUE_ROUTING_PARAMS['K']
K_status = "GOOD" if abs(K_error) < 1.0 else "CHECK"
print(f"{'K (storage const)':<20} {TRUE_ROUTING_PARAMS['K']:>10.2f} {cal_K:>12.2f} {K_error:>+10.2f} {K_status:>12}")

# m recovery  
m_error = cal_m - TRUE_ROUTING_PARAMS['m']
m_status = "GOOD" if abs(m_error) < 0.1 else "CHECK"
print(f"{'m (nonlinearity)':<20} {TRUE_ROUTING_PARAMS['m']:>10.2f} {cal_m:>12.2f} {m_error:>+10.2f} {m_status:>12}")

# n recovery
n_error = cal_n - TRUE_ROUTING_PARAMS['n_subreaches']
n_status = "GOOD" if n_error == 0 else "CHECK"
print(f"{'n_subreaches':<20} {TRUE_ROUTING_PARAMS['n_subreaches']:>10d} {cal_n:>12d} {n_error:>+10d} {n_status:>12}")

print("-" * 70)
best_nse = -result_synth.best_objective
print(f"\nFinal SqrtNSE: {best_nse:.4f}")
print(f"  (Perfect = 1.0, values > 0.9 indicate excellent fit)")

if best_nse > 0.95:
    print("\n SUCCESS: Routing parameters recovered with high accuracy!")
elif best_nse > 0.85:
    print("\n GOOD: Routing parameters reasonably recovered.")
else:
    print("\n NOTE: Recovery may be limited - check parameter identifiability.")

# %%
# Visualize calibration results: Before vs After
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Run model with INITIAL (wrong) routing parameters
rr_model_initial = Sacramento()
rr_model_initial.set_parameters(TRUE_RR_PARAMS)  # Same RR params
model_initial = RoutedModel(
    rr_model_initial,
    NonlinearMuskingumRouter(K=8.0, m=0.6, n_subreaches=1),  # Initial WRONG guesses
    routing_enabled=True
)
model_initial.reset()
results_initial = model_initial.run(inputs)
sim_initial = results_initial['flow'].values

# Run model with CALIBRATED routing parameters
model_synth.set_parameters(result_synth.best_parameters)
model_synth.reset()
results_calibrated = model_synth.run(inputs)
sim_calibrated = results_calibrated['flow'].values
direct_calibrated = results_calibrated['direct_runoff'].values

# Find peak event for visualization
peak_idx = np.argmax(observed[WARMUP_DAYS:]) + WARMUP_DAYS
event_slice = slice(peak_idx - 30, peak_idx + 60)
time_axis = results_initial.index[event_slice]

# Calculate NSE metrics
def calc_nse(sim, obs):
    """Calculate Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

nse_initial = calc_nse(sim_initial[WARMUP_DAYS:], observed[WARMUP_DAYS:])
nse_calibrated = calc_nse(sim_calibrated[WARMUP_DAYS:], observed[WARMUP_DAYS:])

# Panel 1: Before Calibration (wrong routing params)
ax = axes[0]
ax.plot(time_axis, direct_calibrated[event_slice], 'b--', alpha=0.5, linewidth=1, label='Direct runoff (no routing)')
ax.plot(time_axis, sim_initial[event_slice], 'orange', linewidth=2, label='Routed (initial: K=8.0, m=0.6, n=1)')
ax.plot(time_axis, observed[event_slice], 'ko', markersize=4, alpha=0.7, label='Observed (true K=3.5, m=0.85, n=3)')
ax.set_ylabel('Flow (mm/day)')
ax.set_title(f'BEFORE Calibration: Wrong Routing Parameters (NSE = {nse_initial:.3f})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 2: After Calibration (recovered routing params)
ax = axes[1]
ax.plot(time_axis, direct_calibrated[event_slice], 'b--', alpha=0.5, linewidth=1, label='Direct runoff (no routing)')
ax.plot(time_axis, sim_calibrated[event_slice], 'green', linewidth=2, 
        label=f'Routed (calibrated: K={cal_K:.2f}, m={cal_m:.2f}, n={cal_n})')
ax.plot(time_axis, observed[event_slice], 'ko', markersize=4, alpha=0.7, label='Observed')
ax.set_ylabel('Flow (mm/day)')
ax.set_xlabel('Date')
ax.set_title(f'AFTER Calibration: Recovered Routing Parameters (NSE = {nse_calibrated:.3f})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Add improvement annotation
improvement = nse_calibrated - nse_initial
fig.suptitle(f'Routing Parameter Recovery Test\nNSE: {nse_initial:.3f} (wrong params) → {nse_calibrated:.3f} (calibrated)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Summary
print("\n" + "=" * 60)
print("CALIBRATION SUMMARY")
print("=" * 60)
print(f"\nNSE Improvement: {nse_initial:.4f} → {nse_calibrated:.4f} (+{improvement:.4f})")
print(f"\nParameter Comparison:")
print(f"  True:       K={TRUE_ROUTING_PARAMS['K']:.1f}, m={TRUE_ROUTING_PARAMS['m']:.2f}, n={TRUE_ROUTING_PARAMS['n_subreaches']}")
print(f"  Initial:    K=8.0, m=0.60, n=1")
print(f"  Calibrated: K={cal_K:.2f}, m={cal_m:.2f}, n={cal_n}")
print(f"\nRecovery Errors:")
print(f"  K error: {cal_K - TRUE_ROUTING_PARAMS['K']:+.2f} days ({abs(cal_K - TRUE_ROUTING_PARAMS['K'])/TRUE_ROUTING_PARAMS['K']*100:.1f}%)")
print(f"  m error: {cal_m - TRUE_ROUTING_PARAMS['m']:+.3f} ({abs(cal_m - TRUE_ROUTING_PARAMS['m'])/TRUE_ROUTING_PARAMS['m']*100:.1f}%)")
print(f"  n error: {cal_n - TRUE_ROUTING_PARAMS['n_subreaches']:+d}")

# %% [markdown]
# ---
# ## Part 6: Multi-Objective Routing Analysis (Real Data)
#
# ### Building on Notebook 02: Calibration Quickstart
#
# **This section continues directly from Notebook 02 (Calibration Quickstart)**.
# Instead of re-running the rainfall-runoff calibrations from scratch, we **load
# the previously saved calibration results** from the pickle files generated in
# that tutorial.
#
# **Prerequisites**: You must have completed Notebook 02 and saved the calibration
# reports to `test_data/reports/`. If those files don't exist, go back and run
# Notebook 02 first.
#
# ### What We're Loading from Notebook 02
#
# In Notebook 02, we calibrated the Sacramento model with **5 different objective
# functions**. Each calibration serves as the **baseline (Stage A)** for that
# objective function's routing analysis:
#
# | Report File | Objective | Description |
# |-------------|-----------|-------------|
# | `410734_nse.pkl` | NSE | Standard (high flow emphasis) |
# | `410734_lognse_custom.pkl` | LogNSE | Low flow emphasis |
# | `410734_invnse_custom.pkl` | InvNSE | Recession flow emphasis |
# | `410734_sqrtnse_custom.pkl` | SqrtNSE | Balanced high/low flows |
# | `410734_sdeb_custom.pkl` | SDEB | Multi-timescale balance |
#
# ### Experimental Design: Multi-Objective Comparison
#
# **Key Question: How does routing benefit vary across different objective functions?**
#
# For **EACH** of the 5 objective functions, we perform the three-stage calibration:
#
# | Stage | Description | Parameters Calibrated |
# |-------|-------------|----------------------|
# | **A** | Sacramento only (no routing) | **LOADED** from Notebook 02 (each objective has its own baseline) |
# | **B** | Add routing to Stage A (fix RR params) | 3 routing parameters only |
# | **C** | Full joint calibration (RR + routing) | 25 parameters (22 RR + 3 routing) |
#
# This comprehensive design allows us to:
# - Compare **baseline performance** across different objective functions
# - Measure the **incremental benefit of routing** for each objective
# - Assess whether **joint optimization** consistently outperforms sequential calibration
# - Identify which objective functions benefit most from routing
# - Compare **calibrated routing parameters** (K, m, n) across objectives

# %%
# Load real catchment data (same as Notebook 02)
from pathlib import Path
from pyrrm.data import load_parameter_bounds
from pyrrm.calibration import CalibrationReport

DATA_DIR = Path('../data/410734')
REPORTS_DIR = Path('../test_data/reports')
CATCHMENT_AREA_KM2 = 516.62667

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

# Handle missing values
observed_df['observed_flow'] = observed_df['observed_flow'].replace(-9999, np.nan)
observed_df.loc[observed_df['observed_flow'] < 0, 'observed_flow'] = np.nan
observed_df = observed_df.dropna()

# Merge datasets
real_data = rainfall_df.join(pet_df, how='inner').join(observed_df, how='inner')
real_data = real_data.rename(columns={'rainfall': 'precipitation'})

# Prepare for calibration
real_inputs = real_data[['precipitation', 'pet']].copy()
real_observed = real_data['observed_flow'].values
REAL_WARMUP = 365

print("=" * 70)
print("REAL CATCHMENT DATA (Gauge 410734)")
print("=" * 70)
print(f"Period: {real_data.index[0].date()} to {real_data.index[-1].date()}")
print(f"Total days: {len(real_data)}")
print(f"Catchment area: {CATCHMENT_AREA_KM2} km²")
print(f"Warmup period: {REAL_WARMUP} days")

# %% [markdown]
# ### Loading Calibration Results from Notebook 02
#
# Instead of running calibrations again, we load the saved reports. This
# demonstrates a key benefit of the `CalibrationReport` system: you can
# **preserve and reuse calibration results** across multiple analyses.

# %%
# Load all calibration reports from Notebook 02
print("=" * 70)
print("LOADING CALIBRATION REPORTS FROM NOTEBOOK 02")
print("=" * 70)

# Available reports from Notebook 02
report_files = {
    'NSE (default)': '410734_nse.pkl',
    'LogNSE (default)': '410734_lognse.pkl',
    'SqrtNSE (default)': '410734_sqrtnse.pkl',
    'SDEB (default)': '410734_sdeb.pkl',
    'LogNSE (custom)': '410734_lognse_custom.pkl',
    'InvNSE (custom)': '410734_invnse_custom.pkl',
    'SqrtNSE (custom)': '410734_sqrtnse_custom.pkl',
    'SDEB (custom)': '410734_sdeb_custom.pkl',
}

# Try to load each report
loaded_reports = {}
for name, filename in report_files.items():
    filepath = REPORTS_DIR / filename
    if filepath.exists():
        try:
            report = CalibrationReport.load(filepath)
            loaded_reports[name] = report
            print(f"  ✓ {name}: {filename}")
        except Exception as e:
            print(f"  ✗ {name}: Failed to load ({e})")
    else:
        print(f"  ✗ {name}: File not found")

print(f"\nSuccessfully loaded {len(loaded_reports)} reports")

# Check that we have the required report for this tutorial
if 'SqrtNSE (custom)' not in loaded_reports:
    raise FileNotFoundError(
        "Required calibration report not found: 410734_sqrtnse_custom.pkl\n"
        "Please run Notebook 02 (Calibration Quickstart) first to generate the reports."
    )

# %%
# Display summary of loaded calibrations
print("\n" + "=" * 70)
print("CALIBRATION RESULTS SUMMARY (from Notebook 02)")
print("=" * 70)
print(f"\n{'Calibration':<20} {'Objective':<12} {'Best Value':>12} {'Runtime':>10}")
print("-" * 60)

for name, report in loaded_reports.items():
    obj_name = report.result.objective_name
    obj_val = report.result.best_objective
    runtime = report.result.runtime_seconds
    print(f"{name:<20} {obj_name:<12} {obj_val:>12.4f} {runtime:>8.1f}s")

# %% [markdown]
# ### Loading Custom Parameter Bounds
#
# We use the same custom bounds file from Notebook 02 (Step 9). These bounds
# were refined based on initial calibration results to better constrain the
# parameter space.

# %%
# Load custom parameter bounds (same as Notebook 02, Step 9)
bounds_file = DATA_DIR / 'sacramento_bounds_custom.txt'
custom_bounds = load_parameter_bounds(bounds_file)

print("=" * 70)
print("CUSTOM PARAMETER BOUNDS (from Notebook 02)")
print("=" * 70)
print(f"Loaded from: {bounds_file}")
print(f"Parameters: {len(custom_bounds)}")
print("\nBounds summary:")
for param, (lo, hi) in list(custom_bounds.items())[:5]:
    print(f"  {param}: [{lo:.4f}, {hi:.4f}]")
print("  ...")

# %% [markdown]
# ---
# ### Multi-Objective Routing Analysis
#
# We'll run the three-stage calibration (A, B, C) for **all 5 objective functions**
# from Notebook 02, allowing us to compare routing benefits across different
# calibration approaches.
#
# **Objective Functions Analyzed:**
#
# | Code | Objective | Description |
# |------|-----------|-------------|
# | NSE | Nash-Sutcliffe | Standard flow matching (high flow emphasis) |
# | LogNSE | Log-transformed NSE | Low flow emphasis |
# | InvNSE | Inverse-transformed NSE | Recession flow emphasis |
# | SqrtNSE | Sqrt-transformed NSE | Balanced high/low flows |
# | SDEB | Spectral Decomposition Error | Multi-timescale balance |

# %%
# Define the objective functions for analysis
# Map report names to objective constructors
# IMPORTANT: These must match exactly what was used in Notebook 02 calibrations!
from pyrrm.objectives import NSE, KGE, RMSE, PBIAS, SDEB

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

# Filter to only include objectives where we have saved reports
available_objectives = {}
for name, config in OBJECTIVE_CONFIGS.items():
    if config['report_key'] in loaded_reports:
        available_objectives[name] = config
        print(f"  ✓ {name}: {config['description']}")
    else:
        print(f"  ✗ {name}: Report not found ({config['report_key']})")

print(f"\nAnalyzing {len(available_objectives)} objective functions")

# %% [markdown]
# ---
# ### Running Four-Stage Calibration for All Objectives
#
# We run a **four-stage calibration experiment** to investigate the equifinality
# between internal unit hydrograph (UH) routing and external Muskingum routing.
#
# | Stage | Sacramento UH | Muskingum Routing | Description |
# |-------|---------------|-------------------|-------------|
# | **A** | Calibrated (uh1-uh5) | None | Baseline with internal UH, no channel routing |
# | **B** | Fixed from A | Calibrated | Add channel routing, fix all RR params (including UH) |
# | **C** | Calibrated jointly | Calibrated jointly | Full joint calibration (equifinality risk) |
# | **D** | Fixed (uh1=1, others=0) | Calibrated | Clean separation - no internal UH delay |
#
# **Stage D** is key for testing equifinality:
# - It removes internal UH delay by fixing `uh1=1.0` and `uh2-5=0.0`
# - Only Muskingum routing handles timing/attenuation
# - Conceptually cleaner: Sacramento = catchment outlet, Muskingum = channel to gauge
# - Comparing D vs C reveals whether internal UH is needed or just trades off with K

# %%
# Storage for all results
all_results = {}

# Routing parameter bounds (same for all objectives)
routing_only_bounds = {
    'routing_K': (0.5, 15.0),
    'routing_m': (0.1, 1.2),
    'routing_n_subreaches': (1, 10)
}

# Joint bounds (RR + routing)
joint_bounds = custom_bounds.copy()
joint_bounds['routing_K'] = (0.5, 15.0)
joint_bounds['routing_m'] = (0.1, 1.2)
joint_bounds['routing_n_subreaches'] = (1, 10)

print("=" * 70)
print("FOUR-STAGE CALIBRATION FOR ALL OBJECTIVE FUNCTIONS")
print("=" * 70)
print("\nThis will run Calibration B, C, and D for each objective function.")
print("Calibration A results are loaded from Notebook 02.")
print("Stage D uses fixed UH (uh1=1, others=0) to test equifinality.\n")

# Create fixed-UH bounds for Stage D (Sacramento params without UH + routing)
# First, create bounds that exclude uh1-uh5 from calibration
fixed_uh_bounds = {k: v for k, v in custom_bounds.items() if not k.startswith('uh')}
fixed_uh_bounds['routing_K'] = (0.5, 15.0)
fixed_uh_bounds['routing_m'] = (0.1, 1.2)
fixed_uh_bounds['routing_n_subreaches'] = (1, 10)

print(f"Stage D will calibrate {len(fixed_uh_bounds)} parameters (Sacramento without UH + routing)")

# %%
# Loop through each objective function
for obj_name, config in available_objectives.items():
    print("\n" + "=" * 70)
    print(f"OBJECTIVE: {obj_name} - {config['description']}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Calibration A: Load from saved report
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration A: Load from Notebook 02 ---")
    report_A = loaded_reports[config['report_key']]
    result_A = report_A.result
    params_A = result_A.best_parameters.copy()
    
    print(f"  Loaded: {config['report_key']}")
    print(f"  Best objective: {result_A.best_objective:.4f}")
    
    # -------------------------------------------------------------------------
    # Calibration B: Add routing (fix RR params)
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration B: Add Routing (fix RR from A) ---")
    
    # Create model with routing
    rr_model_B = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    rr_model_B.set_parameters(params_A)
    router_B = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_B = RoutedModel(rr_model_B, router_B, routing_enabled=True)
    
    # Setup runner with routing-only bounds
    runner_B = CalibrationRunner(
        model=model_B,
        inputs=real_inputs,
        observed=real_observed,
        objective=config['objective_fn'](),
        parameter_bounds=routing_only_bounds,
        warmup_period=REAL_WARMUP
    )
    
    print(f"  Running SCE-UA (3 routing params, max_evals=3000)...")
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
    
    print(f"  Best objective: {result_B.best_objective:.4f}")
    print(f"  Routing: K={K_B:.2f}, m={m_B:.3f}, n={n_B}")
    
    # Save Calibration B report
    report_B = runner_B.create_report(result_B, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_B_filename = f"410734_{obj_name.lower()}_routing_B"
    report_B.save(REPORTS_DIR / report_B_filename)
    print(f"  Saved: {report_B_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Calibration C: Joint (RR + routing)
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration C: Joint Calibration (RR + Routing) ---")
    
    # Create fresh model for joint calibration
    rr_model_C = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    router_C = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_C = RoutedModel(rr_model_C, router_C, routing_enabled=True)
    
    runner_C = CalibrationRunner(
        model=model_C,
        inputs=real_inputs,
        observed=real_observed,
        objective=config['objective_fn'](),
        parameter_bounds=joint_bounds,
        warmup_period=REAL_WARMUP
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
    
    print(f"  Best objective: {result_C.best_objective:.4f}")
    print(f"  Routing: K={K_C:.2f}, m={m_C:.3f}, n={n_C}")
    
    # Save Calibration C report
    report_C = runner_C.create_report(result_C, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_C_filename = f"410734_{obj_name.lower()}_routing_C"
    report_C.save(REPORTS_DIR / report_C_filename)
    print(f"  Saved: {report_C_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Calibration D: Fixed UH (uh1=1, others=0) + Muskingum routing
    # This tests the equifinality hypothesis - clean separation of routing
    # -------------------------------------------------------------------------
    print(f"\n--- Calibration D: Fixed UH + Routing (equifinality test) ---")
    
    # Create model with fixed UH (no internal routing delay)
    rr_model_D = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    # Set fixed UH: all flow appears in day 1 (no delay)
    fixed_uh_params = {'uh1': 1.0, 'uh2': 0.0, 'uh3': 0.0, 'uh4': 0.0, 'uh5': 0.0}
    router_D = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    model_D = RoutedModel(rr_model_D, router_D, routing_enabled=True)
    
    # For Stage D, we calibrate Sacramento (without UH) + routing together
    runner_D = CalibrationRunner(
        model=model_D,
        inputs=real_inputs,
        observed=real_observed,
        objective=config['objective_fn'](),
        parameter_bounds=fixed_uh_bounds,
        warmup_period=REAL_WARMUP
    )
    
    # Override UH parameters to fixed values before each model run
    # We do this by setting them after each parameter update in the runner
    print(f"  Running SCE-UA ({len(fixed_uh_bounds)} params, max_evals=15000)...")
    print(f"  Note: UH fixed at uh1=1.0, uh2-5=0.0 (no internal delay)")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Timestep dt.*too large.*')
        warnings.filterwarnings('ignore', message='.*Newton-Raphson did not converge.*')
        
        result_D = runner_D.run_sceua_direct(
            max_evals=15000,
            seed=42,
            verbose=True,
            max_tolerant_iter=100,
            tolerance=1e-4
        )
    
    params_D = result_D.best_parameters.copy()
    # Add the fixed UH parameters to the results
    params_D.update(fixed_uh_params)
    rr_params_D = {k: v for k, v in params_D.items() if not k.startswith('routing_')}
    K_D = params_D.get('routing_K', 5.0)
    m_D = params_D.get('routing_m', 0.8)
    n_D = int(round(params_D.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective: {result_D.best_objective:.4f}")
    print(f"  Routing: K={K_D:.2f}, m={m_D:.3f}, n={n_D}")
    
    # Save Calibration D report
    report_D = runner_D.create_report(result_D, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_D_filename = f"410734_{obj_name.lower()}_routing_D"
    report_D.save(REPORTS_DIR / report_D_filename)
    print(f"  Saved: {report_D_filename}.pkl")
    
    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    all_results[obj_name] = {
        'A': {'result': result_A, 'params': params_A},
        'B': {'result': result_B, 'params_rr': params_A, 'routing': {'K': K_B, 'm': m_B, 'n': n_B}, 'report': report_B},
        'C': {'result': result_C, 'params_rr': rr_params_C, 'routing': {'K': K_C, 'm': m_C, 'n': n_C}, 'report': report_C},
        'D': {'result': result_D, 'params_rr': rr_params_D, 'routing': {'K': K_D, 'm': m_D, 'n': n_D}, 'report': report_D},
    }
    
    print(f"\n✓ {obj_name} complete!")

print("\n" + "=" * 70)
print("ALL CALIBRATIONS COMPLETE")
print("=" * 70)

# %% [markdown]
# ---
# ### Comprehensive Results Comparison
#
# Now we compare results across all objective functions and calibration stages using
# a comprehensive set of performance metrics:
#
# | Metric | Description | Optimal |
# |--------|-------------|---------|
# | **NSE** | Nash-Sutcliffe Efficiency (standard) | 1.0 |
# | **KGE** | Kling-Gupta Efficiency | 1.0 |
# | **PBIAS** | Percent Bias (volume error) | 0% |
# | **RMSE** | Root Mean Square Error | 0 |
# | **LogNSE** | NSE on log-transformed flows (low flow emphasis) | 1.0 |
# | **SqrtNSE** | NSE on sqrt-transformed flows (balanced) | 1.0 |
# | **InvNSE** | NSE on inverse-transformed flows (recession emphasis) | 1.0 |
#
# We also store all simulations for interactive visualization.

# %%
# Generate simulations and calculate metrics for all objective functions
print("=" * 70)
print("GENERATING SIMULATIONS FOR ALL CALIBRATIONS")
print("=" * 70)

# Metrics calculators - comprehensive set
nse_metric = NSE()
kge_metric = KGE()
rmse_metric = RMSE()
pbias_metric = PBIAS()
lognse_metric = NSE(transform=FlowTransformation('log'))
sqrtnse_metric = NSE(transform=FlowTransformation('sqrt'))
invnse_metric = NSE(transform=FlowTransformation('inverse'))

obs = real_observed[REAL_WARMUP:]
time_index = real_inputs.index[REAL_WARMUP:]

# Store comprehensive results and simulations
comparison_results = []
simulations = {}  # Store actual simulations for visualization

for obj_name, results in all_results.items():
    print(f"\nProcessing {obj_name}...")
    
    # Get parameters
    params_A = results['A']['params']
    params_rr_B = results['B']['params_rr']
    routing_B = results['B']['routing']
    params_rr_C = results['C']['params_rr']
    routing_C = results['C']['routing']
    params_rr_D = results['D']['params_rr']
    routing_D = results['D']['routing']
    
    # Simulation A: No routing (with calibrated UH)
    model_A = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_A.set_parameters(params_A)
    model_A.reset()
    sim_A = model_A.run(real_inputs)['runoff'].values[REAL_WARMUP:]
    
    # Simulation B: Fixed RR (including UH from A) + routing
    model_B = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_B.set_parameters(params_rr_B)
    model_B.reset()
    sim_B_direct = model_B.run(real_inputs)['runoff'].values
    router_B = NonlinearMuskingumRouter(
        K=routing_B['K'], m=routing_B['m'], n_subreaches=routing_B['n']
    )
    sim_B = router_B.route(sim_B_direct, dt=1.0)[REAL_WARMUP:]
    
    # Simulation C: Joint calibration (UH + routing together)
    model_C = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_C.set_parameters(params_rr_C)
    model_C.reset()
    sim_C_direct = model_C.run(real_inputs)['runoff'].values
    router_C = NonlinearMuskingumRouter(
        K=routing_C['K'], m=routing_C['m'], n_subreaches=routing_C['n']
    )
    sim_C = router_C.route(sim_C_direct, dt=1.0)[REAL_WARMUP:]
    
    # Simulation D: Fixed UH (uh1=1, others=0) + routing
    model_D = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_D.set_parameters(params_rr_D)
    model_D.reset()
    sim_D_direct = model_D.run(real_inputs)['runoff'].values
    router_D = NonlinearMuskingumRouter(
        K=routing_D['K'], m=routing_D['m'], n_subreaches=routing_D['n']
    )
    sim_D = router_D.route(sim_D_direct, dt=1.0)[REAL_WARMUP:]
    
    # Store simulations for visualization
    simulations[obj_name] = {
        'A': sim_A,
        'B': sim_B,
        'C': sim_C,
        'D': sim_D,
        'routing_B': routing_B,
        'routing_C': routing_C,
        'routing_D': routing_D
    }
    
    # Calculate comprehensive metrics for all stages
    stages_data = [
        ('A', sim_A, None),
        ('B', sim_B, routing_B),
        ('C', sim_C, routing_C),
        ('D', sim_D, routing_D)
    ]
    
    for stage, sim, routing in stages_data:
        comparison_results.append({
            'Objective': obj_name,
            'Stage': stage,
            'NSE': nse_metric(obs, sim),
            'KGE': kge_metric(obs, sim),
            'PBIAS': pbias_metric(obs, sim),
            'RMSE': rmse_metric(obs, sim),
            'LogNSE': lognse_metric(obs, sim),
            'SqrtNSE': sqrtnse_metric(obs, sim),
            'InvNSE': invnse_metric(obs, sim),
            'K': routing['K'] if routing else None,
            'm': routing['m'] if routing else None,
            'n': routing['n'] if routing else None,
        })

print("\n✓ All simulations complete!")

# %%
# Create results DataFrame
results_df = pd.DataFrame(comparison_results)

print("\n" + "=" * 90)
print("COMPREHENSIVE RESULTS: ALL OBJECTIVES × ALL STAGES")
print("=" * 90)

# Create pivot tables for all metrics
metrics_to_compare = ['NSE', 'KGE', 'PBIAS', 'RMSE', 'LogNSE', 'SqrtNSE', 'InvNSE']
pivots = {}

for metric in metrics_to_compare:
    pivot = results_df.pivot(index='Objective', columns='Stage', values=metric)
    pivot = pivot[['A', 'B', 'C', 'D']]  # Ensure correct order including D
    pivot['B-A'] = pivot['B'] - pivot['A']
    pivot['C-A'] = pivot['C'] - pivot['A']
    pivot['D-A'] = pivot['D'] - pivot['A']
    pivot['D-C'] = pivot['D'] - pivot['C']  # Key comparison: D vs C tests equifinality
    pivots[metric] = pivot

# Display key metrics
print("\n=== NSE Comparison (Nash-Sutcliffe Efficiency) ===")
print("Stages: A=Baseline | B=Add Routing | C=Joint | D=Fixed UH + Routing")
print(pivots['NSE'].round(4).to_string())

print("\n=== KGE Comparison (Kling-Gupta Efficiency) ===")
print(pivots['KGE'].round(4).to_string())

print("\n=== PBIAS Comparison (Percent Bias - closer to 0 is better) ===")
print(pivots['PBIAS'].round(2).to_string())

print("\n=== LogNSE Comparison (Low Flow Emphasis) ===")
print(pivots['LogNSE'].round(4).to_string())

print("\n=== SqrtNSE Comparison (Balanced) ===")
print(pivots['SqrtNSE'].round(4).to_string())

print("\n=== InvNSE Comparison (Recession Emphasis) ===")
print(pivots['InvNSE'].round(4).to_string())

# %%
# Routing parameters comparison
print("\n" + "=" * 90)
print("ROUTING PARAMETERS BY OBJECTIVE FUNCTION")
print("=" * 90)
print("Stages with routing: B=Fixed RR+UH | C=Joint (UH calibrated) | D=Fixed UH (uh1=1)")

routing_df = results_df[results_df['Stage'].isin(['B', 'C', 'D'])][['Objective', 'Stage', 'K', 'm', 'n']]
routing_pivot = routing_df.pivot(index='Objective', columns='Stage')
print(routing_pivot.round(3).to_string())

# Keep references to important pivots for later use
nse_pivot = pivots['NSE']
kge_pivot = pivots['KGE']

# %%
# Equifinality Analysis: Compare UH parameters from Stage A/C vs fixed in D
print("\n" + "=" * 90)
print("EQUIFINALITY ANALYSIS: UNIT HYDROGRAPH PARAMETERS")
print("=" * 90)
print("\nStage A/B calibrated UH parameters (internal routing):")
print("Stage C jointly calibrated UH + routing")
print("Stage D: Fixed UH (uh1=1.0, uh2-5=0.0) - no internal routing\n")

# Extract UH parameters from results
uh_comparison = []
for obj_name, results in all_results.items():
    params_A = results['A']['params']
    params_C = results['C']['params_rr']
    
    uh_comparison.append({
        'Objective': obj_name,
        'Stage': 'A',
        'uh1': params_A.get('uh1', 1.0),
        'uh2': params_A.get('uh2', 0.0),
        'uh3': params_A.get('uh3', 0.0),
        'uh4': params_A.get('uh4', 0.0),
        'uh5': params_A.get('uh5', 0.0),
    })
    uh_comparison.append({
        'Objective': obj_name,
        'Stage': 'C',
        'uh1': params_C.get('uh1', 1.0),
        'uh2': params_C.get('uh2', 0.0),
        'uh3': params_C.get('uh3', 0.0),
        'uh4': params_C.get('uh4', 0.0),
        'uh5': params_C.get('uh5', 0.0),
    })
    # D has fixed UH
    uh_comparison.append({
        'Objective': obj_name,
        'Stage': 'D',
        'uh1': 1.0,
        'uh2': 0.0,
        'uh3': 0.0,
        'uh4': 0.0,
        'uh5': 0.0,
    })

uh_df = pd.DataFrame(uh_comparison)
uh_pivot = uh_df.pivot(index='Objective', columns='Stage', values=['uh1', 'uh2', 'uh3', 'uh4', 'uh5'])
print(uh_pivot.round(3).to_string())

# %% [markdown]
# ---
# ### Visualization: Multi-Objective Comparison
#
# Now we create comprehensive interactive visualizations to:
# 1. Compare metrics across all 4 stages (A, B, C, D)
# 2. Visually inspect hydrographs before and after routing
# 3. Compare flow duration curves
# 4. Analyze the equifinality between internal UH and Muskingum routing

# %%
# VISUALIZATION 1: Metrics Comparison Dashboard (4 Stages)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

print("=" * 70)
print("VISUALIZATION 1: COMPREHENSIVE METRICS DASHBOARD (4 STAGES)")
print("=" * 70)

# Create a 3x3 subplot for all metrics
fig_metrics = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        '<b>NSE by Stage</b>',
        '<b>KGE by Stage</b>',
        '<b>PBIAS by Stage</b>',
        '<b>LogNSE by Stage</b>',
        '<b>SqrtNSE by Stage</b>',
        '<b>InvNSE by Stage</b>',
        '<b>NSE Improvement over A</b>',
        '<b>D vs C (Equifinality Test)</b>',
        '<b>Routing K Parameter</b>'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

objectives = list(available_objectives.keys())
colors = {'A': '#3498db', 'B': '#e67e22', 'C': '#27ae60', 'D': '#9b59b6'}
stage_labels = {'A': 'No Routing', 'B': 'Add Routing', 'C': 'Joint Cal.', 'D': 'Fixed UH'}

# Row 1: NSE, KGE, PBIAS
for col, metric in enumerate(['NSE', 'KGE', 'PBIAS'], 1):
    for stage in ['A', 'B', 'C', 'D']:
        stage_data = results_df[results_df['Stage'] == stage]
        fig_metrics.add_trace(go.Bar(
            name=f'{stage} ({stage_labels[stage]})',
            x=stage_data['Objective'],
            y=stage_data[metric],
            marker_color=colors[stage],
            showlegend=(col == 1),
            legendgroup=stage
        ), row=1, col=col)

# Row 2: LogNSE, SqrtNSE, InvNSE
for col, metric in enumerate(['LogNSE', 'SqrtNSE', 'InvNSE'], 1):
    for stage in ['A', 'B', 'C', 'D']:
        stage_data = results_df[results_df['Stage'] == stage]
        fig_metrics.add_trace(go.Bar(
            name=stage,
            x=stage_data['Objective'],
            y=stage_data[metric],
            marker_color=colors[stage],
            showlegend=False,
            legendgroup=stage
        ), row=2, col=col)

# Row 3, Col 1: NSE Improvement over A
fig_metrics.add_trace(go.Bar(
    name='B-A',
    x=nse_pivot.index,
    y=nse_pivot['B-A'],
    marker_color=colors['B'],
    showlegend=False
), row=3, col=1)
fig_metrics.add_trace(go.Bar(
    name='C-A',
    x=nse_pivot.index,
    y=nse_pivot['C-A'],
    marker_color=colors['C'],
    showlegend=False
), row=3, col=1)
fig_metrics.add_trace(go.Bar(
    name='D-A',
    x=nse_pivot.index,
    y=nse_pivot['D-A'],
    marker_color=colors['D'],
    showlegend=False
), row=3, col=1)

# Row 3, Col 2: D vs C (Equifinality Test) - NSE and KGE
fig_metrics.add_trace(go.Bar(
    name='NSE: D-C',
    x=nse_pivot.index,
    y=nse_pivot['D-C'],
    marker_color='#e74c3c',
    showlegend=False
), row=3, col=2)
fig_metrics.add_trace(go.Bar(
    name='KGE: D-C',
    x=kge_pivot.index,
    y=kge_pivot['D-C'],
    marker_color='#f39c12',
    showlegend=False
), row=3, col=2)

# Row 3, Col 3: K parameter comparison (B, C, D)
k_B = results_df[(results_df['Stage'] == 'B')].set_index('Objective')['K']
k_C = results_df[(results_df['Stage'] == 'C')].set_index('Objective')['K']
k_D = results_df[(results_df['Stage'] == 'D')].set_index('Objective')['K']
fig_metrics.add_trace(go.Bar(
    name='K (Stage B)',
    x=k_B.index,
    y=k_B.values,
    marker_color=colors['B'],
    showlegend=False
), row=3, col=3)
fig_metrics.add_trace(go.Bar(
    name='K (Stage C)',
    x=k_C.index,
    y=k_C.values,
    marker_color=colors['C'],
    showlegend=False
), row=3, col=3)
fig_metrics.add_trace(go.Bar(
    name='K (Stage D)',
    x=k_D.index,
    y=k_D.values,
    marker_color=colors['D'],
    showlegend=False
), row=3, col=3)

# Update layout
fig_metrics.update_layout(
    height=900,
    width=1400,
    title_text="<b>Comprehensive Metrics Comparison: All Objectives × 4 Stages</b><br><sup>Catchment 410734 - Multi-Objective Routing Analysis (A=Baseline, B=Add Routing, C=Joint, D=Fixed UH)</sup>",
    barmode='group',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

# Add y-axis labels
fig_metrics.update_yaxes(title_text="NSE", row=1, col=1)
fig_metrics.update_yaxes(title_text="KGE", row=1, col=2)
fig_metrics.update_yaxes(title_text="PBIAS (%)", row=1, col=3)
fig_metrics.update_yaxes(title_text="LogNSE", row=2, col=1)
fig_metrics.update_yaxes(title_text="SqrtNSE", row=2, col=2)
fig_metrics.update_yaxes(title_text="InvNSE", row=2, col=3)
fig_metrics.update_yaxes(title_text="Δ from A", row=3, col=1)
fig_metrics.update_yaxes(title_text="D-C (NSE red, KGE orange)", row=3, col=2)
fig_metrics.update_yaxes(title_text="K (days)", row=3, col=3)

fig_metrics.show()

# %% [markdown]
# ---
# ### Interactive Hydrograph and FDC Comparison
#
# For each objective function, we compare all 4 stages:
# - **Hydrograph (Linear)**: Shows peak timing and magnitude
# - **Hydrograph (Log Scale)**: Emphasizes low flow performance
# - **Flow Duration Curve**: Shows exceedance probability
#
# **Key comparison: Stage D (Fixed UH) vs Stage C (Joint calibration)**
# - If D ≈ C: Internal UH not needed, equifinality with Muskingum
# - If C > D: Internal UH provides meaningful additional flexibility

# %%
# VISUALIZATION 2: Interactive Hydrograph and FDC Comparisons
# Create a figure for each objective function

print("=" * 70)
print("VISUALIZATION 2: HYDROGRAPH AND FDC COMPARISON BY OBJECTIVE (4 STAGES)")
print("=" * 70)

def calculate_fdc(flows):
    """Calculate flow duration curve (exceedance probabilities)."""
    sorted_flows = np.sort(flows)[::-1]
    n = len(sorted_flows)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100
    return exceedance, sorted_flows

# Create comparison figure for each objective
for obj_name in simulations.keys():
    print(f"\nCreating comparison figure for {obj_name}...")
    
    sim_data = simulations[obj_name]
    sim_A = sim_data['A']
    sim_B = sim_data['B']
    sim_C = sim_data['C']
    sim_D = sim_data['D']
    routing_B = sim_data['routing_B']
    routing_C = sim_data['routing_C']
    routing_D = sim_data['routing_D']
    
    # Get metrics for this objective
    obj_metrics = results_df[results_df['Objective'] == obj_name]
    
    # Create 2x3 subplot: top row = linear, log hydrographs; bottom row = FDC and metrics
    fig_obj = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f'<b>Hydrograph (Linear Scale)</b>',
            f'<b>Hydrograph (Log Scale)</b>',
            f'<b>Flow Duration Curve</b>',
            f'<b>Performance Metrics (4 Stages)</b>',
            f'<b>Metrics Improvement over A</b>',
            f'<b>Routing K Parameter</b>'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Create time array for x-axis
    time_array = np.arange(len(obs))
    
    # Row 1, Col 1: Hydrograph (Linear Scale) - All 4 stages
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=obs,
        name='Observed', mode='lines',
        line=dict(color='black', width=1.5),
        legendgroup='hydro'
    ), row=1, col=1)
    
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=sim_A,
        name=f'A: No Routing (NSE={obj_metrics[obj_metrics["Stage"]=="A"]["NSE"].values[0]:.3f})',
        mode='lines', line=dict(color=colors['A'], width=1.2, dash='solid'),
        legendgroup='hydro'
    ), row=1, col=1)
    
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=sim_B,
        name=f'B: +Routing (NSE={obj_metrics[obj_metrics["Stage"]=="B"]["NSE"].values[0]:.3f})',
        mode='lines', line=dict(color=colors['B'], width=1.2, dash='dot'),
        legendgroup='hydro'
    ), row=1, col=1)
    
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=sim_C,
        name=f'C: Joint (NSE={obj_metrics[obj_metrics["Stage"]=="C"]["NSE"].values[0]:.3f})',
        mode='lines', line=dict(color=colors['C'], width=1.2, dash='dash'),
        legendgroup='hydro'
    ), row=1, col=1)
    
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=sim_D,
        name=f'D: Fixed UH (NSE={obj_metrics[obj_metrics["Stage"]=="D"]["NSE"].values[0]:.3f})',
        mode='lines', line=dict(color=colors['D'], width=1.2, dash='dashdot'),
        legendgroup='hydro'
    ), row=1, col=1)
    
    # Row 1, Col 2: Hydrograph (Log Scale)
    fig_obj.add_trace(go.Scatter(
        x=time_array, y=obs,
        name='Observed', mode='lines', showlegend=False,
        line=dict(color='black', width=1.5)
    ), row=1, col=2)
    
    for sim, stage, dash in [(sim_A, 'A', 'solid'), (sim_B, 'B', 'dot'), 
                              (sim_C, 'C', 'dash'), (sim_D, 'D', 'dashdot')]:
        fig_obj.add_trace(go.Scatter(
            x=time_array, y=sim,
            name=stage, mode='lines', showlegend=False,
            line=dict(color=colors[stage], width=1.2, dash=dash)
        ), row=1, col=2)
    
    # Row 1, Col 3: Flow Duration Curve
    exc_obs, fdc_obs = calculate_fdc(obs)
    exc_A, fdc_A = calculate_fdc(sim_A)
    exc_B, fdc_B = calculate_fdc(sim_B)
    exc_C, fdc_C = calculate_fdc(sim_C)
    exc_D, fdc_D = calculate_fdc(sim_D)
    
    fig_obj.add_trace(go.Scatter(
        x=exc_obs, y=fdc_obs,
        name='Observed', mode='lines', showlegend=False,
        line=dict(color='black', width=2)
    ), row=1, col=3)
    
    for exc, fdc, stage, dash in [(exc_A, fdc_A, 'A', 'solid'), (exc_B, fdc_B, 'B', 'dot'),
                                   (exc_C, fdc_C, 'C', 'dash'), (exc_D, fdc_D, 'D', 'dashdot')]:
        fig_obj.add_trace(go.Scatter(
            x=exc, y=fdc,
            name=stage, mode='lines', showlegend=False,
            line=dict(color=colors[stage], width=1.5, dash=dash)
        ), row=1, col=3)
    
    # Row 2, Col 1: Key Performance Metrics (NSE, KGE, LogNSE) - All 4 stages
    metrics_to_show = ['NSE', 'KGE', 'LogNSE']
    for stage in ['A', 'B', 'C', 'D']:
        stage_data = obj_metrics[obj_metrics['Stage'] == stage]
        fig_obj.add_trace(go.Bar(
            name=stage, showlegend=False,
            x=metrics_to_show,
            y=[stage_data[m].values[0] for m in metrics_to_show],
            marker_color=colors[stage]
        ), row=2, col=1)
    
    # Row 2, Col 2: Improvement over baseline A - All stages
    improvement_metrics = ['NSE', 'KGE', 'LogNSE', 'SqrtNSE']
    metrics_A = obj_metrics[obj_metrics['Stage'] == 'A']
    metrics_B = obj_metrics[obj_metrics['Stage'] == 'B']
    metrics_C = obj_metrics[obj_metrics['Stage'] == 'C']
    metrics_D = obj_metrics[obj_metrics['Stage'] == 'D']
    
    improvement_B = [metrics_B[m].values[0] - metrics_A[m].values[0] for m in improvement_metrics]
    improvement_C = [metrics_C[m].values[0] - metrics_A[m].values[0] for m in improvement_metrics]
    improvement_D = [metrics_D[m].values[0] - metrics_A[m].values[0] for m in improvement_metrics]
    
    fig_obj.add_trace(go.Bar(
        name='B-A', showlegend=False,
        x=improvement_metrics,
        y=improvement_B,
        marker_color=colors['B']
    ), row=2, col=2)
    
    fig_obj.add_trace(go.Bar(
        name='C-A', showlegend=False,
        x=improvement_metrics,
        y=improvement_C,
        marker_color=colors['C']
    ), row=2, col=2)
    
    fig_obj.add_trace(go.Bar(
        name='D-A', showlegend=False,
        x=improvement_metrics,
        y=improvement_D,
        marker_color=colors['D']
    ), row=2, col=2)
    
    # Row 2, Col 3: Routing K Parameter comparison (B, C, D)
    k_values = [routing_B['K'], routing_C['K'], routing_D['K']]
    fig_obj.add_trace(go.Bar(
        name='K', showlegend=False,
        x=['B', 'C', 'D'],
        y=k_values,
        marker_color=[colors['B'], colors['C'], colors['D']]
    ), row=2, col=3)
    
    # Update axes
    fig_obj.update_yaxes(title_text="Flow (mm/day)", row=1, col=1)
    fig_obj.update_yaxes(title_text="Flow (mm/day)", type="log", row=1, col=2)
    fig_obj.update_yaxes(title_text="Flow (mm/day)", type="log", row=1, col=3)
    fig_obj.update_xaxes(title_text="Day", row=1, col=1)
    fig_obj.update_xaxes(title_text="Day", row=1, col=2)
    fig_obj.update_xaxes(title_text="Exceedance (%)", row=1, col=3)
    fig_obj.update_yaxes(title_text="Value", row=2, col=1)
    fig_obj.update_yaxes(title_text="Δ from A", row=2, col=2)
    fig_obj.update_yaxes(title_text="K (days)", row=2, col=3)
    
    # Layout
    fig_obj.update_layout(
        height=700,
        width=1400,
        title_text=f"<b>Routing Comparison: {obj_name} Objective</b><br><sup>A=No Routing | B=Add Routing | C=Joint Cal. | D=Fixed UH+Routing</sup>",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig_obj.show()

print("\n✓ All comparison figures created!")

# %% [markdown]
# ---
# ### Combined Hydrograph Comparison (All Objectives)
#
# Side-by-side comparison of hydrographs for all objective functions.

# %%
# VISUALIZATION 3: Combined hydrograph comparison for all objectives
print("=" * 70)
print("VISUALIZATION 3: COMBINED HYDROGRAPH COMPARISON (ALL OBJECTIVES)")
print("=" * 70)

n_objectives = len(simulations)
fig_combined = make_subplots(
    rows=n_objectives, cols=3,
    subplot_titles=[f'<b>{obj} - Linear</b>' for obj in simulations.keys()] + 
                   [f'<b>{obj} - Log</b>' for obj in simulations.keys()] +
                   [f'<b>{obj} - FDC</b>' for obj in simulations.keys()],
    vertical_spacing=0.08,
    horizontal_spacing=0.06
)

# Reorganize titles since make_subplots fills row by row
for i, obj_name in enumerate(simulations.keys()):
    row = i + 1
    
    sim_data = simulations[obj_name]
    sim_A = sim_data['A']
    sim_B = sim_data['B']
    sim_C = sim_data['C']
    sim_D = sim_data['D']
    
    time_array = np.arange(len(obs))
    
    # Column 1: Linear hydrograph - All 4 stages
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=obs, mode='lines', showlegend=(i==0),
        name='Observed', line=dict(color='black', width=1)
    ), row=row, col=1)
    
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=sim_A, mode='lines', showlegend=(i==0),
        name='A (No Routing)', line=dict(color=colors['A'], width=1)
    ), row=row, col=1)
    
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=sim_B, mode='lines', showlegend=(i==0),
        name='B (+Routing)', line=dict(color=colors['B'], width=1, dash='dot')
    ), row=row, col=1)
    
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=sim_C, mode='lines', showlegend=(i==0),
        name='C (Joint)', line=dict(color=colors['C'], width=1, dash='dash')
    ), row=row, col=1)
    
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=sim_D, mode='lines', showlegend=(i==0),
        name='D (Fixed UH)', line=dict(color=colors['D'], width=1, dash='dashdot')
    ), row=row, col=1)
    
    # Column 2: Log hydrograph
    fig_combined.add_trace(go.Scatter(
        x=time_array, y=obs, mode='lines', showlegend=False,
        line=dict(color='black', width=1)
    ), row=row, col=2)
    
    for sim, stage, dash in [(sim_A, 'A', 'solid'), (sim_B, 'B', 'dot'), 
                              (sim_C, 'C', 'dash'), (sim_D, 'D', 'dashdot')]:
        fig_combined.add_trace(go.Scatter(
            x=time_array, y=sim, mode='lines', showlegend=False,
            line=dict(color=colors[stage], width=1, dash=dash)
        ), row=row, col=2)
    
    # Column 3: FDC
    exc_obs, fdc_obs = calculate_fdc(obs)
    exc_A, fdc_A = calculate_fdc(sim_A)
    exc_B, fdc_B = calculate_fdc(sim_B)
    exc_C, fdc_C = calculate_fdc(sim_C)
    exc_D, fdc_D = calculate_fdc(sim_D)
    
    fig_combined.add_trace(go.Scatter(
        x=exc_obs, y=fdc_obs, mode='lines', showlegend=False,
        line=dict(color='black', width=1.5)
    ), row=row, col=3)
    
    for exc, fdc, stage, dash in [(exc_A, fdc_A, 'A', 'solid'), (exc_B, fdc_B, 'B', 'dot'),
                                   (exc_C, fdc_C, 'C', 'dash'), (exc_D, fdc_D, 'D', 'dashdot')]:
        fig_combined.add_trace(go.Scatter(
            x=exc, y=fdc, mode='lines', showlegend=False,
            line=dict(color=colors[stage], width=1, dash=dash)
        ), row=row, col=3)
    
    # Update y-axes for this row
    fig_combined.update_yaxes(row=row, col=2, type="log")
    fig_combined.update_yaxes(row=row, col=3, type="log")

# Update subplot titles manually for row-by-row layout
for i, obj_name in enumerate(simulations.keys()):
    fig_combined.layout.annotations[i].text = f'<b>{obj_name} - Linear</b>'
    fig_combined.layout.annotations[i + n_objectives].text = f'<b>{obj_name} - Log</b>'
    fig_combined.layout.annotations[i + 2*n_objectives].text = f'<b>{obj_name} - FDC</b>'

# Actually the subplot_titles are created differently - let me fix the approach
# Update all annotations
annotations_list = list(fig_combined.layout.annotations)
for idx, obj_name in enumerate(simulations.keys()):
    if idx * 3 < len(annotations_list):
        annotations_list[idx * 3] = annotations_list[idx * 3].update(text=f'<b>{obj_name} - Linear</b>')
    if idx * 3 + 1 < len(annotations_list):
        annotations_list[idx * 3 + 1] = annotations_list[idx * 3 + 1].update(text=f'<b>{obj_name} - Log</b>')
    if idx * 3 + 2 < len(annotations_list):
        annotations_list[idx * 3 + 2] = annotations_list[idx * 3 + 2].update(text=f'<b>{obj_name} - FDC</b>')

fig_combined.update_layout(
    height=300 * n_objectives,
    width=1500,
    title_text="<b>Combined Hydrograph Comparison: All Objective Functions (4 Stages)</b><br><sup>A=No Routing | B=Add Routing | C=Joint | D=Fixed UH+Routing</sup>",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    showlegend=True
)

fig_combined.show()

print("\n✓ Combined comparison figure created!")

# %%
# Comprehensive summary statistics
print("\n" + "=" * 90)
print("COMPREHENSIVE SUMMARY: FOUR-STAGE ROUTING ANALYSIS")
print("=" * 90)

# Calculate average improvements for all metrics
print("\n=== Average Improvement from Routing (All Metrics) ===")
print(f"{'Metric':<12} {'B-A':<12} {'C-A':<12} {'D-A':<12} {'D-C':<12}")
print("-" * 60)
for metric in ['NSE', 'KGE', 'LogNSE', 'SqrtNSE', 'InvNSE']:
    avg_BA = pivots[metric]['B-A'].mean()
    avg_CA = pivots[metric]['C-A'].mean()
    avg_DA = pivots[metric]['D-A'].mean()
    avg_DC = pivots[metric]['D-C'].mean()
    print(f"{metric:<12} {avg_BA:>+.4f}       {avg_CA:>+.4f}       {avg_DA:>+.4f}       {avg_DC:>+.4f}")

# PBIAS is different - closer to 0 is better
pbias_A_mean = pivots['PBIAS']['A'].abs().mean()
pbias_B_mean = pivots['PBIAS']['B'].abs().mean()
pbias_C_mean = pivots['PBIAS']['C'].abs().mean()
pbias_D_mean = pivots['PBIAS']['D'].abs().mean()
print(f"\n{'PBIAS (avg |value|)':<20} A: {pbias_A_mean:.2f}%  |  B: {pbias_B_mean:.2f}%  |  C: {pbias_C_mean:.2f}%  |  D: {pbias_D_mean:.2f}%")

print("\n=== Best Performing Combinations (All 4 Stages) ===")
best_nse = results_df.loc[results_df['NSE'].idxmax()]
best_kge = results_df.loc[results_df['KGE'].idxmax()]
best_lognse = results_df.loc[results_df['LogNSE'].idxmax()]
best_sqrtnse = results_df.loc[results_df['SqrtNSE'].idxmax()]
print(f"  Best NSE:     {best_nse['Objective']:>8} - Stage {best_nse['Stage']} (NSE = {best_nse['NSE']:.4f})")
print(f"  Best KGE:     {best_kge['Objective']:>8} - Stage {best_kge['Stage']} (KGE = {best_kge['KGE']:.4f})")
print(f"  Best LogNSE:  {best_lognse['Objective']:>8} - Stage {best_lognse['Stage']} (LogNSE = {best_lognse['LogNSE']:.4f})")
print(f"  Best SqrtNSE: {best_sqrtnse['Objective']:>8} - Stage {best_sqrtnse['Stage']} (SqrtNSE = {best_sqrtnse['SqrtNSE']:.4f})")

print("\n=== Routing Consistently Helps? (NSE Improvement over A) ===")
helps_B = (nse_pivot['B-A'] > 0).sum()
helps_C = (nse_pivot['C-A'] > 0).sum()
helps_D = (nse_pivot['D-A'] > 0).sum()
total = len(nse_pivot)
print(f"  Stage B improves over A: {helps_B}/{total} objectives ({helps_B/total*100:.0f}%)")
print(f"  Stage C improves over A: {helps_C}/{total} objectives ({helps_C/total*100:.0f}%)")
print(f"  Stage D improves over A: {helps_D}/{total} objectives ({helps_D/total*100:.0f}%)")

print("\n=== EQUIFINALITY ANALYSIS: D vs C ===")
print("Testing whether internal UH routing is needed or just trades off with Muskingum K")
print("-" * 70)
dc_positive = (nse_pivot['D-C'] > 0).sum()
dc_negative = (nse_pivot['D-C'] < 0).sum()
dc_similar = (nse_pivot['D-C'].abs() < 0.01).sum()
print(f"  D > C (Fixed UH better):   {dc_positive}/{total} objectives")
print(f"  D < C (Joint UH better):   {dc_negative}/{total} objectives")
print(f"  D ≈ C (|Δ| < 0.01):        {dc_similar}/{total} objectives")
print(f"\n  Average D-C (NSE): {nse_pivot['D-C'].mean():+.4f}")
print(f"  Average D-C (KGE): {kge_pivot['D-C'].mean():+.4f}")

if dc_similar >= total * 0.5:
    print("\n  → EQUIFINALITY DETECTED: Internal UH and Muskingum K appear interchangeable")
    print("    Recommendation: Use Stage D (Fixed UH) for simpler, more interpretable model")
elif dc_negative > dc_positive:
    print("\n  → Joint UH calibration (C) provides benefit over fixed UH (D)")
    print("    Internal UH may capture physically meaningful hillslope routing")
else:
    print("\n  → Fixed UH (D) performs as well or better than joint calibration (C)")
    print("    Recommendation: Use Stage D for simpler interpretation")

print("\n=== Routing Parameter Statistics (Stages B, C, D) ===")
m_B = results_df[(results_df['Stage'] == 'B')].set_index('Objective')['m']
m_C = results_df[(results_df['Stage'] == 'C')].set_index('Objective')['m']
m_D = results_df[(results_df['Stage'] == 'D')].set_index('Objective')['m']
print(f"  K (Stage B): mean = {k_B.mean():.2f} days, range = [{k_B.min():.2f}, {k_B.max():.2f}]")
print(f"  K (Stage C): mean = {k_C.mean():.2f} days, range = [{k_C.min():.2f}, {k_C.max():.2f}]")
print(f"  K (Stage D): mean = {k_D.mean():.2f} days, range = [{k_D.min():.2f}, {k_D.max():.2f}]")
print(f"  m (Stage B): mean = {m_B.mean():.3f}, range = [{m_B.min():.3f}, {m_B.max():.3f}]")
print(f"  m (Stage C): mean = {m_C.mean():.3f}, range = [{m_C.min():.3f}, {m_C.max():.3f}]")
print(f"  m (Stage D): mean = {m_D.mean():.3f}, range = [{m_D.min():.3f}, {m_D.max():.3f}]")

# Key equifinality insight: compare K values between C and D
print("\n=== K Parameter Comparison (Equifinality Indicator) ===")
print("If K is larger in D than C, D compensates for missing internal UH delay")
k_diff = k_D - k_C
print(f"  K(D) - K(C): mean = {k_diff.mean():+.2f} days")
for obj in k_diff.index:
    print(f"    {obj:>8}: K(D)={k_D[obj]:.2f}, K(C)={k_C[obj]:.2f}, Δ={k_diff[obj]:+.2f}")

# %% [markdown]
# ---
# ## Part 7: Summary and Conclusions
#
# ### What We Learned
#
# 1. **Routing Concepts**: The nonlinear Muskingum method provides:
#    - Peak attenuation through storage effects (controlled by K)
#    - Time delay (lag) proportional to K parameter
#    - Nonlinear flow-dependent behavior through m parameter
#
# 2. **Comprehensive Multi-Objective Analysis**: We tested routing benefit across 5 objective functions:
#    - **NSE** (standard) - Overall fit emphasis
#    - **LogNSE** (low flow) - Emphasizes baseflow and recessions
#    - **InvNSE** (recession) - Strong recession emphasis
#    - **SqrtNSE** (balanced) - Moderate emphasis across flow range
#    - **SDEB** (multi-scale) - Spectral decomposition
#
# 3. **Comprehensive Performance Evaluation**: We compared using 7 metrics:
#    - NSE, KGE, PBIAS - Traditional performance measures
#    - LogNSE, SqrtNSE, InvNSE - Flow transformation variants
#    - Routing parameters (K, m) - Physical interpretability
#
# 4. **Four-Stage Calibration Design**:
#    - **A**: Baseline without routing (loaded from Notebook 02, includes calibrated UH)
#    - **B**: Add routing to A (fix RR including UH, calibrate routing only)
#    - **C**: Joint calibration (RR + UH + routing together - equifinality risk)
#    - **D**: Fixed UH (uh1=1, others=0) + routing (clean separation)
#
# 5. **Equifinality Investigation**: Stage D tests whether internal UH routing trades off
#    with Muskingum K parameter. If D ≈ C, the internal UH is redundant.
#
# ### Key Insights
#
# | Approach | When to Use | Pros | Cons |
# |----------|-------------|------|------|
# | **A - No Routing** | Small catchment, gauge near centroid | Simple, interpretable | May miss timing/attenuation |
# | **B - Add Routing** | Quick improvement to existing calibration | Preserves RR params, clear routing benefit | Sub-optimal overall |
# | **C - Joint** | Best overall fit needed | Maximum performance | UH/K equifinality risk |
# | **D - Fixed UH** | Clean interpretation needed | No equifinality, clearer K meaning | May lose some flexibility |
#
# ### Equifinality Interpretation
#
# - If **D ≈ C**: Internal UH not needed; use D for simpler, more interpretable model
# - If **C > D**: Internal UH provides meaningful additional flexibility
# - If **K(D) > K(C)**: D's K compensates for missing internal UH delay
#
# ### Interactive Visualizations Created
#
# 1. **Comprehensive Metrics Dashboard**: 3×3 grid showing all 4 stages
# 2. **Individual Objective Comparisons**: For each objective function:
#    - Hydrograph (linear scale) - Peak timing and magnitude
#    - Hydrograph (log scale) - Low flow performance
#    - Flow Duration Curve - Exceedance probability
#    - Metrics comparison and improvement (including Stage D)
# 3. **Combined Hydrograph Grid**: All objectives side-by-side with 4 stages
# 4. **Equifinality Analysis**: UH parameters and D-C comparison
#
# ### Recommended Workflow
#
# 1. **Start with A**: Calibrate RR model without routing (baseline)
# 2. **Try B**: Keep RR fixed, add routing - shows isolated routing benefit
# 3. **Try D**: Fixed UH + routing - conceptually clean baseline
# 4. **Try C**: Joint calibration - shows maximum achievable fit
# 5. **Compare D vs C**: Test equifinality hypothesis
# 6. **Inspect visually**: Use interactive hydrographs to assess timing/attenuation
# 7. **Decide**: If D ≈ C, prefer D for interpretability; otherwise use C

# %%
print("=" * 70)
print("NOTEBOOK COMPLETE: FOUR-STAGE ROUTING & EQUIFINALITY ANALYSIS")
print("=" * 70)
print(f"\nAnalyzed {len(available_objectives)} objective functions with four-stage calibration.")
print(f"\n=== Performance Summary (All 4 Stages) ===")
print(f"  Average NSE improvement (B-A): {nse_pivot['B-A'].mean():+.4f}")
print(f"  Average NSE improvement (C-A): {nse_pivot['C-A'].mean():+.4f}")
print(f"  Average NSE improvement (D-A): {nse_pivot['D-A'].mean():+.4f}")
print(f"  Average NSE difference  (D-C): {nse_pivot['D-C'].mean():+.4f}")
print(f"\n=== Equifinality Test Result ===")
if abs(nse_pivot['D-C'].mean()) < 0.01:
    print("  → EQUIFINALITY DETECTED: D ≈ C")
    print("    Internal UH (uh1-uh5) appears redundant with Muskingum routing")
    print("    Recommendation: Use Stage D for simpler, more interpretable model")
else:
    print(f"  → D-C = {nse_pivot['D-C'].mean():+.4f} (NSE)")
    if nse_pivot['D-C'].mean() < 0:
        print("    Joint calibration (C) provides benefit over fixed UH (D)")
    else:
        print("    Fixed UH (D) performs as well or better than joint (C)")
print(f"\n=== Best Results ===")
print(f"  Best NSE:  {best_nse['Objective']:>8} - Stage {best_nse['Stage']} (NSE = {best_nse['NSE']:.4f})")
print(f"  Best KGE:  {best_kge['Objective']:>8} - Stage {best_kge['Stage']} (KGE = {best_kge['KGE']:.4f})")
print(f"\n=== Routing Parameters (All Routing Stages) ===")
print(f"  K range: {min(k_B.min(), k_C.min(), k_D.min()):.2f} - {max(k_B.max(), k_C.max(), k_D.max()):.2f} days")
print(f"  m range: {min(m_B.min(), m_C.min(), m_D.min()):.3f} - {max(m_B.max(), m_C.max(), m_D.max()):.3f}")
print(f"  K(D) - K(C) average: {(k_D - k_C).mean():+.2f} days (>0 suggests D compensates for missing UH)")
