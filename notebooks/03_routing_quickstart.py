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
# ## Part 6: How Does Routing Benefit Vary Across Objective Functions?
#
# ### Building on Notebook 02: Calibration Quickstart
#
# **This section continues directly from Notebook 02 (Calibration Quickstart)**.
# We load the previously saved calibration results from the pickle files generated
# in that tutorial to establish our **Stage A baselines**.
#
# **Prerequisites**: You must have completed Notebook 02 and saved the calibration
# reports to `test_data/reports/`. If those files don't exist, go back and run
# Notebook 02 first.
#
# ### Objective Functions from Notebook 02
#
# We analyze ALL objective functions calibrated in Notebook 02:
#
# **NSE-based Objectives:**
#
# | Report File | Objective | Flow Regime Focus |
# |-------------|-----------|-------------------|
# | `410734_sacramento_nse_sceua.pkl` | NSE | High flows |
# | `410734_sacramento_nse_sceua_log.pkl` | LogNSE | Low flows |
# | `410734_sacramento_nse_sceua_inv.pkl` | InvNSE | Very low flows |
# | `410734_sacramento_nse_sceua_sqrt.pkl` | SqrtNSE | Balanced |
#
# **KGE-based Objectives:**
#
# | Report File | Objective | Flow Regime Focus |
# |-------------|-----------|-------------------|
# | `410734_sacramento_kge_sceua.pkl` | KGE | High flows |
# | `410734_sacramento_kge_sceua_inv.pkl` | KGE (1/Q) | Very low flows |
# | `410734_sacramento_kge_sceua_sqrt.pkl` | KGE (√Q) | Balanced |
# | `410734_sacramento_kge_sceua_log.pkl` | KGE (log Q) | Low flows |
# | `410734_sacramento_kge_np_sceua.pkl` | KGE_np | Non-parametric |
#
# **Composite Objective:**
#
# | Report File | Objective | Flow Regime Focus |
# |-------------|-----------|-------------------|
# | `410734_sacramento_sdeb_sceua.pkl` | SDEB | Multi-timescale balance |
#
# ### Experimental Design: Three-Stage Calibration
#
# **Key Question: How does routing benefit vary across different objective functions?**
#
# For **EACH** of the 10 objective functions, we perform three calibration stages:
#
# | Stage | Description | Parameters Calibrated |
# |-------|-------------|----------------------|
# | **A** | Sacramento only (no routing) | **LOADED** from Notebook 02 |
# | **B** | Add routing to Stage A (fix Sacramento params) | 3 routing parameters only |
# | **C** | Full joint calibration (Sacramento + routing) | 25 parameters (22 Sacramento + 3 routing) |
#
# This comprehensive design allows us to:
# - Compare **baseline performance** across different objective functions
# - Measure the **incremental benefit of routing** for each objective
# - Assess whether **joint optimization** consistently outperforms sequential calibration
# - Compare **calibrated routing parameters** (K, m, n) across objectives

# %%
# Load real catchment data (same as Notebook 02)
import warnings
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
# We load ALL saved reports from Notebook 02, including:
# - **NSE variants**: NSE, LogNSE, InvNSE, SqrtNSE
# - **KGE variants**: KGE, KGE_inv, KGE_sqrt, KGE_log
# - **KGE_np variants**: KGE_np, KGE_np_inv, KGE_np_sqrt, KGE_np_log
# - **Composite**: SDEB

# %%
# Load all calibration reports from Notebook 02
print("=" * 70)
print("LOADING CALIBRATION REPORTS FROM NOTEBOOK 02")
print("=" * 70)

# All available reports from Notebook 02 (must match the exact filenames)
report_files = {
    # NSE variants
    'NSE': '410734_sacramento_nse_sceua.pkl',
    'LogNSE': '410734_sacramento_nse_sceua_log.pkl',
    'InvNSE': '410734_sacramento_nse_sceua_inv.pkl',
    'SqrtNSE': '410734_sacramento_nse_sceua_sqrt.pkl',
    # KGE variants
    'KGE': '410734_sacramento_kge_sceua.pkl',
    'KGE_inv': '410734_sacramento_kge_sceua_inv.pkl',
    'KGE_sqrt': '410734_sacramento_kge_sceua_sqrt.pkl',
    'KGE_log': '410734_sacramento_kge_sceua_log.pkl',
    # KGE non-parametric variants
    'KGE_np': '410734_sacramento_kge_np_sceua.pkl',
    'KGE_np_inv': '410734_sacramento_kge_np_sceua_inv.pkl',
    'KGE_np_sqrt': '410734_sacramento_kge_np_sceua_sqrt.pkl',
    'KGE_np_log': '410734_sacramento_kge_np_sceua_log.pkl',
    # Composite
    'SDEB': '410734_sacramento_sdeb_sceua.pkl',
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

if len(loaded_reports) == 0:
    raise FileNotFoundError(
        "No calibration reports found!\n"
        "Please run Notebook 02 (Calibration Quickstart) first to generate the reports."
    )

# %%
# Display summary of loaded calibrations
print("\n" + "=" * 70)
print("CALIBRATION RESULTS SUMMARY (Stage A - from Notebook 02)")
print("=" * 70)
print(f"\n{'Objective':<15} {'Best Value':>12} {'Runtime':>10}")
print("-" * 45)

for name, report in loaded_reports.items():
    obj_val = report.result.best_objective
    runtime = report.result.runtime_seconds
    print(f"{name:<15} {obj_val:>12.4f} {runtime:>8.1f}s")

# %% [markdown]
# ### Loading Custom Parameter Bounds
#
# We use the same custom bounds file from Notebook 02 (Step 9).

# %%
# Load original parameter bounds
bounds_file = DATA_DIR / 'sacramento_bounds.txt'
sac_bounds = load_parameter_bounds(bounds_file)
print(f"Loaded parameter bounds from: {bounds_file}")
print(f"Parameters: {len(sac_bounds)}")

# %% [markdown]
# ---
# ### Define Objective Functions
#
# We need to recreate the objective functions to match what was used in Notebook 02.

# %%
# Define objective functions matching Notebook 02
from pyrrm.objectives import NSE, KGE, KGENonParametric, RMSE, PBIAS, SDEB, FlowTransformation

OBJECTIVE_CONFIGS = {
    'NSE': {
        'objective_fn': lambda: NSE(),
        'description': 'NSE (high flows)'
    },
    'LogNSE': {
        'objective_fn': lambda: NSE(transform=FlowTransformation('log', epsilon_value=0.01)),
        'description': 'LogNSE (low flows)'
    },
    'InvNSE': {
        'objective_fn': lambda: NSE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
        'description': 'InvNSE (very low flows)'
    },
    'SqrtNSE': {
        'objective_fn': lambda: NSE(transform=FlowTransformation('sqrt')),
        'description': 'SqrtNSE (balanced)'
    },
    'KGE': {
        'objective_fn': lambda: KGE(),
        'description': 'KGE (high flows)'
    },
    'KGE_inv': {
        'objective_fn': lambda: KGE(transform=FlowTransformation('inverse', epsilon_value=0.01)),
        'description': 'KGE (1/Q) (very low flows)'
    },
    'KGE_sqrt': {
        'objective_fn': lambda: KGE(transform=FlowTransformation('sqrt')),
        'description': 'KGE (√Q) (balanced)'
    },
    'KGE_log': {
        'objective_fn': lambda: KGE(transform=FlowTransformation('log', epsilon_value=0.01)),
        'description': 'KGE (log Q) (low flows)'
    },
    'KGE_np': {
        'objective_fn': lambda: KGENonParametric(),
        'description': 'KGE_np (high flows)'
    },
    'KGE_np_inv': {
        'objective_fn': lambda: KGENonParametric(transform=FlowTransformation('inverse', epsilon_value=0.01)),
        'description': 'KGE_np (1/Q) (very low flows)'
    },
    'KGE_np_sqrt': {
        'objective_fn': lambda: KGENonParametric(transform=FlowTransformation('sqrt')),
        'description': 'KGE_np (√Q) (balanced)'
    },
    'KGE_np_log': {
        'objective_fn': lambda: KGENonParametric(transform=FlowTransformation('log', epsilon_value=0.01)),
        'description': 'KGE_np (log Q) (low flows)'
    },
    'SDEB': {
        'objective_fn': lambda: SDEB(alpha=0.1, lam=0.5),
        'description': 'SDEB (multi-scale)'
    },
}

# Filter to only include objectives where we have saved reports
available_objectives = {}
for name in loaded_reports.keys():
    if name in OBJECTIVE_CONFIGS:
        available_objectives[name] = OBJECTIVE_CONFIGS[name]
        print(f"  ✓ {name}: {OBJECTIVE_CONFIGS[name]['description']}")

print(f"\nAnalyzing {len(available_objectives)} objective functions")

# %% [markdown]
# ---
# ### Three-Stage Calibration Experiment
#
# For each objective function, we run three calibration stages:
#
# | Stage | Description | Parameters |
# |-------|-------------|------------|
# | **A** | Sacramento only (no routing) | **LOADED** from Notebook 02 (22 params) |
# | **B** | Add routing to A (fix Sacramento) | 3 routing params only |
# | **C** | Full joint calibration | 25 params (22 Sacramento + 3 routing) |

# %%
# Storage for all results
all_results = {}

# Routing parameter bounds (same for all objectives)
routing_only_bounds = {
    'routing_K': (0.5, 15.0),
    'routing_m': (0.1, 1.2),
    'routing_n_subreaches': (1, 10)
}

# Joint bounds (Sacramento + routing)
joint_bounds = sac_bounds.copy()
joint_bounds['routing_K'] = (0.5, 15.0)
joint_bounds['routing_m'] = (0.1, 1.2)
joint_bounds['routing_n_subreaches'] = (1, 10)

CANONICAL_NAMES = {
    'NSE': '410734_sacramento_nse_sceua',
    'LogNSE': '410734_sacramento_nse_sceua_log',
    'InvNSE': '410734_sacramento_nse_sceua_inv',
    'SqrtNSE': '410734_sacramento_nse_sceua_sqrt',
    'KGE': '410734_sacramento_kge_sceua',
    'KGE_inv': '410734_sacramento_kge_sceua_inv',
    'KGE_sqrt': '410734_sacramento_kge_sceua_sqrt',
    'KGE_log': '410734_sacramento_kge_sceua_log',
    'KGE_np': '410734_sacramento_kge_np_sceua',
    'KGE_np_inv': '410734_sacramento_kge_np_sceua_inv',
    'KGE_np_sqrt': '410734_sacramento_kge_np_sceua_sqrt',
    'KGE_np_log': '410734_sacramento_kge_np_sceua_log',
    'SDEB': '410734_sacramento_sdeb_sceua',
}

ROUTING_DIR = REPORTS_DIR / 'routing'
ROUTING_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("THREE-STAGE CALIBRATION FOR ALL OBJECTIVE FUNCTIONS")
print("=" * 70)
print(f"\nStage A: Load from Notebook 02 (22 Sacramento parameters)")
print(f"Stage B: Add routing (3 parameters, fix Sacramento from A)")
print(f"Stage C: Joint calibration (25 parameters total)\n")

# Calibration settings
MAX_EVALS_B = 3000   # Routing only (3 params)
MAX_EVALS_C = 15000  # Joint calibration (25 params)

# %%
# Run three-stage calibration for each objective function
for obj_name, config in available_objectives.items():
    print("\n" + "=" * 70)
    print(f"OBJECTIVE: {obj_name} - {config['description']}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Stage A: Load from saved report (Notebook 02)
    # -------------------------------------------------------------------------
    print(f"\n--- Stage A: Load from Notebook 02 ---")
    report_A = loaded_reports[obj_name]
    result_A = report_A.result
    params_A = result_A.best_parameters.copy()
    
    print(f"  Best objective: {result_A.best_objective:.4f}")
    print(f"  Parameters: {len(params_A)}")
    
    # -------------------------------------------------------------------------
    # Stage B: Add routing (fix Sacramento parameters from A)
    # -------------------------------------------------------------------------
    print(f"\n--- Stage B: Add Routing (fix Sacramento from A) ---")
    
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
    
    print(f"  Running SCE-UA (3 routing params, max_evals={MAX_EVALS_B})...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result_B = runner_B.run_sceua_direct(
            max_evals=MAX_EVALS_B, seed=42, verbose=True,
            max_tolerant_iter=100, tolerance=1e-4
        )
    
    routing_params_B = result_B.best_parameters.copy()
    K_B = routing_params_B.get('routing_K', 5.0)
    m_B = routing_params_B.get('routing_m', 0.8)
    n_B = int(round(routing_params_B.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective: {result_B.best_objective:.4f}")
    print(f"  Routing: K={K_B:.2f} days, m={m_B:.3f}, n={n_B}")
    
    # Save Stage B report
    report_B = runner_B.create_report(result_B, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_B.save(ROUTING_DIR / f"{CANONICAL_NAMES[obj_name]}_routingB")
    
    # -------------------------------------------------------------------------
    # Stage C: Joint calibration (Sacramento + routing)
    # -------------------------------------------------------------------------
    print(f"\n--- Stage C: Joint Calibration (Sacramento + Routing) ---")
    
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
    
    print(f"  Running SCE-UA (25 params, max_evals={MAX_EVALS_C})...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result_C = runner_C.run_sceua_direct(
            max_evals=MAX_EVALS_C, seed=42, verbose=True,
            max_tolerant_iter=100, tolerance=1e-4
        )
    
    params_C = result_C.best_parameters.copy()
    rr_params_C = {k: v for k, v in params_C.items() if not k.startswith('routing_')}
    K_C = params_C.get('routing_K', 5.0)
    m_C = params_C.get('routing_m', 0.8)
    n_C = int(round(params_C.get('routing_n_subreaches', 3)))
    
    print(f"  Best objective: {result_C.best_objective:.4f}")
    print(f"  Routing: K={K_C:.2f} days, m={m_C:.3f}, n={n_C}")
    
    # Save Stage C report
    report_C = runner_C.create_report(result_C, catchment_info={
        'name': 'Queanbeyan River', 'gauge_id': '410734', 'area_km2': CATCHMENT_AREA_KM2
    })
    report_C.save(ROUTING_DIR / f"{CANONICAL_NAMES[obj_name]}_routingC")
    
    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    all_results[obj_name] = {
        'A': {'result': result_A, 'params': params_A},
        'B': {'result': result_B, 'params_rr': params_A, 
              'routing': {'K': K_B, 'm': m_B, 'n': n_B}},
        'C': {'result': result_C, 'params_rr': rr_params_C, 
              'routing': {'K': K_C, 'm': m_C, 'n': n_C}},
    }
    
    print(f"\n✓ {obj_name} complete!")

print("\n" + "=" * 70)
print("ALL CALIBRATIONS COMPLETE")
print("=" * 70)

# %% [markdown]
# ---
# ### Generate Simulations and Calculate Metrics

# %%
# Generate simulations and calculate metrics
print("=" * 70)
print("GENERATING SIMULATIONS FOR ALL CALIBRATIONS")
print("=" * 70)

# Metrics calculators (FlowTransformation already imported above)
nse_metric = NSE()
kge_metric = KGE()
rmse_metric = RMSE()
pbias_metric = PBIAS()
lognse_metric = NSE(transform=FlowTransformation('log', epsilon_value=0.01))
sqrtnse_metric = NSE(transform=FlowTransformation('sqrt'))

obs = real_observed[REAL_WARMUP:]
time_index = real_inputs.index[REAL_WARMUP:]

# Store results and simulations
comparison_results = []
simulations = {}

for obj_name, results in all_results.items():
    print(f"\nProcessing {obj_name}...")
    
    params_A = results['A']['params']
    params_rr_B = results['B']['params_rr']
    routing_B = results['B']['routing']
    params_rr_C = results['C']['params_rr']
    routing_C = results['C']['routing']
    
    # Simulation A: No routing
    model_A = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_A.set_parameters(params_A)
    model_A.reset()
    sim_A = model_A.run(real_inputs)['runoff'].values[REAL_WARMUP:]
    
    # Simulation B: Fixed Sacramento + routing
    model_B = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_B.set_parameters(params_rr_B)
    model_B.reset()
    sim_B_direct = model_B.run(real_inputs)['runoff'].values
    router_B_obj = NonlinearMuskingumRouter(
        K=routing_B['K'], m=routing_B['m'], n_subreaches=routing_B['n']
    )
    sim_B = router_B_obj.route(sim_B_direct, dt=1.0)[REAL_WARMUP:]
    
    # Simulation C: Joint calibration
    model_C = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
    model_C.set_parameters(params_rr_C)
    model_C.reset()
    sim_C_direct = model_C.run(real_inputs)['runoff'].values
    router_C_obj = NonlinearMuskingumRouter(
        K=routing_C['K'], m=routing_C['m'], n_subreaches=routing_C['n']
    )
    sim_C = router_C_obj.route(sim_C_direct, dt=1.0)[REAL_WARMUP:]
    
    # Store simulations
    simulations[obj_name] = {
        'A': sim_A, 'B': sim_B, 'C': sim_C,
        'routing_B': routing_B, 'routing_C': routing_C
    }
    
    # Calculate metrics for all stages
    for stage, sim, routing in [('A', sim_A, None), ('B', sim_B, routing_B), ('C', sim_C, routing_C)]:
        comparison_results.append({
            'Objective': obj_name,
            'Stage': stage,
            'NSE': nse_metric(obs, sim),
            'KGE': kge_metric(obs, sim),
            'PBIAS': pbias_metric(obs, sim),
            'RMSE': rmse_metric(obs, sim),
            'LogNSE': lognse_metric(obs, sim),
            'SqrtNSE': sqrtnse_metric(obs, sim),
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

# Create pivot tables
metrics_to_compare = ['NSE', 'KGE', 'PBIAS', 'LogNSE', 'SqrtNSE']
pivots = {}

for metric in metrics_to_compare:
    pivot = results_df.pivot(index='Objective', columns='Stage', values=metric)
    pivot = pivot[['A', 'B', 'C']]
    pivot['B-A'] = pivot['B'] - pivot['A']
    pivot['C-A'] = pivot['C'] - pivot['A']
    pivot['C-B'] = pivot['C'] - pivot['B']
    pivots[metric] = pivot

# Display key metrics
print("\n=== NSE Comparison ===")
print("A=Sacramento only | B=Add Routing | C=Joint Calibration")
print(pivots['NSE'].round(4).to_string())

print("\n=== KGE Comparison ===")
print(pivots['KGE'].round(4).to_string())

print("\n=== LogNSE Comparison (Low Flow Performance) ===")
print(pivots['LogNSE'].round(4).to_string())

# %% [markdown]
# ---
# ### Visualization: Routing Benefit by Objective Function
#
# For each objective function, we show the improvement from A → B → C.

# %%
# VISUALIZATION: Comprehensive comparison for each objective function
import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors = {'A': '#3498db', 'B': '#e67e22', 'C': '#27ae60'}
stage_labels = {'A': 'Sacramento Only', 'B': 'Add Routing', 'C': 'Joint Calibration'}

print("=" * 70)
print("VISUALIZATION: ROUTING BENEFIT BY OBJECTIVE FUNCTION")
print("=" * 70)

def calculate_fdc(flows):
    """Calculate flow duration curve."""
    sorted_flows = np.sort(flows)[::-1]
    n = len(sorted_flows)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100
    return exceedance, sorted_flows

# Create individual figure for each objective
for obj_name in simulations.keys():
    print(f"\nCreating figure for {obj_name}...")
    
    sim_data = simulations[obj_name]
    sim_A, sim_B, sim_C = sim_data['A'], sim_data['B'], sim_data['C']
    routing_B, routing_C = sim_data['routing_B'], sim_data['routing_C']
    
    obj_metrics = results_df[results_df['Objective'] == obj_name]
    
    # Create 2x3 subplot
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '<b>Hydrograph (Linear)</b>',
            '<b>Hydrograph (Log)</b>',
            '<b>Flow Duration Curve</b>',
            '<b>Performance Metrics</b>',
            '<b>Improvement over A</b>',
            '<b>Routing Parameters</b>'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    time_arr = np.arange(len(obs))
    
    # Row 1: Hydrographs and FDC
    # Linear hydrograph
    fig.add_trace(go.Scatter(x=time_arr, y=obs, name='Observed', 
                             line=dict(color='black', width=1.5)), row=1, col=1)
    for sim, stage in [(sim_A, 'A'), (sim_B, 'B'), (sim_C, 'C')]:
        nse_val = obj_metrics[obj_metrics['Stage']==stage]['NSE'].values[0]
        fig.add_trace(go.Scatter(
            x=time_arr, y=sim, name=f'{stage}: {stage_labels[stage]} (NSE={nse_val:.3f})',
            line=dict(color=colors[stage], width=1)), row=1, col=1)
    
    # Log hydrograph
    fig.add_trace(go.Scatter(x=time_arr, y=obs, showlegend=False,
                             line=dict(color='black', width=1.5)), row=1, col=2)
    for sim, stage in [(sim_A, 'A'), (sim_B, 'B'), (sim_C, 'C')]:
        fig.add_trace(go.Scatter(x=time_arr, y=sim, showlegend=False,
                                 line=dict(color=colors[stage], width=1)), row=1, col=2)
    
    # FDC
    exc_obs, fdc_obs = calculate_fdc(obs)
    fig.add_trace(go.Scatter(x=exc_obs, y=fdc_obs, showlegend=False,
                             line=dict(color='black', width=2)), row=1, col=3)
    for sim, stage in [(sim_A, 'A'), (sim_B, 'B'), (sim_C, 'C')]:
        exc, fdc = calculate_fdc(sim)
        fig.add_trace(go.Scatter(x=exc, y=fdc, showlegend=False,
                                 line=dict(color=colors[stage], width=1.5)), row=1, col=3)
    
    # Row 2: Metrics
    # Performance metrics
    for stage in ['A', 'B', 'C']:
        stage_data = obj_metrics[obj_metrics['Stage'] == stage]
        fig.add_trace(go.Bar(name=stage, showlegend=False, x=['NSE', 'KGE', 'LogNSE'],
                             y=[stage_data['NSE'].values[0], stage_data['KGE'].values[0], 
                                stage_data['LogNSE'].values[0]],
                             marker_color=colors[stage]), row=2, col=1)
    
    # Improvement over A
    metrics_A = obj_metrics[obj_metrics['Stage'] == 'A']
    metrics_B = obj_metrics[obj_metrics['Stage'] == 'B']
    metrics_C = obj_metrics[obj_metrics['Stage'] == 'C']
    
    for stage, metrics_stage, color in [('B', metrics_B, colors['B']), ('C', metrics_C, colors['C'])]:
        improvement = [metrics_stage[m].values[0] - metrics_A[m].values[0] for m in ['NSE', 'KGE', 'LogNSE']]
        fig.add_trace(go.Bar(name=f'{stage}-A', showlegend=False, x=['NSE', 'KGE', 'LogNSE'],
                             y=improvement, marker_color=color), row=2, col=2)
    
    # Routing parameters
    fig.add_trace(go.Bar(name='K (B)', x=['K (days)', 'm'], 
                         y=[routing_B['K'], routing_B['m']], 
                         marker_color=colors['B'], showlegend=False), row=2, col=3)
    fig.add_trace(go.Bar(name='K (C)', x=['K (days)', 'm'], 
                         y=[routing_C['K'], routing_C['m']], 
                         marker_color=colors['C'], showlegend=False), row=2, col=3)
    
    # Update axes
    fig.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Flow (ML/day)", type="log", row=1, col=3)
    fig.update_xaxes(title_text="Exceedance (%)", row=1, col=3)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Δ from A", row=2, col=2)
    
    fig.update_layout(
        height=700, width=1400,
        title_text=f"<b>Routing Benefit: {obj_name}</b><br><sup>{OBJECTIVE_CONFIGS[obj_name]['description']}</sup>",
        barmode='group',
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center')
    )
    
    fig.show()

# %% [markdown]
# ---
# ### Summary Table: Routing Benefit Across All Objectives

# %%
# Summary table
print("\n" + "=" * 90)
print("SUMMARY: ROUTING BENEFIT BY OBJECTIVE FUNCTION")
print("=" * 90)

nse_pivot = pivots['NSE']
kge_pivot = pivots['KGE']

print("\n=== NSE Improvement from Routing ===")
print(f"{'Objective':<12} {'A (Baseline)':<12} {'B (Add Rt)':<12} {'C (Joint)':<12} {'B-A':<10} {'C-A':<10}")
print("-" * 70)
for obj in nse_pivot.index:
    a, b, c = nse_pivot.loc[obj, 'A'], nse_pivot.loc[obj, 'B'], nse_pivot.loc[obj, 'C']
    ba, ca = nse_pivot.loc[obj, 'B-A'], nse_pivot.loc[obj, 'C-A']
    print(f"{obj:<12} {a:<12.4f} {b:<12.4f} {c:<12.4f} {ba:<+10.4f} {ca:<+10.4f}")

# Average improvements
print("\n=== Average Improvement Across All Objectives ===")
print(f"  NSE: B-A = {nse_pivot['B-A'].mean():+.4f}, C-A = {nse_pivot['C-A'].mean():+.4f}")
print(f"  KGE: B-A = {kge_pivot['B-A'].mean():+.4f}, C-A = {kge_pivot['C-A'].mean():+.4f}")

# %%
# Routing parameter summary
print("\n" + "=" * 90)
print("ROUTING PARAMETERS BY OBJECTIVE FUNCTION")
print("=" * 90)

routing_df = results_df[results_df['Stage'].isin(['B', 'C'])][['Objective', 'Stage', 'K', 'm', 'n']]
print("\n=== Stage B (Fixed Sacramento + Routing) ===")
print(routing_df[routing_df['Stage'] == 'B'][['Objective', 'K', 'm', 'n']].to_string(index=False))

print("\n=== Stage C (Joint Calibration) ===")
print(routing_df[routing_df['Stage'] == 'C'][['Objective', 'K', 'm', 'n']].to_string(index=False))

# %%
# Final summary statistics
print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

# Best results
best_nse = results_df.loc[results_df['NSE'].idxmax()]
best_kge = results_df.loc[results_df['KGE'].idxmax()]
best_lognse = results_df.loc[results_df['LogNSE'].idxmax()]

print(f"\n=== Best Performing Combinations ===")
print(f"  Best NSE:    {best_nse['Objective']:>10} - Stage {best_nse['Stage']} (NSE = {best_nse['NSE']:.4f})")
print(f"  Best KGE:    {best_kge['Objective']:>10} - Stage {best_kge['Stage']} (KGE = {best_kge['KGE']:.4f})")
print(f"  Best LogNSE: {best_lognse['Objective']:>10} - Stage {best_lognse['Stage']} (LogNSE = {best_lognse['LogNSE']:.4f})")

# Routing benefit consistency
print(f"\n=== Does Routing Consistently Help? ===")
improves_B = (nse_pivot['B-A'] > 0).sum()
improves_C = (nse_pivot['C-A'] > 0).sum()
total = len(nse_pivot)
print(f"  Stage B improves over A: {improves_B}/{total} objectives ({improves_B/total*100:.0f}%)")
print(f"  Stage C improves over A: {improves_C}/{total} objectives ({improves_C/total*100:.0f}%)")

# K parameter statistics
k_B = results_df[results_df['Stage'] == 'B'].set_index('Objective')['K']
k_C = results_df[results_df['Stage'] == 'C'].set_index('Objective')['K']
print(f"\n=== Routing K Parameter Statistics ===")
print(f"  Stage B: mean = {k_B.mean():.2f} days, range = [{k_B.min():.2f}, {k_B.max():.2f}]")
print(f"  Stage C: mean = {k_C.mean():.2f} days, range = [{k_C.min():.2f}, {k_C.max():.2f}]")

# %% [markdown]
# ---
# ## Part 7: Summary and Conclusions
#
# ### Three-Stage Calibration Design
#
# We investigated how channel routing benefits vary across different objective functions
# using a systematic three-stage calibration approach:
#
# | Stage | Description | Parameters |
# |-------|-------------|------------|
# | **A** | Sacramento only (baseline) | 22 Sacramento parameters |
# | **B** | Add routing (fix Sacramento) | 3 routing parameters |
# | **C** | Joint calibration | 25 parameters (Sacramento + routing) |
#
# ### Objective Functions Analyzed (13 total)
#
# | Family | Objective | Flow Regime Focus |
# |--------|-----------|-------------------|
# | **NSE** | NSE | High flows |
# | | LogNSE | Low flows |
# | | InvNSE | Very low flows |
# | | SqrtNSE | Balanced |
# | **KGE** | KGE | High flows |
# | | KGE_inv | Very low flows |
# | | KGE_sqrt | Balanced |
# | | KGE_log | Low flows |
# | **KGE_np** | KGE_np | High flows (non-parametric) |
# | | KGE_np_inv | Very low flows |
# | | KGE_np_sqrt | Balanced |
# | | KGE_np_log | Low flows |
# | **Composite** | SDEB | Multi-timescale |
#
# ### Key Findings
#
# 1. **Routing Benefit Varies by Objective**:
#    - High-flow focused objectives (NSE, KGE) typically show moderate improvement
#    - Low-flow focused objectives (LogNSE, KGE_log) may show different patterns
#    - The benefit depends on whether timing errors are a dominant source of error
#
# 2. **Stage B vs Stage C**:
#    - Stage B (add routing to existing calibration) is faster and preserves RR parameters
#    - Stage C (joint calibration) can achieve better overall fit but takes longer
#    - The difference (C-B) indicates whether re-optimizing Sacramento parameters helps
#
# 3. **Routing Parameters**:
#    - K (travel time) typically ranges from 0.5-15 days depending on catchment characteristics
#    - m (nonlinearity exponent) affects how attenuation varies with flow magnitude
#    - Different objectives may converge to different routing parameters
#
# ### Recommendations
#
# 1. **Start with Stage A**: Establish baseline without routing
# 2. **Try Stage B**: Quick assessment of routing benefit with fixed RR parameters
# 3. **Compare to Stage C**: Full joint calibration shows maximum achievable fit
# 4. **Analyze by Objective**: Different objectives may reveal different aspects of routing benefit
# 5. **Inspect Visually**: Use hydrographs and FDCs to understand where routing helps

# %%
print("=" * 70)
print("NOTEBOOK COMPLETE: THREE-STAGE ROUTING ANALYSIS")
print("=" * 70)
print(f"\nAnalyzed {len(available_objectives)} objective functions")
print(f"  NSE variants:      NSE, LogNSE, InvNSE, SqrtNSE")
print(f"  KGE variants:      KGE, KGE_inv, KGE_sqrt, KGE_log")
print(f"  KGE_np variants:   KGE_np, KGE_np_inv, KGE_np_sqrt, KGE_np_log")
print(f"  Composite:         SDEB")
print(f"\nThree calibration stages:")
print(f"  A: Sacramento only (22 params) - loaded from Notebook 02")
print(f"  B: Add routing (3 params) - fix Sacramento from A")
print(f"  C: Joint calibration (25 params) - optimize all together")
print(f"\n=== Overall Routing Benefit ===")
print(f"  Average NSE improvement (B-A): {nse_pivot['B-A'].mean():+.4f}")
print(f"  Average NSE improvement (C-A): {nse_pivot['C-A'].mean():+.4f}")
print(f"  Average KGE improvement (B-A): {kge_pivot['B-A'].mean():+.4f}")
print(f"  Average KGE improvement (C-A): {kge_pivot['C-A'].mean():+.4f}")
