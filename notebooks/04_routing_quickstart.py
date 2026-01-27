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
from pyrrm.objectives import NSE as NSE_new, FlowTransformation

# Create SqrtNSE objective for calibration (balanced high/low flow performance)
# Uses pyrrm.objectives.NSE which supports flow transformations
SqrtNSE = lambda: NSE_new(transform=FlowTransformation('sqrt'))

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
# Now let's calibrate the RR model + routing parameters together on the synthetic
# data using the **SqrtNSE objective** (NSE on sqrt-transformed flows). 
# This verifies we can recover the true routing parameters (K=3.5, m=0.85).

# %%
# Create fresh model for calibration
rr_model_synth = Sacramento()
router_synth = NonlinearMuskingumRouter(K=5.0, m=0.9, n_subreaches=3)
model_synth = RoutedModel(rr_model_synth, router_synth, routing_enabled=True)

# Custom bounds - tighter for routing
synth_bounds = model_synth.get_parameter_bounds()
synth_bounds['routing_K'] = (0.5, 10.0)        # True K=3.5
synth_bounds['routing_m'] = (0.5, 1.2)         # True m=0.85
synth_bounds['routing_n_subreaches'] = (1, 10)  # Calibrate n_subreaches

# Create calibration runner
runner_synth = CalibrationRunner(
    model=model_synth,
    inputs=inputs,
    observed=observed,
    objective=SqrtNSE(),  # Use sqrt-transformed NSE for balanced high/low flow performance
    parameter_bounds=synth_bounds,
    warmup_period=WARMUP_DAYS
)

print("Synthetic Data Calibration Setup:")
print("=" * 50)
print(f"Total parameters: {len(synth_bounds)}")
print(f"Routing bounds: K=[0.5, 10], m=[0.5, 1.2], n=[1, 10]")
print(f"True values to recover: K={TRUE_ROUTING_PARAMS['K']}, m={TRUE_ROUTING_PARAMS['m']}, n={TRUE_ROUTING_PARAMS.get('n_subreaches', 3)}")

# %%
# Run calibration (brief for demonstration)
import warnings
print("\nRunning SCE-UA calibration on synthetic data...")

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*Timestep dt.*too large.*')
    warnings.filterwarnings('ignore', message='.*Newton-Raphson did not converge.*')
    
    result_synth = runner_synth.run_sceua(
        n_iterations=5000,
        ngs=7,
        kstop=3,
        pcento=0.01,
        dbname='routing_synth_cal',
        dbformat='csv'
    )

print("\n" + result_synth.summary())

# %%
# Check routing parameter recovery
cal_routing = {k: v for k, v in result_synth.best_parameters.items() 
               if k.startswith('routing_')}

print("=" * 60)
print("ROUTING PARAMETER RECOVERY")
print("=" * 60)
print(f"\n{'Parameter':<20} {'True':>10} {'Calibrated':>12} {'Error':>10}")
print("-" * 55)
print(f"{'K (storage const)':<20} {TRUE_ROUTING_PARAMS['K']:>10.2f} {cal_routing.get('routing_K', 0):>12.2f} {cal_routing.get('routing_K', 0) - TRUE_ROUTING_PARAMS['K']:>+10.2f}")
print(f"{'m (nonlinearity)':<20} {TRUE_ROUTING_PARAMS['m']:>10.2f} {cal_routing.get('routing_m', 0):>12.2f} {cal_routing.get('routing_m', 0) - TRUE_ROUTING_PARAMS['m']:>+10.2f}")

print(f"\nCalibrated NSE: {-result_synth.best_objective:.4f}")

# %% [markdown]
# ---
# ## Part 6: Calibration WITH vs WITHOUT Routing (Real Data)
#
# This section continues from **Notebook 02: Calibration Quickstart** where we
# explored different objective functions and parameter bounds on gauge 410734.
#
# ### Building on Notebook 02
#
# In Notebook 02 (Step 9), we found that using **custom parameter bounds** improved
# calibration performance. Here, we use those same custom bounds to ensure a fair
# comparison when evaluating the benefit of adding channel routing.
#
# **Key Question: Does adding routing to calibration improve model performance?**
#
# We'll perform two calibrations:
# 1. **Without Routing**: Sacramento with custom bounds (same as Notebook 02)
# 2. **With Routing**: Sacramento + Muskingum routing with custom bounds
#
# Both use the **SqrtNSE objective function** (NSE on sqrt-transformed flows) for balanced performance.

# %%
# Load real catchment data (same as Notebook 02)
from pathlib import Path
from pyrrm.data import load_parameter_bounds

DATA_DIR = Path('../data/410734')
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
# ### Calibration A: Sacramento WITHOUT Routing
#
# First, we calibrate the Sacramento model alone (no routing) using custom bounds
# and the SqrtNSE objective function (sqrt-transformed flows).

# %%
# Create model WITHOUT routing
model_no_routing = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)

# Setup calibration runner with custom bounds
runner_no_routing = CalibrationRunner(
    model=model_no_routing,
    inputs=real_inputs,
    observed=real_observed,
    objective=SqrtNSE(),  # sqrt-transformed NSE
    parameter_bounds=custom_bounds,
    warmup_period=REAL_WARMUP
)

print("=" * 70)
print("CALIBRATION A: Sacramento WITHOUT Routing")
print("=" * 70)
print(f"Objective: SqrtNSE (sqrt-transformed flows for balanced performance)")
print(f"Parameter bounds: Custom (from Notebook 02)")
print(f"Parameters to calibrate: {len(custom_bounds)}")
print("\nRunning SCE-UA optimization...")

# Run calibration
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    result_no_routing = runner_no_routing.run_sceua(
        n_iterations=5000,
        ngs=7,
        kstop=5,
        pcento=0.01,
        dbname='cal_no_routing',
        dbformat='csv'
    )

print(f"\n✓ Calibration A complete!")
print(f"  Best SqrtNSE: {result_no_routing.best_objective:.4f}")

# %% [markdown]
# ### Calibration B: Sacramento WITH Routing
#
# Now we calibrate Sacramento + Muskingum routing together, using the same
# custom bounds for RR parameters plus routing parameters.

# %%
# Create model WITH routing
rr_model = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
model_with_routing = RoutedModel(rr_model, router, routing_enabled=True)

# Combine custom RR bounds with routing bounds
bounds_with_routing = custom_bounds.copy()
bounds_with_routing['routing_K'] = (0.5, 15.0)       # Storage constant [days]
bounds_with_routing['routing_m'] = (0.5, 1.2)        # Nonlinearity exponent
bounds_with_routing['routing_n_subreaches'] = (1, 10) # Calibrate n_subreaches

# Setup calibration runner
runner_with_routing = CalibrationRunner(
    model=model_with_routing,
    inputs=real_inputs,
    observed=real_observed,
    objective=SqrtNSE(),  # sqrt-transformed NSE
    parameter_bounds=bounds_with_routing,
    warmup_period=REAL_WARMUP
)

print("=" * 70)
print("CALIBRATION B: Sacramento WITH Routing")
print("=" * 70)
print(f"Objective: SqrtNSE (sqrt-transformed flows for balanced performance)")
print(f"Parameter bounds: Custom RR + Routing")
print(f"Parameters to calibrate: {len(bounds_with_routing)}")
print(f"  RR parameters: {len(custom_bounds)}")
print(f"  Routing parameters: {len(bounds_with_routing) - len(custom_bounds)}")
print("\nRouting parameter bounds:")
print(f"  routing_K: [{bounds_with_routing['routing_K'][0]}, {bounds_with_routing['routing_K'][1]}] days")
print(f"  routing_m: [{bounds_with_routing['routing_m'][0]}, {bounds_with_routing['routing_m'][1]}]")
print(f"  routing_n_subreaches: [{bounds_with_routing['routing_n_subreaches'][0]}, {bounds_with_routing['routing_n_subreaches'][1]}]")
print("\nRunning SCE-UA optimization...")

# Run calibration
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*Timestep dt.*too large.*')
    warnings.filterwarnings('ignore', message='.*Newton-Raphson did not converge.*')
    result_with_routing = runner_with_routing.run_sceua(
        n_iterations=5000,
        ngs=7,
        kstop=5,
        pcento=0.01,
        dbname='cal_with_routing',
        dbformat='csv'
    )

print(f"\n✓ Calibration B complete!")
print(f"  Best SqrtNSE: {result_with_routing.best_objective:.4f}")

# %% [markdown]
# ### Compare Results: With vs Without Routing

# %%
# Run simulations with calibrated parameters
print("=" * 70)
print("GENERATING SIMULATIONS WITH CALIBRATED PARAMETERS")
print("=" * 70)

# Simulation A: Without routing
model_A = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
model_A.set_parameters(result_no_routing.best_parameters)
model_A.reset()
sim_no_routing = model_A.run(real_inputs)['runoff'].values
print(f"Simulation A (no routing): peak = {np.max(sim_no_routing):.1f} ML/day")

# Simulation B: With routing
# Extract RR parameters and routing parameters from calibration result
rr_params_B = {k: v for k, v in result_with_routing.best_parameters.items() 
               if not k.startswith('routing_')}
routing_params_B = {k: v for k, v in result_with_routing.best_parameters.items() 
                    if k.startswith('routing_')}

# Get calibrated routing parameters
cal_K = routing_params_B.get('routing_K', 5.0)
cal_m = routing_params_B.get('routing_m', 0.8)
cal_n = int(routing_params_B.get('routing_n_subreaches', 3))

print(f"\nCalibrated routing parameters:")
print(f"  K = {cal_K:.2f} days, m = {cal_m:.3f}, n = {cal_n}")

# Step 1: Run RR model with calibrated RR parameters to get direct flow
model_B_rr = Sacramento(catchment_area_km2=CATCHMENT_AREA_KM2)
model_B_rr.set_parameters(rr_params_B)
model_B_rr.reset()
sim_direct = model_B_rr.run(real_inputs)['runoff'].values
print(f"\nDirect flow (before routing): peak = {np.max(sim_direct):.1f} ML/day")

# Step 2: Apply routing manually with calibrated parameters
router_B = NonlinearMuskingumRouter(K=cal_K, m=cal_m, n_subreaches=cal_n)
sim_with_routing = router_B.route(sim_direct, dt=1.0)
print(f"Routed flow (after routing): peak = {np.max(sim_with_routing):.1f} ML/day")

# Verify routing effect
routing_effect = np.max(np.abs(sim_direct - sim_with_routing))
print(f"Max routing effect: {routing_effect:.1f} ML/day")
if routing_effect < 0.1:
    print("WARNING: Routing effect is negligible!")

# Calculate metrics (comprehensive evaluation)
from pyrrm.objectives import NSE, KGE, RMSE, PBIAS, FlowTransformation

# Create metric objects
nse_metric = NSE()
kge_metric = KGE()
rmse_metric = RMSE()
pbias_metric = PBIAS()
lognse_metric = NSE(transform=FlowTransformation('log'))  # Log-transformed NSE for low flows

# Standard metrics
nse_A = nse_metric(real_observed[REAL_WARMUP:], sim_no_routing[REAL_WARMUP:])
nse_B = nse_metric(real_observed[REAL_WARMUP:], sim_with_routing[REAL_WARMUP:])
kge_A = kge_metric(real_observed[REAL_WARMUP:], sim_no_routing[REAL_WARMUP:])
kge_B = kge_metric(real_observed[REAL_WARMUP:], sim_with_routing[REAL_WARMUP:])
rmse_A = rmse_metric(real_observed[REAL_WARMUP:], sim_no_routing[REAL_WARMUP:])
rmse_B = rmse_metric(real_observed[REAL_WARMUP:], sim_with_routing[REAL_WARMUP:])

# Low flow metrics (LogNSE)
lognse_A = lognse_metric(real_observed[REAL_WARMUP:], sim_no_routing[REAL_WARMUP:])
lognse_B = lognse_metric(real_observed[REAL_WARMUP:], sim_with_routing[REAL_WARMUP:])

# Percent bias
pbias_A = pbias_metric(real_observed[REAL_WARMUP:], sim_no_routing[REAL_WARMUP:])
pbias_B = pbias_metric(real_observed[REAL_WARMUP:], sim_with_routing[REAL_WARMUP:])

# Routing parameters already extracted above

# Print comparison
print("\n" + "=" * 70)
print("RESULTS COMPARISON: WITH vs WITHOUT ROUTING")
print("=" * 70)
print(f"\n{'Metric':<20} {'Without Routing':>18} {'With Routing':>18} {'Improvement':>15}")
print("-" * 75)
print(f"{'NSE':<20} {nse_A:>18.4f} {nse_B:>18.4f} {nse_B - nse_A:>+15.4f}")
print(f"{'KGE':<20} {kge_A:>18.4f} {kge_B:>18.4f} {kge_B - kge_A:>+15.4f}")
print(f"{'LogNSE (low flows)':<20} {lognse_A:>18.4f} {lognse_B:>18.4f} {lognse_B - lognse_A:>+15.4f}")
print(f"{'PBIAS (%)':<20} {pbias_A:>18.2f} {pbias_B:>18.2f} {abs(pbias_B) - abs(pbias_A):>+15.2f}")
print(f"{'RMSE (ML/day)':<20} {rmse_A:>18.2f} {rmse_B:>18.2f} {rmse_B - rmse_A:>+15.2f}")

print(f"\n{'Calibrated Routing Parameters:'}")
print(f"  K (storage constant): {cal_K:.2f} days")
print(f"  m (nonlinearity):     {cal_m:.3f}")
print(f"  n_subreaches:         {cal_n}")

# Routing effect statistics
routing_diff = sim_direct - sim_with_routing
max_attenuation = np.max(routing_diff)
peak_direct = np.max(sim_direct)
peak_routed = np.max(sim_with_routing)

print(f"\n{'Routing Effect:'}")
print(f"  Peak direct flow:     {peak_direct:.1f} ML/day")
print(f"  Peak routed flow:     {peak_routed:.1f} ML/day")
print(f"  Peak attenuation:     {(1 - peak_routed/peak_direct)*100:.1f}%")
print(f"  Max routing effect:   {max_attenuation:.1f} ML/day")

# %% [markdown]
# ### Comprehensive Comparison: With vs Without Routing

# %%
# VISUALIZATION 1: Event Hydrograph Comparison
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Find the largest peak event after warmup
peak_idx = np.argmax(real_observed[REAL_WARMUP:]) + REAL_WARMUP
event_start = max(REAL_WARMUP, peak_idx - 30)
event_end = min(len(real_data), peak_idx + 60)

dates_event = real_data.index[event_start:event_end]

print("=" * 70)
print("VISUALIZATION 1: Peak Event Hydrograph Comparison")
print("=" * 70)
print(f"Event period: {dates_event[0].date()} to {dates_event[-1].date()}")

# Create 3-panel figure for event analysis
fig1 = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        '<b>Hydrograph Comparison</b>: Observed vs Calibrated Models',
        '<b>Residuals</b>: How well does each model fit?',
        '<b>Routing Effect</b>: Flow change from channel routing (K={:.1f}d)'.format(cal_K)
    ),
    vertical_spacing=0.08,
    row_heights=[0.45, 0.30, 0.25]
)

# Panel 1: Hydrographs
fig1.add_trace(go.Scatter(
    x=dates_event, y=real_observed[event_start:event_end],
    mode='lines+markers', name='Observed',
    line=dict(color='black', width=2.5),
    marker=dict(size=6, symbol='circle')
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=dates_event, y=sim_no_routing[event_start:event_end],
    mode='lines', name=f'Without Routing (NSE={nse_A:.3f})',
    line=dict(color='blue', width=2, dash='dash')
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=dates_event, y=sim_with_routing[event_start:event_end],
    mode='lines', name=f'With Routing (NSE={nse_B:.3f})',
    line=dict(color='red', width=2.5)
), row=1, col=1)

# Panel 2: Residuals comparison
residual_no_routing = real_observed[event_start:event_end] - sim_no_routing[event_start:event_end]
residual_with_routing = real_observed[event_start:event_end] - sim_with_routing[event_start:event_end]

fig1.add_trace(go.Bar(
    x=dates_event, y=residual_no_routing,
    name='Residual (No Routing)',
    marker_color='blue', opacity=0.5
), row=2, col=1)

fig1.add_trace(go.Bar(
    x=dates_event, y=residual_with_routing,
    name='Residual (With Routing)',
    marker_color='red', opacity=0.5
), row=2, col=1)

fig1.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)

# Panel 3: Routing effect (direct vs routed from the WITH routing calibration)
routing_effect = sim_direct[event_start:event_end] - sim_with_routing[event_start:event_end]
fig1.add_trace(go.Scatter(
    x=dates_event, y=routing_effect,
    mode='lines', name='Routing Effect',
    fill='tozeroy',
    line=dict(color='purple', width=2),
    fillcolor='rgba(128, 0, 128, 0.4)',
    showlegend=True
), row=3, col=1)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

# Update layout
fig1.update_layout(
    height=900,
    width=1200,
    title_text=f"<b>Effect of Routing on Calibration</b><br><sup>Catchment 410734 | Routing: K={cal_K:.1f}d, m={cal_m:.2f}</sup>",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    barmode='group'
)

fig1.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
fig1.update_yaxes(title_text="Residual (ML/day)", row=2, col=1)
fig1.update_yaxes(title_text="Attenuation (ML/day)", row=3, col=1)

fig1.show()

# Print event statistics
print(f"\nPeak Event Statistics:")
print(f"  Observed peak:        {np.max(real_observed[event_start:event_end]):.1f} ML/day")
print(f"  Sim (no routing):     {np.max(sim_no_routing[event_start:event_end]):.1f} ML/day")
print(f"  Sim (with routing):   {np.max(sim_with_routing[event_start:event_end]):.1f} ML/day")
print(f"\n  RMSE (no routing):    {np.sqrt(np.mean(residual_no_routing**2)):.1f} ML/day")
print(f"  RMSE (with routing):  {np.sqrt(np.mean(residual_with_routing**2)):.1f} ML/day")

# %%
# VISUALIZATION 2: Full Time Series and Scatter Plot
print("\n" + "=" * 70)
print("VISUALIZATION 2: Full Period Comparison")
print("=" * 70)

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '<b>Full Time Series</b>: Observed vs With Routing',
        '<b>Scatter Plot</b>: Model Performance',
        '<b>Flow Duration Curve</b>',
        '<b>Residual Distribution</b>'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

dates_full = real_data.index[REAL_WARMUP:]
obs_valid = real_observed[REAL_WARMUP:]
sim_A = sim_no_routing[REAL_WARMUP:]
sim_B = sim_with_routing[REAL_WARMUP:]

# Panel 1: Time series
fig2.add_trace(go.Scatter(
    x=dates_full, y=obs_valid,
    mode='lines', name='Observed',
    line=dict(color='black', width=0.8), opacity=0.7
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=dates_full, y=sim_B,
    mode='lines', name='With Routing',
    line=dict(color='red', width=0.8), opacity=0.7
), row=1, col=1)

# Panel 2: Scatter plot with both models
fig2.add_trace(go.Scatter(
    x=obs_valid, y=sim_A,
    mode='markers', name='Without Routing',
    marker=dict(color='blue', size=4, opacity=0.3),
), row=1, col=2)

fig2.add_trace(go.Scatter(
    x=obs_valid, y=sim_B,
    mode='markers', name='With Routing',
    marker=dict(color='red', size=4, opacity=0.3),
), row=1, col=2)

# 1:1 line
max_flow = max(np.max(obs_valid), np.max(sim_A), np.max(sim_B))
fig2.add_trace(go.Scatter(
    x=[0, max_flow], y=[0, max_flow],
    mode='lines', name='1:1 Line',
    line=dict(color='black', dash='dash', width=1),
    showlegend=False
), row=1, col=2)

# Panel 3: Flow Duration Curve
def compute_fdc(flows):
    sorted_flows = np.sort(flows)[::-1]
    exceedance = np.arange(1, len(sorted_flows) + 1) / len(sorted_flows) * 100
    return exceedance, sorted_flows

exc_obs, fdc_obs = compute_fdc(obs_valid)
exc_A, fdc_A = compute_fdc(sim_A)
exc_B, fdc_B = compute_fdc(sim_B)

fig2.add_trace(go.Scatter(
    x=exc_obs, y=fdc_obs,
    mode='lines', name='Observed',
    line=dict(color='black', width=2),
    showlegend=False
), row=2, col=1)

fig2.add_trace(go.Scatter(
    x=exc_A, y=fdc_A,
    mode='lines', name='Without Routing',
    line=dict(color='blue', width=2, dash='dash'),
    showlegend=False
), row=2, col=1)

fig2.add_trace(go.Scatter(
    x=exc_B, y=fdc_B,
    mode='lines', name='With Routing',
    line=dict(color='red', width=2),
    showlegend=False
), row=2, col=1)

# Panel 4: Residual histograms
residual_A = obs_valid - sim_A
residual_B = obs_valid - sim_B

fig2.add_trace(go.Histogram(
    x=residual_A, name='Without Routing',
    marker_color='blue', opacity=0.5,
    nbinsx=50, showlegend=False
), row=2, col=2)

fig2.add_trace(go.Histogram(
    x=residual_B, name='With Routing',
    marker_color='red', opacity=0.5,
    nbinsx=50, showlegend=False
), row=2, col=2)

fig2.add_vline(x=0, line_dash="dash", line_color="black", row=2, col=2)

# Update layout
fig2.update_layout(
    height=900,
    width=1300,
    title_text="<b>Full Period Comparison: Without vs With Routing</b>",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    barmode='overlay'
)

fig2.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
fig2.update_yaxes(title_text="Simulated (ML/day)", row=1, col=2)
fig2.update_xaxes(title_text="Observed (ML/day)", row=1, col=2)
fig2.update_yaxes(title_text="Flow (ML/day)", type="log", row=2, col=1)
fig2.update_xaxes(title_text="Exceedance (%)", row=2, col=1)
fig2.update_yaxes(title_text="Count", row=2, col=2)
fig2.update_xaxes(title_text="Residual (ML/day)", row=2, col=2)

# Add metrics annotation
fig2.add_annotation(
    text=f"<b>Without Routing:</b> NSE={nse_A:.3f}, KGE={kge_A:.3f}<br>" +
         f"<b>With Routing:</b> NSE={nse_B:.3f}, KGE={kge_B:.3f}",
    xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False,
    font=dict(size=12), align="right",
    bgcolor="white", bordercolor="gray", borderwidth=1
)

fig2.show()

# %%
# Print detailed comparison
print("\n" + "=" * 70)
print("DETAILED COMPARISON: WITHOUT vs WITH ROUTING")
print("=" * 70)

print(f"\n{'Metric':<25} {'Without Routing':>15} {'With Routing':>15} {'Difference':>12} {'Better?':>10}")
print("-" * 80)

# Helper to determine which is better
def better(metric_name, val_A, val_B):
    if 'RMSE' in metric_name or 'PBIAS' in metric_name:
        return "Routing" if abs(val_B) < abs(val_A) else "No Routing"
    else:
        return "Routing" if val_B > val_A else "No Routing"

metrics = [
    ('NSE', nse_A, nse_B),
    ('KGE', kge_A, kge_B),
    ('LogNSE (low flows)', lognse_A, lognse_B),
    ('PBIAS (%)', pbias_A, pbias_B),
    ('RMSE (ML/day)', rmse_A, rmse_B),
]

for name, val_A, val_B in metrics:
    diff = val_B - val_A
    winner = better(name, val_A, val_B)
    if 'PBIAS' in name or 'RMSE' in name:
        print(f"{name:<25} {val_A:>15.2f} {val_B:>15.2f} {diff:>+12.2f} {winner:>10}")
    else:
        print(f"{name:<25} {val_A:>15.4f} {val_B:>15.4f} {diff:>+12.4f} {winner:>10}")

# Count how many metrics improved
improved = sum([
    nse_B > nse_A,
    kge_B > kge_A,
    lognse_B > lognse_A,
    abs(pbias_B) < abs(pbias_A),
    rmse_B < rmse_A
])

print(f"\nMetrics improved with routing: {improved}/5")

print(f"\n{'Routing Parameters (Calibrated):'}")
print(f"  K (storage constant): {cal_K:.2f} days")
print(f"  m (nonlinearity):     {cal_m:.3f}")
print(f"  n_subreaches:         {cal_n}")

print(f"\n{'Routing Physical Effect:'}")
print(f"  Peak attenuation:     {(1 - peak_routed/peak_direct)*100:.1f}%")
print(f"  Max flow reduction:   {np.max(routing_diff):.1f} ML/day")

# %%
# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY: DOES ROUTING HELP?")
print("=" * 70)

if nse_B > nse_A:
    print(f"\n✓ YES! Routing improved NSE by {nse_B - nse_A:.4f}")
    print(f"  Without routing: NSE = {nse_A:.4f}")
    print(f"  With routing:    NSE = {nse_B:.4f}")
    print(f"\nRouting appears beneficial for this catchment.")
    print(f"The calibrated K={cal_K:.1f} days suggests a travel time")
    print(f"from catchment centroid to gauge of approximately {cal_K:.0f} days.")
else:
    print(f"\n✗ NO. Routing did not improve NSE (change: {nse_B - nse_A:+.4f})")
    print(f"  Without routing: NSE = {nse_A:.4f}")
    print(f"  With routing:    NSE = {nse_B:.4f}")
    print(f"\nRouting may not be necessary for this catchment.")
    print("This could indicate:")
    print("  - Small catchment with negligible routing effects")
    print("  - Gauge located close to rainfall centroid")
    print("  - Already-attenuated flows in observed data")

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
# 2. **Objective Function Choice**: We used **SqrtNSE** (NSE on sqrt-transformed flows)
#    which provides balanced performance across high and low flows, similar to the
#    approach in Notebook 02.
#
# 3. **Synthetic Data Verification** (Part 5): Successfully recovered known routing
#    parameters using SqrtNSE, validating the implementation works correctly.
#
# 4. **Real Data Comparison** (Part 6): Direct comparison of calibration with and
#    without routing shows the actual benefit for gauge 410734 using custom bounds
#    from Notebook 02.
#
# ### When Routing Helps Most
#
# | Scenario | Routing Likely Helps |
# |----------|---------------------|
# | Large catchment (>100 km²) | Yes |
# | Gauge downstream of catchment | Yes |
# | Systematic timing errors in peaks | Yes |
# | Sharp, flashy hydrograph | Yes |
# | Small headwater catchment | Less likely |
# | Low flows dominate record | Less likely |
#
# ### Recommended Workflow
#
# 1. **Start simple**: Calibrate RR model without routing first (e.g., Notebook 02)
# 2. **Add routing**: If timing errors are evident, add routing and recalibrate
# 3. **Compare**: Use NSE, KGE, LogNSE and visual inspection to assess improvement
# 4. **Decide**: If improvement < 0.01 NSE, routing complexity may not be justified

# %%
print("=" * 70)
print("NOTEBOOK COMPLETE")
print("=" * 70)
print(f"\nKey Results from Part 6 (Real Data):")
print(f"  Without Routing NSE: {nse_A:.4f}")
print(f"  With Routing NSE:    {nse_B:.4f}")
print(f"  Improvement:         {nse_B - nse_A:+.4f}")
print(f"\nCalibrated Routing Parameters:")
print(f"  K = {cal_K:.2f} days (storage constant)")
print(f"  m = {cal_m:.3f} (nonlinearity)")
print(f"  n_subreaches = {cal_n}")
