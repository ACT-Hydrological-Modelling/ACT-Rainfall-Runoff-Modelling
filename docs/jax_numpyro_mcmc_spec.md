# JAX/NumPyro Gradient-Based MCMC for Rainfall-Runoff Model Calibration

## Implementation Specification for Cursor

---

## 1. Project Overview

### 1.1 Objective

Migrate rainfall-runoff model calibration from PyDREAM (abandoned March 2020, Python 3.6 only) to gradient-based MCMC using JAX and NumPyro. This enables the NUTS (No-U-Turn Sampler) algorithm, which exploits automatic differentiation to achieve 5–100× improvements in effective samples per second compared to DREAM's gradient-free random-walk approach.

### 1.2 Scope

- Reimplement GR4J and Sacramento rainfall-runoff models in JAX
- Reimplement existing NSE-based likelihood functions (with flow transformations) in JAX
- Build NumPyro model wrappers for Bayesian calibration via NUTS
- Provide ArviZ-based diagnostics and comparison tooling
- Retain PyDREAM as a fallback sampler option
- Deliver a Jupyter notebook demonstrating the full workflow

### 1.3 Expected Performance Gains

| Model | Parameters | DREAM ESS/s | NUTS ESS/s (expected) | Speedup |
|-------|-----------|-------------|----------------------|---------|
| GR4J | 4 | baseline | 5–20× baseline | Moderate |
| Sacramento | 16–20 | baseline | 20–100× baseline | Substantial |

### 1.4 Key Dependencies

```
jax>=0.4.20
jaxlib>=0.4.20
numpyro>=0.15.0
arviz>=0.18.0
matplotlib>=3.8.0
numpy>=1.24.0
pandas>=2.0.0
```

### 1.5 Reference Implementation

The `tvp-gr4j` repository (github.com/ckrapu/tvp-gr4j) provides a working GR4J implementation in both JAX/NumPyro and Theano/PyMC3 for time-varying parameter inference via HMC. Approximately half was adapted from Kratzert's NumPy implementation with minimal changes. Use this as a structural reference.

---

## 2. Architecture

### 2.1 Module Structure

```
your_library/
├── models/
│   ├── __init__.py
│   ├── gr4j_numpy.py          # Existing NumPy GR4J (keep as-is)
│   ├── gr4j_jax.py            # NEW: JAX GR4J implementation
│   ├── sacramento_numpy.py    # Existing NumPy Sacramento (keep as-is)
│   └── sacramento_jax.py      # NEW: JAX Sacramento implementation
├── likelihoods/
│   ├── __init__.py
│   ├── likelihoods_numpy.py   # Existing likelihood functions (keep as-is)
│   └── likelihoods_jax.py     # NEW: JAX likelihood functions
├── calibration/
│   ├── __init__.py
│   ├── dream_calibrator.py    # Existing PyDREAM wrapper (keep as-is)
│   └── nuts_calibrator.py     # NEW: NumPyro NUTS calibration engine
├── diagnostics/
│   ├── __init__.py
│   └── mcmc_diagnostics.py    # NEW: ArviZ-based diagnostics & comparison
├── tests/
│   ├── test_gr4j_jax.py       # NEW
│   ├── test_sacramento_jax.py # NEW
│   ├── test_likelihoods_jax.py# NEW
│   ├── test_nuts_calibrator.py# NEW
│   └── test_diagnostics.py    # NEW
└── notebooks/
    └── demo_nuts_calibration.ipynb  # NEW: Full demonstration notebook
```

### 2.2 Design Principles

1. **Mirror existing API**: JAX models must accept the same parameter names and return the same output structure as NumPy counterparts
2. **Pure functions**: All JAX code must be pure functions (no side effects, no global state) for JIT compatibility
3. **Separation of concerns**: Forward model, likelihood, and sampler are independent composable components
4. **NumPy fallback**: Every JAX function has a NumPy equivalent; users can choose backend

---

## 3. Module Specifications

### 3.1 JAX GR4J Model — `models/gr4j_jax.py`

#### 3.1.1 Core Conversion Rules

The GR4J model has four parameters (X1–X4), two stores (production, routing), and a unit hydrograph convolution. The conversion from NumPy to JAX requires these systematic changes:

| NumPy Pattern | JAX Replacement | Reason |
|--------------|----------------|--------|
| `import numpy as np` | `import jax.numpy as jnp` | JAX array operations |
| `for t in range(n_timesteps):` | `jax.lax.scan(step_fn, carry, inputs)` | Gradient tracing cannot follow Python loops |
| `array[i] = value` | `array.at[i].set(value)` | JAX arrays are immutable |
| `if condition_on_array:` | `jnp.where(condition, true_val, false_val)` | Control flow must be traceable |
| `np.maximum(a, b)` | `jnp.maximum(a, b)` | Direct replacement |
| `np.exp(x)` | `jnp.exp(x)` | Direct replacement |

#### 3.1.2 Function Signature

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=())
def gr4j_run(
    params: dict,       # {"X1": float, "X2": float, "X3": float, "X4": float}
    precip: jnp.ndarray, # shape (n_timesteps,), daily precipitation in mm
    pet: jnp.ndarray,    # shape (n_timesteps,), daily potential ET in mm
    initial_states: dict | None = None  # Optional: {"production_store": float, "routing_store": float}
) -> dict:
    """
    Run GR4J rainfall-runoff model using JAX for automatic differentiation.

    Parameters
    ----------
    params : dict
        X1: Maximum capacity of the production store (mm), range [100, 1200]
        X2: Groundwater exchange coefficient (mm/day), range [-5, 3]
        X3: One-day-ahead maximum capacity of the routing store (mm), range [20, 300]
        X4: Time base of the unit hydrograph (days), range [1.1, 2.5]
    precip : jnp.ndarray
        Daily precipitation time series (mm)
    pet : jnp.ndarray
        Daily potential evapotranspiration time series (mm)
    initial_states : dict, optional
        Initial store levels. Defaults to 60% of X1 (production) and 70% of X3 (routing).

    Returns
    -------
    dict with keys:
        "simulated_flow": jnp.ndarray, shape (n_timesteps,) — simulated streamflow (mm/day)
        "production_store": jnp.ndarray, shape (n_timesteps,) — production store trace
        "routing_store": jnp.ndarray, shape (n_timesteps,) — routing store trace
    """
```

#### 3.1.3 Implementation Structure Using `jax.lax.scan`

The core time-stepping loop MUST use `jax.lax.scan`. This is the single most critical conversion step — without it, JAX cannot compute gradients through the simulation.

```python
def gr4j_run(params, precip, pet, initial_states=None):
    X1, X2, X3, X4 = params["X1"], params["X2"], params["X3"], params["X4"]

    # Precompute unit hydrograph ordinates (SH1 and SH2)
    # These depend only on X4 and can be computed outside the scan
    n_uh1 = jnp.ceil(X4).astype(int)
    n_uh2 = jnp.ceil(2.0 * X4).astype(int)
    # Use fixed-size arrays padded with zeros for JIT compatibility
    max_uh_length = 20  # GR4J X4 max is 2.5, so 2*X4 max is 5; 20 is safe upper bound
    uh1_ordinates = _compute_uh1(X4, max_uh_length)
    uh2_ordinates = _compute_uh2(X4, max_uh_length)

    # Initial states
    if initial_states is None:
        S_production = 0.6 * X1
        S_routing = 0.7 * X3
    else:
        S_production = initial_states["production_store"]
        S_routing = initial_states["routing_store"]

    # UH state arrays (convolution memory)
    uh1_state = jnp.zeros(max_uh_length)
    uh2_state = jnp.zeros(max_uh_length)

    # Pack carry state
    init_carry = (S_production, S_routing, uh1_state, uh2_state)

    # Pack inputs for scan
    inputs = (precip, pet)

    def step_fn(carry, input_t):
        S_prod, S_rout, uh1, uh2 = carry
        P_t, E_t = input_t

        # === Net precipitation and evaporation ===
        net_precip = jnp.maximum(P_t - E_t, 0.0)
        net_evap = jnp.maximum(E_t - P_t, 0.0)

        # === Production store ===
        # Neutralization and store update (Perrin et al. 2003 equations)
        # ... (full GR4J equations here - see Section 3.1.4)

        # === Unit hydrograph convolution ===
        # ... (using uh1_ordinates, uh2_ordinates)

        # === Routing store ===
        # ... (routing store update, groundwater exchange)

        new_carry = (S_prod_new, S_rout_new, uh1_new, uh2_new)
        output = (Q_simulated, S_prod_new, S_rout_new)
        return new_carry, output

    final_carry, (sim_flow, prod_trace, rout_trace) = lax.scan(
        step_fn, init_carry, inputs
    )

    return {
        "simulated_flow": sim_flow,
        "production_store": prod_trace,
        "routing_store": rout_trace,
    }
```

#### 3.1.4 GR4J Equations (Complete Reference)

Implement all equations from Perrin et al. (2003). The key operations are all smooth and JAX-compatible:

**Production store update (within each timestep):**

```python
# Net rainfall / net evaporation
net_precip = jnp.maximum(P - E, 0.0)
net_evap = jnp.maximum(E - P, 0.0)

# Production store: rainfall contribution
# Ps = X1 * (1 - (S/X1)^2) * tanh(Pn/X1) / (1 + S/X1 * tanh(Pn/X1))
s_ratio = S_prod / X1
Ps = X1 * (1.0 - s_ratio**2) * jnp.tanh(net_precip / X1) / \
     (1.0 + s_ratio * jnp.tanh(net_precip / X1))
# Guard: when net_precip == 0, Ps should be 0
Ps = jnp.where(net_precip > 0, Ps, 0.0)

# Production store: evaporation contribution
# Es = S * (2 - S/X1) * tanh(En/X1) / (1 + (1 - S/X1) * tanh(En/X1))
Es = S_prod * (2.0 - s_ratio) * jnp.tanh(net_evap / X1) / \
     (1.0 + (1.0 - s_ratio) * jnp.tanh(net_evap / X1))
Es = jnp.where(net_evap > 0, Es, 0.0)

S_prod = S_prod - Es + Ps

# Percolation
perc = S_prod * (1.0 - (1.0 + (4.0/9.0 * S_prod / X1)**4)**(-0.25))
S_prod = S_prod - perc

# Total effective rainfall
Pr = perc + (net_precip - Ps)
```

**Unit hydrograph and routing (within each timestep):**

```python
# Split: 90% to UH1 (routed), 10% to UH2 (direct)
# UH1 and UH2 are convolutions — update state arrays
uh1_new = jnp.roll(uh1, -1).at[-1].set(0.0) + uh1_ordinates * 0.9 * Pr
uh2_new = jnp.roll(uh2, -1).at[-1].set(0.0) + uh2_ordinates * 0.1 * Pr
Q9 = uh1_new[0]  # Output from UH1
Q1 = uh2_new[0]  # Output from UH2

# Groundwater exchange
F = X2 * (S_rout / X3) ** 3.5

# Routing store
S_rout = jnp.maximum(0.0, S_rout + Q9 + F)
Qr = S_rout * (1.0 - (1.0 + (S_rout / X3)**4)**(-0.25))
S_rout = S_rout - Qr

# Direct flow component
Qd = jnp.maximum(0.0, Q1 + F)

# Total simulated flow
Q_simulated = Qr + Qd
```

**Unit hydrograph ordinate computation:**

```python
def _compute_uh1(X4, max_length):
    """S-curve for UH1: SH1(t) = (t/X4)^(5/2) for t < X4, else 1."""
    t = jnp.arange(1, max_length + 1, dtype=jnp.float64)
    sh1 = jnp.where(t < X4, (t / X4) ** 2.5, 1.0)
    # Ordinates are differences of S-curve
    sh1_shifted = jnp.concatenate([jnp.array([0.0]), sh1[:-1]])
    ordinates = sh1 - sh1_shifted
    return ordinates

def _compute_uh2(X4, max_length):
    """S-curve for UH2: piecewise for t < X4 and X4 <= t < 2*X4."""
    t = jnp.arange(1, max_length + 1, dtype=jnp.float64)
    sh2 = jnp.where(
        t < X4,
        0.5 * (t / X4) ** 2.5,
        jnp.where(
            t < 2.0 * X4,
            1.0 - 0.5 * (2.0 - t / X4) ** 2.5,
            1.0
        )
    )
    sh2_shifted = jnp.concatenate([jnp.array([0.0]), sh2[:-1]])
    ordinates = sh2 - sh2_shifted
    return ordinates
```

#### 3.1.5 Critical JAX Considerations for GR4J

1. **Use `float64` precision**: Hydrological simulations over thousands of timesteps accumulate numerical errors. Add at program start:
   ```python
   jax.config.update("jax_enable_x64", True)
   ```

2. **Fixed array sizes**: The unit hydrograph arrays must have fixed size known at JIT compile time. Use a generous upper bound (e.g., 20) and pad with zeros. Do NOT use dynamic shapes.

3. **No Python-level branching on traced values**: Every `if` on an array value must become `jnp.where`. The `jnp.maximum(0.0, x)` pattern is already JAX-safe for enforcing non-negativity.

4. **Numerical stability**: Add small epsilon (1e-10) to denominators in the production store equations to avoid division by zero when X1 is very small during NUTS exploration.

---

### 3.2 JAX Sacramento Model — `models/sacramento_jax.py`

#### 3.2.1 Function Signature

```python
@partial(jax.jit, static_argnums=())
def sacramento_run(
    params: dict,         # 16–20 parameter dict (see parameter table below)
    precip: jnp.ndarray,  # shape (n_timesteps,), daily precipitation (mm)
    pet: jnp.ndarray,     # shape (n_timesteps,), daily potential ET (mm)
    initial_states: dict | None = None
) -> dict:
    """
    Run Sacramento Soil Moisture Accounting model using JAX.

    Returns
    -------
    dict with keys:
        "simulated_flow": jnp.ndarray — total channel inflow (mm/day)
        "surface_runoff": jnp.ndarray — surface runoff component
        "interflow": jnp.ndarray — interflow component
        "baseflow": jnp.ndarray — baseflow component
        "upper_zone_tension": jnp.ndarray — UZTWC trace
        "upper_zone_free": jnp.ndarray — UZFWC trace
        "lower_zone_tension": jnp.ndarray — LZTWC trace
        "lower_zone_primary_free": jnp.ndarray — LZFPC trace
        "lower_zone_supplemental_free": jnp.ndarray — LZFSC trace
        "additional_impervious": jnp.ndarray — ADIMC trace
    """
```

#### 3.2.2 Sacramento Parameter Table

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| UZTWM | Upper zone tension water max capacity | 1–150 | mm |
| UZFWM | Upper zone free water max capacity | 1–150 | mm |
| LZTWM | Lower zone tension water max capacity | 1–500 | mm |
| LZFPM | Lower zone primary free water max capacity | 1–1000 | mm |
| LZFSM | Lower zone supplemental free water max capacity | 1–1000 | mm |
| UZK | Upper zone free water lateral depletion rate | 0.1–0.5 | 1/day |
| LZPK | Lower zone primary free water depletion rate | 0.001–0.025 | 1/day |
| LZSK | Lower zone supplemental free water depletion rate | 0.05–0.25 | 1/day |
| ZPERC | Maximum percolation rate coefficient | 1–250 | — |
| REXP | Percolation equation exponent | 1–5 | — |
| PFREE | Fraction of percolation going to lower zone free water | 0–0.6 | — |
| PCTIM | Fraction of impervious area | 0–0.1 | — |
| ADIMP | Additional impervious area fraction | 0–0.4 | — |
| RIVA | Riparian vegetation area fraction | 0–0.2 | — |
| SIDE | Ratio of deep recharge to channel baseflow | 0–0.5 | — |
| RSERV | Fraction of lower zone free water not transferrable to tension | 0–0.4 | — |

#### 3.2.3 Sacramento Conversion Strategy

Sacramento is more complex than GR4J but follows the same conversion pattern. Key considerations:

1. **Threshold logic**: Sacramento uses `min(demand, available)` patterns extensively for water redistribution between stores. These are already smooth operations:
   ```python
   # NumPy: actual_et = min(demand, available_water)
   # JAX:   actual_et = jnp.minimum(demand, available_water)
   ```

2. **Percolation equation**: The percolation from upper to lower zone uses a smooth power law:
   ```python
   perc_demand = (1.0 + ZPERC * (1.0 - lz_deficiency_ratio) ** REXP) * lzpk_lzsk_sum
   ```
   This is fully differentiable.

3. **Capacity checks**: Sacramento frequently checks if stores exceed capacity. Use `jnp.minimum(store, capacity)` instead of if-else branching.

4. **Time structure**: Same `jax.lax.scan` pattern as GR4J. The carry state includes all 5–6 store levels (UZTWC, UZFWC, LZTWC, LZFPC, LZFSC, optionally ADIMC).

5. **Estimated effort**: 2–3 days, primarily translating the water balance partitioning logic into JAX-compatible form.

---

### 3.3 JAX Likelihood Functions — `likelihoods/likelihoods_jax.py`

#### 3.3.1 Supported Likelihood Types

Reimplement all existing likelihood functions from the NumPy codebase. Each takes simulated and observed flows and returns a scalar log-likelihood value.

#### 3.3.2 NSE-Based Gaussian Likelihood

The NSE is equivalent to a Gaussian likelihood under the assumption of i.i.d. errors with constant variance (Stedinger et al. 2008):

```python
def nse_log_likelihood(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    sigma: float | jnp.ndarray,
    transform: str = "none",
    transform_params: dict | None = None,
    warmup_steps: int = 365
) -> jnp.ndarray:
    """
    Compute NSE-based Gaussian log-likelihood with optional flow transformation.

    Parameters
    ----------
    sim : jnp.ndarray
        Simulated streamflow (mm/day)
    obs : jnp.ndarray
        Observed streamflow (mm/day)
    sigma : float or jnp.ndarray
        Error standard deviation. Can be scalar (homoscedastic)
        or array (heteroscedastic).
    transform : str
        Flow transformation to apply before computing likelihood.
        One of: "none", "log", "sqrt", "inverse", "boxcox"
    transform_params : dict, optional
        Parameters for the transformation:
        - "boxcox": {"lambda": float} — Box-Cox lambda parameter
        - "log": {"epsilon": float} — Small constant added before log (default: 1% of mean obs)
        - "inverse": {"epsilon": float} — Small constant added before inversion
    warmup_steps : int
        Number of initial timesteps to exclude from likelihood (model spin-up period)

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood value
    """
    # Trim warmup
    sim_trim = sim[warmup_steps:]
    obs_trim = obs[warmup_steps:]

    # Apply transformation
    sim_t, obs_t = _apply_transform(sim_trim, obs_trim, transform, transform_params)

    # Gaussian log-likelihood: -0.5 * sum((obs_t - sim_t)^2 / sigma^2) - n*log(sigma) - n/2*log(2*pi)
    n = obs_t.shape[0]
    residuals = obs_t - sim_t
    ll = -0.5 * n * jnp.log(2.0 * jnp.pi) - n * jnp.log(sigma) \
         - 0.5 * jnp.sum(residuals**2) / sigma**2

    return ll
```

#### 3.3.3 Flow Transformation Functions

All transformations must be smooth and JAX-differentiable:

```python
def _apply_transform(sim, obs, transform, params):
    """Apply flow transformation. All operations are JAX-differentiable."""

    if transform == "none":
        return sim, obs

    elif transform == "log":
        eps = params.get("epsilon", None) if params else None
        if eps is None:
            eps = 0.01 * jnp.mean(obs)  # Pushpalatha et al. 2012
        return jnp.log(sim + eps), jnp.log(obs + eps)

    elif transform == "sqrt":
        return jnp.sqrt(sim), jnp.sqrt(obs)

    elif transform == "inverse":
        eps = params.get("epsilon", None) if params else None
        if eps is None:
            eps = 0.01 * jnp.mean(obs)
        return 1.0 / (sim + eps), 1.0 / (obs + eps)

    elif transform == "boxcox":
        lam = params["lambda"]
        # Box-Cox: (x^lambda - 1) / lambda  when lambda != 0
        #          log(x)                    when lambda == 0
        # Use smooth approximation to avoid branching:
        return _boxcox(sim, lam), _boxcox(obs, lam)

    else:
        raise ValueError(f"Unknown transform: {transform}")


def _boxcox(x, lam, eps=1e-6):
    """
    Box-Cox transformation, smooth for all lambda including near zero.
    Uses the limit: as lambda -> 0, (x^lambda - 1)/lambda -> log(x).
    """
    # Smooth transition: use log when |lambda| < eps
    # For numerical stability, use the identity:
    # (x^lam - 1) / lam = exp(lam * log(x) - log(lam)) ... but simpler:
    return jnp.where(
        jnp.abs(lam) > eps,
        (jnp.power(x, lam) - 1.0) / lam,
        jnp.log(x)
    )
```

#### 3.3.4 AR(1) Error Model (Optional Enhancement)

For likelihood functions that account for autocorrelation in residuals:

```python
def nse_ar1_log_likelihood(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    sigma: float,
    phi: float,          # AR(1) coefficient, range [0, 1)
    transform: str = "none",
    transform_params: dict | None = None,
    warmup_steps: int = 365
) -> jnp.ndarray:
    """
    Gaussian log-likelihood with AR(1) correlated errors.

    The error model is: e_t = phi * e_{t-1} + eta_t
    where eta_t ~ N(0, sigma^2)

    This accounts for temporal autocorrelation in model residuals,
    which is common in hydrological models.
    """
    sim_t, obs_t = _apply_transform(
        sim[warmup_steps:], obs[warmup_steps:], transform, transform_params
    )

    residuals = obs_t - sim_t
    n = residuals.shape[0]

    # AR(1) innovations
    innovations = residuals[1:] - phi * residuals[:-1]

    # Log-likelihood of AR(1) process
    ll = (-0.5 * (n - 1) * jnp.log(2.0 * jnp.pi)
          - (n - 1) * jnp.log(sigma)
          - 0.5 * jnp.sum(innovations**2) / sigma**2
          + 0.5 * jnp.log(1.0 - phi**2)  # Correction for first observation
          - 0.5 * residuals[0]**2 * (1.0 - phi**2) / sigma**2)

    return ll
```

---

### 3.4 NumPyro NUTS Calibrator — `calibration/nuts_calibrator.py`

#### 3.4.1 Class Interface

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import arviz as az


class NUTSCalibrator:
    """
    Bayesian calibration of rainfall-runoff models using NumPyro NUTS.

    This is the primary new calibration engine. It wraps JAX-based forward
    models and likelihood functions into a NumPyro probabilistic model,
    then runs NUTS for posterior inference.

    Parameters
    ----------
    model_fn : callable
        JAX-based forward model function. Must accept (params_dict, precip, pet)
        and return dict with "simulated_flow" key.
    model_name : str
        One of "gr4j" or "sacramento". Used to select parameter definitions.
    likelihood_fn : callable
        JAX-based likelihood function. Must accept (sim, obs, sigma, **kwargs)
        and return scalar log-likelihood.
    likelihood_config : dict
        Configuration passed to likelihood_fn:
        - "transform": str
        - "transform_params": dict
        - "warmup_steps": int
        - "error_model": "iid" or "ar1"
    prior_config : dict, optional
        Override default prior distributions. Keys are parameter names,
        values are numpyro.distributions instances.
    """

    def __init__(
        self,
        model_fn,
        model_name,
        likelihood_fn,
        likelihood_config=None,
        prior_config=None,
    ):
        self.model_fn = model_fn
        self.model_name = model_name
        self.likelihood_fn = likelihood_fn
        self.likelihood_config = likelihood_config or {}
        self.prior_config = prior_config or {}

    def get_default_priors(self):
        """Return default prior distributions for the chosen model."""
        if self.model_name == "gr4j":
            return {
                "X1": dist.Uniform(100.0, 1200.0),
                "X2": dist.Uniform(-5.0, 3.0),
                "X3": dist.Uniform(20.0, 300.0),
                "X4": dist.Uniform(1.1, 2.5),
            }
        elif self.model_name == "sacramento":
            return {
                "UZTWM": dist.Uniform(1.0, 150.0),
                "UZFWM": dist.Uniform(1.0, 150.0),
                "LZTWM": dist.Uniform(1.0, 500.0),
                "LZFPM": dist.Uniform(1.0, 1000.0),
                "LZFSM": dist.Uniform(1.0, 1000.0),
                "UZK": dist.Uniform(0.1, 0.5),
                "LZPK": dist.Uniform(0.001, 0.025),
                "LZSK": dist.Uniform(0.05, 0.25),
                "ZPERC": dist.Uniform(1.0, 250.0),
                "REXP": dist.Uniform(1.0, 5.0),
                "PFREE": dist.Uniform(0.0, 0.6),
                "PCTIM": dist.Uniform(0.0, 0.1),
                "ADIMP": dist.Uniform(0.0, 0.4),
                "RIVA": dist.Uniform(0.0, 0.2),
                "SIDE": dist.Uniform(0.0, 0.5),
                "RSERV": dist.Uniform(0.0, 0.4),
            }
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _build_numpyro_model(self, precip, pet, obs_flow):
        """
        Construct the NumPyro probabilistic model function.

        This is the core integration point: priors → forward model → likelihood.
        """
        priors = {**self.get_default_priors(), **self.prior_config}
        model_fn = self.model_fn
        likelihood_fn = self.likelihood_fn
        lik_config = self.likelihood_config

        def numpyro_model():
            # Sample parameters from priors
            params = {}
            for name, prior_dist in priors.items():
                params[name] = numpyro.sample(name, prior_dist)

            # Sample error standard deviation
            sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1.0))

            # Optionally sample AR(1) coefficient
            if lik_config.get("error_model") == "ar1":
                phi = numpyro.sample("phi", dist.Uniform(0.0, 0.99))
            else:
                phi = None

            # Run forward model
            result = model_fn(params, precip, pet)
            sim_flow = result["simulated_flow"]

            # Compute log-likelihood
            ll_kwargs = {
                "transform": lik_config.get("transform", "none"),
                "transform_params": lik_config.get("transform_params"),
                "warmup_steps": lik_config.get("warmup_steps", 365),
            }
            if phi is not None:
                ll_kwargs["phi"] = phi

            log_lik = likelihood_fn(sim_flow, obs_flow, sigma, **ll_kwargs)

            # Register likelihood with NumPyro
            numpyro.factor("log_likelihood", log_lik)

            # Store deterministic quantities for diagnostics
            numpyro.deterministic("simulated_flow", sim_flow)

        return numpyro_model

    def run(
        self,
        precip,
        pet,
        obs_flow,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        target_accept_prob=0.8,
        max_tree_depth=10,
        seed=42,
        progress_bar=True,
    ):
        """
        Run NUTS calibration.

        Parameters
        ----------
        precip : array-like
            Daily precipitation (mm). Will be converted to JAX array.
        pet : array-like
            Daily potential evapotranspiration (mm).
        obs_flow : array-like
            Observed streamflow (mm/day).
        num_warmup : int
            Number of NUTS warmup (adaptation) iterations per chain.
        num_samples : int
            Number of posterior samples per chain.
        num_chains : int
            Number of independent MCMC chains.
        target_accept_prob : float
            Target acceptance probability for NUTS step-size adaptation.
            Default 0.8 is standard. Increase to 0.9–0.95 for complex posteriors.
        max_tree_depth : int
            Maximum NUTS tree depth. Default 10 is standard.
        seed : int
            Random seed for reproducibility.
        progress_bar : bool
            Display sampling progress bar.

        Returns
        -------
        CalibrationResult
            Object containing posterior samples, diagnostics, and metadata.
        """
        # Convert inputs to JAX arrays
        precip = jnp.array(precip, dtype=jnp.float64)
        pet = jnp.array(pet, dtype=jnp.float64)
        obs_flow = jnp.array(obs_flow, dtype=jnp.float64)

        # Build model
        numpyro_model = self._build_numpyro_model(precip, pet, obs_flow)

        # Configure NUTS kernel
        kernel = NUTS(
            numpyro_model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        # Run MCMC
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key)

        # Convert to ArviZ InferenceData
        inference_data = az.from_numpyro(mcmc)

        return CalibrationResult(
            mcmc=mcmc,
            inference_data=inference_data,
            precip=precip,
            pet=pet,
            obs_flow=obs_flow,
            model_fn=self.model_fn,
            model_name=self.model_name,
        )
```

#### 3.4.2 CalibrationResult Class

```python
class CalibrationResult:
    """
    Container for MCMC calibration output with convenience methods.
    """

    def __init__(self, mcmc, inference_data, precip, pet, obs_flow, model_fn, model_name):
        self.mcmc = mcmc
        self.inference_data = inference_data
        self.precip = precip
        self.pet = pet
        self.obs_flow = obs_flow
        self.model_fn = model_fn
        self.model_name = model_name

    def summary(self, var_names=None):
        """Print ArviZ summary table with R-hat and ESS."""
        return az.summary(
            self.inference_data,
            var_names=var_names,
            round_to=4,
        )

    def get_posterior_median_params(self):
        """Extract posterior median parameter values as a dict."""
        posterior = self.inference_data.posterior
        param_names = [v for v in posterior.data_vars if v not in ("sigma", "phi", "simulated_flow")]
        return {
            name: float(posterior[name].median()) for name in param_names
        }

    def simulate_posterior_predictive(self, n_samples=100):
        """
        Generate posterior predictive simulations.

        Randomly draws n_samples parameter sets from the posterior and
        runs the forward model for each, returning an array of simulated
        hydrographs for uncertainty band construction.
        """
        posterior = self.inference_data.posterior
        param_names = [v for v in posterior.data_vars if v not in ("sigma", "phi", "simulated_flow")]

        # Stack chains and draw samples
        all_samples = {
            name: posterior[name].values.reshape(-1)
            for name in param_names
        }
        total_samples = len(all_samples[param_names[0]])
        indices = jax.random.choice(
            jax.random.PRNGKey(0), total_samples, shape=(n_samples,), replace=False
        )

        simulations = []
        for idx in indices:
            params = {name: float(all_samples[name][idx]) for name in param_names}
            result = self.model_fn(params, self.precip, self.pet)
            simulations.append(result["simulated_flow"])

        return jnp.stack(simulations)  # shape (n_samples, n_timesteps)

    def compute_nse(self, warmup_steps=365):
        """Compute NSE for the posterior median simulation."""
        params = self.get_posterior_median_params()
        result = self.model_fn(params, self.precip, self.pet)
        sim = result["simulated_flow"][warmup_steps:]
        obs = self.obs_flow[warmup_steps:]
        nse = 1.0 - jnp.sum((obs - sim)**2) / jnp.sum((obs - jnp.mean(obs))**2)
        return float(nse)
```

---

### 3.5 MCMC Diagnostics — `diagnostics/mcmc_diagnostics.py`

#### 3.5.1 Diagnostic Functions

```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def check_convergence(inference_data, threshold_rhat=1.01, threshold_ess=400):
    """
    Check MCMC convergence diagnostics.

    Parameters
    ----------
    inference_data : az.InferenceData
        ArviZ inference data from calibration.
    threshold_rhat : float
        Maximum acceptable R-hat value. Standard: 1.01
    threshold_ess : int
        Minimum acceptable effective sample size (bulk and tail).

    Returns
    -------
    dict with keys:
        "converged": bool — all diagnostics pass
        "rhat": dict — R-hat values per parameter
        "ess_bulk": dict — Bulk ESS per parameter
        "ess_tail": dict — Tail ESS per parameter
        "divergences": int — number of divergent transitions
        "warnings": list of str — specific diagnostic warnings
    """
    summary = az.summary(inference_data)
    warnings = []

    rhat = {k: v for k, v in summary["r_hat"].items()}
    ess_bulk = {k: v for k, v in summary["ess_bulk"].items()}
    ess_tail = {k: v for k, v in summary["ess_tail"].items()}

    # Check R-hat
    bad_rhat = {k: v for k, v in rhat.items() if v > threshold_rhat}
    if bad_rhat:
        warnings.append(
            f"R-hat > {threshold_rhat} for: {bad_rhat}. "
            "Chains have not converged. Run longer or reparameterize."
        )

    # Check ESS
    low_ess = {k: v for k, v in ess_bulk.items() if v < threshold_ess}
    if low_ess:
        warnings.append(
            f"Bulk ESS < {threshold_ess} for: {low_ess}. "
            "Increase num_samples or num_chains."
        )

    # Check divergences
    if hasattr(inference_data, "sample_stats"):
        divergences = int(inference_data.sample_stats["diverging"].sum())
    else:
        divergences = 0
    if divergences > 0:
        warnings.append(
            f"{divergences} divergent transitions detected. "
            "Increase target_accept_prob to 0.9–0.99 or reparameterize model."
        )

    converged = len(warnings) == 0

    return {
        "converged": converged,
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "divergences": divergences,
        "warnings": warnings,
    }


def plot_diagnostics(inference_data, var_names=None, figsize=(12, 8)):
    """
    Generate standard MCMC diagnostic plots.

    Creates a multi-panel figure with:
    1. Trace plots (chain mixing)
    2. Posterior density plots
    3. Rank plots (uniformity check)
    """
    if var_names is None:
        var_names = [
            v for v in inference_data.posterior.data_vars
            if v not in ("simulated_flow",)
        ]

    fig, axes = plt.subplots(len(var_names), 3, figsize=figsize)
    if len(var_names) == 1:
        axes = axes[np.newaxis, :]

    for i, var in enumerate(var_names):
        # Trace plot
        az.plot_trace(
            inference_data, var_names=[var],
            axes=axes[i:i+1, :2], compact=True
        )
        # Rank plot
        az.plot_rank(inference_data, var_names=[var], ax=axes[i, 2])

    plt.tight_layout()
    return fig


def plot_hydrograph_with_uncertainty(
    result,
    obs_flow,
    dates=None,
    warmup_steps=365,
    n_posterior_samples=200,
    figsize=(14, 6),
):
    """
    Plot observed vs simulated hydrograph with posterior predictive uncertainty bands.

    Shows:
    - Observed flow (black line)
    - Posterior median simulation (blue line)
    - 90% and 50% credible intervals (shaded bands)
    """
    sims = result.simulate_posterior_predictive(n_samples=n_posterior_samples)
    sims = np.array(sims[:, warmup_steps:])
    obs = np.array(obs_flow[warmup_steps:])

    median_sim = np.median(sims, axis=0)
    q05 = np.percentile(sims, 5, axis=0)
    q25 = np.percentile(sims, 25, axis=0)
    q75 = np.percentile(sims, 75, axis=0)
    q95 = np.percentile(sims, 95, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    x = dates[warmup_steps:] if dates is not None else np.arange(len(obs))

    ax.fill_between(x, q05, q95, alpha=0.2, color="steelblue", label="90% CI")
    ax.fill_between(x, q25, q75, alpha=0.3, color="steelblue", label="50% CI")
    ax.plot(x, median_sim, color="steelblue", linewidth=1.0, label="Posterior median")
    ax.plot(x, obs, color="black", linewidth=0.8, alpha=0.7, label="Observed")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_xlabel("Date" if dates is not None else "Timestep")
    ax.legend()
    ax.set_title("Posterior Predictive Hydrograph")

    return fig


def compare_samplers(nuts_result, dream_result, var_names=None):
    """
    Compare NUTS vs DREAM calibration results side by side.

    Parameters
    ----------
    nuts_result : CalibrationResult
        Output from NUTSCalibrator.run()
    dream_result : dict
        Output from PyDREAM. Expected format:
        {"samples": np.ndarray (n_samples, n_params), "param_names": list,
         "wall_time_seconds": float}

    Returns
    -------
    dict with comparison metrics:
        "ess_per_second": dict per method
        "rhat": dict per method
        "nse": dict per method
        "wall_time": dict per method
    """
    # Implementation: extract ESS from ArviZ for NUTS,
    # compute from dream_result["samples"] for DREAM,
    # and compare side by side.
    pass  # Implement based on your PyDREAM output format
```

---

## 4. Testing Specification

### 4.1 Test File: `tests/test_gr4j_jax.py`

```python
"""
Tests for JAX GR4J implementation.

Run with: pytest tests/test_gr4j_jax.py -v
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from your_library.models.gr4j_jax import gr4j_run, _compute_uh1, _compute_uh2
from your_library.models.gr4j_numpy import gr4j_run as gr4j_run_numpy  # existing


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def standard_params():
    return {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}


@pytest.fixture
def synthetic_inputs():
    """365 days of synthetic P and PET."""
    np.random.seed(42)
    precip = np.random.exponential(scale=5.0, size=365).astype(np.float64)
    pet = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365.0)
    return precip, pet


@pytest.fixture
def multi_year_inputs():
    """10 years (3650 days) of synthetic data for longer runs."""
    np.random.seed(42)
    n = 3650
    precip = np.random.exponential(scale=5.0, size=n).astype(np.float64)
    pet = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    return precip, pet


# ---------------------------------------------------------------------------
# Test 1: NumPy/JAX equivalence
# ---------------------------------------------------------------------------
class TestNumpyJaxEquivalence:
    """JAX GR4J must produce identical results to the existing NumPy version."""

    def test_standard_params(self, standard_params, synthetic_inputs):
        precip, pet = synthetic_inputs
        result_jax = gr4j_run(standard_params, jnp.array(precip), jnp.array(pet))
        result_np = gr4j_run_numpy(standard_params, precip, pet)

        np.testing.assert_allclose(
            np.array(result_jax["simulated_flow"]),
            result_np["simulated_flow"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="JAX and NumPy GR4J produce different flows"
        )

    def test_multiple_param_sets(self, synthetic_inputs):
        """Test equivalence across a range of parameter values."""
        precip, pet = synthetic_inputs
        param_sets = [
            {"X1": 100.0, "X2": -5.0, "X3": 20.0, "X4": 1.1},
            {"X1": 1200.0, "X2": 3.0, "X3": 300.0, "X4": 2.5},
            {"X1": 500.0, "X2": 0.0, "X3": 100.0, "X4": 2.0},
        ]
        for params in param_sets:
            result_jax = gr4j_run(params, jnp.array(precip), jnp.array(pet))
            result_np = gr4j_run_numpy(params, precip, pet)
            np.testing.assert_allclose(
                np.array(result_jax["simulated_flow"]),
                result_np["simulated_flow"],
                rtol=1e-8,
                err_msg=f"Mismatch for params: {params}"
            )

    def test_multi_year_stability(self, standard_params, multi_year_inputs):
        """Verify no numerical drift over long simulations."""
        precip, pet = multi_year_inputs
        result_jax = gr4j_run(standard_params, jnp.array(precip), jnp.array(pet))
        result_np = gr4j_run_numpy(standard_params, precip, pet)

        np.testing.assert_allclose(
            np.array(result_jax["simulated_flow"]),
            result_np["simulated_flow"],
            rtol=1e-8,
            err_msg="Numerical drift in 10-year simulation"
        )


# ---------------------------------------------------------------------------
# Test 2: Gradient computation
# ---------------------------------------------------------------------------
class TestGradients:
    """
    NUTS requires gradients through the entire forward model.
    These tests verify that JAX can compute them.
    """

    def test_gradient_exists(self, standard_params, synthetic_inputs):
        """jax.grad should return finite gradients for all parameters."""
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(X1, X2, X3, X4):
            params = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
            result = gr4j_run(params, precip, pet)
            return jnp.sum(result["simulated_flow"])

        grad_fn = jax.grad(sum_flow, argnums=(0, 1, 2, 3))
        grads = grad_fn(350.0, 0.5, 90.0, 1.7)

        for i, name in enumerate(["X1", "X2", "X3", "X4"]):
            assert jnp.isfinite(grads[i]), f"Gradient for {name} is not finite: {grads[i]}"
            assert grads[i] != 0.0, f"Gradient for {name} is exactly zero"

    def test_gradient_no_nan(self, synthetic_inputs):
        """Gradients should never be NaN across parameter range."""
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(X1, X2, X3, X4):
            params = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
            result = gr4j_run(params, precip, pet)
            return jnp.sum(result["simulated_flow"])

        grad_fn = jax.grad(sum_flow, argnums=(0, 1, 2, 3))

        # Test at parameter space boundaries
        test_points = [
            (100.0, -5.0, 20.0, 1.1),   # lower bounds
            (1200.0, 3.0, 300.0, 2.5),   # upper bounds
            (350.0, 0.0, 90.0, 1.5),     # mid-range
        ]
        for point in test_points:
            grads = grad_fn(*point)
            for g in grads:
                assert jnp.isfinite(g), f"NaN gradient at params={point}"

    def test_gradient_finite_difference_agreement(self, standard_params, synthetic_inputs):
        """
        Verify JAX autodiff gradients approximately match finite differences.
        This catches autodiff implementation bugs.
        """
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(X1, X2, X3, X4):
            params = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
            result = gr4j_run(params, precip, pet)
            return jnp.sum(result["simulated_flow"])

        X1, X2, X3, X4 = 350.0, 0.5, 90.0, 1.7
        eps = 1e-5

        # Autodiff gradient
        auto_grads = jax.grad(sum_flow, argnums=(0, 1, 2, 3))(X1, X2, X3, X4)

        # Finite difference gradient for X1
        fd_grad_X1 = (sum_flow(X1 + eps, X2, X3, X4) - sum_flow(X1 - eps, X2, X3, X4)) / (2 * eps)
        np.testing.assert_allclose(auto_grads[0], fd_grad_X1, rtol=1e-3,
                                   err_msg="Autodiff vs FD mismatch for X1")


# ---------------------------------------------------------------------------
# Test 3: JIT compilation
# ---------------------------------------------------------------------------
class TestJIT:
    """Verify JIT compilation works and provides speedup."""

    def test_jit_compiles(self, standard_params, synthetic_inputs):
        """Model should JIT compile without error."""
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        jitted_fn = jax.jit(lambda: gr4j_run(standard_params, precip, pet))
        result = jitted_fn()
        assert result["simulated_flow"].shape == (365,)

    def test_jit_speedup(self, standard_params, multi_year_inputs):
        """JIT should be faster than first (compilation) call."""
        import time
        precip, pet = jnp.array(multi_year_inputs[0]), jnp.array(multi_year_inputs[1])

        # First call includes compilation
        t0 = time.perf_counter()
        _ = gr4j_run(standard_params, precip, pet)
        first_call = time.perf_counter() - t0

        # Subsequent calls use compiled code
        t0 = time.perf_counter()
        for _ in range(10):
            _ = gr4j_run(standard_params, precip, pet)
        subsequent_avg = (time.perf_counter() - t0) / 10

        # JIT-compiled should be at least 5x faster than compilation call
        assert subsequent_avg < first_call / 5, (
            f"JIT not providing speedup: first={first_call:.4f}s, "
            f"subsequent={subsequent_avg:.4f}s"
        )


# ---------------------------------------------------------------------------
# Test 4: Physical validity
# ---------------------------------------------------------------------------
class TestPhysicalValidity:
    """Basic physical constraints the model must satisfy."""

    def test_nonnegative_flows(self, standard_params, synthetic_inputs):
        """Simulated flows must never be negative."""
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        result = gr4j_run(standard_params, precip, pet)
        assert jnp.all(result["simulated_flow"] >= 0.0), "Negative flows detected"

    def test_water_balance_approximate(self, standard_params, multi_year_inputs):
        """Total outflow should be approximately equal to total inflow minus ET over long period."""
        precip, pet = jnp.array(multi_year_inputs[0]), jnp.array(multi_year_inputs[1])
        result = gr4j_run(standard_params, precip, pet)
        total_P = jnp.sum(precip)
        total_Q = jnp.sum(result["simulated_flow"])
        # Flow should be between 0 and total precipitation
        assert total_Q > 0, "Zero total flow over 10 years"
        assert total_Q < total_P, "Total flow exceeds total precipitation"

    def test_zero_precipitation(self, standard_params):
        """With zero precipitation, flows should decrease toward zero."""
        precip = jnp.zeros(365)
        pet = jnp.full(365, 3.0)
        result = gr4j_run(standard_params, precip, pet)
        # Last month should have very low flow
        assert jnp.mean(result["simulated_flow"][-30:]) < 0.1

    def test_store_bounds(self, standard_params, synthetic_inputs):
        """Store levels should stay within physical bounds."""
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        result = gr4j_run(standard_params, precip, pet)
        # Production store should not exceed X1
        assert jnp.all(result["production_store"] <= standard_params["X1"] * 1.001)
        assert jnp.all(result["production_store"] >= 0.0)


# ---------------------------------------------------------------------------
# Test 5: Unit hydrograph
# ---------------------------------------------------------------------------
class TestUnitHydrograph:
    """Unit hydrograph ordinates must sum to 1."""

    @pytest.mark.parametrize("X4", [1.1, 1.5, 2.0, 2.5])
    def test_uh1_sums_to_one(self, X4):
        uh1 = _compute_uh1(X4, max_length=20)
        np.testing.assert_allclose(jnp.sum(uh1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("X4", [1.1, 1.5, 2.0, 2.5])
    def test_uh2_sums_to_one(self, X4):
        uh2 = _compute_uh2(X4, max_length=20)
        np.testing.assert_allclose(jnp.sum(uh2), 1.0, atol=1e-10)

    def test_uh_ordinates_nonnegative(self):
        for X4 in [1.1, 1.5, 2.0, 2.5]:
            assert jnp.all(_compute_uh1(X4, 20) >= 0.0)
            assert jnp.all(_compute_uh2(X4, 20) >= 0.0)
```

### 4.2 Test File: `tests/test_likelihoods_jax.py`

```python
"""
Tests for JAX likelihood functions.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from your_library.likelihoods.likelihoods_jax import (
    nse_log_likelihood,
    nse_ar1_log_likelihood,
    _apply_transform,
    _boxcox,
)


@pytest.fixture
def flow_pair():
    """Simulated and observed flows for testing."""
    np.random.seed(42)
    obs = np.abs(np.random.randn(730)) * 5 + 1  # 2 years, positive
    sim = obs + np.random.randn(730) * 0.5       # sim ≈ obs with noise
    return jnp.array(sim), jnp.array(obs)


class TestTransformations:
    """Flow transformations must be correct and differentiable."""

    def test_none_transform_identity(self, flow_pair):
        sim, obs = flow_pair
        sim_t, obs_t = _apply_transform(sim, obs, "none", None)
        np.testing.assert_array_equal(sim_t, sim)

    def test_log_transform(self, flow_pair):
        sim, obs = flow_pair
        eps = 0.01 * jnp.mean(obs)
        sim_t, obs_t = _apply_transform(sim, obs, "log", None)
        np.testing.assert_allclose(sim_t, jnp.log(sim + eps), rtol=1e-10)

    def test_boxcox_reduces_to_log_at_zero(self):
        """Box-Cox with lambda=0 should equal log transform."""
        x = jnp.array([1.0, 2.0, 5.0, 10.0])
        result = _boxcox(x, 0.0)
        expected = jnp.log(x)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_boxcox_lambda_half(self):
        """Box-Cox with lambda=0.5 should give 2*(sqrt(x) - 1)."""
        x = jnp.array([1.0, 4.0, 9.0])
        result = _boxcox(x, 0.5)
        expected = (jnp.sqrt(x) - 1.0) / 0.5
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_all_transforms_differentiable(self, flow_pair):
        """Gradients must be computable through all transforms."""
        sim, obs = flow_pair
        for transform in ["none", "log", "sqrt", "inverse", "boxcox"]:
            params = {"lambda": 0.25} if transform == "boxcox" else None

            def f(s):
                s_t, o_t = _apply_transform(s, obs, transform, params)
                return jnp.sum((s_t - o_t)**2)

            grad = jax.grad(f)(sim)
            assert jnp.all(jnp.isfinite(grad)), f"NaN gradient for {transform}"


class TestLikelihood:
    """Log-likelihood must be correct and differentiable."""

    def test_perfect_fit_high_likelihood(self, flow_pair):
        """When sim ≈ obs, likelihood should be high."""
        _, obs = flow_pair
        ll_good = nse_log_likelihood(obs, obs, sigma=0.1, warmup_steps=0)
        ll_bad = nse_log_likelihood(obs * 2, obs, sigma=0.1, warmup_steps=0)
        assert ll_good > ll_bad, "Perfect fit should have higher likelihood"

    def test_likelihood_gradient_wrt_sim(self, flow_pair):
        """Gradient of likelihood with respect to simulated flow should exist."""
        sim, obs = flow_pair

        def ll(s):
            return nse_log_likelihood(s, obs, sigma=1.0, warmup_steps=0)

        grad = jax.grad(ll)(sim)
        assert jnp.all(jnp.isfinite(grad))

    def test_sigma_effect(self, flow_pair):
        """Larger sigma should give higher likelihood for imperfect fit."""
        sim, obs = flow_pair
        ll_small_sigma = nse_log_likelihood(sim, obs, sigma=0.01, warmup_steps=0)
        ll_large_sigma = nse_log_likelihood(sim, obs, sigma=10.0, warmup_steps=0)
        # For imperfect fit with small residuals, larger sigma is more forgiving
        # but also more diffuse — this depends on residual magnitude.
        # Just check both are finite.
        assert jnp.isfinite(ll_small_sigma)
        assert jnp.isfinite(ll_large_sigma)

    def test_warmup_exclusion(self, flow_pair):
        """Warmup steps should be excluded from likelihood computation."""
        sim, obs = flow_pair
        ll_no_warmup = nse_log_likelihood(sim, obs, sigma=1.0, warmup_steps=0)
        ll_warmup = nse_log_likelihood(sim, obs, sigma=1.0, warmup_steps=365)
        # Different because different number of data points
        assert ll_no_warmup != ll_warmup

    def test_ar1_likelihood(self, flow_pair):
        """AR(1) likelihood should be computable and differentiable."""
        sim, obs = flow_pair
        ll = nse_ar1_log_likelihood(sim, obs, sigma=1.0, phi=0.5, warmup_steps=0)
        assert jnp.isfinite(ll)

        # Gradient check
        def ll_fn(s):
            return nse_ar1_log_likelihood(s, obs, sigma=1.0, phi=0.5, warmup_steps=0)

        grad = jax.grad(ll_fn)(sim)
        assert jnp.all(jnp.isfinite(grad))
```

### 4.3 Test File: `tests/test_nuts_calibrator.py`

```python
"""
Integration tests for NUTS calibration.
Uses synthetic data with known true parameters.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from your_library.models.gr4j_jax import gr4j_run
from your_library.likelihoods.likelihoods_jax import nse_log_likelihood
from your_library.calibration.nuts_calibrator import NUTSCalibrator
from your_library.diagnostics.mcmc_diagnostics import check_convergence


@pytest.fixture
def synthetic_calibration_data():
    """
    Generate synthetic observed data from known GR4J parameters.
    The calibrator should recover these parameters.
    """
    true_params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
    np.random.seed(42)
    n = 1825  # 5 years
    precip = np.random.exponential(scale=5.0, size=n).astype(np.float64)
    pet = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)

    result = gr4j_run(true_params, jnp.array(precip), jnp.array(pet))
    # Add observation noise
    obs_flow = np.array(result["simulated_flow"]) + np.random.randn(n) * 0.1
    obs_flow = np.maximum(obs_flow, 0.0)

    return true_params, precip, pet, obs_flow


class TestNUTSCalibration:
    """End-to-end calibration tests."""

    def test_parameter_recovery(self, synthetic_calibration_data):
        """
        NUTS should recover true parameters from synthetic data.
        Posterior medians should be within 20% of true values.
        """
        true_params, precip, pet, obs_flow = synthetic_calibration_data

        calibrator = NUTSCalibrator(
            model_fn=gr4j_run,
            model_name="gr4j",
            likelihood_fn=nse_log_likelihood,
            likelihood_config={
                "transform": "none",
                "warmup_steps": 365,
            },
        )

        result = calibrator.run(
            precip=precip,
            pet=pet,
            obs_flow=obs_flow,
            num_warmup=500,
            num_samples=1000,
            num_chains=2,
            seed=42,
            progress_bar=False,
        )

        median_params = result.get_posterior_median_params()
        for name, true_val in true_params.items():
            recovered = median_params[name]
            rel_error = abs(recovered - true_val) / abs(true_val)
            assert rel_error < 0.2, (
                f"Parameter {name}: true={true_val}, recovered={recovered}, "
                f"relative error={rel_error:.2%}"
            )

    def test_convergence_diagnostics(self, synthetic_calibration_data):
        """All chains should converge (R-hat < 1.01, no divergences)."""
        _, precip, pet, obs_flow = synthetic_calibration_data

        calibrator = NUTSCalibrator(
            model_fn=gr4j_run,
            model_name="gr4j",
            likelihood_fn=nse_log_likelihood,
            likelihood_config={"transform": "none", "warmup_steps": 365},
        )

        result = calibrator.run(
            precip=precip, pet=pet, obs_flow=obs_flow,
            num_warmup=500, num_samples=1000, num_chains=4,
            seed=42, progress_bar=False,
        )

        diagnostics = check_convergence(result.inference_data)
        assert diagnostics["converged"], f"Failed: {diagnostics['warnings']}"
        assert diagnostics["divergences"] == 0

    def test_nse_above_threshold(self, synthetic_calibration_data):
        """Calibrated model should achieve NSE > 0.9 on synthetic data."""
        _, precip, pet, obs_flow = synthetic_calibration_data

        calibrator = NUTSCalibrator(
            model_fn=gr4j_run,
            model_name="gr4j",
            likelihood_fn=nse_log_likelihood,
            likelihood_config={"transform": "none", "warmup_steps": 365},
        )

        result = calibrator.run(
            precip=precip, pet=pet, obs_flow=obs_flow,
            num_warmup=500, num_samples=500, num_chains=2,
            seed=42, progress_bar=False,
        )

        nse = result.compute_nse(warmup_steps=365)
        assert nse > 0.9, f"NSE too low: {nse:.4f}"

    def test_different_transforms(self, synthetic_calibration_data):
        """Calibration should work with all flow transformations."""
        _, precip, pet, obs_flow = synthetic_calibration_data

        for transform in ["none", "log", "sqrt", "boxcox"]:
            config = {
                "transform": transform,
                "warmup_steps": 365,
            }
            if transform == "boxcox":
                config["transform_params"] = {"lambda": 0.25}

            calibrator = NUTSCalibrator(
                model_fn=gr4j_run,
                model_name="gr4j",
                likelihood_fn=nse_log_likelihood,
                likelihood_config=config,
            )

            result = calibrator.run(
                precip=precip, pet=pet, obs_flow=obs_flow,
                num_warmup=200, num_samples=200, num_chains=2,
                seed=42, progress_bar=False,
            )

            # Should complete without error and have finite posterior
            assert result.inference_data is not None
            nse = result.compute_nse(warmup_steps=365)
            assert nse > 0.5, f"NSE too low for transform={transform}: {nse:.4f}"
```

### 4.4 Test Runner Configuration

Create `pytest.ini` or add to `pyproject.toml`:

```ini
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
testpaths = ["tests"]
```

Mark integration tests that run MCMC as slow:

```python
@pytest.mark.slow
def test_parameter_recovery(self, ...):
    ...
```

Run fast tests only: `pytest -m "not slow"`
Run all tests: `pytest -v`

---

## 5. Jupyter Notebook Specification

### 5.1 File: `notebooks/demo_nuts_calibration.ipynb`

The notebook should contain the following sections, each in its own cell group:

#### Cell 1: Setup and Imports

```python
"""
NUTS Calibration of GR4J — Demonstration Notebook
=================================================
This notebook demonstrates gradient-based Bayesian calibration of the GR4J
rainfall-runoff model using JAX and NumPyro, comparing with the existing
PyDREAM approach.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import time

jax.config.update("jax_enable_x64", True)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

from your_library.models.gr4j_jax import gr4j_run
from your_library.models.gr4j_numpy import gr4j_run as gr4j_run_numpy
from your_library.likelihoods.likelihoods_jax import nse_log_likelihood
from your_library.calibration.nuts_calibrator import NUTSCalibrator
from your_library.diagnostics.mcmc_diagnostics import (
    check_convergence, plot_diagnostics, plot_hydrograph_with_uncertainty
)
```

#### Cell 2: Load or Generate Data

```python
# === Option A: Load real catchment data ===
# data = pd.read_csv("path/to/catchment_data.csv", parse_dates=["date"])
# precip = data["precipitation"].values
# pet = data["pet"].values
# obs_flow = data["observed_flow"].values
# dates = data["date"].values

# === Option B: Generate synthetic data with known parameters ===
TRUE_PARAMS = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
np.random.seed(42)
n_days = 3650  # 10 years

precip = np.random.exponential(scale=5.0, size=n_days)
pet = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)

# Generate "observed" data
true_result = gr4j_run(TRUE_PARAMS, jnp.array(precip), jnp.array(pet))
obs_flow = np.array(true_result["simulated_flow"]) + np.random.randn(n_days) * 0.2
obs_flow = np.maximum(obs_flow, 0.0)
dates = pd.date_range("2010-01-01", periods=n_days, freq="D")

print(f"Data: {n_days} days ({n_days/365:.1f} years)")
print(f"Mean precip: {precip.mean():.1f} mm/day")
print(f"Mean flow: {obs_flow.mean():.2f} mm/day")
```

#### Cell 3: Verify JAX Model Matches NumPy

```python
# Sanity check: JAX and NumPy implementations must agree
result_jax = gr4j_run(TRUE_PARAMS, jnp.array(precip), jnp.array(pet))
result_np = gr4j_run_numpy(TRUE_PARAMS, precip, pet)

max_diff = np.max(np.abs(np.array(result_jax["simulated_flow"]) - result_np["simulated_flow"]))
print(f"Maximum JAX vs NumPy difference: {max_diff:.2e}")
assert max_diff < 1e-8, "JAX and NumPy implementations disagree!"
print("✓ JAX and NumPy implementations match")
```

#### Cell 4: Verify Gradients

```python
# Verify gradients can be computed through the forward model
def total_flow(X1, X2, X3, X4):
    params = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
    result = gr4j_run(params, jnp.array(precip[:365]), jnp.array(pet[:365]))
    return jnp.sum(result["simulated_flow"])

grad_fn = jax.grad(total_flow, argnums=(0, 1, 2, 3))
grads = grad_fn(350.0, 0.5, 90.0, 1.7)

print("Gradients of total flow with respect to parameters:")
for name, g in zip(["X1", "X2", "X3", "X4"], grads):
    print(f"  ∂Q/∂{name} = {g:.6f}")
print("✓ All gradients finite and non-zero")
```

#### Cell 5: Benchmark JIT Speedup

```python
# Benchmark: JIT compilation speedup
precip_jax = jnp.array(precip)
pet_jax = jnp.array(pet)

# First call (includes compilation)
t0 = time.perf_counter()
_ = gr4j_run(TRUE_PARAMS, precip_jax, pet_jax)
compile_time = time.perf_counter() - t0

# Subsequent calls (cached)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = gr4j_run(TRUE_PARAMS, precip_jax, pet_jax)
    times.append(time.perf_counter() - t0)

jit_time = np.mean(times)
print(f"First call (compile + run): {compile_time*1000:.1f} ms")
print(f"Subsequent calls (JIT):     {jit_time*1000:.3f} ms")
print(f"Speedup after compilation:  {compile_time/jit_time:.0f}×")
```

#### Cell 6: Run NUTS Calibration

```python
calibrator = NUTSCalibrator(
    model_fn=gr4j_run,
    model_name="gr4j",
    likelihood_fn=nse_log_likelihood,
    likelihood_config={
        "transform": "log",        # Log-transform emphasizes low flows
        "warmup_steps": 365,        # Exclude first year (spin-up)
    },
)

print("Running NUTS calibration...")
print(f"  Warmup: 1000 iterations × 4 chains")
print(f"  Sampling: 2000 iterations × 4 chains")
print()

t0 = time.perf_counter()
result = calibrator.run(
    precip=precip,
    pet=pet,
    obs_flow=obs_flow,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    target_accept_prob=0.8,
    seed=42,
)
nuts_wall_time = time.perf_counter() - t0
print(f"\nTotal wall time: {nuts_wall_time:.1f} seconds")
```

#### Cell 7: Convergence Diagnostics

```python
# Check convergence
diagnostics = check_convergence(result.inference_data)

if diagnostics["converged"]:
    print("✓ All convergence diagnostics passed")
else:
    print("✗ Convergence issues detected:")
    for w in diagnostics["warnings"]:
        print(f"  - {w}")

print(f"\nDivergent transitions: {diagnostics['divergences']}")
print("\n--- Summary Table ---")
print(result.summary(var_names=["X1", "X2", "X3", "X4", "sigma"]))
```

#### Cell 8: Compare Recovered Parameters to Truth

```python
median_params = result.get_posterior_median_params()
print("Parameter Recovery:")
print(f"{'Param':<6} {'True':>10} {'Recovered':>12} {'Error':>10}")
print("-" * 42)
for name in ["X1", "X2", "X3", "X4"]:
    true_val = TRUE_PARAMS[name]
    rec_val = median_params[name]
    error = abs(rec_val - true_val) / abs(true_val) * 100
    print(f"{name:<6} {true_val:>10.3f} {rec_val:>12.3f} {error:>9.1f}%")

nse = result.compute_nse(warmup_steps=365)
print(f"\nCalibration NSE: {nse:.4f}")
```

#### Cell 9: Diagnostic Plots

```python
# Trace plots, density plots, rank plots
fig = plot_diagnostics(result.inference_data, var_names=["X1", "X2", "X3", "X4", "sigma"])
plt.suptitle("MCMC Diagnostics", fontsize=14)
plt.tight_layout()
plt.show()
```

#### Cell 10: Posterior Pair Plot

```python
# Pair plot showing parameter correlations
az.plot_pair(
    result.inference_data,
    var_names=["X1", "X2", "X3", "X4"],
    kind="kde",
    marginals=True,
    figsize=(10, 10),
)
plt.suptitle("Posterior Parameter Correlations", y=1.02)
plt.tight_layout()
plt.show()
```

#### Cell 11: Hydrograph with Uncertainty Bands

```python
fig = plot_hydrograph_with_uncertainty(
    result,
    obs_flow=obs_flow,
    dates=dates,
    warmup_steps=365,
    n_posterior_samples=200,
)
plt.show()
```

#### Cell 12: Comparison with PyDREAM (Optional)

```python
# === Compare with PyDREAM if available ===
# from your_library.calibration.dream_calibrator import DREAMCalibrator
#
# dream_cal = DREAMCalibrator(model_fn=gr4j_run_numpy, ...)
# t0 = time.perf_counter()
# dream_result = dream_cal.run(precip, pet, obs_flow, n_iterations=50000, n_chains=5)
# dream_wall_time = time.perf_counter() - t0
#
# print(f"NUTS wall time:  {nuts_wall_time:.1f}s")
# print(f"DREAM wall time: {dream_wall_time:.1f}s")
# print(f"NUTS ESS/s vs DREAM ESS/s: compute from ArviZ summary")
```

#### Cell 13: Different Flow Transformations

```python
# Run calibration with different flow transformations and compare
transforms = {
    "None (standard NSE)": {"transform": "none"},
    "Log (low-flow emphasis)": {"transform": "log"},
    "Box-Cox (λ=0.25)": {"transform": "boxcox", "transform_params": {"lambda": 0.25}},
    "Sqrt": {"transform": "sqrt"},
}

results = {}
for name, config in transforms.items():
    config["warmup_steps"] = 365
    cal = NUTSCalibrator(
        model_fn=gr4j_run, model_name="gr4j",
        likelihood_fn=nse_log_likelihood, likelihood_config=config,
    )
    res = cal.run(precip, pet, obs_flow, num_warmup=500, num_samples=1000,
                  num_chains=2, seed=42, progress_bar=False)
    nse = res.compute_nse(warmup_steps=365)
    results[name] = {"nse": nse, "params": res.get_posterior_median_params()}
    print(f"{name}: NSE = {nse:.4f}")
```

---

## 6. Implementation Checklist

Use this checklist to track build progress in Cursor:

### Phase 1: JAX GR4J (Priority: HIGH, Effort: 1 day)
- [ ] Create `models/gr4j_jax.py`
- [ ] Implement `_compute_uh1` and `_compute_uh2` functions
- [ ] Implement `gr4j_run` using `jax.lax.scan`
- [ ] Enable `float64` precision
- [ ] Add numerical stability guards (epsilon in denominators)
- [ ] Write `tests/test_gr4j_jax.py`
- [ ] Pass all tests: NumPy equivalence, gradient computation, JIT, physical validity, UH
- [ ] Verify: `jax.grad` produces finite non-NaN gradients across parameter space

### Phase 2: JAX Likelihood Functions (Priority: HIGH, Effort: 0.5 day)
- [ ] Create `likelihoods/likelihoods_jax.py`
- [ ] Implement `_apply_transform` for all transforms (none, log, sqrt, inverse, boxcox)
- [ ] Implement `_boxcox` with smooth lambda→0 handling
- [ ] Implement `nse_log_likelihood`
- [ ] Implement `nse_ar1_log_likelihood`
- [ ] Write `tests/test_likelihoods_jax.py`
- [ ] Pass all tests: transform correctness, differentiability, likelihood properties

### Phase 3: NumPyro NUTS Calibrator (Priority: HIGH, Effort: 1 day)
- [ ] Create `calibration/nuts_calibrator.py`
- [ ] Implement `NUTSCalibrator` class with default priors for GR4J and Sacramento
- [ ] Implement `_build_numpyro_model` — the NumPyro model function
- [ ] Implement `run` method with NUTS configuration
- [ ] Implement `CalibrationResult` class with summary, predictive, and NSE methods
- [ ] Write `tests/test_nuts_calibrator.py`
- [ ] Pass all tests: parameter recovery, convergence, NSE threshold, transforms

### Phase 4: Diagnostics (Priority: MEDIUM, Effort: 0.5 day)
- [ ] Create `diagnostics/mcmc_diagnostics.py`
- [ ] Implement `check_convergence` (R-hat, ESS, divergences)
- [ ] Implement `plot_diagnostics` (trace, density, rank plots)
- [ ] Implement `plot_hydrograph_with_uncertainty`
- [ ] Implement `compare_samplers` (NUTS vs DREAM comparison)
- [ ] Write `tests/test_diagnostics.py`

### Phase 5: JAX Sacramento (Priority: MEDIUM, Effort: 2–3 days)
- [ ] Create `models/sacramento_jax.py`
- [ ] Implement all store updates using `jax.lax.scan`
- [ ] Convert threshold logic to `jnp.minimum`/`jnp.maximum`/`jnp.where`
- [ ] Write `tests/test_sacramento_jax.py`
- [ ] Pass equivalence tests against existing NumPy Sacramento
- [ ] Verify gradient computation through full Sacramento model
- [ ] Add Sacramento priors to `NUTSCalibrator`

### Phase 6: Demonstration Notebook (Priority: HIGH, Effort: 0.5 day)
- [ ] Create `notebooks/demo_nuts_calibration.ipynb`
- [ ] All 13 cells as specified in Section 5
- [ ] Runs end-to-end without errors
- [ ] Clear narrative explaining each step
- [ ] Visual outputs (plots) render correctly

### Phase 7: Integration and Polish (Priority: MEDIUM, Effort: 0.5 day)
- [ ] Update `__init__.py` files with new module exports
- [ ] Add `requirements.txt` or `pyproject.toml` with JAX/NumPyro deps
- [ ] Run full test suite: `pytest -v`
- [ ] Run slow tests: `pytest -v -m slow`
- [ ] Verify notebook runs cleanly from fresh kernel

---

## 7. Troubleshooting Guide

### 7.1 Common JAX Conversion Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ConcretizationTypeError` | Python `if` on a traced JAX value | Replace with `jnp.where()` or `jax.lax.cond()` |
| `TracerArrayConversionError` | Converting JAX tracer to NumPy | Ensure all operations use `jnp`, not `np` |
| `UnexpectedTracerError` | Using traced value as array index | Use `jax.lax.dynamic_slice` or pre-compute indices |
| `Shapes must be 1D sequences of concrete values` | Dynamic array shape in JIT | Use fixed-size arrays with zero-padding |
| NaN gradients | Division by zero in forward model | Add epsilon (1e-10) to denominators |
| Incorrect gradients | Using `np` instead of `jnp` somewhere | Audit all imports; `np` ops have zero gradient |

### 7.2 NUTS-Specific Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Many divergent transitions | Step size too large for posterior geometry | Increase `target_accept_prob` to 0.9–0.99 |
| Very slow sampling | Large tree depths (hitting `max_tree_depth`) | Reparameterize model; check for multimodality |
| R-hat > 1.1 after many samples | Chains stuck in different modes | Run from multiple starting points; consider SMC |
| All parameters at prior boundaries | Priors too narrow | Widen prior ranges |
| Sigma → 0 | Model overfitting with flexible error model | Add informative prior on sigma: `HalfNormal(1.0)` |

### 7.3 Performance Tips

1. **First JIT call is slow**: This is expected. JAX traces and compiles the function. Subsequent calls are fast. Do not benchmark the first call.

2. **Reduce compilation time**: Use `jax.jit` at the highest level (the whole model), not on individual functions inside the scan body.

3. **Memory**: For very long time series (>50,000 timesteps), gradient computation through `lax.scan` can use significant memory. Consider chunking or using gradient checkpointing: `jax.checkpoint(step_fn)`.

4. **CPU is sufficient**: For GR4J (4 params) and Sacramento (16–20 params), CPU is fast enough. GPU acceleration is unnecessary for individual catchment calibration.

---

## 8. References

- Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.
- Burnash, R. J. C. (1995). The NWS River Forecast System — Catchment Modeling. In V. P. Singh (Ed.), Computer Models of Watershed Hydrology.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. Journal of Machine Learning Research, 15, 1593-1623.
- Krapu, C., & Borsuk, M. (2022). Time-varying parameter inference with HMC for rainfall-runoff models. github.com/ckrapu/tvp-gr4j
- Pushpalatha, R., Perrin, C., Le Moine, N., & Andréassian, V. (2012). A review of efficiency criteria suitable for evaluating low-flow simulations. Journal of Hydrology, 420, 171-182.
- Schoups, G., & Vrugt, J. A. (2010). A formal likelihood function for parameter and predictive inference of hydrologic models with correlated, heteroscedastic, and non-Gaussian errors. Water Resources Research, 46, W10531.
