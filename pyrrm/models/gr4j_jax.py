"""
GR4J Rainfall-Runoff Model -- JAX Implementation.

Pure-function port of _gr4j_core() from pyrrm.models.gr4j for use with
JAX automatic differentiation and NumPyro NUTS sampling.  Every equation
and constant is identical to the NumPy version; only the control-flow
and array mutation patterns are replaced with JAX equivalents.

Reference:
    Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a
    parsimonious model for streamflow simulation. Journal of Hydrology,
    279(1-4), 275-289.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from pyrrm.models.utils.s_curves_jax import (
    compute_uh1_ordinates_jax,
    compute_uh2_ordinates_jax,
)

jax.config.update("jax_enable_x64", True)

_MAX_UH1_SIZE = 20
_MAX_UH2_SIZE = 40
_STORAGE_FRACTION = 0.9
_EPS = 1e-10


def gr4j_run_jax(
    params: dict,
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    production_store_init: float = None,
    routing_store_init: float = None,
) -> dict:
    """
    Run GR4J using JAX (JIT-able, differentiable via jax.grad).

    Args:
        params: ``{"X1": float, "X2": float, "X3": float, "X4": float}``
        precip: Daily precipitation [mm], shape ``(n_timesteps,)``.
        pet: Daily potential evapotranspiration [mm], shape ``(n_timesteps,)``.
        production_store_init: Initial production store [mm].
            Default ``0.3 * X1`` (matches GR4J class).
        routing_store_init: Initial routing store [mm].
            Default ``0.5 * X3`` (matches GR4J class).

    Returns:
        Dict with keys ``"simulated_flow"``, ``"production_store"``,
        ``"routing_store"`` -- each a ``jnp.ndarray`` of shape
        ``(n_timesteps,)``.
    """
    x1 = params["X1"]
    x2 = params["X2"]
    x3 = params["X3"]
    x4 = params["X4"]

    if production_store_init is None:
        prod_store_0 = 0.3 * x1
    else:
        prod_store_0 = production_store_init

    if routing_store_init is None:
        rout_store_0 = 0.5 * x3
    else:
        rout_store_0 = routing_store_init

    o_uh1 = compute_uh1_ordinates_jax(x4, max_length=_MAX_UH1_SIZE)
    o_uh2 = compute_uh2_ordinates_jax(x4, max_length=_MAX_UH2_SIZE)

    uh1_0 = jnp.zeros(_MAX_UH1_SIZE)
    uh2_0 = jnp.zeros(_MAX_UH2_SIZE)

    init_carry = (prod_store_0, rout_store_0, uh1_0, uh2_0)
    inputs = jnp.stack([precip, pet], axis=-1)  # (n_timesteps, 2)

    step_fn = partial(
        _gr4j_step, x1=x1, x2=x2, x3=x3, o_uh1=o_uh1, o_uh2=o_uh2
    )

    final_carry, outputs = lax.scan(step_fn, init_carry, inputs)
    sim_flow, prod_trace, rout_trace = outputs

    return {
        "simulated_flow": sim_flow,
        "production_store": prod_trace,
        "routing_store": rout_trace,
    }


def _gr4j_step(carry, input_t, x1, x2, x3, o_uh1, o_uh2):
    """Single GR4J timestep -- mirrors _gr4j_core inner loop."""
    prod_store, rout_store, uh1, uh2 = carry
    rain = input_t[0]
    evap = input_t[1]

    psf = prod_store / (x1 + _EPS)

    # --- Net evaporation branch (rain <= evap) ---
    net_evap = evap - rain
    snr_dry = jnp.tanh(jnp.minimum(net_evap / (x1 + _EPS), 13.0))
    denom_dry = 1.0 + (1.0 - psf) * snr_dry
    prod_evap = prod_store * (2.0 - psf) * snr_dry / (denom_dry + _EPS)
    prod_store_dry = prod_store - prod_evap
    rout_input_dry = 0.0

    # --- Net precipitation branch (rain > evap) ---
    net_precip = rain - evap
    snr_wet = jnp.tanh(jnp.minimum(net_precip / (x1 + _EPS), 13.0))
    denom_wet = 1.0 + psf * snr_wet
    prod_rainfall = x1 * (1.0 - psf * psf) * snr_wet / (denom_wet + _EPS)
    prod_store_wet = prod_store + prod_rainfall
    rout_input_wet = net_precip - prod_rainfall

    # --- Select branch ---
    is_dry = rain <= evap
    prod_store = jnp.where(is_dry, prod_store_dry, prod_store_wet)
    rout_input = jnp.where(is_dry, rout_input_dry, rout_input_wet)

    prod_store = jnp.maximum(prod_store, 0.0)

    # --- Percolation ---
    psf_p4 = (prod_store / (x1 + _EPS)) ** 4.0
    percolation = prod_store * (1.0 - 1.0 / (1.0 + psf_p4 / 25.62891) ** 0.25)
    prod_store = prod_store - percolation
    rout_input = rout_input + percolation

    # --- UH convolution (shift-and-add, fixed-size) ---
    uh1 = jnp.roll(uh1, -1).at[-1].set(0.0) + o_uh1 * rout_input
    uh2 = jnp.roll(uh2, -1).at[-1].set(0.0) + o_uh2 * rout_input

    # --- Groundwater exchange ---
    gw_exchange = x2 * (rout_store / (x3 + _EPS)) ** 3.5

    # --- Routing store update ---
    rout_store = jnp.maximum(
        0.0, rout_store + uh1[0] * _STORAGE_FRACTION + gw_exchange
    )

    # --- Routing store outflow ---
    rsf_p4 = (rout_store / (x3 + _EPS)) ** 4.0
    rout_flow = rout_store * (1.0 - 1.0 / (1.0 + rsf_p4) ** 0.25)
    rout_store = rout_store - rout_flow

    # --- Direct flow ---
    direct_flow = jnp.maximum(
        0.0, uh2[0] * (1.0 - _STORAGE_FRACTION) + gw_exchange
    )

    sim_flow = rout_flow + direct_flow

    new_carry = (prod_store, rout_store, uh1, uh2)
    outputs = (sim_flow, prod_store, rout_store)
    return new_carry, outputs
