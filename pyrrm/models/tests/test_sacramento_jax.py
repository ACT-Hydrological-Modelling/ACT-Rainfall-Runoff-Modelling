"""
Tests for the JAX Sacramento implementation.

Verifies that sacramento_run_jax reproduces the NumPy Sacramento model,
that JAX gradients are finite, and that JIT compilation succeeds.
"""

import pytest
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pyrrm.models.sacramento import Sacramento
from pyrrm.models.sacramento_jax import sacramento_run_jax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "uztwm": 50.0, "uzfwm": 40.0, "lztwm": 130.0,
    "lzfpm": 60.0, "lzfsm": 25.0, "uzk": 0.3,
    "lzpk": 0.01, "lzsk": 0.05, "zperc": 40.0,
    "rexp": 1.5, "pctim": 0.01, "adimp": 0.0,
    "pfree": 0.06, "rserv": 0.3, "side": 0.0,
    "ssout": 0.0, "sarva": 0.0,
    "uh1": 1.0, "uh2": 0.0, "uh3": 0.0, "uh4": 0.0, "uh5": 0.0,
}


@pytest.fixture
def synthetic_inputs():
    """365 days of synthetic P and PET."""
    rng = np.random.RandomState(42)
    precip = rng.exponential(scale=5.0, size=365).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365.0)).astype(np.float64)
    return precip, pet


@pytest.fixture
def multi_year_inputs():
    """5 years of synthetic data."""
    rng = np.random.RandomState(42)
    n = 1825
    precip = rng.exponential(scale=5.0, size=n).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)).astype(np.float64)
    return precip, pet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_numpy(params, precip, pet):
    """Run the NumPy Sacramento model with reset stores."""
    model = Sacramento()
    model.set_parameters(params)
    model.reset()
    inputs_df = pd.DataFrame({"precipitation": precip, "pet": pet})
    result = model.run(inputs_df)
    return result["runoff"].values


def _run_jax(params, precip, pet):
    """Run the JAX sacramento_run_jax."""
    result = sacramento_run_jax(params, jnp.array(precip), jnp.array(pet))
    return np.array(result["simulated_flow"])


# ---------------------------------------------------------------------------
# NumPy / JAX equivalence
# ---------------------------------------------------------------------------

class TestNumpyJaxEquivalence:

    def test_default_params(self, synthetic_inputs):
        precip, pet = synthetic_inputs
        flow_np = _run_numpy(DEFAULT_PARAMS, precip, pet)
        flow_jax = _run_jax(DEFAULT_PARAMS, precip, pet)
        np.testing.assert_allclose(
            flow_jax, flow_np, rtol=1e-4, atol=1e-8,
            err_msg="Default params mismatch",
        )

    def test_multi_year_stability(self, multi_year_inputs):
        precip, pet = multi_year_inputs
        flow_np = _run_numpy(DEFAULT_PARAMS, precip, pet)
        flow_jax = _run_jax(DEFAULT_PARAMS, precip, pet)
        np.testing.assert_allclose(
            flow_jax, flow_np, rtol=1e-4, atol=1e-8,
            err_msg="Multi-year stability mismatch",
        )


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

class TestGradients:

    def test_gradient_exists(self, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(uztwm, uzfwm, uzk):
            p = dict(DEFAULT_PARAMS)
            p["uztwm"] = uztwm
            p["uzfwm"] = uzfwm
            p["uzk"] = uzk
            return jnp.sum(sacramento_run_jax(p, precip, pet)["simulated_flow"])

        grad_fn = jax.grad(sum_flow, argnums=(0, 1, 2))
        grads = grad_fn(50.0, 40.0, 0.3)
        for i, name in enumerate(["uztwm", "uzfwm", "uzk"]):
            assert jnp.isfinite(grads[i]), f"Gradient for {name} is not finite"


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------

class TestJIT:

    def test_jit_compiles(self, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        jitted = jax.jit(
            lambda: sacramento_run_jax(DEFAULT_PARAMS, precip, pet)
        )
        result = jitted()
        assert result["simulated_flow"].shape == (365,)


# ---------------------------------------------------------------------------
# Physical validity
# ---------------------------------------------------------------------------

class TestPhysicalValidity:

    def test_nonnegative_flows(self, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        result = sacramento_run_jax(DEFAULT_PARAMS, precip, pet)
        assert jnp.all(result["simulated_flow"] >= -1e-10)

    def test_water_balance_approximate(self, multi_year_inputs):
        precip, pet = jnp.array(multi_year_inputs[0]), jnp.array(multi_year_inputs[1])
        result = sacramento_run_jax(DEFAULT_PARAMS, precip, pet)
        total_q = float(jnp.sum(result["simulated_flow"]))
        total_p = float(jnp.sum(precip))
        assert total_q > 0
        assert total_q < total_p
