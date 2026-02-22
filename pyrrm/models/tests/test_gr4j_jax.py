"""
Tests for the JAX GR4J implementation.

Verifies that gr4j_run_jax reproduces _gr4j_core from the NumPy
implementation, that JAX gradients are finite, and that JIT
compilation succeeds.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pyrrm.models.gr4j import _gr4j_core
from pyrrm.models.gr4j_jax import gr4j_run_jax, _MAX_UH1_SIZE, _MAX_UH2_SIZE
from pyrrm.models.utils.s_curves_jax import (
    compute_uh1_ordinates_jax,
    compute_uh2_ordinates_jax,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_params():
    return {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}


@pytest.fixture
def synthetic_inputs():
    """365 days of synthetic P and PET."""
    rng = np.random.RandomState(42)
    precip = rng.exponential(scale=5.0, size=365).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365.0)).astype(np.float64)
    return precip, pet


@pytest.fixture
def multi_year_inputs():
    """10 years of synthetic data."""
    rng = np.random.RandomState(42)
    n = 3650
    precip = rng.exponential(scale=5.0, size=n).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)).astype(np.float64)
    return precip, pet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_numpy(params, precip, pet):
    """Run the NumPy _gr4j_core with default initial states."""
    x1, x2, x3, x4 = params["X1"], params["X2"], params["X3"], params["X4"]
    prod_init = 0.3 * x1
    rout_init = 0.5 * x3
    uh1_init = np.zeros(_MAX_UH1_SIZE)
    uh2_init = np.zeros(_MAX_UH2_SIZE)
    flow, _, _, _, _ = _gr4j_core(
        x1, x2, x3, x4, precip, pet, prod_init, rout_init, uh1_init, uh2_init
    )
    return flow


def _run_jax(params, precip, pet):
    """Run the JAX gr4j_run_jax."""
    result = gr4j_run_jax(params, jnp.array(precip), jnp.array(pet))
    return np.array(result["simulated_flow"])


# ---------------------------------------------------------------------------
# NumPy / JAX equivalence
# ---------------------------------------------------------------------------

class TestNumpyJaxEquivalence:

    def test_standard_params(self, standard_params, synthetic_inputs):
        precip, pet = synthetic_inputs
        flow_np = _run_numpy(standard_params, precip, pet)
        flow_jax = _run_jax(standard_params, precip, pet)
        np.testing.assert_allclose(flow_jax, flow_np, rtol=1e-8, atol=1e-12)

    def test_multiple_param_sets(self, synthetic_inputs):
        precip, pet = synthetic_inputs
        param_sets = [
            {"X1": 100.0, "X2": -5.0, "X3": 20.0, "X4": 1.1},
            {"X1": 1200.0, "X2": 3.0, "X3": 300.0, "X4": 2.5},
            {"X1": 500.0, "X2": 0.0, "X3": 100.0, "X4": 2.0},
        ]
        for params in param_sets:
            flow_np = _run_numpy(params, precip, pet)
            flow_jax = _run_jax(params, precip, pet)
            np.testing.assert_allclose(
                flow_jax, flow_np, rtol=1e-8,
                err_msg=f"Mismatch for params: {params}",
            )

    def test_multi_year_stability(self, standard_params, multi_year_inputs):
        precip, pet = multi_year_inputs
        flow_np = _run_numpy(standard_params, precip, pet)
        flow_jax = _run_jax(standard_params, precip, pet)
        np.testing.assert_allclose(flow_jax, flow_np, rtol=1e-8)


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

class TestGradients:

    def test_gradient_exists(self, standard_params, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(X1, X2, X3, X4):
            p = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
            return jnp.sum(gr4j_run_jax(p, precip, pet)["simulated_flow"])

        grad_fn = jax.grad(sum_flow, argnums=(0, 1, 2, 3))
        grads = grad_fn(350.0, 0.5, 90.0, 1.7)
        for i, name in enumerate(["X1", "X2", "X3", "X4"]):
            assert jnp.isfinite(grads[i]), f"Gradient for {name} is not finite"
            assert grads[i] != 0.0, f"Gradient for {name} is exactly zero"

    def test_no_nan_gradients(self, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])

        def sum_flow(X1, X2, X3, X4):
            p = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
            return jnp.sum(gr4j_run_jax(p, precip, pet)["simulated_flow"])

        grad_fn = jax.grad(sum_flow, argnums=(0, 1, 2, 3))
        test_points = [
            (100.0, -5.0, 20.0, 1.1),
            (1200.0, 3.0, 300.0, 2.5),
            (350.0, 0.0, 90.0, 1.5),
        ]
        for pt in test_points:
            grads = grad_fn(*pt)
            for g in grads:
                assert jnp.isfinite(g), f"NaN gradient at params={pt}"


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------

class TestJIT:

    def test_jit_compiles(self, standard_params, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        jitted = jax.jit(lambda: gr4j_run_jax(standard_params, precip, pet))
        result = jitted()
        assert result["simulated_flow"].shape == (365,)


# ---------------------------------------------------------------------------
# Physical validity
# ---------------------------------------------------------------------------

class TestPhysicalValidity:

    def test_nonnegative_flows(self, standard_params, synthetic_inputs):
        precip, pet = jnp.array(synthetic_inputs[0]), jnp.array(synthetic_inputs[1])
        result = gr4j_run_jax(standard_params, precip, pet)
        assert jnp.all(result["simulated_flow"] >= 0.0)

    def test_water_balance_approximate(self, standard_params, multi_year_inputs):
        precip, pet = jnp.array(multi_year_inputs[0]), jnp.array(multi_year_inputs[1])
        result = gr4j_run_jax(standard_params, precip, pet)
        total_q = jnp.sum(result["simulated_flow"])
        total_p = jnp.sum(precip)
        assert total_q > 0
        assert total_q < total_p

    def test_zero_precipitation(self, standard_params):
        precip = jnp.zeros(365)
        pet = jnp.full(365, 3.0)
        result = gr4j_run_jax(standard_params, precip, pet)
        assert jnp.mean(result["simulated_flow"][-30:]) < 0.1


# ---------------------------------------------------------------------------
# Unit hydrograph ordinates
# ---------------------------------------------------------------------------

class TestUnitHydrograph:

    @pytest.mark.parametrize("x4", [1.1, 1.5, 2.0, 2.5, 5.0])
    def test_uh1_sums_to_one(self, x4):
        uh1 = compute_uh1_ordinates_jax(x4, max_length=_MAX_UH1_SIZE)
        np.testing.assert_allclose(float(jnp.sum(uh1)), 1.0, atol=1e-10)

    @pytest.mark.parametrize("x4", [1.1, 1.5, 2.0, 2.5, 5.0])
    def test_uh2_sums_to_one(self, x4):
        uh2 = compute_uh2_ordinates_jax(x4, max_length=_MAX_UH2_SIZE)
        np.testing.assert_allclose(float(jnp.sum(uh2)), 1.0, atol=1e-10)

    def test_uh_ordinates_nonnegative(self):
        for x4 in [1.1, 1.5, 2.0, 2.5, 5.0]:
            assert jnp.all(compute_uh1_ordinates_jax(x4, _MAX_UH1_SIZE) >= 0.0)
            assert jnp.all(compute_uh2_ordinates_jax(x4, _MAX_UH2_SIZE) >= 0.0)
