"""
Tests for the JAX likelihood functions.

Verifies that JAX transforms match the existing NumPy implementations,
that likelihoods are differentiable, and that the AR(1) model behaves
correctly.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pyrrm.calibration.likelihoods_jax import (
    apply_transform_jax,
    gaussian_log_likelihood_integrated_jax,
    gaussian_log_likelihood_jax,
    transformed_gaussian_log_likelihood_jax,
    transformed_gaussian_log_likelihood_integrated_jax,
    ar1_log_likelihood_jax,
)
from pyrrm.calibration.objective_functions import (
    GaussianLikelihood,
    TransformedGaussianLikelihood,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_pair():
    """Synthetic observed / simulated flows for likelihood tests."""
    rng = np.random.RandomState(99)
    obs = 5.0 + rng.exponential(scale=3.0, size=730).astype(np.float64)
    sim = obs + rng.normal(scale=0.5, size=730).astype(np.float64)
    sim = np.maximum(sim, 0.0)
    return sim, obs


# ---------------------------------------------------------------------------
# Transform equivalence
# ---------------------------------------------------------------------------

class TestTransformEquivalence:

    @pytest.mark.parametrize("transform", ["none", "sqrt", "log", "inverse", "boxcox"])
    def test_transform_matches_numpy(self, flow_pair, transform):
        sim_np, obs_np = flow_pair
        sim_jax = jnp.array(sim_np)
        obs_jax = jnp.array(obs_np)

        jax_result = np.array(apply_transform_jax(sim_jax, obs_jax, transform))

        np_lik = TransformedGaussianLikelihood(transform=transform)
        np_result = np_lik._apply_transform(sim_np, obs_np)

        np.testing.assert_allclose(jax_result, np_result, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Gaussian likelihood equivalence
# ---------------------------------------------------------------------------

class TestGaussianLikelihood:

    def test_integrated_matches_numpy(self, flow_pair):
        sim_np, obs_np = flow_pair
        warmup = 365

        np_lik = GaussianLikelihood()
        np_val = np_lik.calculate(sim_np[warmup:], obs_np[warmup:])

        jax_val = float(gaussian_log_likelihood_integrated_jax(
            jnp.array(sim_np), jnp.array(obs_np), warmup_steps=warmup
        ))

        np.testing.assert_allclose(jax_val, np_val, rtol=1e-6)

    def test_explicit_sigma_negative(self, flow_pair):
        """Log-lik should be negative for non-trivial residuals."""
        sim, obs = flow_pair
        ll = float(gaussian_log_likelihood_jax(
            jnp.array(sim), jnp.array(obs), sigma=1.0, warmup_steps=365
        ))
        assert ll < 0.0

    def test_larger_sigma_lower_lik(self, flow_pair):
        """Wider sigma should lower the likelihood for fixed data."""
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])
        ll_narrow = gaussian_log_likelihood_jax(sim, obs, sigma=0.5, warmup_steps=365)
        ll_wide = gaussian_log_likelihood_jax(sim, obs, sigma=5.0, warmup_steps=365)
        assert float(ll_narrow) > float(ll_wide)


# ---------------------------------------------------------------------------
# Transformed Gaussian likelihood equivalence
# ---------------------------------------------------------------------------

class TestTransformedGaussianLikelihood:

    @pytest.mark.parametrize("transform", ["sqrt", "log", "boxcox"])
    def test_integrated_matches_numpy(self, flow_pair, transform):
        sim_np, obs_np = flow_pair
        warmup = 365

        np_lik = TransformedGaussianLikelihood(transform=transform)
        np_val = np_lik.calculate(sim_np[warmup:], obs_np[warmup:])

        jax_val = float(transformed_gaussian_log_likelihood_integrated_jax(
            jnp.array(sim_np), jnp.array(obs_np),
            transform=transform, warmup_steps=warmup,
        ))

        np.testing.assert_allclose(jax_val, np_val, rtol=1e-5)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

class TestDifferentiability:

    def test_gaussian_grad_wrt_sigma(self, flow_pair):
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])

        grad_fn = jax.grad(
            lambda s: gaussian_log_likelihood_jax(sim, obs, s, warmup_steps=365)
        )
        g = grad_fn(1.0)
        assert jnp.isfinite(g)

    def test_transformed_grad_wrt_sigma(self, flow_pair):
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])

        grad_fn = jax.grad(
            lambda s: transformed_gaussian_log_likelihood_jax(
                sim, obs, s, transform="sqrt", warmup_steps=365
            )
        )
        g = grad_fn(1.0)
        assert jnp.isfinite(g)

    def test_ar1_grad_wrt_phi(self, flow_pair):
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])

        grad_fn = jax.grad(
            lambda phi: ar1_log_likelihood_jax(
                sim, obs, sigma=1.0, phi=phi, warmup_steps=365
            )
        )
        g = grad_fn(0.5)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# AR(1) properties
# ---------------------------------------------------------------------------

class TestAR1:

    def test_phi_zero_close_to_gaussian(self, flow_pair):
        """AR(1) with phi=0 should be close to standard Gaussian."""
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])
        ll_gauss = float(gaussian_log_likelihood_jax(sim, obs, sigma=1.0, warmup_steps=365))
        ll_ar1 = float(ar1_log_likelihood_jax(sim, obs, sigma=1.0, phi=0.0, warmup_steps=365))
        np.testing.assert_allclose(ll_ar1, ll_gauss, rtol=1e-4)

    def test_jit_compiles(self, flow_pair):
        sim, obs = jnp.array(flow_pair[0]), jnp.array(flow_pair[1])
        fn = jax.jit(lambda: ar1_log_likelihood_jax(sim, obs, 1.0, 0.5, warmup_steps=365))
        result = fn()
        assert jnp.isfinite(result)
