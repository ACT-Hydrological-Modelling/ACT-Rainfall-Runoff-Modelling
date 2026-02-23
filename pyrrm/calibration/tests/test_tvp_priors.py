"""
Tests for time-varying parameter (TVP) prior specifications.
"""

import math
import pytest
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

jax.config.update("jax_enable_x64", True)

from pyrrm.calibration.tvp_priors import TVPPrior, GaussianRandomWalk


# ---------------------------------------------------------------------------
# GaussianRandomWalk dataclass
# ---------------------------------------------------------------------------

class TestGaussianRandomWalkInit:

    def test_default_fields(self):
        grw = GaussianRandomWalk()
        assert grw.lower == 0.0
        assert grw.upper == 1000.0
        assert grw.sigma_delta_scale == 3.0
        assert grw.resolution == 1
        assert grw.prefix_zero is False

    def test_custom_fields(self):
        grw = GaussianRandomWalk(
            lower=100.0, upper=1500.0,
            sigma_delta_scale=5.0, resolution=5, prefix_zero=True,
        )
        assert grw.lower == 100.0
        assert grw.upper == 1500.0
        assert grw.sigma_delta_scale == 5.0
        assert grw.resolution == 5
        assert grw.prefix_zero is True

    def test_is_tvp_prior_subclass(self):
        assert issubclass(GaussianRandomWalk, TVPPrior)


# ---------------------------------------------------------------------------
# hyperparameter_names
# ---------------------------------------------------------------------------

class TestHyperparameterNames:

    def test_returns_expected_names(self):
        grw = GaussianRandomWalk()
        names = grw.hyperparameter_names("X1")
        assert names == ["X1_intercept", "X1_sigma_delta", "X1_delta"]

    def test_names_use_parameter_name(self):
        grw = GaussianRandomWalk()
        names = grw.hyperparameter_names("theta")
        assert all(n.startswith("theta_") for n in names)


# ---------------------------------------------------------------------------
# sample_numpyro -- shape and determinism
# ---------------------------------------------------------------------------

class TestSampleNumpyro:

    @staticmethod
    def _trace_grw(grw, name, n_timesteps, seed=0):
        """Run a single forward trace and return the sampled sites."""
        def model():
            grw.sample_numpyro(name, n_timesteps)

        predictive = Predictive(model, num_samples=1)
        samples = predictive(jax.random.PRNGKey(seed))
        return samples

    def test_output_shape_resolution_1(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, resolution=1)
        samples = self._trace_grw(grw, "X1", 365)
        assert samples["X1"].shape == (1, 365)

    def test_output_shape_resolution_5(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, resolution=5)
        samples = self._trace_grw(grw, "X1", 365)
        assert samples["X1"].shape == (1, 365)

    def test_delta_shape_resolution_5(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, resolution=5)
        samples = self._trace_grw(grw, "X1", 100)
        assert samples["X1_delta"].shape == (1, 20)  # ceil(100/5) = 20

    def test_delta_shape_resolution_5_non_divisible(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, resolution=5)
        samples = self._trace_grw(grw, "X1", 103)
        assert samples["X1_delta"].shape == (1, 21)  # ceil(103/5) = 21
        assert samples["X1"].shape == (1, 103)

    def test_prefix_zero_first_delta(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, prefix_zero=True)
        samples = self._trace_grw(grw, "X1", 50)
        tvp = samples["X1"][0]
        intercept = float(samples["X1_intercept"][0])
        np.testing.assert_allclose(float(tvp[0]), intercept, atol=1e-12)

    def test_prefix_zero_delta_shape(self):
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, prefix_zero=True)
        samples = self._trace_grw(grw, "X1", 50)
        assert samples["X1_delta"].shape == (1, 49)  # n_deltas - 1

    def test_reproducibility(self):
        grw = GaussianRandomWalk(lower=0.0, upper=500.0)
        s1 = self._trace_grw(grw, "X1", 100, seed=42)
        s2 = self._trace_grw(grw, "X1", 100, seed=42)
        np.testing.assert_array_equal(np.array(s1["X1"]), np.array(s2["X1"]))

    def test_different_seeds_differ(self):
        grw = GaussianRandomWalk(lower=0.0, upper=500.0)
        s1 = self._trace_grw(grw, "X1", 100, seed=0)
        s2 = self._trace_grw(grw, "X1", 100, seed=1)
        assert not np.allclose(np.array(s1["X1"]), np.array(s2["X1"]))

    def test_intercept_within_bounds(self):
        grw = GaussianRandomWalk(lower=100.0, upper=200.0)

        def model():
            grw.sample_numpyro("X1", 50)

        predictive = Predictive(model, num_samples=200)
        samples = predictive(jax.random.PRNGKey(0))
        intercepts = np.array(samples["X1_intercept"])
        assert np.all(intercepts >= 100.0)
        assert np.all(intercepts <= 200.0)

    def test_sigma_delta_positive(self):
        grw = GaussianRandomWalk(sigma_delta_scale=5.0)

        def model():
            grw.sample_numpyro("X1", 50)

        predictive = Predictive(model, num_samples=100)
        samples = predictive(jax.random.PRNGKey(0))
        sigmas = np.array(samples["X1_sigma_delta"])
        assert np.all(sigmas > 0.0)

    def test_trajectory_is_random_walk(self):
        """Verify trajectory = intercept + cumsum(deltas)."""
        grw = GaussianRandomWalk(lower=100.0, upper=500.0, resolution=1)
        samples = self._trace_grw(grw, "X1", 30)
        tvp = np.array(samples["X1"][0])
        intercept = float(samples["X1_intercept"][0])
        deltas = np.array(samples["X1_delta"][0])
        expected = intercept + np.cumsum(deltas)
        np.testing.assert_allclose(tvp, expected, atol=1e-10)

    def test_resolution_repeats(self):
        """With resolution=5 each delta value spans 5 timesteps."""
        grw = GaussianRandomWalk(lower=0.0, upper=100.0, resolution=5)
        samples = self._trace_grw(grw, "X1", 20)
        tvp = np.array(samples["X1"][0])
        for i in range(4):
            chunk = tvp[i * 5 : (i + 1) * 5]
            np.testing.assert_allclose(
                chunk, chunk[0] * np.ones(5), atol=1e-12,
                err_msg=f"Resolution block {i} not constant",
            )
