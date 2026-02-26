"""
Integration tests for the NumPyro NUTS adapter.

Uses a short synthetic GR4J problem with known true parameters to
verify that the NUTS sampler can recover them.
"""

import pytest
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pyrrm.models.gr4j import GR4J
from pyrrm.models.gr4j_jax import gr4j_run_jax
from pyrrm.calibration.runner import CalibrationRunner
from pyrrm.calibration.numpyro_adapter import run_nuts, NUMPYRO_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed"),
    pytest.mark.slow,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRUE_PARAMS = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}


@pytest.fixture(scope="module")
def synthetic_problem():
    """
    Generate 3 years of synthetic data with known GR4J parameters,
    then add small noise to create observed flow.
    """
    rng = np.random.RandomState(42)
    n = 1095  # 3 years
    precip = rng.exponential(scale=5.0, size=n).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)).astype(np.float64)

    result = gr4j_run_jax(TRUE_PARAMS, jnp.array(precip), jnp.array(pet))
    true_flow = np.array(result["simulated_flow"])

    noise = rng.normal(scale=0.1 * np.mean(true_flow), size=n)
    obs_flow = np.maximum(true_flow + noise, 0.0)

    inputs_df = pd.DataFrame({
        "precipitation": precip,
        "pet": pet,
    })
    return inputs_df, obs_flow


# ---------------------------------------------------------------------------
# Adapter-level test
# ---------------------------------------------------------------------------

class TestRunNuts:

    def test_basic_run(self, synthetic_problem):
        """NUTS runs without error and returns expected keys."""
        inputs, obs = synthetic_problem
        model = GR4J(parameters=TRUE_PARAMS)

        result = run_nuts(
            model=model,
            inputs=inputs,
            observed=obs,
            warmup_period=365,
            num_warmup=100,
            num_samples=100,
            num_chains=1,
            seed=0,
            progress_bar=False,
            verbose=False,
        )

        assert "best_parameters" in result
        assert "all_samples" in result
        assert "inference_data" in result
        assert isinstance(result["all_samples"], pd.DataFrame)
        for name in ["X1", "X2", "X3", "X4"]:
            assert name in result["best_parameters"]

    def test_parameter_recovery(self, synthetic_problem):
        """
        With enough samples the posterior median should be close
        to the true parameters.  We use a wider tolerance since
        this is a short test run.
        """
        inputs, obs = synthetic_problem
        model = GR4J(parameters=TRUE_PARAMS)

        result = run_nuts(
            model=model,
            inputs=inputs,
            observed=obs,
            warmup_period=365,
            num_warmup=200,
            num_samples=400,
            num_chains=2,
            seed=0,
            progress_bar=False,
            verbose=False,
        )

        recovered = result["best_parameters"]
        for name, true_val in TRUE_PARAMS.items():
            assert abs(recovered[name] - true_val) / abs(true_val) < 0.5, (
                f"{name}: recovered={recovered[name]:.2f}, true={true_val:.2f}"
            )


# ---------------------------------------------------------------------------
# CalibrationRunner integration
# ---------------------------------------------------------------------------

class TestCalibrationRunnerNuts:

    def test_runner_run_nuts(self, synthetic_problem):
        """CalibrationRunner.run_nuts() returns a CalibrationResult."""
        inputs, obs = synthetic_problem
        model = GR4J(parameters=TRUE_PARAMS)

        runner = CalibrationRunner(
            model=model,
            inputs=inputs,
            observed=obs,
            warmup_period=365,
        )

        cal_result = runner.run_nuts(
            num_warmup=100,
            num_samples=100,
            num_chains=1,
            seed=1,
            progress_bar=False,
            verbose=False,
        )

        assert cal_result.method == "NUTS (NumPyro)"
        assert len(cal_result.best_parameters) >= 4
        assert cal_result.runtime_seconds > 0
        assert cal_result._raw_result is not None
        assert "inference_data" in cal_result._raw_result

    def test_runner_with_transform(self, synthetic_problem):
        """run_nuts with sqrt transform runs without error."""
        inputs, obs = synthetic_problem
        model = GR4J(parameters=TRUE_PARAMS)

        runner = CalibrationRunner(
            model=model,
            inputs=inputs,
            observed=obs,
            warmup_period=365,
        )

        cal_result = runner.run_nuts(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            transform="sqrt",
            seed=2,
            progress_bar=False,
            verbose=False,
        )

        assert cal_result.method == "NUTS (NumPyro)"
