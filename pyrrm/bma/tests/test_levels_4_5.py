"""
Tests for BMA Levels 4 and 5 (require PyMC, ArviZ, NumPyro).

These tests are skipped when PyMC is not installed (CI Tier 2).
They run in CI Tier 3 (optional-deps) or Tier 4 (JAX/PyMC).
"""

import pytest
import numpy as np

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE, reason="PyMC not installed"
)


@pytest.fixture
def small_training_data():
    """Small dataset for fast MCMC tests."""
    rng = np.random.default_rng(42)
    T = 200
    y = rng.exponential(5, T)
    F = np.column_stack([
        y + rng.normal(0, 1, T),
        y + rng.normal(0, 2, T),
        y + rng.normal(0, 3, T),
    ])
    return F, y


@pytest.fixture
def fast_config():
    """Config with minimal sampling for fast tests."""
    from pyrrm.bma.config import BMAConfig
    return BMAConfig(
        draws=100,
        tune=100,
        chains=2,
        target_accept=0.85,
        init="adapt_diag",
        nuts_sampler="pymc",
        random_seed=42,
        bias_correction="additive",
        heteroscedastic=True,
        dirichlet_alpha="sparse",
        use_manual_loglik=True,
        prediction_intervals=[0.90],
    )


# ═════════════════════════════════════════════════════════════════════════
# Level 4: Global BMA in PyMC
# ═════════════════════════════════════════════════════════════════════════

class TestBuildBMAModel:

    def test_model_compiles(self, small_training_data, fast_config):
        """BMA model should compile without errors."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        assert isinstance(model, pm.Model)

    def test_model_has_weights(self, small_training_data, fast_config):
        """Model should contain Dirichlet weight variable 'w'."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        var_names = [v.name for v in model.free_RVs]
        assert "w" in var_names

    def test_model_has_sigma(self, small_training_data, fast_config):
        """Model should contain sigma variable."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        var_names = [v.name for v in model.free_RVs]
        assert "sigma" in var_names

    def test_sparse_dirichlet(self, small_training_data, fast_config):
        """Sparse Dirichlet alpha=1/K should be used for 3 models."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        fast_config.dirichlet_alpha = "sparse"
        model = build_bma_model(F, y, fast_config)
        assert isinstance(model, pm.Model)

    def test_uniform_dirichlet(self, small_training_data, fast_config):
        """Uniform Dirichlet alpha=1 should compile."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        fast_config.dirichlet_alpha = "uniform"
        model = build_bma_model(F, y, fast_config)
        assert isinstance(model, pm.Model)

    def test_no_bias_correction(self, small_training_data, fast_config):
        """Model without bias correction should compile."""
        from pyrrm.bma.level4_bma import build_bma_model
        F, y = small_training_data
        fast_config.bias_correction = "none"
        model = build_bma_model(F, y, fast_config)
        var_names = [v.name for v in model.free_RVs]
        assert "bias" not in var_names


@pytest.mark.slow
class TestSampleBMA:

    def test_sampling_returns_idata(self, small_training_data, fast_config):
        """Sampling should return ArviZ InferenceData."""
        from pyrrm.bma.level4_bma import build_bma_model, sample_bma
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)
        assert isinstance(idata, az.InferenceData)

    def test_posterior_weights_shape(self, small_training_data, fast_config):
        """Posterior weight array should have shape (chains, draws, K)."""
        from pyrrm.bma.level4_bma import build_bma_model, sample_bma
        F, y = small_training_data
        K = F.shape[1]
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)
        w_shape = idata.posterior["w"].shape
        assert w_shape[-1] == K
        assert w_shape[0] == fast_config.chains
        assert w_shape[1] == fast_config.draws

    def test_weights_sum_to_one(self, small_training_data, fast_config):
        """Posterior weight samples should sum to ~1 per draw."""
        from pyrrm.bma.level4_bma import build_bma_model, sample_bma
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)
        w = idata.posterior["w"].values
        sums = w.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)


class TestCheckConvergence:

    def test_returns_dict(self, small_training_data, fast_config):
        """check_convergence should return a dict (possibly empty)."""
        from pyrrm.bma.level4_bma import (
            build_bma_model, check_convergence, sample_bma,
        )
        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)
        issues = check_convergence(idata)
        assert isinstance(issues, dict)


class TestExtractWeights:

    def test_returns_dataframe_with_model_names(self, small_training_data, fast_config):
        """extract_weights should return DataFrame indexed by model names."""
        from pyrrm.bma.level4_bma import (
            build_bma_model, extract_weights, sample_bma,
        )
        F, y = small_training_data
        names = ["m1", "m2", "m3"]
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)
        df = extract_weights(idata, names)
        assert list(df.index) == names
        assert "mean" in df.columns


# ═════════════════════════════════════════════════════════════════════════
# Level 5: Regime-specific BMA
# ═════════════════════════════════════════════════════════════════════════

class TestComputeRegimeBlendWeights:

    def test_weights_sum_to_one(self):
        """Blend weights should sum to 1 at every timestep."""
        from pyrrm.bma.level5_regime_bma import compute_regime_blend_weights
        flow = np.linspace(0, 100, 500)
        bw = compute_regime_blend_weights(flow, q_high=70, q_low=20, blend_width=5.0)
        total = bw["high"] + bw["medium"] + bw["low"]
        np.testing.assert_allclose(total, 1.0, atol=1e-8)

    def test_high_flow_dominated(self):
        """At very high flows, 'high' weight should dominate."""
        from pyrrm.bma.level5_regime_bma import compute_regime_blend_weights
        flow = np.array([200.0])
        bw = compute_regime_blend_weights(flow, q_high=50, q_low=10, blend_width=5.0)
        assert bw["high"][0] > 0.9

    def test_low_flow_dominated(self):
        """At very low flows, 'low' weight should dominate."""
        from pyrrm.bma.level5_regime_bma import compute_regime_blend_weights
        flow = np.array([0.1])
        bw = compute_regime_blend_weights(flow, q_high=50, q_low=10, blend_width=5.0)
        assert bw["low"][0] > 0.8

    def test_medium_flow_in_between(self):
        """At flow midpoint, medium weight should be substantial."""
        from pyrrm.bma.level5_regime_bma import compute_regime_blend_weights
        flow = np.array([30.0])
        bw = compute_regime_blend_weights(flow, q_high=50, q_low=10, blend_width=5.0)
        assert bw["medium"][0] > 0.3


class TestGenerateBMAPredictions:

    @pytest.mark.slow
    def test_predictions_have_correct_shape(self, small_training_data, fast_config):
        """Posterior predictive samples should have shape (S, T_val)."""
        from pyrrm.bma.level4_bma import build_bma_model, sample_bma
        from pyrrm.bma.prediction import generate_bma_predictions

        F, y = small_training_data
        model = build_bma_model(F, y, fast_config)
        idata = sample_bma(model, fast_config)

        T_val = 50
        F_val = F[:T_val]
        preds = generate_bma_predictions(idata, F_val, fast_config, n_samples=100)
        assert preds["samples"].shape[1] == T_val
        assert preds["mean"].shape == (T_val,)
        assert preds["median"].shape == (T_val,)
        assert 0.90 in preds["intervals"]
