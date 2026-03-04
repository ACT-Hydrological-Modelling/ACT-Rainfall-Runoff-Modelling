"""
Tests for three-step model pre-screening.
"""

import pytest
import numpy as np

from pyrrm.bma.config import BMAConfig
from pyrrm.bma.pre_screen import (
    _kge,
    _nse,
    _pbias,
    hard_threshold_filter,
    pre_screen,
    residual_correlation_clustering,
    preserve_regime_specialists,
)
from pyrrm.bma.tests.fixtures import synthetic_5yr_daily, small_config


# ═════════════════════════════════════════════════════════════════════════
# Internal metric helpers
# ═════════════════════════════════════════════════════════════════════════

class TestInternalMetrics:

    def test_nse_perfect(self):
        """NSE=1 for perfect simulation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _nse(obs, obs) == pytest.approx(1.0, abs=1e-12)

    def test_nse_mean_gives_zero(self):
        """NSE=0 when simulation equals the mean of observations."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.full_like(obs, obs.mean())
        assert _nse(obs, sim) == pytest.approx(0.0, abs=1e-12)

    def test_kge_perfect(self):
        """KGE=1 for perfect simulation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _kge(obs, obs) == pytest.approx(1.0, abs=1e-10)

    def test_kge_negative_for_bad_model(self):
        """KGE should be negative for grossly biased simulation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert _kge(obs, sim) < 0

    def test_pbias_zero_for_unbiased(self):
        """PBIAS=0 when sim equals obs."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _pbias(obs, obs) == pytest.approx(0.0, abs=1e-12)

    def test_pbias_positive_for_overestimation(self):
        """PBIAS>0 when simulation overestimates."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs * 1.1
        assert _pbias(obs, sim) > 0


# ═════════════════════════════════════════════════════════════════════════
# Step 1: Hard threshold filter
# ═════════════════════════════════════════════════════════════════════════

class TestHardThresholdFilter:

    def test_perfect_models_pass(self, synthetic_5yr_daily, small_config):
        """Models that closely match observations should pass."""
        _, F, y_obs = synthetic_5yr_daily
        names = [f"model_{i}" for i in range(F.shape[1])]
        kept = hard_threshold_filter(F, y_obs, names, small_config)
        assert len(kept) > 0

    def test_terrible_model_removed(self):
        """A grossly wrong model should be removed."""
        rng = np.random.default_rng(42)
        y_obs = rng.exponential(5, 500)
        F = np.column_stack([
            y_obs + rng.normal(0, 0.5, 500),
            rng.uniform(100, 200, 500),  # terrible
        ])
        names = ["good", "terrible"]
        cfg = BMAConfig(nse_threshold=0.0, kge_threshold=-0.41, pbias_threshold=25.0)
        kept = hard_threshold_filter(F, y_obs, names, cfg)
        assert 0 in kept
        assert 1 not in kept

    def test_all_removed_returns_empty(self):
        """If all models are bad, return empty list."""
        y_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        F = np.column_stack([np.zeros(5), np.zeros(5)])
        names = ["bad1", "bad2"]
        cfg = BMAConfig(nse_threshold=0.5)
        kept = hard_threshold_filter(F, y_obs, names, cfg)
        assert len(kept) == 0


# ═════════════════════════════════════════════════════════════════════════
# Step 2: Residual correlation clustering
# ═════════════════════════════════════════════════════════════════════════

class TestResidualCorrelationClustering:

    def test_identical_models_clustered(self):
        """Two identical models should be collapsed into one."""
        rng = np.random.default_rng(42)
        y_obs = rng.exponential(5, 500)
        m1 = y_obs + rng.normal(0, 0.5, 500)
        F = np.column_stack([m1, m1.copy(), y_obs + rng.normal(0, 2.0, 500)])
        names = ["m1", "m1_clone", "m2"]
        cfg = BMAConfig(residual_corr_threshold=0.90, cluster_method="ward")
        kept = residual_correlation_clustering(F, y_obs, names, cfg)
        assert len(kept) <= 2

    def test_single_model_preserved(self):
        """Single model should always be kept."""
        y_obs = np.array([1.0, 2.0, 3.0])
        F = y_obs[:, None] + 0.1
        cfg = BMAConfig()
        kept = residual_correlation_clustering(F, y_obs, ["m1"], cfg)
        assert kept == [0]

    def test_diverse_models_all_kept(self):
        """Well-separated models should all survive clustering."""
        rng = np.random.default_rng(42)
        T = 500
        y_obs = rng.exponential(5, T)
        F = np.column_stack([
            y_obs + rng.normal(0, 1, T),
            y_obs * 0.8 + rng.normal(10, 3, T),
            np.sqrt(y_obs) * 5 + rng.normal(0, 2, T),
        ])
        names = ["m1", "m2", "m3"]
        cfg = BMAConfig(residual_corr_threshold=0.95)
        kept = residual_correlation_clustering(F, y_obs, names, cfg)
        assert len(kept) >= 2


# ═════════════════════════════════════════════════════════════════════════
# Step 3: Regime specialist preservation
# ═════════════════════════════════════════════════════════════════════════

class TestPreserveRegimeSpecialists:

    def test_specialist_added_back(self, synthetic_5yr_daily, small_config):
        """A model best for high flows should be added back if removed."""
        _, F, y_obs = synthetic_5yr_daily
        K = F.shape[1]
        names = [f"model_{i}" for i in range(K)]
        kept_without_last = list(range(K - 1))
        result = preserve_regime_specialists(
            F, y_obs, names, kept_without_last, small_config,
        )
        assert len(result) >= len(kept_without_last)

    def test_already_kept_not_duplicated(self, synthetic_5yr_daily, small_config):
        """Models already in the kept set should not be duplicated."""
        _, F, y_obs = synthetic_5yr_daily
        K = F.shape[1]
        names = [f"model_{i}" for i in range(K)]
        kept_all = list(range(K))
        result = preserve_regime_specialists(
            F, y_obs, names, kept_all, small_config,
        )
        assert len(result) == len(set(result))


# ═════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═════════════════════════════════════════════════════════════════════════

class TestPreScreen:

    def test_pre_screen_returns_correct_shapes(self, synthetic_5yr_daily, small_config):
        """Pre-screen should return (F_screened, names, corr_matrix)."""
        _, F, y_obs = synthetic_5yr_daily
        names = [f"model_{i}" for i in range(F.shape[1])]
        F_out, names_out, corr = pre_screen(F, y_obs, names, small_config)
        assert F_out.ndim == 2
        assert F_out.shape[0] == F.shape[0]
        assert F_out.shape[1] == len(names_out)
        assert corr.shape == (len(names_out), len(names_out))

    def test_all_removed_raises(self):
        """If hard thresholds remove everything, should raise ValueError."""
        y_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        F = np.column_stack([np.zeros(5), np.zeros(5)])
        names = ["bad1", "bad2"]
        cfg = BMAConfig(nse_threshold=0.5)
        with pytest.raises(ValueError, match="All models were removed"):
            pre_screen(F, y_obs, names, cfg)
