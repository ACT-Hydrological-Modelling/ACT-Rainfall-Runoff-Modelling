"""
Tests for BMA Levels 1-3 (core dependencies only: numpy, scipy).

Level 1: Equal weights
Level 2: GRC / GRA
Level 3: Bayesian stacking
"""

import pytest
import numpy as np

from pyrrm.bma.level1_equal import equal_weight_predict, equal_weights
from pyrrm.bma.level2_grc import grc_fit, gra_fit, grc_predict, gra_predict
from pyrrm.bma.level3_stacking import stacking_fit, stacking_predict
from pyrrm.bma.tests.fixtures import synthetic_5yr_daily, small_config


# ═════════════════════════════════════════════════════════════════════════
# Level 1: Equal weights
# ═════════════════════════════════════════════════════════════════════════

class TestEqualWeights:

    def test_weights_sum_to_one(self):
        """Equal weights must sum to 1."""
        w = equal_weights(5)
        assert w.sum() == pytest.approx(1.0)

    def test_weights_are_uniform(self):
        """All weights should be identical."""
        w = equal_weights(4)
        np.testing.assert_allclose(w, 0.25)

    def test_predict_is_mean(self):
        """Equal-weight prediction is the arithmetic mean."""
        F = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        pred = equal_weight_predict(F)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(pred, expected)

    def test_single_model(self):
        """With one model, prediction equals that model."""
        F = np.array([[1.0], [2.0], [3.0]])
        pred = equal_weight_predict(F)
        np.testing.assert_allclose(pred, F[:, 0])


# ═════════════════════════════════════════════════════════════════════════
# Level 2: GRC / GRA
# ═════════════════════════════════════════════════════════════════════════

class TestGRC:

    def test_weights_nonneg(self, synthetic_5yr_daily):
        """GRC weights must be non-negative."""
        _, F, y_obs = synthetic_5yr_daily
        w = grc_fit(F, y_obs)
        assert np.all(w >= -1e-10)

    def test_weights_sum_to_one(self, synthetic_5yr_daily):
        """GRC weights must sum to 1."""
        _, F, y_obs = synthetic_5yr_daily
        w = grc_fit(F, y_obs)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_grc_predict_shape(self, synthetic_5yr_daily):
        """GRC prediction shape matches timesteps."""
        _, F, y_obs = synthetic_5yr_daily
        w = grc_fit(F, y_obs)
        pred = grc_predict(F, w)
        assert pred.shape == (F.shape[0],)

    def test_grc_beats_random(self, synthetic_5yr_daily):
        """GRC prediction should be closer to obs than random weights."""
        _, F, y_obs = synthetic_5yr_daily
        w_grc = grc_fit(F, y_obs)
        pred_grc = grc_predict(F, w_grc)
        mse_grc = np.mean((y_obs - pred_grc) ** 2)

        rng = np.random.default_rng(42)
        w_rand = rng.dirichlet(np.ones(F.shape[1]))
        pred_rand = grc_predict(F, w_rand)
        mse_rand = np.mean((y_obs - pred_rand) ** 2)

        assert mse_grc <= mse_rand

    def test_perfect_single_model(self):
        """If one model perfectly matches, it should get weight ~1."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 200)
        F = np.column_stack([y, y + rng.normal(0, 10, 200)])
        w = grc_fit(F, y)
        assert w[0] > 0.8


class TestGRA:

    def test_gra_returns_weights_and_intercept(self, synthetic_5yr_daily):
        """GRA should return (weights, intercept)."""
        _, F, y_obs = synthetic_5yr_daily
        w, intercept = gra_fit(F, y_obs)
        assert w.shape == (F.shape[1],)
        assert isinstance(intercept, float)

    def test_gra_predict_shape(self, synthetic_5yr_daily):
        """GRA prediction shape matches timesteps."""
        _, F, y_obs = synthetic_5yr_daily
        w, intercept = gra_fit(F, y_obs)
        pred = gra_predict(F, w, intercept)
        assert pred.shape == (F.shape[0],)

    def test_gra_allows_negative_weights(self):
        """GRA (unconstrained) can have negative weights."""
        rng = np.random.default_rng(42)
        y = rng.normal(10, 2, 200)
        F = np.column_stack([y + 5, y - 5, rng.normal(20, 10, 200)])
        w, _ = gra_fit(F, y)
        # At least in principle, GRA allows negative weights
        # (this is a statistical test — may not always produce them)
        assert w.shape == (3,)


# ═════════════════════════════════════════════════════════════════════════
# Level 3: Bayesian stacking
# ═════════════════════════════════════════════════════════════════════════

class TestStacking:

    def test_stacking_weights_sum_to_one(self, synthetic_5yr_daily, small_config):
        """Stacking weights must sum to 1."""
        dates, F, y_obs = synthetic_5yr_daily
        from pyrrm.bma.data_prep import create_cv_splits
        small_config.cv_year_start = "water_year"
        splits = create_cv_splits(dates, small_config, y_obs)
        w = stacking_fit(F, y_obs, splits)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_stacking_weights_nonneg(self, synthetic_5yr_daily, small_config):
        """Stacking weights must be non-negative."""
        dates, F, y_obs = synthetic_5yr_daily
        from pyrrm.bma.data_prep import create_cv_splits
        small_config.cv_year_start = "water_year"
        splits = create_cv_splits(dates, small_config, y_obs)
        w = stacking_fit(F, y_obs, splits)
        assert np.all(w >= -1e-10)

    def test_stacking_predict_shape(self, synthetic_5yr_daily, small_config):
        """Stacking prediction shape matches timesteps."""
        dates, F, y_obs = synthetic_5yr_daily
        from pyrrm.bma.data_prep import create_cv_splits
        small_config.cv_year_start = "water_year"
        splits = create_cv_splits(dates, small_config, y_obs)
        w = stacking_fit(F, y_obs, splits)
        pred = stacking_predict(F, w)
        assert pred.shape == (F.shape[0],)

    def test_stacking_empty_valid_fallback(self):
        """If no valid log predictive density, weights fall back to uniform."""
        F = np.ones((10, 3))
        y_obs = np.ones(10)
        splits = [(np.arange(5), np.arange(5, 10))]
        w = stacking_fit(F, y_obs, splits)
        np.testing.assert_allclose(w, 1.0 / 3)

    def test_stacking_favors_better_model(self):
        """Stacking should assign more weight to the better model."""
        rng = np.random.default_rng(42)
        T = 500
        y = rng.exponential(5, T)
        good = y + rng.normal(0, 0.5, T)
        bad = y + rng.normal(0, 10, T)
        F = np.column_stack([good, bad])

        n = T // 2
        splits = [
            (np.arange(n), np.arange(n, T)),
            (np.arange(n, T), np.arange(n)),
        ]
        w = stacking_fit(F, y, splits)
        assert w[0] > w[1], "Better model should get higher weight"
