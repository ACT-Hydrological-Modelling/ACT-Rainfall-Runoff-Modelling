"""
Tests for BMA evaluation metrics (deterministic and probabilistic).
"""

import pytest
import numpy as np

from pyrrm.bma.evaluation import (
    coverage,
    crps_ensemble,
    evaluate_by_regime,
    evaluate_deterministic,
    evaluate_probabilistic,
    fdc_error,
    interval_width,
    pit_uniformity_pvalue,
    pit_values,
)
from pyrrm.bma.tests.fixtures import make_pred_dict


# ═════════════════════════════════════════════════════════════════════════
# Deterministic
# ═════════════════════════════════════════════════════════════════════════

class TestEvaluateDeterministic:

    def test_perfect_simulation(self):
        """Perfect simulation should yield NSE=1, PBIAS=0, RMSE=0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluate_deterministic(y, y)
        assert result["NSE"] == pytest.approx(1.0, abs=1e-6)
        assert result["RMSE"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_with_standard_keys(self):
        """Should return dict with at least NSE, KGE, RMSE."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 100)
        s = y + rng.normal(0, 1, 100)
        result = evaluate_deterministic(y, s)
        assert "NSE" in result
        assert "KGE" in result
        assert "RMSE" in result


# ═════════════════════════════════════════════════════════════════════════
# CRPS
# ═════════════════════════════════════════════════════════════════════════

class TestCRPS:

    def test_crps_nonneg(self):
        """CRPS is non-negative by definition."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 50)
        samples = y[None, :] + rng.normal(0, 2, (100, 50))
        assert crps_ensemble(y, samples) >= 0

    def test_crps_zero_for_perfect(self):
        """CRPS→0 when all ensemble members equal the observation."""
        y = np.array([1.0, 2.0, 3.0])
        samples = np.tile(y, (100, 1))
        assert crps_ensemble(y, samples) == pytest.approx(0.0, abs=1e-10)

    def test_crps_increases_with_noise(self):
        """CRPS should increase when ensemble is noisier."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 50)
        samples_tight = y[None, :] + rng.normal(0, 0.5, (200, 50))
        samples_wide = y[None, :] + rng.normal(0, 10, (200, 50))
        assert crps_ensemble(y, samples_tight) < crps_ensemble(y, samples_wide)


# ═════════════════════════════════════════════════════════════════════════
# PIT
# ═════════════════════════════════════════════════════════════════════════

class TestPIT:

    def test_pit_values_in_01(self):
        """PIT values must lie in [0, 1]."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 50)
        samples = y[None, :] + rng.normal(0, 2, (200, 50))
        pit = pit_values(y, samples)
        assert np.all(pit >= 0) and np.all(pit <= 1)

    def test_pit_length_matches_obs(self):
        """PIT array length must equal number of observations."""
        y = np.array([1.0, 2.0, 3.0])
        samples = np.tile(y, (100, 1)) + 0.1
        pit = pit_values(y, samples)
        assert len(pit) == len(y)

    def test_well_calibrated_pit_is_uniform(self):
        """Samples drawn from the correct distribution → uniform PIT."""
        rng = np.random.default_rng(42)
        T = 500
        y = rng.normal(10, 2, T)
        samples = rng.normal(10, 2, (1000, T))
        pit = pit_values(y, samples)
        p = pit_uniformity_pvalue(pit)
        assert p > 0.01, f"PIT should be ~uniform; KS p-value={p:.4f}"


# ═════════════════════════════════════════════════════════════════════════
# Coverage and interval width
# ═════════════════════════════════════════════════════════════════════════

class TestCoverage:

    def test_perfect_coverage(self):
        """Wide interval should achieve ~100% coverage."""
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([10.0, 10.0, 10.0])
        assert coverage(y, lo, hi) == pytest.approx(1.0)

    def test_zero_coverage(self):
        """Interval that misses all obs → 0% coverage."""
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([10.0, 10.0, 10.0])
        hi = np.array([20.0, 20.0, 20.0])
        assert coverage(y, lo, hi) == pytest.approx(0.0)


class TestIntervalWidth:

    def test_width_positive(self):
        """Interval width should be positive."""
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([5.0, 6.0, 7.0])
        assert interval_width(lo, hi) == pytest.approx(5.0)

    def test_zero_width(self):
        """Zero-width interval (degenerate)."""
        lo = np.array([1.0, 2.0])
        assert interval_width(lo, lo) == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════
# Probabilistic evaluation wrapper
# ═════════════════════════════════════════════════════════════════════════

class TestEvaluateProbabilistic:

    def test_returns_crps_and_pit(self):
        """Should include CRPS and PIT KS p-value."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 100)
        pd = make_pred_dict(y)
        result = evaluate_probabilistic(y, pd)
        assert "CRPS" in result
        assert "PIT_KS_pvalue" in result

    def test_returns_coverage_and_width(self):
        """Should include coverage and width for each interval level."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 100)
        pd = make_pred_dict(y)
        result = evaluate_probabilistic(y, pd)
        assert "coverage_90" in result
        assert "width_90" in result
        assert "coverage_50" in result
        assert "width_50" in result


# ═════════════════════════════════════════════════════════════════════════
# Regime-specific evaluation
# ═════════════════════════════════════════════════════════════════════════

class TestEvaluateByRegime:

    def test_returns_per_regime_metrics(self):
        """Each regime should have its own metrics dict."""
        rng = np.random.default_rng(42)
        T = 300
        y = rng.exponential(5, T)
        pred = y + rng.normal(0, 1, T)
        masks = {
            "high": y > np.percentile(y, 90),
            "medium": (y >= np.percentile(y, 30)) & (y <= np.percentile(y, 90)),
            "low": y < np.percentile(y, 30),
        }
        result = evaluate_by_regime(y, pred, None, masks)
        assert "high" in result
        assert "low" in result
        assert "NSE" in result["high"]

    def test_empty_regime_skipped(self):
        """Regime with no timesteps should be skipped."""
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        masks = {
            "high": np.array([False, False, False]),
            "low": np.array([True, True, True]),
        }
        result = evaluate_by_regime(y, pred, None, masks)
        assert "high" not in result
        assert "low" in result


# ═════════════════════════════════════════════════════════════════════════
# FDC errors
# ═════════════════════════════════════════════════════════════════════════

class TestFDCError:

    def test_fdc_error_keys(self):
        """Should return one key per exceedance segment."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 500)
        pred = y + rng.normal(0, 1, 500)
        result = fdc_error(y, pred)
        assert len(result) >= 4

    def test_fdc_error_nonneg(self):
        """FDC RMSE values should be non-negative."""
        rng = np.random.default_rng(42)
        y = rng.exponential(5, 500)
        pred = y + rng.normal(0, 1, 500)
        for val in fdc_error(y, pred).values():
            assert val >= 0

    def test_perfect_fdc(self):
        """Perfect prediction → zero FDC error."""
        y = np.arange(1, 101, dtype=float)
        result = fdc_error(y, y)
        for val in result.values():
            assert val == pytest.approx(0.0, abs=1e-10)
