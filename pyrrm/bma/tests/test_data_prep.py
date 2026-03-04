"""
Tests for data loading, flow transformations, temporal CV splits,
and regime classification.
"""

import pytest
import numpy as np
import pandas as pd

from pyrrm.bma.config import BMAConfig
from pyrrm.bma.data_prep import (
    apply_transform,
    back_transform,
    classify_regime,
    create_block_cv_splits,
    create_cv_splits,
    create_expanding_window_splits,
    regime_thresholds,
    _identify_complete_years,
    _group_blocks,
    _apply_buffer,
)
from pyrrm.bma.tests.fixtures import (
    synthetic_5yr_daily,
    synthetic_10yr_daily,
    default_config,
    small_config,
)


# ═════════════════════════════════════════════════════════════════════════
# Flow transformations
# ═════════════════════════════════════════════════════════════════════════

class TestApplyTransform:
    """Test forward flow transformations."""

    def test_none_transform_is_identity(self, synthetic_5yr_daily):
        """'none' transform should return copies of input."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="none")
        F_t, y_t, params = apply_transform(F, y_obs, cfg)
        np.testing.assert_array_equal(F_t, F)
        np.testing.assert_array_equal(y_t, y_obs)
        assert params["method"] == "none"

    def test_log_transform_positive(self, synthetic_5yr_daily):
        """Log transform should produce finite values for positive flows."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="log", log_epsilon=0.01)
        F_t, y_t, params = apply_transform(F, y_obs, cfg)
        assert np.all(np.isfinite(F_t))
        assert np.all(np.isfinite(y_t))

    def test_sqrt_transform_nonneg(self, synthetic_5yr_daily):
        """Sqrt transform should return non-negative values."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="sqrt")
        F_t, y_t, _ = apply_transform(F, y_obs, cfg)
        assert np.all(F_t >= 0)
        assert np.all(y_t >= 0)

    def test_boxcox_transform_auto_lambda(self, synthetic_5yr_daily):
        """Box-Cox transform with auto lambda should be finite."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="boxcox")
        F_t, y_t, params = apply_transform(F, y_obs, cfg)
        assert np.all(np.isfinite(F_t))
        assert "lambda" in params

    def test_unknown_transform_raises(self, synthetic_5yr_daily):
        """Unknown transform name should raise ValueError."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="fourier")
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform(F, y_obs, cfg)


class TestBackTransform:
    """Test inverse transformations round-trip to original values."""

    @pytest.mark.parametrize("transform", ["none", "log", "sqrt"])
    def test_roundtrip(self, synthetic_5yr_daily, transform):
        """Forward + back transform should recover original values."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform=transform, log_epsilon=0.01)
        F_t, y_t, params = apply_transform(F, y_obs, cfg)
        F_back = back_transform(F_t, params)
        y_back = back_transform(y_t, params)
        np.testing.assert_allclose(F_back, F, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(y_back, y_obs, rtol=1e-6, atol=1e-6)

    def test_boxcox_roundtrip(self, synthetic_5yr_daily):
        """Box-Cox forward + back should recover original values."""
        _, F, y_obs = synthetic_5yr_daily
        cfg = BMAConfig(transform="boxcox")
        F_t, y_t, params = apply_transform(F, y_obs, cfg)
        F_back = back_transform(F_t, params)
        y_back = back_transform(y_t, params)
        np.testing.assert_allclose(F_back, F, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(y_back, y_obs, rtol=1e-4, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════════
# Year identification helpers
# ═════════════════════════════════════════════════════════════════════════

class TestIdentifyCompleteYears:

    def test_water_year_boundaries(self):
        """Water year identification with July start."""
        dates = pd.date_range("2010-07-01", "2013-06-30", freq="D")
        years = _identify_complete_years(dates, start_month=7)
        assert len(years) == 3
        assert years[0][0] == pd.Timestamp("2010-07-01")
        assert years[0][1] == pd.Timestamp("2011-06-30")

    def test_calendar_year_boundaries(self):
        """Calendar year identification with January start."""
        dates = pd.date_range("2010-01-01", "2014-12-31", freq="D")
        years = _identify_complete_years(dates, start_month=1)
        assert len(years) == 5

    def test_partial_years_excluded(self):
        """Partial years at start/end should be excluded."""
        dates = pd.date_range("2010-03-15", "2013-09-20", freq="D")
        years = _identify_complete_years(dates, start_month=7)
        for begin, end in years:
            assert begin >= dates.min()
            assert end <= dates.max()


class TestGroupBlocks:

    def test_single_year_blocks(self):
        """block_years=1 should return same number of groups as inputs."""
        blocks = [np.arange(365), np.arange(365, 730), np.arange(730, 1095)]
        groups = _group_blocks(blocks, block_years=1)
        assert len(groups) == 3

    def test_two_year_blocks(self):
        """block_years=2 from 4 year-blocks → 2 groups."""
        blocks = [np.arange(i * 365, (i + 1) * 365) for i in range(4)]
        groups = _group_blocks(blocks, block_years=2)
        assert len(groups) == 2
        assert len(groups[0]) == 730

    def test_incomplete_block_dropped(self):
        """Trailing incomplete block should be dropped."""
        blocks = [np.arange(i * 365, (i + 1) * 365) for i in range(5)]
        groups = _group_blocks(blocks, block_years=2)
        assert len(groups) == 2


class TestApplyBuffer:

    def test_buffer_removes_adjacent_train_indices(self):
        """Buffer should remove training indices near validation block."""
        train_idx = np.arange(100)
        val_idx = np.arange(100, 200)
        train_out, val_out = _apply_buffer(train_idx, val_idx, buffer_days=10)
        assert np.all(train_out < 90)
        np.testing.assert_array_equal(val_out, val_idx)

    def test_zero_buffer_is_noop(self):
        """Buffer of 0 should not remove any indices."""
        train_idx = np.arange(100)
        val_idx = np.arange(100, 200)
        train_out, val_out = _apply_buffer(train_idx, val_idx, buffer_days=0)
        np.testing.assert_array_equal(train_out, train_idx)

    def test_empty_val_noop(self):
        """Empty validation set should not affect training indices."""
        train_idx = np.arange(100)
        val_idx = np.array([], dtype=int)
        train_out, val_out = _apply_buffer(train_idx, val_idx, buffer_days=30)
        np.testing.assert_array_equal(train_out, train_idx)


# ═════════════════════════════════════════════════════════════════════════
# Block CV splits
# ═════════════════════════════════════════════════════════════════════════

class TestBlockCVSplits:

    def test_splits_are_non_overlapping(self, synthetic_10yr_daily, small_config):
        """Train and validation indices in each fold must not overlap."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_year_start = "calendar_year"
        splits = create_block_cv_splits(dates, small_config, y_obs)
        for train_idx, val_idx in splits:
            overlap = np.intersect1d(train_idx, val_idx)
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_at_least_two_folds(self, synthetic_10yr_daily, small_config):
        """Must produce at least 2 folds for meaningful CV."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_year_start = "calendar_year"
        splits = create_block_cv_splits(dates, small_config, y_obs)
        assert len(splits) >= 2

    def test_water_year_splits(self, synthetic_5yr_daily, small_config):
        """Water year split should work with 5-year record."""
        dates, _, y_obs = synthetic_5yr_daily
        small_config.cv_year_start = "water_year"
        small_config.cv_block_years = 1
        splits = create_block_cv_splits(dates, small_config, y_obs)
        assert len(splits) >= 2

    def test_multi_year_blocks(self, synthetic_10yr_daily, small_config):
        """2-year blocks from 10 years → ~5 folds."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_year_start = "calendar_year"
        small_config.cv_block_years = 2
        splits = create_block_cv_splits(dates, small_config, y_obs)
        assert len(splits) >= 3

    def test_n_cv_folds_override(self, synthetic_10yr_daily, small_config):
        """Setting n_cv_folds should limit the number of splits."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_year_start = "calendar_year"
        small_config.n_cv_folds = 3
        splits = create_block_cv_splits(dates, small_config, y_obs)
        assert len(splits) == 3

    def test_insufficient_years_raises(self, small_config):
        """Record shorter than 2 complete years should raise ValueError."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        small_config.cv_year_start = "calendar_year"
        with pytest.raises(ValueError, match="at least 2"):
            create_block_cv_splits(dates, small_config)


class TestExpandingWindowSplits:

    def test_expanding_window_train_grows(self, synthetic_10yr_daily, small_config):
        """Training set should grow monotonically across folds."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_strategy = "expanding_window"
        small_config.cv_year_start = "calendar_year"
        small_config.min_train_years = 1.0
        splits = create_expanding_window_splits(dates, small_config, y_obs)
        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_expanding_window_min_train_filter(self, synthetic_10yr_daily, small_config):
        """Folds with insufficient training data should be excluded."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_year_start = "calendar_year"
        small_config.min_train_years = 5.0
        splits = create_expanding_window_splits(dates, small_config, y_obs)
        min_days = int(5.0 * 365.25)
        for train_idx, _ in splits:
            assert len(train_idx) >= min_days * 0.8  # allow buffer removal


class TestCreateCVSplitsDispatch:

    def test_block_dispatch(self, synthetic_10yr_daily, small_config):
        """cv_strategy='block' should dispatch to block splits."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_strategy = "block"
        small_config.cv_year_start = "calendar_year"
        splits = create_cv_splits(dates, small_config, y_obs)
        assert len(splits) >= 2

    def test_expanding_dispatch(self, synthetic_10yr_daily, small_config):
        """cv_strategy='expanding_window' should dispatch correctly."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_strategy = "expanding_window"
        small_config.cv_year_start = "calendar_year"
        small_config.min_train_years = 1.0
        splits = create_cv_splits(dates, small_config, y_obs)
        assert len(splits) >= 1

    def test_unknown_strategy_raises(self, synthetic_10yr_daily, small_config):
        """Unknown cv_strategy should raise ValueError."""
        dates, _, y_obs = synthetic_10yr_daily
        small_config.cv_strategy = "rolling"
        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            create_cv_splits(dates, small_config, y_obs)


# ═════════════════════════════════════════════════════════════════════════
# Flow regime classification
# ═════════════════════════════════════════════════════════════════════════

class TestClassifyRegime:

    def test_three_regimes_returned(self, default_config):
        """Should return masks for high, medium, and low."""
        y = np.arange(1000, dtype=float)
        masks = classify_regime(y, default_config)
        assert set(masks.keys()) == {"high", "medium", "low"}

    def test_regimes_cover_all_timesteps(self, default_config):
        """Union of regime masks must cover every timestep."""
        y = np.arange(1000, dtype=float)
        masks = classify_regime(y, default_config)
        combined = masks["high"] | masks["medium"] | masks["low"]
        assert combined.all()

    def test_regimes_mutually_exclusive(self, default_config):
        """No timestep should belong to more than one regime."""
        y = np.arange(1000, dtype=float)
        masks = classify_regime(y, default_config)
        overlap = (
            (masks["high"] & masks["medium"]).any()
            or (masks["high"] & masks["low"]).any()
            or (masks["medium"] & masks["low"]).any()
        )
        assert not overlap

    def test_nan_handling(self, default_config):
        """NaN values in observations should not crash regime classification."""
        y = np.arange(1000, dtype=float)
        y[100:110] = np.nan
        masks = classify_regime(y, default_config)
        assert "high" in masks


class TestRegimeThresholds:

    def test_thresholds_ordered(self, default_config):
        """q_high > q_low for any non-degenerate data."""
        y = np.arange(1000, dtype=float)
        q_high, q_low = regime_thresholds(y, default_config)
        assert q_high > q_low
