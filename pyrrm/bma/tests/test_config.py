"""
Tests for BMAConfig dataclass and ACT CV presets.
"""

import pytest
from pathlib import Path

from pyrrm.bma.config import BMAConfig, ACT_CV_PRESETS


class TestBMAConfigDefaults:
    """Verify default attribute values are hydrologically sensible."""

    def test_default_cv_strategy(self):
        """Default CV strategy is block temporal."""
        cfg = BMAConfig()
        assert cfg.cv_strategy == "block"

    def test_default_cv_year_start(self):
        """Default year start is water year (July)."""
        cfg = BMAConfig()
        assert cfg.cv_year_start == "water_year"

    def test_default_buffer_days(self):
        """Default buffer days is 60."""
        cfg = BMAConfig()
        assert cfg.buffer_days == 60

    def test_default_regime_quantiles(self):
        """Regime quantiles should define three distinct flow regimes."""
        cfg = BMAConfig()
        assert len(cfg.regime_quantiles) == 2
        assert cfg.regime_quantiles[0] < cfg.regime_quantiles[1]

    def test_default_prediction_intervals(self):
        """Prediction intervals should be sorted probabilities in (0, 1)."""
        cfg = BMAConfig()
        for pi in cfg.prediction_intervals:
            assert 0 < pi < 1

    def test_default_dirichlet_alpha(self):
        """Default Dirichlet alpha is sparse."""
        cfg = BMAConfig()
        assert cfg.dirichlet_alpha == "sparse"


class TestResolvedStartMonth:
    """Test the resolved_start_month property for all cv_year_start modes."""

    def test_water_year_returns_july(self):
        """Water year boundary is July (month 7) for ACT."""
        cfg = BMAConfig(cv_year_start="water_year")
        assert cfg.resolved_start_month == 7

    def test_calendar_year_returns_january(self):
        """Calendar year boundary is January (month 1)."""
        cfg = BMAConfig(cv_year_start="calendar_year")
        assert cfg.resolved_start_month == 1

    def test_custom_returns_specified_month(self):
        """Custom start month should pass through directly."""
        cfg = BMAConfig(cv_year_start="custom", cv_year_start_month=4)
        assert cfg.resolved_start_month == 4

    def test_custom_month_boundary_jan(self):
        """Custom month=1 should work."""
        cfg = BMAConfig(cv_year_start="custom", cv_year_start_month=1)
        assert cfg.resolved_start_month == 1

    def test_custom_month_boundary_dec(self):
        """Custom month=12 should work."""
        cfg = BMAConfig(cv_year_start="custom", cv_year_start_month=12)
        assert cfg.resolved_start_month == 12


class TestPathCoercion:
    """Paths should be coerced to pathlib.Path objects."""

    def test_string_predictions_path_becomes_path(self):
        """String predictions path is converted to Path."""
        cfg = BMAConfig(model_predictions_path="data/pred.csv")
        assert isinstance(cfg.model_predictions_path, Path)

    def test_string_observed_path_becomes_path(self):
        """String observed flow path is converted to Path."""
        cfg = BMAConfig(observed_flow_path="data/obs.csv")
        assert isinstance(cfg.observed_flow_path, Path)

    def test_none_paths_stay_none(self):
        """None paths should remain None."""
        cfg = BMAConfig()
        assert cfg.model_predictions_path is None
        assert cfg.observed_flow_path is None


class TestFromPreset:
    """Test preset-based config creation."""

    def test_standard_preset(self):
        """Standard preset should set water year, 2-year blocks."""
        cfg = BMAConfig.from_preset("standard")
        assert cfg.cv_year_start == "water_year"
        assert cfg.cv_block_years == 2
        assert cfg.buffer_days == 60

    def test_flood_focused_preset(self):
        """Flood-focused preset uses custom start month (April)."""
        cfg = BMAConfig.from_preset("flood_focused")
        assert cfg.cv_year_start == "custom"
        assert cfg.cv_year_start_month == 4
        assert cfg.cv_block_years == 1

    def test_long_term_drought_preset(self):
        """Long-term drought preset uses 5-year blocks."""
        cfg = BMAConfig.from_preset("long_term_drought")
        assert cfg.cv_block_years == 5
        assert cfg.buffer_days == 90

    def test_operational_preset(self):
        """Operational preset uses expanding window strategy."""
        cfg = BMAConfig.from_preset("operational")
        assert cfg.cv_strategy == "expanding_window"

    def test_preset_with_overrides(self):
        """Overrides should take precedence over preset values."""
        cfg = BMAConfig.from_preset("standard", buffer_days=30, draws=500)
        assert cfg.buffer_days == 30
        assert cfg.draws == 500
        assert cfg.cv_year_start == "water_year"

    def test_unknown_preset_raises(self):
        """Unknown preset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            BMAConfig.from_preset("nonexistent")

    def test_all_presets_are_valid(self):
        """Every preset in ACT_CV_PRESETS should produce a valid config."""
        for name in ACT_CV_PRESETS:
            cfg = BMAConfig.from_preset(name)
            assert isinstance(cfg, BMAConfig)
