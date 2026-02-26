"""Tests for COLUMN_ALIASES, resolve_column, and load_catchment_data."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrrm.data.input_handler import (
    COLUMN_ALIASES,
    load_catchment_data,
    resolve_column,
)


# ---------------------------------------------------------------------------
# resolve_column
# ---------------------------------------------------------------------------

class TestResolveColumn:
    """Tests for the resolve_column helper."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"rainfall": [1, 2], "pet": [3, 4]})

    def test_exact_alias_match(self, sample_df):
        assert resolve_column(sample_df, "precipitation") == "rainfall"

    def test_canonical_name_match(self):
        df = pd.DataFrame({"precipitation": [1], "pet": [2]})
        assert resolve_column(df, "precipitation") == "precipitation"

    def test_case_insensitive_fallback(self):
        df = pd.DataFrame({"RAINFALL": [1], "PET": [2]})
        assert resolve_column(df, "precipitation") == "RAINFALL"

    def test_returns_none_when_missing(self, sample_df):
        assert resolve_column(sample_df, "observed_flow") is None

    def test_raise_on_missing(self, sample_df):
        with pytest.raises(ValueError, match="observed_flow"):
            resolve_column(sample_df, "observed_flow", raise_on_missing=True)

    def test_invalid_canonical_name_raises_keyerror(self, sample_df):
        with pytest.raises(KeyError):
            resolve_column(sample_df, "temperature")

    @pytest.mark.parametrize(
        "canonical,col",
        [
            ("precipitation", "precipitation"),
            ("precipitation", "rainfall"),
            ("precipitation", "precip"),
            ("precipitation", "rain"),
            ("precipitation", "P"),
            ("pet", "pet"),
            ("pet", "evapotranspiration"),
            ("pet", "evap"),
            ("pet", "ET"),
            ("observed_flow", "observed_flow"),
            ("observed_flow", "flow"),
            ("observed_flow", "discharge"),
            ("observed_flow", "Q"),
            ("date", "Date"),
            ("date", "datetime"),
        ],
    )
    def test_all_alias_families(self, canonical, col):
        df = pd.DataFrame({col: [0.0]})
        assert resolve_column(df, canonical) == col


# ---------------------------------------------------------------------------
# COLUMN_ALIASES
# ---------------------------------------------------------------------------

class TestColumnAliases:
    """Basic structural tests for the COLUMN_ALIASES constant."""

    def test_required_keys_present(self):
        for key in ("precipitation", "pet", "observed_flow", "date"):
            assert key in COLUMN_ALIASES

    def test_canonical_name_is_first_alias(self):
        for key, aliases in COLUMN_ALIASES.items():
            assert aliases[0] == key, (
                f"First alias for '{key}' should be the canonical name itself"
            )

    def test_no_empty_alias_lists(self):
        for key, aliases in COLUMN_ALIASES.items():
            assert len(aliases) >= 2, f"'{key}' should have at least two aliases"


# ---------------------------------------------------------------------------
# load_catchment_data
# ---------------------------------------------------------------------------

def _write_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path)


class TestLoadCatchmentData:
    """Integration tests for load_catchment_data."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create three CSV files with 100 days of synthetic data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        precip = pd.DataFrame(
            {"Date": dates, "rainfall": np.random.exponential(3.0, 100)}
        )
        pet = pd.DataFrame(
            {"Date": dates, "PET": np.random.uniform(1, 5, 100)}
        )
        flow = pd.DataFrame(
            {"Date": dates, "discharge": np.random.uniform(10, 500, 100)}
        )

        _write_csv(tmp_path / "rain.csv", precip)
        _write_csv(tmp_path / "pet.csv", pet)
        _write_csv(tmp_path / "flow.csv", flow)

        return tmp_path

    def test_basic_load(self, data_dir):
        inputs, observed = load_catchment_data(
            data_dir / "rain.csv",
            data_dir / "pet.csv",
            data_dir / "flow.csv",
        )
        assert list(inputs.columns) == ["precipitation", "pet"]
        assert isinstance(inputs.index, pd.DatetimeIndex)
        assert len(observed) == len(inputs) == 100
        assert observed.dtype == np.float64

    def test_date_slicing(self, data_dir):
        inputs, observed = load_catchment_data(
            data_dir / "rain.csv",
            data_dir / "pet.csv",
            data_dir / "flow.csv",
            start_date="2020-02-01",
            end_date="2020-03-01",
        )
        assert inputs.index[0] == pd.Timestamp("2020-02-01")
        assert inputs.index[-1] == pd.Timestamp("2020-03-01")

    def test_missing_value_replacement(self, data_dir):
        """Sentinel -9999 values in observed flow should be dropped."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        flow_vals = np.random.uniform(10, 500, 50)
        flow_vals[10] = -9999
        flow_vals[20] = -5  # negative → also dropped

        flow = pd.DataFrame({"Date": dates, "Q": flow_vals})
        _write_csv(data_dir / "flow_missing.csv", flow)

        precip = pd.DataFrame(
            {"Date": dates, "precipitation": np.random.exponential(3.0, 50)}
        )
        pet = pd.DataFrame(
            {"Date": dates, "pet": np.random.uniform(1, 5, 50)}
        )
        _write_csv(data_dir / "rain50.csv", precip)
        _write_csv(data_dir / "pet50.csv", pet)

        inputs, observed = load_catchment_data(
            data_dir / "rain50.csv",
            data_dir / "pet50.csv",
            data_dir / "flow_missing.csv",
        )
        assert len(inputs) == 48  # 50 - 2 dropped
        assert np.all(observed >= 0)

    def test_explicit_observed_column(self, data_dir):
        """The observed_value_column kwarg should override auto-detection."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        flow = pd.DataFrame({
            "Date": dates,
            "non_standard_name": np.random.uniform(10, 500, 30),
        })
        _write_csv(data_dir / "flow_explicit.csv", flow)

        precip = pd.DataFrame(
            {"Date": dates, "precipitation": np.random.exponential(3.0, 30)}
        )
        pet = pd.DataFrame(
            {"Date": dates, "pet": np.random.uniform(1, 5, 30)}
        )
        _write_csv(data_dir / "rain30.csv", precip)
        _write_csv(data_dir / "pet30.csv", pet)

        inputs, observed = load_catchment_data(
            data_dir / "rain30.csv",
            data_dir / "pet30.csv",
            data_dir / "flow_explicit.csv",
            observed_value_column="non_standard_name",
        )
        assert len(inputs) == 30

    def test_no_date_column_raises(self, data_dir):
        bad_csv = data_dir / "nodate.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="date column"):
            load_catchment_data(bad_csv, bad_csv, bad_csv)

    def test_empty_after_merge_raises(self, data_dir):
        """Non-overlapping date ranges should raise."""
        d1 = pd.date_range("2020-01-01", periods=10, freq="D")
        d2 = pd.date_range("2025-01-01", periods=10, freq="D")

        _write_csv(
            data_dir / "rain_disjoint.csv",
            pd.DataFrame({"Date": d1, "precipitation": np.ones(10)}),
        )
        _write_csv(
            data_dir / "pet_disjoint.csv",
            pd.DataFrame({"Date": d2, "pet": np.ones(10)}),
        )
        _write_csv(
            data_dir / "flow_disjoint.csv",
            pd.DataFrame({"Date": d1, "Q": np.ones(10)}),
        )

        with pytest.raises(ValueError, match="no data remains"):
            load_catchment_data(
                data_dir / "rain_disjoint.csv",
                data_dir / "pet_disjoint.csv",
                data_dir / "flow_disjoint.csv",
            )
