"""
Tests for CalibrationReport export to Excel and CSV.

Tests:
- export_report(report, path, format='csv') creates four CSV files with expected columns
- export_report(report, path, format='excel') creates one Excel file with four sheets (if openpyxl)
- report.export(path, format=...) delegates correctly
- Content checks: TimeSeries, Best_Calibration, Diagnostics, FDC have expected structure
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrrm.calibration.export import (
    CSV_SUFFIXES,
    SHEET_BEST_CALIBRATION,
    SHEET_DIAGNOSTICS,
    SHEET_FDC,
    SHEET_TIMESERIES,
    export_batch,
    export_report,
)
from pyrrm.calibration.report import CalibrationReport
from pyrrm.calibration.runner import CalibrationResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_calibration_result():
    """Minimal CalibrationResult for building a report."""
    n = 50
    return CalibrationResult(
        best_parameters={"x1": 100.0, "x2": 0.5},
        best_objective=0.85,
        all_samples=pd.DataFrame(
            {"x1": np.random.uniform(80, 120, n), "x2": np.random.uniform(0.3, 0.7, n)}
        ),
        convergence_diagnostics={},
        runtime_seconds=10.0,
        method="SCE-UA",
        objective_name="NSE",
        success=True,
        message="Ok",
    )


@pytest.fixture
def minimal_report(minimal_calibration_result):
    """Minimal CalibrationReport with observed/simulated time series and optional P/PET."""
    n = 100
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    obs = np.maximum(np.random.lognormal(2, 1, n), 0.1)
    sim = obs * np.random.uniform(0.8, 1.2, n)
    precip = np.random.exponential(5, n)
    pet = np.random.uniform(1, 4, n)
    return CalibrationReport(
        result=minimal_calibration_result,
        observed=obs,
        simulated=sim,
        dates=dates,
        precipitation=precip,
        pet=pet,
        inputs=None,
        parameter_bounds={"x1": (50.0, 200.0), "x2": (0.0, 1.0)},
        catchment_info={"name": "Test Catchment", "gauge_id": "410734", "area_km2": 250.0},
        calibration_period=("2020-01-01", "2020-04-10"),
        warmup_days=10,
        model_config={},
        experiment_name="410734_test_nse_sceua",
        created_at="2024-01-15T12:00:00",
    )


@pytest.fixture
def tmp_path_dir(tmp_path):
    """Alias for tmp_path (pytest provides tmp_path)."""
    return tmp_path


# =============================================================================
# CSV export (no optional deps)
# =============================================================================


class TestExportCSV:
    """CSV export creates four files with expected columns."""

    def test_export_csv_creates_four_files(self, minimal_report, tmp_path):
        prefix = str(tmp_path / "report")
        created = export_report(minimal_report, prefix, format="csv")
        assert len(created) == 4
        for p in created:
            assert Path(p).exists()

    def test_export_csv_filenames(self, minimal_report, tmp_path):
        prefix = str(tmp_path / "out")
        export_report(minimal_report, prefix, format="csv")
        for suffix in CSV_SUFFIXES:
            p = tmp_path / ("out" + suffix)
            assert p.exists(), f"Expected {p}"

    def test_export_csv_timeseries_columns(self, minimal_report, tmp_path):
        export_report(minimal_report, tmp_path / "r", format="csv")
        df = pd.read_csv(tmp_path / "r_timeseries.csv")
        assert list(df.columns) == [
            "date",
            "precipitation",
            "pet",
            "observed_flow",
            "simulated_flow",
        ]
        assert len(df) == len(minimal_report.dates)

    def test_export_csv_best_calibration_columns(self, minimal_report, tmp_path):
        export_report(minimal_report, tmp_path / "r", format="csv")
        df = pd.read_csv(tmp_path / "r_best_calibration.csv")
        assert list(df.columns) == ["name", "value"]
        names = set(df["name"])
        assert "x1" in names and "x2" in names
        assert "method" in names and "objective_name" in names
        assert "best_objective" in names and "runtime_seconds" in names
        assert "calibration_period_start" in names
        assert "catchment_name" in names

    def test_export_csv_diagnostics_columns(self, minimal_report, tmp_path):
        export_report(minimal_report, tmp_path / "r", format="csv")
        df = pd.read_csv(tmp_path / "r_diagnostics.csv")
        assert list(df.columns) == ["group", "metric", "value"]
        assert len(df) >= 40  # canonical suite has 48 metrics

    def test_export_csv_fdc_columns(self, minimal_report, tmp_path):
        export_report(minimal_report, tmp_path / "r", format="csv")
        df = pd.read_csv(tmp_path / "r_fdc.csv")
        assert list(df.columns) == ["exceedance_pct", "flow_observed", "flow_simulated"]
        assert len(df) == 99  # 1% to 99% step 1


# =============================================================================
# Excel export (requires openpyxl)
# =============================================================================


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("openpyxl") is None,
    reason="openpyxl not installed",
)
class TestExportExcel:
    """Excel export creates one file with four sheets and expected columns."""

    def test_export_excel_creates_one_file(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        created = export_report(minimal_report, str(xlsx), format="excel")
        assert len(created) == 1
        assert created[0] == str(xlsx)
        assert xlsx.exists()

    def test_export_excel_sheet_names(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        export_report(minimal_report, str(xlsx), format="excel")
        xl = pd.ExcelFile(xlsx)
        assert set(xl.sheet_names) == {
            SHEET_TIMESERIES,
            SHEET_BEST_CALIBRATION,
            SHEET_DIAGNOSTICS,
            SHEET_FDC,
        }

    def test_export_excel_timeseries_sheet(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        export_report(minimal_report, str(xlsx), format="excel")
        df = pd.read_excel(xlsx, sheet_name=SHEET_TIMESERIES)
        assert list(df.columns) == [
            "date",
            "precipitation",
            "pet",
            "observed_flow",
            "simulated_flow",
        ]
        assert len(df) == len(minimal_report.dates)

    def test_export_excel_best_calibration_sheet(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        export_report(minimal_report, str(xlsx), format="excel")
        df = pd.read_excel(xlsx, sheet_name=SHEET_BEST_CALIBRATION)
        assert list(df.columns) == ["name", "value"]
        assert df[df["name"] == "x1"]["value"].iloc[0] == 100.0

    def test_export_excel_diagnostics_sheet(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        export_report(minimal_report, str(xlsx), format="excel")
        df = pd.read_excel(xlsx, sheet_name=SHEET_DIAGNOSTICS)
        assert list(df.columns) == ["group", "metric", "value"]
        assert "NSE" in df["metric"].values

    def test_export_excel_fdc_sheet(self, minimal_report, tmp_path):
        xlsx = tmp_path / "report.xlsx"
        export_report(minimal_report, str(xlsx), format="excel")
        df = pd.read_excel(xlsx, sheet_name=SHEET_FDC)
        assert list(df.columns) == ["exceedance_pct", "flow_observed", "flow_simulated"]
        assert len(df) == 99


# =============================================================================
# format='both' and path behaviour
# =============================================================================


class TestExportBothAndPath:
    """format='both' and path as directory or prefix."""

    def test_export_both_csv_created(self, minimal_report, tmp_path):
        base = tmp_path / "export"
        base.mkdir()
        created = export_report(minimal_report, str(base), format="both")
        # Excel + 4 CSV
        assert len(created) == 5
        csv_count = sum(1 for p in created if p.endswith(".csv"))
        assert csv_count == 4
        assert any(p.endswith(".xlsx") for p in created)

    def test_export_path_is_directory_uses_experiment_name(self, minimal_report, tmp_path):
        (tmp_path / "subdir").mkdir(parents=True)
        export_report(minimal_report, tmp_path / "subdir", format="csv")
        # experiment_name is 410734_test_nse_sceua
        assert (tmp_path / "subdir" / "410734_test_nse_sceua_timeseries.csv").exists()


# =============================================================================
# CalibrationReport.export() and errors
# =============================================================================


class TestReportExportMethod:
    """CalibrationReport.export() delegates to export_report."""

    def test_report_export_csv(self, minimal_report, tmp_path):
        created = minimal_report.export(tmp_path / "r", format="csv")
        assert len(created) == 4
        assert (tmp_path / "r_timeseries.csv").exists()

    def test_report_export_invalid_format(self, minimal_report, tmp_path):
        with pytest.raises(ValueError, match="format must be"):
            minimal_report.export(tmp_path / "r", format="pdf")


class TestExportErrors:
    """Error handling: invalid format, missing openpyxl for excel."""

    def test_export_invalid_format_raises(self, minimal_report, tmp_path):
        with pytest.raises(ValueError, match="format must be"):
            export_report(minimal_report, tmp_path / "r", format="invalid")

    def test_export_excel_without_openpyxl_raises(self, minimal_report, tmp_path):
        try:
            import openpyxl  # noqa: F401
            pytest.skip("openpyxl is installed")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="openpyxl"):
            export_report(minimal_report, tmp_path / "r.xlsx", format="excel")


# =============================================================================
# Edge cases: missing P/PET, empty FDC
# =============================================================================


class TestExportEdgeCases:
    """Report with missing precipitation/PET or minimal data."""

    def test_export_without_precip_pet(self, minimal_calibration_result, tmp_path):
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        obs = np.ones(n) * 10.0
        sim = np.ones(n) * 9.5
        report = CalibrationReport(
            result=minimal_calibration_result,
            observed=obs,
            simulated=sim,
            dates=dates,
            precipitation=None,
            pet=None,
            inputs=None,
            parameter_bounds={},
            catchment_info={},
            calibration_period=("2020-01-01", "2020-01-30"),
            warmup_days=0,
            model_config={},
        )
        created = export_report(report, tmp_path / "e", format="csv")
        assert len(created) == 4
        df = pd.read_csv(tmp_path / "e_timeseries.csv")
        assert "precipitation" in df.columns and "pet" in df.columns
        assert df["precipitation"].isna().all()
        assert df["pet"].isna().all()

    def test_export_precip_from_inputs_rainfall_column(
        self, minimal_calibration_result, tmp_path
    ):
        """When report.precipitation is None but report.inputs has 'rainfall', export uses it."""
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        obs = np.ones(n) * 10.0
        sim = np.ones(n) * 9.5
        rainfall_vals = np.random.exponential(5, n)
        pet_vals = np.random.uniform(1, 4, n)
        inputs_df = pd.DataFrame(
            {"rainfall": rainfall_vals, "pet": pet_vals},
            index=dates,
        )
        report = CalibrationReport(
            result=minimal_calibration_result,
            observed=obs,
            simulated=sim,
            dates=dates,
            precipitation=None,
            pet=None,
            inputs=inputs_df,
            parameter_bounds={},
            catchment_info={},
            calibration_period=("2020-01-01", "2020-01-30"),
            warmup_days=0,
            model_config={},
        )
        created = export_report(report, tmp_path / "e", format="csv")
        assert len(created) == 4
        df = pd.read_csv(tmp_path / "e_timeseries.csv")
        assert "precipitation" in df.columns and "pet" in df.columns
        np.testing.assert_array_almost_equal(
            df["precipitation"].values, rainfall_vals, decimal=5
        )
        np.testing.assert_array_almost_equal(df["pet"].values, pet_vals, decimal=5)


# =============================================================================
# Batch export (export_batch)
# =============================================================================


class TestExportBatch:
    """export_batch iterates over results and exports each report."""

    @staticmethod
    def _make_report(seed, experiment_name):
        """Helper to build a minimal CalibrationReport."""
        np.random.seed(seed)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        obs = np.maximum(np.random.lognormal(2, 1, n), 0.1)
        sim = obs * np.random.uniform(0.8, 1.2, n)
        result = CalibrationResult(
            best_parameters={"a": float(seed)},
            best_objective=0.8 + seed * 0.01,
            all_samples=pd.DataFrame({"a": [float(seed)]}),
            convergence_diagnostics={},
            runtime_seconds=1.0,
            method="SCE-UA",
            objective_name="NSE",
            success=True,
            message="",
        )
        return CalibrationReport(
            result=result,
            observed=obs,
            simulated=sim,
            dates=dates,
            precipitation=np.random.exponential(5, n),
            pet=np.random.uniform(1, 4, n),
            parameter_bounds={"a": (0.0, 10.0)},
            catchment_info={"name": "Test"},
            calibration_period=("2020-01-01", "2020-03-01"),
            warmup_days=0,
            model_config={},
            experiment_name=experiment_name,
        )

    def test_export_batch_csv_creates_subdirs(self, tmp_path):
        from types import SimpleNamespace

        reports = {
            "exp_nse": self._make_report(1, "exp_nse"),
            "exp_kge": self._make_report(2, "exp_kge"),
        }
        batch = SimpleNamespace(results=reports)
        exported = export_batch(batch, str(tmp_path), format="csv")
        assert set(exported.keys()) == {"exp_nse", "exp_kge"}
        for key, paths in exported.items():
            assert len(paths) == 4
            subdir = tmp_path / key
            assert subdir.is_dir(), f"Expected subdirectory {subdir}"
            for p in paths:
                assert Path(p).exists()
                assert Path(p).parent == subdir

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("openpyxl") is None,
        reason="openpyxl not installed",
    )
    def test_export_batch_excel_creates_subdirs(self, tmp_path):
        from types import SimpleNamespace

        reports = {
            "exp_a": self._make_report(10, "exp_a"),
            "exp_b": self._make_report(20, "exp_b"),
        }
        batch = SimpleNamespace(results=reports)
        exported = export_batch(batch, str(tmp_path), format="excel")
        assert len(exported) == 2
        for key, paths in exported.items():
            assert len(paths) == 1
            assert paths[0].endswith(".xlsx")
            xlsx = Path(paths[0])
            assert xlsx.exists()
            assert xlsx.parent == tmp_path / key

    def test_export_batch_empty(self, tmp_path):
        from types import SimpleNamespace

        batch = SimpleNamespace(results={})
        exported = export_batch(batch, str(tmp_path), format="csv")
        assert exported == {}

    def test_export_batch_skip_failures(self, tmp_path):
        from types import SimpleNamespace

        bad_report = "not_a_report"
        good_report = self._make_report(5, "good")
        batch = SimpleNamespace(results={"bad": bad_report, "good": good_report})
        exported = export_batch(batch, str(tmp_path), format="csv", skip_failures=True)
        assert "good" in exported
        assert "bad" not in exported
        assert (tmp_path / "good").is_dir()
