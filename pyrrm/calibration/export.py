"""
Export CalibrationReport to Excel (multi-sheet) or CSV for sharing.

Excel export requires the optional dependency: pip install pyrrm[export]
(adds openpyxl). CSV export has no extra dependencies.

Single-report layout (four sheets / four CSVs):
- TimeSeries: date, precipitation, pet, observed_flow, simulated_flow,
  sim_baseflow, sim_quickflow (Lyne-Hollick separation of simulated flow)
- Best_Calibration: two-column table (name, value) for parameters and run metadata
- Diagnostics: canonical 48-metric suite with group, metric, value
- FDC: exceedance_pct, flow_observed, flow_simulated on a common exceedance grid

Batch export (``export_batch``) iterates over every CalibrationReport inside a
``BatchResult`` and writes one Excel workbook (or CSV set) per experiment into
an output directory.

Example:
    >>> from pyrrm.calibration import CalibrationReport, export_report
    >>> report = CalibrationReport.load('calibrations/410734_sacramento_nse.pkl')
    >>> export_report(report, 'output/410734_report.xlsx', format='excel')
    >>> report.export('output/410734_report', format='both')

    >>> from pyrrm.calibration import BatchResult, export_batch
    >>> batch = BatchResult.load('results/batch_result.pkl')
    >>> export_batch(batch, 'exports/', format='excel')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Sheet names and CSV suffixes (must stay in sync)
SHEET_TIMESERIES = "TimeSeries"
SHEET_BEST_CALIBRATION = "Best_Calibration"
SHEET_DIAGNOSTICS = "Diagnostics"
SHEET_FDC = "FDC"
CSV_SUFFIXES = ("_timeseries.csv", "_best_calibration.csv", "_diagnostics.csv", "_fdc.csv")


def _get_precip_from_report(report: Any, n: int) -> Optional[np.ndarray]:
    """Return precipitation array of length n from report (stored or from inputs)."""
    if report.precipitation is not None and len(report.precipitation) == n:
        return np.asarray(report.precipitation)
    if report.inputs is not None and len(report.inputs) >= report.warmup_days + n:
        from pyrrm.data import resolve_column
        pcol = resolve_column(report.inputs, "precipitation")
        if pcol is not None:
            vals = report.inputs[pcol].values
            return np.asarray(vals[report.warmup_days : report.warmup_days + n])
    return None


def _get_pet_from_report(report: Any, n: int) -> Optional[np.ndarray]:
    """Return PET array of length n from report (stored or from inputs)."""
    if report.pet is not None and len(report.pet) == n:
        return np.asarray(report.pet)
    if report.inputs is not None and len(report.inputs) >= report.warmup_days + n:
        from pyrrm.data import resolve_column
        ecol = resolve_column(report.inputs, "pet")
        if ecol is not None:
            vals = report.inputs[ecol].values
            return np.asarray(vals[report.warmup_days : report.warmup_days + n])
    return None


def _build_timeseries_df(report: Any) -> pd.DataFrame:
    """Build DataFrame with flow timeseries and Lyne-Hollick baseflow separation.

    Columns: date, precipitation, pet, observed_flow, simulated_flow,
    sim_baseflow (Lyne-Hollick slow flow), sim_quickflow (fast flow).
    """
    from pyrrm.analysis.diagnostics import lyne_hollick_baseflow

    n = len(report.dates)
    data = {"date": report.dates}
    precip = _get_precip_from_report(report, n)
    data["precipitation"] = precip if precip is not None else np.full(n, np.nan)
    pet = _get_pet_from_report(report, n)
    data["pet"] = pet if pet is not None else np.full(n, np.nan)
    data["observed_flow"] = report.observed
    data["simulated_flow"] = report.simulated
    sim = np.asarray(report.simulated).flatten()
    bf = lyne_hollick_baseflow(sim)
    data["sim_baseflow"] = bf
    data["sim_quickflow"] = sim - bf
    return pd.DataFrame(data)


def _build_best_calibration_df(report: Any) -> pd.DataFrame:
    """Build two-column table: name, value for parameters and run metadata."""
    rows: List[tuple] = []
    r = report.result
    for name, value in r.best_parameters.items():
        rows.append((name, float(value)))
    rows.append(("method", r.method))
    rows.append(("objective_name", r.objective_name))
    rows.append(("best_objective", float(r.best_objective)))
    rows.append(("runtime_seconds", float(r.runtime_seconds)))
    rows.append(("success", r.success))
    rows.append(("message", str(r.message)))
    rows.append(("calibration_period_start", report.calibration_period[0]))
    rows.append(("calibration_period_end", report.calibration_period[1]))
    rows.append(("warmup_days", report.warmup_days))
    for key in ("name", "gauge_id", "area_km2"):
        if report.catchment_info and key in report.catchment_info:
            rows.append((f"catchment_{key}", report.catchment_info[key]))
    if report.experiment_name:
        rows.append(("experiment_name", report.experiment_name))
    rows.append(("created_at", report.created_at))
    return pd.DataFrame(rows, columns=["name", "value"])


def _build_diagnostics_df(report: Any) -> pd.DataFrame:
    """Build DataFrame: group, metric, value from canonical diagnostics."""
    from pyrrm.analysis.diagnostics import DIAGNOSTIC_GROUPS, compute_diagnostics

    metrics = compute_diagnostics(report.simulated, report.observed)
    rows = []
    for group_name, keys in DIAGNOSTIC_GROUPS.items():
        for k in keys:
            v = metrics.get(k, np.nan)
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                val: Any = ""
            else:
                val = float(v) if hasattr(v, "__float__") else v
            rows.append((group_name, k, val))
    return pd.DataFrame(rows, columns=["group", "metric", "value"])


def _build_fdc_df(
    report: Any,
    exceedance_pct_resolution: float = 1.0,
) -> pd.DataFrame:
    """Build FDC DataFrame on common exceedance grid: exceedance_pct, flow_observed, flow_simulated."""
    from pyrrm.objectives.fdc.curves import compute_fdc

    obs = np.asarray(report.observed).flatten()
    sim = np.asarray(report.simulated).flatten()
    obs = obs[~np.isnan(obs)]
    sim = sim[~np.isnan(sim)]
    if len(obs) == 0 and len(sim) == 0:
        return pd.DataFrame(columns=["exceedance_pct", "flow_observed", "flow_simulated"])

    # Common grid in percent (e.g. 1, 2, ..., 99)
    grid_pct = np.arange(
        exceedance_pct_resolution,
        100.0,
        exceedance_pct_resolution,
        dtype=float,
    )
    grid_frac = grid_pct / 100.0

    flow_obs = np.full_like(grid_pct, np.nan)
    flow_sim = np.full_like(grid_pct, np.nan)

    if len(obs) > 0:
        exc_obs, fdc_obs = compute_fdc(obs)
        if len(exc_obs) > 0:
            flow_obs = np.interp(grid_frac, exc_obs, fdc_obs)
    if len(sim) > 0:
        exc_sim, fdc_sim = compute_fdc(sim)
        if len(exc_sim) > 0:
            flow_sim = np.interp(grid_frac, exc_sim, fdc_sim)

    return pd.DataFrame(
        {
            "exceedance_pct": grid_pct,
            "flow_observed": flow_obs,
            "flow_simulated": flow_sim,
        }
    )


def export_report(
    report: Any,
    path: Union[str, Path],
    format: str = "excel",
    exceedance_pct_resolution: float = 1.0,
    csv_prefix: Optional[str] = None,
) -> List[str]:
    """
    Export a CalibrationReport to Excel and/or CSV files.

    Excel: single file with sheets TimeSeries, Best_Calibration, Diagnostics, FDC.
    CSV: four files named {prefix}_timeseries.csv, {prefix}_best_calibration.csv,
    {prefix}_diagnostics.csv, {prefix}_fdc.csv. Prefix is derived from path (stem)
    unless csv_prefix is provided.

    Args:
        report: CalibrationReport instance (e.g. from CalibrationReport.load(...)).
        path: For format='excel': path to .xlsx file. For format='csv' or 'both':
            directory or file prefix (e.g. 'out/410734' -> out/410734_timeseries.csv).
        format: 'excel', 'csv', or 'both'.
        exceedance_pct_resolution: FDC grid step in percent (default 1.0 -> 1%, 2%, ..., 99%).
        csv_prefix: Override prefix for CSV filenames (default: path stem or path as prefix).

    Returns:
        List of created file paths.

    Raises:
        ImportError: If format is 'excel' or 'both' and openpyxl is not installed.
        ValueError: If format is not one of 'excel', 'csv', 'both'.

    Note:
        Flow units are not stored in the report; add them manually in Best_Calibration if needed.
    """
    path = Path(path)
    format_lower = format.strip().lower()
    if format_lower not in ("excel", "csv", "both"):
        raise ValueError(f"format must be 'excel', 'csv', or 'both'; got {format!r}")

    created: List[str] = []

    df_ts = _build_timeseries_df(report)
    df_best = _build_best_calibration_df(report)
    df_diag = _build_diagnostics_df(report)
    df_fdc = _build_fdc_df(report, exceedance_pct_resolution=exceedance_pct_resolution)

    # Resolve output base: if path is a directory, use experiment_name or default
    if path.is_dir():
        stem = report.experiment_name or "calibration_report"
        output_base = path / stem
    else:
        output_base = path

    if format_lower in ("excel", "both"):
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Excel export requires openpyxl. Install with: pip install pyrrm[export]"
            ) from e
        excel_path = (
            output_base
            if output_base.suffix.lower() in (".xlsx", ".xls")
            else output_base.with_suffix(".xlsx")
        )
        excel_path = Path(excel_path)
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_ts.to_excel(writer, sheet_name=SHEET_TIMESERIES, index=False)
            df_best.to_excel(writer, sheet_name=SHEET_BEST_CALIBRATION, index=False)
            df_diag.to_excel(writer, sheet_name=SHEET_DIAGNOSTICS, index=False)
            df_fdc.to_excel(writer, sheet_name=SHEET_FDC, index=False)
        created.append(str(excel_path))

    if format_lower in ("csv", "both"):
        if csv_prefix is not None:
            prefix_path = Path(csv_prefix)
        else:
            prefix_path = output_base
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        base = str(prefix_path)
        for suffix, df in zip(
            CSV_SUFFIXES,
            (df_ts, df_best, df_diag, df_fdc),
        ):
            csv_path = base + suffix
            df.to_csv(csv_path, index=False)
            created.append(csv_path)

    return created


# =========================================================================
# Batch / Grid / List export
# =========================================================================

logger = logging.getLogger(__name__)


def export_batch(
    batch_result: Any,
    output_dir: Union[str, Path],
    format: str = "excel",
    exceedance_pct_resolution: float = 1.0,
    skip_failures: bool = True,
) -> Dict[str, List[str]]:
    """Export every CalibrationReport in a BatchResult.

    Each experiment is placed in its own subdirectory under *output_dir*
    so the export folder stays tidy::

        output_dir/
        ├── 410734_sacramento_nse_sceua/
        │   ├── 410734_sacramento_nse_sceua.xlsx
        │   ├── 410734_sacramento_nse_sceua_timeseries.csv
        │   └── ...
        └── 410734_sacramento_kge_sceua/
            ├── 410734_sacramento_kge_sceua.xlsx
            └── ...

    Works identically whether the batch was produced by an
    ``ExperimentGrid``, an ``ExperimentList``, or assembled manually.

    Args:
        batch_result: A ``BatchResult`` (or any object with a
            ``.results`` dict mapping ``str`` -> ``CalibrationReport``).
        output_dir: Root directory for the exported files.  Created if
            needed.  Each experiment gets a subdirectory named after its
            experiment key.
        format: ``'excel'``, ``'csv'``, or ``'both'`` (passed to
            :func:`export_report`).
        exceedance_pct_resolution: FDC grid step in percent (default 1.0).
        skip_failures: If *True* (default), silently skip experiments
            that raise during export and log a warning.  If *False*,
            re-raise the first error.

    Returns:
        Dict mapping each experiment key to the list of files created for
        that experiment.  Experiments that failed to export are omitted.

    Example:
        >>> from pyrrm.calibration import BatchResult, export_batch
        >>> batch = BatchResult.load('results/batch_result.pkl')
        >>> files = export_batch(batch, 'exports/', format='excel')
        >>> for key, paths in files.items():
        ...     print(key, paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = getattr(batch_result, "results", {})
    if not results:
        logger.warning("BatchResult has no experiment results to export.")
        return {}

    exported: Dict[str, List[str]] = {}
    for key, report in results.items():
        try:
            exp_dir = output_dir / key
            exp_dir.mkdir(parents=True, exist_ok=True)
            stem = exp_dir / key
            files = export_report(
                report,
                stem,
                format=format,
                exceedance_pct_resolution=exceedance_pct_resolution,
            )
            exported[key] = files
            logger.info("Exported %s  (%d files)", key, len(files))
        except Exception:
            if skip_failures:
                logger.warning("Failed to export %s", key, exc_info=True)
            else:
                raise
    return exported
