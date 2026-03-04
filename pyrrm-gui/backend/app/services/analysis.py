"""
Analysis service for loading, diagnosing, and visualising batch calibration results.

This service loads pre-existing BatchResult pickle files from disk, computes
diagnostics, builds comparison figures, and returns JSON-serialisable data for
the web frontend.
"""

import sys
import uuid
import json
import math
import logging
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# ── pyrrm path setup (same pattern used by other services) ──────────────────
def _setup_pyrrm_path():
    current_file = Path(__file__)
    docker_app_dir = current_file.parents[2]
    if (docker_app_dir / "pyrrm").exists():
        if str(docker_app_dir) not in sys.path:
            sys.path.insert(0, str(docker_app_dir))
        return
    try:
        local_path = current_file.parents[4]
        if str(local_path) not in sys.path:
            sys.path.insert(0, str(local_path))
    except IndexError:
        pass

_setup_pyrrm_path()

from pyrrm.calibration.batch import BatchResult, parse_experiment_key
from pyrrm.calibration.report import CalibrationReport
from pyrrm.analysis.diagnostics import compute_diagnostics, DIAGNOSTIC_GROUPS
from pyrrm.analysis.signatures import (
    SIGNATURE_CATEGORIES,
    SIGNATURE_INFO,
    compute_all_signatures,
    signature_percent_error,
)
from pyrrm.objectives import SDEB

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Headline metrics used for diagnostics tables and clustermaps ─────────────

HEADLINE_METRICS = [
    "NSE", "NSE_sqrt", "NSE_log", "NSE_inv",
    "KGE", "KGE_sqrt", "KGE_log", "KGE_inv",
    "KGE_np", "KGE_np_sqrt", "KGE_np_log", "KGE_np_inv",
    "RMSE", "MAE", "SDEB",
    "PBIAS", "FHV", "FMV", "FLV",
    "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5",
]

HIGHER_IS_BETTER = {
    "NSE", "NSE_sqrt", "NSE_log", "NSE_inv",
    "KGE", "KGE_sqrt", "KGE_log", "KGE_inv",
    "KGE_np", "KGE_np_sqrt", "KGE_np_log", "KGE_np_inv",
}

ERROR_METRICS = {"RMSE", "MAE", "SDEB"}

VOLUME_METRICS = {"PBIAS", "FHV", "FMV", "FLV", "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"}


# ── Color palette for comparison traces ─────────────────────────────────────

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b",
    "#e7969c",
]


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float, returning None for NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


# ── Session dataclass ────────────────────────────────────────────────────────

@dataclass
class AnalysisSession:
    id: str
    name: str
    folder_path: str
    loaded_at: str
    batch_results: Dict[str, BatchResult] = field(default_factory=dict)
    _diagnostics_cache: Dict[str, pd.DataFrame] = field(
        default_factory=dict, repr=False
    )

    @property
    def gauge_ids(self) -> List[str]:
        return sorted(self.batch_results.keys())

    @property
    def total_experiments(self) -> int:
        return sum(len(br.results) for br in self.batch_results.values())

    @property
    def total_failures(self) -> int:
        return sum(len(br.failures) for br in self.batch_results.values())


# ── Main service ─────────────────────────────────────────────────────────────

class AnalysisService:
    """Manages analysis sessions and provides diagnostics/visualization."""

    def __init__(self):
        self._sessions: Dict[str, AnalysisSession] = {}

    # ── Session management ───────────────────────────────────────────────

    def load_session(self, folder_path: str, name: Optional[str] = None) -> AnalysisSession:
        """Scan *folder_path* for batch_result.pkl files and load them."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        pkl_files = sorted(folder.rglob("batch_result.pkl"))
        if not pkl_files:
            raise ValueError(f"No batch_result.pkl files found under {folder_path}")

        session_id = uuid.uuid4().hex[:12]
        session_name = name or folder.name

        batch_results: Dict[str, BatchResult] = {}
        for pkl in pkl_files:
            try:
                br = BatchResult.load(str(pkl))
                gauge_id = self._infer_gauge_id(br, pkl)
                batch_results[gauge_id] = br
                logger.info("Loaded %s (%d experiments)", gauge_id, len(br.results))
            except Exception as e:
                logger.warning("Failed to load %s: %s", pkl, e)

        if not batch_results:
            raise ValueError("All batch_result.pkl files failed to load")

        session = AnalysisSession(
            id=session_id,
            name=session_name,
            folder_path=str(folder),
            loaded_at=datetime.utcnow().isoformat(),
            batch_results=batch_results,
        )
        self._sessions[session_id] = session
        return session

    def list_sessions(self) -> List[AnalysisSession]:
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> AnalysisSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        del self._sessions[session_id]

    # ── Gauge helpers ────────────────────────────────────────────────────

    def _infer_gauge_id(self, br: BatchResult, pkl_path: Path) -> str:
        """Derive gauge ID from experiment keys or folder name."""
        for key in br.results:
            parsed = parse_experiment_key(key)
            cid = parsed.get("catchment", "")
            if cid and cid != "default":
                return cid
        return pkl_path.parent.name

    def get_gauge_batch(self, session_id: str, gauge_id: str) -> BatchResult:
        session = self.get_session(session_id)
        if gauge_id not in session.batch_results:
            raise KeyError(f"Gauge {gauge_id} not found in session {session_id}")
        return session.batch_results[gauge_id]

    def get_gauge_summary(self, session_id: str, gauge_id: str) -> Dict[str, Any]:
        br = self.get_gauge_batch(session_id, gauge_id)
        best = {}
        for obj_name, (key, val) in br.best_by_objective().items():
            best[obj_name] = {"key": key, "value": _safe_float(val)}

        metadata = self._extract_gauge_metadata(br, gauge_id)

        return {
            "gauge_id": gauge_id,
            "n_experiments": len(br.results),
            "n_failures": len(br.failures),
            "best_by_objective": best,
            "metadata": metadata,
        }

    def _extract_gauge_metadata(self, br: BatchResult, gauge_id: str) -> Dict[str, Any]:
        """Extract catchment/gauge metadata from the first available CalibrationReport."""
        meta: Dict[str, Any] = {
            "area_km2": None,
            "gauge_name": None,
            "record_start": None,
            "record_end": None,
            "record_years": None,
            "n_days": None,
            "mean_precip_mm_day": None,
            "mean_pet_mm_day": None,
            "total_precip_mm_yr": None,
            "total_pet_mm_yr": None,
            "mean_flow": None,
            "median_flow": None,
            "aridity_index": None,
            "runoff_ratio": None,
        }

        if not br.results:
            return meta

        report: CalibrationReport = next(iter(br.results.values()))

        info = getattr(report, "catchment_info", {}) or {}
        meta["area_km2"] = _safe_float(info.get("area_km2"))
        meta["gauge_name"] = info.get("name") or info.get("gauge_name")

        if report.dates is not None and len(report.dates) > 1:
            meta["record_start"] = str(report.dates[0].date()) if hasattr(report.dates[0], "date") else str(report.dates[0])
            meta["record_end"] = str(report.dates[-1].date()) if hasattr(report.dates[-1], "date") else str(report.dates[-1])
            meta["n_days"] = int(len(report.dates))
            meta["record_years"] = _safe_float(len(report.dates) / 365.25)

        if report.precipitation is not None and len(report.precipitation) > 0:
            p = np.asarray(report.precipitation)
            meta["mean_precip_mm_day"] = _safe_float(np.nanmean(p))
            n_years = len(p) / 365.25 if len(p) > 0 else 1.0
            meta["total_precip_mm_yr"] = _safe_float(np.nansum(p) / n_years)

        if report.pet is not None and len(report.pet) > 0:
            e = np.asarray(report.pet)
            meta["mean_pet_mm_day"] = _safe_float(np.nanmean(e))
            n_years = len(e) / 365.25 if len(e) > 0 else 1.0
            meta["total_pet_mm_yr"] = _safe_float(np.nansum(e) / n_years)

        if report.observed is not None and len(report.observed) > 0:
            obs = np.asarray(report.observed)
            meta["mean_flow"] = _safe_float(np.nanmean(obs))
            meta["median_flow"] = _safe_float(np.nanmedian(obs))

        if meta["total_pet_mm_yr"] and meta["total_precip_mm_yr"] and meta["total_precip_mm_yr"] > 0:
            meta["aridity_index"] = _safe_float(
                meta["total_pet_mm_yr"] / meta["total_precip_mm_yr"]
            )

        if (meta["mean_precip_mm_day"] and meta["mean_flow"] is not None
                and meta["area_km2"] and meta["area_km2"] > 0):
            mean_precip_m3_s = (meta["mean_precip_mm_day"] / 1000.0) * (meta["area_km2"] * 1e6) / 86400.0
            if mean_precip_m3_s > 0:
                meta["runoff_ratio"] = _safe_float(meta["mean_flow"] / mean_precip_m3_s)

        return meta

    # ── Summary table ────────────────────────────────────────────────────

    def get_combined_summary(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        rows: List[Dict[str, Any]] = []
        for gauge_id, br in session.batch_results.items():
            df = br.to_dataframe()
            for _, row in df.iterrows():
                params = {
                    k.replace("param_", ""): _safe_float(row[k])
                    for k in df.columns if k.startswith("param_")
                }
                rows.append({
                    "key": row.get("key", ""),
                    "gauge_id": gauge_id,
                    "model": row.get("model", ""),
                    "objective": row.get("objective", ""),
                    "algorithm": row.get("algorithm", ""),
                    "transformation": row.get("transformation") or None,
                    "best_objective": _safe_float(row.get("best_objective")),
                    "runtime_seconds": _safe_float(row.get("runtime_seconds")),
                    "success": bool(row.get("success", True)),
                    "parameters": params,
                })
        return rows

    # ── Experiment listing ───────────────────────────────────────────────

    def get_experiments(self, session_id: str, gauge_id: str) -> List[Dict[str, Any]]:
        br = self.get_gauge_batch(session_id, gauge_id)
        diag_df = self._compute_diagnostics_df(session_id, gauge_id)

        experiments: List[Dict[str, Any]] = []
        for key, report in br.results.items():
            parsed = parse_experiment_key(key)
            headline = {}
            if key in diag_df.index:
                for m in HEADLINE_METRICS:
                    headline[m] = _safe_float(diag_df.loc[key, m]) if m in diag_df.columns else None

            experiments.append({
                "key": key,
                "model": parsed.get("model", ""),
                "objective": parsed.get("objective", ""),
                "algorithm": parsed.get("algorithm", ""),
                "transformation": parsed.get("transformation"),
                "best_objective": _safe_float(report.result.best_objective),
                "runtime_seconds": _safe_float(report.result.runtime_seconds),
                "success": report.result.success,
                "headline_metrics": headline,
            })
        return experiments

    # ── Diagnostics ──────────────────────────────────────────────────────

    def _compute_diagnostics_df(self, session_id: str, gauge_id: str) -> pd.DataFrame:
        """Compute (or return cached) diagnostics DataFrame for a gauge."""
        session = self.get_session(session_id)
        cache_key = f"{session_id}:{gauge_id}"
        if cache_key in session._diagnostics_cache:
            return session._diagnostics_cache[cache_key]

        br = self.get_gauge_batch(session_id, gauge_id)
        sdeb_obj = SDEB(alpha=0.1, lam=0.5)
        rows = {}
        for key, report in br.results.items():
            try:
                diag = compute_diagnostics(report.simulated, report.observed)
                diag["SDEB"] = sdeb_obj(report.observed, report.simulated)
                rows[key] = {m: diag.get(m, np.nan) for m in HEADLINE_METRICS}
            except Exception as e:
                logger.warning("Diagnostics failed for %s: %s", key, e)

        df = pd.DataFrame(rows).T
        df.index.name = "experiment_key"
        session._diagnostics_cache[cache_key] = df
        return df

    def _normalise_higher_is_better(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise metrics so higher is always better, then min-max to [0, 1]."""
        norm = df.copy()
        for col in norm.columns:
            if col in ERROR_METRICS:
                norm[col] = -norm[col]
            elif col in VOLUME_METRICS:
                norm[col] = -norm[col].abs()
        for col in norm.columns:
            cmin, cmax = norm[col].min(), norm[col].max()
            if cmax - cmin > 1e-12:
                norm[col] = (norm[col] - cmin) / (cmax - cmin)
            else:
                norm[col] = 0.5
        return norm

    def get_diagnostics(self, session_id: str, gauge_id: str) -> Dict[str, Any]:
        """Return raw + normalised diagnostics with clustermap data."""
        df_raw = self._compute_diagnostics_df(session_id, gauge_id)
        if df_raw.empty:
            return {
                "raw_table": [],
                "normalised_table": [],
                "clustermap": None,
                "top_experiments": [],
                "metric_groups": dict(DIAGNOSTIC_GROUPS),
            }

        df_norm = self._normalise_higher_is_better(df_raw)

        raw_table = [
            {"experiment_key": k, "metrics": {m: _safe_float(v) for m, v in row.items()}}
            for k, row in df_raw.iterrows()
        ]
        norm_table = [
            {"experiment_key": k, "metrics": {m: _safe_float(v) for m, v in row.items()}}
            for k, row in df_norm.iterrows()
        ]

        top_exps = (
            df_norm.mean(axis=1)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"index": "experiment_key", 0: "mean_score"})
        )
        top_experiments = [
            {"experiment_key": r["experiment_key"], "mean_score": _safe_float(r["mean_score"])}
            for _, r in top_exps.iterrows()
        ]

        clustermap = self._build_clustermap_data(df_norm)

        return {
            "raw_table": raw_table,
            "normalised_table": norm_table,
            "clustermap": clustermap,
            "top_experiments": top_experiments,
            "metric_groups": dict(DIAGNOSTIC_GROUPS),
        }

    def _build_clustermap_data(self, df_norm: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Compute Ward linkage and dendrogram data for Plotly rendering."""
        if not SCIPY_AVAILABLE or df_norm.shape[0] < 2:
            return None

        values = df_norm.fillna(0).values

        row_link = linkage(values, method="ward", metric="euclidean")
        col_link = linkage(values.T, method="ward", metric="euclidean")

        row_dend = dendrogram(row_link, no_plot=True)
        col_dend = dendrogram(col_link, no_plot=True)

        row_order = row_dend["leaves"]
        col_order = col_dend["leaves"]

        reordered = values[row_order][:, col_order]
        row_labels = [df_norm.index[i] for i in row_order]
        col_labels = [df_norm.columns[i] for i in col_order]

        annotations = [
            [f"{reordered[r][c]:.2f}" if not np.isnan(reordered[r][c]) else ""
             for c in range(reordered.shape[1])]
            for r in range(reordered.shape[0])
        ]

        def _dend_to_dict(d):
            return {
                "icoord": [[float(x) for x in row] for row in d["icoord"]],
                "dcoord": [[float(x) for x in row] for row in d["dcoord"]],
                "leaves": [int(x) for x in d["leaves"]],
            }

        return {
            "heatmap_values": [
                [_safe_float(reordered[r][c]) for c in range(reordered.shape[1])]
                for r in range(reordered.shape[0])
            ],
            "row_labels": row_labels,
            "col_labels": col_labels,
            "row_dendrogram": _dend_to_dict(row_dend),
            "col_dendrogram": _dend_to_dict(col_dend),
            "annotations": annotations,
        }

    # ── Comparison figures ───────────────────────────────────────────────

    def build_comparison_hydrograph(
        self,
        session_id: str,
        gauge_id: str,
        log_scale: bool = False,
        experiment_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Plotly JSON: observed + N experiment hydrographs."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        br = self.get_gauge_batch(session_id, gauge_id)
        reports = self._select_reports(br, experiment_keys)
        if not reports:
            return {"data": [], "layout": {}}

        ref_key, ref_report = next(iter(reports.items()))
        dates = [d.isoformat() for d in ref_report.dates]

        has_precip = (
            ref_report.precipitation is not None
            and len(ref_report.precipitation) > 0
            and float(np.nanmax(ref_report.precipitation)) > 0
        )

        if has_precip:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.18, 0.82],
                shared_xaxes=True,
                vertical_spacing=0.02,
            )
            fig.add_trace(go.Scatter(
                x=dates,
                y=ref_report.precipitation.tolist(),
                name="Precipitation",
                fill="tozeroy",
                fillcolor="rgba(70,130,180,0.4)",
                line=dict(color="steelblue", width=0.5),
                mode="lines",
                showlegend=False,
            ), row=1, col=1)
            fig.update_yaxes(
                title_text="P (mm)", autorange="reversed",
                fixedrange=False, row=1, col=1,
            )
            flow_row = 2
        else:
            fig = go.Figure()
            flow_row = None

        obs_trace = go.Scatter(
            x=dates,
            y=ref_report.observed.tolist(),
            name="Observed",
            mode="lines",
            line=dict(color="black", width=2),
        )
        if flow_row:
            fig.add_trace(obs_trace, row=flow_row, col=1)
        else:
            fig.add_trace(obs_trace)

        for i, (key, report) in enumerate(reports.items()):
            short = self._short_label(key)
            trace = go.Scatter(
                x=[d.isoformat() for d in report.dates],
                y=report.simulated.tolist(),
                name=short,
                mode="lines",
                line=dict(color=PALETTE[i % len(PALETTE)], width=1),
            )
            if flow_row:
                fig.add_trace(trace, row=flow_row, col=1)
            else:
                fig.add_trace(trace)

        fig.update_layout(
            height=600 if has_precip else 500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        if flow_row:
            fig.update_yaxes(
                title_text="Flow",
                type="log" if log_scale else "linear",
                autorange=True,
                fixedrange=False,
                row=flow_row, col=1,
            )
            fig.update_xaxes(
                type="date", fixedrange=False,
                row=flow_row, col=1,
            )
            fig.update_xaxes(fixedrange=False, row=1, col=1)
        else:
            fig.update_yaxes(
                title_text="Flow",
                type="log" if log_scale else "linear",
                autorange=True,
                fixedrange=False,
            )
            fig.update_xaxes(type="date", fixedrange=False)

        return json.loads(fig.to_json())

    def build_comparison_fdc(
        self,
        session_id: str,
        gauge_id: str,
        log_scale: bool = True,
        experiment_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Plotly JSON: observed FDC + N experiment FDCs."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        br = self.get_gauge_batch(session_id, gauge_id)
        reports = self._select_reports(br, experiment_keys)
        if not reports:
            return {"data": [], "layout": {}}

        ref_report = next(iter(reports.values()))
        fig = go.Figure()

        obs_sorted = np.sort(ref_report.observed)[::-1]
        exc = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
        fig.add_trace(go.Scatter(
            x=exc.tolist(), y=obs_sorted.tolist(),
            name="Observed", mode="lines",
            line=dict(color="black", width=2),
        ))

        for i, (key, report) in enumerate(reports.items()):
            sim_sorted = np.sort(report.simulated)[::-1]
            sim_exc = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
            fig.add_trace(go.Scatter(
                x=sim_exc.tolist(), y=sim_sorted.tolist(),
                name=self._short_label(key), mode="lines",
                line=dict(color=PALETTE[i % len(PALETTE)], width=1),
            ))

        # FHV / FMV / FLV reference zones
        fig.add_vrect(x0=0, x1=2, fillcolor="red", opacity=0.05,
                       annotation_text="FHV", annotation_position="top left")
        fig.add_vrect(x0=20, x1=70, fillcolor="blue", opacity=0.03,
                       annotation_text="FMV", annotation_position="top left")
        fig.add_vrect(x0=70, x1=100, fillcolor="green", opacity=0.05,
                       annotation_text="FLV", annotation_position="top left")

        fig.update_layout(
            xaxis_title="Exceedance Probability (%)",
            yaxis_title="Flow",
            yaxis_type="log" if log_scale else "linear",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        return json.loads(fig.to_json())

    def build_comparison_scatter(
        self,
        session_id: str,
        gauge_id: str,
        log_scale: bool = False,
        experiment_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Plotly JSON: all experiments overlaid on a single obs-vs-sim scatter."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        br = self.get_gauge_batch(session_id, gauge_id)
        reports = self._select_reports(br, experiment_keys)
        if not reports:
            return {"data": [], "layout": {}}

        fig = go.Figure()

        all_vals: list = []
        for report in reports.values():
            all_vals.extend(report.observed.tolist())
            all_vals.extend(report.simulated.tolist())
        global_max = max(all_vals) if all_vals else 1
        positive_vals = [v for v in all_vals if v > 0]
        global_min_pos = min(positive_vals) if positive_vals else 0.01

        for idx, (key, report) in enumerate(reports.items()):
            fig.add_trace(go.Scatter(
                x=report.observed.tolist(),
                y=report.simulated.tolist(),
                mode="markers",
                marker=dict(color=PALETTE[idx % len(PALETTE)], size=3, opacity=0.4),
                name=self._short_label(key),
            ))

        if log_scale:
            line_min = global_min_pos * 0.5
            line_max = global_max * 1.5
        else:
            line_min = 0
            line_max = global_max
        fig.add_trace(go.Scatter(
            x=[line_min, line_max], y=[line_min, line_max],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="1:1",
            showlegend=False,
        ))

        axis_type = "log" if log_scale else "linear"
        fig.update_layout(
            height=500,
            xaxis_title="Observed",
            yaxis_title="Simulated",
            xaxis_type=axis_type,
            yaxis_type=axis_type,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=20, t=50, b=50),
        )
        if not log_scale:
            fig.update_xaxes(range=[0, global_max * 1.05])
            fig.update_yaxes(range=[0, global_max * 1.05])

        return json.loads(fig.to_json())

    # ── Individual report card ───────────────────────────────────────────

    def get_report_card(self, session_id: str, gauge_id: str, exp_key: str) -> Dict[str, Any]:
        """
        Return separate Plotly JSON figures for a single experiment's report card.
        
        Returns a dictionary with independent figures that can be rendered reliably:
        - header: Metadata dict (not a figure)
        - hydrograph_linear: Plotly JSON
        - hydrograph_log: Plotly JSON
        - metrics_table: Plotly JSON
        - fdc: Plotly JSON
        - scatter: Plotly JSON
        - parameters: Plotly JSON (may be None)
        - signatures_table: Plotly JSON (may be None)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        br = self.get_gauge_batch(session_id, gauge_id)
        if exp_key not in br.results:
            raise KeyError(f"Experiment {exp_key} not found")

        report = br.results[exp_key]

        from pyrrm.visualization.report_plots import plot_report_card_components
        components = plot_report_card_components(report)
        
        result = {"header": components["header"]}
        for key in ["hydrograph_linear", "hydrograph_log", "metrics_table", "fdc", "scatter", "parameters", "signatures_table"]:
            fig = components.get(key)
            result[key] = json.loads(fig.to_json()) if fig is not None else None
        
        return result

    # ── Report card export ─────────────────────────────────────────────────

    def export_report_card(
        self,
        session_id: str,
        gauge_id: str,
        exp_key: str,
        format: str = "pdf",
        sections: Optional[List[str]] = None,
    ) -> tuple:
        """
        Export a single experiment's report card.
        
        Args:
            session_id: Session identifier
            gauge_id: Gauge identifier
            exp_key: Experiment key
            format: 'pdf' (landscape single-page), 'html' (landscape static), 
                    or 'interactive' (HTML with interactive Plotly figures)
            sections: List of section IDs to include (None = all)
        
        Returns:
            Tuple of (content_bytes, filename, media_type)
        """
        br = self.get_gauge_batch(session_id, gauge_id)
        if exp_key not in br.results:
            raise KeyError(f"Experiment {exp_key} not found")
        
        report = br.results[exp_key]
        
        # Build filename
        catchment = report.catchment_info.get('name', 'unknown').replace(' ', '_')
        gauge = report.catchment_info.get('gauge_id', gauge_id).replace(' ', '_')
        safe_exp_key = exp_key.replace('/', '_').replace(' ', '_')[:50]
        
        if format == "pdf":
            # Landscape single-page PDF
            from pyrrm.visualization.report_export import export_report_card_landscape_pdf
            content = export_report_card_landscape_pdf(report)
            filename = f"{catchment}_{gauge}_{safe_exp_key}_report.pdf"
            media_type = "application/pdf"
        elif format == "interactive":
            # Interactive HTML with Plotly
            from pyrrm.visualization.report_export import export_report_card_interactive_html
            content = export_report_card_interactive_html(report).encode('utf-8')
            filename = f"{catchment}_{gauge}_{safe_exp_key}_report_interactive.html"
            media_type = "text/html; charset=utf-8"
        else:
            # Static HTML (landscape)
            from pyrrm.visualization.report_export import export_report_card_landscape_html
            content = export_report_card_landscape_html(report).encode('utf-8')
            filename = f"{catchment}_{gauge}_{safe_exp_key}_report.html"
            media_type = "text/html; charset=utf-8"
        
        return content, filename, media_type

    def export_batch_report_card(
        self,
        session_id: str,
        gauge_id: str,
        experiment_keys: Optional[List[str]] = None,
        format: str = "pdf",
        sections: Optional[List[str]] = None,
    ) -> tuple:
        """
        Export multiple experiments' report cards as a single document.
        
        Args:
            session_id: Session identifier
            gauge_id: Gauge identifier
            experiment_keys: List of experiment keys (None = all)
            format: 'pdf' (landscape, one page per experiment), 
                    'html' (landscape static), or 'interactive'
            sections: List of section IDs to include (None = all)
        
        Returns:
            Tuple of (content_bytes, filename, media_type)
        """
        br = self.get_gauge_batch(session_id, gauge_id)
        reports = self._select_reports(br, experiment_keys)
        
        if not reports:
            raise KeyError("No experiments found")
        
        # Build filename and title
        first_report = next(iter(reports.values()))
        catchment = first_report.catchment_info.get('name', 'unknown').replace(' ', '_')
        gauge = first_report.catchment_info.get('gauge_id', gauge_id).replace(' ', '_')
        title = f"Batch Calibration Report - {catchment} ({gauge})"
        
        if format == "pdf":
            # Landscape PDF with one page per experiment
            from pyrrm.visualization.report_export import export_batch_report_landscape_pdf
            content = export_batch_report_landscape_pdf(reports, title=title)
            filename = f"{catchment}_{gauge}_batch_report.pdf"
            media_type = "application/pdf"
        elif format == "interactive":
            # Concatenate interactive HTML reports
            from pyrrm.visualization.report_export import export_report_card_interactive_html
            pages = []
            for i, (exp_key, report) in enumerate(reports.items()):
                page_html = export_report_card_interactive_html(report)
                if i > 0:
                    pages.append('<hr style="margin: 30px 0; border: none; border-top: 3px solid #1a5276;">')
                # Extract body content and add
                import re
                body_match = re.search(r'<body>(.*?)</body>', page_html, re.DOTALL)
                if body_match:
                    pages.append(f'<div style="margin-bottom: 40px;">{body_match.group(1)}</div>')
            
            # Wrap in full HTML document
            from datetime import datetime
            generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; background: #f5f6fa; padding: 20px; }}
        h1 {{ color: #1a5276; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="subtitle">{len(reports)} experiments • Generated: {generated_date}</div>
    {''.join(pages)}
</body>
</html>
"""
            content = combined_html.encode('utf-8')
            filename = f"{catchment}_{gauge}_batch_report_interactive.html"
            media_type = "text/html; charset=utf-8"
        else:
            # Static landscape HTML with page breaks
            from pyrrm.visualization.report_export import export_report_card_landscape_html, REPORT_CSS_LANDSCAPE
            pages = []
            for i, (exp_key, report) in enumerate(reports.items()):
                page_html = export_report_card_landscape_html(report, include_css=False)
                import re
                body_match = re.search(r'<body>(.*?)</body>', page_html, re.DOTALL)
                if body_match:
                    style = 'page-break-before: always;' if i > 0 else ''
                    pages.append(f'<div style="{style}">{body_match.group(1)}</div>')
            
            from datetime import datetime
            generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{REPORT_CSS_LANDSCAPE}</style>
</head>
<body>
    {''.join(pages)}
</body>
</html>
"""
            content = combined_html.encode('utf-8')
            filename = f"{catchment}_{gauge}_batch_report.html"
            media_type = "text/html; charset=utf-8"
        
        return content, filename, media_type

    # ── Private helpers ──────────────────────────────────────────────────

    def _select_reports(
        self, br: BatchResult, experiment_keys: Optional[List[str]] = None
    ) -> Dict[str, CalibrationReport]:
        if experiment_keys:
            return OrderedDict(
                (k, br.results[k]) for k in experiment_keys if k in br.results
            )
        return OrderedDict(br.results)

    @staticmethod
    def _short_label(key: str) -> str:
        """Create a compact label from an experiment key."""
        parsed = parse_experiment_key(key)
        parts = [parsed.get("model", ""), parsed.get("objective", "")]
        if parsed.get("transformation"):
            parts.append(parsed["transformation"])
        parts.append(parsed.get("algorithm", ""))
        return "_".join(p for p in parts if p)

    # ── Signature comparison ──────────────────────────────────────────────

    def build_signature_comparison(
        self,
        session_id: str,
        gauge_id: str,
        experiment_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build signature comparison data for Plotly rendering.

        Computes hydrological signatures for observed flow and selected
        experiments, returning category-organized data plus pre-built
        Plotly figures for radar charts, bar charts, and a summary heatmap.

        Args:
            session_id: Analysis session identifier.
            gauge_id: Gauge/catchment identifier.
            experiment_keys: Optional list of experiment keys to include.
                If None, uses all experiments.

        Returns:
            Dictionary with structure:
            {
                "categories": {
                    "Magnitude": {
                        "signatures": ["Q_mean", "Q_median", ...],
                        "observed": {"Q_mean": 1.23, ...},
                        "experiments": {"exp_key_1": {"Q_mean": 1.25, ...}, ...},
                        "percent_errors": {"exp_key_1": {"Q_mean": 1.6, ...}, ...}
                    },
                    ...
                },
                "bar_figures": {...},    # Plotly JSON per category
                "heatmap_figure": {...}  # Overall signature error heatmap
            }
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for signature comparison")

        br = self.get_gauge_batch(session_id, gauge_id)
        reports = self._select_reports(br, experiment_keys)

        if not reports:
            return {
                "categories": {},
                "bar_figures": {},
                "heatmap_figure": None,
            }

        ref_report = next(iter(reports.values()))
        obs = ref_report.observed
        dates = ref_report.dates

        dates_index = None
        if dates is not None:
            dates_index = pd.DatetimeIndex(dates)

        obs_sigs = compute_all_signatures(obs, dates_index)

        exp_sigs: Dict[str, Dict[str, float]] = {}
        for key, report in reports.items():
            exp_sigs[key] = compute_all_signatures(report.simulated, dates_index)

        categories_data: Dict[str, Dict[str, Any]] = {}
        for cat_name, sig_names in SIGNATURE_CATEGORIES.items():
            obs_cat = {s: _safe_float(obs_sigs.get(s)) for s in sig_names}

            exp_cat: Dict[str, Dict[str, Optional[float]]] = {}
            pct_err_cat: Dict[str, Dict[str, Optional[float]]] = {}

            for exp_key, sigs in exp_sigs.items():
                exp_cat[exp_key] = {s: _safe_float(sigs.get(s)) for s in sig_names}
                pct_err_cat[exp_key] = {}
                for s in sig_names:
                    obs_val = obs_sigs.get(s, np.nan)
                    sim_val = sigs.get(s, np.nan)
                    pct_err_cat[exp_key][s] = _safe_float(
                        signature_percent_error(obs_val, sim_val)
                    )

            categories_data[cat_name] = {
                "signatures": sig_names,
                "observed": obs_cat,
                "experiments": exp_cat,
                "percent_errors": pct_err_cat,
            }

        bar_figures = self._build_signature_bar_figures(categories_data, reports)
        heatmap_figure = self._build_signature_heatmap(categories_data, reports)

        return {
            "categories": categories_data,
            "bar_figures": bar_figures,
            "heatmap_figure": heatmap_figure,
        }

    def _build_signature_bar_figures(
        self,
        categories_data: Dict[str, Dict[str, Any]],
        reports: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build grouped bar charts showing percent difference from observed.
        
        Bars are centered at zero, representing how much each experiment
        deviates from the observed value as a percentage.
        """
        bar_figures: Dict[str, Dict[str, Any]] = {}

        for cat_name, cat_data in categories_data.items():
            sig_names = cat_data["signatures"]
            pct_errors = cat_data["percent_errors"]

            fig = go.Figure()

            for idx, (exp_key, errs) in enumerate(pct_errors.items()):
                exp_y = []
                for s in sig_names:
                    err_val = errs.get(s)
                    if err_val is not None and np.isfinite(err_val):
                        capped = max(-100, min(100, err_val))
                        exp_y.append(capped)
                    else:
                        exp_y.append(None)
                
                fig.add_trace(go.Bar(
                    x=sig_names,
                    y=exp_y,
                    name=self._short_label(exp_key),
                    marker_color=PALETTE[idx % len(PALETTE)],
                ))

            fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid")

            num_experiments = len(pct_errors)
            legend_rows = (num_experiments + 3) // 4  # ~4 items per row
            legend_height = max(30, legend_rows * 25)
            top_margin = 60 + legend_height

            fig.update_layout(
                barmode="group",
                xaxis_title="Signature",
                yaxis_title="% Difference from Observed",
                yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="black"),
                height=400 + legend_height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.0,
                    xanchor="center",
                    x=0.5,
                    traceorder="normal",
                    itemwidth=30,
                    font=dict(size=10),
                ),
                margin=dict(l=60, r=20, t=top_margin, b=80),
                title=dict(
                    text=f"{cat_name} (% difference from observed)",
                    x=0.5,
                    y=0.98,
                    yanchor="top",
                ),
            )

            bar_figures[cat_name] = json.loads(fig.to_json())

        return bar_figures

    def _build_signature_heatmap(
        self,
        categories_data: Dict[str, Dict[str, Any]],
        reports: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build percent error heatmap across all signatures and experiments.
        
        Signatures are on rows (y-axis), experiments on columns (x-axis) for
        better readability with many signatures.
        """
        all_sigs: List[str] = []
        sig_to_cat: Dict[str, str] = {}
        for cat_name, cat_data in categories_data.items():
            for s in cat_data["signatures"]:
                all_sigs.append(s)
                sig_to_cat[s] = cat_name

        exp_keys = list(reports.keys())
        if not exp_keys:
            return None

        z_values: List[List[Optional[float]]] = []
        for sig in all_sigs:
            row: List[Optional[float]] = []
            for exp_key in exp_keys:
                cat = sig_to_cat[sig]
                pct_err = categories_data[cat]["percent_errors"].get(exp_key, {}).get(sig)
                row.append(pct_err)
            z_values.append(row)

        short_labels = [self._short_label(k) for k in exp_keys]

        annotations = []
        for i, sig in enumerate(all_sigs):
            for j, exp_key in enumerate(exp_keys):
                val = z_values[i][j]
                text = f"{val:.0f}%" if val is not None else ""
                annotations.append(dict(
                    x=short_labels[j],
                    y=sig,
                    text=text,
                    showarrow=False,
                    font=dict(size=8),
                ))

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=short_labels,
            y=all_sigs,
            colorscale=[
                [0.0, "darkblue"],
                [0.25, "lightblue"],
                [0.5, "green"],
                [0.75, "lightsalmon"],
                [1.0, "darkred"],
            ],
            zmid=0,
            zmin=-100,
            zmax=100,
            colorbar=dict(title="% Error"),
        ))

        num_sigs = len(all_sigs)
        num_exps = len(exp_keys)
        col_width = max(40, min(80, 600 // max(1, num_exps)))
        fig.update_layout(
            title="Signature Percent Errors",
            xaxis_title="Experiment",
            yaxis_title="Signature",
            height=max(400, 18 * num_sigs + 100),
            width=max(400, col_width * num_exps + 180),
            margin=dict(l=120, r=60, t=50, b=100),
            xaxis=dict(tickangle=45, side="bottom"),
            yaxis=dict(autorange="reversed"),
            annotations=annotations,
        )

        return json.loads(fig.to_json())

    # ── Signature reference ─────────────────────────────────────────────────

    def get_signature_reference(self) -> Dict[str, Any]:
        """
        Return complete signature reference documentation.
        
        Returns metadata for all signatures organized by category,
        with JSON-serializable range values.
        """
        def serialize_range(r):
            """Convert numpy inf values to strings for JSON serialization."""
            if r is None:
                return None
            return [
                "inf" if v == np.inf else "-inf" if v == -np.inf else v
                for v in r
            ]
        
        categories_with_info: Dict[str, List[Dict[str, Any]]] = {}
        
        for cat_name, sig_ids in SIGNATURE_CATEGORIES.items():
            cat_signatures = []
            for sig_id in sig_ids:
                info = SIGNATURE_INFO.get(sig_id, {})
                cat_signatures.append({
                    "id": sig_id,
                    "name": info.get("name", sig_id),
                    "category": cat_name,
                    "units": info.get("units", ""),
                    "range": serialize_range(info.get("range")),
                    "description": info.get("description", ""),
                    "formula": info.get("formula", ""),
                    "interpretation": info.get("interpretation", ""),
                    "related": info.get("related", []),
                    "references": info.get("references", []),
                })
            categories_with_info[cat_name] = cat_signatures
        
        return {
            "categories": categories_with_info,
            "category_order": list(SIGNATURE_CATEGORIES.keys()),
            "total_signatures": sum(len(sigs) for sigs in SIGNATURE_CATEGORIES.values()),
        }


# Module-level singleton
analysis_service = AnalysisService()
