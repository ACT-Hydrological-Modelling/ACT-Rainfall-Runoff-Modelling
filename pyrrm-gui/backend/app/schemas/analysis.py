"""
Pydantic schemas for the Batch Analysis API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LoadSessionRequest(BaseModel):
    """Request to load batch results from a folder."""
    folder_path: str = Field(
        ...,
        description="Absolute path to folder containing batch_result.pkl files"
    )
    name: Optional[str] = Field(
        None,
        description="Human-readable session name; defaults to folder basename"
    )


class SessionSummary(BaseModel):
    """Summary of a loaded analysis session."""
    id: str
    name: str
    folder_path: str
    gauge_ids: List[str]
    total_experiments: int
    total_failures: int
    loaded_at: str


class GaugeMetadata(BaseModel):
    """Catchment / gauge physical metadata extracted from CalibrationReports."""
    area_km2: Optional[float] = None
    gauge_name: Optional[str] = None
    record_start: Optional[str] = None
    record_end: Optional[str] = None
    record_years: Optional[float] = None
    n_days: Optional[int] = None
    mean_precip_mm_day: Optional[float] = None
    mean_pet_mm_day: Optional[float] = None
    total_precip_mm_yr: Optional[float] = None
    total_pet_mm_yr: Optional[float] = None
    mean_flow: Optional[float] = None
    median_flow: Optional[float] = None
    aridity_index: Optional[float] = None
    runoff_ratio: Optional[float] = None


class GaugeSummary(BaseModel):
    """Summary of a single gauge within a session."""
    gauge_id: str
    n_experiments: int
    n_failures: int
    best_by_objective: Dict[str, Dict[str, Any]]
    metadata: Optional[GaugeMetadata] = None


class ExperimentInfo(BaseModel):
    """Parsed experiment information with headline metrics."""
    key: str
    model: str
    objective: str
    algorithm: str
    transformation: Optional[str] = None
    best_objective: float
    runtime_seconds: Optional[float] = None
    success: bool = True
    headline_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)


class DiagnosticsRow(BaseModel):
    """Single row of the diagnostics table (one experiment)."""
    experiment_key: str
    metrics: Dict[str, Optional[float]]


class ClustermapData(BaseModel):
    """Data needed to render a clustermap in Plotly."""
    heatmap_values: List[List[Optional[float]]]
    row_labels: List[str]
    col_labels: List[str]
    row_dendrogram: Dict[str, Any]
    col_dendrogram: Dict[str, Any]
    annotations: List[List[Optional[str]]]


class DiagnosticsResponse(BaseModel):
    """Full diagnostics response for a gauge."""
    raw_table: List[DiagnosticsRow]
    normalised_table: List[DiagnosticsRow]
    clustermap: ClustermapData
    top_experiments: List[Dict[str, Any]]
    metric_groups: Dict[str, List[str]]


class SessionDetailResponse(BaseModel):
    """Detailed session info."""
    id: str
    name: str
    folder_path: str
    gauge_ids: List[str]
    total_experiments: int
    total_failures: int
    loaded_at: str
    gauges: List[GaugeSummary]


class SummaryTableRow(BaseModel):
    """Row in the combined summary table."""
    key: str
    gauge_id: str
    model: str
    objective: str
    algorithm: str
    transformation: Optional[str] = None
    best_objective: Optional[float] = None
    runtime_seconds: Optional[float] = None
    success: bool = True
    parameters: Dict[str, Optional[float]] = Field(default_factory=dict)
