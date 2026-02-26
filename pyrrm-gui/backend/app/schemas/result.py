"""
Pydantic schemas for CalibrationResult API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ParameterResponse(BaseModel):
    """Schema for parameter information."""
    name: str
    value: float
    min_bound: float
    max_bound: float
    percent_of_range: float = Field(
        ..., 
        description="Parameter value as percentage of allowed range"
    )
    description: Optional[str] = None
    unit: Optional[str] = None


class MetricsResponse(BaseModel):
    """Schema for performance metrics."""
    NSE: Optional[float] = None
    KGE: Optional[float] = None
    RMSE: Optional[float] = None
    MAE: Optional[float] = None
    PBIAS: Optional[float] = None
    NSE_log: Optional[float] = Field(None, alias="NSE (log Q)")
    NSE_inv: Optional[float] = Field(None, alias="NSE (1/Q)")
    NSE_sqrt: Optional[float] = Field(None, alias="NSE (√Q)")
    
    # KGE components
    KGE_r: Optional[float] = None
    KGE_alpha: Optional[float] = None
    KGE_beta: Optional[float] = None
    
    class Config:
        populate_by_name = True


class TimeSeriesPoint(BaseModel):
    """Single point in a time series."""
    date: str
    observed: Optional[float]
    simulated: Optional[float]


class TimeSeriesResponse(BaseModel):
    """Schema for time series data."""
    dates: List[str]
    observed: List[Optional[float]]
    simulated: List[Optional[float]]
    precipitation: Optional[List[Optional[float]]] = None
    
    # Summary stats
    n_points: int
    start_date: str
    end_date: str


class FDCPoint(BaseModel):
    """Point on flow duration curve."""
    exceedance: float
    observed: float
    simulated: float


class FlowDurationCurveResponse(BaseModel):
    """Schema for flow duration curve data."""
    points: List[FDCPoint]
    n_points: int = 100


class CalibrationResultResponse(BaseModel):
    """Schema for full calibration result."""
    id: str
    experiment_id: str
    
    # Best result
    best_parameters: Dict[str, float]
    best_objective: float
    
    # Metrics
    metrics: MetricsResponse
    
    # Execution info
    runtime_seconds: Optional[float]
    iterations_completed: Optional[int]
    
    # File paths
    report_file_path: Optional[str]
    
    # Timestamps
    created_at: datetime
    
    class Config:
        from_attributes = True


class SimulationRequest(BaseModel):
    """Schema for running a simulation with custom parameters."""
    parameters: Dict[str, float] = Field(
        ...,
        description="Parameter values to use for simulation"
    )


class SimulationResponse(BaseModel):
    """Schema for simulation result."""
    parameters: Dict[str, float]
    metrics: MetricsResponse
    timeseries: TimeSeriesResponse
