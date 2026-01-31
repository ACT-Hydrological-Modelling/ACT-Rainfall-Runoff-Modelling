"""
Pydantic schemas for Experiment API.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator

from app.models.experiment import ExperimentStatus, ModelType


class CalibrationPeriod(BaseModel):
    """Calibration period configuration."""
    start_date: date = Field(..., description="Start date of calibration period")
    end_date: date = Field(..., description="End date of calibration period")
    warmup_days: int = Field(365, ge=0, description="Number of warmup days")


class FlowThreshold(BaseModel):
    """Flow threshold configuration."""
    type: str = Field(..., description="Type: 'absolute' or 'percentile'")
    value: float = Field(..., description="Threshold value")


class FlowTrimmingConfig(BaseModel):
    """Flow range trimming configuration for calibration."""
    enabled: bool = Field(False, description="Enable flow trimming")
    min_threshold: Optional[FlowThreshold] = Field(
        None, 
        description="Minimum flow threshold (exclude flows below)"
    )
    max_threshold: Optional[FlowThreshold] = Field(
        None, 
        description="Maximum flow threshold (exclude flows above)"
    )


class ObjectiveConfig(BaseModel):
    """Objective function configuration."""
    type: str = Field("NSE", description="Objective function type")
    transform: str = Field("none", description="Flow transformation (none, log, sqrt, inverse)")
    weights: Optional[Dict[str, float]] = Field(
        None, 
        description="Weights for composite objectives"
    )
    flow_trimming: Optional[FlowTrimmingConfig] = Field(
        None,
        description="Flow range trimming configuration"
    )


class AlgorithmConfig(BaseModel):
    """Calibration algorithm configuration."""
    method: str = Field("sceua_direct", description="Calibration method")
    max_evals: int = Field(50000, ge=1000, description="Maximum function evaluations")
    n_complexes: Optional[int] = Field(None, description="Number of complexes (SCE-UA)")
    max_workers: int = Field(1, ge=1, description="Number of parallel workers")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    checkpoint_interval: int = Field(5000, ge=0, description="Checkpoint interval")
    # Convergence criteria - defaults match tutorial notebooks
    max_tolerant_iter: int = Field(
        100, 
        ge=10, 
        le=500,
        description="Max iterations without improvement before stopping (default: 100)"
    )
    tolerance: float = Field(
        1e-4, 
        ge=1e-8, 
        le=1e-2,
        description="Improvement threshold for convergence (default: 1e-4)"
    )


class RoutingConfig(BaseModel):
    """Muskingum routing configuration."""
    enabled: bool = Field(False, description="Enable routing")
    K: float = Field(5.0, ge=0.1, le=200.0, description="Storage constant (days)")
    m: float = Field(0.8, ge=0.3, le=1.5, description="Nonlinear exponent")
    n_subreaches: int = Field(3, ge=1, le=20, description="Number of sub-reaches")
    calibrate_routing: bool = Field(True, description="Include routing params in calibration")


class ModelSettings(BaseModel):
    """Model-specific configuration."""
    routing: Optional[RoutingConfig] = None
    initial_states: Optional[Dict[str, float]] = None


class ExperimentBase(BaseModel):
    """Base schema for experiment data."""
    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: Optional[str] = Field(None, description="Description")
    model_type: ModelType = Field(ModelType.SACRAMENTO, description="Model type")
    model_settings: Optional[ModelSettings] = Field(None, description="Model configuration")
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Parameter bounds as {param: [min, max]}"
    )
    calibration_period: Optional[CalibrationPeriod] = Field(
        None,
        description="Calibration period configuration"
    )
    objective_config: Optional[ObjectiveConfig] = Field(
        None,
        description="Objective function configuration"
    )
    algorithm_config: Optional[AlgorithmConfig] = Field(
        None,
        description="Algorithm configuration"
    )


class ExperimentCreate(ExperimentBase):
    """Schema for creating a new experiment."""
    catchment_id: str = Field(..., description="Parent catchment ID")


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment (all fields optional)."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    model_type: Optional[ModelType] = None
    model_settings: Optional[ModelSettings] = None
    parameter_bounds: Optional[Dict[str, List[float]]] = None
    calibration_period: Optional[CalibrationPeriod] = None
    objective_config: Optional[ObjectiveConfig] = None
    algorithm_config: Optional[AlgorithmConfig] = None


class ExperimentResponse(ExperimentBase):
    """Schema for experiment response."""
    id: str
    catchment_id: str
    status: ExperimentStatus
    celery_task_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    runtime_seconds: Optional[float]
    
    # Include result summary if completed
    has_result: bool = False
    best_objective: Optional[float] = None
    
    class Config:
        from_attributes = True


class ExperimentListResponse(BaseModel):
    """Schema for listing experiments."""
    id: str
    catchment_id: str
    name: str
    model_type: ModelType
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    best_objective: Optional[float] = None
    
    class Config:
        from_attributes = True


class ExperimentStatusResponse(BaseModel):
    """Schema for experiment status check."""
    id: str
    status: ExperimentStatus
    progress: Optional[Dict[str, Any]] = Field(
        None,
        description="Progress info (iteration, best_value, etc.)"
    )
    error_message: Optional[str] = None
