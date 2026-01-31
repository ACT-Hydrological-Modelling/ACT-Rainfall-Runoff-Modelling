"""
Pydantic schemas for API request/response validation.
"""

from app.schemas.catchment import (
    CatchmentCreate,
    CatchmentUpdate,
    CatchmentResponse,
    CatchmentListResponse,
)
from app.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    DatasetPreview,
)
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentListResponse,
    ModelSettings,
    CalibrationPeriod,
    ObjectiveConfig,
    AlgorithmConfig,
)
from app.schemas.result import (
    CalibrationResultResponse,
    ParameterResponse,
    MetricsResponse,
    TimeSeriesResponse,
)

__all__ = [
    # Catchment
    "CatchmentCreate",
    "CatchmentUpdate",
    "CatchmentResponse",
    "CatchmentListResponse",
    # Dataset
    "DatasetCreate",
    "DatasetResponse",
    "DatasetPreview",
    # Experiment
    "ExperimentCreate",
    "ExperimentUpdate",
    "ExperimentResponse",
    "ExperimentListResponse",
    "ModelSettings",
    "CalibrationPeriod",
    "ObjectiveConfig",
    "AlgorithmConfig",
    # Result
    "CalibrationResultResponse",
    "ParameterResponse",
    "MetricsResponse",
    "TimeSeriesResponse",
]
