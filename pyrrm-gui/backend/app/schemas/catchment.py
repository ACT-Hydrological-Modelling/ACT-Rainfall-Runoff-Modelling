"""
Pydantic schemas for Catchment API.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class CatchmentBase(BaseModel):
    """Base schema for catchment data."""
    name: str = Field(..., min_length=1, max_length=255, description="Catchment name")
    gauge_id: Optional[str] = Field(None, max_length=50, description="Gauge station ID")
    area_km2: Optional[float] = Field(None, gt=0, description="Catchment area in km²")
    description: Optional[str] = Field(None, description="Description and notes")


class CatchmentCreate(CatchmentBase):
    """Schema for creating a new catchment."""
    pass


class CatchmentUpdate(BaseModel):
    """Schema for updating a catchment (all fields optional)."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    gauge_id: Optional[str] = Field(None, max_length=50)
    area_km2: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None


class DatasetSummary(BaseModel):
    """Summary of a dataset for listing."""
    id: str
    name: str
    type: str
    record_count: Optional[int]
    
    class Config:
        from_attributes = True


class ExperimentSummary(BaseModel):
    """Summary of an experiment for listing."""
    id: str
    name: str
    status: str
    model_type: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class CatchmentResponse(CatchmentBase):
    """Schema for catchment response with all details."""
    id: str
    created_at: datetime
    updated_at: datetime
    datasets: List[DatasetSummary] = []
    experiments: List[ExperimentSummary] = []
    
    class Config:
        from_attributes = True


class CatchmentListResponse(BaseModel):
    """Schema for listing catchments."""
    id: str
    name: str
    gauge_id: Optional[str]
    area_km2: Optional[float]
    created_at: datetime
    dataset_count: int = 0
    experiment_count: int = 0
    
    class Config:
        from_attributes = True
