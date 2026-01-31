"""
Pydantic schemas for Dataset API.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.models.dataset import DatasetType


class DatasetCreate(BaseModel):
    """Schema for creating a dataset (used with file upload)."""
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    type: DatasetType = Field(..., description="Type of data (rainfall, pet, observed_flow)")


class DatasetResponse(BaseModel):
    """Schema for dataset response."""
    id: str
    catchment_id: str
    name: str
    type: DatasetType
    file_path: str
    start_date: Optional[date]
    end_date: Optional[date]
    record_count: Optional[int]
    extra_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    """Schema for dataset preview with statistics."""
    id: str
    name: str
    type: DatasetType
    start_date: Optional[date]
    end_date: Optional[date]
    record_count: int
    
    # Statistics
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Summary statistics (mean, std, min, max, etc.)"
    )
    
    # Sample data
    sample_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First N rows of data"
    )
    
    # Missing data info
    missing_count: int = 0
    missing_percentage: float = 0.0
    
    # Validation messages
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class DatasetValidation(BaseModel):
    """Schema for dataset validation result."""
    is_valid: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    detected_columns: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of detected columns to standard names"
    )
