"""
Dataset database model.

A dataset represents a time series of hydrological data (rainfall, PET, or observed flow)
associated with a catchment.
"""

import uuid
from datetime import datetime, date
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, Date, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum

from app.database import Base


class DatasetType(str, Enum):
    """Type of hydrological dataset."""
    RAINFALL = "rainfall"
    PET = "pet"
    OBSERVED_FLOW = "observed_flow"


class Dataset(Base):
    """
    Represents a hydrological time series dataset.
    
    Attributes:
        id: Unique identifier (UUID)
        catchment_id: Foreign key to parent catchment
        name: Human-readable name (e.g., "SILO rainfall 1985-2024")
        type: Type of data (rainfall, pet, observed_flow)
        file_path: Path to the uploaded CSV file
        start_date: First date in the dataset
        end_date: Last date in the dataset
        record_count: Number of records
        metadata: Additional metadata (source, units, etc.)
        created_at: Timestamp when created
    """
    
    __tablename__ = "datasets"
    __allow_unmapped__ = True
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    catchment_id = Column(String(36), ForeignKey("catchments.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    type = Column(SQLEnum(DatasetType), nullable=False)
    file_path = Column(String(500), nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    record_count = Column(Integer, nullable=True)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    catchment = relationship("Catchment", back_populates="datasets")
    
    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', type={self.type.value})>"
    
    @property
    def date_range_str(self) -> str:
        """Return formatted date range string."""
        if self.start_date and self.end_date:
            return f"{self.start_date} to {self.end_date}"
        return "Unknown"
