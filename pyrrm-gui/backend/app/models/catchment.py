"""
Catchment database model.

A catchment represents a hydrological catchment area with associated datasets
and calibration experiments.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Float, Text, DateTime
from sqlalchemy.orm import relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset
    from app.models.experiment import Experiment


class Catchment(Base):
    """
    Represents a hydrological catchment.
    
    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable name (e.g., "Queanbeyan River at Queanbeyan")
        gauge_id: Gauge station identifier (e.g., "410734")
        area_km2: Catchment area in square kilometers
        description: Optional description and notes
        created_at: Timestamp when created
        updated_at: Timestamp when last updated
    """
    
    __tablename__ = "catchments"
    __allow_unmapped__ = True
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    gauge_id = Column(String(50), nullable=True, index=True)
    area_km2 = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    datasets = relationship(
        "Dataset",
        back_populates="catchment",
        cascade="all, delete-orphan"
    )
    experiments = relationship(
        "Experiment",
        back_populates="catchment",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Catchment(id={self.id}, name='{self.name}', gauge_id='{self.gauge_id}')>"
