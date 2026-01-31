"""
Experiment database model.

An experiment represents a calibration run configuration and its status.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum

from app.database import Base


class ExperimentStatus(str, Enum):
    """Status of a calibration experiment."""
    DRAFT = "draft"              # Configuration saved but not run
    QUEUED = "queued"            # Submitted to task queue
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Finished with error
    CANCELLED = "cancelled"      # Manually cancelled


class ModelType(str, Enum):
    """Available rainfall-runoff model types."""
    SACRAMENTO = "sacramento"
    GR4J = "gr4j"
    GR5J = "gr5j"
    GR6J = "gr6j"


class Experiment(Base):
    """
    Represents a calibration experiment configuration.
    
    Attributes:
        id: Unique identifier (UUID)
        catchment_id: Foreign key to parent catchment
        name: Human-readable experiment name
        description: Optional description
        model_type: Type of rainfall-runoff model
        model_settings: Model-specific configuration (routing, initial states)
        parameter_bounds: Parameter bounds for calibration
        calibration_period: Start date, end date, warmup days
        objective_config: Objective function configuration
        algorithm_config: Calibration algorithm settings
        status: Current experiment status
        celery_task_id: Celery task ID (if running)
        error_message: Error message (if failed)
        created_at: Timestamp when created
        started_at: Timestamp when calibration started
        completed_at: Timestamp when calibration completed
    """
    
    __tablename__ = "experiments"
    __allow_unmapped__ = True
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    catchment_id = Column(String(36), ForeignKey("catchments.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Model configuration
    model_type = Column(SQLEnum(ModelType), nullable=False, default=ModelType.SACRAMENTO)
    model_settings = Column(JSON, nullable=True, default=dict)
    
    # Parameter bounds (JSON: {"param_name": [min, max], ...})
    parameter_bounds = Column(JSON, nullable=True)
    
    # Calibration period (JSON: {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "warmup_days": 365})
    calibration_period = Column(JSON, nullable=True)
    
    # Objective function (JSON: {"type": "NSE", "transform": "none", "weights": {...}})
    objective_config = Column(JSON, nullable=True)
    
    # Algorithm settings (JSON: {"method": "sceua_direct", "max_evals": 50000, ...})
    algorithm_config = Column(JSON, nullable=True)
    
    # Status tracking
    status = Column(SQLEnum(ExperimentStatus), nullable=False, default=ExperimentStatus.DRAFT)
    celery_task_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    catchment = relationship("Catchment", back_populates="experiments")
    result = relationship(
        "CalibrationResult",
        back_populates="experiment",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', status={self.status.value})>"
    
    @property
    def runtime_seconds(self) -> Optional[float]:
        """Calculate runtime in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
