"""
CalibrationResult database model.

Stores the results of a completed calibration experiment.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class CalibrationResult(Base):
    """
    Represents the results of a calibration experiment.
    
    Attributes:
        id: Unique identifier (UUID)
        experiment_id: Foreign key to parent experiment
        best_parameters: Best parameter values found
        best_objective: Best objective function value
        metrics: Calculated performance metrics (NSE, KGE, RMSE, etc.)
        runtime_seconds: Total calibration runtime
        iterations_completed: Number of iterations/evaluations completed
        report_file_path: Path to saved CalibrationReport (.pkl)
        samples_file_path: Path to all samples file (optional)
        created_at: Timestamp when result was saved
    """
    
    __tablename__ = "calibration_results"
    __allow_unmapped__ = True
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False, unique=True)
    
    # Best result
    best_parameters = Column(JSON, nullable=False)
    best_objective = Column(Float, nullable=False)
    
    # Performance metrics (JSON: {"NSE": 0.85, "KGE": 0.82, ...})
    metrics = Column(JSON, nullable=True)
    
    # Execution stats
    runtime_seconds = Column(Float, nullable=True)
    iterations_completed = Column(Integer, nullable=True)
    
    # File paths
    report_file_path = Column(String(500), nullable=True)
    samples_file_path = Column(String(500), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="result")
    
    def __repr__(self) -> str:
        return f"<CalibrationResult(id={self.id}, best_objective={self.best_objective:.4f})>"
