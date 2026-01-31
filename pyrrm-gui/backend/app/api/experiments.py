"""
Experiment API endpoints.

Provides operations for creating, configuring, and running experiments.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.catchment import Catchment
from app.models.experiment import Experiment, ExperimentStatus, ModelType
from app.models.result import CalibrationResult
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentStatusResponse,
)
from app.config import get_settings

settings = get_settings()
router = APIRouter()


def experiment_to_response(exp: Experiment) -> ExperimentResponse:
    """Convert experiment model to response schema."""
    return ExperimentResponse(
        id=exp.id,
        catchment_id=exp.catchment_id,
        name=exp.name,
        description=exp.description,
        model_type=exp.model_type,
        model_settings=exp.model_settings,
        parameter_bounds=exp.parameter_bounds,
        calibration_period=exp.calibration_period,
        objective_config=exp.objective_config,
        algorithm_config=exp.algorithm_config,
        status=exp.status,
        celery_task_id=exp.celery_task_id,
        error_message=exp.error_message,
        created_at=exp.created_at,
        started_at=exp.started_at,
        completed_at=exp.completed_at,
        runtime_seconds=exp.runtime_seconds,
        has_result=exp.result is not None,
        best_objective=exp.result.best_objective if exp.result else None
    )


@router.get("/", response_model=List[ExperimentListResponse])
def list_experiments(
    catchment_id: Optional[str] = None,
    status_filter: Optional[ExperimentStatus] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List experiments with optional filtering.
    """
    query = db.query(Experiment)
    
    if catchment_id:
        query = query.filter(Experiment.catchment_id == catchment_id)
    
    if status_filter:
        query = query.filter(Experiment.status == status_filter)
    
    experiments = query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for exp in experiments:
        result.append(ExperimentListResponse(
            id=exp.id,
            catchment_id=exp.catchment_id,
            name=exp.name,
            model_type=exp.model_type,
            status=exp.status,
            created_at=exp.created_at,
            started_at=exp.started_at,
            completed_at=exp.completed_at,
            best_objective=exp.result.best_objective if exp.result else None
        ))
    
    return result


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    experiment: ExperimentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new experiment configuration.
    
    The experiment is created in DRAFT status and must be started
    separately using the /run endpoint.
    """
    # Verify catchment exists
    catchment = db.query(Catchment).filter(Catchment.id == experiment.catchment_id).first()
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{experiment.catchment_id}' not found"
        )
    
    # Create experiment
    # Use mode='json' to ensure dates are serialized to ISO format strings
    db_experiment = Experiment(
        catchment_id=experiment.catchment_id,
        name=experiment.name,
        description=experiment.description,
        model_type=experiment.model_type,
        model_settings=experiment.model_settings.model_dump(mode='json') if experiment.model_settings else {},
        parameter_bounds=experiment.parameter_bounds,
        calibration_period=experiment.calibration_period.model_dump(mode='json') if experiment.calibration_period else None,
        objective_config=experiment.objective_config.model_dump(mode='json') if experiment.objective_config else None,
        algorithm_config=experiment.algorithm_config.model_dump(mode='json') if experiment.algorithm_config else None,
        status=ExperimentStatus.DRAFT
    )
    
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    
    return experiment_to_response(db_experiment)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get an experiment by ID.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    return experiment_to_response(experiment)


@router.put("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment(
    experiment_id: str,
    experiment_update: ExperimentUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an experiment configuration.
    
    Only experiments in DRAFT status can be updated.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if experiment.status != ExperimentStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only experiments in DRAFT status can be updated"
        )
    
    update_data = experiment_update.model_dump(exclude_unset=True)
    
    # Handle nested objects - use mode='json' to serialize dates properly
    for field in ['model_settings', 'calibration_period', 'objective_config', 'algorithm_config']:
        if field in update_data and update_data[field] is not None:
            if hasattr(update_data[field], 'model_dump'):
                update_data[field] = update_data[field].model_dump(mode='json')
    
    for field, value in update_data.items():
        setattr(experiment, field, value)
    
    db.commit()
    db.refresh(experiment)
    
    return experiment_to_response(experiment)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an experiment.
    
    Running experiments cannot be deleted - cancel them first.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running experiment. Cancel it first."
        )
    
    db.delete(experiment)
    db.commit()
    
    return None


@router.post("/{experiment_id}/clone", response_model=ExperimentResponse)
def clone_experiment(
    experiment_id: str,
    name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Clone an experiment with a new name.
    
    Creates a copy of the experiment configuration in DRAFT status.
    """
    original = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    new_name = name or f"{original.name} (copy)"
    
    db_experiment = Experiment(
        catchment_id=original.catchment_id,
        name=new_name,
        description=original.description,
        model_type=original.model_type,
        model_settings=original.model_settings,
        parameter_bounds=original.parameter_bounds,
        calibration_period=original.calibration_period,
        objective_config=original.objective_config,
        algorithm_config=original.algorithm_config,
        status=ExperimentStatus.DRAFT
    )
    
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    
    return experiment_to_response(db_experiment)


@router.post("/{experiment_id}/run", response_model=ExperimentStatusResponse)
def run_experiment(
    experiment_id: str,
    use_celery: bool = True,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Start a calibration experiment.
    
    The calibration runs asynchronously in the background.
    Use the /status endpoint to check progress or connect via WebSocket.
    
    Args:
        experiment_id: ID of the experiment to run
        use_celery: If True, use Celery task queue (default). If False, use background task.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if experiment.status not in [ExperimentStatus.DRAFT, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment cannot be started from {experiment.status.value} status"
        )
    
    # Update status to QUEUED
    experiment.status = ExperimentStatus.QUEUED
    experiment.error_message = None
    db.commit()
    
    # Queue the calibration task
    task_id = None
    
    if use_celery:
        try:
            from app.tasks.celery_tasks import run_calibration_celery
            result = run_calibration_celery.delay(experiment_id)
            task_id = result.id
            
            # Store task ID
            experiment.celery_task_id = task_id
            db.commit()
        except Exception as e:
            # Fallback to background task if Celery not available
            from app.tasks.calibration import run_calibration_background
            if background_tasks:
                background_tasks.add_task(run_calibration_background, experiment_id)
    else:
        from app.tasks.calibration import run_calibration_background
        if background_tasks:
            background_tasks.add_task(run_calibration_background, experiment_id)
    
    return ExperimentStatusResponse(
        id=experiment.id,
        status=experiment.status,
        progress={"task_id": task_id} if task_id else None
    )


@router.post("/{experiment_id}/cancel", response_model=ExperimentStatusResponse)
def cancel_experiment(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancel a running experiment.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if experiment.status not in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel experiment in {experiment.status.value} status"
        )
    
    # Update status to CANCELLED
    experiment.status = ExperimentStatus.CANCELLED
    experiment.completed_at = datetime.utcnow()
    db.commit()
    
    # TODO: In Phase 2, also revoke Celery task if celery_task_id is set
    
    return ExperimentStatusResponse(
        id=experiment.id,
        status=experiment.status
    )


@router.get("/{experiment_id}/status", response_model=ExperimentStatusResponse)
def get_experiment_status(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current status of an experiment.
    
    For running experiments, this includes progress information.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    progress = None
    
    # TODO: In Phase 2, fetch progress from Redis
    if experiment.status == ExperimentStatus.RUNNING:
        progress = {
            "message": "Calibration in progress...",
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None
        }
    
    return ExperimentStatusResponse(
        id=experiment.id,
        status=experiment.status,
        progress=progress,
        error_message=experiment.error_message
    )
