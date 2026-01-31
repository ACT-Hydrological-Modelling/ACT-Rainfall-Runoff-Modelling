"""
Results API endpoints.

Provides access to calibration results and visualizations.
"""

import math
from typing import Optional, Dict, Any, Union
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session


def sanitize_float(value: Any) -> Optional[float]:
    """Convert NaN/Inf to None for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize a dict, converting NaN/Inf to None."""
    if d is None:
        return {}
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = sanitize_dict(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [sanitize_float(x) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, (int, float)):
            result[k] = sanitize_float(v)
        else:
            result[k] = v
    return result

from app.database import get_db
from app.models.experiment import Experiment, ExperimentStatus
from app.models.result import CalibrationResult
from app.schemas.result import (
    CalibrationResultResponse,
    MetricsResponse,
    TimeSeriesResponse,
    SimulationRequest,
    SimulationResponse,
)
from app.services.visualization import VisualizationService
from app.config import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/experiments/{experiment_id}", response_model=CalibrationResultResponse)
def get_result(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the calibration result for an experiment.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if experiment.status != ExperimentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment has not completed (status: {experiment.status.value})"
        )
    
    if not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No result found for this experiment"
        )
    
    result = experiment.result
    
    # Sanitize metrics to handle NaN/Inf values
    sanitized_metrics = sanitize_dict(result.metrics) if result.metrics else {}
    sanitized_params = sanitize_dict(result.best_parameters) if result.best_parameters else {}
    
    return CalibrationResultResponse(
        id=result.id,
        experiment_id=result.experiment_id,
        best_parameters=sanitized_params,
        best_objective=sanitize_float(result.best_objective),
        metrics=MetricsResponse(**sanitized_metrics) if sanitized_metrics else MetricsResponse(),
        runtime_seconds=sanitize_float(result.runtime_seconds),
        iterations_completed=result.iterations_completed,
        report_file_path=result.report_file_path,
        created_at=result.created_at
    )


@router.get("/experiments/{experiment_id}/parameters")
def get_parameters(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the calibrated parameters with bounds information.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No result found for this experiment"
        )
    
    parameters = experiment.result.best_parameters
    bounds = experiment.parameter_bounds or {}
    
    result = []
    for name, value in parameters.items():
        value = sanitize_float(value) or 0
        min_val, max_val = bounds.get(name, [value, value])
        min_val = sanitize_float(min_val) or 0
        max_val = sanitize_float(max_val) or 0
        
        if max_val != min_val:
            pct = (value - min_val) / (max_val - min_val) * 100
        else:
            pct = 50
        
        result.append({
            "name": name,
            "value": value,
            "min_bound": min_val,
            "max_bound": max_val,
            "percent_of_range": round(sanitize_float(pct) or 50, 2)
        })
    
    return result


@router.get("/experiments/{experiment_id}/metrics")
def get_metrics(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for a calibration result.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id '{experiment_id}' not found"
        )
    
    if not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No result found for this experiment"
        )
    
    metrics = experiment.result.metrics or {}
    
    return VisualizationService.create_metrics_table(metrics)


@router.get("/experiments/{experiment_id}/plots/hydrograph")
def get_hydrograph_plot(
    experiment_id: str,
    log_scale: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get hydrograph comparison plot data.
    
    Returns Plotly figure JSON for frontend rendering.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment or not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment or result not found"
        )
    
    # Load report if available
    report_path = experiment.result.report_file_path
    if report_path and Path(report_path).exists():
        try:
            import pickle
            with open(report_path, 'rb') as f:
                report = pickle.load(f)
            
            dates = [d.strftime('%Y-%m-%d') for d in report.dates]
            observed = report.observed.tolist()
            simulated = report.simulated.tolist()
            precipitation = report.precipitation.tolist() if report.precipitation is not None else None
            
            return VisualizationService.create_hydrograph(
                dates=dates,
                observed=observed,
                simulated=simulated,
                precipitation=precipitation,
                log_scale=log_scale,
                title=f"Hydrograph - {experiment.name}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load report: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )


@router.get("/experiments/{experiment_id}/plots/fdc")
def get_fdc_plot(
    experiment_id: str,
    log_scale: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get flow duration curve plot data.
    
    Returns Plotly figure JSON for frontend rendering.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment or not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment or result not found"
        )
    
    report_path = experiment.result.report_file_path
    if report_path and Path(report_path).exists():
        try:
            import pickle
            with open(report_path, 'rb') as f:
                report = pickle.load(f)
            
            return VisualizationService.create_fdc(
                observed=report.observed.tolist(),
                simulated=report.simulated.tolist(),
                log_scale=log_scale,
                title=f"Flow Duration Curve - {experiment.name}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load report: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )


@router.get("/experiments/{experiment_id}/plots/scatter")
def get_scatter_plot(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get scatter plot data.
    
    Returns Plotly figure JSON for frontend rendering.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment or not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment or result not found"
        )
    
    report_path = experiment.result.report_file_path
    if report_path and Path(report_path).exists():
        try:
            import pickle
            with open(report_path, 'rb') as f:
                report = pickle.load(f)
            
            return VisualizationService.create_scatter(
                observed=report.observed.tolist(),
                simulated=report.simulated.tolist(),
                title=f"Observed vs Simulated - {experiment.name}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load report: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )


@router.get("/experiments/{experiment_id}/plots/parameters")
def get_parameters_plot(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get parameter bounds chart data.
    
    Returns Plotly figure JSON for frontend rendering.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment or not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment or result not found"
        )
    
    parameters = experiment.result.best_parameters
    bounds = experiment.parameter_bounds or {}
    
    # Convert bounds from list to tuple format
    bounds_tuple = {k: v for k, v in bounds.items()}
    
    return VisualizationService.create_parameter_bounds_chart(
        parameters=parameters,
        bounds=bounds_tuple,
        title=f"Parameter Values - {experiment.name}"
    )


@router.get("/experiments/{experiment_id}/report/download")
def download_report(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the CalibrationReport pickle file.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    if not experiment or not experiment.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment or result not found"
        )
    
    report_path = experiment.result.report_file_path
    if not report_path or not Path(report_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )
    
    filename = f"{experiment.name.replace(' ', '_')}_report.pkl"
    
    return FileResponse(
        report_path,
        media_type="application/octet-stream",
        filename=filename
    )
