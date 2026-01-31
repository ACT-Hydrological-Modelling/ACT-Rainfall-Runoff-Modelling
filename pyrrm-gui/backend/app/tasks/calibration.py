"""
Calibration background task.

This module provides the background task for running calibrations.
In Phase 2, this will be replaced with Celery tasks for full async support.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add pyrrm to path - handles both Docker and local environments
def _setup_pyrrm_path():
    current_file = Path(__file__)
    
    # Try Docker path first (/app)
    docker_app_dir = current_file.parents[2]  # /app/app/tasks -> /app
    if (docker_app_dir / "pyrrm").exists():
        if str(docker_app_dir) not in sys.path:
            sys.path.insert(0, str(docker_app_dir))
        return
    
    # Try local development path
    try:
        local_path = current_file.parents[4]
        if str(local_path) not in sys.path:
            sys.path.insert(0, str(local_path))
    except IndexError:
        pass

_setup_pyrrm_path()

import numpy as np
import pandas as pd

from app.database import SessionLocal
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset import Dataset, DatasetType
from app.models.result import CalibrationResult
from app.services.calibration import CalibrationService
from app.services.data_handler import DataHandlerService
from app.config import get_settings

settings = get_settings()


def run_calibration_background(experiment_id: str):
    """
    Run calibration as a background task.
    
    This function:
    1. Loads the experiment configuration from the database
    2. Loads and prepares the data
    3. Creates the model and objective function
    4. Runs the calibration
    5. Saves the results
    
    Args:
        experiment_id: ID of the experiment to run
    """
    db = SessionLocal()
    
    try:
        # Get experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            return
        
        # Check if still queued (might have been cancelled)
        if experiment.status != ExperimentStatus.QUEUED:
            return
        
        # Update status to RUNNING
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        db.commit()
        
        # Load datasets
        datasets = db.query(Dataset).filter(Dataset.catchment_id == experiment.catchment_id).all()
        
        rainfall_df = None
        pet_df = None
        observed_df = None
        
        for ds in datasets:
            df, _ = DataHandlerService.load_csv(ds.file_path, ds.type.value)
            if ds.type == DatasetType.RAINFALL:
                rainfall_df = df
            elif ds.type == DatasetType.PET:
                pet_df = df
            elif ds.type == DatasetType.OBSERVED_FLOW:
                observed_df = df
        
        if rainfall_df is None or pet_df is None:
            raise ValueError("Missing required datasets (rainfall and PET)")
        
        if observed_df is None:
            raise ValueError("Missing observed flow dataset for calibration")
        
        # Merge datasets
        merged = DataHandlerService.merge_datasets(rainfall_df, pet_df, observed_df)
        
        # Apply date filter if specified
        cal_period = experiment.calibration_period or {}
        if cal_period.get('start_date'):
            merged = merged[merged.index >= pd.Timestamp(cal_period['start_date'])]
        if cal_period.get('end_date'):
            merged = merged[merged.index <= pd.Timestamp(cal_period['end_date'])]
        
        # Prepare inputs
        input_cols = ['rainfall', 'pet']
        if 'precipitation' in merged.columns:
            input_cols[0] = 'precipitation'
        
        inputs = merged[input_cols].copy()
        inputs.columns = ['precipitation', 'pet']
        observed = merged['observed_flow'].values
        
        # Get catchment area
        catchment = experiment.catchment
        catchment_area = catchment.area_km2 if catchment else None
        
        # Create model
        model = CalibrationService.create_model(
            model_type=experiment.model_type.value,
            config=experiment.model_settings or {},
            catchment_area_km2=catchment_area
        )
        
        # Get parameter bounds
        if experiment.parameter_bounds:
            parameter_bounds = {k: tuple(v) for k, v in experiment.parameter_bounds.items()}
        else:
            parameter_bounds = model.get_parameter_bounds()
        
        # Create objective
        obj_config = experiment.objective_config or {'type': 'NSE'}
        objective = CalibrationService.create_objective(obj_config)
        
        # Get algorithm config
        alg_config = experiment.algorithm_config or {'method': 'sceua_direct', 'max_evals': 50000}
        warmup_days = cal_period.get('warmup_days', settings.default_warmup_days)
        
        # Create checkpoint directory
        checkpoint_dir = settings.checkpoints_dir / experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Run calibration
        result = CalibrationService.run_calibration(
            model=model,
            inputs=inputs,
            observed=observed,
            parameter_bounds=parameter_bounds,
            objective=objective,
            algorithm_config=alg_config,
            warmup_days=warmup_days,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Calculate metrics
        model.set_parameters(result.best_parameters)
        model.reset()
        output = model.run(inputs)
        
        if 'runoff' in output.columns:
            simulated = output['runoff'].values
        elif 'flow' in output.columns:
            simulated = output['flow'].values
        else:
            simulated = output.iloc[:, 0].values
        
        from pyrrm.calibration.objective_functions import calculate_metrics
        metrics = calculate_metrics(simulated[warmup_days:], observed[warmup_days:])
        
        # Save report
        from pyrrm.calibration.report import CalibrationReport
        
        report = CalibrationReport(
            result=result,
            observed=observed[warmup_days:],
            simulated=simulated[warmup_days:],
            dates=merged.index[warmup_days:],
            precipitation=inputs['precipitation'].values[warmup_days:] if 'precipitation' in inputs.columns else None,
            pet=inputs['pet'].values[warmup_days:] if 'pet' in inputs.columns else None,
            inputs=inputs,
            parameter_bounds={k: list(v) if isinstance(v, tuple) else v for k, v in parameter_bounds.items()},
            catchment_info={
                'name': catchment.name if catchment else '',
                'gauge_id': catchment.gauge_id if catchment else '',
                'area_km2': catchment_area
            },
            calibration_period=(
                str(cal_period.get('start_date', merged.index[0].date())),
                str(cal_period.get('end_date', merged.index[-1].date()))
            ),
            warmup_days=warmup_days,
            model_settings={
                'module': type(model).__module__,
                'class_name': type(model).__name__,
                'init_kwargs': {}
            }
        )
        
        report_path = settings.reports_dir / f"{experiment_id}.pkl"
        report.save(str(report_path))
        
        # Save result to database
        db_result = CalibrationResult(
            experiment_id=experiment_id,
            best_parameters=result.best_parameters,
            best_objective=result.best_objective,
            metrics=metrics,
            runtime_seconds=result.runtime_seconds,
            iterations_completed=len(result.all_samples) if hasattr(result, 'all_samples') and result.all_samples is not None else None,
            report_file_path=str(report_path)
        )
        
        db.add(db_result)
        
        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        # Update experiment with error
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            experiment.completed_at = datetime.utcnow()
            db.commit()
    
    finally:
        db.close()
