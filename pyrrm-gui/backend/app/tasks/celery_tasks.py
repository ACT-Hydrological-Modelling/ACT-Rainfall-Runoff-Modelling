"""
Celery tasks for background calibration.

This module provides Celery tasks for running calibrations with:
- Progress tracking via Redis pub/sub
- Checkpoint support for resumable calibrations
- Proper error handling and status updates
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import redis

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

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset import Dataset, DatasetType
from app.models.result import CalibrationResult
from app.services.calibration import CalibrationService
from app.services.data_handler import DataHandlerService
from app.config import get_settings

settings = get_settings()


def create_progress_callback(
    experiment_id: str,
    redis_client: redis.Redis,
    model: Any,
    inputs: pd.DataFrame,
    observed: np.ndarray,
    warmup_days: int,
    parameter_names: list,
    total_iterations: Optional[int] = None,
    objective_name: str = "NSE",
    sim_update_interval: int = 50
) -> Callable:
    """
    Create a progress callback that publishes updates to Redis.
    
    This callback streams real-time progress information including:
    - Optimization log messages
    - Objective function evolution
    - Periodic simulation updates (hydrograph, FDC, scatter plot)
    
    Args:
        experiment_id: ID of the experiment
        redis_client: Redis client for publishing
        model: The model instance for running simulations
        inputs: Input DataFrame (precipitation, pet)
        observed: Observed flow array
        warmup_days: Number of warmup days to skip
        parameter_names: List of parameter names in order
        total_iterations: Total expected iterations (max_evals)
        objective_name: Name of objective function for display
        sim_update_interval: How often to update simulation plots (iterations)
        
    Returns:
        Callback function for progress updates
    """
    import time
    
    best_so_far = [float('inf')]  # Use list to allow modification in closure (minimization)
    no_improve_count = [0]
    start_time = [None]
    log_buffer = []  # Store recent log messages
    objective_history = []  # Store (nfev, objective) pairs for evolution plot
    last_sim_update = [0]  # Track last simulation update iteration
    
    # Pre-compute observed data for FDC and scatter (excluding warmup)
    obs_valid = observed[warmup_days:]
    obs_valid_clean = obs_valid[~np.isnan(obs_valid)]
    
    def compute_fdc(data: np.ndarray, n_points: int = 100) -> tuple:
        """Compute Flow Duration Curve with n_points."""
        data_clean = data[~np.isnan(data)]
        if len(data_clean) == 0:
            return [], []
        sorted_data = np.sort(data_clean)[::-1]
        exceedance = np.linspace(0, 100, len(sorted_data))
        # Sample to n_points
        indices = np.linspace(0, len(sorted_data) - 1, n_points, dtype=int)
        return exceedance[indices].tolist(), sorted_data[indices].tolist()
    
    def sample_timeseries(dates, observed_ts, simulated_ts, max_points: int = 300) -> dict:
        """Sample timeseries data for plotting."""
        n = len(dates)
        if n <= max_points:
            indices = np.arange(n)
        else:
            indices = np.linspace(0, n - 1, max_points, dtype=int)
        
        return {
            "dates": [str(dates[i].date()) for i in indices],
            "observed": [float(observed_ts[i]) if not np.isnan(observed_ts[i]) else None for i in indices],
            "simulated": [float(simulated_ts[i]) if not np.isnan(simulated_ts[i]) else None for i in indices]
        }
    
    def sample_scatter(observed_arr, simulated_arr, max_points: int = 500) -> dict:
        """Sample scatter plot data."""
        # Remove NaN pairs
        mask = ~(np.isnan(observed_arr) | np.isnan(simulated_arr))
        obs_clean = observed_arr[mask]
        sim_clean = simulated_arr[mask]
        
        n = len(obs_clean)
        if n <= max_points:
            indices = np.arange(n)
        else:
            indices = np.random.choice(n, max_points, replace=False)
            indices.sort()
        
        return {
            "observed": [float(obs_clean[i]) for i in indices],
            "simulated": [float(sim_clean[i]) for i in indices]
        }
    
    def run_simulation_update(best_x: np.ndarray) -> Optional[dict]:
        """Run simulation with best parameters and compute plot data."""
        try:
            # Set parameters and run model
            param_dict = {name: float(val) for name, val in zip(parameter_names, best_x)}
            model.set_parameters(param_dict)
            model.reset()
            output = model.run(inputs)
            
            # Get simulated flow
            if 'runoff' in output.columns:
                simulated = output['runoff'].values
            elif 'flow' in output.columns:
                simulated = output['flow'].values
            else:
                simulated = output.iloc[:, 0].values
            
            # Extract post-warmup data
            sim_valid = simulated[warmup_days:]
            dates = inputs.index[warmup_days:]
            
            # Compute FDC for both
            obs_exc, obs_flow = compute_fdc(obs_valid)
            sim_exc, sim_flow = compute_fdc(sim_valid)
            
            # Sample hydrograph
            hydrograph = sample_timeseries(dates, obs_valid, sim_valid)
            
            # Sample scatter
            scatter = sample_scatter(obs_valid, sim_valid)
            
            return {
                "hydrograph": hydrograph,
                "fdc": {
                    "exceedance": obs_exc,  # Same for both
                    "observed": obs_flow,
                    "simulated": sim_flow
                },
                "scatter": scatter,
                "parameters": param_dict
            }
        except Exception as e:
            # Don't fail calibration due to visualization error
            return None
    
    def callback(info: Dict[str, Any]):
        """
        Progress callback that publishes to Redis.
        
        Args:
            info: Dict with 'iteration', 'nfev', 'best_fun', 'best_x'
        """
        iteration = info.get('iteration', 0)
        nfev = info.get('nfev', 0)
        objective = info.get('best_fun', 0.0)
        best_x = info.get('best_x', [])
        
        # Initialize start time on first call
        if start_time[0] is None:
            start_time[0] = time.time()
            # Send initial log message
            init_msg = f"SCE-UA Optimization Started"
            log_buffer.append({
                "time": datetime.utcnow().isoformat(),
                "message": init_msg,
                "level": "info"
            })
            log_buffer.append({
                "time": datetime.utcnow().isoformat(),
                "message": f"  Objective: {objective_name}, Max evaluations: {total_iterations}",
                "level": "info"
            })
        
        # Track best value (minimization - lower is better)
        improved = False
        if objective < best_so_far[0]:
            improved = True
            best_so_far[0] = objective
            no_improve_count[0] = 0
        else:
            no_improve_count[0] += 1
        
        # Calculate elapsed time
        elapsed = time.time() - start_time[0]
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
        
        # Generate log message similar to verbose SCE-UA output
        # For NSE-like objectives, display as maximization (negate the value)
        display_obj = -objective  # Convert minimization to maximization display
        display_best = -best_so_far[0]
        
        # Track objective history for evolution plot
        objective_history.append({
            "nfev": nfev,
            "objective": display_best,
            "iteration": iteration
        })
        # Keep last 500 points for history
        if len(objective_history) > 500:
            objective_history.pop(0)
        
        # Create log message every 10 iterations or on improvement
        log_message = None
        log_level = "info"
        
        if iteration % 10 == 0 or improved:
            if improved:
                log_message = f"  Iter {iteration:5d}: nfev={nfev:6d}, best {objective_name}={display_best:.6f} ↑ IMPROVED"
                log_level = "success"
            else:
                log_message = f"  Iter {iteration:5d}: nfev={nfev:6d}, best {objective_name}={display_best:.6f}, no_improve={no_improve_count[0]}"
        
        if log_message:
            log_entry = {
                "time": datetime.utcnow().isoformat(),
                "message": log_message,
                "level": log_level
            }
            log_buffer.append(log_entry)
            # Keep only last 100 messages
            if len(log_buffer) > 100:
                log_buffer.pop(0)
        
        # Progress percentage based on evaluations
        progress_pct = (nfev / total_iterations * 100) if total_iterations else None
        
        # Estimate time remaining
        eta_str = None
        if progress_pct and progress_pct > 5:  # Only estimate after 5%
            remaining_pct = 100 - progress_pct
            eta_seconds = (elapsed / progress_pct) * remaining_pct
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
        
        # Check if we should update simulation plots
        # Update on improvement or every sim_update_interval iterations
        simulation_data = None
        should_update_sim = (
            improved or 
            (iteration - last_sim_update[0] >= sim_update_interval) or
            iteration == 1  # First iteration
        )
        
        if should_update_sim and len(best_x) > 0:
            simulation_data = run_simulation_update(np.array(best_x))
            if simulation_data:
                last_sim_update[0] = iteration
        
        # Publish progress update
        message = {
            "type": "progress",
            "data": {
                "experiment_id": experiment_id,
                "iteration": iteration,
                "nfev": nfev,
                "total_iterations": total_iterations,
                "best_objective": display_best,
                "current_objective": display_obj,
                "best_parameters": list(best_x) if hasattr(best_x, '__iter__') else best_x,
                "parameter_names": parameter_names,
                "progress_percent": progress_pct,
                "elapsed_time": elapsed_str,
                "eta": eta_str,
                "no_improve_count": no_improve_count[0],
                "improved": improved,
                "log_messages": log_buffer[-20:],  # Send last 20 log messages
                "objective_history": objective_history[-100:],  # Last 100 points for chart
                "simulation_data": simulation_data,  # May be None
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            redis_client.publish(
                f"calibration:{experiment_id}",
                json.dumps(message)
            )
        except Exception:
            pass  # Don't fail calibration due to publish error
    
    return callback


@celery_app.task(bind=True, name="calibration.run")
def run_calibration_celery(self, experiment_id: str):
    """
    Celery task for running calibration.
    
    This task:
    1. Loads experiment configuration from database
    2. Prepares data and model
    3. Runs calibration with progress callbacks
    4. Saves results and updates database
    
    Args:
        experiment_id: ID of the experiment to run
    """
    db = SessionLocal()
    redis_client = None
    
    try:
        # Connect to Redis for progress updates
        try:
            redis_client = redis.from_url(settings.redis_url)
        except Exception:
            redis_client = None
        
        # Get experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            return {"status": "error", "message": "Experiment not found"}
        
        # Check if still queued
        if experiment.status != ExperimentStatus.QUEUED:
            return {"status": "skipped", "message": f"Experiment status is {experiment.status.value}"}
        
        # Update status to RUNNING
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        experiment.celery_task_id = self.request.id
        db.commit()
        
        # Publish status update
        if redis_client:
            redis_client.publish(
                f"calibration:{experiment_id}",
                json.dumps({"type": "started", "data": {"experiment_id": experiment_id}})
            )
        
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
        
        # Apply date filter
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
        
        # Create objective (pass observed data for percentile-based flow trimming)
        obj_config = experiment.objective_config or {'type': 'NSE'}
        objective = CalibrationService.create_objective(obj_config, observed=observed)
        
        # Get algorithm config
        alg_config = experiment.algorithm_config or {'method': 'sceua_direct', 'max_evals': 50000}
        warmup_days = cal_period.get('warmup_days', settings.default_warmup_days)
        total_iterations = alg_config.get('max_evals', 50000)
        
        # Create progress callback
        progress_callback = None
        if redis_client:
            # Get objective name for display
            obj_type = obj_config.get('type', 'NSE')
            obj_transform = obj_config.get('transform', 'none')
            objective_display = obj_type if obj_transform == 'none' else f"{obj_type}({obj_transform})"
            
            # Get parameter names in order
            parameter_names = list(parameter_bounds.keys())
            
            progress_callback = create_progress_callback(
                experiment_id=experiment_id,
                redis_client=redis_client,
                model=model,
                inputs=inputs,
                observed=observed,
                warmup_days=warmup_days,
                parameter_names=parameter_names,
                total_iterations=total_iterations,
                objective_name=objective_display,
                sim_update_interval=50  # Update plots every 50 iterations or on improvement
            )
        
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
            checkpoint_dir=str(checkpoint_dir),
            progress_callback=progress_callback
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
            precipitation=inputs['precipitation'].values[warmup_days:],
            pet=inputs['pet'].values[warmup_days:],
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
            model_config={
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
        
        # Publish completion
        if redis_client:
            redis_client.publish(
                f"calibration:{experiment_id}",
                json.dumps({
                    "type": "completed",
                    "data": {
                        "experiment_id": experiment_id,
                        "success": True,
                        "best_objective": result.best_objective,
                        "runtime_seconds": result.runtime_seconds
                    }
                })
            )
        
        return {
            "status": "completed",
            "best_objective": result.best_objective,
            "runtime_seconds": result.runtime_seconds
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Update experiment with error
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = error_msg
            experiment.completed_at = datetime.utcnow()
            db.commit()
        
        # Publish failure
        if redis_client:
            redis_client.publish(
                f"calibration:{experiment_id}",
                json.dumps({
                    "type": "completed",
                    "data": {
                        "experiment_id": experiment_id,
                        "success": False,
                        "error": str(e)
                    }
                })
            )
        
        return {"status": "failed", "error": error_msg}
    
    finally:
        db.close()
        if redis_client:
            redis_client.close()
