"""
Calibration service for running model calibrations using pyrrm.

This service provides the interface between the web application and the
pyrrm calibration framework.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime

import numpy as np
import pandas as pd

# Add pyrrm to path - handles both Docker and local environments
def _setup_pyrrm_path():
    # In Docker: /app/app/services/calibration.py -> /app (where pyrrm is mounted)
    # Locally: .../pyrrm-gui/backend/app/services/calibration.py -> .../pyrrm-gui/../../
    current_file = Path(__file__)
    
    # Try Docker path first (/app)
    docker_app_dir = current_file.parents[2]  # /app/app/services -> /app
    if (docker_app_dir / "pyrrm").exists():
        if str(docker_app_dir) not in sys.path:
            sys.path.insert(0, str(docker_app_dir))
        return
    
    # Try local development path
    try:
        local_path = current_file.parents[4]  # pyrrm-gui parent
        if str(local_path) not in sys.path:
            sys.path.insert(0, str(local_path))
    except IndexError:
        pass

_setup_pyrrm_path()

# Import pyrrm components
from pyrrm.models.sacramento import Sacramento
from pyrrm.models.gr4j import GR4J
from pyrrm.models.gr5j import GR5J
from pyrrm.models.gr6j import GR6J
from pyrrm.routing.muskingum import NonlinearMuskingumRouter
from pyrrm.routing.routed_model import RoutedModel
from pyrrm.calibration.runner import CalibrationRunner
from pyrrm.calibration.objective_functions import (
    NSE, KGE, RMSE, MAE, PBIAS, LogNSE,
    calculate_metrics
)

from app.config import get_settings

settings = get_settings()


# Model type mapping
MODEL_CLASSES = {
    'sacramento': Sacramento,
    'gr4j': GR4J,
    'gr5j': GR5J,
    'gr6j': GR6J,
}

# Objective function mapping
OBJECTIVE_CLASSES = {
    'NSE': NSE,
    'KGE': KGE,
    'RMSE': RMSE,
    'MAE': MAE,
    'PBIAS': PBIAS,
    'NSE_log': LogNSE,
}


class TrimmedFlowObjective:
    """
    Wrapper that trims the flow range for objective function calculation.
    
    This class masks observations outside the specified flow range,
    allowing calibration to focus on specific flow regimes.
    
    Args:
        base_objective: The underlying objective function to use
        flow_min: Minimum flow to include (flows below are masked)
        flow_max: Maximum flow to include (flows above are masked)
    """
    
    def __init__(self, base_objective, flow_min=None, flow_max=None):
        self.base_objective = base_objective
        self.flow_min = flow_min
        self.flow_max = flow_max
        
        # Inherit attributes from base objective
        self.direction = getattr(base_objective, 'direction', 'maximize')
        self.maximize = getattr(base_objective, 'maximize', True)
        self.name = getattr(base_objective, 'name', 'TrimmedObjective')
    
    def __call__(self, observed, simulated):
        """Calculate objective only on flows within range."""
        obs = np.asarray(observed)
        sim = np.asarray(simulated)
        
        # Build mask for valid observations
        valid_mask = ~np.isnan(obs) & ~np.isnan(sim)
        
        if self.flow_min is not None:
            valid_mask &= (obs >= self.flow_min)
        
        if self.flow_max is not None:
            valid_mask &= (obs <= self.flow_max)
        
        if valid_mask.sum() < 10:
            # Not enough valid points - return worst possible value
            return -1e10 if self.maximize else 1e10
        
        # Apply mask and calculate base objective
        obs_masked = obs[valid_mask]
        sim_masked = sim[valid_mask]
        
        return self.base_objective(obs_masked, sim_masked)
    
    def calculate(self, simulated, observed):
        """Legacy interface for compatibility."""
        return self.__call__(observed, simulated)
    
    def __repr__(self):
        return f"TrimmedFlowObjective({self.base_objective}, range=[{self.flow_min}, {self.flow_max}])"


class CalibrationService:
    """
    Service for running model calibrations.
    
    Provides functionality for:
    - Creating rainfall-runoff models
    - Configuring objective functions
    - Running calibrations with progress callbacks
    - Saving and loading calibration results
    """
    
    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """Get list of available models with their parameter definitions."""
        models_info = []
        
        for model_type, model_class in MODEL_CLASSES.items():
            model = model_class()
            params = model.get_parameter_bounds()
            
            # Get parameter definitions if available
            param_defs = []
            if hasattr(model, 'parameter_definitions'):
                for p in model.parameter_definitions:
                    param_defs.append({
                        'name': p.name,
                        'default': p.default,
                        'min_bound': p.min_bound,
                        'max_bound': p.max_bound,
                        'description': p.description,
                        'unit': p.unit
                    })
            else:
                for name, (min_val, max_val) in params.items():
                    param_defs.append({
                        'name': name,
                        'default': (min_val + max_val) / 2,
                        'min_bound': min_val,
                        'max_bound': max_val,
                        'description': '',
                        'unit': ''
                    })
            
            models_info.append({
                'type': model_type,
                'name': model_class.__name__,
                'description': model_class.__doc__.split('\n')[0] if model_class.__doc__ else '',
                'n_parameters': len(params),
                'parameters': param_defs
            })
        
        return models_info
    
    @staticmethod
    def get_available_objectives() -> List[Dict[str, Any]]:
        """Get list of available objective functions."""
        objectives = []
        
        for name, obj_class in OBJECTIVE_CLASSES.items():
            obj = obj_class()
            objectives.append({
                'name': name,
                'description': obj_class.__doc__.split('\n')[0] if obj_class.__doc__ else '',
                'maximize': obj.maximize,
                'optimal_value': 1.0 if obj.maximize else 0.0
            })
        
        return objectives
    
    @staticmethod
    def create_model(
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        catchment_area_km2: Optional[float] = None
    ):
        """
        Create a rainfall-runoff model instance.
        
        Args:
            model_type: Type of model ('sacramento', 'gr4j', etc.)
            config: Model configuration (routing, initial states)
            catchment_area_km2: Catchment area for unit conversion
            
        Returns:
            Model instance (possibly wrapped in RoutedModel)
        """
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = config or {}
        
        # Create base model
        model_class = MODEL_CLASSES[model_type]
        model = model_class()
        
        if catchment_area_km2:
            model.set_catchment_area(catchment_area_km2)
        
        # Apply routing if configured
        routing_config = config.get('routing', {})
        if routing_config.get('enabled', False):
            router = NonlinearMuskingumRouter(
                K=routing_config.get('K', 5.0),
                m=routing_config.get('m', 0.8),
                n_subreaches=routing_config.get('n_subreaches', 3)
            )
            model = RoutedModel(model, router)
        
        return model
    
    @staticmethod
    def create_objective(
        config: Dict[str, Any],
        observed: Optional[np.ndarray] = None
    ):
        """
        Create an objective function instance.
        
        Args:
            config: Objective configuration {type, transform, weights, flow_trimming}
            observed: Observed data array (needed for percentile-based trimming)
            
        Returns:
            Objective function instance (possibly wrapped with TrimmedFlowObjective)
        """
        obj_type = config.get('type', 'NSE')
        
        if obj_type not in OBJECTIVE_CLASSES:
            raise ValueError(f"Unknown objective type: {obj_type}")
        
        # Create base objective
        base_objective = OBJECTIVE_CLASSES[obj_type]()
        
        # Check for flow trimming configuration
        flow_trimming = config.get('flow_trimming', {})
        
        if not flow_trimming or not flow_trimming.get('enabled', False):
            return base_objective
        
        # Calculate flow thresholds
        flow_min = None
        flow_max = None
        
        min_threshold = flow_trimming.get('min_threshold')
        max_threshold = flow_trimming.get('max_threshold')
        
        if observed is not None:
            # Clean observed data for percentile calculation
            obs_clean = observed[~np.isnan(observed)]
            obs_clean = obs_clean[obs_clean >= 0]  # Remove negative values
            
            if min_threshold:
                if min_threshold.get('type') == 'absolute':
                    flow_min = min_threshold.get('value')
                elif min_threshold.get('type') == 'percentile':
                    pct = min_threshold.get('value', 0)
                    flow_min = float(np.percentile(obs_clean, pct)) if len(obs_clean) > 0 else 0
            
            if max_threshold:
                if max_threshold.get('type') == 'absolute':
                    flow_max = max_threshold.get('value')
                elif max_threshold.get('type') == 'percentile':
                    pct = max_threshold.get('value', 100)
                    flow_max = float(np.percentile(obs_clean, pct)) if len(obs_clean) > 0 else 1e10
        else:
            # Use absolute values only if no observed data
            if min_threshold and min_threshold.get('type') == 'absolute':
                flow_min = min_threshold.get('value')
            if max_threshold and max_threshold.get('type') == 'absolute':
                flow_max = max_threshold.get('value')
        
        # Wrap with trimmed objective if thresholds are set
        if flow_min is not None or flow_max is not None:
            return TrimmedFlowObjective(base_objective, flow_min, flow_max)
        
        return base_objective
    
    @staticmethod
    def run_calibration(
        model,
        inputs: pd.DataFrame,
        observed: np.ndarray,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective,
        algorithm_config: Dict[str, Any],
        warmup_days: int = 365,
        checkpoint_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run a calibration.
        
        Args:
            model: Rainfall-runoff model instance
            inputs: Input DataFrame with precipitation and pet
            observed: Observed flow array
            parameter_bounds: Parameter bounds for calibration
            objective: Objective function instance
            algorithm_config: Algorithm settings
            warmup_days: Number of warmup days
            checkpoint_dir: Directory for checkpoints
            progress_callback: Optional callback for progress updates
            
        Returns:
            Calibration result dictionary
        """
        # Create runner
        runner = CalibrationRunner(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            parameter_bounds=parameter_bounds,
            warmup_period=warmup_days
        )
        
        method = algorithm_config.get('method', 'sceua_direct')
        
        if method == 'sceua_direct':
            # Note: max_tolerant_iter and tolerance control early stopping
            # The defaults here match the tutorial notebooks for consistent behavior
            result = runner.run_sceua_direct(
                max_evals=algorithm_config.get('max_evals', 50000),
                n_complexes=algorithm_config.get('n_complexes'),
                max_workers=algorithm_config.get('max_workers', 1),
                seed=algorithm_config.get('seed'),
                max_tolerant_iter=algorithm_config.get('max_tolerant_iter', 100),  # Allow 100 iters without improvement
                tolerance=algorithm_config.get('tolerance', 1e-4),  # More lenient improvement threshold
                callback=progress_callback,
                progress_bar=False  # Disable for background execution
            )
        elif method == 'scipy':
            scipy_method = algorithm_config.get('scipy_method', 'differential_evolution')
            result = runner.run_scipy(
                method=scipy_method,
                maxiter=algorithm_config.get('max_evals', 1000),
                workers=algorithm_config.get('max_workers', 1)
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return result
    
    @staticmethod
    def evaluate_parameters(
        model,
        inputs: pd.DataFrame,
        observed: np.ndarray,
        parameters: Dict[str, float],
        warmup_days: int = 365
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate a single parameter set.
        
        Args:
            model: Model instance
            inputs: Input DataFrame
            observed: Observed flow array
            parameters: Parameter values to evaluate
            warmup_days: Warmup period
            
        Returns:
            Tuple of (metrics dict, simulated array)
        """
        # Set parameters
        model.set_parameters(parameters)
        model.reset()
        
        # Run model
        output = model.run(inputs)
        
        # Extract flow
        if 'runoff' in output.columns:
            simulated = output['runoff'].values
        elif 'flow' in output.columns:
            simulated = output['flow'].values
        else:
            simulated = output.iloc[:, 0].values
        
        # Apply warmup
        sim_post = simulated[warmup_days:]
        obs_post = observed[warmup_days:]
        
        # Calculate metrics
        metrics = calculate_metrics(sim_post, obs_post)
        
        return metrics, sim_post
    
    @staticmethod
    def get_default_bounds(model_type: str) -> Dict[str, Tuple[float, float]]:
        """Get default parameter bounds for a model type."""
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = MODEL_CLASSES[model_type]()
        return model.get_parameter_bounds()
