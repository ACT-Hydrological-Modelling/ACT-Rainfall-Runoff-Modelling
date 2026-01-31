"""
Direct SCE-UA (Shuffled Complex Evolution - University of Arizona) adapter for pyrrm.

This module provides calibration using a standalone SCE-UA implementation,
offering an alternative to the SpotPy-based SCE-UA in spotpy_adapter.py.

Key advantages over SpotPy SCE-UA:
- No external dependency (algorithm is vendored)
- More configuration options (PCA recovery, multiple convergence criteria)
- Parallel evaluation via ThreadPoolExecutor
- Initial parameter sets support (x0)
- All evaluations tracked automatically (xv, funv arrays)

References
----------
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global
    optimization for conceptual rainfall-runoff models. Water Resources Research.
- Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global
    optimization method for calibrating watershed models. Journal of Hydrology.
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
import time

from pyrrm.calibration._sceua import minimize, Result

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction


def _should_maximize(objective) -> bool:
    """
    Check if an objective function should be maximized.
    
    Handles both old interface (maximize property) and new interface (direction attribute).
    
    Args:
        objective: Objective function instance
        
    Returns:
        True if objective should be maximized, False otherwise
    """
    if hasattr(objective, 'direction'):
        return objective.direction == 'maximize'
    elif hasattr(objective, 'maximize'):
        return objective.maximize
    else:
        return True  # Default to maximize (NSE, KGE, etc.)


@dataclass
class SCEUACalibrationResult:
    """Container for SCE-UA calibration results.
    
    Attributes
    ----------
    best_parameters : dict
        Best parameter set found {name: value}.
    best_objective : float
        Best objective value (in original units, not negated).
    all_samples : pd.DataFrame
        All evaluated parameter sets with columns for each parameter
        and 'objective' column.
    convergence_info : dict
        Convergence diagnostics including nit, nfev, message, success.
    runtime_seconds : float
        Total optimization runtime.
    success : bool
        Whether optimization completed successfully.
    message : str
        Termination message.
    """
    best_parameters: Dict[str, float]
    best_objective: float
    all_samples: pd.DataFrame
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    success: bool = True
    message: str = ""


class SCEUAModelWrapper:
    """
    Wrapper to make pyrrm models compatible with sceua.minimize().
    
    Handles parameter transformations and objective calculation.
    The SCE-UA algorithm tracks all evaluations internally, so this
    wrapper focuses on model execution and objective direction handling.
    
    Attributes
    ----------
    model : BaseRainfallRunoffModel
        The rainfall-runoff model to calibrate.
    inputs : pd.DataFrame
        Input data (precipitation, PET).
    observed : np.ndarray
        Observed flow values.
    objective : ObjectiveFunction
        Objective function instance.
    warmup_period : int
        Number of timesteps to exclude from objective calculation.
    """
    
    def __init__(
        self,
        model: 'BaseRainfallRunoffModel',
        inputs: pd.DataFrame,
        observed: np.ndarray,
        objective: 'ObjectiveFunction',
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365
    ):
        """
        Initialize wrapper.
        
        Args:
            model: Rainfall-runoff model instance
            inputs: Input DataFrame with precipitation and PET columns
            observed: Observed flow values (1D array)
            objective: Objective function instance
            parameter_bounds: Parameter bounds {name: (min, max)}.
                If None, uses model.get_parameter_bounds()
            warmup_period: Number of timesteps to exclude from objective
        """
        self.model = model
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.objective = objective
        self.warmup_period = warmup_period
        self._maximize = _should_maximize(objective)
        
        # Get bounds
        if parameter_bounds is None:
            self._param_bounds = model.get_parameter_bounds()
        else:
            self._param_bounds = parameter_bounds
        
        self._param_names = list(self._param_bounds.keys())
        
        # Counters for debugging
        self._n_calls = 0
        self._n_failures = 0
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds as list of tuples for sceua.minimize().
        
        Returns:
            List of (min, max) tuples in parameter order
        """
        return [self._param_bounds[name] for name in self._param_names]
    
    def vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert parameter vector to dictionary.
        
        Args:
            vector: 1D array of parameter values
            
        Returns:
            Dictionary {name: value}
        """
        return dict(zip(self._param_names, vector))
    
    def params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to vector.
        
        Args:
            params: Dictionary {name: value}
            
        Returns:
            1D array of parameter values
        """
        return np.array([params.get(name, 0) for name in self._param_names])
    
    @property
    def param_names(self) -> List[str]:
        """Get list of parameter names in order."""
        return self._param_names
    
    def __call__(self, vector: np.ndarray) -> float:
        """
        Evaluate objective function for a parameter vector.
        
        This method is called by sceua.minimize() during optimization.
        Since SCE-UA minimizes, we negate the objective for maximization
        objectives (NSE, KGE, etc.).
        
        Args:
            vector: 1D array of parameter values
            
        Returns:
            Objective value to minimize (negated if maximizing)
        """
        self._n_calls += 1
        params = self.vector_to_params(vector)
        
        # Reset and run model
        self.model.reset()
        self.model.set_parameters(params)
        
        try:
            results = self.model.run(self.inputs)
            
            # Extract simulated flow
            if 'flow' in results.columns:
                simulated = results['flow'].values
            elif 'runoff' in results.columns:
                simulated = results['runoff'].values
            else:
                simulated = results.iloc[:, 0].values
            
            # Apply warmup
            sim = simulated[self.warmup_period:]
            obs = self.observed[self.warmup_period:]
            
            # Calculate objective value - handle both old and new interfaces
            # New interface (pyrrm.objectives): __call__(obs, sim)
            # Old interface (pyrrm.calibration.objective_functions): calculate(sim, obs)
            if hasattr(self.objective, 'calculate'):
                # Old interface: calculate(simulated, observed)
                value = self.objective.calculate(sim, obs)
            else:
                # New interface: __call__(observed, simulated)
                value = self.objective(obs, sim)
            
        except Exception as e:
            self._n_failures += 1
            warnings.warn(f"Simulation failed (call {self._n_calls}): {e}")
            value = np.nan
        
        # Handle NaN - return large penalty value
        if np.isnan(value) or np.isinf(value):
            return 1e10 if self._maximize else 1e10
        
        # SCE-UA minimizes, so negate for maximization objectives
        return -value if self._maximize else value


def run_sceua_direct(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    n_complexes: Optional[int] = None,
    n_points_complex: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_evals: int = 50000,
    max_iter: int = 1000,
    max_tolerant_iter: int = 30,
    tolerance: float = 1e-6,
    x_tolerance: float = 1e-8,
    seed: Optional[int] = None,
    pca_freq: int = 1,
    pca_tol: float = 1e-3,
    x0: Optional[Union[np.ndarray, List[Tuple[float, ...]], Dict[str, float]]] = None,
    max_workers: int = 1,
    callback: Optional[callable] = None,
    verbose: bool = False,
    progress_bar: bool = True,
) -> Dict[str, Any]:
    """
    Run SCE-UA calibration using the direct implementation.
    
    This is an alternative to the SpotPy-based SCE-UA (run_sceua in spotpy_adapter).
    Key advantages:
    - No SpotPy dependency required
    - More configuration options (PCA recovery, convergence criteria)
    - Parallel evaluation support via ThreadPoolExecutor
    - Initial parameter sets can be provided
    
    Parameters
    ----------
    model : BaseRainfallRunoffModel
        Rainfall-runoff model instance to calibrate.
    inputs : pd.DataFrame
        Input data with precipitation and PET columns.
    observed : np.ndarray
        Observed flow values.
    objective : ObjectiveFunction
        Objective function instance (NSE, KGE, etc.).
    parameter_bounds : dict, optional
        Parameter bounds {name: (min, max)}. If None, uses model defaults.
    warmup_period : int, default 365
        Number of timesteps to exclude from objective calculation.
    n_complexes : int, optional
        Number of complexes. If None, automatically determined based on
        dimensionality: min(max(2, log2(n_params) + 5), 15).
    n_points_complex : int, optional
        Points per complex. If None, calculated as 2 * n_complexes + 1.
    alpha : float, default 1.0
        Reflection coefficient for simplex evolution.
    beta : float, default 0.5
        Contraction coefficient for simplex evolution.
    max_evals : int, default 50000
        Maximum number of function evaluations.
    max_iter : int, default 1000
        Maximum number of iterations.
    max_tolerant_iter : int, default 30
        Stop if no improvement for this many iterations.
    tolerance : float, default 1e-6
        Minimum improvement in objective to count as progress.
    x_tolerance : float, default 1e-8
        Stop if parameter ranges shrink below this threshold.
    seed : int, optional
        Random seed for reproducibility.
    pca_freq : int, default 1
        Frequency of PCA recovery (every N iterations).
    pca_tol : float, default 1e-3
        Tolerance for detecting lost dimensions in PCA recovery.
    x0 : array-like or dict, optional
        Initial parameter sets. Can be:
        - np.ndarray of shape (n_sets, n_params)
        - List of tuples [(p1, p2, ...), ...]
        - Dict {name: value} for a single starting point
    max_workers : int, default 1
        Number of parallel workers for function evaluation.
        Set > 1 to enable parallel evaluation via ThreadPoolExecutor.
    callback : callable, optional
        Function called after each iteration with a dict containing:
        ``{'iteration': int, 'nfev': int, 'best_fun': float, 'best_x': array}``.
        Note: best_fun is in minimization space (negated for maximization objectives).
    verbose : bool, default False
        If True, print progress information every 10 iterations.
    progress_bar : bool, default True
        If True and tqdm is available, show a progress bar.
    
    Returns
    -------
    dict
        Calibration results with keys:
        - 'best_parameters': Dict[str, float] - best parameter set
        - 'best_objective': float - best objective value
        - 'all_samples': pd.DataFrame - all evaluated parameter sets
        - 'convergence_diagnostics': dict - nit, nfev, message, success
        - 'runtime_seconds': float - total runtime
    
    Examples
    --------
    >>> from pyrrm.calibration import CalibrationRunner, NSE
    >>> from pyrrm.calibration.sceua_adapter import run_sceua_direct
    >>> 
    >>> result = run_sceua_direct(
    ...     model=model,
    ...     inputs=inputs,
    ...     observed=observed,
    ...     objective=NSE(),
    ...     max_evals=20000,
    ...     n_complexes=5,
    ...     seed=42
    ... )
    >>> print(f"Best NSE: {result['best_objective']:.4f}")
    
    Notes
    -----
    The SCE-UA algorithm is a global optimization method developed for
    calibrating hydrological models. It combines the strengths of:
    - Simplex method (Nelder-Mead) for local search
    - Competitive evolution for global search
    - Shuffling of complexes to share information
    
    This implementation includes enhancements from the literature:
    - PCA recovery for lost dimensions (Chu et al., 2010)
    - Adaptive smoothing parameter (Muttil & Jayawardena, 2008)
    """
    start_time = time.time()
    
    # Create wrapper
    wrapper = SCEUAModelWrapper(
        model, inputs, observed, objective, parameter_bounds, warmup_period
    )
    
    # Get bounds in list format
    bounds = wrapper.get_bounds()
    
    # Handle x0 if provided as dict
    x0_array: Optional[Union[np.ndarray, List[Tuple[float, ...]]]] = None
    if x0 is not None:
        if isinstance(x0, dict):
            # Convert single point dict to array
            x0_array = np.atleast_2d(wrapper.params_to_vector(x0))
        else:
            x0_array = x0
    
    # Run SCE-UA optimization
    # Pass display_maximize so verbose output shows actual objective values
    result: Result = minimize(
        func=wrapper,
        bounds=bounds,
        n_complexes=n_complexes,
        n_points_complex=n_points_complex,
        alpha=alpha,
        beta=beta,
        max_evals=max_evals,
        max_iter=max_iter,
        max_tolerant_iter=max_tolerant_iter,
        tolerance=tolerance,
        x_tolerance=x_tolerance,
        seed=seed,
        pca_freq=pca_freq,
        pca_tol=pca_tol,
        x0=x0_array,
        max_workers=max_workers,
        callback=callback,
        verbose=verbose,
        progress_bar=progress_bar,
        display_maximize=wrapper._maximize,  # Show true objective for maximization
    )
    
    runtime = time.time() - start_time
    
    # Convert best parameters to dict
    best_params = wrapper.vector_to_params(result.x)
    
    # Convert objective back to original scale (un-negate if maximizing)
    best_objective = -result.fun if wrapper._maximize else result.fun
    
    # Build samples DataFrame from all evaluations
    # Convert function values back to original scale
    funv_original = -result.funv if wrapper._maximize else result.funv
    
    all_samples_dict = {
        'iteration': np.arange(len(result.funv)),
        'objective': funv_original,
    }
    # Add parameter columns
    for i, name in enumerate(wrapper.param_names):
        all_samples_dict[name] = result.xv[:, i]
    
    all_samples = pd.DataFrame(all_samples_dict)
    
    return {
        'best_parameters': best_params,
        'best_objective': best_objective,
        'all_samples': all_samples,
        'convergence_diagnostics': {
            'nit': result.nit,
            'nfev': result.nfev,
            'message': result.message,
            'success': result.success,
            'n_simulation_failures': wrapper._n_failures,
        },
        'runtime_seconds': runtime,
    }


def calibrate_sceua(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    max_evals: int = 50000,
    n_complexes: Optional[int] = None,
    seed: Optional[int] = None,
    max_workers: int = 1,
    **kwargs
) -> SCEUACalibrationResult:
    """
    Calibrate model using SCE-UA (convenience function returning dataclass).
    
    This is a simpler interface that returns an SCEUACalibrationResult dataclass.
    For full control over algorithm parameters, use run_sceua_direct() instead.
    
    Parameters
    ----------
    model : BaseRainfallRunoffModel
        Model to calibrate.
    inputs : pd.DataFrame
        Input data.
    observed : np.ndarray
        Observed values.
    objective : ObjectiveFunction
        Objective function.
    parameter_bounds : dict, optional
        Parameter bounds.
    warmup_period : int, default 365
        Warmup period.
    max_evals : int, default 50000
        Maximum function evaluations.
    n_complexes : int, optional
        Number of complexes.
    seed : int, optional
        Random seed.
    max_workers : int, default 1
        Parallel workers.
    **kwargs
        Additional arguments passed to run_sceua_direct().
    
    Returns
    -------
    SCEUACalibrationResult
        Dataclass with calibration results.
    """
    result = run_sceua_direct(
        model=model,
        inputs=inputs,
        observed=observed,
        objective=objective,
        parameter_bounds=parameter_bounds,
        warmup_period=warmup_period,
        max_evals=max_evals,
        n_complexes=n_complexes,
        seed=seed,
        max_workers=max_workers,
        **kwargs
    )
    
    return SCEUACalibrationResult(
        best_parameters=result['best_parameters'],
        best_objective=result['best_objective'],
        all_samples=result['all_samples'],
        convergence_info=result['convergence_diagnostics'],
        runtime_seconds=result['runtime_seconds'],
        success=result['convergence_diagnostics']['success'],
        message=result['convergence_diagnostics']['message'],
    )
