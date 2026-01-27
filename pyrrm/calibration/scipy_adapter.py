"""
SciPy optimization adapter for pyrrm model calibration.

This module provides calibration using scipy.optimize methods:
- differential_evolution: Global optimization (recommended)
- dual_annealing: Simulated annealing
- minimize: Local optimization methods
- basinhopping: Global optimization with local refinement
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Callable
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
import time

from scipy import optimize

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction


def _should_maximize(objective) -> bool:
    """
    Check if an objective function should be maximized.
    
    Handles both old interface (maximize property) and new interface (direction attribute).
    """
    if hasattr(objective, 'direction'):
        return objective.direction == 'maximize'
    elif hasattr(objective, 'maximize'):
        return objective.maximize
    else:
        return True  # Default to maximize


@dataclass
class ScipyCalibrationResult:
    """Container for scipy calibration results."""
    best_parameters: Dict[str, float]
    best_objective: float
    all_samples: pd.DataFrame
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    success: bool = True
    message: str = ""


class ScipyModelWrapper:
    """
    Wrapper to make pyrrm models compatible with scipy.optimize.
    
    Handles parameter transformations, objective calculation, and
    tracking of optimization history.
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
            model: Rainfall-runoff model
            inputs: Input DataFrame
            observed: Observed flow values
            objective: Objective function
            parameter_bounds: Parameter bounds {name: (min, max)}
            warmup_period: Warmup timesteps to exclude
        """
        self.model = model
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.objective = objective
        self.warmup_period = warmup_period
        
        # Get bounds
        if parameter_bounds is None:
            self._param_bounds = model.get_parameter_bounds()
        else:
            self._param_bounds = parameter_bounds
        
        self._param_names = list(self._param_bounds.keys())
        
        # Tracking
        self._history: List[Dict[str, Any]] = []
        self._n_calls = 0
        self._best_value = np.inf if not _should_maximize(objective) else -np.inf
        self._best_params: Dict[str, float] = {}
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds as list of tuples for scipy."""
        return [self._param_bounds[name] for name in self._param_names]
    
    def vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert parameter vector to dictionary."""
        return dict(zip(self._param_names, vector))
    
    def params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to vector."""
        return np.array([params.get(name, 0) for name in self._param_names])
    
    def __call__(self, vector: np.ndarray) -> float:
        """
        Evaluate objective function.
        
        Args:
            vector: Parameter values
            
        Returns:
            Objective value (negated if maximizing, since scipy minimizes)
        """
        self._n_calls += 1
        params = self.vector_to_params(vector)
        
        # Reset and run model
        self.model.reset()
        self.model.set_parameters(params)
        
        try:
            results = self.model.run(self.inputs)
            
            if 'flow' in results.columns:
                simulated = results['flow'].values
            elif 'runoff' in results.columns:
                simulated = results['runoff'].values
            else:
                simulated = results.iloc[:, 0].values
            
            # Apply warmup
            sim = simulated[self.warmup_period:]
            obs = self.observed[self.warmup_period:]
            
            value = self.objective.calculate(sim, obs)
            
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            value = np.nan
        
        # Handle NaN
        if np.isnan(value):
            scipy_value = 1e10  # Large penalty
        else:
            # scipy minimizes, so negate for maximization objectives
            scipy_value = -value if _should_maximize(self.objective) else value
        
        # Track history
        record = {'iteration': self._n_calls, **params, 'objective': value}
        self._history.append(record)
        
        # Track best
        is_better = (
            (_should_maximize(self.objective) and value > self._best_value) or
            (not _should_maximize(self.objective) and value < self._best_value)
        )
        if not np.isnan(value) and is_better:
            self._best_value = value
            self._best_params = params.copy()
        
        return scipy_value
    
    def get_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        return pd.DataFrame(self._history)


def calibrate_differential_evolution(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: Optional[int] = None,
    workers: int = 1,
    polish: bool = True,
    **kwargs
) -> ScipyCalibrationResult:
    """
    Calibrate model using differential evolution.
    
    Differential evolution is a global optimization algorithm that works
    well for multi-modal objective functions common in hydrology.
    
    Args:
        model: Rainfall-runoff model
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        parameter_bounds: Parameter bounds
        warmup_period: Warmup timesteps
        maxiter: Maximum iterations
        popsize: Population size multiplier
        tol: Convergence tolerance
        mutation: Mutation constant or range
        recombination: Recombination constant
        seed: Random seed
        workers: Number of parallel workers (-1 for all CPUs)
        polish: Polish best result with L-BFGS-B
        **kwargs: Additional scipy arguments
        
    Returns:
        ScipyCalibrationResult with best parameters and history
    """
    start_time = time.time()
    
    wrapper = ScipyModelWrapper(
        model, inputs, observed, objective, parameter_bounds, warmup_period
    )
    
    result = optimize.differential_evolution(
        wrapper,
        bounds=wrapper.get_bounds(),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        workers=workers,
        polish=polish,
        **kwargs
    )
    
    runtime = time.time() - start_time
    
    best_params = wrapper.vector_to_params(result.x)
    best_objective = -result.fun if _should_maximize(objective) else result.fun
    
    return ScipyCalibrationResult(
        best_parameters=best_params,
        best_objective=best_objective,
        all_samples=wrapper.get_history(),
        convergence_info={
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev
        },
        runtime_seconds=runtime,
        success=result.success,
        message=result.message
    )


def calibrate_dual_annealing(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    maxiter: int = 1000,
    initial_temp: float = 5230.0,
    restart_temp_ratio: float = 2e-5,
    visit: float = 2.62,
    accept: float = -5.0,
    maxfun: int = 10000000,
    seed: Optional[int] = None,
    **kwargs
) -> ScipyCalibrationResult:
    """
    Calibrate model using dual annealing.
    
    Dual annealing combines classical simulated annealing with fast
    simulated annealing and local search.
    
    Args:
        model: Rainfall-runoff model
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        parameter_bounds: Parameter bounds
        warmup_period: Warmup timesteps
        maxiter: Maximum iterations
        initial_temp: Initial temperature
        restart_temp_ratio: Restart temperature ratio
        visit: Visit parameter
        accept: Acceptance parameter
        maxfun: Maximum function evaluations
        seed: Random seed
        **kwargs: Additional scipy arguments
        
    Returns:
        ScipyCalibrationResult
    """
    start_time = time.time()
    
    wrapper = ScipyModelWrapper(
        model, inputs, observed, objective, parameter_bounds, warmup_period
    )
    
    result = optimize.dual_annealing(
        wrapper,
        bounds=wrapper.get_bounds(),
        maxiter=maxiter,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
        seed=seed,
        **kwargs
    )
    
    runtime = time.time() - start_time
    
    best_params = wrapper.vector_to_params(result.x)
    best_objective = -result.fun if _should_maximize(objective) else result.fun
    
    return ScipyCalibrationResult(
        best_parameters=best_params,
        best_objective=best_objective,
        all_samples=wrapper.get_history(),
        convergence_info={
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev
        },
        runtime_seconds=runtime,
        success=result.success,
        message=result.message
    )


def calibrate_basinhopping(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    niter: int = 100,
    T: float = 1.0,
    stepsize: float = 0.5,
    seed: Optional[int] = None,
    **kwargs
) -> ScipyCalibrationResult:
    """
    Calibrate model using basin-hopping.
    
    Basin-hopping is a two-phase method that combines a global stepping
    algorithm with local minimization.
    
    Args:
        model: Rainfall-runoff model
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        parameter_bounds: Parameter bounds
        warmup_period: Warmup timesteps
        niter: Number of basin-hopping iterations
        T: Temperature for acceptance test
        stepsize: Maximum step size
        seed: Random seed
        **kwargs: Additional scipy arguments
        
    Returns:
        ScipyCalibrationResult
    """
    start_time = time.time()
    
    wrapper = ScipyModelWrapper(
        model, inputs, observed, objective, parameter_bounds, warmup_period
    )
    
    # Initial guess (middle of bounds)
    bounds = wrapper.get_bounds()
    x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
    
    # Create bounds for local minimizer
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds
    }
    
    if seed is not None:
        np.random.seed(seed)
    
    result = optimize.basinhopping(
        wrapper,
        x0,
        niter=niter,
        T=T,
        stepsize=stepsize,
        minimizer_kwargs=minimizer_kwargs,
        **kwargs
    )
    
    runtime = time.time() - start_time
    
    best_params = wrapper.vector_to_params(result.x)
    best_objective = -result.fun if _should_maximize(objective) else result.fun
    
    return ScipyCalibrationResult(
        best_parameters=best_params,
        best_objective=best_objective,
        all_samples=wrapper.get_history(),
        convergence_info={
            'success': True,  # Basin-hopping doesn't have success flag
            'message': result.message if hasattr(result, 'message') else '',
            'nit': result.nit,
            'nfev': result.nfev
        },
        runtime_seconds=runtime,
        success=True,
        message=''
    )


def calibrate_minimize(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    x0: Optional[Dict[str, float]] = None,
    method: str = 'L-BFGS-B',
    **kwargs
) -> ScipyCalibrationResult:
    """
    Calibrate model using scipy.optimize.minimize.
    
    This is a local optimization method - results depend on initial guess.
    Use for polishing results from global methods.
    
    Args:
        model: Rainfall-runoff model
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        parameter_bounds: Parameter bounds
        warmup_period: Warmup timesteps
        x0: Initial parameter guess (dict or None for middle of bounds)
        method: Optimization method ('L-BFGS-B', 'SLSQP', 'TNC', etc.)
        **kwargs: Additional scipy arguments
        
    Returns:
        ScipyCalibrationResult
    """
    start_time = time.time()
    
    wrapper = ScipyModelWrapper(
        model, inputs, observed, objective, parameter_bounds, warmup_period
    )
    
    bounds = wrapper.get_bounds()
    
    # Initial guess
    if x0 is None:
        x0_vec = np.array([(b[0] + b[1]) / 2 for b in bounds])
    else:
        x0_vec = wrapper.params_to_vector(x0)
    
    result = optimize.minimize(
        wrapper,
        x0_vec,
        method=method,
        bounds=bounds,
        **kwargs
    )
    
    runtime = time.time() - start_time
    
    best_params = wrapper.vector_to_params(result.x)
    best_objective = -result.fun if _should_maximize(objective) else result.fun
    
    return ScipyCalibrationResult(
        best_parameters=best_params,
        best_objective=best_objective,
        all_samples=wrapper.get_history(),
        convergence_info={
            'success': result.success,
            'message': result.message,
            'nit': result.nit if hasattr(result, 'nit') else 0,
            'nfev': result.nfev
        },
        runtime_seconds=runtime,
        success=result.success,
        message=result.message
    )


# Convenience alias
def calibrate_scipy(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    method: str = 'differential_evolution',
    **kwargs
) -> ScipyCalibrationResult:
    """
    Calibrate model using scipy.optimize methods.
    
    Args:
        model: Rainfall-runoff model
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        parameter_bounds: Parameter bounds
        warmup_period: Warmup timesteps
        method: Method name ('differential_evolution', 'dual_annealing', 
                'basinhopping', 'minimize')
        **kwargs: Method-specific arguments
        
    Returns:
        ScipyCalibrationResult
    """
    methods = {
        'differential_evolution': calibrate_differential_evolution,
        'dual_annealing': calibrate_dual_annealing,
        'basinhopping': calibrate_basinhopping,
        'minimize': calibrate_minimize,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Available: {list(methods.keys())}")
    
    return methods[method](
        model, inputs, observed, objective, parameter_bounds, warmup_period, **kwargs
    )
