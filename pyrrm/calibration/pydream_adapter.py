"""
PyDREAM adapter for pyrrm model calibration.

This module provides an interface between pyrrm models and the PyDREAM library,
which implements the MT-DREAM(ZS) algorithm (Multi-Try DiffeRential Evolution
Adaptive Metropolis with snooker updates).

PyDREAM Reference:
    Laloy, E. & Vrugt, J. A. (2012). High-dimensional posterior exploration
    of hydrologic models using multiple-try DREAM(ZS) and high-performance
    computing. Water Resources Research, 48, W01526.

Key features of PyDREAM (MT-DREAM(ZS)):
- Multi-try sampling for better mixing
- Snooker updates for jumping between modes
- Parallel tempering support
- More advanced crossover adaptation
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Callable
import logging
import sys
import numpy as np
import pandas as pd
import warnings
import time
import os
import multiprocessing as mp
from datetime import datetime
import fcntl  # For file locking on Unix

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction

# Check for PyDREAM availability
try:
    from pydream.core import run_dream as pydream_run_dream
    from pydream.parameters import SampledParam
    from pydream.convergence import Gelman_Rubin
    PYDREAM_AVAILABLE = True
except ImportError:
    PYDREAM_AVAILABLE = False

# Check for scipy (needed for parameter priors)
try:
    from scipy.stats import uniform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


class PyDREAMProgressWriter:
    """
    Process-safe CSV writer for tracking PyDREAM calibration progress.
    
    This class enables real-time monitoring of PyDREAM calibrations by writing
    each likelihood evaluation to a CSV file.
    Uses file locking for process-safe writes when running with multiple chains.
    """
    
    def __init__(
        self,
        filename: str,
        parameter_names: List[str],
        write_interval: int = 1
    ):
        """
        Initialize the progress writer.
        
        Args:
            filename: Path to the CSV file to write
            parameter_names: List of parameter names (for CSV header)
            write_interval: Write every N evaluations (1 = write all)
        """
        self.filename = filename
        self.parameter_names = parameter_names
        self.write_interval = write_interval
        self._counter = mp.Value('i', 0)  # Shared counter for all processes
        self._initialized = mp.Value('b', False)
        self._lock = mp.Lock()
    
    def initialize_file(self):
        """Create the CSV file with header (called once at start)."""
        with self._lock:
            if not self._initialized.value:
                # Build header: like1, parX, parY, ..., chain, simulation
                header = ['like1']
                for name in self.parameter_names:
                    header.append(f'par{name}')
                header.extend(['chain', 'simulation'])
                
                with open(self.filename, 'w') as f:
                    f.write(','.join(header) + '\n')
                
                self._initialized.value = True
    
    def write_sample(
        self,
        likelihood: float,
        parameters: np.ndarray,
        chain_id: int = 0
    ):
        """
        Write a sample to the CSV file (process-safe).
        
        Args:
            likelihood: Log-likelihood value
            parameters: Parameter values array
            chain_id: Chain identifier
        """
        with self._lock:
            self._counter.value += 1
            counter = self._counter.value
            
            # Only write at specified interval
            if counter % self.write_interval != 0:
                return
            
            # Build row
            row = [f'{likelihood:.10e}']
            for val in parameters:
                row.append(f'{val:.10e}')
            row.append(str(chain_id))
            row.append(str(counter))
            
            # Append to file with locking
            try:
                with open(self.filename, 'a') as f:
                    # Use fcntl for file locking on Unix
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(','.join(row) + '\n')
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (IOError, OSError):
                # Fallback: just write without locking
                with open(self.filename, 'a') as f:
                    f.write(','.join(row) + '\n')
    
    @property
    def sample_count(self) -> int:
        """Return the current sample count."""
        return self._counter.value


class PyDREAMLikelihood:
    """
    Picklable likelihood class for PyDREAM multiprocessing.
    
    PyDREAM uses multiprocessing for multiple chains, which requires
    the likelihood function to be picklable. This class stores all
    necessary data for computing the likelihood.
    
    Supports optional progress tracking via CSV file for real-time monitoring.
    """
    
    def __init__(
        self,
        model_class: type,
        model_params: Dict[str, float],
        inputs_dict: Dict[str, np.ndarray],
        observed: np.ndarray,
        objective_class: type,
        objective_kwargs: Dict[str, Any],
        parameter_names: List[str],
        warmup_period: int,
        catchment_area_km2: Optional[float] = None,
        dbname: Optional[str] = None,
        write_interval: int = 1
    ):
        """
        Initialize the picklable likelihood.
        
        Args:
            model_class: Class of the model (e.g., Sacramento)
            model_params: Current model parameters
            inputs_dict: Input data as dict of arrays (rainfall, pet)
            observed: Observed flow values
            objective_class: Class of objective function
            objective_kwargs: Kwargs for objective function
            parameter_names: List of parameter names in order
            warmup_period: Warmup timesteps
            catchment_area_km2: Optional catchment area for scaling
            dbname: If provided, writes progress to {dbname}.csv for monitoring
            write_interval: Write every N evaluations (default: 1 = all)
        """
        self.model_class = model_class
        self.model_params = model_params
        self.inputs_dict = inputs_dict
        self.observed = observed
        self.objective_class = objective_class
        self.objective_kwargs = objective_kwargs
        self.parameter_names = parameter_names
        self.warmup_period = warmup_period
        self.catchment_area_km2 = catchment_area_km2
        
        # Progress tracking
        self.dbname = dbname
        self.write_interval = write_interval
        self._progress_writer = None
        self._eval_count = 0
        
        # Initialize progress file if dbname provided
        if dbname:
            self._init_progress_file()
    
    def _init_progress_file(self):
        """Initialize the CSV progress file with header."""
        filename = f"{self.dbname}.csv"
        
        # Build header: like1, parX, parY, ..., chain, simulation
        header = ['like1']
        for name in self.parameter_names:
            header.append(f'par{name}')
        header.extend(['chain', 'simulation'])
        
        # Write header
        with open(filename, 'w') as f:
            f.write(','.join(header) + '\n')
    
    def _write_progress(self, likelihood: float, parameters: np.ndarray, chain_id: int = 0):
        """Write a sample to the progress CSV file."""
        if not self.dbname:
            return
        
        self._eval_count += 1
        
        # Only write at specified interval
        if self._eval_count % self.write_interval != 0:
            return
        
        filename = f"{self.dbname}.csv"
        
        # Build row
        row = [f'{likelihood:.10e}']
        for val in parameters.flatten():
            row.append(f'{val:.10e}')
        row.append(str(chain_id))
        row.append(str(self._eval_count))
        
        # Append to file with file locking for process safety
        try:
            with open(filename, 'a') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(','.join(row) + '\n')
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError, AttributeError):
            # Fallback for Windows or if fcntl unavailable
            with open(filename, 'a') as f:
                f.write(','.join(row) + '\n')
    
    def __call__(self, param_array: np.ndarray) -> float:
        """
        Compute log-likelihood for a parameter set.
        
        PyDREAM's ``Model.total_logp`` wraps this function, evaluates
        the prior from ``SampledParam`` objects, and returns the
        ``(log_prior, log_likelihood)`` tuple that the sampler expects.
        This function therefore returns a **single float**.
        
        .. warning::
           Do **not** use ``multitry=2`` with ``parallel=False``.
           PyDREAM's ``mt_evaluate_logps`` has a bug in that code path:
           it passes the full (2, n_params) proposal matrix to
           ``total_logp`` instead of iterating row-by-row, which
           breaks both the prior calculation and this likelihood.
           Use ``multitry=1`` or ``multitry >= 3`` instead.
        
        Args:
            param_array: Array of parameter values
            
        Returns:
            Log-likelihood value (single float)
        """
        # Recreate objects (necessary for multiprocessing)
        model = self.model_class()
        if self.catchment_area_km2 is not None:
            model.set_catchment_area(self.catchment_area_km2)
        
        objective = self.objective_class(**self.objective_kwargs)
        
        # Recreate inputs DataFrame
        inputs = pd.DataFrame(self.inputs_dict)
        
        # Convert array to parameter dictionary (only calibration params)
        calib_params = dict(zip(self.parameter_names, param_array.flatten()))
        
        # IMPORTANT: First set ALL parameters from stored model_params (includes fixed params)
        # Then update with the calibration parameters being sampled
        all_params = self.model_params.copy()
        all_params.update(calib_params)
        
        # Set parameters
        model.reset()
        model.set_parameters(all_params)
        
        try:
            # Run model
            results = model.run(inputs)
            
            # Get flow output
            if 'flow' in results.columns:
                simulated = results['flow'].values
            elif 'runoff' in results.columns:
                simulated = results['runoff'].values
            else:
                simulated = results.iloc[:, 0].values
            
            # Apply warmup
            sim = simulated[self.warmup_period:]
            obs = self.observed[self.warmup_period:]
            
            # Check for valid arrays
            if len(sim) == 0 or len(obs) == 0:
                self._write_progress(-np.inf, param_array)
                return -np.inf
            
            # Calculate objective value
            # Use callable syntax to support both old (calculate method) and new (__call__) interfaces
            if hasattr(objective, 'calculate'):
                obj_value = objective.calculate(sim, obs)
            else:
                # New-style objective function uses __call__ directly
                obj_value = objective(obs, sim)  # Note: new interface takes (obs, sim) not (sim, obs)
            
            if np.isnan(obj_value):
                log_likelihood = -np.inf
            else:
                # PyDREAM maximizes log-likelihood
                if _should_maximize(objective):
                    log_likelihood = obj_value
                else:
                    log_likelihood = -obj_value
            
            # Write progress to CSV
            self._write_progress(log_likelihood, param_array)
            
            return log_likelihood
            
        except Exception as e:
            # Log the actual error for debugging
            import sys
            _log.error("PyDREAM likelihood error: %s: %s", type(e).__name__, e)
            # Write failed evaluation
            self._write_progress(-np.inf, param_array)
            return -np.inf


def create_pydream_likelihood(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_names: List[str],
    warmup_period: int = 365,
    dbname: Optional[str] = None,
    write_interval: int = 1
) -> PyDREAMLikelihood:
    """
    Create a picklable likelihood object compatible with PyDREAM.
    
    PyDREAM expects a likelihood function that takes a parameter array
    and returns a log-likelihood value. This function returns a picklable
    class instance that can be used with multiprocessing.
    
    Args:
        model: Rainfall-runoff model instance
        inputs: Input DataFrame with precipitation and PET
        observed: Observed flow values
        objective: Objective function instance
        parameter_names: List of parameter names in order
        warmup_period: Warmup timesteps (excluded from objective calculation)
        dbname: If provided, writes progress to {dbname}.csv for real-time monitoring
        write_interval: Write every N evaluations (default: 1 = all)
        
    Returns:
        Picklable likelihood object compatible with PyDREAM
    """
    # Get model class and current parameters
    model_class = type(model)
    model_params = model.get_parameters()
    
    # Get catchment area if set
    catchment_area_km2 = getattr(model, '_catchment_area_km2', None)
    
    # Convert inputs to dict of arrays (picklable)
    inputs_dict = {col: inputs[col].values for col in inputs.columns}
    
    # Get objective class and kwargs
    objective_class = type(objective)
    # Most objective functions don't need kwargs, but handle common ones
    objective_kwargs = {}
    if hasattr(objective, 'weights'):
        objective_kwargs['weights'] = objective.weights
    
    observed_flat = np.asarray(observed).flatten()
    
    return PyDREAMLikelihood(
        model_class=model_class,
        model_params=model_params,
        inputs_dict=inputs_dict,
        observed=observed_flat,
        objective_class=objective_class,
        objective_kwargs=objective_kwargs,
        parameter_names=parameter_names,
        warmup_period=warmup_period,
        catchment_area_km2=catchment_area_km2,
        dbname=dbname,
        write_interval=write_interval
    )


def create_pydream_parameters(
    parameter_bounds: Dict[str, Tuple[float, float]]
) -> Tuple[List, List[str]]:
    """
    Convert pyrrm parameter bounds to PyDREAM SampledParam objects.
    
    Creates uniform prior distributions for each parameter based on the
    provided bounds.
    
    Args:
        parameter_bounds: Dictionary of {param_name: (min, max)}
        
    Returns:
        Tuple of (list of SampledParam objects, list of parameter names)
    """
    if not PYDREAM_AVAILABLE:
        raise ImportError(
            "PyDREAM is required. Install from: "
            "https://github.com/LoLab-VU/PyDREAM"
        )
    
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for PyDREAM parameter priors. "
            "Install with: pip install scipy"
        )
    
    parameters = []
    param_names = []
    
    for name, (low, high) in parameter_bounds.items():
        # Create uniform prior using scipy.stats.uniform
        # uniform(loc=a, scale=b-a) gives uniform distribution on [a, b]
        param = SampledParam(uniform, loc=low, scale=high - low)
        parameters.append(param)
        param_names.append(name)
    
    return parameters, param_names


def run_pydream(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    niterations: int = 10000,
    nchains: int = 5,
    multitry: int = 5,
    snooker: float = 0.1,
    nCR: int = 3,
    adapt_crossover: bool = False,  # Disabled by default due to PyDREAM bug
    adapt_gamma: bool = False,
    DEpairs: int = 1,
    gamma_levels: int = 1,
    p_gamma_unity: float = 0.2,
    history_thin: int = 10,
    start: Optional[List[np.ndarray]] = None,
    start_random: bool = True,
    save_history: bool = False,
    model_name: str = 'pyrrm_calibration',
    dbname: Optional[str] = None,
    dbformat: str = 'csv',
    write_interval: int = 1,
    parallel: bool = False,
    mp_context: Optional[str] = None,
    verbose: bool = True,
    nverbose: int = 100,
    hardboundaries: bool = True,
    convergence_check: bool = True,
    convergence_threshold: float = 1.05,
    # Batch mode parameters (optional - enables early stopping)
    batch_size: Optional[int] = None,
    min_iterations: int = 300,
    patience: int = 2,
    post_convergence_iterations: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Run PyDREAM (MT-DREAM(ZS)) calibration algorithm.
    
    PyDREAM implements the Multi-Try DREAM(ZS) algorithm which offers
    several advantages over standard DREAM:
    - Multi-try sampling for better mixing in multi-modal posteriors
    - Snooker updates for jumping between modes
    - Built-in parallel processing
    - Real-time progress monitoring via CSV output
    
    Progress Monitoring:
        Set `dbname` to enable real-time progress tracking. Each likelihood
        evaluation will be written to `{dbname}.csv` in a format compatible
        allowing you to monitor calibration progress using the
        calibration_monitor notebook.
        
        Example:
            result = run_pydream(..., dbname='pydream_calib')
            # Monitor progress: open pydream_calib.csv in calibration_monitor
    
    Parallelization:
        PyDREAM has TWO levels of parallelization:
        
        1. **Chain-level parallelization** (automatic):
           Multiple MCMC chains run in parallel using a custom DreamPool.
           This is controlled by `nchains` - more chains = more parallelism.
           This happens automatically and requires no special setup.
        
        2. **Multi-try parallelization** (optional, via `parallel=True`):
           When `multitry > 1`, the multi-try proposal evaluations can be
           parallelized. Set `parallel=True` to enable this. The number of
           parallel workers equals `multitry`.
        
        For best performance with 8 cores:
        - Set `nchains=8` for 8 parallel chains (automatic)
        - Optionally set `parallel=True` with `multitry=5` for additional
          parallelism within each chain iteration
    
    Args:
        model: Rainfall-runoff model instance
        inputs: Input DataFrame with precipitation and PET columns
        observed: Observed flow values
        objective: Objective function instance
        parameter_bounds: Dict of {param_name: (min, max)}.
                         If None, uses model's default bounds
        warmup_period: Warmup timesteps (excluded from objective)
        niterations: Number of iterations per chain
        nchains: Number of parallel chains (minimum: 2*DEpairs+1).
                 Each chain runs in a separate process automatically.
        multitry: Number of proposal points per iteration (1 = standard DREAM).
                  When parallel=True, these are evaluated in parallel.
        snooker: Probability of snooker update (0 to disable)
        nCR: Number of crossover values
        adapt_crossover: Whether to adapt crossover probabilities.
                        WARNING: Disabled by default due to PyDREAM bug in
                        estimate_crossover_probabilities().
        adapt_gamma: Whether to adapt gamma (step size) levels
        DEpairs: Number of chain pairs for differential evolution
        gamma_levels: Number of gamma levels
        p_gamma_unity: Probability of gamma=1 (full step)
        history_thin: Thinning factor for history storage
        start: List of starting positions for each chain
        start_random: If True, initialize chains randomly from prior
        save_history: Whether to save history files
        model_name: Prefix for saved files
        dbname: Database name for progress tracking. If provided, writes
                progress to {dbname}.csv for real-time monitoring.
        dbformat: Output format ('csv' only for now)
        write_interval: Write every N evaluations (1 = all, 10 = every 10th)
        parallel: Whether to use parallel processing for multi-try evaluations.
                  When True and multitry > 1, proposal points are evaluated
                  in parallel using `multitry` worker processes.
        mp_context: Multiprocessing context ('fork', 'spawn', 'forkserver').
                    If None, uses system default. 'spawn' is safest but slower.
                    'fork' is faster on Unix but may have issues with some libraries.
        verbose: Whether to print progress
        nverbose: Print progress every N iterations (default: 100).
                  Note: Progress from worker processes may not display in
                  Jupyter notebooks due to multiprocessing buffering.
                  Use `dbname` for reliable progress tracking.
        hardboundaries: Whether to enforce hard parameter boundaries
        convergence_check: Whether to check Gelman-Rubin convergence
        convergence_threshold: Gelman-Rubin threshold (default: 1.2 in normal mode, 1.05 recommended for batch mode)
        batch_size: If set (>0), enables batch mode with early stopping.
                   Runs iterations in batches of this size, checking GR convergence
                   after each batch and stopping early when converged.
                   If None or 0, runs all iterations without early stopping (normal mode).
        min_iterations: Minimum iterations before checking convergence (batch mode only)
        patience: Number of consecutive converged batches required before stopping (batch mode only)
        post_convergence_iterations: Additional iterations to run after convergence
                                    is detected, for robust posterior sampling (batch mode only).
                                    Default: 1000 for sufficient posterior samples.
        **kwargs: Additional arguments passed to PyDREAM
        
    Returns:
        Dictionary containing:
            - best_parameters: Best parameter set found
            - best_objective: Best objective value
            - all_samples: DataFrame with all sampled parameters
            - log_likelihoods: Array of log-likelihood values
            - convergence: Gelman-Rubin statistic (if convergence_check=True)
            - runtime_seconds: Total runtime
            
    Example:
        # Run with progress monitoring
        result = run_pydream(
            model, inputs, observed, objective,
            nchains=8,
            niterations=5000,
            dbname='pydream_calib',  # Writes to pydream_calib.csv
            verbose=True
        )
        # Monitor: open pydream_calib.csv in calibration_monitor notebook
    """
    if not PYDREAM_AVAILABLE:
        raise ImportError(
            "PyDREAM is required for this calibration method. "
            "Install from: https://github.com/LoLab-VU/PyDREAM "
            "using: pip install pydream"
        )
    
    start_time = time.time()
    
    # Get parameter bounds
    if parameter_bounds is None:
        parameter_bounds = model.get_parameter_bounds()
    
    # Create PyDREAM parameters
    pydream_params, param_names = create_pydream_parameters(parameter_bounds)
    
    # Create likelihood function with optional progress tracking
    likelihood_func = create_pydream_likelihood(
        model=model,
        inputs=inputs,
        observed=observed,
        objective=objective,
        parameter_names=param_names,
        warmup_period=warmup_period,
        dbname=dbname,
        write_interval=write_interval
    )
    
    if dbname and verbose:
        _log.info("Progress tracking: %s.csv", dbname)
    
    # Validate chain count
    min_chains = 2 * DEpairs + 1
    if nchains < min_chains:
        warnings.warn(
            f"nchains ({nchains}) is less than minimum ({min_chains}). "
            f"Increasing to {min_chains}."
        )
        nchains = min_chains
    
    # Check if batch mode is enabled
    use_batch_mode = batch_size is not None and batch_size > 0
    
    # Run PyDREAM
    if verbose:
        mode_str = "BATCH MODE (with early stopping)" if use_batch_mode else "NORMAL MODE"
        _log.info("PyDREAM (MT-DREAM(ZS)) - %s", mode_str)
        _log.info("  %d iterations/chain%s, %d chains, %d multi-try%s, "
                  "snooker=%.1f%%, DEpairs=%d, %d params",
                  niterations, " (max)" if use_batch_mode else "",
                  nchains, multitry, " (parallel)" if parallel else "",
                  snooker * 100, DEpairs, len(param_names))
        if use_batch_mode:
            _log.info("  batch_size=%d, min_iter=%d, GR<%s for %d batches",
                      batch_size, min_iterations, convergence_threshold, patience)
            if post_convergence_iterations > 0:
                _log.info("  post-convergence: %d iterations", post_convergence_iterations)
        elif convergence_check:
            _log.info("  convergence check: GR < %s", convergence_threshold)
    
    # =========================================================================
    # BATCH MODE: Run in batches with convergence-based early stopping
    # =========================================================================
    if use_batch_mode:
        # Track convergence
        convergence_history = []
        total_iterations = 0
        batches_converged = 0
        early_stopped = False
        convergence_iteration = None
        post_convergence_remaining = 0
        in_post_convergence_phase = False
        
        # Combined results storage
        all_sampled_params = None
        all_log_ps = None
        current_positions = start  # Starting positions (None = random)
        
        batch_num = 0
        while total_iterations < niterations:
            batch_num += 1
            remaining = niterations - total_iterations
            current_batch = min(batch_size, remaining)
            
            if verbose:
                _log.info("[Batch %d] Running %d iterations (total: %d -> %d)",
                          batch_num, current_batch, total_iterations,
                          total_iterations + current_batch)
            
            # Determine if this is a restart
            is_restart = current_positions is not None and batch_num > 1
            
            # Run batch
            try:
                batch_sampled, batch_log_ps = pydream_run_dream(
                    parameters=pydream_params,
                    likelihood=likelihood_func,
                    niterations=current_batch,
                    nchains=nchains,
                    multitry=multitry,
                    snooker=snooker,
                    nCR=nCR,
                    adapt_crossover=False,  # Disabled due to PyDREAM bug with batch restarts
                    adapt_gamma=adapt_gamma,
                    DEpairs=DEpairs,
                    gamma_levels=gamma_levels,
                    p_gamma_unity=p_gamma_unity,
                    history_thin=history_thin,
                    start=current_positions,
                    start_random=(current_positions is None),
                    save_history=save_history,
                    model_name=model_name,
                    parallel=parallel,
                    mp_context=mp_context,
                    verbose=False,  # Suppress per-batch verbose
                    nverbose=nverbose,
                    hardboundaries=hardboundaries,
                    **kwargs
                )
            except Exception as e:
                warnings.warn(f"Batch {batch_num} failed: {e}")
                break
            
            # Update total iterations
            total_iterations += current_batch
            
            # Accumulate results
            if all_sampled_params is None:
                all_sampled_params = batch_sampled
                all_log_ps = batch_log_ps
            else:
                all_sampled_params = [
                    np.vstack([prev, curr]) 
                    for prev, curr in zip(all_sampled_params, batch_sampled)
                ]
                all_log_ps = [
                    np.vstack([prev, curr])
                    for prev, curr in zip(all_log_ps, batch_log_ps)
                ]
            
            # Update starting positions for next batch
            current_positions = [chain[-1, :] for chain in batch_sampled]
            
            # Check convergence (only after minimum iterations)
            if total_iterations >= min_iterations:
                try:
                    gr_values = Gelman_Rubin(all_sampled_params)
                    gr_dict = dict(zip(param_names, gr_values))
                    max_gr = float(np.max(gr_values))
                    mean_gr = float(np.mean(gr_values))
                    is_converged = np.all(gr_values < convergence_threshold)
                    
                    convergence_history.append({
                        'iteration': total_iterations,
                        'max_gr': max_gr,
                        'mean_gr': mean_gr,
                        'converged': is_converged,
                        'gr_values': gr_dict.copy()
                    })
                    
                    if verbose:
                        status = "CONVERGED" if is_converged else "not converged"
                        if in_post_convergence_phase:
                            status += f" (post-conv: {post_convergence_remaining} remaining)"
                        _log.info("GR check: max=%.4f, mean=%.4f -> %s",
                                  max_gr, mean_gr, status)

                    if in_post_convergence_phase:
                        post_convergence_remaining -= current_batch
                        if post_convergence_remaining <= 0:
                            if verbose:
                                _log.info("POST-CONVERGENCE COMPLETE | converged at iter %d, "
                                          "post-conv samples=%d, total=%d",
                                          convergence_iteration,
                                          post_convergence_iterations, total_iterations)
                            break
                    elif is_converged:
                        batches_converged += 1
                        if batches_converged >= patience:
                            early_stopped = True
                            convergence_iteration = total_iterations

                            if post_convergence_iterations > 0:
                                in_post_convergence_phase = True
                                post_convergence_remaining = post_convergence_iterations
                                if verbose:
                                    _log.info("CONVERGENCE DETECTED at iter %d | max GR=%.4f < %s, "
                                              "running %d post-convergence iterations",
                                              convergence_iteration, max_gr,
                                              convergence_threshold,
                                              post_convergence_iterations)
                            else:
                                if verbose:
                                    _log.info("EARLY STOPPING: converged for %d batches | "
                                              "total=%d, max GR=%.4f < %s",
                                              patience, total_iterations, max_gr,
                                              convergence_threshold)
                                break
                    else:
                        batches_converged = 0  # Reset counter
                        
                except Exception as e:
                    if verbose:
                        _log.warning("GR check failed: %s", e)
        
        # Use batch results
        if all_sampled_params is None:
            raise RuntimeError("No samples collected in batch mode")
        
        sampled_params = all_sampled_params
        log_ps = all_log_ps
        
    # =========================================================================
    # NORMAL MODE: Run all iterations at once
    # =========================================================================
    else:
        try:
            sampled_params, log_ps = pydream_run_dream(
                parameters=pydream_params,
                likelihood=likelihood_func,
                niterations=niterations,
                nchains=nchains,
                multitry=multitry,
                snooker=snooker,
                nCR=nCR,
                adapt_crossover=adapt_crossover,
                adapt_gamma=adapt_gamma,
                DEpairs=DEpairs,
                gamma_levels=gamma_levels,
                p_gamma_unity=p_gamma_unity,
                history_thin=history_thin,
                start=start,
                start_random=start_random,
                save_history=save_history,
                model_name=model_name,
                parallel=parallel,
                mp_context=mp_context,
                verbose=verbose,
                nverbose=nverbose,
                hardboundaries=hardboundaries,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"PyDREAM sampling failed: {e}")
        
        # Initialize batch-mode variables for consistent return
        early_stopped = False
        convergence_iteration = None
        convergence_history = []
        total_iterations = niterations
    
    runtime = time.time() - start_time
    
    # Process results
    # sampled_params is a list of arrays [chain][iteration, parameter]
    # log_ps is a list of arrays [chain][iteration, 1]
    
    # Concatenate all chains
    all_samples_array = np.vstack(sampled_params)
    all_log_ps = np.vstack(log_ps).flatten()
    
    # Find best parameters (highest log-likelihood)
    best_idx = np.argmax(all_log_ps)
    best_params_array = all_samples_array[best_idx]
    best_params = dict(zip(param_names, best_params_array))
    best_log_likelihood = all_log_ps[best_idx]
    
    # Convert to objective value
    # Reverse the transformation done in likelihood function
    if _should_maximize(objective):
        best_objective = best_log_likelihood
    else:
        best_objective = -best_log_likelihood
    
    # Build samples DataFrame
    samples_dict = {
        'iteration': np.arange(len(all_samples_array)),
        'log_likelihood': all_log_ps,
    }
    for i, name in enumerate(param_names):
        samples_dict[name] = all_samples_array[:, i]
    
    # Add chain identifiers
    chain_ids = []
    for chain_idx, chain_samples in enumerate(sampled_params):
        chain_ids.extend([chain_idx] * len(chain_samples))
    samples_dict['chain'] = chain_ids
    
    all_samples_df = pd.DataFrame(samples_dict)
    
    # Compute convergence diagnostics
    convergence_diagnostics = {}
    if convergence_check and len(sampled_params) > 1:
        try:
            gr_values = Gelman_Rubin(sampled_params)
            convergence_diagnostics['gelman_rubin'] = dict(zip(param_names, gr_values))
            convergence_diagnostics['converged'] = np.all(gr_values < convergence_threshold)
            convergence_diagnostics['max_gr'] = float(np.max(gr_values))
            convergence_diagnostics['mean_gr'] = float(np.mean(gr_values))
            
            if verbose:
                _log.info("Gelman-Rubin diagnostics (threshold: %s):", convergence_threshold)
                for name, gr in zip(param_names, gr_values):
                    status = "OK" if gr < convergence_threshold else "FAIL"
                    _log.info("  %s %s: %.4f", status, name, gr)
                _log.info("Converged: %s (max GR: %.4f)",
                          convergence_diagnostics['converged'],
                          convergence_diagnostics['max_gr'])
                if not convergence_diagnostics['converged']:
                    _log.warning("Not converged - consider increasing iterations "
                                 "or checking parameter bounds")
        except Exception as e:
            warnings.warn(f"Could not compute Gelman-Rubin: {e}")
            convergence_diagnostics['error'] = str(e)
    
    if verbose:
        if use_batch_mode and early_stopped:
            _log.info("PyDREAM completed (EARLY STOPPED) in %.1fs | total_iter=%d (max=%d)",
                      runtime, total_iterations, niterations)
        else:
            _log.info("PyDREAM completed in %.1fs", runtime)
        _log.info("Best %s: %.6f", objective.name, best_objective)
    
    # Add batch mode info to convergence diagnostics
    if use_batch_mode:
        convergence_diagnostics['early_stopped'] = early_stopped
        convergence_diagnostics['total_iterations'] = total_iterations
        convergence_diagnostics['convergence_iteration'] = convergence_iteration
        convergence_diagnostics['convergence_history'] = convergence_history
    
    return {
        'best_parameters': best_params,
        'best_objective': best_objective,
        'all_samples': all_samples_df,
        'log_likelihoods': all_log_ps,
        'sampled_params_by_chain': sampled_params,
        'log_ps_by_chain': log_ps,
        'convergence_diagnostics': convergence_diagnostics,
        'runtime_seconds': runtime,
        'parameter_names': param_names,
        # Batch mode specific
        'early_stopped': early_stopped if use_batch_mode else None,
        'total_iterations': total_iterations,
        'convergence_history': convergence_history if use_batch_mode else None,
    }


def continue_pydream(
    model: 'BaseRainfallRunoffModel',
    inputs: pd.DataFrame,
    observed: np.ndarray,
    objective: 'ObjectiveFunction',
    previous_result: Dict[str, Any],
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    warmup_period: int = 365,
    niterations: int = 10000,
    model_name: str = 'pyrrm_calibration',
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Continue a previous PyDREAM run from where it left off.
    
    This is useful for extending a run that hasn't converged yet.
    
    Args:
        model: Rainfall-runoff model instance
        inputs: Input DataFrame
        observed: Observed flow values
        objective: Objective function
        previous_result: Result dictionary from previous run_pydream call
        parameter_bounds: Parameter bounds (must match previous run)
        warmup_period: Warmup period
        niterations: Additional iterations to run
        model_name: Model name for history files
        verbose: Print progress
        **kwargs: Additional arguments for PyDREAM
        
    Returns:
        Combined results from both runs
    """
    if not PYDREAM_AVAILABLE:
        raise ImportError("PyDREAM is required for this method.")
    
    # Get last positions from each chain
    sampled_params_by_chain = previous_result['sampled_params_by_chain']
    nchains = len(sampled_params_by_chain)
    
    # Starting positions: last point from each chain
    start_positions = [chain[-1, :] for chain in sampled_params_by_chain]
    
    # Run continuation
    new_result = run_pydream(
        model=model,
        inputs=inputs,
        observed=observed,
        objective=objective,
        parameter_bounds=parameter_bounds,
        warmup_period=warmup_period,
        niterations=niterations,
        nchains=nchains,
        start=start_positions,
        start_random=False,
        model_name=model_name,
        verbose=verbose,
        restart=True,  # PyDREAM restart flag
        **kwargs
    )
    
    # Combine results
    combined_samples = pd.concat([
        previous_result['all_samples'],
        new_result['all_samples']
    ], ignore_index=True)
    
    # Update iteration numbers
    prev_max_iter = previous_result['all_samples']['iteration'].max()
    combined_samples.loc[
        combined_samples.index >= len(previous_result['all_samples']),
        'iteration'
    ] += prev_max_iter + 1
    
    # Combine chain samples
    combined_chain_samples = [
        np.vstack([prev, new]) 
        for prev, new in zip(
            previous_result['sampled_params_by_chain'],
            new_result['sampled_params_by_chain']
        )
    ]
    
    combined_log_ps = [
        np.vstack([prev, new])
        for prev, new in zip(
            previous_result['log_ps_by_chain'],
            new_result['log_ps_by_chain']
        )
    ]
    
    # Update result with combined data
    combined_result = {
        'best_parameters': new_result['best_parameters'],
        'best_objective': new_result['best_objective'],
        'all_samples': combined_samples,
        'log_likelihoods': np.concatenate([
            previous_result['log_likelihoods'],
            new_result['log_likelihoods']
        ]),
        'sampled_params_by_chain': combined_chain_samples,
        'log_ps_by_chain': combined_log_ps,
        'convergence_diagnostics': new_result['convergence_diagnostics'],
        'runtime_seconds': (
            previous_result['runtime_seconds'] + new_result['runtime_seconds']
        ),
        'parameter_names': new_result['parameter_names'],
    }
    
    # Update best if previous was better
    if _should_maximize(objective):
        if previous_result['best_objective'] > new_result['best_objective']:
            combined_result['best_parameters'] = previous_result['best_parameters']
            combined_result['best_objective'] = previous_result['best_objective']
    else:
        if previous_result['best_objective'] < new_result['best_objective']:
            combined_result['best_parameters'] = previous_result['best_parameters']
            combined_result['best_objective'] = previous_result['best_objective']
    
    return combined_result


def check_pydream_convergence(
    result: Dict[str, Any],
    threshold: float = 1.2
) -> Tuple[bool, Dict[str, float]]:
    """
    Check convergence of PyDREAM results using Gelman-Rubin statistic.
    
    Args:
        result: Result dictionary from run_pydream
        threshold: Gelman-Rubin threshold (typically 1.1 or 1.2)
        
    Returns:
        Tuple of (converged: bool, gelman_rubin_values: dict)
    """
    if not PYDREAM_AVAILABLE:
        raise ImportError("PyDREAM is required for this method.")
    
    sampled_params = result['sampled_params_by_chain']
    param_names = result['parameter_names']
    
    if len(sampled_params) < 2:
        warnings.warn("Need at least 2 chains for Gelman-Rubin diagnostic")
        return False, {}
    
    gr_values = Gelman_Rubin(sampled_params)
    gr_dict = dict(zip(param_names, gr_values))
    
    converged = np.all(gr_values < threshold)
    
    return converged, gr_dict
