"""
SPOTPY adapter for pyrrm model calibration.

This module provides an interface between pyrrm models and SPOTPY's
calibration algorithms, including DREAM, SCE-UA, MCMC, and others.

SPOTPY Reference:
    Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015).
    SPOTting model parameters using a ready-made python package.
    PLoS ONE, 10(12), e0145180.
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import warnings

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction

# Import spotpy only when needed
try:
    import spotpy
    from spotpy.parameter import Uniform, Normal, logNormal
    SPOTPY_AVAILABLE = True
except ImportError:
    SPOTPY_AVAILABLE = False


class SPOTPYModelSetup(object):
    """
    SPOTPY-compatible setup class for pyrrm models.
    
    This class provides the interface required by SPOTPY algorithms.
    Parameters are dynamically added as class attributes after instantiation.
    
    Example:
        >>> from pyrrm.models import Sacramento
        >>> from pyrrm.calibration.objective_functions import NSE
        >>> 
        >>> model = Sacramento()
        >>> setup = SPOTPYModelSetup(model, inputs, observed, NSE())
        >>> sampler = spotpy.algorithms.dream(setup, dbname='calib', dbformat='csv')
        >>> sampler.sample(10000)
    """
    
    def __init__(
        self,
        model: 'BaseRainfallRunoffModel',
        inputs: pd.DataFrame,
        observed: np.ndarray,
        objective: 'ObjectiveFunction',
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365,
        parameter_distributions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize SPOTPY setup.
        
        Args:
            model: Rainfall-runoff model instance
            inputs: Input DataFrame with precipitation and PET
            observed: Observed flow values
            objective: Objective function instance
            parameter_bounds: Dict of {param_name: (min, max)}
            warmup_period: Warmup timesteps (not included in objective)
            parameter_distributions: Dict of {param_name: 'uniform'/'normal'/'lognormal'}
        """
        if not SPOTPY_AVAILABLE:
            raise ImportError("SPOTPY is required. Install with: pip install spotpy")
        
        self.model = model
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.objective = objective
        self.warmup_period = warmup_period
        
        # Get parameter bounds from model if not provided
        if parameter_bounds is None:
            self._param_bounds = model.get_parameter_bounds()
        else:
            self._param_bounds = parameter_bounds
        
        self._param_dists = parameter_distributions or {}
        self._param_names = list(self._param_bounds.keys())
        
        # CRITICAL: Add parameters as CLASS attributes (not instance attributes)
        # SpotPy uses class introspection to find parameters
        self._add_parameters_to_class()
    
    def _add_parameters_to_class(self):
        """Add SPOTPY parameters as class attributes."""
        for name in self._param_names:
            low, high = self._param_bounds[name]
            dist_type = self._param_dists.get(name, 'uniform')
            
            if dist_type == 'uniform':
                param = Uniform(name, low=low, high=high)
            elif dist_type == 'normal':
                mean = (low + high) / 2
                std = (high - low) / 6
                param = Normal(name, mean=mean, stddev=std)
            elif dist_type == 'lognormal':
                log_low = np.log(max(low, 1e-10))
                log_high = np.log(high)
                mean = (log_low + log_high) / 2
                std = (log_high - log_low) / 6
                param = logNormal(name, mean=mean, sigma=std)
            else:
                param = Uniform(name, low=low, high=high)
            
            # Set as CLASS attribute, not instance attribute
            setattr(self.__class__, name, param)
    
    def simulation(self, x):
        """
        Run model simulation with parameter vector x.
        
        Args:
            x: Parameter values (named tuple or array)
            
        Returns:
            List of simulated flow values
        """
        try:
            # Build parameter dict - x can be named tuple or array
            if hasattr(x, '_fields'):
                # Named tuple from SpotPy
                params = {name: getattr(x, name) for name in self._param_names}
            else:
                # Regular array/list
                params = dict(zip(self._param_names, x))
            
            # Reset and set parameters
            self.model.reset()
            self.model.set_parameters(params)
            
            # Run model
            results = self.model.run(self.inputs)
            
            # Get flow output
            if 'flow' in results.columns:
                simulated = results['flow'].values
            elif 'runoff' in results.columns:
                simulated = results['runoff'].values
            else:
                simulated = results.iloc[:, 0].values
            
            # Apply warmup - return only post-warmup values
            simulated = simulated[self.warmup_period:]
            
            return list(simulated)
            
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            n_values = len(self.observed) - self.warmup_period
            return [np.nan] * max(1, n_values)
    
    def evaluation(self):
        """Return observed data (post-warmup)."""
        return list(self.observed[self.warmup_period:])
    
    def objectivefunction(self, simulation, evaluation, params=None):
        """
        Calculate objective function value.
        
        Args:
            simulation: Simulated values (post-warmup)
            evaluation: Observed values (post-warmup)
            params: Parameter values (optional, not used)
            
        Returns:
            Objective function value
        """
        try:
            sim = np.array(simulation)
            obs = np.array(evaluation)
            
            # Ensure arrays have same length
            min_len = min(len(sim), len(obs))
            if min_len == 0:
                return -np.inf
            
            sim = sim[:min_len]
            obs = obs[:min_len]
            
            # Remove NaN/inf values
            valid_mask = np.isfinite(sim) & np.isfinite(obs)
            if not np.any(valid_mask):
                return -np.inf
            
            sim_clean = sim[valid_mask]
            obs_clean = obs[valid_mask]
            
            # Calculate objective
            value = self.objective.for_spotpy(sim_clean, obs_clean)
            
            return value
            
        except Exception as e:
            warnings.warn(f"Objective function failed: {e}")
            return -np.inf


def run_dream(
    setup: SPOTPYModelSetup,
    n_iterations: int = 10000,
    n_chains: int = 5,
    convergence_threshold: float = 1.01,
    nCr: int = 3,
    delta: int = 3,
    c: float = 0.1,
    eps: float = 10e-6,
    dbname: str = 'dream_results',
    dbformat: str = 'csv',
    parallel: str = 'seq',
    save_sim: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run DREAM algorithm for Bayesian calibration.
    
    Args:
        setup: SPOTPY setup instance
        n_iterations: Number of MCMC iterations
        n_chains: Number of parallel chains. For MPI, this should match the
                 number of worker processes (total MPI processes - 1).
        convergence_threshold: Gelman-Rubin threshold for convergence
        nCr: Number of crossover values
        delta: Number of pairs for generating jump
        c: Scaling factor for proposal distribution
        eps: Randomization for initial positions
        dbname: Database name for results
        dbformat: Output format ('csv', 'ram', 'sql')
        parallel: Parallelization mode:
            - 'seq': Sequential (default) - single core
            - 'mpi': MPI parallel - requires mpi4py and mpirun
            - 'mpc': Multiprocessing (ordered) - uses pathos
            - 'umpc': Multiprocessing (unordered) - uses pathos, faster but unordered
        save_sim: Whether to save simulations
        **kwargs: Additional arguments for SPOTPY DREAM
        
    Returns:
        Dictionary with:
            - best_parameters: Best parameter set
            - best_objective: Best objective value
            - all_samples: Full chain history
            - convergence_diagnostics: Gelman-Rubin statistics
    
    MPI Parallel Usage:
        To run with MPI parallelization:
        
        1. Install mpi4py: pip install mpi4py
        
        2. Create a script (e.g., run_calibration.py):
           ```python
           from pyrrm.calibration import CalibrationRunner
           runner = CalibrationRunner(model, inputs, observed)
           result = runner.run_spotpy_dream(
               n_iterations=50000,
               n_chains=10,
               parallel='mpi'
           )
           ```
        
        3. Run with mpirun:
           ```bash
           mpirun -n 11 python run_calibration.py
           ```
           Note: Use n_chains + 1 processes (1 master + n_chains workers)
        
        The speedup is approximately linear with the number of chains/cores.
    """
    if not SPOTPY_AVAILABLE:
        raise ImportError("SPOTPY is required. Install with: pip install spotpy")
    
    import time
    import os
    start_time = time.time()
    
    # Force CSV format for reliability - SpotPy's RAM format has issues in some versions
    if dbformat == 'ram':
        warnings.warn("Using 'csv' format instead of 'ram' for reliability. Results will be saved to disk.")
        dbformat = 'csv'
    
    # Test the setup before running full calibration
    print("Testing SpotPy setup...")
    try:
        # Test parameter generation
        test_params = spotpy.parameter.get_parameters_array(setup)
        print(f"  Parameters found: {list(test_params['name'])}")
        print(f"  Parameter count: {len(test_params['name'])}")
        
        # Test simulation with random parameters
        test_random = test_params['random']
        print(f"  Testing simulation with random parameters...")
        test_sim = setup.simulation(test_random)
        print(f"  Simulation returned {len(test_sim)} values")
        
        # Test evaluation
        test_eval = setup.evaluation()
        print(f"  Evaluation returned {len(test_eval)} values")
        
        # Test objective function
        test_obj = setup.objectivefunction(test_sim, test_eval)
        print(f"  Objective function returned: {test_obj}")
        
        if test_obj is None or (isinstance(test_obj, float) and (np.isnan(test_obj) or np.isinf(test_obj))):
            warnings.warn(f"Objective function returned invalid value: {test_obj}")
        
        print("  Setup test passed!")
    except Exception as e:
        print(f"  Setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"SpotPy setup validation failed: {e}")
    
    # DREAM requires nChains >= 2*delta + 1
    min_chains = 2 * delta + 1
    if n_chains < min_chains:
        print(f"  ⚠️ Warning: DREAM requires at least {min_chains} chains for delta={delta}")
        print(f"     Adjusting delta from {delta} to {(n_chains - 1) // 2}")
        delta = (n_chains - 1) // 2
        if delta < 1:
            delta = 1
            n_chains = 3
            print(f"     Using minimum: delta={delta}, n_chains={n_chains}")
    
    # Create sampler
    print(f"\nCreating DREAM sampler with dbname='{dbname}', dbformat='{dbformat}'...")
    sampler = spotpy.algorithms.dream(
        setup,
        dbname=dbname,
        dbformat=dbformat,
        parallel=parallel,
        save_sim=save_sim,
        db_precision=np.float64  # Higher precision for better results
    )
    
    # Run DREAM - returns r_hat convergence diagnostics
    print(f"Starting DREAM sampling with {n_iterations} iterations, {n_chains} chains, delta={delta}...")
    r_hat = sampler.sample(
        repetitions=n_iterations,
        nChains=n_chains,
        convergence_limit=convergence_threshold,
        nCr=nCr,
        delta=delta,
        c=c,
        eps=eps,
        **kwargs
    )
    print("DREAM sampling completed.")
    
    # Load results using SpotPy's analyser (the recommended approach)
    # This works reliably across SpotPy versions
    results_df = None
    
    # Method 1: Use SpotPy's built-in CSV loader (most reliable)
    if dbformat == 'csv':
        csv_file = f"{dbname}.csv"
        if os.path.exists(csv_file):
            try:
                # Use spotpy.analyser for proper loading
                results_structured = spotpy.analyser.load_csv_results(dbname)
                results_df = pd.DataFrame(results_structured)
            except Exception:
                # Fallback to direct pandas read
                results_df = pd.read_csv(csv_file)
    
    # Method 2: Try sampler.getdata() as fallback
    if results_df is None:
        try:
            if hasattr(sampler, 'datawriter') and sampler.datawriter is not None:
                results_structured = sampler.getdata()
                results_df = pd.DataFrame(results_structured)
        except (AttributeError, TypeError):
            pass
    
    if results_df is None or len(results_df) == 0:
        raise RuntimeError(
            f"Could not retrieve DREAM results. "
            f"Check if the CSV file '{dbname}.csv' was created and contains data."
        )
    
    # Find likelihood column (may be 'like1' or 'like')
    like_col = None
    for col in ['like1', 'like', 'likelihood', 'objectivefunction']:
        if col in results_df.columns:
            like_col = col
            break
    
    if like_col is None:
        raise RuntimeError(f"Could not find likelihood column in results. Columns: {list(results_df.columns)}")
    
    # Find best parameters
    if setup.objective.maximize:
        best_idx = results_df[like_col].idxmax()
    else:
        best_idx = results_df[like_col].idxmin()
    
    best_params = {}
    for name in setup._param_names:
        # Try different column naming conventions
        for col_pattern in [f'par{name}', name, f'par_{name}']:
            if col_pattern in results_df.columns:
                best_params[name] = results_df[col_pattern].iloc[best_idx]
                break
    
    # Build samples DataFrame
    samples_dict = {
        'iteration': np.arange(len(results_df)),
        'likelihood': results_df[like_col].values
    }
    for name in setup._param_names:
        for col_pattern in [f'par{name}', name, f'par_{name}']:
            if col_pattern in results_df.columns:
                samples_dict[name] = results_df[col_pattern].values
                break
    
    all_samples = pd.DataFrame(samples_dict)
    
    runtime = time.time() - start_time
    
    # Store Gelman-Rubin diagnostics if available
    convergence_diagnostics = {}
    if r_hat is not None:
        convergence_diagnostics['r_hat'] = r_hat
    
    return {
        'best_parameters': best_params,
        'best_objective': results_df[like_col].iloc[best_idx],
        'all_samples': all_samples,
        'convergence_diagnostics': convergence_diagnostics,
        'runtime_seconds': runtime,
        'raw_results': results_df
    }


class SCEUAModelSetup(SPOTPYModelSetup):
    """
    SPOTPY setup class specifically for SCE-UA algorithm.
    
    SCE-UA always MINIMIZES, so this wrapper negates the objective function
    for maximization objectives (like NSE, KGE) to convert them to minimization.
    """
    
    def objectivefunction(self, simulation, evaluation, params=None):
        """
        Calculate objective function value, negated for maximization objectives.
        
        SCE-UA minimizes, so we negate objectives that should be maximized.
        """
        try:
            sim = np.array(simulation)
            obs = np.array(evaluation)
            
            min_len = min(len(sim), len(obs))
            if min_len == 0:
                return np.inf  # Return large value for minimization
            
            sim = sim[:min_len]
            obs = obs[:min_len]
            
            valid_mask = np.isfinite(sim) & np.isfinite(obs)
            if not np.any(valid_mask):
                return np.inf
            
            sim_clean = sim[valid_mask]
            obs_clean = obs[valid_mask]
            
            # Get the raw objective value
            value = self.objective.calculate(sim_clean, obs_clean)
            
            # For SCE-UA (minimization):
            # - If objective should be maximized (NSE, KGE): negate to minimize -NSE
            # - If objective should be minimized (RMSE): keep as-is
            if self.objective.maximize:
                return -value  # Negate so minimizing -NSE = maximizing NSE
            else:
                return value   # Already a minimization objective
            
        except Exception as e:
            warnings.warn(f"Objective function failed: {e}")
            return np.inf


def run_sceua(
    setup: SPOTPYModelSetup,
    n_iterations: int = 10000,
    ngs: int = 7,
    kstop: int = 3,
    pcento: float = 0.1,
    peps: float = 0.001,
    dbname: str = 'sceua_results',
    dbformat: str = 'csv',
    parallel: str = 'seq',
    **kwargs
) -> Dict[str, Any]:
    """
    Run SCE-UA (Shuffled Complex Evolution) algorithm.
    
    SCE-UA is a global optimization algorithm combining simplex, random search,
    and competitive evolution. It's highly effective for hydrological calibration.
    
    Note on Optimization Direction:
        SCE-UA in SpotPy always MINIMIZES. This function automatically handles
        the conversion for maximization objectives:
        - NSE, KGE (maximize=True): Returns -NSE so minimizing -NSE = maximizing NSE
        - RMSE, MAE (maximize=False): Returns value as-is for direct minimization
    
    Args:
        setup: SPOTPY setup instance (will be wrapped for SCE-UA compatibility)
        n_iterations: Maximum number of iterations
        ngs: Number of complexes. For MPI parallel, this determines parallelism -
             each complex runs on a separate process.
        kstop: Number of shuffle loops for convergence check
        pcento: Percent change in objective for convergence
        peps: Threshold for parameter value convergence
        dbname: Database name for results
        dbformat: Output format ('csv', 'sql')
        parallel: Parallelization mode:
            - 'seq': Sequential (single core)
            - 'mpi': MPI parallel (complexes run in parallel)
            - 'mpc': Multiprocessing ordered
            - 'umpc': Multiprocessing unordered
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with calibration results
        
    MPI Parallel Usage:
        SCE-UA parallelizes by running its complexes (ngs) in parallel.
        Use ngs equal to the number of worker processes for best efficiency.
        
        Example:
            mpirun -n 8 python script.py  # 1 master + 7 workers
            # In script: run_sceua(..., ngs=7, parallel='mpi')
    """
    if not SPOTPY_AVAILABLE:
        raise ImportError("SPOTPY is required. Install with: pip install spotpy")
    
    import time
    start_time = time.time()
    
    # Create SCE-UA specific setup that handles objective negation
    sceua_setup = SCEUAModelSetup(
        model=setup.model,
        inputs=setup.inputs,
        observed=setup.observed,
        objective=setup.objective,
        parameter_bounds=setup._param_bounds,
        warmup_period=setup.warmup_period
    )
    
    # Log what we're doing
    if setup.objective.maximize:
        print(f"  Objective: {setup.objective.name} (maximize)")
        print(f"  SCE-UA minimizes, so returning -{setup.objective.name} to maximize it")
    else:
        print(f"  Objective: {setup.objective.name} (minimize)")
        print(f"  SCE-UA minimizes directly")
    
    sampler = spotpy.algorithms.sceua(
        sceua_setup,
        dbname=dbname,
        dbformat=dbformat,
        parallel=parallel
    )
    
    sampler.sample(
        repetitions=n_iterations,
        ngs=ngs,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        **kwargs
    )
    
    results = sampler.getdata()
    
    # Find index of minimum objective value (SCE-UA minimizes)
    # Use numpy directly for reliable indexing
    best_idx = int(np.argmin(results['like1']))
    
    best_params = {}
    for name in setup._param_names:
        col_name = f'par{name}'
        if col_name in results.dtype.names:
            best_params[name] = float(results[col_name][best_idx])
    
    # Get the stored objective value at best index
    stored_best = float(results['like1'][best_idx])
    
    # Convert stored objective values back to original scale
    # For maximization objectives, we stored -value, so negate back
    stored_values = results['like1']
    if setup.objective.maximize:
        original_values = -stored_values  # Convert back: -(-NSE) = NSE
        best_objective = -stored_best
    else:
        original_values = stored_values
        best_objective = stored_best
    
    # Build samples DataFrame with original (non-negated) values
    samples_dict = {
        'iteration': np.arange(len(results)),
        'likelihood': original_values
    }
    for name in setup._param_names:
        col_name = f'par{name}'
        if col_name in results.dtype.names:
            samples_dict[name] = results[col_name]
    
    all_samples = pd.DataFrame(samples_dict)
    
    runtime = time.time() - start_time
    
    # Debug: show what SpotPy stored vs what we report
    print(f"\n  SpotPy stored (minimized): {stored_best:.6f}")
    if setup.objective.maximize:
        print(f"  Converted back to {setup.objective.name}: {best_objective:.6f}")
    else:
        print(f"  Best {setup.objective.name}: {best_objective:.6f}")
    
    return {
        'best_parameters': best_params,
        'best_objective': best_objective,
        'all_samples': all_samples,
        'convergence_diagnostics': {},
        'runtime_seconds': runtime,
        'raw_results': results
    }


def run_mcmc(
    setup: SPOTPYModelSetup,
    n_iterations: int = 10000,
    dbname: str = 'mcmc_results',
    dbformat: str = 'csv',
    **kwargs
) -> Dict[str, Any]:
    """
    Run standard MCMC algorithm.
    
    Args:
        setup: SPOTPY setup instance
        n_iterations: Number of MCMC iterations
        dbname: Database name for results
        dbformat: Output format
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with calibration results
    """
    if not SPOTPY_AVAILABLE:
        raise ImportError("SPOTPY is required. Install with: pip install spotpy")
    
    import time
    start_time = time.time()
    
    sampler = spotpy.algorithms.mcmc(
        setup,
        dbname=dbname,
        dbformat=dbformat
    )
    
    sampler.sample(repetitions=n_iterations, **kwargs)
    
    results = sampler.getdata()
    
    # Find best
    if setup.objective.maximize:
        best_idx = np.argmax(results['like1'])
    else:
        best_idx = np.argmin(results['like1'])
    
    best_params = {}
    for name in setup._param_names:
        col_name = f'par{name}'
        if col_name in results.dtype.names:
            best_params[name] = results[col_name][best_idx]
    
    samples_dict = {
        'iteration': np.arange(len(results)),
        'likelihood': results['like1']
    }
    for name in setup._param_names:
        col_name = f'par{name}'
        if col_name in results.dtype.names:
            samples_dict[name] = results[col_name]
    
    all_samples = pd.DataFrame(samples_dict)
    
    runtime = time.time() - start_time
    
    return {
        'best_parameters': best_params,
        'best_objective': results['like1'][best_idx],
        'all_samples': all_samples,
        'convergence_diagnostics': {},
        'runtime_seconds': runtime,
        'raw_results': results
    }
