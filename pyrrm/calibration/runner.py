"""
Unified calibration runner for pyrrm models.

This module provides the CalibrationRunner class, which offers a unified
interface for calibrating models using various algorithms from SPOTPY,
PyDREAM, and scipy.optimize.

Supported DREAM implementations:
- SpotPy DREAM: Standard DREAM algorithm with built-in convergence checking
- PyDREAM (MT-DREAM(ZS)): Advanced DREAM with multi-try sampling and snooker updates
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
import warnings

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel

from pyrrm.calibration.objective_functions import (
    ObjectiveFunction, 
    NSE,
    is_new_interface,
    get_calibration_value,
)

# Check for optional dependencies
try:
    from pyrrm.calibration.pydream_adapter import PYDREAM_AVAILABLE
except ImportError:
    PYDREAM_AVAILABLE = False

try:
    from pyrrm.calibration.spotpy_adapter import SPOTPY_AVAILABLE
except ImportError:
    SPOTPY_AVAILABLE = False


@dataclass
class CalibrationResult:
    """
    Container for calibration results.
    
    Attributes:
        best_parameters: Dictionary of best parameter values
        best_objective: Best objective function value achieved
        all_samples: DataFrame with all evaluated parameter sets
        convergence_diagnostics: Method-specific convergence info
        runtime_seconds: Total runtime in seconds
        method: Calibration method used
        objective_name: Name of objective function
        success: Whether calibration converged successfully
        message: Additional information or warnings
    """
    best_parameters: Dict[str, float]
    best_objective: float
    all_samples: pd.DataFrame
    convergence_diagnostics: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    method: str = ""
    objective_name: str = ""
    success: bool = True
    message: str = ""
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "=" * 60,
            "CALIBRATION RESULTS",
            "=" * 60,
            f"Method: {self.method}",
            f"Objective: {self.objective_name}",
            f"Best {self.objective_name}: {self.best_objective:.6f}",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            f"Success: {self.success}",
            "",
            "Best Parameters:",
            "-" * 40,
        ]
        
        for name, value in self.best_parameters.items():
            lines.append(f"  {name}: {value:.6f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"CalibrationResult(method='{self.method}', "
            f"best_{self.objective_name}={self.best_objective:.4f}, "
            f"runtime={self.runtime_seconds:.1f}s)"
        )


class CalibrationRunner:
    """
    Unified interface for model calibration.
    
    Supports multiple calibration methods:
    - DREAM (Bayesian MCMC via SPOTPY or PyDREAM)
    - SCE-UA (Shuffled Complex Evolution via SPOTPY)
    - Differential Evolution (scipy)
    - Dual Annealing (scipy)
    - Basin-hopping (scipy)
    
    DREAM Implementation Comparison:
        SpotPy DREAM:
            - Standard DREAM algorithm
            - Built-in convergence checking
            - Database storage options (CSV, RAM, SQL)
            
        PyDREAM (MT-DREAM(ZS)):
            - Multi-try sampling for better mixing
            - Snooker updates for mode jumping
            - Parallel tempering support
            - Better for multi-modal posteriors
    
    Example:
        >>> from pyrrm.models import GR4J
        >>> from pyrrm.calibration import CalibrationRunner
        >>> from pyrrm.calibration.objective_functions import NSE
        >>> 
        >>> model = GR4J()
        >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
        >>> 
        >>> # Run DREAM with SpotPy (simpler, built-in convergence)
        >>> result = runner.run_dream(implementation='spotpy', n_iterations=10000)
        >>> 
        >>> # Run DREAM with PyDREAM (advanced features)
        >>> result = runner.run_dream(
        ...     implementation='pydream',
        ...     n_iterations=10000,
        ...     multitry=5,       # Multi-try sampling
        ...     snooker=0.1       # Snooker update probability
        ... )
        >>> 
        >>> # Or use scipy differential evolution
        >>> result = runner.run_scipy(method='differential_evolution')
    """
    
    def __init__(
        self,
        model: 'BaseRainfallRunoffModel',
        inputs: pd.DataFrame,
        observed: np.ndarray,
        objective: Optional[ObjectiveFunction] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365
    ):
        """
        Initialize calibration runner.
        
        Args:
            model: Rainfall-runoff model instance
            inputs: Input DataFrame with precipitation and PET
            observed: Observed flow values
            objective: Objective function (default: NSE)
            parameter_bounds: Parameter bounds {name: (min, max)}
                            If None, uses model's default bounds
            warmup_period: Warmup timesteps (excluded from objective)
        """
        self.model = model
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.objective = objective or NSE()
        self.warmup_period = warmup_period
        
        # Get parameter bounds
        if parameter_bounds is None:
            self._param_bounds = model.get_parameter_bounds()
        else:
            self._param_bounds = parameter_bounds
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input data."""
        if len(self.observed) != len(self.inputs):
            raise ValueError(
                f"Length mismatch: observed ({len(self.observed)}) vs "
                f"inputs ({len(self.inputs)})"
            )
        
        if self.warmup_period >= len(self.observed):
            raise ValueError(
                f"Warmup period ({self.warmup_period}) must be less than "
                f"data length ({len(self.observed)})"
            )
        
        if len(self._param_bounds) == 0:
            raise ValueError("No parameters to calibrate")
    
    def run_dream(
        self,
        implementation: str = 'spotpy',
        n_iterations: int = 10000,
        n_chains: int = 5,
        # SpotPy-specific options
        convergence_threshold: float = 1.01,
        dbname: str = 'dream_results',
        dbformat: str = 'csv',
        parallel: str = 'seq',
        # PyDREAM-specific options
        multitry: int = 5,
        snooker: float = 0.1,
        **kwargs
    ) -> CalibrationResult:
        """
        Run DREAM algorithm with choice of implementation.
        
        DREAM (DiffeRential Evolution Adaptive Metropolis) is a Bayesian MCMC 
        algorithm suitable for high-dimensional parameter estimation with 
        complex posterior distributions.
        
        Two implementations are available:
        - 'spotpy': Standard DREAM with built-in convergence checking and MPI support
        - 'pydream': MT-DREAM(ZS) with multi-try sampling and snooker updates
        
        Args:
            implementation: 'spotpy' for SpotPy DREAM or 'pydream' for PyDREAM
            n_iterations: Number of MCMC iterations
            n_chains: Number of parallel chains
            
            SpotPy-specific args:
                convergence_threshold: Gelman-Rubin convergence threshold
                dbname: Database name for results
                dbformat: Output format ('csv', 'sql')
                parallel: Parallelization mode:
                    - 'seq': Sequential (single core)
                    - 'mpi': MPI parallel (for HPC clusters)
                    - 'mpc': Multiprocessing ordered (local cores)
                    - 'umpc': Multiprocessing unordered (local cores, faster)
                
            PyDREAM-specific args:
                multitry: Number of proposal points per iteration (1=standard DREAM)
                snooker: Probability of snooker update (0 to disable)
                
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
        """
        implementation = implementation.lower()
        
        if implementation == 'spotpy':
            return self.run_spotpy_dream(
                n_iterations=n_iterations,
                n_chains=n_chains,
                convergence_threshold=convergence_threshold,
                dbname=dbname,
                dbformat=dbformat,
                parallel=parallel,
                **kwargs
            )
        elif implementation == 'pydream':
            return self.run_pydream(
                n_iterations=n_iterations,
                n_chains=n_chains,
                multitry=multitry,
                snooker=snooker,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown DREAM implementation: '{implementation}'. "
                f"Choose 'spotpy' or 'pydream'."
            )
    
    def run_spotpy_dream(
        self,
        n_iterations: int = 10000,
        n_chains: int = 5,
        convergence_threshold: float = 1.01,
        dbname: str = 'dream_results',
        dbformat: str = 'csv',
        parallel: str = 'seq',
        **kwargs
    ) -> CalibrationResult:
        """
        Run DREAM algorithm using SpotPy implementation.
        
        SpotPy DREAM provides:
        - Standard DREAM algorithm
        - Built-in Gelman-Rubin convergence checking
        - Multiple database storage formats
        - MPI parallel support for HPC clusters
        
        Args:
            n_iterations: Number of MCMC iterations
            n_chains: Number of parallel chains. For MPI, use n_chains = n_processes - 1
            convergence_threshold: Gelman-Rubin convergence threshold
            dbname: Database name for results
            dbformat: Output format ('csv', 'sql'). Note: 'ram' is converted to 'csv'
            parallel: Parallelization mode:
                - 'seq': Sequential (default) - runs on single core
                - 'mpi': MPI parallel - requires mpi4py, run with mpirun
                - 'mpc': Multiprocessing ordered - uses local CPU cores
                - 'umpc': Multiprocessing unordered - faster, uses local cores
            **kwargs: Additional DREAM parameters (nCr, delta, c, eps)
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
            
        MPI Parallel Example:
            To run calibration on a cluster with MPI:
            
            1. Create calibration script (calibrate.py):
               ```python
               runner = CalibrationRunner(model, inputs, observed)
               result = runner.run_spotpy_dream(
                   n_iterations=50000,
                   n_chains=20,
                   parallel='mpi',
                   dbname='mpi_dream_results'
               )
               if result:  # Only master process returns result
                   print(result.summary())
               ```
            
            2. Run with MPI (21 processes = 1 master + 20 workers):
               ```bash
               mpirun -n 21 python calibrate.py
               ```
            
            Speedup scales approximately linearly with number of chains.
        """
        if not SPOTPY_AVAILABLE:
            raise ImportError(
                "SpotPy is required for this calibration method. "
                "Install with: pip install spotpy"
            )
        
        from pyrrm.calibration.spotpy_adapter import SPOTPYModelSetup, run_dream
        
        setup = SPOTPYModelSetup(
            self.model,
            self.inputs,
            self.observed,
            self.objective,
            self._param_bounds,
            self.warmup_period
        )
        
        result = run_dream(
            setup,
            n_iterations=n_iterations,
            n_chains=n_chains,
            convergence_threshold=convergence_threshold,
            dbname=dbname,
            dbformat=dbformat,
            parallel=parallel,
            **kwargs
        )
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='SpotPy-DREAM',
            objective_name=self.objective.name,
            success=True
        )
    
    def run_pydream(
        self,
        n_iterations: int = 10000,
        n_chains: int = 5,
        multitry: int = 5,
        snooker: float = 0.1,
        nCR: int = 3,
        adapt_crossover: bool = True,
        DEpairs: int = 1,
        p_gamma_unity: float = 0.2,
        parallel: bool = False,
        mp_context: str = None,
        dbname: str = None,
        dbformat: str = 'csv',
        write_interval: int = 1,
        verbose: bool = True,
        **kwargs
    ) -> CalibrationResult:
        """
        Run PyDREAM (MT-DREAM(ZS)) calibration algorithm.
        
        PyDREAM implements Multi-Try DREAM with snooker updates, offering
        several advantages over standard DREAM:
        - Multi-try sampling for better mixing in multi-modal posteriors
        - Snooker updates for jumping between modes
        - Adaptive crossover probabilities
        - Built-in parallel processing
        - Real-time progress monitoring via CSV output
        
        Progress Monitoring:
            Set `dbname` to enable progress tracking. Each evaluation is
            written to `{dbname}.csv` in SpotPy-compatible format.
            
            Example:
                result = runner.run_pydream(
                    dbname='pydream_calib',  # Writes to pydream_calib.csv
                    ...
                )
                # Monitor with calibration_monitor notebook
        
        Parallelization:
            PyDREAM has TWO levels of built-in parallelization:
            
            1. **Chain-level (automatic)**: Each chain runs in a separate
               process via DreamPool. Set `n_chains` to match your CPU cores.
               
            2. **Multi-try (optional)**: When `parallel=True` and `multitry>1`,
               proposal evaluations are parallelized. Uses `multitry` workers.
            
            Example for 8-core machine:
                result = runner.run_pydream(
                    n_chains=8,       # 8 chains in parallel (automatic)
                    multitry=5,       # 5 proposals per iteration
                    parallel=True     # Evaluate proposals in parallel
                )
        
        Args:
            n_iterations: Number of iterations per chain
            n_chains: Number of parallel chains (min: 2*DEpairs+1).
                     Each chain runs in a separate process automatically.
            multitry: Number of proposal points per iteration (1=standard DREAM)
            snooker: Probability of snooker update (0 to disable)
            nCR: Number of crossover values
            adapt_crossover: Whether to adapt crossover probabilities
            DEpairs: Number of chain pairs for differential evolution
            p_gamma_unity: Probability of gamma=1 (full step)
            parallel: Enable parallel processing for multi-try evaluations.
                     When True and multitry>1, proposals are evaluated in parallel.
            mp_context: Multiprocessing context ('fork', 'spawn', 'forkserver').
                       None uses system default. 'spawn' is safest, 'fork' fastest.
            dbname: Database name for progress tracking. Writes to {dbname}.csv.
            dbformat: Output format ('csv' only, for SpotPy compatibility)
            write_interval: Write every N evaluations (1=all, 10=every 10th)
            verbose: Whether to print progress
            **kwargs: Additional PyDREAM parameters
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
        """
        if not PYDREAM_AVAILABLE:
            raise ImportError(
                "PyDREAM is required for this calibration method. "
                "Install from: https://github.com/LoLab-VU/PyDREAM "
                "using: pip install pydream"
            )
        
        from pyrrm.calibration.pydream_adapter import run_pydream
        
        result = run_pydream(
            model=self.model,
            inputs=self.inputs,
            observed=self.observed,
            objective=self.objective,
            parameter_bounds=self._param_bounds,
            warmup_period=self.warmup_period,
            niterations=n_iterations,
            nchains=n_chains,
            multitry=multitry,
            snooker=snooker,
            nCR=nCR,
            adapt_crossover=adapt_crossover,
            DEpairs=DEpairs,
            p_gamma_unity=p_gamma_unity,
            parallel=parallel,
            mp_context=mp_context,
            dbname=dbname,
            dbformat=dbformat,
            write_interval=write_interval,
            verbose=verbose,
            **kwargs
        )
        
        # Determine convergence status
        converged = result.get('convergence_diagnostics', {}).get('converged', None)
        success = converged if converged is not None else True
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='PyDREAM (MT-DREAM(ZS))',
            objective_name=self.objective.name,
            success=success
        )
    
    def run_sceua(
        self,
        n_iterations: int = 10000,
        ngs: int = 7,
        kstop: int = 3,
        pcento: float = 0.1,
        peps: float = 0.001,
        dbname: str = 'sceua_results',
        dbformat: str = 'csv',
        parallel: str = 'seq',
        **kwargs
    ) -> CalibrationResult:
        """
        Run SCE-UA (Shuffled Complex Evolution - University of Arizona).
        
        SCE-UA is a global optimization algorithm that combines simplex,
        random search, and evolution strategies. Highly effective for
        hydrological model calibration.
        
        Args:
            n_iterations: Maximum iterations
            ngs: Number of complexes. For MPI parallel, this determines
                 the degree of parallelism (each complex on separate process)
            kstop: Shuffle loops for convergence
            pcento: Percent change threshold
            peps: Parameter convergence threshold
            dbname: Database name
            dbformat: Output format ('csv', 'sql')
            parallel: Parallelization mode:
                - 'seq': Sequential (single core)
                - 'mpi': MPI parallel (complexes in parallel)
                - 'mpc': Multiprocessing ordered
                - 'umpc': Multiprocessing unordered
            **kwargs: Additional parameters
            
        Returns:
            CalibrationResult
            
        MPI Parallel Example:
            SCE-UA runs its complexes (ngs) in parallel.
            
            ```bash
            mpirun -n 8 python script.py  # 1 master + 7 workers
            ```
            
            In script, set ngs to match worker count:
            ```python
            result = runner.run_sceua(ngs=7, parallel='mpi')
            ```
        """
        from pyrrm.calibration.spotpy_adapter import SPOTPYModelSetup, run_sceua
        
        setup = SPOTPYModelSetup(
            self.model,
            self.inputs,
            self.observed,
            self.objective,
            self._param_bounds,
            self.warmup_period
        )
        
        result = run_sceua(
            setup,
            n_iterations=n_iterations,
            ngs=ngs,
            kstop=kstop,
            pcento=pcento,
            peps=peps,
            dbname=dbname,
            dbformat=dbformat,
            parallel=parallel,
            **kwargs
        )
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='SCE-UA',
            objective_name=self.objective.name,
            success=True
        )
    
    def run_scipy(
        self,
        method: str = 'differential_evolution',
        **kwargs
    ) -> CalibrationResult:
        """
        Run scipy.optimize calibration method.
        
        Args:
            method: Optimization method:
                - 'differential_evolution': Global DE algorithm (recommended)
                - 'dual_annealing': Simulated annealing variant
                - 'basinhopping': Global with local refinement
                - 'minimize': Local optimization (L-BFGS-B, etc.)
            **kwargs: Method-specific arguments
            
        Returns:
            CalibrationResult
        """
        from pyrrm.calibration.scipy_adapter import calibrate_scipy
        
        result = calibrate_scipy(
            self.model,
            self.inputs,
            self.observed,
            self.objective,
            self._param_bounds,
            self.warmup_period,
            method=method,
            **kwargs
        )
        
        return CalibrationResult(
            best_parameters=result.best_parameters,
            best_objective=result.best_objective,
            all_samples=result.all_samples,
            convergence_diagnostics=result.convergence_info,
            runtime_seconds=result.runtime_seconds,
            method=f'scipy.{method}',
            objective_name=self.objective.name,
            success=result.success,
            message=result.message
        )
    
    def run_differential_evolution(self, **kwargs) -> CalibrationResult:
        """Convenience method for differential evolution."""
        return self.run_scipy(method='differential_evolution', **kwargs)
    
    def run_dual_annealing(self, **kwargs) -> CalibrationResult:
        """Convenience method for dual annealing."""
        return self.run_scipy(method='dual_annealing', **kwargs)
    
    def run_basinhopping(self, **kwargs) -> CalibrationResult:
        """Convenience method for basin-hopping."""
        return self.run_scipy(method='basinhopping', **kwargs)
    
    def evaluate_parameters(
        self, 
        parameters: Dict[str, float]
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate model with specific parameters.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Tuple of (objective_value, simulated_flow)
        """
        self.model.reset()
        self.model.set_parameters(parameters)
        
        results = self.model.run(self.inputs)
        
        if 'flow' in results.columns:
            simulated = results['flow'].values
        elif 'runoff' in results.columns:
            simulated = results['runoff'].values
        else:
            simulated = results.iloc[:, 0].values
        
        # Calculate objective (excluding warmup)
        sim = simulated[self.warmup_period:]
        obs = self.observed[self.warmup_period:]
        
        # Handle both old and new objective function interfaces
        if is_new_interface(self.objective):
            # New interface: __call__(obs, sim)
            obj_value = self.objective(obs, sim)
        else:
            # Legacy interface: calculate(sim, obs)
            obj_value = self.objective.calculate(sim, obs)
        
        return obj_value, simulated
    
    def get_best_simulation(
        self, 
        result: CalibrationResult
    ) -> pd.DataFrame:
        """
        Run model with best parameters and return full outputs.
        
        Args:
            result: CalibrationResult from calibration
            
        Returns:
            DataFrame with model outputs using best parameters
        """
        self.model.reset()
        self.model.set_parameters(result.best_parameters)
        
        outputs = self.model.run(self.inputs)
        outputs['observed'] = self.observed
        
        return outputs
