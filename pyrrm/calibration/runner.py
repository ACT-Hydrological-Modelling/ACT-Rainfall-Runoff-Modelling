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
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Literal, Union
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import warnings

# Check for optional parquet support
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

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
        _raw_result: Internal storage for implementation-specific data
                    (used for continuation/checkpointing)
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
    _raw_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
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
    
    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Args:
            include_samples: Whether to include all_samples DataFrame
                           (can be large, excluded by default)
        
        Returns:
            Dictionary representation of the calibration result
        """
        data = {
            'best_parameters': self.best_parameters,
            'best_objective': float(self.best_objective),
            'convergence_diagnostics': self._serialize_diagnostics(),
            'runtime_seconds': float(self.runtime_seconds),
            'method': self.method,
            'objective_name': self.objective_name,
            'success': self.success,
            'message': self.message,
            'saved_at': datetime.now().isoformat(),
            'n_samples': len(self.all_samples) if self.all_samples is not None else 0,
        }
        
        if include_samples and self.all_samples is not None:
            data['all_samples'] = self.all_samples.to_dict(orient='records')
        
        return data
    
    def _serialize_diagnostics(self) -> Dict[str, Any]:
        """Convert convergence_diagnostics to JSON-serializable format."""
        serialized = {}
        for key, value in self.convergence_diagnostics.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                serialized[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) 
                    else float(v) if isinstance(v, (np.floating, np.integer))
                    else v
                    for k, v in value.items()
                }
            else:
                serialized[key] = value
        return serialized
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """
        Create CalibrationResult from dictionary.
        
        Args:
            data: Dictionary with calibration result data
            
        Returns:
            CalibrationResult instance
        """
        # Handle all_samples
        if 'all_samples' in data and data['all_samples']:
            all_samples = pd.DataFrame(data['all_samples'])
        else:
            all_samples = pd.DataFrame()
        
        return cls(
            best_parameters=data['best_parameters'],
            best_objective=data['best_objective'],
            all_samples=all_samples,
            convergence_diagnostics=data.get('convergence_diagnostics', {}),
            runtime_seconds=data.get('runtime_seconds', 0.0),
            method=data.get('method', ''),
            objective_name=data.get('objective_name', ''),
            success=data.get('success', True),
            message=data.get('message', ''),
        )
    
    def save(
        self, 
        path: str, 
        include_samples: bool = True,
        include_chains: bool = True
    ) -> List[str]:
        """
        Save calibration result to disk.
        
        Creates multiple files:
        - {path}_meta.json: Metadata, best params, diagnostics
        - {path}_samples.parquet (or .csv): All samples (if include_samples=True)
        - {path}_chains.npz: Chain data for PyDREAM continuation (if available)
        
        Args:
            path: Base path for output files (without extension)
            include_samples: Whether to save all_samples DataFrame
            include_chains: Whether to save chain data for continuation
            
        Returns:
            List of created file paths
            
        Example:
            >>> result.save('calibrations/catchment_410734')
            ['calibrations/catchment_410734_meta.json',
             'calibrations/catchment_410734_samples.parquet']
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # Save metadata as JSON
        meta_path = str(path) + '_meta.json'
        meta_data = self.to_dict(include_samples=False)
        
        # Add chain info if available
        if self._raw_result is not None:
            meta_data['has_chain_data'] = (
                'sampled_params_by_chain' in self._raw_result
            )
            if 'parameter_names' in self._raw_result:
                meta_data['parameter_names'] = self._raw_result['parameter_names']
        else:
            meta_data['has_chain_data'] = False
        
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        created_files.append(meta_path)
        
        # Save samples
        if include_samples and self.all_samples is not None and len(self.all_samples) > 0:
            if PARQUET_AVAILABLE:
                samples_path = str(path) + '_samples.parquet'
                self.all_samples.to_parquet(samples_path, index=False)
            else:
                samples_path = str(path) + '_samples.csv'
                self.all_samples.to_csv(samples_path, index=False)
            created_files.append(samples_path)
        
        # Save chain data for PyDREAM continuation
        if include_chains and self._raw_result is not None:
            if 'sampled_params_by_chain' in self._raw_result:
                chains_path = str(path) + '_chains.npz'
                chain_data = {
                    'sampled_params_by_chain': np.array(
                        self._raw_result['sampled_params_by_chain'], 
                        dtype=object
                    ),
                    'log_ps_by_chain': np.array(
                        self._raw_result['log_ps_by_chain'],
                        dtype=object
                    ),
                }
                if 'parameter_names' in self._raw_result:
                    chain_data['parameter_names'] = np.array(
                        self._raw_result['parameter_names']
                    )
                np.savez_compressed(chains_path, **chain_data)
                created_files.append(chains_path)
        
        return created_files
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationResult':
        """
        Load calibration result from disk.
        
        Args:
            path: Base path used when saving (without extension),
                  or path to _meta.json file
                  
        Returns:
            CalibrationResult instance
            
        Example:
            >>> result = CalibrationResult.load('calibrations/catchment_410734')
            >>> print(result.best_parameters)
        """
        # Handle both base path and full meta path
        path_str = str(path)
        if path_str.endswith('_meta.json'):
            meta_path = path_str
            base_path = path_str[:-10]  # Remove '_meta.json'
        else:
            meta_path = path_str + '_meta.json'
            base_path = path_str
        
        # Load metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Calibration result not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        
        # Try to load samples
        all_samples = pd.DataFrame()
        samples_parquet = base_path + '_samples.parquet'
        samples_csv = base_path + '_samples.csv'
        
        if os.path.exists(samples_parquet):
            if PARQUET_AVAILABLE:
                all_samples = pd.read_parquet(samples_parquet)
            else:
                warnings.warn(
                    f"Parquet file found but pyarrow not installed. "
                    f"Install with: pip install pyarrow"
                )
        elif os.path.exists(samples_csv):
            all_samples = pd.read_csv(samples_csv)
        
        # Load chain data if available
        raw_result = None
        chains_path = base_path + '_chains.npz'
        if os.path.exists(chains_path):
            chain_data = np.load(chains_path, allow_pickle=True)
            raw_result = {
                'sampled_params_by_chain': list(chain_data['sampled_params_by_chain']),
                'log_ps_by_chain': list(chain_data['log_ps_by_chain']),
            }
            if 'parameter_names' in chain_data:
                raw_result['parameter_names'] = list(chain_data['parameter_names'])
        
        return cls(
            best_parameters=meta_data['best_parameters'],
            best_objective=meta_data['best_objective'],
            all_samples=all_samples,
            convergence_diagnostics=meta_data.get('convergence_diagnostics', {}),
            runtime_seconds=meta_data.get('runtime_seconds', 0.0),
            method=meta_data.get('method', ''),
            objective_name=meta_data.get('objective_name', ''),
            success=meta_data.get('success', True),
            message=meta_data.get('message', ''),
            _raw_result=raw_result,
        )
    
    def can_resume(self) -> bool:
        """
        Check if this result can be used to resume calibration.
        
        Returns True if chain data is available (PyDREAM results).
        """
        if self._raw_result is None:
            return False
        return 'sampled_params_by_chain' in self._raw_result


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
        # Checkpoint options
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        resume_from: Optional[str] = None,
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
            
            Checkpoint args:
                checkpoint_dir: Directory for automatic checkpoints (None to disable)
                checkpoint_interval: Save checkpoint every N iterations
                resume_from: Path to checkpoint or CalibrationResult to resume from
                
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
            
        Example with checkpointing:
            >>> result = runner.run_dream(
            ...     implementation='pydream',
            ...     n_iterations=100000,
            ...     checkpoint_dir='./checkpoints',
            ...     checkpoint_interval=5000
            ... )
            
        Example resuming from checkpoint:
            >>> result = runner.run_dream(
            ...     implementation='pydream',
            ...     n_iterations=50000,  # Additional iterations
            ...     resume_from='./checkpoints'  # Load latest from dir
            ... )
        """
        implementation = implementation.lower()
        
        # Handle resume_from
        if resume_from is not None:
            return self._resume_dream(
                resume_from=resume_from,
                implementation=implementation,
                n_iterations=n_iterations,
                n_chains=n_chains,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                **kwargs
            )
        
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
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown DREAM implementation: '{implementation}'. "
                f"Choose 'spotpy' or 'pydream'."
            )
    
    def _resume_dream(
        self,
        resume_from: str,
        implementation: str,
        n_iterations: int,
        n_chains: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        **kwargs
    ) -> CalibrationResult:
        """
        Resume DREAM calibration from checkpoint or saved result.
        
        Args:
            resume_from: Path to checkpoint directory, checkpoint file, or saved result
            implementation: DREAM implementation to use
            n_iterations: Additional iterations to run
            n_chains: Number of chains
            checkpoint_dir: Directory for new checkpoints
            checkpoint_interval: Checkpoint interval
            **kwargs: Additional parameters
            
        Returns:
            CalibrationResult combining previous and new results
        """
        from pyrrm.calibration.checkpoint import CheckpointManager
        
        # Determine what we're resuming from
        resume_path = Path(resume_from)
        
        if resume_path.is_dir():
            # Load from checkpoint directory
            manager = CheckpointManager(str(resume_path))
            previous_result = manager.load_latest_checkpoint()
            if previous_result is None:
                raise ValueError(f"No checkpoints found in {resume_from}")
        elif resume_path.suffix == '.json' or (resume_path.parent / (resume_path.name + '.json')).exists():
            # Load from specific checkpoint or CalibrationResult
            try:
                # Try loading as CalibrationResult
                prev_calib_result = CalibrationResult.load(str(resume_path))
                previous_result = prev_calib_result._raw_result or {}
                previous_result.update({
                    'best_parameters': prev_calib_result.best_parameters,
                    'best_objective': prev_calib_result.best_objective,
                    'all_samples': prev_calib_result.all_samples,
                    'convergence_diagnostics': prev_calib_result.convergence_diagnostics,
                    'runtime_seconds': prev_calib_result.runtime_seconds,
                })
            except FileNotFoundError:
                # Try loading as checkpoint
                manager = CheckpointManager(str(resume_path.parent))
                previous_result = manager.load_checkpoint(
                    str(resume_path).replace('.json', '')
                )
                if previous_result is None:
                    raise ValueError(f"Could not load checkpoint: {resume_from}")
        else:
            raise ValueError(
                f"Cannot resume from {resume_from}. "
                f"Provide a checkpoint directory, checkpoint file, or saved CalibrationResult."
            )
        
        # Continue based on implementation
        if implementation == 'pydream':
            if 'sampled_params_by_chain' not in previous_result:
                raise ValueError(
                    "Cannot resume PyDREAM: previous result does not contain chain data. "
                    "Use SpotPy DREAM or start fresh."
                )
            
            from pyrrm.calibration.pydream_adapter import continue_pydream
            
            result = continue_pydream(
                model=self.model,
                inputs=self.inputs,
                observed=self.observed,
                objective=self.objective,
                previous_result=previous_result,
                parameter_bounds=self._param_bounds,
                warmup_period=self.warmup_period,
                niterations=n_iterations,
                **kwargs
            )
            
            # Store raw result for potential continuation
            raw_result = {
                'sampled_params_by_chain': result.get('sampled_params_by_chain'),
                'log_ps_by_chain': result.get('log_ps_by_chain'),
                'parameter_names': result.get('parameter_names'),
            }
            
            converged = result.get('convergence_diagnostics', {}).get('converged', None)
            success = converged if converged is not None else True
            
            return CalibrationResult(
                best_parameters=result['best_parameters'],
                best_objective=result['best_objective'],
                all_samples=result['all_samples'],
                convergence_diagnostics=result['convergence_diagnostics'],
                runtime_seconds=result['runtime_seconds'],
                method='PyDREAM (MT-DREAM(ZS)) [resumed]',
                objective_name=self.objective.name,
                success=success,
                _raw_result=raw_result,
            )
        
        else:  # SpotPy
            from pyrrm.calibration.spotpy_adapter import continue_spotpy_dream
            
            result = continue_spotpy_dream(
                model=self.model,
                inputs=self.inputs,
                observed=self.observed,
                objective=self.objective,
                previous_result=previous_result,
                parameter_bounds=self._param_bounds,
                warmup_period=self.warmup_period,
                n_iterations=n_iterations,
                n_chains=n_chains,
                **kwargs
            )
            
            return CalibrationResult(
                best_parameters=result['best_parameters'],
                best_objective=result['best_objective'],
                all_samples=result['all_samples'],
                convergence_diagnostics=result['convergence_diagnostics'],
                runtime_seconds=result['runtime_seconds'],
                method='SpotPy-DREAM [resumed]',
                objective_name=self.objective.name,
                success=True,
                _raw_result=result,
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
        
        # Store raw result for potential continuation
        raw_result = {
            'dbname': dbname,
            'dbformat': dbformat,
            'n_chains': n_chains,
            'raw_results': result.get('raw_results'),
        }
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='SpotPy-DREAM',
            objective_name=self.objective.name,
            success=True,
            _raw_result=raw_result,
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
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
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
            checkpoint_dir: Directory for automatic checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N iterations
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
        
        # Save checkpoint if checkpoint_dir specified
        if checkpoint_dir is not None:
            from pyrrm.calibration.checkpoint import CheckpointManager
            manager = CheckpointManager(
                checkpoint_dir, 
                checkpoint_interval=checkpoint_interval
            )
            manager.save_checkpoint(
                result, 
                iteration=n_iterations, 
                method='PyDREAM (MT-DREAM(ZS))'
            )
            if verbose:
                print(f"Checkpoint saved to {checkpoint_dir}")
        
        # Determine convergence status
        converged = result.get('convergence_diagnostics', {}).get('converged', None)
        success = converged if converged is not None else True
        
        # Store raw result for potential continuation
        raw_result = {
            'sampled_params_by_chain': result.get('sampled_params_by_chain'),
            'log_ps_by_chain': result.get('log_ps_by_chain'),
            'parameter_names': result.get('parameter_names'),
        }
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='PyDREAM (MT-DREAM(ZS))',
            objective_name=self.objective.name,
            success=success,
            _raw_result=raw_result,
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
    
    def run_sceua_direct(
        self,
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
    ) -> CalibrationResult:
        """
        Run SCE-UA using the direct (vendored) implementation.
        
        This is an alternative to run_sceua() which uses SpotPy. Key advantages:
        - No SpotPy dependency required
        - More configuration options (PCA recovery, convergence criteria)
        - Parallel evaluation via ThreadPoolExecutor
        - Initial parameter sets can be provided via x0
        - All evaluations tracked automatically
        
        The SCE-UA (Shuffled Complex Evolution - University of Arizona) algorithm
        is a global optimization method developed specifically for calibrating
        hydrological models.
        
        Args:
            n_complexes: Number of complexes. If None, automatically determined
                based on dimensionality: min(max(2, log2(n_params) + 5), 15).
            n_points_complex: Points per complex. If None, calculated as
                2 * n_complexes + 1.
            alpha: Reflection coefficient for simplex evolution (default 1.0).
            beta: Contraction coefficient for simplex evolution (default 0.5).
            max_evals: Maximum number of function evaluations (default 50000).
            max_iter: Maximum number of iterations (default 1000).
            max_tolerant_iter: Stop if no improvement for this many iterations
                (default 30).
            tolerance: Minimum improvement in objective to count as progress
                (default 1e-6).
            x_tolerance: Stop if parameter ranges shrink below this threshold
                (default 1e-8).
            seed: Random seed for reproducibility.
            pca_freq: Frequency of PCA recovery in iterations (default 1).
            pca_tol: Tolerance for detecting lost dimensions (default 1e-3).
            x0: Initial parameter sets. Can be:
                - np.ndarray of shape (n_sets, n_params)
                - List of tuples [(p1, p2, ...), ...]
                - Dict {name: value} for a single starting point
            max_workers: Number of parallel workers (default 1).
                Set > 1 to enable parallel evaluation via ThreadPoolExecutor.
            callback: Optional function called after each iteration with dict
                containing {'iteration', 'nfev', 'best_fun', 'best_x'}.
            verbose: If True, print progress every 10 iterations (default False).
            progress_bar: If True and tqdm is available, show a progress bar
                (default True).
            
        Returns:
            CalibrationResult with best parameters and optimization history.
            
        Example:
            >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
            >>> result = runner.run_sceua_direct(
            ...     max_evals=20000,
            ...     n_complexes=5,
            ...     seed=42,
            ...     max_workers=4  # Parallel evaluation
            ... )
            >>> print(f"Best NSE: {result.best_objective:.4f}")
            
        Notes:
            This implementation includes enhancements from the literature:
            - PCA recovery for lost dimensions (Chu et al., 2010)
            - Adaptive smoothing parameter (Muttil & Jayawardena, 2008)
            
        See Also:
            run_sceua: SpotPy-based SCE-UA implementation
        """
        from pyrrm.calibration.sceua_adapter import run_sceua_direct
        
        result = run_sceua_direct(
            model=self.model,
            inputs=self.inputs,
            observed=self.observed,
            objective=self.objective,
            parameter_bounds=self._param_bounds,
            warmup_period=self.warmup_period,
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
            x0=x0,
            max_workers=max_workers,
            callback=callback,
            verbose=verbose,
            progress_bar=progress_bar,
        )
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method='SCE-UA (direct)',
            objective_name=self.objective.name,
            success=result['convergence_diagnostics']['success'],
            message=result['convergence_diagnostics']['message'],
        )
    
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
    
    @classmethod
    def resume(
        cls,
        checkpoint_path: str,
        model: 'BaseRainfallRunoffModel',
        inputs: pd.DataFrame,
        observed: np.ndarray,
        objective: Optional['ObjectiveFunction'] = None,
        additional_iterations: int = 10000,
        implementation: str = 'pydream',
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        **kwargs
    ) -> CalibrationResult:
        """
        Resume calibration from a saved checkpoint or CalibrationResult.
        
        This is a high-level convenience method that:
        1. Loads the checkpoint/result
        2. Creates a new CalibrationRunner
        3. Continues the calibration
        4. Returns combined results
        
        Args:
            checkpoint_path: Path to checkpoint directory, checkpoint file,
                           or saved CalibrationResult base path
            model: Rainfall-runoff model instance
            inputs: Input DataFrame with precipitation and PET
            observed: Observed flow values
            objective: Objective function (if None, uses NSE)
            additional_iterations: Number of additional iterations to run
            implementation: DREAM implementation ('spotpy' or 'pydream')
            parameter_bounds: Parameter bounds (if None, uses model defaults)
            warmup_period: Warmup timesteps excluded from objective
            checkpoint_dir: Directory for new checkpoints (optional)
            checkpoint_interval: Checkpoint save interval
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            CalibrationResult combining previous and new results
            
        Example:
            >>> # Resume from checkpoint directory
            >>> result = CalibrationRunner.resume(
            ...     checkpoint_path='./checkpoints',
            ...     model=model,
            ...     inputs=inputs,
            ...     observed=observed,
            ...     additional_iterations=50000
            ... )
            
            >>> # Resume from saved CalibrationResult
            >>> result = CalibrationRunner.resume(
            ...     checkpoint_path='calibrations/run1',  # loads run1_meta.json
            ...     model=model,
            ...     inputs=inputs,
            ...     observed=observed,
            ...     additional_iterations=20000
            ... )
        """
        # Create runner
        runner = cls(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective or NSE(),
            parameter_bounds=parameter_bounds,
            warmup_period=warmup_period
        )
        
        # Use the run_dream method with resume_from
        return runner.run_dream(
            implementation=implementation,
            n_iterations=additional_iterations,
            resume_from=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            **kwargs
        )
    
    def create_report(
        self,
        result: CalibrationResult,
        catchment_info: Optional[Dict[str, Any]] = None,
        include_inputs: bool = True
    ) -> 'CalibrationReport':
        """
        Create a CalibrationReport from a calibration result.
        
        This method packages all the data needed for comprehensive visualization
        and analysis into a CalibrationReport object that can be saved and loaded.
        
        Args:
            result: CalibrationResult from any calibration method
            catchment_info: Optional catchment metadata dictionary with keys like:
                - name: Catchment name
                - gauge_id: Gauge identifier
                - area_km2: Catchment area (if different from model)
            include_inputs: Whether to include full input data for re-simulation
            
        Returns:
            CalibrationReport instance
            
        Example:
            >>> runner = CalibrationRunner(model, inputs, observed, objective)
            >>> result = runner.run_sceua_direct(max_evals=10000)
            >>> report = runner.create_report(
            ...     result,
            ...     catchment_info={'name': 'Queanbeyan', 'gauge_id': '410734'}
            ... )
            >>> report.save('calibrations/queanbeyan_nse.pkl')
        """
        from pyrrm.calibration.report import CalibrationReport
        
        # Run simulation with best parameters to get simulated flow
        self.model.set_parameters(result.best_parameters)
        self.model.reset()
        output = self.model.run(self.inputs)
        
        # Extract runoff column
        if 'runoff' in output.columns:
            sim_full = output['runoff'].values
        elif 'flow' in output.columns:
            sim_full = output['flow'].values
        else:
            sim_full = output.iloc[:, 0].values
        
        # Apply warmup
        sim = sim_full[self.warmup_period:]
        obs = self.observed[self.warmup_period:]
        
        # Get dates (post-warmup)
        if hasattr(self.inputs, 'index') and isinstance(self.inputs.index, pd.DatetimeIndex):
            dates = self.inputs.index[self.warmup_period:]
        elif 'Date' in self.inputs.columns:
            dates = pd.DatetimeIndex(self.inputs['Date'].values[self.warmup_period:])
        else:
            # Create dummy date range
            dates = pd.date_range(start='2000-01-01', periods=len(obs), freq='D')
        
        # Extract precipitation and PET if available
        precip = None
        pet = None
        for col in ['precipitation', 'Precipitation', 'P', 'Rain', 'rain']:
            if col in self.inputs.columns:
                precip = self.inputs[col].values[self.warmup_period:]
                break
        for col in ['pet', 'PET', 'evapotranspiration', 'ET']:
            if col in self.inputs.columns:
                pet = self.inputs[col].values[self.warmup_period:]
                break
        
        # Determine calibration period
        cal_start = str(dates[0].date()) if hasattr(dates[0], 'date') else str(dates[0])
        cal_end = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])
        
        # Build model configuration for re-simulation
        model_config = {
            'module': self.model.__class__.__module__,
            'class_name': self.model.__class__.__name__,
            'init_kwargs': {}
        }
        
        # Try to capture init kwargs (catchment_area_km2 is common)
        if hasattr(self.model, 'catchment_area_km2') and self.model.catchment_area_km2 is not None:
            model_config['init_kwargs']['catchment_area_km2'] = self.model.catchment_area_km2
        
        # Build catchment info
        info = catchment_info.copy() if catchment_info else {}
        if 'area_km2' not in info and hasattr(self.model, 'catchment_area_km2'):
            info['area_km2'] = self.model.catchment_area_km2
        
        # Create report
        report = CalibrationReport(
            result=result,
            observed=obs,
            simulated=sim,
            dates=dates,
            precipitation=precip,
            pet=pet,
            inputs=self.inputs.copy() if include_inputs else None,
            parameter_bounds=self._param_bounds.copy(),
            catchment_info=info,
            calibration_period=(cal_start, cal_end),
            warmup_days=self.warmup_period,
            model_config=model_config
        )
        
        return report