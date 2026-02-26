"""
Unified calibration runner for pyrrm models.

This module provides the CalibrationRunner class, which offers a unified
interface for calibrating models using various algorithms:
- PyDREAM (MT-DREAM(ZS)): Bayesian MCMC with multi-try sampling and snooker updates
- SCE-UA (direct): Shuffled Complex Evolution (vendored, no external dependencies)
- SciPy optimizers: Differential Evolution, Dual Annealing, Basin-hopping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, Literal, Union
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import numpy as np
import pandas as pd
import warnings

_log = logging.getLogger(__name__)

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

try:
    from pyrrm.calibration.pydream_adapter import PYDREAM_AVAILABLE
except ImportError:
    PYDREAM_AVAILABLE = False


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
    - DREAM (Bayesian MCMC via PyDREAM MT-DREAM(ZS))
    - SCE-UA (Shuffled Complex Evolution, vendored -- no external dependencies)
    - Differential Evolution (scipy)
    - Dual Annealing (scipy)
    - Basin-hopping (scipy)
    
    Example:
        >>> from pyrrm.models import GR4J
        >>> from pyrrm.calibration import CalibrationRunner
        >>> from pyrrm.calibration.objective_functions import NSE
        >>> 
        >>> model = GR4J()
        >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
        >>> 
        >>> # SCE-UA global optimisation (fast, no external deps)
        >>> result = runner.run_sceua_direct(max_evals=20000, seed=42)
        >>> 
        >>> # Bayesian MCMC with PyDREAM
        >>> result = runner.run_dream(n_iterations=10000, multitry=5)
        >>> 
        >>> # SciPy differential evolution
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
        n_iterations: int = 10000,
        n_chains: int = 5,
        multitry: int = 5,
        snooker: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        resume_from: Optional[str] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Run DREAM algorithm using PyDREAM (MT-DREAM(ZS)).
        
        DREAM (DiffeRential Evolution Adaptive Metropolis) is a Bayesian MCMC 
        algorithm suitable for high-dimensional parameter estimation with 
        complex posterior distributions.  This method uses PyDREAM which
        implements Multi-Try DREAM with snooker updates.
        
        Args:
            n_iterations: Number of MCMC iterations
            n_chains: Number of parallel chains
            multitry: Number of proposal points per iteration (1=standard DREAM)
            snooker: Probability of snooker update (0 to disable)
            checkpoint_dir: Directory for automatic checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N iterations
            resume_from: Path to checkpoint or CalibrationResult to resume from
            **kwargs: Additional PyDREAM parameters (passed to ``run_pydream``)
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
            
        Example with checkpointing:
            >>> result = runner.run_dream(
            ...     n_iterations=100000,
            ...     checkpoint_dir='./checkpoints',
            ...     checkpoint_interval=5000
            ... )
            
        Example resuming from checkpoint:
            >>> result = runner.run_dream(
            ...     n_iterations=50000,
            ...     resume_from='./checkpoints'
            ... )
        """
        if resume_from is not None:
            return self._resume_dream(
                resume_from=resume_from,
                n_iterations=n_iterations,
                n_chains=n_chains,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                **kwargs
            )
        
        return self.run_pydream(
            n_iterations=n_iterations,
            n_chains=n_chains,
            multitry=multitry,
            snooker=snooker,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            **kwargs
        )
    
    def _resume_dream(
        self,
        resume_from: str,
        n_iterations: int,
        n_chains: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        **kwargs
    ) -> CalibrationResult:
        """
        Resume PyDREAM calibration from checkpoint or saved result.
        
        Args:
            resume_from: Path to checkpoint directory, checkpoint file, or saved result
            n_iterations: Additional iterations to run
            n_chains: Number of chains
            checkpoint_dir: Directory for new checkpoints
            checkpoint_interval: Checkpoint interval
            **kwargs: Additional parameters
            
        Returns:
            CalibrationResult combining previous and new results
        """
        from pyrrm.calibration.checkpoint import CheckpointManager
        
        resume_path = Path(resume_from)
        
        if resume_path.is_dir():
            manager = CheckpointManager(str(resume_path))
            previous_result = manager.load_latest_checkpoint()
            if previous_result is None:
                raise ValueError(f"No checkpoints found in {resume_from}")
        elif resume_path.suffix == '.json' or (resume_path.parent / (resume_path.name + '.json')).exists():
            try:
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
        
        if 'sampled_params_by_chain' not in previous_result:
            raise ValueError(
                "Cannot resume: previous result does not contain chain data. "
                "Start a fresh PyDREAM run instead."
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
    
    def run_pydream(
        self,
        n_iterations: int = 10000,
        n_chains: int = 5,
        multitry: int = 5,
        snooker: float = 0.1,
        nCR: int = 3,
        adapt_crossover: bool = False,  # Disabled by default due to PyDREAM bug
        adapt_gamma: bool = False,
        DEpairs: int = 1,
        p_gamma_unity: float = 0.2,
        parallel: bool = False,
        mp_context: str = None,
        dbname: str = None,
        dbformat: str = 'csv',
        write_interval: int = 1,
        verbose: bool = True,
        nverbose: int = 100,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        # Batch mode parameters (optional - enables early stopping)
        batch_size: Optional[int] = None,
        min_iterations: int = 300,
        convergence_threshold: float = 1.05,
        patience: int = 2,
        post_convergence_iterations: int = 1000,
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
            written to `{dbname}.csv`.
            
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
            adapt_crossover: Whether to adapt crossover probabilities.
                           WARNING: Disabled by default due to PyDREAM bug.
            adapt_gamma: Whether to adapt gamma (step size) levels (recommended: True)
            DEpairs: Number of chain pairs for differential evolution
            p_gamma_unity: Probability of gamma=1 (full step)
            parallel: Enable parallel processing for multi-try evaluations.
                     When True and multitry>1, proposals are evaluated in parallel.
            mp_context: Multiprocessing context ('fork', 'spawn', 'forkserver').
                       None uses system default. 'spawn' is safest, 'fork' fastest.
            dbname: Database name for progress tracking. Writes to {dbname}.csv.
            dbformat: Output format ('csv' only)
            write_interval: Write every N evaluations (1=all, 10=every 10th)
            verbose: Whether to print progress
            nverbose: Print progress every N iterations (default: 100).
                     Note: Progress from worker processes may not display in
                     Jupyter notebooks. Use `dbname` for reliable tracking.
            checkpoint_dir: Directory for automatic checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N iterations
            batch_size: If set (>0), enables BATCH MODE with early stopping.
                       Runs iterations in batches, checking GR convergence after
                       each batch and stopping early when converged.
                       If None, runs all iterations without early stopping (normal mode).
            min_iterations: Minimum iterations before checking convergence (batch mode)
            convergence_threshold: Gelman-Rubin threshold for convergence (default: 1.05, strict)
            patience: Consecutive converged batches required to stop (batch mode)
            post_convergence_iterations: Additional iterations after convergence
                                        for robust posterior sampling (default: 1000)
            **kwargs: Additional PyDREAM parameters
            
        Returns:
            CalibrationResult with best parameters and MCMC chain
            
        Batch Mode Example:
            Enable batch mode for automatic early stopping when GR converges:
            
            ```python
            result = runner.run_pydream(
                n_iterations=5000,      # Maximum iterations
                batch_size=100,          # Check GR every 100 iterations
                min_iterations=300,      # Don't check before 300 iterations
                convergence_threshold=1.1,  # Stop when all GR < 1.1
                patience=2,              # Require 2 consecutive converged batches
                post_convergence_iterations=200,  # Run 200 more after convergence
            )
            # May stop early if GR converges before 5000 iterations!
            ```
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
            adapt_gamma=adapt_gamma,
            DEpairs=DEpairs,
            p_gamma_unity=p_gamma_unity,
            parallel=parallel,
            mp_context=mp_context,
            dbname=dbname,
            dbformat=dbformat,
            write_interval=write_interval,
            verbose=verbose,
            nverbose=nverbose,
            # Batch mode parameters
            batch_size=batch_size,
            min_iterations=min_iterations,
            convergence_threshold=convergence_threshold,
            patience=patience,
            post_convergence_iterations=post_convergence_iterations,
            **kwargs
        )
        
        # Save checkpoint if checkpoint_dir specified
        if checkpoint_dir is not None:
            from pyrrm.calibration.checkpoint import CheckpointManager
            manager = CheckpointManager(
                checkpoint_dir, 
                checkpoint_interval=checkpoint_interval
            )
            # Use actual iterations (may differ from n_iterations if early stopped)
            actual_iterations = result.get('total_iterations', n_iterations)
            manager.save_checkpoint(
                result, 
                iteration=actual_iterations, 
                method='PyDREAM (MT-DREAM(ZS))'
            )
            if verbose:
                _log.info("Checkpoint saved to %s", checkpoint_dir)
        
        # Determine convergence status
        converged = result.get('convergence_diagnostics', {}).get('converged', None)
        success = converged if converged is not None else True
        
        # Store raw result for potential continuation
        raw_result = {
            'sampled_params_by_chain': result.get('sampled_params_by_chain'),
            'log_ps_by_chain': result.get('log_ps_by_chain'),
            'parameter_names': result.get('parameter_names'),
            'early_stopped': result.get('early_stopped'),
            'total_iterations': result.get('total_iterations'),
            'convergence_history': result.get('convergence_history'),
        }
        
        # Build method name
        method_name = 'PyDREAM (MT-DREAM(ZS))'
        if batch_size is not None and result.get('early_stopped'):
            method_name += ' [early stopped]'
        
        return CalibrationResult(
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            all_samples=result['all_samples'],
            convergence_diagnostics=result['convergence_diagnostics'],
            runtime_seconds=result['runtime_seconds'],
            method=method_name,
            objective_name=self.objective.name,
            success=success,
            _raw_result=raw_result,
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
        Run SCE-UA (Shuffled Complex Evolution - University of Arizona).
        
        A vendored, dependency-free implementation with:
        - PCA recovery for lost dimensions (Chu et al., 2010)
        - Multiple convergence criteria
        - Parallel evaluation via ThreadPoolExecutor
        - Initial parameter sets can be provided via x0
        - All evaluations tracked automatically
        
        SCE-UA is a global optimization method developed specifically for
        calibrating hydrological models.
        
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
    
    def run_nuts(
        self,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 8,
        likelihood_type: str = "gaussian",
        transform: str = "none",
        transform_params: Optional[dict] = None,
        error_model: str = "iid",
        prior_config: Optional[dict] = None,
        sigma_prior_scale: float = 1.0,
        reparameterize: bool = False,
        use_float64: bool = True,
        max_ninc: Optional[int] = None,
        fast_mode: bool = False,
        tvp_config: Optional[dict] = None,
        seed: int = 42,
        progress_bar: bool = True,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Run NumPyro NUTS (No-U-Turn Sampler) Bayesian calibration.

        NUTS is a gradient-based MCMC sampler that uses JAX automatic
        differentiation to efficiently explore parameter space.  It
        requires a JAX-ported forward model (e.g. ``gr4j_jax``,
        ``sacramento_jax``).

        Args:
            num_warmup: NUTS warmup (adaptation) iterations per chain.
            num_samples: Post-warmup samples per chain.
            num_chains: Number of independent MCMC chains.
            target_accept_prob: Target acceptance probability (0.6-0.95).
            max_tree_depth: Maximum leapfrog tree depth (default 8;
                max leapfrog steps = 2^depth).
            likelihood_type: ``"gaussian"`` or ``"transformed_gaussian"``.
            transform: Flow transformation for the likelihood
                (``"none"``, ``"sqrt"``, ``"log"``, ``"inverse"``,
                ``"boxcox"``).
            transform_params: Optional overrides for the transform.
            error_model: ``"iid"`` (default) or ``"ar1"``.
            prior_config: Dict mapping parameter name to a
                ``numpyro.distributions`` object for custom priors.
            sigma_prior_scale: Scale for the HalfNormal prior on sigma.
            reparameterize: If True, sample Uniform-prior parameters in
                [0, 1] and deterministically transform to physical
                bounds.  Recommended for models with many parameters
                spanning different scales (e.g. Sacramento).
            use_float64: If True (default), run in float64 precision.
                Set to False for ~2x speedup on CPU.
            max_ninc: Override Sacramento inner loop count (default
                ``None`` keeps model default of 20).  Set to 5 for
                faster calibration.  Ignored when ``fast_mode=True``.
            fast_mode: If True, bypass the Sacramento inner sub-daily
                loop entirely (ninc=1), eliminating the nested
                ``lax.scan`` and reducing the XLA graph by ~10-16×.
                Dramatically faster JIT compilation and gradient
                evaluation.  Recommended for calibration; validate
                posterior with ``fast_mode=False``.
            tvp_config: Optional dict mapping parameter names to
                ``TVPPrior`` instances for time-varying parameter
                calibration.  Example::

                    from pyrrm.calibration.tvp_priors import GaussianRandomWalk
                    tvp_config = {"X1": GaussianRandomWalk(lower=1, upper=1500)}

            seed: PRNG seed for reproducibility.
            progress_bar: Show NumPyro sampling progress bar.
            verbose: Print summary to stdout.

        Returns:
            CalibrationResult containing posterior medians as best
            parameters, all MCMC samples, and ArviZ InferenceData
            stored in ``_raw_result["inference_data"]``.

        Example:
            >>> runner = CalibrationRunner(model, inputs, observed)
            >>> result = runner.run_nuts(num_warmup=500, num_samples=1000)
            >>> print(result.best_parameters)
        """
        try:
            from pyrrm.calibration.numpyro_adapter import (
                run_nuts as _adapter_run_nuts,
                NUMPYRO_AVAILABLE as _NUMPYRO_OK,
            )
        except ImportError:
            raise ImportError(
                "NumPyro is required for NUTS calibration. "
                "Install with: pip install jax jaxlib numpyro arviz"
            )

        if not _NUMPYRO_OK:
            raise ImportError(
                "NumPyro is required for NUTS calibration. "
                "Install with: pip install jax jaxlib numpyro arviz"
            )

        result = _adapter_run_nuts(
            model=self.model,
            inputs=self.inputs,
            observed=self.observed,
            parameter_bounds=self._param_bounds,
            warmup_period=self.warmup_period,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            likelihood_type=likelihood_type,
            transform=transform,
            transform_params=transform_params,
            error_model=error_model,
            prior_config=prior_config,
            sigma_prior_scale=sigma_prior_scale,
            reparameterize=reparameterize,
            use_float64=use_float64,
            max_ninc=max_ninc,
            fast_mode=fast_mode,
            tvp_config=tvp_config,
            seed=seed,
            progress_bar=progress_bar,
            verbose=verbose,
        )

        return CalibrationResult(
            best_parameters=result["best_parameters"],
            best_objective=result["best_objective"],
            all_samples=result["all_samples"],
            convergence_diagnostics=result["convergence_diagnostics"],
            runtime_seconds=result["runtime_seconds"],
            method="NUTS (NumPyro)",
            objective_name=f"log_likelihood ({transform})",
            success=result["convergence_diagnostics"].get("converged", False),
            message="",
            _raw_result=result,
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
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
        **kwargs
    ) -> CalibrationResult:
        """
        Resume PyDREAM calibration from a saved checkpoint or CalibrationResult.
        
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
            parameter_bounds: Parameter bounds (if None, uses model defaults)
            warmup_period: Warmup timesteps excluded from objective
            checkpoint_dir: Directory for new checkpoints (optional)
            checkpoint_interval: Checkpoint save interval
            **kwargs: Additional PyDREAM parameters
            
        Returns:
            CalibrationResult combining previous and new results
            
        Example:
            >>> result = CalibrationRunner.resume(
            ...     checkpoint_path='./checkpoints',
            ...     model=model,
            ...     inputs=inputs,
            ...     observed=observed,
            ...     additional_iterations=50000
            ... )
        """
        runner = cls(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective or NSE(),
            parameter_bounds=parameter_bounds,
            warmup_period=warmup_period
        )
        
        return runner.run_dream(
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
        include_inputs: bool = True,
        experiment_name: Optional[str] = None,
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
            experiment_name: Optional canonical experiment key (from the
                naming convention) stored in the report metadata.

        Returns:
            CalibrationReport instance

        Example:
            >>> runner = CalibrationRunner(model, inputs, observed, objective)
            >>> result = runner.run_sceua_direct(max_evals=10000)
            >>> report = runner.create_report(
            ...     result,
            ...     catchment_info={'name': 'Queanbeyan', 'gauge_id': '410734'},
            ...     experiment_name='410734_sacramento_nse_sceua',
            ... )
            >>> report.save('calibrations/410734_sacramento_nse_sceua.pkl')
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
        
        from pyrrm.data import resolve_column

        precip = None
        pet = None
        pcol = resolve_column(self.inputs, "precipitation")
        if pcol is not None:
            precip = self.inputs[pcol].values[self.warmup_period:]
        ecol = resolve_column(self.inputs, "pet")
        if ecol is not None:
            pet = self.inputs[ecol].values[self.warmup_period:]
        
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
            model_config=model_config,
            experiment_name=experiment_name,
        )
        
        return report