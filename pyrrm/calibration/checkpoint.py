"""
Checkpoint management for long-running calibrations.

This module provides the CheckpointManager class for automatic periodic
saving of calibration state, enabling resume from interruptions.

Example:
    >>> manager = CheckpointManager('./checkpoints', checkpoint_interval=1000)
    >>> # During calibration, save checkpoints
    >>> manager.save_checkpoint(result_dict, iteration=1000)
    >>> # After interruption, load latest
    >>> checkpoint = manager.load_latest_checkpoint()
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import os
import warnings

import numpy as np
import pandas as pd


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    path: str
    iteration: int
    timestamp: datetime
    best_objective: float
    method: str
    
    def __repr__(self) -> str:
        return (
            f"CheckpointInfo(iter={self.iteration}, "
            f"best={self.best_objective:.4f}, "
            f"time={self.timestamp.strftime('%Y-%m-%d %H:%M')})"
        )


class CheckpointManager:
    """
    Manages automatic checkpointing during calibration.
    
    Saves periodic checkpoints to allow resuming interrupted calibrations.
    Uses rolling checkpoints to avoid excessive disk usage.
    
    Attributes:
        checkpoint_dir: Directory for checkpoint files
        checkpoint_interval: Save every N iterations
        max_checkpoints: Maximum number of checkpoints to keep
        checkpoint_on_improvement: Save immediately when best objective improves
    
    Example:
        >>> # Create manager
        >>> manager = CheckpointManager(
        ...     './checkpoints',
        ...     checkpoint_interval=5000,
        ...     max_checkpoints=3
        ... )
        >>> 
        >>> # Save checkpoint during calibration
        >>> manager.save_checkpoint(result_dict, iteration=5000)
        >>> 
        >>> # After interruption, load latest
        >>> checkpoint = manager.load_latest_checkpoint()
        >>> if checkpoint:
        ...     # Resume from checkpoint
        ...     continue_pydream(..., previous_result=checkpoint)
    """
    
    MANIFEST_FILE = 'checkpoint_manifest.json'
    
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 3,
        checkpoint_on_improvement: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files (created if needed)
            checkpoint_interval: Save checkpoint every N iterations
            max_checkpoints: Maximum checkpoints to retain (oldest deleted)
            checkpoint_on_improvement: Save when best objective improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoint_on_improvement = checkpoint_on_improvement
        
        # Tracking state
        self._best_objective: Optional[float] = None
        self._last_checkpoint_iteration: int = 0
        self._checkpoints: List[CheckpointInfo] = []
        
        # Create directory and load existing manifest
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load checkpoint manifest if it exists."""
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                self._checkpoints = [
                    CheckpointInfo(
                        path=cp['path'],
                        iteration=cp['iteration'],
                        timestamp=datetime.fromisoformat(cp['timestamp']),
                        best_objective=cp['best_objective'],
                        method=cp.get('method', '')
                    )
                    for cp in data.get('checkpoints', [])
                ]
                self._best_objective = data.get('best_objective')
            except (json.JSONDecodeError, KeyError) as e:
                warnings.warn(f"Could not load checkpoint manifest: {e}")
                self._checkpoints = []
    
    def _save_manifest(self) -> None:
        """Save checkpoint manifest."""
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        data = {
            'checkpoints': [
                {
                    'path': cp.path,
                    'iteration': cp.iteration,
                    'timestamp': cp.timestamp.isoformat(),
                    'best_objective': cp.best_objective,
                    'method': cp.method,
                }
                for cp in self._checkpoints
            ],
            'best_objective': self._best_objective,
            'updated_at': datetime.now().isoformat(),
        }
        with open(manifest_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def should_checkpoint(
        self, 
        iteration: int, 
        current_best: Optional[float] = None,
        maximize: bool = True
    ) -> bool:
        """
        Check if a checkpoint should be saved.
        
        Args:
            iteration: Current iteration number
            current_best: Current best objective value (for improvement check)
            maximize: Whether higher objective is better
            
        Returns:
            True if checkpoint should be saved
        """
        # Check interval
        if iteration - self._last_checkpoint_iteration >= self.checkpoint_interval:
            return True
        
        # Check improvement
        if self.checkpoint_on_improvement and current_best is not None:
            if self._best_objective is None:
                return True
            if maximize and current_best > self._best_objective:
                return True
            if not maximize and current_best < self._best_objective:
                return True
        
        return False
    
    def save_checkpoint(
        self,
        result: Dict[str, Any],
        iteration: int,
        method: str = '',
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            result: Calibration result dictionary containing:
                - best_parameters: Dict[str, float]
                - best_objective: float
                - all_samples: pd.DataFrame (optional)
                - sampled_params_by_chain: List[np.ndarray] (for PyDREAM)
                - log_ps_by_chain: List[np.ndarray] (for PyDREAM)
                - parameter_names: List[str]
            iteration: Current iteration number
            method: Calibration method name
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint filename
        checkpoint_name = f'checkpoint_{iteration:06d}'
        base_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare metadata
        meta = {
            'iteration': iteration,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'best_parameters': result.get('best_parameters', {}),
            'best_objective': float(result.get('best_objective', 0)),
            'convergence_diagnostics': self._serialize_value(
                result.get('convergence_diagnostics', {})
            ),
            'runtime_seconds': float(result.get('runtime_seconds', 0)),
            'n_samples': len(result.get('all_samples', [])),
            'parameter_names': result.get('parameter_names', []),
            'has_chain_data': 'sampled_params_by_chain' in result,
        }
        
        # Save metadata JSON
        meta_path = str(base_path) + '.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Save chain data for PyDREAM
        if 'sampled_params_by_chain' in result:
            chains_path = str(base_path) + '_chains.npz'
            chain_data = {
                'sampled_params_by_chain': np.array(
                    result['sampled_params_by_chain'], dtype=object
                ),
                'log_ps_by_chain': np.array(
                    result['log_ps_by_chain'], dtype=object
                ),
            }
            if 'parameter_names' in result:
                chain_data['parameter_names'] = np.array(result['parameter_names'])
            np.savez_compressed(chains_path, **chain_data)
        
        # Save samples (as CSV for portability)
        if 'all_samples' in result and result['all_samples'] is not None:
            samples = result['all_samples']
            if isinstance(samples, pd.DataFrame) and len(samples) > 0:
                samples_path = str(base_path) + '_samples.csv'
                samples.to_csv(samples_path, index=False)
        
        # Update tracking
        self._last_checkpoint_iteration = iteration
        best_obj = result.get('best_objective', 0)
        if self._best_objective is None or best_obj > self._best_objective:
            self._best_objective = best_obj
        
        # Add to checkpoint list
        checkpoint_info = CheckpointInfo(
            path=str(base_path),
            iteration=iteration,
            timestamp=datetime.now(),
            best_objective=best_obj,
            method=method,
        )
        self._checkpoints.append(checkpoint_info)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        # Save manifest
        self._save_manifest()
        
        return str(base_path)
    
    def _serialize_value(self, value: Any) -> Any:
        """Convert numpy types to JSON-serializable."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            return float(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value
    
    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific checkpoint.
        
        Args:
            path: Base path to checkpoint (without extension)
            
        Returns:
            Dictionary with checkpoint data, or None if not found
        """
        meta_path = path + '.json'
        if not os.path.exists(meta_path):
            return None
        
        # Load metadata
        with open(meta_path, 'r') as f:
            result = json.load(f)
        
        # Load chain data if available
        chains_path = path + '_chains.npz'
        if os.path.exists(chains_path):
            chain_data = np.load(chains_path, allow_pickle=True)
            result['sampled_params_by_chain'] = list(
                chain_data['sampled_params_by_chain']
            )
            result['log_ps_by_chain'] = list(chain_data['log_ps_by_chain'])
            if 'parameter_names' in chain_data:
                result['parameter_names'] = list(chain_data['parameter_names'])
        
        # Load samples if available
        samples_path = path + '_samples.csv'
        if os.path.exists(samples_path):
            result['all_samples'] = pd.read_csv(samples_path)
        else:
            result['all_samples'] = pd.DataFrame()
        
        return result
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Dictionary with checkpoint data, or None if no checkpoints exist
        """
        if not self._checkpoints:
            return None
        
        # Sort by iteration and get latest
        latest = max(self._checkpoints, key=lambda cp: cp.iteration)
        return self.load_checkpoint(latest.path)
    
    def load_best_checkpoint(self, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load the checkpoint with the best objective value.
        
        Args:
            maximize: If True, load checkpoint with highest objective
            
        Returns:
            Dictionary with checkpoint data, or None if no checkpoints exist
        """
        if not self._checkpoints:
            return None
        
        if maximize:
            best = max(self._checkpoints, key=lambda cp: cp.best_objective)
        else:
            best = min(self._checkpoints, key=lambda cp: cp.best_objective)
        
        return self.load_checkpoint(best.path)
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """
        List all available checkpoints.
        
        Returns:
            List of CheckpointInfo objects sorted by iteration
        """
        return sorted(self._checkpoints, key=lambda cp: cp.iteration)
    
    def cleanup_old_checkpoints(self) -> int:
        """
        Remove old checkpoints beyond max_checkpoints limit.
        
        Returns:
            Number of checkpoints removed
        """
        if len(self._checkpoints) <= self.max_checkpoints:
            return 0
        
        # Sort by iteration and remove oldest
        sorted_checkpoints = sorted(
            self._checkpoints, 
            key=lambda cp: cp.iteration
        )
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        removed = 0
        for checkpoint in to_remove:
            try:
                # Remove checkpoint files
                for ext in ['.json', '_chains.npz', '_samples.csv']:
                    file_path = checkpoint.path + ext
                    if os.path.exists(file_path):
                        os.remove(file_path)
                self._checkpoints.remove(checkpoint)
                removed += 1
            except Exception as e:
                warnings.warn(f"Could not remove checkpoint {checkpoint.path}: {e}")
        
        return removed
    
    def clear_all_checkpoints(self) -> int:
        """
        Remove all checkpoints.
        
        Returns:
            Number of checkpoints removed
        """
        removed = 0
        for checkpoint in list(self._checkpoints):
            try:
                for ext in ['.json', '_chains.npz', '_samples.csv']:
                    file_path = checkpoint.path + ext
                    if os.path.exists(file_path):
                        os.remove(file_path)
                self._checkpoints.remove(checkpoint)
                removed += 1
            except Exception as e:
                warnings.warn(f"Could not remove checkpoint {checkpoint.path}: {e}")
        
        # Remove manifest
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            manifest_path.unlink()
        
        self._best_objective = None
        self._last_checkpoint_iteration = 0
        
        return removed
    
    def get_latest_iteration(self) -> int:
        """Get the iteration number of the latest checkpoint."""
        if not self._checkpoints:
            return 0
        return max(cp.iteration for cp in self._checkpoints)
    
    def __repr__(self) -> str:
        return (
            f"CheckpointManager(dir='{self.checkpoint_dir}', "
            f"n_checkpoints={len(self._checkpoints)}, "
            f"interval={self.checkpoint_interval})"
        )
