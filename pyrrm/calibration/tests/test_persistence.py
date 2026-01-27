"""
Tests for calibration persistence (save/load) functionality.

Tests:
- CalibrationResult.save() and load() round-trip
- CalibrationResult.to_dict() and from_dict()
- CheckpointManager save/load/cleanup
- Edge cases and error handling
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrrm.calibration.runner import CalibrationResult
from pyrrm.calibration.checkpoint import CheckpointManager, CheckpointInfo


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_result():
    """Create a sample CalibrationResult for testing."""
    all_samples = pd.DataFrame({
        'iteration': range(100),
        'likelihood': np.random.randn(100) + 0.8,
        'x1': np.random.uniform(100, 500, 100),
        'x2': np.random.uniform(-5, 5, 100),
        'x3': np.random.uniform(10, 300, 100),
    })
    
    return CalibrationResult(
        best_parameters={'x1': 350.0, 'x2': 1.5, 'x3': 100.0},
        best_objective=0.92,
        all_samples=all_samples,
        convergence_diagnostics={
            'gelman_rubin': {'x1': 1.02, 'x2': 1.01, 'x3': 1.03},
            'converged': True
        },
        runtime_seconds=123.45,
        method='PyDREAM (MT-DREAM(ZS))',
        objective_name='NSE',
        success=True,
        message='Converged successfully',
    )


@pytest.fixture
def sample_result_with_chains():
    """Create a sample CalibrationResult with chain data for PyDREAM."""
    all_samples = pd.DataFrame({
        'iteration': range(100),
        'likelihood': np.random.randn(100) + 0.8,
        'x1': np.random.uniform(100, 500, 100),
        'x2': np.random.uniform(-5, 5, 100),
    })
    
    # Simulate chain data (5 chains, 20 iterations each, 2 parameters)
    sampled_params_by_chain = [
        np.random.randn(20, 2) for _ in range(5)
    ]
    log_ps_by_chain = [
        np.random.randn(20, 1) for _ in range(5)
    ]
    
    raw_result = {
        'sampled_params_by_chain': sampled_params_by_chain,
        'log_ps_by_chain': log_ps_by_chain,
        'parameter_names': ['x1', 'x2'],
    }
    
    return CalibrationResult(
        best_parameters={'x1': 350.0, 'x2': 1.5},
        best_objective=0.92,
        all_samples=all_samples,
        convergence_diagnostics={'converged': True},
        runtime_seconds=123.45,
        method='PyDREAM (MT-DREAM(ZS))',
        objective_name='NSE',
        success=True,
        _raw_result=raw_result,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# ============================================================================
# CalibrationResult Tests
# ============================================================================

class TestCalibrationResultSerialization:
    """Tests for CalibrationResult serialization methods."""
    
    def test_to_dict_basic(self, sample_result):
        """Test to_dict returns valid dictionary."""
        data = sample_result.to_dict()
        
        assert isinstance(data, dict)
        assert data['best_parameters'] == sample_result.best_parameters
        assert data['best_objective'] == sample_result.best_objective
        assert data['method'] == sample_result.method
        assert data['objective_name'] == sample_result.objective_name
        assert data['success'] == sample_result.success
        assert 'saved_at' in data
        assert data['n_samples'] == len(sample_result.all_samples)
    
    def test_to_dict_without_samples(self, sample_result):
        """Test to_dict excludes samples by default."""
        data = sample_result.to_dict(include_samples=False)
        assert 'all_samples' not in data
    
    def test_to_dict_with_samples(self, sample_result):
        """Test to_dict can include samples."""
        data = sample_result.to_dict(include_samples=True)
        assert 'all_samples' in data
        assert len(data['all_samples']) == len(sample_result.all_samples)
    
    def test_from_dict_roundtrip(self, sample_result):
        """Test from_dict correctly reconstructs CalibrationResult."""
        data = sample_result.to_dict(include_samples=True)
        restored = CalibrationResult.from_dict(data)
        
        assert restored.best_parameters == sample_result.best_parameters
        assert restored.best_objective == sample_result.best_objective
        assert restored.method == sample_result.method
        assert restored.objective_name == sample_result.objective_name
        assert restored.success == sample_result.success
        assert len(restored.all_samples) == len(sample_result.all_samples)
    
    def test_from_dict_without_samples(self):
        """Test from_dict handles missing samples."""
        data = {
            'best_parameters': {'x1': 100.0},
            'best_objective': 0.9,
        }
        restored = CalibrationResult.from_dict(data)
        
        assert restored.best_parameters == {'x1': 100.0}
        assert len(restored.all_samples) == 0
    
    def test_to_dict_handles_numpy_types(self, sample_result):
        """Test to_dict converts numpy types to JSON-serializable."""
        # Add numpy types to diagnostics
        sample_result.convergence_diagnostics['numpy_float'] = np.float64(1.5)
        sample_result.convergence_diagnostics['numpy_array'] = np.array([1, 2, 3])
        
        data = sample_result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestCalibrationResultSaveLoad:
    """Tests for CalibrationResult save/load methods."""
    
    def test_save_creates_files(self, sample_result, temp_dir):
        """Test save creates expected files."""
        base_path = os.path.join(temp_dir, 'test_result')
        created = sample_result.save(base_path)
        
        assert len(created) >= 1
        assert os.path.exists(base_path + '_meta.json')
        # Either parquet or csv
        assert (os.path.exists(base_path + '_samples.parquet') or 
                os.path.exists(base_path + '_samples.csv'))
    
    def test_save_load_roundtrip(self, sample_result, temp_dir):
        """Test save/load roundtrip preserves data."""
        base_path = os.path.join(temp_dir, 'test_result')
        sample_result.save(base_path)
        
        loaded = CalibrationResult.load(base_path)
        
        assert loaded.best_parameters == sample_result.best_parameters
        assert loaded.best_objective == sample_result.best_objective
        assert loaded.method == sample_result.method
        assert len(loaded.all_samples) == len(sample_result.all_samples)
    
    def test_save_load_with_chains(self, sample_result_with_chains, temp_dir):
        """Test save/load preserves chain data for PyDREAM."""
        base_path = os.path.join(temp_dir, 'test_chains')
        sample_result_with_chains.save(base_path, include_chains=True)
        
        assert os.path.exists(base_path + '_chains.npz')
        
        loaded = CalibrationResult.load(base_path)
        
        assert loaded._raw_result is not None
        assert 'sampled_params_by_chain' in loaded._raw_result
        assert len(loaded._raw_result['sampled_params_by_chain']) == 5
    
    def test_save_without_samples(self, sample_result, temp_dir):
        """Test save without samples."""
        base_path = os.path.join(temp_dir, 'test_no_samples')
        created = sample_result.save(base_path, include_samples=False)
        
        assert len(created) == 1  # Only meta.json
        assert not os.path.exists(base_path + '_samples.parquet')
        assert not os.path.exists(base_path + '_samples.csv')
    
    def test_load_creates_parent_dirs(self, sample_result, temp_dir):
        """Test save creates parent directories."""
        base_path = os.path.join(temp_dir, 'subdir', 'nested', 'result')
        sample_result.save(base_path)
        
        assert os.path.exists(os.path.dirname(base_path))
    
    def test_load_not_found(self, temp_dir):
        """Test load raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            CalibrationResult.load(os.path.join(temp_dir, 'nonexistent'))
    
    def test_load_with_meta_suffix(self, sample_result, temp_dir):
        """Test load works with _meta.json suffix."""
        base_path = os.path.join(temp_dir, 'test_result')
        sample_result.save(base_path)
        
        # Load with explicit _meta.json
        loaded = CalibrationResult.load(base_path + '_meta.json')
        assert loaded.best_objective == sample_result.best_objective
    
    def test_can_resume(self, sample_result, sample_result_with_chains):
        """Test can_resume correctly identifies resumable results."""
        assert not sample_result.can_resume()
        assert sample_result_with_chains.can_resume()


# ============================================================================
# CheckpointManager Tests
# ============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    def test_init_creates_directory(self, temp_dir):
        """Test CheckpointManager creates directory."""
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        manager = CheckpointManager(checkpoint_dir)
        
        assert os.path.isdir(checkpoint_dir)
    
    def test_save_checkpoint(self, temp_dir):
        """Test saving a checkpoint."""
        manager = CheckpointManager(temp_dir)
        
        result = {
            'best_parameters': {'x1': 100.0},
            'best_objective': 0.9,
            'all_samples': pd.DataFrame({'x': [1, 2, 3]}),
        }
        
        path = manager.save_checkpoint(result, iteration=1000, method='test')
        
        assert os.path.exists(path + '.json')
    
    def test_save_checkpoint_with_chains(self, temp_dir):
        """Test saving checkpoint with chain data."""
        manager = CheckpointManager(temp_dir)
        
        result = {
            'best_parameters': {'x1': 100.0},
            'best_objective': 0.9,
            'sampled_params_by_chain': [np.random.randn(10, 2) for _ in range(3)],
            'log_ps_by_chain': [np.random.randn(10, 1) for _ in range(3)],
            'parameter_names': ['x1', 'x2'],
        }
        
        path = manager.save_checkpoint(result, iteration=1000)
        
        assert os.path.exists(path + '_chains.npz')
    
    def test_load_checkpoint(self, temp_dir):
        """Test loading a checkpoint."""
        manager = CheckpointManager(temp_dir)
        
        result = {
            'best_parameters': {'x1': 100.0, 'x2': 50.0},
            'best_objective': 0.9,
        }
        
        path = manager.save_checkpoint(result, iteration=1000)
        loaded = manager.load_checkpoint(path)
        
        assert loaded['best_parameters'] == result['best_parameters']
        assert loaded['best_objective'] == result['best_objective']
    
    def test_load_latest_checkpoint(self, temp_dir):
        """Test loading the latest checkpoint."""
        manager = CheckpointManager(temp_dir)
        
        # Save multiple checkpoints
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.7}, 
            iteration=1000
        )
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.8}, 
            iteration=2000
        )
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.9}, 
            iteration=3000
        )
        
        latest = manager.load_latest_checkpoint()
        
        assert latest['best_objective'] == 0.9
        assert latest['iteration'] == 3000
    
    def test_load_best_checkpoint(self, temp_dir):
        """Test loading the best checkpoint."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoints with different objectives
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.8}, 
            iteration=1000
        )
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.95},  # Best
            iteration=2000
        )
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.85}, 
            iteration=3000
        )
        
        best = manager.load_best_checkpoint(maximize=True)
        
        assert best['best_objective'] == 0.95
    
    def test_cleanup_old_checkpoints(self, temp_dir):
        """Test cleanup removes old checkpoints."""
        manager = CheckpointManager(temp_dir, max_checkpoints=2)
        
        # Save 4 checkpoints
        for i in range(1, 5):
            manager.save_checkpoint(
                {'best_parameters': {}, 'best_objective': 0.5 + i * 0.1}, 
                iteration=i * 1000
            )
        
        # Should only have 2 left
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        
        # Should be the latest ones
        iterations = [cp.iteration for cp in checkpoints]
        assert 3000 in iterations
        assert 4000 in iterations
    
    def test_list_checkpoints(self, temp_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(temp_dir)
        
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.8}, 
            iteration=1000
        )
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.9}, 
            iteration=2000
        )
        
        checkpoints = manager.list_checkpoints()
        
        assert len(checkpoints) == 2
        assert all(isinstance(cp, CheckpointInfo) for cp in checkpoints)
        assert checkpoints[0].iteration < checkpoints[1].iteration
    
    def test_clear_all_checkpoints(self, temp_dir):
        """Test clearing all checkpoints."""
        manager = CheckpointManager(temp_dir)
        
        # Save some checkpoints
        for i in range(3):
            manager.save_checkpoint(
                {'best_parameters': {}, 'best_objective': 0.5}, 
                iteration=(i + 1) * 1000
            )
        
        removed = manager.clear_all_checkpoints()
        
        assert removed == 3
        assert len(manager.list_checkpoints()) == 0
    
    def test_should_checkpoint_interval(self, temp_dir):
        """Test should_checkpoint respects interval."""
        manager = CheckpointManager(temp_dir, checkpoint_interval=1000)
        
        assert manager.should_checkpoint(1000)
        manager._last_checkpoint_iteration = 1000
        
        assert not manager.should_checkpoint(1500)
        assert manager.should_checkpoint(2000)
    
    def test_should_checkpoint_improvement(self, temp_dir):
        """Test should_checkpoint on improvement."""
        manager = CheckpointManager(
            temp_dir, 
            checkpoint_interval=10000,  # Large interval
            checkpoint_on_improvement=True
        )
        
        # First time always checkpoints
        assert manager.should_checkpoint(100, current_best=0.8)
        manager._best_objective = 0.8
        manager._last_checkpoint_iteration = 100
        
        # No checkpoint without improvement
        assert not manager.should_checkpoint(200, current_best=0.8)
        
        # Checkpoint on improvement
        assert manager.should_checkpoint(200, current_best=0.9)
    
    def test_get_latest_iteration(self, temp_dir):
        """Test getting latest iteration."""
        manager = CheckpointManager(temp_dir)
        
        assert manager.get_latest_iteration() == 0
        
        manager.save_checkpoint(
            {'best_parameters': {}, 'best_objective': 0.8}, 
            iteration=5000
        )
        
        assert manager.get_latest_iteration() == 5000


# ============================================================================
# Integration Tests
# ============================================================================

class TestPersistenceIntegration:
    """Integration tests for the persistence system."""
    
    def test_save_load_real_workflow(self, sample_result_with_chains, temp_dir):
        """Test a realistic save/load workflow."""
        # Save result
        result_path = os.path.join(temp_dir, 'calibration_run1')
        sample_result_with_chains.save(result_path)
        
        # Simulate loading in new session
        loaded = CalibrationResult.load(result_path)
        
        # Verify resumable
        assert loaded.can_resume()
        assert loaded._raw_result['parameter_names'] == ['x1', 'x2']
    
    def test_checkpoint_then_load(self, temp_dir):
        """Test checkpointing then loading for resume."""
        manager = CheckpointManager(temp_dir)
        
        # Simulate calibration progress
        for i in range(1, 4):
            result = {
                'best_parameters': {'x1': 100.0 + i * 10},
                'best_objective': 0.7 + i * 0.05,
                'sampled_params_by_chain': [np.random.randn(10, 1) for _ in range(3)],
                'log_ps_by_chain': [np.random.randn(10, 1) for _ in range(3)],
                'parameter_names': ['x1'],
                'runtime_seconds': i * 100,
            }
            manager.save_checkpoint(result, iteration=i * 1000, method='PyDREAM')
        
        # Simulate resuming
        checkpoint = manager.load_latest_checkpoint()
        
        assert checkpoint['iteration'] == 3000
        assert 'sampled_params_by_chain' in checkpoint
        assert len(checkpoint['sampled_params_by_chain']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
