"""
Tests for the direct SCE-UA implementation.

Tests cover:
- Basic optimization with simple objective functions
- Parameter bounds handling
- Objective direction (maximize vs minimize)
- Convergence criteria
- Integration with pyrrm models
"""

import numpy as np
import pandas as pd
import pytest

from pyrrm.calibration._sceua import minimize, Result
from pyrrm.calibration.sceua_adapter import (
    run_sceua_direct,
    calibrate_sceua,
    SCEUAModelWrapper,
    SCEUACalibrationResult,
    _should_maximize,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockObjectiveMaximize:
    """Mock objective function that should be maximized (like NSE)."""
    name = "MockNSE"
    direction = "maximize"
    
    def calculate(self, simulated, observed):
        """Return negative MSE (higher is better)."""
        return -np.mean((simulated - observed) ** 2)


class MockObjectiveMinimize:
    """Mock objective function that should be minimized (like RMSE)."""
    name = "MockRMSE"
    direction = "minimize"
    
    def calculate(self, simulated, observed):
        """Return RMSE (lower is better)."""
        return np.sqrt(np.mean((simulated - observed) ** 2))


class MockObjectiveLegacy:
    """Mock objective function using legacy interface."""
    name = "MockLegacy"
    maximize = True
    
    def calculate(self, simulated, observed):
        """Return negative MSE (higher is better)."""
        return -np.mean((simulated - observed) ** 2)


class MockModel:
    """Simple mock model for testing."""
    
    def __init__(self, parameters=None):
        self.parameters = parameters or {'a': 1.0, 'b': 0.0}
    
    def get_parameter_bounds(self):
        return {'a': (0.0, 5.0), 'b': (-5.0, 5.0)}
    
    def set_parameters(self, params):
        self.parameters = params
    
    def reset(self):
        pass
    
    def run(self, inputs):
        """Simple linear model: y = a * x + b"""
        a = self.parameters.get('a', 1.0)
        b = self.parameters.get('b', 0.0)
        x = inputs['x'].values
        y = a * x + b
        return pd.DataFrame({'flow': y})


# =============================================================================
# Tests for vendored SCE-UA algorithm
# =============================================================================

class TestSCEUAMinimize:
    """Tests for the vendored sceua.minimize function."""
    
    def test_simple_quadratic(self):
        """Test optimization of simple quadratic function."""
        def objective(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2
        
        bounds = [(-10, 10), (-10, 10)]
        result = minimize(
            objective, bounds, seed=42, max_evals=10000,
            max_tolerant_iter=100, progress_bar=False,
        )
        
        assert isinstance(result, Result)
        assert result.success
        np.testing.assert_allclose(result.x, [2.0, 3.0], atol=0.5)
        assert result.fun < 0.25
        assert result.nfev > 0
        assert result.nit > 0
        assert len(result.xv) == result.nfev
        assert len(result.funv) == result.nfev
    
    def test_rosenbrock(self):
        """Test optimization of Rosenbrock function."""
        def rosenbrock(x):
            return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        bounds = [(-5, 5), (-5, 5)]
        result = minimize(
            rosenbrock, bounds, seed=42, max_evals=20000,
            max_tolerant_iter=100, progress_bar=False,
        )
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=0.5)
    
    def test_bounds_respected(self):
        """Test that all evaluations respect bounds."""
        def objective(x):
            return sum(x ** 2)
        
        bounds = [(0, 1), (0, 1)]
        result = minimize(objective, bounds, seed=42, max_evals=1000)
        
        # All evaluated points should be within bounds
        assert np.all(result.xv >= 0)
        assert np.all(result.xv <= 1)
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        def objective(x):
            return sum(x ** 2)
        
        bounds = [(-1, 1), (-1, 1)]
        
        result1 = minimize(objective, bounds, seed=42, max_evals=500)
        result2 = minimize(objective, bounds, seed=42, max_evals=500)
        
        np.testing.assert_array_equal(result1.x, result2.x)
        np.testing.assert_array_equal(result1.xv, result2.xv)
    
    def test_initial_point(self):
        """Test that initial point is used."""
        def objective(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2
        
        bounds = [(-10, 10), (-10, 10)]
        x0 = np.array([[1.9, 2.9]])  # Close to optimum
        
        result = minimize(objective, bounds, x0=x0, seed=42, max_evals=500)
        
        # Should converge quickly since starting near optimum
        np.testing.assert_allclose(result.x, [2.0, 3.0], atol=0.1)
    
    def test_convergence_message(self):
        """Test that convergence message is meaningful."""
        def objective(x):
            return sum(x ** 2)
        
        bounds = [(-1, 1), (-1, 1)]
        result = minimize(objective, bounds, seed=42, max_evals=100)
        
        assert result.message != ""
        assert isinstance(result.message, str)


# =============================================================================
# Tests for objective direction handling
# =============================================================================

class TestObjectiveDirection:
    """Tests for objective direction detection."""
    
    def test_maximize_new_interface(self):
        """Test detection of maximize objective with new interface."""
        obj = MockObjectiveMaximize()
        assert _should_maximize(obj) is True
    
    def test_minimize_new_interface(self):
        """Test detection of minimize objective with new interface."""
        obj = MockObjectiveMinimize()
        assert _should_maximize(obj) is False
    
    def test_maximize_legacy_interface(self):
        """Test detection of maximize objective with legacy interface."""
        obj = MockObjectiveLegacy()
        assert _should_maximize(obj) is True
    
    def test_default_is_maximize(self):
        """Test that default behavior is maximize."""
        class NoDirectionObj:
            name = "test"
            def calculate(self, sim, obs):
                return 0.0
        
        assert _should_maximize(NoDirectionObj()) is True


# =============================================================================
# Tests for SCEUAModelWrapper
# =============================================================================

class TestSCEUAModelWrapper:
    """Tests for the model wrapper class."""
    
    @pytest.fixture
    def wrapper(self):
        """Create a wrapper instance for testing."""
        model = MockModel()
        inputs = pd.DataFrame({'x': np.linspace(0, 10, 100)})
        observed = 2.0 * inputs['x'].values + 1.0  # True: a=2, b=1
        objective = MockObjectiveMaximize()
        
        return SCEUAModelWrapper(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0
        )
    
    def test_bounds_extraction(self, wrapper):
        """Test that bounds are correctly extracted."""
        bounds = wrapper.get_bounds()
        assert bounds == [(0.0, 5.0), (-5.0, 5.0)]
    
    def test_param_conversion(self, wrapper):
        """Test parameter vector/dict conversion."""
        vector = np.array([2.0, 1.0])
        params = wrapper.vector_to_params(vector)
        
        assert params == {'a': 2.0, 'b': 1.0}
        
        # Round-trip
        vector_back = wrapper.params_to_vector(params)
        np.testing.assert_array_equal(vector, vector_back)
    
    def test_call_returns_value(self, wrapper):
        """Test that calling wrapper returns objective value."""
        vector = np.array([2.0, 1.0])  # True parameters
        value = wrapper(vector)
        
        # Since objective is to maximize (neg MSE), and params are perfect,
        # value should be close to 0 (negated for minimization, so also ~0)
        assert isinstance(value, float)
        assert np.abs(value) < 0.01
    
    def test_maximize_negation(self, wrapper):
        """Test that maximization objectives are negated."""
        # Perfect parameters
        perfect = np.array([2.0, 1.0])
        value_perfect = wrapper(perfect)
        
        # Imperfect parameters
        imperfect = np.array([1.0, 0.0])
        value_imperfect = wrapper(imperfect)
        
        # For maximize objective, wrapper negates values for minimization
        # So better (higher) original value -> lower wrapper value
        assert value_perfect < value_imperfect


# =============================================================================
# Tests for run_sceua_direct function
# =============================================================================

class TestRunSCEUADirect:
    """Tests for the main run_sceua_direct function."""
    
    @pytest.fixture
    def calibration_setup(self):
        """Create setup for calibration tests."""
        model = MockModel()
        inputs = pd.DataFrame({'x': np.linspace(0, 10, 100)})
        observed = 2.0 * inputs['x'].values + 1.0  # True: a=2, b=1
        return model, inputs, observed
    
    def test_basic_calibration(self, calibration_setup):
        """Test basic calibration finds correct parameters."""
        model, inputs, observed = calibration_setup
        objective = MockObjectiveMaximize()
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0,
            max_evals=3000,
            seed=42
        )
        
        assert 'best_parameters' in result
        assert 'best_objective' in result
        assert 'all_samples' in result
        assert 'convergence_diagnostics' in result
        assert 'runtime_seconds' in result
        
        # Check parameters are close to true values
        np.testing.assert_allclose(
            result['best_parameters']['a'], 2.0, atol=0.2
        )
        np.testing.assert_allclose(
            result['best_parameters']['b'], 1.0, atol=0.5
        )
    
    def test_all_samples_dataframe(self, calibration_setup):
        """Test that all_samples is properly formatted DataFrame."""
        model, inputs, observed = calibration_setup
        objective = MockObjectiveMaximize()
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0,
            max_evals=500,
            seed=42
        )
        
        samples = result['all_samples']
        assert isinstance(samples, pd.DataFrame)
        assert 'iteration' in samples.columns
        assert 'objective' in samples.columns
        assert 'a' in samples.columns
        assert 'b' in samples.columns
    
    def test_minimize_objective(self, calibration_setup):
        """Test calibration with minimize objective."""
        model, inputs, observed = calibration_setup
        objective = MockObjectiveMinimize()
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0,
            max_evals=3000,
            seed=42
        )
        
        # Best objective should be low (RMSE)
        assert result['best_objective'] < 0.5
        
        # Parameters should still be correct
        np.testing.assert_allclose(
            result['best_parameters']['a'], 2.0, atol=0.2
        )
    
    def test_custom_bounds(self, calibration_setup):
        """Test with custom parameter bounds."""
        model, inputs, observed = calibration_setup
        objective = MockObjectiveMaximize()
        
        custom_bounds = {'a': (1.5, 2.5), 'b': (0.5, 1.5)}
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            parameter_bounds=custom_bounds,
            warmup_period=0,
            max_evals=500,
            seed=42
        )
        
        # Check all samples respect bounds
        samples = result['all_samples']
        assert (samples['a'] >= 1.5).all()
        assert (samples['a'] <= 2.5).all()
        assert (samples['b'] >= 0.5).all()
        assert (samples['b'] <= 1.5).all()
    
    def test_initial_point_dict(self, calibration_setup):
        """Test with initial point as dictionary."""
        model, inputs, observed = calibration_setup
        objective = MockObjectiveMaximize()
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0,
            max_evals=500,
            x0={'a': 1.9, 'b': 0.9},  # Close to optimal
            seed=42
        )
        
        # Should still find good parameters
        assert result['best_parameters']['a'] > 1.5


# =============================================================================
# Tests for calibrate_sceua convenience function
# =============================================================================

class TestCalibrateSCEUA:
    """Tests for the calibrate_sceua convenience function."""
    
    def test_returns_dataclass(self):
        """Test that calibrate_sceua returns SCEUACalibrationResult."""
        model = MockModel()
        inputs = pd.DataFrame({'x': np.linspace(0, 10, 50)})
        observed = 2.0 * inputs['x'].values + 1.0
        objective = MockObjectiveMaximize()
        
        result = calibrate_sceua(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            warmup_period=0,
            max_evals=500,
            seed=42
        )
        
        assert isinstance(result, SCEUACalibrationResult)
        assert isinstance(result.best_parameters, dict)
        assert isinstance(result.best_objective, float)
        assert isinstance(result.all_samples, pd.DataFrame)
        assert isinstance(result.convergence_info, dict)
        assert result.success


# =============================================================================
# Integration tests (if full pyrrm models available)
# =============================================================================

class TestIntegrationWithPyrrm:
    """Integration tests with actual pyrrm models.
    
    These tests are skipped if pyrrm models are not available.
    """
    
    @pytest.fixture
    def sacramento_setup(self):
        """Try to set up Sacramento model with test data."""
        try:
            from pyrrm.models import SacramentoModel
            from pyrrm.calibration import NSE
            
            # Use a simple synthetic dataset for testing
            n_days = 365 * 2
            np.random.seed(42)
            
            # Synthetic inputs
            precipitation = np.random.exponential(5, n_days)
            precipitation[precipitation < 0.5] = 0  # Many dry days
            pet = 3 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Seasonal PET
            
            inputs = pd.DataFrame({
                'precipitation': precipitation,
                'pet': pet
            })
            
            # Create model with known parameters and generate "observed" flow
            model = SacramentoModel()
            model.reset()
            results = model.run(inputs)
            observed = results['flow'].values + np.random.normal(0, 0.5, n_days)
            observed[observed < 0] = 0
            
            return model, inputs, observed, NSE()
            
        except ImportError:
            pytest.skip("pyrrm models not available")
    
    def test_sacramento_calibration(self, sacramento_setup):
        """Test SCE-UA calibration with Sacramento model."""
        model, inputs, observed, objective = sacramento_setup
        
        # Use only a subset of parameters for faster testing
        subset_bounds = {
            'uztwm': (10.0, 200.0),
            'uzfwm': (10.0, 100.0),
            'uzk': (0.1, 0.5),
        }
        
        result = run_sceua_direct(
            model=model,
            inputs=inputs,
            observed=observed,
            objective=objective,
            parameter_bounds=subset_bounds,
            warmup_period=365,
            max_evals=1000,
            seed=42
        )
        
        # Just check that it runs and produces reasonable results
        assert result['convergence_diagnostics']['success']
        assert result['best_objective'] > -1000  # Some reasonable NSE
        assert len(result['all_samples']) > 0
