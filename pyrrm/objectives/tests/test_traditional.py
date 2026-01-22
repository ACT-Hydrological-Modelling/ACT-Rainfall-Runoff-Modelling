"""
Tests for traditional objective functions (NSE, RMSE, MAE, PBIAS, SDEB).
"""

import pytest
import numpy as np
from pyrrm.objectives.metrics.traditional import NSE, RMSE, MAE, PBIAS, SDEB
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
    generate_with_nan,
    generate_biased_data,
)


class TestNSE:
    """Tests for Nash-Sutcliffe Efficiency."""
    
    def test_nse_perfect_match(self):
        """NSE should be 1.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        nse = NSE()
        value = nse(obs, sim)
        assert np.isclose(value, 1.0, atol=1e-10)
    
    def test_nse_imperfect(self):
        """NSE should be less than 1.0 for imperfect simulation."""
        obs, sim = generate_test_data()
        nse = NSE()
        value = nse(obs, sim)
        assert value < 1.0
        assert value > -1.0  # Should still be reasonable
    
    def test_nse_with_nan(self):
        """NSE should handle NaN values."""
        obs, sim = generate_with_nan()
        nse = NSE()
        value = nse(obs, sim)
        assert not np.isnan(value)
    
    def test_nse_with_transform(self):
        """NSE with transformation should work."""
        obs, sim = generate_test_data()
        nse = NSE(transform=FlowTransformation('sqrt'))
        value = nse(obs, sim)
        assert not np.isnan(value)
        assert 'sqrt' in nse.name
    
    def test_nse_mean_simulation(self):
        """NSE of mean simulation should be approximately 0."""
        obs, _ = generate_test_data()
        sim = np.full_like(obs, np.mean(obs))
        nse = NSE()
        value = nse(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_nse_empty_array(self):
        """NSE should raise error for empty arrays."""
        nse = NSE()
        with pytest.raises(ValueError):
            nse(np.array([]), np.array([]))
    
    def test_nse_mismatched_length(self):
        """NSE should raise error for mismatched lengths."""
        nse = NSE()
        with pytest.raises(ValueError):
            nse(np.array([1, 2, 3]), np.array([1, 2]))


class TestRMSE:
    """Tests for Root Mean Square Error."""
    
    def test_rmse_zero_error(self):
        """RMSE should be 0.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        rmse = RMSE()
        value = rmse(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_rmse_direction(self):
        """RMSE should be a minimize metric."""
        rmse = RMSE()
        assert rmse.direction == 'minimize'
        assert rmse.optimal_value == 0.0
    
    def test_rmse_normalized(self):
        """NRMSE should be normalized by mean."""
        obs, sim = generate_test_data()
        rmse = RMSE()
        nrmse = RMSE(normalized=True)
        
        rmse_val = rmse(obs, sim)
        nrmse_val = nrmse(obs, sim)
        
        # NRMSE = RMSE / mean(obs)
        expected_nrmse = rmse_val / np.mean(obs[~np.isnan(obs)])
        assert np.isclose(nrmse_val, expected_nrmse, rtol=0.01)


class TestMAE:
    """Tests for Mean Absolute Error."""
    
    def test_mae_zero_error(self):
        """MAE should be 0.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        mae = MAE()
        value = mae(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_mae_direction(self):
        """MAE should be a minimize metric."""
        mae = MAE()
        assert mae.direction == 'minimize'


class TestPBIAS:
    """Tests for Percent Bias."""
    
    def test_pbias_zero(self):
        """PBIAS should be 0.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        pbias = PBIAS()
        value = pbias(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_pbias_overestimate(self):
        """PBIAS should be positive for overestimation."""
        obs, sim = generate_biased_data(bias_factor=1.2)
        pbias = PBIAS()
        value = pbias(obs, sim)
        assert value > 0  # Overestimate -> positive bias
    
    def test_pbias_underestimate(self):
        """PBIAS should be negative for underestimation."""
        obs, sim = generate_biased_data(bias_factor=0.8)
        pbias = PBIAS()
        value = pbias(obs, sim)
        assert value < 0  # Underestimate -> negative bias


class TestSDEB:
    """Tests for SDEB metric."""
    
    def test_sdeb_perfect_match(self):
        """SDEB should be 0.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        sdeb = SDEB()
        value = sdeb(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_sdeb_direction(self):
        """SDEB should be a minimize metric."""
        sdeb = SDEB()
        assert sdeb.direction == 'minimize'
        assert sdeb.optimal_value == 0.0
    
    def test_sdeb_components(self):
        """SDEB components should be calculated correctly."""
        obs, sim = generate_test_data()
        sdeb = SDEB()
        components = sdeb.get_components(obs, sim)
        
        assert 'chronological_term' in components
        assert 'ranked_term' in components
        assert 'relative_bias' in components
        assert 'bias_penalty' in components
        
        # Verify components are non-negative
        assert components['chronological_term'] >= 0
        assert components['ranked_term'] >= 0
        assert components['relative_bias'] >= 0
        assert components['bias_penalty'] >= 1  # Minimum is 1
    
    def test_sdeb_custom_alpha(self):
        """SDEB with custom alpha should work."""
        obs, sim = generate_test_data()
        
        sdeb_default = SDEB()
        sdeb_custom = SDEB(alpha=0.5)
        
        val_default = sdeb_default(obs, sim)
        val_custom = sdeb_custom(obs, sim)
        
        # Different alpha should give different results
        assert not np.isclose(val_default, val_custom)
    
    def test_sdeb_invalid_alpha(self):
        """SDEB should raise error for invalid alpha."""
        with pytest.raises(ValueError):
            SDEB(alpha=-0.1)
        with pytest.raises(ValueError):
            SDEB(alpha=1.5)
