"""
Tests for KGE family of objective functions.
"""

import pytest
import warnings
import numpy as np
from pyrrm.objectives.metrics.kge import KGE, KGENonParametric
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
    generate_biased_data,
    generate_variable_data,
)


class TestKGE:
    """Tests for Kling-Gupta Efficiency."""
    
    def test_kge_perfect_match(self):
        """KGE should be 1.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        kge = KGE()
        value = kge(obs, sim)
        assert np.isclose(value, 1.0, atol=1e-10)
    
    def test_kge_perfect_components(self):
        """KGE components should all be 1.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        kge = KGE()
        components = kge.get_components(obs, sim)
        
        assert np.isclose(components['r'], 1.0, atol=1e-10)
        assert np.isclose(components['alpha'], 1.0, atol=1e-10)
        assert np.isclose(components['beta'], 1.0, atol=1e-10)
    
    def test_kge_with_bias(self):
        """KGE beta component should reflect bias."""
        obs, sim = generate_biased_data(bias_factor=1.2)
        kge = KGE(variant='2012')
        components = kge.get_components(obs, sim)
        
        # Beta = mu_sim / mu_obs for 2009/2012 variants
        assert np.isclose(components['beta'], 1.2, rtol=0.1)
    
    def test_kge_variants(self):
        """All KGE variants should work."""
        obs, sim = generate_test_data()
        
        for variant in ['2009', '2012', '2021']:
            kge = KGE(variant=variant)
            value = kge(obs, sim)
            assert not np.isnan(value)
            assert f'KGE_{variant}' in kge.name
    
    def test_kge_invalid_variant(self):
        """KGE should raise error for invalid variant."""
        with pytest.raises(ValueError):
            KGE(variant='invalid')
    
    def test_kge_custom_weights(self):
        """KGE with custom weights should work."""
        obs, sim = generate_test_data()
        
        kge_default = KGE(variant='2012')
        kge_weighted = KGE(variant='2012', weights=(2.0, 1.0, 1.0))
        
        val_default = kge_default(obs, sim)
        val_weighted = kge_weighted(obs, sim)
        
        # Different weights should give different results
        assert not np.isclose(val_default, val_weighted)
    
    def test_kge_invalid_weights(self):
        """KGE should raise error for invalid weights."""
        with pytest.raises(ValueError):
            KGE(weights=(1.0, 1.0))  # Wrong length
        with pytest.raises(ValueError):
            KGE(weights=(1.0, -1.0, 1.0))  # Negative weight
    
    def test_kge_log_warning(self):
        """KGE with log transform should emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kge = KGE(transform=FlowTransformation('log'))
            
            assert len(w) == 1
            assert "log transformation" in str(w[0].message).lower()
    
    def test_kge_direction(self):
        """KGE should be a maximize metric."""
        kge = KGE()
        assert kge.direction == 'maximize'
        assert kge.optimal_value == 1.0
    
    def test_kge_with_transform(self):
        """KGE with transformation should work."""
        obs, sim = generate_test_data()
        
        kge = KGE(transform=FlowTransformation('sqrt'))
        value = kge(obs, sim)
        
        assert not np.isnan(value)
        assert 'sqrt' in kge.name


class TestKGENonParametric:
    """Tests for non-parametric KGE."""
    
    def test_kge_np_perfect_match(self):
        """KGE_np should be 1.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        kge_np = KGENonParametric()
        value = kge_np(obs, sim)
        # Allow slight tolerance due to Spearman calculation
        assert np.isclose(value, 1.0, atol=0.01)
    
    def test_kge_np_components(self):
        """KGE_np components should include Spearman correlation."""
        obs, sim = generate_test_data()
        kge_np = KGENonParametric()
        components = kge_np.get_components(obs, sim)
        
        assert 'r_spearman' in components
        assert 'alpha_np' in components
        assert 'beta' in components
    
    def test_kge_np_direction(self):
        """KGE_np should be a maximize metric."""
        kge_np = KGENonParametric()
        assert kge_np.direction == 'maximize'
        assert kge_np.optimal_value == 1.0
    
    def test_kge_np_robust_to_outliers(self):
        """KGE_np should be more robust to outliers than standard KGE."""
        np.random.seed(42)
        n = 365
        
        # Generate normal data
        obs = np.random.lognormal(3, 1, n)
        sim = obs * np.random.normal(1, 0.1, n)
        
        # Add outliers
        sim_outlier = sim.copy()
        sim_outlier[0:5] = sim_outlier[0:5] * 100  # Extreme outliers
        
        kge_std = KGE()
        kge_np = KGENonParametric()
        
        # Standard KGE should be more affected by outliers
        diff_std = abs(kge_std(obs, sim) - kge_std(obs, sim_outlier))
        diff_np = abs(kge_np(obs, sim) - kge_np(obs, sim_outlier))
        
        # Non-parametric should be more robust (smaller difference)
        assert diff_np < diff_std
