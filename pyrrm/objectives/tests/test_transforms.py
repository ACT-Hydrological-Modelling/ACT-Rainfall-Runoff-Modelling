"""
Tests for flow transformation classes.
"""

import pytest
import numpy as np
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
from pyrrm.objectives.core.constants import TRANSFORM_EMPHASIS


class TestFlowTransformation:
    """Tests for FlowTransformation class."""
    
    def test_all_transform_types(self):
        """All 8 transform types should work."""
        transform_types = ['none', 'sqrt', 'log', 'inverse', 
                          'squared', 'inverse_squared', 'power', 'boxcox']
        
        Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        for t_type in transform_types:
            transform = FlowTransformation(t_type)
            result = transform.apply(Q, Q)
            
            assert not np.any(np.isnan(result))
            assert len(result) == len(Q)
    
    def test_transform_emphasis(self):
        """Flow emphasis should be correct for each transform."""
        for t_type, expected_emphasis in TRANSFORM_EMPHASIS.items():
            transform = FlowTransformation(t_type)
            assert transform.flow_emphasis == expected_emphasis
    
    def test_no_transform(self):
        """'none' transform should return original values."""
        Q = np.array([10.0, 20.0, 30.0])
        transform = FlowTransformation('none')
        result = transform.apply(Q, Q)
        
        assert np.allclose(result, Q)
    
    def test_sqrt_transform(self):
        """sqrt transform should apply correctly."""
        Q = np.array([0.0, 4.0, 16.0, 25.0])
        transform = FlowTransformation('sqrt', epsilon_method='fixed', epsilon_value=0.0001)
        result = transform.apply(Q, Q)
        
        # sqrt(4) = 2, sqrt(16) = 4, sqrt(25) = 5
        assert np.isclose(result[1], 2.0, atol=0.01)
        assert np.isclose(result[2], 4.0, atol=0.01)
        assert np.isclose(result[3], 5.0, atol=0.01)
    
    def test_epsilon_methods(self):
        """All 3 epsilon methods should work."""
        Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        for eps_method in ['mean_fraction', 'fixed', 'min_nonzero']:
            transform = FlowTransformation('log', epsilon_method=eps_method)
            result = transform.apply(Q, Q)
            assert not np.any(np.isnan(result))
    
    def test_epsilon_mean_fraction(self):
        """mean_fraction epsilon should be based on mean."""
        Q = np.array([100.0, 200.0, 300.0])
        transform = FlowTransformation('log', 
                                       epsilon_method='mean_fraction', 
                                       epsilon_value=0.01)
        
        eps = transform.get_epsilon(Q)
        expected = np.mean(Q) * 0.01  # 200 * 0.01 = 2.0
        
        assert np.isclose(eps, expected)
    
    def test_epsilon_fixed(self):
        """fixed epsilon should be the specified value."""
        Q = np.array([100.0, 200.0, 300.0])
        transform = FlowTransformation('log', 
                                       epsilon_method='fixed', 
                                       epsilon_value=5.0)
        
        eps = transform.get_epsilon(Q)
        assert eps == 5.0
    
    def test_epsilon_min_nonzero(self):
        """min_nonzero epsilon should be based on minimum positive value."""
        Q = np.array([0.0, 10.0, 20.0, 30.0])
        transform = FlowTransformation('log', 
                                       epsilon_method='min_nonzero', 
                                       epsilon_value=0.1)
        
        eps = transform.get_epsilon(Q)
        expected = 10.0 * 0.1  # min nonzero = 10, * 0.1 = 1.0
        
        assert np.isclose(eps, expected)
    
    def test_power_transform(self):
        """power transform with custom exponent should work."""
        Q = np.array([1.0, 8.0, 27.0])  # 1^(1/3), 8^(1/3), 27^(1/3)
        transform = FlowTransformation('power', p=1/3)
        result = transform.apply(Q, Q)
        
        # With epsilon ~0, should be approximately 1, 2, 3
        assert result[0] < result[1] < result[2]
    
    def test_boxcox_transform(self):
        """boxcox transform should work."""
        Q = np.array([10.0, 20.0, 30.0])
        transform = FlowTransformation('boxcox', lam=0.25)
        result = transform.apply(Q, Q)
        
        assert not np.any(np.isnan(result))
        assert len(result) == len(Q)
    
    def test_invalid_transform_type(self):
        """Invalid transform type should raise error."""
        with pytest.raises(ValueError):
            FlowTransformation('invalid_type')
    
    def test_invalid_epsilon_method(self):
        """Invalid epsilon method should raise error."""
        with pytest.raises(ValueError):
            FlowTransformation('sqrt', epsilon_method='invalid')
    
    def test_invalid_epsilon_value(self):
        """Non-positive epsilon should raise error."""
        with pytest.raises(ValueError):
            FlowTransformation('sqrt', epsilon_value=0)
        with pytest.raises(ValueError):
            FlowTransformation('sqrt', epsilon_value=-1)
    
    def test_transform_equality(self):
        """Transforms with same parameters should be equal."""
        t1 = FlowTransformation('sqrt', epsilon_method='fixed', epsilon_value=0.01)
        t2 = FlowTransformation('sqrt', epsilon_method='fixed', epsilon_value=0.01)
        t3 = FlowTransformation('log', epsilon_method='fixed', epsilon_value=0.01)
        
        assert t1 == t2
        assert t1 != t3
    
    def test_transform_repr(self):
        """repr should include transform type and params."""
        t1 = FlowTransformation('sqrt')
        t2 = FlowTransformation('power', p=0.3)
        
        assert 'sqrt' in repr(t1)
        assert 'power' in repr(t2)
        assert 'p=0.3' in repr(t2)
    
    def test_transform_hashable(self):
        """Transforms should be hashable for use as dict keys."""
        t1 = FlowTransformation('sqrt')
        t2 = FlowTransformation('log')
        
        d = {t1: 'sqrt_value', t2: 'log_value'}
        assert d[t1] == 'sqrt_value'
        assert d[t2] == 'log_value'
