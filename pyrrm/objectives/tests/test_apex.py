"""
Tests for APEX v2 objective function.

Tests the novel APEX objective function with its dynamics multiplier,
optional lag multiplier, and SDEB-based core formula.
"""

import pytest
import numpy as np
from pyrrm.objectives.metrics.apex import APEX
from pyrrm.objectives.metrics.traditional import SDEB
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
    generate_with_nan,
    generate_biased_data,
    generate_timing_error_data,
    generate_variable_data,
)


class TestAPEXBasic:
    """Basic tests for APEX objective function."""
    
    def test_apex_perfect_match(self):
        """APEX should be 0.0 for perfect simulation."""
        obs, sim = generate_perfect_data()
        apex = APEX()
        value = apex(obs, sim)
        assert np.isclose(value, 0.0, atol=1e-10)
    
    def test_apex_imperfect(self):
        """APEX should be positive for imperfect simulation."""
        obs, sim = generate_test_data()
        apex = APEX()
        value = apex(obs, sim)
        assert value > 0
    
    def test_apex_direction(self):
        """APEX should be a minimize metric."""
        apex = APEX()
        assert apex.direction == 'minimize'
        assert apex.optimal_value == 0.0
    
    def test_apex_with_nan(self):
        """APEX should handle NaN values via pairwise deletion."""
        obs, sim = generate_with_nan()
        apex = APEX()
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_empty_array(self):
        """APEX should raise error for empty arrays."""
        apex = APEX()
        with pytest.raises(ValueError):
            apex(np.array([]), np.array([]))
    
    def test_apex_mismatched_length(self):
        """APEX should raise error for mismatched lengths."""
        apex = APEX()
        with pytest.raises(ValueError):
            apex(np.array([1, 2, 3]), np.array([1, 2]))
    
    def test_apex_for_calibration(self):
        """for_calibration should return negated value for minimization."""
        obs, sim = generate_test_data()
        apex = APEX()
        
        raw_value = apex(obs, sim)
        calib_value = apex.for_calibration(sim, obs)
        
        # for_calibration negates minimize objectives
        assert np.isclose(calib_value, -raw_value)


class TestAPEXTransformations:
    """Tests for APEX with different flow transformations."""
    
    def test_apex_power_transform(self):
        """APEX with power transform should work."""
        obs, sim = generate_test_data()
        apex = APEX(transform='power', transform_param=0.5)
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_sqrt_transform(self):
        """APEX with sqrt transform should work."""
        obs, sim = generate_test_data()
        apex = APEX(transform='sqrt')
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_log_transform(self):
        """APEX with log transform should work (low flow emphasis)."""
        obs, sim = generate_test_data()
        apex = APEX(transform='log')
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_inverse_transform(self):
        """APEX with inverse transform should work (strong low flow emphasis)."""
        obs, sim = generate_test_data()
        apex = APEX(transform='inverse')
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_no_transform(self):
        """APEX with no transform should work (high flow emphasis)."""
        obs, sim = generate_test_data()
        apex = APEX(transform='none')
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0
    
    def test_apex_custom_flowtransformation(self):
        """APEX with custom FlowTransformation instance should work."""
        obs, sim = generate_test_data()
        transform = FlowTransformation('power', p=0.3)
        apex = APEX(transform=transform)
        value = apex(obs, sim)
        assert not np.isnan(value)
        assert value >= 0


class TestAPEXDynamicsMultiplier:
    """Tests for the novel dynamics multiplier component."""
    
    def test_dynamics_multiplier_perfect(self):
        """Dynamics multiplier should be 1.0 for perfect dynamics match."""
        obs, sim = generate_perfect_data()
        apex = APEX(dynamics_strength=0.5)
        components = apex.get_components(obs, sim)
        
        assert np.isclose(components['dynamics_multiplier'], 1.0, atol=1e-10)
        assert np.isclose(components['gradient_correlation'], 1.0, atol=1e-10)
    
    def test_dynamics_multiplier_penalizes_mismatch(self):
        """Dynamics multiplier should increase for gradient mismatch."""
        obs, sim = generate_test_data()
        apex = APEX(dynamics_strength=0.5)
        components = apex.get_components(obs, sim)
        
        # Gradient correlation should be < 1 for imperfect sim
        assert components['gradient_correlation'] < 1.0
        # Dynamics multiplier should be > 1
        assert components['dynamics_multiplier'] >= 1.0
    
    def test_dynamics_multiplier_disabled(self):
        """Dynamics multiplier should be 1.0 when dynamics_strength=0."""
        obs, sim = generate_test_data()
        apex = APEX(dynamics_strength=0.0)
        components = apex.get_components(obs, sim)
        
        assert np.isclose(components['dynamics_multiplier'], 1.0)
    
    def test_dynamics_multiplier_strength_effect(self):
        """Higher dynamics_strength should increase penalty."""
        obs, sim = generate_test_data()
        
        apex_low = APEX(dynamics_strength=0.2)
        apex_high = APEX(dynamics_strength=0.8)
        
        comp_low = apex_low.get_components(obs, sim)
        comp_high = apex_high.get_components(obs, sim)
        
        # Same gradient correlation
        assert np.isclose(
            comp_low['gradient_correlation'], 
            comp_high['gradient_correlation']
        )
        
        # Higher strength = higher multiplier (when correlation < 1)
        if comp_low['gradient_correlation'] < 1.0:
            assert comp_high['dynamics_multiplier'] > comp_low['dynamics_multiplier']


class TestAPEXLagMultiplier:
    """Tests for the novel lag multiplier component."""
    
    def test_lag_multiplier_disabled_by_default(self):
        """Lag multiplier should be 1.0 when lag_penalty=False."""
        obs, sim = generate_test_data()
        apex = APEX(lag_penalty=False)
        components = apex.get_components(obs, sim)
        
        assert np.isclose(components['lag_multiplier'], 1.0)
        assert components['optimal_lag'] == 0
    
    def test_lag_multiplier_perfect_timing(self):
        """Lag multiplier should be 1.0 for perfect timing."""
        obs, sim = generate_perfect_data()
        apex = APEX(lag_penalty=True)
        components = apex.get_components(obs, sim)
        
        assert np.isclose(components['lag_multiplier'], 1.0, atol=0.01)
        assert components['optimal_lag'] == 0
    
    def test_lag_multiplier_detects_shift(self):
        """Lag multiplier should detect timing offset."""
        obs, sim = generate_timing_error_data(shift=3)
        apex = APEX(lag_penalty=True, lag_reference=5)
        components = apex.get_components(obs, sim)
        
        # Should detect lag near 3
        assert abs(components['optimal_lag']) > 0
        # Multiplier should be > 1
        assert components['lag_multiplier'] > 1.0
    
    def test_lag_multiplier_increases_value(self):
        """Lag penalty should increase APEX value for shifted data."""
        obs, sim = generate_timing_error_data(shift=3)
        
        apex_no_lag = APEX(lag_penalty=False)
        apex_with_lag = APEX(lag_penalty=True, lag_strength=0.5)
        
        value_no_lag = apex_no_lag(obs, sim)
        value_with_lag = apex_with_lag(obs, sim)
        
        # With lag penalty should give higher (worse) value
        assert value_with_lag > value_no_lag


class TestAPEXRegimeEmphasis:
    """Tests for regime emphasis weighting in ranked term."""
    
    def test_regime_emphasis_uniform(self):
        """Uniform regime emphasis should work."""
        obs, sim = generate_test_data()
        apex = APEX(regime_emphasis='uniform')
        value = apex(obs, sim)
        assert not np.isnan(value)
    
    def test_regime_emphasis_low_flow(self):
        """Low flow regime emphasis should work."""
        obs, sim = generate_test_data()
        apex = APEX(regime_emphasis='low_flow')
        value = apex(obs, sim)
        assert not np.isnan(value)
    
    def test_regime_emphasis_high_flow(self):
        """High flow regime emphasis should work."""
        obs, sim = generate_test_data()
        apex = APEX(regime_emphasis='high_flow')
        value = apex(obs, sim)
        assert not np.isnan(value)
    
    def test_regime_emphasis_balanced(self):
        """Balanced regime emphasis should work."""
        obs, sim = generate_test_data()
        apex = APEX(regime_emphasis='balanced')
        value = apex(obs, sim)
        assert not np.isnan(value)
    
    def test_regime_emphasis_extremes(self):
        """Extremes regime emphasis should work."""
        obs, sim = generate_test_data()
        apex = APEX(regime_emphasis='extremes')
        value = apex(obs, sim)
        assert not np.isnan(value)
    
    def test_invalid_regime_emphasis(self):
        """Invalid regime emphasis should raise error."""
        with pytest.raises(ValueError, match="Unknown regime_emphasis"):
            APEX(regime_emphasis='invalid')


class TestAPEXBiasMultiplier:
    """Tests for bias multiplier component."""
    
    def test_bias_multiplier_no_bias(self):
        """Bias multiplier should be 1.0 for unbiased simulation."""
        obs, sim = generate_perfect_data()
        apex = APEX()
        components = apex.get_components(obs, sim)
        
        assert np.isclose(components['bias_multiplier'], 1.0)
        assert np.isclose(components['relative_bias'], 0.0)
    
    def test_bias_multiplier_with_bias(self):
        """Bias multiplier should increase with bias."""
        obs, sim = generate_biased_data(bias_factor=1.2)  # 20% overestimation
        apex = APEX(bias_strength=1.0)
        components = apex.get_components(obs, sim)
        
        # Should detect ~20% positive bias
        assert components['relative_bias'] > 0.15
        # Multiplier should be > 1
        assert components['bias_multiplier'] > 1.0
    
    def test_bias_strength_effect(self):
        """Higher bias_strength should increase penalty."""
        obs, sim = generate_biased_data(bias_factor=1.2)
        
        apex_low = APEX(bias_strength=0.5)
        apex_high = APEX(bias_strength=2.0)
        
        comp_low = apex_low.get_components(obs, sim)
        comp_high = apex_high.get_components(obs, sim)
        
        # Higher strength = higher multiplier
        assert comp_high['bias_multiplier'] > comp_low['bias_multiplier']
    
    def test_bias_power_effect(self):
        """Higher bias_power should change penalty shape."""
        obs, sim = generate_biased_data(bias_factor=1.2)
        
        apex_linear = APEX(bias_power=1.0)
        apex_quadratic = APEX(bias_power=2.0)
        
        comp_linear = apex_linear.get_components(obs, sim)
        comp_quadratic = apex_quadratic.get_components(obs, sim)
        
        # Both should have same relative bias
        assert np.isclose(comp_linear['relative_bias'], comp_quadratic['relative_bias'])
        
        # Different multipliers due to different power
        # For bias ~0.2: linear penalty = 1 + 0.2 = 1.2
        # For bias ~0.2: quadratic penalty = 1 + 0.04 = 1.04
        assert comp_linear['bias_multiplier'] > comp_quadratic['bias_multiplier']


class TestAPEXComponents:
    """Tests for get_components method."""
    
    def test_components_keys(self):
        """get_components should return all expected keys."""
        obs, sim = generate_test_data()
        apex = APEX(lag_penalty=True)
        components = apex.get_components(obs, sim)
        
        expected_keys = {
            'chronological_term',
            'ranked_term',
            'weighted_error',
            'bias_multiplier',
            'relative_bias',
            'dynamics_multiplier',
            'gradient_correlation',
            'lag_multiplier',
            'optimal_lag',
            'apex_value',
        }
        
        assert set(components.keys()) == expected_keys
    
    def test_components_consistency(self):
        """Component values should be consistent with final value."""
        obs, sim = generate_test_data()
        apex = APEX(lag_penalty=True)
        
        value = apex(obs, sim)
        components = apex.get_components(obs, sim)
        
        # apex_value should equal direct calculation
        assert np.isclose(value, components['apex_value'])
        
        # Verify formula: weighted_error * bias * dynamics * lag
        expected = (
            components['weighted_error'] *
            components['bias_multiplier'] *
            components['dynamics_multiplier'] *
            components['lag_multiplier']
        )
        assert np.isclose(value, expected)


class TestAPEXSDEBEquivalence:
    """Tests for SDEB equivalence when dynamics_strength=0."""
    
    def test_sdeb_equivalence_structure(self):
        """APEX with dynamics_strength=0 should match SDEB structure."""
        obs, sim = generate_test_data()
        
        # APEX with dynamics disabled
        apex = APEX(
            alpha=0.1,
            transform='power',
            transform_param=0.5,
            dynamics_strength=0.0,
            lag_penalty=False
        )
        
        # SDEB with same parameters
        sdeb = SDEB(alpha=0.1, lam=0.5)
        
        apex_value = apex(obs, sim)
        sdeb_value = sdeb(obs, sim)
        
        # Should be close (may not be exactly equal due to epsilon handling)
        # Allow 5% tolerance for numerical differences
        assert np.isclose(apex_value, sdeb_value, rtol=0.05)
    
    def test_dynamics_adds_penalty(self):
        """Enabling dynamics should increase APEX value for imperfect sim."""
        obs, sim = generate_test_data()
        
        apex_sdeb_equiv = APEX(dynamics_strength=0.0)
        apex_with_dynamics = APEX(dynamics_strength=0.5)
        
        value_equiv = apex_sdeb_equiv(obs, sim)
        value_dynamics = apex_with_dynamics(obs, sim)
        
        # Dynamics multiplier >= 1, so value with dynamics >= value without
        assert value_dynamics >= value_equiv


class TestAPEXParameterValidation:
    """Tests for parameter validation."""
    
    def test_invalid_alpha(self):
        """Alpha outside [0, 1] should raise error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            APEX(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            APEX(alpha=1.5)
    
    def test_invalid_transform_param(self):
        """Non-positive transform_param should raise error."""
        with pytest.raises(ValueError, match="transform_param must be positive"):
            APEX(transform_param=0)
        with pytest.raises(ValueError, match="transform_param must be positive"):
            APEX(transform_param=-0.5)
    
    def test_invalid_bias_strength(self):
        """Negative bias_strength should raise error."""
        with pytest.raises(ValueError, match="bias_strength must be non-negative"):
            APEX(bias_strength=-0.5)
    
    def test_invalid_bias_power(self):
        """Non-positive bias_power should raise error."""
        with pytest.raises(ValueError, match="bias_power must be positive"):
            APEX(bias_power=0)
    
    def test_invalid_dynamics_strength(self):
        """Negative dynamics_strength should raise error."""
        with pytest.raises(ValueError, match="dynamics_strength must be non-negative"):
            APEX(dynamics_strength=-0.5)
    
    def test_invalid_lag_strength(self):
        """Negative lag_strength should raise error."""
        with pytest.raises(ValueError, match="lag_strength must be non-negative"):
            APEX(lag_strength=-0.5)
    
    def test_invalid_lag_reference(self):
        """Non-positive lag_reference should raise error."""
        with pytest.raises(ValueError, match="lag_reference must be positive"):
            APEX(lag_reference=0)
    
    def test_invalid_transform(self):
        """Invalid transform string should raise error."""
        with pytest.raises(ValueError, match="Unknown transform"):
            APEX(transform='invalid_transform')


class TestAPEXRepr:
    """Tests for string representation."""
    
    def test_repr_basic(self):
        """Basic repr should include key parameters."""
        apex = APEX()
        repr_str = repr(apex)
        
        assert 'APEX' in repr_str
        assert 'alpha=' in repr_str
        assert 'transform=' in repr_str
        assert 'dynamics_strength=' in repr_str
    
    def test_repr_with_lag(self):
        """Repr should indicate lag_penalty status."""
        apex_no_lag = APEX(lag_penalty=False)
        apex_with_lag = APEX(lag_penalty=True)
        
        assert 'lag_penalty=False' in repr(apex_no_lag)
        assert 'lag_penalty=True' in repr(apex_with_lag)
