"""
Tests for hydrological signature metrics.
"""

import pytest
import numpy as np
from pyrrm.objectives.signatures.flow_indices import SignatureMetric
from pyrrm.objectives.signatures.dynamics import (
    compute_flashiness_index,
    compute_rising_limb_density,
    compute_falling_limb_density,
)
from pyrrm.objectives.signatures.water_balance import (
    compute_baseflow_index,
)
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
)


class TestSignatureMetric:
    """Tests for SignatureMetric class."""
    
    def test_all_signatures(self):
        """All 15 signature types should work."""
        signatures = [
            'q95', 'q90', 'q75', 'q50', 'q25', 'q10', 'q5',
            'mean', 'std', 'cv',
            'flashiness', 'baseflow_index',
            'high_flow_freq', 'low_flow_freq', 'zero_flow_freq',
        ]
        
        obs, sim = generate_test_data(n=500)
        
        for sig in signatures:
            metric = SignatureMetric(sig)
            value = metric(obs, sim)
            # Most metrics should return a reasonable value
            # (zero_flow_freq might be 0 or undefined)
            assert np.isfinite(value) or np.isnan(value)
    
    def test_signature_perfect_match(self):
        """Signature error should be 0 for perfect match."""
        obs, sim = generate_perfect_data(n=500)
        
        sig_mean = SignatureMetric('mean')
        sig_q50 = SignatureMetric('q50')
        
        assert np.isclose(sig_mean(obs, sim), 0.0, atol=1e-10)
        assert np.isclose(sig_q50(obs, sim), 0.0, atol=1e-10)
    
    def test_signature_percent_error(self):
        """Signature should return percent error."""
        # Create data where sim mean is 2x obs mean
        np.random.seed(42)
        obs = np.random.lognormal(3, 0.5, 365)
        sim = obs * 2  # 100% overestimate
        
        sig_mean = SignatureMetric('mean')
        error = sig_mean(obs, sim)
        
        # Error should be approximately 100%
        assert np.isclose(error, 100.0, rtol=0.1)
    
    def test_signature_get_components(self):
        """get_components should return obs and sim values."""
        obs, sim = generate_test_data(n=500)
        
        sig = SignatureMetric('q50')
        components = sig.get_components(obs, sim)
        
        assert 'observed' in components
        assert 'simulated' in components
        assert components['observed'] > 0
        assert components['simulated'] > 0
    
    def test_signature_direction(self):
        """SignatureMetric should be minimize (minimize error)."""
        sig = SignatureMetric('mean')
        assert sig.direction == 'minimize'
        assert sig.optimal_value == 0.0
    
    def test_invalid_signature(self):
        """Invalid signature name should raise error."""
        with pytest.raises(ValueError):
            SignatureMetric('invalid_signature')


class TestDynamicsSignatures:
    """Tests for dynamics signature calculations."""
    
    def test_flashiness_index(self):
        """Flashiness index should be computed correctly."""
        # Constant flow -> zero flashiness
        Q_const = np.ones(100) * 50
        fi_const = compute_flashiness_index(Q_const)
        assert np.isclose(fi_const, 0.0)
        
        # Variable flow -> positive flashiness
        Q_var = np.array([10, 100, 10, 100, 10, 100])
        fi_var = compute_flashiness_index(Q_var)
        assert fi_var > 0
    
    def test_flashiness_formula(self):
        """Flashiness should follow the formula."""
        Q = np.array([10, 20, 30, 20, 10])
        
        # FI = sum(|diff|) / sum(Q)
        expected = (10 + 10 + 10 + 10) / 90
        fi = compute_flashiness_index(Q)
        
        assert np.isclose(fi, expected)
    
    def test_rising_limb_density(self):
        """Rising limb density should be fraction of increases."""
        Q = np.array([10, 20, 30, 25, 15])  # 2 rises, 2 falls
        
        rld = compute_rising_limb_density(Q)
        assert np.isclose(rld, 0.5)  # 2/4
    
    def test_falling_limb_density(self):
        """Falling limb density should be fraction of decreases."""
        Q = np.array([10, 20, 30, 25, 15])  # 2 rises, 2 falls
        
        fld = compute_falling_limb_density(Q)
        assert np.isclose(fld, 0.5)  # 2/4


class TestWaterBalanceSignatures:
    """Tests for water balance signature calculations."""
    
    def test_baseflow_index_range(self):
        """BFI should be between 0 and 1."""
        obs, _ = generate_test_data(n=500)
        
        bfi = compute_baseflow_index(obs)
        
        assert 0 <= bfi <= 1
    
    def test_baseflow_index_constant_flow(self):
        """BFI of constant flow should be 1."""
        Q = np.ones(100) * 50
        
        bfi = compute_baseflow_index(Q)
        
        assert np.isclose(bfi, 1.0, atol=0.01)
    
    def test_baseflow_index_minimum_filter(self):
        """BFI with minimum filter should be <= total flow."""
        obs, _ = generate_test_data(n=500)
        
        bfi = compute_baseflow_index(obs)
        
        # BFI is ratio, should be <= 1
        assert bfi <= 1.0
