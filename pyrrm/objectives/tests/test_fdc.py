"""
Tests for Flow Duration Curve metrics.
"""

import pytest
import numpy as np
from pyrrm.objectives.fdc.curves import (
    compute_fdc,
    compute_fdc_at_exceedance,
    compute_fdc_slope,
    get_fdc_segment,
)
from pyrrm.objectives.fdc.metrics import FDCMetric
from pyrrm.objectives.core.constants import FDC_SEGMENTS
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
)


class TestFDCCurves:
    """Tests for FDC computation utilities."""
    
    def test_compute_fdc(self):
        """FDC computation should return sorted values."""
        Q = np.array([10, 30, 50, 20, 40])
        exc, flows = compute_fdc(Q)
        
        # Flows should be sorted descending
        assert np.all(np.diff(flows) <= 0)
        
        # Exceedance should be increasing
        assert np.all(np.diff(exc) > 0)
        
        # Exceedance should be between 0 and 1
        assert exc[0] > 0
        assert exc[-1] < 1
    
    def test_compute_fdc_weibull(self):
        """FDC should use Weibull plotting position."""
        n = 100
        Q = np.arange(1, n + 1)
        exc, flows = compute_fdc(Q)
        
        # Weibull: p_i = i / (n + 1)
        expected_exc = np.arange(1, n + 1) / (n + 1)
        assert np.allclose(exc, expected_exc)
    
    def test_compute_fdc_handles_nan(self):
        """FDC should handle NaN values."""
        Q = np.array([10, 20, np.nan, 40, 50])
        exc, flows = compute_fdc(Q)
        
        # Should have 4 valid values
        assert len(exc) == 4
        assert len(flows) == 4
        assert not np.any(np.isnan(flows))
    
    def test_compute_fdc_at_exceedance(self):
        """FDC interpolation should work correctly."""
        Q = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # Get Q at 50% exceedance
        Q50 = compute_fdc_at_exceedance(Q, np.array([0.5]))
        
        # Should be approximately median (55)
        assert 40 < Q50[0] < 70
    
    def test_get_fdc_segment(self):
        """FDC segment extraction should work."""
        Q = np.random.lognormal(3, 1, 1000)
        
        exc_seg, flow_seg = get_fdc_segment(Q, 0.2, 0.7)
        
        # All exceedance values should be in range
        assert np.all(exc_seg >= 0.2)
        assert np.all(exc_seg <= 0.7)


class TestFDCMetric:
    """Tests for FDCMetric class."""
    
    def test_fdc_segments(self):
        """All predefined segments should work."""
        obs, sim = generate_test_data(n=500)
        
        for segment in FDC_SEGMENTS.keys():
            fdc = FDCMetric(segment=segment)
            value = fdc(obs, sim)
            assert not np.isnan(value)
    
    def test_fdc_segment_bounds(self):
        """Segment bounds should be correct."""
        fdc = FDCMetric(segment='high')
        assert fdc.bounds == FDC_SEGMENTS['high']
        
        fdc = FDCMetric(segment='low')
        assert fdc.bounds == FDC_SEGMENTS['low']
    
    def test_fdc_metric_types(self):
        """All metric types should work."""
        obs, sim = generate_test_data(n=500)
        
        for metric in ['volume_bias', 'slope', 'rmse', 'correlation']:
            fdc = FDCMetric(segment='all', metric=metric)
            value = fdc(obs, sim)
            assert not np.isnan(value)
    
    def test_fdc_custom_bounds(self):
        """Custom bounds should override segment."""
        obs, sim = generate_test_data(n=500)
        
        fdc = FDCMetric(custom_bounds=(0.1, 0.9))
        value = fdc(obs, sim)
        assert not np.isnan(value)
        assert fdc.bounds == (0.1, 0.9)
    
    def test_fdc_invalid_bounds(self):
        """Invalid bounds should raise error."""
        with pytest.raises(ValueError):
            FDCMetric(custom_bounds=(0.9, 0.1))  # Lower > upper
        with pytest.raises(ValueError):
            FDCMetric(custom_bounds=(-0.1, 0.5))  # Negative
        with pytest.raises(ValueError):
            FDCMetric(custom_bounds=(0.5, 1.5))  # > 1
    
    def test_fdc_log_transform(self):
        """Log transform should be applied."""
        obs, sim = generate_test_data(n=500)
        
        fdc_normal = FDCMetric(segment='low')
        fdc_log = FDCMetric(segment='low', log_transform=True)
        
        val_normal = fdc_normal(obs, sim)
        val_log = fdc_log(obs, sim)
        
        # Log transform should give different result
        assert val_normal != val_log
        assert 'log' in fdc_log.name
    
    def test_fdc_perfect_match(self):
        """FDC metrics should show good results for perfect match."""
        obs, sim = generate_perfect_data(n=500)
        
        fdc_bias = FDCMetric(segment='all', metric='volume_bias')
        fdc_rmse = FDCMetric(segment='all', metric='rmse')
        fdc_corr = FDCMetric(segment='all', metric='correlation')
        
        assert np.isclose(fdc_bias(obs, sim), 0.0, atol=0.01)
        assert np.isclose(fdc_rmse(obs, sim), 0.0, atol=0.01)
        assert np.isclose(fdc_corr(obs, sim), 1.0, atol=0.01)
    
    def test_fdc_direction(self):
        """FDC metrics should have correct direction."""
        fdc_bias = FDCMetric(metric='volume_bias')
        fdc_corr = FDCMetric(metric='correlation')
        
        assert fdc_bias.direction == 'minimize'
        assert fdc_corr.direction == 'maximize'
    
    def test_fdc_get_components(self):
        """FDC get_components should return segment statistics."""
        obs, sim = generate_test_data(n=500)
        
        fdc = FDCMetric(segment='mid')
        components = fdc.get_components(obs, sim)
        
        assert 'obs_segment_mean' in components
        assert 'sim_segment_mean' in components
        assert 'segment_n' in components
