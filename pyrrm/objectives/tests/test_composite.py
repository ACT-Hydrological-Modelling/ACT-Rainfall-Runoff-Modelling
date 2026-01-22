"""
Tests for composite objective functions.
"""

import pytest
import numpy as np
from pyrrm.objectives.composite.weighted import WeightedObjective
from pyrrm.objectives.composite.factories import (
    kge_hilo,
    fdc_multisegment,
    comprehensive_objective,
    nse_multiscale,
)
from pyrrm.objectives.metrics.traditional import NSE, RMSE, PBIAS
from pyrrm.objectives.metrics.kge import KGE
from pyrrm.objectives.tests.fixtures import (
    generate_test_data,
    generate_perfect_data,
)


class TestWeightedObjective:
    """Tests for WeightedObjective class."""
    
    def test_weighted_sum(self):
        """Weighted sum aggregation should work."""
        obs, sim = generate_test_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (KGE(), 0.5),
        ], aggregation='weighted_sum')
        
        value = combined(obs, sim)
        assert not np.isnan(value)
    
    def test_weighted_product(self):
        """Weighted product aggregation should work."""
        obs, sim = generate_test_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (KGE(), 0.5),
        ], aggregation='weighted_product')
        
        value = combined(obs, sim)
        assert not np.isnan(value)
    
    def test_weighted_min(self):
        """Min aggregation should return minimum value."""
        obs, sim = generate_perfect_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (KGE(), 0.5),
        ], aggregation='min', normalize=False)
        
        value = combined(obs, sim)
        # Both should be 1.0, so min should be ~1.0
        assert value > 0.9
    
    def test_weight_normalization(self):
        """Weights should be normalized to sum to 1."""
        combined = WeightedObjective([
            (NSE(), 2.0),
            (KGE(), 3.0),
        ])
        
        # Internal weights should sum to 1
        assert np.isclose(sum(combined._normalized_weights), 1.0)
    
    def test_get_components(self):
        """get_components should return individual values."""
        obs, sim = generate_test_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (KGE(), 0.5),
        ])
        
        components = combined.get_components(obs, sim)
        
        assert 'NSE' in components
        assert 'KGE_2012' in components or 'KGE' in components
    
    def test_direction(self):
        """Composite objectives should always maximize."""
        combined = WeightedObjective([
            (NSE(), 0.5),
            (RMSE(), 0.5),  # minimize metric
        ])
        
        assert combined.direction == 'maximize'
    
    def test_normalization_minmax(self):
        """minmax normalization should scale values to [0, 1]."""
        obs, sim = generate_test_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (PBIAS(), 0.5),
        ], normalize=True, normalize_method='minmax')
        
        value = combined(obs, sim)
        # Value should be positive (normalized to maximize)
        assert value > 0
    
    def test_empty_objectives(self):
        """Empty objectives list should raise error."""
        with pytest.raises(ValueError):
            WeightedObjective([])
    
    def test_negative_weights(self):
        """Negative weights should raise error."""
        with pytest.raises(ValueError):
            WeightedObjective([
                (NSE(), 0.5),
                (KGE(), -0.5),
            ])
    
    def test_evaluate_individual(self):
        """evaluate_individual should return detailed results."""
        obs, sim = generate_test_data()
        
        combined = WeightedObjective([
            (NSE(), 0.5),
            (KGE(), 0.5),
        ])
        
        results = combined.evaluate_individual(obs, sim)
        
        assert 'raw_values' in results
        assert 'normalized_values' in results
        assert 'weights' in results
        assert 'aggregated_value' in results


class TestFactoryFunctions:
    """Tests for composite objective factory functions."""
    
    def test_kge_hilo(self):
        """kge_hilo should create KGE + KGE(inverse) combination."""
        obj = kge_hilo()
        
        assert isinstance(obj, WeightedObjective)
        
        # Should have 2 objectives
        assert len(obj.objectives) == 2
        
        # Test evaluation
        obs, sim = generate_test_data()
        value = obj(obs, sim)
        assert not np.isnan(value)
    
    def test_kge_hilo_weights(self):
        """kge_hilo custom weights should work."""
        obj = kge_hilo(kge_weight=0.7)
        
        # First objective should have 70% weight
        assert obj.objectives[0][1] == 0.7
        assert obj.objectives[1][1] == 0.3
    
    def test_fdc_multisegment(self):
        """fdc_multisegment should create multi-segment FDC objective."""
        obj = fdc_multisegment()
        
        assert isinstance(obj, WeightedObjective)
        
        # Default should have 3 segments
        assert len(obj.objectives) == 3
        
        # Test evaluation
        obs, sim = generate_test_data(n=500)
        value = obj(obs, sim)
        assert not np.isnan(value)
    
    def test_fdc_multisegment_custom(self):
        """fdc_multisegment with custom segments should work."""
        obj = fdc_multisegment(
            segments=['peak', 'high', 'mid', 'low'],
            weights=[0.1, 0.3, 0.3, 0.3]
        )
        
        assert len(obj.objectives) == 4
    
    def test_comprehensive_objective(self):
        """comprehensive_objective should create multi-metric combination."""
        obj = comprehensive_objective()
        
        assert isinstance(obj, WeightedObjective)
        
        # Should have 5 objectives (KGE, KGE_inv, PBIAS, FDC_high, FDC_low)
        assert len(obj.objectives) == 5
        
        # Test evaluation
        obs, sim = generate_test_data(n=500)
        value = obj(obs, sim)
        assert not np.isnan(value)
    
    def test_nse_multiscale(self):
        """nse_multiscale should create multi-transform NSE combination."""
        obj = nse_multiscale()
        
        assert isinstance(obj, WeightedObjective)
        
        # Should have 3 objectives (NSE, NSE_log, NSE_sqrt)
        assert len(obj.objectives) == 3
        
        # Test evaluation
        obs, sim = generate_test_data()
        value = obj(obs, sim)
        assert not np.isnan(value)
    
    def test_factory_perfect_match(self):
        """Factory objectives should give good results for perfect data."""
        obs, sim = generate_perfect_data(n=500)
        
        kge_hi = kge_hilo()
        fdc_multi = fdc_multisegment()
        comprehensive = comprehensive_objective()
        
        # All should give high values for perfect match
        # (after normalization to maximize)
        assert kge_hi(obs, sim) > 0.9
