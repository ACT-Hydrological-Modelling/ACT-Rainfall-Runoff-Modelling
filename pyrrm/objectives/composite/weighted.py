"""
Weighted composite objective functions.

This module provides the WeightedObjective class for combining multiple
objective functions with configurable weights and aggregation methods.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction


class WeightedObjective(ObjectiveFunction):
    """
    Combine multiple objective functions with configurable weights.
    
    Parameters
    ----------
    objectives : list of (ObjectiveFunction, float) tuples
        List of objective functions and their weights.
        Weights will be normalized to sum to 1.
    
    aggregation : str, default='weighted_sum'
        Aggregation method:
        - 'weighted_sum': Σ(wi × fi)
        - 'weighted_product': Π(fi^wi)
        - 'min': min(fi)
    
    normalize : bool, default=True
        Normalize objective values to [0, 1] before combining.
        Required when combining metrics with different scales.
    
    normalize_method : str, default='minmax'
        Normalization method:
        - 'minmax': Scale to [0, 1] based on assumed ranges
        - 'direction': Just flip sign for minimize objectives
    
    Notes
    -----
    Weight normalization:
        Weights are normalized to sum to 1: wi' = wi / Σwi
    
    Value normalization for 'minmax' (approximate):
        - Maximize objectives: value_norm = (value + 1) / 2, clipped to [0, 1]
        - Minimize objectives: value_norm = 1 - |value|/100, clipped to [0, 1]
    
    Aggregation methods:
        - weighted_sum: Best for combining similar-scale metrics
        - weighted_product: All metrics must be positive; penalizes poor performers
        - min: Ensures minimum acceptable performance on all metrics
    
    Examples
    --------
    >>> from pyrrm.objectives.metrics import KGE, PBIAS
    >>> from pyrrm.objectives.transformations import FlowTransformation
    >>> 
    >>> # KGE + KGE(1/Q) for balanced high/low flow calibration
    >>> combined = WeightedObjective([
    ...     (KGE(), 0.5),
    ...     (KGE(transform=FlowTransformation('inverse')), 0.5),
    ... ])
    >>> result = combined.evaluate(obs, sim)
    
    >>> # Multi-metric with different scales
    >>> from pyrrm.objectives.fdc import FDCMetric
    >>> combined = WeightedObjective([
    ...     (KGE(), 0.4),
    ...     (PBIAS(), 0.3),
    ...     (FDCMetric('low', log_transform=True), 0.3),
    ... ], normalize=True)
    """
    
    AGGREGATIONS: Tuple[str, ...] = ('weighted_sum', 'weighted_product', 'min')
    NORMALIZE_METHODS: Tuple[str, ...] = ('minmax', 'direction')
    
    def __init__(self,
                 objectives: List[Tuple[ObjectiveFunction, float]],
                 aggregation: str = 'weighted_sum',
                 normalize: bool = True,
                 normalize_method: str = 'minmax'):
        
        if not objectives:
            raise ValueError("Must provide at least one objective")
        
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Available: {self.AGGREGATIONS}"
            )
        
        if normalize_method not in self.NORMALIZE_METHODS:
            raise ValueError(
                f"Unknown normalize_method '{normalize_method}'. "
                f"Available: {self.NORMALIZE_METHODS}"
            )
        
        # Validate objectives and weights
        for obj, weight in objectives:
            if not isinstance(obj, ObjectiveFunction):
                raise TypeError(f"Expected ObjectiveFunction, got {type(obj)}")
            if weight < 0:
                raise ValueError("Weights must be non-negative")
        
        # Build name
        components = [f"{w:.2f}×{obj.name}" for obj, w in objectives]
        name = f"Composite({' + '.join(components)})"
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        
        self.objectives = objectives
        self.aggregation = aggregation
        self.normalize = normalize
        self.normalize_method = normalize_method
        
        # Normalize weights to sum to 1
        total_weight = sum(w for _, w in objectives)
        if total_weight == 0:
            raise ValueError("Total weight must be positive")
        self._normalized_weights = [w / total_weight for _, w in objectives]
    
    def _normalize_value(self, value: float, obj: ObjectiveFunction) -> float:
        """Normalize objective value to [0, 1] where 1 is optimal."""
        if np.isnan(value):
            return 0.0
        
        if self.normalize_method == 'direction':
            if obj.direction == 'maximize':
                return value
            else:
                return -value
        
        elif self.normalize_method == 'minmax':
            if obj.direction == 'maximize':
                # Assume maximize metrics are roughly in [-1, 1] range (like NSE, KGE)
                return max(0.0, min(1.0, (value + 1) / 2))
            else:
                # Assume minimize metrics: 0 is optimal, 100 is bad (like PBIAS)
                return max(0.0, min(1.0, 1 - abs(value) / 100))
        
        return value
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        values = []
        
        for (obj, _), norm_weight in zip(self.objectives, self._normalized_weights):
            value = obj(obs, sim, **kwargs)
            if self.normalize:
                value = self._normalize_value(value, obj)
            values.append(value)
        
        values = np.array(values)
        weights = np.array(self._normalized_weights)
        
        # Handle NaN values
        valid = ~np.isnan(values)
        if not np.any(valid):
            return np.nan
        
        values = values[valid]
        weights = weights[valid]
        weights = weights / weights.sum()  # Re-normalize
        
        # Aggregate
        if self.aggregation == 'weighted_sum':
            return float(np.sum(weights * values))
        
        elif self.aggregation == 'weighted_product':
            # Ensure positive values for product
            values_pos = np.maximum(values, 1e-10)
            return float(np.prod(values_pos ** weights))
        
        elif self.aggregation == 'min':
            return float(np.min(values))
    
    def get_components(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> Dict[str, float]:
        """Return individual objective function values."""
        result = {}
        for obj, weight in self.objectives:
            result[obj.name] = obj(obs, sim, **kwargs)
            result[f'{obj.name}_weight'] = weight
        return result
    
    def evaluate_individual(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate each objective individually and return detailed results.
        
        Returns
        -------
        dict
            Dictionary with raw values, normalized values, and weights
        """
        result = {
            'raw_values': {},
            'normalized_values': {},
            'weights': {},
            'aggregated_value': self(obs, sim, **kwargs)
        }
        
        for (obj, weight), norm_weight in zip(self.objectives, self._normalized_weights):
            raw = obj(obs, sim, **kwargs)
            result['raw_values'][obj.name] = raw
            result['normalized_values'][obj.name] = self._normalize_value(raw, obj)
            result['weights'][obj.name] = norm_weight
        
        return result
