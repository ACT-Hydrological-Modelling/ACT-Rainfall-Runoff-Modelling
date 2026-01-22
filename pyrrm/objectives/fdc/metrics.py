"""
Flow Duration Curve based objective functions.

This module provides FDC-based metrics for evaluating rainfall-runoff model
performance across different flow regimes.

References
----------
Yilmaz, K.K., Gupta, H.V., Wagener, T. (2008). A process-based diagnostic 
approach to model evaluation: Application to the NWS distributed hydrologic 
model. Water Resources Research, 44(9), W09417.

Westerberg, I.K. et al. (2011). Calibration of hydrological models using 
flow-duration curves. HESS, 15, 2205-2227.
"""

from typing import Optional, Tuple, Dict
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.constants import FDC_SEGMENTS
from pyrrm.objectives.fdc.curves import compute_fdc


class FDCMetric(ObjectiveFunction):
    """
    Flow Duration Curve based objective functions.
    
    The FDC summarizes the frequency distribution of streamflow magnitudes,
    independent of timing. FDC-based metrics enable calibration focused on
    reproducing statistical flow characteristics.
    
    Parameters
    ----------
    segment : str, default='all'
        FDC segment to evaluate:
        - 'peak': 0-2% exceedance (large precipitation events)
        - 'high': 2-20% exceedance (quick runoff)
        - 'mid': 20-70% exceedance (intermediate response)
        - 'low': 70-95% exceedance (baseflow)
        - 'very_low': 95-100% exceedance (drought conditions)
        - 'all': Full range 0-100%
    
    metric : str, default='volume_bias'
        Type of metric to compute:
        - 'volume_bias': Percent bias in segment volume
        - 'slope': Percent bias in segment slope
        - 'rmse': RMSE between FDCs
        - 'correlation': Correlation between FDCs
    
    log_transform : bool, default=False
        Apply log transformation before calculation.
        Recommended for 'low' and 'very_low' segments.
    
    custom_bounds : tuple of float, optional
        Custom exceedance probability bounds (lower, upper).
        Overrides the segment parameter.
    
    Notes
    -----
    FDC segments based on Yilmaz et al. (2008):
    
    | Segment | Exceedance | Hydrological Process |
    |---------|------------|---------------------|
    | peak | 0-2% | Flash response to large events |
    | high | 2-20% | Quick runoff, snowmelt |
    | mid | 20-70% | Intermediate baseflow response |
    | low | 70-95% | Slow baseflow, groundwater |
    | very_low | 95-100% | Drought, minimum flows |
    
    References
    ----------
    Yilmaz, K.K., Gupta, H.V., Wagener, T. (2008). A process-based diagnostic 
    approach to model evaluation. Water Resources Research, 44(9), W09417.
    
    Westerberg, I.K. et al. (2011). Calibration of hydrological models using 
    flow-duration curves. HESS, 15, 2205-2227.
    
    Examples
    --------
    >>> # High flow volume bias
    >>> fdc_high = FDCMetric(segment='high', metric='volume_bias')
    >>> bias = fdc_high(obs, sim)
    
    >>> # Low flow with log transform
    >>> fdc_low = FDCMetric(segment='low', log_transform=True)
    
    >>> # Custom segment (Q10 to Q90)
    >>> fdc_custom = FDCMetric(custom_bounds=(0.10, 0.90), metric='rmse')
    """
    
    METRICS: Tuple[str, ...] = ('volume_bias', 'slope', 'rmse', 'correlation')
    
    def __init__(self,
                 segment: str = 'all',
                 metric: str = 'volume_bias',
                 log_transform: bool = False,
                 custom_bounds: Optional[Tuple[float, float]] = None):
        
        # Validate segment
        if custom_bounds is None and segment not in FDC_SEGMENTS:
            raise ValueError(
                f"Unknown segment '{segment}'. "
                f"Available: {list(FDC_SEGMENTS.keys())} or use custom_bounds"
            )
        
        # Validate metric
        if metric not in self.METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. Available: {self.METRICS}"
            )
        
        # Validate custom bounds
        if custom_bounds is not None:
            if len(custom_bounds) != 2:
                raise ValueError("custom_bounds must have 2 elements")
            if not (0 <= custom_bounds[0] < custom_bounds[1] <= 1):
                raise ValueError("custom_bounds must satisfy 0 <= lower < upper <= 1")
        
        # Build name
        seg_name = segment if custom_bounds is None else f"custom_{custom_bounds}"
        name = f'FDC_{seg_name}_{metric}'
        if log_transform:
            name += '_log'
        
        # Determine direction and optimal value
        if metric in ('volume_bias', 'slope'):
            direction = 'minimize'
            optimal = 0.0
        elif metric == 'rmse':
            direction = 'minimize'
            optimal = 0.0
        else:  # correlation
            direction = 'maximize'
            optimal = 1.0
        
        super().__init__(name=name, direction=direction, optimal_value=optimal)
        
        self.segment = segment
        self.metric = metric
        self.log_transform = log_transform
        self.bounds = custom_bounds if custom_bounds else FDC_SEGMENTS[segment]
    
    def _get_segment_mask(self, exceedance: np.ndarray) -> np.ndarray:
        """Get boolean mask for the specified FDC segment."""
        lower, upper = self.bounds
        return (exceedance >= lower) & (exceedance <= upper)
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        
        # Compute FDCs
        exc_obs, fdc_obs = compute_fdc(obs)
        exc_sim, fdc_sim = compute_fdc(sim)
        
        if len(exc_obs) == 0 or len(exc_sim) == 0:
            return np.nan
        
        # Get segment data
        mask_obs = self._get_segment_mask(exc_obs)
        mask_sim = self._get_segment_mask(exc_sim)
        
        fdc_obs_seg = fdc_obs[mask_obs]
        fdc_sim_seg = fdc_sim[mask_sim]
        
        # Handle length mismatch by truncating to shorter
        n = min(len(fdc_obs_seg), len(fdc_sim_seg))
        if n == 0:
            return np.nan
        
        fdc_obs_seg = fdc_obs_seg[:n]
        fdc_sim_seg = fdc_sim_seg[:n]
        
        # Apply log transform if specified
        if self.log_transform:
            # Use mean-fraction epsilon for numerical stability
            obs_clean = obs[~np.isnan(obs)]
            eps = np.mean(obs_clean) * 0.01 if len(obs_clean) > 0 else 0.01
            fdc_obs_seg = np.log(fdc_obs_seg + eps)
            fdc_sim_seg = np.log(fdc_sim_seg + eps)
        
        # Compute metric
        if self.metric == 'volume_bias':
            return self._compute_volume_bias(fdc_obs_seg, fdc_sim_seg)
        elif self.metric == 'slope':
            return self._compute_slope_bias(fdc_obs_seg, fdc_sim_seg)
        elif self.metric == 'rmse':
            return self._compute_rmse(fdc_obs_seg, fdc_sim_seg)
        elif self.metric == 'correlation':
            return self._compute_correlation(fdc_obs_seg, fdc_sim_seg)
    
    def _compute_volume_bias(self, obs_seg: np.ndarray, sim_seg: np.ndarray) -> float:
        """Compute percent bias in segment volume."""
        sum_obs = np.sum(np.abs(obs_seg))
        if sum_obs < 1e-10:
            return np.nan
        return 100.0 * (np.sum(sim_seg) - np.sum(obs_seg)) / sum_obs
    
    def _compute_slope_bias(self, obs_seg: np.ndarray, sim_seg: np.ndarray) -> float:
        """Compute percent bias in segment slope."""
        if len(obs_seg) < 2:
            return np.nan
        
        # Simple linear slope across segment
        slope_obs = (obs_seg[-1] - obs_seg[0]) / len(obs_seg)
        slope_sim = (sim_seg[-1] - sim_seg[0]) / len(sim_seg)
        
        if abs(slope_obs) < 1e-10:
            return np.nan
        
        return 100.0 * (slope_sim - slope_obs) / abs(slope_obs)
    
    def _compute_rmse(self, obs_seg: np.ndarray, sim_seg: np.ndarray) -> float:
        """Compute RMSE between FDC segments."""
        return np.sqrt(np.mean((obs_seg - sim_seg) ** 2))
    
    def _compute_correlation(self, obs_seg: np.ndarray, sim_seg: np.ndarray) -> float:
        """Compute correlation between FDC segments."""
        if np.std(obs_seg) == 0 or np.std(sim_seg) == 0:
            return np.nan
        
        corr = np.corrcoef(obs_seg, sim_seg)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def get_components(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Return FDC segment statistics.
        
        Returns
        -------
        dict
            Dictionary with segment statistics including:
            - 'obs_segment_mean': Mean observed flow in segment
            - 'sim_segment_mean': Mean simulated flow in segment
            - 'segment_n': Number of points in segment
        """
        exc_obs, fdc_obs = compute_fdc(obs)
        exc_sim, fdc_sim = compute_fdc(sim)
        
        mask_obs = self._get_segment_mask(exc_obs)
        mask_sim = self._get_segment_mask(exc_sim)
        
        fdc_obs_seg = fdc_obs[mask_obs]
        fdc_sim_seg = fdc_sim[mask_sim]
        
        n = min(len(fdc_obs_seg), len(fdc_sim_seg))
        
        return {
            'obs_segment_mean': np.mean(fdc_obs_seg[:n]) if n > 0 else np.nan,
            'sim_segment_mean': np.mean(fdc_sim_seg[:n]) if n > 0 else np.nan,
            'segment_n': n,
        }
