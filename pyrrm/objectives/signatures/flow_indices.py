"""
Hydrological signature-based objective functions.

This module provides SignatureMetric for evaluating model performance
based on hydrological signatures (indices that characterize catchment behavior).

References
----------
Yilmaz, K.K. et al. (2008). A process-based diagnostic approach to 
model evaluation. Water Resources Research, 44(9).

Addor, N. et al. (2018). A ranking of hydrological signatures based 
on their predictability in space. Water Resources Research, 54.

McMillan, H., Westerberg, I., Branger, F. (2017). Five guidelines for 
selecting hydrological signatures. Hydrological Processes, 31, 4757-4761.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.ndimage import minimum_filter1d

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.constants import (
    HIGH_FLOW_THRESHOLD_MULTIPLIER,
    LOW_FLOW_THRESHOLD_MULTIPLIER,
)


class SignatureMetric(ObjectiveFunction):
    """
    Hydrological signature-based objective function.
    
    Computes percent error between observed and simulated signature values.
    Signatures are hydrological indices that characterize specific aspects
    of catchment behavior.
    
    Parameters
    ----------
    signature : str
        Type of signature to compute:
        
        Flow percentiles:
        - 'q95': 95th percentile (5% exceedance, high flow)
        - 'q90': 90th percentile
        - 'q75': 75th percentile
        - 'q50': 50th percentile (median)
        - 'q25': 25th percentile
        - 'q10': 10th percentile
        - 'q5': 5th percentile (95% exceedance, low flow)
        
        Statistics:
        - 'mean': Mean flow
        - 'std': Standard deviation
        - 'cv': Coefficient of variation (std/mean)
        
        Dynamics:
        - 'flashiness': Richards-Baker flashiness index
        - 'baseflow_index': Estimated baseflow / total flow
        - 'high_flow_freq': Frequency of flows > 3×median
        - 'low_flow_freq': Frequency of flows < 0.2×mean
        - 'zero_flow_freq': Frequency of zero flows
    
    Notes
    -----
    Signature selection guidelines (McMillan et al., 2017):
    1. Signatures should relate to hydrological processes
    2. Sensitive to processes at different time scales
    3. Commonly used in literature for comparability
    
    The percent error output allows combining multiple signatures
    with equal weights in composite objectives.
    
    References
    ----------
    Yilmaz, K.K. et al. (2008). A process-based diagnostic approach to 
    model evaluation. Water Resources Research, 44(9).
    
    Addor, N. et al. (2018). A ranking of hydrological signatures based 
    on their predictability in space. Water Resources Research, 54.
    
    Examples
    --------
    >>> # Low flow signature
    >>> q95_metric = SignatureMetric('q95')
    >>> error = q95_metric(obs, sim)  # Percent error in Q95
    
    >>> # Flashiness index
    >>> flash_metric = SignatureMetric('flashiness')
    """
    
    # Signature definitions: (type, parameter)
    # For percentiles, parameter is the numpy percentile value
    # Note: q95 (95th percentile of flow) = 5th percentile in stats (5% exceedance)
    SIGNATURES: Dict[str, Tuple[str, any]] = {
        # Flow percentiles (hydrological convention: q95 = high flow = 5% exceedance)
        'q95': ('percentile', 95),   # 95th percentile = 5% exceedance (high flow)
        'q90': ('percentile', 90),
        'q75': ('percentile', 75),
        'q50': ('percentile', 50),   # Median
        'q25': ('percentile', 25),
        'q10': ('percentile', 10),
        'q5': ('percentile', 5),     # 5th percentile = 95% exceedance (low flow)
        
        # Statistics
        'mean': ('statistic', 'mean'),
        'std': ('statistic', 'std'),
        'cv': ('statistic', 'cv'),
        
        # Dynamics
        'flashiness': ('dynamic', 'flashiness'),
        'baseflow_index': ('dynamic', 'baseflow_index'),
        'high_flow_freq': ('dynamic', 'high_flow_freq'),
        'low_flow_freq': ('dynamic', 'low_flow_freq'),
        'zero_flow_freq': ('dynamic', 'zero_flow_freq'),
    }
    
    def __init__(self, signature: str):
        if signature not in self.SIGNATURES:
            raise ValueError(
                f"Unknown signature '{signature}'. "
                f"Available: {list(self.SIGNATURES.keys())}"
            )
        
        super().__init__(
            name=f'Sig_{signature}',
            direction='minimize',
            optimal_value=0.0
        )
        self.signature = signature
        self._sig_type, self._sig_param = self.SIGNATURES[signature]
    
    def _compute_signature(self, Q: np.ndarray) -> float:
        """Compute the signature value for a flow series."""
        Q_clean = Q[~np.isnan(Q)]
        
        if len(Q_clean) == 0:
            return np.nan
        
        if self._sig_type == 'percentile':
            return np.percentile(Q_clean, self._sig_param)
        
        elif self._sig_type == 'statistic':
            if self._sig_param == 'mean':
                return np.mean(Q_clean)
            elif self._sig_param == 'std':
                return np.std(Q_clean, ddof=0)
            elif self._sig_param == 'cv':
                mean = np.mean(Q_clean)
                if mean == 0:
                    return np.nan
                return np.std(Q_clean, ddof=0) / mean
        
        elif self._sig_type == 'dynamic':
            if self._sig_param == 'flashiness':
                return self._compute_flashiness(Q_clean)
            elif self._sig_param == 'baseflow_index':
                return self._compute_baseflow_index(Q_clean)
            elif self._sig_param == 'high_flow_freq':
                return self._compute_high_flow_freq(Q_clean)
            elif self._sig_param == 'low_flow_freq':
                return self._compute_low_flow_freq(Q_clean)
            elif self._sig_param == 'zero_flow_freq':
                return self._compute_zero_flow_freq(Q_clean)
        
        return np.nan
    
    def _compute_flashiness(self, Q: np.ndarray) -> float:
        """
        Compute Richards-Baker flashiness index.
        
        FI = Σ|Q_t - Q_{t-1}| / Σ Q_t
        
        Higher values indicate more flashy (variable) flow regime.
        """
        if len(Q) < 2:
            return np.nan
        
        sum_Q = np.sum(Q)
        if sum_Q == 0:
            return np.nan
        
        return np.sum(np.abs(np.diff(Q))) / sum_Q
    
    def _compute_baseflow_index(self, Q: np.ndarray) -> float:
        """
        Compute baseflow index using minimum filter method.
        
        BFI = Σ baseflow / Σ total flow
        
        Uses a 5-day minimum filter to estimate baseflow.
        """
        if len(Q) < 5:
            return np.nan
        
        sum_Q = np.sum(Q)
        if sum_Q == 0:
            return np.nan
        
        # Use 5-day minimum filter for baseflow separation
        window = min(5, len(Q))
        baseflow = minimum_filter1d(Q, size=window)
        
        return np.sum(baseflow) / sum_Q
    
    def _compute_high_flow_freq(self, Q: np.ndarray) -> float:
        """
        Compute frequency of high flow events.
        
        High flow = Q > 3 × median(Q)
        """
        threshold = HIGH_FLOW_THRESHOLD_MULTIPLIER * np.median(Q)
        return np.sum(Q > threshold) / len(Q)
    
    def _compute_low_flow_freq(self, Q: np.ndarray) -> float:
        """
        Compute frequency of low flow events.
        
        Low flow = Q < 0.2 × mean(Q)
        """
        threshold = LOW_FLOW_THRESHOLD_MULTIPLIER * np.mean(Q)
        return np.sum(Q < threshold) / len(Q)
    
    def _compute_zero_flow_freq(self, Q: np.ndarray) -> float:
        """Compute frequency of zero flows."""
        return np.sum(Q == 0) / len(Q)
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        """
        Compute percent error in signature value.
        
        Returns
        -------
        float
            100 × (sig_sim - sig_obs) / |sig_obs|
        """
        self._validate_inputs(obs, sim)
        
        sig_obs = self._compute_signature(obs)
        sig_sim = self._compute_signature(sim)
        
        if np.isnan(sig_obs) or sig_obs == 0:
            return np.nan
        
        return 100.0 * (sig_sim - sig_obs) / abs(sig_obs)
    
    def get_components(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> Dict[str, float]:
        """Return observed and simulated signature values."""
        return {
            'observed': self._compute_signature(obs),
            'simulated': self._compute_signature(sim),
        }
