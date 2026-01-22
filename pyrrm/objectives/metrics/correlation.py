"""
Correlation-based objective functions.

This module provides correlation metrics for evaluating rainfall-runoff
model performance:
- PearsonCorrelation: Linear correlation coefficient
- SpearmanCorrelation: Rank-based correlation coefficient
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy import stats

from pyrrm.objectives.core.base import ObjectiveFunction

if TYPE_CHECKING:
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation


class PearsonCorrelation(ObjectiveFunction):
    """
    Pearson correlation coefficient.
    
    Measures the linear relationship between observed and simulated values.
    
    Formula
    -------
    r = Cov(obs, sim) / (σ_obs × σ_sim)
    
    Properties
    ----------
    - Range: [-1, 1]
    - Optimal value: 1 (perfect positive correlation)
    - r = 0: No linear relationship
    - r < 0: Negative correlation (usually indicates model issues)
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    
    Notes
    -----
    - Sensitive to outliers
    - Does not capture bias or variability differences
    - Best used in combination with other metrics
    
    Examples
    --------
    >>> pearson = PearsonCorrelation()
    >>> value = pearson(obs, sim)
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'Pearson_r'
        if transform is not None and transform.transform_type != 'none':
            name = f'{name}({transform.transform_type})'
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        self.transform = transform
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if len(obs_clean) < 2:
            return np.nan
        
        # Apply transformation if specified
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        # Check for constant values
        if np.std(obs_t) == 0 or np.std(sim_t) == 0:
            return np.nan
        
        corr_matrix = np.corrcoef(obs_t, sim_t)
        r = corr_matrix[0, 1]
        
        return r if not np.isnan(r) else 0.0


class SpearmanCorrelation(ObjectiveFunction):
    """
    Spearman rank correlation coefficient.
    
    Measures the monotonic relationship between observed and simulated values
    using ranks instead of actual values. More robust to outliers than Pearson.
    
    Formula
    -------
    r_s = Pearson correlation of rank(obs) and rank(sim)
    
    Properties
    ----------
    - Range: [-1, 1]
    - Optimal value: 1 (perfect monotonic relationship)
    - r_s = 0: No monotonic relationship
    - More robust to outliers than Pearson
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    
    Notes
    -----
    - Based on ranks, so less sensitive to extreme values
    - Captures monotonic (not just linear) relationships
    - Useful for non-Gaussian distributions common in hydrology
    
    Examples
    --------
    >>> spearman = SpearmanCorrelation()
    >>> value = spearman(obs, sim)
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'Spearman_rho'
        if transform is not None and transform.transform_type != 'none':
            name = f'{name}({transform.transform_type})'
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        self.transform = transform
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if len(obs_clean) < 2:
            return np.nan
        
        # Apply transformation if specified
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        # Calculate Spearman correlation
        r_s, p_value = stats.spearmanr(obs_t, sim_t)
        
        return r_s if not np.isnan(r_s) else 0.0
    
    def get_components(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> dict:
        """
        Return Spearman correlation with p-value.
        
        Returns
        -------
        dict
            Dictionary with 'rho' and 'p_value' keys
        """
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if len(obs_clean) < 2:
            return {'rho': np.nan, 'p_value': np.nan}
        
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        r_s, p_value = stats.spearmanr(obs_t, sim_t)
        
        return {
            'rho': r_s if not np.isnan(r_s) else 0.0,
            'p_value': p_value if not np.isnan(p_value) else 1.0,
        }
