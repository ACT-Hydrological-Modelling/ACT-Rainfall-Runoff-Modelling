"""
Traditional hydrological objective functions.

This module provides implementations of commonly used objective functions
for evaluating rainfall-runoff model performance:
- NSE: Nash-Sutcliffe Efficiency
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- PBIAS: Percent Bias
- SDEB: Sum of Daily Flows, Daily Exceedance Curve and Bias

References
----------
Nash, J.E., Sutcliffe, J.V. (1970). River flow forecasting through 
conceptual models part I — A discussion of principles. Journal of 
Hydrology, 10(3), 282-290.

Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
A robust approach for calibrating continuous hydrological models for
daily to sub-daily streamflow simulation. Journal of Hydrology, 494, 
80-91.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.constants import (
    SDEB_DEFAULT_ALPHA,
    SDEB_DEFAULT_LAMBDA,
)

if TYPE_CHECKING:
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation


class NSE(ObjectiveFunction):
    """
    Nash-Sutcliffe Efficiency (Nash and Sutcliffe, 1970).
    
    Formula
    -------
    NSE = 1 - Σ(Qobs - Qsim)² / Σ(Qobs - μobs)²
    
    Properties
    ----------
    - Range: (-∞, 1]
    - Optimal value: 1 (perfect fit)
    - NSE = 0: Model is as good as using the mean
    - NSE < 0: Model is worse than using the mean
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation.
        Use sqrt for balanced evaluation, inverse for low-flow emphasis.
    
    Notes
    -----
    - Emphasizes high flows due to squared error terms
    - Sensitive to systematic bias
    - Does not account for timing errors explicitly
    
    References
    ----------
    Nash, J.E. and Sutcliffe, J.V. (1970). River flow forecasting through 
    conceptual models part I — A discussion of principles. Journal of 
    Hydrology, 10(3), 282-290.
    
    Examples
    --------
    >>> nse = NSE()
    >>> value = nse(obs, sim)
    
    >>> # With square root transformation for balanced evaluation
    >>> from pyrrm.objectives.transformations import FlowTransformation
    >>> nse_sqrt = NSE(transform=FlowTransformation('sqrt'))
    >>> value = nse_sqrt(obs, sim)
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'NSE'
        if transform is not None and transform.transform_type != 'none':
            name = f'NSE({transform.transform_type})'
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        self.transform = transform
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        # Apply transformation if specified
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        # Calculate NSE
        numerator = np.sum((obs_t - sim_t) ** 2)
        denominator = np.sum((obs_t - np.mean(obs_t)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1.0 - numerator / denominator


class RMSE(ObjectiveFunction):
    """
    Root Mean Square Error.
    
    Formula
    -------
    RMSE = √[Σ(Qobs - Qsim)² / n]
    
    Properties
    ----------
    - Range: [0, ∞)
    - Optimal value: 0
    - Same units as input data
    - Emphasizes large errors due to squaring
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    normalized : bool, default=False
        If True, divide by mean observed flow (NRMSE)
    
    Examples
    --------
    >>> rmse = RMSE()
    >>> value = rmse(obs, sim)
    
    >>> # Normalized RMSE
    >>> nrmse = RMSE(normalized=True)
    >>> value = nrmse(obs, sim)
    """
    
    def __init__(self, 
                 transform: Optional['FlowTransformation'] = None,
                 normalized: bool = False):
        name = 'NRMSE' if normalized else 'RMSE'
        if transform is not None and transform.transform_type != 'none':
            name = f'{name}({transform.transform_type})'
        
        super().__init__(name=name, direction='minimize', optimal_value=0.0)
        self.transform = transform
        self.normalized = normalized
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        rmse = np.sqrt(np.mean((obs_t - sim_t) ** 2))
        
        if self.normalized:
            mean_obs = np.mean(obs_t)
            if mean_obs == 0:
                return np.nan
            return rmse / mean_obs
        
        return rmse


class MAE(ObjectiveFunction):
    """
    Mean Absolute Error.
    
    Formula
    -------
    MAE = Σ|Qobs - Qsim| / n
    
    Properties
    ----------
    - Range: [0, ∞)
    - Optimal value: 0
    - Same units as input data
    - Less sensitive to outliers than RMSE
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    
    Examples
    --------
    >>> mae = MAE()
    >>> value = mae(obs, sim)
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'MAE'
        if transform is not None and transform.transform_type != 'none':
            name = f'MAE({transform.transform_type})'
        
        super().__init__(name=name, direction='minimize', optimal_value=0.0)
        self.transform = transform
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        return np.mean(np.abs(obs_t - sim_t))


class PBIAS(ObjectiveFunction):
    """
    Percent Bias.
    
    Formula
    -------
    PBIAS = 100 × Σ(Qsim - Qobs) / Σ(Qobs)
    
    Properties
    ----------
    - Range: (-∞, ∞)
    - Optimal value: 0
    - Positive values indicate overestimation
    - Negative values indicate underestimation
    
    Notes
    -----
    Measures systematic tendency to over/under-predict.
    Does not account for timing or variability.
    
    Examples
    --------
    >>> pbias = PBIAS()
    >>> value = pbias(obs, sim)
    """
    
    def __init__(self):
        super().__init__(name='PBIAS', direction='minimize', optimal_value=0.0)
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        sum_obs = np.sum(obs_clean)
        if sum_obs == 0:
            return np.nan
        
        return 100.0 * np.sum(sim_clean - obs_clean) / sum_obs


class SDEB(ObjectiveFunction):
    """
    Sum of Daily Flows, Daily Exceedance Curve and Bias (Lerat et al., 2013).
    
    The SDEB metric combines three terms:
    1. Sum of errors on power-transformed flows (chronological, timing-sensitive)
    2. Sum of errors on sorted/ranked flows (FDC-based, timing-insensitive)
    3. Relative simulation bias as a penalty multiplier
    
    Formula
    -------
    SDEB = (α × Σ[Q_obs^λ - Q_sim^λ]² + (1-α) × Σ[R_obs^λ - R_sim^λ]²) × (1 + |bias|)
    
    where:
        α = weighting factor (default 0.1)
        λ = power transform exponent (default 0.5)
        R = ranked (sorted) flows
        bias = |Σ Q_sim - Σ Q_obs| / Σ Q_obs
    
    Properties
    ----------
    - Range: [0, ∞)
    - Optimal value: 0 (perfect fit)
    - Direction: minimize
    
    Parameters
    ----------
    alpha : float, default=0.1
        Weighting factor for chronological term (0 to 1).
        Low values (e.g., 0.1) reduce impact of timing errors.
    lam : float, default=0.5
        Power transform exponent. 0.5 (sqrt) balances high/low flows.
    
    Notes
    -----
    - Used in Australian SOURCE modeling platform
    - Power transform reduces weight on high flow errors
    - Low alpha reduces timing error impact (useful when peak timing uncertain)
    - The ranked term ensures FDC shape is preserved regardless of timing
    - Designed for daily data
    
    References
    ----------
    Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
    A robust approach for calibrating continuous hydrological models for
    daily to sub-daily streamflow simulation. Journal of Hydrology, 494, 
    80-91.
    
    Coron, L., Andréassian, V., Perrin, C., Bourqui, M., Hendrickx, F. (2012).
    On the lack of robustness of hydrological models regarding water balance
    simulation. Hydrological Sciences Journal, 57(6), 1121-1135.
    
    Examples
    --------
    >>> sdeb = SDEB()
    >>> value = sdeb(obs, sim)
    
    >>> # Custom parameters
    >>> sdeb = SDEB(alpha=0.3, lam=0.3)
    >>> value = sdeb(obs, sim)
    """
    
    def __init__(self, 
                 alpha: float = SDEB_DEFAULT_ALPHA, 
                 lam: float = SDEB_DEFAULT_LAMBDA):
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        
        super().__init__(name='SDEB', direction='minimize', optimal_value=0.0)
        self.alpha = alpha
        self.lam = lam
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        n = len(obs_clean)
        
        # Apply power transform
        obs_transformed = obs_clean ** self.lam
        sim_transformed = sim_clean ** self.lam
        
        # Sort flows for ranked term (descending order for FDC convention)
        obs_sorted = np.sort(obs_clean)[::-1]
        sim_sorted = np.sort(sim_clean)[::-1]
        
        obs_sorted_transformed = obs_sorted ** self.lam
        sim_sorted_transformed = sim_sorted ** self.lam
        
        # Term 1: Chronological (timing-sensitive)
        chronological_term = np.sum((obs_transformed - sim_transformed) ** 2)
        
        # Term 2: Ranked/FDC (timing-insensitive)
        ranked_term = np.sum((obs_sorted_transformed - sim_sorted_transformed) ** 2)
        
        # Combined error term
        combined_error = self.alpha * chronological_term + (1 - self.alpha) * ranked_term
        
        # Bias penalty
        sum_obs = np.sum(obs_clean)
        if sum_obs == 0:
            return np.nan
        
        relative_bias = np.abs(np.sum(sim_clean) - sum_obs) / sum_obs
        bias_penalty = 1 + relative_bias
        
        return combined_error * bias_penalty
    
    def get_components(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> dict:
        """
        Return component breakdown of SDEB.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'chronological_term': Timing-sensitive error term
            - 'ranked_term': FDC-based error term
            - 'bias_penalty': Multiplicative bias penalty
            - 'relative_bias': Relative volume bias
        """
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        # Apply power transform
        obs_transformed = obs_clean ** self.lam
        sim_transformed = sim_clean ** self.lam
        
        # Sort flows
        obs_sorted = np.sort(obs_clean)[::-1]
        sim_sorted = np.sort(sim_clean)[::-1]
        
        obs_sorted_transformed = obs_sorted ** self.lam
        sim_sorted_transformed = sim_sorted ** self.lam
        
        # Calculate components
        chronological_term = np.sum((obs_transformed - sim_transformed) ** 2)
        ranked_term = np.sum((obs_sorted_transformed - sim_sorted_transformed) ** 2)
        
        sum_obs = np.sum(obs_clean)
        relative_bias = np.abs(np.sum(sim_clean) - sum_obs) / sum_obs if sum_obs != 0 else np.nan
        bias_penalty = 1 + relative_bias if not np.isnan(relative_bias) else np.nan
        
        return {
            'chronological_term': chronological_term,
            'ranked_term': ranked_term,
            'relative_bias': relative_bias,
            'bias_penalty': bias_penalty,
        }
