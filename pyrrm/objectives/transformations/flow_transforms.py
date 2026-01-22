"""
Flow transformation classes for objective functions.

This module provides the FlowTransformation class for applying mathematical
transformations to streamflow data, shifting the emphasis between high and
low flows in objective function calculations.

References
----------
Pushpalatha, R., Perrin, C., Le Moine, N., Andréassian, V. (2012). A review 
of efficiency criteria suitable for evaluating low-flow simulations. 
Journal of Hydrology, 420-421, 171-182.

Santos, L., Thirel, G., Perrin, C. (2018). Technical note: Pitfalls in using 
log-transformed flows within the KGE criterion. HESS, 22, 4583-4591.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np

from pyrrm.objectives.core.constants import (
    TRANSFORM_EMPHASIS,
    DEFAULT_EPSILON_FRACTION,
)


class FlowTransformation:
    """
    Apply mathematical transformations to streamflow data.
    
    Transformations shift the emphasis of objective functions between
    high and low flows by changing the relative weight of errors at
    different flow magnitudes.
    
    Parameters
    ----------
    transform_type : str, default='none'
        Type of transformation:
        - 'none': No transformation (Q)
        - 'sqrt': Square root (√Q)
        - 'log': Natural logarithm (ln(Q))
        - 'inverse': Inverse (1/Q)
        - 'squared': Square (Q²)
        - 'inverse_squared': Inverse square (1/Q²)
        - 'power': Power (Q^p, default p=0.2)
        - 'boxcox': Box-Cox ((Q^λ - 1)/λ, default λ=0.25)
    
    epsilon_method : str, default='mean_fraction'
        Method for handling zero/near-zero flows:
        - 'mean_fraction': epsilon = mean(obs) × epsilon_value
        - 'fixed': epsilon = epsilon_value
        - 'min_nonzero': epsilon = min(obs[obs > 0]) × epsilon_value
    
    epsilon_value : float, default=0.01
        Value used in epsilon calculation
    
    **params : dict
        Additional parameters for specific transforms:
        - 'p': Exponent for power transform (default 0.2)
        - 'lam': Lambda for Box-Cox transform (default 0.25)
    
    Attributes
    ----------
    flow_emphasis : str
        Which flow regime this transformation emphasizes:
        'very_high', 'high', 'balanced', 'low_medium', 'low', 'very_low'
    
    Notes
    -----
    Flow emphasis by transformation:
    
    | Transform | Emphasis | Use Case |
    |-----------|----------|----------|
    | squared | very_high | Peak flow estimation |
    | none | high | Flood forecasting |
    | sqrt | balanced | General purpose |
    | boxcox | balanced | Adaptive applications |
    | power | low_medium | Low flow studies |
    | log | low | Baseflow applications |
    | inverse | low | Low flow indices |
    | inverse_squared | very_low | Drought indices |
    
    WARNING: Log transformation with KGE causes numerical issues.
    Use sqrt or inverse instead (Santos et al., 2018).
    
    Examples
    --------
    >>> # Square root transformation for balanced calibration
    >>> transform = FlowTransformation('sqrt')
    >>> Q_transformed = transform.apply(Q, Q)
    
    >>> # Inverse transformation for low-flow emphasis
    >>> transform = FlowTransformation('inverse', epsilon_method='mean_fraction')
    >>> Q_transformed = transform.apply(Q, Q_obs)
    
    >>> # Power transformation with custom exponent
    >>> transform = FlowTransformation('power', p=0.3)
    """
    
    # Transform functions: (Q, epsilon, **params) -> transformed Q
    _TRANSFORMS: Dict[str, Callable] = {
        'none': lambda Q, eps, **kw: Q,
        'sqrt': lambda Q, eps, **kw: np.sqrt(Q + eps),
        'log': lambda Q, eps, **kw: np.log(Q + eps),
        'inverse': lambda Q, eps, **kw: 1.0 / (Q + eps),
        'squared': lambda Q, eps, **kw: Q ** 2,
        'inverse_squared': lambda Q, eps, **kw: 1.0 / (Q + eps) ** 2,
        'power': lambda Q, eps, p=0.2, **kw: (Q + eps) ** p,
        'boxcox': lambda Q, eps, lam=0.25, **kw: (
            ((Q + eps) ** lam - 1) / lam if lam != 0 else np.log(Q + eps)
        ),
    }
    
    # Valid epsilon methods
    _EPSILON_METHODS = ('mean_fraction', 'fixed', 'min_nonzero')
    
    def __init__(self,
                 transform_type: str = 'none',
                 epsilon_method: str = 'mean_fraction',
                 epsilon_value: float = DEFAULT_EPSILON_FRACTION,
                 **params):
        """
        Initialize the flow transformation.
        
        Parameters
        ----------
        transform_type : str
            Type of transformation to apply
        epsilon_method : str
            Method for calculating epsilon for zero-flow handling
        epsilon_value : float
            Value used in epsilon calculation
        **params : dict
            Additional parameters (p for power, lam for boxcox)
        
        Raises
        ------
        ValueError
            If transform_type or epsilon_method is invalid, or epsilon_value <= 0
        """
        if transform_type not in self._TRANSFORMS:
            raise ValueError(
                f"Unknown transform_type '{transform_type}'. "
                f"Available: {list(self._TRANSFORMS.keys())}"
            )
        
        if epsilon_method not in self._EPSILON_METHODS:
            raise ValueError(
                f"Unknown epsilon_method '{epsilon_method}'. "
                f"Available: {list(self._EPSILON_METHODS)}"
            )
        
        if epsilon_value <= 0:
            raise ValueError("epsilon_value must be positive")
        
        self.transform_type = transform_type
        self.epsilon_method = epsilon_method
        self.epsilon_value = epsilon_value
        self.params = params
    
    @property
    def flow_emphasis(self) -> str:
        """
        Return which flow regime this transformation emphasizes.
        
        Returns
        -------
        str
            One of: 'very_high', 'high', 'balanced', 'low_medium', 'low', 'very_low'
        """
        return TRANSFORM_EMPHASIS.get(self.transform_type, 'unknown')
    
    def get_epsilon(self, obs: np.ndarray) -> float:
        """
        Calculate epsilon value for zero-flow handling.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed flow values (used for relative calculations)
        
        Returns
        -------
        float
            Epsilon value to add before transformation
        """
        obs_clean = obs[~np.isnan(obs)]
        
        if len(obs_clean) == 0:
            return self.epsilon_value
        
        if self.epsilon_method == 'mean_fraction':
            return np.mean(obs_clean) * self.epsilon_value
        elif self.epsilon_method == 'fixed':
            return self.epsilon_value
        elif self.epsilon_method == 'min_nonzero':
            nonzero = obs_clean[obs_clean > 0]
            if len(nonzero) == 0:
                return self.epsilon_value
            return np.min(nonzero) * self.epsilon_value
        
        return self.epsilon_value
    
    def apply(self, Q: np.ndarray, obs_for_eps: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to flow data.
        
        Parameters
        ----------
        Q : np.ndarray
            Flow data to transform
        obs_for_eps : np.ndarray
            Observed data used for epsilon calculation
            (typically the observed flow array)
        
        Returns
        -------
        np.ndarray
            Transformed flow data
        """
        eps = self.get_epsilon(obs_for_eps)
        transform_fn = self._TRANSFORMS[self.transform_type]
        return transform_fn(Q, eps, **self.params)
    
    def __repr__(self) -> str:
        params_str = ""
        if self.params:
            params_str = ", " + ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"FlowTransformation('{self.transform_type}'{params_str})"
    
    def __eq__(self, other) -> bool:
        """Check equality for testing purposes."""
        if not isinstance(other, FlowTransformation):
            return False
        return (
            self.transform_type == other.transform_type and
            self.epsilon_method == other.epsilon_method and
            self.epsilon_value == other.epsilon_value and
            self.params == other.params
        )
    
    def __hash__(self) -> int:
        """Make hashable for use as dict keys."""
        return hash((
            self.transform_type,
            self.epsilon_method,
            self.epsilon_value,
            tuple(sorted(self.params.items()))
        ))
