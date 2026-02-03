"""
APEX: Adaptive Process-Explicit Objective Function.

This module provides the APEX objective function, which extends SDEB
(Lerat et al., 2013) with novel dynamics and lag penalty multipliers
for comprehensive rainfall-runoff model calibration.

Novel Contributions
-------------------
1. Dynamics Multiplier: Penalizes mismatch in gradient/rate-of-change patterns
2. Lag Multiplier: Penalizes systematic timing offsets (optional)
3. Regime-weighted ranked term: Continuous weighting by exceedance probability

References
----------
Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
A robust approach for calibrating continuous hydrological models for
daily to sub-daily streamflow simulation. Journal of Hydrology, 494, 80-91.
"""

from typing import Optional, Dict, Union, Callable
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.constants import (
    APEX_DEFAULT_ALPHA,
    APEX_DEFAULT_TRANSFORM_PARAM,
    APEX_DEFAULT_BIAS_STRENGTH,
    APEX_DEFAULT_BIAS_POWER,
    APEX_DEFAULT_DYNAMICS_STRENGTH,
    APEX_DEFAULT_LAG_STRENGTH,
    APEX_DEFAULT_LAG_REFERENCE,
    REGIME_EMPHASIS_FUNCTIONS,
)
from pyrrm.objectives.transformations.flow_transforms import FlowTransformation


class APEX(ObjectiveFunction):
    """
    APEX: Adaptive Process-Explicit Objective Function.
    
    Extends SDEB (Lerat et al., 2013) with dynamics and lag penalty
    multipliers for comprehensive rainfall-runoff model calibration.
    
    Formula
    -------
    APEX = [α × E_chron + (1-α) × E_ranked] × BiasMultiplier × DynamicsMultiplier × [LagMultiplier]
    
    where:
        E_chron = Σ[T(Q_obs) - T(Q_sim)]² (chronological, timing-sensitive)
        E_ranked = Σ w(p) × [T(R_obs) - T(R_sim)]² (ranked/FDC, timing-insensitive)
        BiasMultiplier = 1 + β × |bias|^γ
        DynamicsMultiplier = 1 + κ × (1 - ρ_gradient) [NOVEL]
        LagMultiplier = 1 + λ × |optimal_lag| / τ [NOVEL, optional]
    
    Parameters
    ----------
    alpha : float, default=0.1
        Weight for chronological term (timing sensitivity).
        Low values (e.g., 0.1) reduce timing error impact.
        Range: [0, 1]
    
    transform : str or FlowTransformation, default='power'
        Flow transformation to apply. Options:
        - 'power': Q^p (default p=0.5, equivalent to sqrt)
        - 'sqrt': √Q (balanced emphasis)
        - 'log': ln(Q) (low flow emphasis)
        - 'inverse': 1/Q (strong low flow emphasis)
        - 'none': No transformation (high flow emphasis)
        - FlowTransformation instance for custom configuration
    
    transform_param : float, default=0.5
        Parameter for power transformation (λ exponent).
        λ=0.5 (sqrt) balances high/low flows.
        λ<0.5 increases low flow weight.
    
    regime_emphasis : str, default='uniform'
        Flow regime weighting in ranked term. Options:
        - 'uniform': Equal weights (original SDEB)
        - 'low_flow': w(p) = p (higher weight at high exceedance)
        - 'high_flow': w(p) = 1-p (higher weight at low exceedance)
        - 'balanced': w(p) = 4p(1-p) (emphasize mid-range)
        - 'extremes': w(p) = 1-4p(1-p) (emphasize both tails)
    
    bias_strength : float, default=1.0
        Bias penalty strength (β). Higher = stricter volume matching.
    
    bias_power : float, default=1.0
        Bias penalty exponent (γ). 1=linear, 2=quadratic.
    
    dynamics_strength : float, default=0.5
        Dynamics multiplier strength (κ). [NOVEL CONTRIBUTION]
        Penalizes mismatch in gradient correlation.
        Set to 0 for SDEB-equivalent behavior.
    
    lag_penalty : bool, default=False
        Enable lag multiplier. [NOVEL CONTRIBUTION]
    
    lag_strength : float, default=0.3
        Lag penalty strength (λ). Only used if lag_penalty=True.
    
    lag_reference : int, default=5
        Reference lag in timesteps (τ). Normalizes the lag penalty.
    
    Properties
    ----------
    - Range: [0, ∞)
    - Optimal value: 0 (perfect fit)
    - Direction: minimize
    
    Notes
    -----
    **SDEB Foundation:**
    The core formula follows SDEB (Lerat et al., 2013) which combines:
    1. Chronological term: Point-wise errors preserving timing information
    2. Ranked term: FDC-like comparison independent of timing
    3. Bias multiplier: Volume balance penalty
    
    **Novel Contributions:**
    1. **Dynamics Multiplier**: Ensures the model captures rate-of-change patterns,
       not just levels. Based on correlation of first differences.
    2. **Lag Multiplier**: Penalizes systematic timing offsets detected via
       cross-correlation.
    3. **Regime Weighting**: Continuous flow-regime weighting in the ranked term,
       avoiding discontinuities of segment-based approaches.
    
    **Advantages over weighted composite approaches:**
    - No normalization issues (all terms in same error space)
    - Multiplicative structure ensures all aspects must be satisfied
    - Fewer hyperparameters to tune
    - Mathematically principled
    
    References
    ----------
    Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
    A robust approach for calibrating continuous hydrological models for
    daily to sub-daily streamflow simulation. Journal of Hydrology, 494, 80-91.
    
    Examples
    --------
    >>> from pyrrm.objectives import APEX
    >>> 
    >>> # Default: SDEB-like with dynamics multiplier
    >>> apex = APEX()
    >>> value = apex(observed, simulated)
    >>> 
    >>> # Low flow emphasis
    >>> apex_low = APEX(
    ...     transform='log',
    ...     regime_emphasis='low_flow',
    ...     bias_strength=1.5
    ... )
    >>> 
    >>> # High flow / flood focus
    >>> apex_flood = APEX(
    ...     transform_param=0.7,  # Higher λ = more high-flow weight
    ...     regime_emphasis='high_flow',
    ...     dynamics_strength=0.7  # Emphasize dynamics
    ... )
    >>> 
    >>> # With lag penalty for timing-critical applications
    >>> apex_timing = APEX(lag_penalty=True, lag_strength=0.5)
    >>> 
    >>> # SDEB-equivalent (disable dynamics multiplier)
    >>> apex_sdeb = APEX(dynamics_strength=0.0)
    >>> 
    >>> # Get component breakdown
    >>> components = apex.get_components(observed, simulated)
    >>> print(f"Chronological term: {components['chronological_term']:.4f}")
    >>> print(f"Dynamics multiplier: {components['dynamics_multiplier']:.4f}")
    
    See Also
    --------
    SDEB : Original Sum of Daily Flows, Daily Exceedance Curve and Bias
    """
    
    def __init__(self,
                 alpha: float = APEX_DEFAULT_ALPHA,
                 transform: Union[str, FlowTransformation] = 'power',
                 transform_param: float = APEX_DEFAULT_TRANSFORM_PARAM,
                 regime_emphasis: str = 'uniform',
                 bias_strength: float = APEX_DEFAULT_BIAS_STRENGTH,
                 bias_power: float = APEX_DEFAULT_BIAS_POWER,
                 dynamics_strength: float = APEX_DEFAULT_DYNAMICS_STRENGTH,
                 lag_penalty: bool = False,
                 lag_strength: float = APEX_DEFAULT_LAG_STRENGTH,
                 lag_reference: int = APEX_DEFAULT_LAG_REFERENCE):
        """Initialize the APEX objective function."""
        # Validate alpha
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        # Validate transform_param
        if transform_param <= 0:
            raise ValueError(f"transform_param must be positive, got {transform_param}")
        
        # Validate regime_emphasis
        if regime_emphasis not in REGIME_EMPHASIS_FUNCTIONS:
            raise ValueError(
                f"Unknown regime_emphasis '{regime_emphasis}'. "
                f"Available: {list(REGIME_EMPHASIS_FUNCTIONS.keys())}"
            )
        
        # Validate bias parameters
        if bias_strength < 0:
            raise ValueError(f"bias_strength must be non-negative, got {bias_strength}")
        if bias_power <= 0:
            raise ValueError(f"bias_power must be positive, got {bias_power}")
        
        # Validate dynamics parameters
        if dynamics_strength < 0:
            raise ValueError(f"dynamics_strength must be non-negative, got {dynamics_strength}")
        
        # Validate lag parameters
        if lag_strength < 0:
            raise ValueError(f"lag_strength must be non-negative, got {lag_strength}")
        if lag_reference <= 0:
            raise ValueError(f"lag_reference must be positive, got {lag_reference}")
        
        # Build name
        name = 'APEX'
        if dynamics_strength == 0:
            name = 'APEX(SDEB-equiv)'
        elif lag_penalty:
            name = 'APEX(+lag)'
        
        super().__init__(name=name, direction='minimize', optimal_value=0.0)
        
        # Store parameters
        self.alpha = alpha
        self.transform_param = transform_param
        self.regime_emphasis = regime_emphasis
        self.bias_strength = bias_strength
        self.bias_power = bias_power
        self.dynamics_strength = dynamics_strength
        self.lag_penalty = lag_penalty
        self.lag_strength = lag_strength
        self.lag_reference = lag_reference
        
        # Set up transformation
        if isinstance(transform, FlowTransformation):
            self._transform = transform
        elif transform == 'power':
            # Power transform with custom exponent
            self._transform = FlowTransformation('power', p=transform_param)
        elif transform in ('sqrt', 'log', 'inverse', 'none', 'boxcox'):
            self._transform = FlowTransformation(transform)
        else:
            raise ValueError(
                f"Unknown transform '{transform}'. "
                f"Use 'power', 'sqrt', 'log', 'inverse', 'none', 'boxcox', "
                f"or a FlowTransformation instance."
            )
        
        # Store regime weight function
        self._regime_weight_fn = REGIME_EMPHASIS_FUNCTIONS[regime_emphasis]
    
    def _apply_transform(self, Q: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Apply the configured flow transformation."""
        return self._transform.apply(Q, obs)
    
    def _compute_chronological_term(self, 
                                     obs: np.ndarray, 
                                     sim: np.ndarray) -> float:
        """
        Compute chronological error term (timing-sensitive).
        
        E_chron = Σ[T(Q_obs) - T(Q_sim)]²
        """
        obs_t = self._apply_transform(obs, obs)
        sim_t = self._apply_transform(sim, obs)
        
        return np.sum((obs_t - sim_t) ** 2)
    
    def _compute_ranked_term(self, 
                              obs: np.ndarray, 
                              sim: np.ndarray) -> float:
        """
        Compute ranked error term with optional flow-regime weighting.
        
        E_ranked = Σ w(p) × [T(R_obs) - T(R_sim)]²
        
        where R = sorted flows (descending), p = exceedance probability
        """
        # Sort flows in descending order (FDC convention)
        obs_sorted = np.sort(obs)[::-1]
        sim_sorted = np.sort(sim)[::-1]
        
        # Apply transformation
        obs_t = self._apply_transform(obs_sorted, obs)
        sim_t = self._apply_transform(sim_sorted, obs)
        
        # Compute exceedance probabilities
        n = len(obs_sorted)
        exceedance = np.arange(1, n + 1) / (n + 1)
        
        # Compute regime weights
        if self.regime_emphasis == 'uniform':
            weights = np.ones(n)
        else:
            weights = np.array([self._regime_weight_fn(p) for p in exceedance])
        
        # Normalize weights to maintain scale comparability
        weights = weights / np.mean(weights)
        
        return np.sum(weights * (obs_t - sim_t) ** 2)
    
    def _compute_bias_multiplier(self, 
                                  obs: np.ndarray, 
                                  sim: np.ndarray) -> tuple:
        """
        Compute bias multiplier.
        
        BiasMultiplier = 1 + β × |bias|^γ
        
        Returns
        -------
        tuple
            (multiplier, relative_bias)
        """
        sum_obs = np.sum(obs)
        if sum_obs == 0:
            return 1.0, 0.0
        
        relative_bias = (np.sum(sim) - sum_obs) / sum_obs
        multiplier = 1.0 + self.bias_strength * np.abs(relative_bias) ** self.bias_power
        
        return multiplier, relative_bias
    
    def _compute_dynamics_multiplier(self, 
                                      obs: np.ndarray, 
                                      sim: np.ndarray) -> tuple:
        """
        Compute dynamics multiplier based on gradient correlation.
        
        This is a NOVEL CONTRIBUTION.
        
        ρ_gradient = corr(diff(Q_obs), diff(Q_sim))
        DynamicsMultiplier = 1 + κ × (1 - ρ_gradient)
        
        The dynamics multiplier ensures the model captures rate-of-change
        patterns, not just levels. A model that matches levels but has
        wrong recession behavior will be penalized.
        
        Returns
        -------
        tuple
            (multiplier, gradient_correlation)
        """
        if self.dynamics_strength == 0:
            return 1.0, 1.0
        
        # Compute gradients (first differences)
        grad_obs = np.diff(obs)
        grad_sim = np.diff(sim)
        
        # Handle edge cases
        if len(grad_obs) < 2:
            return 1.0, 1.0
        
        std_obs = np.std(grad_obs)
        std_sim = np.std(grad_sim)
        
        if std_obs == 0 or std_sim == 0:
            # Constant or near-constant gradients
            return 1.0, 1.0
        
        # Compute gradient correlation
        rho_gradient = np.corrcoef(grad_obs, grad_sim)[0, 1]
        
        # Handle NaN (can occur with numerical issues)
        if np.isnan(rho_gradient):
            rho_gradient = 0.0
        
        # Clip for safety
        rho_gradient = np.clip(rho_gradient, -1.0, 1.0)
        
        # Compute multiplier
        multiplier = 1.0 + self.dynamics_strength * (1.0 - rho_gradient)
        
        return multiplier, rho_gradient
    
    def _compute_lag_multiplier(self, 
                                 obs: np.ndarray, 
                                 sim: np.ndarray) -> tuple:
        """
        Compute lag multiplier based on cross-correlation optimal lag.
        
        This is a NOVEL CONTRIBUTION.
        
        LagMultiplier = 1 + λ × |optimal_lag| / τ
        
        The lag multiplier penalizes systematic timing offsets. A model
        that consistently leads or lags observations will be penalized.
        
        Returns
        -------
        tuple
            (multiplier, optimal_lag)
        """
        if not self.lag_penalty:
            return 1.0, 0
        
        # Handle edge cases
        std_obs = np.std(obs)
        std_sim = np.std(sim)
        
        if std_obs == 0 or std_sim == 0:
            return 1.0, 0
        
        # Normalized cross-correlation
        obs_norm = (obs - np.mean(obs)) / std_obs
        sim_norm = (sim - np.mean(sim)) / std_sim
        
        # Compute cross-correlation
        corr = np.correlate(obs_norm, sim_norm, mode='full')
        
        # Find optimal lag (positive = sim lags obs, negative = sim leads obs)
        optimal_lag = np.argmax(corr) - len(obs) + 1
        
        # Compute multiplier
        multiplier = 1.0 + self.lag_strength * abs(optimal_lag) / self.lag_reference
        
        return multiplier, optimal_lag
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        """
        Calculate the APEX objective function value.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        
        Returns
        -------
        float
            APEX value (minimize towards 0)
        """
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        # Compute core error terms
        chron_term = self._compute_chronological_term(obs_clean, sim_clean)
        ranked_term = self._compute_ranked_term(obs_clean, sim_clean)
        
        # Weighted combination
        weighted_error = self.alpha * chron_term + (1 - self.alpha) * ranked_term
        
        # Compute multipliers
        bias_mult, _ = self._compute_bias_multiplier(obs_clean, sim_clean)
        dynamics_mult, _ = self._compute_dynamics_multiplier(obs_clean, sim_clean)
        lag_mult, _ = self._compute_lag_multiplier(obs_clean, sim_clean)
        
        # Final APEX value
        return weighted_error * bias_mult * dynamics_mult * lag_mult
    
    def get_components(self, 
                       obs: np.ndarray, 
                       sim: np.ndarray, 
                       **kwargs) -> Dict[str, float]:
        """
        Return component breakdown of APEX.
        
        This method is useful for diagnostics and understanding
        which aspects of the simulation need improvement.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'chronological_term': Timing-sensitive error term
            - 'ranked_term': FDC-based error term
            - 'weighted_error': Combined error before multipliers
            - 'bias_multiplier': Volume balance multiplier
            - 'relative_bias': Relative volume bias
            - 'dynamics_multiplier': Gradient correlation multiplier [NOVEL]
            - 'gradient_correlation': Correlation of first differences [NOVEL]
            - 'lag_multiplier': Timing offset multiplier [NOVEL]
            - 'optimal_lag': Detected timing offset [NOVEL]
            - 'apex_value': Final APEX value
        """
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        # Compute all components
        chron_term = self._compute_chronological_term(obs_clean, sim_clean)
        ranked_term = self._compute_ranked_term(obs_clean, sim_clean)
        weighted_error = self.alpha * chron_term + (1 - self.alpha) * ranked_term
        
        bias_mult, relative_bias = self._compute_bias_multiplier(obs_clean, sim_clean)
        dynamics_mult, gradient_corr = self._compute_dynamics_multiplier(obs_clean, sim_clean)
        lag_mult, optimal_lag = self._compute_lag_multiplier(obs_clean, sim_clean)
        
        apex_value = weighted_error * bias_mult * dynamics_mult * lag_mult
        
        return {
            # Core SDEB components
            'chronological_term': chron_term,
            'ranked_term': ranked_term,
            'weighted_error': weighted_error,
            
            # Bias multiplier
            'bias_multiplier': bias_mult,
            'relative_bias': relative_bias,
            
            # Novel: Dynamics multiplier
            'dynamics_multiplier': dynamics_mult,
            'gradient_correlation': gradient_corr,
            
            # Novel: Lag multiplier
            'lag_multiplier': lag_mult,
            'optimal_lag': optimal_lag,
            
            # Final value
            'apex_value': apex_value,
        }
    
    def __repr__(self) -> str:
        return (
            f"APEX(alpha={self.alpha}, transform={self._transform.transform_type}, "
            f"dynamics_strength={self.dynamics_strength}, lag_penalty={self.lag_penalty})"
        )
