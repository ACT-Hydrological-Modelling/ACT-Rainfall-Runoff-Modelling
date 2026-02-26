"""
Objective functions for model calibration.

This module provides a comprehensive set of objective functions for evaluating
rainfall-runoff model performance, including:
- Standard metrics (NSE, KGE, RMSE, PBIAS, SDEB)
- Flow duration curve metrics
- Weighted/composite objectives
- Hydrological signature metrics
- KGE variants (2009, 2012, 2021)
- Flow transformations for different flow emphasis

NOTE: For new code, prefer importing from pyrrm.objectives which provides
a more comprehensive and modern interface. This module maintains backward
compatibility with the original pyrrm.calibration interface.

Example (new style, recommended):
    >>> from pyrrm.objectives import NSE, KGE, FlowTransformation
    >>> nse = NSE()
    >>> kge_inv = KGE(transform=FlowTransformation('inverse'))

Example (legacy style, for backward compatibility):
    >>> from pyrrm.calibration.objective_functions import NSE, KGE
    >>> nse = NSE()
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
import numpy as np

# =============================================================================
# Re-exports from pyrrm.objectives (new interface)
# =============================================================================
# These provide access to the new objective functions while maintaining
# backward compatibility with the original module location.

try:
    # Import new implementations with aliases for backward compat
    from pyrrm.objectives import (
        # Base classes and utilities
        ObjectiveFunction as NewObjectiveFunction,
        MetricResult,
        FlowTransformation,
        evaluate_all as new_evaluate_all,
        print_evaluation_report,
        # Traditional metrics
        NSE as NewNSE,
        RMSE as NewRMSE,
        MAE as NewMAE,
        PBIAS as NewPBIAS,
        SDEB,
        # KGE family
        KGE as NewKGE,
        KGENonParametric,
        # Correlation
        PearsonCorrelation,
        SpearmanCorrelation,
        # FDC
        FDCMetric,
        compute_fdc,
        # Signatures
        SignatureMetric,
        # Composite
        WeightedObjective as NewWeightedObjective,
        kge_hilo,
        fdc_multisegment,
        comprehensive_objective,
        # Compatibility
        LegacyObjectiveAdapter,
        adapt_objective,
    )
    NEW_OBJECTIVES_AVAILABLE = True
except ImportError:
    NEW_OBJECTIVES_AVAILABLE = False


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.
    
    All objective functions must implement:
    - calculate(): Compute the objective value
    - maximize: Property indicating optimization direction
    """
    
    @abstractmethod
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """
        Calculate the objective function value.
        
        Args:
            simulated: Simulated values
            observed: Observed values
            
        Returns:
            Objective function value
        """
        pass
    
    @property
    @abstractmethod
    def maximize(self) -> bool:
        """True if function should be maximized, False for minimization."""
        pass
    
    @property
    def name(self) -> str:
        """Name of the objective function."""
        return self.__class__.__name__
    
    def __call__(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """Allow calling instance directly."""
        return self.calculate(simulated, observed)
    
    def for_calibration_legacy(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """
        Return value oriented for maximisation-based calibration.
        
        Maximisation objectives return the value as-is; minimisation
        objectives are negated so that all optimisers can maximise.
        """
        value = self.calculate(simulated, observed)
        return value if self.maximize else -value

    # Backward-compatible alias
    for_spotpy = for_calibration_legacy


# =============================================================================
# Standard Metrics
# =============================================================================

class NSE(ObjectiveFunction):
    """
    Nash-Sutcliffe Efficiency.
    
    NSE = 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)
    
    Range: (-inf, 1], where 1 is perfect match.
    """
    
    @property
    def maximize(self) -> bool:
        return True
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        obs_mean = np.mean(obs)
        numerator = np.sum((sim - obs) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1.0 - numerator / denominator


class KGE(ObjectiveFunction):
    """
    Kling-Gupta Efficiency.
    
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    
    where:
        r = correlation coefficient
        alpha = std(sim) / std(obs)
        beta = mean(sim) / mean(obs)
    
    Range: (-inf, 1], where 1 is perfect match.
    """
    
    @property
    def maximize(self) -> bool:
        return True
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        # Correlation
        if np.std(sim) == 0 or np.std(obs) == 0:
            r = 0.0
        else:
            r = np.corrcoef(sim, obs)[0, 1]
        
        # Variability ratio
        alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else 0
        
        # Bias ratio
        beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else 0
        
        return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


class RMSE(ObjectiveFunction):
    """
    Root Mean Square Error.
    
    RMSE = sqrt(mean((sim - obs)^2))
    
    Range: [0, inf), where 0 is perfect match.
    """
    
    @property
    def maximize(self) -> bool:
        return False
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        return np.sqrt(np.mean((sim - obs) ** 2))


class MAE(ObjectiveFunction):
    """
    Mean Absolute Error.
    
    MAE = mean(|sim - obs|)
    
    Range: [0, inf), where 0 is perfect match.
    """
    
    @property
    def maximize(self) -> bool:
        return False
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        return np.mean(np.abs(sim - obs))


class PBIAS(ObjectiveFunction):
    """
    Percent Bias (volume error).
    
    PBIAS = 100 * sum(sim - obs) / sum(obs)
    
    Range: (-inf, inf), where 0 is perfect match.
    Positive values indicate overestimation.
    """
    
    @property
    def maximize(self) -> bool:
        return False  # Minimize absolute bias
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0 or np.sum(obs) == 0:
            return np.nan
        
        return 100.0 * np.sum(sim - obs) / np.sum(obs)


class LogNSE(ObjectiveFunction):
    """
    Log-transformed Nash-Sutcliffe Efficiency.
    
    Better for evaluating low flows.
    Applies log transform before computing NSE.
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        Args:
            epsilon: Small value added before log transform to avoid log(0)
        """
        self.epsilon = epsilon
    
    @property
    def maximize(self) -> bool:
        return True
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        # Add epsilon and take log
        sim_log = np.log(sim + self.epsilon)
        obs_log = np.log(obs + self.epsilon)
        
        obs_mean = np.mean(obs_log)
        numerator = np.sum((sim_log - obs_log) ** 2)
        denominator = np.sum((obs_log - obs_mean) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1.0 - numerator / denominator


class GaussianLikelihood(ObjectiveFunction):
    """
    Gaussian likelihood with measurement error integrated out.
    
    This is the recommended likelihood function for DREAM calibration.
    Implements the Gaussian likelihood with measurement error integrated out:
    
        p = -n/2 * log(sum(e_t(x)^2))
    
    where e_t is the error residual (observed - simulated).
    
    Higher values indicate better fit (maximized).
    
    NOTE: This formulation heavily emphasizes HIGH FLOWS because squared
    errors in original units are dominated by high-flow periods. For balanced
    or low-flow emphasis, use TransformedGaussianLikelihood with an appropriate
    flow transformation.
    
    Reference:
        Vrugt 2016: Markov chain Monte Carlo simulation using the DREAM 
        software package.
    """
    
    @property
    def maximize(self) -> bool:
        return True
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return -np.inf
        
        # Calculate error residuals
        error = obs - sim
        sum_squared_error = np.sum(error ** 2)
        
        if sum_squared_error <= 0:
            return -np.inf
        
        # Gaussian likelihood with measurement error integrated out
        n = len(obs)
        return -n / 2.0 * np.log(sum_squared_error)
    
    def for_calibration_legacy(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """Already in log-likelihood form, directly usable by DREAM."""
        return self.calculate(simulated, observed)

    for_spotpy = for_calibration_legacy


class TransformedGaussianLikelihood(ObjectiveFunction):
    """
    Gaussian likelihood with flow transformation for balanced/low-flow emphasis.
    
    This class applies a flow transformation before computing the Gaussian
    log-likelihood, allowing MCMC calibration (PyDREAM) to
    emphasize different flow regimes - analogous to transformed NSE/KGE.
    
    Formula
    -------
    log_lik = -n/2 * log(Σ(T(obs) - T(sim))²)
    
    where T() is the flow transformation function.
    
    Parameters
    ----------
    transform : str or FlowTransformation, default='sqrt'
        Flow transformation to apply. Can be:
        - String: 'none', 'sqrt', 'log', 'inverse', 'power', 'boxcox'
        - FlowTransformation instance for full control
        
        Default is 'sqrt' for balanced flow emphasis.
    
    epsilon_value : float, default=0.01
        Value for handling zero flows (as fraction of mean observed flow).
        Only used when transform is a string.
    
    Flow Emphasis by Transformation
    -------------------------------
    | Transform | Emphasis | Equivalent to |
    |-----------|----------|---------------|
    | none | High flows | GaussianLikelihood, NSE |
    | sqrt | Balanced | NSE(sqrt) |
    | power (p=0.2) | Low-medium | NSE(power) |
    | log | Low flows | LogNSE |
    | inverse | Very low flows | InvNSE |
    
    Notes
    -----
    - For general-purpose calibration, 'sqrt' provides good balance
    - For baseflow/recession emphasis, use 'log' or 'inverse'
    - For peak flow focus, use 'none' (equivalent to GaussianLikelihood)
    
    WARNING: Log transformation can cause numerical issues with near-zero
    flows. The epsilon parameter helps mitigate this.
    
    References
    ----------
    Vrugt 2016: Markov chain Monte Carlo simulation using the DREAM 
    software package.
    
    Pushpalatha et al. (2012): A review of efficiency criteria suitable 
    for evaluating low-flow simulations. Journal of Hydrology.
    
    Examples
    --------
    >>> # Balanced likelihood (sqrt transform) - recommended default
    >>> likelihood = TransformedGaussianLikelihood('sqrt')
    
    >>> # Low-flow emphasis
    >>> likelihood = TransformedGaussianLikelihood('log')
    
    >>> # Very low-flow emphasis (inverse)
    >>> likelihood = TransformedGaussianLikelihood('inverse')
    
    >>> # Custom transformation with full control
    >>> from pyrrm.objectives import FlowTransformation
    >>> transform = FlowTransformation('power', p=0.3)
    >>> likelihood = TransformedGaussianLikelihood(transform=transform)
    """
    
    # Flow emphasis descriptions for documentation
    _TRANSFORM_EMPHASIS = {
        'none': 'high',
        'sqrt': 'balanced',
        'log': 'low',
        'inverse': 'very_low',
        'power': 'low_medium',
        'boxcox': 'balanced',
        'squared': 'very_high',
        'inverse_squared': 'extreme_low',
    }
    
    def __init__(
        self, 
        transform: Optional[str] = 'sqrt',
        epsilon_value: float = 0.01
    ):
        """
        Initialize transformed Gaussian likelihood.
        
        Parameters
        ----------
        transform : str or FlowTransformation, optional
            Flow transformation. String options: 'none', 'sqrt', 'log', 
            'inverse', 'power', 'boxcox'. Default is 'sqrt' for balanced
            flow emphasis.
        epsilon_value : float, default=0.01
            Epsilon value for zero-flow handling (fraction of mean flow).
        """
        # Handle FlowTransformation import
        try:
            from pyrrm.objectives import FlowTransformation
            self._FlowTransformation = FlowTransformation
        except ImportError:
            # Fallback if new objectives module not available
            self._FlowTransformation = None
        
        # Store original arguments for pickling/recreation
        self._transform_arg = transform
        self._epsilon_value = epsilon_value
        
        # Create or store the transformation
        if transform is None or transform == 'none':
            self._transform = None
            self._transform_type = 'none'
        elif isinstance(transform, str):
            if self._FlowTransformation is not None:
                self._transform = self._FlowTransformation(
                    transform_type=transform,
                    epsilon_method='mean_fraction',
                    epsilon_value=epsilon_value
                )
            else:
                # Fallback: store transform type for manual application
                self._transform = None
                self._transform_type = transform
            self._transform_type = transform
        else:
            # Assume it's a FlowTransformation instance
            self._transform = transform
            self._transform_type = getattr(transform, 'transform_type', 'custom')
        
        self._flow_emphasis = self._TRANSFORM_EMPHASIS.get(self._transform_type, 'custom')
    
    @property
    def maximize(self) -> bool:
        return True
    
    @property
    def flow_emphasis(self) -> str:
        """Return the flow regime this likelihood emphasizes."""
        return self._flow_emphasis
    
    @property
    def transform_type(self) -> str:
        """Return the transformation type."""
        return self._transform_type
    
    def __repr__(self) -> str:
        return f"TransformedGaussianLikelihood(transform='{self._transform_type}', emphasis='{self._flow_emphasis}')"
    
    def _apply_transform(self, values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Apply the flow transformation.
        
        Parameters
        ----------
        values : np.ndarray
            Values to transform (simulated or observed)
        reference : np.ndarray
            Reference values for epsilon calculation (typically observed)
        
        Returns
        -------
        np.ndarray
            Transformed values
        """
        if self._transform is not None:
            return self._transform.apply(values, reference)
        elif self._transform_type == 'none':
            return values
        else:
            # Manual transformation fallback
            eps = np.mean(reference[reference > 0]) * self._epsilon_value if np.any(reference > 0) else self._epsilon_value
            
            if self._transform_type == 'sqrt':
                return np.sqrt(values + eps)
            elif self._transform_type == 'log':
                return np.log(values + eps)
            elif self._transform_type == 'inverse':
                return 1.0 / (values + eps)
            elif self._transform_type == 'power':
                return (values + eps) ** 0.2
            elif self._transform_type == 'boxcox':
                lam = 0.25
                return ((values + eps) ** lam - 1) / lam
            elif self._transform_type == 'squared':
                return values ** 2
            elif self._transform_type == 'inverse_squared':
                return 1.0 / (values + eps) ** 2
            else:
                return values
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """
        Calculate transformed Gaussian log-likelihood.
        
        Parameters
        ----------
        simulated : np.ndarray
            Simulated flow values
        observed : np.ndarray
            Observed flow values
        
        Returns
        -------
        float
            Log-likelihood value (higher is better)
        """
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return -np.inf
        
        # Apply transformation
        obs_t = self._apply_transform(obs, obs)
        sim_t = self._apply_transform(sim, obs)  # Use obs as reference for epsilon
        
        # Check for invalid values after transformation
        if np.any(np.isnan(obs_t)) or np.any(np.isnan(sim_t)):
            return -np.inf
        if np.any(np.isinf(obs_t)) or np.any(np.isinf(sim_t)):
            return -np.inf
        
        # Calculate error residuals in transformed space
        error = obs_t - sim_t
        sum_squared_error = np.sum(error ** 2)
        
        if sum_squared_error <= 0:
            return -np.inf
        
        # Gaussian likelihood with measurement error integrated out
        n = len(obs)
        return -n / 2.0 * np.log(sum_squared_error)
    
    def for_calibration_legacy(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        """Already in log-likelihood form, directly usable by DREAM."""
        return self.calculate(simulated, observed)

    for_spotpy = for_calibration_legacy
    
    def __getstate__(self):
        """Support pickling for multiprocessing."""
        state = self.__dict__.copy()
        # Remove unpicklable FlowTransformation class reference
        state['_FlowTransformation'] = None
        return state
    
    def __setstate__(self, state):
        """Support unpickling for multiprocessing."""
        self.__dict__.update(state)
        # Recreate transformation on unpickle
        try:
            from pyrrm.objectives import FlowTransformation
            self._FlowTransformation = FlowTransformation
            if self._transform_arg is not None and isinstance(self._transform_arg, str) and self._transform_arg != 'none':
                self._transform = FlowTransformation(
                    transform_type=self._transform_arg,
                    epsilon_method='mean_fraction',
                    epsilon_value=self._epsilon_value
                )
        except ImportError:
            self._FlowTransformation = None


# =============================================================================
# Flow Duration Curve Metrics
# =============================================================================

class FDCError(ObjectiveFunction):
    """
    Flow Duration Curve Error.
    
    Computes error between simulated and observed flow duration curves
    at specified exceedance probabilities.
    """
    
    def __init__(
        self, 
        exceedance_points: List[float] = None,
        use_log: bool = True
    ):
        """
        Args:
            exceedance_points: List of exceedance probabilities (0-1)
                              Default: [0.02, 0.1, 0.2, 0.5, 0.7, 0.9, 0.98]
            use_log: If True, compute error in log space
        """
        self.exceedance_points = exceedance_points or [0.02, 0.1, 0.2, 0.5, 0.7, 0.9, 0.98]
        self.use_log = use_log
    
    @property
    def maximize(self) -> bool:
        return False
    
    def _compute_fdc(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute flow duration curve."""
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        exceedance = np.arange(1, n + 1) / (n + 1)
        return exceedance, sorted_data
    
    def _interpolate_fdc(
        self, 
        exceedance: np.ndarray, 
        values: np.ndarray, 
        target_exc: List[float]
    ) -> np.ndarray:
        """Interpolate FDC at target exceedance probabilities."""
        return np.interp(target_exc, exceedance, values)
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        # Compute FDCs
        exc_sim, val_sim = self._compute_fdc(sim)
        exc_obs, val_obs = self._compute_fdc(obs)
        
        # Interpolate at target points
        sim_at_points = self._interpolate_fdc(exc_sim, val_sim, self.exceedance_points)
        obs_at_points = self._interpolate_fdc(exc_obs, val_obs, self.exceedance_points)
        
        if self.use_log:
            # Add small value to avoid log(0)
            epsilon = 0.001
            sim_at_points = np.log(sim_at_points + epsilon)
            obs_at_points = np.log(obs_at_points + epsilon)
        
        # RMSE of FDC points
        return np.sqrt(np.mean((sim_at_points - obs_at_points) ** 2))


# =============================================================================
# Hydrological Signatures
# =============================================================================

class FlowSignatureError(ObjectiveFunction):
    """
    Error in hydrological flow signatures.
    
    Computes error between simulated and observed flow signatures:
    - Q10, Q50, Q90 (flow percentiles)
    - Baseflow Index (BFI)
    - High/Low flow frequency
    - Rising/Falling limb density
    """
    
    def __init__(
        self, 
        signatures: List[str] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            signatures: List of signatures to compute
                       Options: 'Q10', 'Q50', 'Q90', 'mean', 'cv', 'high_freq', 'low_freq'
            weights: Weights for each signature (default: equal weights)
        """
        self.signatures = signatures or ['Q10', 'Q50', 'Q90', 'mean', 'cv']
        self.weights = weights or [1.0] * len(self.signatures)
        
        if len(self.weights) != len(self.signatures):
            raise ValueError("Number of weights must match number of signatures")
    
    @property
    def maximize(self) -> bool:
        return False
    
    def _compute_signature(self, data: np.ndarray, sig_name: str) -> float:
        """Compute a single signature."""
        if sig_name == 'Q10':
            return np.percentile(data, 90)  # High flow
        elif sig_name == 'Q50':
            return np.percentile(data, 50)  # Median flow
        elif sig_name == 'Q90':
            return np.percentile(data, 10)  # Low flow
        elif sig_name == 'mean':
            return np.mean(data)
        elif sig_name == 'cv':
            return np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
        elif sig_name == 'high_freq':
            threshold = np.percentile(data, 90)
            return np.sum(data > threshold) / len(data)
        elif sig_name == 'low_freq':
            threshold = np.percentile(data, 10)
            return np.sum(data < threshold) / len(data)
        else:
            raise ValueError(f"Unknown signature: {sig_name}")
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        sim = np.asarray(simulated).flatten()
        obs = np.asarray(observed).flatten()
        
        mask = ~(np.isnan(sim) | np.isnan(obs))
        sim, obs = sim[mask], obs[mask]
        
        if len(obs) == 0:
            return np.nan
        
        total_error = 0.0
        total_weight = 0.0
        
        for sig, weight in zip(self.signatures, self.weights):
            sim_sig = self._compute_signature(sim, sig)
            obs_sig = self._compute_signature(obs, sig)
            
            if obs_sig != 0:
                rel_error = abs(sim_sig - obs_sig) / abs(obs_sig)
            else:
                rel_error = abs(sim_sig - obs_sig)
            
            total_error += weight * rel_error
            total_weight += weight
        
        return total_error / total_weight if total_weight > 0 else np.nan


# =============================================================================
# Composite Objectives
# =============================================================================

class WeightedObjective(ObjectiveFunction):
    """
    Weighted combination of multiple objective functions.
    
    Example:
        >>> obj = WeightedObjective([
        ...     (NSE(), 0.5),
        ...     (PBIAS(), 0.3),
        ...     (FDCError(), 0.2)
        ... ])
    """
    
    def __init__(
        self, 
        objectives: List[Tuple[ObjectiveFunction, float]],
        normalize: bool = True
    ):
        """
        Args:
            objectives: List of (objective_function, weight) tuples
            normalize: If True, normalize values before combining
        """
        self.objectives = objectives
        self.normalize = normalize
        self._direction = None
    
    @property
    def maximize(self) -> bool:
        # Composite objectives are always returned for maximization
        return True
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        total = 0.0
        total_weight = 0.0
        
        for obj, weight in self.objectives:
            value = obj.calculate(simulated, observed)
            
            if np.isnan(value):
                continue
            
            # Normalize: convert to 0-1 scale where 1 is best
            if self.normalize:
                if _should_maximize(obj):
                    # For NSE, KGE: clip to [-1, 1] then scale to [0, 1]
                    value = (np.clip(value, -1, 1) + 1) / 2
                else:
                    # For RMSE, MAE, PBIAS: use exponential decay
                    value = np.exp(-abs(value) / 10)
            else:
                # Just flip sign for minimization objectives
                if not _should_maximize(obj):
                    value = -value
            
            total += weight * value
            total_weight += weight
        
        return total / total_weight if total_weight > 0 else np.nan


class CustomObjective(ObjectiveFunction):
    """
    Custom objective function from user-provided callable.
    
    Example:
        >>> def my_objective(sim, obs):
        ...     return 1 - np.mean(np.abs(sim - obs) / obs)
        >>> obj = CustomObjective(my_objective, maximize=True)
    """
    
    def __init__(
        self, 
        func: Callable[[np.ndarray, np.ndarray], float],
        maximize: bool = True,
        name: str = "custom"
    ):
        """
        Args:
            func: Callable that takes (simulated, observed) and returns float
            maximize: Whether to maximize (True) or minimize (False)
            name: Name for the objective function
        """
        self._func = func
        self._maximize = maximize
        self._name = name
    
    @property
    def maximize(self) -> bool:
        return self._maximize
    
    @property
    def name(self) -> str:
        return self._name
    
    def calculate(self, simulated: np.ndarray, observed: np.ndarray) -> float:
        return self._func(simulated, observed)


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_metrics(
    simulated: np.ndarray,
    observed: np.ndarray
) -> dict:
    """
    Calculate a basic set of 6 performance metrics.

    .. deprecated::
        Use ``pyrrm.analysis.diagnostics.compute_diagnostics`` instead for the
        full 48-metric canonical diagnostic suite (NSE variants, KGE variants
        with components, KGE non-parametric variants, error metrics, FDC volume
        biases, raw signatures, and hydrological signature errors).

    Args:
        simulated: Simulated values
        observed: Observed values

    Returns:
        Dictionary with metric names and values (NSE, KGE, RMSE, MAE, PBIAS,
        LogNSE).
    """
    metrics = {
        'NSE': NSE(),
        'KGE': KGE(),
        'RMSE': RMSE(),
        'MAE': MAE(),
        'PBIAS': PBIAS(),
        'LogNSE': LogNSE(),
    }

    return {name: obj.calculate(simulated, observed) for name, obj in metrics.items()}


# =============================================================================
# Interface Compatibility Utilities
# =============================================================================

def is_new_interface(objective) -> bool:
    """
    Check if an objective uses the new pyrrm.objectives interface.
    
    The new interface has:
    - direction attribute ('maximize' or 'minimize')
    - for_calibration(simulated, observed) method
    
    The legacy interface has:
    - maximize property (bool)
    - for_calibration_legacy(simulated, observed) method
    
    Args:
        objective: Objective function to check
        
    Returns:
        True if using new interface, False for legacy
    """
    return hasattr(objective, 'direction')


def _should_maximize(objective) -> bool:
    """
    Check if an objective function should be maximized.
    
    Handles both old interface (maximize property) and new interface (direction attribute).
    
    Args:
        objective: Objective function to check
        
    Returns:
        True if objective should be maximized, False if minimized
    """
    if hasattr(objective, 'direction'):
        return objective.direction == 'maximize'
    elif hasattr(objective, 'maximize'):
        return objective.maximize
    else:
        return True  # Default to maximize


def get_calibration_value(objective, simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    Get calibration-ready value from either old or new interface.
    
    This utility function handles both interfaces, returning a value
    suitable for maximisation-based optimisation.
    
    Args:
        objective: Objective function (old or new interface)
        simulated: Simulated values
        observed: Observed values
        
    Returns:
        Objective value suitable for maximisation
    """
    if is_new_interface(objective):
        return objective.for_calibration(simulated, observed)
    else:
        return objective.for_calibration_legacy(simulated, observed)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
# These aliases provide the old names for the new FDC and Signature classes

if NEW_OBJECTIVES_AVAILABLE:
    # Alias for backward compatibility
    FDCError_new = FDCMetric
    FlowSignatureError_new = SignatureMetric
