# Hydrological Objective Functions Library
## Technical Specification Document

**Version:** 1.0.0  
**Date:** January 2025  
**Status:** Draft Specification  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Scope](#2-project-goals-and-scope)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Specifications](#4-module-specifications)
5. [Core Classes](#5-core-classes)
6. [Metric Implementations](#6-metric-implementations)
7. [Flow Transformations](#7-flow-transformations)
8. [Flow Duration Curve Metrics](#8-flow-duration-curve-metrics)
9. [Hydrological Signatures](#9-hydrological-signatures)
10. [Composite Objective Functions](#10-composite-objective-functions)
11. [Utility Functions](#11-utility-functions)
12. [Testing Requirements](#12-testing-requirements)
13. [Documentation Requirements](#13-documentation-requirements)
14. [Dependencies](#14-dependencies)
15. [Mathematical Reference](#15-mathematical-reference)
16. [Literature References](#16-literature-references)

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies the design and implementation requirements for a Python library that enables construction of custom weighted objective functions for rainfall-runoff model calibration. The library goes beyond traditional metrics (RMSE, NSE) to support:

- High-flow and low-flow specific calibration
- Bias control
- Flow duration curve matching
- Hydrological signature-based evaluation
- Flexible weighted combinations of multiple objectives

### 1.2 Target Users

- Hydrological modelers calibrating rainfall-runoff models
- Researchers developing new calibration approaches
- Water resource engineers evaluating model performance
- Developers of hydrological modeling frameworks

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| Modular metrics | Plug-and-play objective function components |
| Flow transformations | Shift emphasis between high/low flows |
| FDC-based metrics | Calibrate to flow frequency distributions |
| Weighted composites | Combine multiple objectives with custom weights |
| Signature library | Process-based hydrological indices |
| Vectorized computation | Efficient numpy-based calculations |

---

## 2. Project Goals and Scope

### 2.1 In Scope

1. **Traditional Metrics**
   - Nash-Sutcliffe Efficiency (NSE)
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - Percent Bias (PBIAS)
   - Correlation coefficients (Pearson, Spearman)

2. **Kling-Gupta Efficiency Family**
   - KGE 2009 (original)
   - KGE' 2012 (modified)
   - KGE'' 2021 (for near-zero means)
   - Non-parametric KGE
   - Weighted KGE with custom component scaling

3. **Flow Transformations**
   - Square root, logarithmic, inverse, power, Box-Cox
   - Zero-flow handling with configurable epsilon
   - Warnings for problematic combinations

4. **Flow Duration Curve Metrics**
   - Segment-based evaluation (peak, high, mid, low, very low)
   - Volume bias, slope, RMSE metrics
   - Custom segment definitions

5. **Hydrological Signatures**
   - Flow percentiles (Q5, Q10, Q50, Q75, Q90, Q95)
   - Runoff ratio, baseflow index
   - Flashiness index
   - Flow frequency metrics

6. **Composite Objectives**
   - Weighted sum aggregation
   - Weighted product aggregation
   - Minimum value aggregation
   - Normalization options

### 2.2 Out of Scope (Future Versions)

- Multi-objective Pareto optimization algorithms
- Automatic calibration routines
- Uncertainty quantification
- Seasonal/event-based subsetting
- Spatial (multi-site) metrics

---

## 3. Architecture Overview

### 3.1 Package Structure

```
hydro_objectives/
├── __init__.py                 # Package exports
├── core/
│   ├── __init__.py
│   ├── base.py                 # Abstract base classes
│   ├── result.py               # MetricResult container
│   └── utils.py                # Utility functions
├── metrics/
│   ├── __init__.py
│   ├── traditional.py          # NSE, RMSE, MAE, PBIAS
│   ├── kge.py                  # KGE family
│   └── correlation.py          # Correlation metrics
├── transformations/
│   ├── __init__.py
│   └── flow_transforms.py      # Flow transformation classes
├── fdc/
│   ├── __init__.py
│   ├── curves.py               # FDC computation utilities
│   └── metrics.py              # FDC-based objective functions
├── signatures/
│   ├── __init__.py
│   ├── flow_indices.py         # Flow percentiles, statistics
│   ├── dynamics.py             # Flashiness, recession, etc.
│   └── water_balance.py        # Runoff ratio, baseflow index
├── composite/
│   ├── __init__.py
│   ├── weighted.py             # Weighted combination builder
│   └── factories.py            # Convenience factory functions
└── tests/
    ├── __init__.py
    ├── test_traditional.py
    ├── test_kge.py
    ├── test_transforms.py
    ├── test_fdc.py
    ├── test_signatures.py
    ├── test_composite.py
    └── fixtures.py             # Test data generators
```

### 3.2 Design Principles

1. **Composability**: All objective functions share a common interface and can be combined
2. **Immutability**: Objective function instances are configured at creation, not modified
3. **Vectorization**: Use numpy operations for computational efficiency
4. **Defensive programming**: Validate inputs, handle edge cases gracefully
5. **Documentation**: Every public class/function has docstrings with examples
6. **Type hints**: Full type annotation for IDE support and static analysis

### 3.3 Class Hierarchy

```
ObjectiveFunction (ABC)
├── NSE
├── RMSE
├── MAE
├── PBIAS
├── KGE
├── KGENonParametric
├── FDCMetric
├── SignatureMetric
└── WeightedObjective
```

---

## 4. Module Specifications

### 4.1 `hydro_objectives.core.base`

#### ObjectiveFunction (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class ObjectiveFunction(ABC):
    """
    Abstract base class for all objective functions.
    
    All concrete objective functions must inherit from this class
    and implement the __call__ method.
    
    Attributes
    ----------
    name : str
        Human-readable name for the objective function
    direction : str
        Optimization direction: 'maximize' or 'minimize'
    optimal_value : float
        The optimal (best possible) value for this metric
    
    Methods
    -------
    __call__(obs, sim, **kwargs) -> float
        Calculate the objective function value (abstract)
    evaluate(obs, sim, **kwargs) -> MetricResult
        Calculate value and return with components
    get_components(obs, sim, **kwargs) -> Optional[Dict[str, float]]
        Return component breakdown if applicable
    """
    
    def __init__(self,
                 name: str,
                 direction: str = 'maximize',
                 optimal_value: float = 1.0):
        """
        Parameters
        ----------
        name : str
            Descriptive name for the objective function
        direction : str
            'maximize' or 'minimize'
        optimal_value : float
            The optimal (best) value for this metric
            
        Raises
        ------
        ValueError
            If direction is not 'maximize' or 'minimize'
        """
        if direction not in ('maximize', 'minimize'):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{direction}'")
        
        self.name = name
        self.direction = direction
        self.optimal_value = optimal_value
    
    @abstractmethod
    def __call__(self, 
                 obs: np.ndarray, 
                 sim: np.ndarray,
                 **kwargs) -> float:
        """
        Calculate the objective function value.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values (1D array)
        sim : np.ndarray
            Simulated values (1D array, same length as obs)
        **kwargs : dict
            Additional parameters specific to the metric
            
        Returns
        -------
        float
            Objective function value
            
        Raises
        ------
        ValueError
            If obs and sim have different lengths
        """
        pass
    
    def evaluate(self, 
                 obs: np.ndarray, 
                 sim: np.ndarray, 
                 **kwargs) -> 'MetricResult':
        """
        Evaluate and return a MetricResult with components.
        
        This method calls __call__ and get_components, packaging
        the results into a MetricResult object.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result object containing value and component breakdown
        """
        value = self(obs, sim, **kwargs)
        components = self.get_components(obs, sim, **kwargs) or {}
        return MetricResult(value=value, components=components, name=self.name)
    
    def get_components(self, 
                       obs: np.ndarray, 
                       sim: np.ndarray,
                       **kwargs) -> Optional[Dict[str, float]]:
        """
        Return component breakdown (if applicable).
        
        Override in subclasses for multi-component metrics like KGE.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        dict or None
            Dictionary of component names and values, or None
        """
        return None
    
    def _validate_inputs(self, obs: np.ndarray, sim: np.ndarray) -> None:
        """
        Validate input arrays.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
            
        Raises
        ------
        ValueError
            If inputs are invalid
        TypeError
            If inputs are not array-like
        """
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        if not isinstance(sim, np.ndarray):
            sim = np.asarray(sim)
        
        if obs.ndim != 1 or sim.ndim != 1:
            raise ValueError("obs and sim must be 1-dimensional arrays")
        
        if len(obs) != len(sim):
            raise ValueError(f"obs and sim must have same length, got {len(obs)} and {len(sim)}")
        
        if len(obs) == 0:
            raise ValueError("obs and sim cannot be empty")
    
    def _clean_data(self, 
                    obs: np.ndarray, 
                    sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove NaN values using pairwise deletion.
        
        Parameters
        ----------
        obs : np.ndarray
            Observed values
        sim : np.ndarray
            Simulated values
            
        Returns
        -------
        tuple of np.ndarray
            Cleaned obs and sim arrays with NaN pairs removed
            
        Raises
        ------
        ValueError
            If all values are NaN
        """
        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs_clean = obs[mask]
        sim_clean = sim[mask]
        
        if len(obs_clean) == 0:
            raise ValueError("No valid (non-NaN) data pairs")
        
        return obs_clean, sim_clean
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self) -> str:
        return self.name
```

### 4.2 `hydro_objectives.core.result`

#### MetricResult (Data Container)

```python
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class MetricResult:
    """
    Container for metric evaluation results.
    
    Attributes
    ----------
    value : float
        The primary metric value
    components : dict
        Dictionary of component values (e.g., r, alpha, beta for KGE)
    name : str
        Name of the metric
    
    Examples
    --------
    >>> result = MetricResult(value=0.85, components={'r': 0.95, 'alpha': 0.9, 'beta': 1.0}, name='KGE')
    >>> print(result)
    KGE=0.8500 (r=0.950, alpha=0.900, beta=1.000)
    >>> result.value
    0.85
    >>> result.components['r']
    0.95
    """
    
    value: float
    components: Dict[str, float] = field(default_factory=dict)
    name: str = ""
    
    def __repr__(self) -> str:
        if self.components:
            # Filter out internal statistics (mu, sigma) for cleaner output
            display_components = {k: v for k, v in self.components.items() 
                                  if not k.startswith(('mu_', 'sigma_'))}
            if display_components:
                comp_str = ", ".join(f"{k}={v:.3f}" for k, v in display_components.items())
                return f"{self.name}={self.value:.4f} ({comp_str})"
        return f"{self.name}={self.value:.4f}"
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary including value and all components.
        
        Returns
        -------
        dict
            Dictionary with 'value' key and all component keys
        """
        result = {'value': self.value}
        result.update(self.components)
        return result
```

---

## 5. Core Classes

### 5.1 Constants and Configuration

```python
# hydro_objectives/core/constants.py

# FDC segment definitions (exceedance probability bounds)
FDC_SEGMENTS = {
    'peak': (0.00, 0.02),      # 0-2%: Peak flows from large events
    'high': (0.02, 0.20),      # 2-20%: High flows, quick runoff
    'mid': (0.20, 0.70),       # 20-70%: Mid-range, intermediate response
    'low': (0.70, 0.95),       # 70-95%: Low flows, baseflow
    'very_low': (0.95, 1.00),  # 95-100%: Very low flows, drought
    'all': (0.00, 1.00),       # Full range
}

# KGE benchmark value (Knoben et al., 2019)
# KGE values above this indicate improvement over mean flow benchmark
KGE_BENCHMARK = -0.41

# Default epsilon fraction for zero-flow handling
DEFAULT_EPSILON_FRACTION = 0.01

# Transformation types and their flow emphasis
TRANSFORM_EMPHASIS = {
    'none': 'high',
    'squared': 'very_high',
    'sqrt': 'balanced',
    'power': 'low_medium',
    'log': 'low',
    'boxcox': 'balanced',
    'inverse': 'low',
    'inverse_squared': 'very_low',
}
```

---

## 6. Metric Implementations

### 6.1 Traditional Metrics

#### 6.1.1 Nash-Sutcliffe Efficiency (NSE)

```python
# hydro_objectives/metrics/traditional.py

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
        obs, sim = self._clean_data(obs, sim)
        
        # Apply transformation if specified
        if self.transform is not None:
            obs = self.transform.apply(obs, obs)
            sim = self.transform.apply(sim, obs)
        
        # Calculate NSE
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1.0 - numerator / denominator
```

#### 6.1.2 RMSE, MAE, PBIAS

```python
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
        obs, sim = self._clean_data(obs, sim)
        
        if self.transform is not None:
            obs = self.transform.apply(obs, obs)
            sim = self.transform.apply(sim, obs)
        
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        
        if self.normalized:
            mean_obs = np.mean(obs)
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
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'MAE'
        if transform is not None and transform.transform_type != 'none':
            name = f'MAE({transform.transform_type})'
        
        super().__init__(name=name, direction='minimize', optimal_value=0.0)
        self.transform = transform
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs, sim = self._clean_data(obs, sim)
        
        if self.transform is not None:
            obs = self.transform.apply(obs, obs)
            sim = self.transform.apply(sim, obs)
        
        return np.mean(np.abs(obs - sim))


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
    """
    
    def __init__(self):
        super().__init__(name='PBIAS', direction='minimize', optimal_value=0.0)
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        obs, sim = self._clean_data(obs, sim)
        
        sum_obs = np.sum(obs)
        if sum_obs == 0:
            return np.nan
        
        return 100.0 * np.sum(sim - obs) / sum_obs
```

### 6.2 Kling-Gupta Efficiency

```python
# hydro_objectives/metrics/kge.py

class KGE(ObjectiveFunction):
    """
    Kling-Gupta Efficiency with configurable variant and component weights.
    
    Formula
    -------
    KGE = 1 - √[(sr·(r-1))² + (sα·(α-1))² + (sβ·(β-1))²]
    
    where:
        r = Pearson correlation coefficient
        α = variability ratio (depends on variant)
        β = bias ratio (depends on variant)
        sr, sα, sβ = scaling factors (weights)
    
    Variants
    --------
    '2009' (Gupta et al., 2009):
        α = σsim / σobs
        β = μsim / μobs
        
    '2012' (Kling et al., 2012):
        α = (σsim/μsim) / (σobs/μobs) = CVsim / CVobs
        β = μsim / μobs
        
    '2021' (Tang et al., 2021):
        α = σsim / σobs
        β = (μsim - μobs) / σobs
    
    Parameters
    ----------
    variant : str, default='2012'
        KGE variant: '2009', '2012', or '2021'
    weights : tuple of float, default=(1.0, 1.0, 1.0)
        Scaling factors (sr, sα, sβ) for each component
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    
    Notes
    -----
    - KGE > -0.41 indicates improvement over mean benchmark (Knoben et al., 2019)
    - Avoid log transformation with KGE (Santos et al., 2018)
    - Use inverse or sqrt transforms for low-flow emphasis
    - The 2012 variant ensures bias and variability are not cross-correlated
    - The 2021 variant handles near-zero means better
    
    References
    ----------
    Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). Decomposition 
    of the mean squared error and NSE performance criteria: Implications for 
    improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.
    
    Kling, H., Fuchs, M., Paulin, M. (2012). Runoff conditions in the upper 
    Danube basin under an ensemble of climate change scenarios. Journal of 
    Hydrology, 424, 264-277.
    
    Tang, G., Clark, M.P., Papalexiou, S.M. (2021). SC-earth: a station-based 
    serially complete earth dataset from 1950 to 2019. Journal of Climate, 
    34(16), 6493-6511.
    
    Examples
    --------
    >>> # Standard KGE (2012 variant)
    >>> kge = KGE()
    >>> result = kge.evaluate(obs, sim)
    >>> print(f"KGE = {result.value:.3f}")
    >>> print(f"Correlation = {result.components['r']:.3f}")
    
    >>> # KGE with custom weights emphasizing correlation
    >>> kge_corr = KGE(weights=(2.0, 1.0, 1.0))
    
    >>> # KGE on inverse flows for low-flow emphasis
    >>> kge_inv = KGE(transform=FlowTransformation('inverse'))
    """
    
    VARIANTS = ('2009', '2012', '2021')
    
    def __init__(self,
                 variant: str = '2012',
                 weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 transform: Optional['FlowTransformation'] = None):
        
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got '{variant}'")
        
        if len(weights) != 3:
            raise ValueError(f"weights must have 3 elements, got {len(weights)}")
        
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        
        # Build name
        name = f'KGE_{variant}'
        if transform is not None and transform.transform_type != 'none':
            name = f'{name}({transform.transform_type})'
        if weights != (1.0, 1.0, 1.0):
            name = f'{name}_weighted'
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        
        self.variant = variant
        self.weights = weights
        self.transform = transform
        
        # Warn about log transformation
        if transform is not None and transform.transform_type == 'log':
            import warnings
            warnings.warn(
                "Log transformation with KGE can cause numerical instabilities "
                "and unit-dependence (Santos et al., 2018). Consider using "
                "'sqrt' or 'inverse' transformation instead.",
                UserWarning
            )
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        components = self.get_components(obs, sim, **kwargs)
        
        if components is None:
            return np.nan
        
        sr, sa, sb = self.weights
        r = components['r']
        alpha = components['alpha']
        beta = components['beta']
        
        # Euclidean distance from ideal point (1, 1, 1)
        ed = np.sqrt(
            (sr * (r - 1)) ** 2 +
            (sa * (alpha - 1)) ** 2 +
            (sb * (beta - 1)) ** 2
        )
        
        return 1.0 - ed
    
    def get_components(self, 
                       obs: np.ndarray, 
                       sim: np.ndarray, 
                       **kwargs) -> Optional[Dict[str, float]]:
        """
        Calculate KGE components.
        
        Returns
        -------
        dict or None
            Dictionary with keys:
            - 'r': Pearson correlation coefficient
            - 'alpha': Variability ratio
            - 'beta': Bias ratio
            - 'mu_obs', 'mu_sim': Mean values
            - 'sigma_obs', 'sigma_sim': Standard deviations
        """
        self._validate_inputs(obs, sim)
        obs_clean, sim_clean = self._clean_data(obs, sim)
        
        if len(obs_clean) < 2:
            return None
        
        # Apply transformation if specified
        if self.transform is not None:
            obs_t = self.transform.apply(obs_clean, obs_clean)
            sim_t = self.transform.apply(sim_clean, obs_clean)
        else:
            obs_t = obs_clean
            sim_t = sim_clean
        
        # Basic statistics
        mu_obs = np.mean(obs_t)
        mu_sim = np.mean(sim_t)
        sigma_obs = np.std(obs_t, ddof=0)
        sigma_sim = np.std(sim_t, ddof=0)
        
        # Check for degenerate cases
        if sigma_obs == 0:
            return None
        if self.variant == '2012' and (mu_obs == 0 or mu_sim == 0):
            return None
        
        # Correlation
        r = np.corrcoef(obs_t, sim_t)[0, 1]
        if np.isnan(r):
            r = 0.0  # Handle constant arrays
        
        # Variability ratio (depends on variant)
        if self.variant == '2009':
            alpha = sigma_sim / sigma_obs
        elif self.variant == '2012':
            cv_obs = sigma_obs / mu_obs
            cv_sim = sigma_sim / mu_sim
            alpha = cv_sim / cv_obs
        elif self.variant == '2021':
            alpha = sigma_sim / sigma_obs
        
        # Bias ratio (depends on variant)
        if self.variant in ('2009', '2012'):
            beta = mu_sim / mu_obs
        elif self.variant == '2021':
            beta = (mu_sim - mu_obs) / sigma_obs
        
        return {
            'r': r,
            'alpha': alpha,
            'beta': beta,
            'mu_obs': mu_obs,
            'mu_sim': mu_sim,
            'sigma_obs': sigma_obs,
            'sigma_sim': sigma_sim,
        }
```

---

## 7. Flow Transformations

```python
# hydro_objectives/transformations/flow_transforms.py

class FlowTransformation:
    """
    Apply mathematical transformations to streamflow data.
    
    Transformations shift the emphasis of objective functions between
    high and low flows by changing the relative weight of errors at
    different flow magnitudes.
    
    Parameters
    ----------
    transform_type : str
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
    | squared | Very high | Peak flow estimation |
    | none | High | Flood forecasting |
    | sqrt | Balanced | General purpose |
    | boxcox | Balanced | Adaptive applications |
    | power | Low-medium | Low flow studies |
    | log | Low | Baseflow applications |
    | inverse | Low | Low flow indices |
    | inverse_squared | Very low | Drought indices |
    
    WARNING: Log transformation with KGE causes numerical issues.
    Use sqrt or inverse instead (Santos et al., 2018).
    
    References
    ----------
    Pushpalatha, R. et al. (2012). A review of efficiency criteria suitable 
    for evaluating low-flow simulations. Journal of Hydrology, 420, 171-182.
    
    Santos, L. et al. (2018). Technical note: Pitfalls in using log-transformed 
    flows within the KGE criterion. HESS, 22, 4583-4591.
    
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
    _TRANSFORMS = {
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
    
    _FLOW_EMPHASIS = {
        'none': 'high',
        'squared': 'very_high',
        'sqrt': 'balanced',
        'power': 'low_medium',
        'log': 'low',
        'boxcox': 'balanced',
        'inverse': 'low',
        'inverse_squared': 'very_low',
    }
    
    def __init__(self,
                 transform_type: str = 'none',
                 epsilon_method: str = 'mean_fraction',
                 epsilon_value: float = 0.01,
                 **params):
        
        if transform_type not in self._TRANSFORMS:
            raise ValueError(
                f"Unknown transform_type '{transform_type}'. "
                f"Available: {list(self._TRANSFORMS.keys())}"
            )
        
        if epsilon_method not in ('mean_fraction', 'fixed', 'min_nonzero'):
            raise ValueError(
                f"Unknown epsilon_method '{epsilon_method}'. "
                f"Available: 'mean_fraction', 'fixed', 'min_nonzero'"
            )
        
        if epsilon_value <= 0:
            raise ValueError("epsilon_value must be positive")
        
        self.transform_type = transform_type
        self.epsilon_method = epsilon_method
        self.epsilon_value = epsilon_value
        self.params = params
    
    @property
    def flow_emphasis(self) -> str:
        """Return which flow regime this transformation emphasizes."""
        return self._FLOW_EMPHASIS.get(self.transform_type, 'unknown')
    
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
        if not isinstance(other, FlowTransformation):
            return False
        return (self.transform_type == other.transform_type and
                self.epsilon_method == other.epsilon_method and
                self.epsilon_value == other.epsilon_value and
                self.params == other.params)
```

---

## 8. Flow Duration Curve Metrics

```python
# hydro_objectives/fdc/metrics.py

class FDCMetric(ObjectiveFunction):
    """
    Flow Duration Curve based objective functions.
    
    The FDC summarizes the frequency distribution of streamflow magnitudes,
    independent of timing. FDC-based metrics enable calibration focused on
    reproducing statistical flow characteristics.
    
    Parameters
    ----------
    segment : str
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
    
    SEGMENTS = {
        'peak': (0.00, 0.02),
        'high': (0.02, 0.20),
        'mid': (0.20, 0.70),
        'low': (0.70, 0.95),
        'very_low': (0.95, 1.00),
        'all': (0.00, 1.00),
    }
    
    METRICS = ('volume_bias', 'slope', 'rmse', 'correlation')
    
    def __init__(self,
                 segment: str = 'all',
                 metric: str = 'volume_bias',
                 log_transform: bool = False,
                 custom_bounds: Optional[tuple[float, float]] = None):
        
        # Validate segment
        if custom_bounds is None and segment not in self.SEGMENTS:
            raise ValueError(
                f"Unknown segment '{segment}'. "
                f"Available: {list(self.SEGMENTS.keys())} or use custom_bounds"
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
        self.bounds = custom_bounds if custom_bounds else self.SEGMENTS[segment]
    
    @staticmethod
    def compute_fdc(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute flow duration curve.
        
        Parameters
        ----------
        Q : np.ndarray
            Flow time series
        
        Returns
        -------
        exceedance : np.ndarray
            Exceedance probabilities (0 to 1)
        flows : np.ndarray
            Sorted flows (descending order)
        """
        Q_clean = Q[~np.isnan(Q)]
        Q_sorted = np.sort(Q_clean)[::-1]  # Descending order
        n = len(Q_sorted)
        
        # Weibull plotting position
        exceedance = np.arange(1, n + 1) / (n + 1)
        
        return exceedance, Q_sorted
    
    def _get_segment_mask(self, exceedance: np.ndarray) -> np.ndarray:
        """Get boolean mask for the specified FDC segment."""
        lower, upper = self.bounds
        return (exceedance >= lower) & (exceedance <= upper)
    
    def __call__(self, obs: np.ndarray, sim: np.ndarray, **kwargs) -> float:
        self._validate_inputs(obs, sim)
        
        # Compute FDCs
        exc_obs, fdc_obs = self.compute_fdc(obs)
        exc_sim, fdc_sim = self.compute_fdc(sim)
        
        # Get segment data
        mask_obs = self._get_segment_mask(exc_obs)
        mask_sim = self._get_segment_mask(exc_sim)
        
        fdc_obs_seg = fdc_obs[mask_obs]
        fdc_sim_seg = fdc_sim[mask_sim]
        
        # Handle length mismatch
        n = min(len(fdc_obs_seg), len(fdc_sim_seg))
        if n == 0:
            return np.nan
        
        fdc_obs_seg = fdc_obs_seg[:n]
        fdc_sim_seg = fdc_sim_seg[:n]
        
        # Apply log transform if specified
        if self.log_transform:
            eps = np.mean(obs[~np.isnan(obs)]) * 0.01
            fdc_obs_seg = np.log(fdc_obs_seg + eps)
            fdc_sim_seg = np.log(fdc_sim_seg + eps)
        
        # Compute metric
        if self.metric == 'volume_bias':
            sum_obs = np.sum(np.abs(fdc_obs_seg))
            if sum_obs < 1e-10:
                return np.nan
            return 100.0 * (np.sum(fdc_sim_seg) - np.sum(fdc_obs_seg)) / sum_obs
        
        elif self.metric == 'slope':
            if len(fdc_obs_seg) < 2:
                return np.nan
            slope_obs = (fdc_obs_seg[-1] - fdc_obs_seg[0]) / len(fdc_obs_seg)
            slope_sim = (fdc_sim_seg[-1] - fdc_sim_seg[0]) / len(fdc_sim_seg)
            if abs(slope_obs) < 1e-10:
                return np.nan
            return 100.0 * (slope_sim - slope_obs) / abs(slope_obs)
        
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((fdc_obs_seg - fdc_sim_seg) ** 2))
        
        elif self.metric == 'correlation':
            if np.std(fdc_obs_seg) == 0 or np.std(fdc_sim_seg) == 0:
                return np.nan
            return np.corrcoef(fdc_obs_seg, fdc_sim_seg)[0, 1]
```

---

## 9. Hydrological Signatures

```python
# hydro_objectives/signatures/flow_indices.py

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
    
    SIGNATURES = {
        # Flow percentiles (note: q95 = 5th percentile = high flow)
        'q95': ('percentile', 5),
        'q90': ('percentile', 10),
        'q75': ('percentile', 25),
        'q50': ('percentile', 50),
        'q25': ('percentile', 75),
        'q10': ('percentile', 90),
        'q5': ('percentile', 95),
        
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
                return np.std(Q_clean)
            elif self._sig_param == 'cv':
                mean = np.mean(Q_clean)
                if mean == 0:
                    return np.nan
                return np.std(Q_clean) / mean
        
        elif self._sig_type == 'dynamic':
            if self._sig_param == 'flashiness':
                # Richards-Baker flashiness index
                if len(Q_clean) < 2:
                    return np.nan
                return np.sum(np.abs(np.diff(Q_clean))) / np.sum(Q_clean)
            
            elif self._sig_param == 'baseflow_index':
                # Simple baseflow estimate using minimum filter
                # (More sophisticated methods could be added)
                window = min(5, len(Q_clean))
                from scipy.ndimage import minimum_filter1d
                baseflow = minimum_filter1d(Q_clean, size=window)
                return np.sum(baseflow) / np.sum(Q_clean)
            
            elif self._sig_param == 'high_flow_freq':
                threshold = 3 * np.median(Q_clean)
                return np.sum(Q_clean > threshold) / len(Q_clean)
            
            elif self._sig_param == 'low_flow_freq':
                threshold = 0.2 * np.mean(Q_clean)
                return np.sum(Q_clean < threshold) / len(Q_clean)
            
            elif self._sig_param == 'zero_flow_freq':
                return np.sum(Q_clean == 0) / len(Q_clean)
        
        return np.nan
    
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
```

---

## 10. Composite Objective Functions

```python
# hydro_objectives/composite/weighted.py

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
        - Maximize objectives: value_norm = max(0, value)
        - Minimize objectives: value_norm = max(0, 1 - |value|/100)
    
    Aggregation methods:
        - weighted_sum: Best for combining similar-scale metrics
        - weighted_product: All metrics must be positive; penalizes poor performers
        - min: Ensures minimum acceptable performance on all metrics
    
    Examples
    --------
    >>> # KGE + KGE(1/Q) for balanced high/low flow calibration
    >>> combined = WeightedObjective([
    ...     (KGE(), 0.5),
    ...     (KGE(transform=FlowTransformation('inverse')), 0.5),
    ... ])
    >>> result = combined.evaluate(obs, sim)
    
    >>> # Multi-metric with different scales
    >>> combined = WeightedObjective([
    ...     (KGE(), 0.4),
    ...     (PBIAS(), 0.3),
    ...     (FDCMetric('low', log_transform=True), 0.3),
    ... ], normalize=True)
    """
    
    AGGREGATIONS = ('weighted_sum', 'weighted_product', 'min')
    
    def __init__(self,
                 objectives: list[tuple[ObjectiveFunction, float]],
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
        
        # Validate weights
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
                # Assume maximize metrics are roughly in [-1, 1] range
                return max(0.0, min(1.0, (value + 1) / 2))
            else:
                # Assume minimize metrics: 0 is optimal, 100 is bad
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
        return {
            obj.name: obj(obs, sim, **kwargs) 
            for obj, _ in self.objectives
        }
```

---

## 11. Utility Functions

```python
# hydro_objectives/composite/factories.py

def kge_hilo(kge_weight: float = 0.5, 
             variant: str = '2012') -> WeightedObjective:
    """
    Create KGE + KGE(1/Q) combination for balanced calibration.
    
    This combination is recommended by Garcia et al. (2017) for 
    achieving good performance on both high and low flow simulation.
    
    Parameters
    ----------
    kge_weight : float, default=0.5
        Weight for standard KGE (1 - kge_weight for inverse KGE)
    variant : str, default='2012'
        KGE variant to use
    
    Returns
    -------
    WeightedObjective
        Combined objective function
    
    References
    ----------
    Garcia, F. et al. (2017). Which objective function to calibrate 
    rainfall–runoff models for low-flow index simulations?
    Hydrological Sciences Journal, 62(7), 1149-1166.
    
    Examples
    --------
    >>> obj = kge_hilo(0.5)
    >>> value = obj(obs, sim)
    """
    from hydro_objectives import KGE, FlowTransformation
    
    kge_standard = KGE(variant=variant)
    kge_inverse = KGE(
        variant=variant,
        transform=FlowTransformation('inverse')
    )
    
    return WeightedObjective(
        objectives=[
            (kge_standard, kge_weight),
            (kge_inverse, 1 - kge_weight),
        ],
        normalize=False  # KGE values already comparable
    )


def fdc_multisegment(segments: list[str] = None,
                     weights: list[float] = None,
                     metric: str = 'volume_bias',
                     log_low: bool = True) -> WeightedObjective:
    """
    Create multi-segment FDC objective function.
    
    Parameters
    ----------
    segments : list of str, default=['high', 'mid', 'low']
        FDC segments to include
    weights : list of float, optional
        Weights for each segment (default: equal weights)
    metric : str, default='volume_bias'
        FDC metric type
    log_low : bool, default=True
        Apply log transform to low flow segments
    
    Returns
    -------
    WeightedObjective
    """
    from hydro_objectives import FDCMetric
    
    if segments is None:
        segments = ['high', 'mid', 'low']
    
    if weights is None:
        weights = [1.0] * len(segments)
    
    if len(segments) != len(weights):
        raise ValueError("segments and weights must have same length")
    
    objectives = []
    for seg, w in zip(segments, weights):
        use_log = log_low and seg in ('low', 'very_low')
        obj = FDCMetric(segment=seg, metric=metric, log_transform=use_log)
        objectives.append((obj, w))
    
    return WeightedObjective(objectives, normalize=True)


def comprehensive_objective(
    kge_weight: float = 0.4,
    kge_inv_weight: float = 0.2,
    pbias_weight: float = 0.15,
    fdc_high_weight: float = 0.1,
    fdc_low_weight: float = 0.15,
    kge_variant: str = '2012'
) -> WeightedObjective:
    """
    Create comprehensive multi-metric objective function.
    
    Combines KGE, KGE on inverse flows, percent bias, and FDC metrics
    for thorough model evaluation.
    
    Parameters
    ----------
    kge_weight : float
        Weight for standard KGE
    kge_inv_weight : float
        Weight for KGE on inverse flows
    pbias_weight : float
        Weight for percent bias
    fdc_high_weight : float
        Weight for high-flow FDC metric
    fdc_low_weight : float
        Weight for low-flow FDC metric
    kge_variant : str
        KGE variant to use
    
    Returns
    -------
    WeightedObjective
    """
    from hydro_objectives import KGE, PBIAS, FDCMetric, FlowTransformation
    
    objectives = [
        (KGE(variant=kge_variant), kge_weight),
        (KGE(variant=kge_variant, transform=FlowTransformation('inverse')), kge_inv_weight),
        (PBIAS(), pbias_weight),
        (FDCMetric('high', 'volume_bias'), fdc_high_weight),
        (FDCMetric('low', 'volume_bias', log_transform=True), fdc_low_weight),
    ]
    
    return WeightedObjective(objectives, normalize=True)


# hydro_objectives/core/utils.py

def evaluate_all(obs: np.ndarray,
                 sim: np.ndarray,
                 objectives: list[ObjectiveFunction] = None) -> dict[str, MetricResult]:
    """
    Evaluate multiple objective functions at once.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values
    objectives : list of ObjectiveFunction, optional
        List of objective functions. If None, uses standard set:
        NSE, KGE, KGE(inverse), PBIAS, FDC_high, FDC_low
    
    Returns
    -------
    dict
        Dictionary mapping objective names to MetricResult objects
    """
    if objectives is None:
        from hydro_objectives import NSE, KGE, PBIAS, FDCMetric, FlowTransformation
        objectives = [
            NSE(),
            KGE(variant='2012'),
            KGE(variant='2012', transform=FlowTransformation('inverse')),
            PBIAS(),
            FDCMetric('high', 'volume_bias'),
            FDCMetric('low', 'volume_bias', log_transform=True),
        ]
    
    return {obj.name: obj.evaluate(obs, sim) for obj in objectives}


def print_evaluation_report(obs: np.ndarray, 
                            sim: np.ndarray,
                            objectives: list[ObjectiveFunction] = None) -> None:
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values
    objectives : list of ObjectiveFunction, optional
        Objectives to evaluate
    """
    results = evaluate_all(obs, sim, objectives)
    
    print("\n" + "=" * 60)
    print("Model Performance Evaluation Report")
    print("=" * 60)
    print(f"Sample size: {len(obs[~np.isnan(obs)])}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"\n{result}")
```

---

## 12. Testing Requirements

### 12.1 Test Categories

1. **Unit Tests**: Each class and function individually
2. **Integration Tests**: Combined functionality
3. **Validation Tests**: Against known implementations (R hydroGOF)
4. **Edge Case Tests**: NaN handling, zero flows, constant values
5. **Performance Tests**: Large array benchmarks

### 12.2 Test Cases

```python
# tests/fixtures.py

import numpy as np

def generate_test_data(n: int = 365, 
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic observed and simulated flow data."""
    np.random.seed(seed)
    
    # Lognormal distribution typical of streamflow
    obs = np.random.lognormal(3, 1, n)
    
    # Simulated with some error
    sim = obs * np.random.normal(1.0, 0.15, n)
    sim = np.maximum(sim, 0)  # No negative flows
    
    return obs, sim


def generate_perfect_data(n: int = 365) -> tuple[np.ndarray, np.ndarray]:
    """Generate identical obs and sim for testing optimal values."""
    obs = np.random.lognormal(3, 1, n)
    return obs, obs.copy()


def generate_with_nan(n: int = 365, 
                      nan_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with random NaN values."""
    obs, sim = generate_test_data(n)
    
    nan_idx = np.random.choice(n, int(n * nan_fraction), replace=False)
    obs[nan_idx[:len(nan_idx)//2]] = np.nan
    sim[nan_idx[len(nan_idx)//2:]] = np.nan
    
    return obs, sim
```

### 12.3 Required Test Cases

| Test | Expected Result |
|------|-----------------|
| `test_nse_perfect_match` | NSE = 1.0 |
| `test_kge_perfect_match` | KGE = 1.0, r = 1, α = 1, β = 1 |
| `test_kge_with_bias` | β ≈ bias_factor |
| `test_rmse_zero_error` | RMSE = 0.0 |
| `test_pbias_overestimate` | PBIAS > 0 |
| `test_pbias_underestimate` | PBIAS < 0 |
| `test_nan_handling` | Valid result (not NaN) |
| `test_empty_array` | Raises ValueError |
| `test_mismatched_length` | Raises ValueError |
| `test_transform_emphasis` | Correct flow emphasis |
| `test_fdc_segments` | Correct segment bounds |
| `test_weighted_combination` | Weights sum correctly |
| `test_kge_hilo_factory` | Returns WeightedObjective |

---

## 13. Documentation Requirements

### 13.1 Docstring Format

Use NumPy style docstrings with the following sections:

1. Short summary (one line)
2. Extended summary (optional)
3. Parameters
4. Returns
5. Raises (if applicable)
6. Notes (optional)
7. References (optional)
8. Examples

### 13.2 API Documentation

Generate with Sphinx and autodoc:

```
docs/
├── conf.py
├── index.rst
├── installation.rst
├── quickstart.rst
├── api/
│   ├── metrics.rst
│   ├── transformations.rst
│   ├── fdc.rst
│   ├── signatures.rst
│   └── composite.rst
└── tutorials/
    ├── basic_usage.rst
    ├── custom_objectives.rst
    └── calibration_example.rst
```

---

## 14. Dependencies

### 14.1 Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Array operations |
| scipy | ≥1.7 | Statistical functions (Spearman, filters) |

### 14.2 Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3 | DataFrame integration |
| matplotlib | ≥3.4 | Visualization utilities |

### 14.3 Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Testing |
| pytest-cov | Coverage reporting |
| sphinx | Documentation |
| black | Code formatting |
| mypy | Type checking |

---

## 15. Mathematical Reference

### 15.1 Nash-Sutcliffe Efficiency

$$NSE = 1 - \frac{\sum_{t=1}^{T}(Q_{obs,t} - Q_{sim,t})^2}{\sum_{t=1}^{T}(Q_{obs,t} - \bar{Q}_{obs})^2}$$

### 15.2 Kling-Gupta Efficiency

$$KGE = 1 - \sqrt{(s_r(r-1))^2 + (s_\alpha(\alpha-1))^2 + (s_\beta(\beta-1))^2}$$

**Components:**

$$r = \frac{\text{Cov}(Q_{obs}, Q_{sim})}{\sigma_{obs} \cdot \sigma_{sim}}$$

**Variant 2009:**
$$\alpha = \frac{\sigma_{sim}}{\sigma_{obs}}, \quad \beta = \frac{\mu_{sim}}{\mu_{obs}}$$

**Variant 2012:**
$$\alpha = \frac{CV_{sim}}{CV_{obs}} = \frac{\sigma_{sim}/\mu_{sim}}{\sigma_{obs}/\mu_{obs}}, \quad \beta = \frac{\mu_{sim}}{\mu_{obs}}$$

**Variant 2021:**
$$\alpha = \frac{\sigma_{sim}}{\sigma_{obs}}, \quad \beta = \frac{\mu_{sim} - \mu_{obs}}{\sigma_{obs}}$$

### 15.3 Flow Duration Curve

For a time series of $T$ values sorted in descending order:

$$p_i = \frac{i}{T+1}$$

where $p_i$ is the exceedance probability for rank $i$.

### 15.4 Flashiness Index

$$FI = \frac{\sum_{t=2}^{T}|Q_t - Q_{t-1}|}{\sum_{t=1}^{T}Q_t}$$

---

## 16. Literature References

1. **Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009).** Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1-2), 80-91.

2. **Kling, H., Fuchs, M., Paulin, M. (2012).** Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. *Journal of Hydrology*, 424, 264-277.

3. **Knoben, W.J.M., Freer, J.E., Woods, R.A. (2019).** Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. *Hydrology and Earth System Sciences*, 23, 4323-4331.

4. **Santos, L., Thirel, G., Perrin, C. (2018).** Technical note: Pitfalls in using log-transformed flows within the KGE criterion. *Hydrology and Earth System Sciences*, 22, 4583-4591.

5. **Tang, G., Clark, M.P., Papalexiou, S.M. (2021).** SC-earth: a station-based serially complete earth dataset from 1950 to 2019. *Journal of Climate*, 34(16), 6493-6511.

6. **Garcia, F., Folton, N., Oudin, L. (2017).** Which objective function to calibrate rainfall–runoff models for low-flow index simulations? *Hydrological Sciences Journal*, 62(7), 1149-1166.

7. **Yilmaz, K.K., Gupta, H.V., Wagener, T. (2008).** A process-based diagnostic approach to model evaluation: Application to the NWS distributed hydrologic model. *Water Resources Research*, 44(9), W09417.

8. **Westerberg, I.K. et al. (2011).** Calibration of hydrological models using flow-duration curves. *Hydrology and Earth System Sciences*, 15, 2205-2227.

9. **Pushpalatha, R., Perrin, C., Le Moine, N., Andréassian, V. (2012).** A review of efficiency criteria suitable for evaluating low-flow simulations. *Journal of Hydrology*, 420-421, 171-182.

10. **Pfannerstill, M., Guse, B., Fohrer, N. (2014).** Smart low flow signature metrics for an improved overall performance evaluation of hydrological models. *Journal of Hydrology*, 510, 447-458.

11. **Nash, J.E., Sutcliffe, J.V. (1970).** River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10(3), 282-290.

12. **Thirel, G., Santos, L., Delaigue, O., Perrin, C. (2024).** On the use of streamflow transformations for hydrological model calibration. *Hydrology and Earth System Sciences*, 28, 4837-4860.

13. **Pool, S., Vis, M., Seibert, J. (2018).** Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency. *Hydrological Sciences Journal*, 63(13-14), 1941-1953.

14. **McMillan, H., Westerberg, I., Branger, F. (2017).** Five guidelines for selecting hydrological signatures. *Hydrological Processes*, 31, 4757-4761.

15. **Addor, N. et al. (2018).** A ranking of hydrological signatures based on their predictability in space. *Water Resources Research*, 54, 8792-8812.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Jan 2025 | - | Initial specification |

---

*End of Specification Document*
