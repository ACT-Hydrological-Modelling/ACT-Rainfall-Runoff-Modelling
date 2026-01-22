"""
Kling-Gupta Efficiency family of objective functions.

This module provides implementations of KGE variants:
- KGE: Kling-Gupta Efficiency with configurable variants (2009, 2012, 2021)
- KGENonParametric: Non-parametric variant using Spearman correlation

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

Pool, S., Vis, M., Seibert, J. (2018). Evaluating model performance: 
towards a non-parametric variant of the Kling-Gupta efficiency. 
Hydrological Sciences Journal, 63(13-14), 1941-1953.

Santos, L., Thirel, G., Perrin, C. (2018). Technical note: Pitfalls in 
using log-transformed flows within the KGE criterion. HESS, 22, 4583-4591.
"""

from typing import Optional, Dict, Tuple, TYPE_CHECKING
import warnings
import numpy as np
from scipy import stats

from pyrrm.objectives.core.base import ObjectiveFunction

if TYPE_CHECKING:
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation


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
    >>> from pyrrm.objectives.transformations import FlowTransformation
    >>> kge_inv = KGE(transform=FlowTransformation('inverse'))
    """
    
    VARIANTS: Tuple[str, ...] = ('2009', '2012', '2021')
    
    def __init__(self,
                 variant: str = '2012',
                 weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
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
        if sigma_sim == 0:
            r = 0.0
        else:
            corr_matrix = np.corrcoef(obs_t, sim_t)
            r = corr_matrix[0, 1]
            if np.isnan(r):
                r = 0.0
        
        # Variability ratio (depends on variant)
        if self.variant == '2009':
            alpha = sigma_sim / sigma_obs
        elif self.variant == '2012':
            cv_obs = sigma_obs / mu_obs
            cv_sim = sigma_sim / mu_sim if mu_sim != 0 else 0
            alpha = cv_sim / cv_obs if cv_obs != 0 else 0
        elif self.variant == '2021':
            alpha = sigma_sim / sigma_obs
        
        # Bias ratio (depends on variant)
        if self.variant in ('2009', '2012'):
            beta = mu_sim / mu_obs if mu_obs != 0 else 0
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


class KGENonParametric(ObjectiveFunction):
    """
    Non-parametric Kling-Gupta Efficiency (Pool et al., 2018).
    
    Uses Spearman rank correlation instead of Pearson correlation,
    and normalized flow duration curves for variability assessment.
    This variant is more robust to non-Gaussian distributions and
    outliers common in streamflow data.
    
    Formula
    -------
    KGE_np = 1 - √[(r_s - 1)² + (α_np - 1)² + (β - 1)²]
    
    where:
        r_s = Spearman rank correlation coefficient
        α_np = variability ratio using normalized FDC
        β = bias ratio (μsim / μobs)
    
    Parameters
    ----------
    transform : FlowTransformation, optional
        Flow transformation to apply before calculation
    
    Notes
    -----
    - More robust to outliers than standard KGE
    - Spearman correlation captures monotonic relationships
    - Better suited for highly skewed flow distributions
    
    References
    ----------
    Pool, S., Vis, M., Seibert, J. (2018). Evaluating model performance: 
    towards a non-parametric variant of the Kling-Gupta efficiency. 
    Hydrological Sciences Journal, 63(13-14), 1941-1953.
    
    Examples
    --------
    >>> kge_np = KGENonParametric()
    >>> result = kge_np.evaluate(obs, sim)
    >>> print(f"KGE_np = {result.value:.3f}")
    """
    
    def __init__(self, transform: Optional['FlowTransformation'] = None):
        name = 'KGE_np'
        if transform is not None and transform.transform_type != 'none':
            name = f'{name}({transform.transform_type})'
        
        super().__init__(name=name, direction='maximize', optimal_value=1.0)
        self.transform = transform
        
        # Warn about log transformation
        if transform is not None and transform.transform_type == 'log':
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
        
        r_s = components['r_spearman']
        alpha_np = components['alpha_np']
        beta = components['beta']
        
        # Euclidean distance from ideal point (1, 1, 1)
        ed = np.sqrt(
            (r_s - 1) ** 2 +
            (alpha_np - 1) ** 2 +
            (beta - 1) ** 2
        )
        
        return 1.0 - ed
    
    def get_components(self, 
                       obs: np.ndarray, 
                       sim: np.ndarray, 
                       **kwargs) -> Optional[Dict[str, float]]:
        """
        Calculate non-parametric KGE components.
        
        Returns
        -------
        dict or None
            Dictionary with keys:
            - 'r_spearman': Spearman rank correlation coefficient
            - 'alpha_np': Non-parametric variability ratio
            - 'beta': Bias ratio
            - 'mu_obs', 'mu_sim': Mean values
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
        
        if mu_obs == 0:
            return None
        
        # Spearman rank correlation
        r_spearman, _ = stats.spearmanr(obs_t, sim_t)
        if np.isnan(r_spearman):
            r_spearman = 0.0
        
        # Non-parametric variability ratio using normalized FDC
        # Sort flows (descending for FDC)
        obs_sorted = np.sort(obs_t)[::-1]
        sim_sorted = np.sort(sim_t)[::-1]
        
        # Normalize by mean
        obs_norm = obs_sorted / mu_obs
        sim_norm = sim_sorted / mu_sim if mu_sim != 0 else np.zeros_like(sim_sorted)
        
        # Calculate variability as the standard deviation of normalized FDC
        # Alternative: use coefficient of variation of FDC values
        alpha_np = np.std(sim_norm) / np.std(obs_norm) if np.std(obs_norm) != 0 else 0
        
        # Bias ratio
        beta = mu_sim / mu_obs
        
        return {
            'r_spearman': r_spearman,
            'alpha_np': alpha_np,
            'beta': beta,
            'mu_obs': mu_obs,
            'mu_sim': mu_sim,
        }
