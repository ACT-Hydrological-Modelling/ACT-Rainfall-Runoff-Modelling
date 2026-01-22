"""
MetricResult container for objective function evaluation results.

This module provides the MetricResult dataclass for storing objective
function values along with their component breakdowns.
"""

from dataclasses import dataclass, field
from typing import Dict


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
    >>> result = MetricResult(
    ...     value=0.85, 
    ...     components={'r': 0.95, 'alpha': 0.9, 'beta': 1.0}, 
    ...     name='KGE'
    ... )
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
        """
        Return string representation with component breakdown.
        
        Internal statistics (mu_, sigma_) are filtered out for cleaner output.
        """
        if self.components:
            # Filter out internal statistics for cleaner output
            display_components = {
                k: v for k, v in self.components.items() 
                if not k.startswith(('mu_', 'sigma_'))
            }
            if display_components:
                comp_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in display_components.items()
                )
                return f"{self.name}={self.value:.4f} ({comp_str})"
        return f"{self.name}={self.value:.4f}"
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary including value and all components.
        
        Returns
        -------
        dict
            Dictionary with 'value' key and all component keys
        
        Examples
        --------
        >>> result = MetricResult(value=0.85, components={'r': 0.95}, name='KGE')
        >>> result.to_dict()
        {'value': 0.85, 'r': 0.95}
        """
        result = {'value': self.value}
        result.update(self.components)
        return result
