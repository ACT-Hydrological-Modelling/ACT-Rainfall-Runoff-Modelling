"""
Factory functions for common composite objective configurations.

This module provides convenience functions for creating commonly used
combinations of objective functions for hydrological model calibration.

References
----------
Garcia, F., Folton, N., Oudin, L. (2017). Which objective function to 
calibrate rainfall–runoff models for low-flow index simulations?
Hydrological Sciences Journal, 62(7), 1149-1166.
"""

from typing import List, Optional

from pyrrm.objectives.composite.weighted import WeightedObjective


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
        KGE variant to use ('2009', '2012', '2021')
    
    Returns
    -------
    WeightedObjective
        Combined objective function
    
    Notes
    -----
    - Standard KGE emphasizes high flows due to variance calculation
    - KGE on inverse flows (1/Q) emphasizes low flows
    - Equal weights (0.5/0.5) provide balanced calibration
    
    References
    ----------
    Garcia, F. et al. (2017). Which objective function to calibrate 
    rainfall–runoff models for low-flow index simulations?
    Hydrological Sciences Journal, 62(7), 1149-1166.
    
    Examples
    --------
    >>> obj = kge_hilo(0.5)
    >>> value = obj(obs, sim)
    
    >>> # Emphasize high flows
    >>> obj = kge_hilo(0.7)  # 70% high flow, 30% low flow
    """
    from pyrrm.objectives.metrics.kge import KGE
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
    
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


def fdc_multisegment(segments: Optional[List[str]] = None,
                      weights: Optional[List[float]] = None,
                      metric: str = 'volume_bias',
                      log_low: bool = True) -> WeightedObjective:
    """
    Create multi-segment FDC objective function.
    
    Combines FDC metrics for multiple flow regimes to ensure
    good performance across the entire flow duration curve.
    
    Parameters
    ----------
    segments : list of str, default=['high', 'mid', 'low']
        FDC segments to include. Options: 'peak', 'high', 'mid', 'low', 'very_low'
    weights : list of float, optional
        Weights for each segment (default: equal weights)
    metric : str, default='volume_bias'
        FDC metric type to use for all segments
    log_low : bool, default=True
        Apply log transform to low flow segments ('low', 'very_low')
    
    Returns
    -------
    WeightedObjective
        Combined multi-segment FDC objective
    
    Examples
    --------
    >>> # Default: high, mid, low with equal weights
    >>> obj = fdc_multisegment()
    
    >>> # Custom segments with custom weights
    >>> obj = fdc_multisegment(
    ...     segments=['peak', 'high', 'mid', 'low'],
    ...     weights=[0.1, 0.3, 0.3, 0.3]
    ... )
    """
    from pyrrm.objectives.fdc.metrics import FDCMetric
    
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
    for thorough model evaluation across multiple performance dimensions.
    
    Parameters
    ----------
    kge_weight : float, default=0.4
        Weight for standard KGE (overall performance)
    kge_inv_weight : float, default=0.2
        Weight for KGE on inverse flows (low flow performance)
    pbias_weight : float, default=0.15
        Weight for percent bias (volume error)
    fdc_high_weight : float, default=0.1
        Weight for high-flow FDC metric
    fdc_low_weight : float, default=0.15
        Weight for low-flow FDC metric
    kge_variant : str, default='2012'
        KGE variant to use
    
    Returns
    -------
    WeightedObjective
        Comprehensive combined objective function
    
    Notes
    -----
    Default weights are designed to:
    - Prioritize overall fit (KGE at 40%)
    - Ensure reasonable low flow performance (KGE_inv + FDC_low = 35%)
    - Control volume bias (PBIAS at 15%)
    - Maintain high flow shape (FDC_high at 10%)
    
    Examples
    --------
    >>> obj = comprehensive_objective()
    >>> value = obj(obs, sim)
    
    >>> # Custom weights emphasizing low flows
    >>> obj = comprehensive_objective(
    ...     kge_weight=0.3,
    ...     kge_inv_weight=0.3,
    ...     fdc_low_weight=0.2
    ... )
    """
    from pyrrm.objectives.metrics.kge import KGE
    from pyrrm.objectives.metrics.traditional import PBIAS
    from pyrrm.objectives.fdc.metrics import FDCMetric
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
    
    objectives = [
        (KGE(variant=kge_variant), kge_weight),
        (KGE(variant=kge_variant, transform=FlowTransformation('inverse')), kge_inv_weight),
        (PBIAS(), pbias_weight),
        (FDCMetric('high', 'volume_bias'), fdc_high_weight),
        (FDCMetric('low', 'volume_bias', log_transform=True), fdc_low_weight),
    ]
    
    return WeightedObjective(objectives, normalize=True)


def nse_multiscale(nse_weight: float = 0.5,
                    log_nse_weight: float = 0.3,
                    sqrt_nse_weight: float = 0.2) -> WeightedObjective:
    """
    Create multi-scale NSE objective combining different transformations.
    
    Combines standard NSE, log-transformed NSE, and sqrt-transformed NSE
    to balance performance across different flow magnitudes.
    
    Parameters
    ----------
    nse_weight : float, default=0.5
        Weight for standard NSE (high flow emphasis)
    log_nse_weight : float, default=0.3
        Weight for log-NSE (low flow emphasis)
    sqrt_nse_weight : float, default=0.2
        Weight for sqrt-NSE (balanced)
    
    Returns
    -------
    WeightedObjective
        Multi-scale NSE objective
    
    Examples
    --------
    >>> obj = nse_multiscale()
    >>> value = obj(obs, sim)
    """
    from pyrrm.objectives.metrics.traditional import NSE
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
    
    objectives = [
        (NSE(), nse_weight),
        (NSE(transform=FlowTransformation('log')), log_nse_weight),
        (NSE(transform=FlowTransformation('sqrt')), sqrt_nse_weight),
    ]
    
    return WeightedObjective(objectives, normalize=False)
