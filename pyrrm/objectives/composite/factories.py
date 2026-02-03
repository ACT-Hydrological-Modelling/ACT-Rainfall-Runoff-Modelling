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
from pyrrm.objectives.composite.adaptive import apex_adaptive  # Import for re-export


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


def apex_objective(
    # Core performance (40%)
    core_metric_1_weight: float = 0.25,
    core_metric_2_weight: float = 0.15,
    
    # FDC multi-segment (30%)
    fdc_high_weight: float = 0.10,
    fdc_mid_weight: float = 0.10,
    fdc_low_weight: float = 0.10,
    
    # Process signatures (20%)
    baseflow_index_weight: float = 0.10,
    flashiness_weight: float = 0.10,
    
    # Bias/timing (10%)
    pbias_weight: float = 0.05,
    timing_correlation_weight: float = 0.05,
    
    # Core metric configuration
    core_metric_type: str = 'kge',
    core_metric_1_transform: Optional[str] = None,
    core_metric_2_transform: str = 'sqrt',
    kge_variant: str = '2012'
) -> WeightedObjective:
    """
    APEX: Adaptive Process-Explicit Objective Function.
    
    A state-of-the-art composite objective combining multi-scale metrics,
    multi-segment FDC evaluation, hydrological signatures, and bias/timing
    controls for comprehensive rainfall-runoff model calibration.
    
    APEX addresses limitations of single-objective calibration by explicitly
    evaluating multiple aspects of hydrological performance:
    - Overall fit and flow magnitude distribution (KGE or NSE metrics)
    - Flow frequency distribution across regimes (FDC segments)
    - Process-based hydrological realism (signatures)
    - Systematic bias and timing correlation
    
    Parameters
    ----------
    core_metric_1_weight : float, default=0.25
        Weight for first core metric (overall performance, typically high flow emphasis)
    core_metric_2_weight : float, default=0.15
        Weight for second core metric (typically with transformation for balance)
    fdc_high_weight : float, default=0.10
        Weight for high flow FDC segment (2-20% exceedance)
    fdc_mid_weight : float, default=0.10
        Weight for mid flow FDC segment (20-70% exceedance)
    fdc_low_weight : float, default=0.10
        Weight for low flow FDC segment (70-95% exceedance, log-transformed)
    baseflow_index_weight : float, default=0.10
        Weight for baseflow index signature (groundwater contribution)
    flashiness_weight : float, default=0.10
        Weight for flashiness index signature (response dynamics)
    pbias_weight : float, default=0.05
        Weight for percent bias (volume error control)
    timing_correlation_weight : float, default=0.05
        Weight for temporal correlation (timing accuracy)
    core_metric_type : str, default='kge'
        Type of core metric to use: 'kge' or 'nse'
    core_metric_1_transform : str, optional
        Flow transformation for first core metric. Options: None, 'sqrt', 'log', 
        'inverse', 'power', 'boxcox'. Default: None (no transformation)
    core_metric_2_transform : str, default='sqrt'
        Flow transformation for second core metric. Options: 'sqrt', 'log', 
        'inverse', 'power', 'boxcox', None. Default: 'sqrt' (balanced high/low flows)
    kge_variant : str, default='2012'
        KGE variant to use when core_metric_type='kge': '2009', '2012', or '2021'
    
    Returns
    -------
    WeightedObjective
        APEX composite objective function ready for calibration
    
    Notes
    -----
    Default weight distribution (sums to 1.0):
    - 40% core performance metrics (2 metrics, configurable type and transforms)
    - 30% FDC segment matching (high + mid + low flows)
    - 20% process signatures (baseflow index + flashiness)
    - 10% bias/timing control (PBIAS + correlation)
    
    **Core Metric Configuration:**
    
    The core metrics can use either KGE or NSE with optional transformations:
    
    - **Default (KGE + KGE-sqrt)**: Recommended for general use
      - Untransformed: High flow emphasis
      - Sqrt-transformed: Balanced high/low flow performance
    
    - **NSE-based**: Alternative for traditional calibration
      - NSE: High flow emphasis (squared errors)
      - NSE-log: Low flow emphasis
      - NSE-sqrt: Balanced performance
      - NSE-inverse: Strong low flow emphasis
    
    - **Mixed approaches**: Any combination is valid
    
    Transform options: None, 'sqrt', 'log', 'inverse', 'power', 'boxcox'
    
    **Advantages over SDEB (Lerat et al., 2013):**
    - Explicit multi-segment FDC evaluation (vs. single ranked term)
    - Process-based signatures ensure hydrological realism
    - Separate timing correlation metric (vs. combined chronological term)
    - Modular design allows component-level diagnostics
    - Flexible core metric selection (KGE or NSE with transforms)
    - Catchment-specific tuning via weighting
    
    **FDC segments** target specific flow regimes:
    - High (2-20%): Flood response and quick runoff
    - Mid (20-70%): Typical flow conditions
    - Low (70-95%): Baseflow and drought conditions
    
    **Process signatures** provide diagnostic information:
    - Baseflow index: Groundwater vs. surface runoff partitioning
    - Flashiness: Catchment response dynamics and hydrograph variability
    
    References
    ----------
    Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). 
    Decomposition of the mean squared error and NSE performance criteria.
    Journal of Hydrology, 377(1-2), 80-91.
    
    Yilmaz, K.K., Gupta, H.V., Wagener, T. (2008). A process-based 
    diagnostic approach to model evaluation. Water Resources Research, 44(9).
    
    Westerberg, I.K. et al. (2011). Calibration of hydrological models 
    using flow-duration curves. HESS, 15, 2205-2227.
    
    Pool, S., Vis, M., Seibert, J. (2018). Evaluating model performance: 
    towards a non-parametric variant of the Kling-Gupta efficiency.
    Hydrological Sciences Journal, 63(13-14), 1941-1953.
    
    Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Kuczera, G. (2013).
    A robust approach for calibrating continuous hydrological models.
    Journal of Hydrology, 494, 80-91.
    
    Examples
    --------
    >>> from pyrrm.objectives import apex_objective
    >>> from pyrrm.calibration import CalibrationRunner
    >>> 
    >>> # Default: KGE + KGE(sqrt)
    >>> apex = apex_objective()
    >>> 
    >>> # NSE-based with log transformation for low flows
    >>> apex_nse = apex_objective(
    ...     core_metric_type='nse',
    ...     core_metric_1_transform=None,      # NSE (high flow emphasis)
    ...     core_metric_2_transform='log'      # Log-NSE (low flow emphasis)
    ... )
    >>> 
    >>> # KGE with inverse transformation for strong low flow focus
    >>> apex_lowflow = apex_objective(
    ...     core_metric_type='kge',
    ...     core_metric_1_transform=None,      # Standard KGE
    ...     core_metric_2_transform='inverse', # KGE(1/Q)
    ...     core_metric_2_weight=0.20,
    ...     fdc_low_weight=0.15
    ... )
    >>> 
    >>> # NSE balanced approach
    >>> apex_nse_balanced = apex_objective(
    ...     core_metric_type='nse',
    ...     core_metric_1_transform='sqrt',    # NSE(sqrt) - balanced
    ...     core_metric_2_transform='log',     # NSE(log) - low flows
    ...     core_metric_1_weight=0.25,
    ...     core_metric_2_weight=0.15
    ... )
    >>> 
    >>> # Custom weights for flashy catchments (with default KGE)
    >>> apex_flashy = apex_objective(
    ...     fdc_high_weight=0.15,
    ...     flashiness_weight=0.15,
    ...     baseflow_index_weight=0.05
    ... )
    >>> 
    >>> # Calibrate model
    >>> runner = CalibrationRunner(model, inputs, observed, objective=apex)
    >>> result = runner.run_sceua_direct(max_iterations=10000)
    >>> 
    >>> # Evaluate components
    >>> components = apex.evaluate_individual(observed, simulated)
    >>> print(components['raw_values'])  # Individual metric values
    
    See Also
    --------
    APEX : New standalone APEX class with dynamics/lag multipliers (recommended)
    comprehensive_objective : Alternative multi-metric objective
    kge_hilo : Simple KGE + KGE(inverse) combination
    fdc_multisegment : FDC-only multi-segment objective
    
    .. deprecated::
        This function is deprecated. Use the new APEX class instead:
        ``from pyrrm.objectives import APEX``
        
        The new APEX class provides:
        - Simpler interface with fewer parameters
        - Novel dynamics multiplier (gradient correlation)
        - Optional lag multiplier (timing offset penalty)
        - SDEB-based multiplicative structure (no normalization issues)
    """
    import warnings
    warnings.warn(
        "apex_objective() is deprecated and will be removed in a future version. "
        "Use the new APEX class instead: from pyrrm.objectives import APEX",
        DeprecationWarning,
        stacklevel=2
    )
    
    from pyrrm.objectives.metrics.kge import KGE
    from pyrrm.objectives.metrics.traditional import NSE
    from pyrrm.objectives.metrics.traditional import PBIAS
    from pyrrm.objectives.metrics.correlation import PearsonCorrelation
    from pyrrm.objectives.fdc.metrics import FDCMetric
    from pyrrm.objectives.signatures.flow_indices import SignatureMetric
    from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
    
    # Validate core metric type
    if core_metric_type not in ['kge', 'nse']:
        raise ValueError(
            f"core_metric_type must be 'kge' or 'nse', got '{core_metric_type}'"
        )
    
    # Build core metrics with optional transformations
    objectives = []
    
    # === CORE PERFORMANCE (40%) ===
    if core_metric_type == 'kge':
        # Core metric 1: KGE (optionally with transformation)
        objectives.append((
            KGE(
                variant=kge_variant,
                transform=FlowTransformation(core_metric_1_transform) if core_metric_1_transform else None
            ), 
            core_metric_1_weight
        ))
        
        # Core metric 2: KGE with transformation for balance
        objectives.append((
            KGE(
                variant=kge_variant,
                transform=FlowTransformation(core_metric_2_transform) if core_metric_2_transform else None
            ), 
            core_metric_2_weight
        ))
    else:  # nse
        # Core metric 1: NSE (optionally with transformation)
        objectives.append((
            NSE(
                transform=FlowTransformation(core_metric_1_transform) if core_metric_1_transform else None
            ), 
            core_metric_1_weight
        ))
        
        # Core metric 2: NSE with transformation for balance
        objectives.append((
            NSE(
                transform=FlowTransformation(core_metric_2_transform) if core_metric_2_transform else None
            ), 
            core_metric_2_weight
        ))
    
    # === FDC MULTI-SEGMENT (30%) ===
    # High flows (2-20% exceedance): flood response
    objectives.append((FDCMetric(segment='high', metric='volume_bias'), fdc_high_weight))
    
    # Mid flows (20-70% exceedance): typical conditions
    objectives.append((FDCMetric(segment='mid', metric='volume_bias'), fdc_mid_weight))
    
    # Low flows (70-95% exceedance): baseflow with log transform
    objectives.append((FDCMetric(segment='low', metric='volume_bias', 
               log_transform=True), fdc_low_weight))
    
    # === PROCESS SIGNATURES (20%) ===
    # Baseflow index: groundwater contribution
    objectives.append((SignatureMetric('baseflow_index'), baseflow_index_weight))
    
    # Flashiness: response dynamics
    objectives.append((SignatureMetric('flashiness'), flashiness_weight))
    
    # === BIAS & TIMING (10%) ===
    # Volume bias
    objectives.append((PBIAS(), pbias_weight))
    
    # Timing correlation (independent of magnitude)
    objectives.append((PearsonCorrelation(), timing_correlation_weight))
    
    return WeightedObjective(
        objectives=objectives,
        aggregation='weighted_sum',
        normalize=True,
        normalize_method='minmax'
    )
