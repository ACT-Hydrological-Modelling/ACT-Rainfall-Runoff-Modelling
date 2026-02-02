"""
Adaptive weighting for APEX objective function.

This module provides utilities for automatically adapting APEX component weights
based on observed flow regime characteristics, enabling catchment-specific
calibration strategies.

References
----------
Sawicz, K., Wagener, T., Sivapalan, M., Troch, P.A., Carrillo, G. (2011).
Catchment classification: empirical analysis of hydrologic similarity based on 
catchment function in the eastern USA. Hydrology and Earth System Sciences, 15,
2895-2911.

Pool, S., Vis, M., Seibert, J. (2018). Evaluating model performance: towards a 
non-parametric variant of the Kling-Gupta efficiency. Hydrological Sciences 
Journal, 63(13-14), 1941-1953.
"""

from typing import Dict, Optional
import numpy as np
from scipy.ndimage import minimum_filter1d

from pyrrm.objectives.composite.weighted import WeightedObjective


def characterize_flow_regime(Q_obs: np.ndarray, 
                              window: int = 5) -> Dict[str, float]:
    """
    Analyze observed flow to characterize catchment flow regime.
    
    Computes key hydrological indices that describe the catchment's flow
    behavior, which can be used to adapt calibration objectives.
    
    Parameters
    ----------
    Q_obs : np.ndarray
        Observed flow time series
    window : int, default=5
        Window size for baseflow separation (days)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'cv': Coefficient of variation (std/mean)
        - 'bfi': Baseflow index (baseflow/total flow)
        - 'flashiness': Flashiness index (sum of abs changes / sum of flows)
        - 'intermittency': Fraction of very low/zero flows
        - 'mean_flow': Mean flow value
        - 'q95': 95th percentile (high flow threshold)
        - 'q5': 5th percentile (low flow threshold)
        - 'regime_type': Classification string
    
    Examples
    --------
    >>> regime = characterize_flow_regime(observed_flow)
    >>> print(f"Catchment type: {regime['regime_type']}")
    >>> print(f"Coefficient of variation: {regime['cv']:.2f}")
    >>> print(f"Baseflow index: {regime['bfi']:.2f}")
    """
    Q_clean = Q_obs[~np.isnan(Q_obs)]
    
    if len(Q_clean) < window * 2:
        raise ValueError(f"Need at least {window * 2} valid observations")
    
    # Basic statistics
    mean_flow = float(np.mean(Q_clean))
    std_flow = float(np.std(Q_clean))
    cv = std_flow / mean_flow if mean_flow > 0 else np.nan
    
    # Flow percentiles
    q95 = float(np.percentile(Q_clean, 95))
    q5 = float(np.percentile(Q_clean, 5))
    
    # Baseflow index using minimum filter method
    baseflow = Q_clean.copy()
    for _ in range(3):  # 3 passes for smoothing
        baseflow = minimum_filter1d(baseflow, size=window)
    baseflow = np.minimum(baseflow, Q_clean)
    bfi = float(np.sum(baseflow) / np.sum(Q_clean))
    
    # Flashiness index (Richards-Baker)
    flashiness = float(np.sum(np.abs(np.diff(Q_clean))) / np.sum(Q_clean))
    
    # Intermittency (fraction of flows < 0.1 * mean)
    threshold = 0.1 * mean_flow
    intermittency = float(np.sum(Q_clean < threshold) / len(Q_clean))
    
    # Classify regime type based on characteristics
    if intermittency > 0.1:
        regime_type = 'intermittent'
    elif cv > 1.5:
        regime_type = 'flashy'
    elif bfi > 0.7:
        regime_type = 'baseflow_dominated'
    else:
        regime_type = 'balanced'
    
    return {
        'cv': cv,
        'bfi': bfi,
        'flashiness': flashiness,
        'intermittency': intermittency,
        'mean_flow': mean_flow,
        'q95': q95,
        'q5': q5,
        'regime_type': regime_type,
    }


def adapt_apex_weights(flow_characteristics: Dict[str, float],
                        base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Adjust APEX weights based on flow regime characteristics.
    
    Applies heuristic rules to modify APEX component weights based on
    catchment flow regime, emphasizing relevant aspects of performance.
    
    Parameters
    ----------
    flow_characteristics : dict
        Flow regime characteristics from characterize_flow_regime()
    base_weights : dict, optional
        Base weight configuration to modify. If None, uses APEX defaults.
    
    Returns
    -------
    dict
        Adapted weight configuration suitable for apex_objective()
    
    Notes
    -----
    Adaptation rules:
    
    Flashy catchments (CV > 1.5):
        - Increase flashiness_weight (emphasize dynamics)
        - Increase fdc_high_weight (flood response important)
        - Decrease baseflow_index_weight (less groundwater dominated)
    
    Baseflow-dominated (BFI > 0.7):
        - Increase baseflow_index_weight (groundwater important)
        - Increase fdc_low_weight (low flows critical)
        - Decrease flashiness_weight (stable response)
        - Decrease fdc_high_weight (floods less important)
    
    Intermittent (zero flow > 10%):
        - Increase fdc_low_weight (low flow matching critical)
        - Increase baseflow_index_weight (understand dry periods)
        - Reduce kge_sqrt_weight (sqrt transform problematic with zeros)
    
    Balanced:
        - Use default weights
    
    Examples
    --------
    >>> regime = characterize_flow_regime(observed_flow)
    >>> adapted_weights = adapt_apex_weights(regime)
    >>> apex = apex_objective(**adapted_weights)
    """
    # Default APEX weights
    if base_weights is None:
        weights = {
            'core_metric_1_weight': 0.25,
            'core_metric_2_weight': 0.15,
            'fdc_high_weight': 0.10,
            'fdc_mid_weight': 0.10,
            'fdc_low_weight': 0.10,
            'baseflow_index_weight': 0.10,
            'flashiness_weight': 0.10,
            'pbias_weight': 0.05,
            'timing_correlation_weight': 0.05,
        }
    else:
        weights = base_weights.copy()
    
    regime_type = flow_characteristics['regime_type']
    cv = flow_characteristics['cv']
    bfi = flow_characteristics['bfi']
    intermittency = flow_characteristics['intermittency']
    
    # Apply adaptations based on regime type
    if regime_type == 'flashy':
        # Emphasize dynamics and high flows
        weights['flashiness_weight'] = 0.15
        weights['fdc_high_weight'] = 0.15
        weights['baseflow_index_weight'] = 0.05
        weights['core_metric_1_weight'] = 0.25
        weights['core_metric_2_weight'] = 0.10
        weights['fdc_mid_weight'] = 0.10
        weights['fdc_low_weight'] = 0.08
        weights['pbias_weight'] = 0.05
        weights['timing_correlation_weight'] = 0.07
        
    elif regime_type == 'baseflow_dominated':
        # Emphasize low flows and baseflow
        weights['baseflow_index_weight'] = 0.15
        weights['fdc_low_weight'] = 0.15
        weights['flashiness_weight'] = 0.05
        weights['fdc_high_weight'] = 0.08
        weights['core_metric_1_weight'] = 0.22
        weights['core_metric_2_weight'] = 0.15
        weights['fdc_mid_weight'] = 0.10
        weights['pbias_weight'] = 0.05
        weights['timing_correlation_weight'] = 0.05
        
    elif regime_type == 'intermittent':
        # Emphasize low flow matching and process understanding
        weights['fdc_low_weight'] = 0.15
        weights['baseflow_index_weight'] = 0.12
        weights['core_metric_1_weight'] = 0.28
        weights['core_metric_2_weight'] = 0.10  # Reduced (sqrt transform problematic with zeros)
        weights['fdc_high_weight'] = 0.10
        weights['fdc_mid_weight'] = 0.10
        weights['flashiness_weight'] = 0.08
        weights['pbias_weight'] = 0.05
        weights['timing_correlation_weight'] = 0.02
        
    else:  # balanced
        # Use defaults (already set)
        pass
    
    # Ensure weights sum to 1.0 (normalize)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights


def apex_adaptive(Q_obs: np.ndarray, 
                   core_metric_type: str = 'kge',
                   core_metric_1_transform: Optional[str] = None,
                   core_metric_2_transform: str = 'sqrt',
                   kge_variant: str = '2012',
                   window: int = 5,
                   verbose: bool = True) -> WeightedObjective:
    """
    Create APEX with automatically adapted weights based on flow regime.
    
    Analyzes the observed flow time series to characterize the catchment's
    flow regime, then adapts APEX component weights to emphasize relevant
    aspects of hydrological performance.
    
    Parameters
    ----------
    Q_obs : np.ndarray
        Observed flow time series
    core_metric_type : str, default='kge'
        Type of core metric to use: 'kge' or 'nse'
    core_metric_1_transform : str, optional
        Flow transformation for first core metric (None, 'sqrt', 'log', 'inverse')
    core_metric_2_transform : str, default='sqrt'
        Flow transformation for second core metric
    kge_variant : str, default='2012'
        KGE variant to use when core_metric_type='kge': '2009', '2012', or '2021'
    window : int, default=5
        Window size for baseflow separation (days)
    verbose : bool, default=True
        If True, print regime characterization and weight adjustments
    
    Returns
    -------
    WeightedObjective
        APEX objective function with adapted weights
    
    Notes
    -----
    This function provides an automated way to configure APEX for catchments
    with different hydrological behaviors. The adaptation emphasizes:
    
    - Flood response for flashy catchments
    - Baseflow processes for groundwater-dominated catchments
    - Low flow matching for intermittent streams
    - Balanced performance for typical catchments
    
    The adapted weights can still be manually overridden if needed by
    extracting the weights and creating a custom apex_objective().
    
    Examples
    --------
    >>> from pyrrm.objectives import apex_adaptive
    >>> from pyrrm.calibration import CalibrationRunner
    >>> 
    >>> # Automatic adaptation with default KGE
    >>> apex = apex_adaptive(observed_flow)
    >>> 
    >>> # Automatic adaptation with NSE-based metrics
    >>> apex_nse = apex_adaptive(observed_flow, core_metric_type='nse',
    ...                           core_metric_1_transform=None,
    ...                           core_metric_2_transform='log')
    >>> 
    >>> # Calibrate
    >>> runner = CalibrationRunner(model, inputs, observed_flow, objective=apex)
    >>> result = runner.run_sceua_direct(max_iterations=10000)
    
    >>> # Check what regime was detected
    >>> from pyrrm.objectives.composite.adaptive import characterize_flow_regime
    >>> regime = characterize_flow_regime(observed_flow)
    >>> print(f"Detected regime: {regime['regime_type']}")
    >>> print(f"CV: {regime['cv']:.2f}, BFI: {regime['bfi']:.2f}")
    
    See Also
    --------
    apex_objective : Create APEX with manual weight specification
    characterize_flow_regime : Analyze flow regime characteristics
    adapt_apex_weights : Get adapted weights without creating objective
    """
    from pyrrm.objectives.composite.factories import apex_objective
    
    # Characterize flow regime
    regime = characterize_flow_regime(Q_obs, window=window)
    
    # Adapt weights
    adapted_weights = adapt_apex_weights(regime)
    
    if verbose:
        print("\n" + "="*70)
        print("APEX ADAPTIVE WEIGHTING")
        print("="*70)
        print(f"\nFlow Regime Characteristics:")
        print(f"  Regime type:        {regime['regime_type']}")
        print(f"  CV:                 {regime['cv']:.3f}")
        print(f"  Baseflow index:     {regime['bfi']:.3f}")
        print(f"  Flashiness:         {regime['flashiness']:.3f}")
        print(f"  Intermittency:      {regime['intermittency']:.3f}")
        print(f"  Mean flow:          {regime['mean_flow']:.2f}")
        
        print(f"\nCore Metric Configuration:")
        print(f"  Type:               {core_metric_type.upper()}")
        print(f"  Metric 1 transform: {core_metric_1_transform or 'None'}")
        print(f"  Metric 2 transform: {core_metric_2_transform or 'None'}")
        if core_metric_type == 'kge':
            print(f"  KGE variant:        {kge_variant}")
        
        print(f"\nAdapted APEX Weights:")
        for name, weight in sorted(adapted_weights.items()):
            print(f"  {name:30s}: {weight:.3f}")
        print(f"\nTotal: {sum(adapted_weights.values()):.3f}")
        print("="*70 + "\n")
    
    # Create APEX with adapted weights
    return apex_objective(
        **adapted_weights, 
        core_metric_type=core_metric_type,
        core_metric_1_transform=core_metric_1_transform,
        core_metric_2_transform=core_metric_2_transform,
        kge_variant=kge_variant
    )
