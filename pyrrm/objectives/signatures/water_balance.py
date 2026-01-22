"""
Water balance hydrological signatures.

This module provides functions for computing water balance-related
hydrological signatures such as runoff ratio and baseflow index.

References
----------
Sawicz, K., Wagener, T., Sivapalan, M., Troch, P.A., Carrillo, G. (2011).
Catchment classification: empirical analysis of hydrologic similarity 
based on catchment function in the eastern USA. Hydrology and Earth 
System Sciences, 15, 2895-2911.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import minimum_filter1d


def compute_runoff_ratio(Q: np.ndarray, 
                          P: np.ndarray,
                          area_km2: Optional[float] = None) -> float:
    """
    Compute the runoff ratio (runoff coefficient).
    
    The runoff ratio is the fraction of precipitation that becomes
    streamflow, indicating catchment wetness and permeability.
    
    Formula
    -------
    RR = Σ Q / Σ P
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series (mm or volume units)
    P : np.ndarray
        Precipitation time series (same units as Q)
    area_km2 : float, optional
        Catchment area in km². If provided and Q is in m³/s,
        Q is converted to mm before calculation.
    
    Returns
    -------
    float
        Runoff ratio (typically 0.1 to 0.8)
        - Low values: Dry/permeable catchments
        - High values: Wet/impermeable catchments
    
    Notes
    -----
    - Should be calculated over complete water years
    - Values > 1 indicate measurement errors or external inputs
    """
    Q_clean = Q.copy()
    P_clean = P.copy()
    
    # Remove NaN pairs
    mask = ~(np.isnan(Q_clean) | np.isnan(P_clean))
    Q_clean = Q_clean[mask]
    P_clean = P_clean[mask]
    
    if len(Q_clean) == 0:
        return np.nan
    
    # Convert Q from m³/s to mm if area provided
    if area_km2 is not None:
        # Assume daily data: m³/s × 86400 s/day / (km² × 10⁶ m²/km²) × 1000 mm/m
        Q_clean = Q_clean * 86400 / (area_km2 * 1e6) * 1000
    
    sum_P = np.sum(P_clean)
    if sum_P == 0:
        return np.nan
    
    return np.sum(Q_clean) / sum_P


def compute_baseflow_index(Q: np.ndarray, 
                            window: int = 5,
                            passes: int = 3) -> float:
    """
    Compute the Baseflow Index (BFI) using digital filter method.
    
    The BFI is the ratio of baseflow to total flow, indicating
    groundwater contribution to streamflow.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    window : int, default=5
        Window size for minimum filter (days)
    passes : int, default=3
        Number of filter passes for smoothing
    
    Returns
    -------
    float
        Baseflow index (0 to 1)
        - Low values: Surface runoff dominated
        - High values: Groundwater dominated
    
    Notes
    -----
    Uses the UKIH (UK Institute of Hydrology) smoothed minima method:
    1. Find local minima using minimum filter
    2. Apply multiple passes for smoothing
    3. Ensure baseflow never exceeds total flow
    
    References
    ----------
    Institute of Hydrology (1980). Low Flow Studies report.
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < window * 2:
        return np.nan
    
    sum_Q = np.sum(Q_clean)
    if sum_Q == 0:
        return np.nan
    
    # Initial baseflow estimate using minimum filter
    baseflow = Q_clean.copy()
    
    for _ in range(passes):
        # Apply minimum filter
        baseflow = minimum_filter1d(baseflow, size=window)
    
    # Ensure baseflow doesn't exceed total flow
    baseflow = np.minimum(baseflow, Q_clean)
    
    return np.sum(baseflow) / sum_Q


def compute_baseflow_recession_constant(Q: np.ndarray,
                                         window: int = 5) -> float:
    """
    Compute baseflow recession constant.
    
    Estimates the recession constant from baseflow time series,
    characterizing aquifer drainage dynamics.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    window : int, default=5
        Window for baseflow separation
    
    Returns
    -------
    float
        Recession constant (typically 0.95-0.99 for daily data)
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < window * 2:
        return np.nan
    
    # Separate baseflow
    baseflow = minimum_filter1d(Q_clean, size=window)
    baseflow = np.minimum(baseflow, Q_clean)
    
    # Find recession periods (decreasing baseflow)
    dB = np.diff(baseflow)
    recession_mask = dB < 0
    
    if np.sum(recession_mask) < 10:
        return np.nan
    
    # Calculate recession ratios
    ratios = baseflow[1:][recession_mask] / baseflow[:-1][recession_mask]
    ratios = ratios[(ratios > 0) & (ratios < 1)]
    
    if len(ratios) == 0:
        return np.nan
    
    return np.median(ratios)


def compute_streamflow_elasticity(Q: np.ndarray, 
                                   P: np.ndarray) -> float:
    """
    Compute streamflow elasticity with respect to precipitation.
    
    Measures how sensitive streamflow is to changes in precipitation.
    
    Formula
    -------
    ε = median((dQ/Q) / (dP/P))
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    P : np.ndarray
        Precipitation time series
    
    Returns
    -------
    float
        Elasticity coefficient
        - ε = 1: Linear response
        - ε > 1: Amplified response (wet catchments)
        - ε < 1: Dampened response (dry catchments)
    
    Notes
    -----
    Typically calculated on annual or seasonal aggregates.
    """
    # Remove NaN pairs
    mask = ~(np.isnan(Q) | np.isnan(P))
    Q_clean = Q[mask]
    P_clean = P[mask]
    
    if len(Q_clean) < 3:
        return np.nan
    
    mean_Q = np.mean(Q_clean)
    mean_P = np.mean(P_clean)
    
    if mean_Q == 0 or mean_P == 0:
        return np.nan
    
    # Percent changes
    dQ_Q = (Q_clean - mean_Q) / mean_Q
    dP_P = (P_clean - mean_P) / mean_P
    
    # Avoid division by zero
    valid = np.abs(dP_P) > 0.01
    
    if np.sum(valid) < 3:
        return np.nan
    
    elasticity = dQ_Q[valid] / dP_P[valid]
    
    return np.median(elasticity)
