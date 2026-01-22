"""
Hydrological dynamics signatures.

This module provides functions for computing dynamics-related hydrological
signatures such as flashiness, recession characteristics, and flow variability.

References
----------
Baker, D.B., Richards, R.P., Loftus, T.T., Kramer, J.W. (2004). A new 
flashiness index: characteristics and applications to midwestern rivers 
and streams. Journal of the American Water Resources Association, 
40(2), 503-522.
"""

from typing import Tuple, Optional
import numpy as np


def compute_flashiness_index(Q: np.ndarray) -> float:
    """
    Compute the Richards-Baker flashiness index.
    
    The flashiness index measures the oscillation in flow relative to
    total flow, indicating how quickly flow changes in a catchment.
    
    Formula
    -------
    FI = Σ|Q_t - Q_{t-1}| / Σ Q_t
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series (1D array)
    
    Returns
    -------
    float
        Flashiness index value
        - Higher values indicate more flashy (variable) regime
        - Typically ranges from 0.1 (stable) to 1.0+ (very flashy)
    
    Notes
    -----
    - Sensitive to time step (designed for daily data)
    - Catchments with large storage have lower FI
    - Urban catchments typically have higher FI
    
    References
    ----------
    Baker, D.B. et al. (2004). A new flashiness index: characteristics 
    and applications. JAWRA, 40(2), 503-522.
    
    Examples
    --------
    >>> Q = np.array([10, 15, 50, 30, 12, 8, 7])
    >>> fi = compute_flashiness_index(Q)
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < 2:
        return np.nan
    
    sum_Q = np.sum(Q_clean)
    if sum_Q == 0:
        return np.nan
    
    return np.sum(np.abs(np.diff(Q_clean))) / sum_Q


def compute_rising_limb_density(Q: np.ndarray) -> float:
    """
    Compute the rising limb density.
    
    Measures the frequency of flow increases, indicating catchment
    responsiveness to rainfall events.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    
    Returns
    -------
    float
        Fraction of time steps with increasing flow (0 to 1)
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < 2:
        return np.nan
    
    dQ = np.diff(Q_clean)
    return np.sum(dQ > 0) / len(dQ)


def compute_falling_limb_density(Q: np.ndarray) -> float:
    """
    Compute the falling limb density.
    
    Measures the frequency of flow decreases, related to recession
    and drainage characteristics.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    
    Returns
    -------
    float
        Fraction of time steps with decreasing flow (0 to 1)
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < 2:
        return np.nan
    
    dQ = np.diff(Q_clean)
    return np.sum(dQ < 0) / len(dQ)


def extract_recession_segments(Q: np.ndarray, 
                                min_length: int = 5) -> list:
    """
    Extract recession segments from a flow time series.
    
    A recession segment is a continuous period of decreasing flow.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    min_length : int, default=5
        Minimum number of consecutive decreasing days to qualify
    
    Returns
    -------
    list of np.ndarray
        List of recession segments (each as array of flows)
    """
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) < min_length:
        return []
    
    segments = []
    current_segment = [Q_clean[0]]
    
    for i in range(1, len(Q_clean)):
        if Q_clean[i] < Q_clean[i - 1]:
            current_segment.append(Q_clean[i])
        else:
            if len(current_segment) >= min_length:
                segments.append(np.array(current_segment))
            current_segment = [Q_clean[i]]
    
    # Check last segment
    if len(current_segment) >= min_length:
        segments.append(np.array(current_segment))
    
    return segments


def compute_recession_constant(Q: np.ndarray, 
                                 method: str = 'linear') -> float:
    """
    Compute average recession constant from flow data.
    
    The recession constant (k) describes the rate of flow decrease
    during recession periods: Q_t = Q_0 × k^t
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    method : str, default='linear'
        Fitting method:
        - 'linear': Linear regression on log(Q)
        - 'ratio': Average of Q_t/Q_{t-1} ratios
    
    Returns
    -------
    float
        Recession constant (typically 0.9 to 0.99 for daily data)
    
    Notes
    -----
    - Values close to 1 indicate slow recession (large storage)
    - Values close to 0 indicate fast recession (flashy)
    """
    segments = extract_recession_segments(Q, min_length=5)
    
    if len(segments) == 0:
        return np.nan
    
    k_values = []
    
    for segment in segments:
        if method == 'ratio':
            # Average ratio method
            ratios = segment[1:] / segment[:-1]
            ratios = ratios[(ratios > 0) & (ratios < 1)]
            if len(ratios) > 0:
                k_values.append(np.mean(ratios))
        
        elif method == 'linear':
            # Linear regression on log(Q)
            segment_positive = segment[segment > 0]
            if len(segment_positive) >= 3:
                t = np.arange(len(segment_positive))
                log_Q = np.log(segment_positive)
                
                # Simple linear regression
                slope = np.polyfit(t, log_Q, 1)[0]
                k = np.exp(slope)
                if 0 < k < 1:
                    k_values.append(k)
    
    if len(k_values) == 0:
        return np.nan
    
    return np.mean(k_values)
