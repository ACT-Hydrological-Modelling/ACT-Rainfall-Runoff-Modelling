"""
Flow Duration Curve computation utilities.

This module provides functions for computing and analyzing flow duration curves,
which summarize the frequency distribution of streamflow magnitudes.

References
----------
Yilmaz, K.K., Gupta, H.V., Wagener, T. (2008). A process-based diagnostic 
approach to model evaluation: Application to the NWS distributed hydrologic 
model. Water Resources Research, 44(9), W09417.

Westerberg, I.K. et al. (2011). Calibration of hydrological models using 
flow-duration curves. HESS, 15, 2205-2227.
"""

from typing import Tuple, Optional
import numpy as np


def compute_fdc(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute flow duration curve using Weibull plotting position.
    
    The FDC shows the percentage of time a given flow is equalled or exceeded.
    Flows are sorted in descending order, and exceedance probabilities are
    calculated using the Weibull plotting position formula.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series (1D array)
    
    Returns
    -------
    exceedance : np.ndarray
        Exceedance probabilities (0 to 1), representing the fraction of time
        each flow is equalled or exceeded
    flows : np.ndarray
        Sorted flows in descending order (highest flows first)
    
    Notes
    -----
    Uses Weibull plotting position formula:
        p_i = i / (n + 1)
    
    where i is the rank (1 = highest flow) and n is the sample size.
    This is unbiased for all distributions.
    
    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([10, 5, 20, 15, 8, 3, 12])
    >>> exc, flows = compute_fdc(Q)
    >>> # exc[0] is ~0.125 (12.5%), flows[0] is 20 (highest)
    >>> # exc[-1] is ~0.875 (87.5%), flows[-1] is 3 (lowest)
    """
    # Remove NaN values
    Q_clean = Q[~np.isnan(Q)]
    
    if len(Q_clean) == 0:
        return np.array([]), np.array([])
    
    # Sort in descending order (highest flows first)
    Q_sorted = np.sort(Q_clean)[::-1]
    n = len(Q_sorted)
    
    # Weibull plotting position
    exceedance = np.arange(1, n + 1) / (n + 1)
    
    return exceedance, Q_sorted


def compute_fdc_at_exceedance(Q: np.ndarray, 
                               exceedance_probs: np.ndarray) -> np.ndarray:
    """
    Compute flow values at specified exceedance probabilities.
    
    Uses linear interpolation to estimate flow values at arbitrary
    exceedance probabilities.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    exceedance_probs : np.ndarray
        Target exceedance probabilities (0 to 1)
    
    Returns
    -------
    np.ndarray
        Interpolated flow values at each exceedance probability
    
    Examples
    --------
    >>> Q = np.random.lognormal(3, 1, 365)
    >>> Q10 = compute_fdc_at_exceedance(Q, np.array([0.10]))[0]  # 10% exceedance
    >>> Q90 = compute_fdc_at_exceedance(Q, np.array([0.90]))[0]  # 90% exceedance
    """
    exc, flows = compute_fdc(Q)
    
    if len(exc) == 0:
        return np.full_like(exceedance_probs, np.nan)
    
    # Interpolate (note: exc is increasing, flows is decreasing)
    return np.interp(exceedance_probs, exc, flows)


def compute_fdc_slope(Q: np.ndarray, 
                       lower_exc: float = 0.33,
                       upper_exc: float = 0.66) -> float:
    """
    Compute the slope of the FDC between two exceedance probabilities.
    
    The FDC slope characterizes the variability of flow regime.
    Steep slopes indicate flashy catchments, while gentle slopes
    indicate more sustained baseflow.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    lower_exc : float, default=0.33
        Lower exceedance probability (higher flow end)
    upper_exc : float, default=0.66
        Upper exceedance probability (lower flow end)
    
    Returns
    -------
    float
        Slope of FDC in the specified segment (typically negative)
    
    Notes
    -----
    Slope is calculated in log-space to better represent the
    typical log-linear relationship in FDCs:
        slope = (log(Q_upper) - log(Q_lower)) / (upper_exc - lower_exc)
    
    Examples
    --------
    >>> slope = compute_fdc_slope(Q, lower_exc=0.33, upper_exc=0.66)
    """
    flows = compute_fdc_at_exceedance(Q, np.array([lower_exc, upper_exc]))
    
    Q_lower = flows[0]  # Higher flow (lower exceedance)
    Q_upper = flows[1]  # Lower flow (higher exceedance)
    
    if Q_lower <= 0 or Q_upper <= 0:
        return np.nan
    
    # Slope in log-space
    log_slope = (np.log(Q_upper) - np.log(Q_lower)) / (upper_exc - lower_exc)
    
    return log_slope


def get_fdc_segment(Q: np.ndarray, 
                     lower_bound: float, 
                     upper_bound: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a segment of the FDC between exceedance probability bounds.
    
    Parameters
    ----------
    Q : np.ndarray
        Flow time series
    lower_bound : float
        Lower exceedance probability (0 to 1)
    upper_bound : float
        Upper exceedance probability (0 to 1), must be > lower_bound
    
    Returns
    -------
    exceedance_segment : np.ndarray
        Exceedance probabilities within the segment
    flow_segment : np.ndarray
        Corresponding flow values
    
    Examples
    --------
    >>> # Get high flow segment (2-20% exceedance)
    >>> exc_high, flow_high = get_fdc_segment(Q, 0.02, 0.20)
    """
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be less than upper_bound")
    
    exc, flows = compute_fdc(Q)
    
    if len(exc) == 0:
        return np.array([]), np.array([])
    
    mask = (exc >= lower_bound) & (exc <= upper_bound)
    
    return exc[mask], flows[mask]
