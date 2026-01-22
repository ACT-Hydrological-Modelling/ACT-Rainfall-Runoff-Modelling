"""
S-curve functions for unit hydrograph generation.

These functions are used by the GR family of models (GR4J, GR5J, GR6J)
to generate unit hydrograph ordinates.

Ported from the Rust implementation in hydrogr.
"""

import numpy as np
from typing import Union


def s_curve1(t: Union[int, float], x4: float, exp: float = 2.5) -> float:
    """
    Unit hydrograph ordinates for UH1 derived from S-curves.
    
    This S-curve is used for the portion of flow routed through the
    routing store in GR models.
    
    Args:
        t: Time index (integer or float)
        x4: Unit hydrograph time constant [days]
        exp: Exponent for the S-curve shape (default 2.5)
        
    Returns:
        S-curve ordinate value at time t
        
    Example:
        >>> s_curve1(1, 2.0, 2.5)
        0.1767766952966369
    """
    t = float(t)
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** exp
    return 1.0


def s_curve2(t: Union[int, float], x4: float, exp: float = 2.5) -> float:
    """
    Unit hydrograph ordinates for UH2 derived from S-curves.
    
    This S-curve is used for the direct flow component in GR models.
    It has a longer base than UH1 (2 * x4 instead of x4).
    
    Args:
        t: Time index (integer or float)
        x4: Unit hydrograph time constant [days]
        exp: Exponent for the S-curve shape (default 2.5)
        
    Returns:
        S-curve ordinate value at time t
        
    Example:
        >>> s_curve2(1, 2.0, 2.5)
        0.08838834764831845
    """
    t = float(t)
    if t <= 0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** exp
    elif t < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** exp
    return 1.0


def compute_uh1_ordinates(x4: float, exp: float = 2.5) -> np.ndarray:
    """
    Compute the unit hydrograph UH1 ordinates.
    
    Args:
        x4: Unit hydrograph time constant [days]
        exp: Exponent for the S-curve shape (default 2.5)
        
    Returns:
        Array of UH1 ordinates
    """
    n_uh1 = int(np.ceil(x4))
    ordinates = np.zeros(n_uh1)
    
    for i in range(1, n_uh1 + 1):
        ordinates[i - 1] = s_curve1(i, x4, exp) - s_curve1(i - 1, x4, exp)
    
    return ordinates


def compute_uh2_ordinates(x4: float, exp: float = 2.5) -> np.ndarray:
    """
    Compute the unit hydrograph UH2 ordinates.
    
    Args:
        x4: Unit hydrograph time constant [days]
        exp: Exponent for the S-curve shape (default 2.5)
        
    Returns:
        Array of UH2 ordinates
    """
    n_uh2 = int(np.ceil(2.0 * x4))
    ordinates = np.zeros(n_uh2)
    
    for i in range(1, n_uh2 + 1):
        ordinates[i - 1] = s_curve2(i, x4, exp) - s_curve2(i - 1, x4, exp)
    
    return ordinates
