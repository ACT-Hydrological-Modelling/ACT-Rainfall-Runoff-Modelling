"""
Constants and configuration values for hydrological objective functions.

This module provides standardized constants used across the objectives library,
including FDC segment definitions, benchmark values, and transformation mappings.
"""

from typing import Dict, Tuple

# =============================================================================
# Flow Duration Curve Segments
# =============================================================================

# FDC segment definitions (exceedance probability bounds)
# Based on Yilmaz et al. (2008) process-based diagnostic approach
FDC_SEGMENTS: Dict[str, Tuple[float, float]] = {
    'peak': (0.00, 0.02),      # 0-2%: Peak flows from large precipitation events
    'high': (0.02, 0.20),      # 2-20%: High flows, quick runoff, snowmelt
    'mid': (0.20, 0.70),       # 20-70%: Mid-range, intermediate baseflow response
    'low': (0.70, 0.95),       # 70-95%: Low flows, slow baseflow, groundwater
    'very_low': (0.95, 1.00),  # 95-100%: Very low flows, drought conditions
    'all': (0.00, 1.00),       # Full range
}

# =============================================================================
# Benchmark Values
# =============================================================================

# KGE benchmark value (Knoben et al., 2019)
# KGE values above this indicate improvement over mean flow benchmark
# Reference: Knoben, W.J.M., Freer, J.E., Woods, R.A. (2019). Technical note:
# Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency
# scores. HESS, 23, 4323-4331.
KGE_BENCHMARK: float = -0.41

# =============================================================================
# Zero-Flow Handling
# =============================================================================

# Default epsilon fraction for zero-flow handling in transformations
# Used when epsilon_method='mean_fraction' to calculate: epsilon = mean(obs) * fraction
DEFAULT_EPSILON_FRACTION: float = 0.01

# =============================================================================
# Flow Transformation Emphasis
# =============================================================================

# Transformation types and their flow emphasis
# Indicates which flow regime each transformation emphasizes
# Reference: Pushpalatha et al. (2012), Santos et al. (2018)
TRANSFORM_EMPHASIS: Dict[str, str] = {
    'none': 'high',           # No transformation - emphasizes high flows
    'squared': 'very_high',   # Q² - strongly emphasizes peak flows
    'sqrt': 'balanced',       # √Q - balanced between high and low
    'power': 'low_medium',    # Q^p (p<1) - moderate low flow emphasis
    'log': 'low',             # ln(Q) - emphasizes low flows
    'boxcox': 'balanced',     # Box-Cox - adaptive, generally balanced
    'inverse': 'low',         # 1/Q - emphasizes low flows
    'inverse_squared': 'very_low',  # 1/Q² - strongly emphasizes very low flows
}

# =============================================================================
# SDEB Defaults
# =============================================================================

# Default parameters for SDEB metric (Lerat et al., 2013)
SDEB_DEFAULT_ALPHA: float = 0.1   # Low weight on chronological term
SDEB_DEFAULT_LAMBDA: float = 0.5  # Square root transform

# =============================================================================
# Signature Thresholds
# =============================================================================

# Default thresholds for flow frequency signatures
HIGH_FLOW_THRESHOLD_MULTIPLIER: float = 3.0   # Q > 3 * median
LOW_FLOW_THRESHOLD_MULTIPLIER: float = 0.2    # Q < 0.2 * mean

# =============================================================================
# APEX v2 Defaults
# =============================================================================

# APEX extends SDEB with novel dynamics and lag penalty multipliers.
# Reference: Built upon Lerat et al. (2013) with novel contributions.

# Core SDEB parameters (inherited)
APEX_DEFAULT_ALPHA: float = 0.1              # Weight for chronological term
APEX_DEFAULT_TRANSFORM_PARAM: float = 0.5    # Power transform exponent (sqrt)

# Bias multiplier parameters
APEX_DEFAULT_BIAS_STRENGTH: float = 1.0      # Bias penalty strength (β)
APEX_DEFAULT_BIAS_POWER: float = 1.0         # Bias penalty exponent (γ)

# Novel: Dynamics multiplier parameters
APEX_DEFAULT_DYNAMICS_STRENGTH: float = 0.5  # Gradient correlation penalty (κ)

# Novel: Lag multiplier parameters
APEX_DEFAULT_LAG_STRENGTH: float = 0.3       # Timing offset penalty (λ)
APEX_DEFAULT_LAG_REFERENCE: int = 5          # Reference lag in timesteps (τ)

# =============================================================================
# APEX Regime Emphasis Functions
# =============================================================================

# Functions for weighting errors in the ranked term by exceedance probability.
# p = exceedance probability (0 = highest flow, 1 = lowest flow)
# These provide continuous weighting without segment discontinuities.

from typing import Callable

REGIME_EMPHASIS_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    # Uniform: Equal weight across all flows (original SDEB behavior)
    'uniform': lambda p: 1.0,
    
    # Low flow emphasis: Higher weight at high exceedance (low flows)
    'low_flow': lambda p: p,
    
    # High flow emphasis: Higher weight at low exceedance (high flows)
    'high_flow': lambda p: 1.0 - p,
    
    # Balanced: Emphasize mid-range flows (parabolic, peaks at p=0.5)
    'balanced': lambda p: 4.0 * p * (1.0 - p),
    
    # Extremes: Emphasize both tails (inverse parabolic)
    'extremes': lambda p: 1.0 - 4.0 * p * (1.0 - p),
}
