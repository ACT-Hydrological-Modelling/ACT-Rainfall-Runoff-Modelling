"""
Nonlinear Muskingum river routing implementation.

This module provides the NonlinearMuskingumRouter class which implements
the nonlinear Muskingum method with S = K * Q^m storage-discharge relationship.

Mathematical Background:
    The nonlinear Muskingum method is based on two fundamental equations:
    
    1. Continuity Equation (Mass Balance):
       dS/dt = I(t) - Q(t)
       
    2. Nonlinear Storage-Discharge Relationship:
       S = K * Q^m
       
    Where:
        S = storage volume in the reach [L³]
        I(t) = inflow rate [L³/T]
        Q(t) = outflow rate [L³/T]
        K = storage constant [T] (approximately the travel time)
        m = nonlinear exponent [dimensionless]
            - m = 1.0: Linear reservoir behavior
            - m < 1.0: More attenuation at low flows (typical)
            - m > 1.0: More attenuation at high flows (rare)

Performance:
    This module uses Numba JIT compilation when available for significant
    speedup (10-50x faster). Falls back to pure Python if Numba is not installed.

References:
    - Gill, M.A. (1978). Flood routing by the Muskingum method.
      Journal of Hydrology, 36(3-4), 353-363.
    - O'Sullivan, J.J., et al. (2012). Nonlinear Muskingum flood routing.
      Journal of Hydraulic Engineering.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import warnings

from pyrrm.routing.base import BaseRouter

# =============================================================================
# Numba JIT Compilation (Optional - falls back to pure Python if unavailable)
# =============================================================================

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator that just returns the function unchanged
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _solve_timestep_jit(
    I_current: float,
    I_next: float,
    Q_current: float,
    K_sub: float,
    m: float,
    dt: float,
    min_Q: float,
    max_iter: int,
    abs_tol: float,
    rel_tol: float
) -> Tuple[float, int, bool]:
    """
    JIT-compiled Newton-Raphson solver for single timestep.
    
    Solves: K*Q^m + (dt/2)*Q - C = 0
    where C = K*Q_old^m - (dt/2)*Q_old + dt*I_avg
    """
    # Average inflow over timestep
    I_avg = (I_current + I_next) / 2.0
    
    # Ensure minimum values
    Q_old = Q_current if Q_current > min_Q else min_Q
    
    # Known constant from current timestep
    C = K_sub * (Q_old ** m) - (dt / 2.0) * Q_old + dt * I_avg
    
    # Initial guess
    Q = Q_old
    
    for iteration in range(max_iter):
        # Ensure Q is positive
        if Q < min_Q:
            Q = min_Q
        
        # Evaluate function: f(Q) = K*Q^m + (dt/2)*Q - C
        f = K_sub * (Q ** m) + (dt / 2.0) * Q - C
        
        # Evaluate derivative: f'(Q) = K*m*Q^(m-1) + dt/2
        f_prime = K_sub * m * (Q ** (m - 1)) + dt / 2.0
        
        # Newton-Raphson update
        if abs(f_prime) < 1e-15:
            break
        
        delta_Q = f / f_prime
        Q_new = Q - delta_Q
        
        # Ensure non-negative
        if Q_new < min_Q:
            Q_new = min_Q
        
        # Check convergence
        if abs(f) < abs_tol:
            return Q_new, iteration + 1, True
        
        if abs(delta_Q) < rel_tol * abs(Q):
            return Q_new, iteration + 1, True
        
        Q = Q_new
    
    # Didn't converge - return best estimate
    return Q, max_iter, False


@jit(nopython=True, cache=True)
def _route_jit(
    inflow: np.ndarray,
    K_sub: float,
    m: float,
    n_subreaches: int,
    dt: float,
    initial_outflow: float,
    min_Q: float,
    max_iter: int,
    abs_tol: float,
    rel_tol: float
) -> np.ndarray:
    """
    JIT-compiled main routing loop.
    
    This is the performance-critical function that benefits most from
    Numba compilation - it eliminates Python loop overhead.
    """
    n_timesteps = len(inflow)
    outflow = np.zeros(n_timesteps)
    
    # Initialize sub-reach outflows
    subreach_outflows = np.full(n_subreaches, initial_outflow)
    outflow[0] = initial_outflow
    
    # Pre-allocate array for old outflows
    old_subreach_outflows = np.zeros(n_subreaches)
    
    # Route through time
    for t in range(n_timesteps - 1):
        # Copy current outflows
        for j in range(n_subreaches):
            old_subreach_outflows[j] = subreach_outflows[j]
        
        # Route through cascade of sub-reaches
        for j in range(n_subreaches):
            if j == 0:
                I_current = inflow[t]
                I_next = inflow[t + 1]
            else:
                I_current = old_subreach_outflows[j - 1]
                I_next = subreach_outflows[j - 1]
            
            Q_next, _, _ = _solve_timestep_jit(
                I_current, I_next,
                old_subreach_outflows[j],
                K_sub, m, dt,
                min_Q, max_iter, abs_tol, rel_tol
            )
            
            subreach_outflows[j] = Q_next
        
        outflow[t + 1] = subreach_outflows[n_subreaches - 1]
    
    return outflow


class NonlinearMuskingumRouter(BaseRouter):
    """
    Nonlinear Muskingum river routing with S = K * Q^m relationship.
    
    This router transforms an inflow hydrograph into a routed outflow
    hydrograph by solving the nonlinear storage-discharge relationship
    using Newton-Raphson iteration with Crank-Nicolson time discretization.
    
    The routing produces two effects on the hydrograph:
    1. Translation (lag): Peak is delayed by approximately K time units
    2. Attenuation: Peak is reduced and spread due to storage effects
    
    Parameters
    ----------
    K : float
        Storage constant for the entire reach [time units matching dt].
        Represents the approximate travel time through the reach.
        Larger K = more storage, more attenuation, longer lag.
        Typical range: 0.5 to 50 days depending on reach length/slope.
        
    m : float
        Nonlinear exponent controlling the storage-discharge relationship.
        - m = 1.0: Linear reservoir (storage proportional to discharge)
        - m < 1.0: More attenuation at low flows, less at high flows
                   (common in natural channels, typical range 0.6-1.0)
        - m > 1.0: Less attenuation at low flows, more at high flows (rare)
        Typical range: 0.6 to 1.0 for natural channels.
        
    n_subreaches : int, optional
        Number of sub-reaches for numerical routing. Default is 1.
        More sub-reaches produce smoother attenuation and better
        numerical stability. The storage constant is divided among
        sub-reaches (K_sub = K / n_subreaches), while m stays constant.
        Rule of thumb: n_subreaches >= K / dt for stability.
        
    solver_config : dict, optional
        Newton-Raphson solver configuration:
        - max_iterations: Maximum iterations (default: 50)
        - abs_tolerance: Absolute convergence tolerance (default: 1e-9)
        - rel_tolerance: Relative convergence tolerance (default: 1e-6)
        - min_discharge: Minimum discharge threshold (default: 1e-10)
          Prevents numerical issues when Q approaches zero.
    
    Attributes
    ----------
    K : float
        Storage constant for the total reach
    m : float
        Nonlinear exponent
    n_subreaches : int
        Number of sub-reaches
    K_sub : float
        Storage constant per sub-reach (K / n_subreaches)
    
    Examples
    --------
    Basic usage:
    
    >>> router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    >>> outflow = router.route(inflow, dt=1.0)  # dt in days
    
    With calibration:
    
    >>> from pyrrm.routing import RoutedModel
    >>> model = RoutedModel(rr_model, router)
    >>> # router parameters automatically included in calibration
    
    Notes
    -----
    The storage constant K should be in the same time units as dt.
    For daily models, K is typically in days. For hourly models,
    K would be in hours.
    
    The method conserves mass - total inflow volume equals total
    outflow volume plus change in storage.
    """
    
    # Default solver configuration
    # NOTE: Tolerances relaxed for better performance (1e-6/1e-4 vs 1e-9/1e-6)
    # These are sufficient for hydrological applications
    DEFAULT_SOLVER_CONFIG = {
        'max_iterations': 20,       # Reduced from 50 (usually converges in 2-5)
        'abs_tolerance': 1e-6,      # Relaxed from 1e-9
        'rel_tolerance': 1e-4,      # Relaxed from 1e-6
        'min_discharge': 1e-10
    }
    
    # High precision config available if needed
    HIGH_PRECISION_SOLVER_CONFIG = {
        'max_iterations': 50,
        'abs_tolerance': 1e-9,
        'rel_tolerance': 1e-6,
        'min_discharge': 1e-10
    }
    
    # Parameter bounds for calibration (with routing_ prefix)
    PARAMETER_BOUNDS = {
        'routing_K': (0.1, 200.0),           # Storage constant [days]
        'routing_m': (0.3, 1.5),             # Nonlinear exponent [-]
        'routing_n_subreaches': (1, 20),     # Number of sub-reaches [-]
    }
    
    def __init__(
        self,
        K: float = 5.0,
        m: float = 0.8,
        n_subreaches: int = 1,
        solver_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the NonlinearMuskingumRouter.
        
        Args:
            K: Storage constant [time units, typically days]
            m: Nonlinear exponent (1.0 = linear)
            n_subreaches: Number of sub-reaches for routing
            solver_config: Optional solver configuration dict
        """
        # Validate and set parameters
        self._validate_parameters(K, m, n_subreaches)
        
        self.K = float(K)
        self.m = float(m)
        self.n_subreaches = int(n_subreaches)
        
        # Compute derived parameter
        self.K_sub = self.K / self.n_subreaches
        
        # Solver configuration
        self._solver_config = {**self.DEFAULT_SOLVER_CONFIG}
        if solver_config is not None:
            self._solver_config.update(solver_config)
        
        # Internal state: outflow at each sub-reach
        self._subreach_outflows: List[float] = []
        self.reset()
    
    def _validate_parameters(
        self,
        K: float,
        m: float,
        n_subreaches: int
    ) -> None:
        """Validate parameter values."""
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")
        if n_subreaches < 1:
            raise ValueError(f"n_subreaches must be >= 1, got {n_subreaches}")
        
        # Warn if parameters are outside typical ranges
        if K > 200:
            warnings.warn(f"K={K} is unusually large (typical: 0.5-50 days)")
        if m < 0.3 or m > 1.5:
            warnings.warn(f"m={m} is outside typical range (0.3-1.5)")
    
    # =========================================================================
    # Core Routing Methods
    # =========================================================================
    
    def route(
        self,
        inflow: np.ndarray,
        dt: float,
        initial_outflow: Optional[float] = None,
        use_jit: bool = True
    ) -> np.ndarray:
        """
        Route an inflow hydrograph through the reach.
        
        Args:
            inflow: Inflow time series. Shape: (n_timesteps,)
            dt: Timestep duration (must match K units, e.g., days)
            initial_outflow: Initial outflow at t=0. If None, uses
                           steady-state assumption (Q_0 = I_0).
            use_jit: If True, use Numba JIT-compiled version when available.
                    Set to False to use pure Python (for debugging).
                           
        Returns:
            Routed outflow time series. Shape: (n_timesteps,)
            
        Raises:
            ValueError: If inflow contains negative values.
        """
        # Validate inputs
        inflow = np.asarray(inflow, dtype=np.float64).flatten()
        self._validate_inputs(inflow, dt)
        
        # Initialize state
        if initial_outflow is None:
            initial_outflow = max(inflow[0], self._solver_config['min_discharge'])
        
        # Use JIT-compiled version if available and requested
        if use_jit and NUMBA_AVAILABLE:
            outflow = _route_jit(
                inflow,
                self.K_sub,
                self.m,
                self.n_subreaches,
                dt,
                initial_outflow,
                self._solver_config['min_discharge'],
                self._solver_config['max_iterations'],
                self._solver_config['abs_tolerance'],
                self._solver_config['rel_tolerance']
            )
            # Update internal state for consistency
            self._subreach_outflows = [outflow[-1]] * self.n_subreaches
            return outflow
        
        # Fall back to pure Python implementation
        return self._route_python(inflow, dt, initial_outflow)
    
    def _route_python(
        self,
        inflow: np.ndarray,
        dt: float,
        initial_outflow: float
    ) -> np.ndarray:
        """
        Pure Python routing implementation (fallback when Numba unavailable).
        """
        n_timesteps = len(inflow)
        outflow = np.zeros(n_timesteps)
        
        # Initialize sub-reach outflows
        self._subreach_outflows = [initial_outflow] * self.n_subreaches
        outflow[0] = initial_outflow
        
        # Route through time
        for t in range(n_timesteps - 1):
            # Store old sub-reach outflows before updating (needed for Crank-Nicolson)
            old_subreach_outflows = self._subreach_outflows.copy()
            
            # Route through cascade of sub-reaches
            for j in range(self.n_subreaches):
                # Determine inflow to this sub-reach at current and next timestep
                if j == 0:
                    I_current = inflow[t]
                    I_next = inflow[t + 1]
                else:
                    I_current = old_subreach_outflows[j - 1]
                    I_next = self._subreach_outflows[j - 1]
                
                # Solve for outflow at next timestep
                Q_next, n_iter, converged = self._solve_timestep(
                    I_current=I_current,
                    I_next=I_next,
                    Q_current=old_subreach_outflows[j],
                    K_sub=self.K_sub,
                    dt=dt
                )
                
                if not converged:
                    warnings.warn(
                        f"Newton-Raphson did not converge at t={t}, "
                        f"sub-reach={j} after {n_iter} iterations"
                    )
                
                self._subreach_outflows[j] = Q_next
            
            # Final outflow is from last sub-reach
            outflow[t + 1] = self._subreach_outflows[-1]
        
        return outflow
    
    def _validate_inputs(self, inflow: np.ndarray, dt: float) -> None:
        """Validate routing inputs."""
        if len(inflow) == 0:
            raise ValueError("Inflow array is empty")
        
        if np.any(inflow < 0):
            n_negative = np.sum(inflow < 0)
            raise ValueError(
                f"Inflow contains {n_negative} negative values. "
                f"Routing requires non-negative inflows."
            )
        
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        
        # Stability warning
        if dt > 2 * self.K_sub:
            warnings.warn(
                f"Timestep dt={dt} may be too large for K_sub={self.K_sub:.2f}. "
                f"For stability, dt should be <= 2*K_sub. "
                f"Consider increasing n_subreaches."
            )
    
    def _solve_timestep(
        self,
        I_current: float,
        I_next: float,
        Q_current: float,
        K_sub: float,
        dt: float
    ) -> Tuple[float, int, bool]:
        """
        Solve for outflow at next timestep using Newton-Raphson.
        
        The discretized equation (Crank-Nicolson) is:
        
        K*Q^m + (dt/2)*Q - C = 0
        
        where C = K*Q_old^m - (dt/2)*Q_old + dt*I_avg
        
        Args:
            I_current: Inflow at current timestep
            I_next: Inflow at next timestep
            Q_current: Outflow at current timestep
            K_sub: Storage constant for this sub-reach
            dt: Timestep duration
            
        Returns:
            Tuple of (Q_next, n_iterations, converged)
        """
        min_Q = self._solver_config['min_discharge']
        max_iter = self._solver_config['max_iterations']
        abs_tol = self._solver_config['abs_tolerance']
        rel_tol = self._solver_config['rel_tolerance']
        
        # Average inflow over timestep
        I_avg = (I_current + I_next) / 2.0
        
        # Ensure minimum values
        Q_old = max(Q_current, min_Q)
        
        # Known constant from current timestep
        # C = K*Q_old^m - (dt/2)*Q_old + dt*I_avg
        C = K_sub * (Q_old ** self.m) - (dt / 2.0) * Q_old + dt * I_avg
        
        # Initial guess: use current outflow or average of inflow
        Q = max(Q_old, min_Q)
        
        for iteration in range(max_iter):
            # Ensure Q is positive
            Q = max(Q, min_Q)
            
            # Evaluate function: f(Q) = K*Q^m + (dt/2)*Q - C
            f = K_sub * (Q ** self.m) + (dt / 2.0) * Q - C
            
            # Evaluate derivative: f'(Q) = K*m*Q^(m-1) + dt/2
            f_prime = K_sub * self.m * (Q ** (self.m - 1)) + dt / 2.0
            
            # Newton-Raphson update
            if abs(f_prime) < 1e-15:
                # Derivative too small, use bisection
                break
            
            delta_Q = f / f_prime
            Q_new = Q - delta_Q
            
            # Ensure non-negative
            Q_new = max(Q_new, min_Q)
            
            # Check convergence
            if abs(f) < abs_tol:
                return Q_new, iteration + 1, True
            
            if abs(delta_Q) < rel_tol * abs(Q):
                return Q_new, iteration + 1, True
            
            Q = Q_new
        
        # Newton-Raphson didn't converge, try bisection fallback
        Q_bisect = self._bisection_solve(Q_old, I_avg, K_sub, dt, C)
        return Q_bisect, max_iter, False
    
    def _bisection_solve(
        self,
        Q_old: float,
        I_avg: float,
        K_sub: float,
        dt: float,
        C: float,
        max_iter: int = 100
    ) -> float:
        """
        Bisection fallback solver for non-convergent cases.
        
        Args:
            Q_old: Previous outflow
            I_avg: Average inflow
            K_sub: Sub-reach storage constant
            dt: Timestep
            C: Known constant
            max_iter: Maximum bisection iterations
            
        Returns:
            Estimated outflow
        """
        min_Q = self._solver_config['min_discharge']
        
        # Define search bounds
        Q_low = min_Q
        Q_high = max(Q_old, I_avg) * 10  # Upper bound
        
        def f(Q):
            return K_sub * (Q ** self.m) + (dt / 2.0) * Q - C
        
        # Check if solution exists in bounds
        f_low = f(Q_low)
        f_high = f(Q_high)
        
        if f_low * f_high > 0:
            # No sign change, return best guess
            return Q_old if abs(f(Q_old)) < abs(f(I_avg)) else I_avg
        
        for _ in range(max_iter):
            Q_mid = (Q_low + Q_high) / 2.0
            f_mid = f(Q_mid)
            
            if abs(f_mid) < self._solver_config['abs_tolerance']:
                return Q_mid
            
            if f_mid * f_low < 0:
                Q_high = Q_mid
            else:
                Q_low = Q_mid
            
            if Q_high - Q_low < self._solver_config['rel_tolerance']:
                return Q_mid
        
        return (Q_low + Q_high) / 2.0
    
    # =========================================================================
    # Parameter Management (BaseRouter interface)
    # =========================================================================
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Return routing parameter bounds for calibration.
        
        Returns:
            Dictionary with 'routing_' prefixed parameter names and bounds.
        """
        return self.PARAMETER_BOUNDS.copy()
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Set routing parameters.
        
        Accepts parameters with or without 'routing_' prefix.
        
        Args:
            params: Dictionary of parameter values
        """
        # Normalize parameter names (handle both prefixed and unprefixed)
        normalized = {}
        for key, value in params.items():
            if key.startswith('routing_'):
                normalized[key[8:]] = value  # Remove prefix
            else:
                normalized[key] = value
        
        # Update parameters
        if 'K' in normalized:
            K = float(normalized['K'])
            if K <= 0:
                raise ValueError(f"K must be positive, got {K}")
            self.K = K
        
        if 'm' in normalized:
            m = float(normalized['m'])
            if m <= 0:
                raise ValueError(f"m must be positive, got {m}")
            self.m = m
        
        if 'n_subreaches' in normalized:
            n = int(round(normalized['n_subreaches']))
            # Clamp to minimum of 1 (don't raise error - optimizer may sample edge values)
            self.n_subreaches = max(1, n)
        
        # Update derived parameter
        self.K_sub = self.K / self.n_subreaches
        
        # Reset state with new parameters
        self.reset()
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current routing parameters with 'routing_' prefix.
        
        Returns:
            Dictionary of current parameter values.
        """
        return {
            'routing_K': self.K,
            'routing_m': self.m,
            'routing_n_subreaches': float(self.n_subreaches),
        }
    
    def reset(self) -> None:
        """Reset router state to initial conditions."""
        self._subreach_outflows = [0.0] * self.n_subreaches
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Get current router state."""
        return {
            'subreach_outflows': self._subreach_outflows.copy(),
            'K': self.K,
            'm': self.m,
            'n_subreaches': self.n_subreaches,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore router state."""
        if 'subreach_outflows' in state:
            self._subreach_outflows = list(state['subreach_outflows'])
        if 'K' in state:
            self.K = state['K']
        if 'm' in state:
            self.m = state['m']
        if 'n_subreaches' in state:
            self.n_subreaches = state['n_subreaches']
            self.K_sub = self.K / self.n_subreaches
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def calculate_storage(self, outflow: float) -> float:
        """
        Calculate storage volume for given outflow.
        
        Uses the nonlinear relationship: S = K * Q^m
        
        Args:
            outflow: Discharge rate [L³/T]
            
        Returns:
            Storage volume [L³*T] (e.g., ML*day if Q is ML/day)
        """
        Q = max(outflow, self._solver_config['min_discharge'])
        return self.K * (Q ** self.m)
    
    def estimate_lag(self) -> float:
        """
        Estimate the approximate lag time through the reach.
        
        For the nonlinear Muskingum method, the lag is approximately
        equal to K under steady-state conditions.
        
        Returns:
            Estimated lag time [same units as K]
        """
        return self.K
    
    def recommend_subreaches(self, dt: float) -> int:
        """
        Recommend number of sub-reaches for given timestep.
        
        Uses the rule of thumb: n >= K/dt for numerical stability.
        
        Args:
            dt: Timestep duration [same units as K]
            
        Returns:
            Recommended number of sub-reaches
        """
        n_recommended = int(np.ceil(self.K / dt))
        return max(1, n_recommended)
    
    def summary(self) -> str:
        """Return detailed summary of router configuration."""
        jit_status = "Enabled (Numba)" if NUMBA_AVAILABLE else "Disabled (Pure Python)"
        lines = [
            "NonlinearMuskingumRouter Configuration",
            "=" * 40,
            f"Storage constant K: {self.K:.2f}",
            f"Nonlinear exponent m: {self.m:.3f}",
            f"Number of sub-reaches: {self.n_subreaches}",
            f"K per sub-reach: {self.K_sub:.2f}",
            "",
            "Solver Configuration:",
            f"  Max iterations: {self._solver_config['max_iterations']}",
            f"  Abs tolerance: {self._solver_config['abs_tolerance']:.0e}",
            f"  Rel tolerance: {self._solver_config['rel_tolerance']:.0e}",
            f"  Min discharge: {self._solver_config['min_discharge']:.0e}",
            "",
            f"Performance: JIT Compilation {jit_status}",
            f"Estimated lag time: ~{self.estimate_lag():.1f} time units",
        ]
        return "\n".join(lines)
    
    @staticmethod
    def is_jit_available() -> bool:
        """Check if Numba JIT compilation is available."""
        return NUMBA_AVAILABLE
    
    def copy(self) -> 'NonlinearMuskingumRouter':
        """Return a deep copy of the router."""
        new_router = NonlinearMuskingumRouter(
            K=self.K,
            m=self.m,
            n_subreaches=self.n_subreaches,
            solver_config=self._solver_config.copy()
        )
        new_router._subreach_outflows = self._subreach_outflows.copy()
        return new_router
    
    def __repr__(self) -> str:
        return (
            f"NonlinearMuskingumRouter(K={self.K}, m={self.m}, "
            f"n_subreaches={self.n_subreaches})"
        )
