"""
RoutedModel wrapper for combining rainfall-runoff models with channel routing.

This module provides the RoutedModel class which wraps any BaseRainfallRunoffModel
with optional channel routing, enabling seamless integration with the calibration
framework.
"""

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

from pyrrm.models.base import BaseRainfallRunoffModel, ModelParameter, ModelState
from pyrrm.routing.base import BaseRouter

if TYPE_CHECKING:
    from pyrrm.routing.muskingum import NonlinearMuskingumRouter


class RoutedModel(BaseRainfallRunoffModel):
    """
    Wrapper combining rainfall-runoff model with optional channel routing.
    
    This class provides a unified interface for coupled RR+routing simulation
    and calibration. It implements the full BaseRainfallRunoffModel interface,
    so it can be used directly with CalibrationRunner and all existing
    calibration methods.
    
    Data Flow:
    
        Precipitation ──► RR Model ──► Direct Runoff ──► Router ──► Routed Outflow
                              │                            │
                              │    (if routing disabled)   │
                              └────────────────────────────┘
    
    The key benefit is that routing parameters are automatically included
    in calibration when routing is enabled. All routing parameters use the
    'routing_' prefix to distinguish them from RR model parameters.
    
    Parameters
    ----------
    rr_model : BaseRainfallRunoffModel
        The rainfall-runoff model instance (Sacramento, GR4J, etc.)
        
    router : BaseRouter, optional
        Router instance (e.g., NonlinearMuskingumRouter).
        If None, no routing is applied and the model behaves exactly
        like the underlying RR model.
        
    routing_enabled : bool, optional
        Whether routing is active. Default True if router is provided.
        Allows temporarily disabling routing without removing the router.
    
    Attributes
    ----------
    name : str
        Model identifier including routing status
    frequency : str
        Timestep frequency inherited from RR model
    
    Examples
    --------
    Basic usage with routing:
    
    >>> from pyrrm.models import Sacramento
    >>> from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
    >>> 
    >>> rr_model = Sacramento()
    >>> router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
    >>> model = RoutedModel(rr_model, router)
    >>> 
    >>> results = model.run(inputs)  # Routing applied automatically
    
    Calibration with routing parameters:
    
    >>> from pyrrm.calibration import CalibrationRunner
    >>> runner = CalibrationRunner(model, inputs, observed, objective=NSE())
    >>> result = runner.run_differential_evolution()
    >>> # result.best_parameters contains both RR and routing params
    
    Comparing routed vs unrouted:
    
    >>> model.disable_routing()
    >>> unrouted = model.run(inputs)
    >>> model.enable_routing()
    >>> routed = model.run(inputs)
    
    Notes
    -----
    The timestep for routing is inferred from the input DataFrame index.
    For daily models, dt=1.0 day. Ensure K is specified in the same
    time units as the model timestep.
    """
    
    name = "routed_model"
    description = "Coupled rainfall-runoff model with channel routing"
    
    def __init__(
        self,
        rr_model: BaseRainfallRunoffModel,
        router: Optional[BaseRouter] = None,
        routing_enabled: bool = True
    ):
        """
        Initialize the RoutedModel wrapper.
        
        Args:
            rr_model: The rainfall-runoff model to wrap
            router: Optional router for channel routing
            routing_enabled: Whether routing is active (default True)
        """
        self._rr_model = rr_model
        self._router = router
        self._routing_enabled = routing_enabled and router is not None
        
        # Inherit properties from RR model
        self.frequency = rr_model.frequency
        self.name = f"routed_{rr_model.name}" if router else rr_model.name
        
        # Inherit catchment area setting
        if hasattr(rr_model, '_catchment_area_km2'):
            self._catchment_area_km2 = rr_model._catchment_area_km2
    
    # =========================================================================
    # BaseRainfallRunoffModel Interface Implementation
    # =========================================================================
    
    @property
    def parameter_definitions(self) -> List[ModelParameter]:
        """
        Combined parameter definitions from RR model and router.
        
        Returns:
            List of ModelParameter objects for all calibratable parameters.
        """
        definitions = list(self._rr_model.parameter_definitions)
        
        if self._router is not None:
            # Add routing parameters
            definitions.extend([
                ModelParameter(
                    name='routing_K',
                    default=5.0,
                    min_bound=0.1,
                    max_bound=200.0,
                    description='Muskingum storage constant (travel time)',
                    unit='days'
                ),
                ModelParameter(
                    name='routing_m',
                    default=0.8,
                    min_bound=0.3,
                    max_bound=1.5,
                    description='Nonlinear exponent (1.0=linear)',
                    unit='-'
                ),
                ModelParameter(
                    name='routing_n_subreaches',
                    default=1,
                    min_bound=1,
                    max_bound=20,
                    description='Number of sub-reaches for routing',
                    unit='-'
                ),
            ])
        
        return definitions
    
    def get_parameter_bounds(
        self,
        include_routing: bool = True,
        include_rr: bool = True,
        routing_params: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds with fine-grained control.
        
        This method provides flexible control over which parameters
        are included for calibration.
        
        Args:
            include_routing: If True, include routing parameters
            include_rr: If True, include RR model parameters
            routing_params: If provided, only include these specific routing
                           parameters (e.g., ['routing_K', 'routing_m']).
                           If None, includes all routing parameters.
                           
        Returns:
            Dictionary mapping parameter names to (min, max) bounds.
            
        Examples:
            >>> # All parameters (default)
            >>> bounds = model.get_parameter_bounds()
            >>> 
            >>> # Only RR parameters
            >>> bounds = model.get_parameter_bounds(include_routing=False)
            >>> 
            >>> # Only routing parameters  
            >>> bounds = model.get_parameter_bounds(include_rr=False)
            >>> 
            >>> # RR + only K and m (not n_subreaches)
            >>> bounds = model.get_parameter_bounds(
            ...     routing_params=['routing_K', 'routing_m']
            ... )
        """
        bounds = {}
        
        if include_rr:
            bounds.update(self._rr_model.get_parameter_bounds())
        
        if include_routing and self._router is not None and self._routing_enabled:
            routing_bounds = self._router.get_parameter_bounds()
            if routing_params is not None:
                # Filter to only requested routing params
                routing_bounds = {
                    k: v for k, v in routing_bounds.items()
                    if k in routing_params
                }
            bounds.update(routing_bounds)
        
        return bounds
    
    def get_rr_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get ONLY the rainfall-runoff model parameter bounds.
        
        Use this when you want to calibrate the RR model parameters
        while keeping routing parameters fixed.
        
        Returns:
            Dictionary of RR model parameter bounds (no routing_ prefix params)
        """
        return self._rr_model.get_parameter_bounds()
    
    def get_routing_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get ONLY the routing parameter bounds.
        
        Use this when you want to calibrate routing parameters
        while keeping RR model parameters fixed.
        
        Returns:
            Dictionary of routing parameter bounds (routing_ prefixed params)
        """
        if self._router is None:
            return {}
        return self._router.get_parameter_bounds()
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Set parameters for both RR model and router.
        
        Parameters are automatically routed to the appropriate component
        based on the 'routing_' prefix.
        
        Args:
            params: Dictionary of parameter names to values.
                   Parameters starting with 'routing_' go to the router.
                   All other parameters go to the RR model.
        """
        # Separate parameters by component
        rr_params = {}
        routing_params = {}
        
        for name, value in params.items():
            if name.startswith('routing_'):
                routing_params[name] = value
            else:
                rr_params[name] = value
        
        # Set RR model parameters
        if rr_params:
            self._rr_model.set_parameters(rr_params)
        
        # Set routing parameters (with integer handling for n_subreaches)
        if routing_params and self._router is not None:
            if 'routing_n_subreaches' in routing_params:
                routing_params['routing_n_subreaches'] = int(
                    round(routing_params['routing_n_subreaches'])
                )
            self._router.set_parameters(routing_params)
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values from both RR model and router.
        
        Returns:
            Dictionary mapping parameter names to current values.
            Includes both RR model parameters and routing parameters
            (prefixed with 'routing_').
        """
        params = self._rr_model.get_parameters().copy()
        
        if self._router is not None:
            params.update(self._router.get_parameters())
        
        return params
    
    def set_fixed_routing_parameters(
        self,
        K: Optional[float] = None,
        m: Optional[float] = None,
        n_subreaches: Optional[int] = None
    ) -> None:
        """
        Set routing parameters to fixed values (not calibrated).
        
        Call this before calibration to fix routing parameters at
        known/estimated values. Then use get_rr_parameter_bounds()
        when creating the CalibrationRunner to exclude routing from
        calibration.
        
        Args:
            K: Fixed storage constant value [days]
            m: Fixed nonlinear exponent value
            n_subreaches: Fixed number of sub-reaches
            
        Example:
            >>> model.set_fixed_routing_parameters(K=4.5, m=0.75, n_subreaches=3)
            >>> runner = CalibrationRunner(
            ...     model=model,
            ...     inputs=inputs,
            ...     observed=observed,
            ...     parameter_bounds=model.get_rr_parameter_bounds()
            ... )
        """
        params = {}
        if K is not None:
            params['routing_K'] = K
        if m is not None:
            params['routing_m'] = m
        if n_subreaches is not None:
            params['routing_n_subreaches'] = n_subreaches
        
        if params and self._router is not None:
            self._router.set_parameters(params)
    
    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Run coupled simulation: RR model followed by optional routing.
        
        Args:
            inputs: DataFrame with DatetimeIndex containing:
                - 'precipitation' (or 'rainfall'): Daily precipitation [mm]
                - 'pet' (or 'evapotranspiration'): Daily potential ET [mm]
                
        Returns:
            DataFrame with model outputs. If routing is enabled:
                - 'runoff' or 'flow': Final routed hydrograph
                - 'direct_runoff': Pre-routing runoff from RR model
            If routing is disabled, returns unmodified RR model output.
        """
        # Step 1: Run RR model
        results = self._rr_model.run(inputs)
        
        # Step 2: Apply routing if enabled
        if self._routing_enabled and self._router is not None:
            # Identify flow column
            if 'flow' in results.columns:
                flow_col = 'flow'
            elif 'runoff' in results.columns:
                flow_col = 'runoff'
            else:
                # Use first numeric column
                flow_col = results.select_dtypes(include=[np.number]).columns[0]
            
            direct_runoff = results[flow_col].values
            
            # Calculate timestep from index
            dt = self._infer_timestep(inputs)
            
            # Route the hydrograph
            routed_flow = self._router.route(direct_runoff, dt=dt)
            
            # Store both in results
            results['direct_runoff'] = direct_runoff
            results[flow_col] = routed_flow  # Replace with routed
        
        return results
    
    def _infer_timestep(self, inputs: pd.DataFrame) -> float:
        """
        Infer timestep duration from input DataFrame index.
        
        Args:
            inputs: DataFrame with DatetimeIndex
            
        Returns:
            Timestep duration in days
        """
        if hasattr(inputs.index, 'freq') and inputs.index.freq is not None:
            # Use frequency attribute if available
            freq = inputs.index.freq
            # Convert to timedelta using pd.Timedelta (avoids deprecation warning)
            try:
                td = pd.Timedelta(freq)
                return td.days + td.seconds / 86400
            except (ValueError, TypeError):
                # Fallback if conversion fails
                if hasattr(freq, 'n'):
                    # Assume daily if freq.n exists
                    return float(freq.n)
        
        # Infer from first two timestamps
        if len(inputs) >= 2:
            delta = inputs.index[1] - inputs.index[0]
            return delta.days + delta.seconds / 86400
        
        # Default to daily
        return 1.0
    
    def run_timestep(self, precipitation: float, pet: float) -> Dict[str, float]:
        """
        Run model for a single timestep.
        
        Note: Routing is NOT applied in timestep mode, as routing
        requires the full hydrograph. Use run() for routed results.
        
        Args:
            precipitation: Precipitation for this timestep [mm]
            pet: Potential evapotranspiration for this timestep [mm]
            
        Returns:
            Dictionary with model outputs (unrouted)
        """
        return self._rr_model.run_timestep(precipitation, pet)
    
    def reset(self) -> None:
        """Reset both RR model and router state."""
        self._rr_model.reset()
        if self._router is not None:
            self._router.reset()
    
    def get_state(self) -> ModelState:
        """
        Get combined state from RR model and router.
        
        Returns:
            ModelState containing state from both components
        """
        rr_state = self._rr_model.get_state()
        
        state_values = rr_state.values.copy()
        state_values['_is_routed_model'] = True
        
        if self._router is not None:
            router_state = self._router.get_state()
            state_values['_router_state'] = router_state
        
        return ModelState(
            values=state_values,
            timestamp=rr_state.timestamp
        )
    
    def set_state(self, state: ModelState) -> None:
        """
        Restore combined state for RR model and router.
        
        Args:
            state: ModelState from previous get_state() call
        """
        # Restore RR model state
        rr_values = {
            k: v for k, v in state.values.items()
            if not k.startswith('_')
        }
        rr_state = ModelState(values=rr_values, timestamp=state.timestamp)
        self._rr_model.set_state(rr_state)
        
        # Restore router state if present
        if self._router is not None and '_router_state' in state.values:
            self._router.set_state(state.values['_router_state'])
    
    # =========================================================================
    # Routing Control Methods
    # =========================================================================
    
    def enable_routing(self) -> None:
        """
        Enable channel routing.
        
        Has no effect if no router is configured.
        """
        if self._router is not None:
            self._routing_enabled = True
    
    def disable_routing(self) -> None:
        """
        Disable channel routing (pass-through mode).
        
        When disabled, run() returns the direct RR model output
        without applying routing.
        """
        self._routing_enabled = False
    
    @property
    def is_routing_enabled(self) -> bool:
        """Check if routing is currently enabled."""
        return self._routing_enabled and self._router is not None
    
    @property
    def router(self) -> Optional[BaseRouter]:
        """Get the router instance (or None if not configured)."""
        return self._router
    
    @property
    def rr_model(self) -> BaseRainfallRunoffModel:
        """Get the underlying rainfall-runoff model."""
        return self._rr_model
    
    def set_router(self, router: Optional[BaseRouter]) -> None:
        """
        Set or replace the router.
        
        Args:
            router: New router instance, or None to remove routing
        """
        self._router = router
        if router is None:
            self._routing_enabled = False
        else:
            self._routing_enabled = True
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values for both RR model and router."""
        defaults = self._rr_model.get_default_parameters()
        
        if self._router is not None:
            # Add routing defaults from parameter definitions
            for param in self.parameter_definitions:
                if param.name.startswith('routing_'):
                    defaults[param.name] = param.default
        
        return defaults
    
    def validate_parameters(self, params: Dict[str, float]) -> List[str]:
        """
        Validate parameter values against bounds.
        
        Args:
            params: Dictionary of parameter values to validate
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Separate and validate RR vs routing params
        rr_params = {k: v for k, v in params.items() if not k.startswith('routing_')}
        routing_params = {k: v for k, v in params.items() if k.startswith('routing_')}
        
        # Validate RR params
        errors.extend(self._rr_model.validate_parameters(rr_params))
        
        # Validate routing params
        if routing_params and self._router is not None:
            routing_bounds = self._router.get_parameter_bounds()
            for name, value in routing_params.items():
                if name in routing_bounds:
                    min_val, max_val = routing_bounds[name]
                    if not (min_val <= value <= max_val):
                        errors.append(
                            f"Parameter '{name}' value {value} is outside bounds "
                            f"[{min_val}, {max_val}]"
                        )
        
        return errors
    
    def summary(self) -> str:
        """Return summary of the routed model configuration."""
        lines = [
            "RoutedModel Configuration",
            "=" * 50,
            f"RR Model: {self._rr_model.name}",
            f"Routing enabled: {self.is_routing_enabled}",
        ]
        
        if self._router is not None:
            router_params = self._router.get_parameters()
            lines.append("")
            lines.append("Routing Parameters:")
            for name, value in router_params.items():
                lines.append(f"  {name}: {value}")
        else:
            lines.append("Router: None (no routing)")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        router_str = repr(self._router) if self._router else "None"
        return (
            f"RoutedModel(rr_model={self._rr_model.name}, "
            f"router={router_str}, "
            f"routing_enabled={self._routing_enabled})"
        )
