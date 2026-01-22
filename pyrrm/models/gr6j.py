"""
GR6J Rainfall-Runoff Model - Python Implementation

A 6-parameter daily lumped rainfall-runoff model developed by IRSTEA/INRAE.
GR6J extends GR5J with an additional exponential store for improved low-flow
simulation.

This is a pure Python port of the Rust implementation from the hydrogr library.

Reference:
    Pushpalatha, R., Perrin, C., Le Moine, N., & Andréassian, V. (2011).
    A downward structural sensitivity analysis of hydrological models to 
    improve low-flow simulation. Journal of Hydrology, 411(1-2), 66-76.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings

from pyrrm.models.base import BaseRainfallRunoffModel, ModelParameter, ModelState
from pyrrm.models.utils.s_curves import s_curve1, s_curve2


def _gr6j_core(
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    x5: float,
    x6: float,
    precipitation: np.ndarray,
    evapotranspiration: np.ndarray,
    production_store: float,
    routing_store: float,
    exponential_store: float,
    uh1_stores: np.ndarray,
    uh2_stores: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    """
    Core GR6J model algorithm.
    
    Args:
        x1: Production store capacity [mm]
        x2: Groundwater exchange coefficient [mm/d]
        x3: Routing store capacity [mm]
        x4: Unit hydrograph time constant [d]
        x5: Groundwater exchange threshold [-]
        x6: Exponential store capacity [mm]
        precipitation: Precipitation time series [mm]
        evapotranspiration: PET time series [mm]
        production_store: Initial production store level [mm]
        routing_store: Initial routing store level [mm]
        exponential_store: Initial exponential store level [mm]
        uh1_stores: Initial UH1 convolution stores
        uh2_stores: Initial UH2 convolution stores
        
    Returns:
        Tuple of (flow, final_production_store, final_routing_store, 
                  final_exponential_store, final_uh1_stores, final_uh2_stores)
    """
    n_timesteps = len(precipitation)
    flow = np.zeros(n_timesteps)
    
    storage_fraction = 0.9
    exp_fraction = 0.4
    exp = 2.5
    
    # Initialize unit hydrographs
    n_uh1 = max(1, int(np.ceil(x4)))
    n_uh2 = max(1, int(np.ceil(2.0 * x4)))
    
    # Compute UH ordinates
    o_uh1 = np.zeros(n_uh1)
    o_uh2 = np.zeros(n_uh2)
    
    for i in range(1, n_uh1 + 1):
        o_uh1[i - 1] = s_curve1(i, x4, exp) - s_curve1(i - 1, x4, exp)
    for i in range(1, n_uh2 + 1):
        o_uh2[i - 1] = s_curve2(i, x4, exp) - s_curve2(i - 1, x4, exp)
    
    # Ensure UH stores are correct size
    if len(uh1_stores) < n_uh1:
        uh1 = np.zeros(n_uh1)
        uh1[:len(uh1_stores)] = uh1_stores
    else:
        uh1 = uh1_stores[:n_uh1].copy()
        
    if len(uh2_stores) < n_uh2:
        uh2 = np.zeros(n_uh2)
        uh2[:len(uh2_stores)] = uh2_stores
    else:
        uh2 = uh2_stores[:n_uh2].copy()
    
    # State variables
    prod_store = production_store
    rout_store = routing_store
    exp_store = exponential_store
    
    # Main loop
    for t in range(n_timesteps):
        rain = precipitation[t]
        evap = evapotranspiration[t]
        
        rout_input = 0.0
        psf = prod_store / x1 if x1 > 0 else 0.0  # Production store filling ratio
        
        if rain <= evap:
            # Net evaporation case
            scaled_net_rain = (evap - rain) / x1 if x1 > 0 else 0.0
            scaled_net_rain = min(scaled_net_rain, 13.0)
            scaled_net_rain = np.tanh(scaled_net_rain)
            
            # Evaporation from production store
            denom = 1.0 + (1.0 - psf) * scaled_net_rain
            if denom > 0:
                prod_evap = prod_store * (2.0 - psf) * scaled_net_rain / denom
            else:
                prod_evap = 0.0
            
            prod_store -= prod_evap
        else:
            # Net precipitation case
            net_rainfall = rain - evap
            scaled_net_rain = net_rainfall / x1 if x1 > 0 else 0.0
            scaled_net_rain = min(scaled_net_rain, 13.0)
            scaled_net_rain = np.tanh(scaled_net_rain)
            
            # Rainfall to production store
            denom = 1.0 + psf * scaled_net_rain
            if denom > 0:
                prod_rainfall = x1 * (1.0 - psf * psf) * scaled_net_rain / denom
            else:
                prod_rainfall = 0.0
            
            rout_input = net_rainfall - prod_rainfall
            prod_store += prod_rainfall
        
        # Ensure non-negative
        if prod_store < 0.0:
            prod_store = 0.0
        
        # Production store percolation
        psf_p4 = (prod_store / x1) ** 4.0 if x1 > 0 else 0.0
        percolation = prod_store * (1.0 - 1.0 / (1.0 + psf_p4 / 25.62890625) ** 0.25)
        
        prod_store -= percolation
        rout_input += percolation
        
        # UH1 convolution
        for i in range(n_uh1 - 1):
            uh1[i] = uh1[i + 1] + o_uh1[i] * rout_input
        uh1[n_uh1 - 1] = o_uh1[n_uh1 - 1] * rout_input
        
        # UH2 convolution
        for i in range(n_uh2 - 1):
            uh2[i] = uh2[i + 1] + o_uh2[i] * rout_input
        uh2[n_uh2 - 1] = o_uh2[n_uh2 - 1] * rout_input
        
        # Groundwater exchange (with threshold X5)
        groundwater_exchange = x2 * (rout_store / x3 - x5) if x3 > 0 else 0.0
        
        # Flow to routing store (only fraction not going to exponential store)
        rout_store += uh1[0] * storage_fraction * (1.0 - exp_fraction) + groundwater_exchange
        
        if rout_store < 0.0:
            rout_store = 0.0
        
        # Routing store outflow
        rsf_p4 = (rout_store / x3) ** 4.0 if x3 > 0 else 0.0
        rout_flow = rout_store * (1.0 - 1.0 / (1.0 + rsf_p4) ** 0.25)
        rout_store -= rout_flow
        
        # Exponential store
        exp_store += uh1[0] * storage_fraction * exp_fraction + groundwater_exchange
        
        # Limit ar to prevent overflow
        ar = exp_store / x6 if x6 > 0 else 0.0
        ar = np.clip(ar, -33.0, 33.0)
        
        # Exponential store outflow
        if ar > 7.0:
            exp_flow = exp_store + x6 / np.exp(ar)
        elif ar < -7.0:
            exp_flow = x6 * np.exp(ar)
        else:
            exp_flow = x6 * np.log(np.exp(ar) + 1.0)
        
        exp_store -= exp_flow
        
        # Direct flow
        direct_flow = uh2[0] * (1.0 - storage_fraction) + groundwater_exchange
        if direct_flow < 0.0:
            direct_flow = 0.0
        
        flow[t] = rout_flow + direct_flow + exp_flow
    
    return flow, prod_store, rout_store, exp_store, uh1, uh2


class GR6J(BaseRainfallRunoffModel):
    """
    GR6J (Génie Rural à 6 paramètres Journalier) rainfall-runoff model.
    
    A 6-parameter daily lumped conceptual rainfall-runoff model.
    GR6J extends GR5J with an additional exponential store for improved
    simulation of low flows.
    
    Parameters:
        X1: Production store capacity [mm] (typical range: 100-1200)
        X2: Groundwater exchange coefficient [mm/d] (typical range: -5 to 3)
        X3: Routing store capacity [mm] (typical range: 20-300)
        X4: Unit hydrograph time base [d] (typical range: 1.1-2.9)
        X5: Groundwater exchange threshold [-] (typical range: -4 to 4)
        X6: Exponential store capacity [mm] (typical range: 1-300)
    
    Example:
        >>> model = GR6J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7, 'X5': 0.1, 'X6': 50})
        >>> results = model.run(input_data)
    """
    
    name = "gr6j"
    description = "GR6J 6-parameter daily rainfall-runoff model (INRAE)"
    frequency = 'D'
    
    _MAX_UH1_SIZE = 20
    _MAX_UH2_SIZE = 40
    
    def __init__(
        self, 
        parameters: Optional[Dict[str, float]] = None,
        catchment_area_km2: Optional[float] = None
    ):
        """
        Initialize GR6J model.
        
        Args:
            parameters: Dictionary with keys 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'
                       If None, uses default parameters.
            catchment_area_km2: Optional catchment area in km². When set,
                               outputs are scaled from mm to ML/day.
        """
        # Initialize catchment area for unit conversion (from base class)
        self._catchment_area_km2: Optional[float] = None
        if catchment_area_km2 is not None:
            self.set_catchment_area(catchment_area_km2)
        
        # Initialize state
        self._production_store = 0.3
        self._routing_store = 0.5
        self._exponential_store = 0.0  # In mm (not fraction)
        self._uh1 = np.zeros(self._MAX_UH1_SIZE)
        self._uh2 = np.zeros(self._MAX_UH2_SIZE)
        
        # Parameters
        self._params: Dict[str, float] = {}
        
        if parameters is not None:
            self.set_parameters(parameters)
        else:
            self.set_parameters(self.get_default_parameters())
    
    @property
    def parameter_definitions(self) -> List[ModelParameter]:
        """Return parameter definitions for GR6J."""
        return [
            ModelParameter(
                name='X1',
                default=350.0,
                min_bound=100.0,
                max_bound=1200.0,
                description='Production store capacity',
                unit='mm'
            ),
            ModelParameter(
                name='X2',
                default=0.0,
                min_bound=-5.0,
                max_bound=3.0,
                description='Groundwater exchange coefficient',
                unit='mm/d'
            ),
            ModelParameter(
                name='X3',
                default=90.0,
                min_bound=20.0,
                max_bound=300.0,
                description='Routing store capacity',
                unit='mm'
            ),
            ModelParameter(
                name='X4',
                default=1.7,
                min_bound=0.5,
                max_bound=10.0,
                description='Unit hydrograph time base',
                unit='d'
            ),
            ModelParameter(
                name='X5',
                default=0.0,
                min_bound=-4.0,
                max_bound=4.0,
                description='Groundwater exchange threshold',
                unit='-'
            ),
            ModelParameter(
                name='X6',
                default=50.0,
                min_bound=1.0,
                max_bound=300.0,
                description='Exponential store capacity',
                unit='mm'
            ),
        ]
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Set model parameters with validation."""
        errors = self.validate_parameters(params)
        if errors:
            for err in errors:
                warnings.warn(err)
        
        # Apply thresholds
        x1 = max(params.get('X1', 350.0), 0.01)
        x2 = params.get('X2', 0.0)
        x3 = max(params.get('X3', 90.0), 0.01)
        x4 = max(params.get('X4', 1.7), 0.5)
        x5 = params.get('X5', 0.0)
        x6 = max(params.get('X6', 50.0), 0.01)
        
        self._params = {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5, 'X6': x6}
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return dict(self._params)
    
    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Run model for entire input time series.
        
        Args:
            inputs: DataFrame with DatetimeIndex containing:
                - 'precipitation' (or 'rainfall'): Daily precipitation [mm]
                - 'pet' (or 'evapotranspiration'): Daily potential ET [mm]
                
        Returns:
            DataFrame with 'flow' column.
            Units are mm if catchment_area_km2 is not set, ML/day if it is set.
        """
        # Extract input columns
        if 'precipitation' in inputs.columns:
            precip = inputs['precipitation'].values.astype(float)
        elif 'rainfall' in inputs.columns:
            precip = inputs['rainfall'].values.astype(float)
        else:
            raise ValueError("Input must contain 'precipitation' or 'rainfall' column")
        
        if 'evapotranspiration' in inputs.columns:
            pet = inputs['evapotranspiration'].values.astype(float)
        elif 'pet' in inputs.columns:
            pet = inputs['pet'].values.astype(float)
        else:
            raise ValueError("Input must contain 'evapotranspiration' or 'pet' column")
        
        # Get initial states in mm
        prod_store_mm = self._production_store * self._params['X1']
        rout_store_mm = self._routing_store * self._params['X3']
        
        # Run core algorithm
        flow, prod_final, rout_final, exp_final, uh1_final, uh2_final = _gr6j_core(
            x1=self._params['X1'],
            x2=self._params['X2'],
            x3=self._params['X3'],
            x4=self._params['X4'],
            x5=self._params['X5'],
            x6=self._params['X6'],
            precipitation=precip,
            evapotranspiration=pet,
            production_store=prod_store_mm,
            routing_store=rout_store_mm,
            exponential_store=self._exponential_store,
            uh1_stores=self._uh1,
            uh2_stores=self._uh2,
        )
        
        # Update states
        self._production_store = prod_final / self._params['X1'] if self._params['X1'] > 0 else 0
        self._routing_store = rout_final / self._params['X3'] if self._params['X3'] > 0 else 0
        self._exponential_store = exp_final
        
        self._uh1 = np.zeros(self._MAX_UH1_SIZE)
        self._uh1[:len(uh1_final)] = uh1_final
        self._uh2 = np.zeros(self._MAX_UH2_SIZE)
        self._uh2[:len(uh2_final)] = uh2_final
        
        # Apply catchment area scaling if set
        flow = self._scale_array_to_volume(flow)
        
        return pd.DataFrame({'flow': flow}, index=inputs.index)
    
    def run_timestep(self, precipitation: float, pet: float) -> Dict[str, float]:
        """
        Run model for a single timestep.
        
        Returns:
            Dictionary with 'runoff' (same as 'flow').
            Units are mm if catchment_area_km2 is not set, ML/day if it is set.
        """
        precip_arr = np.array([precipitation])
        pet_arr = np.array([pet])
        
        prod_store_mm = self._production_store * self._params['X1']
        rout_store_mm = self._routing_store * self._params['X3']
        
        flow, prod_final, rout_final, exp_final, uh1_final, uh2_final = _gr6j_core(
            x1=self._params['X1'],
            x2=self._params['X2'],
            x3=self._params['X3'],
            x4=self._params['X4'],
            x5=self._params['X5'],
            x6=self._params['X6'],
            precipitation=precip_arr,
            evapotranspiration=pet_arr,
            production_store=prod_store_mm,
            routing_store=rout_store_mm,
            exponential_store=self._exponential_store,
            uh1_stores=self._uh1,
            uh2_stores=self._uh2,
        )
        
        self._production_store = prod_final / self._params['X1'] if self._params['X1'] > 0 else 0
        self._routing_store = rout_final / self._params['X3'] if self._params['X3'] > 0 else 0
        self._exponential_store = exp_final
        self._uh1 = np.zeros(self._MAX_UH1_SIZE)
        self._uh1[:len(uh1_final)] = uh1_final
        self._uh2 = np.zeros(self._MAX_UH2_SIZE)
        self._uh2[:len(uh2_final)] = uh2_final
        
        # Apply catchment area scaling if set
        scaled_flow = self._scale_to_volume(flow[0])
        return {'runoff': scaled_flow, 'flow': scaled_flow}
    
    def reset(self) -> None:
        """Reset model to initial state."""
        self._production_store = 0.3
        self._routing_store = 0.5
        self._exponential_store = 0.0
        self._uh1 = np.zeros(self._MAX_UH1_SIZE)
        self._uh2 = np.zeros(self._MAX_UH2_SIZE)
    
    def get_state(self) -> ModelState:
        """Get current model state."""
        return ModelState(
            values={
                'production_store': self._production_store,
                'routing_store': self._routing_store,
                'exponential_store': self._exponential_store,
                'uh1': self._uh1.tolist(),
                'uh2': self._uh2.tolist(),
            }
        )
    
    def set_state(self, state: ModelState) -> None:
        """Set model state."""
        if 'production_store' in state.values:
            self._production_store = float(state.values['production_store'])
        if 'routing_store' in state.values:
            self._routing_store = float(state.values['routing_store'])
        if 'exponential_store' in state.values:
            self._exponential_store = float(state.values['exponential_store'])
        if 'uh1' in state.values:
            uh1 = np.array(state.values['uh1'])
            self._uh1 = np.zeros(self._MAX_UH1_SIZE)
            self._uh1[:min(len(uh1), self._MAX_UH1_SIZE)] = uh1[:self._MAX_UH1_SIZE]
        if 'uh2' in state.values:
            uh2 = np.array(state.values['uh2'])
            self._uh2 = np.zeros(self._MAX_UH2_SIZE)
            self._uh2[:min(len(uh2), self._MAX_UH2_SIZE)] = uh2[:self._MAX_UH2_SIZE]
    
    def set_initial_states(
        self,
        production_store: Optional[float] = None,
        routing_store: Optional[float] = None,
        exponential_store: Optional[float] = None
    ) -> None:
        """
        Set initial store levels.
        
        Args:
            production_store: Initial production store level (0-1, fraction of X1)
            routing_store: Initial routing store level (0-1, fraction of X3)
            exponential_store: Initial exponential store level [mm]
        """
        if production_store is not None:
            self._production_store = np.clip(production_store, 0.0, 1.0)
        if routing_store is not None:
            self._routing_store = np.clip(routing_store, 0.0, 1.0)
        if exponential_store is not None:
            self._exponential_store = exponential_store
