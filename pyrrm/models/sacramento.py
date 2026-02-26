"""
Sacramento Rainfall-Runoff Model - pyrrm Integration

This module provides the Sacramento model adapted to the pyrrm BaseRainfallRunoffModel
interface. It wraps the original Sacramento implementation to provide a consistent
API with other pyrrm models.

The Sacramento model is a conceptual rainfall-runoff model originally developed by
the US National Weather Service. This implementation is ported from the C# version
used in IQQM by NSW Department of Natural Resources.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import warnings
import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from pyrrm.models.base import BaseRainfallRunoffModel, ModelParameter, ModelState

try:
    from pyrrm.models.numba_kernels import (
        NUMBA_AVAILABLE,
        _sacramento_run_numba,
        _sacramento_run_numba_fast,
    )
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

LENGTH_OF_UNIT_HYDROGRAPH: int = 5
PDN20: float = 5.08
PDNOR: float = 25.4
TOLERANCE: float = 1e-10


class InvalidParameterSetException(Exception):
    """Raised when model parameters produce invalid results."""
    pass


# =============================================================================
# UnitHydrograph Class (internal)
# =============================================================================

@dataclass
class _UnitHydrograph:
    """Internal unit hydrograph for Sacramento model."""
    s_curve: List[float] = field(default_factory=list)
    _stores: List[float] = field(default_factory=list)
    
    def initialise_hydrograph(self, proportions: List[float]) -> None:
        self.s_curve = list(proportions)
        self._stores = [0.0] * len(proportions)
    
    def run_time_step(self, input_value: float) -> float:
        if not self.s_curve:
            return input_value
        for i in range(len(self._stores)):
            self._stores[i] += input_value * self.s_curve[i]
        output = self._stores[0]
        for i in range(len(self._stores) - 1):
            self._stores[i] = self._stores[i + 1]
        self._stores[-1] = 0.0
        return output
    
    def reset(self) -> None:
        self._stores = [0.0] * len(self.s_curve) if self.s_curve else []
    
    def proportion_for_item(self, i: int) -> float:
        if 0 <= i < len(self.s_curve):
            return self.s_curve[i]
        return 0.0


# =============================================================================
# Sacramento Model Class
# =============================================================================

class Sacramento(BaseRainfallRunoffModel):
    """
    Sacramento Rainfall-Runoff Model.
    
    A conceptual rainfall-runoff model with multiple soil moisture accounting
    zones (upper zone tension/free water, lower zone tension/free water).
    
    Parameters:
        uztwm: Upper zone tension water maximum storage [mm]
        uzfwm: Upper zone free water maximum storage [mm]
        lztwm: Lower zone tension water maximum storage [mm]
        lzfpm: Lower zone primary free water maximum storage [mm]
        lzfsm: Lower zone supplemental free water maximum storage [mm]
        uzk: Upper zone lateral drainage rate [1/day]
        lzpk: Lower zone primary drainage rate [1/day]
        lzsk: Lower zone supplemental drainage rate [1/day]
        zperc: Percolation demand scale parameter
        rexp: Percolation equation exponent
        pctim: Permanent impervious area fraction
        adimp: Additional impervious area fraction
        pfree: Fraction of percolation going to free water
        rserv: Fraction of lower zone unavailable for transpiration
        side: Side flow ratio
        ssout: Subsurface outflow [mm]
        sarva: Riparian vegetation area fraction
        uh1-uh5: Unit hydrograph components
    
    Example:
        >>> model = Sacramento()
        >>> model.set_parameters({'uztwm': 50, 'lztwm': 130, 'uzk': 0.3})
        >>> results = model.run(input_data)
    """
    
    name = "sacramento"
    description = "Sacramento conceptual rainfall-runoff model (NSW IQQM version)"
    frequency = 'D'
    
    VERSION: int = 1
    
    def __init__(
        self, 
        parameters: Optional[Dict[str, float]] = None,
        catchment_area_km2: Optional[float] = None,
        numba_fastmath: bool = True,
    ):
        """
        Initialize Sacramento model.
        
        Args:
            parameters: Optional dictionary of parameter values.
                       If None, uses default parameters.
            catchment_area_km2: Optional catchment area in km². When set,
                               outputs are scaled from mm to ML/day.
            numba_fastmath: Use the fastmath Numba kernel (default True).
                Enables LLVM floating-point reassociation and FMA for
                ~10-16 % faster single-run execution.  Results agree
                with the strict kernel to within rtol=1e-6.  Set False
                to recover bit-exact IEEE-754 behaviour.
        """
        self._numba_fastmath = numba_fastmath

        # Unit hydrograph components
        self._unscaled_unit_hydrograph: List[float] = [0.0] * LENGTH_OF_UNIT_HYDROGRAPH
        self._unit_hydrograph: _UnitHydrograph = _UnitHydrograph()
        
        # Initialize catchment area for unit conversion (from base class)
        self._catchment_area_km2: Optional[float] = None
        if catchment_area_km2 is not None:
            self.set_catchment_area(catchment_area_km2)
        
        # Initialize all internal attributes
        self._init_all_attributes()
        
        # Set default parameters
        self._init_default_parameters()
        
        # Update derived states
        self._update_internal_states()
        
        # Apply user parameters if provided
        if parameters is not None:
            self.set_parameters(parameters)
        
        self.snapshot_entity_id: UUID = uuid4()
    
    @property
    def parameter_definitions(self) -> List[ModelParameter]:
        """Return parameter definitions for Sacramento model."""
        return [
            ModelParameter('uztwm', 50.0, 25.0, 125.0, 'Upper zone tension water max storage', 'mm'),
            ModelParameter('uzfwm', 40.0, 10.0, 75.0, 'Upper zone free water max storage', 'mm'),
            ModelParameter('lztwm', 130.0, 75.0, 300.0, 'Lower zone tension water max storage', 'mm'),
            ModelParameter('lzfpm', 60.0, 40.0, 600.0, 'Lower zone primary free water max storage', 'mm'),
            ModelParameter('lzfsm', 25.0, 15.0, 300.0, 'Lower zone supplemental free water max storage', 'mm'),
            ModelParameter('uzk', 0.3, 0.2, 0.5, 'Upper zone lateral drainage rate', '1/d'),
            ModelParameter('lzpk', 0.01, 0.001, 0.015, 'Lower zone primary drainage rate', '1/d'),
            ModelParameter('lzsk', 0.05, 0.03, 0.2, 'Lower zone supplemental drainage rate', '1/d'),
            ModelParameter('zperc', 40.0, 20.0, 300.0, 'Percolation demand scale parameter', '-'),
            ModelParameter('rexp', 1.0, 1.4, 3.5, 'Percolation equation exponent', '-'),
            ModelParameter('pctim', 0.01, 0.0, 0.05, 'Permanent impervious area fraction', '-'),
            ModelParameter('adimp', 0.0, 0.0, 0.2, 'Additional impervious area fraction', '-'),
            ModelParameter('pfree', 0.06, 0.0, 0.5, 'Fraction of percolation to free water', '-'),
            ModelParameter('rserv', 0.3, 0.0, 0.4, 'Lower zone unavailable for transpiration', '-'),
            ModelParameter('side', 0.0, 0.0, 0.8, 'Side flow ratio', '-'),
            ModelParameter('ssout', 0.0, 0.0, 0.1, 'Subsurface outflow', 'mm'),
            ModelParameter('sarva', 0.0, 0.0, 0.1, 'Riparian vegetation area fraction', '-'),
            ModelParameter('uh1', 1.0, 0.0, 1.0, 'Unit hydrograph component 1', '-'),
            ModelParameter('uh2', 0.0, 0.0, 1.0, 'Unit hydrograph component 2', '-'),
            ModelParameter('uh3', 0.0, 0.0, 1.0, 'Unit hydrograph component 3', '-'),
            ModelParameter('uh4', 0.0, 0.0, 1.0, 'Unit hydrograph component 4', '-'),
            ModelParameter('uh5', 0.0, 0.0, 1.0, 'Unit hydrograph component 5', '-'),
        ]
    
    def _init_all_attributes(self) -> None:
        """Initialize all model attributes to default values."""
        # Inputs
        self.pet: float = 0.0
        self.rainfall: float = 0.0
        self._pliq: float = 0.0
        
        # Outputs
        self.runoff: float = 0.0
        self.baseflow: float = 0.0
        
        # Parameters
        self.uztwm: float = 0.0
        self.uzfwm: float = 0.0
        self.lztwm: float = 0.0
        self.lzfpm: float = 0.0
        self.lzfsm: float = 0.0
        self.rserv: float = 0.0
        self.adimp: float = 0.0
        self.uzk: float = 0.0
        self.lzpk: float = 0.0
        self.lzsk: float = 0.0
        self.zperc: float = 0.0
        self.rexp: float = 0.0
        self.pctim: float = 0.0
        self.pfree: float = 0.0
        self.side: float = 0.0
        self.ssout: float = 0.0
        self.sarva: float = 0.0
        
        # State variables
        self.uztwc: float = 0.0
        self.uzfwc: float = 0.0
        self.lztwc: float = 0.0
        self.lzfsc: float = 0.0
        self.lzfpc: float = 0.0
        
        # Adjusted lower zone variables
        self.alzfsc: float = 0.0
        self.alzfpc: float = 0.0
        self.alzfsm: float = 0.0
        self.alzfpm: float = 0.0
        
        # Additional state variables
        self.adimc: float = 0.0
        self.flobf: float = 0.0
        self.flosf: float = 0.0
        self.floin: float = 0.0
        self.flwbf: float = 0.0
        self.flwsf: float = 0.0
        self.roimp: float = 0.0
        
        # Evaporation components
        self.evap_uztw: float = 0.0
        self.evap_uzfw: float = 0.0
        self.e3: float = 0.0
        self.e5: float = 0.0
        self.evaporation_channel_water: float = 0.0
        
        # Other state variables
        self.perc: float = 0.0
        self.pbase: float = 0.0
        self.channel_flow: float = 0.0
        self.reserved_lower_zone: float = 0.0
        self.sum_lower_zone_capacities: float = 0.0
        self.hydrograph_store: float = 0.0
        
        # Previous state variables for mass balance
        self._prev_uztwc: float = 0.0
        self._prev_uzfwc: float = 0.0
        self._prev_lztwc: float = 0.0
        self._prev_hydrograph_store: float = 0.0
        self._prev_lzfsc: float = 0.0
        self._prev_lzfpc: float = 0.0
    
    def _init_default_parameters(self) -> None:
        """Initialize model parameters to default values."""
        for i in range(len(self._unscaled_unit_hydrograph)):
            self._unscaled_unit_hydrograph[i] = 0.0
        self._unscaled_unit_hydrograph[0] = 1.0
        self._set_unit_hydrograph_components()
        
        self.uztwm = 50.0
        self.uzfwm = 40.0
        self.lztwm = 130.0
        self.lzfpm = 60.0
        self.lzfsm = 25.0
        self.rserv = 0.3
        self.adimp = 0.0
        self.uzk = 0.3
        self.lzpk = 0.01
        self.lzsk = 0.05
        self.zperc = 40.0
        self.rexp = 1.0
        self.pctim = 0.01
        self.pfree = 0.06
        self.side = 0.0
        self.ssout = 0.0
        self.sarva = 0.0
    
    # =========================================================================
    # Unit Hydrograph Properties
    # =========================================================================
    
    @property
    def uh1(self) -> float:
        return self._unscaled_unit_hydrograph[0]
    
    @uh1.setter
    def uh1(self, value: float) -> None:
        self._unscaled_unit_hydrograph[0] = max(0.0, value)
    
    @property
    def uh2(self) -> float:
        return self._unscaled_unit_hydrograph[1]
    
    @uh2.setter
    def uh2(self, value: float) -> None:
        self._unscaled_unit_hydrograph[1] = max(0.0, value)
    
    @property
    def uh3(self) -> float:
        return self._unscaled_unit_hydrograph[2]
    
    @uh3.setter
    def uh3(self, value: float) -> None:
        self._unscaled_unit_hydrograph[2] = max(0.0, value)
    
    @property
    def uh4(self) -> float:
        return self._unscaled_unit_hydrograph[3]
    
    @uh4.setter
    def uh4(self, value: float) -> None:
        self._unscaled_unit_hydrograph[3] = max(0.0, value)
    
    @property
    def uh5(self) -> float:
        return self._unscaled_unit_hydrograph[4]
    
    @uh5.setter
    def uh5(self, value: float) -> None:
        self._unscaled_unit_hydrograph[4] = max(0.0, value)
    
    @property
    def mass_balance(self) -> float:
        """Calculate mass balance error for current time step."""
        aet = (self.evap_uztw + self.evap_uzfw + self.e3 + 
               self.evaporation_channel_water + self.e5)
        
        delta_s = ((self.uztwc - self._prev_uztwc) + 
                   (self.uzfwc - self._prev_uzfwc) + 
                   (self.lztwc - self._prev_lztwc) +
                   (self.lzfsc - self._prev_lzfsc) + 
                   (self.lzfpc - self._prev_lzfpc)) * (1.0 - self.pctim) + \
                  (self.hydrograph_store - self._prev_hydrograph_store)
        
        baseflow_loss = ((self.alzfsc - self.lzfsc) + 
                         (self.alzfpc - self.lzfpc) + 
                         (self.flobf - self.flwbf)) * (1.0 - self.pctim)
        
        return (self._pliq - aet - self.runoff - delta_s - 
                min(self.ssout, self.flwbf + self.flwsf) - baseflow_loss)
    
    # =========================================================================
    # BaseRainfallRunoffModel Interface Implementation
    # =========================================================================
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Set model parameters."""
        param_map = {
            'uztwm': 'uztwm', 'uzfwm': 'uzfwm', 'lztwm': 'lztwm',
            'lzfpm': 'lzfpm', 'lzfsm': 'lzfsm', 'rserv': 'rserv',
            'adimp': 'adimp', 'uzk': 'uzk', 'lzpk': 'lzpk',
            'lzsk': 'lzsk', 'zperc': 'zperc', 'rexp': 'rexp',
            'pctim': 'pctim', 'pfree': 'pfree', 'side': 'side',
            'ssout': 'ssout', 'sarva': 'sarva',
            'uh1': 'uh1', 'uh2': 'uh2', 'uh3': 'uh3', 'uh4': 'uh4', 'uh5': 'uh5'
        }
        
        for key, value in params.items():
            key_lower = key.lower()
            if key_lower in param_map:
                setattr(self, param_map[key_lower], value)
        
        self._set_unit_hydrograph_components()
        self._update_internal_states()
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {
            'uztwm': self.uztwm, 'uzfwm': self.uzfwm, 'lztwm': self.lztwm,
            'lzfpm': self.lzfpm, 'lzfsm': self.lzfsm, 'rserv': self.rserv,
            'adimp': self.adimp, 'uzk': self.uzk, 'lzpk': self.lzpk,
            'lzsk': self.lzsk, 'zperc': self.zperc, 'rexp': self.rexp,
            'pctim': self.pctim, 'pfree': self.pfree, 'side': self.side,
            'ssout': self.ssout, 'sarva': self.sarva,
            'uh1': self.uh1, 'uh2': self.uh2, 'uh3': self.uh3,
            'uh4': self.uh4, 'uh5': self.uh5,
        }
    
    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Run model for entire input time series.
        
        Args:
            inputs: DataFrame with 'precipitation'/'rainfall' and 'pet'/'evapotranspiration'
            
        Returns:
            DataFrame with 'runoff', 'baseflow', 'channel_flow' columns.
            Units are mm if catchment_area_km2 is not set, ML/day if it is set.
        """
        from pyrrm.data import resolve_column

        pcol = resolve_column(inputs, "precipitation", raise_on_missing=True)
        precip = inputs[pcol].values

        ecol = resolve_column(inputs, "pet", raise_on_missing=True)
        pet = inputs[ecol].values
        
        n_timesteps = len(precip)

        if NUMBA_AVAILABLE:
            uh_scurve = np.array(self._unit_hydrograph.s_curve, dtype=np.float64)
            uh_stores = np.array(self._unit_hydrograph._stores, dtype=np.float64)

            _kernel = (
                _sacramento_run_numba_fast
                if self._numba_fastmath
                else _sacramento_run_numba
            )

            (runoff_out, baseflow_out, channel_flow_out,
             uztwc_f, uzfwc_f, lztwc_f, lzfsc_f, lzfpc_f,
             alzfsc_f, alzfpc_f, adimc_f, hs_f,
             uh_stores_f) = _kernel(
                precip.astype(np.float64), pet.astype(np.float64),
                self.uztwm, self.uzfwm, self.lztwm, self.lzfpm, self.lzfsm,
                self.uzk, self.lzpk, self.lzsk,
                self.zperc, self.rexp, self.pctim, self.adimp,
                self.pfree, self.rserv, self.side, self.ssout, self.sarva,
                uh_scurve,
                self.uztwc, self.uzfwc, self.lztwc, self.lzfsc, self.lzfpc,
                self.adimc, self.hydrograph_store,
                uh_stores,
            )

            self.uztwc = uztwc_f
            self.uzfwc = uzfwc_f
            self.lztwc = lztwc_f
            self.lzfsc = lzfsc_f
            self.lzfpc = lzfpc_f
            self.alzfsc = alzfsc_f
            self.alzfpc = alzfpc_f
            self.adimc = adimc_f
            self.hydrograph_store = hs_f
            for i in range(len(uh_stores_f)):
                self._unit_hydrograph._stores[i] = uh_stores_f[i]
            self.runoff = runoff_out[-1] if n_timesteps > 0 else 0.0
            self.baseflow = baseflow_out[-1] if n_timesteps > 0 else 0.0
            self.channel_flow = channel_flow_out[-1] if n_timesteps > 0 else 0.0
        else:
            runoff_out = np.zeros(n_timesteps)
            baseflow_out = np.zeros(n_timesteps)
            channel_flow_out = np.zeros(n_timesteps)

            for t in range(n_timesteps):
                self.rainfall = float(precip[t])
                self.pet = float(pet[t])
                self._run_time_step()
                runoff_out[t] = self.runoff
                baseflow_out[t] = self.baseflow
                channel_flow_out[t] = self.channel_flow
        
        # Apply catchment area scaling if set
        runoff_out = self._scale_array_to_volume(runoff_out)
        baseflow_out = self._scale_array_to_volume(baseflow_out)
        channel_flow_out = self._scale_array_to_volume(channel_flow_out)
        
        return pd.DataFrame({
            'runoff': runoff_out,
            'flow': runoff_out,  # Alias
            'baseflow': baseflow_out,
            'channel_flow': channel_flow_out,
        }, index=inputs.index)
    
    def run_timestep(self, precipitation: float, pet: float) -> Dict[str, float]:
        """
        Run model for a single timestep.
        
        Args:
            precipitation: Precipitation for this timestep [mm]
            pet: Potential evapotranspiration for this timestep [mm]
            
        Returns:
            Dictionary with 'runoff', 'baseflow', 'channel_flow', 'mass_balance'.
            Flow units are mm if catchment_area_km2 is not set, ML/day if it is set.
        """
        self.rainfall = precipitation
        self.pet = pet
        self._run_time_step()
        return {
            'runoff': self._scale_to_volume(self.runoff),
            'flow': self._scale_to_volume(self.runoff),
            'baseflow': self._scale_to_volume(self.baseflow),
            'channel_flow': self._scale_to_volume(self.channel_flow),
            'mass_balance': self.mass_balance,  # Keep in mm (internal diagnostic)
        }
    
    def reset(self) -> None:
        """Reset model to initial state."""
        self._set_stores_and_fluxes_to_zero()
        self._update_internal_states()
        self._set_unit_hydrograph_components()
        self._set_prev_variables_for_mass_balance()
    
    def get_state(self) -> ModelState:
        """Get current model state."""
        return ModelState(
            values={
                'uztwc': self.uztwc,
                'uzfwc': self.uzfwc,
                'lztwc': self.lztwc,
                'lzfsc': self.lzfsc,
                'lzfpc': self.lzfpc,
                'adimc': self.adimc,
                'alzfpc': self.alzfpc,
                'alzfsc': self.alzfsc,
                'hydrograph_store': self.hydrograph_store,
            }
        )
    
    def set_state(self, state: ModelState) -> None:
        """Set model state."""
        if 'uztwc' in state.values:
            self.uztwc = state.values['uztwc']
        if 'uzfwc' in state.values:
            self.uzfwc = state.values['uzfwc']
        if 'lztwc' in state.values:
            self.lztwc = state.values['lztwc']
        if 'lzfsc' in state.values:
            self.lzfsc = state.values['lzfsc']
        if 'lzfpc' in state.values:
            self.lzfpc = state.values['lzfpc']
        if 'adimc' in state.values:
            self.adimc = state.values['adimc']
        if 'alzfpc' in state.values:
            self.alzfpc = state.values['alzfpc']
        if 'alzfsc' in state.values:
            self.alzfsc = state.values['alzfsc']
    
    def init_stores_full(self) -> None:
        """Initialize all water stores to their full capacity."""
        self.uzfwc = self.uzfwm
        self.uztwc = self.uztwm
        self.lztwc = self.lztwm
        self.lzfsc = self.lzfsm
        self.lzfpc = self.lzfpm
        self._update_internal_states()
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _set_stores_and_fluxes_to_zero(self) -> None:
        """Set all stores and fluxes to zero."""
        self._pliq = 0.0
        self.rainfall = 0.0
        self.pet = 0.0
        self.uzfwc = 0.0
        self.uztwc = 0.0
        self.lzfpc = 0.0
        self.lzfsc = 0.0
        self.lztwc = 0.0
        self.roimp = 0.0
        self.flobf = 0.0
        self.flosf = 0.0
        self.floin = 0.0
        self.flwbf = 0.0
        self.flwsf = 0.0
        self.evap_uztw = 0.0
        self.evap_uzfw = 0.0
        self.e5 = 0.0
        self.e3 = 0.0
        self.reserved_lower_zone = 0.0
        self._unit_hydrograph.reset()
        self.hydrograph_store = 0.0
        self.channel_flow = 0.0
        self.evaporation_channel_water = 0.0
        self.perc = 0.0
    
    def _set_unit_hydrograph_components(self) -> None:
        """Set the unit hydrograph components from unscaled values."""
        self._unit_hydrograph.initialise_hydrograph(
            self._normalise(self._unscaled_unit_hydrograph)
        )
    
    def _normalise(self, unscaled: List[float]) -> List[float]:
        """Normalise unit hydrograph components to sum to 1.0."""
        total = sum(unscaled)
        tmp = [0.0] * len(unscaled)
        
        if all(v == 0 for v in unscaled):
            tmp[0] = 1.0
            self._unit_hydrograph.initialise_hydrograph(tmp)
            return tmp
        
        if abs(total) < TOLERANCE:
            raise ValueError("Sum of unit hydrograph components is zero")
        
        for i in range(len(tmp)):
            tmp[i] = unscaled[i] / total
        
        return tmp
    
    def _update_internal_states(self) -> None:
        """Update internal derived state variables."""
        self.alzfsm = self.lzfsm * (1.0 + self.side)
        self.alzfpm = self.lzfpm * (1.0 + self.side)
        self.alzfsc = self.lzfsc * (1.0 + self.side)
        self.alzfpc = self.lzfpc * (1.0 + self.side)
        self.pbase = self.alzfsm * self.lzsk + self.alzfpm * self.lzpk
        self.adimc = self.uztwc + self.lztwc
        self.reserved_lower_zone = self.rserv * (self.lzfpm + self.lzfsm)
        self.sum_lower_zone_capacities = self.lztwm + self.lzfpm + self.lzfsm
    
    def _set_prev_variables_for_mass_balance(self) -> None:
        """Store current values for mass balance calculation."""
        self._prev_uztwc = self.uztwc
        self._prev_uzfwc = self.uzfwc
        self._prev_lztwc = self.lztwc
        self._prev_hydrograph_store = self.hydrograph_store
        self._prev_lzfsc = self.lzfsc
        self._prev_lzfpc = self.lzfpc
    
    def _do_unit_hydrograph_routing(self) -> None:
        """Perform routing of surface runoff via unit hydrograph."""
        self.flwsf = self._unit_hydrograph.run_time_step(
            self.flosf + self.roimp + self.floin
        )
        self.hydrograph_store += (self.flosf + self.roimp + self.floin - self.flwsf)
    
    def _run_time_step(self) -> None:
        """Execute one time step of the Sacramento model."""
        # Store current values for mass balance
        self._set_prev_variables_for_mass_balance()
        
        self.reserved_lower_zone = self.rserv * (self.lzfpm + self.lzfsm)
        evapt = self.pet
        self._pliq = self.rainfall
        
        # Evaporation from upper zone tension water
        if self.uztwm > 0.0:
            self.evap_uztw = evapt * self.uztwc / self.uztwm
        else:
            self.evap_uztw = 0.0
        
        # Evaporation from upper zone free water
        if self.uztwc < self.evap_uztw:
            self.evap_uztw = self.uztwc
            self.uztwc = 0.0
            self.evap_uzfw = min((evapt - self.evap_uztw), self.uzfwc)
            self.uzfwc = self.uzfwc - self.evap_uzfw
        else:
            self.uztwc = self.uztwc - self.evap_uztw
            self.evap_uzfw = 0.0
        
        # Transfer free water to tension water if needed
        if self.uztwm > 0.0:
            ratio_uztw = self.uztwc / self.uztwm
        else:
            ratio_uztw = 1.0
        
        if self.uzfwm > 0.0:
            ratio_uzfw = self.uzfwc / self.uzfwm
        else:
            ratio_uzfw = 1.0
        
        if ratio_uztw < ratio_uzfw:
            ratio_uztw = (self.uztwc + self.uzfwc) / (self.uztwm + self.uzfwm)
            self.uztwc = self.uztwm * ratio_uztw
            self.uzfwc = self.uzfwm * ratio_uztw
        
        # Evaporation from lower zone
        if self.uztwm + self.lztwm > 0.0:
            self.e3 = min(
                (evapt - self.evap_uztw - self.evap_uzfw) * self.lztwc / (self.uztwm + self.lztwm),
                self.lztwc
            )
            self.e5 = min(
                self.evap_uztw + ((evapt - self.evap_uztw - self.evap_uzfw) * 
                                  (self.adimc - self.evap_uztw - self.uztwc) / 
                                  (self.uztwm + self.lztwm)),
                self.adimc
            )
        else:
            self.e3 = 0.0
            self.e5 = 0.0
        
        self.lztwc = self.lztwc - self.e3
        self.adimc = self.adimc - self.e5
        self.evap_uztw = self.evap_uztw * (1 - self.adimp - self.pctim)
        self.evap_uzfw = self.evap_uzfw * (1 - self.adimp - self.pctim)
        self.e3 = self.e3 * (1 - self.adimp - self.pctim)
        self.e5 = self.e5 * self.adimp
        
        # Lower zone tension water resupply
        if self.lztwm > 0.0:
            ratio_lztw = self.lztwc / self.lztwm
        else:
            ratio_lztw = 1.0
        
        if self.alzfpm + self.alzfsm - self.reserved_lower_zone + self.lztwm > 0.0:
            ratio_lzfw = ((self.alzfpc + self.alzfsc - self.reserved_lower_zone + self.lztwc) /
                          (self.alzfpm + self.alzfsm - self.reserved_lower_zone + self.lztwm))
        else:
            ratio_lzfw = 1.0
        
        if ratio_lztw < ratio_lzfw:
            transfered = (ratio_lzfw - ratio_lztw) * self.lztwm
            self.lztwc = self.lztwc + transfered
            self.alzfsc = self.alzfsc - transfered
            if self.alzfsc < 0:
                self.alzfpc = self.alzfpc + self.alzfsc
                self.alzfsc = 0.0
        
        # Runoff from impervious area
        self.roimp = self._pliq * self.pctim
        
        # Upper zone processing
        pav = self._pliq + self.uztwc - self.uztwm
        if pav < 0:
            self.adimc = self.adimc + self._pliq
            self.uztwc = self.uztwc + self._pliq
            pav = 0.0
        else:
            self.adimc = self.adimc + self.uztwm - self.uztwc
            self.uztwc = self.uztwm
        
        # Determine number of increments
        if pav <= PDN20:
            adj = 1.0
            itime = 2
        else:
            if pav < PDNOR:
                adj = 0.5 * math.sqrt(pav / PDNOR)
            else:
                adj = 1.0 - 0.5 * PDNOR / pav
            itime = 1
        
        self.flobf = 0.0
        self.flosf = 0.0
        self.floin = 0.0
        
        hpl = self.alzfpm / (self.alzfpm + self.alzfsm) if (self.alzfpm + self.alzfsm) > 0 else 0.5
        
        for ii in range(itime, 3):
            ninc = int(math.floor((self.uzfwc * adj + pav) * 0.2)) + 1
            dinc = 1.0 / ninc
            pinc = pav * dinc
            dinc = dinc * adj
            
            if ninc == 1 and adj >= 1.0:
                duz = self.uzk
                dlzp = self.lzpk
                dlzs = self.lzsk
            else:
                duz = 1.0 - math.pow((1.0 - self.uzk), dinc) if self.uzk < 1.0 else 1.0
                dlzp = 1.0 - math.pow((1.0 - self.lzpk), dinc) if self.lzpk < 1.0 else 1.0
                dlzs = 1.0 - math.pow((1.0 - self.lzsk), dinc) if self.lzsk < 1.0 else 1.0
            
            for inc in range(1, ninc + 1):
                ratio = (self.adimc - self.uztwc) / self.lztwm if self.lztwm > 0 else 0
                addro = pinc * ratio * ratio
                
                # Baseflow from lower zone
                if self.alzfpc > 0.0:
                    bf = self.alzfpc * dlzp
                else:
                    self.alzfpc = 0.0
                    bf = 0.0
                
                self.flobf = self.flobf + bf
                self.alzfpc = self.alzfpc - bf
                
                if self.alzfsc > 0.0:
                    bf = self.alzfsc * dlzs
                else:
                    self.alzfsc = 0.0
                    bf = 0.0
                
                self.alzfsc = self.alzfsc - bf
                self.flobf = self.flobf + bf
                
                # Percolation and interflow
                if self.uzfwc > 0.0:
                    lzair = (self.lztwm - self.lztwc + self.alzfsm - self.alzfsc + 
                             self.alzfpm - self.alzfpc)
                    if lzair > 0.0:
                        self.perc = (self.pbase * dinc * self.uzfwc) / self.uzfwm if self.uzfwm > 0 else 0
                        total_lz = self.alzfpm + self.alzfsm + self.lztwm
                        current_lz = self.alzfpc + self.alzfsc + self.lztwc
                        if total_lz > 0:
                            deficit_ratio = 1.0 - current_lz / total_lz
                            self.perc = min(
                                self.uzfwc,
                                self.perc * (1.0 + (self.zperc * math.pow(deficit_ratio, self.rexp)))
                            )
                        self.perc = min(lzair, self.perc)
                        self.uzfwc = self.uzfwc - self.perc
                    else:
                        self.perc = 0.0
                    
                    # Interflow
                    transfered = duz * self.uzfwc
                    self.floin = self.floin + transfered
                    self.uzfwc = self.uzfwc - transfered
                    
                    # Distribute percolation
                    perctw = min(self.perc * (1.0 - self.pfree), self.lztwm - self.lztwc)
                    percfw = self.perc - perctw
                    
                    lzair = self.alzfsm - self.alzfsc + self.alzfpm - self.alzfpc
                    if percfw > lzair:
                        perctw = perctw + percfw - lzair
                        percfw = lzair
                    self.lztwc = self.lztwc + perctw
                    
                    if percfw > 0.0:
                        ratlp = 1.0 - self.alzfpc / self.alzfpm if self.alzfpm > 0 else 0
                        ratls = 1.0 - self.alzfsc / self.alzfsm if self.alzfsm > 0 else 0
                        if ratlp + ratls > 0:
                            percs = min(
                                self.alzfsm - self.alzfsc,
                                percfw * (1.0 - hpl * (2.0 * ratlp) / (ratlp + ratls))
                            )
                        else:
                            percs = 0
                        self.alzfsc = self.alzfsc + percs
                        if self.alzfsc > self.alzfsm:
                            percs = percs - self.alzfsc + self.alzfsm
                            self.alzfsc = self.alzfsm
                        self.alzfpc = self.alzfpc + percfw - percs
                        if self.alzfpc > self.alzfpm:
                            self.alzfsc = self.alzfsc + self.alzfpc - self.alzfpm
                            self.alzfpc = self.alzfpm
                
                # Fill upper zone free water
                if pinc > 0.0:
                    pav_local = pinc
                    if pav_local - self.uzfwm + self.uzfwc <= 0:
                        self.uzfwc = self.uzfwc + pav_local
                    else:
                        pav_local = pav_local - self.uzfwm + self.uzfwc
                        self.uzfwc = self.uzfwm
                        self.flosf = self.flosf + pav_local
                        addro = addro + pav_local * (1.0 - addro / pinc)
                
                self.adimc = self.adimc + pinc - addro
                self.roimp = self.roimp + addro * self.adimp
            
            adj = 1.0 - adj
            pav = 0.0
        
        # Final computations
        self.flosf = self.flosf * (1.0 - self.pctim - self.adimp)
        self.floin = self.floin * (1.0 - self.pctim - self.adimp)
        self.flobf = self.flobf * (1.0 - self.pctim - self.adimp)
        
        self.lzfsc = self.alzfsc / (1.0 + self.side)
        self.lzfpc = self.alzfpc / (1.0 + self.side)
        
        self._do_unit_hydrograph_routing()
        
        self.flwbf = self.flobf / (1.0 + self.side)
        if self.flwbf < 0.0:
            self.flwbf = 0.0
        
        total_before_channel_losses = self.flwbf + self.flwsf
        ratio_baseflow = self.flwbf / total_before_channel_losses if total_before_channel_losses > 0 else 0.0
        
        self.channel_flow = max(0.0, (self.flwbf + self.flwsf - self.ssout))
        self.evaporation_channel_water = min(evapt * self.sarva, self.channel_flow)
        
        self.runoff = self.channel_flow - self.evaporation_channel_water
        self.baseflow = self.runoff * ratio_baseflow
        
        if math.isnan(self.runoff):
            raise InvalidParameterSetException("Runoff is NaN - invalid parameter set")
