"""
Sacramento Rainfall-Runoff Model - Python Implementation

An implementation of the version of the Sacramento rainfall-runoff model used
as part of IQQM by the New South Wales Department of Natural Resources.

This model was ported from the C# implementation in TIME.Models.RainfallRunoff.Sacramento.
The results (model outputs and state variables) are designed to reproduce the C# version exactly.

Original port notes:
- The original port consisted of a port of the fortran 90 core subroutines SNT7A and STR7B
  in the Sacramento implementation (Geoff Podger - NSW Department of Land and Water Conservation).
- Most variable names were kept as they were in the original implementation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4


# =============================================================================
# Constants
# =============================================================================

LENGTH_OF_UNIT_HYDROGRAPH: int = 5

# Undocumented constants from the original Fortran implementation
PDN20: float = 5.08
PDNOR: float = 25.4

# Model version for snapshot compatibility
VERSION: int = 1

# Tolerance for floating-point comparisons
TOLERANCE: float = 1e-10

# Error message constants
RUNOFF_IS_NAN: str = "Runoff is NaN - invalid parameter set"


# =============================================================================
# Custom Exceptions
# =============================================================================

class InvalidParameterSetException(Exception):
    """Raised when model parameters produce invalid results (e.g., NaN runoff)."""
    pass


# =============================================================================
# Parameter Validation
# =============================================================================

# Parameter warning ranges as per C# WarningRangeUnitValidator
PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    'adimp': (0.0, 0.2),
    'lzfpm': (40.0, 600.0),
    'lzfsm': (15.0, 300.0),
    'lzpk': (0.001, 0.015),
    'lzsk': (0.03, 0.2),
    'lztwm': (75.0, 300.0),
    'pctim': (0.0, 0.05),
    'pfree': (0.0, 0.5),
    'rexp': (1.4, 3.5),
    'rserv': (0.0, 0.4),
    'sarva': (0.0, 0.1),
    'side': (0.0, 0.8),
    'ssout': (0.0, 0.1),
    'uzfwm': (10.0, 75.0),
    'uzk': (0.2, 0.5),
    'uztwm': (25.0, 125.0),
    'zperc': (20.0, 300.0),
    'uh1': (0.0, 1.0),
    'uh2': (0.0, 1.0),
    'uh3': (0.0, 1.0),
    'uh4': (0.0, 1.0),
    'uh5': (0.0, 1.0),
}


def validate_parameter(name: str, value: float, warn: bool = True) -> None:
    """
    Validate a parameter value against its warning range.
    
    Args:
        name: Parameter name (lowercase)
        value: Parameter value to validate
        warn: If True, issue a warning for out-of-range values
    """
    if name in PARAMETER_RANGES:
        min_val, max_val = PARAMETER_RANGES[name]
        if not (min_val <= value <= max_val):
            if warn:
                warnings.warn(
                    f"Parameter '{name}' value {value} is outside recommended range "
                    f"[{min_val}, {max_val}]",
                    UserWarning
                )


# =============================================================================
# UnitHydrograph Class
# =============================================================================

@dataclass
class UnitHydrograph:
    """
    A discrete convolution unit hydrograph for routing surface runoff.
    
    The unit hydrograph distributes input flows across future time steps
    based on specified proportions (s_curve), then releases accumulated
    flows each time step.
    """
    s_curve: list[float] = field(default_factory=list)
    _stores: list[float] = field(default_factory=list)
    
    def initialise_hydrograph(self, proportions: list[float]) -> None:
        """
        Initialize the unit hydrograph with the given proportions.
        
        Args:
            proportions: List of proportions (should sum to 1.0)
        """
        self.s_curve = list(proportions)
        self._stores = [0.0] * len(proportions)
    
    def run_time_step(self, input_value: float) -> float:
        """
        Process one time step of the unit hydrograph.
        
        Args:
            input_value: Input flow to be routed
            
        Returns:
            Output flow for this time step
        """
        if not self.s_curve:
            return input_value
        
        # Add input distributed across stores according to s_curve
        for i in range(len(self._stores)):
            self._stores[i] += input_value * self.s_curve[i]
        
        # Output is the first store
        output = self._stores[0]
        
        # Shift stores (cascade down)
        for i in range(len(self._stores) - 1):
            self._stores[i] = self._stores[i + 1]
        self._stores[-1] = 0.0
        
        return output
    
    def reset(self) -> None:
        """Reset all stores to zero."""
        self._stores = [0.0] * len(self.s_curve) if self.s_curve else []
    
    def proportion_for_item(self, i: int) -> float:
        """
        Get the proportion for a specific lag index.
        
        Args:
            i: Index (0-based)
            
        Returns:
            Proportion value, or 0.0 if index is out of range
        """
        if 0 <= i < len(self.s_curve):
            return self.s_curve[i]
        return 0.0


# =============================================================================
# SacramentoSnapshot Dataclass
# =============================================================================

@dataclass
class SacramentoSnapshot:
    """
    Snapshot of Sacramento model state for persistence/restoration.
    """
    snapshot_entity_id: UUID
    version: int
    uztwc: float
    uzfwc: float
    lztwc: float
    adimc: float
    alzfpc: float
    alzfsc: float


@dataclass 
class StateValidationResult:
    """Result of validating a snapshot against the model."""
    is_valid: bool
    model_name: str
    entity_id: UUID
    message: str = ""
    
    @classmethod
    def valid(cls) -> StateValidationResult:
        """Create a valid result."""
        return cls(is_valid=True, model_name="Sacramento", entity_id=UUID(int=0))
    
    @classmethod
    def invalid_version(cls, model_name: str, entity_id: UUID, 
                        snapshot_version: int, expected_version: int) -> StateValidationResult:
        """Create an invalid version result."""
        return cls(
            is_valid=False,
            model_name=model_name,
            entity_id=entity_id,
            message=f"Version mismatch: snapshot version {snapshot_version}, expected {expected_version}"
        )


# =============================================================================
# Sacramento Model Class
# =============================================================================

class Sacramento:
    """
    Sacramento Rainfall-Runoff Model.
    
    An implementation of the version of the Sacramento rainfall-runoff model used
    as part of IQQM by the New South Wales Department of Natural Resources.
    
    Attributes:
        pet: Potential evapotranspiration input (mm)
        rainfall: Rainfall input (mm)
        runoff: Total runoff output (mm)
        baseflow: Baseflow component output (mm)
    """
    
    VERSION: int = 1
    
    def __init__(self) -> None:
        """Initialize the Sacramento model with default parameters."""
        # Unit hydrograph components
        self._unscaled_unit_hydrograph: list[float] = [0.0] * LENGTH_OF_UNIT_HYDROGRAPH
        self._unit_hydrograph: UnitHydrograph = UnitHydrograph()
        
        # Initialize all attributes to avoid AttributeError
        self._init_all_attributes()
        
        # Set default parameters
        self._init_parameters()
        
        # Update derived states
        self._update_internal_states()
        
        # Model metadata
        self.parameter_set_name: str = "<< Default Sacramento parameters. >>"
        
        # Snapshot entity ID
        self.snapshot_entity_id: UUID = uuid4()
    
    def _init_all_attributes(self) -> None:
        """Initialize all model attributes to default values."""
        # Inputs
        self.pet: float = 0.0
        self.rainfall: float = 0.0
        self._pliq: float = 0.0
        
        # Outputs
        self.runoff: float = 0.0
        self.baseflow: float = 0.0
        
        # Parameters (will be set by _init_parameters)
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
        
        # State variables - current storage
        self.uztwc: float = 0.0  # Upper zone tension water current storage
        self.uzfwc: float = 0.0  # Upper zone free water current storage
        self.lztwc: float = 0.0  # Lower zone tension water current storage
        self.lzfsc: float = 0.0  # Lower zone free water supplemental current storage
        self.lzfpc: float = 0.0  # Lower zone free water primary current storage
        
        # Adjusted lower zone variables (includes Side factor)
        self.alzfsc: float = 0.0
        self.alzfpc: float = 0.0
        self.alzfsm: float = 0.0
        self.alzfpm: float = 0.0
        
        # Additional state variables
        self.adimc: float = 0.0  # Additional impervious area content
        self.flobf: float = 0.0  # Baseflow accumulator
        self.flosf: float = 0.0  # Surface flow accumulator
        self.floin: float = 0.0  # Interflow accumulator
        self.flwbf: float = 0.0  # Weighted baseflow
        self.flwsf: float = 0.0  # Weighted surface flow
        self.roimp: float = 0.0  # Runoff from impervious area
        
        # Evaporation components
        self.evap_uztw: float = 0.0  # Evaporation from upper zone tension water
        self.evap_uzfw: float = 0.0  # Evaporation from upper zone free water
        self.e3: float = 0.0  # Lower zone tension water evaporation
        self.e5: float = 0.0  # Additional impervious evaporation
        self.evaporation_channel_water: float = 0.0  # Channel evaporation
        
        # Other state variables
        self.perc: float = 0.0  # Percolation
        self.pbase: float = 0.0  # Base percolation rate
        self.channel_flow: float = 0.0  # Total channel flow
        self.reserved_lower_zone: float = 0.0  # Reserved storage
        self.sum_lower_zone_capacities: float = 0.0  # Sum of LZ capacities
        self.hydrograph_store: float = 0.0  # Unit hydrograph storage tracker
        self.lzmpd: float = 0.0  # Lower zone max percolation demand
        
        # Previous state variables for mass balance
        self._prev_uztwc: float = 0.0
        self._prev_uzfwc: float = 0.0
        self._prev_lztwc: float = 0.0
        self._prev_hydrograph_store: float = 0.0
        self._prev_lzfsc: float = 0.0
        self._prev_lzfpc: float = 0.0
    
    # =========================================================================
    # Unit Hydrograph Properties (UH1-UH5)
    # =========================================================================
    
    @property
    def uh1(self) -> float:
        """First component of the unit hydrograph (proportion of runoff not lagged)."""
        return self._unscaled_unit_hydrograph[0]
    
    @uh1.setter
    def uh1(self, value: float) -> None:
        self._unscaled_unit_hydrograph[0] = self._bound_unit_hydrograph_component(value)
    
    @property
    def uh2(self) -> float:
        """Second component of the unit hydrograph (proportion lagged by 1 time step)."""
        return self._unscaled_unit_hydrograph[1]
    
    @uh2.setter
    def uh2(self, value: float) -> None:
        self._unscaled_unit_hydrograph[1] = self._bound_unit_hydrograph_component(value)
    
    @property
    def uh3(self) -> float:
        """Third component of the unit hydrograph (proportion lagged by 2 time steps)."""
        return self._unscaled_unit_hydrograph[2]
    
    @uh3.setter
    def uh3(self, value: float) -> None:
        self._unscaled_unit_hydrograph[2] = self._bound_unit_hydrograph_component(value)
    
    @property
    def uh4(self) -> float:
        """Fourth component of the unit hydrograph (proportion lagged by 3 time steps)."""
        return self._unscaled_unit_hydrograph[3]
    
    @uh4.setter
    def uh4(self, value: float) -> None:
        self._unscaled_unit_hydrograph[3] = self._bound_unit_hydrograph_component(value)
    
    @property
    def uh5(self) -> float:
        """Fifth component of the unit hydrograph (proportion lagged by 4 time steps)."""
        return self._unscaled_unit_hydrograph[4]
    
    @uh5.setter
    def uh5(self, value: float) -> None:
        self._unscaled_unit_hydrograph[4] = self._bound_unit_hydrograph_component(value)
    
    # =========================================================================
    # Mass Balance Property
    # =========================================================================
    
    @property
    def mass_balance(self) -> float:
        """
        Calculate the mass balance error for the current time step.
        
        Returns:
            Mass balance error (should be close to zero for valid simulation)
        """
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
    # Public Methods
    # =========================================================================
    
    def reset(self) -> None:
        """Reset the model to initial state (all stores and fluxes to zero)."""
        self._set_stores_and_fluxes_to_zero()
        self._update_internal_states()
        self._set_unit_hydrograph_components()
        self._set_prev_variables_for_mass_balance()
    
    def init_stores_full(self) -> None:
        """Initialize all water stores to their full capacity."""
        self.uzfwc = self.uzfwm
        self.uztwc = self.uztwm
        self.lztwc = self.lztwm
        self.lzfsc = self.lzfsm
        self.lzfpc = self.lzfpm
        self._update_internal_states()
    
    def run_time_step(self) -> None:
        """
        Execute one time step of the Sacramento model.
        
        This is the core algorithm that computes runoff and updates all
        state variables based on the current rainfall and PET inputs.
        """
        # Local variables matching original Fortran/C# naming
        # For reference and traceability: some variables named a and b (sic) in the original code
        # were used in alternance for different purposes
        ratio_uztw: float
        ratio_uzfw: float
        ratio_lztw: float
        ratio_lzfw: float
        transfered: float  # was named 'del' in DLWC code
        
        itime: int
        ninc: int
        addro: float
        adj: float
        bf: float
        dinc: float
        dlzp: float
        dlzs: float
        duz: float
        hpl: float
        lzair: float
        pav: float
        percfw: float
        percs: float
        perctw: float
        pinc: float
        ratio: float
        ratlp: float
        ratls: float
        
        evapt: float  # was evap / evapt in original subroutines
        
        # Store current values for mass balance calc
        self._set_prev_variables_for_mass_balance()
        
        self.reserved_lower_zone = self.rserv * (self.lzfpm + self.lzfsm)
        
        # At this point in the Fortran implementation, there were some pan factors applied.
        # This is not included here. A modified time series should be fed into the PET.
        evapt = self.pet
        
        self._pliq = self.rainfall
        
        # Determine evaporation from upper zone tension water store
        if self.uztwm > 0.0:
            self.evap_uztw = evapt * self.uztwc / self.uztwm
        else:
            self.evap_uztw = 0.0
        
        # Determine evaporation from upper zone free water
        if self.uztwc < self.evap_uztw:
            self.evap_uztw = self.uztwc
            self.uztwc = 0.0
            # Determine evaporation from free water surface
            self.evap_uzfw = min((evapt - self.evap_uztw), self.uzfwc)
            self.uzfwc = self.uzfwc - self.evap_uzfw
        else:
            self.uztwc = self.uztwc - self.evap_uztw
            self.evap_uzfw = 0.0
        
        # If the upper zone free water ratio exceeded the upper tension zone
        # content ratio, then transfer the free water into tension until the ratios are equal
        if self.uztwm > 0.0:
            ratio_uztw = self.uztwc / self.uztwm
        else:
            ratio_uztw = 1.0
        
        if self.uzfwm > 0.0:
            ratio_uzfw = self.uzfwc / self.uzfwm
        else:
            ratio_uzfw = 1.0
        
        if ratio_uztw < ratio_uzfw:
            # equivalent to the tension zone "sucking" the free water
            ratio_uztw = (self.uztwc + self.uzfwc) / (self.uztwm + self.uzfwm)
            self.uztwc = self.uztwm * ratio_uztw
            self.uzfwc = self.uzfwm * ratio_uztw
        
        # Evaporation from Adimp (additional impervious area) and Lower zone tension water
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
        
        # Compute the *transpiration* loss from the lower zone tension
        self.lztwc = self.lztwc - self.e3
        # Adjust the impervious area store
        self.adimc = self.adimc - self.e5
        self.evap_uztw = self.evap_uztw * (1 - self.adimp - self.pctim)
        self.evap_uzfw = self.evap_uzfw * (1 - self.adimp - self.pctim)
        self.e3 = self.e3 * (1 - self.adimp - self.pctim)
        self.e5 = self.e5 * self.adimp
        
        # Resupply the lower zone tension with water from the lower zone
        # free water, if more water is available there.
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
            # Transfer water from the lower zone secondary free water to lower zone
            # tension water store
            self.lztwc = self.lztwc + transfered
            self.alzfsc = self.alzfsc - transfered
            if self.alzfsc < 0:
                # Transfer primary free water if secondary free water is inadequate
                self.alzfpc = self.alzfpc + self.alzfsc
                self.alzfsc = 0.0
        
        # Runoff from the impervious or water covered area
        self.roimp = self._pliq * self.pctim
        
        # Reduce the rain by the amount of upper zone tension water deficiency
        pav = self._pliq + self.uztwc - self.uztwm
        if pav < 0:
            # Fill the upper zone tension water as much as rain permits
            self.adimc = self.adimc + self._pliq
            self.uztwc = self.uztwc + self._pliq
            pav = 0.0
        else:
            self.adimc = self.adimc + self.uztwm - self.uztwc
            self.uztwc = self.uztwm
        
        # The rest of this method is very close to the original Fortran implementation;
        # Given the look of it I doubt I can get things to reproduce from first principle.
        if pav <= PDN20:
            adj = 1.0
            itime = 2
        else:
            if pav < PDNOR:
                # Effective rainfall in a period is assumed to be half of the
                # period length for rain equal to the normal rainy period
                adj = 0.5 * math.sqrt(pav / PDNOR)
            else:
                adj = 1.0 - 0.5 * PDNOR / pav
            itime = 1
        
        self.flobf = 0.0
        self.flosf = 0.0
        self.floin = 0.0
        
        # Here again, being blindly faithful to original implementation
        hpl = self.alzfpm / (self.alzfpm + self.alzfsm)
        
        for ii in range(itime, 3):  # ii from itime to 2 (inclusive)
            # using int(math.floor()) to reproduce the fortran INT cast
            ninc = int(math.floor((self.uzfwc * adj + pav) * 0.2)) + 1
            dinc = 1.0 / ninc
            pinc = pav * dinc
            dinc = dinc * adj
            
            if ninc == 1 and adj >= 1.0:
                duz = self.uzk
                dlzp = self.lzpk
                dlzs = self.lzsk
            else:
                if self.uzk < 1.0:
                    duz = 1.0 - math.pow((1.0 - self.uzk), dinc)
                else:
                    duz = 1.0
                
                if self.lzpk < 1.0:
                    dlzp = 1.0 - math.pow((1.0 - self.lzpk), dinc)
                else:
                    dlzp = 1.0
                
                if self.lzsk < 1.0:
                    dlzs = 1.0 - math.pow((1.0 - self.lzsk), dinc)
                else:
                    dlzs = 1.0
            
            # Drainage and percolation loop
            for inc in range(1, ninc + 1):  # inc from 1 to ninc (inclusive)
                ratio = (self.adimc - self.uztwc) / self.lztwm
                addro = pinc * ratio * ratio
                
                # Compute the baseflow from the lower zone
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
                
                # Adjust the upper zone for percolation and interflow
                if self.uzfwc > 0.0:
                    # Determine percolation from the upper zone free water
                    # limited to available water and lower zone air space
                    lzair = (self.lztwm - self.lztwc + self.alzfsm - self.alzfsc + 
                             self.alzfpm - self.alzfpc)
                    if lzair > 0.0:
                        self.perc = (self.pbase * dinc * self.uzfwc) / self.uzfwm
                        self.perc = min(
                            self.uzfwc,
                            self.perc * (1.0 + (self.zperc * math.pow(
                                (1.0 - (self.alzfpc + self.alzfsc + self.lztwc) / 
                                 (self.alzfpm + self.alzfsm + self.lztwm)),
                                self.rexp
                            )))
                        )
                        self.perc = min(lzair, self.perc)
                        self.uzfwc = self.uzfwc - self.perc
                    else:
                        self.perc = 0.0
                    
                    # Compute the interflow
                    transfered = duz * self.uzfwc
                    self.floin = self.floin + transfered
                    self.uzfwc = self.uzfwc - transfered
                    
                    # Distribute water to lower zone tension and free water stores
                    perctw = min(self.perc * (1.0 - self.pfree), self.lztwm - self.lztwc)
                    percfw = self.perc - perctw
                    
                    # Shift any excess lower zone free water percolation to the
                    # lower zone tension water store
                    lzair = self.alzfsm - self.alzfsc + self.alzfpm - self.alzfpc
                    if percfw > lzair:
                        perctw = perctw + percfw - lzair
                        percfw = lzair
                    self.lztwc = self.lztwc + perctw
                    
                    # Distribute water between LZ free water supplemental and primary
                    if percfw > 0.0:
                        ratlp = 1.0 - self.alzfpc / self.alzfpm
                        ratls = 1.0 - self.alzfsc / self.alzfsm
                        percs = min(
                            self.alzfsm - self.alzfsc,
                            percfw * (1.0 - hpl * (2.0 * ratlp) / (ratlp + ratls))
                        )
                        self.alzfsc = self.alzfsc + percs
                        # Check for spill from supplemental to primary
                        if self.alzfsc > self.alzfsm:
                            percs = percs - self.alzfsc + self.alzfsm
                            self.alzfsc = self.alzfsm
                        self.alzfpc = self.alzfpc + percfw - percs
                        # Check for spill from primary to supplemental
                        if self.alzfpc > self.alzfpm:
                            self.alzfsc = self.alzfsc + self.alzfpc - self.alzfpm
                            self.alzfpc = self.alzfpm
                
                # Fill upper zone free water with tension water spill
                if pinc > 0.0:
                    pav = pinc
                    if pav - self.uzfwm + self.uzfwc <= 0:
                        self.uzfwc = self.uzfwc + pav
                    else:
                        pav = pav - self.uzfwm + self.uzfwc
                        self.uzfwc = self.uzfwm
                        self.flosf = self.flosf + pav
                        addro = addro + pav * (1.0 - addro / pinc)
                
                self.adimc = self.adimc + pinc - addro
                self.roimp = self.roimp + addro * self.adimp
            
            adj = 1.0 - adj
            pav = 0.0
        
        # Compute the storage volumes, runoff components and evaporation
        # Note evapotranspiration losses from the water surface and
        # riparian vegetation areas are computed in stn7a
        self.flosf = self.flosf * (1.0 - self.pctim - self.adimp)
        self.floin = self.floin * (1.0 - self.pctim - self.adimp)
        self.flobf = self.flobf * (1.0 - self.pctim - self.adimp)
        
        # End of call to stn7b
        # Following code to the end of the subroutine is part of stn7a
        
        self.lzfsc = self.alzfsc / (1.0 + self.side)
        self.lzfpc = self.alzfpc / (1.0 + self.side)
        
        # Adjust flow for unit hydrograph
        # Replacement / original code: using an object unitHydrograph
        self._do_unit_hydrograph_routing()
        
        self.flwbf = self.flobf / (1.0 + self.side)
        if self.flwbf < 0.0:
            self.flwbf = 0.0
        
        # Calculate the BFI prior to losses, in order to keep
        # this ratio in the final runoff and baseflow components.
        total_before_channel_losses = self.flwbf + self.flwsf
        ratio_baseflow = 0.0
        if total_before_channel_losses > 0:
            ratio_baseflow = self.flwbf / total_before_channel_losses
        
        # Subtract losses from the total channel flow (going to the subsurface discharge)
        self.channel_flow = max(0.0, (self.flwbf + self.flwsf - self.ssout))
        # following was e4
        self.evaporation_channel_water = min(evapt * self.sarva, self.channel_flow)
        
        self.runoff = self.channel_flow - self.evaporation_channel_water
        self.baseflow = self.runoff * ratio_baseflow
        
        if math.isnan(self.runoff):
            raise InvalidParameterSetException(RUNOFF_IS_NAN)
    
    # =========================================================================
    # Snapshot Methods
    # =========================================================================
    
    def get_snapshot(self) -> SacramentoSnapshot:
        """
        Get a snapshot of the current model state.
        
        Returns:
            SacramentoSnapshot containing current state variables
        """
        return SacramentoSnapshot(
            snapshot_entity_id=self.snapshot_entity_id,
            version=self.VERSION,
            uztwc=self.uztwc,
            uzfwc=self.uzfwc,
            lztwc=self.lztwc,
            adimc=self.adimc,
            alzfpc=self.alzfpc,
            alzfsc=self.alzfsc
        )
    
    def set_snapshot(self, snapshot: SacramentoSnapshot) -> None:
        """
        Restore model state from a snapshot.
        
        Args:
            snapshot: SacramentoSnapshot to restore from
        """
        self.uztwc = snapshot.uztwc
        self.uzfwc = snapshot.uzfwc
        self.lztwc = snapshot.lztwc
        self.adimc = snapshot.adimc
        self.alzfpc = snapshot.alzfpc
        self.alzfsc = snapshot.alzfsc
    
    def is_valid(self, snapshot: SacramentoSnapshot) -> StateValidationResult:
        """
        Validate a snapshot for compatibility with this model.
        
        Args:
            snapshot: SacramentoSnapshot to validate
            
        Returns:
            StateValidationResult indicating validity
        """
        if not isinstance(snapshot, SacramentoSnapshot):
            return StateValidationResult(
                is_valid=False,
                model_name="Sacramento",
                entity_id=self.snapshot_entity_id,
                message=f"State is incorrect type - expected: SacramentoSnapshot received: {type(snapshot).__name__}"
            )
        
        if snapshot.version != self.VERSION:
            return StateValidationResult.invalid_version(
                "Sacramento", self.snapshot_entity_id, snapshot.version, self.VERSION
            )
        
        return StateValidationResult.valid()
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _init_parameters(self) -> None:
        """Initialize model parameters to default values."""
        # Initialize unit hydrograph
        for i in range(len(self._unscaled_unit_hydrograph)):
            self._unscaled_unit_hydrograph[i] = 0.0
        
        self._unscaled_unit_hydrograph[0] = 1.0
        self._set_unit_hydrograph_components()
        
        # Some relatively arbitrary (though sensible) default values
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
        """Set the unit hydrograph components from the unscaled values."""
        self._unit_hydrograph.initialise_hydrograph(
            self._normalise(self._unscaled_unit_hydrograph)
        )
    
    def _get_unit_hydrograph_components(self) -> None:
        """Get unit hydrograph components from the unit hydrograph object."""
        if (self._unit_hydrograph is None or 
            not self._unit_hydrograph.s_curve or 
            self._unscaled_unit_hydrograph is None):
            return
        
        for i in range(LENGTH_OF_UNIT_HYDROGRAPH):
            self._unscaled_unit_hydrograph[i] = self._unit_hydrograph.proportion_for_item(i)
        
        self._set_unit_hydrograph_components()
    
    def _normalise(self, unscaled: list[float]) -> list[float]:
        """
        Normalise the unit hydrograph components to sum to 1.0.
        
        Args:
            unscaled: List of unscaled unit hydrograph components
            
        Returns:
            Normalised list that sums to 1.0
        """
        total = sum(unscaled)
        tmp = [0.0] * len(unscaled)
        
        if self._all_zeros(unscaled):
            # This can happen and cannot be prevented with intuitive behavior
            tmp[0] = 1.0  # default to no effect
            self._unit_hydrograph.initialise_hydrograph(tmp)
            return tmp
        
        if abs(total) < TOLERANCE:
            raise ValueError(
                "The sum of the unscaled components of the unit hydrograph is zero. "
                "This 'class' should not have let this happen - there is an issue"
            )
        
        for i in range(len(tmp)):
            tmp[i] = unscaled[i] / total
        
        return tmp
    
    def _all_zeros(self, values: list[float]) -> bool:
        """Check if all values in a list are zero."""
        for v in values:
            if v != 0:
                return False
        return True
    
    def _bound_unit_hydrograph_component(self, value: float) -> float:
        """Bound a unit hydrograph component to be non-negative."""
        return max(0.0, value)
    
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
        """Perform the routing of the surface runoff via unit hydrograph."""
        self.flwsf = self._unit_hydrograph.run_time_step(
            self.flosf + self.roimp + self.floin
        )
        self.hydrograph_store += (self.flosf + self.roimp + self.floin - self.flwsf)
    
    def clone(self) -> Sacramento:
        """
        Create a deep copy of this model.
        
        Returns:
            New Sacramento instance with same state
        """
        result = Sacramento()
        
        # Copy parameters
        result.uztwm = self.uztwm
        result.uzfwm = self.uzfwm
        result.lztwm = self.lztwm
        result.lzfpm = self.lzfpm
        result.lzfsm = self.lzfsm
        result.rserv = self.rserv
        result.adimp = self.adimp
        result.uzk = self.uzk
        result.lzpk = self.lzpk
        result.lzsk = self.lzsk
        result.zperc = self.zperc
        result.rexp = self.rexp
        result.pctim = self.pctim
        result.pfree = self.pfree
        result.side = self.side
        result.ssout = self.ssout
        result.sarva = self.sarva
        
        # Copy unit hydrograph
        result._unscaled_unit_hydrograph = list(self._unscaled_unit_hydrograph)
        result._unit_hydrograph = UnitHydrograph()
        result._unit_hydrograph.initialise_hydrograph(list(self._unit_hydrograph.s_curve))
        result._unit_hydrograph._stores = list(self._unit_hydrograph._stores)
        
        # Copy state variables
        result.uztwc = self.uztwc
        result.uzfwc = self.uzfwc
        result.lztwc = self.lztwc
        result.lzfsc = self.lzfsc
        result.lzfpc = self.lzfpc
        result.alzfsc = self.alzfsc
        result.alzfpc = self.alzfpc
        result.alzfsm = self.alzfsm
        result.alzfpm = self.alzfpm
        result.adimc = self.adimc
        result.hydrograph_store = self.hydrograph_store
        
        # Copy other state
        result.parameter_set_name = self.parameter_set_name
        result.snapshot_entity_id = self.snapshot_entity_id
        
        return result


# =============================================================================
# Module-level convenience functions
# =============================================================================

def create_sacramento_model(**params) -> Sacramento:
    """
    Create a Sacramento model with custom parameters.
    
    Args:
        **params: Parameter values to set (e.g., uztwm=50, lztwm=130)
        
    Returns:
        Configured Sacramento model instance
    """
    model = Sacramento()
    
    for name, value in params.items():
        if hasattr(model, name):
            setattr(model, name, value)
            validate_parameter(name, value)
        else:
            raise AttributeError(f"Sacramento model has no parameter '{name}'")
    
    model._update_internal_states()
    return model


if __name__ == "__main__":
    # Simple test/demo
    model = Sacramento()
    model.reset()
    
    # Run a few time steps with some rainfall
    for day in range(10):
        model.rainfall = 5.0 if day % 3 == 0 else 0.0
        model.pet = 3.0
        model.run_time_step()
        print(f"Day {day}: rainfall={model.rainfall:.1f}, runoff={model.runoff:.4f}, "
              f"baseflow={model.baseflow:.4f}, mass_balance={model.mass_balance:.2e}")
