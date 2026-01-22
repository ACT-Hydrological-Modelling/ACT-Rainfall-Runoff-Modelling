"""
Base classes and interfaces for rainfall-runoff models.

This module defines the abstract base class that all rainfall-runoff models
in pyrrm must inherit from, ensuring a consistent interface across models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class ModelParameter:
    """
    Definition of a model parameter with bounds and metadata.
    
    Attributes:
        name: Parameter name (identifier used in code)
        default: Default value for the parameter
        min_bound: Minimum allowed value (for calibration)
        max_bound: Maximum allowed value (for calibration)
        description: Human-readable description of the parameter
        unit: Physical unit of the parameter
    """
    name: str
    default: float
    min_bound: float
    max_bound: float
    description: str
    unit: str
    
    def validate(self, value: float) -> bool:
        """Check if a value is within the parameter bounds."""
        return self.min_bound <= value <= self.max_bound
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter definition to dictionary."""
        return {
            'name': self.name,
            'default': self.default,
            'min_bound': self.min_bound,
            'max_bound': self.max_bound,
            'description': self.description,
            'unit': self.unit
        }


@dataclass
class ModelState:
    """
    Container for model state variables.
    
    This class stores the internal state of a model at a given point in time,
    allowing for state saving/restoration (e.g., for split-sample testing).
    
    Attributes:
        values: Dictionary mapping state variable names to their values
        timestamp: Optional timestamp associated with this state
    """
    values: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None
    
    def copy(self) -> 'ModelState':
        """Create a deep copy of the state."""
        import copy
        return ModelState(
            values=copy.deepcopy(self.values),
            timestamp=self.timestamp
        )


class BaseRainfallRunoffModel(ABC):
    """
    Abstract base class for all rainfall-runoff models in pyrrm.
    
    This class defines the interface that all models must implement,
    ensuring consistency across different model types (Sacramento, GR4J, etc.).
    
    Class Attributes:
        name: Model identifier (e.g., 'sacramento', 'gr4j')
        description: Brief description of the model
        frequency: Timestep frequency ('D' for daily)
    
    Instance Attributes:
        catchment_area_km2: Optional catchment area in km². When set, outputs
            are automatically scaled from mm to ML/day.
    
    Output Units:
        - If catchment_area_km2 is None (default): outputs are in mm/day
        - If catchment_area_km2 is set: outputs are in ML/day
        
        The conversion is: Flow (ML/day) = Depth (mm/day) × Area (km²)
    
    Example:
        >>> from pyrrm.models import GR4J
        >>> model = GR4J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7})
        >>> results = model.run(input_data)  # Outputs in mm
        >>> 
        >>> # With catchment scaling:
        >>> model.set_catchment_area(150.5)  # 150.5 km²
        >>> results = model.run(input_data)  # Outputs in ML/day
    """
    
    name: str = "base"
    description: str = "Base rainfall-runoff model"
    frequency: str = 'D'  # Daily timestep (all models in this library are daily)
    
    # Catchment area for unit conversion (None = outputs in mm)
    _catchment_area_km2: Optional[float] = None
    
    @property
    @abstractmethod
    def parameter_definitions(self) -> List[ModelParameter]:
        """
        Return list of model parameters with metadata.
        
        Returns:
            List of ModelParameter objects defining all model parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary mapping parameter names to values
            
        Raises:
            ValueError: If required parameters are missing or values are invalid
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary mapping parameter names to current values
        """
        pass
    
    @abstractmethod
    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Run model for entire input time series.
        
        Args:
            inputs: DataFrame with DatetimeIndex containing:
                - 'precipitation' (or 'rainfall'): Daily precipitation [mm]
                - 'pet' (or 'evapotranspiration'): Daily potential ET [mm]
                
        Returns:
            DataFrame with DatetimeIndex containing model outputs:
                - 'runoff': Total runoff [mm or ML/day depending on catchment_area_km2]
                - 'baseflow': Baseflow component [mm or ML/day] (if available)
                - Additional model-specific outputs
                
        Note:
            Output units depend on whether catchment_area_km2 is set:
            - If None (default): outputs are in mm/day
            - If set: outputs are in ML/day (scaled by catchment area)
        """
        pass
    
    @abstractmethod
    def run_timestep(self, precipitation: float, pet: float) -> Dict[str, float]:
        """
        Run model for a single timestep.
        
        Args:
            precipitation: Precipitation for this timestep [mm]
            pet: Potential evapotranspiration for this timestep [mm]
            
        Returns:
            Dictionary with model outputs for this timestep:
                - 'runoff': Total runoff [mm or ML/day depending on catchment_area_km2]
                - 'baseflow': Baseflow component [mm or ML/day] (if available)
                - Additional model-specific outputs
                
        Note:
            Output units depend on whether catchment_area_km2 is set:
            - If None (default): outputs are in mm/day
            - If set: outputs are in ML/day (scaled by catchment area)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset model to initial state.
        
        This sets all internal stores and state variables to their
        initial values (typically zero or model-specific defaults).
        """
        pass
    
    @abstractmethod
    def get_state(self) -> ModelState:
        """
        Get current model state.
        
        Returns:
            ModelState object containing all internal state variables
        """
        pass
    
    @abstractmethod
    def set_state(self, state: ModelState) -> None:
        """
        Set model state.
        
        Args:
            state: ModelState object to restore
            
        Raises:
            ValueError: If state is incompatible with this model
        """
        pass
    
    # Custom parameter bounds override (set via set_parameter_bounds or load_parameter_bounds)
    _custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds as a dictionary.
        
        If custom bounds have been set via `set_parameter_bounds()` or 
        `load_parameter_bounds()`, those will be returned. Otherwise,
        returns the default bounds from parameter_definitions.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        if self._custom_bounds is not None:
            # Merge custom bounds with defaults (custom takes precedence)
            default_bounds = {
                p.name: (p.min_bound, p.max_bound) 
                for p in self.parameter_definitions
            }
            return {**default_bounds, **self._custom_bounds}
        
        return {
            p.name: (p.min_bound, p.max_bound) 
            for p in self.parameter_definitions
        }
    
    def set_parameter_bounds(
        self, 
        bounds: Dict[str, Tuple[float, float]],
        validate: bool = True
    ) -> None:
        """
        Set custom parameter bounds for calibration.
        
        This overrides the default bounds from parameter_definitions.
        The custom bounds will be used by get_parameter_bounds() and
        by calibration methods.
        
        Args:
            bounds: Dictionary mapping parameter names to (min, max) tuples
            validate: If True, validate that bounds are valid (min <= max)
                     and warn about unrecognized parameters
                     
        Raises:
            ValueError: If validate=True and bounds are invalid
            
        Example:
            >>> model = Sacramento()
            >>> model.set_parameter_bounds({
            ...     'uztwm': (30.0, 100.0),  # Narrower range
            ...     'lztwm': (100.0, 250.0),
            ... })
        """
        if validate:
            errors = []
            valid_params = [p.name for p in self.parameter_definitions]
            
            for name, (min_val, max_val) in bounds.items():
                if min_val > max_val:
                    errors.append(
                        f"Parameter '{name}': min ({min_val}) > max ({max_val})"
                    )
                if name not in valid_params:
                    import warnings
                    warnings.warn(
                        f"Parameter '{name}' not recognized by {self.name} model"
                    )
            
            if errors:
                raise ValueError(
                    "Invalid parameter bounds:\n" + 
                    "\n".join(f"  - {e}" for e in errors)
                )
        
        # Initialize if needed
        if self._custom_bounds is None:
            self._custom_bounds = {}
        
        # Update custom bounds
        self._custom_bounds.update(bounds)
    
    def clear_custom_bounds(self) -> None:
        """
        Clear all custom parameter bounds.
        
        After calling this, get_parameter_bounds() will return the
        default bounds from parameter_definitions.
        """
        self._custom_bounds = None
    
    def load_parameter_bounds(self, filepath: str) -> Dict[str, Tuple[float, float]]:
        """
        Load parameter bounds from a configuration file.
        
        Supports both text format (.txt) and CSV format (.csv).
        The loaded bounds are automatically applied to the model.
        
        Args:
            filepath: Path to the bounds file
            
        Returns:
            Dictionary of loaded bounds
            
        Example:
            >>> model = Sacramento()
            >>> bounds = model.load_parameter_bounds('custom_bounds.txt')
            >>> print(f"Loaded {len(bounds)} parameter bounds")
        """
        from pathlib import Path
        from pyrrm.data.parameter_bounds import (
            load_parameter_bounds as _load_txt,
            load_parameter_bounds_csv as _load_csv,
        )
        
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            bounds = _load_csv(filepath)
        else:
            bounds = _load_txt(filepath)
        
        self.set_parameter_bounds(bounds)
        return bounds
    
    def save_parameter_bounds(
        self, 
        filepath: str,
        include_descriptions: bool = True
    ) -> None:
        """
        Save current parameter bounds to a configuration file.
        
        The format is determined by the file extension:
        - .txt: Human-readable text format with comments
        - .csv: Standard CSV format
        
        Args:
            filepath: Output file path
            include_descriptions: If True, include parameter descriptions
            
        Example:
            >>> model = Sacramento()
            >>> model.save_parameter_bounds('sacramento_bounds.txt')
        """
        from pathlib import Path
        from pyrrm.data.parameter_bounds import (
            save_parameter_bounds as _save_txt,
            save_parameter_bounds_csv as _save_csv,
        )
        
        filepath = Path(filepath)
        bounds = self.get_parameter_bounds()
        
        # Get descriptions from parameter definitions
        descriptions = {}
        if include_descriptions:
            for p in self.parameter_definitions:
                desc = p.description
                if p.unit:
                    desc = f"{desc} [{p.unit}]"
                descriptions[p.name] = desc
        
        if filepath.suffix.lower() == '.csv':
            _save_csv(bounds, filepath, descriptions if include_descriptions else None)
        else:
            _save_txt(bounds, filepath, model_name=self.name.title(), descriptions=descriptions)
    
    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values.
        
        Returns:
            Dictionary mapping parameter names to default values
        """
        return {p.name: p.default for p in self.parameter_definitions}
    
    def validate_parameters(self, params: Dict[str, float]) -> List[str]:
        """
        Validate parameter values against bounds.
        
        Args:
            params: Dictionary of parameter values to validate
            
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        param_defs = {p.name: p for p in self.parameter_definitions}
        
        for name, value in params.items():
            if name in param_defs:
                p = param_defs[name]
                if not p.validate(value):
                    errors.append(
                        f"Parameter '{name}' value {value} is outside bounds "
                        f"[{p.min_bound}, {p.max_bound}]"
                    )
        
        # Check for missing required parameters
        for p in self.parameter_definitions:
            if p.name not in params:
                errors.append(f"Missing required parameter: '{p.name}'")
        
        return errors
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    # =========================================================================
    # Catchment Area and Unit Conversion
    # =========================================================================
    
    @property
    def catchment_area_km2(self) -> Optional[float]:
        """
        Get the catchment area in km².
        
        Returns:
            Catchment area in km², or None if not set
        """
        return self._catchment_area_km2
    
    @catchment_area_km2.setter
    def catchment_area_km2(self, value: Optional[float]) -> None:
        """
        Set the catchment area in km².
        
        Args:
            value: Catchment area in km², or None to disable scaling
        """
        if value is not None and value <= 0:
            raise ValueError(f"Catchment area must be positive, got {value}")
        self._catchment_area_km2 = value
    
    def set_catchment_area(self, area_km2: Optional[float]) -> None:
        """
        Set the catchment area for unit conversion.
        
        When set, model outputs will be automatically converted from mm/day
        to ML/day using the formula: Flow (ML/day) = Depth (mm/day) × Area (km²)
        
        Args:
            area_km2: Catchment area in square kilometers, or None to disable
            
        Example:
            >>> model = Sacramento()
            >>> model.set_catchment_area(150.5)  # 150.5 km² catchment
            >>> print(model.output_units)  # 'ML/day'
        """
        self.catchment_area_km2 = area_km2
    
    @property
    def output_units(self) -> str:
        """
        Get the current output units based on catchment area setting.
        
        Returns:
            'mm' if no catchment area is set, 'ML/day' if catchment area is set
        """
        return 'ML/day' if self._catchment_area_km2 is not None else 'mm'
    
    def _scale_to_volume(self, depth_mm: float) -> float:
        """
        Convert runoff depth to volumetric flow if catchment area is set.
        
        Args:
            depth_mm: Runoff depth in mm
            
        Returns:
            If catchment_area_km2 is set: flow in ML/day
            Otherwise: unchanged depth in mm
        """
        if self._catchment_area_km2 is not None:
            return depth_mm * self._catchment_area_km2
        return depth_mm
    
    def _scale_array_to_volume(self, depth_mm: np.ndarray) -> np.ndarray:
        """
        Convert runoff depth array to volumetric flow if catchment area is set.
        
        Args:
            depth_mm: Runoff depth array in mm
            
        Returns:
            If catchment_area_km2 is set: flow array in ML/day
            Otherwise: unchanged depth array in mm
        """
        if self._catchment_area_km2 is not None:
            return depth_mm * self._catchment_area_km2
        return depth_mm
