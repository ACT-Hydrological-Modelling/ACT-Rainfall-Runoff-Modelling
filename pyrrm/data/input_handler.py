"""
Input data handling for rainfall-runoff models.

This module provides utilities for validating and preparing input data
for pyrrm models.
"""

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple, List
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel


class InputDataHandler:
    """
    Validates and prepares input data for rainfall-runoff models.
    
    This class ensures that input data meets model requirements in terms of:
    - Required columns (precipitation, PET)
    - Data frequency (daily)
    - Data types and missing values
    
    Attributes:
        model: The rainfall-runoff model
        data: Input DataFrame
        n_inputs: Number of input timesteps
        start_date: First date in the data
        end_date: Last date in the data
    
    Example:
        >>> from pyrrm.models import GR4J
        >>> from pyrrm.data import InputDataHandler
        >>> 
        >>> model = GR4J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7})
        >>> handler = InputDataHandler(model, df)
        >>> train, test = handler.split_train_test(0.7)
    """
    
    # Standard column names
    PRECIPITATION_COLUMNS = ['precipitation', 'rainfall', 'precip', 'rain', 'P']
    PET_COLUMNS = ['evapotranspiration', 'pet', 'evap', 'ET', 'PET']
    FLOW_COLUMNS = ['flow', 'runoff', 'discharge', 'Q', 'streamflow']
    
    def __init__(
        self, 
        model: 'BaseRainfallRunoffModel', 
        data: pd.DataFrame,
        precipitation_col: Optional[str] = None,
        pet_col: Optional[str] = None,
        validate: bool = True
    ):
        """
        Initialize the input data handler.
        
        Args:
            model: The rainfall-runoff model to use
            data: Input DataFrame with DatetimeIndex
            precipitation_col: Name of precipitation column (auto-detected if None)
            pet_col: Name of PET column (auto-detected if None)
            validate: If True, validate data on initialization
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(data).__name__}"
            )
        
        self.model = model
        self._original_data = data.copy()
        
        # Auto-detect or use specified column names
        self._precip_col = precipitation_col or self._find_column(data, self.PRECIPITATION_COLUMNS)
        self._pet_col = pet_col or self._find_column(data, self.PET_COLUMNS)
        
        # Standardize column names
        self.data = self._standardize_columns(data)
        
        if validate:
            self._validate_data()
            self._validate_frequency()
        
        self.n_inputs = len(self.data)
        self.start_date = self.data.index[0] if len(self.data) > 0 else None
        self.end_date = self.data.index[-1] if len(self.data) > 0 else None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
            # Case-insensitive search
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to 'precipitation' and 'pet'."""
        result = df.copy()
        
        # Rename precipitation column
        if self._precip_col and self._precip_col != 'precipitation':
            if self._precip_col in result.columns:
                result = result.rename(columns={self._precip_col: 'precipitation'})
        
        # Rename PET column
        if self._pet_col and self._pet_col != 'pet':
            if self._pet_col in result.columns:
                result = result.rename(columns={self._pet_col: 'pet'})
        
        return result
    
    def _validate_data(self) -> None:
        """Validate input data structure and contents."""
        # Check index type
        if not (isinstance(self.data.index, pd.DatetimeIndex) or 
                isinstance(self.data.index, pd.PeriodIndex)):
            raise TypeError(
                f"Index must be DatetimeIndex or PeriodIndex, got {type(self.data.index).__name__}"
            )
        
        # Convert PeriodIndex to DatetimeIndex
        if isinstance(self.data.index, pd.PeriodIndex):
            self.data.index = self.data.index.to_timestamp()
        
        # Check required columns
        if 'precipitation' not in self.data.columns:
            raise ValueError(
                f"Could not find precipitation column. "
                f"Looked for: {self.PRECIPITATION_COLUMNS}"
            )
        
        if 'pet' not in self.data.columns:
            raise ValueError(
                f"Could not find PET column. "
                f"Looked for: {self.PET_COLUMNS}"
            )
        
        # Check data types
        for col in ['precipitation', 'pet']:
            if not np.issubdtype(self.data[col].dtype, np.number):
                raise ValueError(f"Column '{col}' must be numeric")
        
        # Check for missing values
        for col in ['precipitation', 'pet']:
            n_missing = self.data[col].isna().sum()
            if n_missing > 0:
                warnings.warn(f"Found {n_missing} missing values in '{col}'")
        
        # Check for negative values
        for col in ['precipitation', 'pet']:
            n_negative = (self.data[col] < 0).sum()
            if n_negative > 0:
                warnings.warn(f"Found {n_negative} negative values in '{col}'")
    
    def _validate_frequency(self) -> None:
        """Validate data frequency matches model requirements."""
        freq = self.data.index.freq
        if freq is None:
            freq = pd.infer_freq(self.data.index)
        
        if freq is not None:
            freq_name = freq if isinstance(freq, str) else freq.name
            # Handle annual frequency with month suffix
            freq_base = freq_name.split('-')[0] if freq_name else None
            
            if freq_base and freq_base not in ['D', 'B', 'C']:
                warnings.warn(
                    f"Data frequency '{freq_name}' may not match model frequency "
                    f"'{self.model.frequency}'. Expected daily frequency."
                )
    
    def get_period(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> 'InputDataHandler':
        """
        Extract data for a specific time period.
        
        Args:
            start_date: Start of period
            end_date: End of period
            
        Returns:
            New InputDataHandler with data for the specified period
        """
        if start_date < self.start_date:
            warnings.warn(
                f"Start date {start_date} is before data start {self.start_date}"
            )
        if end_date > self.end_date:
            warnings.warn(
                f"End date {end_date} is after data end {self.end_date}"
            )
        
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        return InputDataHandler(self.model, self.data.loc[mask], validate=False)
    
    def split_train_test(
        self, 
        train_ratio: float = 0.7,
        warmup_years: int = 1
    ) -> Tuple['InputDataHandler', 'InputDataHandler']:
        """
        Split data into training and testing sets.
        
        Args:
            train_ratio: Fraction of data for training (default 0.7)
            warmup_years: Years at start for warmup (excluded from test metrics)
            
        Returns:
            Tuple of (training_handler, testing_handler)
        """
        n = len(self.data)
        train_end_idx = int(n * train_ratio)
        
        train_data = self.data.iloc[:train_end_idx]
        test_data = self.data.iloc[train_end_idx:]
        
        train_handler = InputDataHandler(self.model, train_data, validate=False)
        test_handler = InputDataHandler(self.model, test_data, validate=False)
        
        return train_handler, test_handler
    
    def get_observed_flow(self) -> Optional[np.ndarray]:
        """
        Get observed flow data if available.
        
        Returns:
            Observed flow array or None if not available
        """
        flow_col = self._find_column(self.data, self.FLOW_COLUMNS)
        if flow_col:
            return self.data[flow_col].values
        return None
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get precipitation and PET as numpy arrays.
        
        Returns:
            Tuple of (precipitation, pet) arrays
        """
        return (
            self.data['precipitation'].values.astype(float),
            self.data['pet'].values.astype(float)
        )
    
    def __len__(self) -> int:
        return self.n_inputs
    
    def __repr__(self) -> str:
        return (
            f"InputDataHandler(n_inputs={self.n_inputs}, "
            f"start='{self.start_date}', end='{self.end_date}')"
        )


def load_csv(
    filepath: str,
    date_column: str = 'date',
    date_format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load input data from a CSV file.
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        date_format: Date format string (optional)
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with DatetimeIndex
    """
    df = pd.read_csv(filepath, **kwargs)
    
    if date_column in df.columns:
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    
    return df


def load_excel(
    filepath: str,
    sheet_name: str = 0,
    date_column: str = 'date',
    **kwargs
) -> pd.DataFrame:
    """
    Load input data from an Excel file.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Sheet name or index
        date_column: Name of date column
        **kwargs: Additional arguments passed to pd.read_excel
        
    Returns:
        DataFrame with DatetimeIndex
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    
    return df


# =============================================================================
# Unit Conversion Functions
# =============================================================================

def mm_to_ml_per_day(
    depth_mm: np.ndarray, 
    catchment_area_km2: float
) -> np.ndarray:
    """
    Convert runoff depth (mm/day) to volumetric flow (ML/day).
    
    The conversion is based on the relationship:
        1 mm over 1 km² = 1,000,000 L = 1 ML
    
    Therefore:
        Flow (ML/day) = Depth (mm/day) × Area (km²)
    
    Args:
        depth_mm: Runoff depth in mm/day (can be scalar or array)
        catchment_area_km2: Catchment area in square kilometers
        
    Returns:
        Volumetric flow in ML/day (Megalitres per day)
        
    Example:
        >>> import numpy as np
        >>> from pyrrm.data import mm_to_ml_per_day
        >>> 
        >>> # 10 mm/day over a 100 km² catchment
        >>> flow_ml = mm_to_ml_per_day(10.0, 100.0)
        >>> print(f"{flow_ml:.1f} ML/day")  # 1000.0 ML/day
    """
    if catchment_area_km2 <= 0:
        raise ValueError(f"Catchment area must be positive, got {catchment_area_km2}")
    
    return np.asarray(depth_mm) * catchment_area_km2


def ml_per_day_to_mm(
    flow_ml: np.ndarray, 
    catchment_area_km2: float
) -> np.ndarray:
    """
    Convert volumetric flow (ML/day) to runoff depth (mm/day).
    
    The conversion is based on the relationship:
        1 ML over 1 km² = 1 mm
    
    Therefore:
        Depth (mm/day) = Flow (ML/day) / Area (km²)
    
    Args:
        flow_ml: Volumetric flow in ML/day (can be scalar or array)
        catchment_area_km2: Catchment area in square kilometers
        
    Returns:
        Runoff depth in mm/day
        
    Example:
        >>> import numpy as np
        >>> from pyrrm.data import ml_per_day_to_mm
        >>> 
        >>> # 1000 ML/day from a 100 km² catchment
        >>> depth_mm = ml_per_day_to_mm(1000.0, 100.0)
        >>> print(f"{depth_mm:.1f} mm/day")  # 10.0 mm/day
    """
    if catchment_area_km2 <= 0:
        raise ValueError(f"Catchment area must be positive, got {catchment_area_km2}")
    
    return np.asarray(flow_ml) / catchment_area_km2


def cumecs_to_ml_per_day(flow_cumecs: np.ndarray) -> np.ndarray:
    """
    Convert flow from cumecs (m³/s) to ML/day.
    
    Conversion: 1 m³/s = 86.4 ML/day
    (1 m³/s × 86400 s/day × 1 ML/1000 m³ = 86.4 ML/day)
    
    Args:
        flow_cumecs: Flow in cubic metres per second
        
    Returns:
        Flow in ML/day
    """
    return np.asarray(flow_cumecs) * 86.4


def ml_per_day_to_cumecs(flow_ml: np.ndarray) -> np.ndarray:
    """
    Convert flow from ML/day to cumecs (m³/s).
    
    Conversion: 1 ML/day = 1/86.4 m³/s ≈ 0.01157 m³/s
    
    Args:
        flow_ml: Flow in ML/day
        
    Returns:
        Flow in cubic metres per second
    """
    return np.asarray(flow_ml) / 86.4
