"""
Data handling module for pyrrm.

Provides utilities for:
- Input data validation and handling
- Data readers for various formats (CSV, Excel, pickle)
- Data validators and preprocessors
- Unit conversion functions (mm to ML/day)
- Parameter bounds loading and saving from configuration files
- Canonical column-name aliases and resolution helpers
"""

from pyrrm.data.input_handler import (
    COLUMN_ALIASES,
    InputDataHandler,
    load_catchment_data,
    load_csv,
    load_excel,
    resolve_column,
    mm_to_ml_per_day,
    ml_per_day_to_mm,
    cumecs_to_ml_per_day,
    ml_per_day_to_cumecs,
)

from pyrrm.data.parameter_bounds import (
    load_parameter_bounds,
    save_parameter_bounds,
    load_parameter_bounds_csv,
    save_parameter_bounds_csv,
    validate_bounds,
)

__all__ = [
    # Column aliases & helpers
    "COLUMN_ALIASES",
    "resolve_column",
    "load_catchment_data",
    # Input handling
    "InputDataHandler",
    "load_csv",
    "load_excel",
    # Unit conversions
    "mm_to_ml_per_day",
    "ml_per_day_to_mm",
    "cumecs_to_ml_per_day",
    "ml_per_day_to_cumecs",
    # Parameter bounds
    "load_parameter_bounds",
    "save_parameter_bounds",
    "load_parameter_bounds_csv",
    "save_parameter_bounds_csv",
    "validate_bounds",
]
