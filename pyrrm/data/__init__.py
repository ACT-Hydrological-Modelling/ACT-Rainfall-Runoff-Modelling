"""
Data handling module for pyrrm.

Provides utilities for:
- Input data validation and handling
- Data readers for various formats (CSV, Excel, pickle)
- Data validators and preprocessors
- Unit conversion functions (mm to ML/day)
- Parameter bounds loading and saving from configuration files
"""

from pyrrm.data.input_handler import (
    InputDataHandler,
    load_csv,
    load_excel,
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
