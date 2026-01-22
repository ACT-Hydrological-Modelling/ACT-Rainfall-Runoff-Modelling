"""
Parameter bounds loading and saving for rainfall-runoff models.

This module provides utilities for loading and saving parameter bounds
from human-readable text files, enabling easy customization of calibration
ranges without modifying code.

Supported Formats:
    - Text format (.txt): Simple key=min,max format with comments
    - CSV format (.csv): Standard CSV with columns for parameter, min, max

Example Text Format:
    # Sacramento Model Parameter Bounds
    # Format: parameter = min, max  [# optional comment]
    uztwm = 25.0, 125.0      # Upper zone tension water max
    uzfwm = 10.0, 75.0       # Upper zone free water max
"""

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def load_parameter_bounds(
    filepath: Union[str, Path]
) -> Dict[str, Tuple[float, float]]:
    """
    Load parameter bounds from a text file.
    
    The file format is simple key-value pairs with comments:
        parameter = min, max  # optional comment
    
    Lines starting with '#' are treated as comments.
    Blank lines are ignored.
    
    Args:
        filepath: Path to the bounds file (.txt format)
        
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        
    Example:
        >>> bounds = load_parameter_bounds('sacramento_bounds.txt')
        >>> print(bounds['uztwm'])
        (25.0, 125.0)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Parameter bounds file not found: {filepath}")
    
    bounds: Dict[str, Tuple[float, float]] = {}
    errors: List[str] = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            # Parse key = value format
            if '=' not in line:
                errors.append(f"Line {line_num}: Missing '=' in '{line}'")
                continue
            
            parts = line.split('=', 1)
            if len(parts) != 2:
                errors.append(f"Line {line_num}: Invalid format in '{line}'")
                continue
            
            param_name = parts[0].strip().lower()
            value_str = parts[1].strip()
            
            # Parse min, max values
            try:
                values = [v.strip() for v in value_str.split(',')]
                if len(values) != 2:
                    errors.append(
                        f"Line {line_num}: Expected 'min, max' format, got '{value_str}'"
                    )
                    continue
                
                min_val = float(values[0])
                max_val = float(values[1])
                
                # Validate bounds
                if min_val > max_val:
                    errors.append(
                        f"Line {line_num}: min ({min_val}) > max ({max_val}) for '{param_name}'"
                    )
                    continue
                
                bounds[param_name] = (min_val, max_val)
                
            except ValueError as e:
                errors.append(f"Line {line_num}: Invalid numeric value in '{value_str}': {e}")
    
    if errors:
        error_msg = "Errors parsing parameter bounds file:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    if not bounds:
        warnings.warn(f"No parameter bounds found in {filepath}")
    
    return bounds


def save_parameter_bounds(
    bounds: Dict[str, Tuple[float, float]],
    filepath: Union[str, Path],
    model_name: str = "Sacramento",
    descriptions: Optional[Dict[str, str]] = None
) -> None:
    """
    Save parameter bounds to a human-readable text file.
    
    Creates a formatted text file with comments explaining the format
    and optional parameter descriptions.
    
    Args:
        bounds: Dictionary mapping parameter names to (min, max) tuples
        filepath: Output file path
        model_name: Model name for header comment
        descriptions: Optional dict of parameter descriptions
        
    Example:
        >>> bounds = {'uztwm': (25.0, 125.0), 'uzfwm': (10.0, 75.0)}
        >>> save_parameter_bounds(bounds, 'my_bounds.txt')
    """
    filepath = Path(filepath)
    descriptions = descriptions or {}
    
    # Default descriptions for Sacramento parameters
    default_descriptions = {
        'uztwm': 'Upper zone tension water max [mm]',
        'uzfwm': 'Upper zone free water max [mm]',
        'lztwm': 'Lower zone tension water max [mm]',
        'lzfpm': 'Lower zone primary free water max [mm]',
        'lzfsm': 'Lower zone supplemental free water max [mm]',
        'uzk': 'Upper zone lateral drainage rate [1/day]',
        'lzpk': 'Lower zone primary drainage rate [1/day]',
        'lzsk': 'Lower zone supplemental drainage rate [1/day]',
        'zperc': 'Percolation demand scale parameter',
        'rexp': 'Percolation equation exponent',
        'pctim': 'Permanent impervious area fraction',
        'adimp': 'Additional impervious area fraction',
        'pfree': 'Fraction of percolation to free water',
        'rserv': 'Lower zone unavailable for transpiration',
        'side': 'Side flow ratio',
        'ssout': 'Subsurface outflow [mm]',
        'sarva': 'Riparian vegetation area fraction',
        'uh1': 'Unit hydrograph component 1',
        'uh2': 'Unit hydrograph component 2',
        'uh3': 'Unit hydrograph component 3',
        'uh4': 'Unit hydrograph component 4',
        'uh5': 'Unit hydrograph component 5',
    }
    
    # Merge with provided descriptions
    all_descriptions = {**default_descriptions, **descriptions}
    
    lines = [
        f"# {model_name} Model Parameter Bounds",
        "# " + "=" * 40,
        "#",
        "# Format: parameter = min, max  [# description]",
        "#",
        "# Edit the min and max values to customize calibration ranges.",
        "# Lines starting with '#' are comments and will be ignored.",
        "#",
        ""
    ]
    
    # Group parameters by type for better organization
    storage_params = ['uztwm', 'uzfwm', 'lztwm', 'lzfpm', 'lzfsm']
    rate_params = ['uzk', 'lzpk', 'lzsk']
    perc_params = ['zperc', 'rexp']
    frac_params = ['pctim', 'adimp', 'pfree', 'rserv', 'side', 'ssout', 'sarva']
    uh_params = ['uh1', 'uh2', 'uh3', 'uh4', 'uh5']
    
    def write_section(title: str, param_list: List[str]) -> None:
        section_params = [p for p in param_list if p in bounds]
        if section_params:
            lines.append(f"# {title}")
            for param in section_params:
                min_val, max_val = bounds[param]
                desc = all_descriptions.get(param, '')
                comment = f"  # {desc}" if desc else ""
                lines.append(f"{param} = {min_val}, {max_val}{comment}")
            lines.append("")
    
    write_section("Storage Parameters (mm)", storage_params)
    write_section("Rate Parameters (1/day)", rate_params)
    write_section("Percolation Parameters", perc_params)
    write_section("Fraction Parameters (dimensionless)", frac_params)
    write_section("Unit Hydrograph Components", uh_params)
    
    # Write any remaining parameters not in standard groups
    other_params = [p for p in bounds.keys() 
                    if p not in storage_params + rate_params + perc_params + 
                       frac_params + uh_params]
    write_section("Other Parameters", other_params)
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def load_parameter_bounds_csv(
    filepath: Union[str, Path]
) -> Dict[str, Tuple[float, float]]:
    """
    Load parameter bounds from a CSV file.
    
    Expected CSV format:
        parameter,min,max
        uztwm,25.0,125.0
        uzfwm,10.0,75.0
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Parameter bounds CSV not found: {filepath}")
    
    bounds: Dict[str, Tuple[float, float]] = {}
    errors: List[str] = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        # Check required columns
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header")
        
        required_cols = {'parameter', 'min', 'max'}
        # Handle case-insensitive column names
        col_map = {col.lower(): col for col in reader.fieldnames}
        
        missing = required_cols - set(col_map.keys())
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Expected: parameter, min, max"
            )
        
        for row_num, row in enumerate(reader, 2):  # Start at 2 (1 is header)
            try:
                param_name = row[col_map['parameter']].strip().lower()
                min_val = float(row[col_map['min']])
                max_val = float(row[col_map['max']])
                
                if min_val > max_val:
                    errors.append(
                        f"Row {row_num}: min ({min_val}) > max ({max_val}) for '{param_name}'"
                    )
                    continue
                
                bounds[param_name] = (min_val, max_val)
                
            except (ValueError, KeyError) as e:
                errors.append(f"Row {row_num}: {e}")
    
    if errors:
        error_msg = "Errors parsing parameter bounds CSV:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return bounds


def save_parameter_bounds_csv(
    bounds: Dict[str, Tuple[float, float]],
    filepath: Union[str, Path],
    descriptions: Optional[Dict[str, str]] = None
) -> None:
    """
    Save parameter bounds to a CSV file.
    
    Creates a CSV with columns: parameter, min, max, description
    
    Args:
        bounds: Dictionary mapping parameter names to (min, max) tuples
        filepath: Output file path
        descriptions: Optional dict of parameter descriptions
    """
    filepath = Path(filepath)
    descriptions = descriptions or {}
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'min', 'max', 'description'])
        
        for param, (min_val, max_val) in sorted(bounds.items()):
            desc = descriptions.get(param, '')
            writer.writerow([param, min_val, max_val, desc])


def validate_bounds(
    bounds: Dict[str, Tuple[float, float]],
    valid_parameters: Optional[List[str]] = None
) -> List[str]:
    """
    Validate parameter bounds.
    
    Args:
        bounds: Dictionary mapping parameter names to (min, max) tuples
        valid_parameters: Optional list of valid parameter names
        
    Returns:
        List of warning messages (empty if all valid)
    """
    warnings_list: List[str] = []
    
    for param, (min_val, max_val) in bounds.items():
        if min_val > max_val:
            warnings_list.append(
                f"Parameter '{param}': min ({min_val}) > max ({max_val})"
            )
        
        if min_val == max_val:
            warnings_list.append(
                f"Parameter '{param}': min == max ({min_val}), parameter will be fixed"
            )
        
        if valid_parameters and param not in valid_parameters:
            warnings_list.append(
                f"Parameter '{param}': not recognized by model"
            )
    
    return warnings_list
