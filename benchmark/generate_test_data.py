#!/usr/bin/env python3
"""
Synthetic Test Data Generator for Sacramento Model Verification

Generates synthetic rainfall and PET time series data for benchmarking
the Python Sacramento implementation against the C# reference.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(n_days: int = 1095, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic rainfall and PET data for model testing.
    
    Args:
        n_days: Number of days to generate (default 3 years = 1095 days)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: timestep, rainfall, pet
    """
    np.random.seed(seed)
    
    # Generate rainfall with mixed distribution
    rainfall = np.zeros(n_days)
    for i in range(n_days):
        p = np.random.random()
        if p < 0.70:
            # Dry day (70% probability)
            rainfall[i] = 0.0
        elif p < 0.90:
            # Light rain (20% probability, 1-10 mm exponential)
            rainfall[i] = np.random.exponential(5.0)
        elif p < 0.98:
            # Moderate rain (8% probability, 10-30 mm)
            rainfall[i] = np.random.uniform(10, 30)
        else:
            # Heavy rain/storm (2% probability, 30-100 mm)
            rainfall[i] = np.random.uniform(30, 100)
    
    # Generate seasonal PET (sinusoidal pattern)
    days = np.arange(n_days)
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * days / 365)
    pet = np.maximum(pet, 0.5)  # Minimum PET of 0.5 mm
    
    return pd.DataFrame({
        'timestep': days,
        'rainfall': np.round(rainfall, 4),
        'pet': np.round(pet, 4)
    })


def generate_zero_rainfall_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate data with zero rainfall (evaporation only test).
    
    Args:
        n_days: Number of days
        seed: Random seed
        
    Returns:
        DataFrame with zero rainfall and seasonal PET
    """
    np.random.seed(seed)
    
    days = np.arange(n_days)
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * days / 365)
    pet = np.maximum(pet, 0.5)
    
    return pd.DataFrame({
        'timestep': days,
        'rainfall': np.zeros(n_days),
        'pet': np.round(pet, 4)
    })


def generate_storm_event_data(storm_mm: float = 100.0, 
                               pre_days: int = 30,
                               post_days: int = 60) -> pd.DataFrame:
    """
    Generate data with a single storm event for testing peak and recession.
    
    Args:
        storm_mm: Storm rainfall amount in mm
        pre_days: Days before storm
        post_days: Days after storm
        
    Returns:
        DataFrame with single storm event
    """
    n_days = pre_days + 1 + post_days
    days = np.arange(n_days)
    
    rainfall = np.zeros(n_days)
    rainfall[pre_days] = storm_mm  # Storm on specified day
    
    # Constant PET for simplicity
    pet = np.full(n_days, 3.0)
    
    return pd.DataFrame({
        'timestep': days,
        'rainfall': rainfall,
        'pet': pet
    })


def generate_long_dry_spell_data(dry_days: int = 180, seed: int = 42) -> pd.DataFrame:
    """
    Generate data with a long dry spell for storage depletion testing.
    
    Args:
        dry_days: Number of dry days
        seed: Random seed
        
    Returns:
        DataFrame with no rainfall
    """
    np.random.seed(seed)
    
    days = np.arange(dry_days)
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * days / 365)
    pet = np.maximum(pet, 0.5)
    
    return pd.DataFrame({
        'timestep': days,
        'rainfall': np.zeros(dry_days),
        'pet': np.round(pet, 4)
    })


def create_parameter_sets() -> dict:
    """
    Create the parameter sets for testing different model behaviors.
    
    Returns:
        Dictionary of parameter sets
    """
    return {
        "default": {
            "description": "Default parameters from C# initParameters()",
            "params": {
                "uztwm": 50.0,
                "uzfwm": 40.0,
                "lztwm": 130.0,
                "lzfpm": 60.0,
                "lzfsm": 25.0,
                "rserv": 0.3,
                "adimp": 0.0,
                "uzk": 0.3,
                "lzpk": 0.01,
                "lzsk": 0.05,
                "zperc": 40.0,
                "rexp": 1.0,
                "pctim": 0.01,
                "pfree": 0.06,
                "side": 0.0,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 1.0,
                "uh2": 0.0,
                "uh3": 0.0,
                "uh4": 0.0,
                "uh5": 0.0
            }
        },
        "wet_catchment": {
            "description": "High storage, fast response catchment",
            "params": {
                "uztwm": 100.0,
                "uzfwm": 60.0,
                "lztwm": 200.0,
                "lzfpm": 100.0,
                "lzfsm": 50.0,
                "rserv": 0.3,
                "adimp": 0.0,
                "uzk": 0.4,
                "lzpk": 0.01,
                "lzsk": 0.08,
                "zperc": 60.0,
                "rexp": 1.5,
                "pctim": 0.01,
                "pfree": 0.1,
                "side": 0.0,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 1.0,
                "uh2": 0.0,
                "uh3": 0.0,
                "uh4": 0.0,
                "uh5": 0.0
            }
        },
        "dry_catchment": {
            "description": "Low storage, slow response catchment",
            "params": {
                "uztwm": 30.0,
                "uzfwm": 20.0,
                "lztwm": 80.0,
                "lzfpm": 40.0,
                "lzfsm": 20.0,
                "rserv": 0.2,
                "adimp": 0.0,
                "uzk": 0.25,
                "lzpk": 0.005,
                "lzsk": 0.04,
                "zperc": 30.0,
                "rexp": 1.0,
                "pctim": 0.01,
                "pfree": 0.04,
                "side": 0.0,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 1.0,
                "uh2": 0.0,
                "uh3": 0.0,
                "uh4": 0.0,
                "uh5": 0.0
            }
        },
        "impervious": {
            "description": "High impervious fraction catchment",
            "params": {
                "uztwm": 50.0,
                "uzfwm": 40.0,
                "lztwm": 130.0,
                "lzfpm": 60.0,
                "lzfsm": 25.0,
                "rserv": 0.3,
                "adimp": 0.15,
                "uzk": 0.3,
                "lzpk": 0.01,
                "lzsk": 0.05,
                "zperc": 40.0,
                "rexp": 1.0,
                "pctim": 0.04,
                "pfree": 0.06,
                "side": 0.0,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 1.0,
                "uh2": 0.0,
                "uh3": 0.0,
                "uh4": 0.0,
                "uh5": 0.0
            }
        },
        "deep_groundwater": {
            "description": "Significant baseflow contribution",
            "params": {
                "uztwm": 50.0,
                "uzfwm": 40.0,
                "lztwm": 130.0,
                "lzfpm": 400.0,
                "lzfsm": 200.0,
                "rserv": 0.3,
                "adimp": 0.0,
                "uzk": 0.3,
                "lzpk": 0.008,
                "lzsk": 0.04,
                "zperc": 100.0,
                "rexp": 2.0,
                "pctim": 0.01,
                "pfree": 0.2,
                "side": 0.3,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 1.0,
                "uh2": 0.0,
                "uh3": 0.0,
                "uh4": 0.0,
                "uh5": 0.0
            }
        },
        "unit_hydrograph": {
            "description": "Lagged response using unit hydrograph",
            "params": {
                "uztwm": 50.0,
                "uzfwm": 40.0,
                "lztwm": 130.0,
                "lzfpm": 60.0,
                "lzfsm": 25.0,
                "rserv": 0.3,
                "adimp": 0.0,
                "uzk": 0.3,
                "lzpk": 0.01,
                "lzsk": 0.05,
                "zperc": 40.0,
                "rexp": 1.0,
                "pctim": 0.01,
                "pfree": 0.06,
                "side": 0.0,
                "ssout": 0.0,
                "sarva": 0.0,
                "uh1": 0.3,
                "uh2": 0.4,
                "uh3": 0.2,
                "uh4": 0.1,
                "uh5": 0.0
            }
        }
    }


def generate_all_test_data(output_dir: Path) -> None:
    """
    Generate all test data files.
    
    Args:
        output_dir: Directory to write output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic test data...")
    
    # Main synthetic dataset (3 years)
    df_main = generate_synthetic_data(n_days=1095, seed=42)
    df_main.to_csv(output_dir / "synthetic_inputs.csv", index=False)
    print(f"  Created synthetic_inputs.csv ({len(df_main)} records)")
    
    # Zero rainfall dataset
    df_zero = generate_zero_rainfall_data(n_days=365, seed=42)
    df_zero.to_csv(output_dir / "zero_rainfall_inputs.csv", index=False)
    print(f"  Created zero_rainfall_inputs.csv ({len(df_zero)} records)")
    
    # Storm event dataset
    df_storm = generate_storm_event_data(storm_mm=100.0)
    df_storm.to_csv(output_dir / "storm_event_inputs.csv", index=False)
    print(f"  Created storm_event_inputs.csv ({len(df_storm)} records)")
    
    # Long dry spell dataset
    df_dry = generate_long_dry_spell_data(dry_days=180, seed=42)
    df_dry.to_csv(output_dir / "long_dry_spell_inputs.csv", index=False)
    print(f"  Created long_dry_spell_inputs.csv ({len(df_dry)} records)")
    
    # Parameter sets
    param_sets = create_parameter_sets()
    with open(output_dir / "parameter_sets.json", "w") as f:
        json.dump(param_sets, f, indent=2)
    print(f"  Created parameter_sets.json ({len(param_sets)} parameter sets)")
    
    print("\nTest data generation complete!")


def main():
    """Main entry point for test data generation."""
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "test_data"
    
    generate_all_test_data(output_dir)


if __name__ == "__main__":
    main()
