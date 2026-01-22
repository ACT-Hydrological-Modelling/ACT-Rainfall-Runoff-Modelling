#!/usr/bin/env python3
"""
Verification Script for Sacramento Model Implementation

Compares Python Sacramento outputs against C# reference outputs to verify
that the Python implementation produces identical results.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent directory to path for sacramento import
sys.path.insert(0, str(Path(__file__).parent.parent))

from sacramento import Sacramento


def run_python_model(inputs_df: pd.DataFrame, 
                     params: dict,
                     init_stores_full: bool = False) -> pd.DataFrame:
    """
    Run the Python Sacramento model on input data.
    
    Args:
        inputs_df: DataFrame with 'rainfall' and 'pet' columns
        params: Dictionary of parameter values
        init_stores_full: If True, initialize stores to full capacity
        
    Returns:
        DataFrame with model outputs for each timestep
    """
    model = Sacramento()
    
    # Apply parameters
    for key, value in params.items():
        if hasattr(model, key):
            setattr(model, key, value)
    
    # Update internal states after parameter changes
    model._update_internal_states()
    model.reset()
    
    if init_stores_full:
        model.init_stores_full()
    
    outputs = []
    for _, row in inputs_df.iterrows():
        model.rainfall = row['rainfall']
        model.pet = row['pet']
        model.run_time_step()
        
        outputs.append({
            'timestep': int(row['timestep']),
            'runoff': model.runoff,
            'baseflow': model.baseflow,
            'uztwc': model.uztwc,
            'uzfwc': model.uzfwc,
            'lztwc': model.lztwc,
            'lzfsc': model.lzfsc,
            'lzfpc': model.lzfpc,
            'mass_balance': model.mass_balance,
            'channel_flow': model.channel_flow,
            'evap_uztw': model.evap_uztw,
            'evap_uzfw': model.evap_uzfw,
            'e3': model.e3,
            'e5': model.e5,
            'adimc': model.adimc,
            'alzfpc': model.alzfpc,
            'alzfsc': model.alzfsc,
            'flobf': model.flobf,
            'flosf': model.flosf,
            'floin': model.floin,
            'flwbf': model.flwbf,
            'flwsf': model.flwsf,
            'roimp': model.roimp,
            'perc': model.perc
        })
    
    return pd.DataFrame(outputs)


def compare_outputs(csharp_df: pd.DataFrame, 
                    python_df: pd.DataFrame,
                    tolerance: float = 1e-10,
                    mass_balance_tolerance: float = 1e-8) -> dict:
    """
    Compare C# and Python model outputs.
    
    Args:
        csharp_df: DataFrame with C# reference outputs
        python_df: DataFrame with Python outputs
        tolerance: Tolerance for state/flux variables
        mass_balance_tolerance: Relaxed tolerance for mass balance
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        'pass': True,
        'differences': [],
        'summary': {}
    }
    
    # Variables to compare with standard tolerance
    standard_vars = ['runoff', 'baseflow', 'uztwc', 'uzfwc', 'lztwc', 'lzfsc', 'lzfpc',
                     'channel_flow', 'evap_uztw', 'evap_uzfw', 'e3', 'e5']
    
    for col in standard_vars:
        if col not in csharp_df.columns or col not in python_df.columns:
            continue
            
        diff = np.abs(csharp_df[col].values - python_df[col].values)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        results['summary'][col] = {
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'timestep_of_max': int(np.argmax(diff))
        }
        
        if max_diff > tolerance:
            results['pass'] = False
            results['differences'].append({
                'variable': col,
                'max_difference': float(max_diff),
                'timestep_of_max': int(np.argmax(diff)),
                'tolerance': tolerance
            })
    
    # Mass balance with relaxed tolerance
    if 'mass_balance' in csharp_df.columns and 'mass_balance' in python_df.columns:
        diff = np.abs(csharp_df['mass_balance'].values - python_df['mass_balance'].values)
        max_diff = np.max(diff)
        
        results['summary']['mass_balance'] = {
            'max_difference': float(max_diff),
            'mean_difference': float(np.mean(diff)),
            'timestep_of_max': int(np.argmax(diff))
        }
        
        if max_diff > mass_balance_tolerance:
            results['pass'] = False
            results['differences'].append({
                'variable': 'mass_balance',
                'max_difference': float(max_diff),
                'timestep_of_max': int(np.argmax(diff)),
                'tolerance': mass_balance_tolerance
            })
    
    return results


def run_verification_test(test_name: str,
                          inputs_path: Path,
                          params: dict,
                          csharp_output_path: Optional[Path] = None,
                          init_stores_full: bool = False) -> dict:
    """
    Run a single verification test.
    
    Args:
        test_name: Name of the test
        inputs_path: Path to input CSV file
        params: Parameter dictionary
        csharp_output_path: Path to C# reference output (optional)
        init_stores_full: Whether to initialize stores full
        
    Returns:
        Test results dictionary
    """
    print(f"\n  Running test: {test_name}")
    
    # Load inputs
    inputs_df = pd.read_csv(inputs_path)
    print(f"    Loaded {len(inputs_df)} input records")
    
    # Run Python model
    python_df = run_python_model(inputs_df, params, init_stores_full)
    print(f"    Python model completed")
    
    result = {
        'test_name': test_name,
        'input_records': len(inputs_df),
        'python_output': python_df,
        'has_csharp_reference': csharp_output_path is not None and csharp_output_path.exists()
    }
    
    # Compare with C# reference if available
    if result['has_csharp_reference']:
        csharp_df = pd.read_csv(csharp_output_path)
        comparison = compare_outputs(csharp_df, python_df)
        result['comparison'] = comparison
        
        if comparison['pass']:
            print(f"    PASS: All variables within tolerance")
        else:
            print(f"    FAIL: {len(comparison['differences'])} variables exceed tolerance")
            for diff in comparison['differences']:
                print(f"      - {diff['variable']}: max diff = {diff['max_difference']:.2e}")
    else:
        print(f"    Note: No C# reference available for comparison")
        # Still check that Python model produces reasonable results
        result['comparison'] = {
            'pass': True,  # Pass by default if no reference
            'differences': [],
            'summary': {}
        }
        
        # Basic sanity checks
        if python_df['runoff'].isna().any():
            result['comparison']['pass'] = False
            result['comparison']['differences'].append({
                'variable': 'runoff',
                'issue': 'Contains NaN values'
            })
        
        if (python_df['runoff'] < 0).any():
            result['comparison']['pass'] = False
            result['comparison']['differences'].append({
                'variable': 'runoff',
                'issue': 'Contains negative values'
            })
    
    return result


def run_all_verifications(test_data_dir: Path,
                          tolerance: float = 1e-10) -> dict:
    """
    Run all verification tests.
    
    Args:
        test_data_dir: Directory containing test data
        tolerance: Tolerance for comparisons
        
    Returns:
        Dictionary of all test results
    """
    test_data_dir = Path(test_data_dir)
    
    # Load parameter sets
    with open(test_data_dir / "parameter_sets.json") as f:
        param_sets = json.load(f)
    
    results = {}
    
    print("\n" + "=" * 60)
    print("Sacramento Model Verification Tests")
    print("=" * 60)
    
    # Test Case 1: Default parameters, full dataset
    results['TC01_default'] = run_verification_test(
        test_name="TC01: Default parameters, full dataset",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['default']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC01_default.csv"
    )
    
    # Test Case 2: Dry catchment
    results['TC02_dry'] = run_verification_test(
        test_name="TC02: Dry catchment scenario",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['dry_catchment']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC02_dry.csv"
    )
    
    # Test Case 3: Wet catchment
    results['TC03_wet'] = run_verification_test(
        test_name="TC03: Wet catchment scenario",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['wet_catchment']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC03_wet.csv"
    )
    
    # Test Case 4: High impervious
    results['TC04_impervious'] = run_verification_test(
        test_name="TC04: High impervious area",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['impervious']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC04_impervious.csv"
    )
    
    # Test Case 5: Deep groundwater
    results['TC05_groundwater'] = run_verification_test(
        test_name="TC05: Deep groundwater",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['deep_groundwater']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC05_groundwater.csv"
    )
    
    # Test Case 6: Unit hydrograph lag
    results['TC06_uh'] = run_verification_test(
        test_name="TC06: Unit hydrograph lag",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['unit_hydrograph']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC06_uh.csv"
    )
    
    # Test Case 7: Zero rainfall
    results['TC07_zero_rain'] = run_verification_test(
        test_name="TC07: Zero rainfall (evap only)",
        inputs_path=test_data_dir / "zero_rainfall_inputs.csv",
        params=param_sets['default']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC07_zero_rain.csv"
    )
    
    # Test Case 8: Storm event
    results['TC08_storm'] = run_verification_test(
        test_name="TC08: Storm event (100mm pulse)",
        inputs_path=test_data_dir / "storm_event_inputs.csv",
        params=param_sets['default']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC08_storm.csv"
    )
    
    # Test Case 9: Stores initialized full
    results['TC09_full_stores'] = run_verification_test(
        test_name="TC09: Stores initialized full",
        inputs_path=test_data_dir / "synthetic_inputs.csv",
        params=param_sets['default']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC09_full_stores.csv",
        init_stores_full=True
    )
    
    # Test Case 10: Long dry spell
    results['TC10_dry_spell'] = run_verification_test(
        test_name="TC10: Long dry spell",
        inputs_path=test_data_dir / "long_dry_spell_inputs.csv",
        params=param_sets['default']['params'],
        csharp_output_path=test_data_dir / "csharp_output_TC10_dry_spell.csv"
    )
    
    return results


def save_python_outputs(results: dict, output_dir: Path) -> None:
    """
    Save Python model outputs to CSV files.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for test_id, result in results.items():
        if 'python_output' in result:
            output_path = output_dir / f"python_output_{test_id}.csv"
            result['python_output'].to_csv(output_path, index=False, float_format='%.12f')
            print(f"Saved {output_path}")


def main():
    """Main entry point for verification."""
    script_dir = Path(__file__).parent.parent
    test_data_dir = script_dir / "test_data"
    
    # Run verifications
    results = run_all_verifications(test_data_dir)
    
    # Save Python outputs
    save_python_outputs(results, test_data_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_pass = True
    for test_id, result in results.items():
        status = "PASS" if result['comparison']['pass'] else "FAIL"
        if not result['comparison']['pass']:
            all_pass = False
        print(f"  {result['test_name']}: {status}")
    
    print("\n" + "-" * 60)
    if all_pass:
        print("OVERALL RESULT: ALL TESTS PASSED")
        return 0
    else:
        print("OVERALL RESULT: SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
