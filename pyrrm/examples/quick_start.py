"""
Quick Start Example for pyrrm Library

This script demonstrates basic usage of the pyrrm library for
rainfall-runoff modeling.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import pyrrm models
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
from pyrrm.calibration import CalibrationRunner
from pyrrm.calibration.objective_functions import NSE, KGE


def generate_synthetic_data(n_days: int = 365 * 3) -> pd.DataFrame:
    """Generate synthetic input data for demonstration."""
    np.random.seed(42)
    
    # Create date index
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate rainfall (seasonal pattern with random events)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    
    # Random rain events
    rain_prob = 0.3 + 0.2 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    rain_events = np.random.random(n_days) < rain_prob
    rain_amount = np.random.exponential(scale=seasonal, size=n_days)
    rainfall = np.where(rain_events, rain_amount, 0)
    
    # Generate PET (seasonal pattern)
    pet = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    pet = np.maximum(pet, 0.5)
    
    return pd.DataFrame({
        'precipitation': rainfall,
        'pet': pet
    }, index=dates)


def example_sacramento():
    """Example: Running the Sacramento model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Sacramento Model")
    print("=" * 60)
    
    # Generate data
    inputs = generate_synthetic_data()
    print(f"Generated {len(inputs)} days of input data")
    
    # Create and configure model
    model = Sacramento()
    model.set_parameters({
        'uztwm': 60,
        'lztwm': 150,
        'uzk': 0.35,
        'lzpk': 0.01
    })
    
    print("\nRunning Sacramento model...")
    results = model.run(inputs)
    
    print(f"\nResults:")
    print(f"  Total Precipitation: {inputs['precipitation'].sum():.1f} mm")
    print(f"  Total Runoff:        {results['runoff'].sum():.1f} mm")
    print(f"  Mean Daily Runoff:   {results['runoff'].mean():.2f} mm/d")
    print(f"  Max Daily Runoff:    {results['runoff'].max():.2f} mm/d")
    
    return model, inputs, results


def example_gr4j():
    """Example: Running the GR4J model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: GR4J Model")
    print("=" * 60)
    
    # Generate data
    inputs = generate_synthetic_data()
    
    # Create model with custom parameters
    model = GR4J({
        'X1': 350,  # Production store capacity (mm)
        'X2': 0,    # Groundwater exchange (mm/d)
        'X3': 90,   # Routing store capacity (mm)
        'X4': 1.7   # Unit hydrograph time (d)
    })
    
    print(f"Model: {model.name}")
    print(f"Parameters: {model.get_parameters()}")
    
    print("\nRunning GR4J model...")
    results = model.run(inputs)
    
    print(f"\nResults:")
    print(f"  Mean Flow: {results['flow'].mean():.2f} mm/d")
    print(f"  Max Flow:  {results['flow'].max():.2f} mm/d")
    
    return model, inputs, results


def example_comparison():
    """Example: Comparing multiple models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Comparison")
    print("=" * 60)
    
    inputs = generate_synthetic_data()
    
    # Create multiple models with default parameters
    models = {
        'Sacramento': Sacramento(),
        'GR4J': GR4J(),
        'GR5J': GR5J(),
        'GR6J': GR6J(),
    }
    
    print("Running all models...")
    results = {}
    for name, model in models.items():
        model.reset()
        results[name] = model.run(inputs)
        print(f"  {name}: Mean flow = {results[name]['flow'].mean():.2f} mm/d")
    
    return models, inputs, results


def example_single_timestep():
    """Example: Running model timestep by timestep."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Timestep-by-Timestep Simulation")
    print("=" * 60)
    
    model = GR4J()
    model.reset()
    
    # Run for 10 days
    rain = [10, 5, 0, 0, 15, 8, 2, 0, 0, 0]
    pet = [3, 3, 4, 4, 3, 3, 4, 4, 4, 5]
    
    print("Day  Precip  PET    Flow")
    print("-" * 35)
    
    flows = []
    for day, (p, e) in enumerate(zip(rain, pet), 1):
        output = model.run_timestep(precipitation=p, pet=e)
        flows.append(output['flow'])
        print(f"{day:3d}   {p:5.1f}  {e:4.1f}  {output['flow']:6.3f}")
    
    print(f"\nTotal runoff: {sum(flows):.2f} mm")
    
    return model, flows


def example_state_management():
    """Example: Saving and restoring model state."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: State Management")
    print("=" * 60)
    
    model = GR4J()
    inputs = generate_synthetic_data(365)
    
    # Run for first half of year
    first_half = inputs.iloc[:180]
    model.reset()
    results1 = model.run(first_half)
    
    # Save state
    saved_state = model.get_state()
    print(f"Saved state after 180 days")
    print(f"Production store: {saved_state.values['production_store']:.3f}")
    print(f"Routing store: {saved_state.values['routing_store']:.3f}")
    
    # Run second half
    second_half = inputs.iloc[180:]
    results2 = model.run(second_half)
    
    # Reset and restore
    model.reset()
    model.set_state(saved_state)
    results2_check = model.run(second_half)
    
    # Verify
    assert np.allclose(results2['flow'].values, results2_check['flow'].values)
    print("\nState restoration verified!")
    
    return model, saved_state


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("  pyrrm Library - Quick Start Examples")
    print("#" * 60)
    
    example_sacramento()
    example_gr4j()
    example_comparison()
    example_single_timestep()
    example_state_management()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
