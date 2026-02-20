#!/usr/bin/env python
"""
Parallel Calibration Example

This script demonstrates how to run PyDREAM (MT-DREAM(ZS)) calibration
with built-in multiprocessing for speedup on multi-core systems.

PyDREAM automatically parallelises chains via its internal DreamPool.
Each chain runs in a separate process, and multi-try proposals can also
be evaluated in parallel.

Usage:
    python parallel_calibration_example.py

Performance Notes:
    - Set n_chains to match your available CPU cores
    - When multitry > 1 and parallel=True, proposal evaluations
      are also parallelised across the multi-try workers
    - For I/O-bound models, increasing n_chains beyond CPU count
      may still help due to interleaved waiting

Author: pyrrm development team
"""

import numpy as np
import pandas as pd


def main():
    """Run calibration example."""
    from pyrrm.models import Sacramento
    from pyrrm.calibration import CalibrationRunner
    from pyrrm.calibration.objective_functions import NSE

    print("=" * 60)
    print("PARALLEL CALIBRATION EXAMPLE (PyDREAM)")
    print("=" * 60)

    # =========================================================================
    # DATA SETUP
    # =========================================================================

    np.random.seed(42)
    n_days = 1000

    dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
    rainfall = np.random.exponential(5, n_days)
    rainfall[rainfall < 2] = 0
    pet = 3 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365)

    inputs = pd.DataFrame({
        'rainfall': rainfall,
        'pet': pet
    }, index=dates)

    true_params = {
        'uztwm': 50.0, 'uzfwm': 40.0, 'lztwm': 130.0,
        'lzfpm': 60.0, 'lzfsm': 25.0, 'uzk': 0.3,
        'lzpk': 0.01, 'lzsk': 0.05, 'zperc': 40.0,
        'rexp': 1.5, 'pctim': 0.01, 'adimp': 0.0,
        'pfree': 0.06, 'rserv': 0.3, 'side': 0.0,
        'ssout': 0.0, 'sarva': 0.0
    }

    true_model = Sacramento()
    true_model.set_parameters(true_params)
    true_results = true_model.run(inputs)
    observed = true_results['flow'].values + np.random.normal(0, 0.5, n_days)
    observed = np.maximum(observed, 0)

    # =========================================================================
    # CALIBRATION SETUP
    # =========================================================================

    model = Sacramento()

    param_bounds = {
        'uztwm': (25.0, 125.0),
        'uzfwm': (10.0, 75.0),
        'lztwm': (75.0, 300.0),
        'uzk': (0.1, 0.5),
        'lzpk': (0.001, 0.025),
        'lzsk': (0.01, 0.25),
    }

    runner = CalibrationRunner(
        model=model,
        inputs=inputs,
        observed=observed,
        objective=NSE(),
        parameter_bounds=param_bounds,
        warmup_period=100
    )

    # =========================================================================
    # RUN CALIBRATION
    # =========================================================================

    import os
    n_chains = max(3, os.cpu_count() - 1)

    print(f"Number of chains: {n_chains}")
    print(f"Multi-try proposals: 5")
    print(f"Iterations: 2000")
    print()

    result = runner.run_dream(
        n_iterations=2000,
        n_chains=n_chains,
        multitry=5,
        snooker=0.1,
        dbname='pydream_parallel_example',
    )

    # =========================================================================
    # RESULTS
    # =========================================================================

    print()
    print(result.summary())

    print("\nParameter Recovery:")
    print("-" * 50)
    print(f"{'Parameter':15s} {'True':>10s} {'Calibrated':>12s} {'Error %':>10s}")
    print("-" * 50)
    for param in param_bounds.keys():
        true_val = true_params.get(param, np.nan)
        cal_val = result.best_parameters.get(param, np.nan)
        if not np.isnan(true_val) and not np.isnan(cal_val):
            error_pct = 100 * (cal_val - true_val) / true_val
            print(f"{param:15s} {true_val:10.4f} {cal_val:12.4f} {error_pct:10.1f}%")

    print()
    print("Results saved to: pydream_parallel_example.csv")
    print("Use the calibration_monitor notebook to visualize progress")


if __name__ == '__main__':
    main()
