#!/usr/bin/env python
"""
MPI Parallel Calibration Example

This script demonstrates how to run SpotPy-DREAM calibration using MPI
parallelization for significant speedup on multi-core systems or HPC clusters.

Requirements:
    - mpi4py: pip install mpi4py
    - An MPI implementation (OpenMPI, MPICH, etc.)

Usage:
    Sequential (single core):
        python mpi_calibration_example.py
    
    MPI Parallel (e.g., 11 processes = 1 master + 10 workers):
        mpirun -n 11 python mpi_calibration_example.py
    
    On HPC clusters (example with SLURM):
        srun -n 21 python mpi_calibration_example.py

Performance Notes:
    - Speedup is approximately linear with number of chains
    - Use n_processes = n_chains + 1 (1 master + n_chains workers)
    - For best performance, match n_chains to available CPU cores - 1
    - MPI overhead is minimal for model runs > 0.1 seconds

Author: pyrrm development team
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Check if running under MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    rank = 0
    size = 1
    MPI_AVAILABLE = False


def main():
    """Run calibration example."""
    
    # Only master process prints info
    if rank == 0:
        print("=" * 60)
        print("MPI PARALLEL CALIBRATION EXAMPLE")
        print("=" * 60)
        if MPI_AVAILABLE:
            print(f"Running with MPI: {size} processes")
            print(f"  Master process: rank 0")
            print(f"  Worker processes: ranks 1-{size-1}")
        else:
            print("Running without MPI (sequential mode)")
            print("Install mpi4py and run with mpirun for parallel execution")
        print()
    
    # Import pyrrm (after MPI check to avoid import errors on workers)
    from pyrrm.models import Sacramento
    from pyrrm.calibration import CalibrationRunner
    from pyrrm.calibration.objective_functions import NSE
    
    # =========================================================================
    # DATA SETUP (only needed on master, but we do it everywhere for simplicity)
    # =========================================================================
    
    # For this example, we'll create synthetic data
    # In practice, you would load your observed data here
    np.random.seed(42)
    n_days = 1000
    
    # Generate synthetic inputs
    dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
    rainfall = np.random.exponential(5, n_days)  # mm/day
    rainfall[rainfall < 2] = 0  # Many dry days
    pet = 3 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Seasonal PET
    
    inputs = pd.DataFrame({
        'rainfall': rainfall,
        'pet': pet
    }, index=dates)
    
    # Generate "observed" data using Sacramento with known parameters
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
    observed = true_results['flow'].values + np.random.normal(0, 0.5, n_days)  # Add noise
    observed = np.maximum(observed, 0)  # No negative flows
    
    # =========================================================================
    # CALIBRATION SETUP
    # =========================================================================
    
    # Create model instance for calibration
    model = Sacramento()
    
    # Define parameter bounds (subset for faster example)
    param_bounds = {
        'uztwm': (25.0, 125.0),
        'uzfwm': (10.0, 75.0),
        'lztwm': (75.0, 300.0),
        'uzk': (0.1, 0.5),
        'lzpk': (0.001, 0.025),
        'lzsk': (0.01, 0.25),
    }
    
    # Create calibration runner
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
    
    # Determine parallelization mode
    if MPI_AVAILABLE and size > 1:
        parallel_mode = 'mpi'
        n_chains = size - 1  # Workers = total processes - master
    else:
        parallel_mode = 'seq'
        n_chains = 5
    
    if rank == 0:
        print(f"Parallel mode: {parallel_mode}")
        print(f"Number of chains: {n_chains}")
        print(f"Iterations: 2000")
        print()
    
    # Run DREAM calibration
    result = runner.run_spotpy_dream(
        n_iterations=2000,
        n_chains=n_chains,
        convergence_threshold=1.2,
        dbname='mpi_dream_example',
        dbformat='csv',
        parallel=parallel_mode,
        # DREAM algorithm parameters
        nCr=3,
        delta=3,
        c=0.1,
        eps=1e-6
    )
    
    # =========================================================================
    # RESULTS (only on master process)
    # =========================================================================
    
    if rank == 0 and result is not None:
        print()
        print(result.summary())
        
        # Compare with true parameters
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
        print("Results saved to: mpi_dream_example.csv")
        print("Use the calibration_monitor notebook to visualize progress")


if __name__ == '__main__':
    main()
