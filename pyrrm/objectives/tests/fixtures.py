"""
Test fixtures for pyrrm.objectives module.

Provides synthetic data generators for testing objective functions.
"""

from typing import Tuple
import numpy as np


def generate_test_data(n: int = 365, 
                       seed: int = 42,
                       mean_flow: float = 50.0,
                       std_log: float = 1.0,
                       noise_std: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic observed and simulated flow data.
    
    Creates lognormal "observed" data and adds multiplicative noise
    to create "simulated" data.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points (days)
    seed : int, default=42
        Random seed for reproducibility
    mean_flow : float, default=50.0
        Approximate mean flow value
    std_log : float, default=1.0
        Standard deviation in log space
    noise_std : float, default=0.15
        Relative noise standard deviation for simulated data
    
    Returns
    -------
    obs : np.ndarray
        Observed flow values (lognormally distributed)
    sim : np.ndarray
        Simulated flow values (obs * multiplicative noise)
    
    Examples
    --------
    >>> obs, sim = generate_test_data(n=365)
    >>> len(obs)
    365
    >>> np.corrcoef(obs, sim)[0, 1] > 0.9  # High correlation
    True
    """
    np.random.seed(seed)
    
    # Generate lognormal "observed" data
    # Mean in log space to achieve target mean_flow
    mu_log = np.log(mean_flow) - 0.5 * std_log**2
    obs = np.random.lognormal(mean=mu_log, sigma=std_log, size=n)
    
    # Generate simulated data with multiplicative noise
    noise = np.random.normal(1.0, noise_std, size=n)
    sim = obs * noise
    
    # Ensure positive values
    sim = np.maximum(sim, 0.001)
    
    return obs, sim


def generate_perfect_data(n: int = 365, 
                          seed: int = 42,
                          mean_flow: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate identical observed and simulated data for testing optimal values.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points
    seed : int, default=42
        Random seed
    mean_flow : float, default=50.0
        Approximate mean flow
    
    Returns
    -------
    obs : np.ndarray
        Observed flow values
    sim : np.ndarray
        Identical simulated values (sim == obs)
    
    Examples
    --------
    >>> obs, sim = generate_perfect_data()
    >>> np.allclose(obs, sim)
    True
    """
    np.random.seed(seed)
    
    # Generate lognormal data
    mu_log = np.log(mean_flow) - 0.5
    obs = np.random.lognormal(mean=mu_log, sigma=1.0, size=n)
    
    # Perfect simulation
    sim = obs.copy()
    
    return obs, sim


def generate_with_nan(n: int = 365, 
                      nan_fraction: float = 0.1,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with random NaN values.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points
    nan_fraction : float, default=0.1
        Fraction of values to set as NaN (0 to 1)
    seed : int, default=42
        Random seed
    
    Returns
    -------
    obs : np.ndarray
        Observed values with NaNs
    sim : np.ndarray
        Simulated values with NaNs (different positions from obs)
    
    Examples
    --------
    >>> obs, sim = generate_with_nan(n=100, nan_fraction=0.1)
    >>> np.sum(np.isnan(obs)) > 0
    True
    """
    obs, sim = generate_test_data(n=n, seed=seed)
    
    np.random.seed(seed + 1)
    
    # Insert NaN values at random positions
    n_nan_obs = int(n * nan_fraction)
    n_nan_sim = int(n * nan_fraction)
    
    nan_idx_obs = np.random.choice(n, n_nan_obs, replace=False)
    nan_idx_sim = np.random.choice(n, n_nan_sim, replace=False)
    
    obs[nan_idx_obs] = np.nan
    sim[nan_idx_sim] = np.nan
    
    return obs, sim


def generate_biased_data(n: int = 365, 
                          bias_factor: float = 1.2,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with known bias.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points
    bias_factor : float, default=1.2
        Multiplicative bias (1.2 = 20% overestimation)
    seed : int, default=42
        Random seed
    
    Returns
    -------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values with known bias
    
    Examples
    --------
    >>> obs, sim = generate_biased_data(bias_factor=1.2)
    >>> np.mean(sim) / np.mean(obs)  # Approximately 1.2
    """
    np.random.seed(seed)
    
    # Generate lognormal data
    obs = np.random.lognormal(mean=np.log(50), sigma=1.0, size=n)
    
    # Apply constant bias
    sim = obs * bias_factor
    
    return obs, sim


def generate_timing_error_data(n: int = 365,
                                 shift: int = 1,
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with timing (phase) error.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points
    shift : int, default=1
        Number of timesteps to shift (positive = delayed simulation)
    seed : int, default=42
        Random seed
    
    Returns
    -------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values shifted in time
    """
    np.random.seed(seed)
    
    obs = np.random.lognormal(mean=np.log(50), sigma=1.0, size=n)
    
    # Shift simulation
    sim = np.roll(obs, shift)
    
    return obs, sim


def generate_variable_data(n: int = 365,
                            var_ratio: float = 0.8,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with different variability.
    
    Parameters
    ----------
    n : int, default=365
        Number of data points
    var_ratio : float, default=0.8
        Ratio of simulated to observed standard deviation
    seed : int, default=42
        Random seed
    
    Returns
    -------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values with different variability
    """
    np.random.seed(seed)
    
    obs = np.random.lognormal(mean=np.log(50), sigma=1.0, size=n)
    
    # Scale variability while preserving mean
    mean_obs = np.mean(obs)
    sim = mean_obs + var_ratio * (obs - mean_obs)
    
    # Ensure positive
    sim = np.maximum(sim, 0.001)
    
    return obs, sim
