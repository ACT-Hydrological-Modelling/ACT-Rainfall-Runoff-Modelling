"""
Shared fixtures for BMA test suite.

All synthetic data uses fixed seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrrm.bma.config import BMAConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_5yr_daily() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """5 years of daily data: 4 model predictions + observed flow.

    Returns (dates, F, y_obs) where F is (T, 4) and y_obs is (T,).
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-07-01", "2015-06-30", freq="D")
    T = len(dates)
    K = 4

    base = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(T) / 365.25)
    noise = rng.exponential(2.0, size=T)
    y_obs = np.maximum(base + noise, 0.1)

    F = np.column_stack([
        y_obs + rng.normal(0, 1.5, T),
        y_obs * 1.05 + rng.normal(0, 2.0, T),
        y_obs * 0.95 + rng.normal(0, 1.0, T),
        y_obs + rng.normal(3, 5.0, T),  # biased model
    ])
    F = np.maximum(F, 0.0)
    return dates, F, y_obs


@pytest.fixture
def synthetic_10yr_daily() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """10 years of daily data for CV split testing."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2005-01-01", "2014-12-31", freq="D")
    T = len(dates)
    y_obs = 5.0 + rng.exponential(3.0, T)
    F = np.column_stack([y_obs + rng.normal(0, 1, T) for _ in range(3)])
    F = np.maximum(F, 0.0)
    return dates, F, y_obs


@pytest.fixture
def default_config() -> BMAConfig:
    return BMAConfig()


@pytest.fixture
def small_config() -> BMAConfig:
    """Config with relaxed thresholds for testing with noisy synthetic data."""
    return BMAConfig(
        nse_threshold=-1.0,
        kge_threshold=-2.0,
        pbias_threshold=100.0,
        residual_corr_threshold=0.99,
        buffer_days=0,
        min_train_years=0.5,
        min_val_years=0.2,
        require_seasonal_coverage=False,
        random_seed=42,
    )


def make_pred_dict(
    y_obs: np.ndarray, noise_scale: float = 2.0, n_samples: int = 200,
) -> dict:
    """Create a synthetic prediction dict matching BMA output format."""
    rng = np.random.default_rng(42)
    T = len(y_obs)
    samples = y_obs[None, :] + rng.normal(0, noise_scale, (n_samples, T))
    return {
        "samples": samples,
        "mean": samples.mean(axis=0),
        "median": np.median(samples, axis=0),
        "intervals": {
            0.90: (
                np.quantile(samples, 0.05, axis=0),
                np.quantile(samples, 0.95, axis=0),
            ),
            0.50: (
                np.quantile(samples, 0.25, axis=0),
                np.quantile(samples, 0.75, axis=0),
            ),
        },
    }
