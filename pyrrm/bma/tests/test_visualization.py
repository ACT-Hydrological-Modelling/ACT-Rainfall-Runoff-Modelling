"""
Tests for BMA visualization functions.

Validates that plot functions return matplotlib Figure objects and
don't crash on typical inputs. We don't verify pixel-level correctness.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from pyrrm.bma.visualization import (
    plot_method_comparison,
    plot_pit_histogram,
    plot_prediction_bands,
    plot_weight_comparison,
)
from pyrrm.bma.tests.fixtures import make_pred_dict


class TestPlotWeightComparison:

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        names = ["m1", "m2", "m3"]
        weights_dict = {
            "equal": np.array([1 / 3, 1 / 3, 1 / 3]),
            "grc": np.array([0.5, 0.3, 0.2]),
        }
        fig = plot_weight_comparison(names, weights_dict)
        assert isinstance(fig, Figure)

    def test_handles_single_method(self):
        """Should work with a single method."""
        names = ["m1", "m2"]
        weights_dict = {"equal": np.array([0.5, 0.5])}
        fig = plot_weight_comparison(names, weights_dict)
        assert isinstance(fig, Figure)


class TestPlotPredictionBands:

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        rng = np.random.default_rng(42)
        T = 100
        dates = pd.date_range("2020-01-01", periods=T, freq="D")
        y_obs = rng.exponential(5, T)
        pred = make_pred_dict(y_obs)
        fig = plot_prediction_bands(dates, y_obs, pred)
        assert isinstance(fig, Figure)


class TestPlotPITHistogram:

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 200)
        fig = plot_pit_histogram(pit)
        assert isinstance(fig, Figure)


class TestPlotMethodComparison:

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        df = pd.DataFrame(
            {"NSE": [0.8, 0.85, 0.9], "KGE": [0.7, 0.75, 0.8]},
            index=["equal", "grc", "stacking"],
        )
        fig = plot_method_comparison(df)
        assert isinstance(fig, Figure)

    def test_subset_metrics(self):
        """Should work with a subset of metrics."""
        df = pd.DataFrame(
            {"NSE": [0.8, 0.9], "KGE": [0.7, 0.8], "RMSE": [5.0, 3.0]},
            index=["equal", "grc"],
        )
        fig = plot_method_comparison(df, metrics=["NSE", "KGE"])
        assert isinstance(fig, Figure)
