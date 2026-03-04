"""
Tests for posterior predictive generation and back-transform.

These tests use synthetic InferenceData-like structures to avoid
requiring PyMC. The actual PyMC sampling is tested in test_levels_4_5.py.
"""

import pytest
import numpy as np

from pyrrm.bma.data_prep import apply_transform, back_transform
from pyrrm.bma.prediction import back_transform_predictions
from pyrrm.bma.config import BMAConfig
from pyrrm.bma.tests.fixtures import make_pred_dict


class TestBackTransformPredictions:

    @pytest.mark.parametrize("transform", ["none", "log", "sqrt"])
    def test_roundtrip_all_arrays(self, transform):
        """Back-transform should invert forward transform on all pred arrays."""
        rng = np.random.default_rng(42)
        T = 100
        y_obs = rng.exponential(5, T) + 0.1
        cfg = BMAConfig(transform=transform, log_epsilon=0.01)
        _, y_t, params = apply_transform(
            y_obs[:, None], y_obs, cfg,
        )

        pred_t = make_pred_dict(y_t, noise_scale=0.5, n_samples=50)
        pred_orig = back_transform_predictions(pred_t, params)

        mean_back = back_transform(pred_t["mean"], params)
        np.testing.assert_allclose(pred_orig["mean"], mean_back, rtol=1e-10)

    def test_intervals_roundtrip(self):
        """Interval bounds should also be back-transformed correctly."""
        rng = np.random.default_rng(42)
        T = 50
        y_obs = rng.exponential(5, T) + 0.1
        cfg = BMAConfig(transform="sqrt")
        _, y_t, params = apply_transform(y_obs[:, None], y_obs, cfg)
        pred_t = make_pred_dict(y_t, noise_scale=0.3)
        pred_orig = back_transform_predictions(pred_t, params)

        for level in pred_t["intervals"]:
            lo_t, hi_t = pred_t["intervals"][level]
            lo_expected = back_transform(lo_t, params)
            hi_expected = back_transform(hi_t, params)
            np.testing.assert_allclose(
                pred_orig["intervals"][level][0], lo_expected, rtol=1e-10,
            )
            np.testing.assert_allclose(
                pred_orig["intervals"][level][1], hi_expected, rtol=1e-10,
            )

    def test_samples_shape_preserved(self):
        """Back-transform should preserve (S, T) shape of samples."""
        rng = np.random.default_rng(42)
        T, S = 50, 100
        y_obs = rng.exponential(5, T) + 0.1
        cfg = BMAConfig(transform="log", log_epsilon=0.01)
        _, y_t, params = apply_transform(y_obs[:, None], y_obs, cfg)
        pred_t = make_pred_dict(y_t, n_samples=S)
        pred_orig = back_transform_predictions(pred_t, params)
        assert pred_orig["samples"].shape == (S, T)
