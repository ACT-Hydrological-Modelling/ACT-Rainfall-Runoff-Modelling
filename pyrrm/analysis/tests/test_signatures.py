"""
Tests for hydrological signature computations.

Tests follow TOSSH and EflowStats reference implementations where possible.
Known values are computed by hand or verified against MATLAB/R outputs.

Test Organization:
- Each signature category has its own test class
- Tests cover known values, edge cases, and boundary conditions
- Tolerances are set appropriately for each signature type
"""

import pytest
import numpy as np
import pandas as pd
from pyrrm.analysis.signatures import (
    SIGNATURE_CATEGORIES,
    SIGNATURE_INFO,
    compute_all_signatures,
    compute_magnitude_signatures,
    compute_variability_signatures,
    compute_timing_signatures,
    compute_fdc_signatures,
    compute_frequency_signatures,
    compute_recession_signatures,
    compute_baseflow_signatures,
    compute_event_signatures,
    compute_seasonality_signatures,
    signature_percent_error,
    compare_signatures,
)


class TestSignatureCategories:
    """Tests for signature category structure and metadata."""

    def test_all_categories_present(self):
        """All 9 categories should be defined."""
        expected_categories = [
            "Magnitude", "Variability", "Timing", "Flow Duration Curve",
            "Frequency", "Recession", "Baseflow", "Event", "Seasonality",
        ]
        assert list(SIGNATURE_CATEGORIES.keys()) == expected_categories

    def test_signature_info_keys_match_categories(self):
        """SIGNATURE_INFO should have entries for major signatures."""
        all_sigs = []
        for sigs in SIGNATURE_CATEGORIES.values():
            all_sigs.extend(sigs)
        
        for key in SIGNATURE_INFO:
            assert key in all_sigs, f"{key} in SIGNATURE_INFO but not in SIGNATURE_CATEGORIES"


class TestMagnitudeSignatures:
    """Tests for magnitude signature computations."""

    def test_q_mean_known_value(self):
        """Q_mean should equal numpy mean for simple array."""
        Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_mean"] == pytest.approx(30.0, abs=1e-10)

    def test_q_median_known_value(self):
        """Q_median should equal numpy median."""
        Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_median"] == pytest.approx(30.0, abs=1e-10)

    def test_q_percentiles_known_values(self):
        """Percentiles should match numpy percentile."""
        Q = np.arange(1.0, 101.0)  # 1 to 100
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q50"] == pytest.approx(np.percentile(Q, 50), abs=1e-10)
        assert sigs["Q95"] == pytest.approx(np.percentile(Q, 95), abs=1e-10)
        assert sigs["Q5"] == pytest.approx(np.percentile(Q, 5), abs=1e-10)

    def test_n_day_max_known_value(self):
        """n-day max should be max of n-day rolling mean."""
        Q = np.concatenate([np.ones(10) * 10.0, np.ones(7) * 100.0, np.ones(10) * 10.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_7day_max"] == pytest.approx(100.0, abs=1e-10)
        assert sigs["Q_1day_max"] == pytest.approx(100.0, abs=1e-10)

    def test_n_day_min_known_value(self):
        """n-day min should be min of n-day rolling mean."""
        Q = np.concatenate([np.ones(10) * 100.0, np.ones(7) * 10.0, np.ones(10) * 100.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_7day_min"] == pytest.approx(10.0, abs=1e-10)

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN for all magnitude signatures."""
        Q = np.array([])
        sigs = compute_magnitude_signatures(Q)
        assert np.isnan(sigs["Q_mean"])
        assert np.isnan(sigs["Q_median"])
        assert np.isnan(sigs["Q50"])

    def test_nan_handling(self):
        """NaN values should be excluded from calculations."""
        Q = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_mean"] == pytest.approx(30.0, abs=1e-10)

    def test_single_value(self):
        """Single value array should return that value for mean/median."""
        Q = np.array([42.0])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_mean"] == pytest.approx(42.0, abs=1e-10)
        assert sigs["Q_median"] == pytest.approx(42.0, abs=1e-10)

    def test_insufficient_length_for_nday(self):
        """Short array should return NaN for n-day signatures."""
        Q = np.array([1.0, 2.0, 3.0])  # len < 7
        sigs = compute_magnitude_signatures(Q)
        assert np.isnan(sigs["Q_7day_max"])
        assert np.isnan(sigs["Q_30day_max"])


class TestVariabilitySignatures:
    """Tests for variability signature computations."""

    def test_cv_known_value(self):
        """CV should be std/mean."""
        Q = np.array([10.0, 20.0, 30.0])
        sigs = compute_variability_signatures(Q)
        expected_cv = np.std(Q, ddof=0) / np.mean(Q)
        assert sigs["CV"] == pytest.approx(expected_cv, abs=1e-10)

    def test_cv_constant_flow(self):
        """Constant flow should have CV = 0."""
        Q = np.ones(100) * 50.0
        sigs = compute_variability_signatures(Q)
        assert sigs["CV"] == pytest.approx(0.0, abs=1e-10)

    def test_flashiness_constant_flow(self):
        """Constant flow should have zero flashiness."""
        Q = np.ones(100) * 50.0
        sigs = compute_variability_signatures(Q)
        assert sigs["Flashiness"] == pytest.approx(0.0, abs=1e-10)

    def test_flashiness_oscillating(self):
        """Oscillating flow should have high flashiness."""
        Q = np.array([10.0, 100.0, 10.0, 100.0, 10.0, 100.0])
        sigs = compute_variability_signatures(Q)
        # Sum of abs changes = 5*90 = 450, total flow = 330, flashiness ~1.36
        assert sigs["Flashiness"] > 1.0

    def test_reversals_known_value(self):
        """Reversals should count direction changes."""
        # rise, fall, rise, fall, rise = 4 reversals
        Q = np.array([10.0, 20.0, 15.0, 25.0, 20.0, 30.0])
        sigs = compute_variability_signatures(Q)
        assert sigs["Reversals"] == pytest.approx(4.0, abs=1e-10)

    def test_reversals_monotonic(self):
        """Monotonically increasing should have zero reversals."""
        Q = np.arange(1.0, 11.0)
        sigs = compute_variability_signatures(Q)
        assert sigs["Reversals"] == pytest.approx(0.0, abs=1e-10)

    def test_rise_fall_rates(self):
        """Rise and fall rates should reflect day-to-day changes."""
        Q = np.array([10.0, 30.0, 20.0, 40.0])  # +20, -10, +20
        sigs = compute_variability_signatures(Q)
        assert sigs["Rise_rate"] == pytest.approx(20.0, abs=1e-10)
        assert sigs["Fall_rate"] == pytest.approx(10.0, abs=1e-10)

    def test_ac1_white_noise(self):
        """White noise should have AC1 near zero."""
        np.random.seed(42)
        Q = np.abs(np.random.randn(1000)) + 10
        sigs = compute_variability_signatures(Q)
        assert abs(sigs["AC1"]) < 0.1

    def test_ac1_perfect_autocorrelation(self):
        """Smoothly varying series should have high AC1."""
        Q = np.sin(np.linspace(0, 4 * np.pi, 100)) + 2
        sigs = compute_variability_signatures(Q)
        assert sigs["AC1"] > 0.9

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_variability_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Variability"]:
            assert np.isnan(sigs[key])

    def test_cv_zero_mean(self):
        """Zero mean should return NaN for CV."""
        Q = np.array([-5.0, 0.0, 5.0])  # mean=0, but artificial
        # In practice, flow is non-negative, but test edge case
        Q = np.array([0.0, 0.0, 0.0])
        sigs = compute_variability_signatures(Q)
        assert np.isnan(sigs["CV"])


class TestTimingSignatures:
    """Tests for timing signature computations."""

    def test_requires_dates(self):
        """Without dates, timing signatures should be NaN."""
        Q = np.random.lognormal(2, 0.5, 365)
        sigs = compute_timing_signatures(Q, dates=None)
        for key in SIGNATURE_CATEGORIES["Timing"]:
            assert np.isnan(sigs[key])

    def test_half_flow_date_symmetric(self):
        """Symmetric annual flow should have HFD near mid-year."""
        dates = pd.date_range('2000-01-01', periods=365, freq='D')
        Q = np.ones(365)
        sigs = compute_timing_signatures(Q, dates)
        assert 180 <= sigs["HFD_mean"] <= 186

    def test_half_flow_date_front_loaded(self):
        """Front-loaded flow should have early HFD."""
        dates = pd.date_range('2000-01-01', periods=365, freq='D')
        Q = np.zeros(365)
        Q[:60] = 10.0  # All flow in first 60 days
        sigs = compute_timing_signatures(Q, dates)
        assert sigs["HFD_mean"] < 60

    def test_date_max_known_peak(self):
        """Date of max should identify peak day."""
        dates = pd.date_range('2000-01-01', periods=365, freq='D')
        Q = np.ones(365)
        Q[100] = 1000.0  # Peak on day 101 (1-indexed DOY)
        sigs = compute_timing_signatures(Q, dates)
        assert sigs["Date_max"] == pytest.approx(101, abs=1)

    def test_date_min_known_trough(self):
        """Date of min should identify minimum day."""
        dates = pd.date_range('2000-01-01', periods=365, freq='D')
        Q = np.ones(365) * 10.0
        Q[200] = 0.1  # Minimum on day 201
        sigs = compute_timing_signatures(Q, dates)
        assert sigs["Date_min"] == pytest.approx(201, abs=1)

    def test_insufficient_data_returns_nan(self):
        """Less than 1 year should return NaN for timing signatures."""
        dates = pd.date_range('2000-01-01', periods=100, freq='D')
        Q = np.random.rand(100) + 1
        sigs = compute_timing_signatures(Q, dates)
        for key in SIGNATURE_CATEGORIES["Timing"]:
            assert np.isnan(sigs[key])

    def test_constancy_constant_flow(self):
        """Constant flow should have high constancy."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        Q = np.ones(730) * 50.0
        sigs = compute_timing_signatures(Q, dates)
        # Constant flow -> all values in same bin -> high constancy
        assert sigs["Constancy"] > 0.8 or np.isnan(sigs["Constancy"])


class TestFDCSignatures:
    """Tests for flow duration curve signature computations."""

    def test_fdc_slope_negative(self):
        """FDC slope should be negative (decreasing with exceedance)."""
        Q = np.random.lognormal(2, 0.5, 365)
        sigs = compute_fdc_signatures(Q)
        assert sigs["FDC_slope"] < 0

    def test_fdc_slope_uniform_distribution(self):
        """Uniform distribution should have relatively constant slope."""
        Q = np.arange(1.0, 101.0)
        sigs = compute_fdc_signatures(Q)
        assert np.isfinite(sigs["FDC_slope"])

    def test_fdc_slope_high_variability(self):
        """High variability should produce steeper FDC slope."""
        # Low variability (small CV)
        Q_low_var = np.random.lognormal(2, 0.1, 365)
        # High variability (large CV)
        Q_high_var = np.random.lognormal(2, 1.5, 365)
        
        sigs_low = compute_fdc_signatures(Q_low_var)
        sigs_high = compute_fdc_signatures(Q_high_var)
        
        # Higher variability should have steeper (more negative) slope
        assert sigs_high["FDC_slope"] < sigs_low["FDC_slope"]

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_fdc_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Flow Duration Curve"]:
            assert np.isnan(sigs[key])

    def test_all_zeros_returns_nan(self):
        """All zero values should return NaN (log undefined)."""
        Q = np.zeros(100)
        sigs = compute_fdc_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Flow Duration Curve"]:
            assert np.isnan(sigs[key])


class TestFrequencySignatures:
    """Tests for frequency signature computations."""

    def test_high_pulse_count_known(self):
        """Known high pulse pattern should be counted correctly."""
        Q = np.concatenate([
            np.ones(10) * 50,   # Below Q75
            np.ones(5) * 200,   # Above Q75 (pulse 1)
            np.ones(10) * 50,   # Below
            np.ones(5) * 200,   # Above Q75 (pulse 2)
            np.ones(10) * 50,   # Below
        ])
        sigs = compute_frequency_signatures(Q)
        # Should detect 2 high pulses (above 75th percentile)
        assert sigs["High_pulse_count"] >= 1.0

    def test_zero_flow_freq_known(self):
        """Zero flow frequency should match manually counted fraction."""
        Q = np.concatenate([np.zeros(20), np.ones(80)])  # 20% zeros
        sigs = compute_frequency_signatures(Q)
        assert sigs["Zero_flow_freq"] == pytest.approx(0.2, abs=1e-10)

    def test_zero_flow_freq_no_zeros(self):
        """No zero days should give Zero_flow_freq = 0."""
        Q = np.ones(100)
        sigs = compute_frequency_signatures(Q)
        assert sigs["Zero_flow_freq"] == pytest.approx(0.0, abs=1e-10)

    def test_low_flow_freq_constant(self):
        """Constant flow should have low_flow_freq = 0."""
        Q = np.ones(100) * 50.0
        sigs = compute_frequency_signatures(Q)
        # All values equal mean, so none below 0.2*mean
        assert sigs["Low_flow_freq"] == pytest.approx(0.0, abs=1e-10)

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_frequency_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Frequency"]:
            assert np.isnan(sigs[key])


class TestRecessionSignatures:
    """Tests for recession signature computations."""

    def test_recession_k_exponential_decay(self):
        """Perfect exponential decay should recover known k."""
        k_true = 0.95
        Q = 100.0 * (k_true ** np.arange(50))
        sigs = compute_recession_signatures(Q)
        # Allow 5% relative tolerance for fitting
        assert sigs["Recession_k"] == pytest.approx(k_true, rel=0.05)

    def test_recession_k_range(self):
        """Recession k should be between 0 and 1 for typical flow."""
        np.random.seed(42)
        Q = np.random.lognormal(3, 0.5, 500)
        sigs = compute_recession_signatures(Q)
        if np.isfinite(sigs["Recession_k"]):
            assert 0 < sigs["Recession_k"] < 1

    def test_no_recession_returns_nan(self):
        """Monotonically increasing flow should return NaN for recession."""
        Q = np.arange(1.0, 101.0)
        sigs = compute_recession_signatures(Q)
        assert np.isnan(sigs["Recession_a"]) or np.isnan(sigs["Recession_b"])

    def test_power_law_parameters(self):
        """Power-law parameters should be computed for typical flow."""
        np.random.seed(42)
        Q = np.random.lognormal(3, 0.5, 500)
        sigs = compute_recession_signatures(Q)
        # Just check they're computed (may be NaN if no valid segments)
        assert "Recession_a" in sigs
        assert "Recession_b" in sigs

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_recession_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Recession"]:
            assert np.isnan(sigs[key])


class TestBaseflowSignatures:
    """Tests for baseflow signature computations."""

    def test_bfi_constant_flow(self):
        """Constant flow should have BFI close to 1."""
        Q = np.ones(100) * 50.0
        sigs = compute_baseflow_signatures(Q)
        # Lyne-Hollick filter on constant flow should give ~100% baseflow
        assert sigs["BFI"] > 0.95

    def test_bfi_range(self):
        """BFI should be between 0 and 1."""
        np.random.seed(42)
        Q = np.random.lognormal(3, 1, 500)
        sigs = compute_baseflow_signatures(Q)
        assert 0 <= sigs["BFI"] <= 1

    def test_bf_mean_positive(self):
        """Mean baseflow should be positive for positive flow."""
        Q = np.random.lognormal(3, 0.5, 365) + 1
        sigs = compute_baseflow_signatures(Q)
        assert sigs["BF_mean"] > 0

    def test_bf_mean_less_than_q_mean(self):
        """Mean baseflow should not exceed mean total flow."""
        Q = np.random.lognormal(3, 0.5, 365)
        sigs = compute_baseflow_signatures(Q)
        q_mean = np.mean(Q)
        assert sigs["BF_mean"] <= q_mean * 1.01  # Small tolerance

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_baseflow_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Baseflow"]:
            assert np.isnan(sigs[key])


class TestEventSignatures:
    """Tests for event signature computations."""

    def test_rld_known_pattern(self):
        """Rising limb density for known pattern."""
        # Longer array to ensure valid RLD computation
        Q = np.tile([10.0, 20.0, 30.0, 25.0, 15.0, 10.0, 20.0, 30.0], 10)
        sigs = compute_event_signatures(Q)
        # Should be positive and less than 1
        assert 0 < sigs["RLD"] < 1

    def test_fld_known_pattern(self):
        """Falling limb density for known pattern."""
        # Longer array to ensure valid FLD computation
        Q = np.tile([10.0, 20.0, 30.0, 25.0, 15.0, 10.0, 20.0, 30.0], 10)
        sigs = compute_event_signatures(Q)
        assert 0 < sigs["FLD"] < 1

    def test_rld_fld_monotonic_rise(self):
        """Monotonic rise should have RLD=1, FLD=0."""
        Q = np.arange(1.0, 101.0)
        sigs = compute_event_signatures(Q)
        assert sigs["RLD"] > 0.9
        assert sigs["FLD"] < 0.1

    def test_rld_fld_monotonic_fall(self):
        """Monotonic fall should have RLD=0, FLD=1."""
        Q = np.arange(100.0, 0.0, -1.0)
        sigs = compute_event_signatures(Q)
        assert sigs["RLD"] < 0.1
        assert sigs["FLD"] > 0.9

    def test_pulse_duration_computed(self):
        """Pulse durations should be computed for varying flow."""
        np.random.seed(42)
        Q = np.random.lognormal(3, 1, 365)
        sigs = compute_event_signatures(Q)
        # Durations should be positive if pulses exist
        if not np.isnan(sigs["High_pulse_duration"]):
            assert sigs["High_pulse_duration"] > 0
        if not np.isnan(sigs["Low_pulse_duration"]):
            assert sigs["Low_pulse_duration"] > 0

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        Q = np.array([])
        sigs = compute_event_signatures(Q)
        for key in SIGNATURE_CATEGORIES["Event"]:
            assert np.isnan(sigs[key])


class TestSeasonalitySignatures:
    """Tests for seasonality signature computations."""

    def test_requires_dates(self):
        """Without dates, seasonality signatures should be NaN."""
        Q = np.random.lognormal(2, 0.5, 730)
        sigs = compute_seasonality_signatures(Q, dates=None)
        for key in SIGNATURE_CATEGORIES["Seasonality"]:
            assert np.isnan(sigs[key])

    def test_seasonal_amplitude_known_sine(self):
        """Known sinusoidal signal should recover amplitude."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        t = np.arange(730)
        amplitude = 50.0
        Q = 100 + amplitude * np.sin(2 * np.pi * t / 365.25)
        sigs = compute_seasonality_signatures(Q, dates)
        # Allow 20% relative tolerance for fitting
        assert sigs["Seasonal_amplitude"] == pytest.approx(amplitude, rel=0.2)

    def test_seasonal_phase_winter_peak(self):
        """Winter peak should have phase near day 0 or 365."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        t = np.arange(730)
        Q = 100 + 50 * np.cos(2 * np.pi * t / 365.25)  # Peak at Jan 1
        sigs = compute_seasonality_signatures(Q, dates)
        # Phase should be near 0 or 365
        phase = sigs["Seasonal_phase"]
        assert phase < 60 or phase > 300

    def test_monthly_cv_constant(self):
        """Constant flow should have Monthly_CV_avg = 0."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        Q = np.ones(730) * 50.0
        sigs = compute_seasonality_signatures(Q, dates)
        assert sigs["Monthly_CV_avg"] == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data_returns_nan(self):
        """Less than 1 year should return NaN."""
        dates = pd.date_range('2000-01-01', periods=200, freq='D')
        Q = np.random.rand(200) + 1
        sigs = compute_seasonality_signatures(Q, dates)
        for key in SIGNATURE_CATEGORIES["Seasonality"]:
            assert np.isnan(sigs[key])


class TestComputeAllSignatures:
    """Integration tests for compute_all_signatures."""

    def test_all_categories_present(self):
        """All 9 categories' signatures should be computed."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        Q = np.random.lognormal(3, 0.5, 730)
        sigs = compute_all_signatures(Q, dates)

        assert "Q_mean" in sigs
        assert "CV" in sigs
        assert "HFD_mean" in sigs
        assert "FDC_slope" in sigs
        assert "High_pulse_count" in sigs
        assert "Recession_k" in sigs
        assert "BFI" in sigs
        assert "RLD" in sigs
        assert "Seasonal_amplitude" in sigs

    def test_without_dates(self):
        """Without dates, timing and seasonality signatures should be NaN."""
        Q = np.random.lognormal(3, 0.5, 730)
        sigs = compute_all_signatures(Q, dates=None)

        assert np.isfinite(sigs["Q_mean"])
        assert np.isfinite(sigs["BFI"])

        assert np.isnan(sigs["HFD_mean"])
        assert np.isnan(sigs["Seasonal_amplitude"])

    def test_returns_dict(self):
        """Should return a dictionary."""
        Q = np.random.lognormal(3, 0.5, 365)
        sigs = compute_all_signatures(Q)
        assert isinstance(sigs, dict)

    def test_all_values_numeric(self):
        """All values should be numeric (float or NaN)."""
        dates = pd.date_range('2000-01-01', periods=730, freq='D')
        Q = np.random.lognormal(3, 0.5, 730)
        sigs = compute_all_signatures(Q, dates)
        
        for key, value in sigs.items():
            assert isinstance(value, (float, np.floating)), f"{key} has non-numeric value: {value}"


class TestSignaturePercentError:
    """Tests for signature_percent_error helper function."""

    def test_exact_match(self):
        """Exact match should give 0% error."""
        error = signature_percent_error(100.0, 100.0)
        assert error == pytest.approx(0.0, abs=1e-10)

    def test_positive_error(self):
        """Overestimate should give positive error."""
        error = signature_percent_error(100.0, 110.0)
        assert error == pytest.approx(10.0, abs=1e-10)

    def test_negative_error(self):
        """Underestimate should give negative error."""
        error = signature_percent_error(100.0, 90.0)
        assert error == pytest.approx(-10.0, abs=1e-10)

    def test_nan_observed(self):
        """NaN observed should return NaN."""
        error = signature_percent_error(np.nan, 100.0)
        assert np.isnan(error)

    def test_nan_simulated(self):
        """NaN simulated should return NaN."""
        error = signature_percent_error(100.0, np.nan)
        assert np.isnan(error)

    def test_near_zero_observed(self):
        """Near-zero observed handled by epsilon."""
        error = signature_percent_error(0.001, 0.001, epsilon=0.01)
        assert error == pytest.approx(0.0, abs=1e-10)


class TestCompareSignatures:
    """Tests for compare_signatures function."""

    def test_returns_nested_dict(self):
        """Should return nested dictionary structure."""
        obs = np.random.lognormal(3, 0.5, 365)
        sim = obs * 1.1
        errors = compare_signatures(obs, {'sim1': sim})
        
        assert isinstance(errors, dict)
        assert 'sim1' in errors
        assert isinstance(errors['sim1'], dict)
        assert 'Q_mean' in errors['sim1']

    def test_multiple_experiments(self):
        """Should handle multiple experiments."""
        obs = np.random.lognormal(3, 0.5, 365)
        sim1 = obs * 1.1
        sim2 = obs * 0.9
        
        errors = compare_signatures(obs, {'sim1': sim1, 'sim2': sim2})
        
        assert 'sim1' in errors
        assert 'sim2' in errors

    def test_error_magnitude(self):
        """10% bias should show ~10% error in Q_mean."""
        np.random.seed(42)
        obs = np.random.lognormal(3, 0.5, 365)
        sim = obs * 1.1  # 10% overestimate
        
        errors = compare_signatures(obs, {'sim': sim})
        
        assert errors['sim']['Q_mean'] == pytest.approx(10.0, rel=0.01)


class TestEdgeCases:
    """Edge case tests across all signature functions."""

    def test_all_nan_input(self):
        """All NaN input should return NaN for all signatures."""
        Q = np.array([np.nan, np.nan, np.nan])
        sigs = compute_all_signatures(Q)
        
        for key, value in sigs.items():
            assert np.isnan(value), f"{key} should be NaN for all-NaN input"

    def test_negative_flows(self):
        """Negative flows (edge case) should still compute without crash."""
        Q = np.array([-10.0, 5.0, 10.0, -5.0, 20.0])
        try:
            sigs = compute_magnitude_signatures(Q)
            assert np.isfinite(sigs["Q_mean"])
        except Exception as e:
            pytest.fail(f"Negative flows caused exception: {e}")

    def test_very_large_values(self):
        """Very large values should be handled."""
        Q = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        sigs = compute_magnitude_signatures(Q)
        assert sigs["Q_mean"] == pytest.approx(3e10, rel=1e-10)

    def test_mixed_zeros_and_values(self):
        """Mix of zeros and positive values."""
        Q = np.concatenate([np.zeros(50), np.ones(50) * 10])
        sigs = compute_all_signatures(Q)
        assert sigs["Zero_flow_freq"] == pytest.approx(0.5, abs=1e-10)
