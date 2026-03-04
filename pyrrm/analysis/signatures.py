"""
Hydrological signature computations for streamflow characterization.

This module provides a comprehensive set of hydrological signatures organized
into 9 categories following the conventions of TOSSH (McMillan, 2021) and
EflowStats (Henriksen et al., 2006). Signatures quantify different aspects
of the flow regime and are useful for:

- Model evaluation and calibration assessment
- Catchment classification and comparison
- Understanding dominant hydrological processes

Categories
----------
1. Magnitude: Mean, median, percentiles, n-day extremes
2. Variability: CV, flashiness, rise/fall rates, reversals, autocorrelation
3. Timing: Half-flow date, date of extremes, constancy, predictability
4. Flow Duration Curve: Slopes at different exceedance ranges
5. Frequency: Pulse counts, high/low flow frequency
6. Recession: Exponential and power-law recession parameters
7. Baseflow: Baseflow index, mean baseflow, recession constant
8. Event: Rising/falling limb density, pulse durations
9. Seasonality: Seasonal amplitude, phase, monthly variability

References
----------
McMillan, H. (2021). Linking hydrologic signatures to hydrologic processes:
    A review. Hydrological Processes, 35(4), e14132. (TOSSH toolbox)

Henriksen, J.A., Heasley, J., Kennen, J.G., Nieswand, S. (2006). Users' manual
    for the Hydroecological Integrity Assessment Process software (including
    the New Jersey Assessment Tools). USGS Open-File Report 2006-1093.
    (EflowStats / HIT)

Addor, N., Nearing, G., Prieto, C., Newman, A.J., Le Vine, N., Clark, M.P.
    (2018). A ranking of hydrological signatures based on their predictability
    in space. Water Resources Research, 54(11), 8792-8812.

Example
-------
>>> import numpy as np
>>> import pandas as pd
>>> from pyrrm.analysis.signatures import compute_all_signatures
>>>
>>> dates = pd.date_range('2000-01-01', periods=730, freq='D')
>>> Q = np.random.lognormal(2, 0.5, 730)
>>> sigs = compute_all_signatures(Q, dates)
>>> print(f"Mean flow: {sigs['Q_mean']:.2f}")
>>> print(f"BFI: {sigs['BFI']:.3f}")
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

from pyrrm.analysis.diagnostics import lyne_hollick_baseflow
from pyrrm.objectives.signatures.dynamics import (
    compute_flashiness_index,
    compute_rising_limb_density,
    compute_falling_limb_density,
    extract_recession_segments,
    compute_recession_constant,
)
from pyrrm.objectives.signatures.water_balance import (
    compute_baseflow_index,
    compute_baseflow_recession_constant,
)


# =============================================================================
# Signature Categories Registry
# =============================================================================

SIGNATURE_CATEGORIES: OrderedDict = OrderedDict([
    ("Magnitude", [
        "Q_mean", "Q_median",
        "Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95",
        "Q_1day_max", "Q_7day_max", "Q_30day_max",
        "Q_1day_min", "Q_7day_min", "Q_30day_min",
    ]),
    ("Variability", [
        "CV", "Flashiness", "Rise_rate", "Fall_rate", "Reversals", "AC1",
    ]),
    ("Timing", [
        "HFD_mean", "Date_max", "Date_min", "Constancy", "Predictability",
    ]),
    ("Flow Duration Curve", [
        "FDC_slope", "FDC_slope_low", "FDC_slope_high",
    ]),
    ("Frequency", [
        "High_pulse_count", "Low_pulse_count",
        "High_flow_freq", "Low_flow_freq", "Zero_flow_freq",
    ]),
    ("Recession", [
        "Recession_k", "Recession_a", "Recession_b",
    ]),
    ("Baseflow", [
        "BFI", "BF_mean", "BF_recession_k",
    ]),
    ("Event", [
        "RLD", "FLD", "High_pulse_duration", "Low_pulse_duration",
    ]),
    ("Seasonality", [
        "Seasonal_amplitude", "Seasonal_phase", "Monthly_CV_avg",
    ]),
])
"""Ordered dictionary mapping category names to lists of signature names."""


SIGNATURE_INFO: Dict[str, Dict] = {
    # ── Magnitude signatures ────────────────────────────────────────────────────
    "Q_mean": {
        "name": "Mean Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Mean daily streamflow over the entire record.",
        "formula": "Q_mean = (1/n) × Σ Q_i",
        "interpretation": "Higher values indicate wetter catchments or larger "
                         "drainage areas. Fundamental baseline metric for "
                         "normalizing other signatures and comparing catchments.",
        "related": ["Q_median", "Q50"],
        "references": ["TOSSH sig_Q_mean", "EflowStats ma1"],
    },
    "Q_median": {
        "name": "Median Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Median (50th percentile) of daily streamflow.",
        "formula": "Q_median = Q at 50th percentile",
        "interpretation": "More robust to outliers than mean. Lower median "
                         "relative to mean indicates positive skew (frequent "
                         "low flows with occasional high peaks).",
        "related": ["Q_mean", "Q50"],
        "references": ["EflowStats ma2"],
    },
    "Q5": {
        "name": "5th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "5th percentile of daily flow (high flow indicator).",
        "formula": "Q5 = Q at 5th percentile (exceeded 95% of time)",
        "interpretation": "Characterizes low-frequency high flows. Higher values "
                         "indicate flashier catchments or those with frequent "
                         "storm events.",
        "related": ["Q10", "Q95", "Q_1day_max"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q10": {
        "name": "10th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "10th percentile of daily flow (high flow indicator).",
        "formula": "Q10 = Q at 10th percentile (exceeded 90% of time)",
        "interpretation": "High flow indicator. Used to define high flow "
                         "thresholds and characterize flood-prone conditions.",
        "related": ["Q5", "Q25", "High_flow_freq"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q25": {
        "name": "25th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "25th percentile (first quartile) of daily flow.",
        "formula": "Q25 = Q at 25th percentile",
        "interpretation": "Upper-moderate flow threshold. The difference "
                         "Q25-Q75 represents the interquartile range of flows.",
        "related": ["Q50", "Q75"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q50": {
        "name": "50th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "50th percentile (median) of daily flow.",
        "formula": "Q50 = Q at 50th percentile",
        "interpretation": "Identical to Q_median. Central tendency measure "
                         "that is robust to extreme values.",
        "related": ["Q_median", "Q25", "Q75"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q75": {
        "name": "75th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "75th percentile (third quartile) of daily flow.",
        "formula": "Q75 = Q at 75th percentile (exceeded 25% of time)",
        "interpretation": "Lower-moderate flow threshold. Often used as a "
                         "baseflow indicator in perennial streams.",
        "related": ["Q50", "Q90", "BFI"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q90": {
        "name": "90th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "90th percentile of daily flow (low flow indicator).",
        "formula": "Q90 = Q at 90th percentile (exceeded 10% of time)",
        "interpretation": "Low flow indicator. Higher Q90 indicates more "
                         "sustained baseflow and groundwater contribution.",
        "related": ["Q95", "Q75", "Low_flow_freq"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q95": {
        "name": "95th Percentile Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "95th percentile of daily flow (low flow indicator).",
        "formula": "Q95 = Q at 95th percentile (exceeded 5% of time)",
        "interpretation": "Critical low flow indicator. Q95=0 indicates "
                         "intermittent streams. Important for environmental "
                         "flow requirements and drought analysis.",
        "related": ["Q90", "Q_7day_min", "Zero_flow_freq"],
        "references": ["TOSSH", "EflowStats"],
    },
    "Q_1day_max": {
        "name": "1-Day Maximum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Maximum single-day flow in the record.",
        "formula": "Q_1day_max = max(Q_i)",
        "interpretation": "Peak flood indicator. High values indicate flashy "
                         "catchments or extreme events. Sensitive to record length.",
        "related": ["Q_7day_max", "Q5", "High_pulse_count"],
        "references": ["TOSSH", "EflowStats dh1"],
    },
    "Q_7day_max": {
        "name": "7-Day Maximum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Maximum 7-day moving average flow.",
        "formula": "Q_7day_max = max(7-day rolling mean of Q)",
        "interpretation": "Sustained high flow indicator. Less sensitive to "
                         "single-day spikes than Q_1day_max. Relevant for "
                         "flood duration impacts.",
        "related": ["Q_1day_max", "Q_30day_max", "High_pulse_duration"],
        "references": ["TOSSH", "EflowStats dh3"],
    },
    "Q_30day_max": {
        "name": "30-Day Maximum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Maximum 30-day moving average flow.",
        "formula": "Q_30day_max = max(30-day rolling mean of Q)",
        "interpretation": "Seasonal high flow indicator. Reflects wet season "
                         "or prolonged flood conditions rather than peak events.",
        "related": ["Q_7day_max", "Seasonal_amplitude"],
        "references": ["TOSSH", "EflowStats dh5"],
    },
    "Q_1day_min": {
        "name": "1-Day Minimum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Minimum single-day flow in the record.",
        "formula": "Q_1day_min = min(Q_i)",
        "interpretation": "Extreme low flow indicator. Zero values indicate "
                         "intermittent flow. Critical for aquatic habitat "
                         "during drought.",
        "related": ["Q_7day_min", "Q95", "Zero_flow_freq"],
        "references": ["TOSSH", "EflowStats dl1"],
    },
    "Q_7day_min": {
        "name": "7-Day Minimum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Minimum 7-day moving average flow.",
        "formula": "Q_7day_min = min(7-day rolling mean of Q)",
        "interpretation": "Sustained low flow indicator (7Q10 basis). Less "
                         "sensitive to measurement error than Q_1day_min. "
                         "Standard for water quality permits.",
        "related": ["Q_1day_min", "Q_30day_min", "BFI"],
        "references": ["TOSSH", "EflowStats dl4"],
    },
    "Q_30day_min": {
        "name": "30-Day Minimum Flow",
        "category": "Magnitude",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Minimum 30-day moving average flow.",
        "formula": "Q_30day_min = min(30-day rolling mean of Q)",
        "interpretation": "Seasonal low flow indicator. Reflects dry season "
                         "baseflow conditions and aquifer storage capacity.",
        "related": ["Q_7day_min", "BF_mean", "Seasonal_amplitude"],
        "references": ["TOSSH", "EflowStats dl5"],
    },

    # ── Variability signatures ──────────────────────────────────────────────────
    "CV": {
        "name": "Coefficient of Variation",
        "category": "Variability",
        "units": "dimensionless",
        "range": [0, "inf"],
        "description": "Standard deviation divided by mean flow.",
        "formula": "CV = σ(Q) / μ(Q)",
        "interpretation": "Measures overall flow variability. Higher CV "
                         "indicates more variable flow regimes (flashy "
                         "catchments, arid climates). Typically 0.5-2.0.",
        "related": ["Flashiness", "AC1"],
        "references": ["TOSSH Q_CoV"],
    },
    "Flashiness": {
        "name": "Richards-Baker Flashiness Index",
        "category": "Variability",
        "units": "dimensionless",
        "range": [0, "inf"],
        "description": "Sum of absolute day-to-day changes divided by total flow.",
        "formula": "Flashiness = Σ|Q_i - Q_{i-1}| / Σ Q_i",
        "interpretation": "Measures how quickly flow changes. Higher values "
                         "indicate urban catchments, steep terrain, or low "
                         "storage. Typically 0.1-1.0.",
        "related": ["CV", "Rise_rate", "Fall_rate"],
        "references": ["Baker et al. (2004)", "TOSSH"],
    },
    "Rise_rate": {
        "name": "Mean Rise Rate",
        "category": "Variability",
        "units": "same as input per day",
        "range": [0, "inf"],
        "description": "Mean rate of positive flow changes.",
        "formula": "Rise_rate = mean(Q_i - Q_{i-1}) for Q_i > Q_{i-1}",
        "interpretation": "Measures typical hydrograph rise speed. Higher "
                         "values indicate rapid storm response (urban, steep, "
                         "or impervious catchments).",
        "related": ["Fall_rate", "Flashiness", "RLD"],
        "references": ["TOSSH", "EflowStats ra1"],
    },
    "Fall_rate": {
        "name": "Mean Fall Rate",
        "category": "Variability",
        "units": "same as input per day",
        "range": ["-inf", 0],
        "description": "Mean rate of negative flow changes (reported as negative).",
        "formula": "Fall_rate = mean(Q_i - Q_{i-1}) for Q_i < Q_{i-1}",
        "interpretation": "Measures typical recession speed. Slower fall rates "
                         "(less negative) indicate greater catchment storage "
                         "and groundwater contribution.",
        "related": ["Rise_rate", "Recession_k", "FLD"],
        "references": ["TOSSH", "EflowStats ra3"],
    },
    "Reversals": {
        "name": "Number of Reversals",
        "category": "Variability",
        "units": "count per year",
        "range": [0, "inf"],
        "description": "Number of times flow switches from rising to falling or vice versa.",
        "formula": "Reversals = count of sign changes in (Q_i - Q_{i-1})",
        "interpretation": "Measures flow oscillation frequency. Higher values "
                         "indicate variable precipitation or regulated flows. "
                         "Natural streams: 50-150/year.",
        "related": ["Flashiness", "High_pulse_count"],
        "references": ["TOSSH", "EflowStats ra8"],
    },
    "AC1": {
        "name": "Lag-1 Autocorrelation",
        "category": "Variability",
        "units": "dimensionless",
        "range": [-1, 1],
        "description": "Correlation between flow on consecutive days.",
        "formula": "AC1 = corr(Q_t, Q_{t-1})",
        "interpretation": "Measures flow persistence. Higher values (near 1) "
                         "indicate slow-responding catchments with sustained "
                         "baseflow. Lower values indicate flashy response.",
        "related": ["Recession_k", "BFI"],
        "references": ["TOSSH", "Montanari & Koutsoyiannis (2012)"],
    },

    # ── Timing signatures ───────────────────────────────────────────────────────
    "HFD_mean": {
        "name": "Half-Flow Date",
        "category": "Timing",
        "units": "day of year",
        "range": [1, 366],
        "description": "Mean day of year when 50% of annual flow volume has passed.",
        "formula": "HFD = day when Σ Q (from Oct 1) reaches 0.5 × annual total",
        "interpretation": "Indicates flow timing centre of mass. Earlier dates "
                         "suggest snowmelt-dominated or winter-wet regimes. "
                         "Climate-sensitive indicator.",
        "related": ["Date_max", "Seasonal_phase"],
        "references": ["Court (1962)", "TOSSH sig_HFD_mean"],
    },
    "Date_max": {
        "name": "Date of Maximum Flow",
        "category": "Timing",
        "units": "day of year",
        "range": [1, 366],
        "description": "Mean day of year when annual maximum flow occurs.",
        "formula": "Date_max = mean DOY of annual peak flows",
        "interpretation": "Indicates typical flood season timing. Useful for "
                         "identifying dominant flood-generating mechanisms "
                         "(snowmelt, monsoon, frontal storms).",
        "related": ["HFD_mean", "Q_1day_max", "Seasonal_phase"],
        "references": ["TOSSH", "EflowStats th1"],
    },
    "Date_min": {
        "name": "Date of Minimum Flow",
        "category": "Timing",
        "units": "day of year",
        "range": [1, 366],
        "description": "Mean day of year when annual minimum flow occurs.",
        "formula": "Date_min = mean DOY of annual minimum flows",
        "interpretation": "Indicates typical low flow season. Important for "
                         "water supply planning and environmental flow timing.",
        "related": ["HFD_mean", "Q_1day_min", "Seasonal_phase"],
        "references": ["TOSSH", "EflowStats tl1"],
    },
    "Constancy": {
        "name": "Colwell's Constancy",
        "category": "Timing",
        "units": "dimensionless",
        "range": [0, 1],
        "description": "Degree to which flow is constant across all time periods.",
        "formula": "C = 1 - H(X)/log(s), where H is entropy, s is states",
        "interpretation": "Higher values (near 1) indicate uniform flow year-round. "
                         "Lower values indicate seasonal or variable regimes. "
                         "Groundwater-fed streams have high constancy.",
        "related": ["Predictability", "CV", "Seasonal_amplitude"],
        "references": ["Colwell (1974)", "TOSSH"],
    },
    "Predictability": {
        "name": "Colwell's Predictability",
        "category": "Timing",
        "units": "dimensionless",
        "range": [0, 1],
        "description": "Sum of constancy and contingency (seasonal regularity).",
        "formula": "P = C + M (constancy + contingency)",
        "interpretation": "Higher values indicate flows are predictable—either "
                         "constant or following a regular seasonal pattern. "
                         "Low values indicate erratic, unpredictable regimes.",
        "related": ["Constancy", "Seasonal_amplitude"],
        "references": ["Colwell (1974)", "TOSSH"],
    },

    # ── Flow Duration Curve signatures ──────────────────────────────────────────
    "FDC_slope": {
        "name": "FDC Slope (Mid-range)",
        "category": "Flow Duration Curve",
        "units": "dimensionless",
        "range": ["-inf", 0],
        "description": "Slope of log-transformed FDC between 33rd and 66th percentiles.",
        "formula": "FDC_slope = (log(Q33) - log(Q66)) / (0.66 - 0.33)",
        "interpretation": "Indicates flow variability in the mid-range. Steeper "
                         "(more negative) slopes indicate more variable flows. "
                         "Related to catchment storage and permeability.",
        "related": ["FDC_slope_low", "FDC_slope_high", "CV"],
        "references": ["TOSSH sig_FDC_slope", "Sawicz et al. (2011)"],
    },
    "FDC_slope_low": {
        "name": "FDC Slope (Low Flows)",
        "category": "Flow Duration Curve",
        "units": "dimensionless",
        "range": ["-inf", 0],
        "description": "Slope of log-transformed FDC between 66th and 90th percentiles.",
        "formula": "FDC_slope_low = (log(Q66) - log(Q90)) / (0.90 - 0.66)",
        "interpretation": "Characterizes low flow variability. Steeper slopes "
                         "indicate rapid baseflow depletion. Related to aquifer "
                         "properties and dry season behaviour.",
        "related": ["FDC_slope", "BFI", "Recession_k"],
        "references": ["TOSSH", "Sawicz et al. (2011)"],
    },
    "FDC_slope_high": {
        "name": "FDC Slope (High Flows)",
        "category": "Flow Duration Curve",
        "units": "dimensionless",
        "range": ["-inf", 0],
        "description": "Slope of log-transformed FDC between 10th and 33rd percentiles.",
        "formula": "FDC_slope_high = (log(Q10) - log(Q33)) / (0.33 - 0.10)",
        "interpretation": "Characterizes high flow variability. Steeper slopes "
                         "indicate greater flood magnitude variability. Related "
                         "to storm response and drainage network efficiency.",
        "related": ["FDC_slope", "Flashiness", "High_pulse_count"],
        "references": ["TOSSH", "Sawicz et al. (2011)"],
    },

    # ── Frequency signatures ────────────────────────────────────────────────────
    "High_pulse_count": {
        "name": "High Pulse Count",
        "category": "Frequency",
        "units": "count per year",
        "range": [0, "inf"],
        "description": "Average number of high flow pulses per year (above 75th percentile).",
        "formula": "High_pulse_count = count(events where Q > Q75) / years",
        "interpretation": "Measures flood frequency. Higher counts indicate more "
                         "frequent storm events or variable precipitation. "
                         "Ecologically important for disturbance regime.",
        "related": ["Low_pulse_count", "High_flow_freq", "Reversals"],
        "references": ["TOSSH", "EflowStats fh1"],
    },
    "Low_pulse_count": {
        "name": "Low Pulse Count",
        "category": "Frequency",
        "units": "count per year",
        "range": [0, "inf"],
        "description": "Average number of low flow pulses per year (below 25th percentile).",
        "formula": "Low_pulse_count = count(events where Q < Q25) / years",
        "interpretation": "Measures drought frequency. Higher counts indicate "
                         "intermittent or flashy streams with frequent low "
                         "flow periods.",
        "related": ["High_pulse_count", "Low_flow_freq", "Zero_flow_freq"],
        "references": ["TOSSH", "EflowStats fl1"],
    },
    "High_flow_freq": {
        "name": "High Flow Frequency",
        "category": "Frequency",
        "units": "fraction",
        "range": [0, 1],
        "description": "Fraction of days with flow above 3× median.",
        "formula": "High_flow_freq = count(Q > 3 × Q_median) / n",
        "interpretation": "Measures proportion of high flow days. Higher values "
                         "indicate flashier regimes. Typical range 0.01-0.10.",
        "related": ["Low_flow_freq", "High_pulse_count", "Flashiness"],
        "references": ["TOSSH", "Clausen & Biggs (2000)"],
    },
    "Low_flow_freq": {
        "name": "Low Flow Frequency",
        "category": "Frequency",
        "units": "fraction",
        "range": [0, 1],
        "description": "Fraction of days with flow below 0.2× median.",
        "formula": "Low_flow_freq = count(Q < 0.2 × Q_median) / n",
        "interpretation": "Measures proportion of low flow days. Higher values "
                         "indicate seasonal or intermittent streams. Related "
                         "to drought stress periods.",
        "related": ["High_flow_freq", "Zero_flow_freq", "Q95"],
        "references": ["TOSSH", "Olden & Poff (2003)"],
    },
    "Zero_flow_freq": {
        "name": "Zero Flow Frequency",
        "category": "Frequency",
        "units": "fraction",
        "range": [0, 1],
        "description": "Fraction of days with zero flow.",
        "formula": "Zero_flow_freq = count(Q = 0) / n",
        "interpretation": "Indicates stream intermittency. Zero for perennial "
                         "streams. Higher values indicate ephemeral systems. "
                         "Critical for aquatic habitat assessment.",
        "related": ["Low_flow_freq", "Q_1day_min", "Q95"],
        "references": ["TOSSH", "Kennard et al. (2010)"],
    },

    # ── Recession signatures ────────────────────────────────────────────────────
    "Recession_k": {
        "name": "Recession Constant (Exponential)",
        "category": "Recession",
        "units": "dimensionless",
        "range": [0, 1],
        "description": "Exponential decay constant for recession limbs.",
        "formula": "Q_t = Q_0 × k^t, where k is fitted to recession segments",
        "interpretation": "Higher values (near 1) indicate slow recession and "
                         "high storage (deep aquifers). Lower values indicate "
                         "rapid drainage (shallow soils, urban).",
        "related": ["BF_recession_k", "AC1", "FDC_slope_low"],
        "references": ["TOSSH sig_BaseflowRecessionK", "Brutsaert & Nieber (1977)"],
    },
    "Recession_a": {
        "name": "Recession Parameter a (Power Law)",
        "category": "Recession",
        "units": "varies",
        "range": [0, "inf"],
        "description": "Coefficient in power-law recession model -dQ/dt = aQ^b.",
        "formula": "-dQ/dt = a × Q^b (a is the coefficient)",
        "interpretation": "Scale parameter for recession. Combines with b to "
                         "characterize nonlinear storage-discharge relationship. "
                         "Related to aquifer transmissivity.",
        "related": ["Recession_b", "Recession_k"],
        "references": ["Brutsaert & Nieber (1977)", "TOSSH"],
    },
    "Recession_b": {
        "name": "Recession Parameter b (Power Law)",
        "category": "Recession",
        "units": "dimensionless",
        "range": [0, "inf"],
        "description": "Exponent in power-law recession model -dQ/dt = aQ^b.",
        "formula": "-dQ/dt = a × Q^b (b is the exponent)",
        "interpretation": "Shape parameter. b=1 indicates linear reservoir. "
                         "b>1 indicates nonlinear storage (common). Higher b "
                         "means faster initial recession that slows down.",
        "related": ["Recession_a", "Recession_k"],
        "references": ["Brutsaert & Nieber (1977)", "TOSSH"],
    },

    # ── Baseflow signatures ─────────────────────────────────────────────────────
    "BFI": {
        "name": "Baseflow Index",
        "category": "Baseflow",
        "units": "dimensionless",
        "range": [0, 1],
        "description": "Ratio of baseflow volume to total streamflow volume.",
        "formula": "BFI = Σ Q_baseflow / Σ Q_total (Lyne-Hollick filter)",
        "interpretation": "Higher values (>0.7) indicate groundwater-dominated "
                         "streams. Lower values (<0.3) indicate surface runoff "
                         "dominated or urban catchments.",
        "related": ["BF_mean", "BF_recession_k", "Q75"],
        "references": ["TOSSH sig_BFI", "EflowStats ml17-20", "Lyne & Hollick (1979)"],
    },
    "BF_mean": {
        "name": "Mean Baseflow",
        "category": "Baseflow",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Mean baseflow from Lyne-Hollick separation.",
        "formula": "BF_mean = mean(Q_baseflow)",
        "interpretation": "Indicates sustained groundwater contribution. Higher "
                         "values indicate larger or more permeable aquifers. "
                         "BF_mean/Q_mean ≈ BFI.",
        "related": ["BFI", "Q_mean", "Q_30day_min"],
        "references": ["TOSSH", "Lyne & Hollick (1979)"],
    },
    "BF_recession_k": {
        "name": "Baseflow Recession Constant",
        "category": "Baseflow",
        "units": "dimensionless",
        "range": [0, 1],
        "description": "Recession constant fitted to baseflow component only.",
        "formula": "Q_bf(t) = Q_bf(0) × k^t (fitted to baseflow series)",
        "interpretation": "Characterizes aquifer drainage timescale. Higher "
                         "values indicate slower aquifer response and greater "
                         "storage. Related to aquifer hydraulic properties.",
        "related": ["Recession_k", "BFI", "AC1"],
        "references": ["TOSSH", "Brutsaert (2008)"],
    },

    # ── Event signatures ────────────────────────────────────────────────────────
    "RLD": {
        "name": "Rising Limb Density",
        "category": "Event",
        "units": "events per day of rise",
        "range": [0, "inf"],
        "description": "Ratio of rising limb count to total rising days.",
        "formula": "RLD = number of rise events / total rising days",
        "interpretation": "Measures hydrograph compactness on rising limbs. "
                         "Higher values indicate more distinct, rapid rises. "
                         "Lower values indicate gradual, sustained rises.",
        "related": ["FLD", "Rise_rate", "Flashiness"],
        "references": ["TOSSH sig_RisingLimbDensity"],
    },
    "FLD": {
        "name": "Falling Limb Density",
        "category": "Event",
        "units": "events per day of fall",
        "range": [0, "inf"],
        "description": "Ratio of falling limb count to total falling days.",
        "formula": "FLD = number of fall events / total falling days",
        "interpretation": "Measures hydrograph compactness on falling limbs. "
                         "Lower values indicate long, slow recessions (high "
                         "storage). Higher values indicate multiple peaks.",
        "related": ["RLD", "Fall_rate", "Recession_k"],
        "references": ["TOSSH sig_FallingLimbDensity"],
    },
    "High_pulse_duration": {
        "name": "Mean High Pulse Duration",
        "category": "Event",
        "units": "days",
        "range": [0, "inf"],
        "description": "Mean duration of high flow pulses (above 75th percentile).",
        "formula": "High_pulse_duration = mean(duration of Q > Q75 events)",
        "interpretation": "Measures typical flood duration. Longer durations "
                         "indicate sustained high flows (snowmelt, large "
                         "catchments). Shorter indicates flashy response.",
        "related": ["Low_pulse_duration", "High_pulse_count", "Q_7day_max"],
        "references": ["TOSSH", "EflowStats dh15"],
    },
    "Low_pulse_duration": {
        "name": "Mean Low Pulse Duration",
        "category": "Event",
        "units": "days",
        "range": [0, "inf"],
        "description": "Mean duration of low flow pulses (below 25th percentile).",
        "formula": "Low_pulse_duration = mean(duration of Q < Q25 events)",
        "interpretation": "Measures typical drought duration. Longer durations "
                         "indicate sustained dry periods. Critical for "
                         "ecological stress assessment.",
        "related": ["High_pulse_duration", "Low_pulse_count", "Q_7day_min"],
        "references": ["TOSSH", "EflowStats dl18"],
    },

    # ── Seasonality signatures ──────────────────────────────────────────────────
    "Seasonal_amplitude": {
        "name": "Seasonal Amplitude",
        "category": "Seasonality",
        "units": "same as input",
        "range": [0, "inf"],
        "description": "Amplitude of fitted sinusoidal seasonal cycle to monthly means.",
        "formula": "Q(t) = Q_mean + A×sin(2π×t/12 + φ); A is amplitude",
        "interpretation": "Measures strength of seasonal flow variation. Higher "
                         "values indicate strong wet/dry seasons. Near-zero "
                         "indicates uniform year-round flow.",
        "related": ["Seasonal_phase", "Monthly_CV_avg", "Constancy"],
        "references": ["TOSSH", "Kennard et al. (2010)"],
    },
    "Seasonal_phase": {
        "name": "Seasonal Phase",
        "category": "Seasonality",
        "units": "month (1-12)",
        "range": [1, 12],
        "description": "Phase of fitted sinusoidal cycle (peak flow month).",
        "formula": "Q(t) = Q_mean + A×sin(2π×t/12 + φ); peak at month = (3 - φ×6/π) mod 12",
        "interpretation": "Indicates timing of seasonal peak. Reflects dominant "
                         "precipitation regime (monsoon, snowmelt, frontal). "
                         "Compare with Date_max for validation.",
        "related": ["Seasonal_amplitude", "HFD_mean", "Date_max"],
        "references": ["TOSSH", "Kennard et al. (2010)"],
    },
    "Monthly_CV_avg": {
        "name": "Average Monthly CV",
        "category": "Seasonality",
        "units": "dimensionless",
        "range": [0, "inf"],
        "description": "Mean of the 12 monthly coefficients of variation.",
        "formula": "Monthly_CV_avg = (1/12) × Σ CV_month",
        "interpretation": "Measures intra-month variability averaged across year. "
                         "Higher values indicate variable flows within each month. "
                         "Complements Seasonal_amplitude (inter-month variation).",
        "related": ["CV", "Seasonal_amplitude", "Predictability"],
        "references": ["TOSSH", "EflowStats"],
    },
}
"""Metadata for each signature including name, units, formula, interpretation, and references."""


def get_signature_info(sig_id: str) -> Optional[Dict]:
    """
    Get metadata for a specific signature.
    
    Args:
        sig_id: Signature identifier (e.g., 'Q_mean', 'BFI').
        
    Returns:
        Dictionary with signature metadata, or None if not found.
    """
    return SIGNATURE_INFO.get(sig_id)


# =============================================================================
# Magnitude Signatures
# =============================================================================

def compute_magnitude_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute magnitude-related hydrological signatures.
    
    Magnitude signatures characterize the overall flow levels including
    central tendency (mean, median), distribution (percentiles), and
    extremes (n-day maxima and minima).
    
    Args:
        Q: Daily streamflow time series (1D array). Units are preserved
            in output (typically mm/day or m³/s).
        dates: Optional DatetimeIndex for Q. Not used for magnitude
            signatures but included for API consistency.
    
    Returns:
        Dictionary with signature names as keys and computed values as floats.
        Returns NaN for signatures that cannot be computed.
    
    Notes:
        - NaN values in Q are excluded from calculations
        - n-day max/min use rolling windows; require len(Q) >= n
        - Percentiles use numpy's linear interpolation method
    
    References:
        Henriksen et al. (2006). EflowStats ma1-ma2, dh1-dh5, dl1-dl5.
        McMillan (2021). TOSSH sig_Q_mean, sig_x_percentile, sig_Q_n_day_max.
    
    Example:
        >>> Q = np.array([10, 20, 30, 40, 50])
        >>> sigs = compute_magnitude_signatures(Q)
        >>> print(f"Mean: {sigs['Q_mean']}")  # 30.0
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) == 0:
        for key in SIGNATURE_CATEGORIES["Magnitude"]:
            sigs[key] = np.nan
        return sigs
    
    sigs["Q_mean"] = float(np.mean(Q_clean))
    sigs["Q_median"] = float(np.median(Q_clean))
    
    for p in [5, 10, 25, 50, 75, 90, 95]:
        sigs[f"Q{p}"] = float(np.percentile(Q_clean, p))
    
    for n in [1, 7, 30]:
        if len(Q_clean) >= n:
            rolling_mean = np.convolve(Q_clean, np.ones(n) / n, mode='valid')
            sigs[f"Q_{n}day_max"] = float(np.max(rolling_mean))
            sigs[f"Q_{n}day_min"] = float(np.min(rolling_mean))
        else:
            sigs[f"Q_{n}day_max"] = np.nan
            sigs[f"Q_{n}day_min"] = np.nan
    
    return sigs


# =============================================================================
# Variability Signatures
# =============================================================================

def compute_variability_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute variability-related hydrological signatures.
    
    Variability signatures characterize the temporal dynamics and
    responsiveness of streamflow, including flashiness, rate of change,
    and persistence (autocorrelation).
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Used for annualizing reversals.
    
    Returns:
        Dictionary with signature names as keys:
        - CV: Coefficient of variation (std/mean)
        - Flashiness: Richards-Baker flashiness index
        - Rise_rate: Mean of positive day-to-day changes
        - Fall_rate: Mean of negative day-to-day changes (absolute value)
        - Reversals: Number of flow direction changes per year
        - AC1: Lag-1 autocorrelation coefficient
    
    Notes:
        - CV returns NaN if mean is zero
        - Rise/Fall rates exclude zero changes
        - Reversals are annualized if dates provided, otherwise per-record
        - AC1 uses Pearson correlation between Q[t] and Q[t-1]
    
    References:
        Baker et al. (2004). Richards-Baker flashiness index.
        Henriksen et al. (2006). EflowStats ra1, ra3, ra8.
        McMillan (2021). TOSSH sig_Autocorrelation, sig_FlashinessIndex.
    
    Example:
        >>> Q = np.array([10, 20, 15, 25, 20, 30])
        >>> sigs = compute_variability_signatures(Q)
        >>> print(f"Reversals: {sigs['Reversals']}")  # 4.0
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) < 2:
        for key in SIGNATURE_CATEGORIES["Variability"]:
            sigs[key] = np.nan
        return sigs
    
    mean_Q = np.mean(Q_clean)
    if mean_Q > 0:
        sigs["CV"] = float(np.std(Q_clean, ddof=0) / mean_Q)
    else:
        sigs["CV"] = np.nan
    
    sigs["Flashiness"] = float(compute_flashiness_index(Q_clean))
    
    dQ = np.diff(Q_clean)
    rises = dQ[dQ > 0]
    falls = dQ[dQ < 0]
    
    sigs["Rise_rate"] = float(np.mean(rises)) if len(rises) > 0 else np.nan
    sigs["Fall_rate"] = float(np.mean(np.abs(falls))) if len(falls) > 0 else np.nan
    
    sign_changes = np.diff(np.sign(dQ))
    reversals = np.sum(sign_changes != 0)
    
    if dates is not None and len(dates) > 1:
        n_years = (dates[-1] - dates[0]).days / 365.25
        sigs["Reversals"] = float(reversals / n_years) if n_years > 0 else np.nan
    else:
        sigs["Reversals"] = float(reversals)
    
    if len(Q_clean) >= 3:
        ac1 = np.corrcoef(Q_clean[:-1], Q_clean[1:])[0, 1]
        sigs["AC1"] = float(ac1) if np.isfinite(ac1) else np.nan
    else:
        sigs["AC1"] = np.nan
    
    return sigs


# =============================================================================
# Timing Signatures
# =============================================================================

def compute_timing_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute timing-related hydrological signatures.
    
    Timing signatures characterize when flow events occur within the year,
    including the center of mass of annual flow and predictability metrics.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: DatetimeIndex corresponding to Q values. Required for timing
            signatures; returns NaN for all if not provided.
    
    Returns:
        Dictionary with signature names as keys:
        - HFD_mean: Half-flow date (day when 50% of annual flow has passed)
        - Date_max: Mean Julian date of annual maximum flow
        - Date_min: Mean Julian date of annual minimum flow
        - Constancy: Colwell's constancy index (temporal stability)
        - Predictability: Colwell's predictability index
    
    Notes:
        - Requires at least 365 days of data for meaningful results
        - HFD computed per water year, then averaged
        - Constancy and Predictability range from 0 to 1
    
    References:
        Court, A. (1962). Measures of streamflow timing.
        Colwell, R.K. (1974). Predictability, constancy, and contingency.
        Henriksen et al. (2006). EflowStats ta1, ta2, th1, tl1.
        McMillan (2021). TOSSH sig_HFD_mean.
    
    Example:
        >>> dates = pd.date_range('2000-01-01', periods=365, freq='D')
        >>> Q = np.ones(365)  # Constant flow
        >>> sigs = compute_timing_signatures(Q, dates)
        >>> print(f"HFD: {sigs['HFD_mean']:.0f}")  # ~183
    """
    sigs: Dict[str, float] = {}
    
    if dates is None or len(Q) < 365:
        for key in SIGNATURE_CATEGORIES["Timing"]:
            sigs[key] = np.nan
        return sigs
    
    Q_clean = np.where(np.isnan(Q), 0, Q)
    
    df = pd.DataFrame({'Q': Q_clean, 'date': dates})
    df['doy'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    
    hfd_values = []
    for year, group in df.groupby('year'):
        if len(group) >= 360:
            cumsum = np.cumsum(group['Q'].values)
            if cumsum[-1] > 0:
                half_total = cumsum[-1] / 2
                idx = np.searchsorted(cumsum, half_total)
                if idx < len(group):
                    hfd_values.append(group['doy'].iloc[idx])
    
    sigs["HFD_mean"] = float(np.mean(hfd_values)) if hfd_values else np.nan
    
    date_max_values = []
    date_min_values = []
    for year, group in df.groupby('year'):
        if len(group) >= 360:
            max_idx = group['Q'].idxmax()
            min_idx = group['Q'].idxmin()
            date_max_values.append(group.loc[max_idx, 'doy'])
            date_min_values.append(group.loc[min_idx, 'doy'])
    
    sigs["Date_max"] = float(np.mean(date_max_values)) if date_max_values else np.nan
    sigs["Date_min"] = float(np.mean(date_min_values)) if date_min_values else np.nan
    
    constancy, predictability = _compute_colwell_indices(Q_clean, dates)
    sigs["Constancy"] = constancy
    sigs["Predictability"] = predictability
    
    return sigs


def _compute_colwell_indices(
    Q: np.ndarray,
    dates: pd.DatetimeIndex,
    n_states: int = 11,
    n_seasons: int = 12,
) -> Tuple[float, float]:
    """
    Compute Colwell's constancy and predictability indices.
    
    Args:
        Q: Streamflow array (NaN-free).
        dates: DatetimeIndex.
        n_states: Number of flow states (bins).
        n_seasons: Number of seasonal periods (12 = monthly).
    
    Returns:
        Tuple of (constancy, predictability), both in range [0, 1].
    
    References:
        Colwell, R.K. (1974). Predictability, constancy, and contingency
        of periodic phenomena. Ecology, 55(5), 1148-1153.
    """
    if len(Q) < 365 or np.all(Q == Q[0]):
        return np.nan, np.nan
    
    Q_pos = Q[Q > 0]
    if len(Q_pos) < 10:
        return np.nan, np.nan
    
    log_Q = np.log10(Q_pos + 1e-10)
    bins = np.linspace(log_Q.min(), log_Q.max() + 1e-10, n_states + 1)
    
    df = pd.DataFrame({'Q': Q, 'date': dates})
    df['month'] = df['date'].dt.month
    df['log_Q'] = np.log10(df['Q'] + 1e-10)
    df['state'] = np.digitize(df['log_Q'], bins) - 1
    df['state'] = df['state'].clip(0, n_states - 1)
    
    contingency = np.zeros((n_states, n_seasons))
    for _, row in df.iterrows():
        state = int(row['state'])
        season = int(row['month']) - 1
        contingency[state, season] += 1
    
    N = contingency.sum()
    if N == 0:
        return np.nan, np.nan
    
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    
    H_Y = -np.sum((row_sums / N) * np.log2(row_sums / N + 1e-10))
    H_X = -np.sum((col_sums / N) * np.log2(col_sums / N + 1e-10))
    
    H_XY = 0.0
    for i in range(n_states):
        for j in range(n_seasons):
            if contingency[i, j] > 0:
                p_ij = contingency[i, j] / N
                H_XY -= p_ij * np.log2(p_ij)
    
    H_max = np.log2(n_states)
    
    if H_max > 0:
        constancy = 1 - H_Y / H_max
        predictability = 1 - (H_XY - H_X) / H_max
    else:
        constancy = np.nan
        predictability = np.nan
    
    return float(constancy), float(predictability)


# =============================================================================
# Flow Duration Curve Signatures
# =============================================================================

def compute_fdc_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute flow duration curve (FDC) signatures.
    
    FDC signatures characterize the slope of the flow duration curve at
    different exceedance probability ranges, indicating flow variability
    and storage characteristics.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Not used but included for API consistency.
    
    Returns:
        Dictionary with signature names as keys:
        - FDC_slope: Slope between 33rd and 66th percentile (mid-range)
        - FDC_slope_low: Slope between 70th and 90th percentile (low flows)
        - FDC_slope_high: Slope between 10th and 30th percentile (high flows)
    
    Notes:
        - Slopes computed on log-transformed FDC
        - Negative slopes indicate decreasing flow with increasing exceedance
        - Steeper slopes indicate more variable flow regime
        - Requires positive flows for log transformation
    
    References:
        Sawicz et al. (2011). Catchment classification.
        McMillan (2021). TOSSH sig_FDC_slope.
    
    Example:
        >>> Q = np.random.lognormal(2, 0.5, 365)
        >>> sigs = compute_fdc_signatures(Q)
        >>> print(f"FDC slope: {sigs['FDC_slope']:.3f}")
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)]
    Q_pos = Q_clean[Q_clean > 0]
    
    if len(Q_pos) < 10:
        for key in SIGNATURE_CATEGORIES["Flow Duration Curve"]:
            sigs[key] = np.nan
        return sigs
    
    Q_sorted = np.sort(Q_pos)[::-1]
    n = len(Q_sorted)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100
    
    log_Q = np.log10(Q_sorted + 1e-10)
    
    def _fdc_slope(exc_low: float, exc_high: float) -> float:
        """Compute slope between two exceedance percentiles."""
        idx_low = np.searchsorted(exceedance, exc_low)
        idx_high = np.searchsorted(exceedance, exc_high)
        
        if idx_high <= idx_low or idx_high >= n:
            return np.nan
        
        x = exceedance[idx_low:idx_high]
        y = log_Q[idx_low:idx_high]
        
        if len(x) < 2:
            return np.nan
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    sigs["FDC_slope"] = _fdc_slope(33, 66)
    sigs["FDC_slope_low"] = _fdc_slope(70, 90)
    sigs["FDC_slope_high"] = _fdc_slope(10, 30)
    
    return sigs


# =============================================================================
# Frequency Signatures
# =============================================================================

def compute_frequency_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute frequency-related hydrological signatures.
    
    Frequency signatures characterize how often different flow conditions
    occur, including high and low pulse events.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Used for annualizing counts.
    
    Returns:
        Dictionary with signature names as keys:
        - High_pulse_count: Events above 75th percentile per year
        - Low_pulse_count: Events below 25th percentile per year
        - High_flow_freq: Events above 3x median per year
        - Low_flow_freq: Fraction of days below 0.2x mean
        - Zero_flow_freq: Fraction of days with zero flow
    
    Notes:
        - Pulse = consecutive days above/below threshold
        - Event counts are annualized if dates provided
        - Zero_flow_freq always returns fraction (not annualized)
    
    References:
        Henriksen et al. (2006). EflowStats fh1, fl1.
        McMillan (2021). TOSSH sig_x_Q_frequency.
    
    Example:
        >>> Q = np.random.lognormal(2, 0.5, 730)  # 2 years
        >>> dates = pd.date_range('2000-01-01', periods=730, freq='D')
        >>> sigs = compute_frequency_signatures(Q, dates)
        >>> print(f"High pulse count: {sigs['High_pulse_count']:.1f}/yr")
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) < 10:
        for key in SIGNATURE_CATEGORIES["Frequency"]:
            sigs[key] = np.nan
        return sigs
    
    n_years = 1.0
    if dates is not None and len(dates) > 1:
        n_years = max((dates[-1] - dates[0]).days / 365.25, 1.0)
    
    q75 = np.percentile(Q_clean, 75)
    q25 = np.percentile(Q_clean, 25)
    median_Q = np.median(Q_clean)
    mean_Q = np.mean(Q_clean)
    
    def _count_pulses(condition: np.ndarray) -> int:
        """Count number of distinct pulse events."""
        in_pulse = False
        count = 0
        for c in condition:
            if c and not in_pulse:
                count += 1
                in_pulse = True
            elif not c:
                in_pulse = False
        return count
    
    high_pulse_events = _count_pulses(Q_clean > q75)
    low_pulse_events = _count_pulses(Q_clean < q25)
    
    sigs["High_pulse_count"] = float(high_pulse_events / n_years)
    sigs["Low_pulse_count"] = float(low_pulse_events / n_years)
    
    high_flow_events = _count_pulses(Q_clean > 3 * median_Q)
    sigs["High_flow_freq"] = float(high_flow_events / n_years)
    
    sigs["Low_flow_freq"] = float(np.sum(Q_clean < 0.2 * mean_Q) / len(Q_clean))
    
    sigs["Zero_flow_freq"] = float(np.sum(Q_clean == 0) / len(Q_clean))
    
    return sigs


# =============================================================================
# Recession Signatures
# =============================================================================

def compute_recession_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute recession-related hydrological signatures.
    
    Recession signatures characterize how streamflow decreases after
    rainfall events, reflecting catchment storage and drainage properties.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Not used but included for API consistency.
    
    Returns:
        Dictionary with signature names as keys:
        - Recession_k: Exponential recession constant (Q_t = Q_0 * k^t)
        - Recession_a: Power-law coefficient (dQ/dt = -a * Q^b)
        - Recession_b: Power-law exponent
    
    Notes:
        - Recession segments extracted as consecutive decreasing flows
        - Minimum 5 days required for valid recession segment
        - k values typically 0.9-0.99 for daily data
        - Power-law fitting uses log-log linear regression
    
    References:
        Brutsaert & Nieber (1977). Regionalized drought flow hydrographs.
        McMillan (2021). TOSSH sig_RecessionAnalysis, sig_BaseflowRecessionK.
    
    Example:
        >>> k_true = 0.95
        >>> Q = 100 * (k_true ** np.arange(50))  # Pure exponential decay
        >>> sigs = compute_recession_signatures(Q)
        >>> print(f"Recession k: {sigs['Recession_k']:.3f}")  # ~0.95
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) < 10:
        for key in SIGNATURE_CATEGORIES["Recession"]:
            sigs[key] = np.nan
        return sigs
    
    k = compute_recession_constant(Q_clean, method='ratio')
    sigs["Recession_k"] = float(k) if np.isfinite(k) else np.nan
    
    segments = extract_recession_segments(Q_clean, min_length=5)
    
    if len(segments) == 0:
        sigs["Recession_a"] = np.nan
        sigs["Recession_b"] = np.nan
        return sigs
    
    all_Q = []
    all_dQdt = []
    
    for seg in segments:
        Q_seg = seg[:-1]
        dQ = np.diff(seg)
        
        valid = (Q_seg > 0) & (dQ < 0)
        if np.sum(valid) > 0:
            all_Q.extend(Q_seg[valid].tolist())
            all_dQdt.extend((-dQ[valid]).tolist())
    
    if len(all_Q) < 5:
        sigs["Recession_a"] = np.nan
        sigs["Recession_b"] = np.nan
        return sigs
    
    all_Q = np.array(all_Q)
    all_dQdt = np.array(all_dQdt)
    
    valid = (all_Q > 0) & (all_dQdt > 0)
    if np.sum(valid) < 5:
        sigs["Recession_a"] = np.nan
        sigs["Recession_b"] = np.nan
        return sigs
    
    log_Q = np.log10(all_Q[valid])
    log_dQdt = np.log10(all_dQdt[valid])
    
    try:
        b, log_a = np.polyfit(log_Q, log_dQdt, 1)
        a = 10 ** log_a
        sigs["Recession_a"] = float(a)
        sigs["Recession_b"] = float(b)
    except (np.linalg.LinAlgError, ValueError):
        sigs["Recession_a"] = np.nan
        sigs["Recession_b"] = np.nan
    
    return sigs


# =============================================================================
# Baseflow Signatures
# =============================================================================

def compute_baseflow_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute baseflow-related hydrological signatures.
    
    Baseflow signatures characterize the groundwater contribution to
    streamflow, separated using digital filter methods.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Not used but included for API consistency.
    
    Returns:
        Dictionary with signature names as keys:
        - BFI: Baseflow index (baseflow / total flow)
        - BF_mean: Mean baseflow magnitude
        - BF_recession_k: Baseflow-specific recession constant
    
    Notes:
        - Uses Lyne-Hollick digital filter (alpha=0.925) for separation
        - BFI typically ranges 0.2-0.8 for natural catchments
        - Higher BFI indicates more groundwater-dominated flow regime
    
    References:
        Lyne & Hollick (1979). Stochastic time-variable rainfall-runoff.
        Institute of Hydrology (1980). Low Flow Studies report.
        McMillan (2021). TOSSH sig_BFI, sig_BaseflowMagnitude.
    
    Example:
        >>> Q = np.random.lognormal(2, 0.5, 365)
        >>> sigs = compute_baseflow_signatures(Q)
        >>> print(f"BFI: {sigs['BFI']:.3f}")
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) < 10:
        for key in SIGNATURE_CATEGORIES["Baseflow"]:
            sigs[key] = np.nan
        return sigs
    
    baseflow = lyne_hollick_baseflow(Q_clean, alpha=0.925)
    
    total_Q = np.sum(Q_clean)
    if total_Q > 0:
        sigs["BFI"] = float(np.sum(baseflow) / total_Q)
    else:
        sigs["BFI"] = np.nan
    
    sigs["BF_mean"] = float(np.mean(baseflow))
    
    bf_k = compute_baseflow_recession_constant(Q_clean, window=5)
    sigs["BF_recession_k"] = float(bf_k) if np.isfinite(bf_k) else np.nan
    
    return sigs


# =============================================================================
# Event Signatures
# =============================================================================

def compute_event_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute event-related hydrological signatures.
    
    Event signatures characterize the shape and duration of flow events,
    including rising and falling limb characteristics.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: Optional DatetimeIndex. Used for annualizing durations.
    
    Returns:
        Dictionary with signature names as keys:
        - RLD: Rising limb density (fraction of rising days)
        - FLD: Falling limb density (fraction of falling days)
        - High_pulse_duration: Mean duration of high pulses (days)
        - Low_pulse_duration: Mean duration of low pulses (days)
    
    Notes:
        - RLD + FLD may not equal 1 due to constant-flow days
        - Pulse duration = consecutive days above/below threshold
        - High pulse: above 75th percentile; Low pulse: below 25th
    
    References:
        Henriksen et al. (2006). EflowStats dh15, dl16.
        McMillan (2021). TOSSH sig_RisingLimbDensity.
    
    Example:
        >>> Q = np.array([10, 20, 30, 25, 15, 10, 20, 30])
        >>> sigs = compute_event_signatures(Q)
        >>> print(f"RLD: {sigs['RLD']:.2f}")
    """
    sigs: Dict[str, float] = {}
    
    Q_clean = Q[~np.isnan(Q)] if len(Q) > 0 else np.array([])
    
    if len(Q_clean) < 10:
        for key in SIGNATURE_CATEGORIES["Event"]:
            sigs[key] = np.nan
        return sigs
    
    sigs["RLD"] = float(compute_rising_limb_density(Q_clean))
    sigs["FLD"] = float(compute_falling_limb_density(Q_clean))
    
    q75 = np.percentile(Q_clean, 75)
    q25 = np.percentile(Q_clean, 25)
    
    def _pulse_durations(condition: np.ndarray) -> List[int]:
        """Extract durations of consecutive True periods."""
        durations = []
        current_duration = 0
        for c in condition:
            if c:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return durations
    
    high_durations = _pulse_durations(Q_clean > q75)
    low_durations = _pulse_durations(Q_clean < q25)
    
    sigs["High_pulse_duration"] = float(np.mean(high_durations)) if high_durations else np.nan
    sigs["Low_pulse_duration"] = float(np.mean(low_durations)) if low_durations else np.nan
    
    return sigs


# =============================================================================
# Seasonality Signatures
# =============================================================================

def compute_seasonality_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute seasonality-related hydrological signatures.
    
    Seasonality signatures characterize the annual cycle of streamflow,
    including amplitude and timing of seasonal variations.
    
    Args:
        Q: Daily streamflow time series (1D array).
        dates: DatetimeIndex corresponding to Q values. Required for
            seasonality signatures; returns NaN for all if not provided.
    
    Returns:
        Dictionary with signature names as keys:
        - Seasonal_amplitude: Amplitude of fitted sine curve
        - Seasonal_phase: Phase of seasonal cycle (day of year)
        - Monthly_CV_avg: Average coefficient of variation by month
    
    Notes:
        - Seasonal amplitude from sine curve fit to monthly means
        - Requires at least 365 days for meaningful results
        - Phase indicates timing of seasonal maximum
    
    References:
        Archfield et al. (2013). Magnificent Seven statistics.
        Henriksen et al. (2006). EflowStats ma24-ma35.
    
    Example:
        >>> dates = pd.date_range('2000-01-01', periods=730, freq='D')
        >>> t = np.arange(730)
        >>> Q = 100 + 50 * np.sin(2 * np.pi * t / 365.25)  # Known amplitude=50
        >>> sigs = compute_seasonality_signatures(Q, dates)
        >>> print(f"Amplitude: {sigs['Seasonal_amplitude']:.1f}")  # ~50
    """
    sigs: Dict[str, float] = {}
    
    if dates is None or len(Q) < 365:
        for key in SIGNATURE_CATEGORIES["Seasonality"]:
            sigs[key] = np.nan
        return sigs
    
    Q_clean = np.where(np.isnan(Q), np.nanmean(Q), Q)
    
    df = pd.DataFrame({'Q': Q_clean, 'date': dates})
    df['month'] = df['date'].dt.month
    
    monthly_means = df.groupby('month')['Q'].mean()
    monthly_stds = df.groupby('month')['Q'].std()
    monthly_cvs = monthly_stds / monthly_means
    monthly_cvs = monthly_cvs.replace([np.inf, -np.inf], np.nan)
    
    sigs["Monthly_CV_avg"] = float(monthly_cvs.mean()) if not monthly_cvs.isna().all() else np.nan
    
    if len(monthly_means) < 12:
        sigs["Seasonal_amplitude"] = np.nan
        sigs["Seasonal_phase"] = np.nan
        return sigs
    
    months = np.arange(1, 13)
    y = monthly_means.reindex(months).values
    
    if np.any(np.isnan(y)):
        sigs["Seasonal_amplitude"] = np.nan
        sigs["Seasonal_phase"] = np.nan
        return sigs
    
    t = (months - 1) * 30.44
    
    try:
        mean_y = np.mean(y)
        y_centered = y - mean_y
        
        sin_t = np.sin(2 * np.pi * t / 365.25)
        cos_t = np.cos(2 * np.pi * t / 365.25)
        
        A = np.sum(y_centered * cos_t) * 2 / 12
        B = np.sum(y_centered * sin_t) * 2 / 12
        
        amplitude = np.sqrt(A**2 + B**2)
        phase = np.arctan2(B, A) * 365.25 / (2 * np.pi)
        if phase < 0:
            phase += 365.25
        
        sigs["Seasonal_amplitude"] = float(amplitude)
        sigs["Seasonal_phase"] = float(phase)
    except (ValueError, np.linalg.LinAlgError):
        sigs["Seasonal_amplitude"] = np.nan
        sigs["Seasonal_phase"] = np.nan
    
    return sigs


# =============================================================================
# Main Computation Function
# =============================================================================

def compute_all_signatures(
    Q: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, float]:
    """
    Compute all hydrological signatures for a streamflow time series.
    
    This is the main entry point for computing the full suite of ~45
    signatures across all 9 categories. Signatures that require dates
    will return NaN if dates are not provided.
    
    Args:
        Q: Daily streamflow time series (1D array). Should be in consistent
            units (mm/day or m³/s). NaN values are handled per-signature.
        dates: Optional DatetimeIndex corresponding to Q values. Required
            for timing and seasonality signatures; other categories will
            compute successfully without dates.
    
    Returns:
        Dictionary with all signature names as keys and computed values
        as floats. Keys are organized by category in SIGNATURE_CATEGORIES.
        Returns NaN for signatures that cannot be computed.
    
    Notes:
        - Computation order: Magnitude, Variability, Timing, FDC, Frequency,
          Recession, Baseflow, Event, Seasonality
        - Each category function handles its own edge cases
        - Total of ~45 signatures returned
    
    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyrrm.analysis.signatures import compute_all_signatures
        >>>
        >>> # Generate synthetic data
        >>> dates = pd.date_range('2000-01-01', periods=730, freq='D')
        >>> Q = np.random.lognormal(2, 0.5, 730)
        >>>
        >>> # Compute all signatures
        >>> sigs = compute_all_signatures(Q, dates)
        >>>
        >>> # Access specific signatures
        >>> print(f"Mean flow: {sigs['Q_mean']:.2f}")
        >>> print(f"BFI: {sigs['BFI']:.3f}")
        >>> print(f"Flashiness: {sigs['Flashiness']:.4f}")
    """
    Q = np.asarray(Q).flatten()
    
    all_sigs: Dict[str, float] = {}
    
    all_sigs.update(compute_magnitude_signatures(Q, dates))
    all_sigs.update(compute_variability_signatures(Q, dates))
    all_sigs.update(compute_timing_signatures(Q, dates))
    all_sigs.update(compute_fdc_signatures(Q, dates))
    all_sigs.update(compute_frequency_signatures(Q, dates))
    all_sigs.update(compute_recession_signatures(Q, dates))
    all_sigs.update(compute_baseflow_signatures(Q, dates))
    all_sigs.update(compute_event_signatures(Q, dates))
    all_sigs.update(compute_seasonality_signatures(Q, dates))
    
    return all_sigs


def signature_percent_error(obs_val: float, sim_val: float, epsilon: float = 0.01) -> float:
    """
    Compute percent error between observed and simulated signature values.
    
    Args:
        obs_val: Observed signature value.
        sim_val: Simulated signature value.
        epsilon: Minimum observed value for percent calculation.
    
    Returns:
        Percent error: 100 * (sim - obs) / |obs|.
        Returns 0 if both values are below epsilon.
        Returns 100 if sim > epsilon but obs <= epsilon.
        Returns NaN if obs_val is NaN.
    
    Example:
        >>> error = signature_percent_error(100, 110)  # 10% overestimate
        >>> print(f"{error:.1f}%")  # 10.0%
    """
    if np.isnan(obs_val) or np.isnan(sim_val):
        return np.nan
    
    if abs(obs_val) > epsilon:
        return 100.0 * (sim_val - obs_val) / abs(obs_val)
    elif abs(sim_val) <= epsilon:
        return 0.0
    else:
        return 100.0


def compare_signatures(
    obs: np.ndarray,
    sim_dict: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare signatures between observed and multiple simulated series.
    
    Args:
        obs: Observed streamflow time series.
        sim_dict: Dictionary mapping experiment names to simulated arrays.
        dates: Optional DatetimeIndex for timing/seasonality signatures.
    
    Returns:
        Nested dictionary: {experiment_name: {signature_name: percent_error}}.
    
    Example:
        >>> obs = np.random.lognormal(2, 0.5, 365)
        >>> sim1 = obs * 1.1  # 10% bias
        >>> sim2 = obs * 0.9  # -10% bias
        >>> errors = compare_signatures(obs, {'sim1': sim1, 'sim2': sim2})
        >>> print(f"sim1 Q_mean error: {errors['sim1']['Q_mean']:.1f}%")
    """
    obs_sigs = compute_all_signatures(obs, dates)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for exp_name, sim in sim_dict.items():
        sim_sigs = compute_all_signatures(sim, dates)
        errors: Dict[str, float] = {}
        
        for sig_name in obs_sigs:
            errors[sig_name] = signature_percent_error(
                obs_sigs[sig_name],
                sim_sigs.get(sig_name, np.nan),
            )
        
        results[exp_name] = errors
    
    return results
