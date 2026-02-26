"""
Model diagnostics and performance analysis.

This module provides tools for diagnosing model performance
and analyzing simulation results.

The canonical diagnostic suite -- ``compute_diagnostics`` -- produces 48
metrics covering NSE variants, KGE variants with components, KGE
non-parametric variants with components, error metrics, FDC volume
biases, raw hydrological signature values, and hydrological signature
errors.  All notebooks and the ``CalibrationReport`` class should use
this function so that diagnostic output is consistent everywhere.

The module also exposes ``lyne_hollick_baseflow`` for separating a
streamflow hydrograph into baseflow (slow flow) and quickflow (fast
flow) components using the Lyne-Hollick single-pass digital filter.

Naming convention
-----------------
All metric families follow the same pattern:

    {metric}[_{transformation}][_{component}]

where ``transformation`` is one of ``log``, ``sqrt``, ``inv`` (meaning
1/Q), and ``component`` is one of ``r``, ``alpha``, ``beta``.  NSE
variants therefore use ``NSE``, ``NSE_log``, ``NSE_sqrt``, ``NSE_inv``
(matching the KGE naming) rather than the older ``LogNSE``, ``SqrtNSE``,
``InvNSE`` names.

References
----------
Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009).
    Decomposition of the mean squared error and NSE performance criteria.
    Journal of Hydrology, 377(1-2), 80-91.

Pool, S., Vis, M., Seibert, J. (2018). Evaluating model performance:
    towards a non-parametric variant of the Kling-Gupta efficiency.
    Hydrological Sciences Journal, 63(13-14), 1941-1953.

Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008).
    A process-based diagnostic approach to model evaluation.
    Water Resources Research, 44(1).

Baker, D. B., Richards, R. P., Loftus, T. T., & Kramer, J. W. (2004).
    A new flashiness index: characteristics and applications to midwestern
    rivers and streams. Journal of the American Water Resources Association,
    40(2), 503-522.

Lyne, V. & Hollick, M. (1979). Stochastic time-variable rainfall-runoff
    modelling. Institute of Engineers Australia National Conference,
    pp. 89-93.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from pyrrm.calibration.objective_functions import (
    NSE, KGE, RMSE, MAE, PBIAS, LogNSE, calculate_metrics
)


# =========================================================================
# Lyne-Hollick baseflow separation
# =========================================================================

def lyne_hollick_baseflow(q: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    """Lyne-Hollick single-pass digital filter for baseflow separation.

    Separates a streamflow hydrograph into baseflow (slow flow) and
    quickflow (fast flow) components.  Quickflow is simply
    ``q - lyne_hollick_baseflow(q)``.

    Args:
        q: Total streamflow array (must be non-negative).
        alpha: Filter parameter (default 0.925 per Lyne & Hollick, 1979).

    Returns:
        Baseflow array of the same length as *q*.

    References:
        Lyne, V. & Hollick, M. (1979). Stochastic time-variable
        rainfall-runoff modelling. Institute of Engineers Australia
        National Conference, pp. 89-93.

    Example:
        >>> import numpy as np
        >>> q = np.array([1.0, 5.0, 3.0, 2.0, 1.5])
        >>> bf = lyne_hollick_baseflow(q)
        >>> qf = q - bf
    """
    q = np.asarray(q, dtype=float)
    bf = np.zeros_like(q)
    bf[0] = q[0]
    for t in range(1, len(q)):
        bf[t] = alpha * bf[t - 1] + (1 - alpha) / 2 * (q[t] + q[t - 1])
        bf[t] = min(bf[t], q[t])
        bf[t] = max(bf[t], 0.0)
    return bf


# =========================================================================
# Canonical diagnostic metric suite
# =========================================================================

def compute_diagnostics(sim, obs) -> OrderedDict:
    """Compute the full suite of diagnostic metrics.

    Returns an OrderedDict with 48 metrics:

    - NSE variants: NSE, NSE_log, NSE_sqrt, NSE_inv
    - KGE(Q) + components: KGE, KGE_r, KGE_alpha, KGE_beta
    - KGE(log Q) + components
    - KGE(sqrt Q) + components
    - KGE(1/Q) + components
    - KGE_np(Q) + components: KGE_np, KGE_np_r, KGE_np_alpha, KGE_np_beta
    - KGE_np(log Q) + components
    - KGE_np(sqrt Q) + components
    - KGE_np(1/Q) + components
    - Error metrics: RMSE, MAE, PBIAS
    - FDC segment volume biases (Yilmaz et al. 2008): FHV, FMV, FLV
    - Raw hydrological signatures: BFI_obs, BFI_sim
    - Hydrological signatures (% error): Sig_BFI, Sig_Flash, Sig_Q95, Sig_Q5

    Args:
        sim: Simulated flow values (array-like).
        obs: Observed flow values (array-like).

    Returns:
        OrderedDict of metric_name -> float.

    Notes:
        For ephemeral or low-flow catchments, some metrics use fallbacks:
        - **Sig_Q95** (and **Sig_Q5**): Percent error in the 5th (resp. 95th)
          percentile. When the observed percentile is zero or below epsilon,
          returns 0 if simulated is also low, else 100 (sim has flow when obs
          has none).
        - **FLV** (low-flow volume bias): Uses the lowest 30% of flows by rank.
          If no observed values in that segment exceed epsilon, returns 0 if
          simulated low-flow volume is also negligible, else 100. If observed
          low flows are constant (zero spread in log space), uses volume bias
          in the low segment (same as FHV/FMV) so the metric is a continuous %.
        - **FHV**, **FMV**: Return NaN if the corresponding observed segment
          sum is zero (no fallback).
    """
    sim = np.asarray(sim).flatten()
    obs = np.asarray(obs).flatten()
    mask = ~(np.isnan(sim) | np.isnan(obs) | np.isinf(sim) | np.isinf(obs))
    s, o = sim[mask], obs[mask]
    if len(s) == 0:
        return OrderedDict()

    m = OrderedDict()
    eps_flow = 0.01

    # --- Pre-compute transformed series used by both NSE and KGE -------------
    pos = o > 0
    if pos.sum() > 0:
        log_o, log_s = np.log(o[pos] + 1), np.log(np.maximum(s[pos], 0) + 1)
    else:
        log_o = log_s = np.array([])

    sqrt_o, sqrt_s = np.sqrt(np.maximum(o, 0)), np.sqrt(np.maximum(s, 0))

    inv_mask = o > eps_flow
    if inv_mask.sum() > 0:
        inv_o = 1.0 / o[inv_mask]
        inv_s = 1.0 / np.maximum(s[inv_mask], eps_flow)
    else:
        inv_o = inv_s = np.array([])

    # --- NSE variants --------------------------------------------------------
    def _nse(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    m['NSE'] = _nse(o, s)
    m['NSE_log'] = _nse(log_o, log_s) if len(log_o) > 0 else np.nan
    m['NSE_sqrt'] = _nse(sqrt_o, sqrt_s)
    m['NSE_inv'] = _nse(inv_o, inv_s) if len(inv_o) > 0 else np.nan

    # --- KGE helper (Pearson r) ----------------------------------------------
    def _kge(a, b):
        if len(a) < 2:
            return np.nan, np.nan, np.nan, np.nan
        r = np.corrcoef(a, b)[0, 1]
        alpha = np.std(b) / np.std(a) if np.std(a) > 0 else np.nan
        beta = np.mean(b) / np.mean(a) if np.mean(a) != 0 else np.nan
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return kge, r, alpha, beta

    # --- KGE_np helper (Spearman r, FDC-based alpha) -------------------------
    def _kge_np(a, b):
        """Non-parametric KGE (Pool et al. 2018)."""
        if len(a) < 2:
            return np.nan, np.nan, np.nan, np.nan
        r_s = _scipy_stats.spearmanr(a, b).statistic
        n = len(a)
        ranks_obs = _scipy_stats.rankdata(a) / n
        ranks_sim = _scipy_stats.rankdata(b) / n
        alpha_np = 1 - 0.5 * np.mean(np.abs(ranks_sim - ranks_obs))
        beta = np.mean(b) / np.mean(a) if np.mean(a) != 0 else np.nan
        kge_np = 1 - np.sqrt(
            (r_s - 1) ** 2 + (alpha_np - 1) ** 2 + (beta - 1) ** 2
        )
        return kge_np, r_s, alpha_np, beta

    # KGE(Q)
    kge, r, alpha, beta = _kge(o, s)
    m['KGE'] = kge
    m['KGE_r'] = r
    m['KGE_alpha'] = alpha
    m['KGE_beta'] = beta

    # KGE(log Q)
    kge_l, r_l, a_l, b_l = _kge(log_o, log_s) if len(log_o) > 0 else (np.nan,) * 4
    m['KGE_log'] = kge_l
    m['KGE_log_r'] = r_l
    m['KGE_log_alpha'] = a_l
    m['KGE_log_beta'] = b_l

    # KGE(sqrt Q)
    kge_s, r_s, a_s, b_s = _kge(sqrt_o, sqrt_s)
    m['KGE_sqrt'] = kge_s
    m['KGE_sqrt_r'] = r_s
    m['KGE_sqrt_alpha'] = a_s
    m['KGE_sqrt_beta'] = b_s

    # KGE(1/Q)
    kge_i, r_i, a_i, b_i = _kge(inv_o, inv_s) if len(inv_o) > 0 else (np.nan,) * 4
    m['KGE_inv'] = kge_i
    m['KGE_inv_r'] = r_i
    m['KGE_inv_alpha'] = a_i
    m['KGE_inv_beta'] = b_i

    # KGE_np(Q)
    knp, knp_r, knp_a, knp_b = _kge_np(o, s)
    m['KGE_np'] = knp
    m['KGE_np_r'] = knp_r
    m['KGE_np_alpha'] = knp_a
    m['KGE_np_beta'] = knp_b

    # KGE_np(log Q)
    knpl, knpl_r, knpl_a, knpl_b = _kge_np(log_o, log_s) if len(log_o) > 0 else (np.nan,) * 4
    m['KGE_np_log'] = knpl
    m['KGE_np_log_r'] = knpl_r
    m['KGE_np_log_alpha'] = knpl_a
    m['KGE_np_log_beta'] = knpl_b

    # KGE_np(sqrt Q)
    knps, knps_r, knps_a, knps_b = _kge_np(sqrt_o, sqrt_s)
    m['KGE_np_sqrt'] = knps
    m['KGE_np_sqrt_r'] = knps_r
    m['KGE_np_sqrt_alpha'] = knps_a
    m['KGE_np_sqrt_beta'] = knps_b

    # KGE_np(1/Q)
    knpi, knpi_r, knpi_a, knpi_b = _kge_np(inv_o, inv_s) if len(inv_o) > 0 else (np.nan,) * 4
    m['KGE_np_inv'] = knpi
    m['KGE_np_inv_r'] = knpi_r
    m['KGE_np_inv_alpha'] = knpi_a
    m['KGE_np_inv_beta'] = knpi_b

    # --- Error metrics -------------------------------------------------------
    m['RMSE'] = np.sqrt(np.mean((s - o) ** 2))
    m['MAE'] = np.mean(np.abs(s - o))
    m['PBIAS'] = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan

    # --- FDC segment volume biases (Yilmaz et al. 2008) ----------------------
    obs_sorted = np.sort(o)[::-1]
    sim_sorted = np.sort(s)[::-1]
    n = len(obs_sorted)

    h = max(int(0.02 * n), 1)
    sum_obs_h = np.sum(obs_sorted[:h])
    m['FHV'] = (
        100 * (np.sum(sim_sorted[:h]) - sum_obs_h) / sum_obs_h
        if sum_obs_h > 0 else np.nan
    )

    i20 = int(0.20 * n)
    i70 = min(int(0.70 * n), n)
    sum_obs_m = np.sum(obs_sorted[i20:i70])
    m['FMV'] = (
        100 * (np.sum(sim_sorted[i20:i70]) - sum_obs_m) / sum_obs_m
        if sum_obs_m > 0 else np.nan
    )

    i70_start = int(0.70 * n)
    obs_low = obs_sorted[i70_start:]
    sim_low = sim_sorted[i70_start:]
    low_pos = obs_low > eps_flow
    if low_pos.sum() > 0:
        log_obs_low = np.log(obs_low[low_pos])
        log_sim_low = np.log(np.maximum(sim_low[low_pos], eps_flow))
        min_log_obs = np.min(log_obs_low)
        sum_log_obs = np.sum(log_obs_low - min_log_obs)
        sum_log_sim = np.sum(log_sim_low - min_log_obs)
        if sum_log_obs > 0:
            m['FLV'] = 100 * (sum_log_sim - sum_log_obs) / sum_log_obs
        else:
            # Observed low flows are constant (zero spread in log space), e.g.
            # small or ephemeral catchments. Use volume bias in the low segment
            # (same idea as FHV/FMV) so FLV is a continuous % instead of ±100.
            sum_obs_low = np.sum(obs_low[low_pos])
            sum_sim_low = np.sum(sim_low[low_pos])
            if sum_obs_low > 0:
                m['FLV'] = 100 * (sum_sim_low - sum_obs_low) / sum_obs_low
            else:
                m['FLV'] = 0.0 if sum_sim_low <= eps_flow else 100.0
    else:
        # No observed flow above epsilon in low 30%: define as 0 if sim also none, else 100%
        m['FLV'] = 100.0 if np.sum(sim_low) > eps_flow else 0.0

    # --- Hydrological signatures ------------------------------------------------
    def _bfi(q, alpha=0.925):
        """Baseflow index via Lyne-Hollick (ratio of baseflow to total flow)."""
        bf = lyne_hollick_baseflow(q, alpha)
        total = np.sum(q)
        return np.sum(bf) / total if total > 0 else np.nan

    def _flashiness(q):
        """Richards-Baker Flashiness Index."""
        total = np.sum(q)
        return np.sum(np.abs(np.diff(q))) / total if total > 0 else np.nan

    def _sig_pct_error(sig_sim, sig_obs):
        """Percent error; when obs is zero or below eps, use 0 if sim also low, else 100."""
        if sig_obs > eps_flow:
            return 100 * (sig_sim - sig_obs) / sig_obs
        return 0.0 if sig_sim <= eps_flow else 100.0

    bfi_obs = _bfi(o)
    bfi_sim = _bfi(s)
    m['BFI_obs'] = bfi_obs
    m['BFI_sim'] = bfi_sim
    m['Sig_BFI'] = _sig_pct_error(bfi_sim, bfi_obs)

    flash_obs = _flashiness(o)
    flash_sim = _flashiness(s)
    m['Sig_Flash'] = _sig_pct_error(flash_sim, flash_obs)

    q95_obs = np.percentile(o, 5)   # exceeded 95% of the time
    q95_sim = np.percentile(s, 5)
    m['Sig_Q95'] = _sig_pct_error(q95_sim, q95_obs)

    q5_obs = np.percentile(o, 95)   # exceeded 5% of the time
    q5_sim = np.percentile(s, 95)
    m['Sig_Q5'] = _sig_pct_error(q5_sim, q5_obs)

    return m


DIAGNOSTIC_GROUPS: OrderedDict = OrderedDict([
    ("NSE variants", ["NSE", "NSE_log", "NSE_sqrt", "NSE_inv"]),
    ("KGE(Q)", ["KGE", "KGE_r", "KGE_alpha", "KGE_beta"]),
    ("KGE(log Q)", ["KGE_log", "KGE_log_r", "KGE_log_alpha", "KGE_log_beta"]),
    ("KGE(sqrt Q)", ["KGE_sqrt", "KGE_sqrt_r", "KGE_sqrt_alpha", "KGE_sqrt_beta"]),
    ("KGE(1/Q)", ["KGE_inv", "KGE_inv_r", "KGE_inv_alpha", "KGE_inv_beta"]),
    ("KGE_np(Q)", ["KGE_np", "KGE_np_r", "KGE_np_alpha", "KGE_np_beta"]),
    ("KGE_np(log Q)", ["KGE_np_log", "KGE_np_log_r", "KGE_np_log_alpha", "KGE_np_log_beta"]),
    ("KGE_np(sqrt Q)", ["KGE_np_sqrt", "KGE_np_sqrt_r", "KGE_np_sqrt_alpha", "KGE_np_sqrt_beta"]),
    ("KGE_np(1/Q)", ["KGE_np_inv", "KGE_np_inv_r", "KGE_np_inv_alpha", "KGE_np_inv_beta"]),
    ("Error metrics", ["RMSE", "MAE", "PBIAS"]),
    ("FDC volume bias", ["FHV", "FMV", "FLV"]),
    ("Signatures (raw)", ["BFI_obs", "BFI_sim"]),
    ("Signatures (% error)", ["Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"]),
])
"""Ordered groups for the 48 canonical diagnostic metrics."""


def print_diagnostics(metrics: dict, label: str = "") -> None:
    """Print a grouped diagnostics table.

    Args:
        metrics: Dictionary returned by ``compute_diagnostics``.
        label: Optional header label (e.g. objective name or run ID).
    """
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTIC METRICS{f'  --  {label}' if label else ''}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Value':>12}")
    for group_name, keys in DIAGNOSTIC_GROUPS.items():
        print(f"  {'-' * 40}")
        print(f"  {group_name}")
        for k in keys:
            v = metrics.get(k, np.nan)
            if isinstance(v, float) and np.isnan(v):
                print(f"    {k:<23} {'N/A':>12}")
            else:
                print(f"    {k:<23} {v:>12.4f}")
    print(f"{'=' * 60}")


class ModelDiagnostics:
    """
    Comprehensive model diagnostics and performance analysis.
    
    Provides:
    - Standard performance metrics
    - Flow regime analysis
    - Residual analysis
    - Seasonal performance breakdown
    
    Example:
        >>> diag = ModelDiagnostics(simulated, observed, dates)
        >>> print(diag.summary())
        >>> diag.get_monthly_performance()
    """
    
    def __init__(
        self,
        simulated: np.ndarray,
        observed: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        flow_units: str = 'mm/d'
    ):
        """
        Initialize diagnostics.
        
        Args:
            simulated: Simulated flow values
            observed: Observed flow values
            dates: DatetimeIndex for temporal analysis
            flow_units: Units for flow values
        """
        self.simulated = np.asarray(simulated).flatten()
        self.observed = np.asarray(observed).flatten()
        self.dates = dates
        self.flow_units = flow_units
        
        if len(self.simulated) != len(self.observed):
            raise ValueError("Simulated and observed arrays must have same length")
        
        # Calculate residuals
        self.residuals = self.simulated - self.observed
        
        # Valid mask (non-NaN)
        self._valid = ~(np.isnan(self.simulated) | np.isnan(self.observed))
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate the canonical diagnostic metric suite.
        
        Returns:
            OrderedDict with 48 metric names and values.
        """
        sim = self.simulated[self._valid]
        obs = self.observed[self._valid]
        
        return compute_diagnostics(sim, obs)
    
    def get_flow_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different flow regimes.
        
        Returns:
            Dictionary with metrics for 'low', 'medium', 'high' flows
        """
        sim = self.simulated[self._valid]
        obs = self.observed[self._valid]
        
        # Define thresholds
        q10 = np.percentile(obs, 10)  # Low flow threshold
        q90 = np.percentile(obs, 90)  # High flow threshold
        
        results = {}
        
        # Low flows (Q < Q10)
        low_mask = obs <= q10
        if np.sum(low_mask) > 10:
            results['low_flow'] = calculate_metrics(sim[low_mask], obs[low_mask])
        
        # Medium flows (Q10 < Q < Q90)
        med_mask = (obs > q10) & (obs < q90)
        if np.sum(med_mask) > 10:
            results['medium_flow'] = calculate_metrics(sim[med_mask], obs[med_mask])
        
        # High flows (Q > Q90)
        high_mask = obs >= q90
        if np.sum(high_mask) > 10:
            results['high_flow'] = calculate_metrics(sim[high_mask], obs[high_mask])
        
        return results
    
    def get_monthly_performance(self) -> pd.DataFrame:
        """
        Calculate monthly performance breakdown.
        
        Returns:
            DataFrame with metrics by month
        """
        if self.dates is None:
            raise ValueError("Dates required for monthly analysis")
        
        df = pd.DataFrame({
            'simulated': self.simulated,
            'observed': self.observed,
            'month': self.dates.month
        })
        
        results = []
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                metrics = calculate_metrics(
                    month_data['simulated'].values,
                    month_data['observed'].values
                )
                metrics['month'] = month
                results.append(metrics)
        
        return pd.DataFrame(results).set_index('month')
    
    def get_annual_performance(self) -> pd.DataFrame:
        """
        Calculate annual performance breakdown.
        
        Returns:
            DataFrame with metrics by year
        """
        if self.dates is None:
            raise ValueError("Dates required for annual analysis")
        
        df = pd.DataFrame({
            'simulated': self.simulated,
            'observed': self.observed,
            'year': self.dates.year
        })
        
        results = []
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            if len(year_data) > 30:  # At least 30 days
                metrics = calculate_metrics(
                    year_data['simulated'].values,
                    year_data['observed'].values
                )
                metrics['year'] = year
                results.append(metrics)
        
        return pd.DataFrame(results).set_index('year')
    
    def get_residual_statistics(self) -> Dict[str, float]:
        """
        Calculate residual statistics.
        
        Returns:
            Dictionary with residual statistics
        """
        res = self.residuals[self._valid]
        
        return {
            'mean': np.mean(res),
            'std': np.std(res),
            'median': np.median(res),
            'min': np.min(res),
            'max': np.max(res),
            'skewness': float(pd.Series(res).skew()),
            'kurtosis': float(pd.Series(res).kurtosis()),
            'n_positive': int(np.sum(res > 0)),
            'n_negative': int(np.sum(res < 0)),
        }
    
    def get_volume_statistics(self) -> Dict[str, float]:
        """
        Calculate volume-related statistics.
        
        Returns:
            Dictionary with volume statistics
        """
        sim = self.simulated[self._valid]
        obs = self.observed[self._valid]
        
        total_sim = np.sum(sim)
        total_obs = np.sum(obs)
        
        return {
            'total_simulated': total_sim,
            'total_observed': total_obs,
            'volume_error': total_sim - total_obs,
            'volume_error_percent': 100 * (total_sim - total_obs) / total_obs if total_obs > 0 else np.nan,
            'mean_simulated': np.mean(sim),
            'mean_observed': np.mean(obs),
        }
    
    def get_timing_statistics(self) -> Dict[str, Any]:
        """
        Calculate timing-related statistics for peaks.
        
        Returns:
            Dictionary with timing statistics
        """
        if self.dates is None:
            return {}
        
        # Find peaks (simple approach: local maxima above threshold)
        threshold = np.percentile(self.observed[self._valid], 90)
        
        obs_peaks = []
        sim_peaks = []
        
        for i in range(1, len(self.observed) - 1):
            if (self.observed[i] > threshold and 
                self.observed[i] > self.observed[i-1] and 
                self.observed[i] > self.observed[i+1]):
                obs_peaks.append(i)
                
                # Find corresponding simulated peak within ±3 days
                window = slice(max(0, i-3), min(len(self.simulated), i+4))
                sim_peak_idx = np.argmax(self.simulated[window]) + max(0, i-3)
                sim_peaks.append(sim_peak_idx)
        
        if len(obs_peaks) == 0:
            return {'n_peaks': 0}
        
        timing_errors = [sim_peaks[i] - obs_peaks[i] for i in range(len(obs_peaks))]
        
        return {
            'n_peaks': len(obs_peaks),
            'mean_timing_error_days': np.mean(timing_errors),
            'std_timing_error_days': np.std(timing_errors),
            'peaks_early': sum(1 for e in timing_errors if e < 0),
            'peaks_late': sum(1 for e in timing_errors if e > 0),
            'peaks_exact': sum(1 for e in timing_errors if e == 0),
        }
    
    def summary(self) -> str:
        """Generate text summary of diagnostics."""
        metrics = self.get_metrics()
        volume = self.get_volume_statistics()
        residuals = self.get_residual_statistics()
        
        lines = [
            "=" * 60,
            "MODEL DIAGNOSTICS SUMMARY",
            "=" * 60,
            "",
            "Performance Metrics:",
            "-" * 40,
            f"  NSE:    {metrics.get('NSE', np.nan):8.4f}",
            f"  KGE:    {metrics.get('KGE', np.nan):8.4f}",
            f"  RMSE:   {metrics.get('RMSE', np.nan):8.4f} {self.flow_units}",
            f"  MAE:    {metrics.get('MAE', np.nan):8.4f} {self.flow_units}",
            f"  PBIAS:  {metrics.get('PBIAS', np.nan):8.2f} %",
            f"  NSE_log: {metrics.get('NSE_log', np.nan):8.4f}",
            "",
            "Volume Statistics:",
            "-" * 40,
            f"  Total Observed:    {volume['total_observed']:12.2f} {self.flow_units}",
            f"  Total Simulated:   {volume['total_simulated']:12.2f} {self.flow_units}",
            f"  Volume Error:      {volume['volume_error_percent']:12.2f} %",
            "",
            "Residual Statistics:",
            "-" * 40,
            f"  Mean Residual:     {residuals['mean']:12.4f}",
            f"  Std Residual:      {residuals['std']:12.4f}",
            f"  Overestimations:   {residuals['n_positive']:12d}",
            f"  Underestimations:  {residuals['n_negative']:12d}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
