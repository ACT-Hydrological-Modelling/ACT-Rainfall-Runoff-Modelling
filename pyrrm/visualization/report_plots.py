"""
Report visualization functions for CalibrationReport.

This module provides plotting functions for generating comprehensive calibration
report cards, including both Matplotlib (static) and Plotly (interactive) versions.

Functions:
    Matplotlib:
    - plot_report_card_matplotlib: Multi-panel summary figure
    - plot_hydrograph_comparison: Observed vs simulated hydrograph
    - plot_fdc_comparison: Flow duration curves
    - plot_scatter_comparison: Scatter plot with 1:1 line
    - plot_parameter_bounds_chart: Parameters as % of bounds
    
    Plotly:
    - plot_report_card_plotly: Interactive HTML dashboard
    - plot_hydrograph_plotly: Interactive hydrograph
    - plot_fdc_plotly: Interactive FDC
    - plot_parameter_bounds_plotly: Interactive parameter bounds chart

Example:
    >>> from pyrrm.calibration import CalibrationReport
    >>> from pyrrm.visualization.report_plots import plot_report_card_matplotlib
    >>> report = CalibrationReport.load('my_calibration.pkl')
    >>> fig = plot_report_card_matplotlib(report)
    >>> fig.savefig('report.png', dpi=300)
"""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyrrm.calibration.report import CalibrationReport
    from matplotlib.figure import Figure

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Styling Constants
# =============================================================================

# Light theme colors
COLORS_LIGHT = {
    'observed': '#e74c3c',      # Red
    'simulated': '#3498db',     # Blue
    'one_to_one': '#2ecc71',    # Green
    'grid': '#bdc3c7',          # Light gray
    'text': '#2c3e50',          # Dark blue-gray
}

# Dark theme colors
COLORS_DARK = {
    'observed': '#ff6b6b',      # Coral red
    'simulated': '#4ecdc4',     # Turquoise
    'one_to_one': '#95e1d3',    # Light green
    'grid': '#34495e',          # Dark gray
    'text': '#ecf0f1',          # Light gray
    'background': '#1a1a2e',
    'panel': '#16213e',
}


def _apply_dark_style():
    """Apply dark style to matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        'figure.facecolor': COLORS_DARK['background'],
        'axes.facecolor': COLORS_DARK['panel'],
        'axes.edgecolor': COLORS_DARK['text'],
        'axes.labelcolor': COLORS_DARK['text'],
        'text.color': COLORS_DARK['text'],
        'xtick.color': COLORS_DARK['text'],
        'ytick.color': COLORS_DARK['text'],
        'grid.color': COLORS_DARK['grid'],
        'legend.facecolor': COLORS_DARK['panel'],
        'legend.edgecolor': COLORS_DARK['text'],
    })


def _reset_style():
    """Reset matplotlib to default style."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update(plt.rcParamsDefault)


# =============================================================================
# Helper Functions
# =============================================================================

def _calculate_basic_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Calculate comprehensive performance metrics directly."""
    # Filter valid data
    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs_v = obs[mask]
    sim_v = sim[mask]
    
    if len(obs_v) == 0:
        return {}
    
    metrics = {}
    
    # ==========================================================================
    # NSE and variants
    # ==========================================================================
    # Standard NSE
    ss_res = np.sum((obs_v - sim_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    metrics['NSE'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # Log NSE (with offset for zeros)
    obs_pos_mask = obs_v > 0
    obs_pos = obs_v[obs_pos_mask]
    sim_pos = sim_v[obs_pos_mask]
    if len(obs_pos) > 0:
        log_obs = np.log(obs_pos + 1)
        log_sim = np.log(np.maximum(sim_pos, 0) + 1)
        ss_res_log = np.sum((log_obs - log_sim) ** 2)
        ss_tot_log = np.sum((log_obs - np.mean(log_obs)) ** 2)
        metrics['NSE_log'] = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else np.nan
    else:
        metrics['NSE_log'] = np.nan
    
    # Inverse NSE (1/Q transformation)
    obs_pos_inv = obs_v[obs_v > 0.01]
    sim_pos_inv = sim_v[obs_v > 0.01]
    if len(obs_pos_inv) > 0:
        inv_obs = 1.0 / obs_pos_inv
        inv_sim = 1.0 / np.maximum(sim_pos_inv, 0.01)
        ss_res_inv = np.sum((inv_obs - inv_sim) ** 2)
        ss_tot_inv = np.sum((inv_obs - np.mean(inv_obs)) ** 2)
        metrics['NSE_inv'] = 1 - ss_res_inv / ss_tot_inv if ss_tot_inv > 0 else np.nan
    else:
        metrics['NSE_inv'] = np.nan
    
    # Sqrt NSE
    sqrt_obs = np.sqrt(np.maximum(obs_v, 0))
    sqrt_sim = np.sqrt(np.maximum(sim_v, 0))
    ss_res_sqrt = np.sum((sqrt_obs - sqrt_sim) ** 2)
    ss_tot_sqrt = np.sum((sqrt_obs - np.mean(sqrt_obs)) ** 2)
    metrics['NSE_sqrt'] = 1 - ss_res_sqrt / ss_tot_sqrt if ss_tot_sqrt > 0 else np.nan
    
    # ==========================================================================
    # KGE (standard) and components
    # ==========================================================================
    r = np.corrcoef(obs_v, sim_v)[0, 1] if len(obs_v) > 1 else np.nan
    alpha = np.std(sim_v) / np.std(obs_v) if np.std(obs_v) > 0 else np.nan
    beta = np.mean(sim_v) / np.mean(obs_v) if np.mean(obs_v) != 0 else np.nan
    metrics['KGE'] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r) else np.nan
    metrics['KGE_r'] = r
    metrics['KGE_alpha'] = alpha
    metrics['KGE_beta'] = beta
    
    # ==========================================================================
    # KGE with transformations (each with components)
    # ==========================================================================
    # KGE(log Q)
    if len(obs_pos) > 0:
        log_obs_kge = np.log(obs_pos + 1)
        log_sim_kge = np.log(np.maximum(sim_pos, 0) + 1)
        r_log = np.corrcoef(log_obs_kge, log_sim_kge)[0, 1] if len(log_obs_kge) > 1 else np.nan
        alpha_log = np.std(log_sim_kge) / np.std(log_obs_kge) if np.std(log_obs_kge) > 0 else np.nan
        beta_log = np.mean(log_sim_kge) / np.mean(log_obs_kge) if np.mean(log_obs_kge) != 0 else np.nan
        metrics['KGE_log'] = 1 - np.sqrt((r_log - 1)**2 + (alpha_log - 1)**2 + (beta_log - 1)**2) if not np.isnan(r_log) else np.nan
        metrics['KGE_log_r'] = r_log
        metrics['KGE_log_alpha'] = alpha_log
        metrics['KGE_log_beta'] = beta_log
    else:
        metrics['KGE_log'] = np.nan
        metrics['KGE_log_r'] = np.nan
        metrics['KGE_log_alpha'] = np.nan
        metrics['KGE_log_beta'] = np.nan
    
    # KGE(sqrt Q)
    r_sqrt = np.corrcoef(sqrt_obs, sqrt_sim)[0, 1] if len(sqrt_obs) > 1 else np.nan
    alpha_sqrt = np.std(sqrt_sim) / np.std(sqrt_obs) if np.std(sqrt_obs) > 0 else np.nan
    beta_sqrt = np.mean(sqrt_sim) / np.mean(sqrt_obs) if np.mean(sqrt_obs) != 0 else np.nan
    metrics['KGE_sqrt'] = 1 - np.sqrt((r_sqrt - 1)**2 + (alpha_sqrt - 1)**2 + (beta_sqrt - 1)**2) if not np.isnan(r_sqrt) else np.nan
    metrics['KGE_sqrt_r'] = r_sqrt
    metrics['KGE_sqrt_alpha'] = alpha_sqrt
    metrics['KGE_sqrt_beta'] = beta_sqrt
    
    # KGE(1/Q)
    if len(obs_pos_inv) > 0:
        inv_obs_kge = 1.0 / obs_pos_inv
        inv_sim_kge = 1.0 / np.maximum(sim_pos_inv, 0.01)
        r_inv = np.corrcoef(inv_obs_kge, inv_sim_kge)[0, 1] if len(inv_obs_kge) > 1 else np.nan
        alpha_inv = np.std(inv_sim_kge) / np.std(inv_obs_kge) if np.std(inv_obs_kge) > 0 else np.nan
        beta_inv = np.mean(inv_sim_kge) / np.mean(inv_obs_kge) if np.mean(inv_obs_kge) != 0 else np.nan
        metrics['KGE_inv'] = 1 - np.sqrt((r_inv - 1)**2 + (alpha_inv - 1)**2 + (beta_inv - 1)**2) if not np.isnan(r_inv) else np.nan
        metrics['KGE_inv_r'] = r_inv
        metrics['KGE_inv_alpha'] = alpha_inv
        metrics['KGE_inv_beta'] = beta_inv
    else:
        metrics['KGE_inv'] = np.nan
        metrics['KGE_inv_r'] = np.nan
        metrics['KGE_inv_alpha'] = np.nan
        metrics['KGE_inv_beta'] = np.nan
    
    # ==========================================================================
    # KGE non-parametric and variants
    # ==========================================================================
    from scipy import stats as scipy_stats
    
    # KGE non-parametric uses Spearman correlation and normalized flow duration curves
    def kge_nonparametric(obs_arr, sim_arr):
        """Compute KGE non-parametric: uses Spearman r, relative variability from FDC."""
        if len(obs_arr) < 3:
            return np.nan, np.nan, np.nan, np.nan
        # Spearman correlation
        rho, _ = scipy_stats.spearmanr(obs_arr, sim_arr)
        # Relative variability from FDC
        obs_sorted = np.sort(obs_arr)[::-1]
        sim_sorted = np.sort(sim_arr)[::-1]
        alpha_np = np.mean(sim_sorted) / np.mean(obs_sorted) if np.mean(obs_sorted) > 0 else np.nan
        # Beta as bias ratio
        beta_np = np.mean(sim_arr) / np.mean(obs_arr) if np.mean(obs_arr) != 0 else np.nan
        if np.isnan(rho) or np.isnan(alpha_np) or np.isnan(beta_np):
            return np.nan, rho, alpha_np, beta_np
        kge_np = 1 - np.sqrt((rho - 1)**2 + (alpha_np - 1)**2 + (beta_np - 1)**2)
        return kge_np, rho, alpha_np, beta_np
    
    # Standard KGE_np
    kge_np_val, kge_np_rho, kge_np_alpha, kge_np_beta = kge_nonparametric(obs_v, sim_v)
    metrics['KGE_np'] = kge_np_val
    
    # KGE_np on sqrt(Q)
    kge_np_sqrt_val, _, _, _ = kge_nonparametric(sqrt_obs, sqrt_sim)
    metrics['KGE_np_sqrt'] = kge_np_sqrt_val
    
    # KGE_np on log(Q)
    if len(obs_pos) > 0:
        log_obs_np = np.log(obs_pos + 1)
        log_sim_np = np.log(np.maximum(sim_pos, 0) + 1)
        kge_np_log_val, _, _, _ = kge_nonparametric(log_obs_np, log_sim_np)
        metrics['KGE_np_log'] = kge_np_log_val
    else:
        metrics['KGE_np_log'] = np.nan
    
    # KGE_np on 1/Q
    if len(obs_pos_inv) > 0:
        inv_obs_np = 1.0 / obs_pos_inv
        inv_sim_np = 1.0 / np.maximum(sim_pos_inv, 0.01)
        kge_np_inv_val, _, _, _ = kge_nonparametric(inv_obs_np, inv_sim_np)
        metrics['KGE_np_inv'] = kge_np_inv_val
    else:
        metrics['KGE_np_inv'] = np.nan
    
    # ==========================================================================
    # Traditional error metrics
    # ==========================================================================
    metrics['RMSE'] = np.sqrt(np.mean((sim_v - obs_v) ** 2))
    metrics['MAE'] = np.mean(np.abs(sim_v - obs_v))
    metrics['PBIAS'] = 100 * np.sum(sim_v - obs_v) / np.sum(obs_v) if np.sum(obs_v) != 0 else np.nan
    metrics['R2'] = r ** 2 if not np.isnan(r) else np.nan
    
    # SDEB - Standard Deviation of Error Bias (captures timing/shape errors)
    residuals = sim_v - obs_v
    metrics['SDEB'] = np.std(residuals)
    
    # ==========================================================================
    # FDC-based metrics (segmented flow errors)
    # ==========================================================================
    # Sort flows for FDC
    obs_sorted = np.sort(obs_v)[::-1]
    sim_sorted = np.sort(sim_v)[::-1]
    n = len(obs_sorted)
    
    # Flow percentiles
    def get_percentile_value(arr, pct):
        idx = int(pct / 100 * len(arr))
        idx = min(idx, len(arr) - 1)
        return arr[idx]
    
    # High flows (Q5, Q10) - top 5-10% exceedance
    q5_obs = get_percentile_value(obs_sorted, 5)
    q5_sim = get_percentile_value(sim_sorted, 5)
    q10_obs = get_percentile_value(obs_sorted, 10)
    q10_sim = get_percentile_value(sim_sorted, 10)
    
    metrics['Q5_bias'] = 100 * (q5_sim - q5_obs) / q5_obs if q5_obs > 0 else np.nan
    metrics['Q10_bias'] = 100 * (q10_sim - q10_obs) / q10_obs if q10_obs > 0 else np.nan
    
    # Mid flows (Q50)
    q50_obs = get_percentile_value(obs_sorted, 50)
    q50_sim = get_percentile_value(sim_sorted, 50)
    metrics['Q50_bias'] = 100 * (q50_sim - q50_obs) / q50_obs if q50_obs > 0 else np.nan
    
    # Low flows (Q90, Q95) - bottom 90-95% exceedance
    q90_obs = get_percentile_value(obs_sorted, 90)
    q90_sim = get_percentile_value(sim_sorted, 90)
    q95_obs = get_percentile_value(obs_sorted, 95)
    q95_sim = get_percentile_value(sim_sorted, 95)
    
    metrics['Q90_bias'] = 100 * (q90_sim - q90_obs) / q90_obs if q90_obs > 0 else np.nan
    metrics['Q95_bias'] = 100 * (q95_sim - q95_obs) / q95_obs if q95_obs > 0 else np.nan
    
    # FHV, FMV, FLV - Flow volume errors (high, mid, low flow segments)
    # FHV: High flow volume error (top 2% of flows)
    idx_2pct = max(1, int(0.02 * n))
    fhv_obs = np.sum(obs_sorted[:idx_2pct])
    fhv_sim = np.sum(sim_sorted[:idx_2pct])
    metrics['FHV'] = 100 * (fhv_sim - fhv_obs) / fhv_obs if fhv_obs > 0 else np.nan
    
    # FMV: Mid flow volume error (20-70% exceedance)
    idx_20pct = int(0.20 * n)
    idx_70pct = int(0.70 * n)
    fmv_obs = np.sum(obs_sorted[idx_20pct:idx_70pct])
    fmv_sim = np.sum(sim_sorted[idx_20pct:idx_70pct])
    metrics['FMV'] = 100 * (fmv_sim - fmv_obs) / fmv_obs if fmv_obs > 0 else np.nan
    
    # FLV: Low flow volume error (bottom 30% of flows, log-transformed)
    idx_70pct_start = int(0.70 * n)
    lfv_obs = obs_sorted[idx_70pct_start:]
    lfv_sim = sim_sorted[idx_70pct_start:]
    # Use log for low flows to give them more weight
    if np.all(lfv_obs > 0) and np.all(lfv_sim > 0):
        log_lfv_obs = np.sum(np.log(lfv_obs + 1))
        log_lfv_sim = np.sum(np.log(np.maximum(lfv_sim, 0.001) + 1))
        metrics['FLV'] = 100 * (log_lfv_sim - log_lfv_obs) / log_lfv_obs if log_lfv_obs > 0 else np.nan
    else:
        metrics['FLV'] = np.nan
    
    # FDC slope (mid-section: 33-66% exceedance)
    idx_33 = int(0.33 * n)
    idx_66 = int(0.66 * n)
    if idx_66 > idx_33:
        exc_mid = np.linspace(33, 66, idx_66 - idx_33)
        obs_mid = obs_sorted[idx_33:idx_66]
        sim_mid = sim_sorted[idx_33:idx_66]
        
        # FDC slope in log space
        if np.all(obs_mid > 0) and np.all(sim_mid > 0):
            log_obs_mid = np.log10(obs_mid)
            log_sim_mid = np.log10(sim_mid)
            slope_obs = (log_obs_mid[-1] - log_obs_mid[0]) / (exc_mid[-1] - exc_mid[0]) if len(exc_mid) > 1 else np.nan
            slope_sim = (log_sim_mid[-1] - log_sim_mid[0]) / (exc_mid[-1] - exc_mid[0]) if len(exc_mid) > 1 else np.nan
            metrics['FDC_slope_obs'] = slope_obs
            metrics['FDC_slope_sim'] = slope_sim
            metrics['FDC_slope_error'] = 100 * (slope_sim - slope_obs) / abs(slope_obs) if slope_obs != 0 else np.nan
        else:
            metrics['FDC_slope_obs'] = np.nan
            metrics['FDC_slope_sim'] = np.nan
            metrics['FDC_slope_error'] = np.nan
    
    # ==========================================================================
    # Hydrologic signatures (as percent errors: 100 * (sim - obs) / obs)
    # ==========================================================================
    # Runoff coefficient error
    runoff_obs = np.sum(obs_v)
    runoff_sim = np.sum(sim_v)
    metrics['Runoff_coef_error'] = 100 * (runoff_sim - runoff_obs) / runoff_obs if runoff_obs > 0 else np.nan
    
    # Mean flow error
    mean_obs = np.mean(obs_v)
    mean_sim = np.mean(sim_v)
    metrics['Mean_flow_error'] = 100 * (mean_sim - mean_obs) / mean_obs if mean_obs > 0 else np.nan
    
    # Coefficient of variation error
    cv_obs = np.std(obs_v) / np.mean(obs_v) if np.mean(obs_v) > 0 else np.nan
    cv_sim = np.std(sim_v) / np.mean(sim_v) if np.mean(sim_v) > 0 else np.nan
    metrics['CV_error'] = 100 * (cv_sim - cv_obs) / cv_obs if cv_obs > 0 and not np.isnan(cv_obs) else np.nan
    
    # High flow frequency error (days above mean)
    high_flow_threshold = np.mean(obs_v)
    hf_obs = np.sum(obs_v > high_flow_threshold) / len(obs_v)
    hf_sim = np.sum(sim_v > high_flow_threshold) / len(sim_v)
    metrics['High_flow_freq_error'] = 100 * (hf_sim - hf_obs) / hf_obs if hf_obs > 0 else np.nan
    
    # Low flow frequency error (days below Q90)
    low_flow_threshold = q90_obs
    lf_obs = np.sum(obs_v < low_flow_threshold) / len(obs_v)
    lf_sim = np.sum(sim_v < low_flow_threshold) / len(sim_v)
    metrics['Low_flow_freq_error'] = 100 * (lf_sim - lf_obs) / lf_obs if lf_obs > 0 else np.nan
    
    # Zero flow days error
    zf_obs = np.sum(obs_v <= 0) / len(obs_v) * 100
    zf_sim = np.sum(sim_v <= 0) / len(sim_v) * 100
    # For zero flow: if obs has none, report absolute difference
    if zf_obs > 0:
        metrics['Zero_flow_error'] = 100 * (zf_sim - zf_obs) / zf_obs
    else:
        metrics['Zero_flow_error'] = zf_sim - zf_obs  # Absolute difference in %
    
    # ==========================================================================
    # Signature-based metrics (percent error from observed)
    # ==========================================================================
    # Sig_BFI: Baseflow Index error
    # Simple baseflow separation using Eckhardt filter approximation
    def baseflow_index(flow):
        """Simple BFI using min-max approach over 5-day blocks."""
        if len(flow) < 5:
            return np.nan
        block_size = 5
        n_blocks = len(flow) // block_size
        if n_blocks == 0:
            return np.nan
        baseflow = np.zeros(len(flow))
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            baseflow[start:end] = np.min(flow[start:end])
        # Handle remainder
        if len(flow) % block_size > 0:
            baseflow[n_blocks * block_size:] = np.min(flow[n_blocks * block_size:])
        return np.sum(baseflow) / np.sum(flow) if np.sum(flow) > 0 else np.nan
    
    bfi_obs = baseflow_index(obs_v)
    bfi_sim = baseflow_index(sim_v)
    if not np.isnan(bfi_obs) and bfi_obs > 0:
        metrics['Sig_BFI'] = 100 * (bfi_sim - bfi_obs) / bfi_obs
    else:
        metrics['Sig_BFI'] = np.nan
    
    # Sig_Flash: Flashiness Index error (Richards-Baker)
    def flashiness_index(flow):
        """Richards-Baker flashiness index."""
        if len(flow) < 2:
            return np.nan
        path_length = np.sum(np.abs(np.diff(flow)))
        total_flow = np.sum(flow)
        return path_length / total_flow if total_flow > 0 else np.nan
    
    flash_obs = flashiness_index(obs_v)
    flash_sim = flashiness_index(sim_v)
    if not np.isnan(flash_obs) and flash_obs > 0:
        metrics['Sig_Flash'] = 100 * (flash_sim - flash_obs) / flash_obs
    else:
        metrics['Sig_Flash'] = np.nan
    
    # Sig_Q95: Low flow signature (Q95 = flow exceeded 95% of time)
    q95_pct_obs = np.percentile(obs_v, 5)  # Q95 is 5th percentile (exceeded 95% of time)
    q95_pct_sim = np.percentile(sim_v, 5)
    if q95_pct_obs > 0:
        metrics['Sig_Q95'] = 100 * (q95_pct_sim - q95_pct_obs) / q95_pct_obs
    else:
        metrics['Sig_Q95'] = np.nan
    
    # Sig_Q5: High flow signature (Q5 = flow exceeded 5% of time)
    q5_pct_obs = np.percentile(obs_v, 95)  # Q5 is 95th percentile (exceeded 5% of time)
    q5_pct_sim = np.percentile(sim_v, 95)
    if q5_pct_obs > 0:
        metrics['Sig_Q5'] = 100 * (q5_pct_sim - q5_pct_obs) / q5_pct_obs
    else:
        metrics['Sig_Q5'] = np.nan
    
    return metrics


def _get_metric_color(value: float, metric_name: str) -> str:
    """Get color code for metric value (green=good, red=bad)."""
    if np.isnan(value):
        return '#888888'
    
    # Efficiency metrics (higher is better, 1.0 is perfect)
    # Include all NSE variants with their display names
    efficiency_metrics = [
        'NSE', 'KGE', 'NSE_log', 'NSE_sqrt', 'NSE_inv',
        'NSE (high flows)', 'NSE_log (low flows)', 'NSE_sqrt (balanced)', 'NSE_inv (very low flows)',
        'KGE_log', 'KGE_sqrt', 'KGE_inv', 'KGE(log)', 'KGE(√Q)', 'KGE(1/Q)',
        'KGE_r', 'r', 'R2', 'R²', 'r (corr)', 'R² (variance explained)'
    ]
    
    # Check if metric name contains any efficiency metric substring
    is_efficiency = any(eff in metric_name for eff in ['NSE', 'KGE', 'R²', 'R2'])
    # But not if it's an error metric
    is_error = 'error' in metric_name.lower() or 'Error' in metric_name
    
    if metric_name in efficiency_metrics or (is_efficiency and not is_error and 'α' not in metric_name and 'β' not in metric_name):
        if value >= 0.75:
            return '#27ae60'  # Green - excellent
        elif value >= 0.5:
            return '#f39c12'  # Orange - good
        elif value >= 0.0:
            return '#e67e22'  # Dark orange - acceptable
        else:
            return '#e74c3c'  # Red - poor
    
    # PBIAS and percent errors (closer to 0 is better)
    elif 'PBIAS' in metric_name or 'bias' in metric_name.lower() or 'error' in metric_name.lower():
        if abs(value) <= 10:
            return '#27ae60'  # Green
        elif abs(value) <= 25:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red
    
    # KGE components alpha and beta (closer to 1 is better)
    elif metric_name in ['KGE_alpha', 'KGE_beta', 'α', 'β', 'α (var)', 'β (bias)'] or '  α' in metric_name or '  β' in metric_name:
        if 0.8 <= value <= 1.2:
            return '#27ae60'
        elif 0.5 <= value <= 1.5:
            return '#f39c12'
        else:
            return '#e74c3c'
    
    return '#2c3e50'  # Default


# =============================================================================
# Matplotlib Functions
# =============================================================================

def plot_report_card_matplotlib(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (20, 24),
    dark_theme: bool = False
) -> 'Figure':
    """
    Generate a comprehensive matplotlib report card figure.
    
    Layout (2 columns):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  HEADER: Catchment Name, Method, Objective, Period                          │
    ├─────────────────────────────────┬───────────────────────────────────────────┤
    │                                 │                                           │
    │  Hydrograph (Linear Scale)      │   COMPREHENSIVE METRICS TABLE             │
    │                                 │                                           │
    ├─────────────────────────────────┤   - NSE Variants                          │
    │                                 │   - KGE (Q) + components                  │
    │  Hydrograph (Log Scale)         │   - KGE (log Q) + components              │
    │                                 │   - KGE (√Q) + components                 │
    ├─────────────────────────────────┤   - KGE (1/Q) + components                │
    │                                 │   - Error Metrics                         │
    │  Flow Duration Curve            │   - FDC Segmented Flow Errors             │
    │                                 │   - Hydrologic Signatures                 │
    ├─────────────────────────────────┤                                           │
    │                                 │                                           │
    │  Scatter Plot                   │                                           │
    │                                 │                                           │
    ├─────────────────────────────────┤                                           │
    │                                 │                                           │
    │  Parameter Bounds Chart         │                                           │
    │                                 │                                           │
    └─────────────────────────────────┴───────────────────────────────────────────┘
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size (width, height)
        dark_theme: Use dark background theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    bg_color = COLORS_DARK['background'] if dark_theme else '#ffffff'
    panel_color = COLORS_DARK['panel'] if dark_theme else '#f8f9fa'
    text_color = COLORS_DARK['text'] if dark_theme else '#2c3e50'
    
    # Create figure with 2-column layout
    # Left column: 5 plots stacked vertically
    # Right column: Single metrics table spanning full height
    fig = plt.figure(figsize=figsize)
    
    # Main grid: header row + content row
    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.04, 0.96], hspace=0.02)
    
    # Content grid: 2 columns (plots | table)
    gs_content = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], 
                                                   width_ratios=[1.1, 0.9], wspace=0.08)
    
    # Left column: 5 plots
    gs_plots = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_content[0], 
                                                 height_ratios=[1, 1, 1, 1, 1.2], hspace=0.25)
    
    fig.patch.set_facecolor(bg_color)
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    # Calculate comprehensive metrics
    metrics = _calculate_basic_metrics(obs, sim)
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    ax_header = fig.add_subplot(gs_main[0])
    ax_header.axis('off')
    ax_header.set_facecolor(bg_color)
    
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    area = report.catchment_info.get('area_km2', '')
    
    title_text = f"{catchment_name}"
    if gauge_id:
        title_text += f" ({gauge_id})"
    
    ax_header.text(0.5, 0.65, title_text, transform=ax_header.transAxes,
                   fontsize=18, fontweight='bold', ha='center', va='center',
                   color=text_color)
    
    subtitle = f"Method: {report.result.method} | Objective: {report.result.objective_name}"
    subtitle += f" | Best: {report.result.best_objective:.4f}"
    if area:
        subtitle += f" | Area: {area} km²"
    subtitle += f" | Period: {report.calibration_period[0]} to {report.calibration_period[1]}"
    
    ax_header.text(0.5, 0.15, subtitle, transform=ax_header.transAxes,
                   fontsize=10, ha='center', va='center', color=text_color, alpha=0.8)
    
    # ==========================================================================
    # LEFT COLUMN - PLOTS
    # ==========================================================================
    
    # Plot 1: Hydrograph (Linear Scale)
    ax_hydro = fig.add_subplot(gs_plots[0])
    ax_hydro.set_facecolor(panel_color)
    ax_hydro.plot(dates, obs, color=colors['observed'], linewidth=0.7, 
                  label='Observed', alpha=0.9)
    ax_hydro.plot(dates, sim, color=colors['simulated'], linewidth=0.7, 
                  label='Simulated', alpha=0.9)
    ax_hydro.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_hydro.set_title('Hydrograph (Linear Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_hydro.legend(loc='upper right', fontsize=8)
    ax_hydro.grid(True, alpha=0.3)
    ax_hydro.tick_params(labelsize=8)
    ax_hydro.set_xticklabels([])
    
    # Plot 2: Hydrograph (Log Scale)
    ax_hydro_log = fig.add_subplot(gs_plots[1])
    ax_hydro_log.set_facecolor(panel_color)
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    ax_hydro_log.semilogy(dates, obs_log, color=colors['observed'], linewidth=0.7,
                          label='Observed', alpha=0.9)
    ax_hydro_log.semilogy(dates, sim_log, color=colors['simulated'], linewidth=0.7,
                          label='Simulated', alpha=0.9)
    ax_hydro_log.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_hydro_log.set_title('Hydrograph (Log Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_hydro_log.legend(loc='upper right', fontsize=8)
    ax_hydro_log.grid(True, alpha=0.3, which='both')
    ax_hydro_log.tick_params(labelsize=8)
    ax_hydro_log.set_xticklabels([])
    
    # Plot 3: Flow Duration Curve
    ax_fdc = fig.add_subplot(gs_plots[2])
    ax_fdc.set_facecolor(panel_color)
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    ax_fdc.semilogy(exc_obs, obs_sorted, color=colors['observed'], linewidth=1.5,
                    label='Observed')
    ax_fdc.semilogy(exc_sim, sim_sorted, color=colors['simulated'], linewidth=1.5,
                    linestyle='--', label='Simulated')
    ax_fdc.set_xlabel('Exceedance (%)', fontsize=9)
    ax_fdc.set_ylabel('Flow (ML/day)', fontsize=9)
    ax_fdc.set_title('Flow Duration Curve', fontweight='bold', fontsize=10, pad=3)
    ax_fdc.legend(loc='upper right', fontsize=8)
    ax_fdc.grid(True, alpha=0.3, which='both')
    ax_fdc.set_xlim(0, 100)
    ax_fdc.tick_params(labelsize=8)
    
    # Plot 4: Scatter Plot (Log-Log scale)
    ax_scatter = fig.add_subplot(gs_plots[3])
    ax_scatter.set_facecolor(panel_color)
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    # Filter for positive values only (required for log scale)
    pos_mask = (obs_valid > 0) & (sim_valid > 0)
    obs_pos = obs_valid[pos_mask]
    sim_pos = sim_valid[pos_mask]
    
    ax_scatter.scatter(obs_pos, sim_pos, alpha=0.3, s=8, color=colors['simulated'])
    
    # 1:1 line for log scale
    min_val = max(np.nanmin(obs_pos), np.nanmin(sim_pos), 0.01)
    max_val = max(np.nanmax(obs_pos), np.nanmax(sim_pos))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], '--', color=colors['one_to_one'],
                    linewidth=1.5, label='1:1 Line')
    
    ax_scatter.set_xlabel('Observed (ML/day)', fontsize=9)
    ax_scatter.set_ylabel('Simulated (ML/day)', fontsize=9)
    ax_scatter.set_title('Scatter Plot (Log Scale)', fontweight='bold', fontsize=10, pad=3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.legend(loc='upper left', fontsize=8)
    ax_scatter.grid(True, alpha=0.3, which='both')
    ax_scatter.tick_params(labelsize=8)
    r_squared = metrics.get('R2', np.nan)
    if not np.isnan(r_squared):
        ax_scatter.text(0.95, 0.05, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes,
                       fontsize=9, ha='right', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 5: Parameter Bounds Chart
    ax_bounds = fig.add_subplot(gs_plots[4])
    ax_bounds.set_facecolor(panel_color)
    
    if report.parameter_bounds:
        param_names = list(report.result.best_parameters.keys())
        norm_values = []
        actual_values = []
        bar_colors = []
        
        for param in param_names:
            value = report.result.best_parameters[param]
            actual_values.append(value)
            if param in report.parameter_bounds:
                low, high = report.parameter_bounds[param]
                norm = (value - low) / (high - low) * 100 if high > low else 50
                norm = np.clip(norm, 0, 100)
                norm_values.append(norm)
                # Red ONLY if within the shaded regions (0-10% or 90-100%)
                if norm <= 10 or norm >= 90:
                    bar_colors.append('#e74c3c')  # Red - inside shaded warning zone
                else:
                    bar_colors.append(colors['simulated'])  # Normal blue
            else:
                norm_values.append(50)
                bar_colors.append(colors['simulated'])
        
        y_pos_bounds = np.arange(len(param_names))
        
        # Add shaded warning regions (0-10% and 90-100%)
        ax_bounds.axvspan(0, 10, alpha=0.15, color='#e74c3c', zorder=0)
        ax_bounds.axvspan(90, 100, alpha=0.15, color='#e74c3c', zorder=0)
        
        bars = ax_bounds.barh(y_pos_bounds, norm_values, color=bar_colors, alpha=0.8, height=0.6, zorder=2)
        ax_bounds.axvline(x=50, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
        ax_bounds.axvline(x=10, linestyle=':', color='#e74c3c', alpha=0.5, linewidth=1, zorder=1)
        ax_bounds.axvline(x=90, linestyle=':', color='#e74c3c', alpha=0.5, linewidth=1, zorder=1)
        
        for bar, val, norm in zip(bars, actual_values, norm_values):
            label_x = norm + 2 if norm < 85 else norm - 3
            ha = 'left' if norm < 85 else 'right'
            ax_bounds.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', ha=ha, fontsize=7, fontweight='bold', zorder=3)
        
        ax_bounds.set_yticks(y_pos_bounds)
        ax_bounds.set_yticklabels(param_names, fontsize=7)
        ax_bounds.set_xlabel('Position within Bounds (%)', fontsize=9)
        ax_bounds.set_xlim(-5, 105)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=10, pad=3)
        ax_bounds.invert_yaxis()
        ax_bounds.tick_params(labelsize=8)
        
        # Add legend for shaded regions
        ax_bounds.text(0.5, -0.08, 'Shaded regions: parameters within 10% of bounds (may be constrained)',
                      transform=ax_bounds.transAxes, fontsize=7, ha='center', va='top', 
                      color='#e74c3c', style='italic')
    else:
        ax_bounds.text(0.5, 0.5, 'No parameter bounds available', ha='center', va='center',
                       transform=ax_bounds.transAxes, fontsize=10)
        ax_bounds.set_title('Calibrated Parameters vs Bounds', fontweight='bold', fontsize=10, pad=3)
    
    # ==========================================================================
    # RIGHT COLUMN - METRICS TABLE
    # ==========================================================================
    ax_table = fig.add_subplot(gs_content[1])
    ax_table.axis('off')
    ax_table.set_facecolor(bg_color)
    
    # Build comprehensive metrics data for the table
    # Format: (Metric Name, Value, Type for coloring)
    def fmt_value(val, metric_type):
        """Format value based on type."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        if metric_type == 'bias':
            return f'{val:+.2f}%'
        elif metric_type == 'error':
            return f'{val:.2f}'
        elif metric_type == 'ratio':
            return f'{val:.3f}'
        elif metric_type == 'percent':
            return f'{val:.1f}%'
        else:
            return f'{val:.4f}'
    
    # Build table data: list of (section_header, [(name, value, type), ...])
    # Flow regime clarifications added in parentheses
    table_sections = [
        ('NSE Variants (Flow Regime Focus)', [
            ('NSE (high flows)', metrics.get('NSE', np.nan), 'efficiency'),
            ('NSE_log (low flows)', metrics.get('NSE_log', np.nan), 'efficiency'),
            ('NSE_sqrt (balanced)', metrics.get('NSE_sqrt', np.nan), 'efficiency'),
            ('NSE_inv (very low flows)', metrics.get('NSE_inv', np.nan), 'efficiency'),
        ]),
        ('KGE (Q) - High Flows', [
            ('KGE', metrics.get('KGE', np.nan), 'efficiency'),
            ('  r (correlation)', metrics.get('KGE_r', np.nan), 'efficiency'),
            ('  α (variability)', metrics.get('KGE_alpha', np.nan), 'ratio'),
            ('  β (bias)', metrics.get('KGE_beta', np.nan), 'ratio'),
        ]),
        ('KGE (log Q) - Low Flows', [
            ('KGE(log)', metrics.get('KGE_log', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_log_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_log_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_log_beta', np.nan), 'ratio'),
        ]),
        ('KGE (√Q) - Balanced', [
            ('KGE(√Q)', metrics.get('KGE_sqrt', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_sqrt_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_sqrt_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_sqrt_beta', np.nan), 'ratio'),
        ]),
        ('KGE (1/Q) - Very Low Flows', [
            ('KGE(1/Q)', metrics.get('KGE_inv', np.nan), 'efficiency'),
            ('  r', metrics.get('KGE_inv_r', np.nan), 'efficiency'),
            ('  α', metrics.get('KGE_inv_alpha', np.nan), 'ratio'),
            ('  β', metrics.get('KGE_inv_beta', np.nan), 'ratio'),
        ]),
        ('Error Metrics (Overall)', [
            ('R² (variance explained)', metrics.get('R2', np.nan), 'efficiency'),
            ('RMSE (high flow errors)', metrics.get('RMSE', np.nan), 'error'),
            ('MAE (balanced errors)', metrics.get('MAE', np.nan), 'error'),
            ('PBIAS (volume bias %)', metrics.get('PBIAS', np.nan), 'bias'),
        ]),
        ('FDC Segmented Flow Errors (%)', [
            ('Q5 (peak flows)', metrics.get('Q5_bias', np.nan), 'bias'),
            ('Q10 (high flows)', metrics.get('Q10_bias', np.nan), 'bias'),
            ('Q50 (median flows)', metrics.get('Q50_bias', np.nan), 'bias'),
            ('Q90 (low flows)', metrics.get('Q90_bias', np.nan), 'bias'),
            ('Q95 (very low flows)', metrics.get('Q95_bias', np.nan), 'bias'),
            ('FDC slope (regime shape)', metrics.get('FDC_slope_error', np.nan), 'bias'),
        ]),
        ('Hydrologic Signature Errors (%)', [
            ('Runoff coef (water balance)', metrics.get('Runoff_coef_error', np.nan), 'bias'),
            ('Mean flow (volume bias)', metrics.get('Mean_flow_error', np.nan), 'bias'),
            ('CV (variability)', metrics.get('CV_error', np.nan), 'bias'),
            ('High flow freq (>mean)', metrics.get('High_flow_freq_error', np.nan), 'bias'),
            ('Low flow freq (<Q90)', metrics.get('Low_flow_freq_error', np.nan), 'bias'),
            ('Zero flow days', metrics.get('Zero_flow_error', np.nan), 'bias'),
        ]),
    ]
    
    # Build table rows
    cell_text = []
    row_colors = []
    
    for section_header, section_metrics in table_sections:
        # Add section header row
        cell_text.append([section_header, ''])
        row_colors.append('header')
        
        # Add metric rows
        for name, val, mtype in section_metrics:
            val_str = fmt_value(val, mtype)
            cell_text.append([name, val_str])
            row_colors.append(mtype)
    
    # Create the table
    table = ax_table.table(
        cellText=cell_text,
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        colLoc='left',
        loc='upper center',
        colWidths=[0.65, 0.35]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    
    # Style header row
    for col in range(2):
        cell = table[(0, col)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_height(0.03)
    
    # Style data rows
    for row_idx, (row_type, (text_row)) in enumerate(zip(row_colors, cell_text), start=1):
        for col in range(2):
            cell = table[(row_idx, col)]
            
            if row_type == 'header':
                # Section header
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white', fontweight='bold', fontsize=9)
            else:
                # Data row - alternate colors
                bg = '#f8f9fa' if row_idx % 2 == 0 else '#ffffff'
                cell.set_facecolor(bg)
                
                # Color the value cell based on metric type
                if col == 1:
                    val_text = text_row[1]
                    metric_name = text_row[0].strip()
                    if val_text != 'N/A':
                        try:
                            # Parse value for coloring
                            val_clean = val_text.replace('%', '').replace('+', '')
                            val_num = float(val_clean)
                            color = _get_metric_color(val_num, metric_name)
                            cell.set_text_props(color=color, fontweight='bold')
                        except:
                            pass
    
    # Add title above table
    ax_table.text(0.5, 1.0, 'Comprehensive Performance Metrics', 
                  transform=ax_table.transAxes,
                  fontsize=12, fontweight='bold', ha='center', va='bottom',
                  color=text_color)
    
    # Add color legends at TOP (between title and table)
    # Legend for efficiency metrics (NSE, KGE, R²)
    ax_table.text(0.02, 0.975, 'Efficiency:', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color=text_color, fontweight='bold')
    ax_table.text(0.15, 0.975, '● ≥0.75', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#27ae60', fontweight='bold')
    ax_table.text(0.27, 0.975, '● ≥0.50', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#f39c12', fontweight='bold')
    ax_table.text(0.39, 0.975, '● <0.50', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#e74c3c', fontweight='bold')
    
    # Legend for percent errors (bias, FDC errors, signature errors)
    ax_table.text(0.54, 0.975, '% Errors:', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color=text_color, fontweight='bold')
    ax_table.text(0.67, 0.975, '● ≤±10%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#27ae60', fontweight='bold')
    ax_table.text(0.79, 0.975, '● ≤±25%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#f39c12', fontweight='bold')
    ax_table.text(0.91, 0.975, '● >±25%', transform=ax_table.transAxes,
                  fontsize=8, ha='left', va='top', color='#e74c3c', fontweight='bold')
    
    if dark_theme:
        _reset_style()
    
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.06, right=0.98)
    return fig


def plot_hydrograph_comparison(
    report: 'CalibrationReport',
    log_scale: bool = False,
    figsize: Tuple[int, int] = (14, 6),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot observed vs simulated hydrograph.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    if log_scale:
        obs = np.where(obs > 0, obs, np.nan)
        sim = np.where(sim > 0, sim, np.nan)
        ax.semilogy(dates, obs, color=colors['observed'], linewidth=1,
                    label='Observed', alpha=0.9)
        ax.semilogy(dates, sim, color=colors['simulated'], linewidth=1,
                    label='Simulated', alpha=0.9)
    else:
        ax.plot(dates, obs, color=colors['observed'], linewidth=1,
                label='Observed', alpha=0.9)
        ax.plot(dates, sim, color=colors['simulated'], linewidth=1,
                label='Simulated', alpha=0.9)
    
    ax.set_ylabel('Flow (ML/day)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    scale_label = 'Log Scale' if log_scale else 'Linear Scale'
    ax.set_title(f'Hydrograph Comparison ({scale_label})', fontweight='bold')
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_fdc_comparison(
    report: 'CalibrationReport',
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot flow duration curves for observed and simulated.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    if log_scale:
        ax.semilogy(exc_obs, obs_sorted, color=colors['observed'], linewidth=2,
                    label='Observed')
        ax.semilogy(exc_sim, sim_sorted, color=colors['simulated'], linewidth=2,
                    linestyle='--', label='Simulated')
    else:
        ax.plot(exc_obs, obs_sorted, color=colors['observed'], linewidth=2,
                label='Observed')
        ax.plot(exc_sim, sim_sorted, color=colors['simulated'], linewidth=2,
                linestyle='--', label='Simulated')
    
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Flow (ML/day)')
    ax.set_title('Flow Duration Curve', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
    ax.set_xlim(0, 100)
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_scatter_comparison(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (8, 8),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot observed vs simulated scatter plot with 1:1 line.
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    obs = report.observed
    sim = report.simulated
    
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    ax.scatter(obs_valid, sim_valid, alpha=0.3, s=10, color=colors['simulated'])
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    min_val = min(np.nanmin(obs_valid), np.nanmin(sim_valid))
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=colors['one_to_one'],
            linewidth=2, label='1:1 Line')
    
    # Add regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], p([min_val, max_val]), ':',
            color=colors['observed'], linewidth=2, label='Regression')
    
    ax.set_xlabel('Observed Flow (ML/day)')
    ax.set_ylabel('Simulated Flow (ML/day)')
    ax.set_title('Observed vs Simulated', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add metrics annotation
    try:
        metrics = report.calculate_metrics()
        metrics_text = f"NSE: {metrics['NSE']:.3f}\nKGE: {metrics['KGE']:.3f}"
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception:
        pass
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


def plot_parameter_bounds_chart(
    report: 'CalibrationReport',
    figsize: Tuple[int, int] = (10, 8),
    dark_theme: bool = False
) -> 'Figure':
    """
    Plot calibrated parameters as percentage of their bounds.
    
    Args:
        report: CalibrationReport instance
        figsize: Figure size
        dark_theme: Use dark theme
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function")
    
    if not report.parameter_bounds:
        raise ValueError("No parameter bounds available in report")
    
    if dark_theme:
        _apply_dark_style()
    
    colors = COLORS_DARK if dark_theme else COLORS_LIGHT
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor(COLORS_DARK['background'])
        ax.set_facecolor(COLORS_DARK['panel'])
    
    param_names = list(report.result.best_parameters.keys())
    values = []
    norm_values = []
    
    for param in param_names:
        value = report.result.best_parameters[param]
        values.append(value)
        
        if param in report.parameter_bounds:
            low, high = report.parameter_bounds[param]
            norm = (value - low) / (high - low) * 100 if high > low else 50
            norm_values.append(np.clip(norm, 0, 100))
        else:
            norm_values.append(50)
    
    y_pos = np.arange(len(param_names))
    bars = ax.barh(y_pos, norm_values, color=colors['simulated'], alpha=0.7)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        label_x = width + 2 if width < 90 else width - 5
        ha = 'left' if width < 90 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', ha=ha, fontsize=8)
    
    ax.axvline(x=50, linestyle='--', color=colors['observed'], alpha=0.7,
               label='50% (middle of bounds)')
    ax.axvline(x=0, linestyle='-', color='gray', alpha=0.3)
    ax.axvline(x=100, linestyle='-', color='gray', alpha=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Position within Parameter Bounds (%)')
    ax.set_xlim(-5, 110)
    ax.set_title('Calibrated Parameters vs Bounds', fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    if dark_theme:
        _reset_style()
    
    plt.tight_layout()
    return fig


# =============================================================================
# Plotly Functions
# =============================================================================

def plot_report_card_components(
    report: 'CalibrationReport',
) -> dict:
    """
    Generate separate Plotly figures for each report card component.
    
    This approach returns independent figures that can be rendered sequentially,
    avoiding the layout complexity issues of a single combined figure.
    
    Args:
        report: CalibrationReport instance
        
    Returns:
        Dictionary with keys:
        - 'header': dict with catchment_name, gauge_id, subtitle info
        - 'hydrograph_linear': Plotly Figure for linear-scale hydrograph
        - 'hydrograph_log': Plotly Figure for log-scale hydrograph
        - 'metrics_table': Plotly Figure with diagnostic metrics table
        - 'fdc': Plotly Figure for flow duration curve
        - 'scatter': Plotly Figure for scatter plot
        - 'parameters': Plotly Figure for parameter bounds chart
        - 'signatures_table': Plotly Figure with hydrologic signatures table
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function. Install with: pip install plotly")
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    # Calculate comprehensive metrics
    metrics = _calculate_basic_metrics(obs, sim)
    
    # Compute hydrologic signatures for observed and simulated
    try:
        from pyrrm.analysis.signatures import compute_all_signatures, signature_percent_error
        obs_sigs = compute_all_signatures(obs, dates)
        sim_sigs = compute_all_signatures(sim, dates)
        pct_errors = {}
        for sig_name in obs_sigs:
            pct_errors[sig_name] = signature_percent_error(
                obs_sigs[sig_name],
                sim_sigs.get(sig_name, np.nan),
            )
        has_signatures = True
    except Exception:
        has_signatures = False
        obs_sigs = {}
        sim_sigs = {}
        pct_errors = {}
    
    # Colors
    color_obs = '#e74c3c'
    color_sim = '#3498db'
    color_excellent = '#27ae60'
    color_good = '#f39c12'
    color_poor = '#e74c3c'
    
    # Common layout settings
    common_layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        font=dict(family='Arial, sans-serif'),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    
    result = {}
    
    # ==========================================================================
    # Header info
    # ==========================================================================
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    area = report.catchment_info.get('area_km2', '')
    
    result['header'] = {
        'catchment_name': catchment_name,
        'gauge_id': gauge_id,
        'area': area,
        'method': report.result.method,
        'objective_name': report.result.objective_name,
        'best_objective': report.result.best_objective,
        'period_start': str(report.calibration_period[0]),
        'period_end': str(report.calibration_period[1]),
    }
    
    # ==========================================================================
    # Hydrograph (Linear Scale)
    # ==========================================================================
    fig_hydro = go.Figure()
    fig_hydro.add_trace(go.Scatter(x=dates, y=obs, name='Observed', line=dict(color=color_obs, width=1)))
    fig_hydro.add_trace(go.Scatter(x=dates, y=sim, name='Simulated', line=dict(color=color_sim, width=1)))
    fig_hydro.update_layout(
        title='Hydrograph (Linear Scale)',
        yaxis_title='Flow (ML/day)',
        height=300,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        **common_layout
    )
    result['hydrograph_linear'] = fig_hydro
    
    # ==========================================================================
    # Hydrograph (Log Scale)
    # ==========================================================================
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    fig_hydro_log = go.Figure()
    fig_hydro_log.add_trace(go.Scatter(x=dates, y=obs_log, name='Observed', line=dict(color=color_obs, width=1)))
    fig_hydro_log.add_trace(go.Scatter(x=dates, y=sim_log, name='Simulated', line=dict(color=color_sim, width=1)))
    fig_hydro_log.update_layout(
        title='Hydrograph (Log Scale)',
        yaxis_title='Flow (ML/day)',
        yaxis_type='log',
        height=300,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        **common_layout
    )
    result['hydrograph_log'] = fig_hydro_log
    
    # ==========================================================================
    # Diagnostic Metrics Table
    # ==========================================================================
    HEADLINE_METRICS = [
        "NSE", "NSE_sqrt", "NSE_log", "NSE_inv",
        "KGE", "KGE_sqrt", "KGE_log", "KGE_inv",
        "KGE_np", "KGE_np_sqrt", "KGE_np_log", "KGE_np_inv",
        "RMSE", "MAE", "SDEB",
        "PBIAS", "FHV", "FMV", "FLV",
        "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5",
    ]
    EFFICIENCY_METRICS = {"NSE", "NSE_sqrt", "NSE_log", "NSE_inv", "KGE", "KGE_sqrt", "KGE_log", "KGE_inv",
                         "KGE_np", "KGE_np_sqrt", "KGE_np_log", "KGE_np_inv"}
    ERROR_METRICS = {"RMSE", "MAE", "SDEB"}
    PERCENT_ERROR_METRICS = {"PBIAS", "FHV", "FMV", "FLV", "Sig_BFI", "Sig_Flash", "Sig_Q95", "Sig_Q5"}
    METRIC_DESCRIPTIONS = {
        "NSE": "Nash-Sutcliffe (high flows)", "NSE_sqrt": "NSE on √Q (balanced)",
        "NSE_log": "NSE on log(Q) (low flows)", "NSE_inv": "NSE on 1/Q (very low flows)",
        "KGE": "Kling-Gupta (high flows)", "KGE_sqrt": "KGE on √Q (balanced)",
        "KGE_log": "KGE on log(Q) (low flows)", "KGE_inv": "KGE on 1/Q (very low flows)",
        "KGE_np": "KGE non-parametric (high flows)", "KGE_np_sqrt": "KGE_np on √Q (balanced)",
        "KGE_np_log": "KGE_np on log(Q) (low flows)", "KGE_np_inv": "KGE_np on 1/Q (very low flows)",
        "RMSE": "Root Mean Square Error", "MAE": "Mean Absolute Error", "SDEB": "Spectral Decomp. Error Bias",
        "PBIAS": "Volume Bias (%)", "FHV": "High Flow Volume Error (%)",
        "FMV": "Mid Flow Volume Error (%)", "FLV": "Low Flow Volume Error (%)",
        "Sig_BFI": "Baseflow Index Error (%)", "Sig_Flash": "Flashiness Error (%)",
        "Sig_Q95": "Q95 (low flow) Error (%)", "Sig_Q5": "Q5 (high flow) Error (%)",
    }
    
    def get_metric_color(value, metric_name):
        if np.isnan(value):
            return '#888888'
        if metric_name in EFFICIENCY_METRICS:
            if value >= 0.7: return color_excellent
            elif value >= 0.5: return color_good
            else: return color_poor
        if metric_name in ERROR_METRICS:
            return '#2c3e50'
        if metric_name in PERCENT_ERROR_METRICS:
            if abs(value) <= 10: return color_excellent
            elif abs(value) <= 20: return color_good
            else: return color_poor
        return '#2c3e50'
    
    metric_names, metric_values, metric_descs, cell_colors = [], [], [], []
    for metric in HEADLINE_METRICS:
        val = metrics.get(metric, np.nan)
        metric_names.append(metric)
        metric_descs.append(METRIC_DESCRIPTIONS.get(metric, ""))
        if np.isnan(val):
            metric_values.append("N/A")
            cell_colors.append('#f8f9fa')
        elif metric in PERCENT_ERROR_METRICS:
            metric_values.append(f"{val:+.2f}%")
            cell_colors.append(get_metric_color(val, metric))
        elif metric in ERROR_METRICS:
            metric_values.append(f"{val:.2f}")
            cell_colors.append('#f8f9fa')
        else:
            metric_values.append(f"{val:.4f}")
            cell_colors.append(get_metric_color(val, metric))
    
    fig_metrics = go.Figure(go.Table(
        header=dict(values=['<b>Metric</b>', '<b>Value</b>', '<b>Description</b>'],
                    fill_color='#34495e', font=dict(color='white', size=11), align='left', height=28),
        cells=dict(values=[metric_names, metric_values, metric_descs],
                   fill_color=['#f8f9fa', cell_colors, '#f8f9fa'],
                   font=dict(color=[['#2c3e50']*len(metric_names),
                                    ['white' if c not in ['#f8f9fa','#888888'] else '#2c3e50' for c in cell_colors],
                                    ['#7f8c8d']*len(metric_names)], size=10),
                   align='left', height=22),
        columnwidth=[0.22, 0.18, 0.60]
    ))
    fig_metrics.update_layout(title='Diagnostic Metrics', height=650, margin=dict(l=10, r=10, t=40, b=10))
    result['metrics_table'] = fig_metrics
    
    # ==========================================================================
    # Flow Duration Curve
    # ==========================================================================
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    fig_fdc = go.Figure()
    fig_fdc.add_trace(go.Scatter(x=exc_obs, y=obs_sorted, name='Observed', line=dict(color=color_obs, width=2)))
    fig_fdc.add_trace(go.Scatter(x=exc_sim, y=sim_sorted, name='Simulated', line=dict(color=color_sim, width=2, dash='dash')))
    fig_fdc.update_layout(
        title='Flow Duration Curve',
        xaxis_title='Exceedance (%)', yaxis_title='Flow (ML/day)',
        yaxis_type='log', xaxis_range=[0, 100],
        height=350,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        **common_layout
    )
    result['fdc'] = fig_fdc
    
    # ==========================================================================
    # Scatter Plot
    # ==========================================================================
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid, sim_valid = obs[valid], sim[valid]
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=obs_valid, y=sim_valid, mode='markers', name='Points',
                                      marker=dict(color=color_sim, size=4, opacity=0.3)))
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='1:1 Line',
                                      line=dict(color='green', dash='dash', width=2)))
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=p([0, max_val]), mode='lines',
                                      name=f'Fit (y={z[0]:.2f}x+{z[1]:.1f})',
                                      line=dict(color=color_obs, dash='dot', width=2)))
    fig_scatter.update_layout(
        title='Scatter Plot (Observed vs Simulated)',
        xaxis_title='Observed (ML/day)', yaxis_title='Simulated (ML/day)',
        height=400,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        **common_layout
    )
    result['scatter'] = fig_scatter
    
    # ==========================================================================
    # Parameter Bounds Chart
    # ==========================================================================
    if report.parameter_bounds:
        param_names = list(report.result.best_parameters.keys())
        norm_values, actual_values, bar_colors = [], [], []
        for param in param_names:
            value = report.result.best_parameters[param]
            actual_values.append(value)
            if param in report.parameter_bounds:
                low, high = report.parameter_bounds[param]
                norm = (value - low) / (high - low) * 100 if high > low else 50
                norm = np.clip(norm, 0, 100)
                norm_values.append(norm)
                if norm < 5 or norm > 95: bar_colors.append(color_poor)
                elif norm < 15 or norm > 85: bar_colors.append(color_good)
                else: bar_colors.append(color_sim)
            else:
                norm_values.append(50)
                bar_colors.append(color_sim)
        
        fig_params = go.Figure(go.Bar(
            y=param_names, x=norm_values, orientation='h', marker_color=bar_colors,
            text=[f'{v:.3f}' for v in actual_values], textposition='outside',
            hovertemplate='%{y}<br>Value: %{text}<br>Position: %{x:.1f}%<extra></extra>'
        ))
        fig_params.update_layout(
            title='Calibrated Parameters (Position within Bounds)',
            xaxis_title='Position within Bounds (%)',
            xaxis_range=[-5, 115], xaxis_fixedrange=True,
            yaxis_autorange='reversed', yaxis_fixedrange=True,
            height=max(300, 25 * len(param_names) + 80),
            margin=dict(l=100, r=60, t=40, b=40),
        )
        result['parameters'] = fig_params
    else:
        result['parameters'] = None
    
    # ==========================================================================
    # Hydrologic Signatures Table
    # ==========================================================================
    if has_signatures:
        try:
            from pyrrm.analysis.signatures import SIGNATURE_CATEGORIES, SIGNATURE_INFO
        except ImportError:
            SIGNATURE_CATEGORIES, SIGNATURE_INFO = {}, {}
        
        ALL_SIGNATURES = []
        for cat_name, sig_ids in SIGNATURE_CATEGORIES.items():
            for sig_id in sig_ids:
                info = SIGNATURE_INFO.get(sig_id, {})
                ALL_SIGNATURES.append((cat_name, sig_id, info.get("name", sig_id)))
        
        def fmt_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
            if abs(val) >= 1000: return f"{val:.1f}"
            elif abs(val) >= 1: return f"{val:.2f}"
            else: return f"{val:.3f}"
        
        def err_color(pct_err):
            if pct_err is None or (isinstance(pct_err, float) and np.isnan(pct_err)): return '#888888'
            if abs(pct_err) <= 10: return color_excellent
            elif abs(pct_err) <= 20: return color_good
            else: return color_poor
        
        sig_cats, sig_names, sig_obs, sig_sim, sig_errs, sig_colors = [], [], [], [], [], []
        for cat, sig_id, sig_display in ALL_SIGNATURES:
            sig_cats.append(cat)
            sig_names.append(sig_display)
            sig_obs.append(fmt_val(obs_sigs.get(sig_id)))
            sig_sim.append(fmt_val(sim_sigs.get(sig_id)))
            pct_err = pct_errors.get(sig_id)
            if pct_err is not None and not (isinstance(pct_err, float) and np.isnan(pct_err)):
                sig_errs.append(f"{pct_err:+.1f}%")
            else:
                sig_errs.append("N/A")
            sig_colors.append(err_color(pct_err))
        
        fig_sigs = go.Figure(go.Table(
            header=dict(values=['<b>Category</b>', '<b>Signature</b>', '<b>Observed</b>', '<b>Simulated</b>', '<b>% Error</b>'],
                        fill_color='#34495e', font=dict(color='white', size=10), align='left', height=26),
            cells=dict(values=[sig_cats, sig_names, sig_obs, sig_sim, sig_errs],
                       fill_color=['#f8f9fa', '#f8f9fa', '#f8f9fa', '#f8f9fa', sig_colors],
                       font=dict(color=[['#2c3e50']*len(sig_names)]*4 +
                                       [['white' if c!='#888888' else '#2c3e50' for c in sig_colors]], size=9),
                       align='left', height=18),
            columnwidth=[0.20, 0.32, 0.16, 0.16, 0.16]
        ))
        fig_sigs.update_layout(title='Hydrologic Signatures', height=1000, margin=dict(l=10, r=10, t=40, b=10))
        result['signatures_table'] = fig_sigs
    else:
        result['signatures_table'] = None
    
    return result


def plot_report_card_plotly(
    report: 'CalibrationReport',
    height: int = 2400
):
    """
    Generate an interactive Plotly report card (legacy single-figure version).
    
    NOTE: This function attempts to combine multiple components into a single figure,
    which can cause layout issues. Consider using plot_report_card_components() instead,
    which returns separate figures that render reliably.
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Use the components version and combine into a simple vertical layout
    components = plot_report_card_components(report)
    
    # For backwards compatibility, return a simple combined figure
    # Just stack the hydrographs and FDC vertically
    from plotly.subplots import make_subplots
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    color_obs = '#e74c3c'
    color_sim = '#3498db'
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Hydrograph (Linear Scale)', 'Hydrograph (Log Scale)', 
                       'Flow Duration Curve', 'Scatter Plot'),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.08
    )
    
    # Hydrograph linear
    fig.add_trace(go.Scatter(x=dates, y=obs, name='Observed', line=dict(color=color_obs, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sim, name='Simulated', line=dict(color=color_sim, width=1)), row=1, col=1)
    fig.update_yaxes(title_text="Flow (ML/day)", row=1, col=1)
    
    # Hydrograph log
    obs_log = np.where(obs > 0, obs, np.nan)
    sim_log = np.where(sim > 0, sim, np.nan)
    fig.add_trace(go.Scatter(x=dates, y=obs_log, name='Observed', line=dict(color=color_obs, width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sim_log, name='Simulated', line=dict(color=color_sim, width=1), showlegend=False), row=2, col=1)
    fig.update_yaxes(type="log", title_text="Flow (ML/day)", row=2, col=1)
    
    # FDC
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    fig.add_trace(go.Scatter(x=exc_obs, y=obs_sorted, name='Observed', line=dict(color=color_obs, width=2), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=exc_sim, y=sim_sorted, name='Simulated', line=dict(color=color_sim, width=2, dash='dash'), showlegend=False), row=3, col=1)
    fig.update_yaxes(type="log", title_text="Flow (ML/day)", row=3, col=1)
    fig.update_xaxes(title_text="Exceedance (%)", range=[0, 100], row=3, col=1)
    
    # Scatter
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid, sim_valid = obs[valid], sim[valid]
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    fig.add_trace(go.Scatter(x=obs_valid, y=sim_valid, mode='markers', marker=dict(color=color_sim, size=4, opacity=0.3), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='green', dash='dash', width=2), showlegend=False), row=4, col=1)
    fig.update_xaxes(title_text="Observed (ML/day)", row=4, col=1)
    fig.update_yaxes(title_text="Simulated (ML/day)", row=4, col=1)
    
    # Header
    catchment_name = report.catchment_info.get('name', 'Unknown Catchment')
    gauge_id = report.catchment_info.get('gauge_id', '')
    title_text = f"<b>{catchment_name}</b>"
    if gauge_id:
        title_text += f" ({gauge_id})"
    subtitle = f"Method: {report.result.method} | Objective: {report.result.objective_name} | Best: {report.result.best_objective:.4f}"
    
    fig.update_layout(
        title=dict(text=f"{title_text}<br><sup>{subtitle}</sup>", x=0.5, xanchor='center'),
        height=1200,
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
    )
    
    return fig


def plot_hydrograph_plotly(
    report: 'CalibrationReport',
    log_scale: bool = False,
    height: int = 400
):
    """
    Create an interactive Plotly hydrograph.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    dates = report.dates
    
    if log_scale:
        obs = np.where(obs > 0, obs, np.nan)
        sim = np.where(sim > 0, sim, np.nan)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=obs, name='Observed',
        line=dict(color='#e74c3c', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=sim, name='Simulated',
        line=dict(color='#3498db', width=1)
    ))
    
    scale_label = 'Log Scale' if log_scale else 'Linear Scale'
    fig.update_layout(
        title=f'Hydrograph Comparison ({scale_label})',
        xaxis_title='Date',
        yaxis_title='Flow (ML/day)',
        height=height,
        yaxis_type='log' if log_scale else 'linear'
    )
    
    return fig


def plot_fdc_plotly(
    report: 'CalibrationReport',
    log_scale: bool = True,
    height: int = 400
):
    """
    Create an interactive Plotly flow duration curve.
    
    Args:
        report: CalibrationReport instance
        log_scale: Use logarithmic y-axis
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    
    obs_sorted = np.sort(obs[~np.isnan(obs)])[::-1]
    sim_sorted = np.sort(sim[~np.isnan(sim)])[::-1]
    exc_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1) * 100
    exc_sim = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=exc_obs, y=obs_sorted, name='Observed',
        line=dict(color='#e74c3c', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=exc_sim, y=sim_sorted, name='Simulated',
        line=dict(color='#3498db', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Flow Duration Curve',
        xaxis_title='Exceedance Probability (%)',
        yaxis_title='Flow (ML/day)',
        height=height,
        yaxis_type='log' if log_scale else 'linear',
        xaxis_range=[0, 100]
    )
    
    return fig


def plot_scatter_plotly(
    report: 'CalibrationReport',
    height: int = 500
):
    """
    Create an interactive Plotly scatter plot of observed vs simulated.
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    obs = report.observed
    sim = report.simulated
    
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    # Calculate metrics for annotation
    metrics = _calculate_basic_metrics(obs, sim)
    r_squared = metrics.get('KGE_r', np.nan) ** 2 if not np.isnan(metrics.get('KGE_r', np.nan)) else np.nan
    
    # Regression line
    z = np.polyfit(obs_valid, sim_valid, 1)
    p = np.poly1d(z)
    
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    min_val = min(np.nanmin(obs_valid), np.nanmin(sim_valid))
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=obs_valid, y=sim_valid, mode='markers',
        marker=dict(color='#3498db', size=5, opacity=0.4),
        name='Data points',
        hovertemplate='Observed: %{x:.1f}<br>Simulated: %{y:.1f}<extra></extra>'
    ))
    
    # 1:1 line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line=dict(color='green', dash='dash', width=2),
        name='1:1 Line'
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=p([min_val, max_val]), mode='lines',
        line=dict(color='#e74c3c', dash='dot', width=2),
        name=f'Regression (y={z[0]:.2f}x+{z[1]:.1f})'
    ))
    
    # Add R² annotation
    annotations = []
    if not np.isnan(r_squared):
        annotations.append(dict(
            x=0.95, y=0.05, xref='paper', yref='paper',
            text=f"<b>R² = {r_squared:.4f}</b><br>NSE = {metrics.get('NSE', np.nan):.4f}",
            showarrow=False, font=dict(size=12),
            bgcolor='white', bordercolor='gray', borderwidth=1
        ))
    
    fig.update_layout(
        title='Observed vs Simulated Flow',
        xaxis_title='Observed Flow (ML/day)',
        yaxis_title='Simulated Flow (ML/day)',
        height=height,
        annotations=annotations,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Make axes equal
    fig.update_xaxes(range=[0, max_val * 1.05])
    fig.update_yaxes(range=[0, max_val * 1.05], scaleanchor='x', scaleratio=1)
    
    return fig


def plot_parameter_bounds_plotly(
    report: 'CalibrationReport',
    height: int = 500
):
    """
    Create an interactive Plotly parameter bounds chart.
    
    Args:
        report: CalibrationReport instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    if not report.parameter_bounds:
        raise ValueError("No parameter bounds available in report")
    
    param_names = list(report.result.best_parameters.keys())
    norm_values = []
    actual_values = []
    hover_texts = []
    
    for param in param_names:
        value = report.result.best_parameters[param]
        actual_values.append(value)
        
        if param in report.parameter_bounds:
            low, high = report.parameter_bounds[param]
            norm = (value - low) / (high - low) * 100 if high > low else 50
            norm_values.append(np.clip(norm, 0, 100))
            hover_texts.append(f"{param}: {value:.4f}<br>Bounds: [{low:.3f}, {high:.3f}]<br>Position: {norm:.1f}%")
        else:
            norm_values.append(50)
            hover_texts.append(f"{param}: {value:.4f}<br>No bounds available")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=param_names,
        x=norm_values,
        orientation='h',
        marker_color='#3498db',
        text=[f'{v:.3f}' for v in actual_values],
        textposition='inside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50%")
    
    fig.update_layout(
        title='Calibrated Parameters vs Bounds',
        xaxis_title='Position within Bounds (%)',
        xaxis_range=[0, 100],
        height=height,
        yaxis_autorange='reversed'
    )
    
    return fig
