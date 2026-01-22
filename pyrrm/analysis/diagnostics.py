"""
Model diagnostics and performance analysis.

This module provides tools for diagnosing model performance
and analyzing simulation results.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from pyrrm.calibration.objective_functions import (
    NSE, KGE, RMSE, MAE, PBIAS, LogNSE, calculate_metrics
)


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
        Calculate all standard performance metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        sim = self.simulated[self._valid]
        obs = self.observed[self._valid]
        
        return calculate_metrics(sim, obs)
    
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
            f"  LogNSE: {metrics.get('LogNSE', np.nan):8.4f}",
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
