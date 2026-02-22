"""
Calibration report for comprehensive result storage and visualization.

This module provides the CalibrationReport class which encapsulates all data
needed to reproduce, visualize, and analyze a calibration result, including:
- The calibration result (best parameters, samples, diagnostics)
- Input and observed data for visualization
- Model configuration for re-simulation
- Methods for generating comprehensive report cards

Example:
    >>> from pyrrm.calibration import CalibrationRunner, CalibrationReport
    >>> runner = CalibrationRunner(model, inputs, observed, objective)
    >>> result = runner.run_sceua_direct(max_evals=10000)
    >>> report = runner.create_report(result, catchment_info={'name': 'Queanbeyan'})
    >>> report.save('calibrations/my_calibration.pkl')
    
    >>> # Later, in a new session
    >>> report = CalibrationReport.load('calibrations/my_calibration.pkl')
    >>> fig = report.plot_report_card()
    >>> fig.savefig('report_card.png', dpi=300)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import importlib
import pickle
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyrrm.calibration.runner import CalibrationResult
    from pyrrm.models.base import BaseRainfallRunoffModel
    from matplotlib.figure import Figure


@dataclass
class CalibrationReport:
    """
    Complete calibration result with all data needed for report generation.
    
    This class stores everything required to:
    1. Visualize calibration results (hydrographs, FDCs, scatter plots)
    2. Re-run simulations with different parameters
    3. Calculate comprehensive performance metrics
    4. Generate report cards in multiple formats
    
    Attributes:
        result: CalibrationResult containing best parameters and samples
        observed: Observed flow values (post-warmup)
        simulated: Simulated flow values from calibrated model (post-warmup)
        dates: DatetimeIndex for the calibration period (post-warmup)
        precipitation: Precipitation values (optional, post-warmup)
        pet: PET values (optional, post-warmup)
        inputs: Full input DataFrame (for re-simulation)
        parameter_bounds: Parameter bounds used in calibration
        catchment_info: Catchment metadata (name, gauge_id, area_km2)
        calibration_period: Start and end dates as strings
        warmup_days: Number of warmup days used
        model_config: Model class and initialization kwargs for re-simulation
        created_at: Timestamp when report was created
        
    Example:
        >>> report = CalibrationReport.load('my_calibration.pkl')
        >>> metrics = report.calculate_metrics()
        >>> print(f"NSE: {metrics['NSE']:.3f}")
        >>> fig = report.plot_report_card()
    """
    
    # Core calibration result
    result: 'CalibrationResult'
    
    # Time series data (post-warmup)
    observed: np.ndarray
    simulated: np.ndarray
    dates: pd.DatetimeIndex
    precipitation: Optional[np.ndarray] = None
    pet: Optional[np.ndarray] = None
    
    # Full input data (for re-simulation)
    inputs: Optional[pd.DataFrame] = None
    
    # Configuration
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    catchment_info: Dict[str, Any] = field(default_factory=dict)
    calibration_period: Tuple[str, str] = ("", "")
    warmup_days: int = 0
    
    # Model configuration (for re-simulation)
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment identifier (canonical key from naming convention)
    experiment_name: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        # Ensure arrays are numpy arrays
        if not isinstance(self.observed, np.ndarray):
            self.observed = np.asarray(self.observed)
        if not isinstance(self.simulated, np.ndarray):
            self.simulated = np.asarray(self.simulated)
        if self.precipitation is not None and not isinstance(self.precipitation, np.ndarray):
            self.precipitation = np.asarray(self.precipitation)
        if self.pet is not None and not isinstance(self.pet, np.ndarray):
            self.pet = np.asarray(self.pet)
            
        # Validate lengths match
        n_obs = len(self.observed)
        n_sim = len(self.simulated)
        n_dates = len(self.dates)
        
        if n_obs != n_sim:
            warnings.warn(f"Length mismatch: observed ({n_obs}) vs simulated ({n_sim})")
        if n_obs != n_dates:
            warnings.warn(f"Length mismatch: observed ({n_obs}) vs dates ({n_dates})")
    
    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    
    def save(self, path: str) -> str:
        """
        Save the complete calibration report to a pickle file.
        
        Args:
            path: Path for the output file (will add .pkl if not present)
            
        Returns:
            Path to the saved file
            
        Example:
            >>> report.save('calibrations/410734_lognse')
            'calibrations/410734_lognse.pkl'
        """
        path = Path(path)
        
        # Add .pkl extension if not present
        if path.suffix != '.pkl':
            path = path.with_suffix('.pkl')
        
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return str(path)
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationReport':
        """
        Load a calibration report from a pickle file.
        
        Args:
            path: Path to the pickle file
            
        Returns:
            CalibrationReport instance
            
        Example:
            >>> report = CalibrationReport.load('calibrations/410734_lognse.pkl')
            >>> print(report.result.best_objective)
        """
        path = Path(path)
        
        # Add .pkl extension if not present
        if path.suffix != '.pkl' and not path.exists():
            path = path.with_suffix('.pkl')
        
        if not path.exists():
            raise FileNotFoundError(f"Report file not found: {path}")
        
        with open(path, 'rb') as f:
            report = pickle.load(f)
        
        # Check by class name (more robust with module reloading)
        if type(report).__name__ != 'CalibrationReport':
            raise TypeError(f"Loaded object is not a CalibrationReport: {type(report)}")
        
        return report
    
    # =========================================================================
    # Model Re-simulation
    # =========================================================================
    
    def _get_model_class(self) -> type:
        """Get the model class from stored configuration."""
        if not self.model_config:
            raise ValueError("No model configuration stored. Cannot re-run simulation.")
        
        module_name = self.model_config.get('module')
        class_name = self.model_config.get('class_name')
        
        if not module_name or not class_name:
            raise ValueError("Model configuration missing 'module' or 'class_name'")
        
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            return model_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import model class {module_name}.{class_name}: {e}")
    
    def rerun_simulation(
        self, 
        parameters: Optional[Dict[str, float]] = None,
        return_full: bool = False
    ) -> np.ndarray:
        """
        Re-run the model simulation with specified or best parameters.
        
        Args:
            parameters: Parameter dictionary (uses best_parameters if None)
            return_full: If True, return full output including warmup period
            
        Returns:
            Simulated flow array (post-warmup unless return_full=True)
            
        Example:
            >>> # Re-run with best parameters
            >>> sim = report.rerun_simulation()
            
            >>> # Re-run with modified parameters
            >>> new_params = report.result.best_parameters.copy()
            >>> new_params['uztwm'] = 100.0
            >>> sim = report.rerun_simulation(new_params)
        """
        if self.inputs is None:
            raise ValueError("No input data stored. Cannot re-run simulation.")
        
        # Get model class and instantiate
        model_class = self._get_model_class()
        init_kwargs = self.model_config.get('init_kwargs', {})
        model = model_class(**init_kwargs)
        
        # Set parameters
        params = parameters if parameters is not None else self.result.best_parameters
        model.set_parameters(params)
        model.reset()
        
        # Run model
        output = model.run(self.inputs)
        
        # Extract runoff column
        if 'runoff' in output.columns:
            sim = output['runoff'].values
        elif 'flow' in output.columns:
            sim = output['flow'].values
        else:
            sim = output.iloc[:, 0].values
        
        # Apply warmup
        if return_full:
            return sim
        else:
            return sim[self.warmup_days:]
    
    def can_rerun(self) -> bool:
        """Check if re-simulation is possible."""
        return (
            self.inputs is not None and 
            self.model_config is not None and
            'module' in self.model_config and
            'class_name' in self.model_config
        )
    
    # =========================================================================
    # Metrics Calculation
    # =========================================================================
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate standard performance metrics.
        
        Returns:
            Dictionary with NSE, KGE, RMSE, MAE, PBIAS, LogNSE
            
        Example:
            >>> metrics = report.calculate_metrics()
            >>> print(f"NSE: {metrics['NSE']:.3f}, KGE: {metrics['KGE']:.3f}")
        """
        from pyrrm.calibration.objective_functions import calculate_metrics
        return calculate_metrics(self.simulated, self.observed)
    
    def calculate_comprehensive_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics including all NSE and KGE variants.
        
        Returns:
            Dictionary with extended metrics including:
            - NSE variants: NSE, LogNSE, SqrtNSE, InvNSE
            - KGE variants: KGE, KGE_log, KGE_sqrt, KGE_inv
            - KGE components: r, alpha, beta
            - Error metrics: RMSE, MAE, PBIAS, R²
            
        Example:
            >>> metrics = report.calculate_comprehensive_metrics()
            >>> for name, value in metrics.items():
            ...     print(f"{name}: {value:.4f}")
        """
        from pyrrm.calibration.objective_functions import (
            NSE, KGE, RMSE, MAE, PBIAS, LogNSE
        )
        
        obs = self.observed
        sim = self.simulated
        
        # Filter valid data
        mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
        obs_valid = obs[mask]
        sim_valid = sim[mask]
        
        metrics = {}
        
        # ==========================================================================
        # NSE and variants
        # ==========================================================================
        metrics['NSE'] = NSE().calculate(sim_valid, obs_valid)
        
        try:
            metrics['LogNSE'] = LogNSE().calculate(sim_valid, obs_valid)
        except Exception:
            metrics['LogNSE'] = np.nan
        
        # SqrtNSE (calculated directly)
        try:
            sqrt_obs = np.sqrt(np.maximum(obs_valid, 0))
            sqrt_sim = np.sqrt(np.maximum(sim_valid, 0))
            ss_res_sqrt = np.sum((sqrt_obs - sqrt_sim) ** 2)
            ss_tot_sqrt = np.sum((sqrt_obs - np.mean(sqrt_obs)) ** 2)
            metrics['SqrtNSE'] = 1 - ss_res_sqrt / ss_tot_sqrt if ss_tot_sqrt > 0 else np.nan
        except Exception:
            metrics['SqrtNSE'] = np.nan
        
        # InvNSE (calculated directly)
        try:
            obs_pos_inv = obs_valid[obs_valid > 0.01]
            sim_pos_inv = sim_valid[obs_valid > 0.01]
            if len(obs_pos_inv) > 0:
                inv_obs = 1.0 / obs_pos_inv
                inv_sim = 1.0 / np.maximum(sim_pos_inv, 0.01)
                ss_res_inv = np.sum((inv_obs - inv_sim) ** 2)
                ss_tot_inv = np.sum((inv_obs - np.mean(inv_obs)) ** 2)
                metrics['InvNSE'] = 1 - ss_res_inv / ss_tot_inv if ss_tot_inv > 0 else np.nan
            else:
                metrics['InvNSE'] = np.nan
        except Exception:
            metrics['InvNSE'] = np.nan
        
        # ==========================================================================
        # KGE (standard) and components
        # ==========================================================================
        metrics['KGE'] = KGE().calculate(sim_valid, obs_valid)
        
        # KGE components
        r = np.corrcoef(obs_valid, sim_valid)[0, 1] if len(obs_valid) > 1 else np.nan
        alpha = np.std(sim_valid) / np.std(obs_valid) if np.std(obs_valid) > 0 else np.nan
        beta = np.mean(sim_valid) / np.mean(obs_valid) if np.mean(obs_valid) != 0 else np.nan
        
        metrics['KGE_r'] = r
        metrics['KGE_alpha'] = alpha
        metrics['KGE_beta'] = beta
        
        # ==========================================================================
        # KGE with transformations
        # ==========================================================================
        # KGE(log Q)
        obs_pos_mask = obs_valid > 0
        obs_pos = obs_valid[obs_pos_mask]
        sim_pos = sim_valid[obs_pos_mask]
        if len(obs_pos) > 0:
            log_obs = np.log(obs_pos + 1)
            log_sim = np.log(np.maximum(sim_pos, 0) + 1)
            r_log = np.corrcoef(log_obs, log_sim)[0, 1] if len(log_obs) > 1 else np.nan
            alpha_log = np.std(log_sim) / np.std(log_obs) if np.std(log_obs) > 0 else np.nan
            beta_log = np.mean(log_sim) / np.mean(log_obs) if np.mean(log_obs) != 0 else np.nan
            metrics['KGE_log'] = 1 - np.sqrt((r_log - 1)**2 + (alpha_log - 1)**2 + (beta_log - 1)**2) if not np.isnan(r_log) else np.nan
        else:
            metrics['KGE_log'] = np.nan
        
        # KGE(sqrt Q)
        sqrt_obs = np.sqrt(np.maximum(obs_valid, 0))
        sqrt_sim = np.sqrt(np.maximum(sim_valid, 0))
        r_sqrt = np.corrcoef(sqrt_obs, sqrt_sim)[0, 1] if len(sqrt_obs) > 1 else np.nan
        alpha_sqrt = np.std(sqrt_sim) / np.std(sqrt_obs) if np.std(sqrt_obs) > 0 else np.nan
        beta_sqrt = np.mean(sqrt_sim) / np.mean(sqrt_obs) if np.mean(sqrt_obs) != 0 else np.nan
        metrics['KGE_sqrt'] = 1 - np.sqrt((r_sqrt - 1)**2 + (alpha_sqrt - 1)**2 + (beta_sqrt - 1)**2) if not np.isnan(r_sqrt) else np.nan
        
        # KGE(1/Q)
        obs_pos_inv = obs_valid[obs_valid > 0.01]
        sim_pos_inv = sim_valid[obs_valid > 0.01]
        if len(obs_pos_inv) > 0:
            inv_obs = 1.0 / obs_pos_inv
            inv_sim = 1.0 / np.maximum(sim_pos_inv, 0.01)
            r_inv = np.corrcoef(inv_obs, inv_sim)[0, 1] if len(inv_obs) > 1 else np.nan
            alpha_inv = np.std(inv_sim) / np.std(inv_obs) if np.std(inv_obs) > 0 else np.nan
            beta_inv = np.mean(inv_sim) / np.mean(inv_obs) if np.mean(inv_obs) != 0 else np.nan
            metrics['KGE_inv'] = 1 - np.sqrt((r_inv - 1)**2 + (alpha_inv - 1)**2 + (beta_inv - 1)**2) if not np.isnan(r_inv) else np.nan
        else:
            metrics['KGE_inv'] = np.nan
        
        # ==========================================================================
        # Other metrics
        # ==========================================================================
        metrics['RMSE'] = RMSE().calculate(sim_valid, obs_valid)
        metrics['MAE'] = MAE().calculate(sim_valid, obs_valid)
        metrics['PBIAS'] = PBIAS().calculate(sim_valid, obs_valid)
        metrics['R2'] = r ** 2 if not np.isnan(r) else np.nan
        
        return metrics
    
    # =========================================================================
    # Summary and Display
    # =========================================================================
    
    def summary(self) -> str:
        """Generate a text summary of the calibration report."""
        lines = [
            "=" * 70,
            "CALIBRATION REPORT",
            "=" * 70,
            "",
            f"Created: {self.created_at}",
            "",
        ]
        
        # Catchment info
        if self.catchment_info:
            lines.append("Catchment Information:")
            for key, value in self.catchment_info.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Calibration info
        lines.append("Calibration Configuration:")
        lines.append(f"  Period: {self.calibration_period[0]} to {self.calibration_period[1]}")
        lines.append(f"  Warmup: {self.warmup_days} days")
        lines.append(f"  Method: {self.result.method}")
        lines.append(f"  Objective: {self.result.objective_name}")
        lines.append(f"  Runtime: {self.result.runtime_seconds:.1f} seconds")
        lines.append("")
        
        # Best result
        lines.append("Best Result:")
        lines.append(f"  {self.result.objective_name}: {self.result.best_objective:.6f}")
        lines.append("")
        
        # Performance metrics
        try:
            metrics = self.calculate_metrics()
            lines.append("Performance Metrics:")
            for name, value in metrics.items():
                lines.append(f"  {name}: {value:.4f}")
            lines.append("")
        except Exception as e:
            lines.append(f"Could not calculate metrics: {e}")
            lines.append("")
        
        # Best parameters
        lines.append("Best Parameters:")
        lines.append("-" * 40)
        for name, value in self.result.best_parameters.items():
            lines.append(f"  {name}: {value:.6f}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        catchment = self.catchment_info.get('name', 'Unknown')
        return (
            f"CalibrationReport(catchment='{catchment}', "
            f"method='{self.result.method}', "
            f"best_{self.result.objective_name}={self.result.best_objective:.4f})"
        )
    
    # =========================================================================
    # Visualization Methods (delegate to report_plots module)
    # =========================================================================
    
    def plot_report_card(
        self,
        figsize: Tuple[int, int] = (16, 12),
        dark_theme: bool = False
    ) -> 'Figure':
        """
        Generate a comprehensive matplotlib report card figure.
        
        Args:
            figsize: Figure size (width, height)
            dark_theme: Use dark background theme
            
        Returns:
            matplotlib Figure with multi-panel report card
            
        Example:
            >>> fig = report.plot_report_card()
            >>> fig.savefig('report_card.png', dpi=300, bbox_inches='tight')
        """
        from pyrrm.visualization.report_plots import plot_report_card_matplotlib
        return plot_report_card_matplotlib(self, figsize=figsize, dark_theme=dark_theme)
    
    def plot_report_card_interactive(self, height: int = 1000):
        """
        Generate an interactive Plotly report card.
        
        Args:
            height: Figure height in pixels
            
        Returns:
            Plotly Figure object (can be saved as HTML)
            
        Example:
            >>> fig = report.plot_report_card_interactive()
            >>> fig.write_html('report_card.html')
        """
        from pyrrm.visualization.report_plots import plot_report_card_plotly
        return plot_report_card_plotly(self, height=height)
    
    def plot_hydrograph(
        self,
        log_scale: bool = False,
        figsize: Tuple[int, int] = (14, 6),
        dark_theme: bool = False,
        backend: str = 'matplotlib',
        height: int = 400
    ):
        """
        Plot observed vs simulated hydrograph.
        
        Args:
            log_scale: Use logarithmic y-axis
            figsize: Figure size (matplotlib only)
            dark_theme: Use dark theme (matplotlib only)
            backend: 'matplotlib' or 'plotly'
            height: Figure height in pixels (plotly only)
            
        Returns:
            matplotlib Figure or Plotly Figure object
            
        Example:
            >>> fig = report.plot_hydrograph(backend='matplotlib')
            >>> fig = report.plot_hydrograph(backend='plotly')
        """
        if backend.lower() == 'plotly':
            from pyrrm.visualization.report_plots import plot_hydrograph_plotly
            return plot_hydrograph_plotly(self, log_scale=log_scale, height=height)
        else:
            from pyrrm.visualization.report_plots import plot_hydrograph_comparison
            return plot_hydrograph_comparison(
                self, log_scale=log_scale, figsize=figsize, dark_theme=dark_theme
            )
    
    def plot_fdc(
        self,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        dark_theme: bool = False,
        backend: str = 'matplotlib',
        height: int = 400
    ):
        """
        Plot flow duration curves.
        
        Args:
            log_scale: Use logarithmic y-axis
            figsize: Figure size (matplotlib only)
            dark_theme: Use dark theme (matplotlib only)
            backend: 'matplotlib' or 'plotly'
            height: Figure height in pixels (plotly only)
            
        Returns:
            matplotlib Figure or Plotly Figure object
            
        Example:
            >>> fig = report.plot_fdc(backend='matplotlib')
            >>> fig = report.plot_fdc(backend='plotly')
        """
        if backend.lower() == 'plotly':
            from pyrrm.visualization.report_plots import plot_fdc_plotly
            return plot_fdc_plotly(self, log_scale=log_scale, height=height)
        else:
            from pyrrm.visualization.report_plots import plot_fdc_comparison
            return plot_fdc_comparison(
                self, log_scale=log_scale, figsize=figsize, dark_theme=dark_theme
            )
    
    def plot_scatter(
        self,
        figsize: Tuple[int, int] = (8, 8),
        dark_theme: bool = False,
        backend: str = 'matplotlib',
        height: int = 500
    ):
        """
        Plot observed vs simulated scatter plot with 1:1 line.
        
        Args:
            figsize: Figure size (matplotlib only)
            dark_theme: Use dark theme (matplotlib only)
            backend: 'matplotlib' or 'plotly'
            height: Figure height in pixels (plotly only)
            
        Returns:
            matplotlib Figure or Plotly Figure object
            
        Example:
            >>> fig = report.plot_scatter(backend='matplotlib')
            >>> fig = report.plot_scatter(backend='plotly')
        """
        if backend.lower() == 'plotly':
            from pyrrm.visualization.report_plots import plot_scatter_plotly
            return plot_scatter_plotly(self, height=height)
        else:
            from pyrrm.visualization.report_plots import plot_scatter_comparison
            return plot_scatter_comparison(self, figsize=figsize, dark_theme=dark_theme)
    
    def plot_parameter_bounds(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dark_theme: bool = False,
        backend: str = 'matplotlib',
        height: int = 500
    ):
        """
        Plot calibrated parameters as percentage of bounds.
        
        Args:
            figsize: Figure size (matplotlib only)
            dark_theme: Use dark theme (matplotlib only)
            backend: 'matplotlib' or 'plotly'
            height: Figure height in pixels (plotly only)
            
        Returns:
            matplotlib Figure or Plotly Figure object
            
        Example:
            >>> fig = report.plot_parameter_bounds(backend='matplotlib')
            >>> fig = report.plot_parameter_bounds(backend='plotly')
        """
        if backend.lower() == 'plotly':
            from pyrrm.visualization.report_plots import plot_parameter_bounds_plotly
            return plot_parameter_bounds_plotly(self, height=height)
        else:
            from pyrrm.visualization.report_plots import plot_parameter_bounds_chart
            return plot_parameter_bounds_chart(self, figsize=figsize, dark_theme=dark_theme)
