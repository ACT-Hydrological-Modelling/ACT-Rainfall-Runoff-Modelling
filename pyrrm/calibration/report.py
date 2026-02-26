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
            >>> report.save('calibrations/410734_sacramento_nse_sceua_log')
            'calibrations/410734_sacramento_nse_sceua_log.pkl'
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
            >>> report = CalibrationReport.load('calibrations/410734_sacramento_nse_sceua_log.pkl')
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
        Calculate the canonical 48-metric diagnostic suite.

        Returns:
            OrderedDict with:
            - NSE variants: NSE, NSE_log, NSE_sqrt, NSE_inv
            - KGE(Q) + components: KGE, KGE_r, KGE_alpha, KGE_beta
            - KGE(log Q) + components
            - KGE(sqrt Q) + components
            - KGE(1/Q) + components
            - Error metrics: RMSE, MAE, PBIAS
            - FDC volume biases: FHV, FMV, FLV

        Example:
            >>> metrics = report.calculate_comprehensive_metrics()
            >>> for name, value in metrics.items():
            ...     print(f"{name}: {value:.4f}")
        """
        from pyrrm.analysis.diagnostics import compute_diagnostics
        return compute_diagnostics(self.simulated, self.observed)
    
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
        
        # Performance metrics (canonical 48-metric suite)
        try:
            from pyrrm.analysis.diagnostics import compute_diagnostics, DIAGNOSTIC_GROUPS
            metrics = compute_diagnostics(self.simulated, self.observed)
            lines.append("Performance Metrics:")
            for group_name, keys in DIAGNOSTIC_GROUPS.items():
                lines.append(f"  {group_name}:")
                for k in keys:
                    v = metrics.get(k, float('nan'))
                    if isinstance(v, float) and np.isnan(v):
                        lines.append(f"    {k:<23} {'N/A':>10}")
                    else:
                        lines.append(f"    {k:<23} {v:>10.4f}")
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
    # Export (Excel / CSV)
    # =========================================================================

    def export(
        self,
        path: str,
        format: str = 'excel',
        exceedance_pct_resolution: float = 1.0,
        csv_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Export this report to Excel and/or CSV files for sharing.

        Excel: single file with sheets TimeSeries, Best_Calibration, Diagnostics, FDC.
        Requires optional dependency: pip install pyrrm[export]

        CSV: four files (timeseries, best_calibration, diagnostics, fdc) with
        the same content as the Excel sheets.

        Args:
            path: For format='excel': path to .xlsx file. For format='csv' or 'both':
                directory or file prefix (e.g. 'out/410734' -> out/410734.xlsx and
                out/410734_timeseries.csv, etc.). If path is a directory, output
                filename uses experiment_name or 'calibration_report'.
            format: 'excel', 'csv', or 'both'.
            exceedance_pct_resolution: FDC grid step in percent (default 1.0).
            csv_prefix: Override prefix for CSV filenames (default: derived from path).

        Returns:
            List of created file paths.

        Example:
            >>> report = CalibrationReport.load('calibrations/410734.pkl')
            >>> report.export('output/410734_report.xlsx', format='excel')
            >>> report.export('output/410734', format='both')
        """
        from pyrrm.calibration.export import export_report
        return export_report(
            self,
            path,
            format=format,
            exceedance_pct_resolution=exceedance_pct_resolution,
            csv_prefix=csv_prefix,
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
