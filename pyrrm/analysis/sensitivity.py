"""
Sensitivity analysis for rainfall-runoff models.

This module provides Sobol sensitivity analysis for parameter importance
assessment using SALib (Sensitivity Analysis Library).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import warnings

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction

# Import SALib only when needed
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False


@dataclass
class SobolResult:
    """
    Container for Sobol sensitivity analysis results.
    
    Attributes:
        S1: First-order sensitivity indices
        S1_conf: Confidence intervals for S1
        ST: Total-order sensitivity indices
        ST_conf: Confidence intervals for ST
        S2: Second-order interaction indices (if calculated)
        parameter_names: List of parameter names
        n_samples: Number of samples used
    """
    S1: Dict[str, float]
    S1_conf: Dict[str, float]
    ST: Dict[str, float]
    ST_conf: Dict[str, float]
    S2: Optional[pd.DataFrame] = None
    parameter_names: List[str] = field(default_factory=list)
    n_samples: int = 0
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "=" * 60,
            "SOBOL SENSITIVITY ANALYSIS RESULTS",
            "=" * 60,
            f"Number of samples: {self.n_samples}",
            "",
            "First-Order Indices (S1):",
            "-" * 40,
        ]
        
        # Sort by importance
        sorted_params = sorted(
            self.parameter_names, 
            key=lambda p: self.ST.get(p, 0), 
            reverse=True
        )
        
        for name in sorted_params:
            s1 = self.S1.get(name, 0)
            s1_conf = self.S1_conf.get(name, 0)
            st = self.ST.get(name, 0)
            st_conf = self.ST_conf.get(name, 0)
            lines.append(f"  {name:15s}: S1={s1:7.4f} ± {s1_conf:.4f}, ST={st:7.4f} ± {st_conf:.4f}")
        
        lines.append("")
        lines.append("Parameter Ranking by Total Sensitivity (ST):")
        lines.append("-" * 40)
        
        for i, name in enumerate(sorted_params, 1):
            st = self.ST.get(name, 0)
            lines.append(f"  {i}. {name}: {st:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = {
            'parameter': self.parameter_names,
            'S1': [self.S1.get(p, np.nan) for p in self.parameter_names],
            'S1_conf': [self.S1_conf.get(p, np.nan) for p in self.parameter_names],
            'ST': [self.ST.get(p, np.nan) for p in self.parameter_names],
            'ST_conf': [self.ST_conf.get(p, np.nan) for p in self.parameter_names],
        }
        
        df = pd.DataFrame(data)
        return df.sort_values('ST', ascending=False)


class SobolSensitivityAnalysis:
    """
    Sobol sensitivity analysis for parameter importance.
    
    Uses variance-based global sensitivity analysis to identify:
    - First-order effects (S1): Individual parameter contributions
    - Total-order effects (ST): Parameter contributions including interactions
    - Second-order effects (S2): Pairwise parameter interactions
    
    Example:
        >>> from pyrrm.models import GR4J
        >>> from pyrrm.analysis import SobolSensitivityAnalysis
        >>> from pyrrm.calibration.objective_functions import NSE
        >>> 
        >>> model = GR4J()
        >>> analysis = SobolSensitivityAnalysis(model, inputs, observed, NSE())
        >>> result = analysis.run(n_samples=1024)
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        model: 'BaseRainfallRunoffModel',
        inputs: pd.DataFrame,
        observed: np.ndarray,
        objective: 'ObjectiveFunction',
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warmup_period: int = 365
    ):
        """
        Initialize Sobol analysis.
        
        Args:
            model: Rainfall-runoff model
            inputs: Input DataFrame
            observed: Observed flow values
            objective: Objective function for evaluation
            parameter_bounds: Parameter bounds {name: (min, max)}
            warmup_period: Warmup timesteps to exclude
        """
        if not SALIB_AVAILABLE:
            raise ImportError(
                "SALib is required for sensitivity analysis. "
                "Install with: pip install SALib"
            )
        
        self.model = model
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.objective = objective
        self.warmup_period = warmup_period
        
        # Get parameter bounds
        if parameter_bounds is None:
            self._param_bounds = model.get_parameter_bounds()
        else:
            self._param_bounds = parameter_bounds
        
        self._param_names = list(self._param_bounds.keys())
    
    def _build_problem(self) -> Dict[str, Any]:
        """Build SALib problem definition."""
        bounds = [list(self._param_bounds[name]) for name in self._param_names]
        
        return {
            'num_vars': len(self._param_names),
            'names': self._param_names,
            'bounds': bounds
        }
    
    def _evaluate_sample(self, param_values: np.ndarray) -> float:
        """Evaluate objective for a single parameter set."""
        params = dict(zip(self._param_names, param_values))
        
        self.model.reset()
        self.model.set_parameters(params)
        
        try:
            results = self.model.run(self.inputs)
            
            if 'flow' in results.columns:
                simulated = results['flow'].values
            elif 'runoff' in results.columns:
                simulated = results['runoff'].values
            else:
                simulated = results.iloc[:, 0].values
            
            # Apply warmup
            sim = simulated[self.warmup_period:]
            obs = self.observed[self.warmup_period:]
            
            value = self.objective.calculate(sim, obs)
            
            return value if not np.isnan(value) else 0.0
            
        except Exception:
            return 0.0
    
    def run(
        self,
        n_samples: int = 1024,
        calc_second_order: bool = True,
        conf_level: float = 0.95,
        print_progress: bool = True,
        seed: Optional[int] = None
    ) -> SobolResult:
        """
        Run Sobol sensitivity analysis.
        
        Args:
            n_samples: Base number of samples (actual samples = N*(2D+2))
            calc_second_order: Whether to calculate second-order indices
            conf_level: Confidence level for intervals
            print_progress: Whether to print progress
            seed: Random seed for reproducibility
            
        Returns:
            SobolResult with sensitivity indices
        """
        problem = self._build_problem()
        
        # Generate samples using Saltelli method
        if seed is not None:
            np.random.seed(seed)
        
        param_values = saltelli.sample(
            problem, 
            n_samples, 
            calc_second_order=calc_second_order
        )
        
        n_total = len(param_values)
        if print_progress:
            print(f"Running {n_total} model evaluations...")
        
        # Evaluate all samples
        Y = np.zeros(n_total)
        
        for i, params in enumerate(param_values):
            if print_progress and (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_total} evaluations...")
            
            Y[i] = self._evaluate_sample(params)
        
        # Analyze results
        Si = sobol.analyze(
            problem, 
            Y, 
            calc_second_order=calc_second_order,
            conf_level=conf_level
        )
        
        # Build result
        S1 = dict(zip(self._param_names, Si['S1']))
        S1_conf = dict(zip(self._param_names, Si['S1_conf']))
        ST = dict(zip(self._param_names, Si['ST']))
        ST_conf = dict(zip(self._param_names, Si['ST_conf']))
        
        # Second-order indices
        S2 = None
        if calc_second_order and 'S2' in Si:
            S2_matrix = Si['S2']
            S2 = pd.DataFrame(
                S2_matrix,
                index=self._param_names,
                columns=self._param_names
            )
        
        return SobolResult(
            S1=S1,
            S1_conf=S1_conf,
            ST=ST,
            ST_conf=ST_conf,
            S2=S2,
            parameter_names=self._param_names,
            n_samples=n_samples
        )
    
    def run_fast(
        self,
        n_samples: int = 512,
        m: int = 4,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run FAST (Fourier Amplitude Sensitivity Test) analysis.
        
        FAST is faster than Sobol but only provides first-order indices.
        
        Args:
            n_samples: Number of samples
            m: Interference factor
            seed: Random seed
            
        Returns:
            Dictionary of first-order sensitivity indices
        """
        from SALib.sample import fast_sampler
        from SALib.analyze import fast
        
        problem = self._build_problem()
        
        if seed is not None:
            np.random.seed(seed)
        
        param_values = fast_sampler.sample(problem, n_samples, m)
        
        Y = np.array([self._evaluate_sample(p) for p in param_values])
        
        Si = fast.analyze(problem, Y, m)
        
        return dict(zip(self._param_names, Si['S1']))
