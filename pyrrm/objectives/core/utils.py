"""
Utility functions for objective function evaluation.

This module provides convenience functions for evaluating multiple
objective functions and generating performance reports.
"""

from typing import List, Dict, Optional
import numpy as np

from pyrrm.objectives.core.base import ObjectiveFunction
from pyrrm.objectives.core.result import MetricResult


def evaluate_all(obs: np.ndarray,
                 sim: np.ndarray,
                 objectives: Optional[List[ObjectiveFunction]] = None) -> Dict[str, MetricResult]:
    """
    Evaluate multiple objective functions at once.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values
    objectives : list of ObjectiveFunction, optional
        List of objective functions to evaluate.
        If None, uses a standard set of metrics.
    
    Returns
    -------
    dict
        Dictionary mapping objective names to MetricResult objects
    
    Examples
    --------
    >>> from pyrrm.objectives import NSE, KGE, PBIAS
    >>> results = evaluate_all(obs, sim, [NSE(), KGE(), PBIAS()])
    >>> for name, result in results.items():
    ...     print(f"{name}: {result.value:.4f}")
    
    >>> # Use default metrics
    >>> results = evaluate_all(obs, sim)
    """
    if objectives is None:
        # Import here to avoid circular imports
        from pyrrm.objectives.metrics.traditional import NSE, PBIAS
        from pyrrm.objectives.metrics.kge import KGE
        from pyrrm.objectives.fdc.metrics import FDCMetric
        from pyrrm.objectives.transformations.flow_transforms import FlowTransformation
        
        objectives = [
            NSE(),
            KGE(variant='2012'),
            KGE(variant='2012', transform=FlowTransformation('inverse')),
            PBIAS(),
            FDCMetric('high', 'volume_bias'),
            FDCMetric('low', 'volume_bias', log_transform=True),
        ]
    
    return {obj.name: obj.evaluate(obs, sim) for obj in objectives}


def print_evaluation_report(obs: np.ndarray, 
                             sim: np.ndarray,
                             objectives: Optional[List[ObjectiveFunction]] = None,
                             title: str = "Model Performance Evaluation Report") -> None:
    """
    Print formatted evaluation report.
    
    Generates a nicely formatted console report showing all
    objective function values and their components.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values
    objectives : list of ObjectiveFunction, optional
        Objectives to evaluate (uses defaults if None)
    title : str, default="Model Performance Evaluation Report"
        Report title
    
    Examples
    --------
    >>> print_evaluation_report(obs, sim)
    
    ============================================================
    Model Performance Evaluation Report
    ============================================================
    Sample size: 365
    
    NSE: 0.8523
    KGE_2012: 0.8912 (r=0.934, alpha=0.987, beta=1.023)
    ...
    ============================================================
    """
    results = evaluate_all(obs, sim, objectives)
    
    # Calculate sample size
    obs_clean = np.asarray(obs).flatten()
    sim_clean = np.asarray(sim).flatten()
    mask = ~(np.isnan(obs_clean) | np.isnan(sim_clean))
    n_valid = np.sum(mask)
    
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Sample size: {n_valid} valid pairs")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"\n{result}")
    
    print("\n" + "=" * 60)


def calculate_metrics_summary(obs: np.ndarray,
                               sim: np.ndarray,
                               objectives: Optional[List[ObjectiveFunction]] = None) -> Dict[str, float]:
    """
    Calculate multiple metrics and return as a simple dictionary.
    
    Unlike evaluate_all, this returns only the primary metric values
    without the MetricResult wrapper, useful for quick comparisons.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    sim : np.ndarray
        Simulated values
    objectives : list of ObjectiveFunction, optional
        Objectives to evaluate
    
    Returns
    -------
    dict
        Dictionary of {metric_name: value}
    
    Examples
    --------
    >>> metrics = calculate_metrics_summary(obs, sim)
    >>> print(f"NSE: {metrics['NSE']:.3f}, KGE: {metrics['KGE_2012']:.3f}")
    """
    results = evaluate_all(obs, sim, objectives)
    return {name: result.value for name, result in results.items()}


def compare_simulations(obs: np.ndarray,
                         simulations: Dict[str, np.ndarray],
                         objectives: Optional[List[ObjectiveFunction]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple simulations against observations.
    
    Useful for comparing different model configurations or
    calibration results.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    simulations : dict
        Dictionary mapping simulation names to simulated arrays
    objectives : list of ObjectiveFunction, optional
        Objectives to evaluate
    
    Returns
    -------
    dict
        Nested dictionary: {sim_name: {metric_name: value}}
    
    Examples
    --------
    >>> simulations = {
    ...     'Default': sim_default,
    ...     'Calibrated': sim_calibrated,
    ...     'Validation': sim_validation,
    ... }
    >>> comparison = compare_simulations(obs, simulations)
    >>> for sim_name, metrics in comparison.items():
    ...     print(f"{sim_name}: NSE={metrics['NSE']:.3f}")
    """
    return {
        sim_name: calculate_metrics_summary(obs, sim, objectives)
        for sim_name, sim in simulations.items()
    }


def rank_simulations(obs: np.ndarray,
                      simulations: Dict[str, np.ndarray],
                      metric_name: str = 'NSE') -> List[tuple]:
    """
    Rank simulations by a specified metric.
    
    Parameters
    ----------
    obs : np.ndarray
        Observed values
    simulations : dict
        Dictionary mapping simulation names to simulated arrays
    metric_name : str, default='NSE'
        Name of metric to rank by
    
    Returns
    -------
    list of tuples
        List of (sim_name, metric_value) sorted by performance
        (best first for maximize metrics, lowest first for minimize)
    
    Examples
    --------
    >>> ranking = rank_simulations(obs, simulations, metric_name='KGE_2012')
    >>> print("Best simulation:", ranking[0][0])
    """
    comparison = compare_simulations(obs, simulations)
    
    # Extract metric values
    results = []
    for sim_name, metrics in comparison.items():
        if metric_name in metrics:
            results.append((sim_name, metrics[metric_name]))
    
    # Sort (assume maximize for efficiency metrics)
    # For minimize metrics, user should interpret accordingly
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results
