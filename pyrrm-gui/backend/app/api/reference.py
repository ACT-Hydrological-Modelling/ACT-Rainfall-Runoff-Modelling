"""
Reference Data API endpoints.

Provides information about available models, objectives, and algorithms.
"""

from typing import List, Dict, Any

from fastapi import APIRouter

from app.services.calibration import CalibrationService

router = APIRouter()


@router.get("/models")
def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available rainfall-runoff models.
    
    Returns model information including:
    - Type identifier
    - Display name
    - Description
    - Number of parameters
    - Parameter definitions with defaults and bounds
    """
    return CalibrationService.get_available_models()


@router.get("/models/{model_type}/bounds")
def get_model_bounds(model_type: str) -> Dict[str, List[float]]:
    """
    Get default parameter bounds for a model type.
    
    Returns bounds as {parameter_name: [min, max]}.
    """
    bounds = CalibrationService.get_default_bounds(model_type)
    return {k: list(v) for k, v in bounds.items()}


@router.get("/objectives")
def get_available_objectives() -> List[Dict[str, Any]]:
    """
    Get list of available objective functions.
    
    Returns:
    - Name/identifier
    - Description
    - Whether to maximize (True for NSE/KGE, False for RMSE)
    - Optimal value
    """
    return CalibrationService.get_available_objectives()


@router.get("/algorithms")
def get_available_algorithms() -> List[Dict[str, Any]]:
    """
    Get list of available calibration algorithms.
    """
    return [
        {
            "id": "sceua_direct",
            "name": "SCE-UA (Direct)",
            "description": "Shuffled Complex Evolution - University of Arizona. Global optimization algorithm designed for calibration of hydrological models.",
            "parameters": [
                {
                    "name": "max_evals",
                    "type": "integer",
                    "default": 50000,
                    "min": 1000,
                    "description": "Maximum function evaluations"
                },
                {
                    "name": "n_complexes",
                    "type": "integer",
                    "default": None,
                    "description": "Number of complexes (auto-determined if None)"
                },
                {
                    "name": "max_workers",
                    "type": "integer",
                    "default": 1,
                    "min": 1,
                    "description": "Number of parallel workers"
                },
                {
                    "name": "seed",
                    "type": "integer",
                    "default": None,
                    "description": "Random seed for reproducibility"
                }
            ]
        },
        {
            "id": "scipy_de",
            "name": "Differential Evolution (SciPy)",
            "description": "Global optimization using differential evolution algorithm from SciPy.",
            "parameters": [
                {
                    "name": "maxiter",
                    "type": "integer",
                    "default": 1000,
                    "description": "Maximum iterations"
                },
                {
                    "name": "popsize",
                    "type": "integer",
                    "default": 15,
                    "description": "Population size multiplier"
                },
                {
                    "name": "workers",
                    "type": "integer",
                    "default": 1,
                    "description": "Number of parallel workers"
                }
            ]
        },
        {
            "id": "spotpy_dream",
            "name": "DREAM (SpotPy)",
            "description": "Differential Evolution Adaptive Metropolis. Bayesian MCMC algorithm for uncertainty quantification.",
            "parameters": [
                {
                    "name": "n_iterations",
                    "type": "integer",
                    "default": 10000,
                    "description": "Number of iterations per chain"
                },
                {
                    "name": "n_chains",
                    "type": "integer",
                    "default": 5,
                    "description": "Number of Markov chains"
                }
            ],
            "requires": "spotpy"
        },
        {
            "id": "pydream",
            "name": "MT-DREAM(ZS) (PyDREAM)",
            "description": "Multi-Try DREAM with snooker updates. Advanced Bayesian calibration method.",
            "parameters": [
                {
                    "name": "n_iterations",
                    "type": "integer",
                    "default": 10000,
                    "description": "Number of iterations"
                },
                {
                    "name": "n_chains",
                    "type": "integer",
                    "default": 5,
                    "description": "Number of chains"
                },
                {
                    "name": "multitry",
                    "type": "integer",
                    "default": 5,
                    "description": "Multi-try proposals"
                }
            ],
            "requires": "pydream"
        }
    ]


@router.get("/transforms")
def get_available_transforms() -> List[Dict[str, Any]]:
    """
    Get list of available flow transformations.
    """
    return [
        {
            "id": "none",
            "name": "None",
            "description": "No transformation (emphasizes high flows)"
        },
        {
            "id": "log",
            "name": "Logarithm",
            "description": "Natural log transformation (emphasizes low flows)"
        },
        {
            "id": "sqrt",
            "name": "Square Root",
            "description": "Square root transformation (balanced)"
        },
        {
            "id": "inverse",
            "name": "Inverse (1/Q)",
            "description": "Inverse transformation (strong low flow emphasis)"
        }
    ]
