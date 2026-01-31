"""
Vendored SCE-UA (Shuffled Complex Evolution - University of Arizona) algorithm.

This is a standalone implementation of the SCE-UA global optimization algorithm,
vendored from https://github.com/cheginit/sceua for use in pyrrm calibration.

The SCE-UA algorithm is particularly well-suited for calibrating hydrological
models due to its ability to handle multi-modal objective functions and
high-dimensional parameter spaces.

References
----------
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global
    optimization for conceptual rainfall-runoff models. Water Resources Research.
- Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global
    optimization method for calibrating watershed models. Journal of Hydrology.
"""

from pyrrm.calibration._sceua.sceua import Result, minimize

__all__ = ["minimize", "Result"]
