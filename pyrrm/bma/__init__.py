"""
pyrrm.bma — Bayesian Model Averaging for multi-model hydrological ensembles.

Provides five combination levels ordered by complexity:

1. **Equal weights** — simple arithmetic mean.
2. **GRC** — constrained regression weights.
3. **Bayesian stacking** — cross-validated optimisation.
4. **BMA (global)** — full probabilistic Dirichlet mixture in PyMC.
5. **Regime-specific BMA** — separate weights per flow regime with
   sigmoid blending.

Quick start::

    from pyrrm.bma import BMAConfig, BMARunner

    config = BMAConfig.from_preset("standard",
        model_predictions_path="predictions.csv",
        observed_flow_path="observed.csv",
    )
    runner = BMARunner(config)
    runner.load_data()
    runner.pre_screen()
    cv_df = runner.run_cross_validation()
    runner.fit_final()
    runner.save_results()
"""

from pyrrm.bma.config import BMAConfig, ACT_CV_PRESETS
from pyrrm.bma.pipeline import BMARunner

__all__ = [
    "BMAConfig",
    "ACT_CV_PRESETS",
    "BMARunner",
]
