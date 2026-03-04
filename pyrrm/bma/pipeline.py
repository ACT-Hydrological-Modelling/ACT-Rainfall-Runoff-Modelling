"""
BMARunner — orchestration class for the full BMA pipeline.

Ties together data loading, pre-screening, cross-validated evaluation
of all 5 levels, final model fitting, and results export.

Example::

    from pyrrm.bma import BMAConfig, BMARunner

    config = BMAConfig.from_preset("standard",
        model_predictions_path="data/predictions.csv",
        observed_flow_path="data/observed.csv",
    )
    runner = BMARunner(config)
    runner.load_data()
    runner.pre_screen()
    cv_results = runner.run_cross_validation()
    runner.fit_final()
    runner.save_results()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from pyrrm.bma.config import BMAConfig
from pyrrm.bma.data_prep import (
    apply_transform,
    back_transform,
    classify_regime,
    create_cv_splits,
    load_from_batch_result,
    load_from_csv,
    load_from_reports,
    regime_thresholds,
)
from pyrrm.bma.pre_screen import pre_screen
from pyrrm.bma.level1_equal import equal_weight_predict, equal_weights
from pyrrm.bma.level2_grc import grc_fit, grc_predict
from pyrrm.bma.level3_stacking import stacking_fit, stacking_predict
from pyrrm.bma.evaluation import (
    evaluate_by_regime,
    evaluate_deterministic,
    evaluate_probabilistic,
    fdc_error,
    pit_values,
)

if TYPE_CHECKING:
    from pyrrm.calibration.batch import BatchResult
    from pyrrm.calibration.report import CalibrationReport

logger = logging.getLogger(__name__)


class BMARunner:
    """End-to-end BMA ensemble pipeline.

    Attributes:
        config: BMAConfig instance.
        F_raw: Raw prediction DataFrame (T, K).
        y_obs: Observed flow Series.
        dates: DatetimeIndex.
        F_screened: Prediction matrix after pre-screening.
        kept_models: Model names retained after pre-screening.
        cv_results: Cross-validation results DataFrame.
        final_idata: ArviZ InferenceData from the final fit.
    """

    def __init__(self, config: BMAConfig) -> None:
        self.config = config
        self.F_raw: Optional[pd.DataFrame] = None
        self.y_obs: Optional[pd.Series] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.F_screened: Optional[np.ndarray] = None
        self.kept_models: Optional[List[str]] = None
        self._corr_matrix: Optional[np.ndarray] = None
        self._transform_params: Optional[Dict[str, Any]] = None
        self.cv_results: Optional[pd.DataFrame] = None
        self.final_idata: Optional[Any] = None
        self._final_weights_df: Optional[pd.DataFrame] = None

    # ── Data loading ─────────────────────────────────────────────────

    def load_data(
        self,
        batch_result: Optional["BatchResult"] = None,
        reports: Optional[Dict[str, "CalibrationReport"]] = None,
    ) -> None:
        """Load prediction matrix and observations.

        Exactly one of ``batch_result``, ``reports``, or
        ``config.model_predictions_path`` must be usable.
        """
        if batch_result is not None:
            self.F_raw, self.y_obs, self.dates = load_from_batch_result(batch_result)
        elif reports is not None:
            self.F_raw, self.y_obs, self.dates = load_from_reports(reports)
        elif self.config.model_predictions_path is not None:
            self.F_raw, self.y_obs, self.dates = load_from_csv(
                self.config.model_predictions_path,
                self.config.observed_flow_path,
            )
        else:
            raise ValueError(
                "Provide batch_result, reports, or set "
                "config.model_predictions_path / config.observed_flow_path."
            )
        logger.info(
            "Loaded %d models, %d timesteps.",
            self.F_raw.shape[1], len(self.dates),
        )

    # ── Pre-screening ────────────────────────────────────────────────

    def pre_screen(self) -> None:
        """Run three-step pre-screening (hard thresholds, clustering,
        regime specialist preservation)."""
        self._require_loaded()
        F_arr = self.F_raw.values
        y_arr = self.y_obs.values
        names = list(self.F_raw.columns)

        self.F_screened, self.kept_models, self._corr_matrix = pre_screen(
            F_arr, y_arr, names, self.config,
        )
        logger.info("Kept %d models: %s", len(self.kept_models), self.kept_models)

    # ── Cross-validation ─────────────────────────────────────────────

    def run_cross_validation(self) -> pd.DataFrame:
        """Cross-validated comparison of all 5 levels.

        Returns a DataFrame with one row per method and columns for
        each metric (averaged across folds).
        """
        self._require_screened()

        F_t, y_t, tparams = apply_transform(
            self.F_screened, self.y_obs.values, self.config,
        )
        self._transform_params = tparams

        cv_splits = create_cv_splits(self.dates, self.config, self.y_obs.values)
        n_folds = len(cv_splits)
        logger.info("Running %d-fold cross-validation...", n_folds)

        all_results: Dict[str, List[Dict[str, float]]] = {
            "equal": [], "grc": [], "stacking": [],
            "bma_global": [], "bma_regime": [],
        }

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info("─── CV Fold %d / %d ───", fold + 1, n_folds)
            F_train, F_val = F_t[train_idx], F_t[val_idx]
            y_train, y_val = y_t[train_idx], y_t[val_idx]
            y_val_orig = back_transform(y_val, tparams)

            # Level 1
            pred_eq = back_transform(equal_weight_predict(F_val), tparams)
            all_results["equal"].append(evaluate_deterministic(y_val_orig, pred_eq))

            # Level 2
            w_grc = grc_fit(F_train, y_train)
            pred_grc = back_transform(grc_predict(F_val, w_grc), tparams)
            all_results["grc"].append(evaluate_deterministic(y_val_orig, pred_grc))

            # Level 3
            inner_splits = create_cv_splits(
                self.dates[train_idx], self.config,
            )
            mapped_splits = []
            for ti, vi in inner_splits:
                mapped_splits.append((ti, vi))
            w_stack = stacking_fit(F_train, y_train, mapped_splits)
            pred_stack = back_transform(stacking_predict(F_val, w_stack), tparams)
            all_results["stacking"].append(evaluate_deterministic(y_val_orig, pred_stack))

            # Levels 4-5 (require PyMC)
            try:
                from pyrrm.bma.level4_bma import (
                    build_bma_model, check_convergence, sample_bma,
                )
                from pyrrm.bma.prediction import (
                    back_transform_predictions, generate_bma_predictions,
                )
                from pyrrm.bma.level5_regime_bma import (
                    build_regime_bma, compute_regime_blend_weights,
                    regime_blend_predict,
                )

                # Level 4
                bma_model = build_bma_model(F_train, y_train, self.config)
                bma_idata = sample_bma(bma_model, self.config)
                issues = check_convergence(bma_idata)
                if issues:
                    logger.warning("BMA convergence issues fold %d: %s", fold + 1, issues)
                bma_preds = generate_bma_predictions(bma_idata, F_val, self.config)
                bma_preds_orig = back_transform_predictions(bma_preds, tparams)
                bma_det = evaluate_deterministic(y_val_orig, bma_preds_orig["mean"])
                bma_prob = evaluate_probabilistic(y_val_orig, bma_preds_orig)
                all_results["bma_global"].append({**bma_det, **bma_prob})

                # Level 5
                regime_masks_train = classify_regime(y_train, self.config)
                q_high, q_low = regime_thresholds(y_train, self.config)
                blend_w_range = self.config.regime_blend_width * (q_high - q_low)

                regime_results = build_regime_bma(
                    F_train, y_train, regime_masks_train, self.config,
                )
                if regime_results:
                    flow_proxy = F_val.mean(axis=1)
                    blend_weights = compute_regime_blend_weights(
                        flow_proxy, q_high, q_low, blend_w_range,
                    )
                    regime_preds = regime_blend_predict(
                        F_val, regime_results, blend_weights, self.config,
                    )
                    regime_preds_orig = back_transform_predictions(regime_preds, tparams)
                    r_det = evaluate_deterministic(y_val_orig, regime_preds_orig["mean"])
                    r_prob = evaluate_probabilistic(y_val_orig, regime_preds_orig)
                    all_results["bma_regime"].append({**r_det, **r_prob})
                else:
                    all_results["bma_regime"].append(all_results["bma_global"][-1])

            except ImportError:
                logger.warning(
                    "PyMC not available — skipping Levels 4-5 for fold %d.", fold + 1,
                )

        self.cv_results = self._aggregate_cv(all_results)
        return self.cv_results

    @staticmethod
    def _aggregate_cv(
        all_results: Dict[str, List[Dict[str, float]]],
    ) -> pd.DataFrame:
        """Average metrics across folds for each method."""
        rows = []
        for method, fold_results in all_results.items():
            if not fold_results:
                continue
            keys = fold_results[0].keys()
            avg: Dict[str, float] = {}
            for k in keys:
                vals = [r[k] for r in fold_results if k in r and r[k] is not None and not np.isnan(r.get(k, np.nan))]
                avg[k] = float(np.mean(vals)) if vals else np.nan
            rows.append({"method": method, **avg})
        return pd.DataFrame(rows).set_index("method")

    # ── Final fit on full data ───────────────────────────────────────

    def fit_final(self) -> None:
        """Fit BMA on the full dataset to get production weights."""
        self._require_screened()

        F_t, y_t, tparams = apply_transform(
            self.F_screened, self.y_obs.values, self.config,
        )
        self._transform_params = tparams

        from pyrrm.bma.level4_bma import (
            build_bma_model, extract_weights, sample_bma,
        )

        logger.info("Fitting final BMA on full dataset...")
        model = build_bma_model(F_t, y_t, self.config)
        self.final_idata = sample_bma(model, self.config)
        self._final_weights_df = extract_weights(self.final_idata, self.kept_models)
        logger.info("Final BMA weights:\n%s", self._final_weights_df)

    # ── Save results ─────────────────────────────────────────────────

    def save_results(self) -> None:
        """Save CV results, final weights, and InferenceData to disk."""
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        if self.cv_results is not None:
            path = out / "cv_results.csv"
            self.cv_results.to_csv(path)
            logger.info("Saved CV results → %s", path)

        if self._final_weights_df is not None:
            path = out / "posterior_weights.csv"
            self._final_weights_df.to_csv(path)
            logger.info("Saved weights → %s", path)

        if self.final_idata is not None:
            path = out / "final_bma_model.nc"
            self.final_idata.to_netcdf(str(path))
            logger.info("Saved InferenceData → %s", path)

    # ── Quick summary plots ──────────────────────────────────────────

    def plot_summary(self) -> Dict[str, Any]:
        """Generate summary plots and return them in a dict.

        Keys: ``'weight_comparison'``, ``'method_comparison'``,
        and optionally ``'posterior_weights'``.
        """
        from pyrrm.bma.visualization import (
            plot_method_comparison,
            plot_posterior_weights,
            plot_weight_comparison,
        )

        figs: Dict[str, Any] = {}

        if self.cv_results is not None:
            figs["method_comparison"] = plot_method_comparison(self.cv_results)

        if self.final_idata is not None and self.kept_models is not None:
            figs["posterior_weights"] = plot_posterior_weights(
                self.final_idata, self.kept_models,
            )

        return figs

    # ── Helpers ───────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self.F_raw is None:
            raise RuntimeError("Call load_data() first.")

    def _require_screened(self) -> None:
        self._require_loaded()
        if self.F_screened is None:
            raise RuntimeError("Call pre_screen() first.")
