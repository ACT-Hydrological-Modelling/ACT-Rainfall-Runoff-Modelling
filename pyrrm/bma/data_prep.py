"""
Data loading, transformation, temporal CV splits, and flow regime classification.

Supports two ingestion paths:

1. ``load_from_batch_result`` -- extract an aligned (T, K) prediction matrix
   directly from a ``BatchResult`` (or dict of ``CalibrationReport``).
2. ``load_from_csv`` -- load from CSV files (standalone mode).

Cross-validation splits are water-year-aligned (or calendar/custom year)
block temporal splits with configurable block size and buffer zones.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from pyrrm.bma.config import BMAConfig
    from pyrrm.calibration.batch import BatchResult
    from pyrrm.calibration.report import CalibrationReport

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════

def load_from_batch_result(
    batch_result: "BatchResult",
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Extract an aligned prediction matrix from *batch_result*.

    Each successful experiment in the ``BatchResult`` contributes one
    column (keyed by experiment name) to the returned DataFrame.

    Returns:
        (F, y_obs, dates) where F is (T, K) with model names as columns,
        y_obs is (T,), and dates is the shared DatetimeIndex.

    Raises:
        ValueError: If no successful experiments exist or dates cannot
            be aligned.
    """
    reports: Dict[str, "CalibrationReport"] = batch_result.results
    if not reports:
        raise ValueError("BatchResult contains no successful experiments.")

    first_key = next(iter(reports))
    ref = reports[first_key]
    dates = ref.dates
    y_obs = pd.Series(ref.observed, index=dates, name="observed")

    predictions: Dict[str, np.ndarray] = {}
    for key, report in reports.items():
        if len(report.simulated) != len(dates):
            warnings.warn(
                f"Skipping '{key}': length {len(report.simulated)} "
                f"!= reference length {len(dates)}."
            )
            continue
        if not report.dates.equals(dates):
            warnings.warn(
                f"Skipping '{key}': date index does not match reference."
            )
            continue
        predictions[key] = report.simulated

    if not predictions:
        raise ValueError(
            "No experiments have matching date indices after alignment."
        )

    F = pd.DataFrame(predictions, index=dates)
    return F, y_obs, dates


def load_from_reports(
    reports: Dict[str, "CalibrationReport"],
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Convenience wrapper accepting a plain dict of CalibrationReports."""
    from pyrrm.calibration.batch import BatchResult

    br = BatchResult(results=reports)
    return load_from_batch_result(br)


def load_from_csv(
    predictions_path: Path,
    observed_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load predictions and observations from CSV files.

    ``predictions_path`` should be a CSV with a ``date`` column and one
    column per model.  ``observed_path`` should have ``date`` and ``Q``
    (or ``observed_flow``) columns.
    """
    pred_df = pd.read_csv(predictions_path, parse_dates=["date"], index_col="date")
    obs_df = pd.read_csv(observed_path, parse_dates=["date"], index_col="date")

    q_col = None
    for col in ("Q", "observed_flow", "flow", "q"):
        if col in obs_df.columns:
            q_col = col
            break
    if q_col is None:
        raise ValueError(
            f"Cannot identify observed flow column in {observed_path}. "
            "Expected one of: Q, observed_flow, flow, q."
        )

    common = pred_df.index.intersection(obs_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between predictions and observations.")

    common = common.sort_values()
    F = pred_df.loc[common]
    y_obs = obs_df.loc[common, q_col]
    y_obs.name = "observed"

    mask = ~(F.isna().any(axis=1) | y_obs.isna())
    F = F.loc[mask]
    y_obs = y_obs.loc[mask]
    dates = F.index

    logger.info("Loaded %d models, %d timesteps from CSV.", F.shape[1], len(dates))
    return F, y_obs, dates


# ═════════════════════════════════════════════════════════════════════════
# Flow transformations
# ═════════════════════════════════════════════════════════════════════════

def apply_transform(
    F: np.ndarray,
    y_obs: np.ndarray,
    config: "BMAConfig",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply a flow transformation to predictions and observations.

    Returns (F_t, y_t, params) where *params* stores everything needed
    for ``back_transform``.
    """
    method = config.transform
    params: Dict[str, Any] = {"method": method}

    if method == "none":
        return F.copy(), y_obs.copy(), params

    if method == "log":
        eps = config.log_epsilon
        params["epsilon"] = eps
        return np.log(F + eps), np.log(y_obs + eps), params

    if method == "sqrt":
        return np.sqrt(np.maximum(F, 0.0)), np.sqrt(np.maximum(y_obs, 0.0)), params

    if method == "boxcox":
        lam = config.boxcox_lambda
        if lam is None:
            _, lam = sp_stats.boxcox(np.maximum(y_obs, 1e-6))
        params["lambda"] = lam
        if abs(lam) < 1e-10:
            return np.log(np.maximum(F, 1e-6)), np.log(np.maximum(y_obs, 1e-6)), params
        return (
            (np.maximum(F, 1e-6) ** lam - 1) / lam,
            (np.maximum(y_obs, 1e-6) ** lam - 1) / lam,
            params,
        )

    raise ValueError(
        f"Unknown transform '{method}'. Choose from: 'none', 'log', 'sqrt', 'boxcox'."
    )


def back_transform(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Inverse-transform predictions back to original flow units."""
    method = params["method"]

    if method == "none":
        return values.copy()
    if method == "log":
        return np.exp(values) - params["epsilon"]
    if method == "sqrt":
        return np.maximum(values, 0.0) ** 2
    if method == "boxcox":
        lam = params["lambda"]
        if abs(lam) < 1e-10:
            return np.exp(values)
        return np.maximum(lam * values + 1, 1e-10) ** (1.0 / lam)

    raise ValueError(f"Unknown transform '{method}'.")


# ═════════════════════════════════════════════════════════════════════════
# Temporal cross-validation splits
# ═════════════════════════════════════════════════════════════════════════

def _resolve_start_month(config: "BMAConfig") -> int:
    return config.resolved_start_month


def _identify_complete_years(
    dates: pd.DatetimeIndex,
    start_month: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Identify complete year periods in the record.

    A year period runs from the 1st of ``start_month`` in one calendar
    year to the day before ``start_month`` in the next.  Only periods
    fully covered by *dates* are returned.
    """
    first, last = dates.min(), dates.max()

    start_year = first.year if first.month < start_month else first.year
    if first > pd.Timestamp(start_year, start_month, 1):
        start_year += 1

    years: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    yr = start_year
    while True:
        begin = pd.Timestamp(yr, start_month, 1)
        end = pd.Timestamp(yr + 1, start_month, 1) - pd.Timedelta(days=1)
        if begin < first or end > last:
            yr += 1
            if begin > last:
                break
            continue
        years.append((begin, end))
        yr += 1

    return years


def _years_to_indices(
    year_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    dates: pd.DatetimeIndex,
) -> List[np.ndarray]:
    """Convert year periods to arrays of integer indices into *dates*."""
    date_series = dates.to_series()
    blocks: List[np.ndarray] = []
    for begin, end in year_periods:
        mask = (date_series >= begin) & (date_series <= end)
        blocks.append(np.where(mask.values)[0])
    return blocks


def _group_blocks(
    year_blocks: List[np.ndarray],
    block_years: int,
) -> List[np.ndarray]:
    """Merge consecutive year-blocks into multi-year blocks."""
    groups: List[np.ndarray] = []
    for i in range(0, len(year_blocks), block_years):
        chunk = year_blocks[i : i + block_years]
        if len(chunk) == block_years:
            groups.append(np.concatenate(chunk))
    return groups


def _apply_buffer(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    buffer_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove buffer_days of indices from train that are adjacent to val."""
    if buffer_days <= 0 or len(val_idx) == 0 or len(train_idx) == 0:
        return train_idx, val_idx

    val_min, val_max = int(val_idx.min()), int(val_idx.max())
    keep = (
        (train_idx < val_min - buffer_days)
        | (train_idx > val_max + buffer_days)
    )
    return train_idx[keep], val_idx


def _check_fold_quality(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dates: pd.DatetimeIndex,
    y_obs: np.ndarray,
    fold_num: int,
    config: "BMAConfig",
) -> None:
    """Log warnings if a fold fails quality checks."""
    min_train = int(config.min_train_years * 365.25)
    min_val = int(config.min_val_years * 365.25)

    if len(train_idx) < min_train:
        logger.warning(
            "Fold %d: training set (%d days) < min_train_years (%.1f yr = %d days).",
            fold_num, len(train_idx), config.min_train_years, min_train,
        )
    if len(val_idx) < min_val:
        logger.warning(
            "Fold %d: validation set (%d days) < min_val_years (%.1f yr = %d days).",
            fold_num, len(val_idx), config.min_val_years, min_val,
        )

    if config.require_seasonal_coverage and len(train_idx) > 0:
        months = dates[train_idx].month
        for season_name, season_months in [
            ("DJF", {12, 1, 2}), ("MAM", {3, 4, 5}),
            ("JJA", {6, 7, 8}), ("SON", {9, 10, 11}),
        ]:
            frac = np.isin(months, list(season_months)).mean()
            if frac < 0.05:
                logger.warning(
                    "Fold %d: season %s has only %.1f%% of training data.",
                    fold_num, season_name, frac * 100,
                )

    if len(val_idx) > 0:
        val_flow = y_obs[val_idx]
        valid = val_flow[~np.isnan(val_flow)]
        if len(valid) > 0:
            p95 = np.percentile(y_obs[~np.isnan(y_obs)], 95)
            has_flood = np.any(valid > p95)
            if not has_flood:
                logger.warning(
                    "Fold %d: no flood events (>Q95) in validation set.",
                    fold_num,
                )

    nan_frac_train = np.isnan(y_obs[train_idx]).mean() if len(train_idx) > 0 else 0
    nan_frac_val = np.isnan(y_obs[val_idx]).mean() if len(val_idx) > 0 else 0
    if nan_frac_train > 0.05:
        logger.warning("Fold %d: %.1f%% NaN in training.", fold_num, nan_frac_train * 100)
    if nan_frac_val > 0.05:
        logger.warning("Fold %d: %.1f%% NaN in validation.", fold_num, nan_frac_val * 100)


def create_block_cv_splits(
    dates: pd.DatetimeIndex,
    config: "BMAConfig",
    y_obs: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create block temporal CV splits with flexible year boundary and
    block size.

    Args:
        dates: DatetimeIndex for the full record.
        config: BMAConfig with CV parameters.
        y_obs: Observed flow (optional, used for fold quality checks).

    Returns:
        List of ``(train_indices, val_indices)`` tuples.
    """
    start_month = _resolve_start_month(config)
    year_periods = _identify_complete_years(dates, start_month)

    if len(year_periods) < 2:
        raise ValueError(
            f"Need at least 2 complete years for CV; found {len(year_periods)} "
            f"(start month={start_month})."
        )

    year_blocks = _years_to_indices(year_periods, dates)
    groups = _group_blocks(year_blocks, config.cv_block_years)

    if len(groups) < 2:
        raise ValueError(
            f"Block size {config.cv_block_years} yields only {len(groups)} "
            f"blocks from {len(year_periods)} years -- need at least 2."
        )

    n_folds = config.n_cv_folds if config.n_cv_folds is not None else len(groups)
    n_folds = min(n_folds, len(groups))

    if n_folds < len(groups):
        merged: List[np.ndarray] = []
        base, extra = divmod(len(groups), n_folds)
        idx = 0
        for i in range(n_folds):
            size = base + (1 if i < extra else 0)
            merged.append(np.concatenate(groups[idx : idx + size]))
            idx += size
        groups = merged

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(groups)):
        val_idx = groups[i]
        train_parts = [groups[j] for j in range(len(groups)) if j != i]
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        train_idx, val_idx = _apply_buffer(train_idx, val_idx, config.buffer_days)
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)

        if y_obs is not None:
            _check_fold_quality(train_idx, val_idx, dates, y_obs, i + 1, config)

        splits.append((train_idx, val_idx))

    logger.info(
        "Created %d block CV folds (block=%d yr, start_month=%d, buffer=%d d).",
        len(splits), config.cv_block_years, start_month, config.buffer_days,
    )
    return splits


def create_expanding_window_splits(
    dates: pd.DatetimeIndex,
    config: "BMAConfig",
    y_obs: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV with flexible year boundary and block size.

    Each fold: train on all data up to block N, validate on block N+1.
    """
    start_month = _resolve_start_month(config)
    year_periods = _identify_complete_years(dates, start_month)
    year_blocks = _years_to_indices(year_periods, dates)
    groups = _group_blocks(year_blocks, config.cv_block_years)

    min_train_days = int(config.min_train_years * 365.25)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(groups)):
        train_idx = np.concatenate(groups[:i])
        val_idx = groups[i]

        if len(train_idx) < min_train_days:
            continue

        train_idx, val_idx = _apply_buffer(train_idx, val_idx, config.buffer_days)
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)

        if y_obs is not None:
            _check_fold_quality(train_idx, val_idx, dates, y_obs, len(splits) + 1, config)

        splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError(
            f"No valid expanding-window folds (min_train_years={config.min_train_years})."
        )

    logger.info(
        "Created %d expanding-window folds (block=%d yr, start_month=%d, buffer=%d d).",
        len(splits), config.cv_block_years, start_month, config.buffer_days,
    )
    return splits


def create_cv_splits(
    dates: pd.DatetimeIndex,
    config: "BMAConfig",
    y_obs: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dispatch to the appropriate CV strategy based on *config*."""
    if config.cv_strategy == "block":
        return create_block_cv_splits(dates, config, y_obs)
    if config.cv_strategy == "expanding_window":
        return create_expanding_window_splits(dates, config, y_obs)
    raise ValueError(
        f"Unknown cv_strategy '{config.cv_strategy}'. "
        "Choose from: 'block', 'expanding_window'."
    )


# ═════════════════════════════════════════════════════════════════════════
# Flow regime classification
# ═════════════════════════════════════════════════════════════════════════

def classify_regime(
    y_train: np.ndarray,
    config: "BMAConfig",
) -> Dict[str, np.ndarray]:
    """Classify timesteps into flow regimes using training-set quantiles.

    Returns a dict mapping ``'high'``, ``'medium'``, ``'low'`` to
    boolean masks of the same length as *y_train*.  Quantile thresholds
    are derived from ``config.regime_quantiles``.
    """
    q_lo, q_hi = config.regime_quantiles
    valid = y_train[~np.isnan(y_train)]

    thresh_high = np.percentile(valid, 100 * (1 - q_lo))
    thresh_low = np.percentile(valid, 100 * (1 - q_hi))

    return {
        "high": y_train >= thresh_high,
        "medium": (y_train >= thresh_low) & (y_train < thresh_high),
        "low": y_train < thresh_low,
    }


def regime_thresholds(
    y_train: np.ndarray,
    config: "BMAConfig",
) -> Tuple[float, float]:
    """Return (q_high, q_low) flow thresholds from training data."""
    q_lo, q_hi = config.regime_quantiles
    valid = y_train[~np.isnan(y_train)]
    return (
        float(np.percentile(valid, 100 * (1 - q_lo))),
        float(np.percentile(valid, 100 * (1 - q_hi))),
    )
