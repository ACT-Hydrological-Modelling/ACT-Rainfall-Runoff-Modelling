"""
Model pre-screening: reduce K models to ~10-15 before BMA fitting.

Three-step process (spec Section 4):

1. **Hard threshold removal** -- drop models with NSE < 0, KGE < -0.41,
   or |PBIAS| > 25 %.
2. **Residual correlation clustering** -- hierarchical clustering on
   Pearson correlation of model residuals, keeping one representative
   per cluster (best mean KGE).
3. **Regime specialist preservation** -- protect models that are
   best-in-class for any flow regime even if they would be clustered away.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from pyrrm.bma.config import BMAConfig

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Step 1 — Hard threshold removal
# ═════════════════════════════════════════════════════════════════════════

def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    ss_res = np.nansum((obs - sim) ** 2)
    ss_tot = np.nansum((obs - np.nanmean(obs)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    valid = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[valid], sim[valid]
    if len(o) < 2:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = float(np.std(s) / np.std(o)) if np.std(o) > 0 else np.nan
    beta = float(np.mean(s) / np.mean(o)) if np.mean(o) != 0 else np.nan
    if np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    s = np.nansum(obs)
    return float(100.0 * np.nansum(sim - obs) / s) if abs(s) > 0 else np.nan


def hard_threshold_filter(
    F: np.ndarray,
    y_obs: np.ndarray,
    model_names: List[str],
    config: "BMAConfig",
) -> List[int]:
    """Return indices of models that pass all hard thresholds."""
    kept: List[int] = []
    for k, name in enumerate(model_names):
        sim = F[:, k]
        nse_val = _nse(y_obs, sim)
        kge_val = _kge(y_obs, sim)
        pbias_val = _pbias(y_obs, sim)

        if nse_val < config.nse_threshold:
            logger.info("Removing '%s': NSE=%.3f < %.2f", name, nse_val, config.nse_threshold)
            continue
        if kge_val < config.kge_threshold:
            logger.info("Removing '%s': KGE=%.3f < %.2f", name, kge_val, config.kge_threshold)
            continue
        if abs(pbias_val) > config.pbias_threshold:
            logger.info("Removing '%s': |PBIAS|=%.1f%% > %.1f%%", name, abs(pbias_val), config.pbias_threshold)
            continue
        kept.append(k)

    logger.info(
        "Hard threshold: kept %d / %d models.", len(kept), len(model_names)
    )
    return kept


# ═════════════════════════════════════════════════════════════════════════
# Step 2 — Residual correlation clustering
# ═════════════════════════════════════════════════════════════════════════

def residual_correlation_clustering(
    F: np.ndarray,
    y_obs: np.ndarray,
    model_names: List[str],
    config: "BMAConfig",
) -> List[int]:
    """Cluster models by residual similarity, keep best per cluster.

    Returns indices into the *current* model set (post-Step-1).
    """
    K = F.shape[1]
    if K <= 1:
        return list(range(K))

    residuals = F - y_obs[:, None]  # (T, K)
    corr = np.corrcoef(residuals.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)

    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=config.cluster_method)
    cluster_ids = fcluster(Z, t=1.0 - config.residual_corr_threshold, criterion="distance")

    kge_scores = np.array([_kge(y_obs, F[:, k]) for k in range(K)])

    kept: List[int] = []
    for cid in np.unique(cluster_ids):
        members = np.where(cluster_ids == cid)[0]
        best = members[np.nanargmax(kge_scores[members])]
        kept.append(int(best))
        if len(members) > 1:
            dropped = [model_names[m] for m in members if m != best]
            logger.info(
                "Cluster %d: keeping '%s' (KGE=%.3f), dropping %s",
                cid, model_names[best], kge_scores[best], dropped,
            )

    logger.info(
        "Residual clustering: kept %d / %d models (threshold=%.2f).",
        len(kept), K, config.residual_corr_threshold,
    )
    return sorted(kept)


# ═════════════════════════════════════════════════════════════════════════
# Step 3 — Regime specialist preservation
# ═════════════════════════════════════════════════════════════════════════

def preserve_regime_specialists(
    F_full: np.ndarray,
    y_obs: np.ndarray,
    all_names: List[str],
    kept_indices: List[int],
    config: "BMAConfig",
) -> List[int]:
    """Add back models that are best-in-class for any flow regime.

    Operates on the *full* pre-Step-1 model set so that a model removed
    by clustering can be recovered if it excels in a regime.
    """
    from pyrrm.bma.data_prep import classify_regime

    regimes = classify_regime(y_obs, config)
    kept_set = set(kept_indices)
    added: List[str] = []

    for regime_name, mask in regimes.items():
        if mask.sum() < 10:
            continue
        obs_regime = y_obs[mask]
        best_k, best_kge = -1, -np.inf
        for k in range(F_full.shape[1]):
            score = _kge(obs_regime, F_full[mask, k])
            if score > best_kge:
                best_kge = score
                best_k = k
        if best_k >= 0 and best_k not in kept_set:
            kept_set.add(best_k)
            added.append(f"{all_names[best_k]} (best {regime_name}, KGE={best_kge:.3f})")

    if added:
        logger.info("Regime specialists added back: %s", added)

    return sorted(kept_set)


# ═════════════════════════════════════════════════════════════════════════
# Combined pre-screening pipeline
# ═════════════════════════════════════════════════════════════════════════

def pre_screen(
    F: np.ndarray,
    y_obs: np.ndarray,
    model_names: List[str],
    config: "BMAConfig",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Run the full three-step pre-screening pipeline.

    Args:
        F: (T, K) predictions.
        y_obs: (T,) observations.
        model_names: length-K list of model names.
        config: BMAConfig.

    Returns:
        (F_screened, kept_names, residual_corr_matrix)
    """
    K_orig = F.shape[1]

    # Step 1
    step1 = hard_threshold_filter(F, y_obs, model_names, config)
    if not step1:
        raise ValueError("All models were removed by hard thresholds.")

    F1 = F[:, step1]
    names1 = [model_names[i] for i in step1]

    # Step 2
    step2 = residual_correlation_clustering(F1, y_obs, names1, config)
    if not step2:
        raise ValueError("All models were removed by residual clustering.")

    step2_global = [step1[i] for i in step2]

    # Step 3
    if config.preserve_regime_specialists:
        final_global = preserve_regime_specialists(
            F, y_obs, model_names, step2_global, config,
        )
    else:
        final_global = step2_global

    F_out = F[:, final_global]
    names_out = [model_names[i] for i in final_global]

    residuals = F_out - y_obs[:, None]
    corr = np.corrcoef(residuals.T)

    logger.info(
        "Pre-screening complete: %d → %d models.", K_orig, len(names_out),
    )
    return F_out, names_out, corr
