"""
Level 4: Global BMA in PyMC.

Dirichlet-weighted Gaussian mixture with optional bias correction and
heteroscedastic residual variance.  Uses the NUTS sampler (preferably
via the NUMPyro JAX backend for speed).

Key design notes:

* **Label switching is NOT a problem** — unlike unsupervised mixtures,
  component *k* is always model *k*'s predictions, so the posterior is
  identified without post-processing.
* **Sparse Dirichlet (alpha = 1/K)** encourages the posterior to zero
  out redundant models.
* **Manual logsumexp likelihood** (``pm.Potential``) is the default
  because ``pm.Mixture`` can have shape issues with large K.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyrrm.bma.config import BMAConfig

logger = logging.getLogger(__name__)

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

_PYMC_ERR = (
    "PyMC is required for BMA Levels 4-5. "
    "Install with: pip install 'pyrrm[bma]'  "
    "(or: pip install pymc arviz numpyro jax jaxlib)"
)


def _require_pymc() -> None:
    if not PYMC_AVAILABLE:
        raise ImportError(_PYMC_ERR)


# ═════════════════════════════════════════════════════════════════════════
# Model construction
# ═════════════════════════════════════════════════════════════════════════

def build_bma_model(
    F_train: np.ndarray,
    y_train: np.ndarray,
    config: "BMAConfig",
) -> "pm.Model":
    """Build a PyMC BMA model as a Dirichlet-weighted Gaussian mixture.

    Args:
        F_train: (T, K) array of model predictions (transformed if applicable).
        y_train: (T,) array of observations (transformed if applicable).
        config: BMAConfig instance.

    Returns:
        A ``pm.Model`` ready for sampling.
    """
    _require_pymc()

    K = F_train.shape[1]

    if config.dirichlet_alpha == "sparse":
        alpha_val = 1.0 / K
    elif config.dirichlet_alpha == "uniform":
        alpha_val = 1.0
    else:
        alpha_val = float(config.dirichlet_alpha)

    with pm.Model() as model:
        # ── Weights on the K-simplex ─────────────────────────────────
        w = pm.Dirichlet("w", a=np.ones(K) * alpha_val)

        # ── Bias correction ──────────────────────────────────────────
        if config.bias_correction == "additive":
            bias = pm.Normal(
                "bias", mu=0, sigma=config.bias_prior_sigma, shape=K,
            )
            mu = F_train + bias[None, :]
        elif config.bias_correction == "linear":
            a = pm.Normal(
                "a", mu=0, sigma=config.bias_prior_sigma, shape=K,
            )
            b = pm.Normal("b", mu=1, sigma=0.2, shape=K)
            mu = a[None, :] + b[None, :] * F_train
        else:
            mu = pt.as_tensor_variable(F_train)

        # ── Residual variance ────────────────────────────────────────
        if config.heteroscedastic:
            sigma = pm.HalfNormal(
                "sigma", sigma=config.sigma_prior_sigma, shape=K,
            )
        else:
            sigma = pm.HalfNormal(
                "sigma", sigma=config.sigma_prior_sigma,
            )

        # ── Likelihood ───────────────────────────────────────────────
        if config.use_manual_loglik:
            sig = sigma[None, :] if config.heteroscedastic else sigma
            log_comp = (
                -0.5 * pt.log(2 * np.pi)
                - pt.log(sig)
                - 0.5 * ((y_train[:, None] - mu) / sig) ** 2
            )
            log_w = pt.log(w)[None, :]
            log_lik = pm.math.logsumexp(log_w + log_comp, axis=1)
            pm.Potential("loglik", log_lik.sum())
        else:
            if config.heteroscedastic:
                comp_dists = pm.Normal.dist(
                    mu=mu,
                    sigma=sigma[None, :] * pt.ones_like(mu),
                )
            else:
                comp_dists = pm.Normal.dist(mu=mu, sigma=sigma)
            pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=y_train)

    return model


# ═════════════════════════════════════════════════════════════════════════
# Sampling
# ═════════════════════════════════════════════════════════════════════════

def sample_bma(
    model: "pm.Model",
    config: "BMAConfig",
) -> "az.InferenceData":
    """Sample the BMA model and return ArviZ InferenceData."""
    _require_pymc()

    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            init=config.init,
            nuts_sampler=config.nuts_sampler,
            random_seed=config.random_seed,
        )
    return idata


# ═════════════════════════════════════════════════════════════════════════
# Convergence diagnostics
# ═════════════════════════════════════════════════════════════════════════

def check_convergence(idata: "az.InferenceData") -> Dict[str, Any]:
    """Check R-hat, ESS, and divergences.

    Returns a dict of issues found.  Empty dict means all is well.
    """
    _require_pymc()

    issues: Dict[str, Any] = {}

    rhat = az.rhat(idata)
    for var in rhat:
        max_rhat = float(rhat[var].max())
        if max_rhat > 1.01:
            issues[f"rhat_{var}"] = max_rhat

    for kind in ("bulk", "tail"):
        try:
            ess = az.ess(idata, kind=kind)
        except TypeError:
            ess = az.ess(idata, method=kind)
        for var in ess:
            min_ess = float(ess[var].min())
            if min_ess < 400:
                issues[f"ess_{kind}_{var}"] = min_ess

    if hasattr(idata, "sample_stats"):
        div_field = idata.sample_stats.get("diverging", None)
        if div_field is not None:
            n_div = int(div_field.values.sum())
            if n_div > 0:
                issues["divergences"] = n_div

    return issues


def extract_weights(
    idata: "az.InferenceData",
    model_names: List[str],
) -> pd.DataFrame:
    """Extract posterior weight summary with model names as index."""
    _require_pymc()

    summary = az.summary(idata, var_names=["w"], hdi_prob=0.94, kind="stats")
    summary.index = model_names
    return summary
