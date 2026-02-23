"""
Time-varying parameter (TVP) prior specifications for NumPyro NUTS.

Provides prior distributions that produce per-timestep parameter
trajectories inside a NumPyro model.  Currently implements the
Gaussian Random Walk (GRW); the ``TVPPrior`` protocol is designed
so that additional priors (GP, changepoint, seasonal) can be added
without modifying the adapter or forward model.

Usage inside ``CalibrationRunner.run_nuts``::

    from pyrrm.calibration.tvp_priors import GaussianRandomWalk

    tvp_config = {
        "X1": GaussianRandomWalk(lower=1.0, upper=1500.0, resolution=5),
    }
    result = runner.run_nuts(tvp_config=tvp_config, ...)

Reference:
    Santos, L., Thirel, G., & Perrin, C. (2022).  Continuous state-space
    representation of a bucket-type rainfall-runoff model.  WRR.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

try:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    _NUMPYRO_OK = True
except ImportError:
    _NUMPYRO_OK = False


class TVPPrior(ABC):
    """Protocol for time-varying parameter priors.

    Subclasses must implement:

    * ``sample_numpyro(name, n_timesteps)`` -- called inside a NumPyro
      model context to register the appropriate sample sites and return
      a JAX array of shape ``(n_timesteps,)`` representing the parameter
      trajectory.

    * ``hyperparameter_names(name)`` -- return the list of NumPyro
      sample-site names that this prior creates (used by post-processing
      to identify which InferenceData variables belong to the TVP).
    """

    @abstractmethod
    def sample_numpyro(self, name: str, n_timesteps: int):
        """Sample the TVP trajectory inside a NumPyro model.

        Args:
            name: Base parameter name (e.g. ``"X1"``).
            n_timesteps: Length of the forcing time series.

        Returns:
            ``jnp.ndarray`` of shape ``(n_timesteps,)`` registered as
            ``numpyro.deterministic(name, ...)``.
        """

    @abstractmethod
    def hyperparameter_names(self, name: str) -> List[str]:
        """Return NumPyro sample-site names created by this prior."""


@dataclass
class GaussianRandomWalk(TVPPrior):
    r"""Gaussian Random Walk prior for a time-varying parameter.

    The trajectory is built as:

    .. math::

        \alpha &\sim \mathrm{Uniform}(\text{lower}, \text{upper}) \\
        \sigma_\delta &\sim \mathrm{HalfNormal}(\text{sigma\_delta\_scale}) \\
        \delta_{t'} &\sim \mathcal{N}(0, \sigma_\delta^2)
            \quad\text{for } t' \in \{1, 2, \ldots, T/r\} \\
        \theta_t &= \alpha + \sum_{t' \leq \lceil t/r \rceil} \delta_{t'}

    where *r* is the ``resolution`` (timesteps per increment).

    Args:
        lower: Lower bound for the intercept Uniform prior.
        upper: Upper bound for the intercept Uniform prior.
        sigma_delta_scale: Scale of the HalfNormal prior on the
            innovation standard deviation (default 3.0).
        resolution: Number of data timesteps per GRW increment
            (default 1 = one increment per timestep).
        prefix_zero: If True, the first delta is forced to zero so that
            the walk starts exactly at the intercept (default False).
    """

    lower: float = 0.0
    upper: float = 1000.0
    sigma_delta_scale: float = 3.0
    resolution: int = 1
    prefix_zero: bool = False

    def sample_numpyro(self, name: str, n_timesteps: int):
        if not _NUMPYRO_OK:
            raise ImportError(
                "NumPyro is required for TVP priors. "
                "Install with: pip install jax jaxlib numpyro"
            )

        intercept = numpyro.sample(
            f"{name}_intercept", dist.Uniform(self.lower, self.upper)
        )
        sigma_delta = numpyro.sample(
            f"{name}_sigma_delta", dist.HalfNormal(self.sigma_delta_scale)
        )

        n_deltas = math.ceil(n_timesteps / self.resolution)

        if self.prefix_zero:
            raw_delta = numpyro.sample(
                f"{name}_delta", dist.Normal(0.0, sigma_delta).expand([n_deltas - 1])
            )
            delta = jnp.concatenate([jnp.zeros(1), raw_delta])
        else:
            delta = numpyro.sample(
                f"{name}_delta", dist.Normal(0.0, sigma_delta).expand([n_deltas])
            )

        cumulative = jnp.cumsum(delta)

        if self.resolution > 1:
            tvp = intercept + jnp.repeat(cumulative, self.resolution)[:n_timesteps]
        else:
            tvp = intercept + cumulative

        return numpyro.deterministic(name, tvp)

    def hyperparameter_names(self, name: str) -> List[str]:
        return [f"{name}_intercept", f"{name}_sigma_delta", f"{name}_delta"]
