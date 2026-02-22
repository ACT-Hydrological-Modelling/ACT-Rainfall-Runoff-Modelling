"""
JAX-compatible log-likelihood functions for MCMC calibration.

Pure-function ports of GaussianLikelihood and TransformedGaussianLikelihood
from pyrrm.calibration.objective_functions, plus an AR(1) error model
for autocorrelated residuals.

All functions are JIT-compatible and differentiable via jax.grad.
"""

import jax.numpy as jnp

_EPS = 1e-10


# -----------------------------------------------------------------------
# Flow transformations (mirrors TransformedGaussianLikelihood._apply_transform)
# -----------------------------------------------------------------------

def apply_transform_jax(
    values: jnp.ndarray,
    reference: jnp.ndarray,
    transform: str = "none",
    transform_params: dict = None,
) -> jnp.ndarray:
    """
    Apply a flow transformation in JAX-compatible fashion.

    The transform string is resolved *before* JIT tracing so Python
    ``if/elif`` dispatch is fine.

    Args:
        values: Array to transform (simulated or observed flows).
        reference: Reference array for epsilon calculation (typically obs).
        transform: One of ``"none"``, ``"sqrt"``, ``"log"``, ``"inverse"``,
            ``"power"``, ``"boxcox"``, ``"squared"``, ``"inverse_squared"``.
        transform_params: Optional overrides:
            ``{"epsilon_value": float, "power_exp": float, "boxcox_lambda": float}``.

    Returns:
        Transformed array.
    """
    if transform_params is None:
        transform_params = {}
    epsilon_frac = transform_params.get("epsilon_value", 0.01)
    eps = jnp.mean(jnp.where(reference > 0, reference, 0.0)) * epsilon_frac + _EPS

    if transform == "none":
        return values
    elif transform == "sqrt":
        return jnp.sqrt(values + eps)
    elif transform == "log":
        return jnp.log(values + eps)
    elif transform == "inverse":
        return 1.0 / (values + eps)
    elif transform == "power":
        p = transform_params.get("power_exp", 0.2)
        return (values + eps) ** p
    elif transform == "boxcox":
        lam = transform_params.get("boxcox_lambda", 0.25)
        return ((values + eps) ** lam - 1.0) / lam
    elif transform == "squared":
        return values ** 2
    elif transform == "inverse_squared":
        return 1.0 / (values + eps) ** 2
    else:
        return values


# -----------------------------------------------------------------------
# Gaussian log-likelihood (sigma integrated out -- matches existing code)
# -----------------------------------------------------------------------

def gaussian_log_likelihood_integrated_jax(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    warmup_steps: int = 365,
) -> jnp.ndarray:
    """
    Gaussian log-likelihood with sigma integrated out.

    Matches ``GaussianLikelihood.calculate()``::

        log_lik = -n/2 * log(SSE)

    Args:
        sim: Simulated flow array.
        obs: Observed flow array.
        warmup_steps: Number of leading timesteps to discard.

    Returns:
        Scalar log-likelihood.
    """
    sim_t = sim[warmup_steps:]
    obs_t = obs[warmup_steps:]
    n = obs_t.shape[0]
    sse = jnp.sum((obs_t - sim_t) ** 2)
    return -n / 2.0 * jnp.log(sse + _EPS)


# -----------------------------------------------------------------------
# Gaussian log-likelihood (explicit sigma -- for NUTS sampling)
# -----------------------------------------------------------------------

def gaussian_log_likelihood_jax(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    sigma: float,
    warmup_steps: int = 365,
) -> jnp.ndarray:
    """
    Gaussian log-likelihood with explicit sigma (sampled by NUTS).

    Args:
        sim: Simulated flow array.
        obs: Observed flow array.
        sigma: Standard deviation of residuals (sampled as MCMC parameter).
        warmup_steps: Number of leading timesteps to discard.

    Returns:
        Scalar log-likelihood.
    """
    sim_t = sim[warmup_steps:]
    obs_t = obs[warmup_steps:]
    n = obs_t.shape[0]
    residuals = obs_t - sim_t
    return (
        -0.5 * n * jnp.log(2.0 * jnp.pi)
        - n * jnp.log(sigma + _EPS)
        - 0.5 * jnp.sum(residuals ** 2) / (sigma ** 2 + _EPS)
    )


# -----------------------------------------------------------------------
# Transformed Gaussian log-likelihood (explicit sigma)
# -----------------------------------------------------------------------

def transformed_gaussian_log_likelihood_jax(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    sigma: float,
    transform: str = "sqrt",
    transform_params: dict = None,
    warmup_steps: int = 365,
) -> jnp.ndarray:
    """
    Gaussian log-likelihood in transformed flow space.

    Matches ``TransformedGaussianLikelihood.calculate()`` but with
    explicit sigma for NUTS.

    Args:
        sim: Simulated flow array.
        obs: Observed flow array.
        sigma: Residual standard deviation in transformed space.
        transform: Flow transformation name.
        transform_params: Optional transform parameters.
        warmup_steps: Leading timesteps to discard.

    Returns:
        Scalar log-likelihood.
    """
    sim_t = sim[warmup_steps:]
    obs_t = obs[warmup_steps:]
    obs_trans = apply_transform_jax(obs_t, obs_t, transform, transform_params)
    sim_trans = apply_transform_jax(sim_t, obs_t, transform, transform_params)
    n = obs_t.shape[0]
    residuals = obs_trans - sim_trans
    return (
        -0.5 * n * jnp.log(2.0 * jnp.pi)
        - n * jnp.log(sigma + _EPS)
        - 0.5 * jnp.sum(residuals ** 2) / (sigma ** 2 + _EPS)
    )


# -----------------------------------------------------------------------
# Transformed Gaussian log-likelihood (sigma integrated out)
# -----------------------------------------------------------------------

def transformed_gaussian_log_likelihood_integrated_jax(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    transform: str = "sqrt",
    transform_params: dict = None,
    warmup_steps: int = 365,
) -> jnp.ndarray:
    """
    Transformed Gaussian log-likelihood with sigma integrated out.

    Matches ``TransformedGaussianLikelihood.calculate()``.
    """
    sim_t = sim[warmup_steps:]
    obs_t = obs[warmup_steps:]
    obs_trans = apply_transform_jax(obs_t, obs_t, transform, transform_params)
    sim_trans = apply_transform_jax(sim_t, obs_t, transform, transform_params)
    n = obs_t.shape[0]
    sse = jnp.sum((obs_trans - sim_trans) ** 2)
    return -n / 2.0 * jnp.log(sse + _EPS)


# -----------------------------------------------------------------------
# AR(1) log-likelihood (new -- autocorrelated residuals)
# -----------------------------------------------------------------------

def ar1_log_likelihood_jax(
    sim: jnp.ndarray,
    obs: jnp.ndarray,
    sigma: float,
    phi: float,
    transform: str = "none",
    transform_params: dict = None,
    warmup_steps: int = 365,
) -> jnp.ndarray:
    """
    Gaussian log-likelihood with AR(1) error model.

    Accounts for temporal autocorrelation in residuals, common in
    hydrological time series.

    The AR(1) model:  e_t = phi * e_{t-1} + w_t,  w_t ~ N(0, sigma^2)

    Args:
        sim: Simulated flow array.
        obs: Observed flow array.
        sigma: Innovation standard deviation.
        phi: AR(1) autocorrelation coefficient (|phi| < 1).
        transform: Flow transformation name.
        transform_params: Optional transform parameters.
        warmup_steps: Leading timesteps to discard.

    Returns:
        Scalar log-likelihood.
    """
    sim_t = sim[warmup_steps:]
    obs_t = obs[warmup_steps:]

    if transform != "none":
        obs_t = apply_transform_jax(obs_t, obs_t, transform, transform_params)
        sim_t = apply_transform_jax(sim_t, obs_t, transform, transform_params)

    residuals = obs_t - sim_t
    n = residuals.shape[0]

    innovations = residuals[1:] - phi * residuals[:-1]
    n_inn = innovations.shape[0]

    ll_first = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma + _EPS) - 0.5 * (residuals[0] / (sigma + _EPS)) ** 2
    ll_rest = (
        -0.5 * n_inn * jnp.log(2.0 * jnp.pi)
        - n_inn * jnp.log(sigma + _EPS)
        - 0.5 * jnp.sum(innovations ** 2) / (sigma ** 2 + _EPS)
    )
    return ll_first + ll_rest
