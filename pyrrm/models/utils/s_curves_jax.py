"""
JAX-compatible S-curve functions for unit hydrograph generation.

Port of s_curves.py for use with JAX automatic differentiation
and jax.lax.scan-based model implementations. All functions are
pure and JIT-compatible.
"""

import jax.numpy as jnp


def s_curve1_jax(t: jnp.ndarray, x4: float, exp: float = 2.5) -> jnp.ndarray:
    """
    S-curve for UH1 (routed flow component).

    JAX equivalent of s_curves.s_curve1, vectorised over t.

    Args:
        t: Time indices (array or scalar).
        x4: Unit hydrograph time constant [days].
        exp: S-curve exponent (default 2.5).

    Returns:
        S-curve values at each t.
    """
    return jnp.where(t <= 0, 0.0, jnp.where(t < x4, (t / x4) ** exp, 1.0))


def s_curve2_jax(t: jnp.ndarray, x4: float, exp: float = 2.5) -> jnp.ndarray:
    """
    S-curve for UH2 (direct flow component).

    JAX equivalent of s_curves.s_curve2, vectorised over t.

    Args:
        t: Time indices (array or scalar).
        x4: Unit hydrograph time constant [days].
        exp: S-curve exponent (default 2.5).

    Returns:
        S-curve values at each t.
    """
    return jnp.where(
        t <= 0,
        0.0,
        jnp.where(
            t < x4,
            0.5 * (t / x4) ** exp,
            jnp.where(
                t < 2.0 * x4,
                1.0 - 0.5 * jnp.maximum(2.0 - t / x4, 0.0) ** exp,
                1.0,
            ),
        ),
    )


def compute_uh1_ordinates_jax(
    x4: float, max_length: int = 20, exp: float = 2.5
) -> jnp.ndarray:
    """
    Compute fixed-size UH1 ordinates for JIT compatibility.

    Unlike the NumPy version which returns a variably-sized array,
    this always returns an array of length *max_length* padded with
    zeros beyond the active ordinates.

    Args:
        x4: Unit hydrograph time constant [days].
        max_length: Fixed output length (must cover ceil(x4)).
        exp: S-curve exponent (default 2.5).

    Returns:
        Array of shape (max_length,) with UH1 ordinates.
    """
    t = jnp.arange(1, max_length + 1).astype(jnp.result_type(1.0))
    sh1 = s_curve1_jax(t, x4, exp)
    sh1_prev = jnp.concatenate([jnp.array([0.0]), sh1[:-1]])
    return sh1 - sh1_prev


def compute_uh2_ordinates_jax(
    x4: float, max_length: int = 40, exp: float = 2.5
) -> jnp.ndarray:
    """
    Compute fixed-size UH2 ordinates for JIT compatibility.

    Args:
        x4: Unit hydrograph time constant [days].
        max_length: Fixed output length (must cover ceil(2*x4)).
        exp: S-curve exponent (default 2.5).

    Returns:
        Array of shape (max_length,) with UH2 ordinates.
    """
    t = jnp.arange(1, max_length + 1).astype(jnp.result_type(1.0))
    sh2 = s_curve2_jax(t, x4, exp)
    sh2_prev = jnp.concatenate([jnp.array([0.0]), sh2[:-1]])
    return sh2 - sh2_prev
