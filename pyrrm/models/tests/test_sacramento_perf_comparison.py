"""
Comprehensive Sacramento performance comparison across backends.

Benchmarks:
  1. Single-run: Python, Numba, Numba-fastmath, JAX, MLX
  2. Batch calibration: Sequential Numba vs Parallel Numba (prange)

Also verifies numerical equivalence between all backends.

Run with:
    pytest -m slow -s pyrrm/models/tests/test_sacramento_perf_comparison.py
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pyrrm.models.numba_kernels import (
    NUMBA_AVAILABLE,
    _sacramento_run_numba,
    _sacramento_run_numba_fast,
    _sacramento_batch_numba,
)
from pyrrm.models.sacramento import Sacramento

try:
    from pyrrm.models.sacramento_mlx import sacramento_run_mlx, MLX_AVAILABLE
except ImportError:
    MLX_AVAILABLE = False

try:
    from pyrrm.models.sacramento_jax import sacramento_run_jax
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed"),
]


# =========================================================================
# Test data helpers
# =========================================================================

SAC_PARAMS = {
    "uztwm": 50.0, "uzfwm": 40.0, "lztwm": 130.0,
    "lzfpm": 60.0, "lzfsm": 25.0, "uzk": 0.3,
    "lzpk": 0.01, "lzsk": 0.05, "zperc": 40.0,
    "rexp": 1.5, "pctim": 0.01, "adimp": 0.0,
    "pfree": 0.06, "rserv": 0.3, "side": 0.0,
    "ssout": 0.0, "sarva": 0.0,
    "uh1": 1.0, "uh2": 0.0, "uh3": 0.0, "uh4": 0.0, "uh5": 0.0,
}

DATA_410734_DIR = Path(__file__).resolve().parents[3] / "data" / "410734"


def _make_inputs(n_years):
    rng = np.random.RandomState(42)
    n = n_years * 365
    precip = rng.exponential(scale=5.0, size=n).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)).astype(np.float64)
    return precip, pet


def _time_fn(fn, repeats=5):
    """Time a function, returning median elapsed seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _setup_model():
    model = Sacramento()
    model.set_parameters(SAC_PARAMS)
    model.reset()
    return model


def _get_kernel_args(model):
    uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
    uh_stores = np.array(model._unit_hydrograph._stores, dtype=np.float64)
    return (
        model.uztwm, model.uzfwm, model.lztwm, model.lzfpm, model.lzfsm,
        model.uzk, model.lzpk, model.lzsk,
        model.zperc, model.rexp, model.pctim, model.adimp,
        model.pfree, model.rserv, model.side, model.ssout, model.sarva,
        uh_scurve, 0., 0., 0., 0., 0., 0., 0., uh_stores.copy(),
    )


def _get_jax_params():
    return {k: float(v) for k, v in SAC_PARAMS.items()}


def _sac_python(precip, pet):
    model = Sacramento()
    model.set_parameters(SAC_PARAMS)
    model.reset()
    n = len(precip)
    for t in range(n):
        model.rainfall = float(precip[t])
        model.pet = float(pet[t])
        model._run_time_step()


# =========================================================================
# Single-run benchmarks
# =========================================================================

class TestSingleRunBenchmarks:

    @pytest.mark.parametrize("n_years", [1, 10, 25])
    def test_all_backends(self, n_years):
        """Compare all available backends for a single Sacramento run."""
        precip, pet = _make_inputs(n_years)
        model = _setup_model()
        args = _get_kernel_args(model)

        # Warm up Numba
        _sacramento_run_numba(precip[:365], pet[:365], *args)
        _sacramento_run_numba_fast(precip[:365], pet[:365], *args)

        results = {}

        # Python
        t_py = _time_fn(lambda: _sac_python(precip, pet))
        results["Python"] = t_py

        # Numba standard
        t_nb = _time_fn(lambda: _sacramento_run_numba(precip, pet, *args))
        results["Numba"] = t_nb

        # Numba fastmath
        t_fm = _time_fn(lambda: _sacramento_run_numba_fast(precip, pet, *args))
        results["Numba-fast"] = t_fm

        # JAX (if available)
        if JAX_AVAILABLE:
            jax_params = _get_jax_params()
            import jax.numpy as jnp
            p_jax = jnp.array(precip)
            e_jax = jnp.array(pet)
            sacramento_run_jax(jax_params, p_jax, e_jax)  # warm up
            t_jax = _time_fn(lambda: sacramento_run_jax(jax_params, p_jax, e_jax))
            results["JAX"] = t_jax

        # MLX (if available)
        if MLX_AVAILABLE:
            sacramento_run_mlx(precip[:365], pet[:365], *args)  # warm up
            t_mlx = _time_fn(lambda: sacramento_run_mlx(precip, pet, *args))
            results["MLX"] = t_mlx

        # Print results
        print(f"\n  Sacramento {n_years:2d}yr single-run benchmarks:")
        print(f"  {'Backend':<14} {'Time (ms)':>10} {'vs Python':>10} {'vs Numba':>10}")
        print(f"  {'-'*46}")
        for name, t in results.items():
            vs_py = t_py / t if t > 0 else float("inf")
            vs_nb = t_nb / t if t > 0 else float("inf")
            print(f"  {name:<14} {t*1000:10.2f} {vs_py:9.1f}x {vs_nb:9.1f}x")

        assert t_nb < t_py, "Numba should be faster than Python"


# =========================================================================
# Fastmath equivalence within relaxed tolerance
# =========================================================================

class TestFastmathEquivalence:

    @pytest.mark.parametrize("n_years", [1, 10])
    def test_fastmath_matches_standard(self, n_years):
        """Fastmath results should be close to standard Numba."""
        precip, pet = _make_inputs(n_years)
        model = _setup_model()
        args = _get_kernel_args(model)

        result_std = _sacramento_run_numba(precip, pet, *args)
        result_fm = _sacramento_run_numba_fast(precip, pet, *args)

        np.testing.assert_allclose(
            result_fm[0], result_std[0],
            rtol=1e-6, atol=1e-8,
            err_msg="Fastmath runoff diverges from standard Numba",
        )
        np.testing.assert_allclose(
            result_fm[1], result_std[1],
            rtol=1e-6, atol=1e-8,
            err_msg="Fastmath baseflow diverges from standard Numba",
        )
        np.testing.assert_allclose(
            result_fm[2], result_std[2],
            rtol=1e-6, atol=1e-8,
            err_msg="Fastmath channel_flow diverges from standard Numba",
        )

    def test_fastmath_nonneg_and_finite(self):
        precip, pet = _make_inputs(10)
        model = _setup_model()
        args = _get_kernel_args(model)
        result = _sacramento_run_numba_fast(precip, pet, *args)
        assert np.all(np.isfinite(result[0])), "NaN in fastmath runoff"
        assert np.all(result[0] >= 0), "Negative fastmath runoff"


# =========================================================================
# MLX equivalence
# =========================================================================

class TestMLXEquivalence:

    pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")

    @pytest.mark.parametrize("n_years", [1])
    def test_mlx_matches_numba(self, n_years):
        """MLX should produce identical results to standard Numba."""
        precip, pet = _make_inputs(n_years)
        model = _setup_model()
        args = _get_kernel_args(model)

        result_nb = _sacramento_run_numba(precip, pet, *args)
        result_mlx = sacramento_run_mlx(precip, pet, *args)

        np.testing.assert_allclose(
            result_mlx[0], result_nb[0],
            rtol=1e-10, atol=1e-12,
            err_msg="MLX runoff diverges from Numba",
        )
        np.testing.assert_allclose(
            result_mlx[1], result_nb[1],
            rtol=1e-10, atol=1e-12,
            err_msg="MLX baseflow diverges from Numba",
        )


# =========================================================================
# Batch calibration benchmarks
# =========================================================================

class TestBatchCalibrationBenchmarks:

    @pytest.mark.parametrize("n_evals", [100, 500, 1000])
    def test_sequential_vs_parallel(self, n_evals):
        """Compare sequential vs parallel batch Sacramento evaluation."""
        precip, pet = _make_inputs(10)
        model = _setup_model()

        rng = np.random.RandomState(99)
        uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
        n_uh = len(uh_scurve)

        param_matrix = np.zeros((n_evals, 24), dtype=np.float64)
        uh_scurve_matrix = np.tile(uh_scurve, (n_evals, 1))
        uh_stores_matrix = np.zeros((n_evals, n_uh), dtype=np.float64)

        for i in range(n_evals):
            param_matrix[i, 0] = rng.uniform(25, 125)     # uztwm
            param_matrix[i, 1] = rng.uniform(20, 80)      # uzfwm
            param_matrix[i, 2] = rng.uniform(75, 300)      # lztwm
            param_matrix[i, 3] = rng.uniform(30, 120)      # lzfpm
            param_matrix[i, 4] = rng.uniform(10, 60)       # lzfsm
            param_matrix[i, 5] = rng.uniform(0.2, 0.5)     # uzk
            param_matrix[i, 6] = rng.uniform(0.005, 0.02)  # lzpk
            param_matrix[i, 7] = rng.uniform(0.02, 0.15)   # lzsk
            param_matrix[i, 8] = rng.uniform(20, 200)      # zperc
            param_matrix[i, 9] = rng.uniform(1.0, 4.0)     # rexp
            param_matrix[i, 10] = rng.uniform(0.005, 0.05) # pctim
            param_matrix[i, 11] = 0.0                       # adimp
            param_matrix[i, 12] = rng.uniform(0.02, 0.3)   # pfree
            param_matrix[i, 13] = 0.3                       # rserv
            param_matrix[i, 14] = 0.0                       # side
            param_matrix[i, 15] = 0.0                       # ssout
            param_matrix[i, 16] = 0.0                       # sarva
            # init states (17-23) all zero

        # Warm up both paths
        _sacramento_run_numba(precip[:365], pet[:365],
                              *_get_kernel_args(model))
        _sacramento_batch_numba(
            precip[:365], pet[:365],
            param_matrix[:2], uh_scurve_matrix[:2], uh_stores_matrix[:2],
        )

        # Sequential: loop calling _sacramento_run_numba
        def run_sequential():
            for i in range(n_evals):
                p = param_matrix[i]
                _sacramento_run_numba(
                    precip, pet,
                    p[0], p[1], p[2], p[3], p[4],
                    p[5], p[6], p[7], p[8], p[9], p[10],
                    p[11], p[12], p[13], p[14], p[15], p[16],
                    uh_scurve_matrix[i],
                    p[17], p[18], p[19], p[20], p[21], p[22], p[23],
                    uh_stores_matrix[i].copy(),
                )

        # Parallel: single call to _sacramento_batch_numba
        def run_parallel():
            _sacramento_batch_numba(
                precip, pet,
                param_matrix, uh_scurve_matrix, uh_stores_matrix,
            )

        t_seq = _time_fn(run_sequential, repeats=3)
        t_par = _time_fn(run_parallel, repeats=3)
        speedup = t_seq / t_par if t_par > 0 else float("inf")

        print(f"\n  Batch calibration ({n_evals} evals x 10yr):")
        print(f"    Sequential: {t_seq:.2f} s ({t_seq/n_evals*1000:.2f} ms/eval)")
        print(f"    Parallel:   {t_par:.2f} s ({t_par/n_evals*1000:.2f} ms/eval)")
        print(f"    Speedup:    {speedup:.1f}x")


# =========================================================================
# Batch-parallel equivalence (must match sequential exactly)
# =========================================================================

class TestBatchEquivalence:

    def test_parallel_matches_sequential(self):
        """Parallel batch results must be bit-identical to sequential."""
        precip, pet = _make_inputs(5)
        model = _setup_model()

        rng = np.random.RandomState(42)
        uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
        n_uh = len(uh_scurve)
        n_evals = 20

        param_matrix = np.zeros((n_evals, 24), dtype=np.float64)
        uh_scurve_matrix = np.tile(uh_scurve, (n_evals, 1))
        uh_stores_matrix = np.zeros((n_evals, n_uh), dtype=np.float64)

        for i in range(n_evals):
            param_matrix[i, 0] = rng.uniform(25, 125)
            param_matrix[i, 1] = rng.uniform(20, 80)
            param_matrix[i, 2] = rng.uniform(75, 300)
            param_matrix[i, 3] = rng.uniform(30, 120)
            param_matrix[i, 4] = rng.uniform(10, 60)
            param_matrix[i, 5] = rng.uniform(0.2, 0.5)
            param_matrix[i, 6] = rng.uniform(0.005, 0.02)
            param_matrix[i, 7] = rng.uniform(0.02, 0.15)
            param_matrix[i, 8] = rng.uniform(20, 200)
            param_matrix[i, 9] = rng.uniform(1.0, 4.0)
            param_matrix[i, 10] = rng.uniform(0.005, 0.05)
            param_matrix[i, 11] = 0.0
            param_matrix[i, 12] = rng.uniform(0.02, 0.3)
            param_matrix[i, 13] = 0.3
            param_matrix[i, 14] = 0.0
            param_matrix[i, 15] = 0.0
            param_matrix[i, 16] = 0.0

        # Sequential reference
        runoff_seq = np.empty((n_evals, len(precip)))
        for i in range(n_evals):
            p = param_matrix[i]
            result = _sacramento_run_numba(
                precip, pet,
                p[0], p[1], p[2], p[3], p[4],
                p[5], p[6], p[7], p[8], p[9], p[10],
                p[11], p[12], p[13], p[14], p[15], p[16],
                uh_scurve_matrix[i],
                p[17], p[18], p[19], p[20], p[21], p[22], p[23],
                uh_stores_matrix[i].copy(),
            )
            runoff_seq[i] = result[0]

        # Parallel
        runoff_par = _sacramento_batch_numba(
            precip, pet,
            param_matrix, uh_scurve_matrix, uh_stores_matrix,
        )

        np.testing.assert_allclose(
            runoff_par, runoff_seq,
            rtol=1e-12, atol=1e-14,
            err_msg="Parallel batch results diverge from sequential",
        )


# =========================================================================
# Real-data benchmark (Gauge 410734)
# =========================================================================

SAC_410734_PARAMS = {
    "uztwm": 70.6442117562136,
    "uzfwm": 21.9039674996516,
    "lztwm": 108.247049338186,
    "lzfpm": 50.5919796985246,
    "lzfsm": 32.9531229692213,
    "uzk": 0.2,
    "lzpk": 0.00665711482219168,
    "lzsk": 0.104690625552841,
    "zperc": 167.655607344567,
    "rexp": 3.5,
    "pctim": 0.0312567350242689,
    "adimp": 0.101782859860568,
    "pfree": 0.246803255046147,
    "rserv": 0.0,
    "side": 0.491133663960067,
    "ssout": 0.005977407539568,
    "sarva": 0.00167606208695788,
    "uh1": 0.566400040334827,
    "uh2": 1.0,
    "uh3": 0.244659569272703,
    "uh4": 0.0698556515828799,
    "uh5": 0.130286870916207,
}


def _can_skip_410734():
    rain = DATA_410734_DIR / "Default Input Set - Rain_QBN01.csv"
    pet = DATA_410734_DIR / "Default Input Set - Mwet_QBN01.csv"
    return not (rain.exists() and pet.exists())


@pytest.fixture(scope="module")
def data_410734():
    rain_path = DATA_410734_DIR / "Default Input Set - Rain_QBN01.csv"
    pet_path = DATA_410734_DIR / "Default Input Set - Mwet_QBN01.csv"
    rain_df = pd.read_csv(rain_path, parse_dates=["Date"], index_col="Date")
    pet_df = pd.read_csv(pet_path, parse_dates=["Date"], index_col="Date")
    merged = rain_df.join(pet_df, how="inner")
    merged.columns = ["precipitation", "pet"]
    return merged


class TestRealDataBenchmark:

    pytestmark = pytest.mark.skipif(
        _can_skip_410734(), reason="410734 data files not found",
    )

    def test_real_data_all_backends(self, data_410734):
        """Benchmark all backends on ~135 years of real data."""
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)
        n_days = len(precip)

        model = Sacramento()
        model.set_parameters(SAC_410734_PARAMS)
        model.reset()
        args = _get_kernel_args(model)

        # Warm up
        _sacramento_run_numba(precip[:365], pet[:365], *args)
        _sacramento_run_numba_fast(precip[:365], pet[:365], *args)

        results = {}

        t_nb = _time_fn(lambda: _sacramento_run_numba(precip, pet, *args))
        results["Numba"] = t_nb

        t_fm = _time_fn(lambda: _sacramento_run_numba_fast(precip, pet, *args))
        results["Numba-fast"] = t_fm

        if JAX_AVAILABLE:
            import jax.numpy as jnp
            jax_params = {k: float(v) for k, v in SAC_410734_PARAMS.items()}
            p_jax = jnp.array(precip)
            e_jax = jnp.array(pet)
            sacramento_run_jax(jax_params, p_jax, e_jax)
            t_jax = _time_fn(lambda: sacramento_run_jax(jax_params, p_jax, e_jax))
            results["JAX"] = t_jax

        if MLX_AVAILABLE:
            sacramento_run_mlx(precip[:365], pet[:365], *args)
            t_mlx = _time_fn(lambda: sacramento_run_mlx(precip, pet, *args), repeats=1)
            results["MLX"] = t_mlx

        print(f"\n  Real data: Gauge 410734 ({n_days} days, {n_days/365.25:.0f} years)")
        print(f"  {'Backend':<14} {'Time (ms)':>10} {'vs Numba':>10}")
        print(f"  {'-'*36}")
        for name, t in results.items():
            vs_nb = t_nb / t if t > 0 else float("inf")
            print(f"  {name:<14} {t*1000:10.2f} {vs_nb:9.2f}x")

    def test_fastmath_real_data_equivalence(self, data_410734):
        """Fastmath equivalence on real data with calibrated parameters."""
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)

        model = Sacramento()
        model.set_parameters(SAC_410734_PARAMS)
        model.reset()
        args = _get_kernel_args(model)

        result_std = _sacramento_run_numba(precip, pet, *args)
        result_fm = _sacramento_run_numba_fast(precip, pet, *args)

        np.testing.assert_allclose(
            result_fm[0], result_std[0],
            rtol=1e-6, atol=1e-8,
            err_msg="Fastmath runoff on 410734 diverges from standard Numba",
        )

        cum_std = np.cumsum(result_std[0])
        cum_fm = np.cumsum(result_fm[0])
        np.testing.assert_allclose(
            cum_fm, cum_std, rtol=1e-5,
            err_msg="Fastmath cumulative runoff drift on 410734",
        )


# =========================================================================
# Summary table
# =========================================================================

class TestSummaryTable:

    def test_print_comprehensive_summary(self):
        """Print a consolidated performance summary across all backends."""
        model = _setup_model()
        args = _get_kernel_args(model)

        # Warm up
        p1, e1 = _make_inputs(1)
        _sacramento_run_numba(p1, e1, *args)
        _sacramento_run_numba_fast(p1, e1, *args)

        rows = []
        for label, n_years in [("1yr", 1), ("10yr", 10), ("25yr", 25)]:
            precip, pet = _make_inputs(n_years)

            t_py = _time_fn(lambda: _sac_python(precip, pet))
            t_nb = _time_fn(lambda: _sacramento_run_numba(precip, pet, *args))
            t_fm = _time_fn(lambda: _sacramento_run_numba_fast(precip, pet, *args))

            rows.append(("Python", label, t_py))
            rows.append(("Numba", label, t_nb))
            rows.append(("Numba-fast", label, t_fm))

            if JAX_AVAILABLE:
                import jax.numpy as jnp
                jax_params = _get_jax_params()
                p_jax, e_jax = jnp.array(precip), jnp.array(pet)
                sacramento_run_jax(jax_params, p_jax, e_jax)
                t_jax = _time_fn(lambda: sacramento_run_jax(jax_params, p_jax, e_jax))
                rows.append(("JAX", label, t_jax))

            if MLX_AVAILABLE:
                sacramento_run_mlx(precip[:365], pet[:365], *args)
                t_mlx = _time_fn(
                    lambda: sacramento_run_mlx(precip, pet, *args),
                    repeats=3 if n_years <= 10 else 1,
                )
                rows.append(("MLX", label, t_mlx))

        print("\n" + "=" * 64)
        print(f"  {'Backend':<14} {'Period':<8} {'Time (ms)':>12} {'Speedup':>10}")
        print("-" * 64)

        py_times = {r[1]: r[2] for r in rows if r[0] == "Python"}
        for backend, period, t in rows:
            t_py = py_times.get(period, 1.0)
            speedup = t_py / t if t > 0 else float("inf")
            print(f"  {backend:<14} {period:<8} {t*1000:12.2f} {speedup:9.1f}x")
        print("=" * 64)
