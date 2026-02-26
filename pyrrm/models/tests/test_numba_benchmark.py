"""
Performance benchmarks comparing Numba-compiled kernels against
pure-Python implementations for all four rainfall-runoff models.

Marked with @pytest.mark.slow -- run with:
    pytest -m slow -s pyrrm/models/tests/test_numba_benchmark.py
"""

import time
import pytest
import numpy as np
import pandas as pd

from pyrrm.models.numba_kernels import (
    NUMBA_AVAILABLE,
    _gr4j_core_numba,
    _gr5j_core_numba,
    _gr6j_core_numba,
    _sacramento_run_numba,
)
from pyrrm.models.gr4j import _gr4j_core
from pyrrm.models.gr5j import _gr5j_core
from pyrrm.models.gr6j import _gr6j_core
from pyrrm.models.sacramento import Sacramento

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed"),
]


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


# =========================================================================
# GR4J benchmarks
# =========================================================================

class TestGR4JBenchmark:

    @pytest.mark.parametrize("n_years", [1, 10, 25])
    def test_gr4j_speedup(self, n_years):
        precip, pet = _make_inputs(n_years)
        uh1, uh2 = np.zeros(20), np.zeros(40)
        args = (350., 0.5, 90., 1.7, precip, pet, 105., 45.)

        # Warm up Numba (already cached but ensure loaded)
        _gr4j_core_numba(*args, uh1.copy(), uh2.copy())

        t_py = _time_fn(lambda: _gr4j_core(*args, uh1.copy(), uh2.copy()))
        t_nb = _time_fn(lambda: _gr4j_core_numba(*args, uh1.copy(), uh2.copy()))
        speedup = t_py / t_nb if t_nb > 0 else float("inf")

        print(f"\n  GR4J {n_years:2d}yr | Python: {t_py*1000:8.2f} ms | "
              f"Numba: {t_nb*1000:8.2f} ms | Speedup: {speedup:6.1f}x")
        assert speedup > 1.0, "Numba should be faster than Python"


# =========================================================================
# GR5J benchmarks
# =========================================================================

class TestGR5JBenchmark:

    @pytest.mark.parametrize("n_years", [1, 10, 25])
    def test_gr5j_speedup(self, n_years):
        precip, pet = _make_inputs(n_years)
        uh2 = np.zeros(40)
        args = (350., 0.5, 90., 1.7, 0.5, precip, pet, 105., 45.)

        _gr5j_core_numba(*args, uh2.copy())

        t_py = _time_fn(lambda: _gr5j_core(*args, uh2.copy()))
        t_nb = _time_fn(lambda: _gr5j_core_numba(*args, uh2.copy()))
        speedup = t_py / t_nb if t_nb > 0 else float("inf")

        print(f"\n  GR5J {n_years:2d}yr | Python: {t_py*1000:8.2f} ms | "
              f"Numba: {t_nb*1000:8.2f} ms | Speedup: {speedup:6.1f}x")
        assert speedup > 1.0


# =========================================================================
# GR6J benchmarks
# =========================================================================

class TestGR6JBenchmark:

    @pytest.mark.parametrize("n_years", [1, 10, 25])
    def test_gr6j_speedup(self, n_years):
        precip, pet = _make_inputs(n_years)
        uh1, uh2 = np.zeros(20), np.zeros(40)
        args = (350., 0.5, 90., 1.7, 0.1, 50., precip, pet, 105., 45., 0.)

        _gr6j_core_numba(*args, uh1.copy(), uh2.copy())

        t_py = _time_fn(lambda: _gr6j_core(*args, uh1.copy(), uh2.copy()))
        t_nb = _time_fn(lambda: _gr6j_core_numba(*args, uh1.copy(), uh2.copy()))
        speedup = t_py / t_nb if t_nb > 0 else float("inf")

        print(f"\n  GR6J {n_years:2d}yr | Python: {t_py*1000:8.2f} ms | "
              f"Numba: {t_nb*1000:8.2f} ms | Speedup: {speedup:6.1f}x")
        assert speedup > 1.0


# =========================================================================
# Sacramento benchmarks
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


def _sac_python(precip, pet):
    model = Sacramento()
    model.set_parameters(SAC_PARAMS)
    model.reset()
    n = len(precip)
    for t in range(n):
        model.rainfall = float(precip[t])
        model.pet = float(pet[t])
        model._run_time_step()


def _sac_numba(precip, pet, model):
    uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
    uh_stores = np.array(model._unit_hydrograph._stores, dtype=np.float64)
    _sacramento_run_numba(
        precip, pet,
        model.uztwm, model.uzfwm, model.lztwm, model.lzfpm, model.lzfsm,
        model.uzk, model.lzpk, model.lzsk,
        model.zperc, model.rexp, model.pctim, model.adimp,
        model.pfree, model.rserv, model.side, model.ssout, model.sarva,
        uh_scurve, 0., 0., 0., 0., 0., 0., 0., uh_stores.copy(),
    )


class TestSacramentoBenchmark:

    @pytest.mark.parametrize("n_years", [1, 10, 25])
    def test_sacramento_speedup(self, n_years):
        precip, pet = _make_inputs(n_years)

        model = Sacramento()
        model.set_parameters(SAC_PARAMS)
        model.reset()

        # Warm up
        _sac_numba(precip[:365], pet[:365], model)

        t_py = _time_fn(lambda: _sac_python(precip, pet))
        t_nb = _time_fn(lambda: _sac_numba(precip, pet, model))
        speedup = t_py / t_nb if t_nb > 0 else float("inf")

        print(f"\n  Sacramento {n_years:2d}yr | Python: {t_py*1000:8.2f} ms | "
              f"Numba: {t_nb*1000:8.2f} ms | Speedup: {speedup:6.1f}x")
        assert speedup > 1.0


# =========================================================================
# Simulated calibration loop (1000 evaluations)
# =========================================================================

class TestCalibrationLoopBenchmark:

    def test_gr4j_calibration_loop(self):
        """1000 model evaluations with parameter resets."""
        precip, pet = _make_inputs(10)
        uh1, uh2 = np.zeros(20), np.zeros(40)
        rng = np.random.RandomState(99)

        x1_vals = rng.uniform(100, 1200, 1000)
        x2_vals = rng.uniform(-5, 3, 1000)
        x3_vals = rng.uniform(20, 300, 1000)
        x4_vals = rng.uniform(1.1, 2.9, 1000)

        # Warm up
        _gr4j_core_numba(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                         uh1.copy(), uh2.copy())

        t0 = time.perf_counter()
        for i in range(1000):
            _gr4j_core(x1_vals[i], x2_vals[i], x3_vals[i], x4_vals[i],
                       precip, pet, 0.3 * x1_vals[i], 0.5 * x3_vals[i],
                       uh1.copy(), uh2.copy())
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(1000):
            _gr4j_core_numba(x1_vals[i], x2_vals[i], x3_vals[i], x4_vals[i],
                             precip, pet, 0.3 * x1_vals[i], 0.5 * x3_vals[i],
                             uh1.copy(), uh2.copy())
        t_nb = time.perf_counter() - t0

        speedup = t_py / t_nb if t_nb > 0 else float("inf")
        print(f"\n  GR4J calibration loop (1000 x 10yr) | "
              f"Python: {t_py:.1f} s | Numba: {t_nb:.1f} s | "
              f"Speedup: {speedup:.1f}x")
        assert speedup > 2.0

    def test_sacramento_calibration_loop(self):
        """1000 model evaluations via class interface."""
        precip, pet = _make_inputs(10)
        df = pd.DataFrame({"precipitation": precip, "pet": pet})

        # Warm up Numba kernel
        m = Sacramento()
        m.set_parameters(SAC_PARAMS)
        m.reset()
        m.run(df)

        rng = np.random.RandomState(99)
        param_list = []
        for _ in range(1000):
            p = dict(SAC_PARAMS)
            p["uztwm"] = rng.uniform(25, 125)
            p["lztwm"] = rng.uniform(75, 300)
            p["uzk"] = rng.uniform(0.2, 0.5)
            param_list.append(p)

        t0 = time.perf_counter()
        for p in param_list:
            m.set_parameters(p)
            m.reset()
            m.run(df)
        t_nb = time.perf_counter() - t0

        print(f"\n  Sacramento calibration loop (1000 x 10yr) | "
              f"Numba: {t_nb:.1f} s | "
              f"Per eval: {t_nb/1000*1000:.2f} ms")
        assert t_nb < 120, "1000 Sacramento evals should complete in <2 min"


# =========================================================================
# JIT compilation overhead
# =========================================================================

class TestCompilationOverhead:

    def test_first_call_overhead(self):
        """Measure the first-call JIT compilation time."""
        precip, pet = _make_inputs(1)

        # Force fresh compilation by calling with different type shapes
        # (in practice, the cache handles this, so we just measure import+first-call)
        t0 = time.perf_counter()
        _gr4j_core_numba(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                         np.zeros(20), np.zeros(40))
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        _gr4j_core_numba(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                         np.zeros(20), np.zeros(40))
        t_cached = time.perf_counter() - t0

        print(f"\n  GR4J first call: {t_first*1000:.1f} ms | "
              f"Cached call: {t_cached*1000:.3f} ms")


# =========================================================================
# Summary table
# =========================================================================

class TestSummaryTable:

    def test_print_summary(self):
        """Print a consolidated performance summary table."""
        precip_1, pet_1 = _make_inputs(1)
        precip_10, pet_10 = _make_inputs(10)
        precip_25, pet_25 = _make_inputs(25)

        model = Sacramento()
        model.set_parameters(SAC_PARAMS)
        model.reset()

        rows = []
        for label, n_years, p, e in [
            ("1yr", 1, precip_1, pet_1),
            ("10yr", 10, precip_10, pet_10),
            ("25yr", 25, precip_25, pet_25),
        ]:
            n = n_years * 365
            uh1, uh2 = np.zeros(20), np.zeros(40)

            # GR4J
            t_py = _time_fn(lambda: _gr4j_core(350., 0.5, 90., 1.7, p, e, 105., 45., uh1.copy(), uh2.copy()))
            t_nb = _time_fn(lambda: _gr4j_core_numba(350., 0.5, 90., 1.7, p, e, 105., 45., uh1.copy(), uh2.copy()))
            rows.append(("GR4J", label, t_py, t_nb))

            # Sacramento
            t_py_s = _time_fn(lambda: _sac_python(p, e))
            t_nb_s = _time_fn(lambda: _sac_numba(p, e, model))
            rows.append(("Sacramento", label, t_py_s, t_nb_s))

        print("\n" + "=" * 72)
        print(f"  {'Model':<12} {'Period':<8} {'Python (ms)':>12} {'Numba (ms)':>12} {'Speedup':>10}")
        print("-" * 72)
        for model_name, period, t_py, t_nb in rows:
            speedup = t_py / t_nb if t_nb > 0 else float("inf")
            print(f"  {model_name:<12} {period:<8} {t_py*1000:12.2f} {t_nb*1000:12.2f} {speedup:9.1f}x")
        print("=" * 72)
