"""
Tests verifying Numba-compiled kernels produce identical results to the
pure-Python implementations for Sacramento, GR4J, GR5J, and GR6J.

These tests call both backends directly and compare outputs at tight
numerical tolerances. They also verify physical validity (non-negative
flows, no NaNs) and that the class-level interface behaves correctly
when Numba is active.

The ``TestSyntheticPythonVsNumba*`` classes force Python and Numba
paths through the *class-level* ``model.run()`` interface using
monkey-patching, so they test the complete dispatch and state-update
logic -- not just the raw kernel functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest import mock

from pyrrm.models.numba_kernels import (
    NUMBA_AVAILABLE,
    _s_curve1,
    _s_curve2,
    _uh_step,
    _gr4j_core_numba,
    _gr5j_core_numba,
    _gr6j_core_numba,
    _sacramento_run_numba,
)
from pyrrm.models.utils.s_curves import s_curve1, s_curve2
from pyrrm.models.gr4j import _gr4j_core, GR4J
from pyrrm.models.gr5j import _gr5j_core, GR5J
from pyrrm.models.gr6j import _gr6j_core, GR6J
from pyrrm.models.sacramento import Sacramento, _UnitHydrograph
import pyrrm.models.gr4j as _gr4j_mod
import pyrrm.models.gr5j as _gr5j_mod
import pyrrm.models.gr6j as _gr6j_mod
import pyrrm.models.sacramento as _sac_mod

pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not installed"
)

TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "test_data"


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def synthetic_1yr():
    rng = np.random.RandomState(42)
    precip = rng.exponential(scale=5.0, size=365).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365.0)).astype(np.float64)
    return precip, pet


@pytest.fixture
def synthetic_10yr():
    rng = np.random.RandomState(42)
    n = 3650
    precip = rng.exponential(scale=5.0, size=n).astype(np.float64)
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365.0)).astype(np.float64)
    return precip, pet


@pytest.fixture
def zero_precip():
    precip = np.zeros(365, dtype=np.float64)
    pet = np.full(365, 3.0, dtype=np.float64)
    return precip, pet


@pytest.fixture
def storm_burst():
    """30 days of intense rainfall followed by 335 days of drought."""
    precip = np.zeros(365, dtype=np.float64)
    precip[:30] = 50.0
    pet = np.full(365, 3.0, dtype=np.float64)
    return precip, pet


@pytest.fixture
def alternating_wet_dry():
    """Alternating 30-day wet / 30-day dry cycles for 1 year."""
    precip = np.zeros(365, dtype=np.float64)
    for start in range(0, 365, 60):
        precip[start:min(start + 30, 365)] = 15.0
    pet = (2.0 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365.0)).astype(np.float64)
    return precip, pet


def _make_df(precip, pet):
    return pd.DataFrame({"precipitation": precip, "pet": pet})


def _run_model_python(model_cls, params, precip, pet, **kwargs):
    """Run a model class with Numba forcibly disabled."""
    mod_map = {
        GR4J: _gr4j_mod,
        GR5J: _gr5j_mod,
        GR6J: _gr6j_mod,
        Sacramento: _sac_mod,
    }
    target_mod = mod_map[model_cls]
    model = model_cls(**kwargs)
    if isinstance(params, dict):
        model.set_parameters(params)
    model.reset()
    with mock.patch.object(target_mod, "NUMBA_AVAILABLE", False):
        result = model.run(_make_df(precip, pet))
    state = model.get_state()
    return result, state


def _run_model_numba(model_cls, params, precip, pet, **kwargs):
    """Run a model class with Numba enabled (the default when installed)."""
    model = model_cls(**kwargs)
    if isinstance(params, dict):
        model.set_parameters(params)
    model.reset()
    result = model.run(_make_df(precip, pet))
    state = model.get_state()
    return result, state


# =========================================================================
# S-curve equivalence
# =========================================================================

class TestSCurves:

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    @pytest.mark.parametrize("x4", [1.0, 1.7, 2.5, 5.0])
    def test_s_curve1_matches(self, t, x4):
        expected = s_curve1(t, x4, 2.5)
        got = _s_curve1(t, x4, 2.5)
        assert got == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    @pytest.mark.parametrize("x4", [1.0, 1.7, 2.5, 5.0])
    def test_s_curve2_matches(self, t, x4):
        expected = s_curve2(t, x4, 2.5)
        got = _s_curve2(t, x4, 2.5)
        assert got == pytest.approx(expected, abs=1e-15)


# =========================================================================
# Unit hydrograph step equivalence
# =========================================================================

class TestUnitHydrograph:

    def test_uh_step_matches(self):
        scurve = [0.7, 0.2, 0.1]
        inputs = [10.0, 5.0, 0.0, 3.0, 8.0]

        uh_py = _UnitHydrograph()
        uh_py.initialise_hydrograph(scurve)

        nb_stores = np.zeros(3, dtype=np.float64)
        nb_scurve = np.array(scurve, dtype=np.float64)

        for inp in inputs:
            out_py = uh_py.run_time_step(inp)
            out_nb = _uh_step(inp, nb_stores, nb_scurve)
            assert out_nb == pytest.approx(out_py, abs=1e-14)


# =========================================================================
# GR4J equivalence
# =========================================================================

GR4J_PARAM_SETS = [
    {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7},
    {"X1": 100.0, "X2": -5.0, "X3": 20.0, "X4": 1.1},
    {"X1": 1200.0, "X2": 3.0, "X3": 300.0, "X4": 2.5},
    {"X1": 500.0, "X2": 0.0, "X3": 100.0, "X4": 2.0},
    {"X1": 200.0, "X2": -2.0, "X3": 50.0, "X4": 5.0},
]


class TestGR4JNumbaEquivalence:

    @pytest.mark.parametrize("params", GR4J_PARAM_SETS)
    def test_matches_python(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        x1, x2, x3, x4 = params["X1"], params["X2"], params["X3"], params["X4"]
        prod_init = 0.3 * x1
        rout_init = 0.5 * x3
        uh1 = np.zeros(20)
        uh2 = np.zeros(40)

        flow_py, ps_py, rs_py, uh1_py, uh2_py = _gr4j_core(
            x1, x2, x3, x4, precip, pet, prod_init, rout_init,
            uh1.copy(), uh2.copy(),
        )
        flow_nb, ps_nb, rs_nb, uh1_nb, uh2_nb = _gr4j_core_numba(
            x1, x2, x3, x4, precip, pet, prod_init, rout_init,
            uh1.copy(), uh2.copy(),
        )

        np.testing.assert_allclose(flow_nb, flow_py, rtol=1e-12, atol=1e-14,
                                   err_msg=f"Flow mismatch for {params}")
        assert ps_nb == pytest.approx(ps_py, rel=1e-12)
        assert rs_nb == pytest.approx(rs_py, rel=1e-12)

    def test_10yr_stability(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        uh1 = np.zeros(20)
        uh2 = np.zeros(40)
        flow_py, *_ = _gr4j_core(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                                  uh1.copy(), uh2.copy())
        flow_nb, *_ = _gr4j_core_numba(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                                        uh1.copy(), uh2.copy())
        np.testing.assert_allclose(flow_nb, flow_py, rtol=1e-12)

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        uh1 = np.zeros(20)
        uh2 = np.zeros(40)
        flow_nb, *_ = _gr4j_core_numba(350., 0.5, 90., 1.7, precip, pet, 105., 45.,
                                        uh1, uh2)
        assert np.all(np.isfinite(flow_nb))

    def test_nonnegative_flows(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        for params in GR4J_PARAM_SETS:
            x1, x2, x3, x4 = params["X1"], params["X2"], params["X3"], params["X4"]
            flow, *_ = _gr4j_core_numba(x1, x2, x3, x4, precip, pet,
                                        0.3 * x1, 0.5 * x3,
                                        np.zeros(20), np.zeros(40))
            assert np.all(flow >= 0), f"Negative flow for {params}"


# =========================================================================
# GR5J equivalence
# =========================================================================

GR5J_PARAM_SETS = [
    {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.5},
    {"X1": 100.0, "X2": -5.0, "X3": 20.0, "X4": 1.1, "X5": 0.0},
    {"X1": 1200.0, "X2": 3.0, "X3": 300.0, "X4": 2.5, "X5": 1.0},
]


class TestGR5JNumbaEquivalence:

    @pytest.mark.parametrize("params", GR5J_PARAM_SETS)
    def test_matches_python(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        x1, x2, x3, x4, x5 = (params["X1"], params["X2"], params["X3"],
                                params["X4"], params["X5"])
        uh2 = np.zeros(40)

        flow_py, ps_py, rs_py, uh2_py = _gr5j_core(
            x1, x2, x3, x4, x5, precip, pet, 0.3 * x1, 0.5 * x3, uh2.copy(),
        )
        flow_nb, ps_nb, rs_nb, uh2_nb = _gr5j_core_numba(
            x1, x2, x3, x4, x5, precip, pet, 0.3 * x1, 0.5 * x3, uh2.copy(),
        )

        np.testing.assert_allclose(flow_nb, flow_py, rtol=1e-12, atol=1e-14,
                                   err_msg=f"Flow mismatch for {params}")
        assert ps_nb == pytest.approx(ps_py, rel=1e-12)
        assert rs_nb == pytest.approx(rs_py, rel=1e-12)


# =========================================================================
# GR6J equivalence
# =========================================================================

GR6J_PARAM_SETS = [
    {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.1, "X6": 50.0},
    {"X1": 100.0, "X2": -5.0, "X3": 20.0, "X4": 1.1, "X5": -2.0, "X6": 1.0},
    {"X1": 1200.0, "X2": 3.0, "X3": 300.0, "X4": 2.5, "X5": 2.0, "X6": 300.0},
]


class TestGR6JNumbaEquivalence:

    @pytest.mark.parametrize("params", GR6J_PARAM_SETS)
    def test_matches_python(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        x1, x2, x3, x4, x5, x6 = (params["X1"], params["X2"], params["X3"],
                                    params["X4"], params["X5"], params["X6"])
        uh1 = np.zeros(20)
        uh2 = np.zeros(40)

        flow_py, ps_py, rs_py, es_py, uh1_py, uh2_py = _gr6j_core(
            x1, x2, x3, x4, x5, x6, precip, pet,
            0.3 * x1, 0.5 * x3, 0.0, uh1.copy(), uh2.copy(),
        )
        flow_nb, ps_nb, rs_nb, es_nb, uh1_nb, uh2_nb = _gr6j_core_numba(
            x1, x2, x3, x4, x5, x6, precip, pet,
            0.3 * x1, 0.5 * x3, 0.0, uh1.copy(), uh2.copy(),
        )

        np.testing.assert_allclose(flow_nb, flow_py, rtol=1e-12, atol=1e-14,
                                   err_msg=f"Flow mismatch for {params}")
        assert ps_nb == pytest.approx(ps_py, rel=1e-12)
        assert rs_nb == pytest.approx(rs_py, rel=1e-12)
        assert es_nb == pytest.approx(es_py, rel=1e-12)


# =========================================================================
# Sacramento equivalence
# =========================================================================

SAC_DEFAULT_PARAMS = {
    "uztwm": 50.0, "uzfwm": 40.0, "lztwm": 130.0,
    "lzfpm": 60.0, "lzfsm": 25.0, "uzk": 0.3,
    "lzpk": 0.01, "lzsk": 0.05, "zperc": 40.0,
    "rexp": 1.5, "pctim": 0.01, "adimp": 0.0,
    "pfree": 0.06, "rserv": 0.3, "side": 0.0,
    "ssout": 0.0, "sarva": 0.0,
    "uh1": 1.0, "uh2": 0.0, "uh3": 0.0, "uh4": 0.0, "uh5": 0.0,
}

SAC_PARAM_SETS = [
    SAC_DEFAULT_PARAMS,
    {**SAC_DEFAULT_PARAMS, "uztwm": 100.0, "lztwm": 200.0, "rexp": 2.5,
     "adimp": 0.1, "side": 0.3},
    {**SAC_DEFAULT_PARAMS, "uzk": 0.5, "lzpk": 0.015, "lzsk": 0.2,
     "pctim": 0.05, "pfree": 0.5},
]


def _run_sacramento_python(params, precip, pet):
    """Run Sacramento purely via Python (no Numba)."""
    model = Sacramento()
    model.set_parameters(params)
    model.reset()

    n = len(precip)
    runoff = np.zeros(n)
    baseflow = np.zeros(n)
    channel_flow = np.zeros(n)
    for t in range(n):
        model.rainfall = float(precip[t])
        model.pet = float(pet[t])
        model._run_time_step()
        runoff[t] = model.runoff
        baseflow[t] = model.baseflow
        channel_flow[t] = model.channel_flow
    return runoff, baseflow, channel_flow, model


def _run_sacramento_numba(params, precip, pet):
    """Run Sacramento via the Numba kernel directly."""
    model = Sacramento()
    model.set_parameters(params)
    model.reset()

    uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
    uh_stores = np.array(model._unit_hydrograph._stores, dtype=np.float64)

    result = _sacramento_run_numba(
        precip.astype(np.float64), pet.astype(np.float64),
        model.uztwm, model.uzfwm, model.lztwm, model.lzfpm, model.lzfsm,
        model.uzk, model.lzpk, model.lzsk,
        model.zperc, model.rexp, model.pctim, model.adimp,
        model.pfree, model.rserv, model.side, model.ssout, model.sarva,
        uh_scurve,
        model.uztwc, model.uzfwc, model.lztwc, model.lzfsc, model.lzfpc,
        model.adimc, model.hydrograph_store,
        uh_stores,
    )
    return result[0], result[1], result[2]


class TestSacramentoNumbaEquivalence:

    @pytest.mark.parametrize("params", SAC_PARAM_SETS)
    def test_matches_python(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        ro_py, bf_py, cf_py, _ = _run_sacramento_python(params, precip, pet)
        ro_nb, bf_nb, cf_nb = _run_sacramento_numba(params, precip, pet)

        np.testing.assert_allclose(ro_nb, ro_py, rtol=1e-10, atol=1e-12,
                                   err_msg=f"Runoff mismatch")
        np.testing.assert_allclose(bf_nb, bf_py, rtol=1e-10, atol=1e-12,
                                   err_msg=f"Baseflow mismatch")
        np.testing.assert_allclose(cf_nb, cf_py, rtol=1e-10, atol=1e-12,
                                   err_msg=f"Channel flow mismatch")

    def test_10yr_stability(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        ro_py, bf_py, _, _ = _run_sacramento_python(SAC_DEFAULT_PARAMS, precip, pet)
        ro_nb, bf_nb, _ = _run_sacramento_numba(SAC_DEFAULT_PARAMS, precip, pet)
        np.testing.assert_allclose(ro_nb, ro_py, rtol=1e-10, atol=1e-12)

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        ro_nb, _, _ = _run_sacramento_numba(SAC_DEFAULT_PARAMS, precip, pet)
        assert np.all(np.isfinite(ro_nb))
        assert np.all(ro_nb >= 0)

    def test_full_stores_init(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        model = Sacramento()
        model.set_parameters(SAC_DEFAULT_PARAMS)
        model.reset()
        model.init_stores_full()

        uh_scurve = np.array(model._unit_hydrograph.s_curve, dtype=np.float64)
        uh_stores = np.array(model._unit_hydrograph._stores, dtype=np.float64)

        result = _sacramento_run_numba(
            precip, pet,
            model.uztwm, model.uzfwm, model.lztwm, model.lzfpm, model.lzfsm,
            model.uzk, model.lzpk, model.lzsk,
            model.zperc, model.rexp, model.pctim, model.adimp,
            model.pfree, model.rserv, model.side, model.ssout, model.sarva,
            uh_scurve,
            model.uztwc, model.uzfwc, model.lztwc, model.lzfsc, model.lzfpc,
            model.adimc, model.hydrograph_store,
            uh_stores,
        )
        ro_nb = result[0]
        assert np.all(np.isfinite(ro_nb))
        assert np.all(ro_nb >= 0)

    def test_csharp_reference(self, synthetic_1yr):
        """Verify Numba Sacramento against the C# reference output."""
        csv_path = TEST_DATA_DIR / "csharp_output_TC01_default.csv"
        if not csv_path.exists():
            pytest.skip("C# reference data not available")

        ref = pd.read_csv(csv_path)
        if "Rainfall" not in ref.columns or "Evaporation" not in ref.columns:
            pytest.skip("C# reference data missing expected columns")

        precip = ref["Rainfall"].values.astype(np.float64)
        pet = ref["Evaporation"].values.astype(np.float64)

        model = Sacramento()
        model.reset()
        inputs_df = pd.DataFrame({"precipitation": precip, "pet": pet})
        result = model.run(inputs_df)

        if "Runoff" in ref.columns:
            ref_runoff = ref["Runoff"].values.astype(np.float64)
            np.testing.assert_allclose(
                result["runoff"].values, ref_runoff,
                rtol=1e-4, atol=1e-4,
                err_msg="Numba Sacramento vs C# reference mismatch",
            )

    def test_nonnegative_flows(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        for params in SAC_PARAM_SETS:
            ro, bf, cf = _run_sacramento_numba(params, precip, pet)
            assert np.all(ro >= 0), "Negative runoff"
            assert np.all(bf >= 0), "Negative baseflow"
            assert np.all(cf >= 0), "Negative channel flow"

    def test_no_nans(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        for params in SAC_PARAM_SETS:
            ro, bf, cf = _run_sacramento_numba(params, precip, pet)
            assert np.all(np.isfinite(ro)), "NaN in runoff"
            assert np.all(np.isfinite(bf)), "NaN in baseflow"
            assert np.all(np.isfinite(cf)), "NaN in channel flow"


# =========================================================================
# Class-level interface tests (Numba active)
# =========================================================================

class TestClassInterface:
    """Verify that the model class methods work correctly with Numba dispatch."""

    def test_gr4j_run(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = GR4J({"X1": 350, "X2": 0.5, "X3": 90, "X4": 1.7})
        result = model.run(df)
        assert "flow" in result.columns
        assert len(result) == 365
        assert np.all(result["flow"].values >= 0)

    def test_gr4j_run_timestep(self):
        model = GR4J({"X1": 350, "X2": 0.5, "X3": 90, "X4": 1.7})
        out = model.run_timestep(10.0, 3.0)
        assert "flow" in out
        assert out["flow"] >= 0

    def test_gr4j_reset_and_rerun(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = GR4J({"X1": 350, "X2": 0.5, "X3": 90, "X4": 1.7})
        r1 = model.run(df)
        model.reset()
        r2 = model.run(df)
        np.testing.assert_array_equal(r1["flow"].values, r2["flow"].values)

    def test_gr5j_run(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = GR5J({"X1": 350, "X2": 0.5, "X3": 90, "X4": 1.7, "X5": 0.5})
        result = model.run(df)
        assert "flow" in result.columns
        assert np.all(result["flow"].values >= 0)

    def test_gr6j_run(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = GR6J({"X1": 350, "X2": 0.5, "X3": 90, "X4": 1.7, "X5": 0.1, "X6": 50})
        result = model.run(df)
        assert "flow" in result.columns
        assert np.all(result["flow"].values >= 0)

    def test_sacramento_run(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = Sacramento()
        model.set_parameters(SAC_DEFAULT_PARAMS)
        model.reset()
        result = model.run(df)
        assert "runoff" in result.columns
        assert "baseflow" in result.columns
        assert np.all(result["runoff"].values >= 0)

    def test_sacramento_reset_and_rerun(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = Sacramento()
        model.set_parameters(SAC_DEFAULT_PARAMS)
        model.reset()
        r1 = model.run(df)
        model.reset()
        r2 = model.run(df)
        np.testing.assert_allclose(
            r1["runoff"].values, r2["runoff"].values, rtol=1e-12,
        )

    def test_sacramento_get_set_state(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})
        model = Sacramento()
        model.set_parameters(SAC_DEFAULT_PARAMS)
        model.reset()
        model.run(df)
        state = model.get_state()
        assert "uztwc" in state.values
        assert "lzfpc" in state.values

    def test_sacramento_catchment_area_scaling(self, synthetic_1yr):
        precip, pet = synthetic_1yr
        df = pd.DataFrame({"precipitation": precip, "pet": pet})

        model_mm = Sacramento()
        model_mm.set_parameters(SAC_DEFAULT_PARAMS)
        model_mm.reset()
        r_mm = model_mm.run(df)

        model_ml = Sacramento(catchment_area_km2=100.0)
        model_ml.set_parameters(SAC_DEFAULT_PARAMS)
        model_ml.reset()
        r_ml = model_ml.run(df)

        np.testing.assert_allclose(
            r_ml["runoff"].values, r_mm["runoff"].values * 100.0, rtol=1e-12,
        )


# =========================================================================
# Synthetic class-level Python vs Numba (monkey-patched dispatch)
# =========================================================================

class TestSyntheticPythonVsNumbaGR4J:
    """Force Python and Numba paths via model.run() and compare everything."""

    @pytest.mark.parametrize("params", GR4J_PARAM_SETS)
    def test_flow_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        res_py, st_py = _run_model_python(GR4J, params, precip, pet)
        res_nb, st_nb = _run_model_numba(GR4J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"GR4J class-level flow mismatch for {params}",
        )

    @pytest.mark.parametrize("params", GR4J_PARAM_SETS)
    def test_final_state_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        _, st_py = _run_model_python(GR4J, params, precip, pet)
        _, st_nb = _run_model_numba(GR4J, params, precip, pet)
        for key in ("production_store", "routing_store"):
            assert st_nb.values[key] == pytest.approx(
                st_py.values[key], rel=1e-12
            ), f"GR4J state '{key}' mismatch"
        np.testing.assert_allclose(
            np.array(st_nb.values["uh1"]),
            np.array(st_py.values["uh1"]),
            rtol=1e-12, atol=1e-14,
        )
        np.testing.assert_allclose(
            np.array(st_nb.values["uh2"]),
            np.array(st_py.values["uh2"]),
            rtol=1e-12, atol=1e-14,
        )

    def test_10yr_flow_identical(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
        res_py, _ = _run_model_python(GR4J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR4J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values, rtol=1e-12,
        )

    def test_storm_burst(self, storm_burst):
        precip, pet = storm_burst
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
        res_py, _ = _run_model_python(GR4J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR4J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )

    def test_alternating_wet_dry(self, alternating_wet_dry):
        precip, pet = alternating_wet_dry
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
        res_py, _ = _run_model_python(GR4J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR4J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7}
        res_py, _ = _run_model_python(GR4J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR4J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )


class TestSyntheticPythonVsNumbaGR5J:

    @pytest.mark.parametrize("params", GR5J_PARAM_SETS)
    def test_flow_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        res_py, _ = _run_model_python(GR5J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR5J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"GR5J class-level flow mismatch for {params}",
        )

    @pytest.mark.parametrize("params", GR5J_PARAM_SETS)
    def test_final_state_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        _, st_py = _run_model_python(GR5J, params, precip, pet)
        _, st_nb = _run_model_numba(GR5J, params, precip, pet)
        for key in ("production_store", "routing_store"):
            assert st_nb.values[key] == pytest.approx(
                st_py.values[key], rel=1e-12
            ), f"GR5J state '{key}' mismatch"
        np.testing.assert_allclose(
            np.array(st_nb.values["uh2"]),
            np.array(st_py.values["uh2"]),
            rtol=1e-12, atol=1e-14,
        )

    def test_10yr_flow_identical(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.5}
        res_py, _ = _run_model_python(GR5J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR5J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values, rtol=1e-12,
        )

    def test_storm_burst(self, storm_burst):
        precip, pet = storm_burst
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.5}
        res_py, _ = _run_model_python(GR5J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR5J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.5}
        res_py, _ = _run_model_python(GR5J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR5J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )


class TestSyntheticPythonVsNumbaGR6J:

    @pytest.mark.parametrize("params", GR6J_PARAM_SETS)
    def test_flow_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        res_py, _ = _run_model_python(GR6J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR6J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"GR6J class-level flow mismatch for {params}",
        )

    @pytest.mark.parametrize("params", GR6J_PARAM_SETS)
    def test_final_state_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        _, st_py = _run_model_python(GR6J, params, precip, pet)
        _, st_nb = _run_model_numba(GR6J, params, precip, pet)
        for key in ("production_store", "routing_store", "exponential_store"):
            assert st_nb.values[key] == pytest.approx(
                st_py.values[key], rel=1e-12
            ), f"GR6J state '{key}' mismatch"
        np.testing.assert_allclose(
            np.array(st_nb.values["uh1"]),
            np.array(st_py.values["uh1"]),
            rtol=1e-12, atol=1e-14,
        )
        np.testing.assert_allclose(
            np.array(st_nb.values["uh2"]),
            np.array(st_py.values["uh2"]),
            rtol=1e-12, atol=1e-14,
        )

    def test_10yr_flow_identical(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.1, "X6": 50.0}
        res_py, _ = _run_model_python(GR6J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR6J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values, rtol=1e-12,
        )

    def test_storm_burst(self, storm_burst):
        precip, pet = storm_burst
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.1, "X6": 50.0}
        res_py, _ = _run_model_python(GR6J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR6J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        params = {"X1": 350.0, "X2": 0.5, "X3": 90.0, "X4": 1.7, "X5": 0.1, "X6": 50.0}
        res_py, _ = _run_model_python(GR6J, params, precip, pet)
        res_nb, _ = _run_model_numba(GR6J, params, precip, pet)
        np.testing.assert_allclose(
            res_nb["flow"].values, res_py["flow"].values,
            rtol=1e-12, atol=1e-14,
        )


class TestSyntheticPythonVsNumbaSacramento:

    @pytest.mark.parametrize("params", SAC_PARAM_SETS)
    def test_flow_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        res_py, _ = _run_model_python(Sacramento, params, precip, pet)
        res_nb, _ = _run_model_numba(Sacramento, params, precip, pet)
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
                err_msg=f"Sacramento class-level '{col}' mismatch",
            )

    @pytest.mark.parametrize("params", SAC_PARAM_SETS)
    def test_final_state_identical(self, params, synthetic_1yr):
        precip, pet = synthetic_1yr
        _, st_py = _run_model_python(Sacramento, params, precip, pet)
        _, st_nb = _run_model_numba(Sacramento, params, precip, pet)
        for key in ("uztwc", "uzfwc", "lztwc", "lzfsc", "lzfpc",
                     "adimc", "hydrograph_store"):
            assert st_nb.values[key] == pytest.approx(
                st_py.values[key], rel=1e-10, abs=1e-12,
            ), f"Sacramento state '{key}' mismatch"

    def test_10yr_flow_identical(self, synthetic_10yr):
        precip, pet = synthetic_10yr
        res_py, _ = _run_model_python(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        res_nb, _ = _run_model_numba(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
            )

    def test_storm_burst(self, storm_burst):
        precip, pet = storm_burst
        res_py, _ = _run_model_python(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        res_nb, _ = _run_model_numba(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
            )

    def test_alternating_wet_dry(self, alternating_wet_dry):
        precip, pet = alternating_wet_dry
        res_py, _ = _run_model_python(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        res_nb, _ = _run_model_numba(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
            )

    def test_zero_precip(self, zero_precip):
        precip, pet = zero_precip
        res_py, _ = _run_model_python(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        res_nb, _ = _run_model_numba(Sacramento, SAC_DEFAULT_PARAMS, precip, pet)
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
            )


# =========================================================================
# Real-data equivalence: Gauge 410734 (Queanbeyan)
# =========================================================================

DATA_410734_DIR = Path(__file__).resolve().parents[3] / "data" / "410734"

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
    """Load the full 410734 rainfall and PET series."""
    rain_path = DATA_410734_DIR / "Default Input Set - Rain_QBN01.csv"
    pet_path = DATA_410734_DIR / "Default Input Set - Mwet_QBN01.csv"
    rain_df = pd.read_csv(rain_path, parse_dates=["Date"], index_col="Date")
    pet_df = pd.read_csv(pet_path, parse_dates=["Date"], index_col="Date")

    merged = rain_df.join(pet_df, how="inner")
    merged.columns = ["precipitation", "pet"]
    return merged


class TestRealData410734Equivalence:
    """Python vs Numba on the full 410734 Queanbeyan dataset with calibrated
    Sacramento parameters.  This is the strongest available integration test:
    ~135 years of real daily data with realistic parameter values."""

    pytestmark = pytest.mark.skipif(
        _can_skip_410734(),
        reason="410734 data files not found",
    )

    def test_full_record_flow_identical(self, data_410734):
        """Run entire ~49k-day record and compare every timestep."""
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)

        res_py, st_py = _run_model_python(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        res_nb, st_nb = _run_model_numba(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )

        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
                err_msg=f"410734 Sacramento '{col}' mismatch over full record",
            )

    def test_full_record_final_state_identical(self, data_410734):
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)

        _, st_py = _run_model_python(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        _, st_nb = _run_model_numba(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        for key in ("uztwc", "uzfwc", "lztwc", "lzfsc", "lzfpc",
                     "adimc", "hydrograph_store"):
            assert st_nb.values[key] == pytest.approx(
                st_py.values[key], rel=1e-10, abs=1e-12,
            ), f"410734 Sacramento final state '{key}' mismatch"

    def test_full_record_nonneg_and_finite(self, data_410734):
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)

        res_nb, _ = _run_model_numba(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        for col in ("runoff", "baseflow", "channel_flow"):
            vals = res_nb[col].values
            assert np.all(np.isfinite(vals)), f"NaN/Inf in 410734 '{col}'"
            assert np.all(vals >= 0), f"Negative values in 410734 '{col}'"

    def test_subset_2000_2023_flow_identical(self, data_410734):
        """Subset matching the 410734_norouting.csv period (2000-2023)."""
        sub = data_410734.loc["2000-01-01":"2023-11-01"]
        precip = sub["precipitation"].values.astype(np.float64)
        pet = sub["pet"].values.astype(np.float64)

        res_py, _ = _run_model_python(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        res_nb, _ = _run_model_numba(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        for col in ("runoff", "baseflow", "channel_flow"):
            np.testing.assert_allclose(
                res_nb[col].values, res_py[col].values,
                rtol=1e-10, atol=1e-12,
                err_msg=f"410734 2000-2023 '{col}' mismatch",
            )

    def test_mass_balance_consistency(self, data_410734):
        """Verify Python and Numba produce identical cumulative volumes.
        Any drift in mass balance would amplify over 135 years."""
        precip = data_410734["precipitation"].values.astype(np.float64)
        pet = data_410734["pet"].values.astype(np.float64)

        res_py, _ = _run_model_python(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )
        res_nb, _ = _run_model_numba(
            Sacramento, SAC_410734_PARAMS, precip, pet,
        )

        for col in ("runoff", "baseflow", "channel_flow"):
            cum_py = np.cumsum(res_py[col].values)
            cum_nb = np.cumsum(res_nb[col].values)
            np.testing.assert_allclose(
                cum_nb, cum_py, rtol=1e-9,
                err_msg=f"410734 cumulative '{col}' drift between backends",
            )
