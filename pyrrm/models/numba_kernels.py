"""
Numba JIT-compiled kernels for rainfall-runoff models.

Provides accelerated versions of the core computational loops for
Sacramento, GR4J, GR5J, and GR6J models. When Numba is available,
the model classes automatically dispatch to these kernels for
50-200x speedups over the pure-Python implementations.

All functions use @njit(cache=True) to avoid recompilation across
sessions. The cache files (.nbi/.nbc) are written next to this module.
"""

import math
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def prange(*args):
        """Fallback to range when Numba is not installed."""
        return range(*args)

    def njit(*args, **kwargs):
        """No-op decorator when Numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =========================================================================
# S-curve functions (used by GR4J, GR5J, GR6J)
# =========================================================================

@njit(cache=True)
def _s_curve1(t, x4, exponent):
    if t <= 0.0:
        return 0.0
    if t < x4:
        return (t / x4) ** exponent
    return 1.0


@njit(cache=True)
def _s_curve2(t, x4, exponent):
    if t <= 0.0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** exponent
    elif t < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** exponent
    return 1.0


# =========================================================================
# Sacramento unit hydrograph step
# =========================================================================

@njit(cache=True)
def _uh_step(input_value, uh_stores, uh_scurve):
    n = len(uh_stores)
    if n == 0:
        return input_value
    for i in range(n):
        uh_stores[i] += input_value * uh_scurve[i]
    output = uh_stores[0]
    for i in range(n - 1):
        uh_stores[i] = uh_stores[i + 1]
    uh_stores[n - 1] = 0.0
    return output


# =========================================================================
# GR4J core
# =========================================================================

@njit(cache=True)
def _gr4j_core_numba(
    x1, x2, x3, x4,
    precipitation, evapotranspiration,
    production_store, routing_store,
    uh1_stores, uh2_stores,
):
    n_timesteps = len(precipitation)
    flow = np.empty(n_timesteps)

    storage_fraction = 0.9
    exponent = 2.5

    n_uh1 = max(1, int(math.ceil(x4)))
    n_uh2 = max(1, int(math.ceil(2.0 * x4)))

    o_uh1 = np.zeros(n_uh1)
    o_uh2 = np.zeros(n_uh2)

    for i in range(1, n_uh1 + 1):
        o_uh1[i - 1] = _s_curve1(float(i), x4, exponent) - _s_curve1(float(i - 1), x4, exponent)
    for i in range(1, n_uh2 + 1):
        o_uh2[i - 1] = _s_curve2(float(i), x4, exponent) - _s_curve2(float(i - 1), x4, exponent)

    if len(uh1_stores) < n_uh1:
        uh1 = np.zeros(n_uh1)
        for i in range(len(uh1_stores)):
            uh1[i] = uh1_stores[i]
    else:
        uh1 = uh1_stores[:n_uh1].copy()

    if len(uh2_stores) < n_uh2:
        uh2 = np.zeros(n_uh2)
        for i in range(len(uh2_stores)):
            uh2[i] = uh2_stores[i]
    else:
        uh2 = uh2_stores[:n_uh2].copy()

    prod_store = production_store
    rout_store = routing_store

    for t in range(n_timesteps):
        rain = precipitation[t]
        evap = evapotranspiration[t]

        rout_input = 0.0
        psf = prod_store / x1 if x1 > 0.0 else 0.0

        if rain <= evap:
            scaled_net_rain = (evap - rain) / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + (1.0 - psf) * scaled_net_rain
            if denom > 0.0:
                prod_evap = prod_store * (2.0 - psf) * scaled_net_rain / denom
            else:
                prod_evap = 0.0

            prod_store -= prod_evap
        else:
            net_rainfall = rain - evap
            scaled_net_rain = net_rainfall / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + psf * scaled_net_rain
            if denom > 0.0:
                prod_rainfall = x1 * (1.0 - psf * psf) * scaled_net_rain / denom
            else:
                prod_rainfall = 0.0

            rout_input = net_rainfall - prod_rainfall
            prod_store += prod_rainfall

        if prod_store < 0.0:
            prod_store = 0.0

        psf_p4 = (prod_store / x1) ** 4.0 if x1 > 0.0 else 0.0
        percolation = prod_store * (1.0 - 1.0 / (1.0 + psf_p4 / 25.62891) ** 0.25)

        prod_store -= percolation
        rout_input += percolation

        for i in range(n_uh1 - 1):
            uh1[i] = uh1[i + 1] + o_uh1[i] * rout_input
        uh1[n_uh1 - 1] = o_uh1[n_uh1 - 1] * rout_input

        for i in range(n_uh2 - 1):
            uh2[i] = uh2[i + 1] + o_uh2[i] * rout_input
        uh2[n_uh2 - 1] = o_uh2[n_uh2 - 1] * rout_input

        groundwater_exchange = x2 * (rout_store / x3) ** 3.5 if x3 > 0.0 else 0.0
        rout_store += uh1[0] * storage_fraction + groundwater_exchange

        if rout_store < 0.0:
            rout_store = 0.0

        rsf_p4 = (rout_store / x3) ** 4.0 if x3 > 0.0 else 0.0
        rout_flow = rout_store * (1.0 - 1.0 / (1.0 + rsf_p4) ** 0.25)

        direct_flow = uh2[0] * (1.0 - storage_fraction) + groundwater_exchange
        if direct_flow < 0.0:
            direct_flow = 0.0

        rout_store -= rout_flow
        flow[t] = rout_flow + direct_flow

    return flow, prod_store, rout_store, uh1, uh2


# =========================================================================
# GR5J core
# =========================================================================

@njit(cache=True)
def _gr5j_core_numba(
    x1, x2, x3, x4, x5,
    precipitation, evapotranspiration,
    production_store, routing_store,
    uh2_stores,
):
    n_timesteps = len(precipitation)
    flow = np.empty(n_timesteps)

    storage_fraction = 0.9
    exponent = 2.5

    n_uh2 = max(1, int(math.ceil(2.0 * x4)))

    o_uh2 = np.zeros(n_uh2)
    for i in range(1, n_uh2 + 1):
        o_uh2[i - 1] = _s_curve2(float(i), x4, exponent) - _s_curve2(float(i - 1), x4, exponent)

    if len(uh2_stores) < n_uh2:
        uh2 = np.zeros(n_uh2)
        for i in range(len(uh2_stores)):
            uh2[i] = uh2_stores[i]
    else:
        uh2 = uh2_stores[:n_uh2].copy()

    prod_store = production_store
    rout_store = routing_store

    for t in range(n_timesteps):
        rain = precipitation[t]
        evap = evapotranspiration[t]

        rout_input = 0.0
        psf = prod_store / x1 if x1 > 0.0 else 0.0

        if rain <= evap:
            scaled_net_rain = (evap - rain) / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + (1.0 - psf) * scaled_net_rain
            if denom > 0.0:
                prod_evap = prod_store * (2.0 - psf) * scaled_net_rain / denom
            else:
                prod_evap = 0.0

            prod_store -= prod_evap
        else:
            net_rainfall = rain - evap
            scaled_net_rain = net_rainfall / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + psf * scaled_net_rain
            if denom > 0.0:
                prod_rainfall = x1 * (1.0 - psf * psf) * scaled_net_rain / denom
            else:
                prod_rainfall = 0.0

            rout_input = net_rainfall - prod_rainfall
            prod_store += prod_rainfall

        if prod_store < 0.0:
            prod_store = 0.0

        psf_p4 = (prod_store / x1) ** 4.0 if x1 > 0.0 else 0.0
        percolation = prod_store * (1.0 - 1.0 / (1.0 + psf_p4 / 25.62890625) ** 0.25)

        prod_store -= percolation
        rout_input += percolation

        for i in range(n_uh2 - 1):
            uh2[i] = uh2[i + 1] + o_uh2[i] * rout_input
        uh2[n_uh2 - 1] = o_uh2[n_uh2 - 1] * rout_input

        groundwater_exchange = x2 * (rout_store / x3 - x5) if x3 > 0.0 else 0.0
        rout_store += uh2[0] * storage_fraction + groundwater_exchange

        if rout_store < 0.0:
            rout_store = 0.0

        rsf_p4 = (rout_store / x3) ** 4.0 if x3 > 0.0 else 0.0
        rout_flow = rout_store * (1.0 - 1.0 / (1.0 + rsf_p4) ** 0.25)

        direct_flow = uh2[0] * (1.0 - storage_fraction) + groundwater_exchange
        if direct_flow < 0.0:
            direct_flow = 0.0

        rout_store -= rout_flow
        flow[t] = rout_flow + direct_flow

    return flow, prod_store, rout_store, uh2


# =========================================================================
# GR6J core
# =========================================================================

@njit(cache=True)
def _gr6j_core_numba(
    x1, x2, x3, x4, x5, x6,
    precipitation, evapotranspiration,
    production_store, routing_store, exponential_store,
    uh1_stores, uh2_stores,
):
    n_timesteps = len(precipitation)
    flow = np.empty(n_timesteps)

    storage_fraction = 0.9
    exp_fraction = 0.4
    exponent = 2.5

    n_uh1 = max(1, int(math.ceil(x4)))
    n_uh2 = max(1, int(math.ceil(2.0 * x4)))

    o_uh1 = np.zeros(n_uh1)
    o_uh2 = np.zeros(n_uh2)

    for i in range(1, n_uh1 + 1):
        o_uh1[i - 1] = _s_curve1(float(i), x4, exponent) - _s_curve1(float(i - 1), x4, exponent)
    for i in range(1, n_uh2 + 1):
        o_uh2[i - 1] = _s_curve2(float(i), x4, exponent) - _s_curve2(float(i - 1), x4, exponent)

    if len(uh1_stores) < n_uh1:
        uh1 = np.zeros(n_uh1)
        for i in range(len(uh1_stores)):
            uh1[i] = uh1_stores[i]
    else:
        uh1 = uh1_stores[:n_uh1].copy()

    if len(uh2_stores) < n_uh2:
        uh2 = np.zeros(n_uh2)
        for i in range(len(uh2_stores)):
            uh2[i] = uh2_stores[i]
    else:
        uh2 = uh2_stores[:n_uh2].copy()

    prod_store = production_store
    rout_store = routing_store
    exp_store = exponential_store

    for t in range(n_timesteps):
        rain = precipitation[t]
        evap = evapotranspiration[t]

        rout_input = 0.0
        psf = prod_store / x1 if x1 > 0.0 else 0.0

        if rain <= evap:
            scaled_net_rain = (evap - rain) / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + (1.0 - psf) * scaled_net_rain
            if denom > 0.0:
                prod_evap = prod_store * (2.0 - psf) * scaled_net_rain / denom
            else:
                prod_evap = 0.0

            prod_store -= prod_evap
        else:
            net_rainfall = rain - evap
            scaled_net_rain = net_rainfall / x1 if x1 > 0.0 else 0.0
            if scaled_net_rain > 13.0:
                scaled_net_rain = 13.0
            scaled_net_rain = math.tanh(scaled_net_rain)

            denom = 1.0 + psf * scaled_net_rain
            if denom > 0.0:
                prod_rainfall = x1 * (1.0 - psf * psf) * scaled_net_rain / denom
            else:
                prod_rainfall = 0.0

            rout_input = net_rainfall - prod_rainfall
            prod_store += prod_rainfall

        if prod_store < 0.0:
            prod_store = 0.0

        psf_p4 = (prod_store / x1) ** 4.0 if x1 > 0.0 else 0.0
        percolation = prod_store * (1.0 - 1.0 / (1.0 + psf_p4 / 25.62890625) ** 0.25)

        prod_store -= percolation
        rout_input += percolation

        for i in range(n_uh1 - 1):
            uh1[i] = uh1[i + 1] + o_uh1[i] * rout_input
        uh1[n_uh1 - 1] = o_uh1[n_uh1 - 1] * rout_input

        for i in range(n_uh2 - 1):
            uh2[i] = uh2[i + 1] + o_uh2[i] * rout_input
        uh2[n_uh2 - 1] = o_uh2[n_uh2 - 1] * rout_input

        groundwater_exchange = x2 * (rout_store / x3 - x5) if x3 > 0.0 else 0.0

        rout_store += uh1[0] * storage_fraction * (1.0 - exp_fraction) + groundwater_exchange

        if rout_store < 0.0:
            rout_store = 0.0

        rsf_p4 = (rout_store / x3) ** 4.0 if x3 > 0.0 else 0.0
        rout_flow = rout_store * (1.0 - 1.0 / (1.0 + rsf_p4) ** 0.25)
        rout_store -= rout_flow

        exp_store += uh1[0] * storage_fraction * exp_fraction + groundwater_exchange

        ar = exp_store / x6 if x6 > 0.0 else 0.0
        if ar > 33.0:
            ar = 33.0
        elif ar < -33.0:
            ar = -33.0

        if ar > 7.0:
            exp_flow = exp_store + x6 / math.exp(ar)
        elif ar < -7.0:
            exp_flow = x6 * math.exp(ar)
        else:
            exp_flow = x6 * math.log(math.exp(ar) + 1.0)

        exp_store -= exp_flow

        direct_flow = uh2[0] * (1.0 - storage_fraction) + groundwater_exchange
        if direct_flow < 0.0:
            direct_flow = 0.0

        flow[t] = rout_flow + direct_flow + exp_flow

    return flow, prod_store, rout_store, exp_store, uh1, uh2


# =========================================================================
# Sacramento core — full time-series version
# =========================================================================

LENGTH_OF_UNIT_HYDROGRAPH = 5
_PDN20 = 5.08
_PDNOR = 25.4


@njit(cache=True)
def _sacramento_run_numba(
    precip, pet,
    uztwm, uzfwm, lztwm, lzfpm, lzfsm,
    uzk, lzpk, lzsk,
    zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
    uh_scurve,
    init_uztwc, init_uzfwc, init_lztwc, init_lzfsc, init_lzfpc,
    init_adimc, init_hydrograph_store,
    init_uh_stores,
):
    n = len(precip)
    runoff_out = np.empty(n)
    baseflow_out = np.empty(n)
    channel_flow_out = np.empty(n)

    alzfsm = lzfsm * (1.0 + side)
    alzfpm = lzfpm * (1.0 + side)

    uztwc = init_uztwc
    uzfwc = init_uzfwc
    lztwc = init_lztwc
    lzfsc = init_lzfsc
    lzfpc = init_lzfpc
    adimc = init_adimc
    hydrograph_store = init_hydrograph_store

    alzfsc = lzfsc * (1.0 + side)
    alzfpc = lzfpc * (1.0 + side)
    pbase = alzfsm * lzsk + alzfpm * lzpk

    uh_stores = init_uh_stores.copy()
    n_uh = len(uh_scurve)

    for t in range(n):
        reserved_lower_zone = rserv * (lzfpm + lzfsm)
        evapt = pet[t]
        pliq = precip[t]

        # Evaporation from upper zone tension water
        if uztwm > 0.0:
            evap_uztw = evapt * uztwc / uztwm
        else:
            evap_uztw = 0.0

        if uztwc < evap_uztw:
            evap_uztw = uztwc
            uztwc = 0.0
            evap_uzfw = min((evapt - evap_uztw), uzfwc)
            uzfwc = uzfwc - evap_uzfw
        else:
            uztwc = uztwc - evap_uztw
            evap_uzfw = 0.0

        if uztwm > 0.0:
            ratio_uztw = uztwc / uztwm
        else:
            ratio_uztw = 1.0

        if uzfwm > 0.0:
            ratio_uzfw = uzfwc / uzfwm
        else:
            ratio_uzfw = 1.0

        if ratio_uztw < ratio_uzfw:
            ratio_uztw = (uztwc + uzfwc) / (uztwm + uzfwm)
            uztwc = uztwm * ratio_uztw
            uzfwc = uzfwm * ratio_uztw

        if uztwm + lztwm > 0.0:
            e3 = min(
                (evapt - evap_uztw - evap_uzfw) * lztwc / (uztwm + lztwm),
                lztwc,
            )
            e5 = min(
                evap_uztw + ((evapt - evap_uztw - evap_uzfw)
                             * (adimc - evap_uztw - uztwc)
                             / (uztwm + lztwm)),
                adimc,
            )
        else:
            e3 = 0.0
            e5 = 0.0

        lztwc = lztwc - e3
        adimc = adimc - e5
        evap_uztw = evap_uztw * (1.0 - adimp - pctim)
        evap_uzfw = evap_uzfw * (1.0 - adimp - pctim)
        e3 = e3 * (1.0 - adimp - pctim)
        e5 = e5 * adimp

        if lztwm > 0.0:
            ratio_lztw = lztwc / lztwm
        else:
            ratio_lztw = 1.0

        denom_lzfw = alzfpm + alzfsm - reserved_lower_zone + lztwm
        if denom_lzfw > 0.0:
            ratio_lzfw = (alzfpc + alzfsc - reserved_lower_zone + lztwc) / denom_lzfw
        else:
            ratio_lzfw = 1.0

        if ratio_lztw < ratio_lzfw:
            transfered = (ratio_lzfw - ratio_lztw) * lztwm
            lztwc = lztwc + transfered
            alzfsc = alzfsc - transfered
            if alzfsc < 0.0:
                alzfpc = alzfpc + alzfsc
                alzfsc = 0.0

        roimp = pliq * pctim

        pav = pliq + uztwc - uztwm
        if pav < 0.0:
            adimc = adimc + pliq
            uztwc = uztwc + pliq
            pav = 0.0
        else:
            adimc = adimc + uztwm - uztwc
            uztwc = uztwm

        if pav <= _PDN20:
            adj = 1.0
            itime = 2
        else:
            if pav < _PDNOR:
                adj = 0.5 * math.sqrt(pav / _PDNOR)
            else:
                adj = 1.0 - 0.5 * _PDNOR / pav
            itime = 1

        flobf = 0.0
        flosf = 0.0
        floin = 0.0

        hpl = alzfpm / (alzfpm + alzfsm) if (alzfpm + alzfsm) > 0.0 else 0.5

        for ii in range(itime, 3):
            ninc = int(math.floor((uzfwc * adj + pav) * 0.2)) + 1
            dinc = 1.0 / ninc
            pinc = pav * dinc
            dinc = dinc * adj

            if ninc == 1 and adj >= 1.0:
                duz = uzk
                dlzp = lzpk
                dlzs = lzsk
            else:
                duz = 1.0 - (1.0 - uzk) ** dinc if uzk < 1.0 else 1.0
                dlzp = 1.0 - (1.0 - lzpk) ** dinc if lzpk < 1.0 else 1.0
                dlzs = 1.0 - (1.0 - lzsk) ** dinc if lzsk < 1.0 else 1.0

            for inc in range(1, ninc + 1):
                ratio = (adimc - uztwc) / lztwm if lztwm > 0.0 else 0.0
                addro = pinc * ratio * ratio

                if alzfpc > 0.0:
                    bf = alzfpc * dlzp
                else:
                    alzfpc = 0.0
                    bf = 0.0

                flobf = flobf + bf
                alzfpc = alzfpc - bf

                if alzfsc > 0.0:
                    bf = alzfsc * dlzs
                else:
                    alzfsc = 0.0
                    bf = 0.0

                alzfsc = alzfsc - bf
                flobf = flobf + bf

                if uzfwc > 0.0:
                    lzair = (lztwm - lztwc + alzfsm - alzfsc + alzfpm - alzfpc)
                    if lzair > 0.0:
                        perc = (pbase * dinc * uzfwc) / uzfwm if uzfwm > 0.0 else 0.0
                        total_lz = alzfpm + alzfsm + lztwm
                        current_lz = alzfpc + alzfsc + lztwc
                        if total_lz > 0.0:
                            deficit_ratio = 1.0 - current_lz / total_lz
                            perc = min(
                                uzfwc,
                                perc * (1.0 + (zperc * deficit_ratio ** rexp)),
                            )
                        perc = min(lzair, perc)
                        uzfwc = uzfwc - perc
                    else:
                        perc = 0.0

                    transfered_if = duz * uzfwc
                    floin = floin + transfered_if
                    uzfwc = uzfwc - transfered_if

                    perctw = min(perc * (1.0 - pfree), lztwm - lztwc)
                    percfw = perc - perctw

                    lzair = alzfsm - alzfsc + alzfpm - alzfpc
                    if percfw > lzair:
                        perctw = perctw + percfw - lzair
                        percfw = lzair
                    lztwc = lztwc + perctw

                    if percfw > 0.0:
                        ratlp = 1.0 - alzfpc / alzfpm if alzfpm > 0.0 else 0.0
                        ratls = 1.0 - alzfsc / alzfsm if alzfsm > 0.0 else 0.0
                        if ratlp + ratls > 0.0:
                            percs = min(
                                alzfsm - alzfsc,
                                percfw * (1.0 - hpl * (2.0 * ratlp) / (ratlp + ratls)),
                            )
                        else:
                            percs = 0.0
                        alzfsc = alzfsc + percs
                        if alzfsc > alzfsm:
                            percs = percs - alzfsc + alzfsm
                            alzfsc = alzfsm
                        alzfpc = alzfpc + percfw - percs
                        if alzfpc > alzfpm:
                            alzfsc = alzfsc + alzfpc - alzfpm
                            alzfpc = alzfpm

                if pinc > 0.0:
                    pav_local = pinc
                    if pav_local - uzfwm + uzfwc <= 0.0:
                        uzfwc = uzfwc + pav_local
                    else:
                        pav_local = pav_local - uzfwm + uzfwc
                        uzfwc = uzfwm
                        flosf = flosf + pav_local
                        addro = addro + pav_local * (1.0 - addro / pinc)

                adimc = adimc + pinc - addro
                roimp = roimp + addro * adimp

            adj = 1.0 - adj
            pav = 0.0

        flosf = flosf * (1.0 - pctim - adimp)
        floin = floin * (1.0 - pctim - adimp)
        flobf = flobf * (1.0 - pctim - adimp)

        lzfsc = alzfsc / (1.0 + side)
        lzfpc = alzfpc / (1.0 + side)

        # Unit hydrograph routing (inline)
        uh_input = flosf + roimp + floin
        if n_uh == 0:
            flwsf = uh_input
        else:
            for i in range(n_uh):
                uh_stores[i] += uh_input * uh_scurve[i]
            flwsf = uh_stores[0]
            for i in range(n_uh - 1):
                uh_stores[i] = uh_stores[i + 1]
            uh_stores[n_uh - 1] = 0.0
        hydrograph_store += (uh_input - flwsf)

        flwbf = flobf / (1.0 + side)
        if flwbf < 0.0:
            flwbf = 0.0

        total_before_channel_losses = flwbf + flwsf
        ratio_baseflow = flwbf / total_before_channel_losses if total_before_channel_losses > 0.0 else 0.0

        channel_flow = flwbf + flwsf - ssout
        if channel_flow < 0.0:
            channel_flow = 0.0
        evaporation_channel_water = min(evapt * sarva, channel_flow)

        runoff = channel_flow - evaporation_channel_water
        baseflow = runoff * ratio_baseflow

        if runoff != runoff:  # NaN check without math.isnan
            runoff = 0.0
            baseflow = 0.0
            channel_flow = 0.0

        runoff_out[t] = runoff
        baseflow_out[t] = baseflow
        channel_flow_out[t] = channel_flow

    return (
        runoff_out, baseflow_out, channel_flow_out,
        uztwc, uzfwc, lztwc, lzfsc, lzfpc,
        alzfsc, alzfpc, adimc, hydrograph_store,
        uh_stores,
    )


# =========================================================================
# Sacramento core — fastmath variant (relaxed IEEE-754 for ~10-30% speedup)
# =========================================================================

@njit(cache=True, fastmath=True)
def _sacramento_run_numba_fast(
    precip, pet,
    uztwm, uzfwm, lztwm, lzfpm, lzfsm,
    uzk, lzpk, lzsk,
    zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
    uh_scurve,
    init_uztwc, init_uzfwc, init_lztwc, init_lzfsc, init_lzfpc,
    init_adimc, init_hydrograph_store,
    init_uh_stores,
):
    n = len(precip)
    runoff_out = np.empty(n)
    baseflow_out = np.empty(n)
    channel_flow_out = np.empty(n)

    alzfsm = lzfsm * (1.0 + side)
    alzfpm = lzfpm * (1.0 + side)

    uztwc = init_uztwc
    uzfwc = init_uzfwc
    lztwc = init_lztwc
    lzfsc = init_lzfsc
    lzfpc = init_lzfpc
    adimc = init_adimc
    hydrograph_store = init_hydrograph_store

    alzfsc = lzfsc * (1.0 + side)
    alzfpc = lzfpc * (1.0 + side)
    pbase = alzfsm * lzsk + alzfpm * lzpk

    uh_stores = init_uh_stores.copy()
    n_uh = len(uh_scurve)

    for t in range(n):
        reserved_lower_zone = rserv * (lzfpm + lzfsm)
        evapt = pet[t]
        pliq = precip[t]

        if uztwm > 0.0:
            evap_uztw = evapt * uztwc / uztwm
        else:
            evap_uztw = 0.0

        if uztwc < evap_uztw:
            evap_uztw = uztwc
            uztwc = 0.0
            evap_uzfw = min((evapt - evap_uztw), uzfwc)
            uzfwc = uzfwc - evap_uzfw
        else:
            uztwc = uztwc - evap_uztw
            evap_uzfw = 0.0

        if uztwm > 0.0:
            ratio_uztw = uztwc / uztwm
        else:
            ratio_uztw = 1.0

        if uzfwm > 0.0:
            ratio_uzfw = uzfwc / uzfwm
        else:
            ratio_uzfw = 1.0

        if ratio_uztw < ratio_uzfw:
            ratio_uztw = (uztwc + uzfwc) / (uztwm + uzfwm)
            uztwc = uztwm * ratio_uztw
            uzfwc = uzfwm * ratio_uztw

        if uztwm + lztwm > 0.0:
            e3 = min(
                (evapt - evap_uztw - evap_uzfw) * lztwc / (uztwm + lztwm),
                lztwc,
            )
            e5 = min(
                evap_uztw + ((evapt - evap_uztw - evap_uzfw)
                             * (adimc - evap_uztw - uztwc)
                             / (uztwm + lztwm)),
                adimc,
            )
        else:
            e3 = 0.0
            e5 = 0.0

        lztwc = lztwc - e3
        adimc = adimc - e5
        evap_uztw = evap_uztw * (1.0 - adimp - pctim)
        evap_uzfw = evap_uzfw * (1.0 - adimp - pctim)
        e3 = e3 * (1.0 - adimp - pctim)
        e5 = e5 * adimp

        if lztwm > 0.0:
            ratio_lztw = lztwc / lztwm
        else:
            ratio_lztw = 1.0

        denom_lzfw = alzfpm + alzfsm - reserved_lower_zone + lztwm
        if denom_lzfw > 0.0:
            ratio_lzfw = (alzfpc + alzfsc - reserved_lower_zone + lztwc) / denom_lzfw
        else:
            ratio_lzfw = 1.0

        if ratio_lztw < ratio_lzfw:
            transfered = (ratio_lzfw - ratio_lztw) * lztwm
            lztwc = lztwc + transfered
            alzfsc = alzfsc - transfered
            if alzfsc < 0.0:
                alzfpc = alzfpc + alzfsc
                alzfsc = 0.0

        roimp = pliq * pctim

        pav = pliq + uztwc - uztwm
        if pav < 0.0:
            adimc = adimc + pliq
            uztwc = uztwc + pliq
            pav = 0.0
        else:
            adimc = adimc + uztwm - uztwc
            uztwc = uztwm

        if pav <= _PDN20:
            adj = 1.0
            itime = 2
        else:
            if pav < _PDNOR:
                adj = 0.5 * math.sqrt(pav / _PDNOR)
            else:
                adj = 1.0 - 0.5 * _PDNOR / pav
            itime = 1

        flobf = 0.0
        flosf = 0.0
        floin = 0.0

        hpl = alzfpm / (alzfpm + alzfsm) if (alzfpm + alzfsm) > 0.0 else 0.5

        for ii in range(itime, 3):
            ninc = int(math.floor((uzfwc * adj + pav) * 0.2)) + 1
            dinc = 1.0 / ninc
            pinc = pav * dinc
            dinc = dinc * adj

            if ninc == 1 and adj >= 1.0:
                duz = uzk
                dlzp = lzpk
                dlzs = lzsk
            else:
                duz = 1.0 - (1.0 - uzk) ** dinc if uzk < 1.0 else 1.0
                dlzp = 1.0 - (1.0 - lzpk) ** dinc if lzpk < 1.0 else 1.0
                dlzs = 1.0 - (1.0 - lzsk) ** dinc if lzsk < 1.0 else 1.0

            for inc in range(1, ninc + 1):
                ratio = (adimc - uztwc) / lztwm if lztwm > 0.0 else 0.0
                addro = pinc * ratio * ratio

                if alzfpc > 0.0:
                    bf = alzfpc * dlzp
                else:
                    alzfpc = 0.0
                    bf = 0.0

                flobf = flobf + bf
                alzfpc = alzfpc - bf

                if alzfsc > 0.0:
                    bf = alzfsc * dlzs
                else:
                    alzfsc = 0.0
                    bf = 0.0

                alzfsc = alzfsc - bf
                flobf = flobf + bf

                if uzfwc > 0.0:
                    lzair = (lztwm - lztwc + alzfsm - alzfsc + alzfpm - alzfpc)
                    if lzair > 0.0:
                        perc = (pbase * dinc * uzfwc) / uzfwm if uzfwm > 0.0 else 0.0
                        total_lz = alzfpm + alzfsm + lztwm
                        current_lz = alzfpc + alzfsc + lztwc
                        if total_lz > 0.0:
                            deficit_ratio = 1.0 - current_lz / total_lz
                            perc = min(
                                uzfwc,
                                perc * (1.0 + (zperc * deficit_ratio ** rexp)),
                            )
                        perc = min(lzair, perc)
                        uzfwc = uzfwc - perc
                    else:
                        perc = 0.0

                    transfered_if = duz * uzfwc
                    floin = floin + transfered_if
                    uzfwc = uzfwc - transfered_if

                    perctw = min(perc * (1.0 - pfree), lztwm - lztwc)
                    percfw = perc - perctw

                    lzair = alzfsm - alzfsc + alzfpm - alzfpc
                    if percfw > lzair:
                        perctw = perctw + percfw - lzair
                        percfw = lzair
                    lztwc = lztwc + perctw

                    if percfw > 0.0:
                        ratlp = 1.0 - alzfpc / alzfpm if alzfpm > 0.0 else 0.0
                        ratls = 1.0 - alzfsc / alzfsm if alzfsm > 0.0 else 0.0
                        if ratlp + ratls > 0.0:
                            percs = min(
                                alzfsm - alzfsc,
                                percfw * (1.0 - hpl * (2.0 * ratlp) / (ratlp + ratls)),
                            )
                        else:
                            percs = 0.0
                        alzfsc = alzfsc + percs
                        if alzfsc > alzfsm:
                            percs = percs - alzfsc + alzfsm
                            alzfsc = alzfsm
                        alzfpc = alzfpc + percfw - percs
                        if alzfpc > alzfpm:
                            alzfsc = alzfsc + alzfpc - alzfpm
                            alzfpc = alzfpm

                if pinc > 0.0:
                    pav_local = pinc
                    if pav_local - uzfwm + uzfwc <= 0.0:
                        uzfwc = uzfwc + pav_local
                    else:
                        pav_local = pav_local - uzfwm + uzfwc
                        uzfwc = uzfwm
                        flosf = flosf + pav_local
                        addro = addro + pav_local * (1.0 - addro / pinc)

                adimc = adimc + pinc - addro
                roimp = roimp + addro * adimp

            adj = 1.0 - adj
            pav = 0.0

        flosf = flosf * (1.0 - pctim - adimp)
        floin = floin * (1.0 - pctim - adimp)
        flobf = flobf * (1.0 - pctim - adimp)

        lzfsc = alzfsc / (1.0 + side)
        lzfpc = alzfpc / (1.0 + side)

        uh_input = flosf + roimp + floin
        if n_uh == 0:
            flwsf = uh_input
        else:
            for i in range(n_uh):
                uh_stores[i] += uh_input * uh_scurve[i]
            flwsf = uh_stores[0]
            for i in range(n_uh - 1):
                uh_stores[i] = uh_stores[i + 1]
            uh_stores[n_uh - 1] = 0.0
        hydrograph_store += (uh_input - flwsf)

        flwbf = flobf / (1.0 + side)
        if flwbf < 0.0:
            flwbf = 0.0

        total_before_channel_losses = flwbf + flwsf
        ratio_baseflow = flwbf / total_before_channel_losses if total_before_channel_losses > 0.0 else 0.0

        channel_flow = flwbf + flwsf - ssout
        if channel_flow < 0.0:
            channel_flow = 0.0
        evaporation_channel_water = min(evapt * sarva, channel_flow)

        runoff = channel_flow - evaporation_channel_water
        baseflow = runoff * ratio_baseflow

        if runoff != runoff:
            runoff = 0.0
            baseflow = 0.0
            channel_flow = 0.0

        runoff_out[t] = runoff
        baseflow_out[t] = baseflow
        channel_flow_out[t] = channel_flow

    return (
        runoff_out, baseflow_out, channel_flow_out,
        uztwc, uzfwc, lztwc, lzfsc, lzfpc,
        alzfsc, alzfpc, adimc, hydrograph_store,
        uh_stores,
    )


# =========================================================================
# Sacramento batch-parallel kernel for calibration workloads
# =========================================================================

@njit(parallel=True, cache=True)
def _sacramento_batch_numba(
    precip, pet,
    param_matrix,
    uh_scurve_matrix,
    init_uh_stores_matrix,
):
    """Run N independent Sacramento evaluations in parallel.

    Args:
        precip: Precipitation array, shape (n_timesteps,).
        pet: PET array, shape (n_timesteps,).
        param_matrix: Parameter matrix, shape (n_evals, 24).
            Columns: uztwm, uzfwm, lztwm, lzfpm, lzfsm,
                     uzk, lzpk, lzsk, zperc, rexp, pctim,
                     adimp, pfree, rserv, side, ssout, sarva,
                     init_uztwc, init_uzfwc, init_lztwc,
                     init_lzfsc, init_lzfpc, init_adimc,
                     init_hydrograph_store.
        uh_scurve_matrix: UH S-curve arrays, shape (n_evals, n_uh).
        init_uh_stores_matrix: UH store arrays, shape (n_evals, n_uh).

    Returns:
        runoff_all: Runoff for each evaluation, shape (n_evals, n_timesteps).
    """
    n_evals = param_matrix.shape[0]
    n_timesteps = len(precip)
    runoff_all = np.empty((n_evals, n_timesteps))

    for i in prange(n_evals):
        p = param_matrix[i]
        result = _sacramento_run_numba(
            precip, pet,
            p[0], p[1], p[2], p[3], p[4],
            p[5], p[6], p[7], p[8], p[9], p[10],
            p[11], p[12], p[13], p[14], p[15], p[16],
            uh_scurve_matrix[i],
            p[17], p[18], p[19], p[20], p[21], p[22], p[23],
            init_uh_stores_matrix[i].copy(),
        )
        runoff_all[i] = result[0]

    return runoff_all
