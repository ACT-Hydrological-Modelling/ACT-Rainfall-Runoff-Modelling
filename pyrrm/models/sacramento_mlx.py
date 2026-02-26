"""
Sacramento Rainfall-Runoff Model -- Apple MLX Implementation.

Port of the Sacramento kernel for benchmarking on Apple Silicon.
MLX lacks scan/while_loop primitives, so this uses a Python-level
timestep loop dispatching MLX scalar operations.  This demonstrates
the GPU-dispatch overhead penalty for sequential state-machine models.

This module is intended for **benchmarking only** -- the Numba kernel
is the recommended backend for production use.
"""

import math
import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

LENGTH_OF_UNIT_HYDROGRAPH = 5
_PDN20 = 5.08
_PDNOR = 25.4


def sacramento_run_mlx(
    precip_np, pet_np,
    uztwm, uzfwm, lztwm, lzfpm, lzfsm,
    uzk, lzpk, lzsk,
    zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
    uh_scurve_np,
    init_uztwc, init_uzfwc, init_lztwc, init_lzfsc, init_lzfpc,
    init_adimc, init_hydrograph_store,
    init_uh_stores_np,
):
    """Run Sacramento model using MLX arrays on Apple Silicon.

    Accepts NumPy arrays for inputs and returns NumPy arrays for outputs.
    Internally converts to MLX arrays for GPU-accelerated arithmetic,
    though the sequential loop structure limits parallelism.

    The signature matches ``_sacramento_run_numba`` for drop-in comparison.
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not installed. Install with: pip install mlx")

    n = len(precip_np)
    precip = mx.array(precip_np)
    pet = mx.array(pet_np)
    uh_scurve = mx.array(uh_scurve_np)
    n_uh = len(uh_scurve_np)

    runoff_list = []
    baseflow_list = []
    channel_flow_list = []

    alzfsm = lzfsm * (1.0 + side)
    alzfpm = lzfpm * (1.0 + side)

    uztwc = float(init_uztwc)
    uzfwc = float(init_uzfwc)
    lztwc = float(init_lztwc)
    lzfsc = float(init_lzfsc)
    lzfpc = float(init_lzfpc)
    adimc_v = float(init_adimc)
    hydrograph_store = float(init_hydrograph_store)

    alzfsc = lzfsc * (1.0 + side)
    alzfpc = lzfpc * (1.0 + side)
    pbase = alzfsm * lzsk + alzfpm * lzpk

    uh_stores = list(init_uh_stores_np.astype(float))

    # Sequential timestep loop -- the bottleneck for MLX.
    # Each iteration depends on the previous state, preventing vectorisation.
    for t in range(n):
        reserved_lower_zone = rserv * (lzfpm + lzfsm)
        evapt = float(pet_np[t])
        pliq = float(precip_np[t])

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
                             * (adimc_v - evap_uztw - uztwc)
                             / (uztwm + lztwm)),
                adimc_v,
            )
        else:
            e3 = 0.0
            e5 = 0.0

        lztwc = lztwc - e3
        adimc_v = adimc_v - e5
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
            adimc_v = adimc_v + pliq
            uztwc = uztwc + pliq
            pav = 0.0
        else:
            adimc_v = adimc_v + uztwm - uztwc
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
                ratio = (adimc_v - uztwc) / lztwm if lztwm > 0.0 else 0.0
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

                adimc_v = adimc_v + pinc - addro
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
                uh_stores[i] += uh_input * float(uh_scurve_np[i])
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

        runoff_list.append(runoff)
        baseflow_list.append(baseflow)
        channel_flow_list.append(channel_flow)

    runoff_out = np.array(runoff_list, dtype=np.float64)
    baseflow_out = np.array(baseflow_list, dtype=np.float64)
    channel_flow_out = np.array(channel_flow_list, dtype=np.float64)

    return (
        runoff_out, baseflow_out, channel_flow_out,
        uztwc, uzfwc, lztwc, lzfsc, lzfpc,
        alzfsc, alzfpc, adimc_v, hydrograph_store,
        np.array(uh_stores, dtype=np.float64),
    )
