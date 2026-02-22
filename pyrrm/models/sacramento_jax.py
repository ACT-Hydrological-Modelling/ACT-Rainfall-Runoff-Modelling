"""
Sacramento Rainfall-Runoff Model -- JAX Implementation.

Line-for-line port of Sacramento._run_time_step() from
pyrrm.models.sacramento for use with JAX automatic differentiation
and NumPyro NUTS sampling.

Adaptations for JAX compatibility:
- Variable inner loop count (ninc) fixed to MAX_NINC with masking
- Two-pass outer loop (itime) unrolled with branch masking
- All if/else replaced with jnp.where / jnp.minimum / jnp.maximum
- Unit hydrograph uses fixed-size arrays

Reference:
    Burnash, R.J.C., Ferral, R.L. & McGuire, R.A. (1973).
    A generalized streamflow simulation system.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

jax.config.update("jax_enable_x64", True)

LENGTH_OF_UNIT_HYDROGRAPH = 5
PDN20 = 5.08
PDNOR = 25.4
MAX_NINC = 20
_EPS = 1e-10


def sacramento_run_jax(
    params: dict,
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    initial_states: dict = None,
    max_ninc: int = MAX_NINC,
    fast_mode: bool = False,
) -> dict:
    """
    Run Sacramento model using JAX (JIT-able, differentiable).

    Args:
        params: Dict with keys ``uztwm``, ``uzfwm``, ``lztwm``,
            ``lzfpm``, ``lzfsm``, ``uzk``, ``lzpk``, ``lzsk``,
            ``zperc``, ``rexp``, ``pctim``, ``adimp``, ``pfree``,
            ``rserv``, ``side``, ``ssout``, ``sarva``,
            ``uh1``..``uh5``.
        precip: Daily precipitation [mm], shape ``(n_timesteps,)``.
        pet: Daily PET [mm], shape ``(n_timesteps,)``.
        initial_states: Optional dict with initial store levels.
        max_ninc: Maximum sub-daily inner loop iterations.  Default
            ``MAX_NINC`` (20) matches the NumPy reference.  Use a
            smaller value (e.g. 5) during calibration for faster
            gradient evaluation; ``ninc`` rarely exceeds 3 in practice.
            Ignored when ``fast_mode=True``.
        fast_mode: If True, bypass the inner sub-daily ``lax.scan``
            entirely by forcing ``ninc=1`` and using daily drainage
            rates directly.  This eliminates the nested scan,
            reducing the XLA graph by ~10-16× and dramatically
            speeding up JIT compilation and NUTS gradient evaluation.
            Recommended for calibration; validate final posterior with
            ``fast_mode=False``.

    Returns:
        Dict with ``"simulated_flow"``, ``"baseflow"``,
        ``"surface_runoff"`` arrays.
    """
    uztwm = params["uztwm"]
    uzfwm = params["uzfwm"]
    lztwm = params["lztwm"]
    lzfpm = params["lzfpm"]
    lzfsm = params["lzfsm"]
    uzk = params["uzk"]
    lzpk = params["lzpk"]
    lzsk = params["lzsk"]
    zperc = params["zperc"]
    rexp = params["rexp"]
    pctim = params["pctim"]
    adimp = params["adimp"]
    pfree = params["pfree"]
    rserv = params["rserv"]
    side = params["side"]
    ssout = params["ssout"]
    sarva = params["sarva"]

    raw_uh = jnp.array([
        params.get("uh1", 1.0), params.get("uh2", 0.0),
        params.get("uh3", 0.0), params.get("uh4", 0.0),
        params.get("uh5", 0.0),
    ])
    uh_total = jnp.sum(raw_uh)
    uh_default = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    uh_proportions = jnp.where(uh_total > _EPS, raw_uh / (uh_total + _EPS), uh_default)

    alzfsm = lzfsm * (1.0 + side)
    alzfpm = lzfpm * (1.0 + side)

    if initial_states is None:
        uztwc0 = 0.0
        uzfwc0 = 0.0
        lztwc0 = 0.0
        lzfsc0 = 0.0
        lzfpc0 = 0.0
        adimc0 = 0.0
    else:
        uztwc0 = initial_states.get("uztwc", 0.0)
        uzfwc0 = initial_states.get("uzfwc", 0.0)
        lztwc0 = initial_states.get("lztwc", 0.0)
        lzfsc0 = initial_states.get("lzfsc", 0.0) * (1.0 + side)
        lzfpc0 = initial_states.get("lzfpc", 0.0) * (1.0 + side)
        adimc0 = initial_states.get("adimc", 0.0)

    uh_stores0 = jnp.zeros(LENGTH_OF_UNIT_HYDROGRAPH)

    static_params = (
        uztwm, uzfwm, lztwm, lzfpm, lzfsm, uzk, lzpk, lzsk,
        zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
        alzfsm, alzfpm, uh_proportions, max_ninc,
    )

    carry0 = (uztwc0, uzfwc0, lztwc0, lzfsc0, lzfpc0, adimc0, uh_stores0)
    inputs_stacked = jnp.stack([precip, pet], axis=-1)

    if fast_mode:
        step_fn = partial(_sacramento_step_fast, static_params=static_params)
    else:
        step_fn = partial(_sacramento_step, static_params=static_params)
    _, outputs = lax.scan(step_fn, carry0, inputs_stacked)

    runoff, baseflow, surface_flow = outputs
    return {
        "simulated_flow": runoff,
        "baseflow": baseflow,
        "surface_runoff": surface_flow,
    }


def _sacramento_step(carry, input_t, static_params):
    """Single Sacramento timestep -- mirrors _run_time_step()."""
    (uztwm, uzfwm, lztwm, lzfpm, lzfsm, uzk, lzpk, lzsk,
     zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
     alzfsm, alzfpm, uh_proportions, max_ninc) = static_params

    uztwc, uzfwc, lztwc, alzfsc, alzfpc, adimc, uh_stores = carry
    rainfall = input_t[0]
    evapt = input_t[1]

    reserved_lower_zone = rserv * (lzfpm + lzfsm)
    pbase = alzfsm * lzsk + alzfpm * lzpk

    # --- Evaporation from upper zone tension water ---
    evap_uztw = jnp.where(uztwm > 0.0, evapt * uztwc / (uztwm + _EPS), 0.0)

    excess_evap = evap_uztw > uztwc
    evap_uztw_a = uztwc
    uztwc_a = 0.0
    evap_uzfw_a = jnp.minimum(evapt - evap_uztw_a, uzfwc)
    uzfwc_a = uzfwc - evap_uzfw_a

    evap_uztw_b = evap_uztw
    uztwc_b = uztwc - evap_uztw_b
    evap_uzfw_b = 0.0
    uzfwc_b = uzfwc

    evap_uztw = jnp.where(excess_evap, evap_uztw_a, evap_uztw_b)
    uztwc = jnp.where(excess_evap, uztwc_a, uztwc_b)
    evap_uzfw = jnp.where(excess_evap, evap_uzfw_a, evap_uzfw_b)
    uzfwc = jnp.where(excess_evap, uzfwc_a, uzfwc_b)

    # --- Transfer free water to tension water ---
    ratio_uztw = jnp.where(uztwm > 0.0, uztwc / (uztwm + _EPS), 1.0)
    ratio_uzfw = jnp.where(uzfwm > 0.0, uzfwc / (uzfwm + _EPS), 1.0)

    do_transfer = ratio_uztw < ratio_uzfw
    combined_ratio = (uztwc + uzfwc) / (uztwm + uzfwm + _EPS)
    uztwc_t = jnp.where(do_transfer, uztwm * combined_ratio, uztwc)
    uzfwc_t = jnp.where(do_transfer, uzfwm * combined_ratio, uzfwc)
    uztwc = uztwc_t
    uzfwc = uzfwc_t

    # --- Evaporation from lower zone ---
    denom_lz = uztwm + lztwm + _EPS
    e3 = jnp.minimum(
        (evapt - evap_uztw - evap_uzfw) * lztwc / denom_lz,
        lztwc,
    )
    e5 = jnp.minimum(
        evap_uztw + (evapt - evap_uztw - evap_uzfw) * (adimc - evap_uztw - uztwc) / denom_lz,
        adimc,
    )

    lztwc = lztwc - e3
    adimc = adimc - e5
    evap_uztw = evap_uztw * (1.0 - adimp - pctim)
    evap_uzfw = evap_uzfw * (1.0 - adimp - pctim)
    e3 = e3 * (1.0 - adimp - pctim)
    e5 = e5 * adimp

    # --- Lower zone tension water resupply ---
    ratio_lztw = jnp.where(lztwm > 0.0, lztwc / (lztwm + _EPS), 1.0)
    denom_lzfw = alzfpm + alzfsm - reserved_lower_zone + lztwm + _EPS
    ratio_lzfw = (alzfpc + alzfsc - reserved_lower_zone + lztwc) / denom_lzfw

    do_lz_transfer = ratio_lztw < ratio_lzfw
    transferred = (ratio_lzfw - ratio_lztw) * lztwm
    lztwc = jnp.where(do_lz_transfer, lztwc + transferred, lztwc)
    alzfsc_adj = jnp.where(do_lz_transfer, alzfsc - transferred, alzfsc)
    overflow = jnp.minimum(alzfsc_adj, 0.0)
    alzfpc = jnp.where(do_lz_transfer, alzfpc + overflow, alzfpc)
    alzfsc = jnp.where(do_lz_transfer, jnp.maximum(alzfsc_adj, 0.0), alzfsc)

    # --- Runoff from impervious area ---
    roimp = rainfall * pctim

    # --- Upper zone processing ---
    pav = rainfall + uztwc - uztwm
    pav_neg = pav < 0
    adimc = jnp.where(pav_neg, adimc + rainfall, adimc + uztwm - uztwc)
    uztwc = jnp.where(pav_neg, uztwc + rainfall, uztwm)
    pav = jnp.maximum(pav, 0.0)

    # --- Determine adj and itime ---
    is_small_pav = pav <= PDN20
    adj_small = 1.0
    adj_large = jnp.where(
        pav < PDNOR,
        0.5 * jnp.sqrt(jnp.maximum(pav, _EPS) / (PDNOR + _EPS)),
        1.0 - 0.5 * PDNOR / (pav + _EPS),
    )
    adj_init = jnp.where(is_small_pav, adj_small, adj_large)
    run_pass1 = ~is_small_pav

    flobf = 0.0
    flosf = 0.0
    floin = 0.0

    hpl = jnp.where(
        alzfpm + alzfsm > 0,
        alzfpm / (alzfpm + alzfsm + _EPS),
        0.5,
    )

    # --- Pass 1 (only active when itime == 1, i.e. pav > PDN20) ---
    (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin) = _outer_pass(
        pav, adj_init, run_pass1,
        uzfwc, uztwc, uzfwm, lztwc, lztwm,
        alzfsc, alzfsm, alzfpc, alzfpm,
        adimc, adimp, roimp, pbase,
        uzk, lzpk, lzsk, zperc, rexp, pfree,
        hpl, flobf, flosf, floin, max_ninc,
    )

    # --- Pass 2 (always active) ---
    # When pav <= PDN20 (itime==2): this is the ONLY active pass, adj=1.0, pav=pav
    # When pav > PDN20  (itime==1): this is the second pass, adj=1-adj_init, pav=0
    adj2 = jnp.where(is_small_pav, 1.0, 1.0 - adj_init)
    pav2 = jnp.where(is_small_pav, pav, 0.0)
    (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin) = _outer_pass(
        pav2, adj2, jnp.bool_(True),
        uzfwc, uztwc, uzfwm, lztwc, lztwm,
        alzfsc, alzfsm, alzfpc, alzfpm,
        adimc, adimp, roimp, pbase,
        uzk, lzpk, lzsk, zperc, rexp, pfree,
        hpl, flobf, flosf, floin, max_ninc,
    )

    # --- Final computations ---
    area_factor = 1.0 - pctim - adimp
    flosf = flosf * area_factor
    floin = floin * area_factor
    flobf = flobf * area_factor

    lzfsc = alzfsc / (1.0 + side)
    lzfpc = alzfpc / (1.0 + side)

    # Unit hydrograph routing
    uh_input = flosf + roimp + floin
    uh_stores = uh_stores + uh_input * uh_proportions
    flwsf = uh_stores[0]
    uh_stores = jnp.roll(uh_stores, -1).at[-1].set(0.0)

    flwbf = jnp.maximum(0.0, flobf / (1.0 + side))

    total_before_losses = flwbf + flwsf + _EPS
    ratio_baseflow = flwbf / total_before_losses

    channel_flow = jnp.maximum(0.0, flwbf + flwsf - ssout)
    evap_channel = jnp.minimum(evapt * sarva, channel_flow)

    runoff = channel_flow - evap_channel
    baseflow = runoff * ratio_baseflow

    new_carry = (uztwc, uzfwc, lztwc, alzfsc, alzfpc, adimc, uh_stores)
    return new_carry, (runoff, baseflow, flosf)


def _outer_pass(
    pav, adj, is_active,
    uzfwc, uztwc, uzfwm, lztwc, lztwm,
    alzfsc, alzfsm, alzfpc, alzfpm,
    adimc, adimp, roimp, pbase,
    uzk, lzpk, lzsk, zperc, rexp, pfree,
    hpl, flobf, flosf, floin, max_ninc,
):
    """One pass of the outer loop (itime). Masked by *is_active*."""
    ninc_raw = jnp.floor((uzfwc * adj + pav) * 0.2).astype(jnp.int32) + 1
    ninc = jnp.clip(ninc_raw, 1, max_ninc)
    dinc = 1.0 / ninc
    pinc = pav * dinc
    dinc_adj = dinc * adj

    simple = (ninc == 1) & (adj >= 1.0)
    duz = jnp.where(simple, uzk, 1.0 - jnp.power(jnp.maximum(1.0 - uzk, _EPS), dinc_adj))
    dlzp = jnp.where(simple, lzpk, 1.0 - jnp.power(jnp.maximum(1.0 - lzpk, _EPS), dinc_adj))
    dlzs = jnp.where(simple, lzsk, 1.0 - jnp.power(jnp.maximum(1.0 - lzsk, _EPS), dinc_adj))

    init_inner = (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin)

    def inner_step(carry_inner, step_idx):
        (uzfwc_i, lztwc_i, alzfsc_i, alzfpc_i, adimc_i,
         roimp_i, flobf_i, flosf_i, floin_i) = carry_inner

        active = step_idx < ninc

        ratio = jnp.where(lztwm > 0, (adimc_i - uztwc) / (lztwm + _EPS), 0.0)
        addro = pinc * ratio * ratio

        # Baseflow from lower zone primary
        bf_p = jnp.where(alzfpc_i > 0.0, alzfpc_i * dlzp, 0.0)
        flobf_n = flobf_i + bf_p
        alzfpc_n = jnp.maximum(alzfpc_i - bf_p, 0.0)

        # Baseflow from lower zone supplemental
        bf_s = jnp.where(alzfsc_i > 0.0, alzfsc_i * dlzs, 0.0)
        alzfsc_n = jnp.maximum(alzfsc_i - bf_s, 0.0)
        flobf_n = flobf_n + bf_s

        # Percolation and interflow
        lzair = (lztwm - lztwc_i + alzfsm - alzfsc_n + alzfpm - alzfpc_n)
        perc_base = jnp.where(
            uzfwm > 0, pbase * dinc_adj * uzfwc_i / (uzfwm + _EPS), 0.0
        )
        total_lz = alzfpm + alzfsm + lztwm + _EPS
        current_lz = alzfpc_n + alzfsc_n + lztwc_i
        deficit_ratio = jnp.maximum(1.0 - current_lz / total_lz, 0.0)
        perc_full = jnp.minimum(
            uzfwc_i,
            perc_base * (1.0 + zperc * jnp.power(deficit_ratio + _EPS, rexp)),
        )
        perc = jnp.minimum(jnp.maximum(lzair, 0.0), perc_full)
        perc = jnp.where((uzfwc_i > 0.0) & (lzair > 0.0), perc, 0.0)
        uzfwc_n = uzfwc_i - perc

        # Interflow
        transfered = duz * uzfwc_n
        floin_n = floin_i + jnp.where(uzfwc_i > 0.0, transfered, 0.0)
        uzfwc_n = jnp.where(uzfwc_i > 0.0, uzfwc_n - transfered, uzfwc_n)

        # Distribute percolation
        perctw = jnp.minimum(perc * (1.0 - pfree), lztwm - lztwc_i)
        percfw = perc - perctw
        lzair_fw = alzfsm - alzfsc_n + alzfpm - alzfpc_n
        overflow_perc = percfw > lzair_fw
        perctw = jnp.where(overflow_perc, perctw + percfw - lzair_fw, perctw)
        percfw = jnp.where(overflow_perc, lzair_fw, percfw)
        lztwc_n = lztwc_i + perctw

        ratlp = jnp.where(alzfpm > 0, 1.0 - alzfpc_n / (alzfpm + _EPS), 0.0)
        ratls = jnp.where(alzfsm > 0, 1.0 - alzfsc_n / (alzfsm + _EPS), 0.0)
        sum_rat = ratlp + ratls + _EPS
        percs = jnp.where(
            (percfw > 0.0) & (sum_rat > _EPS),
            jnp.minimum(
                alzfsm - alzfsc_n,
                percfw * (1.0 - hpl * (2.0 * ratlp) / sum_rat),
            ),
            0.0,
        )
        alzfsc_n2 = alzfsc_n + percs
        over_s = alzfsc_n2 > alzfsm
        percs = jnp.where(over_s, percs - alzfsc_n2 + alzfsm, percs)
        alzfsc_n2 = jnp.minimum(alzfsc_n2, alzfsm)
        alzfpc_n2 = alzfpc_n + percfw - percs
        over_p = alzfpc_n2 > alzfpm
        alzfsc_n2 = jnp.where(over_p, alzfsc_n2 + alzfpc_n2 - alzfpm, alzfsc_n2)
        alzfpc_n2 = jnp.minimum(alzfpc_n2, alzfpm)

        # Use percfw only when uzfwc_i > 0
        alzfsc_n_final = jnp.where(uzfwc_i > 0.0, alzfsc_n2, alzfsc_n)
        alzfpc_n_final = jnp.where(uzfwc_i > 0.0, alzfpc_n2, alzfpc_n)
        lztwc_n = jnp.where(uzfwc_i > 0.0, lztwc_n, lztwc_i)

        # Fill upper zone free water
        has_pinc = pinc > 0.0
        room = uzfwm - uzfwc_n
        can_absorb = pinc <= room
        uzfwc_fill = jnp.where(can_absorb, uzfwc_n + pinc, uzfwm)
        pav_excess = jnp.where(can_absorb, 0.0, pinc - room)
        flosf_n = flosf_i + jnp.where(has_pinc, pav_excess, 0.0)
        addro = jnp.where(
            has_pinc & (~can_absorb),
            addro + pav_excess * (1.0 - addro / (pinc + _EPS)),
            addro,
        )
        uzfwc_n = jnp.where(has_pinc, uzfwc_fill, uzfwc_n)

        adimc_n = adimc_i + pinc - addro
        roimp_n = roimp_i + addro * adimp

        new_inner = (uzfwc_n, lztwc_n, alzfsc_n_final, alzfpc_n_final,
                     adimc_n, roimp_n, flobf_n, flosf_n, floin_n)
        masked = jax.tree.map(
            lambda n, o: jnp.where(active, n, o), new_inner, carry_inner
        )
        return masked, None

    steps = jnp.arange(max_ninc)
    result_inner, _ = lax.scan(inner_step, init_inner, steps)

    final = jax.tree.map(
        lambda n, o: jnp.where(is_active, n, o), result_inner, init_inner
    )
    return final


# =====================================================================
# Fast-mode implementation (ninc=1, no inner lax.scan)
# =====================================================================


def _sacramento_step_fast(carry, input_t, static_params):
    """Single Sacramento timestep — fast mode (ninc=1, no inner scan).

    Identical to ``_sacramento_step`` except that the two
    ``_outer_pass`` calls are replaced by ``_outer_pass_fast`` which
    evaluates a single sub-daily increment using daily drainage rates,
    eliminating the nested ``lax.scan`` entirely.
    """
    (uztwm, uzfwm, lztwm, lzfpm, lzfsm, uzk, lzpk, lzsk,
     zperc, rexp, pctim, adimp, pfree, rserv, side, ssout, sarva,
     alzfsm, alzfpm, uh_proportions, _max_ninc) = static_params

    uztwc, uzfwc, lztwc, alzfsc, alzfpc, adimc, uh_stores = carry
    rainfall = input_t[0]
    evapt = input_t[1]

    reserved_lower_zone = rserv * (lzfpm + lzfsm)
    pbase = alzfsm * lzsk + alzfpm * lzpk

    # --- Evaporation from upper zone tension water ---
    evap_uztw = jnp.where(uztwm > 0.0, evapt * uztwc / (uztwm + _EPS), 0.0)

    excess_evap = evap_uztw > uztwc
    evap_uztw_a = uztwc
    uztwc_a = 0.0
    evap_uzfw_a = jnp.minimum(evapt - evap_uztw_a, uzfwc)
    uzfwc_a = uzfwc - evap_uzfw_a

    evap_uztw_b = evap_uztw
    uztwc_b = uztwc - evap_uztw_b
    evap_uzfw_b = 0.0
    uzfwc_b = uzfwc

    evap_uztw = jnp.where(excess_evap, evap_uztw_a, evap_uztw_b)
    uztwc = jnp.where(excess_evap, uztwc_a, uztwc_b)
    evap_uzfw = jnp.where(excess_evap, evap_uzfw_a, evap_uzfw_b)
    uzfwc = jnp.where(excess_evap, uzfwc_a, uzfwc_b)

    # --- Transfer free water to tension water ---
    ratio_uztw = jnp.where(uztwm > 0.0, uztwc / (uztwm + _EPS), 1.0)
    ratio_uzfw = jnp.where(uzfwm > 0.0, uzfwc / (uzfwm + _EPS), 1.0)

    do_transfer = ratio_uztw < ratio_uzfw
    combined_ratio = (uztwc + uzfwc) / (uztwm + uzfwm + _EPS)
    uztwc_t = jnp.where(do_transfer, uztwm * combined_ratio, uztwc)
    uzfwc_t = jnp.where(do_transfer, uzfwm * combined_ratio, uzfwc)
    uztwc = uztwc_t
    uzfwc = uzfwc_t

    # --- Evaporation from lower zone ---
    denom_lz = uztwm + lztwm + _EPS
    e3 = jnp.minimum(
        (evapt - evap_uztw - evap_uzfw) * lztwc / denom_lz,
        lztwc,
    )
    e5 = jnp.minimum(
        evap_uztw + (evapt - evap_uztw - evap_uzfw) * (adimc - evap_uztw - uztwc) / denom_lz,
        adimc,
    )

    lztwc = lztwc - e3
    adimc = adimc - e5
    evap_uztw = evap_uztw * (1.0 - adimp - pctim)
    evap_uzfw = evap_uzfw * (1.0 - adimp - pctim)
    e3 = e3 * (1.0 - adimp - pctim)
    e5 = e5 * adimp

    # --- Lower zone tension water resupply ---
    ratio_lztw = jnp.where(lztwm > 0.0, lztwc / (lztwm + _EPS), 1.0)
    denom_lzfw = alzfpm + alzfsm - reserved_lower_zone + lztwm + _EPS
    ratio_lzfw = (alzfpc + alzfsc - reserved_lower_zone + lztwc) / denom_lzfw

    do_lz_transfer = ratio_lztw < ratio_lzfw
    transferred = (ratio_lzfw - ratio_lztw) * lztwm
    lztwc = jnp.where(do_lz_transfer, lztwc + transferred, lztwc)
    alzfsc_adj = jnp.where(do_lz_transfer, alzfsc - transferred, alzfsc)
    overflow = jnp.minimum(alzfsc_adj, 0.0)
    alzfpc = jnp.where(do_lz_transfer, alzfpc + overflow, alzfpc)
    alzfsc = jnp.where(do_lz_transfer, jnp.maximum(alzfsc_adj, 0.0), alzfsc)

    # --- Runoff from impervious area ---
    roimp = rainfall * pctim

    # --- Upper zone processing ---
    pav = rainfall + uztwc - uztwm
    pav_neg = pav < 0
    adimc = jnp.where(pav_neg, adimc + rainfall, adimc + uztwm - uztwc)
    uztwc = jnp.where(pav_neg, uztwc + rainfall, uztwm)
    pav = jnp.maximum(pav, 0.0)

    # --- Determine adj and itime ---
    is_small_pav = pav <= PDN20
    adj_large = jnp.where(
        pav < PDNOR,
        0.5 * jnp.sqrt(jnp.maximum(pav, _EPS) / (PDNOR + _EPS)),
        1.0 - 0.5 * PDNOR / (pav + _EPS),
    )
    adj_init = jnp.where(is_small_pav, 1.0, adj_large)
    run_pass1 = ~is_small_pav

    flobf = 0.0
    flosf = 0.0
    floin = 0.0

    hpl = jnp.where(
        alzfpm + alzfsm > 0,
        alzfpm / (alzfpm + alzfsm + _EPS),
        0.5,
    )

    # --- Pass 1 (only active when pav > PDN20) ---
    (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin
     ) = _outer_pass_fast(
        pav, adj_init, run_pass1,
        uzfwc, uztwc, uzfwm, lztwc, lztwm,
        alzfsc, alzfsm, alzfpc, alzfpm,
        adimc, adimp, roimp, pbase,
        uzk, lzpk, lzsk, zperc, rexp, pfree,
        hpl, flobf, flosf, floin,
    )

    # --- Pass 2 (always active) ---
    adj2 = jnp.where(is_small_pav, 1.0, 1.0 - adj_init)
    pav2 = jnp.where(is_small_pav, pav, 0.0)
    (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin
     ) = _outer_pass_fast(
        pav2, adj2, jnp.bool_(True),
        uzfwc, uztwc, uzfwm, lztwc, lztwm,
        alzfsc, alzfsm, alzfpc, alzfpm,
        adimc, adimp, roimp, pbase,
        uzk, lzpk, lzsk, zperc, rexp, pfree,
        hpl, flobf, flosf, floin,
    )

    # --- Final computations ---
    area_factor = 1.0 - pctim - adimp
    flosf = flosf * area_factor
    floin = floin * area_factor
    flobf = flobf * area_factor

    lzfsc = alzfsc / (1.0 + side)
    lzfpc = alzfpc / (1.0 + side)

    uh_input = flosf + roimp + floin
    uh_stores = uh_stores + uh_input * uh_proportions
    flwsf = uh_stores[0]
    uh_stores = jnp.roll(uh_stores, -1).at[-1].set(0.0)

    flwbf = jnp.maximum(0.0, flobf / (1.0 + side))

    total_before_losses = flwbf + flwsf + _EPS
    ratio_baseflow = flwbf / total_before_losses

    channel_flow = jnp.maximum(0.0, flwbf + flwsf - ssout)
    evap_channel = jnp.minimum(evapt * sarva, channel_flow)

    runoff = channel_flow - evap_channel
    baseflow = runoff * ratio_baseflow

    new_carry = (uztwc, uzfwc, lztwc, alzfsc, alzfpc, adimc, uh_stores)
    return new_carry, (runoff, baseflow, flosf)


def _outer_pass_fast(
    pav, adj, is_active,
    uzfwc, uztwc, uzfwm, lztwc, lztwm,
    alzfsc, alzfsm, alzfpc, alzfpm,
    adimc, adimp, roimp, pbase,
    uzk, lzpk, lzsk, zperc, rexp, pfree,
    hpl, flobf, flosf, floin,
):
    """One outer-loop pass with ninc=1 — no inner scan.

    Uses daily drainage rates directly (``duz=uzk``, ``dlzp=lzpk``,
    ``dlzs=lzsk``) when ``adj >= 1``; otherwise applies the fractional
    adjustment ``1 - (1-k)^adj``.  Precipitation is applied in a single
    lump (``pinc=pav``).  This avoids the nested ``lax.scan`` that
    dominates compilation and gradient cost.
    """
    pinc = pav

    simple = adj >= 1.0
    duz = jnp.where(simple, uzk, 1.0 - jnp.power(jnp.maximum(1.0 - uzk, _EPS), adj))
    dlzp = jnp.where(simple, lzpk, 1.0 - jnp.power(jnp.maximum(1.0 - lzpk, _EPS), adj))
    dlzs = jnp.where(simple, lzsk, 1.0 - jnp.power(jnp.maximum(1.0 - lzsk, _EPS), adj))

    # --- Single increment (the body of inner_step, inlined) ---
    ratio = jnp.where(lztwm > 0, (adimc - uztwc) / (lztwm + _EPS), 0.0)
    addro = pinc * ratio * ratio

    bf_p = jnp.where(alzfpc > 0.0, alzfpc * dlzp, 0.0)
    flobf_n = flobf + bf_p
    alzfpc_n = jnp.maximum(alzfpc - bf_p, 0.0)

    bf_s = jnp.where(alzfsc > 0.0, alzfsc * dlzs, 0.0)
    alzfsc_n = jnp.maximum(alzfsc - bf_s, 0.0)
    flobf_n = flobf_n + bf_s

    lzair = (lztwm - lztwc + alzfsm - alzfsc_n + alzfpm - alzfpc_n)
    perc_base = jnp.where(
        uzfwm > 0, pbase * adj * uzfwc / (uzfwm + _EPS), 0.0
    )
    total_lz = alzfpm + alzfsm + lztwm + _EPS
    current_lz = alzfpc_n + alzfsc_n + lztwc
    deficit_ratio = jnp.maximum(1.0 - current_lz / total_lz, 0.0)
    perc_full = jnp.minimum(
        uzfwc,
        perc_base * (1.0 + zperc * jnp.power(deficit_ratio + _EPS, rexp)),
    )
    perc = jnp.minimum(jnp.maximum(lzair, 0.0), perc_full)
    perc = jnp.where((uzfwc > 0.0) & (lzair > 0.0), perc, 0.0)
    uzfwc_n = uzfwc - perc

    transfered = duz * uzfwc_n
    floin_n = floin + jnp.where(uzfwc > 0.0, transfered, 0.0)
    uzfwc_n = jnp.where(uzfwc > 0.0, uzfwc_n - transfered, uzfwc_n)

    perctw = jnp.minimum(perc * (1.0 - pfree), lztwm - lztwc)
    percfw = perc - perctw
    lzair_fw = alzfsm - alzfsc_n + alzfpm - alzfpc_n
    overflow_perc = percfw > lzair_fw
    perctw = jnp.where(overflow_perc, perctw + percfw - lzair_fw, perctw)
    percfw = jnp.where(overflow_perc, lzair_fw, percfw)
    lztwc_n = lztwc + perctw

    ratlp = jnp.where(alzfpm > 0, 1.0 - alzfpc_n / (alzfpm + _EPS), 0.0)
    ratls = jnp.where(alzfsm > 0, 1.0 - alzfsc_n / (alzfsm + _EPS), 0.0)
    sum_rat = ratlp + ratls + _EPS
    percs = jnp.where(
        (percfw > 0.0) & (sum_rat > _EPS),
        jnp.minimum(
            alzfsm - alzfsc_n,
            percfw * (1.0 - hpl * (2.0 * ratlp) / sum_rat),
        ),
        0.0,
    )
    alzfsc_n2 = alzfsc_n + percs
    over_s = alzfsc_n2 > alzfsm
    percs = jnp.where(over_s, percs - alzfsc_n2 + alzfsm, percs)
    alzfsc_n2 = jnp.minimum(alzfsc_n2, alzfsm)
    alzfpc_n2 = alzfpc_n + percfw - percs
    over_p = alzfpc_n2 > alzfpm
    alzfsc_n2 = jnp.where(over_p, alzfsc_n2 + alzfpc_n2 - alzfpm, alzfsc_n2)
    alzfpc_n2 = jnp.minimum(alzfpc_n2, alzfpm)

    alzfsc_n_final = jnp.where(uzfwc > 0.0, alzfsc_n2, alzfsc_n)
    alzfpc_n_final = jnp.where(uzfwc > 0.0, alzfpc_n2, alzfpc_n)
    lztwc_n = jnp.where(uzfwc > 0.0, lztwc_n, lztwc)

    has_pinc = pinc > 0.0
    room = uzfwm - uzfwc_n
    can_absorb = pinc <= room
    uzfwc_fill = jnp.where(can_absorb, uzfwc_n + pinc, uzfwm)
    pav_excess = jnp.where(can_absorb, 0.0, pinc - room)
    flosf_n = flosf + jnp.where(has_pinc, pav_excess, 0.0)
    addro = jnp.where(
        has_pinc & (~can_absorb),
        addro + pav_excess * (1.0 - addro / (pinc + _EPS)),
        addro,
    )
    uzfwc_n = jnp.where(has_pinc, uzfwc_fill, uzfwc_n)

    adimc_n = adimc + pinc - addro
    roimp_n = roimp + addro * adimp

    result = (uzfwc_n, lztwc_n, alzfsc_n_final, alzfpc_n_final,
              adimc_n, roimp_n, flobf_n, flosf_n, floin_n)
    init = (uzfwc, lztwc, alzfsc, alzfpc, adimc, roimp, flobf, flosf, floin)

    final = jax.tree.map(
        lambda n, o: jnp.where(is_active, n, o), result, init
    )
    return final
