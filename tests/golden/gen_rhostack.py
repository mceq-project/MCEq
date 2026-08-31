"""Golden section for the numpy EM ρ-stack kernels.

``solv_numpy_etd2_rho_stack`` and ``solv_numpy_etd2_rho_stack_multirhs``
interpolate the EM interaction matrix in air density: LPM suppression of
bremsstrahlung and pair production is density dependent, so the EM cascade
wants a different ``int_m`` high in the atmosphere than at sea level.

**This section is a behaviour pin, not a physics reference.** No EM database on
this machine carries an ``/electromagnetic/<medium>/rho_grid`` dataset, so
``MCEqRun.enable_em_density_interpolation()`` cannot build a stack from data and
the interpolation has never run on real per-density matrices. The three slices
here are synthetic — ``int_m`` with its EM rows scaled by a fixed ramp. A green
golden says the kernels still compute what they computed at the Phase-0 tree; it
says nothing about whether the LPM interpolation is correct.

The blend under test is log-linear in ρ_eff = 1/rho_inv[k] at every step k:

    int_m_eff = (1 - w) · stack[lo] + w · stack[lo+1],
    w = (log10 ρ_eff - log10 ρ_grid[lo]) / (log10 ρ_grid[lo+1] - log10 ρ_grid[lo]),

clamped to the stack endpoints. On the pinned path 105 of the 133 steps land
strictly inside a bracket, 26 clamp low and 2 clamp high, and both brackets of
the 3-slice grid are used — so a regression in either the weight or the slice
selection moves the answer.

Distinct slices are the point: a degenerate stack (identical slices) cannot
detect a blend-weight regression. It is exercised separately, as the self-check
under ``selfcheck/``.

The ETD2 EM block is only conditionally stable with e± in the system. At the
energy cut pinned here (0.1 .. 1e5 GeV) and zenith 60 deg the solve stays
finite, max|φ| = 1.2; a variant that widens the cut or raises the zenith has no
such guarantee, hence the finiteness assert.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import scipy.sparse as sp

from ._harness import HOST_RTOL, make_provenance, sparse_digest
from .make_goldens import SectionUnavailable

SECTION = "rhostack"

#: The EM databases carry a 140-bin energy grid and the reduced database 31;
#: combining them dies in scipy with "index pointer size 141 should be 32". The
#: ρ-stack therefore needs the full v140 database, trimmed by e_min/e_max.
HADRONIC_DB = "mceq_db_lext_dpm193_v140.h5"
EM_DB = "mceq_db_EM_Tsai_Max_v131.h5"

#: ``config.em_db_fname``'s package default names a file that is not shipped
#: with MCEq; the built EM databases live in the maintenance-tools checkout.
EM_DB_FALLBACK = (
    "/ceph/sharedfs/work/SATORI/anatoli/devel/mceq-maintenance-tools/dbfiles/"
    "mceq_db_EM_Tsai_Max_v131.h5"
)

ZENITH_DEG = 60.0

#: Slice densities (g/cm³) and the factor each slice applies to the EM rows of
#: ``int_m``. The ramp is what makes the slices distinct; it carries no physics.
RHO_GRID = (1e-5, 1e-4, 1.225e-3)
EM_ROW_SCALE = (1.00, 1.05, 1.10)

#: The multi-RHS kernel is linear in ``phi``, so proportional columns would test
#: nothing beyond broadcasting: column 1 carries a sqrt(E/GeV) spectral tilt.
MULTIRHS_TILT_EXPONENT = 0.5

#: Config globals the section fixes. ``numpy_bsr_blocksize`` is read inside the
#: kernel at call time (solvers.py:1651) and 11, 7 and None give three distinct
#: bit patterns for the same inputs. ``em_adaptive_step`` must stay False: True
#: caps dX_max through ``_em_cascade_dx_cap`` and the 133-step path becomes a
#: 1032-step one. ``muon_helicity_dependence`` is pinned because MCEqRun forces
#: it off under ``enable_em`` (core.py:341) and that write has to be undone.
CONFIG_PINS = {
    "debug_level": 0,
    "mceq_db_fname": HADRONIC_DB,
    "enable_em": True,
    "muon_helicity_dependence": True,
    "e_min": 1e-1,
    "e_max": 1e5,
    "kernel_config": "numpy_etd2",
    "numpy_bsr_blocksize": 11,
    "em_adaptive_step": False,
    "em_air_density": None,
    "density_model": ("CORSIKA", ("BK_USStd", None)),
    "X_start": 0.0,
    "etd2_path": {"eps": 0.3, "dX_max": 20.0, "dX_min": 0.01, "fd_span": 0.01},
    # The atmosphere is built from these, so an ambient h_obs moves the path and
    # with it every solution key.
    "h_obs": 0.0,
    "h_atm": 112.8,
    "r_E": 6371.315,
}

#: e± are disabled by default; the EM cascade is the whole point here.
ADV_SET_PINS = {"disabled_particles": []}


def build():
    """Produce (arrays, provenance) for the ρ-stack section."""
    from MCEq import config

    em_db = _locate_em_db(config.data_dir)
    saved_config = {k: getattr(config, k) for k in CONFIG_PINS}
    saved_config["em_db_fname"] = config.em_db_fname
    saved_adv_set = copy.deepcopy(config.adv_set)
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)
        config.em_db_fname = em_db
        config.adv_set.update(ADV_SET_PINS)
        arrays, provenance = _generate(em_db)
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)

    db = provenance["databases"].get("em_db_fname", {})
    assert "sha256" in db, f"EM DB not resolvable, provenance incomplete: {db}"

    return arrays, provenance


def _locate_em_db(data_dir):
    """Absolute path of an EM database carrying the full 140-bin energy grid."""
    for path in (os.path.join(data_dir, EM_DB), EM_DB_FALLBACK):
        if os.path.isfile(path):
            return path
    raise SectionUnavailable(
        f"EM database {EM_DB} is in neither {data_dir} nor {EM_DB_FALLBACK}"
    )


def _generate(em_db):
    import crflux.models as pm

    from MCEq import solvers
    from MCEq.core import MCEqRun

    mceq = MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )
    try:
        mceq.set_zenith_azimuth(ZENITH_DEG)
        # force=True: the path cache key holds neither the zenith nor the
        # density model, so a path cached earlier would be served instead.
        mceq._calculate_integration_path(None, "X", force=True)
        nsteps, dX, rho_inv, _ = mceq.integration_path
        grid_idcs = [0, nsteps // 3, 2 * nsteps // 3, nsteps - 1]

        int_m = mceq.int_m.tocsr()
        em_rows = np.zeros(mceq.dim_states, dtype=bool)
        for p in mceq.pman.all_particles:
            if p.is_em and p.mceqidx >= 0:
                em_rows[p.lidx : p.uidx] = True

        rho_grid = np.array(RHO_GRID)
        stack = [_scale_em_rows(int_m, em_rows, s) for s in EM_ROW_SCALE]
        # solvers._build_step_blend_indices has no caller in src — both kernels
        # inline the same searchsorted map — so these keys pin the helper, and
        # sol/* is what catches the kernels drifting away from it.
        lo_idx, weight = solvers._build_step_blend_indices(rho_inv, rho_grid)

        phi0 = mceq._phi0
        sol, grid_sol = solvers.solv_numpy_etd2_rho_stack(
            nsteps, dX, rho_inv, stack, rho_grid, mceq.dec_m, phi0.copy(), grid_idcs
        )

        tilt = np.tile(mceq._energy_grid.c, mceq.dim_states // mceq.dim)
        phi_multi = np.column_stack([phi0, phi0 * tilt**MULTIRHS_TILT_EXPONENT])
        sol_multi, grid_sol_multi = solvers.solv_numpy_etd2_rho_stack_multirhs(
            nsteps, dX, rho_inv, stack, rho_grid, mceq.dec_m, phi_multi, grid_idcs
        )

        # A degenerate stack collapses the blend to (1-w)·M@x + w·M@x, so the
        # ρ-stack kernel has to land on the plain kernel: measured rel-L2
        # 1.51e-16, four orders inside the 1e-12 host budget. Phase 2
        # reproduces this to show its CompiledOperator variant is still a
        # no-op on identical slices.
        sol_degenerate, _ = solvers.solv_numpy_etd2_rho_stack(
            nsteps, dX, rho_inv, [int_m] * 3, rho_grid, mceq.dec_m, phi0.copy(), []
        )
        sol_plain, _ = solvers.solv_numpy_etd2(
            nsteps, dX, rho_inv, int_m, mceq.dec_m, phi0.copy(), []
        )
        degenerate_gap = _rel_l2(sol_degenerate, sol_plain)
        # The multi-RHS kernel splits its slices into CSR where the single-RHS
        # one uses BSR, so column 0 sits one rounding step from the single run.
        multirhs_gap = _rel_l2(sol_multi[:, 0], sol)

        blocks = [p for p in mceq.pman.all_particles if p.mceqidx >= 0]
        arrays = {
            "path/nsteps": np.int64(nsteps),
            "path/dX": dX,
            "path/rho_inv": rho_inv,
            "snapshot/grid_idcs": np.asarray(grid_idcs, dtype=np.int64),
            "blend/lo_idx": lo_idx,
            "blend/weight": weight,
            "stack/rho_grid": rho_grid,
            "stack/em_row_scale": np.asarray(EM_ROW_SCALE),
            "stack/em_row_count": np.int64(em_rows.sum()),
            "stack/dim_states": np.int64(mceq.dim_states),
            "stack/slice_digest": np.array(
                [
                    [sparse_digest(m)[part] for part in ("data", "indices", "indptr")]
                    for m in stack
                ]
            ),
            "sol/single": sol,
            "sol/single_grid": grid_sol,
            "sol/multirhs": sol_multi,
            "sol/multirhs_grid": grid_sol_multi,
            "sol/degenerate": sol_degenerate,
            "sol/plain": sol_plain,
            "reduce/block_name": np.array([p.name for p in blocks]),
            "reduce/block_sum": np.array([sol[p.lidx : p.uidx].sum() for p in blocks]),
            "selfcheck/degenerate_vs_plain_rel_l2": degenerate_gap,
            "selfcheck/multirhs_vs_single_rel_l2": multirhs_gap,
        }

        assert np.all(np.isfinite(sol)), "the pinned ρ-stack solve must stay finite"

        provenance = make_provenance(
            SECTION,
            note=(
                "Behaviour pin for solv_numpy_etd2_rho_stack and its multi-RHS"
                " twin, NOT a physics reference. No EM database carries"
                " /electromagnetic/<medium>/rho_grid, so the LPM density"
                " interpolation has never run on per-density matrices from"
                " data; the three slices are int_m with its EM rows scaled by"
                f" {'/'.join(f'{s:.2f}' for s in EM_ROW_SCALE)}. Green pins the"
                " blend arithmetic and the kernel bookkeeping, nothing more."
            ),
            tolerances=_tolerances(degenerate_gap, multirhs_gap),
            extra={
                "interaction_model": "SIBYLL21",
                "primary_model": "HillasGaisser2012 H3a",
                "zenith_deg": ZENITH_DEG,
                "em_db_path": os.path.realpath(em_db),
                "stack_is_synthetic": True,
                "em_rows": int(em_rows.sum()),
                "steps_interior_weight": int(((weight > 0) & (weight < 1)).sum()),
                "steps_clamped_low": int((weight == 0).sum()),
                "steps_clamped_high": int((weight == 1).sum()),
                "degenerate_vs_plain_rel_l2": degenerate_gap,
                "multirhs_vs_single_rel_l2": multirhs_gap,
                "host_budget_rel_l2": HOST_RTOL,
            },
        )
    finally:
        mceq.close()

    return arrays, provenance


def _tolerances(degenerate_gap, multirhs_gap):
    """Per-key comparison modes; everything not named here is bitwise.

    The two ``selfcheck/`` keys hold float noise, which cannot be pinned
    bitwise across a legitimate reassociation. Their rtol is set so a
    regenerated gap passes exactly while it stays inside the host budget.
    """
    return {
        "selfcheck/degenerate_vs_plain_rel_l2": {
            "mode": "rel_l2",
            "rtol": _budget_rtol(degenerate_gap),
        },
        "selfcheck/multirhs_vs_single_rel_l2": {
            "mode": "rel_l2",
            "rtol": _budget_rtol(multirhs_gap),
        },
    }


def _scale_em_rows(int_m, em_rows, scale):
    """``int_m`` with its e±/γ rows multiplied by ``scale``, canonical CSR."""
    scaled = (sp.diags(np.where(em_rows, scale, 1.0)) @ int_m).tocsr()
    scaled.sort_indices()
    return scaled


def _rel_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def _budget_rtol(gap):
    """rtol that passes iff a regenerated ``gap`` stays inside the host budget.

    Clamped at zero: a measured gap already outside the budget would otherwise
    give a negative tolerance, which no comparison can satisfy.
    """
    if gap <= 0.0:
        return 0.0
    return max(0.0, HOST_RTOL / gap - 1.0)
