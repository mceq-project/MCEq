"""Golden section ``emphoton``: the EM cascade on the PPM production database.

The first golden that executes the EM cascade at all — photon primary on the
PPM production EM database, solved on the standalone EM grid — closing the
refactor-plan gap "EM transport is never executed by any golden"
(``solve1d``/``species``/``operators*`` carry e+/e-/gamma through the system
but none of them transports an EM shower). One solve: a 1 PeV photon at
zenith 60 deg on a four-depth grid. Measured on this machine (2026-09-06):
grid d = 150, 134 integration steps, wall ~0.2 s, finite throughout.

Host **and** slow. Host: three of the six stable keys are sha256 digests and
therefore bitwise by construction, and the state key is compared against a
host golden exactly like ``solve1d``'s states. Slow: neither database is in
the repo or published on the releases page ``config.ensure_db_available``
downloads from; both are absolute paths into the ``mceq-em-integration`` runs
directory on the shared filesystem — the shower-library convention that the
payloads live outside the checkout, not gitignored symlinks into it. A build
machine without either file skips rather than fails (``SectionUnavailable``,
turning into a pytest skip in ``test_goldens._build``); the harness's
``db_stanza`` / ``_check_database_identity`` compares the sha256s against the
provenance afterwards, so :func:`build` itself only checks existence.

Why exactly these two files: they are the PPM production pair the 2026-06-30
cross-code photon comparison used —

* EM: ``mceq_db_real-ppm-drop-10dec_EM.h5``,
  sha256 ``5f1b02cd12080e47fac487c236054c7f3b81556744add38efc8d9100134e9b3d``
* hadronic: ``mceq_db_v13_1mev_compact_lext_DPMJETIII193.h5``,
  sha256 ``a6694d740949071c487944a056631ba231da40337e7071ccd9bb5dfee14c32f8``

With ``em_standalone_grid = True`` the energy grid is read from the EM file —
d = 150 with bin edges from 1 MeV to 1 EeV (centers 1.12e-3 to 8.91e11 GeV) —
and ``HDF5Backend._em_standalone`` skips the hadronic interaction and decay
matrices. The hadronic database is not inert: its
``continuous_losses/air/ionization`` e+/e- curves are interpolated onto the EM
grid by the standalone branches of ``hdf5_backend``, and doubling them alone
moves the solved state by max |Δ| 4.3e+29 — which is why its sha256 is part of
the provenance identity, not incidental.

``muon_helicity_dependence`` is pinned True like every other generator's
table, but ``MCEqRun.__init__`` forces it False under ``enable_em`` (helicity
rows destabilise the EM cascade; core prints "forcing
muon_helicity_dependence=False"). The forcing is a behaviour no array value
can reveal, so ``meta/helicity_forced_off`` pins that it happened.

The drop EM database carries no ``rho_grid``, so ``em_air_density`` stays
``None`` and this section always reads the medium's air tables; the rho-stack
path is the ``emrho`` section's job, not this one's. The drop DB also carries
a ``subfloor_dedx`` dataset which this backend does not read — no consumer in
``src/MCEq`` — so it is carried for the maintenance tools' external scorer,
not used by ``MCEqRun``.

The comparison modes: the digest keys and the meta stanza are bitwise (the
default). ``grid_sol`` — ``m.grid_sol`` as stored, one depth snapshot per row
— is scored ``per_species_max`` at 1e-11 through the species table in
``meta/``, the same treatment gen_solve1d's state keys get. Measured on this
run, though, the sign-definite guard admits none of the species/depth lanes, so
``compare_key`` falls back to whole-array rel-L2 dominated by the final depth
snapshot (norms by depth 1.2e2, 1.7e5, 4.5e9, 2.6e26); the per-species bound
still applies through the 1e-12 x peak floor. The disposition of that fallback
is an open maintainer ruling. ``spectra`` is a stacked readout the per-species
metric cannot slice (its rows are species, not energy), so it is compared at
rel-L2 1e-11, the 1D per-species bound — which gen_solve1d's single-depth flux
keys clear bitwise anyway.
"""

from __future__ import annotations

import copy
import hashlib
import pathlib
import time

import numpy as np

from . import _flux_metric
from ._harness import array_digest, load_section, make_provenance, sparse_digest
from .make_goldens import SectionUnavailable

SECTION = "emphoton"

#: Absolute paths of the two external payloads; see the module docstring for
#: the run directories and the sha256s the harness checks through provenance.
EM_DB_PATH = pathlib.Path(
    "/ceph/sharedfs/work/SATORI/anatoli/devel/mceq-em-integration/runs/"
    "2026-06-26_ppm-prod-realistic-drop/outputs/"
    "mceq_db_real-ppm-drop-10dec_EM.h5"
)
HAD_DB_PATH = pathlib.Path(
    "/ceph/sharedfs/work/SATORI/anatoli/devel/mceq-em-integration/runs/"
    "2026-05-25_em-db-v13-1mev/outputs/"
    "mceq_db_v13_1mev_compact_lext_DPMJETIII193.h5"
)

INTERACTION_MODEL = "DPMJETIII193"
PRIMARY_PDG = 22
PRIMARY_ENERGY = 1e6  # GeV, inside the 1 MeV - 110 PeV standalone grid
THETA_DEG = 60.0

#: Depths in g/cm2 for the grid solve; the scored spectra are read per index.
INT_GRID = [10.0, 100.0, 300.0, 800.0]

#: The cascade species, row order of the ``spectra`` key. Bare names resolve
#: through ``pman.pname2pref`` without a ``total_`` prefix.
SPECTRA_NAMES = ("e-", "e+", "gamma")

#: rel-L2 bound for the stacked ``spectra`` key — the 1D per-species bound
#: (``_flux_metric.RTOL_1D``), justified in the module docstring.
SPECTRA_RTOL = 1e-11

#: Config globals the section fixes. The autouse fixture in tests/conftest.py
#: restores only six of these, so a golden built inside a full pytest session
#: has to set — and afterwards restore — the rest itself. ``kernel_config`` is
#: deliberately left ambient like in gen_solve1d; provenance records it.
CONFIG_PINS = {
    "data_dir": pathlib.Path(""),  # the DB paths below are absolute
    "mceq_db_fname": str(HAD_DB_PATH),
    "em_db_fname": str(EM_DB_PATH),
    "enable_em": True,
    "em_standalone_grid": True,
    "em_air_density": None,  # the drop DB has no rho_grid (see docstring)
    "e_min": 1e-3,
    "e_max": 1.1e14,
    "enable_energy_loss": True,
    "enable_em_ion": True,
    "enable_cont_rad_loss": True,
    "generic_losses_all_charged": True,
    "enable_default_tracking": False,
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "return_as": "kinetic energy",
    "excpt_on_missing_particle": True,
    "density_model": ("CORSIKA", ("BK_USStd", None)),
    "r_E": 6371.315e3,
    "h_obs": 0.0,
    "h_atm": 112.8e3,
    "X_start": 0.0,
    "interaction_medium": "air",
    "A_target": "auto",
    "floatlen": None,
    "cuda_fp_precision": 64,
    "etd2_path": {"eps": 0.3, "dX_max": 20.0, "dX_min": 0.01, "fd_span": 0.01},
    "em_adaptive_step": False,
    "em_step_safety": 0.12,
    "em_step_dense_eig_max": 4000,
    "minimal_primary_energy": 3.0,
    "fallback_to_air_cs": True,
    "average_loss_operator": False,
    "loss_step_for_average": 1e-1,
    "loss_stencil_method": "expfit_low_upwind2",
    "loss_stencil_low_upwind_rows": 8,
    "loss_stencil_alpha0": 3.0,
    "use_isospin_sym": True,
    "muon_helicity_dependence": True,  # forced False by MCEqRun; see docstring
    "muon_multiple_scattering": True,
    "assume_nucleon_interactions_for_exotics": True,
    "prompt_ctau": 2.6842,
    "low_energy_extension": {
        "model": None,
        "he_le_transition": 80,
        "he_le_trwidth": 0.3,
        "use_unknown_cs": True,
    },
}

ADV_SET_PINS = {
    "disabled_particles": [13, -13],
    "disable_interactions_of_unstable": False,
    "disable_charm_pprod": False,
    "allowed_projectiles": [],
    "disable_direct_leptons": False,
    "disable_leading_mesons": False,
    "disable_decays": [],
    "force_resonance": [],
    "forced_int_cs": None,
    "replace_meson_cross_sections_with": None,
}

#: The one non-bitwise numeric key. ``grid_sol`` stores ``m.grid_sol`` as it
#: comes out of the solver — one depth snapshot per row of the flat
#: (depth, species x dim) state — and carries the same metric gen_solve1d's
#: stacks get: per_species_max at 1e-11 (maintainer ruling 2026-09-02) with
#: the 1e-12 x peak floor and the sign-definite guard on the grid less its top
#: 3 bins, against the species table in ``meta/``. The measured 1D retune
#: spread the bound carries (3.4x worst over all state keys) is the
#: calibration; a fresh EM shower is young and sign-definite almost anywhere,
#: so the guard admits essentially every row.
TOLERANCES = {
    "grid_sol": {
        "mode": "per_species_max",
        "rtol": _flux_metric.RTOL_1D,
        "floor": _flux_metric.FLOOR,
        "guard": _flux_metric.DEFAULT_GUARD,
        "trim_top_bins": _flux_metric.TRIM_TOP_BINS,
    },
    # The stacked (3, n_depths, d) readout has species on axis 0 and depth on
    # axis 1, not energy on the last axis of a (dim_states, K) state, so the
    # metric cannot slice it into species lanes; it keeps an L2 bound at the
    # 1D flux tolerance instead — see the module docstring.
    "spectra": {"mode": "rel_l2", "rtol": SPECTRA_RTOL},
}


def _require_databases():
    """Existence check for both payloads, or ``SectionUnavailable``.

    No checksums here: ``make_provenance`` digests both files through
    ``db_stanza`` and ``test_goldens._check_database_identity`` compares them
    against the stored provenance, which is where a swapped file must fail.
    """
    missing = [str(p) for p in (EM_DB_PATH, HAD_DB_PATH) if not p.exists()]
    if missing:
        raise SectionUnavailable(
            "PPM EM golden database not present: " + "; ".join(missing)
        )


def _species_digest(pman):
    """The gen_species ``_species_digest`` idiom, reused rather than imported.

    sha256 over the ``array_digest`` of the cascade names and of the
    int64-coerced mceqidx array (mceqidx mixes int and numpy.int64 — tracking
    particles take theirs from ``np.max(...) + 1`` — so the indices are
    coerced, exactly as there).
    """
    names = np.array([p.name for p in pman.cascade_particles])
    indices = np.array([int(p.mceqidx) for p in pman.cascade_particles], dtype=np.int64)
    h = hashlib.sha256()
    h.update(array_digest(names).encode())
    h.update(array_digest(indices).encode())
    return h.hexdigest()


def _csr_digest(mat) -> str:
    """sha256 hex folding the three sparse CSR buffer digests of `mat`.

    The same per-buffer hashing :func:`._harness.sparse_digest` performs —
    sort a copy, then hash dtype/shape/bytes of `data`, `indices`, `indptr`
    (operators1d stores those three digests as separate keys; this section
    has one key for int_m, so it folds the three hex digests in that order).
    Full int_m, not the EM block: restricting it would mean re-deriving the
    species-block boundaries from pman internals for no extra signal, since
    the digest already covers every buffer the solver binds.
    """
    digest = sparse_digest(mat)
    h = hashlib.sha256()
    for part in ("data", "indices", "indptr"):
        h.update(digest[part].encode())
    return h.hexdigest()


def _record_meta(arrays, mceq):
    """The layout stanza ``per_species_max`` slices ``grid_sol`` with.

    Same fields and encoding as gen_solve1d's ``<case>/meta/``, so
    ``_flux_metric.layout_for`` resolves it for the state key. The
    ``grid_*`` fingerprints pin that the standalone grid really came from the
    EM file (d = 150 over 1 MeV - 110 PeV).
    """
    arrays["meta/e_grid"] = mceq.e_grid
    arrays["meta/e_bins"] = mceq.e_bins
    arrays["meta/e_widths"] = mceq.e_widths
    arrays["meta/dim"] = np.asarray(mceq.dim)
    arrays["meta/dim_states"] = np.asarray(mceq.dim_states)
    arrays["meta/species"] = np.array(
        [f"{q.mceqidx}:{q.name}" for q in mceq.pman.cascade_particles], dtype="U32"
    )
    arrays["meta/grid_d"] = np.asarray(int(mceq._energy_grid.d))
    arrays["meta/grid_c0"] = np.asarray(float(mceq.e_grid[0]))
    arrays["meta/grid_cN"] = np.asarray(float(mceq.e_grid[-1]))


def build():
    """Produce (arrays, provenance) for the PPM EM photon section."""
    from MCEq import config

    missing = sorted(k for k in CONFIG_PINS if not hasattr(config, k))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    _require_databases()

    saved_config = {k: getattr(config, k) for k in CONFIG_PINS}
    saved_adv_set = copy.deepcopy(config.adv_set)
    arrays = {}
    seconds = {}
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)
        config.adv_set.update(ADV_SET_PINS)

        from MCEq.core import MCEqRun

        started = time.perf_counter()
        mceq = MCEqRun(
            interaction_model=INTERACTION_MODEL,
            theta_deg=THETA_DEG,
            primary_model=None,
        )
        seconds["construct"] = time.perf_counter() - started
        try:
            # The constructor pins the forcing away from the pin table: a
            # config refactor that stopped forcing would move nothing else
            # here, because with [13, -13] disabled the helicity lanes do not
            # exist in this system at all.
            arrays["meta/helicity_forced_off"] = np.asarray(
                int(config.muon_helicity_dependence is False)
            )
            assert mceq.dim == 150, (
                f"standalone EM grid is d={mceq.dim}, expected 150 — "
                f"config.grid did not come from {EM_DB_PATH.name}"
            )
            _record_meta(arrays, mceq)

            mceq.set_single_primary_particle(E=PRIMARY_ENERGY, pdg_id=PRIMARY_PDG)
            arrays["meta/phi0"] = np.copy(mceq._phi0)

            started = time.perf_counter()
            mceq.solve(int_grid=INT_GRID, dX_max=20.0)
            seconds["solve"] = time.perf_counter() - started

            nsteps, dX, _, grid_idcs = mceq.integration_path
            # One float array: the step count, then the X values the steps end
            # at (cumsum of the step sizes; X_start is pinned 0.0). gen
            # _solve1d keeps the same tuple split across three keys; this
            # section has one run and folds it into the `path` key, with the
            # bookkeeping (grid_idcs, int_grid) in the provenance extra.
            arrays["path"] = np.concatenate(
                (
                    np.asarray([float(nsteps)]),
                    np.cumsum(np.asarray(dX, dtype=np.float64)),
                )
            )

            # (4, dim_states) as the solver stacks it; state_lanes slices the
            # rows into species for the flux metric via the meta/ stanza.
            arrays["grid_sol"] = np.asarray(mceq.grid_sol)

            arrays["spectra"] = np.stack(
                [
                    [
                        mceq.get_solution(name, grid_idx=j, integrate=True)
                        for j in range(len(INT_GRID))
                    ]
                    for name in SPECTRA_NAMES
                ]
            )

            arrays["partdigest"] = np.asarray(_species_digest(mceq.pman))
            arrays["csdigest"] = np.asarray(
                hashlib.sha256(
                    np.ascontiguousarray(
                        mceq._mceq_db.cs_db(INTERACTION_MODEL)["index_d"][PRIMARY_PDG]
                    ).tobytes()
                ).hexdigest()
            )
            # Full int_m, digested like operators1d digests int_m through
            # _operator_sweep (sort a copy, hash the three CSR buffers).
            arrays["emdigest"] = np.asarray(_csr_digest(mceq.int_m))

            # Captured before close(): close() pops _backend_cache from the
            # instance dict.
            backends = sorted(str(k) for k in mceq._backend_cache)
            sizes = {
                "d": int(mceq._energy_grid.d),
                "dim": int(mceq.dim),
                "dim_states": int(mceq.dim_states),
                "n_species": int(mceq.pman.n_cparticles),
            }
        finally:
            mceq.close()

        assert np.isfinite(arrays["grid_sol"]).all(), "EM solve went non-finite"
        empty = [
            name
            for i, name in enumerate(SPECTRA_NAMES)
            if not np.any(arrays["spectra"][i])
        ]
        assert not empty, (
            f"identically zero spectra: {empty} — a missing or silently "
            f"decoupled EM species would freeze into a golden that can never fail"
        )

        seconds["total"] = seconds["construct"] + seconds["solve"]
        provenance = make_provenance(
            SECTION,
            note=(
                "Photon primary (pdg 22, 1e6 GeV, zenith 60 deg) on the PPM"
                " production EM database with the standalone grid (d = 150,"
                " edges 1 MeV - 1 EeV, centers 1.12e-3 - 8.91e11 GeV, read"
                " from the EM file; the hadronic DB's interaction and decay"
                " matrices are skipped but its continuous_losses/air/"
                "ionization e+- curves are interpolated onto the EM grid and"
                " do feed the solve). This is the pair the"
                " 2026-06-30 cross-code photon comparison used; both payloads"
                " live in the shared-filesystem runs tree and neither is"
                " carried by CI, so the section is slow, and it is host"
                " because three keys are sha256 digests. muon_helicity_"
                "dependence is pinned True but MCEqRun forces it False under"
                " enable_em (meta/helicity_forced_off pins the forcing). The"
                " drop DB carries no rho_grid (em_air_density None — the"
                " rho-stack path is the emrho section's job) and its"
                " subfloor_dedx dataset is not read by this backend (no"
                " consumer in src/MCEq; it serves the maintenance tools'"
                " external scorer). disabled_particles [13, -13]: the golden"
                " transports the EM shower, not muons. grid_sol is compared"
                " per_species_max at 1e-11 like the solve1d states, but here"
                " the sign-definite guard admits no lane so compare_key falls"
                " back to whole-array rel-L2 (an open maintainer ruling);"
                " spectra"
                " (integrated e-/e+/gamma per depth) is a stacked readout the"
                " metric cannot slice and keeps rel-L2 at the same 1e-11;"
                " partdigest is the gen_species species-table digest, csdigest"
                " the gamma total-inelastic curve the run uses, emdigest the"
                " full int_m CSR data buffer as operators1d digests it."
            ),
            tolerances=TOLERANCES,
            extra={
                "seconds": seconds,
                **sizes,
                "int_grid": INT_GRID,
                "primary": {"pdg_id": PRIMARY_PDG, "E": PRIMARY_ENERGY},
                "theta_deg": THETA_DEG,
                "interaction_model": INTERACTION_MODEL,
                "backends_executed": backends,
                "kernel_config_executed": config.kernel_config,
            },
        )
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)

    for key in ("mceq_db_fname", "em_db_fname"):
        db = provenance["databases"].get(key, {})
        assert "sha256" in db, f"{key} not resolvable, provenance incomplete: {db}"

    return arrays, provenance


def diff_report(golden: dict, produced: dict) -> list[str]:
    """Per-species worst and the fixed-energy table for the state key.

    Same shape as gen_solve1d's: the species layout is in the golden's own
    ``meta/`` arrays and the stored provenance is consulted only for the
    floor, guard and trim the file was written with.
    """
    _, prov = load_section(SECTION)
    entry = prov.get("tolerances", {}).get("grid_sol", {})
    return _flux_metric.flux_report(
        golden,
        produced,
        keys=("grid_sol",),
        rtol=float(entry.get("rtol", _flux_metric.RTOL_1D)),
        floor=float(entry.get("floor", _flux_metric.FLOOR)),
        guard=entry.get("guard", _flux_metric.DEFAULT_GUARD),
        trim=int(entry.get("trim_top_bins", _flux_metric.TRIM_TOP_BINS)),
    )
