"""Golden section for the 2D FLUKA rc7 sec(theta) solver routes.

Plan section 8.1: the four secant entry points on a production-shaped 2D
system (`mceq_db_v2_fluka2d_rc7.h5`, FLUKA20251, CORSIKA/USStd,
dim_states 2170 x n_k 48 = 104160 stitched, 22 coupled modes):

``solve``
    single axis at zenith 30 deg, K = 1, on a depth grid.
``solve_batch(phi0_matrix, int_grid=...)``
    shared-path multi-RHS, K = 8: seven proton primaries spread over the
    energy grid plus a zero column, which stays zero because the cascade is
    linear.
``solve_batch(conditions=..., carousel_K=8)``
    the LPT carousel over 12 zeniths. Twelve, not eight: with K_total == K_pipe
    every pixel gets its own slot and no slot is ever reset, so eight
    conditions would leave the carousel's reset path (`solvers.py:1252`)
    unexercised. Twelve pixels through eight slots fire four resets and cost
    2.2 s more than eight.
``solve_fullsky(zenith, azimuth, carousel_K=2)``
    2 zeniths x 2 azimuths. `CorsikaAtmosphere.depends_on_azimuth` is False, so
    the azimuth columns of a zenith are bitwise identical and the grid pins the
    condition dedup, the `(i_zen, i_az)` pixel order and the skymap readout
    rather than azimuth-dependent transport. The 4x4 grid the plan names costs
    54.6 s against 11.2 s here (its 80 deg lane alone is 4 x 1162 steps) and
    pins nothing the 2x2 does not; K_pipe 2 < K 4 keeps the resets.

The path knobs are passed explicitly on every call. `config.etd2_path`
(eps 0.3, dX_max 20) is unstable on this database -- the mu-_l rows reach
|state| ~ 1e22 with negative muon fluxes, at every energy range -- and the
cliff sits between dX_max 8 and 6.

`e_max` is 1e2 so the grid is 31 bins: `secant_theta_e_max` = 31.6 GeV then
splits it into 26 low-energy and 5 high-energy bins, which is what makes the
low-E-first permutation and the one-way T_P support non-trivial. The primary
sits at 50 GeV because `set_single_primary_particle` at or above the top grid
edge (89.13 GeV) raises IndexError.

`config.secant_theta_transport` is pinned True ("require") rather than "auto":
a kernel without a secant route then raises instead of silently downgrading to
the paraxial transport and comparing paraxial numbers against secant goldens.
`config.kernel_config` is left ambient. numpy_etd2 and mkl_etd2 agree bitwise
on 49 of the 61 keys, including every batched state; the twelve that move are
the K = 1 ones, by at most rel-L2 4.0e-15, three orders inside the declared
budget. Pinning numpy_etd2 for its own sake would cost 147 s instead of 57 s.

The section costs 57 s on mkl_etd2: 5.1 construct, 1.8 single, 5.9 multi-RHS,
33.3 carousel, 11.1 full sky.
"""

# ruff: noqa: E402  -- the OpenBLAS pool is sized before numpy loads.
from __future__ import annotations

import copy
import pathlib
import time

#: Thread count for every BLAS pool. `config.set_mkl_threads` applies it to MKL
#: and OpenBLAS alike, so the dense mode-coupling GEMMs of the secant routes are
#: capped on both the single-axis and the batched route. OpenBLAS switches
#: microkernel between 1 and >= 2 threads (4.6e-9 max-relative on the stitched
#: state); 2, 4, 8, 16 and 48 threads all agree bitwise above that.
BLAS_THREADS = 4

import numpy as np

from ._harness import HOST_RTOL, array_digest, file_digest, make_provenance
from .make_goldens import SectionUnavailable

SECTION = "solve2d"

#: 329 MB, linked into src/MCEq/data by hand. It is not published on the
#: releases page `config.ensure_db_available` downloads from, and CI caches
#: only the reduced 1D database.
DB_NAME = "mceq_db_v2_fluka2d_rc7.h5"

INTERACTION_MODEL = "FLUKA20251"  # "FLUKA" raises Unknown selections
PRIMARY_PDG = 2212
PRIMARY_ENERGY = 50.0
SINGLE_THETA = 30.0

#: Config globals the section fixes. The autouse fixture in tests/conftest.py
#: restores only six of these, and tests/test_solvers_2d.py leaks e_max,
#: muon_multiple_scattering and secant_theta_transport, so the section sets --
#: and afterwards restores -- all of them itself.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": DB_NAME,
    "return_as": "kinetic energy",
    "excpt_on_missing_particle": True,
    "density_model": ("CORSIKA", ("USStd", None)),
    "r_E": 6371.315e3,
    "h_obs": 0.0,
    "h_atm": 112.8e3,
    "X_start": 0.0,
    "interaction_medium": "air",
    "A_target": "auto",
    "e_min": 1e-1,
    "e_max": 1e2,
    "enable_em": False,
    "em_air_density": None,
    "floatlen": None,
    "etd2_path": {"eps": 0.3, "dX_max": 20.0, "dX_min": 0.01, "fd_span": 0.01},
    "minimal_primary_energy": 3.0,
    "enable_default_tracking": True,
    "enable_energy_loss": True,
    "generic_losses_all_charged": True,
    "enable_cont_rad_loss": True,
    "enable_em_ion": True,
    "fallback_to_air_cs": True,
    "average_loss_operator": False,
    "loss_step_for_average": 1e-1,
    "loss_stencil_method": "expfit_low_upwind2",
    "loss_stencil_low_upwind_rows": 8,
    "loss_stencil_alpha0": 3.0,
    "use_isospin_sym": True,
    "muon_helicity_dependence": True,
    "muon_multiple_scattering": True,
    "assume_nucleon_interactions_for_exotics": True,
    "prompt_ctau": 2.6842,
    "low_energy_extension": {
        "model": None,
        "he_le_transition": 80,
        "he_le_trwidth": 0.3,
        "use_unknown_cs": True,
    },
    "secant_theta_transport": True,
    "secant_theta_cap_deg": 75.0,
    "secant_theta_row_kmax": 50.0,
    "secant_theta_lam_rel": 1e-9,
    "secant_theta_w_flat": 1.0,
    "secant_theta_e_max": 31.6,
}

#: e+/e- stay out of the system: with them in, the 2D ETD2 solve diverges above
#: 60 deg and the NaN mask is backend dependent.
ADV_SET_PINS = {
    "disabled_particles": [11, -11],
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

#: ETD2 path knobs, passed to every solve. dX_max <= 5 is the stability
#: requirement; the empirical scan at zenith 30 deg gives max|state| 3.8e22 at
#: dX_max 20, 2.8e15 at 10, 126 at 8 and 41 at 6 and 5.
PATH = {"X_start": 0.0, "eps": 1.0, "dX_max": 5.0, "dX_min": 0.01, "fd_span": 0.01}

#: Depths in g/cm2 for the snapshots of the two shared-path routes.
INT_GRID = [300.0, 700.0]

#: Proton primaries seeding the multi-RHS columns, one per decade-ish so the
#: right-hand sides are linearly independent rather than scaled copies. An
#: eighth, zero column follows.
MULTIRHS_ENERGIES = (50.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0)

CAROUSEL_ZENITHS = (
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    55.0,
    60.0,
    65.0,
    68.0,
    70.0,
    72.0,
)
CAROUSEL_K = 8

SKY_ZENITH = (0.0, 60.0)
SKY_AZIMUTH = (0.0, 180.0)
SKY_CAROUSEL_K = 2
SKYMAP_ENERGY = 1.0  # GeV kinetic, inside the 0.089 - 89.13 GeV grid

SPECIES = ("total_mu+", "total_mu-", "total_numu", "total_nue")
SPECTRUM_MAG = 3.0

#: Ceiling for the stability check. The routes peak at |state| ~ 42.
STATE_PEAK_MAX = 1e6

#: rel-L2 budget for the secant states and spectra. Across OpenBLAS
#: microkernels the fullsky state moves 3.02e-12, the carousel 1.93e-12 and the
#: single axis 1.17e-12 -- above the plan's 1e-12.
SECANT_RTOL = 1e-11

TOLERANCES = {
    "state/": {"mode": "rel_l2", "rtol": SECANT_RTOL},
    "spectrum/": {"mode": "rel_l2", "rtol": SECANT_RTOL},
    # np.linalg.eig orders the two near-degenerate eigenvalues of S_P
    # (1.000000000212, 0.999999951399) by microkernel; sorted they agree to
    # 6.3e-15. Everything else under ops/ is read off the T disk cache without
    # further BLAS and stays bitwise.
    "ops/lam_sorted": {"mode": "rel_l2", "rtol": HOST_RTOL},
}


def _require_db(config):
    """Path of the 2D database, or `SectionUnavailable` if it is not here.

    `MCEqRun.__init__` calls `config.ensure_db_available`, which would try to
    download a file the releases page does not carry, so the check has to come
    before the constructor.
    """
    path = pathlib.Path(config.data_dir) / DB_NAME
    if not path.exists():
        raise SectionUnavailable(
            f"{DB_NAME} is not at {path}. The 329 MB 2D FLUKA rc7 database is "
            "not published on the MCEq releases page and is not cached by CI; "
            "link it into src/MCEq/data to build this section."
        )
    return path


def _mode_blocks(state, n_k):
    """View a stitched state as `(n_k, dim_states, ...)`.

    `solve` broadcasts the initial condition with `np.tile(phi0, n_k)`, so the
    stitched index is `k * dim_states + i` and the Hankel mode is the outer
    axis.
    """
    state = np.asarray(state)
    return state.reshape((n_k, -1) + state.shape[1:])


def _record_state(arrays, tag, state, n_k, *, monopole=True):
    """Reduce a stitched state to per-mode L2 norms and its monopole block.

    The four raw states are 24 MB together, so the section compares these
    reductions and carries the sha256 of each raw state in the provenance
    instead. A max-relative on the raw state is not usable: it reaches 5.5e-5
    across OpenBLAS microkernels purely from components at ~1e-16 of the peak,
    while the same run is 7.3e-11 on components above 1e-10 of the maximum.
    """
    blocks = _mode_blocks(state, n_k)
    arrays[f"state/{tag}_mode_l2"] = np.linalg.norm(blocks, axis=1)
    if monopole:
        arrays[f"state/{tag}_mode0"] = np.ascontiguousarray(blocks[0])


def _check_stable(tag, state):
    """Return max|state|, refusing to golden a diverged one.

    The ETD2 path sits on a stability cliff on this database: at the config
    default dX_max = 20 the mu-_l rows reach |state| ~ 1e22 with negative muon
    fluxes. Pinning a blown-up state would freeze it into the golden.
    """
    peak = float(np.abs(np.asarray(state)).max())
    if not np.isfinite(peak) or peak > STATE_PEAK_MAX:
        raise RuntimeError(
            f"{tag}: |state| peaks at {peak:.3e}, above {STATE_PEAK_MAX:.0e} -- "
            f"the ETD2 path knobs {PATH} have gone unstable on {DB_NAME}"
        )
    return peak


def _record_batch_spectra(arrays, tag, result, selectors):
    """Named E^3-weighted spectra, one row per selector, stacked in column order."""
    for name in SPECIES:
        arrays[f"spectrum/{tag}_{name}"] = np.stack(
            [result.get_solution(name, mag=SPECTRUM_MAG, **sel) for sel in selectors]
        )


def _secant_cache_entry(config, ops):
    """Identify the disk-cached T operator behind `ops`.

    `secant_coupling_matrix` zeroes every row with kappa > row_kmax before
    storing the full `(n_k, n_k)` matrix under
    `data/secant_cache/T_<hash>.npy`, hashed on the MCEq version, so T
    reconstructs exactly from `P` and `T_P`. A cold cache costs 180 s to
    rebuild.
    """
    n_k = int(ops["n_k"])
    coupling = np.zeros((n_k, n_k))
    coupling[np.asarray(ops["P"])] = ops["T_P"]
    cache_dir = pathlib.Path(config.data_dir) / "secant_cache"
    hit = next(
        (
            path
            for path in sorted(cache_dir.glob("T_*.npy"))
            if np.array_equal(np.load(path), coupling)
        ),
        None,
    )
    return {
        "operator_digest": array_digest(coupling),
        "cache_file": hit.name if hit is not None else None,
        "cache_sha256": file_digest(hit) if hit is not None else None,
    }


def build():
    """Produce (arrays, provenance) for the 2D FLUKA rc7 secant section."""
    from MCEq import config

    missing = sorted(key for key in CONFIG_PINS if not hasattr(config, key))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    db_path = _require_db(config)

    saved_config = {key: getattr(config, key) for key in CONFIG_PINS}
    saved_config["mkl_threads"] = config.mkl_threads
    saved_adv_set = copy.deepcopy(config.adv_set)
    arrays = {}
    seconds = {}
    peaks = {}
    digests = {}
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)
        config.adv_set.update(ADV_SET_PINS)

        from MCEq.core import MCEqRun

        config.set_mkl_threads(BLAS_THREADS)
        started = time.perf_counter()
        mceq = MCEqRun(
            interaction_model=INTERACTION_MODEL,
            primary_model=None,
            theta_deg=SINGLE_THETA,
            density_model=CONFIG_PINS["density_model"],
        )
        seconds["construct"] = time.perf_counter() - started
        try:
            n_k = int(mceq._mceq_db.n_k)

            # --- the constant sec(theta) operator set ------------------
            ops = mceq._build_secant_ops()
            arrays["ops/k_grid"] = np.asarray(mceq._mceq_db.k_grid)
            arrays["ops/e_grid"] = np.asarray(mceq.e_grid)
            arrays["ops/n_k"] = np.asarray(n_k)
            arrays["ops/P"] = np.asarray(ops["P"])
            arrays["ops/T_P"] = np.asarray(ops["T_P"])
            arrays["ops/T_PP"] = np.asarray(ops["T_PP"])
            arrays["ops/low_e_idx"] = np.asarray(ops["low_e_idx"])
            # V, Vi and lam themselves are not goldened: eig returns the
            # near-degenerate eigenvalues in a microkernel-dependent order
            # and the column signs of V are arbitrary.
            arrays["ops/lam_sorted"] = np.sort(np.asarray(ops["lam"]))
            secant_cache = _secant_cache_entry(config, ops)

            # --- multi-RHS columns, then the canonical primary ---------
            columns = []
            for energy in MULTIRHS_ENERGIES:
                mceq.set_single_primary_particle(E=energy, pdg_id=PRIMARY_PDG)
                columns.append(np.copy(mceq._phi0))
            columns.append(np.zeros_like(columns[0]))
            phi0_matrix = np.stack(columns, axis=1)
            mceq.set_single_primary_particle(E=PRIMARY_ENERGY, pdg_id=PRIMARY_PDG)
            # phi0 lands under state/ because set_single_primary_particle
            # equalises three moments through scipy.linalg.solve, i.e.
            # LAPACK, so it is not one of the BLAS-free operator reads.
            arrays["state/phi0"] = np.copy(mceq._phi0)
            arrays["state/phi0_matrix_l2"] = np.linalg.norm(phi0_matrix, axis=0)

            # --- 1. single axis, K = 1 ---------------------------------
            started = time.perf_counter()
            mceq.solve(int_grid=INT_GRID, **PATH)
            seconds["single"] = time.perf_counter() - started

            nsteps, dX, rho_inv, grid_idcs = mceq.integration_path
            arrays["path/single_nsteps"] = np.asarray(nsteps)
            arrays["path/single_dX"] = np.asarray(dX)
            arrays["path/single_rho_inv"] = np.asarray(rho_inv)
            arrays["path/single_grid_idcs"] = np.asarray(grid_idcs)
            arrays["path/int_grid"] = np.asarray(INT_GRID)

            peaks["single"] = _check_stable("single", mceq._solution)
            digests["single"] = array_digest(mceq._solution)
            _record_state(arrays, "single", mceq._solution, n_k)
            for index, snapshot in enumerate(mceq.grid_sol):
                _record_state(
                    arrays, f"single_grid{index}", snapshot, n_k, monopole=False
                )
            for name in SPECIES:
                arrays[f"spectrum/single_{name}"] = mceq.get_solution(
                    name, mag=SPECTRUM_MAG
                )
                arrays[f"spectrum/single_grid_{name}"] = np.stack(
                    [
                        mceq.get_solution(name, mag=SPECTRUM_MAG, grid_idx=index)
                        for index in range(len(INT_GRID))
                    ]
                )

            # --- 2. shared-path multi-RHS, K = 8 -----------------------
            started = time.perf_counter()
            multirhs = mceq.solve_batch(phi0_matrix, int_grid=INT_GRID, **PATH)
            seconds["multirhs"] = time.perf_counter() - started

            peaks["multirhs"] = _check_stable("multirhs", multirhs.sol)
            digests["multirhs"] = array_digest(multirhs.sol)
            _record_state(arrays, "multirhs", multirhs.sol, n_k)
            for index, snapshot in enumerate(multirhs.grid_sol):
                _record_state(
                    arrays, f"multirhs_grid{index}", snapshot, n_k, monopole=False
                )
            _record_batch_spectra(
                arrays,
                "multirhs",
                multirhs,
                [{"k": k} for k in range(multirhs.K)],
            )
            for name in SPECIES:
                arrays[f"spectrum/multirhs_grid_{name}"] = np.stack(
                    [
                        multirhs.get_solution(
                            name, k=0, mag=SPECTRUM_MAG, grid_idx=index
                        )
                        for index in range(len(INT_GRID))
                    ]
                )
            # A zero right-hand side stays zero: the cascade is linear.
            assert not np.any(multirhs.sol[:, -1]), "zero column picked up flux"

            # --- 3. LPT carousel over 12 zeniths, K_pipe = 8 -----------
            started = time.perf_counter()
            carousel = mceq.solve_batch(
                conditions=[{"zenith_deg": z} for z in CAROUSEL_ZENITHS],
                carousel_K=CAROUSEL_K,
                **PATH,
            )
            seconds["carousel"] = time.perf_counter() - started

            arrays["path/carousel_zeniths"] = np.asarray(CAROUSEL_ZENITHS)
            arrays["path/carousel_nsteps_per_col"] = np.asarray(carousel.nsteps_per_col)
            peaks["carousel"] = _check_stable("carousel", carousel.sol)
            digests["carousel"] = array_digest(carousel.sol)
            _record_state(arrays, "carousel", carousel.sol, n_k)
            _record_batch_spectra(
                arrays,
                "carousel",
                carousel,
                [{"k": k} for k in range(carousel.K)],
            )

            # --- 4. full sky, 2 zeniths x 2 azimuths -------------------
            started = time.perf_counter()
            fullsky = mceq.solve_fullsky(
                np.asarray(SKY_ZENITH),
                np.asarray(SKY_AZIMUTH),
                carousel_K=SKY_CAROUSEL_K,
                geomagnetic_cutoff=False,  # never reach for gtracr
                **PATH,
            )
            seconds["fullsky"] = time.perf_counter() - started

            arrays["path/fullsky_zenith_grid"] = np.asarray(fullsky.zenith_grid)
            arrays["path/fullsky_azimuth_grid"] = np.asarray(fullsky.azimuth_grid)
            arrays["path/fullsky_pixel_index"] = np.asarray(fullsky.pixel_index)
            arrays["path/fullsky_nsteps_per_col"] = np.asarray(fullsky.nsteps_per_col)
            peaks["fullsky"] = _check_stable("fullsky", fullsky.sol)
            digests["fullsky"] = array_digest(fullsky.sol)
            _record_state(arrays, "fullsky", fullsky.sol, n_k)
            # Selecting by pixel rather than by k pins that column_index
            # flattens (i_zen, i_az) with azimuth innermost.
            _record_batch_spectra(
                arrays,
                "fullsky",
                fullsky,
                [
                    {"pixel": (i_zen, i_az)}
                    for i_zen in range(len(SKY_ZENITH))
                    for i_az in range(len(SKY_AZIMUTH))
                ],
            )
            for name in SPECIES:
                arrays[f"spectrum/fullsky_skymap_{name}"] = fullsky.skymap(
                    name, SKYMAP_ENERGY, mag=SPECTRUM_MAG
                )
            azimuth_degenerate = all(
                np.array_equal(
                    fullsky.sol[:, i_zen * len(SKY_AZIMUTH)],
                    fullsky.sol[:, i_zen * len(SKY_AZIMUTH) + i_az],
                )
                for i_zen in range(len(SKY_ZENITH))
                for i_az in range(len(SKY_AZIMUTH))
            )

            backends = sorted(str(key) for key in mceq._backend_cache)
            sizes = {
                "dim": int(mceq.dim),
                "dim_states": int(mceq.dim_states),
                "n_k": n_k,
                "stitched": int(mceq.dim_states) * n_k,
                "n_coupled_modes": int(np.size(ops["P"])),
                "n_low_e_columns": int(np.size(ops["low_e_idx"])),
                "n_species": len(mceq.pman.cascade_particles),
            }
        finally:
            mceq.close()

        seconds["total"] = sum(seconds.values())

        zeros = sorted(
            key
            for key, value in arrays.items()
            if key.startswith("spectrum/") and not np.any(value)
        )
        assert not zeros, f"identically zero spectra: {zeros}"
        nonfinite = sorted(
            key
            for key, value in arrays.items()
            if np.asarray(value).dtype.kind == "f" and not np.all(np.isfinite(value))
        )
        assert not nonfinite, f"non-finite entries: {nonfinite}"

        provenance = make_provenance(
            SECTION,
            note=(
                "2D FLUKA rc7 secant routes: single axis (K=1), shared-path "
                "multi-RHS (K=8), LPT carousel over 12 zeniths at carousel_K=8 "
                "and solve_fullsky on a 2x2 sky grid at carousel_K=2. Every "
                "solve passes the ETD2 path knobs explicitly; the config "
                "defaults (eps 0.3, dX_max 20) drive the mu-_l rows to "
                "|state| ~ 1e22 with negative muon fluxes on this database, "
                "the cliff sitting between dX_max 8 and 6. state/ and "
                "spectrum/ keys carry rel-L2 1e-11 rather than the plan's "
                "1e-12: across OpenBLAS microkernels the fullsky state moves "
                "3.02e-12 rel-L2, the carousel 1.93e-12 and the single axis "
                "1.17e-12. On one host with the OpenBLAS pool at >= 2 threads "
                "every key is bitwise, and mkl_etd2 reproduces every key "
                "bitwise except the K=1 state (rel-L2 2.1e-14). ops/ keys are "
                "read off the secant T disk cache without further BLAS and "
                "stay bitwise; ops/lam_sorted is 1e-12 because np.linalg.eig "
                "orders the two near-degenerate eigenvalues of S_P by "
                "microkernel. The raw (104160, K) states are 24 MB together, "
                "so they are compared through per-mode L2 norms, the monopole "
                "block and the named spectra, with their sha256 in "
                "extra.state_digests."
            ),
            tolerances=TOLERANCES,
            extra={
                "db_path": str(db_path),
                "backends_executed": backends,
                "kernel_config_executed": config.kernel_config,
                "blas_threads": BLAS_THREADS,
                "path_knobs": dict(PATH),
                "int_grid": INT_GRID,
                "multirhs_energies": list(MULTIRHS_ENERGIES),
                "carousel_zeniths": list(CAROUSEL_ZENITHS),
                "carousel_K": CAROUSEL_K,
                "sky_zenith": list(SKY_ZENITH),
                "sky_azimuth": list(SKY_AZIMUTH),
                "sky_carousel_K": SKY_CAROUSEL_K,
                "sizes": sizes,
                "seconds": seconds,
                "state_digests": digests,
                "state_peaks": peaks,
                "secant_cache": secant_cache,
                # CorsikaAtmosphere.depends_on_azimuth is False, so the azimuth
                # columns of a zenith come out identical; only the
                # MSIS*LocationCentered models make that axis physical.
                "fullsky_azimuth_degenerate": azimuth_degenerate,
            },
        )
    finally:
        config.set_mkl_threads(saved_config.pop("mkl_threads"))
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)

    db = provenance["databases"].get("mceq_db_fname", {})
    assert "sha256" in db, f"{DB_NAME} not resolvable, provenance incomplete: {db}"

    return arrays, provenance
