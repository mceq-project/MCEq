"""Golden section for the 1D SIBYLL21 solve and the ``get_solution`` surface.

Plan section 8.1: SIBYLL21 on the reduced database at zenith 0/60/89 deg, with
the state vector, the integration path, the named spectra behind
:meth:`MCEqRun.get_solution`, and the particle counters
:meth:`n_particles` / :meth:`n_mu` / :meth:`n_e`.

Two cases, because the EM block is only conditionally stable under ETD2:

``emoff``
    ``adv_set["disabled_particles"] = [11, -11]`` -- the production default --
    at theta = 0, 60 and 89 deg. Every entry is finite and the system is well
    conditioned: a 1-ulp perturbation of ``_phi0`` moves the solution by
    rel-L2 ~2e-16, and numpy_etd2 and mkl_etd2 agree bitwise. Every key is
    therefore compared bitwise on the host.
``emon``
    ``adv_set["disabled_particles"] = []`` -- what ``tests/conftest.py``'s
    ``mceq_sib21`` fixture uses -- at theta = 0 ONLY. With e+/e- in the system
    the solve diverges with zenith: max|phi| is 9.5e-02 at 0 deg, 3.1e+21 at
    60 deg, 4.4e+208 at 85 deg and NaN at 89 deg. The NaN mask is backend
    dependent (at 89 deg mkl_etd2 produces 192 NaN entries and numpy_etd2 186,
    the six extra being antinue), so above 0 deg no cross-backend golden of
    this case exists.

``get_solution`` returns zeros for an unrecognised particle name whenever
``config.excpt_on_missing_particle`` is False, so :func:`build` pins that flag
True and additionally rejects any spectrum that comes out identically zero --
a typo would otherwise freeze into a golden that can never fail.

The spectra span the arguments that change the answer, not just the defaults:
``mag`` in {0, 1, 3}, ``dont_sum_helicities``, ``integrate`` for a lepton and a
hadron, and ``return_as="total energy"`` at ``mag=1`` (at ``mag=0`` the total-
and kinetic-energy branches return numerically identical arrays, so only a
non-zero ``mag`` separates ``etot_grid`` from ``e_grid``).

``config.kernel_config`` is deliberately left ambient: the golden test drives
this section through numpy, MKL and CUDA against one stored file, comparing the
host backends bitwise and CUDA at ``CUDA_RTOL``, so the executed backend goes
into the provenance rather than into a compared key.

The section is hermetic apart from one binding it cannot control:
``get_solution``'s ``return_as`` default is evaluated when ``MCEq.core`` is
imported (core.py:560), so a process that sets ``config.return_as`` before that
import shifts 17 of the weighted keys. :func:`build` asserts the resolved
default rather than trusting the pin.
"""

from __future__ import annotations

import copy
import inspect

import numpy as np

from ._harness import array_digest, make_provenance, sparse_digest

SECTION = "solve1d"

#: Config globals the section fixes. The autouse fixture in tests/conftest.py
#: restores only six of these, so a golden built inside a full pytest session
#: has to set -- and afterwards restore -- the rest itself.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": "mceq_db_v140reduced_compact.h5",
    "return_as": "kinetic energy",
    "excpt_on_missing_particle": True,
    "density_model": ("CORSIKA", ("BK_USStd", None)),
    "r_E": 6371.315e3,
    "h_obs": 0.0,
    "h_atm": 112.8e3,
    "X_start": 0.0,
    "interaction_medium": "air",
    "A_target": "auto",
    "e_min": 0.1,
    "e_max": 1e11,
    "enable_em": False,
    "em_air_density": None,
    "floatlen": None,
    "cuda_fp_precision": 64,
    "etd2_path": {"eps": 0.3, "dX_max": 20.0, "dX_min": 0.01, "fd_span": 0.01},
    "em_adaptive_step": False,
    "em_step_safety": 0.12,
    "em_step_dense_eig_max": 4000,
    "minimal_primary_energy": 3.0,
    "enable_default_tracking": True,
    "enable_energy_loss": True,
    "generic_losses_all_charged": True,
    "enable_cont_rad_loss": True,
    "fallback_to_air_cs": True,
    "enable_em_ion": True,
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
}

ADV_SET_PINS = {
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

#: Spectra recorded at every zenith. Covers the bare, ``total_``, ``conv_``
#: (total - prompt), ``pr_`` (prcas_ + prres_ + em_) and the ``pi_``/``k_``
#: tracking prefixes. The reduced database carries no charm, so ``pr_`` resolves
#: through ``prres_`` alone.
NAMES = (
    "total_mu+",
    "total_mu-",
    "total_numu",
    "total_antinumu",
    "total_nue",
    "total_antinue",
    "total_gamma",
    "total_p+",
    "conv_mu+",
    "conv_mu-",
    "conv_numu",
    "conv_nue",
    "pr_mu+",
    "pr_numu",
    "pr_nue",
    "pi_numu",
    "pi_mu+",
    "k_numu",
    "k_mu+",
)

#: Names carried through the E^mag weighting and the ``return_as`` branches.
WEIGHTED_NAMES = ("total_mu+", "total_numu", "total_gamma", "total_p+")

#: Integrated spectra: one charged lepton, one neutrino, one hadron, since
#: n_particles/n_mu/n_e all funnel through ``integrate=True``.
INTEGRATED_NAMES = ("total_mu+", "total_numu", "total_p+")

#: Spectra recorded per depth-grid snapshot.
GRID_NAMES = ("total_mu+", "total_numu", "total_gamma")

THETAS_EMOFF = (0.0, 60.0, 89.0)
THETAS_EMON = (0.0,)

#: Depths in g/cm2 for the grid solve.
INT_GRID = [10.0, 100.0, 500.0, 1000.0]


def _record_sparse(arrays, prefix, matrix):
    """Store shape, nnz, dtype and the CSR buffer digests of `matrix`."""
    digest = sparse_digest(matrix)
    arrays[prefix + "/shape"] = np.asarray(digest["shape"])
    arrays[prefix + "/nnz"] = np.asarray(digest["nnz"])
    arrays[prefix + "/dtype"] = np.asarray(digest["dtype"])
    for part in ("data", "indices", "indptr"):
        arrays[prefix + "/" + part] = np.asarray(digest[part])


def _record_meta(arrays, case, mceq):
    """Store the energy grid, the species layout and the operator digests."""
    p = f"{case}/meta/"
    arrays[p + "e_grid"] = mceq.e_grid
    arrays[p + "e_bins"] = mceq.e_bins
    arrays[p + "e_widths"] = mceq.e_widths
    arrays[p + "dim"] = np.asarray(mceq.dim)
    arrays[p + "dim_states"] = np.asarray(mceq.dim_states)
    arrays[p + "species"] = np.array(
        [f"{q.mceqidx}:{q.name}" for q in mceq.pman.cascade_particles], dtype="U32"
    )
    arrays[p + "phi0"] = np.copy(mceq._phi0)
    # etot_grid shifts e_bins by the particle mass, so a massive and a massless
    # species pin both halves of the shift.
    for name in ("mu+", "p+", "gamma"):
        arrays[p + "etot_grid/" + name] = mceq.etot_grid(name)
    _record_sparse(arrays, p + "int_m", mceq.int_m)
    _record_sparse(arrays, p + "dec_m", mceq.dec_m)


def _record_spectra(arrays, tag, mceq, grid_idx=None):
    """Store the `get_solution` surface at one solve state."""
    for name in NAMES:
        arrays[f"{tag}/sol/{name}"] = mceq.get_solution(name, grid_idx=grid_idx)
    for name in WEIGHTED_NAMES:
        for mag in (1.0, 3.0):
            arrays[f"{tag}/sol_mag{mag:g}/{name}"] = mceq.get_solution(
                name, mag=mag, grid_idx=grid_idx
            )
        arrays[f"{tag}/sol_etot_mag1/{name}"] = mceq.get_solution(
            name, mag=1.0, grid_idx=grid_idx, return_as="total energy"
        )
    for name in INTEGRATED_NAMES:
        arrays[f"{tag}/sol_int/{name}"] = mceq.get_solution(
            name, integrate=True, grid_idx=grid_idx
        )
    for name in ("total_mu+", "total_mu-"):
        arrays[f"{tag}/sol_helicity/{name}"] = mceq.get_solution(
            name, grid_idx=grid_idx, dont_sum_helicities=True
        )


def _record_solve(arrays, case, mceq, theta, with_e):
    """Solve at one zenith and store the path, the state and the spectra."""
    tag = f"{case}/theta{theta:g}"
    mceq.set_zenith_azimuth(theta)
    mceq.solve()

    nsteps, dX, rho_inv, _ = mceq.integration_path
    arrays[tag + "/nsteps"] = np.asarray(nsteps)
    arrays[tag + "/path_dX"] = np.asarray(dX)
    arrays[tag + "/path_rho_inv"] = np.asarray(rho_inv)
    arrays[tag + "/state"] = np.copy(mceq._solution)

    _record_spectra(arrays, tag, mceq)

    arrays[tag + "/n_mu"] = np.asarray(mceq.n_mu())
    # A cutoff above e_bins[0] = 794.33 GeV exercises the bin selection in
    # n_particles rather than reusing the whole grid.
    arrays[tag + "/n_mu_cut1e3"] = np.asarray(mceq.n_mu(min_energy_cutoff=1e3))
    for label in ("total_mu+", "total_numu"):
        arrays[tag + "/n_particles/" + label] = np.asarray(mceq.n_particles(label))
    if with_e:
        arrays[tag + "/n_e"] = np.asarray(mceq.n_e())


def _record_grid_solve(arrays, case, mceq):
    """Solve at theta = 0 on an explicit depth grid and store the snapshots.

    `grid_sol` pins the snapshots themselves; the per-index spectra pin that
    `get_solution(grid_idx=...)` selects the matching row, including the clamp
    that maps an index past the end of the grid onto the last snapshot.
    """
    p = f"{case}/grid"
    mceq.set_zenith_azimuth(0.0)
    mceq.solve(int_grid=INT_GRID)

    nsteps, dX, _, grid_idcs = mceq.integration_path
    arrays[p + "/int_grid"] = np.asarray(INT_GRID)
    arrays[p + "/nsteps"] = np.asarray(nsteps)
    arrays[p + "/path_dX"] = np.asarray(dX)
    arrays[p + "/grid_idcs"] = np.asarray(grid_idcs)
    arrays[p + "/grid_sol"] = np.copy(mceq.grid_sol)
    arrays[p + "/grid_sol_shape"] = np.asarray(mceq.grid_sol.shape)
    for i in list(range(len(INT_GRID))) + [len(INT_GRID) + 5]:
        for name in GRID_NAMES:
            arrays[f"{p}/idx{i}/sol/{name}"] = mceq.get_solution(name, grid_idx=i)
        arrays[f"{p}/idx{i}/n_mu"] = np.asarray(mceq.n_mu(grid_idx=i))


def build():
    """Produce (arrays, provenance) for the 1D SIBYLL21 solve section."""
    import crflux.models as pm

    from MCEq import config

    missing = sorted(k for k in CONFIG_PINS if not hasattr(config, k))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    saved_config = {k: getattr(config, k) for k in CONFIG_PINS}
    saved_adv_set = copy.deepcopy(config.adv_set)
    arrays = {}
    backends = set()
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)
        config.adv_set.update(ADV_SET_PINS)

        from MCEq.core import MCEqRun

        # get_solution binds return_as when core.py is imported while
        # _get_solution_from_state resolves it per call, so unifying the two is
        # a behaviour change that no array value can reveal: at mag=0 the
        # kinetic- and total-energy branches return the same numbers.
        bound_default = (
            inspect.signature(MCEqRun.get_solution).parameters["return_as"].default
        )
        assert bound_default == CONFIG_PINS["return_as"], (
            f"MCEq.core was imported with config.return_as={bound_default!r}; "
            f"the weighted keys of this section are only reproducible at "
            f"{CONFIG_PINS['return_as']!r}"
        )
        arrays["meta/get_solution_return_as_default"] = np.asarray(bound_default)

        for case, disabled, thetas, with_e in (
            ("emoff", [11, -11], THETAS_EMOFF, False),
            ("emon", [], THETAS_EMON, True),
        ):
            config.adv_set["disabled_particles"] = disabled
            mceq = MCEqRun(
                interaction_model="SIBYLL21",
                theta_deg=0.0,
                primary_model=(pm.HillasGaisser2012, "H3a"),
            )
            try:
                _record_meta(arrays, case, mceq)
                for theta in thetas:
                    _record_solve(arrays, case, mceq, theta, with_e)
                _record_grid_solve(arrays, case, mceq)
                backends.update(str(k) for k in mceq._backend_cache)
            finally:
                mceq.close()

        # excpt_on_missing_particle is pinned True above, so a typo raises here
        # rather than reaching this guard. What it catches is a valid name whose
        # flux is identically zero on this database, which would freeze into a
        # golden that can never fail.
        zeros = sorted(k for k, v in arrays.items() if "/sol" in k and not np.any(v))
        assert not zeros, f"identically zero spectra: {zeros}"

        nonfinite = sorted(
            key
            for key, value in arrays.items()
            if key.startswith("emoff/")
            and np.asarray(value).dtype.kind == "f"
            and not np.all(np.isfinite(value))
        )
        assert not nonfinite, f"emoff must stay finite at every zenith: {nonfinite}"

        provenance = make_provenance(
            SECTION,
            note=(
                "1D SIBYLL21 on the reduced DB. Case emoff (disabled_particles"
                " [11,-11]) at theta 0/60/89 deg; case emon (disabled_particles"
                " []) at theta 0 deg only, where the EM block is still finite."
                " Host backends agree bitwise; the golden test re-runs the"
                " section under cuda_etd2 and compares it against this file at"
                " rel-L2 1e-9 (measured worst case 9.8e-14)."
            ),
            extra={
                "backends_executed": sorted(backends),
                "int_grid": INT_GRID,
                "names": list(NAMES),
                "state_digest": {
                    key: array_digest(value)
                    for key, value in arrays.items()
                    if key.endswith("/state")
                },
            },
        )
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)

    db = provenance["databases"].get("mceq_db_fname", {})
    assert "sha256" in db, f"reduced DB not resolvable, provenance incomplete: {db}"

    return arrays, provenance
