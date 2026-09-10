"""Golden section ``emrho``: the EM transport chain on a synthetic rho-stack.

CI-runnable and hermetic: both databases are synthetic and written into a temp
directory by :func:`build` in well under a second (the hadronic one is the
``tests/data/make_hdf5_fixtures.py`` fixture via ``build_database(path,
tuple_width=4)`` — the by-file-path import idiom of ``tests/conftest.py``,
since ``tests/data`` is not an importable package; the EM one is the rho-stack
of :func:`build_rho_stack_em` below). Physical HDF5 file digests depend on
the HDF5 build and container metadata, so they are provenance only for
these generated fixtures. The decoded inputs, assembled operators and
solved states below provide the portable regression checks.

What executes is the full chain — HDF5 -> ``MCEq.data.em_tables`` rho-slice
selection -> species setup -> operator assembly -> ETD2 solve — with the slice
choice observable in the result, so it complements the host-only ``emphoton``
section (real PPM EM database) with a file no external payload is needed for.
Neither slow nor host-pinned: nothing here is a sha256 of a spline or a solve
on a payload CI lacks.

Layout of the EM file
    ``electromagnetic/air/rho_{i:02d}`` each with an ``emca_mats`` pack (built
    by the fixture module's own ``_write_pack`` / ``em_cs_table``, so the pack
    and cs encoding stays the validated one) and a ``cs`` table, plus
    ``electromagnetic/air/rho_grid``. Deliberately NO flat
    ``electromagnetic/air/emca_mats`` or ``/cs`` on the parent group: with
    ``em_air_density`` set the selector must resolve into a slice and must not
    fall back, and a stack-only file read with ``em_air_density=None`` raises
    a ``KeyError`` on the missing ``cs`` dataset from the flat-layout reader
    (validated) — that is the reader's contract for a file without a flat
    layout, not a case this section handles.

Slices: ``rho_grid = [1e-6, 1e-3, 1e-2]`` g/cm3, slice scales ``[0.5, 2.0,
8.0]``. The three cases run ``em_air_density`` = 1.2e-6, 9e-4, 8e-3, which the
closest-in-log10 rule maps to slice 0, 1, 2 respectively, and the ``slice_idx``
key (produced by calling ``select_em_rho_slice`` directly on the store) pins
that mapping itself. The cs scale distinguishes the slices through the cs path
(``em_cs_table(scale)`` multiplies the whole table, pinned directly by
``gamma_cs``), and since slice ``i`` writes the channel list rotated left by
``i`` (see :func:`build_rho_stack_em`), the decoded ``channel_block`` yield
matrices are slice-distinct too: the assembled interaction operator — and with
it the solve — moves with the yield-slice selection itself, not only with the
cs path. ``gamma_cs`` therefore pins the cs leg, ``yield_operator`` pins the
assembled operator leg, and ``grid_sol`` and ``emin_spectrum`` pin the solved
dynamics against both EM inputs at once.

The extended channel set below is why the slice is visible at all: the
fixture's minimal EM set has no gamma *parent*, so a gamma primary would
produce no e± and every slice would solve to the same zero e- spectrum. With
``(22, 11)`` and friends present, a gamma primary yields a nonzero e- spectrum
and the solved state depends on the selected slice (validated).

``config.em_standalone_grid`` is pinned False: set True, the backend would
take the energy grid from the EM file, which a rho-stack does not carry (the
stack has no ``/common`` group at all) — another reason the flag must stay off
here. ``disabled_particles = [13, -13, 3112]`` likewise carries a hard
requirement: the fixture's positive-control channel ``(-211, 3112)`` puts a
Sigma- on the unstable list with no decay pack and ``MCEqRun.__init__``
raises 'Unstable particle without decay distribution' unless 3112 is
disabled.

Tolerances: rel-L2 1e-11 on the three solved numeric keys and 1e-11 on
``yield_operator``. The latter compares the assembled matrix numerically:
its expfit loss stencil solves a small linear system whose rounding differs
across BLAS/LAPACK builds, so a SHA256
cannot serve as a portable comparison. Per-slice digests remain diagnostic
provenance. Slice indices and operator sparsity remain bitwise.

What a mutant has to break to fail this section
    Dropping the EM injection leaves the e- spectrum identically zero, which
    trips the build-time positivity assert before any golden comparison, and a
    wrong-slice selection moves ``slice_idx``, ``gamma_cs``, ``int_m`` and
    ``grid_sol`` together: with the per-slice channel rotation each slice
    decodes its own yield matrices, so reading the yield pack from the wrong
    slice changes the assembled operator and the solved state at once. The
    flattened solved state digests per slice (measured on this machine, and
    mirrored in ``extra.slice_digest``) are 282c8794..., 935bf70e... and
    7ebc6800..., and the per-slice ``int_m`` CSR-data digests (mirrored in
    ``extra.yield_digest``) are 65e92ade..., fe830377... and 172ff289..., all
    three distinct in each set. A slice that silently fell back to a flat
    layout could not hide: there is no flat layout in the file.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import pathlib
import shutil
import tempfile

import h5py
import numpy as np

from ._harness import array_digest, make_provenance

SECTION = "emrho"

# Built from the fixture recipe on every run; raw HDF5 bytes are not a
# portable identity. Numerical and operator-digest comparisons still apply.
SYNTHETIC_DATABASES = True

#: EM channels of the rho-stack file: the fixture's minimal set plus the
#: gamma-parent channels. Without a gamma parent a gamma primary produces no
#: e± and the slice scale would be invisible; (22, 22) / (22, 11) / (22, -11)
#: give the photon an interaction matrix and (22, 211) / (22, -211) let it
#: feed the (stable here) hadron sector.
EM_CHANNELS = [
    (11, 11),
    (11, 22),
    (22, 22),
    (22, 11),
    (22, -11),
    (22, 211),
    (22, -211),
    (-11, -11),
    (-11, 22),
]

#: g/cm3, written to ``electromagnetic/air/rho_grid``.
RHO_GRID = [1e-6, 1e-3, 1e-2]

#: cs-table scale factor of each slice; what differs between the slices
#: through the cs path (the yield pack differs too, by the per-slice channel
#: rotation — see :func:`build_rho_stack_em`).
SLICE_SCALES = [0.5, 2.0, 8.0]

#: ``(em_air_density, expected slice)`` per case; closest in log10.
CASES = ((1.2e-6, 0), (9e-4, 1), (8e-3, 2))

INT_GRID = [10.0, 50.0]
DX_MAX = 20.0

#: The expected e- peak magnitudes validated interactively for slices 0/1/2
#: are ~1.6e-7, 6.6e-7 and 2.6e-6 — this section asserts positivity only and
#: pins the numbers through the compared arrays, never through a literal.
E_PEAK_ORDER = (1.6e-7, 6.6e-7, 2.6e-6)

TOLERANCES = {
    key: {"mode": "rel_l2", "rtol": 1e-11}
    for key in ("grid_sol", "emin_spectrum", "gamma_cs")
}
TOLERANCES["yield_operator"] = {"mode": "rel_l2", "rtol": 1e-11}


def _fixture_builder_module():
    """``tests/data/make_hdf5_fixtures.py``, imported by file path.

    Same idiom as the ``hdf5_fixture_dbs`` fixture in ``tests/conftest.py``:
    ``tests/data`` is not an importable package.
    """
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "data" / "make_hdf5_fixtures.py"
    )
    spec = importlib.util.spec_from_file_location("make_hdf5_fixtures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_rho_stack_em(path, scales, rho_grid):
    """Write a rho-stacked EM database and return its path.

    ``build_em_database`` writes only the flat layout, so the stack is built
    here, but the pack and cs encodings come from the fixture module
    (``_write_pack`` / ``em_cs_table`` / ``EM_CS_PARENTS``) so this file is
    encoded exactly like the databases the decode tests validate. Slice ``i``
    is ``electromagnetic/air/rho_{i:02d}`` and carries ``scales[i]`` in its
    ``cs`` table. There is deliberately no flat ``emca_mats``/``cs`` on the
    parent group — see the module docstring for what that pins.

    Slice ``i`` writes the channel list ROTATED LEFT by ``i``
    (``EM_CHANNELS[i:] + EM_CHANNELS[:i]``), which makes the stored
    ``emca_mats`` payload distinct per slice by construction: ``channel_block``
    scales every matrix by the channel's INDEX in the pack, so a rotated
    order decodes to different yield matrices even though the written channel
    multiset is unchanged. The species system, the energy dim, and the cs
    coupling story are therefore identical in every slice — only the decoded
    matrices move, which is what makes a wrong-slice read on the yield path
    observable. Slice 0 is written unrotated, so it reproduces the
    unrotated digests exactly.
    """
    F = _fixture_builder_module()
    path = pathlib.Path(path)
    with h5py.File(path, "w") as db:
        for idx, scale in enumerate(scales):
            group = db.create_group(f"electromagnetic/air/rho_{idx:02d}")
            channels = EM_CHANNELS[idx:] + EM_CHANNELS[:idx]
            F._write_pack(group, "emca_mats", channels, 1, 2, None)
            cs = group.create_dataset("cs", data=F.em_cs_table(scale))
            cs.attrs["projectiles"] = np.array(F.EM_CS_PARENTS, dtype=np.int64)
        db.create_dataset(
            "electromagnetic/air/rho_grid", data=np.asarray(rho_grid, dtype=np.float64)
        )
    return path


def build():
    """Produce (arrays, provenance) for the rho-slice EM section."""
    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.data import em_tables, hdf5_store

    # Snapshot everything this section touches: the config globals it pins
    # (MCEqRun additionally forces muon_helicity_dependence off while
    # enable_em is on, so that one is saved and restored too) and adv_set,
    # deep-copied because entries are mutated in place — same discipline as
    # the other generators, which a full pytest session relies on.
    saved = {
        key: getattr(config, key)
        for key in (
            "debug_level",
            "enable_em",
            "enable_em_ion",
            "enable_energy_loss",
            "interaction_medium",
            "data_dir",
            "mceq_db_fname",
            "em_db_fname",
            "em_air_density",
            "em_standalone_grid",
            "muon_helicity_dependence",
        )
    }
    saved_adv = copy.deepcopy(config.adv_set)
    workdir = tempfile.mkdtemp(prefix="mceq_golden_emrho_")
    arrays = {}
    slice_digests = {}
    yield_digests = {}
    yield_operators = []
    try:
        F = _fixture_builder_module()
        had_path = str(
            F.build_database(pathlib.Path(workdir) / "fixture_1d.h5", tuple_width=4)
        )
        em_path = str(
            build_rho_stack_em(
                pathlib.Path(workdir) / "em_rho_stack.h5", SLICE_SCALES, RHO_GRID
            )
        )

        config.debug_level = 0
        config.enable_em = True
        config.enable_em_ion = True
        config.enable_energy_loss = True
        config.interaction_medium = "air"
        # Empty data_dir: both filenames below are absolute temp paths, so
        # the join in db_stanza / HDF5Backend resolves them as they are.
        config.data_dir = pathlib.Path("")
        config.mceq_db_fname = had_path
        config.em_db_fname = em_path
        # em_standalone_grid must stay off: True would read the energy grid
        # from the EM file, which a rho-stack does not carry (no /common
        # group there at all) — another reason the flag is pinned.
        config.em_standalone_grid = False
        # 3112 is required: the fixture's positive-control channel
        # (-211, 3112) puts a Sigma- on the unstable list with no decay
        # pack of its own and MCEqRun.__init__ raises
        # 'Unstable particle without decay distribution' unless it is
        # disabled. The muons go with it: pi -> mu nu would otherwise
        # drag muon states into the system this section does not need.
        config.adv_set["disabled_particles"] = [13, -13, 3112]

        store = hdf5_store.HDF5Store(em_path)
        grid_sol_rows = []
        emin_rows = []
        gamma_cs_rows = []
        slice_idx = []
        for density, expected in CASES:
            idx = em_tables.select_em_rho_slice(store, "air", density)
            assert idx == expected, (
                f"rho-slice selection for {density:g} g/cm3 gave {idx}, "
                f"expected {expected}"
            )
            slice_idx.append(idx)

            config.em_air_density = density
            mceq = MCEqRun(
                interaction_model="SIBYLL21",
                theta_deg=0.0,
                primary_model=None,
            )
            try:
                mceq.set_single_primary_particle(E=30.0, pdg_id=22)
                mceq.solve(int_grid=INT_GRID, dX_max=DX_MAX)

                # sha256 over the contiguous bytes of the assembled
                # operator's CSR data buffer: with the per-slice channel
                # rotation this is the digest of the yield-slice selection
                # itself, not of a slice-invariant payload.
                yield_operators.append(mceq.int_m.toarray())
                int_m_data = np.ascontiguousarray(mceq.int_m.data)
                yield_digests[f"slice{idx}"] = hashlib.sha256(
                    int_m_data.tobytes()
                ).hexdigest()

                grid = np.asarray(mceq.grid_sol)
                pman = mceq.pman
                assert pman.dim == 20, (
                    f"energy dim {pman.dim}, expected 20: the fixture grid "
                    "is 1-100 GeV at 10/decade with no cuts applied"
                )
                assert pman.dim_states == pman.n_cparticles * pman.dim
                assert grid.shape == (len(INT_GRID), pman.dim_states), (
                    f"grid_sol is {grid.shape}, expected "
                    f"{(len(INT_GRID), pman.dim_states)}"
                )
                assert np.all(np.isfinite(grid)), "grid_sol not finite"
                emin = np.asarray(mceq.get_solution("e-", grid_idx=1))
                assert emin.size > 0 and np.any(emin > 0.0), (
                    "e- spectrum at grid_idx 1 carries no positive entry: the "
                    "EM injection did not happen or the slice has no "
                    "photon-initiated pair"
                )
                # The cs_db call of InteractionCrossSections.load resolved
                # for this run: the gamma row is the one read from the
                # selected slice's cs table, so this pins the coupling
                # between the slice and the cross sections in play.
                cs_gamma = np.asarray(mceq._int_cs.index_d[22])
                assert cs_gamma.size == emin.size

                slice_digests[f"slice{idx}"] = array_digest(grid.ravel())
                grid_sol_rows.append(grid.ravel())
                emin_rows.append(emin)
                gamma_cs_rows.append(cs_gamma)
            finally:
                mceq.close()

        assert len(slice_digests) == len(CASES)
        assert len({*slice_digests.values()}) == len(CASES), (
            f"solved states coincide across slices: {slice_digests}"
        )
        assert len({*yield_digests.values()}) == len(CASES), (
            f"assembled operators coincide across slices, so the yield-slice "
            f"selection is not observable: {yield_digests}"
        )

        arrays["grid_sol"] = np.vstack(grid_sol_rows)
        arrays["emin_spectrum"] = np.vstack(emin_rows)
        arrays["gamma_cs"] = np.vstack(gamma_cs_rows)
        arrays["slice_idx"] = np.asarray(slice_idx, dtype=np.int64)
        arrays["yield_operator"] = np.stack(yield_operators)
        arrays["yield_nonzero"] = arrays["yield_operator"] != 0

        provenance = make_provenance(
            SECTION,
            note=(
                "EM transport on a synthetic rho-stack: hadronic fixture"
                " (make_hdf5_fixtures.build_database, tuple_width 4) plus a"
                " 3-slice EM file whose cs tables carry scales 0.5/2.0/8.0 and"
                " whose channel packs are the same multiset rotated left by the"
                " slice index, so each slice decodes distinct yield matrices."
                " Densities 1.2e-6/9e-4/8e-3 g/cm3 select slices 0/1/2"
                " (closest in log10, pinned by slice_idx); gamma primary at"
                " 30 GeV, solve on int_grid [10, 50] g/cm2 requesting dX_max 20; the assembled loss-stencil guard can tighten this ceiling."
                " gamma_cs pins the cs-to-slice coupling directly, yield_operator"
                " pins the assembled operator against the yield-slice selection,"
                " and grid_sol and emin_spectrum pin that the solved dynamics"
                " move with both. Numerically strict (rel-L2 1e-11; measured"
                " run-to-run drift is zero, both within one process and across"
                " fresh ones) because the whole fixture is analytic — no"
                " real-physics noise to absorb. Per-slice solved-state digests"
                " are in extra.slice_digest and the per-slice int_m CSR-data"
                " digests in extra.yield_digest; the three differ in each set,"
                " so a wrong-slice mutant moves int_m, the solved state and"
                " yield_operator together and cannot pass, and a dropped EM"
                " injection trips the e- positivity assert at build time."
            ),
            tolerances=TOLERANCES,
            extra={
                "rho_grid": list(RHO_GRID),
                "slice_scales": list(SLICE_SCALES),
                "densities": [density for density, _ in CASES],
                "expected_slices": [expected for _, expected in CASES],
                "slice_digest": slice_digests,
                "yield_digest": yield_digests,
                "em_channels": [list(ch) for ch in EM_CHANNELS],
                "e_peak_order": list(E_PEAK_ORDER),
            },
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        for key, value in saved.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv)

    dbs = provenance["databases"]
    for key in ("mceq_db_fname", "em_db_fname"):
        assert "sha256" in dbs.get(key, {}), (
            f"fixture {key} not resolvable, provenance incomplete: {dbs.get(key)}"
        )
    return arrays, provenance
