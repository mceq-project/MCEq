"""Golden section for the data layer: species table, production modifiers,
channel packs, assembled operators and the DDM goldens.

Fixture: the 1D SIBYLL21 reduced database under the ``mceq_sib21`` config of
tests/conftest.py — EM cascade enabled (``disabled_particles = []``), muon
helicity dependence on. 72 cascade species, 0 resonances, dim 31,
dim_states 2232.

Plan section 8.1 asks for the ``[(name, mceqidx)]`` species table, the
``mod_pprod`` keys for primaries 2212/2112 and the DDM goldens; the Phase 4
acceptance adds CSR-pack checksums and a float32 dtype test. The section
covers them in four layers, so a Phase 4 failure localises to one of them:

``raw/``
    the CSR packs as they sit in HDF5 — the ``(2, len_data)`` data/column
    rows, the per-channel indptrs and the ``tuple_idcs`` / ``len_data``
    attributes. This is the input side of the planned
    ``HDF5Store.read_channel_pack``, and it catches a wrong node path before
    the decode digest runs. The decay dataset name is read from
    ``Decays._default_decay_dset`` rather than assumed: data.py:1244 picks
    ``polarized`` or ``unpolarized`` from ``config.muon_helicity_dependence``,
    and a raw digest taken over the other dataset would silently describe data
    the run never decoded.

``channels/``
    all five keys ``_gen_db_dictionary`` (data.py:302) returns — parents,
    particles, relations, index_d and description — for interactions and
    decays, plus the cross-section and continuous-loss maps. Parent order is
    load-bearing (``Interactions.load`` filters the list in place and
    ``Decays.load(parent_list=...)`` consumes it) and description is the only
    provenance the decoded pack carries, so a rewrite that reorders parents or
    drops description fails here rather than passing on an index_d digest.

``operators/``
    int_m and dec_m as CSR buffer digests plus per-species reductions. The
    matrices are 2232x2232 here and 176256x176256 with 2.1e7 nonzeros in 2D,
    so they are digested, never stored; the row-block sums and nonzero counts
    make a digest mismatch localisable to a species instead of a bare boolean.

``ddm/``
    the two z-factors of the (2212, 211, ebeam=158.0) entry, the spline entry
    behind them, the 17 data-driven matrices and the operators after
    ``inject_ddm``.

Config sensitivity: the species table depends on two knobs the conftest
fixture moves away from the production default, so each variant carries its
own digest under ``sweep/``. Measured on this database and model:
``disabled_particles=[]`` gives 72 species / dim_states 2232, ``[11,-11]``
gives 66 / 2046, and ``muon_helicity_dependence=False`` gives 54 / 1674 and
switches the decay dataset to ``unpolarized``. ``config.floatlen =
np.float32`` leaves the table identical but propagates to index_d and to the
int_m/dec_m dtype; since the digests carry ``dtype.str``, that sweep gets its
own keys under ``f32/`` rather than a loosened tolerance.

Two bugs are pinned as they behave today, to be fixed in their own commits
after the behaviour is frozen:

B1 (data.py:1082-1085, plan section 9)
    ``prim_pdg, symm_pdg = 2212, 2112`` overwrites the caller's primary and
    the next line tests the overwritten variable, so the neutron branch is
    dead and a 2112 primary is treated as a proton. It is numerically
    observable, not just a mislabelled key: ``set_mod_pprod(2112, 321, ...)``
    files both the direct and the isospin modification under ``(2112, 321)``,
    and ``get_matrix`` (data.py:1215-1224) multiplies every entry of a key in,
    so n -> K+ is scaled by 1.45^2 = 2.1025 while (2212, 321) is never
    modified at all.

The eta/omega/phi coupling (data.py:1095)
    ``np.any([p in self.parents for p in [221, 223, 333]])`` tests bare PDG
    ids against ``Interactions.parents``, which holds ``(pdg, helicity)``
    tuples, so the branch is unreachable for any database and no unflavoured
    key is ever written. ``mod_pprod/unflavoured_membership`` pins the three
    membership results at 0; a Phase 4 cleanup of that comparison would enable
    six new modifier keys and move fluxes.

A third defect, not in the plan's ledger: ``DataDrivenModel.e_min`` /
``e_max`` (ddm.py:712-713) are stored and never read — ``ddm_matrices``
(ddm.py:739) takes the full grid from ``mceq.e_bins`` — so the conftest
``data_driven_model`` fixture's ``e_min=5.0, e_max=500.0`` is a no-op. That
fixture also cannot be injected at all (``ddm_matrices`` asserts (2212, 321)
is present when ``enable_K0_from_isospin`` is True, which its
``enable_channels=[(2212, 211)]`` filter removes), so this section uses the
unfiltered ``DataDrivenModel()``: 13 channels, 17 matrices.

Everything is compared bitwise except the four z-factors, which come out of
adaptive quadrature (``scipy.integrate.quad``, limit 150, epsabs 1e-5) and
``jacobi.propagate``: they are bit-stable across processes today, but a scipy
or jacobi bump may legitimately move the last ulp, so they are compared at
rel-L2 1e-12 — five orders tighter than the ``np.allclose`` of
tests/test_ddm.py:292, which hides a real 6e-10 gap against its 10-digit
literals.
"""

from __future__ import annotations

import copy
import hashlib
import os

import numpy as np

from ._harness import HOST_RTOL, array_digest, make_provenance, sparse_digest

SECTION = "species"

#: Config globals this section fixes. The autouse fixture in tests/conftest.py
#: restores six of them, so a section built inside a full pytest session has to
#: set — and afterwards restore — the rest itself. MKL thread count is left
#: ambient on purpose: every digest here is unchanged at 1, 2 and 4 threads.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": "mceq_db_v140reduced_compact.h5",
    "e_min": 0.1,
    "e_max": 1e11,
    "floatlen": None,
    "interaction_medium": "air",
    "A_target": "auto",
    "enable_em": False,
    "enable_em_ion": True,
    "enable_energy_loss": True,
    "enable_cont_rad_loss": True,
    "generic_losses_all_charged": True,
    "average_loss_operator": False,
    "loss_step_for_average": 1e-1,
    "loss_stencil_method": "expfit_low_upwind2",
    "loss_stencil_low_upwind_rows": 8,
    "loss_stencil_alpha0": 3.0,
    "use_isospin_sym": True,
    "muon_helicity_dependence": True,
    "muon_multiple_scattering": True,
    "assume_nucleon_interactions_for_exotics": True,
    "enable_default_tracking": True,
    "fallback_to_air_cs": True,
    "prompt_ctau": 2.6842,
    "minimal_primary_energy": 3.0,
    "low_energy_extension": {
        "model": None,
        "he_le_transition": 80,
        "he_le_trwidth": 0.3,
        "use_unknown_cs": True,
    },
}

ADV_SET_PINS = {
    "disabled_particles": [],
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

#: ``(label, primary, secondary, value)`` for the modifier sweep: the pion,
#: kaon, leading-nucleon and charm branches of ``_set_mod_pprod``, each with a
#: proton and a neutron primary. The ``_n`` rows are where B1 shows.
MOD_PPROD_CASES = (
    ("pi+_p", 2212, 211, 0.15),
    ("pi+_n", 2112, 211, 0.15),
    ("pi-_p", 2212, -211, 0.12),
    ("K+_p", 2212, 321, 0.45),
    ("K+_n", 2112, 321, 0.45),
    ("lead_p", 2212, 2212, 0.10),
    ("lead_n", 2112, 2112, 0.10),
    ("charm_p", 2212, 411, 0.10),
    ("charm_n", 2112, 411, 0.10),
)

#: The DDM entry the plan's two literals come from (tests/test_ddm.py:292,311).
DDM_ENTRY = dict(projectile=2212, secondary=211, ebeam=158.0)

TOLERANCES = {
    "ddm/zfactors": {"mode": "rel_l2", "rtol": HOST_RTOL},
    "ddm/zfactors_tuned": {"mode": "rel_l2", "rtol": HOST_RTOL},
}


# --------------------------------------------------------------------------
# digests over the decoded maps
# --------------------------------------------------------------------------


def _particle_key(particle):
    """``(pdg, helicity)`` as plain ints; the decoded keys mix int and int64."""
    return int(particle[0]), int(particle[1])


def _channel_key(channel):
    """``((parent), (child))`` flattened to four ints."""
    parent, child = channel
    return _particle_key(parent) + _particle_key(child)


def _map_digest(mapping, key_of):
    """sha256 over a ``{key: array}`` map: normalised keys in sorted order,
    each followed by the digest of the array it carries."""
    h = hashlib.sha256()
    for key in sorted(mapping, key=key_of):
        h.update(repr(key_of(key)).encode())
        h.update(array_digest(mapping[key]).encode())
    return h.hexdigest()


def _relations_digest(relations):
    """sha256 over ``parent -> [child, ...]``, child order preserved.

    The child lists drive the membership test in ``Interactions.get_matrix``
    and the parent list handed to ``Decays.load``, so their order is part of
    the pack contract even though it never reaches an array value.
    """
    h = hashlib.sha256()
    for parent in sorted(relations, key=_particle_key):
        h.update(repr(_particle_key(parent)).encode())
        h.update(repr([_particle_key(c) for c in relations[parent]]).encode())
    return h.hexdigest()


def _species_table(pman):
    """``[(name, mceqidx)]`` in cascade order, split into a U array and an
    int64 array. ``mceqidx`` mixes int and numpy.int64 — tracking particles
    take theirs from ``np.max(...) + 1`` (particlemanager.py:955) — so the
    indices are coerced rather than stored as they come.
    """
    names = np.array([p.name for p in pman.cascade_particles])
    indices = np.array([int(p.mceqidx) for p in pman.cascade_particles], dtype=np.int64)
    return names, indices


def _species_digest(pman):
    """One digest over the whole species table, for the config sweeps."""
    names, indices = _species_table(pman)
    h = hashlib.sha256()
    h.update(array_digest(names).encode())
    h.update(array_digest(indices).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# recorders
# --------------------------------------------------------------------------


def _record_fixture(arrays, mceq):
    """Identify the tables the run actually decoded, read live."""
    from MCEq import config

    arrays["fixture/db"] = np.asarray(config.mceq_db_fname)
    arrays["fixture/model"] = np.asarray(mceq._interactions.iam)
    arrays["fixture/medium"] = np.asarray(mceq._mceq_db.medium)
    arrays["fixture/decay_dset"] = np.asarray(mceq._decays._default_decay_dset)
    arrays["fixture/floatlen"] = np.asarray(str(config.floatlen))
    arrays["fixture/is_2d"] = np.asarray(int(mceq._mceq_db.is_2d))
    arrays["fixture/n_k"] = np.asarray(int(mceq._mceq_db.n_k))


def _record_species(arrays, mceq):
    """The species table and the dimensions derived from it."""
    pman = mceq.pman
    names, indices = _species_table(pman)
    arrays["species/names"] = names
    arrays["species/mceqidx"] = indices
    arrays["species/n_cascade"] = np.asarray(int(pman.n_cparticles))
    arrays["species/n_resonances"] = np.asarray(len(pman.resonances))
    arrays["species/dim"] = np.asarray(int(pman.dim))
    arrays["species/dim_states"] = np.asarray(int(pman.dim_states))
    arrays["species/pdg2mceqidx"] = np.array(
        sorted(
            (int(pdg), int(helicity), int(idx))
            for (pdg, helicity), idx in pman.pdg2mceqidx.items()
        ),
        dtype=np.int64,
    )
    arrays["species/e_grid"] = np.asarray(mceq._energy_grid.c)
    arrays["species/e_bins"] = np.asarray(mceq._energy_grid.b)


def _record_sweep(arrays, prefix, mceq):
    """Species digest and counts for one config variant."""
    arrays[prefix + "/species_digest"] = np.asarray(_species_digest(mceq.pman))
    arrays[prefix + "/n_cascade"] = np.asarray(int(mceq.pman.n_cparticles))
    arrays[prefix + "/dim_states"] = np.asarray(int(mceq.pman.dim_states))


def _record_raw_pack(arrays, prefix, db_path, group, dataset):
    """Digest one CSR pack as stored in HDF5, before any decode."""
    import h5py

    with h5py.File(db_path, "r") as db:
        node, indptrs = db[group][dataset], db[group][dataset + "_indptrs"]
        arrays[prefix + "/node"] = np.asarray(f"{group}/{dataset}")
        arrays[prefix + "/file_version"] = np.asarray(str(db.attrs.get("version", "?")))
        arrays[prefix + "/mat_data_shape"] = np.asarray(node.shape)
        arrays[prefix + "/mat_data"] = np.asarray(array_digest(node[:, :]))
        arrays[prefix + "/indptrs"] = np.asarray(array_digest(indptrs[:]))
        arrays[prefix + "/tuple_idcs"] = np.asarray(
            array_digest(node.attrs["tuple_idcs"])
        )
        arrays[prefix + "/len_data"] = np.asarray(array_digest(node.attrs["len_data"]))


def _record_channel_pack(arrays, prefix, table):
    """Store all five products of the pack decode for one table."""
    arrays[prefix + "/parents"] = np.array(
        [_particle_key(p) for p in table.parents], dtype=np.int64
    )
    arrays[prefix + "/particles"] = np.array(
        [_particle_key(p) for p in table.particles], dtype=np.int64
    )
    arrays[prefix + "/relations"] = np.asarray(_relations_digest(table.relations))
    arrays[prefix + "/index_d"] = np.asarray(_map_digest(table.index_d, _channel_key))
    arrays[prefix + "/n_channels"] = np.asarray(len(table.index_d))
    arrays[prefix + "/description"] = np.asarray(str(table.description))
    channel = next(iter(table.index_d.values()))
    arrays[prefix + "/channel_dtype"] = np.asarray(channel.dtype.str)
    arrays[prefix + "/channel_shape"] = np.asarray(np.shape(channel))


def _record_operator(arrays, prefix, matrix, n_species):
    """CSR buffer digests plus per-species reductions of an assembled operator.

    The state vector is laid out species-major, so folding the row sums and
    row nonzero counts into ``(n_species, dim)`` blocks names the species a
    digest mismatch comes from.
    """
    digest = sparse_digest(matrix)
    arrays[prefix + "/shape"] = np.asarray(digest["shape"])
    arrays[prefix + "/nnz"] = np.asarray(digest["nnz"])
    arrays[prefix + "/dtype"] = np.asarray(digest["dtype"])
    for part in ("data", "indices", "indptr"):
        arrays[prefix + "/" + part] = np.asarray(digest[part])

    csr = matrix.tocsr()
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    arrays[prefix + "/row_sums_by_species"] = row_sums.reshape(n_species, -1).sum(
        axis=1
    )
    arrays[prefix + "/nnz_by_species"] = (
        np.diff(csr.indptr).reshape(n_species, -1).sum(axis=1).astype(np.int64)
    )


def _weight(xmat, egrid, arg_name, value):
    """The modification function of tests/test_core.py::test_set_mod_pprod: a
    flat ``1 + value`` scaling of the whole x distribution."""
    return (1.0 + value) * np.ones_like(xmat)


def _mod_pprod_entries(mod_pprod):
    """``"prim,sec:func:arg=value"`` for every registered modification.

    ``mod_pprod`` is a ``defaultdict(dict)``: indexing a key inserts it (as
    tests/test_core.py:551 does), so the store is only ever iterated.
    """
    return sorted(
        f"{int(prim)},{int(sec)}:{func_name}:{args[0]}={args[1]:g}"
        for (prim, sec), entries in mod_pprod.items()
        for func_name, args in entries
    )


def _mod_pprod_digest(mod_pprod):
    """sha256 over the store including the modification matrices themselves."""
    h = hashlib.sha256()
    for key, entries in sorted(
        mod_pprod.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))
    ):
        h.update(repr((int(key[0]), int(key[1]))).encode())
        for entry_key, kmat in sorted(entries.items(), key=lambda kv: repr(kv[0])):
            func_name, args = entry_key
            h.update(repr((str(func_name), str(args[0]), float(args[1]))).encode())
            h.update(array_digest(kmat).encode())
    return h.hexdigest()


def _record_mod_pprod(arrays, mceq):
    """The modifier keys and the factor they apply, for both nucleon primaries.

    The applied factor is ``get_matrix`` with the modification divided by
    ``get_matrix`` without it, over the nonzero entries; it is NaN for a
    channel the model does not carry (charm: ``2212 -> 411`` raises "trying to
    get empty matrix"), so an unexpected failure elsewhere cannot pass as a
    silently shrunken golden.
    """
    interactions = mceq._interactions
    n_cases = len(MOD_PPROD_CASES)
    returned = np.zeros(n_cases, dtype=np.int64)
    n_keys = np.zeros(n_cases, dtype=np.int64)
    factors = np.full((n_cases, 2), np.nan)

    for i, (label, prim, sec, value) in enumerate(MOD_PPROD_CASES):
        mceq.unset_mod_pprod(dont_fill=True)
        has_channel = (sec, 0) in interactions.relations.get((prim, 0), [])
        base = (
            np.copy(interactions.get_matrix((prim, 0), (sec, 0)))
            if has_channel
            else None
        )

        returned[i] = mceq.set_mod_pprod(prim, sec, _weight, ("a", value))
        arrays[f"mod_pprod/{label}/keys"] = np.array(
            sorted(
                f"{int(prim_pdg)},{int(sec_pdg)}"
                for prim_pdg, sec_pdg in interactions.mod_pprod
            )
        )
        arrays[f"mod_pprod/{label}/entries"] = np.array(
            _mod_pprod_entries(interactions.mod_pprod)
        )
        arrays[f"mod_pprod/{label}/digest"] = np.asarray(
            _mod_pprod_digest(interactions.mod_pprod)
        )
        n_keys[i] = len(interactions.mod_pprod)

        if has_channel:
            modified = interactions.get_matrix((prim, 0), (sec, 0))
            nonzero = base > 0
            ratio = modified[nonzero] / base[nonzero]
            factors[i] = (np.min(ratio), np.max(ratio))

    mceq.unset_mod_pprod(dont_fill=True)

    arrays["mod_pprod/labels"] = np.array([label for label, _, _, _ in MOD_PPROD_CASES])
    arrays["mod_pprod/calls"] = np.array(
        [(prim, sec) for _, prim, sec, _ in MOD_PPROD_CASES], dtype=np.int64
    )
    arrays["mod_pprod/arg_values"] = np.array(
        [value for _, _, _, value in MOD_PPROD_CASES], dtype=np.float64
    )
    arrays["mod_pprod/set_returned"] = returned
    arrays["mod_pprod/n_keys"] = n_keys
    arrays["mod_pprod/applied_factor"] = factors
    # data.py:1095 verbatim: bare PDG ids against a list of (pdg, helicity)
    # tuples, so the unflavoured coupling never fires.
    arrays["mod_pprod/unflavoured_membership"] = np.array(
        [int(pdg in interactions.parents) for pdg in (221, 223, 333)], dtype=np.int64
    )


def _record_ddm(arrays, mceq, n_species):
    """The z-factors, the spline entry behind them, and the DDM matrices.

    ``inject_ddm`` rebuilds int_m and dec_m, so this runs last on the fixture.
    """
    from MCEq import ddm

    full = ddm.DataDrivenModel()
    entry = full.spline_db.get_entry(**DDM_ENTRY)
    z_jacobi, err_jacobi = entry.calc_zfactor_and_error()
    z_quad, err_quad = entry.calc_zfactor_and_error2()
    arrays["ddm/zfactors"] = np.array(
        [z_jacobi, err_jacobi, z_quad, err_quad], dtype=np.float64
    )
    arrays["ddm/entry/ebeam"] = np.asarray(str(entry.ebeam))
    arrays["ddm/entry/scalars"] = np.array(
        [entry.fl_ebeam, entry.x_min, entry.tv, entry.te], dtype=np.float64
    )
    arrays["ddm/entry/ints"] = np.array(
        [entry.n_knots, entry.spl_idx, entry.tck[2], int(entry.x17)], dtype=np.int64
    )
    arrays["ddm/entry/knots"] = np.asarray(entry.tck[0], dtype=np.float64)
    arrays["ddm/entry/coefficients"] = np.asarray(entry.tck[1], dtype=np.float64)
    arrays["ddm/entry/cov"] = np.asarray(entry.cov, dtype=np.float64)

    arrays["ddm/channels"] = np.array(
        sorted(
            (int(channel.projectile), int(channel.secondary))
            for channel in full.spline_db.channels
        ),
        dtype=np.int64,
    )

    matrices = full.ddm_matrices(mceq)
    keys = sorted(matrices, key=lambda k: (int(k[0]), int(k[1])))
    arrays["ddm/matrix_keys"] = np.array([f"{int(p)},{int(s)}" for p, s in keys])
    arrays["ddm/matrix_digests"] = np.array([array_digest(matrices[k]) for k in keys])
    arrays["ddm/matrix_sums"] = np.array(
        [matrices[k].sum() for k in keys], dtype=np.float64
    )
    arrays["ddm/matrix_shape"] = np.asarray(np.shape(matrices[keys[0]]))

    mceq.inject_ddm(full)
    _record_operator(arrays, "ddm/int_m", mceq.int_m, n_species)
    _record_operator(arrays, "ddm/dec_m", mceq.dec_m, n_species)

    # A second model for the tuning: the first DDMSplineDB built in a process
    # aliases the class-level ``_ddm_splines`` dict (ddm.py:460), and
    # apply_tuning mutates its entry in place; a later instance reloads from
    # file, so tuning here leaves nothing behind for other golden sections.
    tuned = ddm.DataDrivenModel()
    tuned.apply_tuning(**DDM_ENTRY, tv=0.5, te=0.8)
    tuned_entry = tuned.spline_db.get_entry(**DDM_ENTRY)
    arrays["ddm/zfactors_tuned"] = np.array(
        tuned_entry.calc_zfactor_and_error2(), dtype=np.float64
    )


def _record_float32(arrays, mceq, n_species):
    """The float32 sweep: dtype propagates from ``config.floatlen`` into the
    decoded channels and the assembled operators, while the species table is
    unchanged. Digests carry ``dtype.str``, so these are separate keys."""
    from MCEq import config

    arrays["f32/floatlen"] = np.asarray(str(config.floatlen))
    arrays["f32/species_digest"] = np.asarray(_species_digest(mceq.pman))
    for name, table in (("interactions", mceq._interactions), ("decays", mceq._decays)):
        channel = next(iter(table.index_d.values()))
        arrays[f"f32/{name}/channel_dtype"] = np.asarray(channel.dtype.str)
        arrays[f"f32/{name}/index_d"] = np.asarray(
            _map_digest(table.index_d, _channel_key)
        )
    _record_operator(arrays, "f32/int_m", mceq.int_m, n_species)
    _record_operator(arrays, "f32/dec_m", mceq.dec_m, n_species)


def _mceq_run():
    """The conftest ``mceq_sib21`` run, built under the config in force."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    return MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


def build():
    """Produce (arrays, provenance) for the data-layer section."""
    from MCEq import config

    missing = sorted(k for k in CONFIG_PINS if not hasattr(config, k))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    saved_config = {k: getattr(config, k) for k in CONFIG_PINS}
    saved_adv_set = copy.deepcopy(config.adv_set)
    # DDMSplineDB._ddm_splines is a class attribute, and the first instance in a
    # process writes into the class dict rather than an instance one
    # (ddm.py:460, 503-504), so building a DataDrivenModel here leaks 13 channel
    # entries into every later test in the same session.
    from MCEq.ddm import DDMSplineDB

    saved_ddm_splines = dict(DDMSplineDB._ddm_splines)
    arrays = {}
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)
        config.adv_set.update(ADV_SET_PINS)
        db_path = os.path.join(config.data_dir, config.mceq_db_fname)

        mceq = _mceq_run()
        try:
            n_species = int(mceq.pman.n_cparticles)
            _record_fixture(arrays, mceq)
            _record_species(arrays, mceq)
            _record_sweep(arrays, "sweep/em_on", mceq)
            _record_raw_pack(
                arrays,
                "raw/interactions",
                db_path,
                f"hadronic_interactions/{mceq._mceq_db.medium}",
                mceq._interactions.iam,
            )
            _record_raw_pack(
                arrays,
                "raw/decays",
                db_path,
                "decays",
                mceq._decays._default_decay_dset,
            )
            _record_channel_pack(arrays, "channels/interactions", mceq._interactions)
            _record_channel_pack(arrays, "channels/decays", mceq._decays)
            arrays["channels/cross_sections"] = np.asarray(
                _map_digest(mceq._int_cs.index_d, int)
            )
            arrays["channels/n_cross_sections"] = np.asarray(len(mceq._int_cs.index_d))
            arrays["channels/cont_losses"] = np.asarray(
                _map_digest(mceq._cont_losses.index_d, _particle_key)
            )
            arrays["channels/n_cont_losses"] = np.asarray(
                len(mceq._cont_losses.index_d)
            )
            _record_operator(arrays, "operators/int_m", mceq.int_m, n_species)
            _record_operator(arrays, "operators/dec_m", mceq.dec_m, n_species)
            _record_mod_pprod(arrays, mceq)
            _record_ddm(arrays, mceq, n_species)
        finally:
            mceq.close()

        config.adv_set["disabled_particles"] = [11, -11]
        mceq = _mceq_run()
        try:
            _record_sweep(arrays, "sweep/em_disabled", mceq)
        finally:
            mceq.close()
        config.adv_set["disabled_particles"] = ADV_SET_PINS["disabled_particles"]

        config.muon_helicity_dependence = False
        mceq = _mceq_run()
        try:
            _record_sweep(arrays, "sweep/helicity_off", mceq)
            decay_dset = mceq._decays._default_decay_dset
            arrays["sweep/helicity_off/decay_dset"] = np.asarray(decay_dset)
            arrays["sweep/helicity_off/decays_index_d"] = np.asarray(
                _map_digest(mceq._decays.index_d, _channel_key)
            )
            _record_raw_pack(
                arrays, "sweep/helicity_off/raw_decays", db_path, "decays", decay_dset
            )
        finally:
            mceq.close()
        config.muon_helicity_dependence = CONFIG_PINS["muon_helicity_dependence"]

        config.floatlen = np.float32
        mceq = _mceq_run()
        try:
            _record_float32(arrays, mceq, int(mceq.pman.n_cparticles))
        finally:
            mceq.close()
        config.floatlen = CONFIG_PINS["floatlen"]

        provenance = make_provenance(
            SECTION,
            note=(
                "Data layer on the 1D SIBYLL21 reduced DB with the conftest"
                " mceq_sib21 config (disabled_particles [], muon helicity on):"
                " species table, raw and decoded channel packs, assembled"
                " operators, production modifiers and DDM. Bug B1"
                " (data.py:1082-1085, neutron primary treated as a proton) is"
                " pinned as it behaves today and fixed in its own commit later;"
                " mod_pprod/K+_n therefore carries the same modification twice"
                " and applies 1.45^2. mod_pprod/unflavoured_membership pins the"
                " unreachable eta/omega/phi branch (data.py:1095 tests bare PDG"
                " ids against (pdg, helicity) tuples). DataDrivenModel.e_min /"
                " e_max (ddm.py:712-713) are stored and never read — ddm_matrices"
                " takes the full grid from mceq.e_bins — so no DDM key here"
                " depends on them. Bitwise everywhere except ddm/zfactors and"
                " ddm/zfactors_tuned, which come from adaptive quadrature and"
                " jacobi.propagate and are compared at rel-L2 1e-12."
            ),
            tolerances=TOLERANCES,
            extra={
                "mod_pprod_cases": [list(case) for case in MOD_PPROD_CASES],
                "ddm_entry": DDM_ENTRY,
                "species_digest": str(arrays["sweep/em_on/species_digest"]),
                "int_m_digest": str(arrays["operators/int_m/data"]),
                "sweeps": {
                    "em_on": "conftest mceq_sib21: disabled_particles []",
                    "em_disabled": "production default: disabled_particles [11, -11]",
                    "helicity_off": "muon_helicity_dependence False -> unpolarized decays",
                    "f32": "config.floatlen = numpy.float32",
                },
            },
        )
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)
        DDMSplineDB._ddm_splines.clear()
        DDMSplineDB._ddm_splines.update(saved_ddm_splines)

    db = provenance["databases"].get("mceq_db_fname", {})
    assert "sha256" in db, f"reduced DB not resolvable, provenance incomplete: {db}"

    return arrays, provenance
