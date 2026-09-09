"""Model-specific projectile equivalences: the table and the three uses of it.

A database stores production matrices for the projectiles its generator
actually simulated; every other species has to borrow a column. This module
holds the borrowing rules:

* :data:`equivalences`, the per-family ``child -> stand-in`` table;
* :func:`reverse_equivalences` and :func:`apply_equivalences`, the inverse
  lookup and the copy that ``HDF5Backend._gen_db_dictionary`` performs per
  channel when ``assume_nucleon_interactions_for_exotics`` is set;
* :func:`mapped_cross_section` and :func:`family_fallback`, the cross-section
  side.

**Two object identities must be preserved.**

1. ``equivalences["FLUKA"] is equivalences["DPMJET"]`` -- the same dict, not a
   copy, so any edit to one family is seen by the other.
2. :func:`apply_equivalences` aliases rather than copies: the borrowing
   projectile gets *the same* matrix object as its stand-in, and *the same*
   ``relations`` list, which later iterations of the caller's channel loop keep
   appending to. Callers must therefore account for mutations reached through
   either name.

Both are checked in ``tests/test_data_hdf5_decode.py``
(``test_equivalence_injection_aliases_rather_than_copies``,
``test_fluka_and_dpmjet_share_one_equivalence_table``).

**Importing from this module: use ``from … import``, never ``import … as``.**
``MCEq.data`` re-exports the :data:`equivalences` *dict* under the same name as
this submodule, and the re-export runs after the import machinery has bound the
module, so the dict wins: ``MCEq.data.equivalences`` **is the dict**. CPython
resolves ``import MCEq.data.equivalences as x`` by ``getattr``, so that binds
the dict too, and ``MCEq.data.equivalences.family_fallback`` raises
``AttributeError: 'dict' object has no attribute 'family_fallback'`` even after
a plain ``import MCEq.data.equivalences``. Only
``from MCEq.data.equivalences import family_fallback`` (or a ``sys.modules``
lookup) reaches the functions. The dict has to win -- it is the exported public
name (``tests/test_phase4_surface.py``).

No config, no HDF5, no numpy: the module reads only :mod:`MCEq.data.model_names`
and ``misc.info``.
"""

from collections import defaultdict

from MCEq.data.model_names import family_of
from MCEq.misc import info

equivalences = {
    "SIBYLL23": {
        -4132: 4122,
        -4122: 4122,
        -3334: -3312,
        -3322: -2112,
        -3212: -3122,
        -413: -411,
        113: 211,
        221: 211,
        111: 211,
        310: 130,
        413: 411,
        3212: 3122,
        3334: 3312,
    },
    "SIBYLL21": {
        -3322: 2112,
        -3312: 2212,
        -3222: 2212,
        -3212: 2112,
        -3122: 2112,
        -3112: 2212,
        -2212: 2212,
        -2112: 2112,
        310: 130,
        111: 211,
        3112: 2212,
        3122: 2112,
        3212: 2112,
        3222: 2212,
        3312: 2212,
        3322: 2112,
    },
    "QGSJET01": {
        -4122: 2212,
        -3322: 2212,
        -3312: 2212,
        -3222: 2212,
        -3122: 2212,
        -3112: 2212,
        -2212: 2212,
        -2112: 2212,
        -421: 321,
        -411: 321,
        -211: 211,
        -321: 321,
        111: 211,
        221: 211,
        130: 321,
        310: 321,
        411: 321,
        421: 321,
        2112: 2212,
        3112: 2212,
        3122: 2212,
        3222: 2212,
        3312: 2212,
        3322: 2212,
        4122: 2212,
    },
    "QGSJETII": {
        -3122: -2112,
        111: 211,
        113: 211,
        221: 211,
        310: 130,
        3122: 2112,
    },
    "DPMJET": {
        -4122: -3222,
        -3334: -3312,
        -3212: -3122,
        -431: -321,
        -421: -321,
        -413: -321,
        -411: -321,
        310: 130,
        113: 211,
        221: 211,
        111: 211,
        411: 321,
        413: 321,
        421: 321,
        431: 321,
        3212: 3122,
        3334: 3312,
        4122: 3222,
    },
    "EPOSLHC": {
        -3334: 2212,
        -3322: -3122,
        -3312: 2212,
        -3222: -2212,
        -3212: -3122,
        -3112: 2212,
        111: 211,
        113: 211,
        221: 211,
        310: 130,
        3112: -2212,
        3212: 3122,
        3222: 2212,
        3312: -2212,
        3322: 3122,
        3334: -2212,
    },
    "PYTHIA8": {
        -3122: -2112,
        -431: -321,
        -421: -321,
        -413: -321,
        -411: -321,
        111: 211,
        113: 211,
        221: 211,
        310: 321,
        130: 321,
        411: 321,
        413: 321,
        421: 321,
        431: 321,
        3122: 2112,
    },
}

equivalences["FLUKA"] = equivalences["DPMJET"]


def reverse_equivalences(equivalences):
    """``{child: stand_in}`` -> ``{(stand_in, 0): [(child, 0), ...]}``.

    A ``defaultdict(list)`` on purpose: the caller indexes it with every parent
    it decodes, most of which stand in for nothing, and relies on the empty
    list rather than guarding the lookup.
    """
    eqv_lookup = defaultdict(list)
    for k in equivalences:
        eqv_lookup[(equivalences[k], 0)].append((k, 0))
    return eqv_lookup


def apply_equivalences(
    eqv_lookup,
    parent_pdg,
    child_pdg,
    model_particles,
    available_parents,
    particle_list,
    index_d,
    relations,
):
    """Give missing projectiles the simulated parent's channel matrix and list.

    ``parent_pdg`` is the simulated representative; ``eqv_lookup`` lists the
    projectiles that borrow from it.

    Called once per decoded channel by ``HDF5Backend._gen_db_dictionary``, and
    only when ``physics.assume_nucleon_interactions_for_exotics`` is set -- the
    switch stays at the call site so this module reads no configuration.

    Mutates ``particle_list``, ``index_d`` and ``relations`` in place, and
    **aliases**: the borrowing projectile shares the matrix object and the
    ``relations`` list of ``parent_pdg`` (see the module docstring). Returns
    nothing.
    """
    for eqv_parent in eqv_lookup[parent_pdg]:
        if eqv_parent[0] not in model_particles:
            info(
                10,
                "No equiv. replacement needed of",
                eqv_parent,
                "for",
                parent_pdg,
                "parent.",
            )
            continue
        if eqv_parent in available_parents:
            info(
                10,
                f"Parent {eqv_parent[0]} has dedicated simulation.",
            )
            continue
        particle_list.append(eqv_parent)
        index_d[(eqv_parent, child_pdg)] = index_d[(parent_pdg, child_pdg)]
        relations[eqv_parent] = relations[parent_pdg]
        _e = eqv_parent[0]
        _p = parent_pdg[0]
        info(
            15,
            f"equivalence of {_e} and {_p} interactions",
        )


def family_fallback(projectile):
    """Sign-preserving family representatives to try for ``projectile``.

    A negative projectile tries the antiparticle representative first
    (antibaryons carry annihilation channels -- nbar, lambda-bar etc. must map
    to pbar, not p; same logic for K-/pi- when the model stores dedicated
    columns) before falling back to the positive one.

    Check kaon IDs before pion IDs: ID ``130`` (K_L) satisfies both the kaon
    window (``apid in (130, 310, 311)``) and the pion window
    (``100 < apid < 300``), and keeps the kaon representative only because its
    branch is tested first. Reordering the branches would silently reroute K_L
    to the pion column.

    Anything outside all three windows gets ``[]`` -- leptons, the diffractive
    ``10313``/``10323`` ids, nuclei at or above 5000. This is one of three
    ``PDG -> family representative`` policies in the tree and is deliberately
    not unified with the other two; see
    ``tests/test_data_model_names.py::test_the_two_pdg_policies_are_not_unified``,
    which drives both real callables and fails if they are unified.
    """
    apid = abs(projectile)
    sign = -1 if projectile < 0 else 1
    if apid in (130, 310, 311) or 300 < apid < 1000:
        return [sign * 321, 321]
    if 100 < apid < 300:
        return [sign * 211, 211]
    if 1000 < apid < 5000:
        return [sign * 2212, 2212]
    return []


def mapped_cross_section(index_d, projectile, model_name):
    """Return a model cross section using the established equivalences."""
    candidates = [projectile, abs(projectile)]
    family = family_of(model_name)
    model_eqv = None if family is None else equivalences[family]
    if model_eqv is not None:
        mapped = model_eqv.get(projectile, model_eqv.get(abs(projectile)))
        if mapped is not None:
            candidates.extend([mapped, abs(mapped)])

    candidates.extend(family_fallback(projectile))

    for candidate in candidates:
        if candidate in index_d:
            return index_d[candidate]
    return None
