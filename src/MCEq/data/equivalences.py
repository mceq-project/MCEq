"""Projectile equivalences of hadronic models and cross-section mapping."""

# anfe: crazy docstring! AI talking to itself. This module should joined with the model_names to form soem kind of model "settings" or hadr_model_[xomething, missing a word]

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
    """``{child: stand_in}`` -> ``defaultdict {(stand_in, 0): [(child, 0), ...]}``."""
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
    """Give projectiles without own tables the channel of their stand-in ``parent_pdg``.

    Updates ``particle_list``, ``index_d`` and ``relations`` in place. The
    borrowing projectile shares the matrix object and the ``relations`` list of
    ``parent_pdg`` (no copies).
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
    """Family representatives (K+-, pi+-, p) to try for ``projectile``."""
    # Antiparticle representative first (e.g. nbar -> pbar), then the particle.
    apid = abs(projectile)
    sign = -1 if projectile < 0 else 1
    # Kaons before pions: 130 (K_L) is also inside the pion window.
    if apid in (130, 310, 311) or 300 < apid < 1000:
        return [sign * 321, 321]
    if 100 < apid < 300:
        return [sign * 211, 211]
    if 1000 < apid < 5000:
        return [sign * 2212, 2212]
    # leptons, 10313/10323, nuclei: no equivalent
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
