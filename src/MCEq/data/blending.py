"""Runtime blending of a high- and a low-energy hadronic model."""

from collections import defaultdict

import numpy as np

from MCEq.data.equivalences import mapped_cross_section


def he_le_weight(energy, transition, trwidth, dtype):
    """High-energy model weight on the active kinetic-energy grid.

    Args:
        energy: energy grid centers in GeV (``energy_grid.c``).
        transition: HE/LE transition energy in GeV.
        trwidth: full 10--90 % width of the transition in decades; ``0.0``
            makes the weight a hard switch at ``transition``.
        dtype: weight dtype; ``None`` gives float64.
    """
    energy = np.asarray(energy, dtype=float)
    if trwidth == 0.0:
        return (energy >= transition).astype(dtype)
    # Define trwidth as the full 10--90% width in log10 energy.
    arg = 2.0 * np.log(9.0) * np.log10(energy / transition) / trwidth
    arg = np.clip(arg, -700.0, 700.0)
    return np.asarray(1.0 / (1.0 + np.exp(-arg)), dtype=dtype)


def blend_yields(he_index, le_index, he_name, le_name, w_he, transition, trwidth):
    """Blend yield matrices column-wise over the HE model's channel set.

    Channels absent from the LE model keep the HE matrix. The weight
    ``w_he`` of :func:`he_le_weight` depends on the projectile energy (last
    axis) only, so it broadcasts over the Legendre modes of 2D
    ``(n_k, n_e, n_e)`` tensors. Both models must be on the same energy grid,
    with the transition inside both models' coverage. ``transition`` and
    ``trwidth`` only enter the description string.
    """
    w_he = w_he[np.newaxis, :]
    w_le = 1.0 - w_he
    he_yields = he_index["index_d"]
    le_yields = le_index["index_d"]
    blended = {}
    for channel, he_matrix in he_yields.items():
        if channel in le_yields:
            blended[channel] = he_matrix * w_he + le_yields[channel] * w_le
        else:
            blended[channel] = he_matrix

    relations = defaultdict(list)
    particles = set()
    for parent, child in blended:
        relations[parent].append(child)
        particles.add(parent)
        particles.add(child)
    return {
        "parents": sorted(relations),
        "particles": sorted(particles),
        "relations": dict(relations),
        "index_d": blended,
        "description": (
            f"Runtime HE/LE blend: {he_name} + {le_name}; "
            f"transition={transition:g} GeV, "
            f"10-90 width={trwidth:g} decades"
        ),
    }


def blend_cross_sections(he_index, le_index, he_name, le_name, w_he):
    """Blend inelastic cross sections per projectile, mapping across models.

    A projectile carried by only one model is mapped onto the other model's
    equivalent column (:func:`~MCEq.data.equivalences.mapped_cross_section`);
    without an equivalent it stays unblended. ``he_name``/``le_name`` are
    normalised model names and select the equivalence tables.
    """
    w_le = 1.0 - w_he
    blended = {}
    for projectile, he_cs in he_index["index_d"].items():
        le_cs = mapped_cross_section(le_index["index_d"], projectile, le_name)
        # Leave HE-only projectiles unchanged.
        blended[projectile] = he_cs if le_cs is None else he_cs * w_he + le_cs * w_le
    # LE-only projectiles (e.g. FLUKA's n, nbar, K0, hyperons) are kept: own
    # sigma below the transition, the mapped HE equivalent above it. Otherwise
    # get_cs would fall back to the proton column and, for antibaryons, miss
    # the annihilation part.
    for projectile, le_cs in le_index["index_d"].items():
        if projectile in blended:
            continue
        he_cs = mapped_cross_section(he_index["index_d"], projectile, he_name)
        blended[projectile] = le_cs if he_cs is None else he_cs * w_he + le_cs * w_le
    return {"parents": sorted(blended), "index_d": blended}
