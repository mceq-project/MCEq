"""Runtime HE/LE model blending: the sigmoid weight and the two blends.

``HDF5Backend`` loads the model tables and passes their arrays to these
functions; the backend keeps thin delegates.

DTYPE
-----

With ``dtype=None`` (the default), blend weights use float64 and blended
channels promote accordingly. An explicit dtype controls the weights.
Channels without an LE counterpart are passed through unchanged and retain
their stored arrays and dtype.

The weight is applied as ``w_he[np.newaxis, :]``, i.e. along the LAST axis, so
one ``(1, n_e)`` array serves both the 1D ``(n_e, n_e)`` channel matrix and the
2D ``(n_k, n_e, n_e)`` Hankel-mode tensor (it broadcasts as ``(1, 1, n_e)``).
Indexing the first axis instead would still pass a square-matrix test.
"""

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
        dtype: ``grid.dtype``, i.e. ``None`` unless ``config.floatlen`` is set.
            The default result is float64; an explicit dtype is respected in
            both transition modes.
    """
    energy = np.asarray(energy, dtype=float)
    if trwidth == 0.0:
        return (energy >= transition).astype(dtype)
    # Define trwidth as the full 10--90% width in log10 energy.
    arg = 2.0 * np.log(9.0) * np.log10(energy / transition) / trwidth
    arg = np.clip(arg, -700.0, 700.0)
    return np.asarray(1.0 / (1.0 + np.exp(-arg)), dtype=dtype)


def blend_yields(he_index, le_index, he_name, le_name, w_he, transition, trwidth):
    """Blend yield matrices column-wise, using the HE channel set.

    The HE model defines the channel set. Channels absent from the LE
    model remain unchanged, matching the former compiled low-energy
    extension. Model-specific projectile equivalences have already been
    applied by ``_gen_db_dictionary`` before this step.

    On 2D databases the channel matrices are ``(n_k, n_e, n_e)`` Hankel-
    mode tensors and the ``(1, n_e)`` column weights broadcast over every
    mode: the blend weight depends on the projectile energy only, so
    blending commutes with the Hankel transform and applies per kappa.
    Both models must be stored on the database's common grid (a model's
    blocks are zero outside its production range, so the transition
    window must lie inside both models' coverage — the multi-model 2D
    database build guarantees this).

    ``w_he`` is the 1D weight of :func:`he_le_weight`; ``transition`` and
    ``trwidth`` are carried only to render the description string, which is the
    returned index's sole record of how the blend was parameterised.
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

    Unlike :func:`blend_yields` this is not a plain intersection: a projectile
    carried by only one of the two models is mapped onto the other model's
    nearest column through :func:`~MCEq.data.equivalences.mapped_cross_section`,
    and ``None`` from that lookup means "no equivalent" and leaves the value the
    one model does carry unblended. ``he_name``/``le_name`` select the model's
    alias table, so they are the normalised model names, not free-form labels.

    ``w_he`` is the 1D weight of :func:`he_le_weight`, applied without the
    ``np.newaxis`` of :func:`blend_yields`: a cross section is one value per
    energy, so the weight already lines up.
    """
    w_le = 1.0 - w_he
    blended = {}
    for projectile, he_cs in he_index["index_d"].items():
        le_cs = mapped_cross_section(le_index["index_d"], projectile, le_name)
        # Leave HE-only projectiles unchanged.
        blended[projectile] = he_cs if le_cs is None else he_cs * w_he + le_cs * w_le
    # LE-only projectiles (e.g. FLUKA's dedicated n, nbar, K0, hyperon
    # columns) keep their identity instead of being dropped: their own
    # sigma below the transition, the mapped HE equivalent above it.
    # Without this, get_cs falls back to the proton column, which for
    # antibaryons misses the annihilation part entirely.
    for projectile, le_cs in le_index["index_d"].items():
        if projectile in blended:
            continue
        he_cs = mapped_cross_section(he_index["index_d"], projectile, he_name)
        blended[projectile] = le_cs if he_cs is None else he_cs * w_he + le_cs * w_le
    return {"parents": sorted(blended), "index_d": blended}
