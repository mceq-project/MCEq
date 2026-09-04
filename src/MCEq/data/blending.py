"""Runtime HE/LE model blending: the sigmoid weight and the two blends.

Plan section 1's ``data/blending.py``. The three functions were
``HDF5Backend._he_le_weight``, ``HDF5Backend._blend_interaction_dbs`` and the
blend half of ``HDF5Backend.cs_db``; they read nothing off the backend that is
not passed in here, so they are free functions and the backend keeps thin
delegates. Loading stays in ``HDF5Backend`` -- ``cs_db`` still calls
``_cs_db_single`` twice and hands the two indices over.

THE DTYPE BEHAVIOUR IS DELIBERATE, NOT AN OVERSIGHT
---------------------------------------------------

``dtype`` is ``config.grid.dtype``, which is ``None`` by default (``floatlen``
is unset), and ``np.asarray(x, dtype=None)`` PRESERVES ``x``'s dtype while
``x.astype(None)`` FORCES float64. The consequences are pinned in
``tests/test_low_energy_blending.py`` and a "cleanup" here changes numbers:

* :func:`he_le_weight` returns float64 in *both* branches whatever the energy
  grid is, and neither ``dtype=None`` is what makes it so. The hard switch is
  ``.astype`` on a *bool* array, so ``None`` forces float64 where preserving
  would have given bool; the sigmoid is float64 over-determinedly, both from
  the ``dtype=float`` on the energy array and from ``np.log(9.0)`` being an
  ``np.float64`` scalar that promotes under NEP 50.
* the one site where ``dtype=None`` genuinely preserves is the channel-pack
  read in ``_gen_db_dictionary``, so a float32 pack reaches :func:`blend_yields`
  as float32, is multiplied by the float64 weight and comes back float64 --
  while an HE-only channel, passed through untouched, stays float32. That
  asymmetry inside one returned ``index_d`` is today's behaviour.

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
            See the module docstring -- the result is float64 either way.
    """
    energy = np.asarray(energy, dtype=float)
    if trwidth == 0.0:
        return (energy >= transition).astype(dtype)
    # Define trwidth as the full 10--90% width in log10 energy.
    arg = 2.0 * np.log(9.0) * np.log10(energy / transition) / trwidth
    arg = np.clip(arg, -700.0, 700.0)
    return np.asarray(1.0 / (1.0 + np.exp(-arg)), dtype=dtype)


def blend_yields(he_index, le_index, he_name, le_name, w_he, transition, trwidth):
    """Blend yield matrices column-wise, preserving historical semantics.

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
        # Historical behaviour: an HE-only projectile remains unchanged.
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
