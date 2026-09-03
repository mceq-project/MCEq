"""Shared builder for the two solve-free ``operators`` golden sections.

The sections pin the assembled cascade operators themselves — ``int_m``,
``dec_m`` and the differential operator behind the continuous-loss band —
across the full cross product of interior stencils and muon multiple
scattering, on both databases. Nothing here solves; the whole cost is
:meth:`MCEq.core.MatrixBuilder.construct_matrices`.

Why the section exists: Phase 5 moves ``MatrixBuilder`` out of ``core.py``,
extracts the stencil families and the kappa^2 muon damping as pure functions,
and splits ``int_m`` into ``int_m_hadr + dEdx_band``. The 1D operators are
pinned today by ``solve1d`` (``{emoff,emon}/meta/{int_m,dec_m}``) and
``species`` (``{operators,ddm,f32}/…``); the 2D operators are pinned by
nothing, and no section moves ``config.loss_stencil_method`` off its default,
so six of the seven stencil families and the whole 2D assembly path are
unpinned before this one.

Construction, and why it is exact
---------------------------------
One :class:`~MCEq.core.MCEqRun` per database. Per cell the generator sets
``config.loss_stencil_method`` and ``config.muon_multiple_scattering``, calls
``MatrixBuilder._construct_differential_operator()`` and then
``construct_matrices(skip_decay_matrix=False)``, and digests the result.

The explicit ``_construct_differential_operator()`` is load-bearing:
``core.py`` calls it from ``MatrixBuilder.__init__`` and nowhere else, so
``MCEqRun.regenerate_matrices()`` alone does **not** pick up a stencil change
— it rebuilds the blocks against the stale ``op_matrix``. That is pinned as a
behaviour by ``tests/test_operators_pin.py``.

Reusing one run is not an approximation. Measured on this tree, the 28 cells
built the naive way (a fresh ``MCEqRun`` per cell: 3.5 s for 1D, 57.3 s for
2D) and this way (3.1 s / 19.4 s) give sha256-identical ``int_m`` *and*
``dec_m`` in 28 of 28 cells. ``skip_decay_matrix`` stays False for the same
reason: with True the builder returns the ``dec_m`` object it already holds,
and "``dec_m`` is invariant under the stencil and under muon scattering"
would be a tautology instead of a measurement. The extra cost is 5.6 s on the
2D section.

What the sweep buys
-------------------
Across both sections the 28 ``int_m`` digests take 21 distinct values, and the
collision structure is the section's own consistency check:

* ``dec_m`` is invariant — one digest per database over all 14 cells. The
  stencil enters only through ``cont_loss_operator``, and the muon damping
  only through ``_csr_from_blocks(apply_muon_scattering=True)``, which the
  decay path never sets.
* in 1D the two muon-scattering cells of a stencil are **equal**:
  ``_muon_scattering_damping`` returns ``None`` unless ``is_2d``, so the flag
  is a no-op there. 7 distinct of 14.
* in 2D they **differ**, at every stencil: 14 distinct of 14.

BLAS threads are left ambient. ``_fill_matrices`` reaches BLAS through the
``dprop.dot(pprod_mat)`` products of ``_follow_chains``, but rebuilding either
section at 1, 2, 4 and 8 threads reproduces every key bitwise except
``em_step_scale`` — see :data:`EM_SCALE_RTOL`, which is the one tolerance
either section carries.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from ._harness import make_provenance, sparse_digest

#: Every interior stencil ``MatrixBuilder._construct_differential_operator``
#: accepts (core.py:3295-3352). Kept in the order of the method's own dispatch:
#: the two ``expfit_low_upwind*`` composites, the three high-order families,
#: then the two monotone upwind operators. ``tests/test_operators_pin.py``
#: reads the accepted names out of the ``ValueError`` the method raises and
#: fails if this tuple and the source drift apart.
STENCILS = (
    "expfit_low_upwind2",
    "expfit_low_upwind",
    "expfit",
    "centered",
    "biased",
    "upwind",
    "upwind2",
)

#: ``(label, config.muon_multiple_scattering)`` for the second sweep axis.
SCATTERING = (("ms_on", True), ("ms_off", False))

#: ``(label, stencil, muon_scattering)`` for every cell, in build order.
CELLS = tuple(
    (f"{stencil}/{ms_label}", stencil, ms)
    for stencil in STENCILS
    for ms_label, ms in SCATTERING
)

#: rel-L2 budget for ``em_step_scale`` — the only key of either section that is
#: not bitwise. It is ``max|eigvals|`` of a dense, strongly non-normal block
#: through LAPACK, not a reduction whose order the builder fixes, and it does
#: move: rebuilding ``operators1d`` at 1, 2, 4 and 8 BLAS threads reproduces
#: every other key bitwise but shifts ``expfit_low_upwind2`` by 6.9e-13
#: relative (0.09581041044292567 -> ...99196 at 1, 2 and 4 threads, ...92385 at
#: 8) on a 217 x 217 EM block, and ``expfit_low_upwind`` by 7.4e-16. So
#: ``HOST_RTOL`` (1e-12) would carry 1.4x, which is no margin at all for a
#: LAPACK bump or another host's reference job. 1e-9 carries 1400x and still
#: pins the value to ten significant digits; the two closest stencils differ by
#: 1.5e-3 relative (biased 0.10855 against expfit 0.10871), six orders above
#: the bound, so the key keeps discriminating every cell.
EM_SCALE_RTOL = 1e-9


def tolerances() -> dict:
    """The section tolerance table: ``em_step_scale`` on rel-L2, rest bitwise."""
    return {
        f"cells/{label}/em_step_scale": {"mode": "rel_l2", "rtol": EM_SCALE_RTOL}
        for label, _, _ in CELLS
    }


# --------------------------------------------------------------------------
# recorders
# --------------------------------------------------------------------------


def _record_csr(arrays, prefix, matrix):
    """Shape, nnz, dtype and the three CSR buffer digests of one operator."""
    digest = sparse_digest(matrix)
    arrays[prefix + "/shape"] = np.asarray(digest["shape"])
    arrays[prefix + "/nnz"] = np.asarray(digest["nnz"])
    arrays[prefix + "/dtype"] = np.asarray(digest["dtype"])
    for part in ("data", "indices", "indptr"):
        arrays[prefix + "/" + part] = np.asarray(digest[part])
    return digest


def _record_reductions(arrays, prefix, matrix, n_k, n_species):
    """Row-block reductions of one operator along both axes of the layout.

    The stitched state runs over ``(mode, species, energy)``, so the rows fold
    into ``(n_k, n_species, dim)``. Summing out energy and mode names the
    species a digest mismatch comes from; summing out energy and species names
    the Hankel mode, which is the axis the ``-kappa^2 theta_s^2 / 4`` muon
    damping varies along. Both are ``n_k`` or ``n_species`` numbers, against
    the ``n_k * n_species`` a full block table would cost.
    """
    csr = matrix.tocsr()
    rows = np.asarray(csr.sum(axis=1)).ravel().reshape(n_k, n_species, -1)
    counts = np.diff(csr.indptr).reshape(n_k, n_species, -1)
    arrays[prefix + "/row_sums_by_species"] = rows.sum(axis=(0, 2))
    arrays[prefix + "/row_sums_by_mode"] = rows.sum(axis=(1, 2))
    arrays[prefix + "/nnz_by_species"] = counts.sum(axis=(0, 2)).astype(np.int64)
    arrays[prefix + "/nnz_by_mode"] = counts.sum(axis=(1, 2)).astype(np.int64)


def _record_fixture(arrays, mceq):
    """The tables and dimensions the sweep runs against, read live."""
    from MCEq import config

    mb = mceq.matrix_builder
    arrays["fixture/db"] = np.asarray(config.mceq_db_fname)
    arrays["fixture/model"] = np.asarray(mceq._interactions.iam)
    arrays["fixture/medium"] = np.asarray(mceq._mceq_db.medium)
    arrays["fixture/decay_dset"] = np.asarray(mceq._decays._default_decay_dset)
    arrays["fixture/floatlen"] = np.asarray(str(config.floatlen))
    arrays["fixture/is_2d"] = np.asarray(int(mb.is_2d))
    arrays["fixture/n_k"] = np.asarray(int(mb.n_k))
    arrays["fixture/k_grid"] = np.asarray(mb.k_grid, dtype=np.float64)
    arrays["fixture/dim"] = np.asarray(int(mb.dim))
    arrays["fixture/dim_states"] = np.asarray(int(mb.dim_states))
    arrays["fixture/n_species"] = np.asarray(int(mceq.pman.n_cparticles))
    arrays["fixture/species"] = np.array(
        [f"{p.mceqidx}:{p.name}" for p in mceq.pman.cascade_particles], dtype="U32"
    )
    arrays["fixture/em_species"] = np.array(
        sorted(p.name for p in mceq.pman.all_particles if getattr(p, "is_em", False)),
        dtype="U32",
    )
    arrays["fixture/e_grid"] = np.asarray(mceq.e_grid)
    arrays["fixture/e_bins"] = np.asarray(mceq.e_bins)


def _record_cell(arrays, label, stencil, mceq, n_k, n_species):
    """Assemble one cell and store the operators and the EM step scale.

    ``op_matrix`` depends on the stencil alone, so it is stored once per
    stencil under ``stencil/`` and the second cell of a pair only asserts that
    it did not move.

    ``int_m`` / ``dec_m`` are written back onto the run before the step scale
    is read: ``_em_cascade_step_scale`` caches against the identity of
    ``self.int_m``, so a builder-only rebuild would be answered out of the
    stale entry.
    """
    mb = mceq.matrix_builder
    mb._construct_differential_operator()
    int_m, dec_m = mb.construct_matrices(skip_decay_matrix=False)
    mceq.int_m, mceq.dec_m = int_m, dec_m

    op_matrix = np.asarray(mb.op_matrix, dtype=np.float64)
    key = f"stencil/{stencil}/op_matrix"
    if key in arrays:
        assert np.array_equal(arrays[key], op_matrix), (
            f"{label}: muon_multiple_scattering moved op_matrix. The flag "
            f"reaches _csr_from_blocks, never _construct_differential_operator."
        )
    else:
        arrays[key] = op_matrix

    int_digest = _record_csr(arrays, f"cells/{label}/int_m", int_m)
    _record_reductions(arrays, f"cells/{label}/int_m", int_m, n_k, n_species)
    dec_digest = _record_csr(arrays, f"cells/{label}/dec_m", dec_m)

    mceq._em_step_scale_cache = None
    arrays[f"cells/{label}/em_step_scale"] = np.asarray(
        float(mceq._em_cascade_step_scale())
    )
    return int_digest["data"], dec_digest["data"]


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


def _check_invariants(is_2d, int_digests, dec_digests):
    """Assert the collision structure the sweep makes free; return a summary.

    Three statements, each a property of the builder rather than of a number:
    ``dec_m`` sees neither knob; muon scattering is a no-op in 1D and moves
    every stencil in 2D; the seven stencils are mutually distinct at either
    setting of the flag.
    """
    distinct_dec = sorted(set(dec_digests.values()))
    assert len(distinct_dec) == 1, (
        f"dec_m is not invariant under the sweep: {len(distinct_dec)} distinct "
        f"digests over {len(dec_digests)} cells. The stencil reaches int_m "
        f"through cont_loss_operator and the muon damping through "
        f"_csr_from_blocks(apply_muon_scattering=True); neither is on the "
        f"decay path."
    )

    for stencil in STENCILS:
        on = int_digests[f"{stencil}/ms_on"]
        off = int_digests[f"{stencil}/ms_off"]
        if is_2d:
            assert on != off, (
                f"{stencil}: muon_multiple_scattering left int_m unchanged on a "
                f"2D database. _csr_from_blocks adds -kappa^2 theta_s^2 / 4 to "
                f"the muon diagonals of every mode with kappa != 0."
            )
        else:
            assert on == off, (
                f"{stencil}: muon_multiple_scattering moved int_m on a 1D "
                f"database. _muon_scattering_damping returns None unless "
                f"is_2d, so the flag has nothing to apply."
            )

    for ms_label, _ in SCATTERING:
        column = {s: int_digests[f"{s}/{ms_label}"] for s in STENCILS}
        collisions = sorted(
            (a, b)
            for i, a in enumerate(STENCILS)
            for b in STENCILS[i + 1 :]
            if column[a] == column[b]
        )
        assert not collisions, (
            f"{ms_label}: distinct loss_stencil_method values gave the same "
            f"int_m: {collisions}. The section cannot discriminate a stencil "
            f"the builder ignores."
        )

    n_distinct = len(set(int_digests.values()))
    expected = len(STENCILS) * (len(SCATTERING) if is_2d else 1)
    assert n_distinct == expected, (
        f"expected {expected} distinct int_m digests over {len(int_digests)} "
        f"cells, got {n_distinct}"
    )
    return {
        "n_cells": len(int_digests),
        "n_distinct_int_m": n_distinct,
        "n_distinct_dec_m": len(distinct_dec),
        "int_m_digests": dict(int_digests),
        "dec_m_digest": distinct_dec[0],
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_section(section, *, config_pins, adv_set_pins, run_kwargs, note, extra=None):
    """Produce ``(arrays, provenance)`` for one database's operator sweep."""
    from MCEq import config

    missing = sorted(key for key in config_pins if not hasattr(config, key))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    saved_config = {key: getattr(config, key) for key in config_pins}
    saved_adv_set = copy.deepcopy(config.adv_set)
    arrays = {}
    seconds = {}
    try:
        for key, value in config_pins.items():
            setattr(config, key, value)
        config.adv_set.update(adv_set_pins)

        from MCEq.core import MCEqRun

        started = time.perf_counter()
        mceq = MCEqRun(**run_kwargs)
        seconds["construct"] = time.perf_counter() - started
        try:
            _record_fixture(arrays, mceq)
            is_2d = bool(mceq.matrix_builder.is_2d)
            n_k = int(mceq.matrix_builder.n_k)
            n_species = int(mceq.pman.n_cparticles)

            int_digests, dec_digests = {}, {}
            started = time.perf_counter()
            for label, stencil, scattering in CELLS:
                config.loss_stencil_method = stencil
                config.muon_multiple_scattering = scattering
                int_digests[label], dec_digests[label] = _record_cell(
                    arrays, label, stencil, mceq, n_k, n_species
                )
            seconds["cells"] = time.perf_counter() - started
        finally:
            mceq.close()

        seconds["total"] = sum(seconds.values())
        summary = _check_invariants(is_2d, int_digests, dec_digests)

        provenance = make_provenance(
            section,
            note=note,
            tolerances=tolerances(),
            extra={
                "stencils": list(STENCILS),
                "scattering": [label for label, _ in SCATTERING],
                "cells": [label for label, _, _ in CELLS],
                "seconds": seconds,
                **summary,
                **(extra or {}),
            },
        )
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)

    db = provenance["databases"].get("mceq_db_fname", {})
    assert "sha256" in db, f"database not resolvable, provenance incomplete: {db}"

    return arrays, provenance
