"""Shared builder for the two solve-free ``operators`` golden sections.

The sections pin the assembled cascade operators themselves — ``int_m``,
``dec_m`` and the differential operator behind the continuous-loss band —
across the full cross product of interior stencils and muon multiple
scattering, plus one averaged-loss cell beside it, on both databases. Nothing
here solves; the whole cost is
:meth:`MCEq.operators.matrix_builder.MatrixBuilder.construct_matrices`.

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
``config.loss_stencil_method``, ``config.muon_multiple_scattering`` and
``config.average_loss_operator``, calls
``MatrixBuilder._construct_differential_operator()`` and then
``construct_matrices(skip_decay_matrix=False)``, and digests the result. All
three globals are set on every cell, so the averaged one cannot leak into the
cell built after it.

The explicit ``_construct_differential_operator()`` is load-bearing: it is
called from ``MatrixBuilder.__init__`` and nowhere else, so
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

That equivalence is a *gate*, not a measurement left in a docstring:
:func:`build_cell_operators` is the whole shortcut, and
``tests/test_operators_pin.py`` runs two cells of it against a fresh
``MCEqRun`` and compares digests. Without that, a builder that memoised
anything across cells — which the Phase 5 ``int_m_hadr + dEdx_band`` split
invites, since it puts a band matrix on ``self`` — would leave the sweep
self-consistent (comparison and regeneration move together) while the section
silently stopped pinning a fresh build.

What the sweep buys
-------------------
Across both sections the 28 cross-product ``int_m`` digests take 21 distinct
values, and the collision structure is the section's own consistency check.
The two averaged cells add one value each, so the 30 cells take 23:

* ``dec_m`` is invariant — one digest per database over all 15 cells. The
  stencil and the loss averaging enter only through ``cont_loss_operator``,
  and the muon damping only through
  ``_csr_from_blocks(apply_muon_scattering=True)``, which the decay path
  never sets.
* in 1D the two muon-scattering cells of a stencil are **equal**:
  ``_muon_scattering_damping`` returns ``None`` unless ``is_2d``, so the flag
  is a no-op there. 7 distinct of 14.
* in 2D they **differ**, at every stencil: 14 distinct of 14.
* :data:`AVERAGE_CELL` is a fifteenth operator in either section, and a new
  digest in both: ten explicit Euler steps fill the loss band in, so ``int_m``
  goes from 169786 to 187806 nonzeros in 1D (+10.6 %) and 6212352 to 6894720
  in 2D (+11.0 %). That the digest is new is what says the flag reaches
  ``int_m`` at all, which is what covering ``np.linalg.matrix_power`` is for.

BLAS threads are left ambient — unlike ``gen_solve2d``, which pins them with
``config.set_mkl_threads`` and so leaves a process-wide limiter behind.
``_fill_matrices`` reaches BLAS through the ``dprop.dot(pprod_mat)`` products
of ``_follow_chains``, but rebuilding either section at 1, 2, 4 and 8 threads
reproduces every key bitwise except ``em_step_scale`` — see
:data:`EM_SCALE_RTOL`, which is the one tolerance either section carries. What
is ambient still has to be *recorded*, since it is the one input that key
demonstrably depends on: ``extra["blas_threads"]``, from
:func:`._harness.blas_threads`.

``em_step_scale`` is recorded only by the section where it has content. On the
2D fixture ``disabled_particles = [11, -11]`` leaves gamma the only ``is_em``
species and ``enable_em`` is off, so the EM off-diagonal block is empty and
the value is identically 0 in all fourteen cells — a statement about the
fixture, not about the assembly the section exists to pin. Hence the
``em_step_scale`` flag of :func:`build_section`; the mode-0-only indexing that
``_em_cascade_step_scale`` uses on a stitched 2D operator is pinned instead as
a behaviour, database-free, in ``tests/test_operators_pin.py``.
"""

from __future__ import annotations

import contextlib
import copy
import time

import numpy as np

from ._harness import (
    assert_canonical_csr,
    blas_threads,
    make_provenance,
    sparse_digest,
)

#: Every interior stencil ``MatrixBuilder._construct_differential_operator``
#: accepts (:data:`MCEq.operators.loss_stencil.STENCIL_METHODS`, since Phase 5
#: move B; ``core.py:3295-3352`` before it). Kept in the order of the dispatch:
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

#: The one cell outside the cross product: ``average_loss_operator``, at the
#: default stencil with muon scattering on. The flag sends
#: ``cont_loss_operator`` through ``_average_operator`` —
#: ``matrix_power(I + op * step, 1/step) - I`` — and is False in all five
#: generators, with nothing else in the suite setting it, so the branch and its
#: ``np.linalg.matrix_power`` have no golden coverage at all. Both generators
#: pin ``loss_step_for_average`` at 1e-1, so the cell runs at ten explicit
#: Euler steps rather than at whatever the ambient value happens to be.
#:
#: It is not a third axis: the averaging is a property of the band, not of the
#: stencil dispatch or of the assembly, so fourteen more cells would pin the
#: same statement seven times over at 0.9 s each in 2D.
AVERAGE_CELL = ("expfit_low_upwind2/ms_on/avg_loss", "expfit_low_upwind2", True, True)

#: ``(label, stencil, muon_scattering, average_loss_operator)`` per cell, in
#: build order. The averaged cell is last, so every cross-product cell is built
#: before the flag is ever True.
CELLS = tuple(
    (f"{stencil}/{ms_label}", stencil, ms, False)
    for stencil in STENCILS
    for ms_label, ms in SCATTERING
) + (AVERAGE_CELL,)

#: The (stencil x scattering) cross product alone — :data:`CELLS` less the
#: averaged cell, whose operator the collision structure below does not
#: describe. The invariants and the structure tests range over this, so adding
#: the averaged cell leaves them stating exactly what they stated before.
SWEEP_CELLS = tuple(cell for cell in CELLS if not cell[3])

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


def tolerances(em_step_scale: bool = True) -> dict:
    """The section tolerance table: ``em_step_scale`` on rel-L2, rest bitwise.

    Empty for a section that records no ``em_step_scale``. An entry matching no
    key is worse than no entry: a reader credits the section with a 1e-9
    tolerance that :func:`._harness.tolerance_entry_for` never resolves.
    """
    if not em_step_scale:
        return {}
    return {
        f"cells/{label}/em_step_scale": {"mode": "rel_l2", "rtol": EM_SCALE_RTOL}
        for label, *_ in CELLS
    }


# --------------------------------------------------------------------------
# the construct-once shortcut
# --------------------------------------------------------------------------


@contextlib.contextmanager
def pinned_config(config_pins, adv_set_pins):
    """Hold one section's config pins in force, then put the globals back.

    The generator and the fresh-build equivalence test enter the fixture
    through this, so the test cannot drift onto a different one. The
    ``hasattr`` sweep fails loudly on a pin block naming a config global the
    tree has dropped, rather than silently pinning nothing.
    """
    from MCEq import config

    missing = sorted(key for key in config_pins if not hasattr(config, key))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    saved_config = {key: getattr(config, key) for key in config_pins}
    saved_adv_set = copy.deepcopy(config.adv_set)
    try:
        for key, value in config_pins.items():
            setattr(config, key, value)
        config.adv_set.update(adv_set_pins)
        yield config
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)


def build_cell_operators(mceq, stencil, scattering, average=False):
    """Assemble ``(int_m, dec_m)`` for one cell on an already-built run.

    The sweep's whole construct-once shortcut, in one function, so the
    generator and the equivalence test in ``tests/test_operators_pin.py``
    exercise the same three calls instead of two copies that can drift.

    All three globals are set on every cell, ``average_loss_operator``
    included, so the averaged cell cannot leak into the one built after it.

    ``_construct_differential_operator()`` is explicit because it is called
    from ``MatrixBuilder.__init__`` and nowhere else, so
    ``construct_matrices`` alone refills the blocks against the ``op_matrix``
    the constructor left behind. ``skip_decay_matrix`` stays False so ``dec_m``
    is rebuilt and its invariance is measured, not assumed.
    """
    from MCEq import config

    config.loss_stencil_method = stencil
    config.muon_multiple_scattering = scattering
    config.average_loss_operator = average
    builder = mceq.matrix_builder
    builder._construct_differential_operator()
    return builder.construct_matrices(skip_decay_matrix=False)


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


def _record_cell(
    arrays, label, stencil, scattering, average, mceq, n_k, n_species, em_step_scale
):
    """Assemble one cell and store the operators and the EM step scale.

    ``op_matrix`` depends on the stencil alone, so it is stored once per
    stencil under ``stencil/`` and the second cell of a pair only asserts that
    it did not move.

    ``int_m`` / ``dec_m`` are written back onto the run so it describes what
    was built, and because ``_em_cascade_step_scale`` caches against the
    identity of ``self.int_m``: a builder-only rebuild would be answered out
    of the stale entry.
    """
    mb = mceq.matrix_builder
    int_m, dec_m = build_cell_operators(mceq, stencil, scattering, average)
    mceq.int_m, mceq.dec_m = int_m, dec_m

    # Canonical form is the property the digests below cannot see, and the 2D
    # assembly is where it can be lost: the muon damping is scattered in as
    # further COO entries over an occupied diagonal, so it rests on
    # `coo.tocsr()` summing duplicates. 18 ms per operator at 6.2M nonzeros.
    assert_canonical_csr(int_m, f"{label}/int_m")
    assert_canonical_csr(dec_m, f"{label}/dec_m")

    op_matrix = np.asarray(mb.op_matrix, dtype=np.float64)
    key = f"stencil/{stencil}/op_matrix"
    if key in arrays:
        assert np.array_equal(arrays[key], op_matrix), (
            f"{label}: muon_multiple_scattering or average_loss_operator moved "
            f"op_matrix. The first reaches _csr_from_blocks and the second "
            f"cont_loss_operator, never _construct_differential_operator."
        )
    else:
        arrays[key] = op_matrix

    int_digest = _record_csr(arrays, f"cells/{label}/int_m", int_m)
    _record_reductions(arrays, f"cells/{label}/int_m", int_m, n_k, n_species)
    dec_digest = _record_csr(arrays, f"cells/{label}/dec_m", dec_m)

    if em_step_scale:
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

    Four statements, each a property of the builder rather than of a number:
    ``dec_m`` sees no knob; muon scattering is a no-op in 1D and moves every
    stencil in 2D; the seven stencils are mutually distinct at either setting
    of the flag; and ``average_loss_operator`` moves ``int_m`` off the cell it
    shares a stencil and a scattering flag with.

    The first three range over :data:`SWEEP_CELLS` — the cross product alone,
    which is the grid the collision structure describes. The averaged cell is
    one more operator beside it, so the summary reports both counts: distinct
    digests over the whole file, and over the cross product the invariants use.
    """
    sweep = {label: int_digests[label] for label, *_ in SWEEP_CELLS}
    distinct_dec = sorted(set(dec_digests.values()))
    assert len(distinct_dec) == 1, (
        f"dec_m is not invariant under the sweep: {len(distinct_dec)} distinct "
        f"digests over {len(dec_digests)} cells. The stencil reaches int_m "
        f"through cont_loss_operator and the muon damping through "
        f"_csr_from_blocks(apply_muon_scattering=True); neither is on the "
        f"decay path."
    )

    for stencil in STENCILS:
        on = sweep[f"{stencil}/ms_on"]
        off = sweep[f"{stencil}/ms_off"]
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
        column = {s: sweep[f"{s}/{ms_label}"] for s in STENCILS}
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

    n_sweep_distinct = len(set(sweep.values()))
    expected = len(STENCILS) * (len(SCATTERING) if is_2d else 1)
    assert n_sweep_distinct == expected, (
        f"expected {expected} distinct int_m digests over {len(sweep)} cross-"
        f"product cells, got {n_sweep_distinct}"
    )

    # The averaged cell is a cell, not an axis: what it has to state is that
    # the flag reaches int_m at all, which is the whole point of covering the
    # matrix_power branch.
    avg_label, avg_stencil = AVERAGE_CELL[0], AVERAGE_CELL[1]
    assert int_digests[avg_label] != sweep[f"{avg_stencil}/ms_on"], (
        "average_loss_operator left int_m unchanged. cont_loss_operator "
        "returns _average_operator(op_mat) under the flag: matrix_power("
        "I + op * step, 1/step) - I, which is not op."
    )
    return {
        "n_cells": len(int_digests),
        "n_sweep_cells": len(sweep),
        "n_distinct_int_m": len(set(int_digests.values())),
        "n_distinct_sweep_int_m": n_sweep_distinct,
        "n_distinct_dec_m": len(distinct_dec),
        "int_m_digests": dict(int_digests),
        "dec_m_digest": distinct_dec[0],
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_section(
    section,
    *,
    config_pins,
    adv_set_pins,
    run_kwargs,
    note,
    extra=None,
    em_step_scale=True,
):
    """Produce ``(arrays, provenance)`` for one database's operator sweep.

    ``em_step_scale`` False drops the EM step scale and its tolerance entry,
    for a fixture on which the value carries nothing — see the module
    docstring.
    """
    arrays = {}
    seconds = {}
    with pinned_config(config_pins, adv_set_pins):
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
            for label, stencil, scattering, average in CELLS:
                int_digests[label], dec_digests[label] = _record_cell(
                    arrays,
                    label,
                    stencil,
                    scattering,
                    average,
                    mceq,
                    n_k,
                    n_species,
                    em_step_scale,
                )
            seconds["cells"] = time.perf_counter() - started
        finally:
            mceq.close()

        seconds["total"] = sum(seconds.values())
        summary = _check_invariants(is_2d, int_digests, dec_digests)

        provenance = make_provenance(
            section,
            note=note,
            tolerances=tolerances(em_step_scale),
            extra={
                "stencils": list(STENCILS),
                "scattering": [label for label, _ in SCATTERING],
                "cells": [label for label, *_ in CELLS],
                "average_loss_cell": AVERAGE_CELL[0],
                "blas_threads": blas_threads(),
                "seconds": seconds,
                **summary,
                **(extra or {}),
            },
        )

    db = provenance["databases"].get("mceq_db_fname", {})
    assert "sha256" in db, f"database not resolvable, provenance incomplete: {db}"

    return arrays, provenance
