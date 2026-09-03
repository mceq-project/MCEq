"""Behaviour pins for the operators layer, taken before Phase 5 moves it.

Phase 5 of the layered-architecture refactor lifts ``MatrixBuilder`` and
``MCEqRun._em_cascade_step_scale`` out of ``core.py`` into an ``operators/``
package, extracts the loss stencils and the kappa^2 muon damping as pure
functions, and splits ``int_m`` into ``int_m_hadr + dEdx_band``. The numbers
are pinned by the ``operators1d`` / ``operators2d`` golden sections; what is
pinned here is the *behaviour* around them, which no array value records:

* which stencil names the builder accepts, and that it rejects the rest;
* that ``op_matrix`` is built in ``MatrixBuilder.__init__`` and nowhere else,
  so ``regenerate_matrices()`` alone does not pick up a stencil change;
* that every assembly path returns a *canonical* CSR — the one property the
  golden digests cannot see, and the one MKL and cuSPARSE handles depend on;
* that the golden sweep's construct-once shortcut is the same operator as a
  fresh ``MCEqRun``, which was measured and then left in a docstring;
* the two-term invariant ``int_m == int_m_hadr + dEdx_band`` and the two
  properties of the split the golden digests cannot see: that the band is
  reused rather than recomputed, and that a settings change since the assembly
  is refused instead of answered with a mismatched half;
* the identity, invalidation and release contract of the ``_compiled_operator``
  cache;
* the dense-vs-ARPACK gate of the EM step scale, the norm fallback behind
  both, and that on a stitched 2D operator it reads Hankel mode 0 alone.

Everything except the ``regenerate_matrices``, canonical-``int_m`` and
construct-once pins is database-free: the methods under test read a bin-edge
array and a sparse matrix, so a synthetic operator exercises the production
code path without the interaction database. The three that need one use the
reduced database CI carries, never the 329 MB 2D one.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.core import MatrixBuilder, MCEqRun
from MCEq.operators import loss_stencil, scattering
from tests.golden import gen_operators1d
from tests.golden._harness import canonical_csr_problems, sparse_digest
from tests.golden._operator_sweep import (
    STENCILS,
    build_cell_operators,
    pinned_config,
)

# ---------------------------------------------------------------------------
# loss stencils: which names exist, and that each one is a different operator
# ---------------------------------------------------------------------------


def _stencil_stub(dim=31, e_min=1.0, e_max=1e6):
    """A `MatrixBuilder` stand-in carrying only what the stencil builder reads.

    ``_construct_differential_operator`` uses ``_energy_grid.b`` (the bin edges,
    for the log spacings ``h``) and ``_energy_grid.d``, the three stencil
    settings off the injected ``losses`` group and the dtype off ``grid``, and
    writes ``self.op_matrix``. Nothing else of the builder is involved, so the
    real unbound method runs against a log-uniform grid of the production
    shape. The groups are the live views, so the flat names the tests below
    monkeypatch still reach it.
    """
    edges = np.logspace(np.log10(e_min), np.log10(e_max), dim + 1)
    return SimpleNamespace(
        _energy_grid=SimpleNamespace(b=edges, d=dim),
        _grid=config.grid,
        _losses=config.losses,
    )


def _op_matrix(method, stub=None):
    stub = stub or _stencil_stub()
    config.loss_stencil_method = method
    MatrixBuilder._construct_differential_operator(stub)
    return np.copy(stub.op_matrix)


@pytest.fixture
def stencil_config(monkeypatch):
    """Restore the three stencil globals around a test."""
    for key in (
        "loss_stencil_method",
        "loss_stencil_low_upwind_rows",
        "loss_stencil_alpha0",
    ):
        monkeypatch.setattr(config, key, getattr(config, key))
    monkeypatch.setattr(config, "loss_stencil_low_upwind_rows", 8)
    monkeypatch.setattr(config, "loss_stencil_alpha0", 3.0)


def test_stencil_names_match_the_builders_own_error(stencil_config):
    """The golden sweep's `STENCILS` is the set the stencil module accepts.

    The names are read back out of the ``ValueError`` rather than restated, so
    a family added to the builder without a golden cell fails here instead of
    going unswept. ``STENCIL_METHODS`` is checked against the same message: it
    is the module's own catalogue of the dispatch, and a family listed there
    but not raised over is the same silent drift in the other direction.
    """
    with pytest.raises(ValueError) as excinfo:
        _op_matrix("no_such_stencil")
    expected = str(excinfo.value).split("Expected", 1)[1]
    raised = set(re.findall(r"'([a-z0-9_]+)'", expected))
    assert raised == set(STENCILS)
    assert raised == set(loss_stencil.STENCIL_METHODS)


@pytest.mark.parametrize("method", STENCILS)
def test_every_stencil_builds_a_finite_banded_operator(method, stencil_config):
    """Each accepted name gives a real ``(dim, dim)`` operator of bandwidth 3.

    Every family is 7-point and spans at most ``[-3, +3]``, which is what makes
    the interior row range ``range(3, dim - 3)`` uniform across the dispatch.
    """
    op = _op_matrix(method)
    assert op.shape == (31, 31)
    assert np.all(np.isfinite(op))
    assert np.any(op)
    rows, cols = np.nonzero(op)
    assert np.abs(rows - cols).max() <= 3


@pytest.mark.parametrize("method", ["centered", "biased", "upwind", "upwind2"])
def test_polynomial_stencils_annihilate_a_constant(method, stencil_config):
    """The finite-difference families are exact on ``E^0``, every row.

    A constant spectrum has zero ``d/d ln E``, so their rows sum to zero — the
    boundary rows too, which are one-sided polynomial fits of the same order.
    """
    assert np.abs(_op_matrix(method).sum(axis=1)).max() < 1e-12


@pytest.mark.parametrize(
    "method,n_boundary",
    [("expfit", 3), ("expfit_low_upwind", 8), ("expfit_low_upwind2", 8)],
)
def test_expfit_interior_rows_do_not_annihilate_a_constant(
    method, n_boundary, stencil_config
):
    """The expfit interior is exact on ``E^{-alpha}`` near ``alpha0``, not on ``E^0``.

    ``_construct_differential_operator`` fits the seven exponentials
    ``a = -alpha0 + [-1, -0.5, 0, 0.5, 1, 1.5, 2]``, and ``a = 0`` is not among
    them at ``alpha0 = 3``, so a constant is not in the span and the interior
    rows carry a residual. It is the same residual in every interior row: the
    grid is log-uniform, so the one fitted coefficient vector is divided by the
    same ``h``. The low rows (polynomial or upwind, depending on the family) and
    the top three polynomial rows do annihilate it.
    """
    op = _op_matrix(method)
    sums = op.sum(axis=1)
    interior = sums[n_boundary : op.shape[0] - 3]
    assert np.abs(interior - interior[0]).max() < 1e-12
    assert abs(interior[0]) > 1e-3
    assert np.abs(sums[:n_boundary]).max() < 1e-12
    assert np.abs(sums[op.shape[0] - 3 :]).max() < 1e-12


def test_the_seven_stencils_are_seven_different_operators(stencil_config):
    """Pairwise distinct on one grid — what makes the golden sweep informative."""
    matrices = {method: _op_matrix(method) for method in STENCILS}
    for i, a in enumerate(STENCILS):
        for b in STENCILS[i + 1 :]:
            assert not np.array_equal(matrices[a], matrices[b]), (a, b)


def test_low_upwind_rows_replace_the_boundary_layer(stencil_config):
    """``expfit_low_upwind*`` differs from ``expfit`` in the low rows only.

    ``config.loss_stencil_low_upwind_rows`` (8 here) rows are zeroed and
    refilled with the monotone upwind stencil; everything above is the expfit
    interior, so the two operators agree there.
    """
    n_low = 8
    expfit = _op_matrix("expfit")
    for method, width in (("expfit_low_upwind", 2), ("expfit_low_upwind2", 3)):
        op = _op_matrix(method)
        assert np.array_equal(op[n_low:], expfit[n_low:]), method
        assert not np.array_equal(op[:n_low], expfit[:n_low]), method
        # The replaced rows are one-sided toward higher E, `width` wide.
        for row in range(n_low):
            nonzero = np.nonzero(op[row])[0]
            assert nonzero.tolist() == list(range(row, row + width)), (method, row)


# ---------------------------------------------------------------------------
# op_matrix is built in MatrixBuilder.__init__ and nowhere else
# ---------------------------------------------------------------------------


def test_regenerate_matrices_does_not_rebuild_the_differential_operator(mceq_sib21):
    """A stencil change needs `_construct_differential_operator()` explicitly.

    It is called from ``MatrixBuilder.__init__`` only, so
    ``regenerate_matrices()`` refills the blocks against the ``op_matrix`` the
    constructor left behind and ``int_m`` does not move. The golden generators
    depend on this — they call the method themselves — and a Phase 5 that
    changes it changes when a config edit takes effect.
    """
    mceq = mceq_sib21
    saved = config.loss_stencil_method
    before = mceq.int_m.copy()
    try:
        config.loss_stencil_method = "centered"
        mceq.regenerate_matrices()
        assert (mceq.int_m != before).nnz == 0, (
            "regenerate_matrices() picked up the stencil change on its own"
        )

        mceq.matrix_builder._construct_differential_operator()
        mceq.regenerate_matrices()
        assert (mceq.int_m != before).nnz != 0
    finally:
        config.loss_stencil_method = saved
        mceq.matrix_builder._construct_differential_operator()
        mceq.regenerate_matrices()
    assert (mceq.int_m != before).nnz == 0, "the fixture was not restored"
    assert np.array_equal(mceq.int_m.data, before.data)


# ---------------------------------------------------------------------------
# canonical CSR form: the property no golden digest can see
# ---------------------------------------------------------------------------


def _clean_csr():
    """A canonical CSR with two entries in its first row, ready to corrupt."""
    return sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]]))


def _duplicated_csr():
    """:func:`_clean_csr`, with its (0, 0) entry split into a duplicate pair."""
    return sp.csr_matrix(
        (
            np.array([0.5, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([0, 0, 1, 1, 2, 0, 2]),
            np.array([0, 3, 5, 7]),
        ),
        shape=(3, 3),
    )


def test_the_canonical_check_rejects_a_de_canonicalised_matrix():
    """The negative control: without it the checks below prove nothing.

    Four corruptions, all built in memory and all of the *same* operator —
    ``!=`` gives an empty difference against the clean one every time:

    * two entries of a row transposed, with the flags primed first. scipy
      caches them on read, so afterwards the matrix claims to be sorted and
      canonical over data that is neither, and a check written against the
      flags would pass. ``sparse_digest`` cannot see it at all: sorting the
      copy undoes the transposition, so the hash is unchanged.
    * one entry split into a duplicate pair. Its columns still ascend, just not
      strictly, so ``has_sorted_indices`` is honestly True and scipy computes
      ``has_canonical_format`` as False — here the buffer scan is what catches
      it.
    * the same duplicate pair with both flags forced True, which is what an
      in-place edit or a hand-built CSR leaves behind.
    * a canonical matrix whose flag says otherwise, which misleads a backend
      into work it does not need and, worse, is how the checks above would be
      defeated in the other direction.
    """
    clean = _clean_csr()
    assert canonical_csr_problems(clean, "int_m") == []

    transposed = _clean_csr()
    assert transposed.has_sorted_indices and transposed.has_canonical_format
    transposed.indices[[0, 1]] = transposed.indices[[1, 0]]
    transposed.data[[0, 1]] = transposed.data[[1, 0]]
    problems = canonical_csr_problems(transposed, "int_m")
    assert any("descending column step" in line for line in problems), problems
    assert any("has_sorted_indices is True" in line for line in problems), problems
    assert (transposed != clean).nnz == 0, "the corruption changed the operator"
    assert sparse_digest(transposed) == sparse_digest(clean), (
        "sparse_digest can see an unsorted row after all — this test is stale"
    )

    duplicated = _duplicated_csr()
    assert (duplicated != clean).nnz == 0, "the corruption changed the operator"
    # Its columns still ascend, just not strictly, so `has_sorted_indices` is
    # honestly True and scipy computes `has_canonical_format` as False. The
    # buffer scan is what has to catch this one.
    assert canonical_csr_problems(duplicated, "int_m") == [
        "int_m: 1 duplicate (row, column) entry/ies"
    ]

    lying = _duplicated_csr()
    lying.has_sorted_indices = True
    lying.has_canonical_format = True
    problems = canonical_csr_problems(lying, "int_m")
    assert any("duplicate (row, column)" in line for line in problems), problems
    assert any("has_canonical_format is True" in line for line in problems), problems

    stale = _clean_csr()
    stale.has_sorted_indices = False
    problems = canonical_csr_problems(stale, "int_m")
    assert len(problems) == 2, problems
    assert all("and the buffers say True" in line for line in problems), problems


class _StubParticle:
    """A cascade species as the assembly and the muon damping see one."""

    def __init__(self, mceqidx, name, pdg_id, lidx, dim):
        self.mceqidx = mceqidx
        self.name = name
        self.pdg_id = pdg_id
        self.lidx = lidx
        self.uidx = lidx + dim
        self.mass = scattering.MUON_MASS


class _StubPman:
    """The two lookup tables `_csr_from_blocks` and the damping index with."""

    def __init__(self, species):
        self.mceqidx2pref = {p.mceqidx: p for p in species}
        self.pdg2pref = {(p.pdg_id, 0): p for p in species}

    def __getitem__(self, key):
        return self.pdg2pref[key]


class _BlocksStub:
    """A `MatrixBuilder` stand-in carrying only what the assembly reads.

    ``_csr_from_blocks`` reads ``is_2d``, the three dimensions, ``k_grid``,
    ``_grid.dtype`` and the particle manager's ``mceqidx2pref``;
    ``_muon_scattering_damping`` additionally reads ``_energy_grid.c``,
    ``pdg2pref`` and ``_physics.muon_multiple_scattering``. Both are borrowed
    unbound, so the production assembly runs — including the COO scatter, the
    ``block_diag`` stitch and the ``kappa^2`` damping — with no database. The
    two settings groups are the live views, so `assembly_config`'s
    monkeypatches of the flat names still reach them.

    Three species of ``dim`` bins each, one of them a muon so the damping has
    somewhere to land. ``k_grid`` starts at ``kappa = 0``, as a real one does,
    which is the mode the damping skips.
    """

    _csr_from_blocks = MatrixBuilder._csr_from_blocks
    _muon_scattering_damping = MatrixBuilder._muon_scattering_damping

    def __init__(self, is_2d, n_k=3, dim=8):
        self._grid = config.grid
        self._physics = config.physics
        self.is_2d = is_2d
        self.n_k = n_k if is_2d else 1
        self.k_grid = np.arange(self.n_k, dtype=np.float64)
        self.dim = dim
        species = [
            _StubParticle(0, "gamma", 22, 0 * dim, dim),
            _StubParticle(1, "mu+", 13, 1 * dim, dim),
            _StubParticle(2, "pi+", 211, 2 * dim, dim),
        ]
        self.dim_states = len(species) * dim
        self._pman = _StubPman(species)
        self._energy_grid = SimpleNamespace(c=np.logspace(-1, 3, dim), d=dim)

    def blocks(self):
        """``(child, parent) -> block``, shaped as the builder expects.

        The muon self-block has a populated main diagonal, which is what makes
        the damping a duplicate rather than a new entry; the pion -> muon block
        is lower-triangular, the shape a real production block has; the gamma
        row is left empty, so the assembly also has to cope with rows that
        carry nothing.
        """
        dim = self.dim
        diagonal = np.diag(-np.linspace(1.0, 2.0, dim))
        triangular = np.tril(np.full((dim, dim), 0.25))
        blocks = {(1, 1): diagonal, (1, 2): triangular}
        if self.is_2d:
            return {
                key: np.repeat(value[None, :, :], self.n_k, axis=0)
                for key, value in blocks.items()
            }
        return blocks


@pytest.fixture
def assembly_config(monkeypatch):
    """Fix the two globals `_csr_from_blocks` reads through its groups."""
    monkeypatch.setattr(config, "floatlen", np.float64)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)


@pytest.mark.parametrize("is_2d", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize("scatter", [False, True], ids=["ms_off", "ms_on"])
def test_every_assembly_path_returns_a_canonical_operator(
    is_2d, scatter, assembly_config
):
    """Both branches of `_csr_from_blocks`, at either setting of the damping.

    The 2D branch is the one that can lose the property: it scatters the
    channel slabs into COO triplets and adds the damping as *further* entries
    at positions the channel diagonal already holds, so canonical form there
    rests on ``coo.tocsr()`` summing duplicates and on the explicit
    ``sort_indices()`` after it. The 1D branch goes through
    ``csr_matrix(dense)`` and is canonical by construction — pinned anyway,
    because Phase 5 splits ``int_m`` into ``int_m_hadr + dEdx_band`` and a
    hand-built CSR on either branch would be silent in all 28 golden cells.
    """
    stub = _BlocksStub(is_2d)
    matrix = stub._csr_from_blocks(stub.blocks(), apply_muon_scattering=scatter)
    assert matrix.shape == (stub.n_k * stub.dim_states,) * 2
    assert canonical_csr_problems(matrix, "int_m") == []


def test_the_2d_damping_is_added_over_an_occupied_diagonal(assembly_config):
    """So the 2D path really does depend on duplicates being summed.

    Same nonzero count as the undamped assembly and different values: every
    damping entry fell on a ``(row, column)`` the muon self-block already
    occupied. Without this the canonical check above could be passing on an
    assembly that never produced a duplicate in the first place.
    """
    stub = _BlocksStub(is_2d=True)
    damped = stub._csr_from_blocks(stub.blocks(), apply_muon_scattering=True)
    plain = stub._csr_from_blocks(stub.blocks(), apply_muon_scattering=False)

    assert damped.nnz == plain.nnz, "the damping added entries of its own"
    assert (damped != plain).nnz > 0, "the damping did not reach the operator"
    # kappa = 0 is untouched; every other mode is damped.
    per_mode = [
        (damped - plain)[
            k * stub.dim_states : (k + 1) * stub.dim_states,
            k * stub.dim_states : (k + 1) * stub.dim_states,
        ].nnz
        for k in range(stub.n_k)
    ]
    assert per_mode[0] == 0 and all(n == stub.dim for n in per_mode[1:]), per_mode


def test_the_assembled_1d_operators_are_canonical(mceq_sib21):
    """The real `int_m` / `dec_m`, in the state a backend binds them in.

    The synthetic paths above pin the assembly code; this pins the objects a
    run actually hands to ``_compiled_operator``, on the reduced database CI
    carries. The 2D operators would need the 329 MB FLUKA rc7 database, so
    they are covered by the synthetic 2D branch here and by the assertion the
    ``operators2d`` generator makes on all fourteen of its cells.
    """
    for name in ("int_m", "dec_m"):
        assert canonical_csr_problems(getattr(mceq_sib21, name), name) == []


# ---------------------------------------------------------------------------
# int_m == int_m_hadr + dEdx_band
# ---------------------------------------------------------------------------


def _pruned_digest(matrix):
    """`sparse_digest` of a copy with explicit zeros dropped.

    A CSR add prunes a cancelling entry on scipy 1.18 but ``pyproject``
    declares plain ``"scipy"``, so a version that keeps it would otherwise
    hash differently over the same operator.
    """
    csr = matrix.tocsr(copy=True)
    csr.eliminate_zeros()
    return sparse_digest(csr)


def test_int_m_is_the_sum_of_the_hadronic_part_and_the_loss_band(mceq_sib21):
    """The D27 split, bitwise, on the operator the fixture built.

    ``int_m`` is not reassembled from the two halves — the accumulation that
    produces it is untouched and the halves are read off it — so what this
    states is that the two exposed operators really are its parts, which no
    ``operators*`` digest can see. fp64 here (``floatlen`` None); the sum is
    fp64 because ``dEdx_band`` is, and the band is deliberately *not* cast to
    ``grid.dtype`` first: ``block += band`` rounds the promoted sum once, and a
    cast band would round twice and move about one entry in 3000 at float32.
    """
    mb = mceq_sib21.matrix_builder
    int_m, hadr, band = mb.int_m, mb.int_m_hadr, mb.dEdx_band

    assert band.dtype == np.float64, "the band must stay in the dtype it is built in"
    assert hadr.dtype == int_m.dtype
    assert hadr.shape == band.shape == int_m.shape
    assert band.nnz and hadr.nnz > band.nnz

    total = (hadr + band).astype(int_m.dtype)
    total.eliminate_zeros()
    total.sort_indices()
    assert _pruned_digest(total) == _pruned_digest(int_m)
    for name, matrix in (("int_m_hadr", hadr), ("dEdx_band", band)):
        assert canonical_csr_problems(matrix.tocsr(), name) == []


def test_the_two_halves_are_cached_and_reuse_the_stashed_band(mceq_sib21, monkeypatch):
    """Repeat access is the same object, and `cont_loss_operator` is not re-run.

    The band arrays are stashed as the accumulation adds them, so assembling
    ``dEdx_band`` cannot re-enter ``cont_loss_operator`` — which under
    ``losses.average_operator`` goes through ``np.linalg.matrix_power`` and
    which a stencil changed since the assembly would answer differently.
    ``boom`` is what says the property is not quietly recomputing it.
    """
    mb = mceq_sib21.matrix_builder
    mb._int_m_hadr = mb._dEdx_band = None  # force a fresh assembly

    def boom(self, pdg_id):
        raise AssertionError("dEdx_band recomputed the band instead of reusing it")

    monkeypatch.setattr(MatrixBuilder, "cont_loss_operator", boom)
    band, hadr = mb.dEdx_band, mb.int_m_hadr
    assert mb.dEdx_band is band
    assert mb.int_m_hadr is hadr


def test_a_settings_change_since_the_assembly_is_refused(mceq_sib21, monkeypatch):
    """Neither half is served against settings the live ``int_m`` was not built at.

    Both are assembled on demand through ``_csr_from_blocks``, which reads
    ``grid.dtype`` and the muon-damping flag at call time, so a global changed
    since ``construct_matrices`` would return a half of an operator nobody
    holds. ``floatlen`` is the trigger the 1D fixture has: the damping flag
    reaches the assembly only when ``is_2d``, so flipping it here is genuinely
    a no-op and the key says so.
    """
    mb = mceq_sib21.matrix_builder
    assert mb.int_m_hadr is not None

    monkeypatch.setattr(
        config, "muon_multiple_scattering", not config.muon_multiple_scattering
    )
    assert mb.int_m_hadr is not None, "the 1D assembly does not read the damping flag"

    monkeypatch.setattr(config, "floatlen", np.float32)
    for name in ("int_m_hadr", "dEdx_band"):
        with pytest.raises(RuntimeError, match="not be a part of the int_m in hand"):
            getattr(mb, name)


# ---------------------------------------------------------------------------
# the golden sweep's construct-once shortcut == a fresh MCEqRun
# ---------------------------------------------------------------------------

#: The two cells the equivalence is measured on. ``upwind`` takes the
#: early-return branch of the stencil dispatch and the default composite takes
#: the long one, so between them they cover both shapes of
#: ``_construct_differential_operator``.
EQUIVALENCE_CELLS = (("upwind", True), ("expfit_low_upwind2", True))


def _operator_digests(int_m, dec_m):
    return sparse_digest(int_m), sparse_digest(dec_m)


def _reused_and_fresh_digests():
    """Both arms of the equivalence, under the `operators1d` config pins.

    The reused arm is the sweep's own :func:`build_cell_operators`, on one run
    constructed at the pinned default — the generator's exact sequence. The
    fresh arm sets the two globals and constructs a whole ``MCEqRun`` per cell,
    which builds ``op_matrix`` in ``MatrixBuilder.__init__`` and the matrices
    from it, with no state carried in from another cell.
    """
    with pinned_config(gen_operators1d.CONFIG_PINS, gen_operators1d.ADV_SET_PINS):
        mceq = MCEqRun(**gen_operators1d.RUN_KWARGS)
        try:
            reused = {
                cell: _operator_digests(*build_cell_operators(mceq, *cell))
                for cell in EQUIVALENCE_CELLS
            }
        finally:
            mceq.close()

        fresh = {}
        for stencil, scatter in EQUIVALENCE_CELLS:
            config.loss_stencil_method = stencil
            config.muon_multiple_scattering = scatter
            run = MCEqRun(**gen_operators1d.RUN_KWARGS)
            try:
                fresh[stencil, scatter] = _operator_digests(run.int_m, run.dec_m)
            finally:
                run.close()
    return reused, fresh


def _assert_reused_equals_fresh():
    reused, fresh = _reused_and_fresh_digests()
    for cell in EQUIVALENCE_CELLS:
        assert reused[cell] == fresh[cell], (
            f"{cell}: the sweep's reused run and a fresh MCEqRun disagree. "
            f"The operators* sections build one run per database and mutate "
            f"two config globals per cell, so a builder that carries state "
            f"across cells leaves them self-consistent — comparison and "
            f"regeneration move together — while they stop pinning a fresh "
            f"build.\nreused {reused[cell]}\nfresh  {fresh[cell]}"
        )


def test_the_golden_sweeps_reused_run_equals_a_fresh_one():
    """The 28-cell equivalence the two `operators*` sections rest on.

    Measured once at 28 of 28 cells and then left in a docstring; this is the
    gate. Two cells on the reduced database, which is the one ``operators1d``
    uses and the one CI carries: 2.2 s in a cold process, 0.9 s in a session
    that has already opened it.
    """
    _assert_reused_equals_fresh()


def test_a_band_memoised_on_the_builder_breaks_the_equivalence(monkeypatch):
    """The negative control, in the exact shape Phase 5's D27 invites.

    D27 splits ``int_m`` into ``int_m_hadr + dEdx_band``, which puts a band
    matrix on ``self``. Memoised by pdg id — the obvious way, since it does not
    change during a solve — it goes stale on a stencil change, and the sweep
    notices nothing: both the comparison and the regeneration run through the
    same reused run. Here that cache is monkeypatched in and the equivalence
    goes red, which is what says the test above can fail.
    """
    real = MatrixBuilder.cont_loss_operator

    def memoised(self, pdg_id):
        cache = self.__dict__.setdefault("_dEdx_band_cache", {})
        key = tuple(pdg_id)
        if key not in cache:
            cache[key] = real(self, pdg_id)
        return cache[key]

    monkeypatch.setattr(MatrixBuilder, "cont_loss_operator", memoised)
    with pytest.raises(AssertionError, match="disagree"):
        _assert_reused_equals_fresh()


# ---------------------------------------------------------------------------
# the _compiled_operator cache
# ---------------------------------------------------------------------------


class _CacheStub:
    """The smallest object `MCEqRun._compiled_operator` and `close` work on.

    Both are borrowed unbound so the stub runs the production code, including
    the ``self.__dict__`` bookkeeping the cache is built out of.
    """

    _compiled_operator = MCEqRun._compiled_operator
    close = MCEqRun.close

    def __init__(self, dim=6):
        self.int_m = sp.csr_matrix(_toy_matrix(dim, 0.3))
        self.dec_m = sp.csr_matrix(_toy_matrix(dim, 0.1))


def _toy_matrix(dim, offdiag):
    m = -np.eye(dim)
    m[np.arange(dim - 1), np.arange(1, dim)] = offdiag
    return m


def _toy_sec_ops(dim=6, n_k=2):
    """A synthetic secant operator set that `compile_operator` accepts.

    ``secant_layout`` needs the coupled modes to be the leading block and the
    low-E columns sorted; ``secant_coupling`` needs the five dense operators.
    """
    n_p, n_g = 1, 2
    return {
        "n_k": n_k,
        "P": np.arange(n_p),
        "low_e_idx": np.arange(n_g),
        "T_P": np.zeros((n_p, n_k)),
        "T_PP": np.zeros((n_p, n_p)),
        "V": np.eye(n_p),
        "Vi": np.eye(n_p),
        "lam": np.ones(n_p),
    }


def test_compiled_operator_is_cached_by_identity():
    """Repeat access returns the same object; a rebound matrix rebuilds it."""
    stub = _CacheStub()
    op = stub._compiled_operator()
    assert stub._compiled_operator() is op

    stub.int_m = stub.int_m.copy()  # equal values, new object
    rebuilt = stub._compiled_operator()
    assert rebuilt is not op
    assert np.array_equal(rebuilt.d_int, op.d_int)

    stub.dec_m = stub.dec_m.copy()
    assert stub._compiled_operator() is not rebuilt


def test_compiled_operator_keeps_a_paraxial_and_a_coupled_entry():
    """The two layouts coexist under one cache, keyed on `sec_ops is None`."""
    stub = _CacheStub()
    sec_ops = _toy_sec_ops()
    paraxial = stub._compiled_operator()
    coupled = stub._compiled_operator(sec_ops)

    assert coupled is not paraxial
    assert not paraxial.coupled and coupled.coupled
    assert stub._compiled_operator() is paraxial
    assert stub._compiled_operator(sec_ops) is coupled

    # The operator set is compared by identity, not by value: an equal but
    # distinct set is a different transport and has to recompile.
    assert stub._compiled_operator(_toy_sec_ops()) is not coupled


def test_close_releases_the_compiled_operator():
    """`close()` drops the cache, and the next access lazily rebuilds it."""
    stub = _CacheStub()
    op = stub._compiled_operator()
    assert "_operator_cache" in stub.__dict__

    stub.close()
    assert "_operator_cache" not in stub.__dict__
    assert stub._compiled_operator() is not op
    stub.close()  # idempotent


def test_regenerate_matrices_invalidates_the_compiled_operator(mceq_sib21):
    """The production wiring: new matrices, new compiled operator.

    The cache holds the CSR buffers a backend binds its handles to, so an
    ``int_m`` rebuilt behind a stale entry would be integrated with the old
    operator.
    """
    mceq = mceq_sib21
    op = mceq._compiled_operator()
    assert mceq._compiled_operator() is op

    mceq.regenerate_matrices()
    assert mceq._compiled_operator() is not op

    mceq.regenerate_matrices(skip_decay_matrix=True)
    # dec_m is the object the builder already held; int_m is new, which is
    # enough for the identity check to fire.
    assert mceq._compiled_operator() is not op


# ---------------------------------------------------------------------------
# EM step scale: the dense-vs-ARPACK gate and the norm fallback
# ---------------------------------------------------------------------------

#: Off-diagonal EM block ``w * (J - I)``: a positive matrix, so Perron-Frobenius
#: gives it the unique real dominant eigenvalue ``w * (n - 1)``, well separated
#: from the ``-w`` of multiplicity ``n - 1``. ARPACK converges on it in a few
#: iterations, which is what makes it usable for a gate test — the real EM block
#: is non-normal enough that ``eigs`` fails on it (see
#: ``test_em_cascade_step_scale_dense_for_large_nonnormal_block``).
EM_BLOCK_N = 60
EM_BLOCK_W = 0.01
EM_BLOCK_RHO = EM_BLOCK_W * (EM_BLOCK_N - 1)


def _em_matrix(block, dim):
    """One ``dim x dim`` operator block with `block` in its leading corner."""
    matrix = -np.eye(dim)  # benign diagonal, stripped by the off-split
    n_em = block.shape[0]
    matrix[:n_em, :n_em] += block
    return matrix


class _EmStub:
    _em_cascade_step_scale = MCEqRun._em_cascade_step_scale

    def __init__(self, block, n_em, dim):
        self.int_m = sp.csr_matrix(_em_matrix(block, dim))
        self.pman = SimpleNamespace(
            all_particles=[
                SimpleNamespace(is_em=index < n_em, lidx=index, uidx=index + 1)
                for index in range(dim)
            ]
        )
        self._em_step_scale_cache = None


def _positive_em_block(scale=1.0):
    return scale * EM_BLOCK_W * (np.ones((EM_BLOCK_N, EM_BLOCK_N)) - np.eye(EM_BLOCK_N))


def _positive_em_stub():
    return _EmStub(_positive_em_block(), EM_BLOCK_N, EM_BLOCK_N + 4)


@pytest.mark.parametrize(
    "threshold,dense,arpack",
    [(EM_BLOCK_N, 1, 0), (EM_BLOCK_N - 1, 0, 1)],
    ids=["at-the-threshold-dense", "one-over-arpack"],
)
def test_em_step_scale_gate_is_block_size_vs_em_step_dense_eig_max(
    threshold, dense, arpack, monkeypatch
):
    """``n <= config.em_step_dense_eig_max`` picks dense ``eigvals``, else ARPACK.

    Both routes have to return the same spectral radius; the counters are what
    say which one ran, since the number alone cannot tell them apart.
    """
    import scipy.sparse.linalg

    calls = {"dense": 0, "arpack": 0}
    real_eigvals, real_eigs = np.linalg.eigvals, scipy.sparse.linalg.eigs

    def count_eigvals(matrix):
        calls["dense"] += 1
        return real_eigvals(matrix)

    def count_eigs(matrix, **kwargs):
        calls["arpack"] += 1
        return real_eigs(matrix, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvals", count_eigvals)
    monkeypatch.setattr(scipy.sparse.linalg, "eigs", count_eigs)
    monkeypatch.setattr(config, "em_step_dense_eig_max", threshold)

    stub = _positive_em_stub()
    assert MCEqRun._em_cascade_step_scale(stub) == pytest.approx(EM_BLOCK_RHO, rel=1e-8)
    assert calls == {"dense": dense, "arpack": arpack}


def test_em_step_scale_falls_back_to_the_smaller_induced_norm(monkeypatch):
    """When the chosen reduction raises, ``min(||.||_1, ||.||_inf)`` stands in.

    The block is a nilpotent forward shift: spectral radius 0, both induced
    norms 1. The fallback is an upper bound on the radius, so it never
    under-caps the step — and it is visibly not the radius, which is how the
    branch is identified.
    """
    import scipy.sparse.linalg

    def boom(matrix, **kwargs):
        raise RuntimeError("ARPACK did not converge")

    monkeypatch.setattr(scipy.sparse.linalg, "eigs", boom)
    monkeypatch.setattr(config, "em_step_dense_eig_max", 4)

    n_em, dim = 8, 12
    block = np.zeros((n_em, n_em))
    block[np.arange(n_em - 1), np.arange(1, n_em)] = 1.0
    stub = _EmStub(block, n_em, dim)

    assert np.max(np.abs(np.linalg.eigvals(block))) == pytest.approx(0.0, abs=1e-12)
    assert MCEqRun._em_cascade_step_scale(stub) == pytest.approx(1.0)


def test_em_step_scale_is_cached_against_int_m_identity(monkeypatch):
    """One spectral radius per ``int_m``; a rebuilt matrix recomputes it."""
    monkeypatch.setattr(config, "em_step_dense_eig_max", 4000)
    calls = []
    real_eigvals = np.linalg.eigvals
    monkeypatch.setattr(
        np.linalg, "eigvals", lambda m: (calls.append(1), real_eigvals(m))[1]
    )

    stub = _positive_em_stub()
    first = MCEqRun._em_cascade_step_scale(stub)
    assert MCEqRun._em_cascade_step_scale(stub) == first
    assert len(calls) == 1

    stub.int_m = stub.int_m.copy()
    assert MCEqRun._em_cascade_step_scale(stub) == pytest.approx(first)
    assert len(calls) == 2


def test_em_step_scale_reads_hankel_mode_0_of_a_stitched_operator(monkeypatch):
    """``p.lidx`` / ``p.uidx`` are offsets inside one ``dim_states`` block.

    So on the ``(n_k * dim_states)`` operator the 2D assembly produces, the
    method slices the EM rows and columns of mode 0 and never sees the rest.
    The two modes here are given different spectral radii — mode 1 is four
    times mode 0 — so a value that covered both would be visibly different
    rather than accidentally equal.

    This is what the fourteen ``cells/*/em_step_scale`` keys of the
    ``operators2d`` golden section were said to pin. They could not: that
    fixture has ``disabled_particles = [11, -11]`` and ``enable_em`` off, so
    gamma is the only EM species, its off-diagonal block is empty in all 48
    modes and not merely in mode 0, and all fourteen keys were exactly 0.0
    whichever modes were read. They also carried a rel-L2 1e-9 entry that
    ``compare_key`` short-circuits to exact equality in front of an all-zero
    reference. The keys and the entry are gone; the behaviour is pinned here,
    database-free and therefore in CI, where a phase that makes the indexing
    ``n_k``-aware sees a named test instead of nothing at all.
    """
    monkeypatch.setattr(config, "em_step_dense_eig_max", 4000)
    mode_scales = (1.0, 4.0)

    stub = _positive_em_stub()
    dim_states = stub.int_m.shape[0]
    stub.int_m = sp.block_diag(
        [
            sp.csr_matrix(_em_matrix(_positive_em_block(scale), dim_states))
            for scale in mode_scales  # kappa-block per Hankel mode
        ],
        format="csr",
    )
    stub._em_step_scale_cache = None

    assert stub.int_m.shape == (len(mode_scales) * dim_states,) * 2
    assert MCEqRun._em_cascade_step_scale(stub) == pytest.approx(
        mode_scales[0] * EM_BLOCK_RHO, rel=1e-8
    ), "the EM step scale stopped being the mode-0 value"
    assert MCEqRun._em_cascade_step_scale(stub) != pytest.approx(
        max(mode_scales) * EM_BLOCK_RHO, rel=1e-8
    ), (
        "the EM step scale now covers every Hankel mode. That is the Phase 6 "
        "fix, not a regression: update this test, and note that the "
        "operators1d golden section is 1D (n_k == 1) and does not move."
    )
