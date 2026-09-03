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
* the identity, invalidation and release contract of the ``_compiled_operator``
  cache;
* the dense-vs-ARPACK gate of the EM step scale, and the norm fallback behind
  both.

Everything except the ``regenerate_matrices`` pin is database-free: the two
methods under test read a bin-edge array and a sparse matrix respectively, so
a synthetic operator exercises the production code path without the
interaction database.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.core import MatrixBuilder, MCEqRun
from tests.golden._operator_sweep import STENCILS

# ---------------------------------------------------------------------------
# loss stencils: which names exist, and that each one is a different operator
# ---------------------------------------------------------------------------


def _stencil_stub(dim=31, e_min=1.0, e_max=1e6):
    """A `MatrixBuilder` stand-in carrying only what the stencil builder reads.

    ``_construct_differential_operator`` uses ``_energy_grid.b`` (the bin edges,
    for the log spacings ``h``) and ``_energy_grid.d``, and writes
    ``self.op_matrix``. Nothing else of the builder is involved, so the real
    unbound method runs against a log-uniform grid of the production shape.
    """
    edges = np.logspace(np.log10(e_min), np.log10(e_max), dim + 1)
    return SimpleNamespace(_energy_grid=SimpleNamespace(b=edges, d=dim))


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
    """The golden sweep's `STENCILS` is the set `core.py` says it accepts.

    The names are read back out of the ``ValueError`` rather than restated, so
    a family added to the builder without a golden cell fails here instead of
    going unswept.
    """
    with pytest.raises(ValueError) as excinfo:
        _op_matrix("no_such_stencil")
    expected = str(excinfo.value).split("Expected", 1)[1]
    assert set(re.findall(r"'([a-z0-9_]+)'", expected)) == set(STENCILS)


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

    ``core.py`` calls it from ``MatrixBuilder.__init__`` only, so
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


class _EmStub:
    _em_cascade_step_scale = MCEqRun._em_cascade_step_scale

    def __init__(self, block, n_em, dim):
        matrix = -np.eye(dim)  # benign diagonal, stripped by the off-split
        matrix[:n_em, :n_em] += block
        self.int_m = sp.csr_matrix(matrix)
        self.pman = SimpleNamespace(
            all_particles=[
                SimpleNamespace(is_em=index < n_em, lidx=index, uidx=index + 1)
                for index in range(dim)
            ]
        )
        self._em_step_scale_cache = None


def _positive_em_stub():
    block = EM_BLOCK_W * (np.ones((EM_BLOCK_N, EM_BLOCK_N)) - np.eye(EM_BLOCK_N))
    return _EmStub(block, EM_BLOCK_N, EM_BLOCK_N + 4)


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
