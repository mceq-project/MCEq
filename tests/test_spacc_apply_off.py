"""``SpaccApplyOff`` against a fake Accelerate handle, on every platform.

The real ``MCEq.spacc`` loads ``libspacc`` at import and only macOS builds
carry one, so every other Accelerate test skips elsewhere. What is
platform-independent is the *contract* between :class:`MCEq.solvers.
SpaccApplyOff` and the two ctypes entry points it drives:

    gemv_ctargs(alpha, cx, cy)                  y += alpha * M * x
    gemm_ctargs(alpha, nrhs, cB, ldb, cC, ldc)  C += alpha * M * B

with ``B`` and ``C`` COLUMN-major and ``ldb`` / ``ldc`` their column
strides, while the driver's own ``(dim, K)`` buffers are row-major. The
fake below implements exactly that, rebuilding every operand from the raw
pointer it is handed and honouring the leading dimensions — so a missed
transpose, a wrong stride or a tile pointed at the wrong column shows up
as a numerical difference instead of passing silently.

The reference is the same driver on the scipy binding: identical operator,
identical stage order, only ``apply_off`` differs.

The backend lifecycle tests live here for the same reason: the fakes make
``SpaccApplyOff`` bindable on every platform, so the release contract of
``close()`` can be pinned for all three host bindings in one place, without
a production database and without a platform skip.
"""

from __future__ import annotations

import ctypes
import gc
import tracemalloc
import weakref

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.operator_assembly import compile_operator
from MCEq.solvers import (
    HostBackend,
    ScipyApplyOff,
    SpaccApplyOff,
    etd2_driver,
    mkl_backend,
)


class FakeSpaccMatrix:
    """Ctypes-level stand-in for :class:`MCEq.spacc.SpaccMatrix`.

    Takes pointers, not arrays: ``B`` and ``C`` are rebuilt with
    ``np.ctypeslib.as_array`` over ``ldb`` / ``ldc``-strided columns, which
    is the only way the fake can catch a caller that hands over a
    row-major buffer or an off-by-one tile offset. ``calls`` records
    ``(kind, alpha, nrhs)`` per invocation, so a test can assert which
    entry point ran and with which tile widths.

    The multiply itself is scipy's CSR product, the same routine the
    reference binding runs, so the two agree to the bit once the layout is
    right: any deviation the comparison reports is a layout or tiling
    error, never BLAS summation noise.
    """

    def __init__(self, csr, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.csr = sp.csr_matrix(csr, dtype=self.dtype)
        self.nnz = csr.nnz
        self.dim_rows, self.dim_cols = csr.shape
        self._ct = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
        self.calls = []

    def _vec(self, ptr, n):
        """The ``n`` contiguous elements at ``ptr`` as a writable view."""
        return np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(self._ct)), shape=(n,)
        )

    def _block(self, ptr, ld, nrhs):
        """The column-major block at ``ptr``: ``nrhs`` consecutive columns of
        ``ld`` elements each, transposed to ``(dim_rows, nrhs)`` — the layout
        Accelerate's ``CblasColMajor`` walks. A writable view."""
        return self._vec(ptr, ld * nrhs).reshape(nrhs, ld).T[: self.dim_rows]

    def close(self):
        """Idempotent slot release, as the real wrapper has."""
        self.csr = None

    def gemv_ctargs(self, alpha, cx, cy):
        self.calls.append(("gemv", float(alpha), 1))
        x = self._vec(cx, self.dim_cols)
        y = self._vec(cy, self.dim_rows)
        y += (self.csr @ x) * alpha

    def gemm_ctargs(self, alpha, nrhs, cB, ldb, cC, ldc):
        self.calls.append(("gemm", float(alpha), int(nrhs)))
        B = self._block(cB, int(ldb), int(nrhs))
        C = self._block(cC, int(ldc), int(nrhs))
        C += (self.csr @ B) * alpha


def _problem(K, seed=11, size=23, nsteps=17):
    """A small ``(int_m, dec_m, phi0, path)`` whose off-diagonals are dense
    and asymmetric enough that a transposed operand cannot pass by accident."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.015] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.008] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    return dict(
        nsteps=nsteps,
        dX=np.full(nsteps, 0.1),
        # Away from 1.0 at every step, so an alpha of ri is distinguishable
        # from an alpha the binding never applied.
        rho_inv=np.linspace(1.3, 2.0, nsteps),
        int_m=sp.csr_matrix(A),
        dec_m=sp.csr_matrix(B),
        phi0=rng.uniform(0.1, 1.0, size=(size, K)),
    )


def _drop_off_diagonal(csr):
    """``csr`` with its off-diagonal dropped — an nnz = 0 block for the split,
    with the diagonal kept so the solve stays a real ETD2 run."""
    return sp.csr_matrix(np.diag(np.diag(csr.toarray())))


def _fakes(op, dtype=np.float64):
    """The two fake handles of a compiled operator's split; ``None`` for an
    empty block, exactly as :func:`MCEq.solvers.accelerate_backend` builds
    them."""
    return (
        FakeSpaccMatrix(op.int_off, dtype) if op.int_off.nnz else None,
        FakeSpaccMatrix(op.dec_off, dtype) if op.dec_off.nnz else None,
    )


def _path(p, per_lane):
    """``(dX, rho_inv)`` shared across lanes, or one per lane."""
    if not per_lane:
        return p["dX"], p["rho_inv"]
    # (nsteps, K) per-lane paths with a distinct ri per column: this is what
    # hands the binding a (K,) lane row instead of a scalar.
    K = p["phi0"].shape[1]
    scale = np.linspace(0.8, 1.3, K)
    dX = np.repeat(p["dX"][:, None], K, axis=1)
    rho_inv = p["rho_inv"][:, None] * scale[None, :]
    return dX, rho_inv


def _run(p, make_apply_off, per_lane=False, nsteps=None, dtype=np.float64):
    """The driver over ``p`` on the binding ``make_apply_off`` builds."""
    op = compile_operator(p["int_m"], p["dec_m"], None)
    be = HostBackend(op, make_apply_off(op, dtype), dtype)
    dX, rho_inv = _path(p, per_lane)
    n = p["nsteps"] if nsteps is None else nsteps
    try:
        sol, _ = etd2_driver(n, dX[:n], rho_inv[:n], be, p["phi0"], [])
    finally:
        be.close()
    return sol


def _scipy_binding(op, dtype=np.float64):
    return ScipyApplyOff(op.int_off, op.dec_off, dtype)


def _spacc_binding(op, dtype=np.float64):
    return SpaccApplyOff(*_fakes(op, dtype), dtype=dtype)


def _rel(a, b):
    """Max elementwise relative deviation, floored at 1e-12 of the peak so a
    near-zero entry cannot dominate."""
    scale = np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-12)
    return float(np.max(np.abs(a - b) / scale))


@pytest.mark.parametrize("K", [1, 3, 70])
@pytest.mark.parametrize("per_lane", [False, True], ids=["scalar_ri", "lane_ri"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
def test_spacc_apply_off_matches_scipy(K, per_lane, dtype):
    """The Accelerate binding reproduces the scipy binding at the same dtype.

    K = 70 crosses the 64-column ``_SPACC_SPMM_TILE`` boundary, where a
    tiling off-by-one is visible and nowhere else. Both sides run the same
    scipy product in the same order, so the comparison isolates layout: at
    either precision the two agree to the bit unless a stride is wrong.
    """
    p = _problem(K)
    ref = _run(p, _scipy_binding, per_lane, dtype=dtype)
    got = _run(p, _spacc_binding, per_lane, dtype=dtype)
    assert got.shape == ref.shape
    dev = _rel(got, ref)
    assert dev <= 1e-13, f"K={K} per_lane={per_lane}: max rel dev {dev:.3e}"


@pytest.mark.parametrize("K", [1, 3, 70])
@pytest.mark.parametrize("per_lane", [False, True], ids=["scalar_ri", "lane_ri"])
@pytest.mark.parametrize("empty", ["dec_m", "int_m"], ids=["dec_empty", "int_empty"])
def test_spacc_apply_off_matches_scipy_with_empty_block(K, per_lane, empty):
    """An nnz = 0 off-diagonal is skipped, not called on a ``None`` handle."""
    p = _problem(K)
    p[empty] = _drop_off_diagonal(p[empty])
    ref = _run(p, _scipy_binding, per_lane)
    got = _run(p, _spacc_binding, per_lane)
    dev = _rel(got, ref)
    assert dev <= 1e-13, f"{empty} empty, K={K}: max rel dev {dev:.3e}"


def _one_step_calls(K):
    """The fake handles after one driver step at width ``K``."""
    p = _problem(K)
    op = compile_operator(p["int_m"], p["dec_m"], None)
    handles = _fakes(op)
    be = HostBackend(op, SpaccApplyOff(*handles))
    try:
        etd2_driver(1, p["dX"][:1], p["rho_inv"][:1], be, p["phi0"], [])
    finally:
        be.close()
    return p, handles


@pytest.mark.parametrize(("K", "widths"), [(3, [3]), (64, [64]), (70, [64, 6])])
def test_spacc_apply_off_tiles_at_64_columns(K, widths):
    """K > 64 is split into 64-column SpMM tiles; K <= 64 is one call.

    The driver applies the operator twice per step (state and predictor),
    so one pass of tiles per handle becomes two.
    """
    _, (int_fake, _) = _one_step_calls(K)
    assert {c[0] for c in int_fake.calls} == {"gemm"}
    assert [c[2] for c in int_fake.calls] == widths * 2


def test_spacc_apply_off_single_column_uses_gemv():
    """K = 1 needs no staging: row-major and column-major are the same bytes."""
    _, (int_fake, dec_fake) = _one_step_calls(1)
    assert {c[0] for c in int_fake.calls} == {"gemv"}
    assert {c[0] for c in dec_fake.calls} == {"gemv"}


def test_spacc_apply_off_scalar_ri_folds_into_alpha():
    """A scalar ri rides the dec call's alpha, not a second accumulator."""
    p, (int_fake, dec_fake) = _one_step_calls(4)
    assert {c[1] for c in int_fake.calls} == {1.0}
    assert {c[1] for c in dec_fake.calls} == {float(p["rho_inv"][0])}


# --- backend lifecycle -----------------------------------------------------
#
# ``close()`` is the only release point the solvers have: the driver's
# callers run it in a ``finally`` and ``MCEqRun`` runs it when its backend
# cache rotates. The bind-time scratch is the bulk of what a solve allocates
# -- ``step_buffers`` alone is six fp64 planes and a boolean one of the state
# shape -- so a ``close()`` that leaves it reachable pins the working set for
# as long as the backend object lives. Reachability is asserted with
# weakrefs, which is exact and platform-independent, not by measuring memory.


def _bound(p, make_apply_off, per_lane=False, dtype=np.float64, sec_ops=None):
    """A backend still bound after the driver ran over ``p``, and its
    solution: the state ``close()`` has to release."""
    op = compile_operator(p["int_m"], p["dec_m"], sec_ops)
    be = HostBackend(op, make_apply_off(op, dtype), dtype)
    dX, rho_inv = _path(p, per_lane)
    sol, _ = etd2_driver(p["nsteps"], dX, rho_inv, be, p["phi0"], [])
    return be, sol


def _coupled_problem(n_k=36, N=6, K=4, nsteps=6, seed=7):
    """A block-diagonal problem in the secant layout of ``_secant_ops``.

    Reaches the coupled corner, so ``_block`` is allocated, and runs per-lane
    paths, so the ``step_buffers`` planes are ``(dim, K)`` rather than
    ``(dim,)``. ``n_k`` is the smallest that operator set allows -- its
    ``T_P`` carries an off-``P`` column at mode 35 -- and ``N`` the smallest
    its ``low_e_idx`` allows.
    """
    # The operator set is the one the batched secant closure tests use; a
    # sibling test module, hence the import at the use site.
    from test_secant_multirhs import _secant_ops

    rng = np.random.default_rng(seed)

    def blocks(spread, loss):
        """A per-mode upper-triangular production block over a stable loss
        diagonal, block-diagonal over the mode axis."""
        return sp.block_diag(
            [
                sp.csr_matrix(
                    np.triu(rng.uniform(0.0, spread, (N, N)), 1)
                    - np.diag(rng.uniform(*loss, N))
                )
                for _ in range(n_k)
            ],
            format="csr",
        )

    dim = n_k * N
    return dict(
        nsteps=nsteps,
        dim=dim,
        sec_ops=_secant_ops(n_k),
        dX=np.full(nsteps, 0.1),
        rho_inv=np.linspace(1.1, 1.4, nsteps),
        int_m=blocks(0.012, (0.05, 0.13)),
        dec_m=blocks(0.006, (0.01, 0.035)),
        phi0=rng.uniform(0.1, 1.2, size=(dim, K)),
    )


def _host_scratch(be):
    """Every array ``HostBackend.bind`` allocated, flattened."""
    bufs = be._bufs
    return (
        [bufs[k] for k in ("hD", "eD", "phi1", "phi2")]
        + list(bufs["work"])
        + list(be._block or ())
        + list(be._factors or ())
        + ([] if be._work is None else [be._work])
    )


def _alive(refs):
    """How many of ``refs`` still have a target, after a collection."""
    gc.collect()
    return sum(r() is not None for r in refs)


def test_host_close_releases_bind_scratch():
    """``close()`` releases every buffer ``bind()`` allocated, not only the
    pointer memo."""
    p = _problem(3)
    be, sol = _bound(p, _scipy_binding)
    refs = [weakref.ref(a) for a in _host_scratch(be)]
    # Four factor planes and the three phi_work arrays, plus the numpy
    # lowering's scratch where the C extension is not built.
    assert len(refs) == 7 + (be._work is not None)
    del sol
    be.close()
    assert _alive(refs) == 0
    assert (be._bufs, be._block, be._factors, be._work, be._coupling) == (None,) * 5
    assert be._ptr_cache == {}


def test_host_rebind_peaks_at_one_generation():
    """A repeat ``bind()`` drops the previous scratch before allocating the
    next, so the peak is one generation of step buffers and not two.

    The buffers are reachable until the assignment either way, so only the
    allocation peak distinguishes the two orders -- hence tracemalloc rather
    than a weakref.
    """
    dim, K = 40000, 4
    m = sp.diags(-np.linspace(0.05, 0.1, dim), format="csr")
    op = compile_operator(m, m, None)
    be = HostBackend(op, _scipy_binding(op))
    # 6 fp64 planes and a bool, per element of dim x K.
    generation = 49 * dim * K
    tracemalloc.start()
    try:
        be.bind(dim, K, True, 3)
        be.bind(dim, K, True, 3)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        be.close()
    assert peak < 1.5 * generation, f"{peak / generation:.2f} generations live at once"


def test_host_close_is_idempotent_and_rebinds():
    """``close()`` is assignment only: it runs on a backend that was never
    bound and on one already closed, and a bind after it solves as before."""
    p = _problem(3)
    op = compile_operator(p["int_m"], p["dec_m"], None)
    HostBackend(op, _scipy_binding(op)).close()

    be, sol = _bound(p, _scipy_binding)
    sol = np.array(sol)
    be.close()
    be.close()
    dX, rho_inv = _path(p, False)
    try:
        again, _ = etd2_driver(p["nsteps"], dX, rho_inv, be, p["phi0"], [])
    finally:
        be.close()
    np.testing.assert_array_equal(again, sol)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
def test_host_close_releases_coupled_scratch(dtype):
    """The coupled per-lane route allocates the most: ``(dim, K)`` planes,
    the coupled-block factors and phi scratch, and at fp32 the landing
    buffers of the factor cast."""
    p = _coupled_problem()
    be, sol = _bound(p, _scipy_binding, True, dtype, p["sec_ops"])
    assert be._bufs["eD"].shape == (p["dim"], p["phi0"].shape[1])
    assert be._block is not None and be._coupling is not None
    assert (be._factors is None) is (dtype is np.float64)
    refs = [weakref.ref(a) for a in _host_scratch(be)]
    assert np.all(np.isfinite(sol))
    del sol
    be.close()
    assert _alive(refs) == 0
    assert (be._bufs, be._block, be._factors, be._work, be._coupling) == (None,) * 5


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_mkl_close_releases_bind_scratch():
    """The MKL binding's scratch and pointer memo go with ``close()`` too.

    The memo holds the driver's state buffers as well, so leaving it
    populated pins four ``(dim, K)`` planes the backend does not own.
    """
    p = _problem(3)
    op = compile_operator(p["int_m"], p["dec_m"], None)
    be = mkl_backend(op)
    dX, rho_inv = _path(p, False)
    sol, _ = etd2_driver(p["nsteps"], dX, rho_inv, be, p["phi0"], [])
    ao = be._apply_off
    assert len(ao._ptrs) == 4
    refs = [weakref.ref(ao._dec_buf)]
    del sol
    be.close()
    assert _alive(refs) == 0
    assert ao._dec_buf is None
    assert ao._ptrs == {}


@pytest.mark.parametrize("K", [1, 3], ids=["gemv", "staged"])
def test_spacc_close_releases_bind_scratch(K):
    """Both Accelerate paths release their staging buffers on ``close()`` --
    the Fortran-ordered trio and its tile pointers above K = 1, the single
    dec buffer and its pointer at it."""
    p = _problem(K)
    op = compile_operator(p["int_m"], p["dec_m"], None)
    ao = SpaccApplyOff(*_fakes(op))
    be = HostBackend(op, ao)
    dX, rho_inv = _path(p, False)
    sol, _ = etd2_driver(p["nsteps"], dX, rho_inv, be, p["phi0"], [])
    assert ao._staged is (K > 1)
    refs = [
        weakref.ref(a) for a in [ao._dec] + ([ao._x_f, ao._out_f] if ao._staged else [])
    ]
    del sol
    be.close()
    assert _alive(refs) == 0
    assert (ao._x_f, ao._out_f, ao._dec, ao._tiles) == (None,) * 4
    assert (ao._dec_p, ao._x_p, ao._out_p) == (None,) * 3
    assert ao._ptrs == {}
