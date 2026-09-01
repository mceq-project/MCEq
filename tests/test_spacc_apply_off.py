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
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq.operator_assembly import compile_operator
from MCEq.solvers import HostBackend, ScipyApplyOff, SpaccApplyOff, etd2_driver


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
