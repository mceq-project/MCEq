"""Apple Accelerate Sparse BLAS binding of the ETD2 off-diagonal SpMM.

:class:`SpaccApplyOff` bridges the driver's row-major ``(dim, K)`` buffers
to Accelerate's column-major accumulating SpMM through Fortran-ordered
scratch and a fixed column tile; :func:`accelerate_backend` hands it to
:class:`MCEq.solvers.backends.host.HostBackend`. The handles themselves are
:class:`MCEq.spacc.SpaccMatrix`.

Everything Accelerate-shaped belongs here -- the layout staging, the tile
size, the ctypes scalars. ``MCEq.spacc`` loads ``libspacc`` at import and
only macOS builds carry one, so it is imported inside the constructor.
"""

from ctypes import POINTER, c_double, c_float

import numpy as np

from MCEq.solvers.backends.base import _state_dtype
from MCEq.solvers.backends.host import HostBackend

# K-tile size of the Accelerate Sparse BLAS SpMM in :class:`SpaccApplyOff`.
# The K-to-1000 bench (runs/2026-05-21_multi-rhs-etd2-prototype) shows
# ``sparse_matrix_product_dense_double`` peaks at K ≈ 32–64 on the M3 Pro
# (3.0–3.2× /RHS) then drops to ≈ 1.4× at K ≥ 128 — Accelerate's internal
# SpMM tiling stops being cache-friendly past ~64 columns. Splitting larger
# K requests into 64-column tiles restores the peak operating point at all K.
_SPACC_SPMM_TILE = 64


#: ctypes scalar per state dtype for the Accelerate pointer arguments —
#: the whole of the fp64/fp32 difference in :class:`SpaccApplyOff`. The
#: entry-point families themselves are picked by
#: :data:`MCEq.spacc._SPACC_TYPES` when the handle is built.
_SPACC_CTYPES = {np.dtype(np.float64): c_double, np.dtype(np.float32): c_float}


class SpaccApplyOff:
    """``out = int_off x + ri dec_off x`` through Apple Accelerate Sparse BLAS.

    Accelerate's SpMM is column-major and accumulating (``C += alpha M B``,
    no beta) while the driver's ``(dim, K)`` buffers are row-major, so the
    operand and the result are staged through Fortran-ordered scratch
    allocated once in :meth:`bind`. At K = 1 the two layouts are the same
    bytes and the driver's own pointers go straight to the SpMV. The SpMM
    runs one call per :data:`_SPACC_SPMM_TILE` columns: a column block of a
    Fortran-ordered buffer is contiguous, so a tile is that column's
    pointer with ``ldb = ldc = dim``. A scalar ``ri`` is fused into the dec
    SpMM's alpha; a ``(K,)`` lane row scales a separate accumulator.
    ``owns`` closes the handles with the binding. ``dtype`` is the state
    precision and matches the handles'.
    """

    name = "accelerate"

    def __init__(self, spacc_int_off, spacc_dec_off, owns=False, dtype=np.float64):
        self.int_off, self.dec_off = (
            m if m is not None and m.nnz else None
            for m in (spacc_int_off, spacc_dec_off)
        )
        self.handles = [m for m in (self.int_off, self.dec_off) if m is not None]
        self.owns = owns
        self.dtype = np.dtype(dtype)

    def bind(self, dim, K, nsteps):
        self.dim = dim
        self.K = K
        self._ptr_type = POINTER(_SPACC_CTYPES[self.dtype])
        self._ptrs = {}
        self._staged = K > 1
        if self._staged:
            self._x_f, self._out_f, self._dec = (
                np.empty((dim, K), dtype=self.dtype, order="F") for _ in range(3)
            )
            tile = max(1, min(_SPACC_SPMM_TILE, K))

            def at(buf, c0):
                """Pointer to column ``c0`` of a Fortran-ordered buffer; the
                returned ctypes object keeps the view alive."""
                return buf[:, c0:].ctypes.data_as(self._ptr_type)

            self._tiles = [
                (
                    min(tile, K - c0),
                    at(self._x_f, c0),
                    at(self._out_f, c0),
                    at(self._dec, c0),
                )
                for c0 in range(0, K, tile)
            ]
        else:
            self._dec = np.empty((dim, 1), dtype=self.dtype)
            self._dec_p = self._ptr(self._dec)

    def _ptr(self, arr):
        p = self._ptrs.get(id(arr))
        if p is None:
            p = self._ptrs[id(arr)] = arr.ctypes.data_as(self._ptr_type)
        return p

    def _spmm(self, m, alpha, into_dec):
        """``target += alpha m operand`` on the staged operand, where the
        target is the result accumulator or, with ``into_dec``, the dec one.

        One SpMV at K = 1, one accumulating SpMM per column tile above it."""
        if self._staged:
            dim = self.dim
            for nrhs, x_p, out_p, dec_p in self._tiles:
                m.gemm_ctargs(alpha, nrhs, x_p, dim, dec_p if into_dec else out_p, dim)
        else:
            m.gemv_ctargs(alpha, self._x_p, self._dec_p if into_dec else self._out_p)

    def __call__(self, x, out, ri):
        if self._staged:
            np.copyto(self._x_f, x)  # row-major state -> column-major operand
            acc = self._out_f
        else:
            self._x_p, self._out_p = self._ptr(x), self._ptr(out)
            acc = out
        # Accelerate accumulates and takes no beta, so each chain zeroes its
        # accumulator instead of passing beta = 0 on the first call.
        acc.fill(0.0)
        if self.int_off is not None:
            self._spmm(self.int_off, 1.0, False)
        if self.dec_off is not None:
            if np.ndim(ri) == 0:
                self._spmm(self.dec_off, float(ri), False)
            else:
                self._dec.fill(0.0)
                self._spmm(self.dec_off, 1.0, True)
                np.multiply(self._dec, ri, out=self._dec)
                np.add(acc, self._dec, out=acc)
        if self._staged:
            np.copyto(out, self._out_f)

    def close(self):
        if self.owns:
            for m in self.handles:
                m.close()


def accelerate_backend(op, fp_precision=64):
    """Host backend on Apple Accelerate Sparse BLAS; owns the handles it
    creates.

    The import is local because ``MCEq.spacc`` loads ``libspacc`` at import
    and only macOS builds carry one.
    """
    from MCEq.spacc import SpaccMatrix

    dtype = _state_dtype(fp_precision)
    handles = tuple(
        SpaccMatrix(off, dtype=dtype) if off.nnz else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, SpaccApplyOff(*handles, owns=True, dtype=dtype), dtype)
