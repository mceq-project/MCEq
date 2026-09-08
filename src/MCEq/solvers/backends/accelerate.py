"""Apple Accelerate Sparse BLAS binding of the ETD2 off-diagonal SpMM.

:class:`SpaccApplyOff` passes the driver's row-major ``(dim, K)`` buffers
directly to Accelerate. :func:`accelerate_backend` hands it to
:class:`MCEq.solvers.backends.host.HostBackend`. The handles themselves are
:class:`MCEq.solvers._kernels.spacc.SpaccMatrix`, imported only when needed
because their compiled extension is available on macOS only.
"""

from ctypes import POINTER, c_double, c_float

import numpy as np

from MCEq.solvers.backends.base import _state_dtype
from MCEq.solvers.backends.host import HostBackend

#: ctypes scalar per state dtype for the Accelerate pointer arguments.
_SPACC_CTYPES = {np.dtype(np.float64): c_double, np.dtype(np.float32): c_float}


class SpaccApplyOff:
    """``out = int_off x + ri dec_off x`` through Apple Accelerate Sparse BLAS.

    The driver's buffers go directly to SpMM with ``CblasRowMajor`` and
    ``ldb = ldc = K``. At K = 1 the same bytes go to SpMV. Accelerate
    accumulates without beta, so each output is zeroed before use. A scalar
    ``ri`` rides the decay product's alpha; a lane row scales a separate
    decay accumulator. ``owns`` closes the handles with the binding.
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
        self.K = K
        self._ptr_type = POINTER(_SPACC_CTYPES[self.dtype])
        # Drop pointers before replacing scratch: data_as pins its array.
        self._dec = self._dec_p = self._x_p = self._out_p = None
        self._ptrs = {}
        self._dec = np.empty((dim, K), dtype=self.dtype)
        self._dec_p = self._ptr(self._dec)

    def _ptr(self, arr):
        p = self._ptrs.get(id(arr))
        if p is None:
            p = self._ptrs[id(arr)] = arr.ctypes.data_as(self._ptr_type)
        return p

    def _spmm(self, m, alpha, target):
        if self.K == 1:
            m.gemv_ctargs(alpha, self._x_p, target)
        else:
            m.gemm_ctargs(alpha, self.K, self._x_p, self.K, target, self.K, layout=101)

    def __call__(self, x, out, ri):
        self._x_p, self._out_p = self._ptr(x), self._ptr(out)
        out.fill(0.0)
        if self.int_off is not None:
            self._spmm(self.int_off, 1.0, self._out_p)
        if self.dec_off is not None:
            if np.ndim(ri) == 0:
                self._spmm(self.dec_off, float(ri), self._out_p)
            else:
                self._dec.fill(0.0)
                self._spmm(self.dec_off, 1.0, self._dec_p)
                np.multiply(self._dec, ri, out=self._dec)
                np.add(out, self._dec, out=out)

    def close(self):
        self._dec = self._dec_p = self._x_p = self._out_p = None
        self._ptrs = {}
        if self.owns:
            for m in self.handles:
                m.close()


def accelerate_backend(op, fp_precision=64):
    """Host backend on Apple Accelerate Sparse BLAS; owns its sparse handles."""
    from MCEq.solvers._kernels.spacc import SpaccMatrix

    dtype = _state_dtype(fp_precision)
    handles = tuple(
        SpaccMatrix(off, dtype=dtype) if off.nnz else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, SpaccApplyOff(*handles, owns=True, dtype=dtype), dtype)
