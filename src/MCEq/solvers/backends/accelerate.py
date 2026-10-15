"""Apple Accelerate Sparse BLAS binding of the ETD2 off-diagonal SpMM.

The handles are :class:`MCEq.solvers._kernels.spacc.SpaccMatrix`, imported
only when needed because their compiled extension exists on macOS only.
"""

import numpy as np

from MCEq.solvers.backends.host import CtypesApplyOff, HostBackend, state_dtype


class SpaccApplyOff(CtypesApplyOff):
    """The apply-off binding on Accelerate: row-major SpMM with
    ``ldb = ldc = K`` (SpMV at K = 1). Accelerate accumulates without beta,
    so an overwritten output is zeroed first."""

    name = "accelerate"

    def _spmm(self, m, alpha, x, out, overwrite):
        if overwrite:
            out.fill(0.0)
        x_p, out_p = self._ptr(x), self._ptr(out)
        if self.K == 1:
            m.gemv_ctargs(alpha, x_p, out_p)
        else:
            m.gemm_ctargs(alpha, self.K, x_p, self.K, out_p, self.K, layout=101)


def accelerate_backend(op, dtype=np.float64):
    """Host backend on Apple Accelerate Sparse BLAS; owns its sparse handles."""
    from MCEq.solvers._kernels.spacc import SpaccMatrix

    dtype = state_dtype(dtype)
    handles = tuple(
        SpaccMatrix(off, dtype=dtype) if off.nnz else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, SpaccApplyOff(*handles, owns=True, dtype=dtype), dtype)
