"""Intel MKL sparse BLAS binding of the ETD2 off-diagonal SpMM; ``libmkl_rt`` is loaded on the first handle."""

from ctypes import POINTER, Structure, byref, c_double, c_float, c_int, c_void_p

import numpy as np
import scipy.sparse as sp

from MCEq import config
from MCEq.solvers.backends.host import CtypesApplyOff, HostBackend, state_dtype


class _MklMatrixDescr(Structure):
    """``struct matrix_descr`` of the MKL sparse calls.

    Module level: ctypes checks Structure argtypes with isinstance().
    """

    _fields_ = [("type", c_int), ("mode", c_int), ("diag", c_int)]


_MKL_ARGTYPES_SET = False


def _set_mkl_argtypes(mkl):
    """Register ctypes argtypes/restype of the MKL sparse calls, once.

    Without them ctypes marshals scalars by value where MKL expects pointers.
    """
    global _MKL_ARGTYPES_SET
    if _MKL_ARGTYPES_SET:
        return
    fl64, fl32 = c_double, c_float

    # fp64
    mkl.mkl_sparse_d_mv.argtypes = [
        c_int,
        fl64,
        c_void_p,
        _MklMatrixDescr,
        POINTER(fl64),
        fl64,
        POINTER(fl64),
    ]
    mkl.mkl_sparse_d_mv.restype = c_int
    mkl.mkl_sparse_d_mm.argtypes = [
        c_int,
        fl64,
        c_void_p,
        _MklMatrixDescr,
        c_int,
        POINTER(fl64),
        c_int,
        c_int,
        fl64,
        POINTER(fl64),
        c_int,
    ]
    mkl.mkl_sparse_d_mm.restype = c_int
    # fp32
    if hasattr(mkl, "mkl_sparse_s_mv"):
        mkl.mkl_sparse_s_mv.argtypes = [
            c_int,
            fl32,
            c_void_p,
            _MklMatrixDescr,
            POINTER(fl32),
            fl32,
            POINTER(fl32),
        ]
        mkl.mkl_sparse_s_mv.restype = c_int
        mkl.mkl_sparse_s_mm.argtypes = [
            c_int,
            fl32,
            c_void_p,
            _MklMatrixDescr,
            c_int,
            POINTER(fl32),
            c_int,
            c_int,
            fl32,
            POINTER(fl32),
            c_int,
        ]
        mkl.mkl_sparse_s_mm.restype = c_int

    _MKL_ARGTYPES_SET = True


#: MKL Sparse BLAS entry-point family and ctypes scalar per state dtype.
_MKL_TYPES = {
    np.dtype(np.float64): ("d", c_double),
    np.dtype(np.float32): ("s", c_float),
}


class MklSparseMatrix:
    """Thin RAII wrapper around an Intel MKL sparse-matrix handle.

    Holds a CSR view of the off-diagonal block in ``dtype``, on the
    ``mkl_sparse_d_*`` or ``mkl_sparse_s_*`` entry points of
    :data:`_MKL_TYPES`. MKL keeps raw pointers into the backing arrays, so
    the wrapper holds references to them. ``mkl_sparse_set_mv_hint`` and
    ``mkl_sparse_optimize`` run once at construction.

    Args:
      csr (scipy.sparse.csr_matrix): CSR matrix with int32 indices; the
        data is cast to ``dtype`` here, once.
      expected_calls (int): SpMV count hint for MKL planning.
      dtype: float64 or float32 storage for the matrix data.
    """

    def __init__(self, csr, expected_calls=200, dtype=np.float64):
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrix expects a CSR matrix, got {type(csr).__name__}"
            )
        self.dtype = np.dtype(dtype)
        try:
            prec, fl_pr = _MKL_TYPES[self.dtype]
        except KeyError:
            raise TypeError(
                f"MklSparseMatrix expects float64 or float32 data, got {self.dtype}"
            ) from None
        self._prec = prec
        self._ct = fl_pr

        n_orig = csr.shape[0]
        self.n_cols = csr.shape[1]

        mkl = config.load_mkl()
        self._mkl = mkl
        _set_mkl_argtypes(mkl)
        self._mv = getattr(mkl, f"mkl_sparse_{prec}_mv")
        self._mm = getattr(mkl, f"mkl_sparse_{prec}_mm")

        # Real float32/float64 input only; the casts below would silently
        # change complex or integer data.
        if csr.dtype not in _MKL_TYPES:
            raise TypeError(
                f"MklSparseMatrix expects float64 or float32 data, got {csr.dtype}"
            )
        indices = csr.indices.astype(np.int32, copy=False)
        indptr = csr.indptr.astype(np.int32, copy=False)
        data = csr.data.astype(self.dtype, copy=False)
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self.nnz = csr.nnz

        handle = c_void_p()
        data_p = data.ctypes.data_as(POINTER(fl_pr))
        ci_p = indices.ctypes.data_as(POINTER(c_int))
        pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
        pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))

        st = getattr(mkl, f"mkl_sparse_{prec}_create_csr")(
            byref(handle),
            c_int(0),
            c_int(n_orig),
            c_int(self.n_cols),
            pb_p,
            pe_p,
            ci_p,
            data_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_{prec}_create_csr failed with status {st}")
        self._handle = handle

        descr = _MklMatrixDescr()
        descr.type = c_int(20)  # SPARSE_MATRIX_TYPE_GENERAL
        descr.mode = c_int(121)
        descr.diag = c_int(131)
        self._descr = descr
        self._operation = c_int(10)  # SPARSE_OPERATION_NON_TRANSPOSE

        st = mkl.mkl_sparse_set_mv_hint(
            handle, self._operation, descr, c_int(int(expected_calls))
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mv_hint failed with status {st}")
        st = mkl.mkl_sparse_optimize(handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def gemv_ctargs(self, alpha, x_p, beta, y_p):
        """``y = alpha * A * x + beta * y`` via raw pointers of the wrapper's dtype."""
        fl_pr = self._ct
        st = self._mv(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            x_p,
            fl_pr(beta),
            y_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_{self._prec}_mv failed with status {st}")

    def gemm_ctargs(self, alpha, nrhs, B_p, ldb, C_p, ldc, beta=1.0, layout=102):
        """``C = alpha * A * B + beta * C`` via raw pointers of the wrapper's dtype.

        ``layout`` is 102 (column-major) or 101 (row-major, the driver's
        C-contiguous buffers, ``ldb = ldc = K``).
        """
        fl_pr = self._ct
        st = self._mm(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            c_int(int(layout)),
            B_p,
            c_int(int(nrhs)),
            c_int(int(ldb)),
            fl_pr(beta),
            C_p,
            c_int(int(ldc)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_{self._prec}_mm failed with status {st}")

    def set_mm_hint(self, nrhs, expected_calls=200, layout=102):
        """Register the SpMM shape (``layout``, ``nrhs``) with MKL and re-optimise."""
        st = self._mkl.mkl_sparse_set_mm_hint(
            self._handle,
            self._operation,
            self._descr,
            c_int(int(layout)),
            c_int(int(nrhs)),
            c_int(int(expected_calls)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mm_hint failed with status {st}")
        st = self._mkl.mkl_sparse_optimize(self._handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def close(self):
        """Free the MKL handle (idempotent)."""
        handle = getattr(self, "_handle", None)
        mkl = getattr(self, "_mkl", None)
        if handle is None or mkl is None:
            return
        try:
            mkl.mkl_sparse_destroy(handle)
        except Exception:
            pass
        self._handle = None

    def __del__(self):
        # close() also covers a partially initialised instance.
        try:
            self.close()
        except Exception:
            pass


class MklApplyOff(CtypesApplyOff):
    """The apply-off binding on MKL: one row-major ``mkl_sparse_?_mm`` per
    product (``mkl_sparse_?_mv`` at K = 1); the handles' dtype is the state's."""

    name = "mkl"

    def bind(self, dim, K, nsteps):
        if K > 1:
            for m in self.handles:
                m.set_mm_hint(K, expected_calls=2 * nsteps, layout=101)
        super().bind(dim, K, nsteps)

    def _spmm(self, m, alpha, x, out, overwrite):
        beta = 0.0 if overwrite else 1.0
        if self.K == 1:
            m.gemv_ctargs(alpha, self._ptr(x), beta, self._ptr(out))
        else:
            K = self.K
            m.gemm_ctargs(
                alpha, K, self._ptr(x), K, self._ptr(out), K, beta=beta, layout=101
            )


def mkl_backend(op, dtype=np.float64, expected_calls=2000):
    """Host backend on MKL sparse BLAS; owns the handles it creates."""
    dtype = state_dtype(dtype)
    handles = tuple(
        MklSparseMatrix(off, expected_calls=expected_calls, dtype=dtype)
        if off.nnz
        else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, MklApplyOff(*handles, owns=True, dtype=dtype), dtype)
