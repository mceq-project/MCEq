"""Intel MKL sparse BLAS binding of the ETD2 off-diagonal SpMM.

:class:`MklSparseMatrix` is the RAII wrapper around one
``mkl_sparse_{d,s}_*`` handle together with the ctypes argtype registration
its call sites need; :class:`MklApplyOff` is the ``apply_off`` binding built
on a pair of handles, and :func:`mkl_backend` hands that pair to
:class:`MCEq.solvers.backends.host.HostBackend`.

Everything that talks to ``libmkl`` belongs here. The library is loaded on
the first handle, never at import.
"""

from ctypes import POINTER, c_double, c_float

import numpy as np
import scipy.sparse as sp

from MCEq import config
from MCEq.solvers.backends.base import _state_dtype
from MCEq.solvers.backends.host import HostBackend


class _MklMatrixDescr:
    """Shared ``struct matrix_descr`` ctypes wrapper for MKL sparse calls.

    Has to be a module-level class so the argtypes-registered version and
    the per-wrapper instance reference the same Python type — ctypes does
    strict isinstance() checking on Structure argtypes.
    """


def _build_mkl_descr_class():
    from ctypes import Structure, c_int

    class _MatrixDescr(Structure):
        _fields_ = [("type", c_int), ("mode", c_int), ("diag", c_int)]

    return _MatrixDescr


_MklMatrixDescr = _build_mkl_descr_class()


_MKL_ARGTYPES_SET = False


def _set_mkl_argtypes(mkl):
    """Register ctypes argtypes/restype on the MKL functions we call.

    Without explicit argtypes, Python ctypes inspects the value passed at
    each call site to decide how to marshal it. That breaks when callers
    pass ``c_double.from_address(addr)`` where ``POINTER(c_double)`` is
    expected — ctypes marshals the c_double by value (not as a pointer)
    and MKL rejects the call with ``SPARSE_STATUS_INVALID_VALUE``.
    Setting argtypes once after the library load fixes the marshalling.
    """
    global _MKL_ARGTYPES_SET
    if _MKL_ARGTYPES_SET:
        return
    from ctypes import POINTER, c_int, c_void_p
    from ctypes import c_double as fl64
    from ctypes import c_float as fl32

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


#: MKL Sparse BLAS entry-point family and ctypes scalar per state dtype —
#: the whole of the fp64/fp32 difference in :class:`MklSparseMatrix`.
_MKL_TYPES = {
    np.dtype(np.float64): ("d", c_double),
    np.dtype(np.float32): ("s", c_float),
}


class MklSparseMatrix:
    """Thin RAII wrapper around an Intel MKL sparse-matrix handle.

    Holds a CSR view of the off-diagonal block in ``dtype``, on the
    ``mkl_sparse_d_*`` or ``mkl_sparse_s_*`` entry points of
    :data:`_MKL_TYPES`. MKL keeps raw pointers into the backing arrays, so
    the Python objects must outlive the handle — we keep references on the
    wrapper. ``mkl_sparse_set_mv_hint`` + ``mkl_sparse_optimize`` are
    called once at construction so the per-solve loop reuses the optimised
    layout.

    CSR only: BSR was a 1.5x win at fp64 on SIBYLL21, but the BSR block
    microkernels MKL ships are fp64-only on most builds.

    Args:
      csr (scipy.sparse.csr_matrix): CSR matrix with int32 indices; the
        data is cast to ``dtype`` here, once.
      expected_calls (int): SpMV count hint for MKL planning.
      dtype: float64 or float32 storage for the matrix data.
    """

    def __init__(self, csr, expected_calls=200, dtype=np.float64):
        from ctypes import byref, c_int, c_void_p

        config._load_mkl()
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
        self.n_orig = n_orig
        self.n_cols = csr.shape[1]

        mkl = config.mkl
        self._mkl = mkl
        _set_mkl_argtypes(mkl)
        self._mv = getattr(mkl, f"mkl_sparse_{prec}_mv")
        self._mm = getattr(mkl, f"mkl_sparse_{prec}_mm")

        # A complex or integer CSR would otherwise be cast silently, dropping
        # the imaginary part or the fractional one; only the two real float
        # widths convert to each other meaningfully here.
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
        self.n_padded = n_orig

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
        """``y = alpha * A * x + beta * y`` via raw pointers of the wrapper's
        dtype.

        Mirrors :meth:`MCEq.spacc.SpaccMatrix.gemv_ctargs` so the ETD2
        kernels can be written backend-agnostic up to the gemv binding.
        """
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
        """``C = alpha * A * B + beta * C`` via raw pointers of the wrapper's
        dtype.

        Wraps ``mkl_sparse_{d,s}_mm``. The default column-major layout
        (``layout=102``) serves (dim, K) Fortran-contiguous buffers
        without transpose; ``layout=101`` (row-major, ``ldb``/``ldc`` = K)
        serves the C-contiguous buffers of the driver. ``ldb`` and ``ldc``
        are the leading dimensions; per-tile callers offset the pointer
        instead.

        Default ``beta = 1.0`` matches :class:`MCEq.spacc.SpaccMatrix.gemm_ctargs`
        (accumulating SpMM). Caller is responsible for zeroing ``C`` before
        the first call in an accumulator chain.
        """
        from ctypes import c_int

        # SPARSE_LAYOUT_COLUMN_MAJOR = 102, SPARSE_LAYOUT_ROW_MAJOR = 101.
        # Operation enum (10 = non-transpose) comes from self._operation,
        # set in __init__.
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
        """Tell MKL the SpMM-specific shape so it can re-plan.

        ``mkl_sparse_set_mm_hint`` accepts the layout, op, descr, ncols, and
        expected call count; if the layout/ncols differs from the SpMV hint
        registered at construction, the optimiser can pick a different
        kernel. Followed by another ``mkl_sparse_optimize``. Optional —
        callers can skip if the SpMV hint is already adequate.
        """
        from ctypes import c_int

        # SPARSE_LAYOUT_COLUMN_MAJOR = 102, SPARSE_LAYOUT_ROW_MAJOR = 101.
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
        """Free the underlying MKL sparse handle.

        Idempotent — safe to call repeatedly. Prefer this over
        ``del`` or relying on refcount-driven ``__del__`` when the
        backend cache of ``MCEqRun`` rotates; the call below the C
        boundary returns the MKL-internal optimised layout memory, not
        just the Python wrapper.
        """
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
        # Defer to ``close()``; both are idempotent. Guards in ``close()``
        # cover partially-initialised instances (constructor raised before
        # the handle was created).
        try:
            self.close()
        except Exception:
            pass


class MklApplyOff:
    """``out = int_off x + ri dec_off x`` through MKL sparse BLAS.

    The driver's ``(dim, K)`` buffers are C-contiguous, so one row-major
    ``mkl_sparse_d_mm`` per stage covers all lanes (SpMV at K = 1). A
    scalar ``ri`` is fused into the dec SpMM's alpha; a ``(K,)`` lane row
    scales a separate accumulator. BSR handles pad the operator to a
    multiple of the block size; the operand is then staged through
    padded copies. ``owns`` closes the handles with the binding.
    ``dtype`` is the state precision and matches the handles'.
    """

    name = "mkl"

    def __init__(self, mkl_int_off, mkl_dec_off, owns=False, dtype=np.float64):
        self.int_off, self.dec_off = (
            m if m is not None and m.nnz else None for m in (mkl_int_off, mkl_dec_off)
        )
        self.handles = [m for m in (self.int_off, self.dec_off) if m is not None]
        self.owns = owns
        self.dtype = np.dtype(dtype)

    def bind(self, dim, K, nsteps):
        self.K = K
        self.n_padded = max([dim] + [m.n_padded for m in self.handles])
        self.dim = dim
        if K > 1:
            for m in self.handles:
                m.set_mm_hint(K, expected_calls=2 * nsteps, layout=101)
        # Dropped first (the memo pins its arrays), so two generations of
        # staging never coexist.
        self._dec_buf = self._x_pad = self._out_pad = None
        self._ptrs = {}
        self._ptr_type = POINTER(_MKL_TYPES[self.dtype][1])
        self._dec_buf = np.empty((self.n_padded, K), dtype=self.dtype)
        self._pad = self.n_padded != dim
        if self._pad:
            self._x_pad = np.zeros((self.n_padded, K), dtype=self.dtype)
            self._out_pad = np.empty((self.n_padded, K), dtype=self.dtype)

    def _ptr(self, arr):
        p = self._ptrs.get(id(arr))
        if p is None:
            p = self._ptrs[id(arr)] = arr.ctypes.data_as(self._ptr_type)
        return p

    def _spmm(self, m, alpha, x, beta, out):
        if self.K == 1:
            m.gemv_ctargs(alpha, self._ptr(x), beta, self._ptr(out))
        else:
            K = self.K
            m.gemm_ctargs(
                alpha, K, self._ptr(x), K, self._ptr(out), K, beta=beta, layout=101
            )

    def __call__(self, x, out, ri):
        if self._pad:
            self._x_pad[: self.dim] = x
            x, out_full = self._x_pad, out
            out = self._out_pad
        if self.int_off is None:
            out.fill(0.0)
        else:
            self._spmm(self.int_off, 1.0, x, 0.0, out)
        if self.dec_off is not None:
            if np.ndim(ri) == 0:
                self._spmm(self.dec_off, float(ri), x, 1.0, out)
            else:
                self._spmm(self.dec_off, 1.0, x, 0.0, self._dec_buf)
                np.multiply(self._dec_buf, ri, out=self._dec_buf)
                np.add(out, self._dec_buf, out=out)
        if self._pad:
            np.copyto(out_full, out[: self.dim])

    def close(self):
        self._dec_buf = self._x_pad = self._out_pad = None
        self._ptrs = {}
        if self.owns:
            for m in self.handles:
                m.close()


def mkl_backend(op, expected_calls=2000, fp_precision=64):
    """Host backend on MKL sparse BLAS; owns the handles it creates."""
    dtype = _state_dtype(fp_precision)
    handles = tuple(
        MklSparseMatrix(off, expected_calls=expected_calls, dtype=dtype)
        if off.nnz
        else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, MklApplyOff(*handles, owns=True, dtype=dtype), dtype)
