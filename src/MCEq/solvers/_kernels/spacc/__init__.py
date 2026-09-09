import os
import sysconfig
from ctypes import (
    POINTER,
    c_double,
    c_float,
    c_int,
    c_longlong,
    cdll,
)

import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
suffix = sysconfig.get_config_var("EXT_SUFFIX")
# Some Python 2.7 versions don't define EXT_SUFFIX
if suffix is None and "SO" in sysconfig.get_config_vars():
    suffix = sysconfig.get_config_var("SO")

assert suffix is not None, "Shared lib suffix was not identified."

for fn in os.listdir(base):
    if "libspacc" in fn and fn.endswith(suffix):
        spacc = cdll.LoadLibrary(os.path.join(base, fn))
        break

# Declaration of function argument types
spacc.free_mstore.restype = None

spacc.free_mstore_at.restype = None
spacc.free_mstore_at.argtypes = [c_int]

spacc.create_sparse_matrix.restype = c_int
spacc.create_sparse_matrix.argtypes = [
    c_int,
    c_int,
    c_int,
    POINTER(c_longlong),
    POINTER(c_int),
    POINTER(c_double),
]

spacc.gemv.restype = c_int
spacc.gemv.argtypes = [
    c_double,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
]

# gemm: C := alpha * A * B + C with caller-selected dense layout.
# SpaccApplyOff passes row-major driver buffers directly. See spacc.c.
spacc.gemm.restype = c_int
spacc.gemm.argtypes = [
    c_int,
    c_double,
    c_int,
    c_int,
    POINTER(c_double),
    c_int,
    POINTER(c_double),
    c_int,
]

# ---- fp32 bindings ----
spacc.gemv_f32.restype = c_int
spacc.gemv_f32.argtypes = [c_float, c_int, POINTER(c_float), POINTER(c_float)]
spacc.gemm_f32.restype = c_int
spacc.gemm_f32.argtypes = [
    c_int,
    c_float,
    c_int,
    c_int,
    POINTER(c_float),
    c_int,
    POINTER(c_float),
    c_int,
]
spacc.create_sparse_matrix_f32.restype = c_int
spacc.create_sparse_matrix_f32.argtypes = [
    c_int,
    c_int,
    c_int,
    POINTER(c_longlong),
    POINTER(c_int),
    POINTER(c_float),
]

# Initialize
spacc.free_mstore()


#: Accelerate sparse entry-point family and ctypes scalar per state dtype —
#: the whole of the fp64/fp32 difference in :class:`SpaccMatrix`. The mstore
#: slot is typed at creation, so f64 and f32 matrices live in different slots.
_SPACC_TYPES = {
    np.dtype(np.float64): ("create_sparse_matrix", "gemv", "gemm", c_double),
    np.dtype(np.float32): (
        "create_sparse_matrix_f32",
        "gemv_f32",
        "gemm_f32",
        c_float,
    ),
}


class SpaccMatrix:
    """Apple Accelerate sparse-matrix handle over a scipy sparse matrix.

    ``dtype`` is the storage precision, float64 or float32, and selects the
    ``sparse_matrix_*_double`` or ``sparse_matrix_*_float`` entry-point
    family of :data:`_SPACC_TYPES`. fp32 halves the memory bandwidth on the
    ``(dim, K)`` state buffers and benches at ~1.5-2x the fp64 SpMM
    throughput.

    Args:
      scipy_sparse_matrix: any scipy sparse matrix; the data is cast to
        ``dtype`` here, once.
      dtype: float64 or float32 storage for the matrix data.
    """

    def __init__(self, scipy_sparse_matrix, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        try:
            create, gemv, gemm, fl_pr = _SPACC_TYPES[self.dtype]
        except KeyError:
            raise TypeError(
                f"SpaccMatrix expects float64 or float32 data, got {self.dtype}"
            ) from None
        self._ct = fl_pr
        self._create = getattr(spacc, create)
        self._gemv = getattr(spacc, gemv)
        self._gemm = getattr(spacc, gemm)

        self.store_id = None
        csr = scipy_sparse_matrix.tocsr()
        if not csr.has_canonical_format:
            csr = csr.copy()
            csr.sum_duplicates()
        self.dim_rows, self.dim_cols = csr.shape
        if max(csr.shape) > np.iinfo(np.int32).max:
            raise ValueError("Accelerate matrix dimensions must fit in int32")
        self.nnz = csr.nnz
        self._create_matrix(csr)

    def close(self):
        """Free the underlying Accelerate sparse-matrix slot.

        Idempotent — safe to call more than once. Prefer this over
        ``del`` or letting refcount drive ``__del__`` when you need a
        deterministic release of the slot (e.g. when juggling caches
        in ``MCEqRun._etd2_backend``); the slot pool
        (``SIZE_MSTORE``) is fixed-size, so prompt release matters.
        """
        if self.store_id is not None and spacc is not None:
            spacc.free_mstore_at(self.store_id)
            self.store_id = None

    def __del__(self):
        # Defer to ``close()``; both are idempotent.
        try:
            self.close()
        except Exception:
            pass

    def _create_matrix(self, csr):
        # Construction buffers live only until sparse_commit returns. The
        # native matrix owns its entries; retaining COO copies here adds
        # 24 bytes per nonzero at fp64, on top of the compiled CSR operator.
        indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
        data = np.ascontiguousarray(csr.data, dtype=self.dtype)
        store_id = self._create(
            -1,
            self.dim_rows,
            self.dim_cols,
            indptr.ctypes.data_as(POINTER(c_longlong)),
            indices.ctypes.data_as(POINTER(c_int)),
            data.ctypes.data_as(POINTER(self._ct)),
        )
        if store_id < 0:
            raise Exception("Matrix creation failed.")
        # A failed construction never owns a slot (in particular, not -1).
        self.store_id = store_id

    def gemv_ctargs(self, alpha, cx, cy):
        """General Matrix-Vector multiplication, expects arguments
        in correct ctypes.

        Performs y = alpha*M*x + y.
        No dimensional checks are performed.
        """
        if self._gemv(alpha, self.store_id, cx, cy) != 0:
            raise Exception("Sparse matrix-vector multiplication failed.")

    def gemv_npargs(self, alpha, x, y):
        """General Matrix-Vector multiplication, expects numpy arrays as arguments.

        Performs y = alpha*M*x + y.
        """
        assert x.shape[0] == self.dim_cols
        assert y.shape[0] == self.dim_cols
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
        alpha = float(alpha)

        cy = y.ctypes.data_as(POINTER(self._ct))
        cx = x.ctypes.data_as(POINTER(self._ct))
        if self._gemv(alpha, self.store_id, cx, cy) != 0:
            raise Exception("Sparse matrix-vector multiplication failed.")

    def gemm_ctargs(self, alpha, nrhs, cB, ldb, cC, ldc, layout=102):
        """``C := alpha * M * B + C`` through raw ctypes pointers.

        ``layout=102`` selects column-major buffers; ``layout=101`` selects
        row-major buffers. ``ldb`` and ``ldc`` are the corresponding column
        or row strides in elements. B has shape ``(dim_cols, nrhs)`` and C
        has shape ``(dim_rows, nrhs)``. Both must match the handle's dtype.
        Zero C before the first call for a non-accumulating result.
        """
        if self._gemm(layout, alpha, self.store_id, nrhs, cB, ldb, cC, ldc) != 0:
            raise Exception("Sparse matrix-matrix multiplication failed.")
