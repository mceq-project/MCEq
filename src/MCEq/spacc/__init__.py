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
spacc.test.restype = c_int

spacc.free_mstore.restype = None

spacc.free_mstore_at.restype = None
spacc.free_mstore_at.argtypes = [c_int]

spacc.create_sparse_matrix.restype = c_int
spacc.create_sparse_matrix.argtypes = [
    c_int,
    c_int,
    c_int,
    c_int,
    POINTER(c_longlong),
    POINTER(c_longlong),
    POINTER(c_double),
]

spacc.gemv.restype = c_int
spacc.gemv.argtypes = [
    c_double,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
]

# gemm: C := alpha * A * B + C with B, C dense (n_rows_A, nrhs) column-major.
# Wraps Apple Accelerate's sparse_matrix_product_dense_double; used by the
# multi-RHS spacc kernel (solv_spacc_etd2_multirhs). See spacc.c.
spacc.gemm.restype = c_int
spacc.gemm.argtypes = [
    c_double,
    c_int,
    c_int,
    POINTER(c_double),
    c_int,
    POINTER(c_double),
    c_int,
]

spacc.daxpy.restype = None
spacc.daxpy.argtypes = [
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
]

daxpy = spacc.daxpy

# ETD2 fused post-apply kernels — replace the 4-ufunc chains in the
# multi-RHS spacc solver with a single fused (dim, K) pass. Same idea as
# PriNCe's post_apply1/2 ElementwiseKernels on cupy. All matrices are
# column-major (Fortran-contiguous); eD/phi1/phi2 are either (dim,)
# (multirhs) or (dim, K) (multipath). See spacc.c.
spacc.etd2_post_apply1_multirhs.restype = None
spacc.etd2_post_apply1_multirhs.argtypes = [
    c_int,
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
spacc.etd2_post_apply2_multirhs.restype = None
spacc.etd2_post_apply2_multirhs.argtypes = [
    c_int,
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
spacc.etd2_post_apply1_multipath.restype = None
spacc.etd2_post_apply1_multipath.argtypes = [
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
spacc.etd2_post_apply2_multipath.restype = None
spacc.etd2_post_apply2_multipath.argtypes = [
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]

etd2_post_apply1_multirhs = spacc.etd2_post_apply1_multirhs
etd2_post_apply2_multirhs = spacc.etd2_post_apply2_multirhs
etd2_post_apply1_multipath = spacc.etd2_post_apply1_multipath
etd2_post_apply2_multipath = spacc.etd2_post_apply2_multipath

# ---- fp32 bindings ----
spacc.gemv_f32.restype = c_int
spacc.gemv_f32.argtypes = [c_float, c_int, POINTER(c_float), POINTER(c_float)]
spacc.gemm_f32.restype = c_int
spacc.gemm_f32.argtypes = [
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
    c_int,
    POINTER(c_longlong),
    POINTER(c_longlong),
    POINTER(c_float),
]
spacc.etd2_post_apply1_multirhs_f32.restype = None
spacc.etd2_post_apply1_multirhs_f32.argtypes = [
    c_int,
    c_int,
    c_float,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
spacc.etd2_post_apply2_multirhs_f32.restype = None
spacc.etd2_post_apply2_multirhs_f32.argtypes = [
    c_int,
    c_int,
    c_float,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
spacc.etd2_post_apply1_multipath_f32.restype = None
spacc.etd2_post_apply1_multipath_f32.argtypes = [
    c_int,
    c_int,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
spacc.etd2_post_apply2_multipath_f32.restype = None
spacc.etd2_post_apply2_multipath_f32.argtypes = [
    c_int,
    c_int,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]

etd2_post_apply1_multirhs_f32 = spacc.etd2_post_apply1_multirhs_f32
etd2_post_apply2_multirhs_f32 = spacc.etd2_post_apply2_multirhs_f32
etd2_post_apply1_multipath_f32 = spacc.etd2_post_apply1_multipath_f32
etd2_post_apply2_multipath_f32 = spacc.etd2_post_apply2_multipath_f32

# Initialize
spacc.free_mstore()


class SpaccMatrix(object):
    def __init__(self, scipy_sparse_matrix):
        spm = scipy_sparse_matrix.tocoo()
        self.dim_rows, self.dim_cols = spm.shape
        self.nnz = spm.nnz
        self.col = spm.col.astype("longlong")
        self.row = spm.row.astype("longlong")
        self.data = spm.data.astype("double")
        self.store_id = None
        self._create_matrix()

    def close(self):
        """Free the underlying Accelerate sparse-matrix slot.

        Idempotent — safe to call more than once. Prefer this over
        ``del`` or letting refcount drive ``__del__`` when you need a
        deterministic release of the slot (e.g. when juggling caches
        in ``MCEqRun._build_kernel_dispatch``); the slot pool
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

    def _create_matrix(self):
        self.store_id = spacc.create_sparse_matrix(
            -1,
            self.dim_rows,
            self.dim_cols,
            self.nnz,
            self.row.ctypes.data_as(POINTER(c_longlong)),
            self.col.ctypes.data_as(POINTER(c_longlong)),
            self.data.ctypes.data_as(POINTER(c_double)),
        )
        if self.store_id < 0:
            raise Exception("Matrix creation failed.")

    def gemv_ctargs(self, alpha, cx, cy):
        """General Matrix-Vector multiplication, expects arguments
        in correct ctypes.

        Performs y = alpha*M*x + y.
        No dimensional checks are performed.
        """
        if spacc.gemv(alpha, self.store_id, cx, cy) != 0:
            raise Exception("Sparse matrix-vector multiplication failed.")

    def gemv_npargs(self, alpha, x, y):
        """General Matrix-Vector multiplication, expects numpy arrays as arguments.

        Performs y = alpha*M*x + y.
        """
        assert x.shape[0] == self.dim_cols
        assert y.shape[0] == self.dim_cols
        assert x.dtype == "float64"
        assert y.dtype == "float64"
        alpha = float(alpha)

        cy = y.ctypes.data_as(POINTER(c_double))
        cx = x.ctypes.data_as(POINTER(c_double))
        if spacc.gemv(alpha, self.store_id, cx, cy) != 0:
            raise Exception("Sparse matrix-vector multiplication failed.")

    def gemm_ctargs(self, alpha, nrhs, cB, ldb, cC, ldc):
        """Sparse-dense SpMM with raw ctypes pointers.

        Performs ``C := alpha * M * B + C`` where ``B`` and ``C`` are dense
        ``(dim_cols, nrhs)`` / ``(dim_rows, nrhs)`` matrices in column-major
        layout with leading dimensions ``ldb`` / ``ldc``. Accumulating —
        zero ``C`` before the first call if you want a non-accumulating
        result.

        No dimensional checks are performed. Caller is responsible for
        ensuring ``B`` and ``C`` are float64 column-major arrays
        (Fortran-contiguous; ``np.asfortranarray`` is the canonical way).
        """
        if spacc.gemm(alpha, self.store_id, nrhs, cB, ldb, cC, ldc) != 0:
            raise Exception("Sparse matrix-matrix multiplication failed.")


class SpaccMatrixF32(object):
    """Float32 sibling of :class:`SpaccMatrix`.

    Wraps Apple Accelerate's ``sparse_matrix_create_float`` /
    ``sparse_matrix_vector_product_dense_float`` /
    ``sparse_matrix_product_dense_float`` family. The mstore slot is
    typed at creation, so f64 and f32 matrices live in different slots.
    Use the f32 variant when the multi-RHS solver runs in fp32 — halves
    memory bandwidth on the (dim, K) state buffers, and on Accelerate
    benches at ~1.5-2× SpMM throughput vs fp64.
    """

    def __init__(self, scipy_sparse_matrix):
        spm = scipy_sparse_matrix.tocoo()
        self.dim_rows, self.dim_cols = spm.shape
        self.nnz = spm.nnz
        self.col = spm.col.astype("longlong")
        self.row = spm.row.astype("longlong")
        self.data = spm.data.astype("float32")
        self.store_id = None
        self._create_matrix()

    def close(self):
        if self.store_id is not None and spacc is not None:
            spacc.free_mstore_at(self.store_id)
            self.store_id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _create_matrix(self):
        self.store_id = spacc.create_sparse_matrix_f32(
            -1,
            self.dim_rows,
            self.dim_cols,
            self.nnz,
            self.row.ctypes.data_as(POINTER(c_longlong)),
            self.col.ctypes.data_as(POINTER(c_longlong)),
            self.data.ctypes.data_as(POINTER(c_float)),
        )
        if self.store_id < 0:
            raise Exception("F32 sparse matrix creation failed.")

    def gemv_ctargs(self, alpha, cx, cy):
        """Single-precision SpMV: y += alpha * M * x."""
        if spacc.gemv_f32(alpha, self.store_id, cx, cy) != 0:
            raise Exception("F32 sparse matrix-vector multiplication failed.")

    def gemm_ctargs(self, alpha, nrhs, cB, ldb, cC, ldc):
        """Single-precision SpMM: C += alpha * M * B (column-major B, C)."""
        if spacc.gemm_f32(alpha, self.store_id, nrhs, cB, ldb, cC, ldc) != 0:
            raise Exception("F32 sparse matrix-matrix multiplication failed.")
