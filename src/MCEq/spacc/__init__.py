import atexit
import os
import sysconfig
from ctypes import (
    POINTER,
    c_double,
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

spacc.daxpy.restype = None
spacc.daxpy.argtypes = [
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
]

daxpy = spacc.daxpy

# Initialize
spacc.free_mstore()
# On module unload or reload, free the pointers
atexit.register(spacc.free_mstore)


class SpaccMatrix:
    def __init__(self, scipy_sparse_matrix):
        spm = scipy_sparse_matrix.tocoo()
        self.dim_rows, self.dim_cols = spm.shape
        self.nnz = spm.nnz
        self.col = spm.col.astype("longlong")
        self.row = spm.row.astype("longlong")
        self.data = spm.data.astype("double")
        self.store_id = None
        self._create_matrix()

    def __del__(self):
        spacc.free_mstore_at(self.store_id)

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
