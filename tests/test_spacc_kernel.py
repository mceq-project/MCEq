"""Real Accelerate SpMM: precision, dense order, and leading dimensions."""

import ctypes

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from MCEq import config


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("layout", [101, 102], ids=["rowmajor", "colmajor"])
@pytest.mark.parametrize("K", [3, 70])
def test_spacc_gemm_layout_and_strides(dtype, layout, K):
    """Rectangular operands with padding expose swapped dimensions or strides.

    Nonzero C pins accumulating semantics; untouched padding detects writes
    outside the output. Both precisions cross the old 64-column tile boundary.
    """
    from MCEq.solvers._kernels.spacc import SpaccMatrix

    rng = np.random.default_rng(42)
    A = csr_matrix(rng.uniform(-1, 1, (5, 9)).astype(dtype))
    B = rng.uniform(-1, 1, (9, K)).astype(dtype)
    C = rng.uniform(-1, 1, (5, K)).astype(dtype)
    alpha = 0.7
    if layout == 101:
        b_store = np.full((9, K + 3), 123, dtype=dtype)
        c_store = np.full((5, K + 3), 123, dtype=dtype)
        b_view, c_view = b_store[:, :K], c_store[:, :K]
        ldb = ldc = K + 3
        padding = c_store[:, K:]
    else:
        b_store = np.full((12, K), 123, dtype=dtype, order="F")
        c_store = np.full((8, K), 123, dtype=dtype, order="F")
        b_view, c_view = b_store[:9], c_store[:5]
        ldb, ldc = 12, 8
        padding = c_store[5:]
    b_view[:] = B
    c_view[:] = C
    ptr = ctypes.POINTER(ctypes.c_double if dtype is np.float64 else ctypes.c_float)
    handle = SpaccMatrix(A, dtype=dtype)
    try:
        handle.gemm_ctargs(
            alpha,
            K,
            b_view.ctypes.data_as(ptr),
            ldb,
            c_view.ctypes.data_as(ptr),
            ldc,
            layout=layout,
        )
    finally:
        handle.close()
    tol = 1e-12 if dtype is np.float64 else 1e-5
    np.testing.assert_allclose(c_view, C + alpha * (A @ B), rtol=tol, atol=tol)
    np.testing.assert_array_equal(padding, 123)
    np.testing.assert_array_equal(b_view, B)


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_spacc_construction_releases_input_arrays(dtype):
    """Committed handles own their values and keep no CSR/COO construction copy."""
    import gc
    import weakref

    from MCEq.solvers._kernels.spacc import SpaccMatrix

    matrix = csr_matrix([[1, 0, 3], [0, 2, 0], [4, 0, 5]], dtype=dtype)
    refs = [weakref.ref(a) for a in (matrix.data, matrix.indices, matrix.indptr)]
    handle = SpaccMatrix(matrix, dtype=dtype)
    matrix.data.fill(0)
    del matrix
    gc.collect()
    assert all(ref() is None for ref in refs)
    out = np.zeros(3, dtype=dtype)
    try:
        handle.gemv_npargs(1.0, np.ones(3, dtype=dtype), out)
    finally:
        handle.close()
    np.testing.assert_array_equal(out, [4, 2, 9])
