import copy

import numpy as np
import pytest

from MCEq import config
from MCEq.solvers import solv_numpy


def test_solv_numpy_runs(toy_solver_problem):
    phi0 = toy_solver_problem[-2].copy()
    grid_idcs = toy_solver_problem[-1]

    solution, grid_sol = solv_numpy(*toy_solver_problem)
    assert solution.shape == phi0.shape
    assert grid_sol.shape == (len(grid_idcs), phi0.shape[0])
    assert not np.isnan(solution).any()

    for i in range(1, grid_sol.shape[0]):
        assert np.all(grid_sol[i] <= grid_sol[i - 1])


@pytest.mark.skipif(not config.has_cuda, reason="CUDA not available")
def test_solv_CUDA_sparse_matches_numpy(toy_solver_problem):
    from MCEq.solvers import CUDASparseContext, solv_CUDA_sparse

    solution_numpy, _ = solv_numpy(*toy_solver_problem)

    int_m = toy_solver_problem[3]
    dec_m = toy_solver_problem[4]

    ctx = CUDASparseContext(int_m, dec_m, device_id=config.cuda_gpu_id)

    solution_cuda, _ = solv_CUDA_sparse(
        toy_solver_problem[0],  # nsteps
        toy_solver_problem[1],  # dX
        toy_solver_problem[2],  # rho_inv
        ctx,  # CUDASparseContext
        toy_solver_problem[5],  # phi
        toy_solver_problem[6],  # grid_idcs
    )
    assert solution_cuda == pytest.approx(solution_numpy, rel=1e-5, abs=1e-10)


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_solv_MKL_sparse_matches_numpy(toy_solver_problem):
    from MCEq.solvers import solv_MKL_sparse

    toy_solver_problem_mkl = tuple(copy.deepcopy(x) for x in toy_solver_problem)

    solution_numpy, _ = solv_numpy(*toy_solver_problem)

    solution_mkl, _ = solv_MKL_sparse(*toy_solver_problem_mkl)

    assert solution_mkl == pytest.approx(solution_numpy, rel=1e-5, abs=1e-10)


def test_solv_numpy_does_not_modify_input_phi(toy_solver_setup):
    """
    Regression test: ensure solv_numpy doesn't modify the input phi array.

    This was a bug where the NumPy solver modified the input array in place,
    causing subsequent solver calls to start with wrong initial conditions.
    """
    from MCEq.solvers import solv_numpy

    phi_original = toy_solver_setup[-2]
    phi_copy = phi_original.copy()
    # Run solver
    solution, _ = solv_numpy(*toy_solver_setup)

    # The input array should not be modified
    assert np.array_equal(phi_original, phi_copy), (
        "solv_numpy modified the input phi array - this breaks subsequent solver calls"
    )

    # The solution should be different from the input
    assert not np.array_equal(solution, phi_copy), (
        "Solver should produce a different result"
    )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_matrix_creation(toy_solver_problem):
    """SpaccMatrix should be created from a scipy sparse matrix without error."""
    import MCEq.spacc as spacc

    int_m = toy_solver_problem[3]
    sm = spacc.SpaccMatrix(int_m)
    assert sm.store_id is not None
    assert sm.store_id >= 0
    assert sm.dim_rows == int_m.shape[0]
    assert sm.dim_cols == int_m.shape[1]
    assert sm.nnz == int_m.nnz


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_gemv_matches_scipy(toy_solver_problem):
    """gemv_npargs should produce the same result as scipy sparse dot."""
    import MCEq.spacc as spacc

    int_m = toy_solver_problem[3]
    sm = spacc.SpaccMatrix(int_m)

    size = int_m.shape[0]
    x = np.ones(size)
    y = np.zeros(size)
    alpha = 2.0

    sm.gemv_npargs(alpha, x, y)

    expected = alpha * int_m.dot(x)
    assert np.allclose(y, expected, rtol=1e-12), (
        f"gemv result {y} does not match scipy result {expected}"
    )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_double_del_is_safe(toy_solver_problem):
    """Calling __del__ twice on a SpaccMatrix must not crash (double-free guard)."""
    import MCEq.spacc as spacc

    int_m = toy_solver_problem[3]
    sm = spacc.SpaccMatrix(int_m)
    sm.__del__()
    # After __del__, store_id should be set to None to prevent double-free
    assert sm.store_id is None, "store_id should be None after __del__"
    # Second call must not crash
    sm.__del__()


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_del_with_none_store_id():
    """SpaccMatrix.__del__ with store_id=None must not crash (failed-init guard)."""
    from scipy.sparse import eye

    import MCEq.spacc as spacc

    sm = spacc.SpaccMatrix(eye(3, format="coo"))
    sm.store_id = None  # Simulate a failed __init__
    sm.__del__()  # Must not raise or crash


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_solver_matches_numpy(toy_solver_problem):
    """solv_spacc_sparse should produce the same result as solv_numpy."""
    import MCEq.spacc as spacc
    from MCEq.solvers import solv_spacc_sparse

    nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs = toy_solver_problem

    solution_numpy, grid_numpy = solv_numpy(*toy_solver_problem)

    spacc_int_m = spacc.SpaccMatrix(int_m)
    spacc_dec_m = spacc.SpaccMatrix(dec_m)

    solution_spacc, grid_spacc = solv_spacc_sparse(
        nsteps, dX, rho_inv, spacc_int_m, spacc_dec_m, phi.copy(), grid_idcs
    )

    assert solution_spacc == pytest.approx(solution_numpy, rel=1e-12, abs=1e-15), (
        "spacc solver result does not match numpy solver"
    )
    assert np.allclose(grid_spacc, grid_numpy, rtol=1e-12), (
        "spacc solver grid solutions do not match numpy solver"
    )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_matrix_store_full():
    """Filling SIZE_MSTORE (10) slots and then freeing them leaves store clean."""
    from scipy.sparse import eye

    import MCEq.spacc as spacc

    # Clear any leftover matrices from previous tests
    spacc.spacc.free_mstore()

    matrices = []
    # SIZE_MSTORE is 10; fill all slots
    for _ in range(10):
        matrices.append(spacc.SpaccMatrix(eye(3, format="coo")))

    # Free explicitly; after this all slots must be available again
    for m in matrices:
        m.__del__()

    # A fresh matrix should now succeed (store is not full anymore)
    extra = spacc.SpaccMatrix(eye(3, format="coo"))
    assert extra.store_id is not None and extra.store_id >= 0
    extra.__del__()


@pytest.mark.skipif(not config.has_cuda, reason="CUDA not available")
def test_cuda_numpy_solver_consistency(toy_solver_setup):
    """
    Regression test: ensure CUDA and NumPy solvers produce consistent results.

    This validates that both solvers implement the same algorithm correctly.

    Note: Requires CuPy >= 12.0.0 for modern sparse matrix interface compatibility.
    """
    from MCEq import config
    from MCEq.solvers import CUDASparseContext, solv_CUDA_sparse

    # Run NumPy solver
    solution_numpy, _ = solv_numpy(*toy_solver_setup)

    # Run CUDA solver
    nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs = toy_solver_setup

    ctx = CUDASparseContext(int_m, dec_m, device_id=config.cuda_gpu_id)
    solution_cuda, _ = solv_CUDA_sparse(nsteps, dX, rho_inv, ctx, phi, grid_idcs)

    # Results should match within floating-point precision
    # (CUDA uses float32, NumPy uses float64, so we expect small differences)
    assert np.allclose(
        solution_cuda,
        solution_numpy,
        rtol=1e-5,
        atol=1e-8,
    ), "CUDA and NumPy solvers produce different results"
