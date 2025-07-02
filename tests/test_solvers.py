import numpy as np
import copy

from MCEq import config
from MCEq.solvers import solv_numpy

import pytest


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
    from MCEq.solvers import solv_CUDA_sparse, CUDASparseContext

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


@pytest.mark.skipif(not config.has_cuda, reason="CUDA not available")
def test_cuda_numpy_solver_consistency(toy_solver_setup):
    """
    Regression test: ensure CUDA and NumPy solvers produce consistent results.

    This validates that both solvers implement the same algorithm correctly.

    Note: Requires CuPy >= 12.0.0 for modern sparse matrix interface compatibility.
    """
    from MCEq.solvers import solv_CUDA_sparse, CUDASparseContext
    from MCEq import config

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
