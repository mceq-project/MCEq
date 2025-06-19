import numpy as np

from MCEq import config
from MCEq.solvers import solv_numpy

import pytest


def test_solv_numpy_runs(toy_solver_problem):

    phi0 = toy_solver_problem[-2].copy()
    grid_idcs = toy_solver_problem[-1]

    solution, grid_sol = solv_numpy(*toy_solver_problem)
    print(phi0, solution, grid_sol)
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

    solution_numpy, _ = solv_numpy(*toy_solver_problem)

    solution_mkl, _ = solv_MKL_sparse(*toy_solver_problem)

    assert solution_mkl == pytest.approx(solution_numpy, rel=1e-5, abs=1e-10)
