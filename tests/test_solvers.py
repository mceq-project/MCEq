import numpy as np
import pytest

from MCEq import config


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


# ---------------------------------------------------------------------------
# ETD2 (numpy_etd2) tests
# ---------------------------------------------------------------------------
def test_solv_numpy_etd2_runs(toy_solver_problem):
    """ETD2 returns the right shape, no NaN, monotonic decay on the grid.

    The toy fixture has only diagonal int_m / dec_m, so ETD2 collapses to
    phi <- exp(h*D) * phi (no off-diagonal stages). We don't compare against
    a reference here — full-fixture equivalence is covered by the spacc-vs-
    numpy tests below.
    """
    from MCEq.solvers import solv_numpy_etd2

    phi0 = toy_solver_problem[-2].copy()
    grid_idcs = toy_solver_problem[-1]

    solution, grid_sol = solv_numpy_etd2(*toy_solver_problem)
    assert solution.shape == phi0.shape
    assert grid_sol.shape == (len(grid_idcs), phi0.shape[0])
    assert not np.isnan(solution).any()
    assert np.all(np.isfinite(solution))

    for i in range(1, grid_sol.shape[0]):
        assert np.all(grid_sol[i] <= grid_sol[i - 1])


def test_solv_numpy_etd2_does_not_modify_input_phi(toy_solver_setup):
    """Regression: ETD2 must not mutate the input phi array in place."""
    from MCEq.solvers import solv_numpy_etd2

    phi_original = toy_solver_setup[-2]
    phi_copy = phi_original.copy()

    solution, _ = solv_numpy_etd2(*toy_solver_setup)

    assert np.array_equal(phi_original, phi_copy), (
        "solv_numpy_etd2 modified the input phi array - this breaks subsequent "
        "solver calls"
    )
    assert not np.array_equal(solution, phi_copy), (
        "Solver should produce a different result"
    )


@pytest.mark.parametrize("K", [1, 4, 16])
def test_solv_numpy_etd2_multirhs_matches_single_rhs_toy(K):
    """Multi-RHS ETD2 columns match K independent single-RHS solves bit-exactly.

    scipy's CSR ``@`` against a 2-D (n, K) RHS issues per-column SpMVs with
    the same arithmetic as the single-RHS path; the multi-RHS kernel uses
    CSR off-diagonals throughout (bypassing the production BSR conversion
    which doesn't vectorise across K), so the per-column result is
    arithmetically identical to ``solv_numpy_etd2`` with
    ``config.numpy_bsr_blocksize = None``.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solv_numpy_etd2, solv_numpy_etd2_multirhs

    rng = np.random.default_rng(42)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)
    grid_idcs = [5, 15, 25]

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)

    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_multi, grid_multi = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs
    )
    assert sol_multi.shape == (size, K)
    assert grid_multi.shape == (len(grid_idcs), size, K)

    saved_bs = getattr(config, "numpy_bsr_blocksize", 11)
    config.numpy_bsr_blocksize = None
    try:
        for k in range(K):
            try:
                delattr(int_m, "_etd_split_cache_v2")
            except AttributeError:
                pass
            sol_k, grid_k = solv_numpy_etd2(
                nsteps,
                dX,
                rho_inv,
                int_m,
                dec_m,
                phi0_multi[:, k].copy(),
                grid_idcs,
            )
            assert np.array_equal(sol_multi[:, k], sol_k), (
                f"column {k} of multi-RHS solution diverges from single-RHS"
            )
            assert np.array_equal(grid_multi[:, :, k], grid_k), (
                f"column {k} of multi-RHS grid snapshots diverges from single-RHS"
            )
    finally:
        config.numpy_bsr_blocksize = saved_bs


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solve_multirhs_dtype_float32():
    """End-to-end fp32 dispatch through MCEqRun.solve_multirhs.

    Compares the fp32 spacc multirhs path to the fp64 reference at K=4
    on the real SIBYLL21 config; asserts per-cell relative error stays
    below the empirically established 1e-4 budget for the production
    particle set (e± disabled — they're the ``_EM_BLOWUP_CAVEAT`` rows
    whose semi-Lagrangian ETD2 update saturates fp32 dynamic range at
    finite zenith). Stability test
    ``runs/2026-05-21_multi-rhs-etd2-prototype/inputs/test_etd2_fp32.py``
    tracks the per-species figure in more detail.

    Builds a fresh MCEqRun (rather than reusing the ``mceq_sib21``
    fixture which leaves e± enabled) so the test exactly matches the
    production default.
    """
    import crflux.models as pm

    from MCEq.core import MCEqRun

    saved_kernel = config.kernel_config
    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    saved_db = config.mceq_db_fname
    config.kernel_config = "accelerate_etd2"
    config.adv_set["disabled_particles"] = [11, -11]
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=30.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        mceq.solve()
        phi0 = mceq._phi0.copy()
        rng = np.random.default_rng(0)
        K = 4
        phi0_multi = np.stack([s * phi0 for s in rng.uniform(0.5, 1.5, K)], axis=1)

        sol_f64, _ = mceq.solve_multirhs(phi0_multi)
        sol_f32, _ = mceq.solve_multirhs(phi0_multi, dtype=np.float32)
        assert sol_f64.dtype == np.float64
        assert sol_f32.dtype == np.float32

        denom = np.maximum(np.abs(sol_f64), 1e-30)
        rel = np.abs(sol_f32.astype(np.float64) - sol_f64) / denom
        assert rel.max() < 1e-4, (
            f"solve_multirhs fp32 vs fp64 max rel err {rel.max():.2e} "
            f"exceeds 1e-4 budget"
        )
        mceq.close()
    finally:
        config.kernel_config = saved_kernel
        config.adv_set["disabled_particles"] = saved_disabled
        config.mceq_db_fname = saved_db


def test_solv_numpy_etd2_multirhs_rejects_1d_phi():
    """Multi-RHS kernel must reject 1-D phi (caller should use solv_numpy_etd2)."""
    import scipy.sparse as sp

    from MCEq.solvers import solv_numpy_etd2_multirhs

    nsteps = 3
    size = 4
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    grid_idcs = []
    int_m = sp.csr_matrix(-0.1 * np.eye(size))
    dec_m = sp.csr_matrix(-0.05 * np.eye(size))
    phi0_1d = np.ones(size)

    with pytest.raises(ValueError, match="phi must be 2-D"):
        solv_numpy_etd2_multirhs(nsteps, dX, rho_inv, int_m, dec_m, phi0_1d, grid_idcs)


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.parametrize("K", [1, 4, 16])
def test_solv_mkl_etd2_multirhs_matches_numpy_multirhs_toy(K):
    """MKL multi-RHS columns match numpy multi-RHS columns within fp64 ε."""
    import scipy.sparse as sp

    from MCEq.solvers import (
        MklSparseMatrix,
        _etd_split_cache,
        solv_mkl_etd2_multirhs,
        solv_numpy_etd2_multirhs,
    )

    rng = np.random.default_rng(7)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, []
    )
    mkl_int = MklSparseMatrix(int_off) if int_off.nnz else None
    mkl_dec = MklSparseMatrix(dec_off) if dec_off.nnz else None
    try:
        sol_mkl, _ = solv_mkl_etd2_multirhs(
            nsteps, dX, rho_inv, mkl_int, mkl_dec, d_int, d_dec, phi0_multi, []
        )
        np.testing.assert_allclose(sol_mkl, sol_numpy, rtol=5e-13, atol=0)
    finally:
        if mkl_int is not None:
            mkl_int.close()
        if mkl_dec is not None:
            mkl_dec.close()


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.parametrize("K", [1, 4])
def test_solv_mkl_etd2_multirhs_f32_matches_numpy_multirhs_toy(K):
    """fp32 MKL multi-RHS holds 1e-4 rel-L2 vs numpy fp64 reference."""
    import scipy.sparse as sp

    from MCEq.solvers import (
        MklSparseMatrixF32,
        _etd_split_cache,
        solv_mkl_etd2_multirhs_f32,
        solv_numpy_etd2_multirhs,
    )

    rng = np.random.default_rng(7)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, []
    )
    mkl_int32 = (
        MklSparseMatrixF32(int_off.astype(np.float32)) if int_off.nnz else None
    )
    mkl_dec32 = (
        MklSparseMatrixF32(dec_off.astype(np.float32)) if dec_off.nnz else None
    )
    try:
        sol_mkl32, _ = solv_mkl_etd2_multirhs_f32(
            nsteps, dX, rho_inv, mkl_int32, mkl_dec32,
            d_int, d_dec, phi0_multi, [],
        )
        rel_l2 = np.linalg.norm(sol_mkl32 - sol_numpy) / max(
            np.linalg.norm(sol_numpy), 1e-30
        )
        assert rel_l2 < 1e-4, (
            f"mkl multirhs f32 (K={K}) vs numpy fp64 rel-L2 = {rel_l2:.3e}"
        )
    finally:
        if mkl_int32 is not None:
            mkl_int32.close()
        if mkl_dec32 is not None:
            mkl_dec32.close()


@pytest.mark.parametrize("K", [1, 4, 16])
def test_solv_numpy_etd2_rho_stack_multirhs_matches_single_rhs_toy(K):
    """ρ-stack multi-RHS columns match K independent single-RHS ρ-stack solves.

    Toy: 2-slice ρ-stack with scaled interaction matrices, shared decay
    matrix, ramped rho_inv. Multi-RHS columns must equal arithmetic-identical
    single-RHS ρ-stack solves run with BSR disabled (CSR-vs-CSR comparison).
    """
    import scipy.sparse as sp

    from MCEq.solvers import (
        solv_numpy_etd2_rho_stack,
        solv_numpy_etd2_rho_stack_multirhs,
    )

    rng = np.random.default_rng(11)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    # rho_inv spans the ρ-grid so the per-step blend exercises both slices.
    rho_inv = np.linspace(1.0, 4.0, nsteps)
    grid_idcs = [5, 15, 25]

    A_base = rng.standard_normal((size, size)) * 0.05
    A_base[np.abs(A_base) < 0.02] = 0.0
    A_base -= np.diag(np.abs(A_base).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)

    # Two distinct slices to make the per-step blend non-trivial.
    int_m_stack = [
        sp.csr_matrix(A_base),
        sp.csr_matrix(A_base * 0.7),
    ]
    rho_grid = np.array([1e-4, 1e-3])
    dec_m = sp.csr_matrix(B)

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_multi, grid_multi = solv_numpy_etd2_rho_stack_multirhs(
        nsteps, dX, rho_inv, int_m_stack, rho_grid, dec_m, phi0_multi, grid_idcs
    )
    assert sol_multi.shape == (size, K)
    assert grid_multi.shape == (len(grid_idcs), size, K)

    saved_bs = getattr(config, "numpy_bsr_blocksize", 11)
    config.numpy_bsr_blocksize = None
    try:
        for k in range(K):
            for slice_m in int_m_stack:
                try:
                    delattr(slice_m, "_etd_split_cache_v2")
                except AttributeError:
                    pass
            sol_k, grid_k = solv_numpy_etd2_rho_stack(
                nsteps,
                dX,
                rho_inv,
                int_m_stack,
                rho_grid,
                dec_m,
                phi0_multi[:, k].copy(),
                grid_idcs,
            )
            np.testing.assert_allclose(
                sol_multi[:, k],
                sol_k,
                rtol=1e-12,
                atol=0,
                err_msg=f"column {k} of ρ-stack multi-RHS solution diverges",
            )
            np.testing.assert_allclose(
                grid_multi[:, :, k],
                grid_k,
                rtol=1e-12,
                atol=0,
                err_msg=f"column {k} of ρ-stack multi-RHS grid snapshots diverges",
            )
    finally:
        config.numpy_bsr_blocksize = saved_bs


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize("K", [1, 4, 16])
def test_solv_spacc_etd2_multirhs_matches_numpy_multirhs_toy(K):
    """Spacc multi-RHS columns match numpy multi-RHS columns within fp64 eps.

    Both kernels evaluate identical math (Cox–Matthews ETD2 with the same
    diagonal split and accumulated SpMM); the only difference is the
    sparse SpMM backend (scipy CSR vs Apple Accelerate
    ``sparse_matrix_product_dense_double``). Differences are at the few
    ULP level and the test uses np.allclose with a tight tolerance.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solv_numpy_etd2_multirhs, solv_spacc_etd2_multirhs
    from MCEq.spacc import SpaccMatrix

    rng = np.random.default_rng(7)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)
    grid_idcs = [5, 15, 25]

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    # Pre-split for spacc — solv_spacc_etd2_multirhs expects SpaccMatrix-wrapped
    # off-diagonals plus plain numpy diagonals, like the single-RHS spacc kernel.
    from MCEq.solvers import _etd_split_cache

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    spacc_int = SpaccMatrix(int_off) if int_off.nnz else None
    spacc_dec = SpaccMatrix(dec_off) if dec_off.nnz else None

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs
    )
    sol_spacc, _ = solv_spacc_etd2_multirhs(
        nsteps,
        dX,
        rho_inv,
        spacc_int,
        spacc_dec,
        d_int,
        d_dec,
        phi0_multi,
        grid_idcs,
    )

    np.testing.assert_allclose(sol_spacc, sol_numpy, rtol=5e-13, atol=0)

    if spacc_int is not None:
        spacc_int.close()
    if spacc_dec is not None:
        spacc_dec.close()


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
@pytest.mark.parametrize("K", [1, 4, 16])
def test_solv_cuda_etd2_multirhs_matches_numpy_multirhs_toy(K):
    """cupy multi-RHS columns match numpy multi-RHS columns within
    cuSPARSE-reorder tolerance.

    cuSPARSE reorders partial sums (warp-reduction order differs from
    scipy's row-major accumulation), so we tolerate a relative L2 of 1e-10
    rather than round-off. The single-RHS cuda test uses the same bound.
    """
    import scipy.sparse as sp

    from MCEq.solvers import (
        CudaEtd2MultiRHSContext,
        _etd_split_cache,
        solv_cuda_etd2_multirhs,
        solv_numpy_etd2_multirhs,
    )

    rng = np.random.default_rng(7)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)
    grid_idcs = [5, 15, 25]

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, grid_numpy = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs
    )

    ctx = CudaEtd2MultiRHSContext(
        int_off,
        dec_off,
        d_int,
        d_dec,
        K=K,
        device_id=config.cuda_gpu_id,
        fp_precision=64,
    )
    sol_cuda, grid_cuda = solv_cuda_etd2_multirhs(
        nsteps, dX, rho_inv, ctx, phi0_multi, grid_idcs
    )

    rel_l2 = np.linalg.norm(sol_cuda - sol_numpy) / max(
        np.linalg.norm(sol_numpy), 1e-30
    )
    assert rel_l2 < 1e-10, (
        f"cuda multirhs (K={K}) vs numpy multirhs rel-L2 = {rel_l2:.3e}"
    )
    if grid_idcs:
        rel_grid = np.linalg.norm(grid_cuda - grid_numpy) / max(
            np.linalg.norm(grid_numpy), 1e-30
        )
        assert rel_grid < 1e-10, (
            f"cuda multirhs grid snapshots rel-L2 = {rel_grid:.3e}"
        )


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
@pytest.mark.parametrize("K", [1, 4])
def test_solv_cuda_etd2_multirhs_f32_matches_numpy_multirhs_toy(K):
    """fp32 cupy multi-RHS holds 1e-4 rel-L2 vs the fp64 numpy reference.

    Per the multi-RHS handover plan: fp32 stability budget is 1e-4 relative
    error (verified against per-particle MCEq SIBYLL21 fluxes on Mac
    Accelerate; same arithmetic carries to cupy by construction).
    """
    import scipy.sparse as sp

    from MCEq.solvers import (
        CudaEtd2MultiRHSContext,
        _etd_split_cache,
        solv_cuda_etd2_multirhs,
        solv_numpy_etd2_multirhs,
    )

    rng = np.random.default_rng(7)
    nsteps = 30
    size = 24
    dX = np.full(nsteps, 0.1)
    rho_inv = np.linspace(1.0, 2.0, nsteps)

    A = rng.standard_normal((size, size)) * 0.05
    A[np.abs(A) < 0.02] = 0.0
    A -= np.diag(np.abs(A).sum(axis=1) + 0.1)
    B = rng.standard_normal((size, size)) * 0.02
    B[np.abs(B) < 0.01] = 0.0
    B -= np.diag(np.abs(B).sum(axis=1) + 0.05)
    int_m = sp.csr_matrix(A)
    dec_m = sp.csr_matrix(B)

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solv_numpy_etd2_multirhs(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, []
    )
    ctx = CudaEtd2MultiRHSContext(
        int_off,
        dec_off,
        d_int,
        d_dec,
        K=K,
        device_id=config.cuda_gpu_id,
        fp_precision=32,
    )
    sol_cuda32, _ = solv_cuda_etd2_multirhs(
        nsteps, dX, rho_inv, ctx, phi0_multi, []
    )
    rel_l2 = np.linalg.norm(sol_cuda32 - sol_numpy) / max(
        np.linalg.norm(sol_numpy), 1e-30
    )
    assert rel_l2 < 1e-4, (
        f"cuda multirhs f32 (K={K}) vs numpy fp64 rel-L2 = {rel_l2:.3e}"
    )


def _muon_flux(mceq, phi):
    """E^3 * (mu+ + mu-) flux on mceq.e_grid, in arbitrary units."""
    e = mceq.e_grid
    flux = np.zeros_like(e)
    for name in ("mu+", "mu-"):
        sl = mceq.pman[name].lidx, mceq.pman[name].uidx
        flux += phi[sl[0] : sl[1]]
    return e, e**3 * flux


def _solve_with_kernel(mceq, kernel_name):
    """Run mceq.solve() with the given kernel_config; restore on the way out."""
    saved = config.kernel_config
    config.kernel_config = kernel_name
    try:
        # Force re-derivation of the integration path so the chosen kernel
        # actually runs end-to-end.
        mceq.integration_path = None
        mceq.solve()
        return mceq._solution.copy()
    finally:
        config.kernel_config = saved


def test_solv_numpy_etd2_stable_at_high_zenith():
    """Regression: ETD2 must stay finite at theta=89 deg.

    At extreme zenith, rho_inv blows up and forward-Euler-style schemes
    explode on rows with weak diagonal damping. The ETD2 design treats the
    diagonal exactly via an integrating factor; this test locks in that
    property — the integrator must not return non-finite values.

    e+/e- are disabled because their semi-Lagrangian L/R-variant rows have
    no diagonal damping and require a block-ETD generalization (see
    docs/mceq_v1.x_v2_diff.md). This is a known limitation, not a regression.
    """
    import crflux.models as pm

    from MCEq.core import MCEqRun

    saved = list(config.adv_set.get("disabled_particles", []))
    saved_kernel = config.kernel_config
    saved_db = config.mceq_db_fname
    config.adv_set["disabled_particles"] = [11, -11]
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=89.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        phi_etd = _solve_with_kernel(mceq, "numpy_etd2")

        assert np.all(np.isfinite(phi_etd)), "ETD2 blew up at theta=89 deg"

        e, mu_etd = _muon_flux(mceq, phi_etd)
        band = (e > 1.0) & (mu_etd > 1e-30)
        assert band.any(), "no nonzero muon-flux band found"
    finally:
        config.adv_set["disabled_particles"] = saved
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db


def _etd2_oversampled(int_m, dec_m, phi0, dX, rho_inv, oversample):
    """ETD2RK with `oversample` substeps per native step. Mirrors the
    production kernel's update rule so the convergence test exercises the
    same math."""
    from MCEq.solvers import _etd_split_cache

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    phi = phi0.astype(np.float64).copy()

    PHI1_SMALL = 1e-6
    PHI2_SMALL = 1e-3

    for k in range(len(dX)):
        h_full = dX[k]
        ri = rho_inv[k]
        D = d_int + ri * d_dec
        for _ in range(oversample):
            h = h_full / oversample
            hD = h * D
            eD = np.exp(hD)
            phi1 = np.where(
                np.abs(hD) > PHI1_SMALL,
                (eD - 1.0) / np.where(hD != 0.0, hD, 1.0),
                1.0 + 0.5 * hD + hD * hD / 6.0,
            )
            phi2 = np.where(
                np.abs(hD) > PHI2_SMALL,
                (eD - 1.0 - hD) / np.where(hD != 0.0, hD * hD, 1.0),
                0.5 + hD / 6.0 + hD * hD / 24.0,
            )
            F_phi = int_off.dot(phi) + ri * dec_off.dot(phi)
            a = eD * phi + h * phi1 * F_phi
            F_a = int_off.dot(a) + ri * dec_off.dot(a)
            phi = a + h * phi2 * (F_a - F_phi)
    return phi


def test_solv_numpy_etd2_second_order_convergence(mceq_sib21):
    """ETD2 should exhibit observed convergence order ~2 under h-refinement.

    Build the reference via ETD2 at high oversample (rather than oversampled
    Euler). Using the same scheme for truth keeps the truth's residual
    error a factor of ~16 below the test points (since ETD2 is second-order
    and we refine by 8x), so the measured ratio reflects ETD2's own
    asymptotic constant rather than the truth's first-order leftover.

    A floor of 1.8 catches regressions while tolerating constant noise in
    the asymptotic regime.
    """
    mceq_sib21.set_theta_deg(0.0)

    saved_kernel = config.kernel_config
    config.kernel_config = "numpy_etd2"
    try:
        mceq_sib21.integration_path = None
        mceq_sib21._calculate_integration_path(int_grid=None, grid_var="X")
        _, dX_full, rho_inv_full, _ = mceq_sib21.integration_path
    finally:
        config.kernel_config = saved_kernel

    # Use the full ETD2 path. It is much coarser than the old Euler native
    # grid, but the convergence ratio measurement only depends on whether
    # the cumulative dynamics are large enough to lift `phi_h` above the
    # round-off floor — which the assertion below guards.
    dX = dX_full
    rho_inv = rho_inv_full

    int_m = mceq_sib21.int_m.tocsr()
    dec_m = mceq_sib21.dec_m.tocsr()
    phi0 = mceq_sib21._phi0.copy()

    # ETD2 truth at oversample=16. Using ETD2 (rather than Euler) for the
    # reference means the truth's own residual is O((h/16)^2) — far below
    # ETD2 at os=1 or os=2, so it doesn't pollute the order estimate.
    phi_truth = _etd2_oversampled(int_m, dec_m, phi0, dX, rho_inv, oversample=16)
    norm_truth = np.linalg.norm(phi_truth)
    assert norm_truth > 0

    phi_h = _etd2_oversampled(int_m, dec_m, phi0, dX, rho_inv, oversample=1)
    phi_h2 = _etd2_oversampled(int_m, dec_m, phi0, dX, rho_inv, oversample=2)

    err_h = np.linalg.norm(phi_h - phi_truth) / norm_truth
    err_h2 = np.linalg.norm(phi_h2 - phi_truth) / norm_truth

    assert err_h > 1e-10, (
        f"ETD2 native-grid error {err_h:.3e} too small to measure order — "
        "test is in floating-point-noise regime"
    )
    assert err_h2 < err_h, (
        f"ETD2 error did not decrease under h-refinement: "
        f"err(h)={err_h:.3e} err(h/2)={err_h2:.3e}"
    )

    order = np.log2(err_h / err_h2)
    assert order >= 1.8, (
        f"ETD2 observed order {order:.2f} below 1.8 "
        f"(err(h)={err_h:.3e}, err(h/2)={err_h2:.3e})"
    )


# ---------------------------------------------------------------------------
# ETD2 (spacc / Apple Accelerate) tests
# ---------------------------------------------------------------------------
@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solv_spacc_etd2_matches_numpy_etd2_toy(toy_solver_problem):
    """Trivial-matrix smoke test.

    The toy fixture has purely diagonal int_m/dec_m, so both off-diagonals
    are empty (nnz=0). The kernel should detect that and skip the SpMV
    calls; the result is just the integrating-factor `exp(h*D) * phi` per
    step. This catches the empty-matrix code path without requiring a
    full MCEqRun.
    """
    import MCEq.spacc as spacc
    from MCEq.solvers import _etd_split_cache, solv_numpy_etd2, solv_spacc_etd2

    nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs = toy_solver_problem

    sol_numpy, _ = solv_numpy_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi.copy(), grid_idcs
    )

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    # int_off / dec_off may be empty here; the kernel should handle that.
    spacc_int_off = spacc.SpaccMatrix(int_off) if int_off.nnz > 0 else None
    spacc_dec_off = spacc.SpaccMatrix(dec_off) if dec_off.nnz > 0 else None
    sol_spacc, _ = solv_spacc_etd2(
        nsteps,
        dX,
        rho_inv,
        spacc_int_off,
        spacc_dec_off,
        d_int,
        d_dec,
        phi.copy(),
        grid_idcs,
    )
    assert sol_spacc == pytest.approx(sol_numpy, rel=1e-12, abs=1e-15), (
        "spacc_etd2 differs from numpy_etd2 on toy diagonal-only problem"
    )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solv_spacc_etd2_matches_numpy_etd2_real(mceq_sib21):
    """Equivalence test on real MCEq matrices with non-trivial off-diagonals.

    Builds a uniform mid-point sampled path at theta=60 with the default
    (no-mixing) particle treatment, runs both kernels on that fixed path,
    asserts agreement to ~1e-12 — the 4 SpMVs/step are the same operation
    in both backends, so equality is essentially arithmetic round-off.
    """
    import MCEq.spacc as spacc
    from MCEq.solvers import _etd_split_cache, solv_numpy_etd2, solv_spacc_etd2

    mceq_sib21.set_theta_deg(60.0)

    h = 5.0
    max_X = mceq_sib21.density_model.max_X
    n_full = int((max_X - config.X_start) // h)
    tail = (max_X - config.X_start) - n_full * h
    if tail > 1e-9:
        dX = np.full(n_full + 1, h, dtype=np.float64)
        dX[-1] = tail
    else:
        dX = np.full(n_full, h, dtype=np.float64)
    Xs = config.X_start + np.concatenate([[0.0], np.cumsum(dX)[:-1]])
    ri = mceq_sib21.density_model.r_X2rho
    rho_inv = np.array([ri(Xs[i] + 0.5 * dX[i]) for i in range(len(dX))])
    grid_idcs = []
    nsteps = len(dX)
    phi0 = mceq_sib21._phi0.copy()

    sol_numpy, _ = solv_numpy_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq_sib21.int_m,
        mceq_sib21.dec_m,
        phi0.copy(),
        grid_idcs,
    )

    d_int, d_dec, int_off, dec_off = _etd_split_cache(
        mceq_sib21.int_m, mceq_sib21.dec_m
    )
    assert int_off.nnz > 0 and dec_off.nnz > 0, (
        "real matrices should have non-empty off-diagonals"
    )
    spacc_int_off = spacc.SpaccMatrix(int_off)
    spacc_dec_off = spacc.SpaccMatrix(dec_off)
    sol_spacc, _ = solv_spacc_etd2(
        nsteps,
        dX,
        rho_inv,
        spacc_int_off,
        spacc_dec_off,
        d_int,
        d_dec,
        phi0.copy(),
        grid_idcs,
    )

    assert np.all(np.isfinite(sol_spacc)), "spacc_etd2 produced non-finite values"
    rel_l2 = np.linalg.norm(sol_spacc - sol_numpy) / max(
        np.linalg.norm(sol_numpy), 1e-30
    )
    assert rel_l2 < 1e-12, (
        f"spacc_etd2 vs numpy_etd2 rel-L2 = {rel_l2:.3e} (expected < 1e-12)"
    )


# ---------------------------------------------------------------------------
# ETD2 (Intel MKL / NVIDIA CUDA) tests
# ---------------------------------------------------------------------------
# These backend tests build their own MCEqRun rather than reusing the shared
# ``mceq_sib21`` fixture (which is calibrated against the reduced compact DB
# used by the calibration tests in test_core.py). They only need a working
# matrix system; the full DB is the more portable choice and keeps these
# tests independent of the compact-DB calibration regime.
@pytest.fixture(scope="module")
def mceq_sib21_full_db():
    """Module-scoped MCEqRun against the full DB for backend equivalence tests."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    saved_db = config.mceq_db_fname
    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    config.mceq_db_fname = "mceq_db_lext_dpm193_v140.h5"
    config.adv_set["disabled_particles"] = []
    try:
        if config.has_mkl:
            config.set_mkl_threads(2)
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        yield mceq
    finally:
        config.mceq_db_fname = saved_db
        config.adv_set["disabled_particles"] = saved_disabled


def _uniform_path_theta60(mceq, h=5.0):
    """Build a uniform mid-point-sampled path at theta=60 deg.

    Used by the MKL and CUDA equivalence tests so both run on identical
    inputs and produce comparable rel-L2 numbers.
    """
    mceq.set_theta_deg(60.0)
    max_X = mceq.density_model.max_X
    n_full = int((max_X - config.X_start) // h)
    tail = (max_X - config.X_start) - n_full * h
    if tail > 1e-9:
        dX = np.full(n_full + 1, h, dtype=np.float64)
        dX[-1] = tail
    else:
        dX = np.full(n_full, h, dtype=np.float64)
    Xs = config.X_start + np.concatenate([[0.0], np.cumsum(dX)[:-1]])
    ri = mceq.density_model.r_X2rho
    rho_inv = np.array([ri(Xs[i] + 0.5 * dX[i]) for i in range(len(dX))])
    return len(dX), dX, rho_inv


@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.parametrize("blocksize", [None, 6], ids=["csr", "bsr6"])
def test_solv_mkl_etd2_matches_numpy_etd2_real(mceq_sib21_full_db, blocksize):
    """Equivalence test on real MCEq matrices, both CSR and BSR(6) paths.

    CSR is bit-exact vs numpy (~1e-12); BSR reorders the SpMV partial
    sums per-block and lands at ~1e-10 on these matrices — still
    essentially round-off, just looser.
    """
    from MCEq.solvers import (
        MklSparseMatrix,
        _etd_split_cache,
        solv_mkl_etd2,
        solv_numpy_etd2,
    )

    mceq = mceq_sib21_full_db
    nsteps, dX, rho_inv = _uniform_path_theta60(mceq)
    grid_idcs = []
    phi0 = mceq._phi0.copy()

    sol_numpy, _ = solv_numpy_etd2(
        nsteps, dX, rho_inv, mceq.int_m, mceq.dec_m, phi0.copy(), grid_idcs
    )

    d_int, d_dec, int_off, dec_off = _etd_split_cache(mceq.int_m, mceq.dec_m)
    assert int_off.nnz > 0 and dec_off.nnz > 0, (
        "real matrices should have non-empty off-diagonals"
    )
    mkl_int_off = MklSparseMatrix(int_off.tocsr(), blocksize=blocksize)
    mkl_dec_off = MklSparseMatrix(dec_off.tocsr(), blocksize=blocksize)
    sol_mkl, _ = solv_mkl_etd2(
        nsteps,
        dX,
        rho_inv,
        mkl_int_off,
        mkl_dec_off,
        d_int,
        d_dec,
        phi0.copy(),
        grid_idcs,
    )

    assert np.all(np.isfinite(sol_mkl)), "mkl_etd2 produced non-finite values"
    rel_l2 = np.linalg.norm(sol_mkl - sol_numpy) / max(np.linalg.norm(sol_numpy), 1e-30)
    # CSR is the same arithmetic as numpy → bit-exact bound. BSR groups
    # the SpMV per block, which reorders partial sums and loosens to ~1e-10.
    tol = 1e-12 if blocksize is None else 1e-9
    assert rel_l2 < tol, (
        f"mkl_etd2(blocksize={blocksize}) vs numpy_etd2 rel-L2 = {rel_l2:.3e} "
        f"(expected < {tol})"
    )


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_numpy_etd2_empty_dec_off_bsr_padding():
    """Regression: a pure e±/γ EM-cascade solve disables all decays, so dec_m
    has no off-diagonal. With BSR padding on (default blocksize), the empty
    dec_off must pad to the SAME dimension as the non-empty int_off — otherwise
    the per-step ``dec_off.dot(phc)`` crashes on a dim mismatch (N vs N+pad).
    This is the bug that broke native ``solve()`` on the EM-cascade DB. See
    ``_etd_off_to_bsr``.
    """
    import scipy.sparse as sp

    from MCEq import config
    from MCEq.solvers import solv_numpy_etd2

    dim = 50  # 50 % 11 != 0 -> BSR padding is active at the default blocksize
    rng = np.random.default_rng(0)
    int_m = sp.csr_matrix(rng.standard_normal((dim, dim)) * 0.01)
    int_m.setdiag(-0.5)
    int_m = int_m.tocsr()
    dec_m = sp.csr_matrix((dim, dim))  # all decays disabled -> empty off-diagonal

    nsteps = 20
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    phi = np.ones(dim)
    saved = getattr(config, "numpy_bsr_blocksize", None)
    config.numpy_bsr_blocksize = 11
    try:
        sol, _ = solv_numpy_etd2(nsteps, dX, rho_inv, int_m, dec_m, phi, [])
    finally:
        config.numpy_bsr_blocksize = saved
    assert sol.shape == (dim,)
    assert np.isfinite(sol).all()


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_mkl_bsr_handles_padding():
    """BSR with a dimension that's not a multiple of blocksize must round-trip.

    SIBYLL21 happens to give dim=8712 (divisible by 6). This test uses a
    synthetic 7x7 matrix with blocksize=3 to exercise the padding code path
    end-to-end via the kernel. Padding rows/cols stay zero through SpMV, so
    the result should match a CSR run to round-off.
    """
    import scipy.sparse as sp

    from MCEq.solvers import MklSparseMatrix, solv_mkl_etd2

    rng = np.random.default_rng(0)
    dim = 7  # not a multiple of any common blocksize
    int_off = sp.csr_matrix(rng.standard_normal((dim, dim)) * 0.1)
    dec_off = sp.csr_matrix(rng.standard_normal((dim, dim)) * 0.05)
    # zero the diagonals — that's the off-diagonal contract
    int_off.setdiag(0)
    dec_off.setdiag(0)
    int_off.eliminate_zeros()
    dec_off.eliminate_zeros()
    d_int = -0.2 * np.ones(dim)
    d_dec = -0.05 * np.ones(dim)
    phi0 = np.ones(dim)
    nsteps = 5
    dX = np.full(nsteps, 0.5)
    rho_inv = np.linspace(1.0, 0.5, nsteps)

    sol_csr, _ = solv_mkl_etd2(
        nsteps,
        dX,
        rho_inv,
        MklSparseMatrix(int_off, blocksize=None),
        MklSparseMatrix(dec_off, blocksize=None),
        d_int,
        d_dec,
        phi0.copy(),
        [],
    )
    sol_bsr, _ = solv_mkl_etd2(
        nsteps,
        dX,
        rho_inv,
        MklSparseMatrix(int_off, blocksize=3),  # forces padding 7 -> 9
        MklSparseMatrix(dec_off, blocksize=3),
        d_int,
        d_dec,
        phi0.copy(),
        [],
    )
    assert sol_csr.shape == (dim,)
    assert sol_bsr.shape == (dim,)
    assert np.allclose(sol_csr, sol_bsr, rtol=1e-10, atol=1e-12), (
        f"BSR padded path differs from CSR: csr={sol_csr}, bsr={sol_bsr}"
    )


@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_solv_mkl_etd2_stable_at_high_zenith():
    """Regression: MKL ETD2 must stay finite at theta=89 deg.

    Mirrors the numpy_etd2 stability test — at extreme zenith the
    diagonal-exact treatment is the only thing keeping the integrator
    stable. e±/γ disabled per the EM-cascade caveat.
    """
    import crflux.models as pm

    from MCEq.core import MCEqRun

    saved = list(config.adv_set.get("disabled_particles", []))
    saved_kernel = config.kernel_config
    saved_db = config.mceq_db_fname
    config.adv_set["disabled_particles"] = [11, -11]
    config.mceq_db_fname = "mceq_db_lext_dpm193_v140.h5"
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=89.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        config.kernel_config = "mkl_etd2"
        mceq.integration_path = None
        mceq.solve()
        phi = mceq._solution
        assert np.all(np.isfinite(phi)), "MKL ETD2 blew up at theta=89 deg"
    finally:
        config.adv_set["disabled_particles"] = saved
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db


# ---------------------------------------------------------------------------
# ETD2 (NVIDIA CUDA) tests
# ---------------------------------------------------------------------------
@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
def test_solv_cuda_etd2_matches_numpy_etd2_real(mceq_sib21_full_db):
    """Equivalence test on real MCEq matrices.

    cuSPARSE may reorder partial sums vs scipy CSR, so we tolerate a
    rel-L2 of 1e-9 instead of round-off — but anything looser would mask
    a real bug. Path matches the MKL test for parity.
    """
    from MCEq.solvers import (
        CudaEtd2Context,
        _etd_split_cache,
        solv_cuda_etd2,
        solv_numpy_etd2,
    )

    mceq = mceq_sib21_full_db
    nsteps, dX, rho_inv = _uniform_path_theta60(mceq)
    grid_idcs = []
    phi0 = mceq._phi0.copy()

    sol_numpy, _ = solv_numpy_etd2(
        nsteps, dX, rho_inv, mceq.int_m, mceq.dec_m, phi0.copy(), grid_idcs
    )

    d_int, d_dec, int_off, dec_off = _etd_split_cache(mceq.int_m, mceq.dec_m)
    ctx = CudaEtd2Context(
        int_off.tocsr(),
        dec_off.tocsr(),
        d_int,
        d_dec,
        device_id=config.cuda_gpu_id,
        fp_precision=64,
    )
    sol_cuda, _ = solv_cuda_etd2(nsteps, dX, rho_inv, ctx, phi0.copy(), grid_idcs)

    assert np.all(np.isfinite(sol_cuda)), "cuda_etd2 produced non-finite values"
    rel_l2 = np.linalg.norm(sol_cuda - sol_numpy) / max(
        np.linalg.norm(sol_numpy), 1e-30
    )
    # cuSPARSE reorders partial sums vs scipy — round-off-bounded but not
    # bit-exact. 1e-9 catches systematic bugs while tolerating reorder.
    assert rel_l2 < 1e-9, (
        f"cuda_etd2 vs numpy_etd2 rel-L2 = {rel_l2:.3e} (expected < 1e-9)"
    )


@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
def test_solv_cuda_etd2_stable_at_high_zenith():
    """Regression: CUDA ETD2 must stay finite at theta=89 deg."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    saved = list(config.adv_set.get("disabled_particles", []))
    saved_kernel = config.kernel_config
    saved_db = config.mceq_db_fname
    config.adv_set["disabled_particles"] = [11, -11]
    config.mceq_db_fname = "mceq_db_lext_dpm193_v140.h5"
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=89.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        config.kernel_config = "cuda_etd2"
        mceq.integration_path = None
        mceq.solve()
        phi = mceq._solution
        assert np.all(np.isfinite(phi)), "CUDA ETD2 blew up at theta=89 deg"
    finally:
        config.adv_set["disabled_particles"] = saved
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db


# ---------------------------------------------------------------------------
# ETD2 on GeneralizedTarget (uniform-density profile)
# ---------------------------------------------------------------------------
def test_solv_numpy_etd2_generalized_target_convergence():
    """ETD2 must converge with order ~2 on a uniform-density target.

    For a constant-density profile, the non-uniform `ρ`-aware path
    degenerates to uniform `h_max` because `|d ln ρ⁻¹/dX| = 0`. We
    therefore test against literal uniform stepping at decreasing `h` and
    require that the rel-L2 error vs the finest reference is consistent
    with second-order convergence.

    This is a regression test against future kernel changes silently
    breaking on non-atmospheric targets.
    """
    import crflux.models as pm

    from MCEq.core import MCEqRun
    from MCEq.geometry.density_profiles import GeneralizedTarget
    from MCEq.solvers import solv_numpy_etd2

    saved_kernel = config.kernel_config
    saved_db = config.mceq_db_fname
    saved_disabled = list(config.adv_set.get("disabled_particles", []))

    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    config.adv_set["disabled_particles"] = [11, -11]
    try:
        target = GeneralizedTarget(len_target=1000.0, env_density=1.0, env_name="water")
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        mceq.set_density_model(target)
        max_X = mceq.density_model.max_X

        sols = {}
        for h in (4.0, 2.0, 1.0, 0.5):
            n = int(np.ceil(max_X / h))
            dX = np.full(n, h, dtype=np.float64)
            dX[-1] = max_X - (n - 1) * h
            rho_inv = np.full(n, 1.0 / target.env_density, dtype=np.float64)
            sol, _ = solv_numpy_etd2(
                n,
                dX,
                rho_inv,
                mceq.int_m,
                mceq.dec_m,
                mceq._phi0.copy(),
                [],
            )
            assert np.all(np.isfinite(sol)), (
                f"ETD2 on water at h={h} produced non-finite values"
            )
            sols[h] = sol

        ref = sols[0.5]
        norm_ref = np.linalg.norm(ref)
        assert norm_ref > 0
        err = {h: np.linalg.norm(sols[h] - ref) / norm_ref for h in sols}

        # Each halving of h should drop error by ~4× (order 2). Allow some
        # slack in the asymptotic-constant regime.
        for h_coarse, h_fine in ((4.0, 2.0), (2.0, 1.0)):
            ratio = err[h_coarse] / err[h_fine] if err[h_fine] > 0 else float("inf")
            assert ratio > 3.0, (
                f"ETD2 on water: error ratio h={h_coarse}->{h_fine} is "
                f"{ratio:.2f}, below the 3.0 floor expected for O(h²) "
                f"convergence (err({h_coarse})={err[h_coarse]:.2e}, "
                f"err({h_fine})={err[h_fine]:.2e})"
            )

        # And the absolute error at h=4 should already be small.
        assert err[4.0] < 5e-2, (
            f"ETD2 on water at h=4 has rel-L2={err[4.0]:.2e} vs h=0.5 "
            f"reference; expected < 5e-2"
        )
    finally:
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db
        config.adv_set["disabled_particles"] = saved_disabled


# ---------------------------------------------------------------------------
# ETD2 path-parameter wiring through MCEqRun.solve()
# ---------------------------------------------------------------------------
def test_etd2_solve_default_path(mceq_sib21):
    """``mceq.solve()`` with ``kernel_config="numpy_etd2"`` must build a
    non-uniform path automatically (no ``solve_from_integration_path``
    needed) and produce a finite muon spectrum.

    Locks in the wiring: the ETD2 branch in ``_calculate_integration_path``
    is reached, the public ``etd2_nonuniform_path`` builder is invoked,
    and the resulting path populates ``mceq.integration_path`` with a
    step count well below the per-decade-of-X cap.
    """
    mceq_sib21.set_theta_deg(60.0)
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        mceq_sib21.integration_path = None
        mceq_sib21.solve()  # no explicit path injection — should auto-build
        n_etd = mceq_sib21.integration_path[0]
        mu_etd = mceq_sib21.get_solution("total_mu+", 0) + mceq_sib21.get_solution(
            "total_mu-", 0
        )
    finally:
        config.kernel_config = saved_kernel

    # The ETD2 nonuniform path on the standard atmosphere at 60 deg is
    # ~150-300 steps depending on the dX_max cap; both ends should be well
    # under 1000 (an Euler native grid would have ~10000).
    assert n_etd < 1000, f"ETD2 path is suspiciously dense: n_etd={n_etd}"
    assert n_etd > 10, f"ETD2 path is suspiciously sparse: n_etd={n_etd}"
    assert np.all(np.isfinite(mu_etd)), "ETD2 default solve produced non-finite mu"

    e = mceq_sib21.e_grid
    band = (e > 1.0) & (mu_etd > 1e-30)
    assert band.any(), "no nonzero muon-flux band found"


def test_etd2_solve_eps_override_shifts_step_count(mceq_sib21):
    """Passing ``eps`` to ``solve()`` must propagate through the cache and
    actually rebuild the path. Smaller eps → more steps."""
    mceq_sib21.set_theta_deg(60.0)
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        mceq_sib21.integration_path = None
        mceq_sib21.solve(eps=0.3)
        n_default = mceq_sib21.integration_path[0]
        mceq_sib21.solve(eps=0.1)
        n_finer = mceq_sib21.integration_path[0]
        mceq_sib21.solve(eps=1.0)
        n_coarser = mceq_sib21.integration_path[0]
    finally:
        config.kernel_config = saved_kernel

    assert n_finer > n_default > n_coarser, (
        f"eps override did not change the path: "
        f"n(eps=0.1)={n_finer}, n(eps=0.3)={n_default}, n(eps=1.0)={n_coarser}"
    )


def test_etd2_solve_path_cache_invalidates_on_param_change(mceq_sib21):
    """The path cache must invalidate when an ETD2 parameter changes
    between calls; otherwise param overrides would be silently ignored."""
    mceq_sib21.set_theta_deg(60.0)
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        mceq_sib21.integration_path = None
        mceq_sib21.solve()
        path_a = mceq_sib21.integration_path
        mceq_sib21.solve(dX_max=10.0)
        path_b = mceq_sib21.integration_path
    finally:
        config.kernel_config = saved_kernel

    # dX_max=10 must produce more steps than the default dX_max=20
    assert path_b[0] > path_a[0], (
        f"dX_max change did not invalidate cache: "
        f"n(default)={path_a[0]}, n(dX_max=10)={path_b[0]}"
    )


def test_etd2_solve_int_grid_dense(mceq_sib21):
    """A user-supplied int_grid must produce at least len(int_grid) steps
    and a snapshot grid_idcs of matching length, regardless of how dense
    the grid is relative to the natural ETD2 schedule.

    Tests two cases:
      - sparse grid (50 evenly spaced points): natural path dominates,
        but the requested points still land on step boundaries.
      - dense grid (5000 evenly spaced points): grid is much finer than
        the natural ~20 g/cm² bulk step, forcing every bulk step to
        truncate and land on a snapshot.
    """
    mceq_sib21.set_theta_deg(60.0)
    saved_kernel = config.kernel_config
    max_X = mceq_sib21.density_model.max_X
    try:
        config.kernel_config = "numpy_etd2"

        for n_grid in (50, 5000):
            int_grid = np.linspace(max_X / n_grid, max_X, n_grid, dtype=np.float64)
            mceq_sib21.integration_path = None
            mceq_sib21.solve(int_grid=int_grid)
            nsteps, dX, rho_inv, grid_idcs = mceq_sib21.integration_path

            # Step count must be >= len(int_grid) (each grid point lands
            # on a step boundary)
            assert nsteps >= n_grid, f"n_grid={n_grid}: nsteps={nsteps} < len(int_grid)"
            # Every requested snapshot must be recorded
            assert len(grid_idcs) == n_grid, (
                f"n_grid={n_grid}: got {len(grid_idcs)} snapshots, expected {n_grid}"
            )
            # And the cumulative step boundaries must hit each int_grid value
            X_boundaries = np.cumsum(dX)
            recorded_X = X_boundaries[np.asarray(grid_idcs)]
            assert np.allclose(recorded_X, int_grid, rtol=0, atol=1e-9), (
                f"n_grid={n_grid}: snapshot positions don't match int_grid "
                f"(max diff = {np.max(np.abs(recorded_X - int_grid)):.3e})"
            )
            # Grid solutions must have the right shape
            assert mceq_sib21.grid_sol.shape[0] == n_grid, (
                f"n_grid={n_grid}: grid_sol has {mceq_sib21.grid_sol.shape[0]} "
                f"snapshots, expected {n_grid}"
            )
    finally:
        config.kernel_config = saved_kernel


def test_etd2_solve_int_grid_below_X_start_raises(mceq_sib21):
    """An int_grid value strictly below X_start must raise immediately.

    A snapshot exactly at X_start is allowed (records the initial state);
    only points below it are rejected.
    """
    mceq_sib21.set_theta_deg(60.0)
    saved_kernel = config.kernel_config
    saved_X_start = config.X_start
    try:
        config.kernel_config = "numpy_etd2"
        config.X_start = 50.0
        mceq_sib21.integration_path = None
        with pytest.raises(ValueError, match="larger than or equal to X_start"):
            mceq_sib21.solve(int_grid=np.array([10.0, 100.0, 500.0]))
    finally:
        config.kernel_config = saved_kernel
        config.X_start = saved_X_start


# ---------------------------------------------------------------------------
# solve_fullsky — 2-D phi0 (per-pixel initial spectrum)
# ---------------------------------------------------------------------------
def test_solve_fullsky_2d_phi0_tiled_matches_1d(mceq_sib21):
    """Tiling a 1-D phi0 into a 2-D (dim, K) array must produce identical
    per-pixel solutions to passing the 1-D phi0 directly. Locks in the
    invariant that 2-D phi0 with identical columns is the broadcast path.
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([0.0, 30.0, 60.0])
        K = zenith_grid.size
        phi0_1d = mceq_sib21._phi0.copy()

        sol_1d, _ = mceq_sib21.solve_fullsky(zenith_grid)
        phi0_2d = np.broadcast_to(
            phi0_1d[:, None], (mceq_sib21.dim_states, K)
        ).copy()
        sol_2d, _ = mceq_sib21.solve_fullsky(zenith_grid, phi0=phi0_2d)

        assert sol_2d.shape == sol_1d.shape
        np.testing.assert_allclose(sol_2d, sol_1d, rtol=0, atol=0)
    finally:
        config.kernel_config = saved_kernel


def test_solve_fullsky_2d_phi0_per_pixel_matches_serial(mceq_sib21):
    """Per-pixel 2-D phi0 must match K independent 1-D ``solve_fullsky``
    calls (one per zenith with that pixel's phi0 column). This is the
    correctness test for the per-pixel cutoff use case.
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([10.0, 45.0, 70.0])
        K = zenith_grid.size
        rng = np.random.default_rng(20260523)
        phi0_base = mceq_sib21._phi0.copy()
        # Independent per-pixel phi0s: scale phi0_base by a random positive
        # factor and zero a different slice per pixel (simulates the cutoff
        # mask zeroing low-E bins per primary species per direction).
        phi0_2d = np.zeros((mceq_sib21.dim_states, K), dtype=np.float64)
        for k in range(K):
            scale = float(rng.uniform(0.5, 2.0))
            mask = np.ones(mceq_sib21.dim_states, dtype=np.float64)
            cut = int(rng.integers(0, mceq_sib21.dim_states // 4))
            mask[:cut] = 0.0
            phi0_2d[:, k] = scale * mask * phi0_base

        sol_2d, _ = mceq_sib21.solve_fullsky(zenith_grid, phi0=phi0_2d)

        for k in range(K):
            sol_k, _ = mceq_sib21.solve_fullsky(
                zenith_grid[k : k + 1], phi0=phi0_2d[:, k].copy()
            )
            np.testing.assert_allclose(sol_2d[:, k], sol_k[:, 0], rtol=1e-12, atol=0)
    finally:
        config.kernel_config = saved_kernel


def test_solve_fullsky_2d_phi0_carousel_K_invariant(mceq_sib21):
    """The LPT-carousel pipeline width (``carousel_K``) is a scheduling knob
    and must not change the per-pixel result: solving with carousel_K=1 and
    carousel_K=3 must agree bit-for-bit when phi0 is 2-D.
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
        K = zenith_grid.size
        rng = np.random.default_rng(20260524)
        phi0_base = mceq_sib21._phi0.copy()
        phi0_2d = (
            phi0_base[:, None]
            * rng.uniform(0.5, 2.0, size=K).astype(np.float64)[None, :]
        )

        sol_single, _ = mceq_sib21.solve_fullsky(
            zenith_grid, phi0=phi0_2d, carousel_K=1
        )
        sol_pipelined, _ = mceq_sib21.solve_fullsky(
            zenith_grid, phi0=phi0_2d, carousel_K=3
        )
        np.testing.assert_allclose(sol_pipelined, sol_single, rtol=1e-12, atol=0)
    finally:
        config.kernel_config = saved_kernel


def test_solve_fullsky_2d_phi0_shape_validation(mceq_sib21):
    """Reject 2-D phi0 with wrong first axis or wrong K."""
    zenith_grid = np.array([0.0, 30.0])
    dim = mceq_sib21.dim_states
    # Wrong first axis
    with pytest.raises(ValueError, match="first axis"):
        mceq_sib21.solve_fullsky(zenith_grid, phi0=np.zeros((dim - 1, 2)))
    # Wrong second axis (K mismatch)
    with pytest.raises(ValueError, match="second axis"):
        mceq_sib21.solve_fullsky(zenith_grid, phi0=np.zeros((dim, 3)))
    # 3-D phi0 rejected
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        mceq_sib21.solve_fullsky(zenith_grid, phi0=np.zeros((dim, 2, 1)))


# ---------------------------------------------------------------------------
# Cure B: EM-cascade adaptive step cap (config.em_adaptive_step). See the
# wiki lesson ``mceq-loss-averaging-grid-fragility``: the legacy dX_max=20
# over-integrates the stiff e±/γ cascade and biases the charged-shower X_max
# deep (up to +57 g/cm² in homogeneous media). The cap keys off the spectral
# radius of the EM off-diagonal block of int_m and refines the step
# automatically; muon/hadron-only solves (no e± loaded) are never throttled.
#
# These tests are DB-free: they drive the unbound MCEqRun methods on a small
# synthetic stub whose EM off-diagonal block has a known spectral radius, so
# they exercise the exact code path without the gigabyte interaction DB.
# ---------------------------------------------------------------------------

from MCEq.core import MCEqRun  # noqa: E402


class _StubParticle:
    def __init__(self, is_em, lidx, uidx):
        self.is_em, self.lidx, self.uidx = is_em, lidx, uidx


class _StubPMan:
    def __init__(self, particles):
        self.all_particles = particles


class _StubMCEq:
    """Minimal MCEqRun-like object exposing exactly what the cure-B helpers
    and _calculate_integration_path read: int_m, pman, density_model, and the
    path-cache attributes. EM block (indices 0,1) is a symmetric off-diagonal
    coupling [[0, r],[r, 0]] -> spectral radius == r_known."""

    # Borrow the real methods under test (unbound) so the stub exercises the
    # exact production code paths, including their internal self-calls.
    _em_cascade_step_scale = MCEqRun._em_cascade_step_scale
    _em_cascade_dx_cap = MCEqRun._em_cascade_dx_cap
    _calculate_integration_path = MCEqRun._calculate_integration_path

    def __init__(self, r_known=0.5, density_model=None):
        import scipy.sparse as sp

        m = -1.0 * np.eye(5)  # benign diagonal (removed by the off-split)
        m[0, 1] = m[1, 0] = r_known  # EM-EM off-diagonal block
        self.int_m = sp.csr_matrix(m)
        self.pman = _StubPMan(
            [
                _StubParticle(True, 0, 1),   # e- (EM)
                _StubParticle(True, 1, 2),   # e+ (EM)
                _StubParticle(False, 2, 3),  # hadron
                _StubParticle(False, 3, 4),
                _StubParticle(False, 4, 5),
            ]
        )
        self.density_model = density_model
        self.integration_path = None
        self.int_grid = None
        self.grid_var = None


def _const_density_target(max_X=600.0, rho=1e-3):
    """Homogeneous slab: zero density gradient, so the legacy schedule takes
    dX_max everywhere and the EM cap is the only thing that can refine."""
    from MCEq.geometry.density_profiles import GeneralizedTarget

    return GeneralizedTarget(len_target=max_X / rho, env_density=rho, env_name="air")


def test_em_cascade_step_scale_matches_spectral_radius():
    stub = _StubMCEq(r_known=0.5)
    r_em = MCEqRun._em_cascade_step_scale(stub)
    assert r_em == pytest.approx(0.5, rel=1e-6)
    # Cached on int_m identity.
    assert MCEqRun._em_cascade_step_scale(stub) == r_em


def test_em_cascade_step_scale_dense_for_large_nonnormal_block():
    """Regression: a large NON-NORMAL EM off-diagonal block must be reduced
    via the dense spectral radius, not sparse ``eigs``.

    The real EM block is strongly non-normal and ARPACK ``eigs(k=1)`` fails to
    converge on it; the old code then silently substituted ``min(||.||_1,
    ||.||_inf)`` — a 2-3x over-estimate that capped dX far too tight and was
    nondeterministic. Here the block is a big nilpotent forward-shift (spectral
    radius 0, both induced norms = 1) blocked with a small symmetric 2-cycle
    (spectral radius 0.3). True r_EM = 0.3; the norm fallback would return 1.0.
    The block dim (184) is well above the old hard-coded dense cutoff (32), so
    this exercises exactly the path that used to mis-fire.
    """
    import scipy.sparse as sp

    n_nil, dim = 180, 184
    A = -1.0 * np.eye(dim)  # benign diagonal, stripped by the off-split
    for i in range(n_nil - 1):
        A[i, i + 1] = 1.0  # nilpotent forward shift: rho 0, norms 1
    A[n_nil, n_nil + 1] = A[n_nil + 1, n_nil] = 0.3  # symmetric 2-cycle: rho 0.3

    stub = _StubMCEq.__new__(_StubMCEq)
    stub.int_m = sp.csr_matrix(A)
    stub.pman = _StubPMan(
        [_StubParticle(True, i, i + 1) for i in range(n_nil + 2)]
        + [_StubParticle(False, i, i + 1) for i in range(n_nil + 2, dim)]
    )
    stub._em_step_scale_cache = None

    r_em = MCEqRun._em_cascade_step_scale(stub)
    assert r_em == pytest.approx(0.3, abs=1e-6)  # dense spectral radius
    assert r_em < 0.5  # NOT the norm fallback (which would be ~1.0)
    # Deterministic across calls (ARPACK was not).
    stub._em_step_scale_cache = None
    assert MCEqRun._em_cascade_step_scale(stub) == pytest.approx(r_em, rel=1e-12)


def test_em_cascade_dx_cap_gating():
    stub = _StubMCEq(r_known=0.5)
    config.em_adaptive_step = False
    assert MCEqRun._em_cascade_dx_cap(stub) == np.inf

    config.em_adaptive_step = True
    config.em_step_safety = 0.04
    assert MCEqRun._em_cascade_dx_cap(stub) == pytest.approx(0.04 / 0.5)


def test_em_cascade_step_scale_zero_without_em():
    """No e±/γ loaded -> r_EM is 0 and the cap is inf, so non-EM (hadronic /
    muon) solves are never throttled even with the feature enabled."""
    stub = _StubMCEq()
    stub.pman = _StubPMan([_StubParticle(False, i, i + 1) for i in range(5)])
    stub._em_step_scale_cache = None
    assert MCEqRun._em_cascade_step_scale(stub) == 0.0

    config.em_adaptive_step = True
    assert MCEqRun._em_cascade_dx_cap(stub) == np.inf


def test_em_adaptive_step_refines_path():
    """In a homogeneous slab (zero density gradient) the cap is the only
    refiner. Feature ON must increase the step count; OFF reproduces the
    legacy schedule exactly."""
    grid = np.arange(0.0, 600.0 + 0.1, 50.0)
    dm = _const_density_target()

    config.em_adaptive_step = False
    stub_off = _StubMCEq(r_known=0.5, density_model=dm)
    MCEqRun._calculate_integration_path(stub_off, grid, "X", X_start=0.0)
    nsteps_off = stub_off.integration_path[0]

    config.em_adaptive_step = True
    config.em_step_safety = 0.04  # cap = 0.08 g/cm^2 << legacy dX_max=20
    stub_on = _StubMCEq(r_known=0.5, density_model=dm)
    MCEqRun._calculate_integration_path(stub_on, grid, "X", X_start=0.0)
    nsteps_on = stub_on.integration_path[0]

    assert nsteps_on > nsteps_off
    # Effective steps must respect the cap (allowing the snapshot-truncation
    # steps that land exactly on the coarse output grid).
    dX_on = stub_on.integration_path[1]
    assert dX_on.max() <= 0.08 + 1e-9


def test_em_adaptive_step_off_matches_legacy():
    """Feature OFF leaves the path identical to a build that never consulted
    the cap (same nsteps and dX array on repeat)."""
    grid = np.arange(0.0, 600.0 + 0.1, 50.0)
    dm = _const_density_target()
    config.em_adaptive_step = False

    a = _StubMCEq(r_known=0.5, density_model=dm)
    MCEqRun._calculate_integration_path(a, grid, "X", X_start=0.0)
    b = _StubMCEq(r_known=0.5, density_model=dm)
    MCEqRun._calculate_integration_path(b, grid, "X", X_start=0.0)

    assert a.integration_path[0] == b.integration_path[0]
    np.testing.assert_array_equal(a.integration_path[1], b.integration_path[1])


# ---------------------------------------------------------------------------
# solve_batch — unified batched entry point (shared-path + carousel routes)
# ---------------------------------------------------------------------------
def test_solve_batch_shared_matches_solve(mceq_sib21):
    """solve_batch(conditions=None) must reproduce K back-to-back solve()
    calls bit-for-bit (the shared-path multi-RHS kernel is bit-exact vs
    the single-RHS kernel on the CSR path — BSR reorders the SpMV
    partial sums, so it is disabled here), including int_grid snapshots,
    and must not mutate the instance solution state.
    """
    saved_kernel = config.kernel_config
    saved_bs = getattr(config, "numpy_bsr_blocksize", 11)

    def _clear_split_cache():
        for m in (mceq_sib21.int_m, mceq_sib21.dec_m):
            try:
                delattr(m, "_etd_split_cache_v2")
            except AttributeError:
                pass

    try:
        config.kernel_config = "numpy_etd2"
        config.numpy_bsr_blocksize = None
        _clear_split_cache()
        mceq_sib21.set_zenith_azimuth(30.0)
        int_grid = [100.0, 400.0, 900.0]
        phi0_base = mceq_sib21.get_initial_state()
        scales = [0.5, 1.0, 2.0]
        phi0_multi = np.stack([s * phi0_base for s in scales], axis=1)

        res = mceq_sib21.solve_batch(phi0_multi, int_grid=int_grid)

        # Legacy tuple unpacking
        sol, grid_sol = res
        assert sol is res.sol
        assert grid_sol is res.grid_sol
        assert sol.shape == (mceq_sib21.dim_states, 3)
        assert grid_sol.shape == (len(int_grid), mceq_sib21.dim_states, 3)
        assert np.all(res.nsteps_per_col == res.nsteps_per_col[0])

        saved_phi0 = mceq_sib21._phi0.copy()
        for k, s in enumerate(scales):
            mceq_sib21._phi0[:] = s * phi0_base
            mceq_sib21.solve(int_grid=int_grid)
            assert np.array_equal(res.sol[:, k], mceq_sib21._solution), (
                f"column {k} of solve_batch diverges from solve()"
            )
            assert np.array_equal(res.grid_sol[:, :, k], mceq_sib21.grid_sol), (
                f"column {k} snapshots diverge from solve()"
            )
        mceq_sib21._phi0[:] = saved_phi0
    finally:
        config.kernel_config = saved_kernel
        config.numpy_bsr_blocksize = saved_bs
        _clear_split_cache()
        mceq_sib21.set_zenith_azimuth(0.0)


def test_solve_batch_conditions_matches_fullsky(mceq_sib21):
    """A zenith-grid batch through explicit conditions must match
    solve_fullsky over the same grid (both run the LPT carousel on the
    same per-pixel paths).
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([0.0, 30.0, 60.0])
        conditions = [{"zenith_deg": float(z)} for z in zenith_grid]

        res_batch = mceq_sib21.solve_batch(conditions=conditions)
        res_sky, _ = mceq_sib21.solve_fullsky(zenith_grid)

        np.testing.assert_allclose(res_batch.sol, res_sky, rtol=0, atol=0)
        np.testing.assert_array_equal(
            res_batch.nsteps_per_col, res_batch.nsteps_per_col
        )
    finally:
        config.kernel_config = saved_kernel


def test_solve_batch_duplicate_conditions_use_shared_route(mceq_sib21):
    """Conditions that dedup to a single path must take the shared-path
    multi-RHS route and match the conditions=None result bit-for-bit
    (including grid snapshots being available... they are not requested
    here; final state only).
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        phi0_base = mceq_sib21.get_initial_state()
        phi0_multi = np.stack([phi0_base, 2.0 * phi0_base], axis=1)

        mceq_sib21.set_zenith_azimuth(45.0)
        res_none = mceq_sib21.solve_batch(phi0_multi)
        res_dup = mceq_sib21.solve_batch(
            phi0_multi,
            conditions=[{"zenith_deg": 45.0}, {"zenith_deg": 45.0}],
        )
        assert np.array_equal(res_none.sol, res_dup.sol)
        # Shared route is detectable through the legacy tuple layout:
        # (sol, grid_sol) instead of (sol, nsteps_per_col).
        assert res_dup._legacy[1] is res_dup.grid_sol
    finally:
        config.kernel_config = saved_kernel
        mceq_sib21.set_zenith_azimuth(0.0)


def test_solve_batch_density_model_override_matches_serial(mceq_sib21):
    """Per-condition density_model overrides must match serial
    set_density_model + solve() runs, and the instance's density model
    and angle must be restored afterwards.
    """
    saved_kernel = config.kernel_config
    dm_before = mceq_sib21.density_model
    try:
        config.kernel_config = "numpy_etd2"
        mceq_sib21.set_zenith_azimuth(20.0)
        theta_before = mceq_sib21.density_model.theta_deg

        seasons = [("CORSIKA", ("BK_USStd", None)),
                   ("CORSIKA", ("PL_SouthPole", "January"))]
        conditions = [
            {"zenith_deg": 60.0, "density_model": dm} for dm in seasons
        ]
        res = mceq_sib21.solve_batch(conditions=conditions)

        assert mceq_sib21.density_model is dm_before, (
            "density model not restored after solve_batch"
        )
        assert mceq_sib21.density_model.theta_deg == theta_before, (
            "zenith angle not restored after solve_batch"
        )

        saved_phi0 = mceq_sib21._phi0.copy()
        for k, dm in enumerate(seasons):
            mceq_sib21.set_density_model(dm)
            mceq_sib21.set_zenith_azimuth(60.0)
            mceq_sib21.solve()
            # rtol allows for BSR-vs-CSR partial-sum reordering between
            # the single-RHS solve() and the carousel SpMM. Typically
            # ~1e-12 on the e± blowup rows this fixture keeps enabled,
            # but the reordering is BLAS-dependent: macOS-Intel CI hit
            # 1.7e-10 on a single row, so keep two decades of headroom.
            np.testing.assert_allclose(
                res.sol[:, k], mceq_sib21._solution, rtol=1e-8, atol=0
            )
        mceq_sib21._phi0[:] = saved_phi0
    finally:
        config.kernel_config = saved_kernel
        mceq_sib21.set_density_model(dm_before)
        mceq_sib21.set_zenith_azimuth(0.0)


def test_solve_batch_int_grid_heterogeneous_raises(mceq_sib21):
    """int_grid snapshots are only supported on the shared-path route."""
    with pytest.raises(NotImplementedError, match="shared-path"):
        mceq_sib21.solve_batch(
            conditions=[{"zenith_deg": 0.0}, {"zenith_deg": 60.0}],
            int_grid=[100.0, 500.0],
        )


def test_solve_batch_phi0_shape_validation(mceq_sib21):
    """Shape errors carry the same phrasing as the solve_fullsky ones."""
    dim = mceq_sib21.dim_states
    with pytest.raises(ValueError, match="first axis"):
        mceq_sib21.solve_batch(np.zeros((dim - 1, 2)))
    with pytest.raises(ValueError, match="second axis"):
        mceq_sib21.solve_batch(
            np.zeros((dim, 3)), conditions=[{"zenith_deg": 0.0}] * 2
        )
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        mceq_sib21.solve_batch(np.zeros((dim, 2, 2)))
    with pytest.raises(ValueError, match="unknown keys"):
        mceq_sib21.solve_batch(conditions=[{"zenith": 0.0}])


def test_initial_state_builder(mceq_sib21):
    """initial_state() must reproduce the set_single_primary_particle /
    set_initial_spectrum vectors and leave the instance state untouched.
    """
    phi0_before = mceq_sib21.get_initial_state()
    restore_before = list(mceq_sib21._restore_initial_condition)

    # Single primary (proton, superposition path for a nucleus)
    col_p = mceq_sib21.initial_state({"E": 1e5, "pdg_id": 2212})
    col_fe = mceq_sib21.initial_state({"E": 1e6, "corsika_id": 5626})
    assert np.array_equal(mceq_sib21.get_initial_state(), phi0_before), (
        "initial_state mutated the instance phi0"
    )
    assert mceq_sib21._restore_initial_condition == restore_before

    mceq_sib21.set_single_primary_particle(1e5, pdg_id=2212)
    assert np.array_equal(col_p, mceq_sib21._phi0)
    mceq_sib21.set_single_primary_particle(1e6, corsika_id=5626)
    assert np.array_equal(col_fe, mceq_sib21._phi0)

    # Composition: two components == append chain
    col_both = mceq_sib21.initial_state(
        [{"E": 1e5, "pdg_id": 2212}, {"E": 1e6, "corsika_id": 5626}]
    )
    assert np.array_equal(col_both, col_p + col_fe)

    # Spectrum component
    spec = np.ones(mceq_sib21.dim) * 1e-8
    col_spec = mceq_sib21.initial_state({"spectrum": spec, "pdg_id": 2212})
    mceq_sib21.set_initial_spectrum(spec, pdg_id=2212)
    assert np.array_equal(col_spec, mceq_sib21._phi0)

    # Error cases
    with pytest.raises(ValueError, match="components must not be empty"):
        mceq_sib21.initial_state([])
    with pytest.raises(ValueError, match="unknown keys"):
        mceq_sib21.initial_state({"E": 1e5, "pdg": 2212})
    with pytest.raises(ValueError, match="each component needs"):
        mceq_sib21.initial_state({"corsika_id": 5626})

    # Restore the fixture's initial condition (H3a primary model)
    mceq_sib21._phi0[:] = phi0_before
    mceq_sib21._restore_initial_condition = restore_before


def test_batch_result_get_solution_matches_serial(mceq_sib21):
    """MCEqBatchResult.get_solution must agree with MCEqRun.get_solution
    after an equivalent serial solve, for plain, tracking-prefixed and
    magnified spectra, and for the zenith=/pixel= selectors.
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([15.0, 55.0])
        res = mceq_sib21.solve_fullsky(zenith_grid)

        for k, zen in enumerate(zenith_grid):
            mceq_sib21.set_zenith_azimuth(float(zen))
            mceq_sib21.solve()
            for pname, mag in [
                ("total_mu+", 0), ("conv_numu", 3), ("pr_antinumu", 0),
            ]:
                ref = mceq_sib21.get_solution(pname, mag=mag)
                got_k = res.get_solution(pname, k=k, mag=mag)
                got_zen = res.get_solution(pname, zenith=float(zen), mag=mag)
                got_pix = res.get_solution(pname, pixel=(k, 0), mag=mag)
                np.testing.assert_allclose(got_k, ref, rtol=1e-12, atol=0)
                np.testing.assert_array_equal(got_k, got_zen)
                np.testing.assert_array_equal(got_k, got_pix)

        # integrate= and return_as= passthrough
        mceq_sib21.set_zenith_azimuth(float(zenith_grid[-1]))
        mceq_sib21.solve()
        ref_int = mceq_sib21.get_solution("total_mu-", integrate=True)
        np.testing.assert_allclose(
            res.get_solution("total_mu-", k=1, integrate=True),
            ref_int, rtol=1e-12, atol=0,
        )

        # Selector errors
        with pytest.raises(ValueError, match="select one"):
            res.get_solution("total_mu+")
        with pytest.raises(ValueError, match="not in grid"):
            res.get_solution("total_mu+", zenith=33.0)
        with pytest.raises(IndexError):
            res.get_solution("total_mu+", k=5)
    finally:
        config.kernel_config = saved_kernel
        mceq_sib21.set_zenith_azimuth(0.0)


def test_batch_result_skymap(mceq_sib21):
    """skymap() at an exact grid energy equals the per-pixel extraction
    at that bin, with the (n_zen, n_az) layout.
    """
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([0.0, 45.0])
        azimuth_grid = np.array([0.0, 180.0])
        res = mceq_sib21.solve_fullsky(zenith_grid, azimuth_grid)

        e_idx = 30
        e_target = mceq_sib21.e_grid[e_idx]
        smap = res.skymap("total_numu", e_target)
        assert smap.shape == (2, 2)
        for i_zen in range(2):
            for i_az in range(2):
                ref = res.get_solution(
                    "total_numu", pixel=(i_zen, i_az),
                    return_as="kinetic energy",
                )[e_idx]
                np.testing.assert_allclose(
                    smap[i_zen, i_az], ref, rtol=1e-12, atol=0
                )

        # skymap on a non-fullsky result raises
        res_batch = mceq_sib21.solve_batch(
            conditions=[{"zenith_deg": 0.0}, {"zenith_deg": 45.0}]
        )
        with pytest.raises(ValueError, match="solve_fullsky results"):
            res_batch.skymap("total_numu", e_target)
    finally:
        config.kernel_config = saved_kernel


def test_solve_fullsky_2d_phi0_explicit_cutoff_warns(mceq_sib21):
    """Explicitly requesting the geomagnetic cutoff together with a 2-D
    phi0 must emit a warning (the cutoff is not applied on top)."""
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        zenith_grid = np.array([0.0, 30.0])
        phi0_2d = np.broadcast_to(
            mceq_sib21.get_initial_state()[:, None],
            (mceq_sib21.dim_states, 2),
        ).copy()
        with pytest.warns(UserWarning, match="NOT applied"):
            mceq_sib21.solve_fullsky(
                zenith_grid, phi0=phi0_2d, geomagnetic_cutoff=True
            )
    finally:
        config.kernel_config = saved_kernel


def test_solve_multirhs_alias_matches_solve_batch(mceq_sib21):
    """The deprecated solve_multirhs wrapper returns the raw
    (sol, grid_sol) pair of the equivalent solve_batch call."""
    saved_kernel = config.kernel_config
    try:
        config.kernel_config = "numpy_etd2"
        phi0_base = mceq_sib21.get_initial_state()
        phi0_multi = np.stack([phi0_base, 0.5 * phi0_base], axis=1)

        sol_a, grid_a = mceq_sib21.solve_multirhs(phi0_multi)
        res = mceq_sib21.solve_batch(phi0_multi)
        assert isinstance(sol_a, np.ndarray)
        assert np.array_equal(sol_a, res.sol)

        with pytest.raises(ValueError, match="must be 2-D"):
            mceq_sib21.solve_multirhs(phi0_base)
    finally:
        config.kernel_config = saved_kernel


# ---------------------------------------------------------------------------
# cuda fp32 pipeline — fp64-internal diagonal factors
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
def test_cuda_phi_compute_f64diag_accuracy():
    """The fp32-pipeline diag-factor kernel (fp32 buffers, fp64-internal
    arithmetic) must reproduce the fp64 phi factors to fp32 roundoff.

    The pure-fp32 phi1/phi2 cancellations ((e-1)/hd, (e-1-hd)/hd^2)
    lose 3-7 digits around the Taylor-switch thresholds; this test locks
    in the fp64-internal fix so a regression to fp32 arithmetic (or a
    threshold recalibration that reopens the cancellation band) fails
    loudly.
    """
    import cupy as cp

    from MCEq.solvers import _cuda_etd2_kernels

    Kset = _cuda_etd2_kernels()
    rng = np.random.default_rng(11)
    dim, K = 4096, 8
    # Diagonal rates and step sizes spanning the cancellation band
    # |hd| ~ 1e-6 .. 1e2, both signs, including near-threshold values.
    d_int = -np.abs(rng.lognormal(mean=-3.0, sigma=3.0, size=dim))
    d_dec = -np.abs(rng.lognormal(mean=-8.0, sigma=3.0, size=dim))
    h = rng.uniform(0.05, 15.0, (1, K))
    ri = rng.lognormal(mean=8.0, sigma=2.0, size=(1, K))

    args32 = [
        cp.asarray(a, dtype=cp.float32)
        for a in (d_int.reshape(dim, 1), d_dec.reshape(dim, 1), h, ri)
    ]
    outs_mixed = [cp.empty((dim, K), cp.float32) for _ in range(3)]
    Kset.phi_compute_multipath_f64diag(*args32, *outs_mixed)

    # fp64 reference from the same (fp32-quantised) inputs, so the
    # comparison isolates kernel arithmetic from input quantisation.
    args64 = [a.astype(cp.float64) for a in args32]
    outs64 = [cp.empty((dim, K), cp.float64) for _ in range(3)]
    Kset.phi_compute_multipath(*args64, *outs64)

    for name, mixed, ref in zip(
        ("eD", "phi1", "phi2"), outs_mixed, outs64
    ):
        m = cp.asnumpy(mixed).astype(np.float64)
        r = cp.asnumpy(ref)
        mask = np.abs(r) > 1e-15 * np.abs(r).max()
        rel = np.abs(m[mask] - r[mask]) / np.abs(r[mask])
        assert rel.max() < 5e-7, (
            f"{name}: fp64-internal diag kernel rel err {rel.max():.2e} "
            f"exceeds fp32-roundoff budget 5e-7"
        )
