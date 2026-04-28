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


# ---------------------------------------------------------------------------
# ETD2 (numpy_etd2) tests
# ---------------------------------------------------------------------------
def test_solv_numpy_etd2_runs(toy_solver_problem):
    """ETD2 returns the right shape, no NaN, monotonic decay on the grid.

    The toy fixture has only diagonal int_m / dec_m, so ETD2 collapses to
    phi <- exp(h*D) * phi (no off-diagonal stages), which differs from
    forward-Euler at this step size. We don't compare values here — that is
    covered by the full-fixture tests below.
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


@pytest.mark.parametrize("theta_deg", [0.0, 60.0])
def test_solv_numpy_etd2_matches_euler_full_path(mceq_sib21, theta_deg):
    """ETD2 and forward-Euler should agree on the muon spectrum at the
    native step grid for moderate zenith angles.

    Tolerance is loose (2%) because ETD2 is genuinely more accurate than
    Euler — the two schemes don't converge to each other, they both
    converge to the true solution. The 1-2% gap is the known first-order
    error of Euler at the native grid (see docs/etd1_solver.md).
    """
    mceq_sib21.set_theta_deg(theta_deg)

    phi_eul = _solve_with_kernel(mceq_sib21, "numpy")
    phi_etd = _solve_with_kernel(mceq_sib21, "numpy_etd2")

    assert np.all(np.isfinite(phi_etd)), "ETD2 produced non-finite values"

    # Per-species relative-L2 on the major fluxes
    for name in ("mu+", "mu-", "numu", "antinumu", "nue"):
        sl = mceq_sib21.pman[name].lidx, mceq_sib21.pman[name].uidx
        a = phi_eul[sl[0] : sl[1]]
        b = phi_etd[sl[0] : sl[1]]
        denom = np.linalg.norm(a)
        if denom == 0:
            continue
        rel = np.linalg.norm(a - b) / denom
        assert rel < 0.05, f"{name} rel-L2 = {rel:.3e} exceeds 5% at theta={theta_deg}"

    # Headline physics quantity: E^3 * dN/dE for muons above 1 GeV
    e, mu_eul = _muon_flux(mceq_sib21, phi_eul)
    _, mu_etd = _muon_flux(mceq_sib21, phi_etd)
    band = (e > 1.0) & (mu_eul > 1e-30)
    max_rel = np.max(np.abs(mu_etd[band] - mu_eul[band]) / mu_eul[band])
    assert max_rel < 0.02, (
        f"max muon-flux rel diff = {max_rel:.3e} exceeds 2% at theta={theta_deg}"
    )


def test_solv_numpy_etd2_stable_at_high_zenith():
    """Regression: ETD2 must stay finite and within tolerance at theta=89 deg.

    At extreme zenith, rho_inv blows up and forward-Euler-style schemes
    explode on rows with weak diagonal damping. The ETD2 design treats the
    diagonal exactly via an integrating factor; we verify it doesn't
    regress that property.

    e+/e- are disabled because their semi-Lagrangian L/R-variant rows have
    no diagonal damping and require a block-ETD generalization (see
    docs/etd1_solver.md). This is a known limitation, not a regression.
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
        phi_eul = _solve_with_kernel(mceq, "numpy")
        phi_etd = _solve_with_kernel(mceq, "numpy_etd2")

        assert np.all(np.isfinite(phi_etd)), "ETD2 blew up at theta=89 deg"

        e, mu_eul = _muon_flux(mceq, phi_eul)
        _, mu_etd = _muon_flux(mceq, phi_etd)
        band = (e > 1.0) & (mu_eul > 1e-30)
        assert band.any(), "no nonzero muon-flux band found"
        max_rel = np.max(np.abs(mu_etd[band] - mu_eul[band]) / mu_eul[band])
        assert max_rel < 0.05, (
            f"max muon-flux rel diff = {max_rel:.3e} exceeds 5% at theta=89 deg"
        )
    finally:
        config.adv_set["disabled_particles"] = saved
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db


def _euler_oversampled(int_m, dec_m, phi0, dX, rho_inv, oversample):
    """Forward-Euler with each native step subdivided into `oversample`
    substeps; rho_inv held constant within the native step. Used to build
    a high-fidelity reference for the convergence test."""
    phi = phi0.astype(np.float64).copy()
    for k in range(len(dX)):
        h = dX[k] / oversample
        ri = rho_inv[k]
        for _ in range(oversample):
            phi = phi + h * (int_m.dot(phi) + ri * dec_m.dot(phi))
    return phi


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
    config.kernel_config = "numpy"
    try:
        mceq_sib21.integration_path = None
        mceq_sib21._calculate_integration_path(int_grid=None, grid_var="X")
        nsteps_full, dX_full, rho_inv_full, _ = mceq_sib21.integration_path
    finally:
        config.kernel_config = saved_kernel

    # Use the full path; the early steps barely accumulate dynamics and
    # would put ETD2 in the floating-point-noise floor on a short slice.
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
