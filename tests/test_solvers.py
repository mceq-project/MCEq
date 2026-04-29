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
