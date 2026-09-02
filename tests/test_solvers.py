import numpy as np
import pytest

from MCEq import config


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_spacc_matrix_creation(toy_solver_problem, dtype):
    """SpaccMatrix should be created from a scipy sparse matrix without error.

    Both entry-point families of ``MCEq.spacc._SPACC_TYPES``: the mstore slot
    is typed at creation, so fp32 goes through ``create_sparse_matrix_f32``.
    """
    import MCEq.spacc as spacc

    int_m = toy_solver_problem[3]
    sm = spacc.SpaccMatrix(int_m, dtype=dtype)
    assert sm.store_id is not None
    assert sm.store_id >= 0
    assert sm.dim_rows == int_m.shape[0]
    assert sm.dim_cols == int_m.shape[1]
    assert sm.nnz == int_m.nnz
    assert sm.dtype == np.dtype(dtype)
    assert sm.data.dtype == np.dtype(dtype)
    sm.close()


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize(("dtype", "rtol"), [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_spacc_gemv_matches_scipy(toy_solver_problem, dtype, rtol):
    """gemv_npargs should produce the same result as scipy sparse dot."""
    import MCEq.spacc as spacc

    int_m = toy_solver_problem[3]
    sm = spacc.SpaccMatrix(int_m, dtype=dtype)

    size = int_m.shape[0]
    x = np.ones(size, dtype=dtype)
    y = np.zeros(size, dtype=dtype)
    alpha = 2.0

    sm.gemv_npargs(alpha, x, y)

    expected = alpha * int_m.dot(np.ones(size))
    assert np.allclose(y, expected, rtol=rtol), (
        f"gemv result {y} does not match scipy result {expected}"
    )
    sm.close()


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
def test_step_formula_has_one_source():
    """The C kernels spell the step out of ``numerics``' table, not their own.

    ``numerics.PREDICTOR_EXPR`` / ``CORRECTOR_EXPR`` fix the association
    order every backend holds to. The cupy kernels format the strings into
    their bodies at build time and cannot drift; ``etd2_kernels.c`` carries
    them verbatim as its two macros, and this is what stops that copy from
    being edited on its own.
    """
    from pathlib import Path

    import MCEq
    from MCEq.solvers.numerics import CORRECTOR_EXPR, PREDICTOR_EXPR

    source = Path(MCEq.__file__).parent / "etd2_kernels" / "etd2_kernels.c"
    if not source.exists():  # installed as a wheel: only the .so is shipped
        pytest.skip(f"{source} is not in this installation")
    text = source.read_text()
    for macro, expr in (
        ("ETD2_PREDICT", PREDICTOR_EXPR),
        ("ETD2_CORRECT", CORRECTOR_EXPR),
    ):
        assert expr in text, (
            f"{source.name} does not carry numerics' {macro} expression "
            f"{expr!r}; the C kernels and the table have diverged"
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize(
    ("dim", "K", "per_lane"),
    [(4096, 1, 0), (513, 8, 0), (513, 8, 1), (97, 3, 1)],
    ids=["k1", "shared", "per_lane", "small"],
)
def test_c_stages_match_numpy_lowering(dim, K, per_lane, dtype):
    """The compiled stages and the numpy lowering agree to the bit.

    Both lower the same two expressions of ``numerics``, so "one association
    order" holds only if the *compiled* code keeps it — which reading the
    source cannot show. Contracting ``a * b + c`` into an FMA is how this
    breaks: the same association, one rounding fewer, and it is what a
    compiler does by default wherever FMA is baseline (aarch64, or
    ``-march=native`` on x86-64). ``CMakeLists.txt`` passes
    ``-ffp-contract=off`` to prevent it; this notices if that stops working.

    Arguments span 18 decades so the two products land at different
    exponents, where a contracted multiply-add differs from a rounded one.
    """
    from MCEq.solvers import numerics
    from MCEq.solvers.backends.host import _C_POINTER, _fused_stages

    stages = _fused_stages(dtype)
    if stages is None:
        pytest.skip("MCEq.etd2_kernels is not built")
    ptr = _C_POINTER[dtype]
    rng = np.random.default_rng(7)
    shape, fshape = (dim, K), ((dim, K) if per_lane else (dim, 1))

    def sample(shp):
        return (
            rng.uniform(0.5, 2.0, size=shp) * 10.0 ** rng.uniform(-17, 1, shp)
        ).astype(dtype)

    eD, hphi1, hphi2 = (sample(fshape) for _ in range(3))
    x, F, F_a, a = (sample(shape) for _ in range(4))
    work = np.empty(shape, dtype=dtype)

    # The two lowerings take their operands in different orders; both are
    # spelled out here so a change to either signature fails loudly.
    for name, c_stage, c_args, py_stage, py_args in (
        (
            "predictor",
            stages[0],
            (eD, hphi1, x, F),
            numerics.predictor,
            (eD, x, hphi1, F),
        ),
        (
            "corrector",
            stages[1],
            (hphi2, a, F_a, F),
            numerics.corrector,
            (a, hphi2, F_a, F),
        ),
    ):
        got, want = (np.empty(shape, dtype=dtype) for _ in range(2))
        c_stage(dim, K, per_lane, *[b.ctypes.data_as(ptr) for b in (*c_args, got)])
        py_stage(*py_args, want, work)
        bad = int(np.count_nonzero(got.view(np.uint8) != want.view(np.uint8)))
        assert bad == 0, (
            f"{name}: C and numpy lowerings differ in {bad} bytes at dim={dim} "
            f"K={K} per_lane={per_lane} {np.dtype(dtype).name}; max rel "
            f"{np.max(np.abs(got - want) / np.where(want != 0, np.abs(want), 1)):.3e}"
        )


@pytest.mark.parametrize("K", [1, 4], ids=["k1", "multirhs"])
def test_host_solves_without_the_c_extension(toy_solver_problem, K):
    """An unbuilt source tree falls back to numpy and gets the same answer.

    ``MCEq.etd2_kernels`` is a compiled extension; before the step stages
    moved into it the host backend was pure numpy, and a tree that has not
    been built must not lose the solver entirely. Runs the same solve twice
    in subprocesses, once with the extension blocked at import, and requires
    the two to agree to the bit -- the fallback lowers the same expressions
    from the same table.
    """
    import subprocess
    import sys

    script = """
import sys, numpy as np, scipy.sparse as sp
if {block!r}:
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name == "MCEq.etd2_kernels":
                raise ImportError("extension not built")
            return None
    sys.meta_path.insert(0, Block())
    from MCEq.solvers.backends.host import _fused_stages
    assert _fused_stages(np.float64) is None, "the extension was still importable"
import MCEq.solvers as solvers
rng = np.random.default_rng(3)
n = 40
A = rng.standard_normal((n, n)) * 0.05
A -= np.diag(np.abs(A).sum(1) + 0.1)
B = rng.standard_normal((n, n)) * 0.02
B -= np.diag(np.abs(B).sum(1) + 0.05)
sol, _ = solvers.solve_etd2(
    nsteps=25, dX=np.full(25, 0.1), rho_inv=np.linspace(1.3, 2.0, 25),
    int_m=sp.csr_matrix(A), dec_m=sp.csr_matrix(B),
    phi=rng.uniform(0.1, 1.0, (n, {K})), grid_idcs=[], backend="numpy",
)
sys.stdout.buffer.write(np.ascontiguousarray(sol, dtype=np.float64).tobytes())
"""
    out = []
    for block in (False, True):
        run = subprocess.run(
            [sys.executable, "-c", script.format(block=block, K=K)],
            capture_output=True,
        )
        assert run.returncode == 0, run.stderr.decode()[-2000:]
        out.append(np.frombuffer(run.stdout, dtype=np.float64))

    assert out[0].size == 40 * K
    assert np.all(np.isfinite(out[1]))
    assert np.array_equal(out[0], out[1]), (
        "the numpy fallback does not reproduce the C stages bitwise; max rel "
        f"{np.max(np.abs(out[1] - out[0]) / np.abs(out[0])):.3e}"
    )


def test_solve_etd2_numpy_runs(toy_solver_problem):
    """ETD2 returns the right shape, no NaN, monotonic decay on the grid.

    The toy fixture has only diagonal int_m / dec_m, so ETD2 collapses to
    phi <- exp(h*D) * phi (no off-diagonal stages). We don't compare against
    a reference here — full-fixture equivalence is covered by the
    accelerate-vs-numpy tests below.
    """
    from MCEq.solvers import solve_etd2

    phi0 = toy_solver_problem[-2].copy()
    grid_idcs = toy_solver_problem[-1]

    solution, grid_sol = solve_etd2(*toy_solver_problem, backend="numpy")
    assert solution.shape == phi0.shape
    assert grid_sol.shape == (len(grid_idcs), phi0.shape[0])
    assert not np.isnan(solution).any()
    assert np.all(np.isfinite(solution))

    for i in range(1, grid_sol.shape[0]):
        assert np.all(grid_sol[i] <= grid_sol[i - 1])


def test_solve_etd2_numpy_does_not_modify_input_phi(toy_solver_setup):
    """Regression: ETD2 must not mutate the input phi array in place."""
    from MCEq.solvers import solve_etd2

    phi_original = toy_solver_setup[-2]
    phi_copy = phi_original.copy()

    solution, _ = solve_etd2(*toy_solver_setup, backend="numpy")

    assert np.array_equal(phi_original, phi_copy), (
        "solve_etd2 modified the input phi array - this breaks subsequent solver calls"
    )
    assert not np.array_equal(solution, phi_copy), (
        "Solver should produce a different result"
    )


@pytest.mark.parametrize("K", [1, 4, 16])
def test_solve_etd2_numpy_multirhs_matches_single_rhs_toy(K):
    """Multi-RHS ETD2 columns match K independent single-RHS solves bit-exactly.

    scipy's CSR ``@`` against a 2-D (n, K) RHS issues per-column SpMVs with
    the same arithmetic as the single-RHS path, so the per-column result is
    arithmetically identical to the single-RHS solve.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    sol_multi, grid_multi = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs, backend="numpy"
    )
    assert sol_multi.shape == (size, K)
    assert grid_multi.shape == (len(grid_idcs), size, K)

    for k in range(K):
        sol_k, grid_k = solve_etd2(
            nsteps,
            dX,
            rho_inv,
            int_m,
            dec_m,
            phi0_multi[:, k].copy(),
            grid_idcs,
            backend="numpy",
        )
        assert np.array_equal(sol_multi[:, k], sol_k), (
            f"column {k} of multi-RHS solution diverges from single-RHS"
        )
        assert np.array_equal(grid_multi[:, :, k], grid_k), (
            f"column {k} of multi-RHS grid snapshots diverges from single-RHS"
        )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solve_multirhs_dtype_float32():
    """End-to-end fp32 dispatch through MCEqRun.solve_multirhs.

    Compares the fp32 Accelerate multi-RHS path to the fp64 reference at K=4
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


def test_solve_etd2_rank_follows_phi():
    """The solution has the rank of ``phi``: one entry point serves both.

    A 1-D state gives a 1-D solution and a ``(dim, 1)`` batch gives the same
    values as a column, so the batch route needs no name of its own and
    ``solve_etd2`` refuses neither rank. The 2-D requirement that a caller
    asking for a batch does need lives at the API boundary
    (``MCEqRun.solve_multirhs``), which guards it.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

    nsteps = 3
    size = 4
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    grid_idcs = [1]
    int_m = sp.csr_matrix(-0.1 * np.eye(size))
    dec_m = sp.csr_matrix(-0.05 * np.eye(size))
    phi0_1d = np.linspace(0.2, 1.0, size)

    sol_1d, grid_1d = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_1d, grid_idcs, backend="numpy"
    )
    assert sol_1d.shape == (size,)
    assert grid_1d.shape == (len(grid_idcs), size)

    sol_2d, grid_2d = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        int_m,
        dec_m,
        phi0_1d[:, None].copy(),
        grid_idcs,
        backend="numpy",
    )
    assert sol_2d.shape == (size, 1)
    assert np.array_equal(sol_2d[:, 0], sol_1d)
    assert np.array_equal(grid_2d[:, :, 0], grid_1d)


def test_solve_etd2_rejects_unknown_backend():
    """An unknown ``backend`` names the ones that exist."""
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

    int_m = sp.csr_matrix(-0.1 * np.eye(3))
    with pytest.raises(ValueError, match="accelerate, cuda, mkl, numpy"):
        solve_etd2(
            1, np.ones(1), np.ones(1), int_m, int_m, np.ones(3), [], backend="opencl"
        )


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.parametrize("K", [1, 4, 16])
def test_solve_etd2_mkl_multirhs_matches_numpy_multirhs_toy(K):
    """MKL multi-RHS columns match numpy multi-RHS columns within fp64 ε."""
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, [], backend="numpy"
    )
    sol_mkl, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, [], backend="mkl"
    )
    np.testing.assert_allclose(sol_mkl, sol_numpy, rtol=5e-13, atol=0)


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("backend", ["numpy", "mkl"])
def test_solve_etd2_fp32_matches_numpy_multirhs_toy(backend, K):
    """The host backends at fp32 hold 1e-4 rel-L2 vs the fp64 reference."""
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, [], backend="numpy"
    )
    sol_f32, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        int_m,
        dec_m,
        phi0_multi,
        [],
        backend=backend,
        fp_precision=32,
    )
    assert sol_f32.dtype == np.float64  # the driver hands back fp64
    rel_l2 = np.linalg.norm(sol_f32 - sol_numpy) / max(np.linalg.norm(sol_numpy), 1e-30)
    assert rel_l2 < 1e-4, f"{backend} fp32 (K={K}) vs numpy fp64 rel-L2 = {rel_l2:.3e}"


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
@pytest.mark.parametrize("K", [1, 4, 16, 70])
def test_solve_etd2_accelerate_multirhs_matches_numpy_multirhs_toy(K):
    """Accelerate multi-RHS columns match numpy multi-RHS columns within fp64 eps.

    Both runs are the same driver over the same operator; the only
    difference is the ``apply_off`` binding (scipy CSR vs Apple Accelerate
    ``sparse_matrix_product_dense_double`` on column-major staging).
    Differences are at the few ULP level and the test uses np.allclose with
    a tight tolerance. K = 70 crosses the 64-column SpMM tile boundary.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs, backend="numpy"
    )
    sol_spacc, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs, backend="accelerate"
    )

    np.testing.assert_allclose(sol_spacc, sol_numpy, rtol=5e-13, atol=0)


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
@pytest.mark.parametrize("K", [1, 4, 16])
def test_solve_etd2_cuda_multirhs_matches_numpy_multirhs_toy(K):
    """cupy multi-RHS columns match numpy multi-RHS columns within
    cuSPARSE-reorder tolerance.

    cuSPARSE reorders partial sums (warp-reduction order differs from
    scipy's row-major accumulation), so we tolerate a relative L2 of 1e-10
    rather than round-off. The single-RHS cuda test uses the same bound.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, grid_numpy = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, grid_idcs, backend="numpy"
    )
    sol_cuda, grid_cuda = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        int_m,
        dec_m,
        phi0_multi,
        grid_idcs,
        backend="cuda",
        device_id=config.cuda_gpu_id,
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
        assert rel_grid < 1e-10, f"cuda multirhs grid snapshots rel-L2 = {rel_grid:.3e}"


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
@pytest.mark.parametrize("K", [1, 4])
def test_solve_etd2_cuda_multirhs_f32_matches_numpy_multirhs_toy(K):
    """fp32 cupy multi-RHS holds 1e-4 rel-L2 vs the fp64 numpy reference.

    Per the multi-RHS handover plan: fp32 stability budget is 1e-4 relative
    error (verified against per-particle MCEq SIBYLL21 fluxes on Mac
    Accelerate; same arithmetic carries to cupy by construction).
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

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

    phi0_multi = rng.uniform(0.1, 1.0, size=(size, K))

    sol_numpy, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_multi, [], backend="numpy"
    )
    sol_cuda32, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        int_m,
        dec_m,
        phi0_multi,
        [],
        backend="cuda",
        device_id=config.cuda_gpu_id,
        fp_precision=32,
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


def test_etd2_numpy_stable_at_high_zenith():
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
    from MCEq.operator_assembly import split_diagonal

    d_int, d_dec, int_off, dec_off = split_diagonal(int_m, dec_m)
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


def test_solve_etd2_numpy_second_order_convergence():
    """ETD2 must show observed convergence order ~2 on the production path.

    Refinement is done the way production accuracy actually depends on it:
    the non-uniform path is *rebuilt* at successively smaller ``dX_max`` and
    ``eps``, so ``rho_inv`` is re-derived (integral mean per step) at every
    refinement level.  Refining inside frozen native steps instead — what
    ``_etd2_oversampled`` does — is a strictly weaker check.

    The error norm excludes the EM rows (gamma, e+/e-), and that exclusion
    is the whole point of this test.  Measured on this fixture 2026-07-29,
    production path, reference at 128x refinement:

        rows            err @ default     order (1->2, 2->4, 4->8)
        all             2.26e-02          0.708  0.877  0.979
        EM only         2.27e-02          0.708  0.877  0.979
        non-EM          5.95e-03          2.065  2.121  2.030

    i.e. the hadronic/leptonic system is cleanly second order (confirmed at
    theta = 0, 60 and 89 deg; muon, numu and proton fluxes converge at
    1.87-1.93 in relative terms), while the EM rows converge at first order
    and carry essentially the *entire* whole-state error.  That is the
    documented ETD2 EM caveat — the semi-Lagrangian e+/e- and gamma rows
    have no diagonal damping (see docs/mceq_v1.x_v2_diff.md).  Any all-rows
    norm therefore reads order ~1 regardless of the scheme, which is why
    this assertion must be taken over the non-EM block.

    History: this test used to measure an all-rows norm on the production
    path with the ``mceq_sib21`` fixture, which re-enables e+/e-.  The
    coarsest solve then *diverged* (err ~1e8), so log2(err_h/err_h2) came
    out ~33 and the ``>= 1.8`` floor was vacuous.  Whether it diverged was
    round-off sensitive: on macos-15-intel/3.14 it stayed finite and the
    honest all-rows ratio (1.04) surfaced as a CI failure.  The stability
    bound below now rejects a divergence instead of reading it as high order.
    """
    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    saved_kernel = config.kernel_config
    saved_db = config.mceq_db_fname
    config.adv_set["disabled_particles"] = [11, -11]
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        import crflux.models as pm

        from MCEq.core import MCEqRun
        from MCEq.solvers import solve_etd2

        config.kernel_config = "numpy_etd2"
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )

        dX_max_0 = config.etd2_path["dX_max"]
        eps_0 = config.etd2_path["eps"]
        int_m = mceq.int_m.tocsr()
        dec_m = mceq.dec_m.tocsr()

        def solve_refined(refine):
            """Production solve on a path refined by `refine` in dX and eps."""
            mceq._calculate_integration_path(
                None,
                "X",
                force=True,
                dX_max=dX_max_0 / refine,
                eps=eps_0 / refine,
                dX_min=1e-10,
            )
            nsteps, dX, rho_inv, _ = mceq.integration_path
            phi, _ = solve_etd2(
                nsteps,
                dX,
                rho_inv,
                int_m,
                dec_m,
                mceq._phi0.copy(),
                [],
                backend="numpy",
            )
            return phi

        # Mask the EM block out of the error norm (see docstring).
        em_rows = np.zeros(mceq.dim_states, dtype=bool)
        for pdg in (22, 11, -11):
            try:
                part = mceq.pman[pdg]
            except (KeyError, AttributeError):
                continue
            em_rows[part.lidx : part.uidx] = True
        assert em_rows.any(), "expected gamma/e+- rows in the state vector"
        keep = ~em_rows

        # Reference at 32x refinement: 8x beyond the finest test point, so
        # its own O(h^2) residual sits ~64x below it.
        phi_truth = solve_refined(32)
        norm_truth = np.linalg.norm(phi_truth[keep])
        assert norm_truth > 0

        errs = {}
        for refine in (1, 2, 4):
            phi = solve_refined(refine)
            errs[refine] = np.linalg.norm((phi - phi_truth)[keep]) / norm_truth

        # Stability guard: the coarsest solve must be a small perturbation of
        # the reference, not a divergence. Without this a blowup inflates the
        # error ratio and masquerades as high order.
        assert np.isfinite(errs[1]) and errs[1] < 1e-1, (
            f"ETD2 at the default path is not in a perturbative regime "
            f"(err={errs[1]:.3e}) — the order ratios below would be meaningless"
        )
        assert errs[1] > 1e-10, (
            f"ETD2 error {errs[1]:.3e} too small to measure order — "
            "test is in floating-point-noise regime"
        )
        assert errs[4] < errs[2] < errs[1], (
            "ETD2 error did not decrease monotonically under h-refinement: "
            f"{errs[1]:.3e} -> {errs[2]:.3e} -> {errs[4]:.3e}"
        )

        # Measures 2.07 and 2.14 on this fixture.
        for coarse, fine in ((1, 2), (2, 4)):
            order = np.log2(errs[coarse] / errs[fine])
            assert order >= 1.8, (
                f"ETD2 observed order {order:.2f} below 1.8 between refine="
                f"{coarse} and refine={fine} (err={errs[coarse]:.3e} -> "
                f"{errs[fine]:.3e})"
            )
    finally:
        config.adv_set["disabled_particles"] = saved_disabled
        config.kernel_config = saved_kernel
        config.mceq_db_fname = saved_db


# ---------------------------------------------------------------------------
# ETD2 (Apple Accelerate) tests
# ---------------------------------------------------------------------------
@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solve_etd2_accelerate_matches_numpy_etd2_toy(toy_solver_problem):
    """Trivial-matrix smoke test.

    The toy fixture has purely diagonal int_m/dec_m, so both off-diagonals
    are empty (nnz=0). The binding should detect that and skip the SpMV
    calls; the result is just the integrating-factor `exp(h*D) * phi` per
    step. This catches the empty-matrix code path without requiring a
    full MCEqRun.
    """
    from MCEq.solvers import solve_etd2

    nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs = toy_solver_problem

    sol_numpy, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi.copy(), grid_idcs, backend="numpy"
    )
    sol_spacc, _ = solve_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi.copy(), grid_idcs, backend="accelerate"
    )
    assert sol_spacc == pytest.approx(sol_numpy, rel=1e-12, abs=1e-15), (
        "accelerate differs from numpy on toy diagonal-only problem"
    )


@pytest.mark.xdist_group("spacc")
@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_solve_etd2_accelerate_matches_numpy_etd2_real(mceq_sib21):
    """Equivalence test on real MCEq matrices with non-trivial off-diagonals.

    Builds a uniform mid-point sampled path at theta=60 with the default
    (no-mixing) particle treatment, runs both backends on that fixed path,
    asserts agreement to ~1e-12 — the 4 SpMVs/step are the same operation
    in both backends, so equality is essentially arithmetic round-off.
    """
    from MCEq.solvers import solve_etd2

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

    sol_numpy, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq_sib21.int_m,
        mceq_sib21.dec_m,
        phi0.copy(),
        grid_idcs,
        backend="numpy",
    )

    from MCEq.operator_assembly import split_diagonal

    _, _, int_off, dec_off = split_diagonal(mceq_sib21.int_m, mceq_sib21.dec_m)
    assert int_off.nnz > 0 and dec_off.nnz > 0, (
        "real matrices should have non-empty off-diagonals"
    )
    sol_spacc, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq_sib21.int_m,
        mceq_sib21.dec_m,
        phi0.copy(),
        grid_idcs,
        backend="accelerate",
    )

    assert np.all(np.isfinite(sol_spacc)), "accelerate produced non-finite values"
    rel_l2 = np.linalg.norm(sol_spacc - sol_numpy) / max(
        np.linalg.norm(sol_numpy), 1e-30
    )
    assert rel_l2 < 1e-12, (
        f"accelerate vs numpy rel-L2 = {rel_l2:.3e} (expected < 1e-12)"
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
def test_etd2_mkl_matches_numpy_real(mceq_sib21_full_db):
    """Equivalence test on real MCEq matrices, both CSR and BSR(6) paths.

    CSR is bit-exact vs numpy (~1e-12); BSR reorders the SpMV partial
    sums per-block and lands at ~1e-10 on these matrices — still
    essentially round-off, just looser.
    """
    from MCEq.solvers import (
        compile_operator,
        etd2_driver,
        mkl_backend,
        solve_etd2,
        split_diagonal,
    )

    mceq = mceq_sib21_full_db
    nsteps, dX, rho_inv = _uniform_path_theta60(mceq)
    grid_idcs = []
    phi0 = mceq._phi0.copy()

    sol_numpy, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq.int_m,
        mceq.dec_m,
        phi0.copy(),
        grid_idcs,
        backend="numpy",
    )

    _, _, int_off, dec_off = split_diagonal(mceq.int_m, mceq.dec_m)
    assert int_off.nnz > 0 and dec_off.nnz > 0, (
        "real matrices should have non-empty off-diagonals"
    )
    # The storage of the MKL handles is a backend-factory option, not an
    # argument of the entry point: bind the backend here and run the driver
    be = mkl_backend(compile_operator(mceq.int_m, mceq.dec_m))
    try:
        sol_mkl, _ = etd2_driver(nsteps, dX, rho_inv, be, phi0.copy(), grid_idcs)
    finally:
        be.close()

    assert np.all(np.isfinite(sol_mkl)), "mkl_etd2 produced non-finite values"
    rel_l2 = np.linalg.norm(sol_mkl - sol_numpy) / max(np.linalg.norm(sol_numpy), 1e-30)
    # CSR is the same arithmetic as numpy → bit-exact bound. BSR groups
    # the SpMV per block, which reorders partial sums and loosens to ~1e-10.
    tol = 1e-12
    assert rel_l2 < tol, (
        f"mkl_etd2 vs numpy_etd2 rel-L2 = {rel_l2:.3e} (expected < {tol})"
    )


def _per_species_max_rel(mceq, ref, new, floor=1e-12):
    """The D18 metric: per species, the max ``|dphi / phi_ref|`` over the
    energy bins, dropping bins below ``floor`` x that species' peak and
    ``phi_ref == 0``.

    Species blocks are ``[mceqidx * dim : (mceqidx + 1) * dim]``, the slice
    ``get_solution`` reads. ``nan`` for a species with no bin above the floor.
    """
    dim = mceq.dim
    out = {}
    for q in mceq.pman.cascade_particles:
        r = ref[q.mceqidx * dim : (q.mceqidx + 1) * dim]
        n = new[q.mceqidx * dim : (q.mceqidx + 1) * dim]
        peak = r.max()
        keep = (
            (r >= floor * peak) & (r != 0.0)
            if peak > 0.0
            else np.zeros(r.shape, dtype=bool)
        )
        out[q.name] = (
            float(np.max(np.abs(n[keep] - r[keep]) / np.abs(r[keep])))
            if keep.any()
            else np.nan
        )
    return out


@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
def test_etd2_fp32_mkl_vs_cuda_per_species_real(mceq_sib21_full_db):
    """fp32 MKL vs fp32 CUDA on the real operator, per species (D18, D22).

    Two uniform theta=60 paths over one operator and initial state: ``h = 5``
    is inside the explicit-stepping stiffness limit of the EM block, ``h = 20``
    (the production ``etd2_path["dX_max"]``) is past it with
    ``config.em_adaptive_step`` off. Outside e+- the metric is a stable
    property of the operator — fp32 roundoff in the sparse product and the
    state buffers, 6e-6 to 1e-5 measured across both steps and both energy
    grids the shared fixture is built on.

    The e+- lanes are not such a property, which is what excluding them
    buys: over-integrated, they are a sign-oscillating residual already at
    fp64, so the metric either leaves the budget (1e-3) or has no bin left
    above its floor (a reference block with no positive bin). Their value on
    the fine step is reported, not bounded — it moves from 1.1e-5 to 2.3e-3
    with the energy grid alone.
    """
    from MCEq.solvers import solve_etd2

    mceq = mceq_sib21_full_db
    em = {q.name for q in mceq.pman.cascade_particles if abs(q.pdg_id[0]) == 11}
    phi0 = mceq._phi0.copy()

    def run(path, backend):
        nsteps, dX, rho_inv = path
        sol, _ = solve_etd2(
            nsteps,
            dX,
            rho_inv,
            mceq.int_m,
            mceq.dec_m,
            phi0.copy(),
            [],
            backend=backend,
            fp_precision=32,
            device_id=config.cuda_gpu_id,
        )
        return sol

    for h in (5.0, 20.0):
        path = _uniform_path_theta60(mceq, h=h)
        per_species = _per_species_max_rel(mceq, run(path, "mkl"), run(path, "cuda"))
        # nan (nothing above the floor) is unmeasurable, not inside the budget.
        worst_em = max(per_species[name] for name in em)
        worst_other = max(
            v for name, v in per_species.items() if name not in em and np.isfinite(v)
        )
        print(
            f"h={h:g} ({path[0]} steps) mkl32 vs cuda32 per-species max-rel: "
            f"e+- {worst_em:.3e}, other {worst_other:.3e}"
        )
        assert worst_other <= 1e-4, (
            f"h={h:g}: worst non-e+- species = {worst_other:.3e} (e+- {worst_em:.3e})"
        )
        if h > 5.0:
            # Over-integrated, the e+- blocks are either past the budget or have
            # no bin above the floor at all -- either way the exclusion earns its
            # place. Spelled out rather than left to `not nan <= x`.
            assert np.isnan(worst_em) or worst_em > 1e-4, (
                f"h={h:g} over-integrates the EM cascade, so the e+- exclusion "
                f"should be load bearing, but their worst is {worst_em:.3e}"
            )


def test_numpy_etd2_empty_dec_off():
    """A pure e±/γ EM-cascade solve disables all decays, so dec_m has no
    off-diagonal. The kernel must carry an all-zero ``dec_off`` through the
    per-step SpMV at the same dimension as the non-empty ``int_off``.
    """
    import scipy.sparse as sp

    from MCEq.solvers import solve_etd2

    dim = 50
    rng = np.random.default_rng(0)
    int_m = sp.csr_matrix(rng.standard_normal((dim, dim)) * 0.01)
    int_m.setdiag(-0.5)
    int_m = int_m.tocsr()
    dec_m = sp.csr_matrix((dim, dim))  # all decays disabled -> empty off-diagonal

    nsteps = 20
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    phi = np.ones(dim)
    sol, _ = solve_etd2(nsteps, dX, rho_inv, int_m, dec_m, phi, [], backend="numpy")
    assert sol.shape == (dim,)
    assert np.isfinite(sol).all()


@pytest.mark.xdist_group("full_db")
@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_etd2_mkl_stable_at_high_zenith():
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
def test_solve_etd2_cuda_matches_numpy_real(mceq_sib21_full_db):
    """Equivalence test on real MCEq matrices.

    cuSPARSE may reorder partial sums vs scipy CSR, so we tolerate a
    rel-L2 of 1e-9 instead of round-off — but anything looser would mask
    a real bug. Path matches the MKL test for parity.
    """
    from MCEq.solvers import solve_etd2

    mceq = mceq_sib21_full_db
    nsteps, dX, rho_inv = _uniform_path_theta60(mceq)
    grid_idcs = []
    phi0 = mceq._phi0.copy()

    sol_numpy, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq.int_m,
        mceq.dec_m,
        phi0.copy(),
        grid_idcs,
        backend="numpy",
    )
    sol_cuda, _ = solve_etd2(
        nsteps,
        dX,
        rho_inv,
        mceq.int_m,
        mceq.dec_m,
        phi0.copy(),
        grid_idcs,
        backend="cuda",
        device_id=config.cuda_gpu_id,
    )

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
def test_etd2_cuda_stable_at_high_zenith():
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
def test_solve_etd2_numpy_generalized_target_convergence():
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
    from MCEq.solvers import solve_etd2

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
            sol, _ = solve_etd2(
                n,
                dX,
                rho_inv,
                mceq.int_m,
                mceq.dec_m,
                mceq._phi0.copy(),
                [],
                backend="numpy",
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
        phi0_2d = np.broadcast_to(phi0_1d[:, None], (mceq_sib21.dim_states, K)).copy()
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
                _StubParticle(True, 0, 1),  # e- (EM)
                _StubParticle(True, 1, 2),  # e+ (EM)
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
    the single-RHS kernel), including int_grid snapshots, and must not
    mutate the instance solution state.
    """
    saved_kernel = config.kernel_config

    try:
        config.kernel_config = "numpy_etd2"
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

        seasons = [
            ("CORSIKA", ("BK_USStd", None)),
            ("CORSIKA", ("PL_SouthPole", "January")),
        ]
        conditions = [{"zenith_deg": 60.0, "density_model": dm} for dm in seasons]
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
        mceq_sib21.solve_batch(np.zeros((dim, 3)), conditions=[{"zenith_deg": 0.0}] * 2)
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
                ("total_mu+", 0),
                ("conv_numu", 3),
                ("pr_antinumu", 0),
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
            ref_int,
            rtol=1e-12,
            atol=0,
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
                    "total_numu",
                    pixel=(i_zen, i_az),
                    return_as="kinetic energy",
                )[e_idx]
                np.testing.assert_allclose(smap[i_zen, i_az], ref, rtol=1e-12, atol=0)

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
            mceq_sib21.solve_fullsky(zenith_grid, phi0=phi0_2d, geomagnetic_cutoff=True)
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
def test_cuda_diag_factors_f64diag_accuracy():
    """The diag-factor kernel takes fp64 inputs and does fp64 arithmetic
    whatever the output dtype, so its fp32 outputs are the fp64 phi
    factors to fp32 roundoff.

    The pure-fp32 phi1/phi2 cancellations ((e-1)/hd, (e-1-hd)/hd^2)
    lose 3-7 digits around the Taylor-switch thresholds; this test locks
    in the fp64-internal contract so a regression to fp32 arithmetic (or a
    threshold recalibration that reopens the cancellation band) fails
    loudly.
    """
    import cupy as cp

    from MCEq.solvers.backends.cuda import _cuda_etd2_kernels

    Kset = _cuda_etd2_kernels()
    rng = np.random.default_rng(11)
    dim, K = 4096, 8
    # Diagonal rates and step sizes spanning the cancellation band
    # |hd| ~ 1e-6 .. 1e2, both signs, including near-threshold values.
    d_int = -np.abs(rng.lognormal(mean=-3.0, sigma=3.0, size=dim))
    d_dec = -np.abs(rng.lognormal(mean=-8.0, sigma=3.0, size=dim))
    h = rng.uniform(0.05, 15.0, (1, K))
    ri = rng.lognormal(mean=8.0, sigma=2.0, size=(1, K))

    args64 = [
        cp.asarray(a, dtype=cp.float64)
        for a in (d_int.reshape(dim, 1), d_dec.reshape(dim, 1), h, ri)
    ]
    outs_mixed = [cp.empty((dim, K), cp.float32) for _ in range(3)]
    Kset.diag_factors(*args64, *outs_mixed)

    # fp64 reference from the same inputs, so the comparison isolates the
    # output cast from the arithmetic.
    outs64 = [cp.empty((dim, K), cp.float64) for _ in range(3)]
    Kset.diag_factors(*args64, *outs64)

    for name, mixed, ref in zip(("eD", "hphi1", "hphi2"), outs_mixed, outs64):
        m = cp.asnumpy(mixed).astype(np.float64)
        r = cp.asnumpy(ref)
        mask = np.abs(r) > 1e-15 * np.abs(r).max()
        rel = np.abs(m[mask] - r[mask]) / np.abs(r[mask])
        assert rel.max() < 5e-7, (
            f"{name}: fp64-internal diag kernel rel err {rel.max():.2e} "
            f"exceeds fp32-roundoff budget 5e-7"
        )


def test_msis_condition_paths_are_fork_reproducible():
    """A forked path build reproduces the serial one on an MSIS atmosphere.

    `cNRLMSISE00` memoised on altitude alone, which made the azimuth-averaged
    `MSIS00LocationCentered` spline depend on the order the directions were
    evaluated in; a worker pool changes that order, which is what used to look
    like nrlmsise-00 not being fork-safe.
    """
    import numpy as np

    from MCEq.core import MCEqRun

    mceq = MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=None,
        density_model=("MSIS00_IC", ("SouthPole", "January")),
        build_matrices=False,
    )
    try:
        conditions = [
            {"zenith_deg": zenith, "azimuth_deg": azimuth, "density_model": None}
            for zenith in (20.0, 60.0)
            for azimuth in (0.0, 90.0, 180.0)
        ]
        serial = mceq._build_condition_paths(conditions, path_workers=0)
        forked = mceq._build_condition_paths(conditions, path_workers=4)

        assert len(serial) == len(forked) == len(conditions)
        for one, other in zip(serial, forked):
            assert one[0] == other[0]
            assert np.array_equal(one[1], other[1])
            assert np.array_equal(one[2], other[2])
    finally:
        mceq.close()
