"""Closure tests for the batched (multi-RHS / carousel) secant ETD2 driver.

The synthetic fixture has 48 logical Hankel modes with a tiny per-mode
state, so it exercises the production tensor layout, all modes,
independent initial columns, carousel harvest/reset and frozen-lane
pinning without the minutes-scale fitted operator or the external
production database. The batched kernels must reproduce independent
single-axis :func:`solv_numpy_etd2_secant` solves; tolerances are
~1e-11 relative — SpMM and K-fold SpMV need not agree to round-off
across BLAS implementations (the production fp64 CUDA carousel
validation closed at 4.6e-12).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.solvers import (
    MklSparseMatrix,
    _etd_split_cache,
    compile_carousel_schedule,
    schedule_lpt,
    solv_mkl_etd2_secant_carousel,
    solv_mkl_etd2_secant_multirhs,
    solv_numpy_etd2_secant,
    solv_numpy_etd2_secant_carousel,
    solv_numpy_etd2_secant_multirhs,
)

RTOL = 1e-11
ATOL = 1e-13


def _secant_ops(n_k, scale=1.0):
    """Constant operator set in the production layout, small and exact.

    ``S_P = I + T_PP`` is symmetric here, so the eigenbasis is orthogonal
    and its round-trip is well conditioned; two off-``P`` columns
    exercise the one-way ``T_P[:, ~P]`` support.
    """
    P = np.arange(12, dtype=np.int64)
    distance = np.abs(P[:, None] - P[None, :])
    T_PP = scale * 0.025 * np.exp(-distance / 2.5)
    T_P = np.zeros((len(P), n_k), dtype=np.float64)
    T_P[:, P] = T_PP
    T_P[:, 20] = scale * np.linspace(0.002, 0.0002, len(P))
    T_P[:, 35] = scale * np.linspace(-0.0001, 0.001, len(P))
    lam, V = np.linalg.eigh(np.eye(len(P)) + T_PP)
    return {
        "P": P,
        "T_P": T_P,
        "T_PP": T_PP,
        "V": V,
        "Vi": V.T,
        "lam": lam,
        # Conceptual low-energy state columns: hadron, muon, numu, nue.
        "low_e_idx": np.array([0, 2, 4, 5], dtype=np.int64),
        "n_k": n_k,
    }


@pytest.fixture
def secant_48mode_problem():
    rng = np.random.default_rng(20260825)
    n_k, N, K = 48, 6, 4
    dim = n_k * N
    per_mode_int = []
    per_mode_dec = []
    for mode in range(n_k):
        int_block = np.triu(rng.uniform(0.0, 0.012, (N, N)), 1)
        int_block -= np.diag(rng.uniform(0.05, 0.13, N) + mode * 2e-5)
        dec_block = np.triu(rng.uniform(0.0, 0.006, (N, N)), 1)
        dec_block -= np.diag(rng.uniform(0.01, 0.035, N))
        per_mode_int.append(sp.csr_matrix(int_block))
        per_mode_dec.append(sp.csr_matrix(dec_block))
    int_m = sp.block_diag(per_mode_int, format="csr")
    dec_m = sp.block_diag(per_mode_dec, format="csr")

    # Distinct primary columns, including a zero/cutoff-like state.
    # Tiling each per-mode base vector mirrors a collimated 2D primary.
    phi0 = np.empty((dim, K))
    base = rng.uniform(0.1, 1.2, (N, K))
    base[:, 1] *= np.linspace(0.2, 1.0, N)
    base[:, 2] = 0.0
    base[:, 3] *= 1.7
    phi0[:] = np.tile(base, (n_k, 1))

    dX = np.array([0.08, 0.13, 0.05, 0.17, 0.09, 0.11])
    rho_inv = np.array([1.1, 0.8, 1.3, 0.7, 0.95, 1.05])
    # Unequal path lengths force LPT packing, resets and frozen tails.
    paths = [
        (6, dX.copy(), rho_inv.copy(), []),
        (3, dX[:3].copy(), rho_inv[:3].copy(), []),
        (5, dX[:5].copy(), rho_inv[:5].copy(), []),
        (2, dX[:2].copy(), rho_inv[:2].copy(), []),
    ]
    return {
        "n_k": n_k,
        "N": N,
        "K": K,
        "dim": dim,
        "int_m": int_m,
        "dec_m": dec_m,
        "phi0": phi0,
        "dX": dX,
        "rho_inv": rho_inv,
        "paths": paths,
    }


def _single_axis_columns(problem, paths, sec_ops):
    out = []
    for rhs, path in enumerate(paths):
        nsteps, dX, rho_inv, _ = path
        col, _ = solv_numpy_etd2_secant(
            nsteps,
            dX,
            rho_inv,
            problem["int_m"],
            problem["dec_m"],
            problem["phi0"][:, rhs],
            [],
            sec_ops,
        )
        out.append(col)
    return np.stack(out, axis=1)


def _carousel_inputs(problem, K_pipe):
    slots, T = schedule_lpt([path[0] for path in problem["paths"]], K_pipe)
    return compile_carousel_schedule(
        problem["paths"], slots, T, problem["dim"], problem["phi0"]
    )


@pytest.mark.parametrize("operator_scale", [0.7, 1.0])
def test_numpy_secant_multirhs_matches_repeated_single(
    secant_48mode_problem, operator_scale
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"], operator_scale)
    snapshots = [1, 5]
    nsteps = len(p["dX"])
    batched, batched_grid = solv_numpy_etd2_secant_multirhs(
        nsteps, p["dX"], p["rho_inv"], p["int_m"], p["dec_m"], p["phi0"],
        snapshots, ops,
    )

    reference = []
    reference_grid = []
    for rhs in range(p["K"]):
        col, grid = solv_numpy_etd2_secant(
            nsteps, p["dX"], p["rho_inv"], p["int_m"], p["dec_m"],
            p["phi0"][:, rhs], snapshots, ops,
        )
        reference.append(col)
        reference_grid.append(grid)
    reference = np.stack(reference, axis=1)
    reference_grid = np.stack(reference_grid, axis=2)

    np.testing.assert_allclose(batched, reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        batched_grid, reference_grid, rtol=RTOL, atol=ATOL
    )
    # The zero/cutoff-like column stays exactly zero (no cross-talk).
    assert np.count_nonzero(batched[:, 2]) == 0


def test_numpy_secant_carousel_matches_independent_paths(
    secant_48mode_problem,
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    reference = _single_axis_columns(p, p["paths"], ops)

    dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=2)
    carousel = solv_numpy_etd2_secant_carousel(
        p["int_m"], p["dec_m"], dX_c, rho_c, phi_initial, schedule,
        p["phi0"], ops,
    )

    np.testing.assert_allclose(carousel, reference, rtol=RTOL, atol=ATOL)
    # Harvest lands in original pixel order, and the zero/cutoff-like
    # input remains isolated across resets and frozen tails.
    assert np.count_nonzero(carousel[:, 2]) == 0
    for rhs in (0, 1, 3):
        assert np.count_nonzero(carousel[:, rhs]) > 0


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_mkl_secant_multirhs_and_carousel_match_numpy(secant_48mode_problem):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    nsteps = len(p["dX"])
    d_int, d_dec, int_off, dec_off = _etd_split_cache(p["int_m"], p["dec_m"])
    mkl_int = MklSparseMatrix(int_off.tocsr()) if int_off.nnz else None
    mkl_dec = MklSparseMatrix(dec_off.tocsr()) if dec_off.nnz else None
    try:
        numpy_shared, _ = solv_numpy_etd2_secant_multirhs(
            nsteps, p["dX"], p["rho_inv"], p["int_m"], p["dec_m"],
            p["phi0"], [], ops,
        )
        mkl_shared, _ = solv_mkl_etd2_secant_multirhs(
            nsteps, p["dX"], p["rho_inv"], mkl_int, mkl_dec, d_int, d_dec,
            p["phi0"], [], ops,
        )
        np.testing.assert_allclose(
            mkl_shared, numpy_shared, rtol=RTOL, atol=ATOL
        )

        dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=2)
        mkl_carousel = solv_mkl_etd2_secant_carousel(
            mkl_int, mkl_dec, d_int, d_dec, dX_c, rho_c, phi_initial,
            schedule, p["phi0"], ops,
        )
        reference = _single_axis_columns(p, p["paths"], ops)
        np.testing.assert_allclose(
            mkl_carousel, reference, rtol=RTOL, atol=ATOL
        )
        assert np.count_nonzero(mkl_carousel[:, 2]) == 0
    finally:
        for m in (mkl_int, mkl_dec):
            if m is not None:
                m.close()


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy not available")
def test_cuda_secant_multirhs_and_carousel_backend_parity(
    secant_48mode_problem,
):
    from MCEq.solvers import (
        CudaEtd2Context,
        CudaEtd2MultiRHSContext,
        solv_cuda_etd2_secant,
        solv_cuda_etd2_secant_carousel,
        solv_cuda_etd2_secant_multirhs,
    )

    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    nsteps = len(p["dX"])
    d_int, d_dec, int_off, dec_off = _etd_split_cache(p["int_m"], p["dec_m"])
    int_off = int_off.tocsr()
    dec_off = dec_off.tocsr()

    multi_ctx = CudaEtd2MultiRHSContext(
        int_off, dec_off, d_int, d_dec, K=p["K"], device_id=0, fp_precision=64
    )
    cuda_shared, _ = solv_cuda_etd2_secant_multirhs(
        nsteps, p["dX"], p["rho_inv"], multi_ctx, p["phi0"], [], ops
    )
    numpy_shared, _ = solv_numpy_etd2_secant_multirhs(
        nsteps, p["dX"], p["rho_inv"], p["int_m"], p["dec_m"], p["phi0"],
        [], ops,
    )
    np.testing.assert_allclose(cuda_shared, numpy_shared, rtol=2e-11, atol=2e-12)

    dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=p["K"])
    cuda_carousel = solv_cuda_etd2_secant_carousel(
        multi_ctx, dX_c, rho_c, phi_initial, schedule, p["phi0"], ops
    )
    numpy_reference = _single_axis_columns(p, p["paths"], ops)
    np.testing.assert_allclose(
        cuda_carousel, numpy_reference, rtol=2e-11, atol=2e-12
    )
    assert np.count_nonzero(cuda_carousel[:, 2]) == 0

    # The carousel must also agree with the single-axis CUDA kernel run
    # per path (same driver, K = 1 without a schedule).
    single_ctx = CudaEtd2Context(
        int_off, dec_off, d_int, d_dec, device_id=0, fp_precision=64
    )
    cuda_single = []
    for rhs, path in enumerate(p["paths"]):
        ns, dXp, rip, _ = path
        col, _ = solv_cuda_etd2_secant(
            ns, dXp, rip, single_ctx, p["phi0"][:, rhs], [], ops
        )
        cuda_single.append(col)
    np.testing.assert_allclose(
        cuda_carousel, np.stack(cuda_single, axis=1), rtol=2e-11, atol=2e-12
    )


# ---------------------------------------------------------------------------
# solve_batch wiring: the tri-state resolver for the batch entry points is a
# pure function of (config, kernel_config, dtype, rho-stack); test it unbound
# with a stub, like the single-axis tri-state in test_2d_defaults.py.
# ---------------------------------------------------------------------------
def _resolve_for(flag, kernel, dtype=np.float64, is_2d=True, rho_stack=False):
    from MCEq.core import MCEqRun

    stub = SimpleNamespace(
        _mceq_db=SimpleNamespace(is_2d=is_2d),
        _build_secant_ops=lambda: {"marker": True},
    )
    if rho_stack:
        stub._int_m_stack = [None]
    saved = (config.secant_theta_transport, config.kernel_config)
    config.secant_theta_transport = flag
    config.kernel_config = kernel
    try:
        return MCEqRun._resolve_batch_secant(stub, "test", dtype)
    finally:
        config.secant_theta_transport, config.kernel_config = saved


def test_resolve_batch_secant_tri_state():
    # Supported backends build one constant operator set.
    for kernel in ("numpy_etd2", "mkl_etd2", "cuda_etd2"):
        assert _resolve_for("auto", kernel) == {"marker": True}
        assert _resolve_for(True, kernel) == {"marker": True}
    # 1D databases and explicit False stay paraxial without building.
    assert _resolve_for("auto", "numpy_etd2", is_2d=False) is None
    assert _resolve_for(False, "numpy_etd2") is None
    # Unsupported configurations downgrade under "auto", raise under
    # "require": accelerate backend, fp32 outside cuda, EM rho-stack.
    assert _resolve_for("auto", "accelerate_etd2") is None
    assert _resolve_for("auto", "mkl_etd2", dtype=np.float32) is None
    assert _resolve_for("auto", "numpy_etd2", rho_stack=True) is None
    assert _resolve_for(True, "cuda_etd2", dtype=np.float32) == {"marker": True}
    for kwargs in (
        {"kernel": "accelerate_etd2"},
        {"kernel": "mkl_etd2", "dtype": np.float32},
        {"kernel": "numpy_etd2", "rho_stack": True},
    ):
        with pytest.raises(NotImplementedError):
            _resolve_for(True, **kwargs)
