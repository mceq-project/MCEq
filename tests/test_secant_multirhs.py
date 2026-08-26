"""Focused closure tests for secant-coupled multi-RHS ETD2 kernels.

The fixture deliberately has 48 logical Hankel modes while keeping the
per-mode state tiny.  It therefore exercises the production tensor layout,
all modes, independent initial columns, carousel resets, and an exact-J1
theta-integrated readout without requiring the minutes-scale fitted operator
or the external production database.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.special import j1

from MCEq.solvers import (
    CudaEtd2Context,
    CudaEtd2MultiRHSContext,
    _etd_split_cache,
    compile_carousel_schedule,
    schedule_lpt,
    solv_cuda_etd2_secant,
    solv_cuda_etd2_secant_carousel,
    solv_cuda_etd2_secant_multirhs,
    solv_numpy_etd2_secant,
    solv_numpy_etd2_secant_carousel,
    solv_numpy_etd2_secant_multirhs,
)


def _secant_ops(n_k, N, scale=1.0):
    P = np.arange(12, dtype=np.int64)
    distance = np.abs(P[:, None] - P[None, :])
    T_PP = scale * 0.025 * np.exp(-distance / 2.5)
    T_P = np.zeros((len(P), n_k), dtype=np.float64)
    T_P[:, P] = T_PP
    # One-way support from uncoupled modes exercises the T_P[:, ~P] leg.
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
        # Conceptual columns: hadron, muon, numu, nue.
        "gate_idx": np.array([0, 2, 4, 5], dtype=np.int64),
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

    phi0 = rng.uniform(0.05, 1.0, (dim, K))
    # Distinct primary columns, including a zero/cutoff-like state.  Tiling
    # each per-mode base vector also mirrors a collimated 2-D primary.
    base = rng.uniform(0.1, 1.2, (N, K))
    base[:, 1] *= np.linspace(0.2, 1.0, N)
    base[:, 2] = 0.0
    base[:, 3] *= 1.7
    phi0[:] = np.tile(base, (n_k, 1))

    dX = np.array([0.08, 0.13, 0.05, 0.17, 0.09, 0.11])
    rho_inv = np.array([1.1, 0.8, 1.3, 0.7, 0.95, 1.05])
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
        "k_grid": np.linspace(0.0, 80.0, n_k),
        "int_m": int_m,
        "dec_m": dec_m,
        "phi0": phi0,
        "dX": dX,
        "rho_inv": rho_inv,
        "paths": paths,
    }


def _single_numpy_columns(problem, paths, sec_ops):
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


def _theta_integrated(state, k_grid, n_k, N):
    """Exact-J1 observable for every conceptual species/RHS column."""
    modes = state.reshape(n_k, N, -1)
    theta_max = np.pi / 2.0
    k_fine = np.linspace(k_grid[0], k_grid[-1], 2001)
    dk = np.diff(k_fine)
    trap = np.empty_like(k_fine)
    trap[0] = 0.5 * dk[0]
    trap[-1] = 0.5 * dk[-1]
    trap[1:-1] = 0.5 * (dk[:-1] + dk[1:])
    weights = theta_max * j1(k_fine * theta_max) * trap
    fine = CubicSpline(k_grid, modes, axis=0)(k_fine)
    return np.tensordot(weights, fine, axes=(0, 0))


@pytest.mark.parametrize("operator_scale", [0.7, 1.0])
def test_numpy_secant_multirhs_matches_repeated_single_all_modes(
    secant_48mode_problem, operator_scale
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"], p["N"], operator_scale)
    snapshots = [1, 5]
    # Homogeneous closure means genuinely identical RHS columns. Distinct
    # per-RHS primaries and cross-talk/order are exercised by the carousel
    # test below.
    identical_phi0 = np.repeat(p["phi0"][:, :1], p["K"], axis=1)
    batched, batched_grid = solv_numpy_etd2_secant_multirhs(
        len(p["dX"]),
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        identical_phi0,
        snapshots,
        ops,
    )

    reference = []
    reference_grid = []
    for rhs in range(p["K"]):
        col, grid = solv_numpy_etd2_secant(
            len(p["dX"]),
            p["dX"],
            p["rho_inv"],
            p["int_m"],
            p["dec_m"],
            identical_phi0[:, rhs],
            snapshots,
            ops,
        )
        reference.append(col)
        reference_grid.append(grid)
    reference = np.stack(reference, axis=1)
    reference_grid = np.stack(reference_grid, axis=2)

    np.testing.assert_allclose(batched, reference, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(batched_grid, reference_grid, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        _theta_integrated(batched, p["k_grid"], p["n_k"], p["N"]),
        _theta_integrated(reference, p["k_grid"], p["n_k"], p["N"]),
        rtol=5e-13,
        atol=5e-13,
    )


def test_numpy_secant_carousel_matches_independent_paths_and_primary_states(
    secant_48mode_problem,
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"], p["N"])
    reference = _single_numpy_columns(p, p["paths"], ops)

    slots, T = schedule_lpt([path[0] for path in p["paths"]], 2)
    dX_c, rho_c, phi_initial, schedule = compile_carousel_schedule(
        p["paths"], slots, T, p["dim"], p["phi0"]
    )
    carousel = solv_numpy_etd2_secant_carousel(
        p["int_m"],
        p["dec_m"],
        dX_c,
        rho_c,
        phi_initial,
        schedule,
        p["phi0"],
        ops,
    )

    np.testing.assert_allclose(carousel, reference, rtol=3e-14, atol=3e-14)
    # The zero/cutoff-like input remains isolated, and harvest order is the
    # original RHS order rather than LPT slot order.
    assert np.count_nonzero(carousel[:, 2]) == 0
    assert np.count_nonzero(carousel[:, 0]) > 0
    assert np.count_nonzero(carousel[:, 1]) > 0
    assert np.count_nonzero(carousel[:, 3]) > 0
    np.testing.assert_allclose(
        _theta_integrated(carousel, p["k_grid"], p["n_k"], p["N"]),
        _theta_integrated(reference, p["k_grid"], p["n_k"], p["N"]),
        rtol=5e-13,
        atol=5e-13,
    )


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy not available")
def test_cuda_secant_multirhs_and_carousel_backend_parity(secant_48mode_problem):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"], p["N"])
    d_int, d_dec, int_off, dec_off = _etd_split_cache(p["int_m"], p["dec_m"])
    multi_ctx = CudaEtd2MultiRHSContext(
        int_off, dec_off, d_int, d_dec, K=p["K"], device_id=0, fp_precision=64
    )
    width_two_ctx = CudaEtd2MultiRHSContext(
        int_off,
        dec_off,
        d_int,
        d_dec,
        K=2,
        device_id=0,
        fp_precision=64,
        shared_static=multi_ctx,
    )
    assert width_two_ctx.cu_int_off is multi_ctx.cu_int_off
    assert width_two_ctx.cu_dec_off is multi_ctx.cu_dec_off
    assert width_two_ctx.cu_d_int is multi_ctx.cu_d_int
    assert width_two_ctx.cu_d_dec is multi_ctx.cu_d_dec
    assert width_two_ctx.cu_phc is not multi_ctx.cu_phc
    cuda_shared, _ = solv_cuda_etd2_secant_multirhs(
        len(p["dX"]),
        p["dX"],
        p["rho_inv"],
        multi_ctx,
        p["phi0"],
        [],
        ops,
    )
    numpy_shared, _ = solv_numpy_etd2_secant_multirhs(
        len(p["dX"]),
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        [],
        ops,
    )
    np.testing.assert_allclose(cuda_shared, numpy_shared, rtol=2e-11, atol=2e-12)

    slots, T = schedule_lpt([path[0] for path in p["paths"]], p["K"])
    dX_c, rho_c, phi_initial, schedule = compile_carousel_schedule(
        p["paths"], slots, T, p["dim"], p["phi0"]
    )
    cuda_carousel = solv_cuda_etd2_secant_carousel(
        multi_ctx, dX_c, rho_c, phi_initial, schedule, p["phi0"], ops
    )
    numpy_reference = _single_numpy_columns(p, p["paths"], ops)
    np.testing.assert_allclose(cuda_carousel, numpy_reference, rtol=2e-11, atol=2e-12)

    single_ctx = CudaEtd2Context(
        int_off, dec_off, d_int, d_dec, device_id=0, fp_precision=64
    )
    cuda_single = []
    for rhs, path in enumerate(p["paths"]):
        nsteps, dX, rho_inv, _ = path
        col, _ = solv_cuda_etd2_secant(
            nsteps,
            dX,
            rho_inv,
            single_ctx,
            p["phi0"][:, rhs],
            [],
            ops,
        )
        cuda_single.append(col)
    np.testing.assert_allclose(
        cuda_carousel,
        np.stack(cuda_single, axis=1),
        rtol=2e-11,
        atol=2e-12,
    )
