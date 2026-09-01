"""Closure tests for the batched (multi-RHS / carousel) secant ETD2 driver.

The synthetic fixture has 48 logical Hankel modes with a tiny per-mode
state, so it exercises the production tensor layout, all modes,
independent initial columns, carousel harvest/reset and frozen-lane
pinning without the minutes-scale fitted operator or the external
production database. The batched kernels must reproduce independent
single-axis :func:`solve_etd2` solves; tolerances are
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
    compile_carousel_schedule,
    schedule_lpt,
    secant_layout,
    secant_split,
    solve_etd2,
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
        col, _ = solve_etd2(
            nsteps,
            dX,
            rho_inv,
            problem["int_m"],
            problem["dec_m"],
            problem["phi0"][:, rhs],
            [],
            backend="numpy",
            sec_ops=sec_ops,
        )
        out.append(col)
    return np.stack(out, axis=1)


def _carousel_inputs(problem, K_pipe):
    slots, T = schedule_lpt([path[0] for path in problem["paths"]], K_pipe)
    return compile_carousel_schedule(
        problem["paths"], slots, T, problem["dim"], problem["phi0"]
    )


def test_secant_layout_and_split(secant_48mode_problem):
    """Stage 0: in the permuted state the coupled plane is the corner
    block, and the permuted operators act on it as the originals act on
    the original state (bit-identical CSR SpMM: row order preserved)."""
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    lay = secant_layout(ops, p["dim"])
    x = p["phi0"]
    xp = x[lay.perm]
    corner = xp.reshape(lay.n_k, lay.N, p["K"])[: lay.n_P, : lay.n_g]
    plane = x.reshape(lay.n_k, lay.N, p["K"])[np.ix_(ops["P"], ops["low_e_idx"])]
    np.testing.assert_array_equal(corner, plane)
    assert np.array_equal(xp[lay.inv_perm], x)

    d_int, d_dec, int_off, dec_off = secant_split(p["int_m"], p["dec_m"], ops)
    assert np.array_equal(d_int, p["int_m"].diagonal()[lay.perm])
    for m, off in ((p["int_m"], int_off), (p["dec_m"], dec_off)):
        ref = (m - sp.diags(m.diagonal())) @ x
        np.testing.assert_array_equal((off @ xp)[lay.inv_perm], ref)
    # (Caching of the compiled operator is MCEqRun._compiled_operator's
    # job; compile_operator / secant_split are pure.)

    bad = dict(ops, P=np.array([0, 2, 3]))
    with pytest.raises(ValueError):
        secant_layout(bad, p["dim"])


@pytest.mark.parametrize("operator_scale", [0.7, 1.0])
def test_numpy_secant_multirhs_matches_repeated_single(
    secant_48mode_problem, operator_scale
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"], operator_scale)
    snapshots = [1, 5]
    nsteps = len(p["dX"])
    batched, batched_grid = solve_etd2(
        nsteps,
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        snapshots,
        backend="numpy",
        sec_ops=ops,
    )

    reference = []
    reference_grid = []
    for rhs in range(p["K"]):
        col, grid = solve_etd2(
            nsteps,
            p["dX"],
            p["rho_inv"],
            p["int_m"],
            p["dec_m"],
            p["phi0"][:, rhs],
            snapshots,
            backend="numpy",
            sec_ops=ops,
        )
        reference.append(col)
        reference_grid.append(grid)
    reference = np.stack(reference, axis=1)
    reference_grid = np.stack(reference_grid, axis=2)

    np.testing.assert_allclose(batched, reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(batched_grid, reference_grid, rtol=RTOL, atol=ATOL)
    # The zero/cutoff-like column stays exactly zero (no cross-talk).
    assert np.count_nonzero(batched[:, 2]) == 0


def test_numpy_secant_carousel_matches_independent_paths(
    secant_48mode_problem,
):
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    reference = _single_axis_columns(p, p["paths"], ops)

    dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=2)
    carousel = solve_etd2(
        schedule.T,
        dX_c,
        rho_c,
        p["int_m"],
        p["dec_m"],
        phi_initial,
        [],
        backend="numpy",
        sec_ops=ops,
        schedule=schedule,
        phi0_per_pixel=p["phi0"],
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
    numpy_shared, _ = solve_etd2(
        nsteps,
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        [],
        backend="numpy",
        sec_ops=ops,
    )
    mkl_shared, _ = solve_etd2(
        nsteps,
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        [],
        backend="mkl",
        sec_ops=ops,
    )
    np.testing.assert_allclose(mkl_shared, numpy_shared, rtol=RTOL, atol=ATOL)

    dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=2)
    mkl_carousel = solve_etd2(
        schedule.T,
        dX_c,
        rho_c,
        p["int_m"],
        p["dec_m"],
        phi_initial,
        [],
        backend="mkl",
        sec_ops=ops,
        schedule=schedule,
        phi0_per_pixel=p["phi0"],
    )
    reference = _single_axis_columns(p, p["paths"], ops)
    np.testing.assert_allclose(mkl_carousel, reference, rtol=RTOL, atol=ATOL)
    assert np.count_nonzero(mkl_carousel[:, 2]) == 0


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
    p = secant_48mode_problem
    ops = _secant_ops(p["n_k"])
    nsteps = len(p["dX"])

    cuda_shared, _ = solve_etd2(
        nsteps,
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        [],
        backend="cuda",
        sec_ops=ops,
    )
    numpy_shared, _ = solve_etd2(
        nsteps,
        p["dX"],
        p["rho_inv"],
        p["int_m"],
        p["dec_m"],
        p["phi0"],
        [],
        backend="numpy",
        sec_ops=ops,
    )
    np.testing.assert_allclose(cuda_shared, numpy_shared, rtol=2e-11, atol=2e-12)

    dX_c, rho_c, phi_initial, schedule = _carousel_inputs(p, K_pipe=p["K"])
    cuda_carousel = solve_etd2(
        schedule.T,
        dX_c,
        rho_c,
        p["int_m"],
        p["dec_m"],
        phi_initial,
        [],
        backend="cuda",
        sec_ops=ops,
        schedule=schedule,
        phi0_per_pixel=p["phi0"],
    )
    numpy_reference = _single_axis_columns(p, p["paths"], ops)
    np.testing.assert_allclose(cuda_carousel, numpy_reference, rtol=2e-11, atol=2e-12)
    assert np.count_nonzero(cuda_carousel[:, 2]) == 0

    # The carousel must also agree with the single-axis CUDA run per path
    # (same driver, K = 1 without a schedule).
    cuda_single = []
    for rhs, path in enumerate(p["paths"]):
        ns, dXp, rip, _ = path
        col, _ = solve_etd2(
            ns,
            dXp,
            rip,
            p["int_m"],
            p["dec_m"],
            p["phi0"][:, rhs],
            [],
            backend="cuda",
            sec_ops=ops,
        )
        cuda_single.append(col)
    np.testing.assert_allclose(
        cuda_carousel, np.stack(cuda_single, axis=1), rtol=2e-11, atol=2e-12
    )


# ---------------------------------------------------------------------------
# solve_batch wiring: the tri-state resolver for the batch entry points is a
# pure function of (config, kernel_config, dtype); test it unbound with a
# stub, like the single-axis tri-state in test_2d_defaults.py.
# ---------------------------------------------------------------------------
def _resolve_for(flag, kernel, dtype=np.float64, is_2d=True):
    from MCEq.core import MCEqRun

    stub = SimpleNamespace(
        _mceq_db=SimpleNamespace(is_2d=is_2d),
        _build_secant_ops=lambda: {"marker": True},
    )
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
    # "require": accelerate backend, fp32 outside cuda.
    assert _resolve_for("auto", "accelerate_etd2") is None
    assert _resolve_for("auto", "mkl_etd2", dtype=np.float32) is None
    assert _resolve_for(True, "cuda_etd2", dtype=np.float32) == {"marker": True}
    for kwargs in (
        {"kernel": "accelerate_etd2"},
        {"kernel": "mkl_etd2", "dtype": np.float32},
    ):
        with pytest.raises(NotImplementedError):
            _resolve_for(True, **kwargs)
