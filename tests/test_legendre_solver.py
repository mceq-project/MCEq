"""ETD2 with a Legendre operator against the exact matrix exponential.

dPhi/dX = [D + M (A_off + ri B_off) + Band_off] S Phi: the removal M acts on
the leading modes of the production only, the energy-loss band bypasses it,
S is the sec(theta) coupling of the low-energy columns.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp

from MCEq.operators.legendre import backward_removal
from MCEq.operators.secant import build_secant_kernel_ops
from MCEq.solvers import solve_etd2

N_SP, N_E, L_GRID = 2, 6, np.array([0, 1, 2, 3, 7])
N = N_SP * N_E
N_K, N_D = len(L_GRID), 4


def _blocks(rng, density, scale, diag_scale=0.0):
    mats = []
    for _ in range(N_K):
        m = sp.random(N, N, density=density, random_state=rng).toarray() * scale
        np.fill_diagonal(m, -diag_scale * (1 + rng.random(N)))
        mats.append(m)
    return sl.block_diag(*mats)


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(5)
    hadr = _blocks(rng, 0.3, 0.2, diag_scale=1.0)
    dec = _blocks(rng, 0.2, 0.1, diag_scale=0.5)
    band1 = np.diag(-0.3 * np.ones(N)) + np.diag(0.3 * np.ones(N - 1), 1)
    band = sl.block_diag(*[band1] * N_K)
    M = backward_removal(N_D - 1)
    mix = SimpleNamespace(M=M, n_D=N_D, N=N)
    m_ext = np.eye(N_K)
    m_ext[:N_D, :N_D] = M
    big = np.kron(m_ext, np.eye(N))
    return SimpleNamespace(hadr=hadr, dec=dec, band=band, mix=mix, big=big)


def _reference(s, ri, S, X, phi0):
    def off(m):
        return m - np.diag(np.diag(m))

    L = (
        np.diag(np.diag(s.hadr + s.band) + ri * np.diag(s.dec))
        + s.big @ (off(s.hadr) + ri * off(s.dec))
        + off(s.band)
    )
    return sl.expm(L @ S * X) @ phi0


def _solve(s, sec_ops, ri, X, phi0, nsteps=4000, backend="numpy"):
    dX = np.full(nsteps, X / nsteps)
    sol, _ = solve_etd2(
        nsteps,
        dX,
        np.full(nsteps, ri),
        sp.csr_matrix(s.hadr),
        sp.csr_matrix(s.dec),
        phi0,
        backend=backend,
        sec_ops=sec_ops,
        band=sp.csr_matrix(s.band),
        mode_mix=s.mix,
    )
    return sol


@pytest.mark.parametrize("backend", ["numpy", "mkl"])
def test_paraxial_route_matches_expm(system, backend):
    if backend == "mkl":
        pytest.importorskip("MCEq.solvers.backends.mkl")
    phi0 = np.random.default_rng(1).random(N_K * N)
    ref = _reference(system, 0.7, np.eye(N_K * N), 1.5, phi0)
    try:
        sol = _solve(system, None, 0.7, 1.5, phi0, backend=backend)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"{backend} unavailable: {exc}")
    np.testing.assert_allclose(sol, ref, rtol=2e-6, atol=1e-9)


def test_coupled_route_matches_expm(system):
    spec = SimpleNamespace(
        cap_deg=75.0, row_kmax=50.0, lam_rel=1e-9, w_flat=1.0, e_max=0.5
    )
    e_centers = np.logspace(-2, 1, N_E)
    ops = build_secant_kernel_ops(L_GRID, e_centers, N_SP, spec, basis="legendre")
    T = np.zeros((N_K, N_K))
    T[ops["P"][:, None], np.arange(N_K)[None, :]] = ops["T_P"]
    low = np.zeros(N)
    low[ops["low_e_idx"]] = 1.0
    S = np.eye(N_K * N) + np.kron(T, np.diag(low))
    phi0 = np.random.default_rng(2).random(N_K * N)
    ref = _reference(system, 0.4, S, 1.0, phi0)
    # the dense rows couple to the sparse mode l = 7 explicitly: second order
    # convergence to the exact exponential, 4.3e-5 at 4000 steps
    err = [
        np.abs(_solve(system, ops, 0.4, 1.0, phi0, nsteps=n) / ref - 1).max()
        for n in (4000, 8000)
    ]
    assert err[1] < 2e-5
    assert 3.5 < err[0] / err[1] < 4.5


def test_without_removal_the_band_split_is_neutral(system):
    """mode_mix = identity reproduces the unsplit operator int_m = hadr + band."""
    eye = SimpleNamespace(M=np.eye(N_D), n_D=N_D, N=N)
    phi0 = np.random.default_rng(3).random(N_K * N)
    args = (40, np.full(40, 0.02), np.full(40, 0.3))
    split, _ = solve_etd2(
        *args,
        sp.csr_matrix(system.hadr),
        sp.csr_matrix(system.dec),
        phi0,
        band=sp.csr_matrix(system.band),
        mode_mix=eye,
    )
    whole, _ = solve_etd2(
        *args, sp.csr_matrix(system.hadr + system.band), sp.csr_matrix(system.dec), phi0
    )
    np.testing.assert_allclose(split, whole, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "mkl"])
def test_step_loop_runs_openblas_serial_on_mkl(system, backend, monkeypatch):
    """Inside the step loop of the MKL backend OpenBLAS runs at one thread and
    MKL keeps its pool; other backends and the pools after the loop are
    untouched."""
    from threadpoolctl import ThreadpoolController, threadpool_limits

    import MCEq.solvers.etd2 as etd2
    from MCEq import mkl_runtime

    if backend == "mkl":
        pytest.importorskip("MCEq.solvers.backends.mkl")
    seen = []
    make = etd2._BACKENDS[backend]

    def recording(*args):
        be = make(*args)
        stage = be.diag_factors

        def diag_factors(*a):
            seen.append((ThreadpoolController().info(), mkl_threads()))
            return stage(*a)

        be.diag_factors = diag_factors
        return be

    monkeypatch.setitem(etd2._BACKENDS, backend, recording)

    def threads(info, api):
        return {p["num_threads"] for p in info if p["internal_api"] == api}

    def mkl_threads():
        lib = mkl_runtime.lib()
        return None if lib is None else lib.mkl_get_max_threads()

    phi0 = np.random.default_rng(4).random(N_K * N)
    with threadpool_limits(limits=2, user_api="blas"):
        before = ThreadpoolController().info()
        if not threads(before, "openblas"):
            pytest.skip("numpy is not linked against OpenBLAS")
        try:
            _solve(system, None, 0.7, 0.1, phi0, nsteps=3, backend=backend)
        except (OSError, RuntimeError) as exc:
            pytest.skip(f"{backend} unavailable: {exc}")
        after = ThreadpoolController().info(), mkl_threads()
    serial = backend == "mkl"
    assert threads(seen[0][0], "openblas") == ({1} if serial else {2})
    assert seen[0][1] == after[1]
    assert threads(after[0], "openblas") == {2}
