"""ETD2 with a Legendre operator against the exact matrix exponential.

dPhi/dX = [D + M (A_off + ri B_off) + Band_off] S Phi: the removal M acts on
the leading modes of the production only, the energy-loss band bypasses it,
S is the sec(theta) coupling of the low-energy columns. The CUDA cases carry
the ``cuda`` marker and skip without a device.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp

from MCEq import config
from MCEq.operators.legendre import backward_removal
from MCEq.operators.secant import build_secant_kernel_ops
from MCEq.solvers import solve_etd2

N_SP, N_E, L_GRID = 2, 6, np.array([0, 1, 2, 3, 7])
N = N_SP * N_E
N_K, N_D = len(L_GRID), 4
SPEC = SimpleNamespace(cap_deg=75.0, e_max=0.5)
CUDA = pytest.param(
    "cuda",
    marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not config.has_cuda, reason="needs a CUDA device"),
    ],
)


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


def _secant_ops():
    return build_secant_kernel_ops(
        L_GRID, np.logspace(-2, 1, N_E), N_SP, SPEC.cap_deg, e_max=SPEC.e_max
    )


def _solve(s, sec_ops, ri, X, phi0, nsteps=4000, backend="numpy", fp=64):
    """``nsteps`` equal steps over ``X``; ``X`` and ``ri`` of shape ``(K,)``
    give one path per lane of a ``(dim, K)`` state."""
    dX = np.multiply.outer(np.ones(nsteps), np.asarray(X) / nsteps)
    sol, _ = solve_etd2(
        nsteps,
        dX,
        np.broadcast_to(ri, dX.shape),
        sp.csr_matrix(s.hadr),
        sp.csr_matrix(s.dec),
        phi0,
        backend=backend,
        sec_ops=sec_ops,
        band=sp.csr_matrix(s.band),
        mode_mix=s.mix,
        dtype=np.float32 if fp == 32 else np.float64,
    )
    return sol


@pytest.mark.parametrize("backend", ["numpy", "mkl", CUDA])
def test_paraxial_route_matches_expm(system, backend):
    if backend == "mkl" and not config.has_mkl:
        pytest.skip("MKL runtime not available")
    phi0 = np.random.default_rng(1).random(N_K * N)
    ref = _reference(system, 0.7, np.eye(N_K * N), 1.5, phi0)
    try:
        sol = _solve(system, None, 0.7, 1.5, phi0, backend=backend)
    except (OSError, RuntimeError) as exc:
        if backend == "cuda":
            raise
        pytest.skip(f"{backend} unavailable: {exc}")
    np.testing.assert_allclose(sol, ref, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", CUDA])
def test_coupled_route_matches_expm(system, backend):
    ops = _secant_ops()
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
        np.abs(
            _solve(system, ops, 0.4, 1.0, phi0, nsteps=n, backend=backend) / ref - 1
        ).max()
        for n in (4000, 8000)
    ]
    assert err[1] < 2e-5
    assert 3.5 < err[0] / err[1] < 4.5


@pytest.mark.parametrize("backend", ["numpy", CUDA])
def test_coupled_route_in_a_rotated_basis(system, backend):
    """The step is exact in any real basis of ``S_P``: rotating two
    eigenvectors into each other (as for the ``[Re v, Im v]`` plane of a
    split pair) leaves the off-diagonal part to the remainder and the
    solution on the exact exponential."""
    ops = dict(_secant_ops())
    c, s_ = np.cos(0.3), np.sin(0.3)
    R = np.eye(len(ops["lam"]))
    R[:2, :2] = [[c, -s_], [s_, c]]
    ops["V"], ops["Vi"] = ops["V"] @ R, R.T @ ops["Vi"]
    T = np.zeros((N_K, N_K))
    T[ops["P"][:, None], np.arange(N_K)[None, :]] = ops["T_P"]
    low = np.zeros(N)
    low[ops["low_e_idx"]] = 1.0
    S = np.eye(N_K * N) + np.kron(T, np.diag(low))
    phi0 = np.random.default_rng(2).random(N_K * N)
    ref = _reference(system, 0.4, S, 1.0, phi0)
    sol = _solve(system, ops, 0.4, 1.0, phi0, nsteps=4000, backend=backend)
    assert np.abs(sol / ref - 1).max() < 1e-4


@pytest.mark.parametrize("backend", ["numpy", CUDA])
def test_without_removal_the_band_split_is_neutral(system, backend):
    """mode_mix = identity reproduces the unsplit operator int_m = hadr + band."""
    eye = SimpleNamespace(M=np.eye(N_D), n_D=N_D, N=N)
    phi0 = np.random.default_rng(3).random(N_K * N)
    args = (40, np.full(40, 0.02), np.full(40, 0.3))
    split, _ = solve_etd2(
        *args,
        sp.csr_matrix(system.hadr),
        sp.csr_matrix(system.dec),
        phi0,
        backend=backend,
        band=sp.csr_matrix(system.band),
        mode_mix=eye,
    )
    whole, _ = solve_etd2(
        *args,
        sp.csr_matrix(system.hadr + system.band),
        sp.csr_matrix(system.dec),
        phi0,
        backend=backend,
    )
    np.testing.assert_allclose(split, whole, rtol=1e-12, atol=1e-14)


@pytest.mark.cuda
@pytest.mark.skipif(not config.has_cuda, reason="needs a CUDA device")
@pytest.mark.parametrize("coupled", [False, True], ids=["paraxial", "coupled"])
@pytest.mark.parametrize(
    "K, per_lane", [(1, False), (8, False), (8, True), (3, True), (64, True)]
)
@pytest.mark.parametrize("fp", [64, 32])
def test_cuda_matches_numpy(system, coupled, K, per_lane, fp):
    """CUDA against numpy at the same precision, relative to each column's
    peak: one shared path or one path per lane, tuned (1, 8, 64) and untuned
    (3) lane maps of the SpMM kernels. At fp32 the coupled route's own
    rounding is 1.2-1.8e-4 of the peak on both backends (against fp64), so
    the parity bound there is 5e-5 (measured 1.9e-5); paraxial 1e-5
    (measured 1.9e-6)."""
    ops = _secant_ops() if coupled else None
    rng = np.random.default_rng(6)
    phi0 = rng.random(N_K * N) if K == 1 else rng.random((N_K * N, K))
    X, ri = (
        (np.linspace(0.5, 1.5, K), np.linspace(0.2, 0.6, K)) if per_lane else (1.0, 0.4)
    )
    ref, got = (
        _solve(system, ops, ri, X, phi0, nsteps=400, backend=b, fp=fp)
        for b in ("numpy", "cuda")
    )
    assert np.isfinite(got).all()
    tol = {64: 1e-12, 32: 5e-5 if coupled else 1e-5}[fp]
    assert np.max(np.abs(got - ref) / np.abs(ref).max(axis=0)) < tol


@pytest.mark.cuda
@pytest.mark.skipif(not config.has_cuda, reason="needs a CUDA device")
def test_cuda_band_is_uploaded_with_the_operator(system, monkeypatch):
    """The band reaches the device once, with the operator; a solve uploads
    no CSR matrix."""
    from MCEq.operators.compiled import compile_operator
    from MCEq.solvers import etd2_driver
    from MCEq.solvers.backends import cuda

    op = compile_operator(
        sp.csr_matrix(system.hadr),
        sp.csr_matrix(system.dec),
        None,
        band=sp.csr_matrix(system.band),
        mode_mix=system.mix,
    )
    be = cuda.CudaBackend(op)
    assert be.cu_band_off.nnz == op.band_off.nnz

    def no_upload(*args):
        raise AssertionError("CSR upload during a solve")

    monkeypatch.setattr(cuda, "_DeviceCsr", no_upload)
    phi0 = np.random.default_rng(7).random((N_K * N, 2))
    for _ in range(2):
        sol, _ = etd2_driver(20, np.full(20, 0.01), np.full(20, 0.3), be, phi0, ())
    assert np.isfinite(sol).all()


@pytest.mark.parametrize("backend", ["numpy", "mkl"])
def test_step_loop_runs_openblas_serial_on_mkl(system, backend, monkeypatch):
    """Inside the step loop of the MKL backend OpenBLAS runs at one thread and
    MKL keeps its pool; other backends and the pools after the loop are
    untouched."""
    from threadpoolctl import ThreadpoolController, threadpool_limits

    import MCEq.solvers.etd2 as etd2

    if backend == "mkl" and not config.has_mkl:
        pytest.skip("MKL runtime not available")
    seen = []
    make = etd2.make_backend

    def recording(*args):
        be = make(*args)
        stage = be.diag_factors

        def diag_factors(*a):
            seen.append((ThreadpoolController().info(), mkl_threads()))
            return stage(*a)

        be.diag_factors = diag_factors
        return be

    monkeypatch.setattr(etd2, "make_backend", recording)

    def threads(info, api):
        return {p["num_threads"] for p in info if p["internal_api"] == api}

    def mkl_threads():
        lib = config.mkl
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


def test_skewed_cluster_basis_falls_back_to_schur():
    """``S_P = 1.1 I`` on the coupled modes: every basis is an eigenbasis.
    A skewed one turns the l-dependent damping into an exactly integrated
    growth (diagonal weights 10 and -9); ``compile_operator`` replaces it by
    the orthogonal Schur basis and the step stays on the exponential."""
    from MCEq.operators.compiled import compile_operator, secant_coupling

    n, n_k = 2, 3
    blk = np.diag([-1.0, -2.0]) + np.diag([0.3], -1)
    damp = [0.0, 5.0, 5.0]  # multiple-scattering damping of modes l = 0, 1, 2
    int_m = sl.block_diag(*[blk - d * np.eye(n) for d in damp])
    dec_m = 0.1 * sl.block_diag(*[np.diag([-1.0, -0.5])] * n_k)
    T_P = np.array([[0.1, 0.0, 0.05], [0.0, 0.1, 0.02]])
    V = np.array([[1.0, 3.0], [0.3, 1.0]])
    ops = {"P": np.arange(2), "low_e_idx": np.arange(n), "n_k": n_k, "T_P": T_P,
           "T_PP": T_P[:, :2], "V": V, "Vi": np.linalg.inv(V), "lam": np.full(2, 1.1)}  # fmt: skip
    op = compile_operator(sp.csr_matrix(int_m), sp.csr_matrix(dec_m), ops)
    c = op.coupling
    np.testing.assert_allclose(c.V.T @ c.V, np.eye(2), atol=1e-14)
    assert op.exact_slot_diagonals[0].max() <= 0
    op.coupling = secant_coupling(ops)
    assert op.exact_slot_diagonals[0].max() > 40  # the skewed eigenbasis

    T = np.zeros((n_k, n_k))
    T[:2] = T_P
    S = np.eye(n_k * n) + np.kron(T, np.eye(n))
    phi0 = np.random.default_rng(4).random(n_k * n)
    ref = sl.expm((int_m + 0.5 * dec_m) @ S * 2.0) @ phi0
    nsteps = 400
    dX = np.full(nsteps, 2.0 / nsteps)
    sol, _ = solve_etd2(
        nsteps, dX, np.full(nsteps, 0.5), sp.csr_matrix(int_m), sp.csr_matrix(dec_m),
        phi0, sec_ops=ops,
    )  # fmt: skip
    np.testing.assert_allclose(sol, ref, rtol=1e-4)



@pytest.mark.parametrize("backend", ["numpy", CUDA])
def test_stiff_mode_damping_needs_own_column_bases(backend):
    """A damping far above 1/h on the coupled columns of one species (e+-
    multiple scattering, up to ~4e3 cm^2/g at 1 MeV) is integrated exactly in
    per-column bases of ``diag(d_i) S_P``; in the shared basis of ``S_P`` the
    stiff remainder is explicit and the step diverges. Species 0 also decays
    (low-energy muons): the decay diagonal stays explicit. Dense l grid, as
    the coupled block of the production grid (whose spectra are real)."""
    from MCEq.operators.compiled import column_bases, compile_operator

    l_grid, n_sp, n_e = np.arange(8), 2, 5
    n, n_k = n_sp * n_e, 8
    rng = np.random.default_rng(7)
    ops = build_secant_kernel_ops(l_grid, np.logspace(-2, 1, n_e), n_sp, 75.0, e_max=0.5)
    blocks, decs = [], []
    for ell in l_grid:
        m = sp.random(n, n, density=0.3, random_state=rng).toarray() * 0.2
        np.fill_diagonal(m, -(1 + rng.random(n)))
        m[np.arange(n_e), np.arange(n_e)] -= 40.0 * ell * (ell + 1)  # species 0
        blocks.append(m)
        d = sp.random(n, n, density=0.2, random_state=rng).toarray() * 0.1
        np.fill_diagonal(d, -0.5 * (1 + rng.random(n)))
        decs.append(d)
    hadr, dec = sl.block_diag(*blocks), sl.block_diag(*decs)
    T = np.zeros((n_k, n_k))
    T[ops["P"][:, None], np.arange(n_k)[None, :]] = ops["T_P"]
    low = np.zeros(n)
    low[ops["low_e_idx"]] = 1.0
    S = np.eye(n_k * n) + np.kron(T, np.diag(low))
    phi0 = rng.random(n_k * n)
    ri = 0.4
    ref = sl.expm((hadr + ri * dec) @ S) @ phi0
    cols = np.arange(n_e)

    op = compile_operator(sp.csr_matrix(hadr), sp.csr_matrix(dec), ops, exact_columns=cols)
    cb = op.column_basis
    assert len(cb.idx) == np.isin(ops["low_e_idx"], cols).sum() > 0
    assert np.all(cb.lam < 0)
    assert np.array_equal(op.exact_slot_diagonals[0][:, cb.idx], cb.lam)
    assert column_bases(op, [n - 1 + n]) is None  # not a state column of one block

    def solve(nsteps, exact):
        dX = np.full((nsteps, 1), 1.0 / nsteps)
        sol, _ = solve_etd2(
            nsteps, dX, np.full_like(dX, ri), sp.csr_matrix(hadr), sp.csr_matrix(dec),
            phi0[:, None], backend=backend, sec_ops=ops,
            exact_columns=cols if exact else None,
        )
        return sol[:, 0]

    scale = np.abs(ref).max()
    err = [np.abs(solve(k, True) - ref).max() / scale for k in (200, 400, 800, 1600)]
    with np.errstate(all="ignore"):
        shared = [np.abs(solve(k, False) - ref).max() / scale for k in (200, 1600)]
    # stiff order reduction: ~2.3-2.6 per halving (6.7e-4 ... 4.6e-5); the
    # shared basis is 2.4 at 200 steps and 3.3e-2 at 1600
    assert err[-1] < 1e-4
    assert all(a / b > 2.0 for a, b in zip(err, err[1:]))
    assert not np.isfinite(shared[0]) or shared[0] > 100 * err[0]
    assert not np.isfinite(shared[1]) or shared[1] > 100 * err[-1]
