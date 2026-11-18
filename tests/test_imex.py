"""IMEX solver (:mod:`MCEq.solvers.imex`) on small synthetic cascades and against ETD2 on the CI databases."""

import os

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import expm

from MCEq import config
from MCEq.solvers.imex import (
    GAMMA,
    ImexOperator,
    ImexStepper,
    imex_driver,
)

N_BINS = 8


def _matrices(n_k=1, cycle=True, seed=1):
    """A light species with a stiff loss band and a long-lived hadron that produces it and decays into it.

    Per mode: the hadron's production of light states is mode dependent (explicit part); with ``cycle`` the
    loss band reaches one bin up (a cycle the sweep leaves to GMRES). For ``n_k > 1`` the leading two modes are
    mixed and the low-energy states coupled across modes.
    """
    rng = np.random.default_rng(seed)
    N = 2 * N_BINS
    light = np.r_[np.ones(N_BINS, bool), np.zeros(N_BINS, bool)]
    e_tot = np.tile(np.logspace(0, 3, N_BINS), 2)
    band = np.zeros((N, N))
    for k in range(N_BINS):
        band[k, k] = -400.0
        if k + 1 < N_BINS:
            band[k, k + 1] = 380.0  # loss: from bin k + 1 down into bin k
        if cycle and k > 0:
            band[k, k - 1] = 8.0  # stencil entry reaching one bin up
    ints, decs = [], []
    for mode in range(n_k):
        a = np.zeros((N, N))
        b = np.zeros((N, N))
        for k in range(N_BINS):
            h = N_BINS + k
            a[k, k] = -0.03 - 0.01 * mode  # light interaction (+ mode damping)
            a[:k, k] = 0.002  # light -> light, downward
            a[h, h] = -0.02
            a[N_BINS:h, h] = 0.001  # hadron redistribution, downward (implicit)
            a[:k, h] = 0.004 * rng.random(k) / (1 + mode)  # hadron -> light (explicit)
            b[h, h] = -50.0
            b[min(k + 1, N_BINS - 1), h] = 45.0  # decay feeds light one bin up
        ints.append(a)
        decs.append(b)
    int_m = sp.block_diag([sp.csr_matrix(a) for a in ints], format="csr")
    dec_m = sp.block_diag([sp.csr_matrix(b) for b in decs], format="csr")
    band_m = sp.block_diag([sp.csr_matrix(band)] * n_k, format="csr")
    mix = coupling = None
    if n_k > 1:
        mix = np.array([[1.0, 0.0], [0.2, 0.8]])
        T_P = 0.3 * rng.random((2, n_k))
        coupling = (T_P, np.r_[0, 1, 2, N_BINS, N_BINS + 1])
    species = np.arange(N) // N_BINS
    return (int_m, dec_m, band_m, light, species, n_k, mix, coupling), e_tot


def _cascade(n_k=1, cycle=True, xp=np):
    """:func:`_matrices` as an :class:`ImexOperator` and the total energies."""
    args, e_tot = _matrices(n_k, cycle)
    return ImexOperator(*args, xp=xp), e_tot


def _dense(fn, op):
    """Dense matrix of the linear map ``fn`` on mode-major states."""
    dim = op.n_k * op.N
    cols = [fn(np.eye(dim)[j].reshape(op.n_k, op.N)).ravel() for j in range(dim)]
    return np.array(cols).T


@pytest.mark.parametrize("n_k", [1, 3])
def test_split_explicit_is_hadronic(n_k):
    (int_m, _, _, light, species, *_), _ = _matrices(n_k)
    op, _ = _cascade(n_k)
    off = int_m - sp.diags(int_m.diagonal())
    assert abs(op.int_imp + op.int_exp - off).max() == 0
    imp, exp = op.int_imp.tocoo(), op.int_exp.tocoo()

    def stiff(m):
        r, c = m.row % op.N, m.col % op.N
        return (species[r] == species[c]) | (light[r] & light[c])

    assert np.all(stiff(imp))
    assert not np.any(stiff(exp))
    assert np.any(
        species[imp.row % op.N] == 1
    )  # the hadron's own redistribution is implicit


@pytest.mark.parametrize("n_k", [1, 3])
def test_sweep_is_exact_without_cycles(n_k):
    op, e_tot = _cascade(n_k, cycle=False)
    stepper = ImexStepper(op, e_tot)
    c, ri = GAMMA * 2.0, 0.1
    stepper.sweep.prepare(c, ri)
    system = np.eye(op.n_k * op.N) - c * _dense(lambda x: op.implicit(x, ri), op)
    b = np.random.default_rng(0).random((op.n_k, op.N))
    exact = np.linalg.solve(system, b.ravel())
    np.testing.assert_allclose(stepper.sweep.solve(b).ravel(), exact, rtol=1e-12)


@pytest.mark.parametrize("n_k", [1, 3])
@pytest.mark.parametrize("h", [0.1, 10.0])
def test_stage_solve_with_cycles(n_k, h):
    op, e_tot = _cascade(n_k)
    stepper = ImexStepper(op, e_tot, rtol=1e-11)
    c, ri = GAMMA * h, 0.1
    stepper.sweep.prepare(c, ri)
    system = np.eye(op.n_k * op.N) - c * _dense(lambda x: op.implicit(x, ri), op)
    b = np.random.default_rng(0).random((op.n_k, op.N))
    x = stepper._solve(b, c, ri, x0=None)
    np.testing.assert_allclose(x.ravel(), np.linalg.solve(system, b.ravel()), rtol=1e-9)
    assert stepper.iterations[-1] < 20


@pytest.mark.parametrize("n_k", [1, 3])
@pytest.mark.parametrize("ri", [0.1, 0.02])
def test_second_order_against_expm(n_k, ri):
    op, e_tot = _cascade(n_k)
    stepper = ImexStepper(op, e_tot, rtol=1e-12)
    full = _dense(lambda x: op.implicit(x, ri) + op.explicit(x, ri), op)
    phi0 = np.zeros(op.n_k * op.N)
    phi0[op.N - 1 :: op.N] = 1.0  # one hadron in the top bin, every mode
    x_end = 0.5  # the hadron decays by e^-2.5 (ri 0.1) or e^-0.5 (ri 0.02)
    exact = expm(x_end * full) @ phi0
    errors = []
    for nsteps in (5, 10, 20):  # h * (stiff rate) = 40, 20, 10
        dX = np.full(nsteps, x_end / nsteps)
        phi, _ = imex_driver(nsteps, dX, np.full(nsteps, ri), stepper, phi0)
        errors.append(np.abs(phi - exact).max() / np.abs(exact).max())
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert errors[-1] < 2e-3
    assert np.all(orders > 1.8)


def test_large_step_stays_bounded():
    op, e_tot = _cascade(3)
    stepper = ImexStepper(op, e_tot)
    phi0 = np.zeros(op.n_k * op.N)
    phi0[op.N - 1 :: op.N] = 1.0
    # h * |stiff rate| = 400: far outside any explicit stability region
    phi, _ = imex_driver(10, np.full(10, 1.0), np.ones(10), stepper, phi0)
    assert np.all(np.isfinite(phi)) and np.abs(phi).max() < 10


def test_snapshots_in_mceq_order():
    op, e_tot = _cascade(3)
    stepper = ImexStepper(op, e_tot)
    phi0 = np.zeros(op.n_k * op.N)
    phi0[op.N - 1 :: op.N] = 1.0
    nsteps = 6
    dX, ri = np.full(nsteps, 0.1), np.ones(nsteps)
    phi, grid = imex_driver(nsteps, dX, ri, stepper, phi0, grid_idcs=[2, 5])
    assert grid.shape == (2, phi0.size)
    np.testing.assert_array_equal(grid[1], phi)
    phi3, _ = imex_driver(3, dX[:3], ri[:3], stepper, phi0)
    np.testing.assert_allclose(grid[0], phi3, rtol=1e-12)


@pytest.mark.parametrize("n_k", [1, 3])
def test_columns_are_independent(n_k):
    """K columns of very different scale give the single-column solutions (tangent columns)."""
    op, e_tot = _cascade(n_k)
    stepper = ImexStepper(op, e_tot, rtol=1e-12)
    phi = np.random.default_rng(2).random((n_k * op.N, 3)) * np.array([1.0, 1e-6, 1e3])
    dX, ri = np.full(4, 0.3), np.full(4, 0.05)
    many, _ = imex_driver(4, dX, ri, stepper, phi)
    for k in range(3):
        one, _ = imex_driver(4, dX, ri, stepper, phi[:, k])
        np.testing.assert_allclose(many[:, k], one, rtol=1e-9)


def _hadronic_derivative(n_k, seed=3):
    """An off-diagonal hadron -> light production matrix per mode (explicit in the split)."""
    rng = np.random.default_rng(seed)
    N = 2 * N_BINS
    blocks = []
    for _ in range(n_k):
        g = np.zeros((N, N))
        for k in range(N_BINS):
            g[:k, N_BINS + k] = 0.01 * rng.random(k)
        blocks.append(sp.csr_matrix(g))
    return sp.block_diag(blocks, format="csr")


@pytest.mark.parametrize("n_k", [1, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_tangent_columns_are_derivatives(n_k, groups):
    """With G in the explicit part the tangent columns are the derivative of the IMEX solution: central finite
    differences of the operator A + p G agree; a passive column is carried unchanged by the sources."""
    from types import SimpleNamespace

    args, e_tot = _matrices(n_k)
    g = _hadronic_derivative(n_k)
    dim = n_k * 2 * N_BINS
    nsteps, dX, ri = 6, np.full(6, 0.3), np.full(6, 0.05)
    rng = np.random.default_rng(4)
    nominal = rng.random((dim, groups))
    passive = rng.random((dim, groups))

    def solve(p):
        op = ImexOperator(args[0] + p * g, *args[1:])
        out, _ = imex_driver(
            nsteps, dX, ri, ImexStepper(op, e_tot, rtol=1e-13), nominal
        )
        return out

    op = ImexOperator(*args)
    stepper = ImexStepper(op, e_tot, rtol=1e-13)
    tangent = SimpleNamespace(_mats=[g], curvature=False, extra=1, groups=groups)
    phi = np.stack([nominal, np.zeros_like(nominal), passive], axis=2).reshape(dim, -1)
    out, _ = imex_driver(nsteps, dX, ri, stepper, phi, tangent=tangent)
    out = out.reshape(dim, groups, 3)
    fd = (solve(1e-4) - solve(-1e-4)) / 2e-4
    np.testing.assert_allclose(out[:, :, 0], solve(0.0), rtol=1e-10)
    np.testing.assert_allclose(
        out[:, :, 1], fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max()
    )
    passive_only, _ = imex_driver(
        nsteps, dX, ri, ImexStepper(op, e_tot, rtol=1e-13), passive
    )
    np.testing.assert_allclose(out[:, :, 2], passive_only, rtol=1e-9)


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
@pytest.mark.parametrize("n_k", [1, 3])
def test_cuda_kernel_matches_array_sweep(n_k):
    """The fp64 kernel is the array sweep to round-off; the default fp32 kernel to fp32 precision."""
    import cupy as cp

    from MCEq.solvers.backends.imex_cuda import CudaLevelKernel

    b = np.random.default_rng(0).random((n_k, 2 * N_BINS, 3))
    op, e_tot = _cascade(n_k)
    host = ImexStepper(op, e_tot).sweep
    host.prepare(GAMMA * 3.0, 0.1)
    expected = host.solve(b)
    gpu = ImexStepper(*_cascade(n_k, xp=cp)).sweep
    gpu.prepare(GAMMA * 3.0, 0.1)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.solve(cp.asarray(b))), expected, rtol=1e-5
    )
    gpu._kernel = CudaLevelKernel(gpu, np.float64)
    gpu.prepare(GAMMA * 3.0, 0.1)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.solve(cp.asarray(b))), expected, rtol=1e-12
    )


def test_mceqrun_imex_matches_etd2(mceq_sib21, monkeypatch):
    """The atmospheric lepton fluxes of the IMEX path agree with ETD2."""
    monkeypatch.setattr(config, "secant_1d_table", None)
    species = ("total_mu+", "total_numu", "total_nue")
    mceq_sib21.solve()
    etd2 = {name: mceq_sib21.get_solution(name) for name in species}
    etd2_steps = mceq_sib21.integration_path[0]

    monkeypatch.setattr(config, "integrator", "imex")
    mceq_sib21.solve()
    assert mceq_sib21.integration_path[0] < etd2_steps
    for name in species:
        ref, sol = etd2[name], mceq_sib21.get_solution(name)
        populated = ref > ref.max() * 1e-6
        np.testing.assert_allclose(sol[populated], ref[populated], rtol=1e-2)


def test_mceqrun_imex_tangent_matches_etd2(mceq_sib21, monkeypatch):
    """solve_batch tangent columns (derivative of pi+ production by protons): IMEX agrees with ETD2."""
    from MCEq.solvers import TangentOperator

    monkeypatch.setattr(config, "secant_1d_table", None)
    m = mceq_sib21
    p, pip = m.pman[2212], m.pman[211]
    g = sp.lil_matrix(m.int_m.shape)
    g[pip.lidx : pip.uidx, p.lidx : p.uidx] = m.int_m[
        pip.lidx : pip.uidx, p.lidx : p.uidx
    ]
    tangent = TangentOperator([g.tocsr()])
    etd2 = m.solve_batch(tangent=tangent)
    monkeypatch.setattr(config, "integrator", "imex")
    imex = m.solve_batch(tangent=tangent)
    for name in ("total_mu+", "total_numu"):
        for k in (0, 1):
            ref, sol = etd2.get_solution(name, k=k), imex.get_solution(name, k=k)
            populated = np.abs(ref) > np.abs(ref).max() * 1e-6
            np.testing.assert_allclose(sol[populated], ref[populated], rtol=2e-2)


def test_mceqrun_imex_refuses_secant_table(mceq_sib21, monkeypatch):
    from MCEq.operators import secant_table

    monkeypatch.setattr(config, "integrator", "imex")
    monkeypatch.setattr(secant_table, "table_for", lambda mceq: object())
    with pytest.raises(NotImplementedError, match="secant table"):
        mceq_sib21.solve()


def test_auto_integrator(mceq_sib21, monkeypatch):
    """ "auto": ETD2 on a hadronic run and in 1D, IMEX on 2D runs below the stiffness threshold, ETD2 with the
    secant table or without a shared path; "imex" refuses a batch without a shared path."""
    from MCEq.operators import secant_table

    m = mceq_sib21
    monkeypatch.setattr(config, "secant_1d_table", None)
    monkeypatch.setattr(config, "integrator", "auto")
    assert m._continuous_loss_dx_cap() > config.imex_auto_dx
    assert m._integrator() == "etd2"
    monkeypatch.setattr(config, "imex_auto_dx", np.inf)
    assert m._integrator() == "etd2"  # 1D
    monkeypatch.setattr(m._mceq_db, "is_2d", True)
    assert m._integrator() == "imex"
    assert m._integrator(shared=False) == "etd2"
    with monkeypatch.context() as ctx:
        ctx.setattr(secant_table, "table_for", lambda mceq: object())
        assert m._integrator() == "etd2"
    monkeypatch.setattr(config, "integrator", "imex")
    with pytest.raises(NotImplementedError, match="shared path"):
        m.solve_batch(conditions=[{"zenith_deg": 0.0}, {"zenith_deg": 30.0}])


@pytest.mark.xdist_group("fluka2d")
def test_mceqrun_imex_2d_matches_etd2(monkeypatch):
    """On the 2D CI database with the sec(theta) coupling, IMEX and ETD2 agree on the angle-integrated muon and
    neutrino spectra of a proton shower."""
    from MCEq.core import MCEqRun

    fn = "mceq_ci_base_air_2d_legendre_v2.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; python -m MCEq.data.download --file {fn}")
    for name, value in {
        "mceq_db_fname": fn,
        "mceq_db_manifest": "mceq_db_manifest_ci_v2.yaml",
        "e_min": 1e-1,
        "e_max": 1e2,
        "muon_multiple_scattering": True,
        "secant_theta_transport": "require",
        "kernel_config": "numpy_etd2",
    }.items():
        monkeypatch.setattr(config, name, value)
    mceq = MCEqRun(
        interaction_model="FLUKA20251",
        primary_model=None,
        theta_deg=60.0,
        density_model=("CORSIKA", ("USStd", None)),
    )
    mceq.set_single_primary_particle(E=50.0, pdg_id=2212)
    species = ("mu+", "numu", "nue")
    mceq.solve()
    etd2 = {name: mceq.get_solution(name) for name in species}
    monkeypatch.setattr(config, "integrator", "imex")
    mceq.solve()
    for name in species:
        ref, sol = etd2[name], mceq.get_solution(name)
        populated = ref > ref.max() * 1e-4
        np.testing.assert_allclose(sol[populated], ref[populated], rtol=2e-2)
