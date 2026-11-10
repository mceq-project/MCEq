"""Tangent columns of the ETD2 driver against finite differences of the
cascade, on a toy operator and on the CI databases."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.operators.compiled import compile_operator
from MCEq.operators.legendre import backward_removal, dense_count
from MCEq.solvers import etd2_driver, make_backend
from MCEq.solvers.etd2 import TangentOperator

DIM = 12


@pytest.fixture(scope="module")
def toy():
    """Upper-triangular interaction and decay operators (energy flows down),
    an off-diagonal and a diagonal derivative, a density profile."""
    rng = np.random.default_rng(1)

    def triangular(scale, diagonal):
        m = np.triu(rng.uniform(0.0, scale, (DIM, DIM)), 1)
        np.fill_diagonal(m, diagonal)
        return m

    return SimpleNamespace(
        A=triangular(0.05, -rng.uniform(0.5, 1.5, DIM)),
        B=triangular(0.02, -rng.uniform(0.1, 0.4, DIM)),
        G_off=np.triu(rng.uniform(0.0, 0.05, (DIM, DIM)), 1),
        G_diag=np.diag(rng.uniform(-0.3, 0.3, DIM)),
        phi0=rng.uniform(0.5, 1.0, DIM),
        phi1=rng.uniform(0.5, 1.0, DIM),
    )


def _solve(toy, A, nsteps, phi, tangent=None):
    depth = 3.0
    dX = np.full(nsteps, depth / nsteps)
    rho_inv = 0.5 + 0.5 * (np.arange(nsteps) + 0.5) * depth / nsteps
    op = compile_operator(sp.csr_matrix(A), sp.csr_matrix(toy.B), None)
    be = make_backend(op, "numpy", np.float64, 0)
    try:
        return etd2_driver(nsteps, dX, rho_inv, be, phi, [], tangent=tangent)[0]
    finally:
        be.close()


def _tangent_state(toy, tangent, passive=()):
    x = np.zeros((DIM, tangent.columns))
    x[:, 0] = toy.phi0
    for j, p in enumerate(passive):
        x[:, 1 + tangent.derivative_columns + j] = p
    return x


def test_cascade_and_passive_columns(toy):
    t = TangentOperator([toy.G_off, toy.G_diag], curvature=True, extra=1)
    assert t.columns == 1 + 4 + 1
    sol = _solve(toy, toy.A, 50, _tangent_state(toy, t, [toy.phi1]), t)
    np.testing.assert_allclose(sol[:, 0], _solve(toy, toy.A, 50, toy.phi0), rtol=1e-14)
    np.testing.assert_allclose(sol[:, -1], _solve(toy, toy.A, 50, toy.phi1), rtol=1e-14)


def test_off_diagonal_derivatives_equal_finite_differences(toy):
    """Without a diagonal, the tangent columns are the derivatives of the
    ETD2 solution itself, at any step size."""
    t = TangentOperator([toy.G_off], curvature=True)
    sol = _solve(toy, toy.A, 20, _tangent_state(toy, t), t)

    def differences(h):
        plus, minus, mid = (
            _solve(toy, toy.A + s * h * toy.G_off, 20, toy.phi0) for s in (1, -1, 0)
        )
        return (plus - minus) / (2 * h), (plus - 2 * mid + minus) / h**2

    first, second = differences(1e-4)[0], differences(1e-2)[1]
    np.testing.assert_allclose(sol[:, 1], first, rtol=1e-8)
    np.testing.assert_allclose(sol[:, 2], second, rtol=1e-5, atol=1e-9 * second.max())


def test_diagonal_derivative_converges_with_the_step(toy):
    """With a diagonal, tangent and finite difference differ by the step
    error, which falls as h^2."""

    def gap(nsteps):
        t = TangentOperator([toy.G_diag])
        tangent = _solve(toy, toy.A, nsteps, _tangent_state(toy, t), t)[:, 1]
        h = 1e-5
        fd = (
            _solve(toy, toy.A + h * toy.G_diag, nsteps, toy.phi0)
            - _solve(toy, toy.A - h * toy.G_diag, nsteps, toy.phi0)
        ) / (2 * h)
        return np.max(np.abs(tangent - fd)) / np.max(np.abs(fd))

    coarse, fine = gap(40), gap(160)
    assert coarse < 1e-3
    assert 12.0 < coarse / fine < 20.0


def test_binding_is_kept_per_operator_and_dtype(toy):
    G = sp.csr_matrix(toy.G_off)
    t = TangentOperator([G])
    G.data *= 10.0  # the operator keeps its own copy
    np.testing.assert_array_equal(t._mats[0].toarray(), toy.G_off)
    op = compile_operator(sp.csr_matrix(toy.A), sp.csr_matrix(toy.B), None)
    be = make_backend(op, "numpy", np.float64, 0)
    try:
        sources = t.bind(be, np.float64)
        assert t.bind(be, np.float64) is sources
        assert t.bind(be, np.float32) is not sources
    finally:
        be.close()


def test_solve_batch_tangent_columns(mceq_sib21):
    run = mceq_sib21
    M0 = run.int_m
    p = run.pman[2212]
    pi = run.pman[211]
    G = sp.lil_matrix(M0.shape)
    G[pi.lidx : pi.uidx, p.lidx : p.uidx] = M0[pi.lidx : pi.uidx, p.lidx : p.uidx]
    G = G.tocsr()
    t = TangentOperator([G], extra=1)
    phi0 = np.stack([run._phi0, 2.0 * run._phi0], axis=1)
    res = run.solve_batch(phi0=phi0)
    tan = run.solve_batch(phi0=phi0, tangent=t)
    assert tan.sol.shape == (run.dim_states, 3)
    np.testing.assert_allclose(tan.sol[:, 0], res.sol[:, 0], rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(tan.sol[:, 2], res.sol[:, 1], rtol=1e-12, atol=1e-300)
    h = 1e-3
    try:
        run.int_m = M0 + h * G
        plus = run.solve_batch().sol[:, 0]
        run.int_m = M0 - h * G
        minus = run.solve_batch().sol[:, 0]
    finally:
        run.int_m = M0
    fd = (plus - minus) / (2 * h)
    np.testing.assert_allclose(
        tan.sol[:, 1], fd, rtol=1e-7, atol=1e-9 * np.abs(fd).max()
    )

    with pytest.raises(ValueError, match="phi0 with 2 columns"):
        run.solve_batch(phi0=run._phi0, tangent=t)
    with pytest.raises(ValueError, match="one cascade"):
        run.solve_batch(conditions=[{"zenith_deg": 0.0}] * 2, tangent=t)


_CONFIG_2D = (
    "mceq_db_fname",
    "e_min",
    "e_max",
    "secant_theta_transport",
)


@pytest.fixture(scope="module")
def mceq_2d_secant():
    """The 2D CI database with the sec(theta) transport: the Legendre
    backward-removal mixing and the coupled corner."""
    from MCEq.core import MCEqRun

    fn = "mceq_ci_base_air_2d_legendre_v2.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; python -m MCEq.data.download --file {fn}")
    saved = {name: getattr(config, name) for name in _CONFIG_2D}
    config.mceq_db_fname = fn
    config.e_min, config.e_max = 1e-1, 1e2
    config.secant_theta_transport = True
    try:
        yield MCEqRun(
            interaction_model="FLUKA20251",
            primary_model=None,
            theta_deg=75.0,
            density_model=("CORSIKA", ("USStd", None)),
        )
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def test_legendre_secant_off_diagonal_equals_finite_differences(mceq_2d_secant):
    run = mceq_2d_secant
    sec_ops = run._resolve_secant()
    assert sec_ops is not None and config.legendre_backward_removal
    M0 = sp.csr_matrix(run.matrix_builder.int_m_hadr)
    ds, n_k = run.dim_states, M0.shape[0] // run.dim_states

    def rows(pdg):
        p = run.pman[pdg]
        return np.concatenate([np.arange(p.lidx, p.uidx) + k * ds for k in range(n_k)])

    C = M0.tocoo()
    sel = np.isin(C.row, rows(211)) & np.isin(C.col, rows(2212)) & (C.row != C.col)
    G = sp.csr_matrix((C.data[sel], (C.row[sel], C.col[sel])), shape=M0.shape)
    mix = SimpleNamespace(
        M=backward_removal(dense_count(run._mceq_db.k_grid) - 1),
        n_D=dense_count(run._mceq_db.k_grid),
        N=ds,
    )
    run._calculate_integration_path(None, "X")
    nsteps, dX, rho_inv, _ = run.integration_path
    nsteps = min(nsteps, 300)
    phi0 = run._stack_modes(run._phi0)

    def solve(M, phi, tangent=None):
        op = compile_operator(
            M, run.dec_m, sec_ops, band=run.matrix_builder.dEdx_band, mode_mix=mix
        )
        be = make_backend(op, "numpy", np.float64, 0)
        try:
            return etd2_driver(
                nsteps, dX[:nsteps], rho_inv[:nsteps], be, phi, [], tangent=tangent
            )[0]
        finally:
            be.close()

    t = TangentOperator([G])
    sol = solve(M0, np.stack([phi0, np.zeros_like(phi0)], axis=1), t)
    np.testing.assert_allclose(sol[:, 0], solve(M0, phi0), rtol=1e-12, atol=1e-300)
    h = 1e-3
    fd = (solve(M0 + h * G, phi0) - solve(M0 - h * G, phi0)) / (2 * h)
    np.testing.assert_allclose(sol[:, 1], fd, atol=1e-9 * np.abs(fd).max())
