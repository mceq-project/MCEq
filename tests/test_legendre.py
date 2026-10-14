"""Legendre angular modes: Funk-Hecke composition, flat-sky limit, couplings."""

import numpy as np
import pytest
import scipy.special as sp
from scipy.integrate import quad

from MCEq.operators.legendre import (
    l_interpolation,
    legendre_p,
    path_elongation,
    series,
    theta_density,
)
from MCEq.operators.secant import (
    _CLUSTER_RTOL,
    build_secant_kernel_ops,
    legendre_coupling,
)
from tests.helpers.legendre import moments

# Sparse grid from 0 to 2000 (48 modes); its tail above 64 is the sparse part
# of the production grid.
K_GRID = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 28, 33, 38, 43,
     50, 58, 67, 77, 89, 102, 118, 136, 156, 180, 208, 239, 276, 318, 366, 422,
     486, 560, 645, 743, 856, 986, 1136, 1308, 1507, 1736, 2000]
)  # fmt: skip

# Legendre l grid of the 90-mode database: dense l <= 64, sparse above.
L_GRID = np.concatenate([np.arange(65), K_GRID[K_GRID > 64]])


def hg_sample(rng, g, n):
    """``cos alpha`` from the Henyey-Greenstein kernel; ``k_l = g^l``."""
    s = (1 - g * g) / (1 - g + 2 * g * rng.random(n))
    return (1 + g * g - s * s) / (2 * g)


def vmf_sample(rng, kappa, n):
    """``cos alpha`` from the von Mises-Fisher kernel on the sphere."""
    xi = rng.random(n)
    return 1 + np.log(xi + (1 - xi) * np.exp(-2 * kappa)) / kappa


def vmf_moments(kappa, ell):
    """``<P_l(cos alpha)> = I_{l+1/2}(kappa) / I_{1/2}(kappa)``."""
    return sp.ive(ell + 0.5, kappa) / sp.ive(0.5, kappa)


def rotate(mu, cos_alpha, rng):
    """Polar cosine of a direction at ``alpha`` to ``mu``, uniform azimuth."""
    phi = 2 * np.pi * rng.random(mu.size)
    sin_a = np.sqrt(np.clip(1 - cos_alpha**2, 0, None))
    return mu * cos_alpha + np.sqrt(np.clip(1 - mu**2, 0, None)) * sin_a * np.cos(phi)


def test_recurrence_matches_scipy():
    mu = np.linspace(-1, 1, 101)
    p = legendre_p(2000, mu)
    for ell in (0, 1, 2, 7, 50, 333, 2000):
        np.testing.assert_allclose(p[ell], sp.eval_legendre(ell, mu), atol=1e-12)


def test_sparse_moments_match_table():
    mu = np.random.default_rng(1).uniform(-1, 1, 1000)
    w = np.random.default_rng(2).random(1000)
    ell = np.array([0, 3, 3, 17, 64, 1])
    ref = legendre_p(64, mu)[ell] @ w
    np.testing.assert_allclose(moments(mu, ell, w, chunk=77), ref, rtol=1e-12)


def test_series_round_trip():
    """A degree-6 polynomial is reproduced by its first 7 moments."""
    x, w = np.polynomial.legendre.leggauss(16)

    def f(mu):
        return 1 + mu - 2 * mu**3 + 0.5 * mu**6

    a = moments(x, np.arange(7), w * f(x))
    mu = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(series(a, mu), f(mu), atol=1e-12)


@pytest.mark.parametrize("g", [0.2, 0.6, 0.95])
def test_henyey_greenstein_kernel_moments(g):
    """``k_l`` by quadrature of the kernel density is ``g^ell``."""
    x, w = np.polynomial.legendre.leggauss(4000)
    density = 0.5 * (1 - g * g) / (1 + g * g - 2 * g * x) ** 1.5
    ell = np.arange(31)
    np.testing.assert_allclose(moments(x, ell, w * density), g**ell, atol=1e-10)


@pytest.mark.parametrize("parent", ["hg-axis", "uniform-hemisphere"])
def test_funk_hecke_composition_mc(parent):
    """Composition on the sphere multiplies the moments: a_l(s) = k_l a_l(p).

    The parent is either narrow-ish about the axis (HG, g=0.7) or spread over
    the forward hemisphere; the kernel (HG, g=0.6) sends a sizeable fraction
    backward, where planar composition fails.
    """
    rng = np.random.default_rng(42)
    n = 400_000
    ell = np.arange(41)
    if parent == "hg-axis":
        mu_p = hg_sample(rng, 0.7, n)
        a_p = 0.7**ell
    else:
        mu_p = rng.random(n)
        x, w = np.polynomial.legendre.leggauss(64)
        a_p = legendre_p(40, 0.5 * x + 0.5) @ (0.5 * w)  # int_0^1 P_l dmu
    mu_s = rotate(mu_p, hg_sample(rng, 0.6, n), rng)
    a_s = moments(mu_s, ell) / n
    sigma = legendre_p(40, mu_s).std(axis=1) / np.sqrt(n)
    assert np.all(np.abs(a_s - 0.6**ell * a_p) < 5 * sigma + 1e-12)


def test_hilb_limit():
    """``P_l(cos t) = sqrt(t / sin t) J_0((ell + 1/2) t) + O(ell^-3/2)`` on
    ``0 < t <= pi/2``, and ``P_l(cos t) -> J_0((ell + 1/2) t)`` in the core
    ``(ell + 1/2) t <= 5`` with an ``O(t^2)`` remainder."""
    t = np.linspace(1e-6, np.pi / 2, 20001)
    for ell in (10, 100, 1000, 2000):
        p = legendre_p(ell, np.cos(t))[ell]
        j = sp.j0((ell + 0.5) * t)
        assert np.abs(p - np.sqrt(t / np.sin(t)) * j).max() < 0.07 * ell**-1.5
        core = (ell + 0.5) * t <= 5
        assert np.abs(p - j)[core].max() < 0.03 * (5 / (ell + 0.5)) ** 2


@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e6])
def test_narrow_cone_moments_match_hankel(kappa):
    """A von Mises-Fisher cone of width ``sigma = kappa^-1/2`` (5.7, 0.57,
    0.057 deg): its Legendre moments equal the Hankel modes of the planar
    Gaussian, ``exp(-kappa_H^2 sigma^2 / 2)`` at ``kappa_H = ell + 1/2``, to
    ``O(sigma^2)``."""
    ell = K_GRID
    lo = max(-1.0, 1 - 40 / kappa)
    x, w = np.polynomial.legendre.leggauss(4000)
    mu = 0.5 * (1 - lo) * x + 0.5 * (1 + lo)
    density = kappa * np.exp(kappa * (mu - 1)) / (1 - np.exp(-2 * kappa))
    a = moments(mu, ell, 0.5 * (1 - lo) * w * density)
    np.testing.assert_allclose(a, vmf_moments(kappa, ell), atol=1e-10)
    hankel = np.exp(-((ell + 0.5) ** 2) / (2 * kappa))
    assert np.abs(a - hankel).max() < 0.13 / kappa


def _quad_coupling(ell, lp, g, lo=0.0, points=None):
    def f(mu):
        return sp.eval_legendre(ell, mu) * g(mu) * sp.eval_legendre(lp, mu)

    return (2 * lp + 1) / 2 * quad(f, lo, 1, points=points, limit=400)[0]


@pytest.mark.parametrize("cap", [60.0, 75.0, 89.0])
def test_path_elongation(cap):
    """``G`` against direct quadrature of ``P_l g P_l'`` with the kink at
    ``mu_cap``, and its action on a polynomial."""
    l_max = 48
    g_mat = path_elongation(l_max, cap)
    mu_cap = np.cos(np.radians(cap))

    def g(mu):
        return 1 / np.maximum(mu, mu_cap)

    for ell, lp in [(0, 0), (0, 1), (1, 0), (3, 9), (20, 21), (48, 48), (2, 48)]:
        ref = _quad_coupling(ell, lp, g, lo=-1.0, points=[mu_cap])
        assert g_mat[ell, lp] == pytest.approx(ref, rel=1e-9, abs=1e-11)
    x, w = np.polynomial.legendre.leggauss(64)

    def f(mu):
        return 2 - mu + mu**4

    a = moments(x, np.arange(l_max + 1), w * f(x))
    ref = [
        quad(lambda mu, ell=ell: sp.eval_legendre(ell, mu) * g(mu) * f(mu), -1, 1,
             points=[mu_cap], limit=400)[0]
        for ell in range(l_max + 1)
    ]  # fmt: skip
    np.testing.assert_allclose(g_mat @ a, ref, atol=1e-10)


def test_secant_without_elongation_is_identity():
    np.testing.assert_allclose(path_elongation(30, 0.0), np.eye(31), atol=1e-13)
    with pytest.raises(ValueError):
        path_elongation(30, 90.0)


def test_secant_on_collimated_state():
    """A beam along the axis (``a_l = 1`` for all ``ell``) is not elongated,
    ``g(1) = 1``: the low moments of the truncated ``G 1`` converge to 1."""
    err = []
    for l_max in (64, 256, 1024):
        ga = path_elongation(l_max, 75.0) @ np.ones(l_max + 1)
        err.append(np.abs(ga[:11] - 1).max())
    assert err[0] < 1e-2
    assert err[1] < err[0] / 4
    assert err[2] < 1e-3


def test_l_interpolation():
    """Not-a-knot cubic spline: exact on cubics, the identity on a dense grid."""
    l_all = np.arange(L_GRID[-1] + 1)
    cubic = 3.0 - 2e-2 * l_all + 1e-5 * l_all**2 - 2e-9 * l_all**3
    m = l_interpolation(L_GRID, l_all)
    np.testing.assert_allclose(m @ cubic[L_GRID], cubic, rtol=1e-11)
    np.testing.assert_array_equal(
        l_interpolation(np.arange(9), np.arange(9)), np.eye(9)
    )


def test_wide_secant_coupling_on_narrow_cones():
    """The dense rows of ``T`` reach the sparse modes: on cones down to 0.1 deg
    they reproduce the converged ``G`` (rows against l <= 3000) applied to the
    exact moments. ``G`` cut at the dense block misses the top rows by up to
    0.9 a_0 on the same cones."""
    n_d = 65
    t = legendre_coupling(L_GRID, 75.0)
    g_ref = path_elongation(3000, 75.0, n_rows=n_d)
    g_cut = path_elongation(n_d - 1, 75.0)
    for sigma in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0):
        kappa = 1 / np.radians(sigma) ** 2
        ref = g_ref @ vmf_moments(kappa, np.arange(3001))
        a = vmf_moments(kappa, L_GRID)
        np.testing.assert_allclose(a[:n_d] + t[:n_d] @ a, ref, rtol=0, atol=1e-6)
        if sigma <= 1.0:
            assert np.abs(g_cut @ a[:n_d] - ref).max() > 0.4
    # sparse rows are not elongated; a dense-only grid keeps G - 1
    assert not t[n_d:].any()
    np.testing.assert_allclose(
        legendre_coupling(np.arange(n_d), 75.0), g_cut - np.eye(n_d), atol=1e-13
    )


@pytest.mark.parametrize(
    "cap", [0.0, 0.5, 1.0, 5.0, 7.4, 7.5, 7.7, 20.0, 45.0, 50.0, 75.0, 89.0]
)
def test_wide_secant_eigenbasis(cap):
    """``S_P = I + T_PP`` has eigenvalues near ``1/mu_cap`` ~60-fold. At caps
    7.4-7.8 deg the cluster splits into conjugate pairs ``a ± ib`` with
    ``b/a`` up to 2.4e-7; the real basis keeps their 2x2 rotation, and the
    spread of a cluster, off the diagonal and stays well conditioned, down to
    ``S_P = I`` at cap 0."""
    ops = build_secant_kernel_ops(L_GRID, np.logspace(-1, 3, 5), 2, cap)
    v, vi, lam = ops["V"], ops["Vi"], ops["lam"]
    s_p = np.eye(len(lam)) + ops["T_PP"]
    assert np.linalg.cond(v) < 1e4
    np.testing.assert_allclose(v @ vi, np.eye(len(lam)), rtol=0, atol=1e-12)
    rest = vi @ s_p @ v - np.diag(lam)
    assert np.abs(rest).max() < _CLUSTER_RTOL * lam.max()
    if cap == 7.5:
        assert np.abs(rest).max() > 1e-8  # a split pair is present
    # g lies in [1, 1/mu_cap]; truncated to the dense block, the spectrum
    # overshoots 1/mu_cap by 0.4 % at most (cap 20 deg).
    assert 1 - 1e-9 < lam.min() <= lam.max() < 1.005 / np.cos(np.radians(cap))
    if cap == 0.0:
        assert np.abs(ops["T_P"]).max() < 1e-7


@pytest.mark.parametrize("cap", [7.5, 75.0])
def test_secant_basis_independent_of_blas_threads(cap):
    """The degenerate cluster at ``1/mu_cap`` (33 of 65 modes at 75 deg) and
    the conjugate pairs at 7.5 deg get a canonical basis: one and four BLAS
    threads give the same ``V`` to round-off. LAPACK's own eigenvectors
    differed by O(1), which moved 2D solutions by up to 2 %."""
    from threadpoolctl import threadpool_limits

    bases = []
    for threads in (1, 4):
        with threadpool_limits(limits=threads):
            ops = build_secant_kernel_ops(L_GRID, [1.0], 1, cap)
        bases.append(ops["V"])
    np.testing.assert_allclose(bases[0], bases[1], rtol=0, atol=1e-9)


@pytest.mark.slow
def test_wide_secant_cap_sweep():
    """Every cap in [0, 90) on a 0.1 deg lattice builds on the 90-mode grid
    (about 5 min)."""
    for cap in np.arange(0.0, 90.0, 0.1):
        ops = build_secant_kernel_ops(L_GRID, [1.0], 1, cap)
        assert np.linalg.cond(ops["V"]) < 1e4


@pytest.mark.parametrize(
    "cones",
    [
        [(0.9, 1.0), (0.08, 5.0), (0.015, 15.0), (0.005, 40.0)],
        [(0.9, 0.5), (0.08, 3.0), (0.015, 12.0), (0.005, 35.0)],
        [(0.85, 2.0), (0.12, 7.0), (0.025, 20.0), (0.005, 45.0)],
        [(0.93, 0.25), (0.05, 2.0), (0.015, 10.0), (0.005, 30.0)],
    ],
)
def test_theta_density_wide_tail(cones):
    """Narrow core with tails at 1e-7..1e-4 of the peak (``(weight, sigma
    deg)`` vMF cones), moments on the sparse grid only: the readout follows
    the tail at 20-60 deg to 3 %. Linear interpolation in ``l`` misses it by
    factors of 2-10 and goes negative."""
    theta = np.radians(np.linspace(20, 60, 81))
    kappas = [(w, 1 / np.radians(s) ** 2) for w, s in cones]
    a = sum(w * vmf_moments(k, L_GRID) for w, k in kappas)
    f_mu = sum(w * k * np.exp(k * (np.cos(theta) - 1)) / -np.expm1(-2 * k)
               for w, k in kappas)  # fmt: skip
    truth = f_mu * np.sin(theta) / theta
    np.testing.assert_allclose(theta_density(a, L_GRID, theta), truth, rtol=3e-2)


@pytest.mark.parametrize("log_theta", [False, True])
def test_convert_to_theta_space_reads_one_species(log_theta):
    """``convert_to_theta_space`` slices the species out of each depth's
    ``(n_k, dim_states)`` modes and reads them out with ``theta_density``;
    a 5-degree cone integrates to its ``a_0`` in ``theta dtheta``."""
    from types import SimpleNamespace

    from MCEq.core import MCEqRun

    n_e, kappa = 3, 1 / np.radians(5.0) ** 2
    cone = vmf_moments(kappa, L_GRID)
    state = np.zeros((len(L_GRID), 2 * n_e))
    state[:, n_e:] = np.outer(cone, [1.0, 2.0, 4.0])
    run = SimpleNamespace(
        _mceq_db=SimpleNamespace(k_grid=L_GRID),
        e_grid=np.ones(n_e),
        pman=SimpleNamespace(pdg2mceqidx={(14, 0): 1}),
    )
    l_grid, amps, theta, f = MCEqRun.convert_to_theta_space(
        run, [state, 0.5 * state], 14, 0, theta_res=4000, log_theta=log_theta
    )
    np.testing.assert_array_equal(l_grid, L_GRID)
    assert np.asarray(f).shape == (2, n_e, 4000)
    np.testing.assert_array_equal(amps[1][2], 2.0 * cone)
    np.testing.assert_allclose(f[0][1], theta_density(2.0 * cone, L_GRID, theta))
    norm = np.trapezoid(np.asarray(f) * theta, theta, axis=-1)
    np.testing.assert_allclose(norm, [[1, 2, 4], [0.5, 1, 2]], rtol=2e-3)
