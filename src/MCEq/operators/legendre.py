r"""Legendre angular modes for the 2D transport.

An azimuthally symmetric angular distribution ``f(mu)``, ``mu = cos theta``
about the shower axis, is represented by its Legendre moments::

    a_l = int_{-1}^{1} f(mu) P_l(mu) dmu = <P_l(cos theta)>,
    f(mu) = sum_l (2l + 1)/2 a_l P_l(mu).

Funk-Hecke theorem: a secondary emitted at the angle ``alpha`` to its parent,
with ``alpha`` drawn from a kernel that depends only on the relative angle
and a uniform azimuth about the parent, has

    a_l(secondary) = k_l a_l(parent),   k_l = <P_l(cos alpha)>_kernel,

exactly on the sphere. Production, decay and scattering therefore stay
diagonal in ``l``.

Operations that act on ``mu`` couple the modes through

    M[g]_{ll'} = (2l' + 1)/2 int P_l(mu) g(mu) P_l'(mu) dmu,

which maps the moments of ``f`` to the moments of ``g f``, evaluated by
Gauss-Legendre quadrature on the smooth pieces of ``g``. The transport is

    dN/dX = (-Lambda + B Y) G N,

with ``G`` the path elongation, ``Y`` the mode-diagonal production and ``B``
the removal of produced content with ``mu < 0``. A collimated state has
``a_l = 1`` for every ``l``, so a truncated ``M[g]`` converges on it only if
``g`` is smooth: ``G`` therefore continues ``g`` through ``mu = 0`` and the
step of ``B`` acts only on the production, whose moments decay in ``l``.
"""

import numpy as np

__all__ = [
    "backward_removal",
    "dense_count",
    "forward_weights",
    "hemisphere_projection",
    "jackson_weights",
    "l_interpolation",
    "legendre_p",
    "moments",
    "path_elongation",
    "series",
    "theta_density",
]


def legendre_p(l_max, mu):
    """``P_l(mu)`` for ``l = 0..l_max``, shape ``(l_max + 1, *mu.shape)``.

    Bonnet recurrence ``(l + 1) P_{l+1} = (2l + 1) mu P_l - l P_{l-1}``,
    stable for ``|mu| <= 1``.
    """
    mu = np.asarray(mu, dtype=float)
    p = np.empty((l_max + 1,) + mu.shape)
    p[0] = 1.0
    if l_max > 0:
        p[1] = mu
    for n in range(1, l_max):
        p[n + 1] = ((2 * n + 1) * mu * p[n] - n * p[n - 1]) / (n + 1)
    return p


def moments(mu, ls, weights=None, chunk=1 << 16):
    """``sum_i w_i P_l(mu_i)`` for every ``l`` in the integer array ``ls``.

    The recurrence runs to ``max(l)`` over chunks of ``mu``, so sparse high
    ``l`` sets cost ``O(max(l) n_mu)`` time and ``O(chunk)`` memory per row.
    """
    mu = np.ravel(np.asarray(mu, dtype=float))
    w = np.ones_like(mu) if weights is None else np.ravel(weights).astype(float)
    ls = np.asarray(ls, dtype=int)
    out = np.zeros(ls.shape)
    rows = {int(li): np.flatnonzero(ls == li) for li in np.unique(ls)}
    for lo in range(0, mu.size, chunk):
        x, wx = mu[lo : lo + chunk], w[lo : lo + chunk]
        p_prev, p = np.ones_like(x), x.copy()
        for li in range(int(ls.max()) + 1):
            if li in rows:
                out[rows[li]] += wx @ (p_prev if li == 0 else p)
            if li >= 1:
                p_prev, p = p, ((2 * li + 1) * x * p - li * p_prev) / (li + 1)
    return out


def series(a, mu):
    """``f(mu) = sum_l (2l + 1)/2 a_l P_l(mu)`` from the moments ``a[..., l]``,
    ``l = 0..a.shape[-1] - 1``."""
    a = np.asarray(a, dtype=float)
    ell = np.arange(a.shape[-1])
    return (a * (2 * ell + 1) / 2) @ legendre_p(ell.size - 1, mu)


def forward_weights(l_max):
    """``c_l = (2l + 1)/2 int_0^1 P_l dmu``, so ``int_0^1 f dmu = sum_l c_l a_l``.

    ``int_0^1 P_l dmu = (P_{l-1}(0) - P_{l+1}(0)) / (2l + 1)`` for ``l >= 1``.
    """
    p0 = legendre_p(l_max + 1, 0.0)
    c = np.empty(l_max + 1)
    c[0] = 0.5
    c[1:] = 0.5 * (p0[:l_max] - p0[2:])
    return c


def _coupling(l_max, pieces, extra_nodes, n_rows=None):
    """``M[g]`` with ``g`` given as ``[(lo, hi, g_callable), ...]`` on [-1, 1];
    the first ``n_rows`` rows (all by default) against every ``l' <= l_max``."""
    n_rows = l_max + 1 if n_rows is None else n_rows
    x, w = np.polynomial.legendre.leggauss(l_max + 1 + extra_nodes)
    m = np.zeros((n_rows, l_max + 1))
    for lo, hi, g in pieces:
        mu = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        p = legendre_p(l_max, mu)
        m += (p[:n_rows] * (0.5 * (hi - lo) * w * g(mu))) @ p.T
    return m * ((2 * np.arange(l_max + 1) + 1) / 2)[None, :]


def hemisphere_projection(l_max):
    """``B_{ll'} = (2l' + 1)/2 int_0^1 P_l P_l' dmu``: removes ``mu < 0``.

    Gauss-Legendre with ``l_max + 1`` nodes integrates the degree ``2 l_max``
    integrand exactly.
    """
    return _coupling(l_max, [(0.0, 1.0, np.ones_like)], 0)


def path_elongation(l_max, theta_cap_deg=75.0, extra_nodes=64, n_rows=None):
    """``G = M[g]`` with ``g = 1 / max(mu, mu_cap)`` on ``[-1, 1]``.

    Per unit axis depth a particle at ``theta`` crosses ``sec theta`` more
    matter, capped at ``theta_cap_deg < 90``. ``g`` is constant below
    ``mu_cap``, including the backward hemisphere, which the transport keeps
    empty through ``B``; the quadrature is exact there and ``extra_nodes``
    resolve ``1/mu`` above it. ``n_rows`` keeps only the leading rows, each
    against every column ``l' <= l_max``.
    """
    if not 0 <= theta_cap_deg < 90:
        raise ValueError(f"theta_cap_deg must be in [0, 90), got {theta_cap_deg}")
    mu_cap = np.cos(np.radians(theta_cap_deg))
    pieces = [(-1.0, mu_cap, lambda mu: np.full_like(mu, 1 / mu_cap))]
    if mu_cap < 1:
        pieces.append((mu_cap, 1.0, np.reciprocal))
    return _coupling(l_max, pieces, extra_nodes, n_rows)


def jackson_weights(l_max):
    """Jackson damping factors ``sigma_l``, ``l = 0..l_max``.

    ``sum_l sigma_l (2l + 1)/2 a_l P_l`` is a non-negative smoothing of a
    non-negative ``f`` with resolution ``~pi / l_max`` and tails that fall as
    ``(l_max theta)^-4``.
    """
    n = l_max + 1
    ell = np.arange(n)
    a = np.pi / (n + 1)
    return ((n - ell + 1) * np.cos(ell * a) + np.sin(ell * a) / np.tan(a)) / (n + 1)


def backward_removal(l_max):
    """``I - R``: removes the backward hemisphere of produced particles.

    ``R = M[1(mu < 0)] diag(sigma_l)`` takes the backward part of the Jackson
    smoothed distribution. For wide distributions this is the exact removal
    ``hemisphere_projection`` up to the smoothing near ``mu = 0``; collimated
    distributions (``a_l ~ 1`` up to ``l_max``) have no smoothed backward
    tail, where the sharp projection would leave Gibbs residues
    ``~ l_max^-1/2`` at every application.
    """
    r = _coupling(l_max, [(-1.0, 0.0, np.ones_like)], 0) * jackson_weights(l_max)
    return np.eye(l_max + 1) - r


def dense_count(l_grid):
    """Number of leading modes with ``l_grid[i] == i``."""
    l_grid = np.asarray(l_grid)
    match = l_grid == np.arange(len(l_grid))
    return len(l_grid) if match.all() else int(np.argmin(match))


def l_interpolation(l_grid, l_out):
    """``(len(l_out), len(l_grid))`` matrix that maps values on the mode grid
    to values at ``l_out``: a not-a-knot cubic spline in ``l``.

    One map for every place that needs the moments between the sparse modes,
    the readout and the secant coupling. A dense-only grid evaluated at its
    own ``l`` gives the identity.
    """
    from scipy.interpolate import CubicSpline

    l_grid = np.asarray(l_grid)
    l_out = np.asarray(l_out)
    n = len(l_grid)
    if dense_count(l_grid) == n and np.array_equal(l_out, l_grid):
        return np.eye(n)
    return CubicSpline(l_grid, np.eye(n), axis=0, bc_type="not-a-knot")(l_out)


def theta_density(a, l_grid, theta, smooth=True):
    """Angular density from moments ``a[..., mode]`` on the sparse ``l_grid``.

    The moments are interpolated in ``l`` with :func:`l_interpolation` to
    every ``l`` up to ``l_grid[-1]`` and summed as a Legendre series, with
    Jackson damping when ``smooth``. Returns ``f(theta) sin(theta) / theta``,
    the density in the planar measure ``theta dtheta``, so that
    ``int f_planar theta dtheta = int f dmu``.
    """
    a = np.asarray(a, dtype=float)
    l_grid = np.asarray(l_grid)
    l_all = np.arange(int(l_grid[-1]) + 1)
    full = a.reshape(-1, a.shape[-1]) @ l_interpolation(l_grid, l_all).T
    if smooth:
        full = full * jackson_weights(l_all[-1])
    theta = np.asarray(theta, dtype=float)
    f_mu = series(full, np.cos(theta))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(theta > 0, np.sin(theta) / np.where(theta > 0, theta, 1), 1.0)
    return (f_mu * ratio).reshape(a.shape[:-1] + theta.shape)
