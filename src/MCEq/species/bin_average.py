r"""Bin-averaged rates for databases with stencil-averaged tables.

A database built with mceq-maintenance-tools' ``bin_average`` step stores
yield and decay tables that act on the parent's *count density* inside each
bin, reconstructed by a fixed neighbour stencil: the degree-:math:`2r`
polynomial in :math:`\xi = \log_{10}(E/E_j)/h` that reproduces the cell
contents of bins :math:`j-r \ldots j+r`. The count of decays (interactions)
in bin :math:`j` is then the bin integral of :math:`r(E)\,\Phi(E)`, not the
centre value :math:`r(E_j)\,\Phi_j`. With the same reconstruction of
:math:`\Phi`,

.. math::

    N_j = \int_{-1/2}^{1/2} r(\xi)\,\phi(\xi)\,d\xi
        = \sum_{m=-r}^{r} s_{j+m} \sum_k W_{km} \int_{-1/2}^{1/2}
          \xi^k\, r(\xi)\, d\xi,

where :math:`s_j = \Phi_j w_j` are the cell contents and :math:`W` the inverse
of the cell-average matrix. In flux units the rate becomes a banded operator
:math:`R = \mathrm{diag}(1/w)\,R'\,\mathrm{diag}(w)` in place of
:math:`\mathrm{diag}(r_j)`; for a constant rate :math:`R = r\,I` exactly.
The database marks the parents it averaged (``bin_average_parents`` on the
decay group, ``bin_average_projectiles`` on a yield table) and the stencil
radius; only those get :math:`R`, so databases without the marks are
unchanged.
"""

import numpy as np

#: Gauss-Legendre nodes per bin for the rate moments
N_GAUSS = 12


def stencil_weights(r):
    """``W[k, m]``: polynomial coefficient ``c_k = sum_m W[k, m] s_{j+m}`` whose
    integrals over ``[m - 1/2, m + 1/2]`` are the cell contents, m = -r .. r."""
    m = np.arange(-r, r + 1)
    k = np.arange(2 * r + 1)
    a = ((m[:, None] + 0.5) ** (k[None] + 1) - (m[:, None] - 0.5) ** (k[None] + 1)) / (
        k[None] + 1
    )
    return np.linalg.inv(a)


def rate_operator(e_bins, rate, r):
    """Banded rate operator in flux units, ``(d, d)``.

    ``e_bins``: bin edges, uniform in log E (else ``ValueError``); ``rate``:
    callable, the rate at arbitrary energies (same units as the diagonal it
    replaces). Bins within ``r`` of the grid edge keep the centre rate.
    """
    e_bins = np.asarray(e_bins, dtype=np.float64)
    lb = np.log10(e_bins)
    h = np.diff(lb)
    if not np.allclose(h, h[0], rtol=1e-6, atol=0):
        raise ValueError("bin-averaged rates need a grid uniform in log E")
    h = float(h[0])
    d = len(e_bins) - 1
    centres = 10 ** (0.5 * (lb[1:] + lb[:-1]))
    widths = np.diff(e_bins)
    x, wq = np.polynomial.legendre.leggauss(N_GAUSS)
    xi, wq = 0.5 * x, 0.5 * wq
    r_nodes = np.asarray(rate(centres[:, None] * 10 ** (h * xi[None, :])), dtype=float)
    moments = (r_nodes * wq[None, :]) @ (
        xi[:, None] ** np.arange(2 * r + 1)
    )  # (d, 2r+1)
    band = moments @ stencil_weights(r)  # (d, 2r+1): R'[j, j + m]
    out = np.zeros((d, d))
    j = np.arange(r, d - r)
    for mi, m in enumerate(range(-r, r + 1)):
        out[j, j + m] = band[j, mi] * widths[j + m] / widths[j]
    edge = np.r_[np.arange(min(r, d)), np.arange(max(d - r, r), d)]
    out[edge, edge] = np.asarray(rate(centres[edge]), dtype=float)
    return out


def apply_right(block, op, r):
    """``block @ op`` for an ``op`` with half-bandwidth ``r``; ``block`` may carry
    leading (mode) axes. Keeps ``block``'s dtype."""
    out = np.zeros_like(block)
    d = op.shape[0]
    for m in range(-r, r + 1):
        diag = np.diagonal(op, offset=m)  # op[j, j + m]
        if m >= 0:
            out[..., m:] += block[..., : d - m] * diag.astype(block.dtype)
        else:
            out[..., : d + m] += block[..., -m:] * diag.astype(block.dtype)
    return out


def delta_deposition(e_bins, energy, r):
    """Cell contents on bins ``c - r .. c + r`` (``c`` the bin holding ``energy``)
    for one particle at ``energy`` under stencil-averaged tables.

    The tables read the cell contents of bins ``j - r .. j + r`` as the
    degree-``2 r`` polynomial density inside bin ``j`` (bins within ``r`` of the
    grid edge: a constant). The deposition is chosen so that this reconstructed
    density has the log-energy moments ``k <= 2 r`` of a delta at ``energy``:
    the first generation then equals the kernel at ``energy`` to the order of the
    stencil. Returns ``(first_bin, contents)``; contents sum to 1.
    """
    lb = np.log10(np.asarray(e_bins, dtype=np.float64))
    h = float(lb[1] - lb[0])
    d = len(lb) - 1
    c = int(np.searchsorted(lb, np.log10(energy), side="right")) - 1
    if c - r < 0 or c + r > d - 1:
        raise ValueError(
            f"energy {energy:g} GeV is within {r} bins of the grid edge; the "
            "stencil deposition needs its neighbours on the grid"
        )
    x0 = np.log10(energy)
    w = stencil_weights(r)
    xg, wg = np.polynomial.legendre.leggauss(2 * r + 2)
    xi, wg = 0.5 * xg, 0.5 * wg
    powers = xi[:, None] ** np.arange(2 * r + 1)  # (nodes, coefficient)
    kk = np.arange(2 * r + 1)
    b = np.zeros((2 * r + 1, 2 * r + 1))
    for a, src in enumerate(range(c - r, c + r + 1)):
        for j in range(max(src - r, 0), min(src + r, d - 1) + 1):
            if r <= j <= d - 1 - r:
                coef = w[:, src - j + r]  # polynomial of window j per unit s_src
            else:
                if j != src:
                    continue
                coef = np.eye(2 * r + 1)[0]
            p = powers @ coef
            dx = lb[j] + h * (0.5 + xi) - x0
            b[:, a] += (wg * p) @ (dx[:, None] ** kk)
    rhs = np.eye(2 * r + 1)[0]
    return c - r, np.linalg.solve(b, rhs)
