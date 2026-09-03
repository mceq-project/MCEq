r"""Finite-difference families for ``d/du``, ``u = ln E``, on the energy grid.

Continuous energy loss enters the cascade equations as an advection in
``u = ln E``. :func:`differential_operator` returns the ``(dim_e, dim_e)``
banded matrix that approximates ``dPhi/du`` on the (log-uniform) grid; the
builder turns it into the loss band with

    ``-diag(1/E) . D . diag(dE/dX)``

so the accuracy of this one matrix sets the accuracy of the whole
continuous-loss channel.

Exactness conditions
--------------------
With log spacing ``h = Delta u`` a seven-point row ``c_{-3..+3}`` evaluates

    ``(dPhi/du)_i ~ (1/h) sum_k c_k Phi(u_i + k h)``.

On ``Phi = exp(a u)`` this is ``exp(a u_i) (1/h) sum_k c_k exp(a k h)``, so
the row differentiates that exponential *exactly* iff

    ``sum_k c_k exp(a k h) = a h``.                                     (19)

``docs/mceq_v1.x_v2_diff.md`` section 5.2 eq. 19 states the condition for one
rate ``a = -alpha0`` together with polynomial exactness
``sum_k c_k k^p = delta_{p,1}``, ``p = 0..5``, as the six companion equations.
This implementation instead imposes (19) at seven rates at once,

    ``a = -alpha0 + delta``,  ``delta = [-1, -1/2, 0, 1/2, 1, 3/2, 2]``,

seven equations for seven coefficients (:func:`expfit_coefficients`). The
fitted set is therefore a *band* of power laws ``E^{-alpha}`` with
``alpha = alpha0 - delta`` in ``[alpha0 - 2, alpha0 + 1]``, not one power law
plus a polynomial.

Which families are exact on a constant
--------------------------------------
A constant is ``a = 0``, where (19) reads ``sum_k c_k = 0`` -- zero row sum.

* ``centered``, ``biased``, ``upwind``, ``upwind2``: exact in every row.
  They are polynomial fits and ``p = 0`` is in their fit set.
* the ``expfit`` interior: **not** exact. ``a = 0`` needs ``delta = alpha0``,
  and at the default ``alpha0 = 3`` that lies outside the fitted band, so the
  span of the seven fitted exponentials does not contain a constant. Every
  interior row carries the same residual ``(sum_k c_k) / h`` -- the same one
  because the grid is log-uniform, so a single coefficient vector is divided
  by a common ``h``. It is a genuine truncation error on flat spectra, traded
  for round-off-limited accuracy near ``E^{-alpha0}`` (docs 5.2: 5e-16
  against ~1e-4 for ``centered`` on ``exp(-3u)``).
* boundary rows of every family, and the ``expfit_low_upwind*`` replacement
  rows: exact -- they are one-sided polynomial or upwind rows.

Row layout
----------
Every family is seven-point and spans at most ``[-3, +3]``, which is what
makes the interior range ``range(3, dim_e - 3)`` uniform across the dispatch.
The three lowest and three highest rows would reach outside the grid, so the
non-upwind families close them with one-sided polynomial fits of the same
order. Those rows are exact on constants but carry percent-level residuals on
steep spectra and excite non-normal transients at a low-energy grid floor --
the "boundary cliff" of ``docs/mceq_v1.x_v2_diff.md`` section 5.3. The
``expfit_low_upwind*`` composites are the fix in production use: they overwrite
the lowest ``low_upwind_rows`` rows with monotone upwind rows and keep the
expfit interior above.

Energy loss advects the spectrum toward *lower* E, so the upwind direction is
toward higher E (forward in ``u``), which is also the orientation of the
one-sided low-energy row.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STENCIL_METHODS",
    "differential_operator",
    "expfit_coefficients",
    "log_spacings",
]

#: Every interior stencil :func:`differential_operator` accepts, in the order
#: of its own dispatch: the two composites, the three high-order families,
#: then the two monotone upwind operators.
STENCIL_METHODS = (
    "expfit_low_upwind2",
    "expfit_low_upwind",
    "expfit",
    "centered",
    "biased",
    "upwind",
    "upwind2",
)

#: The seven rate offsets ``delta`` of the expfit fit set, ``a = -alpha0 + delta``.
EXPFIT_DELTAS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

#: ``(offsets, coefficients, denominator)`` of the one-sided polynomial rows at
#: the low-energy edge -- rows 0, 1 and 2, each relative to its own row index.
#: Coefficients are truncated Taylor fits of ``d/du`` of the same order as the
#: interior; each set sums to zero, so each row annihilates a constant.
_LOW_EDGE_ROWS = (
    ([0, 1, 2, 3], [-11, 18, -9, 2], 6),
    ([-1, 0, 1, 2, 3], [-3, -10, 18, -6, 1], 12),
    ([-2, -1, 0, 1, 2, 3], [3, -30, -20, 60, -15, 2], 60),
)

#: The high-energy edge rows: the mirror images of :data:`_LOW_EDGE_ROWS`.
#: ``d/du`` is odd under ``u -> -u``, so reflecting the offsets and negating
#: the coefficients turns a forward-biased row into the backward-biased one.
_HIGH_EDGE_ROWS = tuple(
    ([-d for d in offsets[::-1]], [-c for c in coeffs[::-1]], denom)
    for offsets, coeffs, denom in _LOW_EDGE_ROWS
)


def log_spacings(e_bins):
    """Per-bin log spacings ``h_i = ln(E_{i+1}/E_i)`` from the bin edges.

    Constant on a log-uniform grid; kept per-row because the operator divides
    each row by the spacing of *its* bin, which is what makes the boundary rows
    correct on a grid that has been cut at either end.
    """
    e_bins = np.asarray(e_bins)
    return np.log(e_bins[1:] / e_bins[:-1])


def expfit_coefficients(h, alpha0, offsets=None):
    r"""Solve (19) for the seven exponentially-fitted interior coefficients.

    ``A c = b`` with ``A_{ij} = exp(a_i k_j h)`` and ``b_i = a_i h`` over the
    seven rates ``a_i = -alpha0 + delta_i`` of :data:`EXPFIT_DELTAS`. Square and
    nonsingular for distinct rates, so the fit is a direct solve rather than a
    least-squares one, and the resulting row differentiates each of the seven
    power laws ``E^{-alpha0 + delta}`` exactly.

    ``h`` is a single spacing, not the per-bin array: the grid is log-uniform,
    so one fit at the mean spacing serves every interior row.
    """
    if offsets is None:
        offsets = np.arange(-3, 4)
    a = -alpha0 + EXPFIT_DELTAS
    return np.linalg.solve(np.exp(np.outer(a, offsets) * h), a * h)


def _interior_row(method, h, alpha0):
    """``(offsets, coefficients)`` of the interior row of a high-order family.

    The coefficients are the row *before* division by ``h``, so a caller
    divides by the spacing of each row it writes.
    """
    if method == "biased":
        offsets, coeffs, denom = _LOW_EDGE_ROWS[2]
        return np.asarray(offsets), np.asarray(coeffs, dtype=np.float64) / float(denom)
    if method == "centered":
        return (
            np.asarray([-3, -2, -1, 1, 2, 3]),
            np.asarray([-1, 9, -45, 45, -9, 1], dtype=np.float64) / 60.0,
        )
    if method == "expfit":
        offsets = np.arange(-3, 4)
        # One fit at the mean log spacing; the grid is log-uniform.
        return offsets, expfit_coefficients(float(np.mean(h)), float(alpha0), offsets)
    raise ValueError(
        f"Unknown loss_stencil_method: {method!r}. "
        "Expected 'expfit', 'centered', 'biased', 'upwind', 'upwind2', "
        "'expfit_low_upwind', or 'expfit_low_upwind2'."
    )


def _write_upwind_rows(op_matrix, h, rows, order):
    """Fill ``rows`` with the monotone upwind row of the requested order.

    Forward-biased, i.e. toward higher E: ``[-1, 1]/h`` at first order,
    ``[-3/2, 2, -1/2]/h`` at second. Both annihilate a constant, are
    unconditionally stable, and are diffusive at ``O(h)`` / ``O(h^2)``, so they
    converge as the grid refines. Requires ``row + order`` to be on the grid.
    """
    for row in rows:
        if order == 1:
            op_matrix[row, row] = -1.0 / h[row]
            op_matrix[row, row + 1] = 1.0 / h[row]
        else:
            op_matrix[row, row] = -1.5 / h[row]
            op_matrix[row, row + 1] = 2.0 / h[row]
            op_matrix[row, row + 2] = -0.5 / h[row]


def _upwind_operator(h, dim_e, order, dtype):
    """The pure upwind families: one monotone row everywhere, no high-order rows.

    No polynomial boundary rows at all, so the expfit boundary cliff has nothing
    to excite; the price is first-/second-order accuracy in the bulk. Added for
    the fine-grid Nmax convergence study
    (``runs/2026-06-06_em-grid-exact-nmax``). The top ``order`` rows cannot
    reach forward and take the backward-biased mirror row instead.
    """
    op_matrix = np.zeros((dim_e, dim_e), dtype=dtype)
    last = dim_e - 1
    _write_upwind_rows(op_matrix, h, range(dim_e - order), order)
    if order == 1:
        op_matrix[last, last - 1] = -1.0 / h[last - 1]
        op_matrix[last, last] = 1.0 / h[last - 1]
    else:
        for row in (dim_e - 2, last):
            hh = h[row - 2]
            op_matrix[row, row - 2] = 0.5 / hh
            op_matrix[row, row - 1] = -2.0 / hh
            op_matrix[row, row] = 1.5 / hh
    return op_matrix


def differential_operator(
    e_bins,
    dim_e,
    method="expfit_low_upwind2",
    alpha0=3.0,
    low_upwind_rows=8,
    dtype=np.float64,
):
    """The ``(dim_e, dim_e)`` approximation to ``d/du``, ``u = ln E``.

    Args:
      e_bins: energy bin edges, ``dim_e + 1`` of them; only their log spacings
        are used.
      dim_e (int): energy-grid dimension, the size of the returned matrix.
      method (str): one of :data:`STENCIL_METHODS`.
      alpha0 (float): anchor exponent of the ``expfit`` fit set; read only by
        the three ``expfit*`` families.
      low_upwind_rows (int): rows the ``expfit_low_upwind*`` composites replace,
        clamped to ``[3, dim_e - 2]``; read only by those two.
      dtype: dtype of the returned matrix.

    Raises:
      ValueError: on a name outside :data:`STENCIL_METHODS`.
    """
    h = log_spacings(e_bins)
    dim_e = int(dim_e)
    last = dim_e - 1

    low_boundary = None
    if method in ("expfit_low_upwind", "expfit_low_upwind2"):
        low_boundary = method.rsplit("_", 1)[-1]
        method = "expfit"

    if method in ("upwind", "upwind2"):
        return _upwind_operator(h, dim_e, 1 if method == "upwind" else 2, dtype)

    offsets_int, coeffs_int = _interior_row(method, h, alpha0)

    op_matrix = np.zeros((dim_e, dim_e), dtype=dtype)
    for row, (offsets, coeffs, denom) in enumerate(_LOW_EDGE_ROWS):
        op_matrix[row, row + np.asarray(offsets)] = np.asarray(coeffs) / (
            denom * h[row]
        )
    for below, (offsets, coeffs, denom) in enumerate(_HIGH_EDGE_ROWS):
        row = last - below
        op_matrix[row, row + np.asarray(offsets)] = np.asarray(coeffs) / (
            denom * h[row]
        )
    for row in range(3, dim_e - 3):
        op_matrix[row, row + offsets_int] = coeffs_int / h[row]

    if low_boundary is not None:
        n_low = min(max(3, int(low_upwind_rows)), dim_e - 2)
        op_matrix[:n_low, :] = 0.0
        _write_upwind_rows(
            op_matrix, h, range(n_low), 1 if low_boundary == "upwind" else 2
        )
    return op_matrix
