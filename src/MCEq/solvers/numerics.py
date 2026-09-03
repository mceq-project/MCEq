"""The formula table of the ETD2RK step, and the buffers it runs in.

One integration step, with ``D`` the diagonal of the operator, ``F`` the
off-diagonal remainder and ``h`` the step size:

    eD = e^{hD}      hphi1 = h phi1(hD)      hphi2 = h phi2(hD)
    a  = eD x + hphi1 F(x)
    x+ = a + hphi2 (F(a) - F(x))

``h`` is folded into the phi factors at the factor stage, so the predictor
and the corrector are elementwise products of three arrays on every backend
and carry no step size of their own. That fixes the association order --
``h`` onto the phi factor first, the product onto the remainder -- at the one
place it is written, instead of leaving each backend's ufunc chain or kernel
to choose it.

This module is the single source of those formulas, lowered three ways:

* numpy, here -- :func:`phi_factors`, :func:`diagonal_factors` -- serving
  :class:`MCEq.solvers.backends.host.HostBackend` and through it MKL and
  Accelerate;
* C, ``MCEq/etd2_kernels/etd2_kernels.c``, which carries
  :data:`PREDICTOR_EXPR` and :data:`CORRECTOR_EXPR` verbatim as its
  ``ETD2_PREDICT`` / ``ETD2_CORRECT`` macros;
* cupy, :func:`MCEq.solvers.backends.cuda._build_cuda_etd2_kernels`, whose
  ElementwiseKernel bodies are generated from :data:`PHI_C_BODY` and the same
  two expression strings.

numpy only. A new scalar formula of the step loop belongs here; anything
that touches a library handle, a device, or a configuration value does not.
"""

import numpy as np

# --- the formula table -----------------------------------------------------

#: Radii below which the analytic phi quotients are replaced by their Taylor
#: series — order 2 for phi1, order 3 for phi2. phi2 switches at the wider
#: radius because its numerator cancels to second order where phi1's cancels
#: to first: with the numerator formed as ``e^z - 1``, the quotient carries
#: ``~2u/|z|`` for phi1 and ``~2u/z^2`` for phi2 (``u = 2^-53``), and each
#: radius sits where that meets the truncation of the series taken.
#:
#: Measured against a 60-digit reference over ``1e-8 <= |z| <= 1``, both signs
#: (:func:`tests.test_solvers.test_phi_factors_accuracy`), the worst relative
#: error is 7.9e-13 for phi1 at ``z = 1.4e-4`` and 5.4e-12 for phi2 at
#: ``z = 6.3e-3`` — in both cases on the quotient side just outside the
#: radius, which is what makes the radius, not the series order, the thing
#: that sets the accuracy. phi2's third Horner term is what lets its radius
#: widen that far, and buys 39x for one multiply-add.
#:
#: Two improvements are left: phi1's radius optimum on that grid is 2.0e-4
#: (4.9e-13, at identical cost), and forming the numerator with ``expm1``
#: gives phi1 9.1e-14 and phi2 4.4e-12 at these radii, or 2.2e-16 and 8.0e-14
#: with the radii retuned for it too (3.5e-6, 2.2e-3) — but needs a second
#: transcendental over the whole state where ``e^z`` is already there for ``eD``.
_PHI1_SMALL = 1.3e-4
_PHI2_SMALL = 6.31e-3
_INV_6 = 1.0 / 6.0
_INV_24 = 1.0 / 24.0
_INV_120 = 1.0 / 120.0

#: The two state stages as C expressions, with the step size already folded
#: into the phi factors. The compiled lowerings are built from these strings:
#: ``etd2_kernels.c`` carries them as macros and the cupy kernels format them
#: into ElementwiseKernel bodies.
PREDICTOR_EXPR = "(eD) * (x) + (hphi1) * (F)"
CORRECTOR_EXPR = "(a) + (hphi2) * ((F_a) - (F))"


def predictor(eD, x, hphi1, F, out, work):
    """``out = eD x + hphi1 F`` — the numpy lowering of
    :data:`PREDICTOR_EXPR`, associated exactly as written.

    The compiled lowerings are what the host and CUDA backends run; this is
    the reference they are checked against, and the fallback when the C
    extension is not built. ``work`` is a scratch array of ``out``'s shape.
    """
    np.multiply(eD, x, out=out)
    np.multiply(hphi1, F, out=work)
    np.add(out, work, out=out)


def corrector(a, hphi2, F_a, F, out, work):
    """``out = a + hphi2 (F_a - F)`` — the numpy lowering of
    :data:`CORRECTOR_EXPR`, associated exactly as written."""
    np.subtract(F_a, F, out=work)
    np.multiply(hphi2, work, out=work)
    np.add(a, work, out=out)


#: The phi branch as C source: reads a ``double z`` in scope and defines
#: ``e = e^z``, ``p1 = phi1(z)``, ``p2 = phi2(z)``. :func:`phi_factors` is the
#: numpy form of the same table; see its docstring for the formulas.
PHI_C_BODY = f"""
const double e = exp(z);
const double em1 = e - 1.0;
const double az = (z >= 0.0) ? z : -z;
const double p1 = (az > {_PHI1_SMALL!r})
                ? em1 / z
                : 1.0 + z * (0.5 + z * {_INV_6!r});
const double p2 = (az > {_PHI2_SMALL!r})
                ? (em1 - z) / (z * z)
                : 0.5 + z * ({_INV_6!r} + z * ({_INV_24!r} + z * {_INV_120!r}));
"""


def phi_factors(z, e, phi1, phi2, work):
    """``e^z``, ``phi1(z)`` and ``phi2(z)`` elementwise, into given buffers.

        phi1(z) = (e^z - 1) / z        Taylor 1 + z/2 + z^2/6
        phi2(z) = (e^z - 1 - z) / z^2  Taylor 1/2 + z/6 + z^2/24 + z^3/120

    Both quotients cancel as ``z -> 0``, phi1 to first order and phi2 to
    second, so inside :data:`_PHI1_SMALL` / :data:`_PHI2_SMALL` the series is
    evaluated by Horner instead. The numerator is ``e^z - 1``, formed from the
    exponential the step needs anyway for ``eD``, rather than ``expm1``, which
    costs accuracy in a band above each radius; see :data:`_PHI1_SMALL` for
    what that costs and what fixing it would move.

    The Taylor form is written for every entry and the quotient overwrites it
    where the argument is large enough, so the branch costs one mask pass and
    no gather. ``work`` is the ``(em1, t, mask)`` triple of :func:`phi_work`,
    the shape of ``z``; every output and temporary is preallocated because
    this runs once per integration step.

    ``z`` is ``hD`` over the full state, ``(dim,)`` for a shared integration
    path and ``(dim, K)`` per lane, and ``h D0 lam`` over the coupled plane of
    the sec(theta) exact slot. One implementation for all three: the shapes
    differ, the formula does not.
    """
    em1, t, mask = work
    np.exp(z, out=e)
    np.subtract(e, 1.0, out=em1)
    np.abs(z, out=t)

    np.multiply(z, _INV_6, out=phi1)
    np.add(phi1, 0.5, out=phi1)
    np.multiply(z, phi1, out=phi1)
    np.add(phi1, 1.0, out=phi1)
    np.greater(t, _PHI1_SMALL, out=mask)
    np.divide(em1, z, out=phi1, where=mask)

    np.multiply(z, _INV_120, out=phi2)
    np.add(phi2, _INV_24, out=phi2)
    np.multiply(z, phi2, out=phi2)
    np.add(phi2, _INV_6, out=phi2)
    np.multiply(z, phi2, out=phi2)
    np.add(phi2, 0.5, out=phi2)
    np.greater(t, _PHI2_SMALL, out=mask)
    np.subtract(em1, z, out=em1)
    np.multiply(z, z, out=t)
    np.divide(em1, t, out=phi2, where=mask)


def phi_work(shape):
    """The scratch triple :func:`phi_factors` needs for arguments of `shape`."""
    return (
        np.empty(shape, dtype=np.float64),
        np.empty(shape, dtype=np.float64),
        np.empty(shape, dtype=bool),
    )


def step_buffers(shape):
    """Buffers of the diagonal-factor stage, allocated once per solve.

    ``shape`` is ``(dim,)`` when every lane walks one shared integration path
    and ``(dim, K)`` when each carries its own atmosphere path, so that both
    ``h`` and ``ri`` vary along the lane axis. Centralized here so every
    backend shares a layout: this is a hot loop, and an allocation inside it
    dominates the SpMMs once those run on a tuned BLAS.

    Six float64 arrays and one boolean of ``shape``. At the full-sky
    operating point dim=7986, K=3072 that is 1.2 GB; see
    wiki/methods/multi-rhs-etd2-design.md Stage 3.
    """
    return {
        "hD": np.empty(shape, dtype=np.float64),
        "eD": np.empty(shape, dtype=np.float64),
        "phi1": np.empty(shape, dtype=np.float64),
        "phi2": np.empty(shape, dtype=np.float64),
        "work": phi_work(shape),
    }


def diagonal_factors(h, ri, d_int, d_dec, bufs):
    """``eD``, ``h phi1(hD)`` and ``h phi2(hD)`` of one step, in place.

    ``D = d_int + ri d_dec`` is the diagonal of the operator at this step and
    ``hD`` its scaled form. One function for both integration paths: a shared
    path passes scalar ``h`` and ``ri`` against ``(dim,)`` buffers, the
    per-lane paths of the carousel pass ``(K,)`` lane rows against a
    ``(dim, K)`` plane. The shapes differ, the formula does not.

    Folding ``h`` in here is what leaves the predictor and the corrector with
    three arrays and no step size; see the module docstring. A lane with
    ``h == 0`` is frozen by the same arithmetic: ``hD = 0`` gives ``eD = 1``
    and ``hphi1 = hphi2 = 0``, so its step collapses to ``x <- x``.
    """
    hD, eD, phi1, phi2 = (bufs[k] for k in ("hD", "eD", "phi1", "phi2"))
    if hD.ndim == 2:
        d_int, d_dec = d_int[:, None], d_dec[:, None]
        h, ri = h[None, :], ri[None, :]

    # D = d_int + ri d_dec, then hD = h D, both in the one buffer.
    np.multiply(d_dec, ri, out=hD)
    np.add(hD, d_int, out=hD)
    np.multiply(hD, h, out=hD)

    phi_factors(hD, eD, phi1, phi2, bufs["work"])
    np.multiply(phi1, h, out=phi1)
    np.multiply(phi2, h, out=phi2)


def left_matmul(matrix, plane, out=None):
    """Apply a mode-space matrix to the leading axis of a >= 2-D plane.

    The driver carries logical ``(mode, column, lane)`` planes; flattening
    the trailing axes turns each mode transform into one dense GEMM. The
    plane may be a strided view of the state (the low-E block) — BLAS
    takes it through its leading dimension without a copy.
    """
    trailing = plane.shape[1:]
    result_shape = (matrix.shape[0],) + trailing
    plane_2d = plane.reshape(plane.shape[0], -1)
    if out is None:
        return (matrix @ plane_2d).reshape(result_shape)
    np.matmul(matrix, plane_2d, out=out.reshape(matrix.shape[0], -1))
    return out
