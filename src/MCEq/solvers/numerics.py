"""The formula table of the ETD2RK step, and the buffers it runs in.

One step, with ``D`` the diagonal of the operator, ``F`` the off-diagonal
remainder and ``h`` the step size::

    eD = e^{hD}      hphi1 = h phi1(hD)      hphi2 = h phi2(hD)
    a  = eD x + hphi1 F(x)
    x+ = a + hphi2 (F(a) - F(x))

Derivation: :ref:`solver-mathematics`.

This module is the single source of the formulas, lowered three ways:
numpy here; C in ``MCEq/solvers/_kernels/etd2/etd2_kernels.c``, which
carries the two ``*_EXPR`` strings verbatim as macros; cupy in
:func:`MCEq.solvers.backends.cuda._build_cuda_etd2_kernels`, whose kernel
bodies are formatted from the same strings. ``h`` is folded into the phi
factors at the factor stage, so the association order is fixed here and
the predictor and corrector are products of three arrays. numpy only.
"""

import numpy as np

# --- the formula table -----------------------------------------------------

#: Radii below which the phi quotients are replaced by their Taylor series
#: (order 2 for phi1, order 3 for phi2), set where the cancellation of the
#: quotient meets the truncation of the series; worst relative error 7.9e-13
#: (phi1) and 5.4e-12 (phi2) against a 60-digit reference
#: (:func:`tests.test_solvers.test_phi_factors_accuracy`).
_PHI1_SMALL = 1.3e-4
_PHI2_SMALL = 6.31e-3
_INV_6 = 1.0 / 6.0
_INV_24 = 1.0 / 24.0
_INV_120 = 1.0 / 120.0

#: The two state stages as C expressions; ``etd2_kernels.c`` carries them as
#: macros and the cupy kernels format them into their bodies.
PREDICTOR_EXPR = "(eD) * (x) + (hphi1) * (F)"
CORRECTOR_EXPR = "(a) + (hphi2) * ((F_a) - (F))"


def predictor(eD, x, hphi1, F, out, work):
    """``out = eD x + hphi1 F``, the numpy lowering of :data:`PREDICTOR_EXPR`:
    the reference of the compiled stages and the fallback without the C
    extension. ``work`` is a scratch array of ``out``'s shape."""
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
    """``e^z``, ``phi1(z)`` and ``phi2(z)`` elementwise, into given buffers::

        phi1(z) = (e^z - 1) / z        Taylor 1 + z/2 + z^2/6
        phi2(z) = (e^z - 1 - z) / z^2  Taylor 1/2 + z/6 + z^2/24 + z^3/120

    Inside :data:`_PHI1_SMALL` / :data:`_PHI2_SMALL` the series replaces the
    quotient. ``work`` is the ``(em1, t, mask)`` triple of :func:`phi_work`,
    the shape of ``z``.
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
    """Buffers of the diagonal-factor stage, allocated once per solve:
    ``hD, eD, phi1, phi2`` of ``shape`` (``(dim,)`` for a shared path,
    ``(dim, K)`` per lane) in fp64 and the ``work`` triple of
    :func:`phi_work`."""
    return {
        "hD": np.empty(shape, dtype=np.float64),
        "eD": np.empty(shape, dtype=np.float64),
        "phi1": np.empty(shape, dtype=np.float64),
        "phi2": np.empty(shape, dtype=np.float64),
        "work": phi_work(shape),
    }


def diagonal_factors(h, ri, d_int, d_dec, bufs):
    """``eD``, ``h phi1(hD)`` and ``h phi2(hD)`` of one step, in place, for
    ``D = d_int + ri d_dec``. ``h`` and ``ri`` are scalars against ``(dim,)``
    buffers or ``(K,)`` lane rows against ``(dim, K)``. A lane with
    ``h == 0`` gets ``eD = 1`` and ``hphi1 = hphi2 = 0``."""
    hD, eD, phi1, phi2 = (bufs[k] for k in ("hD", "eD", "phi1", "phi2"))
    if hD.ndim == 2:
        d_int, d_dec = d_int[:, None], d_dec[:, None]
        h, ri = h[None, :], ri[None, :]

    # D = d_int + ri d_dec, then hD = h D, in the one buffer
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
