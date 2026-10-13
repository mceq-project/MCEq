"""What every ETD2 backend shares: the precision contract and scipy SpMM.

:data:`_PRECISION_CONTRACT` is the fp32/fp64 rule all four backends
implement, and :func:`_state_dtype` the one place ``fp_precision`` becomes a
dtype. :class:`ScipyApplyOff` is the off-diagonal SpMM binding that needs no
library beyond scipy, and the reference the other bindings are checked
against.

A definition belongs here when more than one backend module needs it and it
depends on no sparse library and no device.
"""

import numpy as np

#: The precision contract every backend implements. ``fp_precision`` (32 or
#: 64) is one knob on the shared backends, not a kernel family of its own,
#: and it means exactly this. Referenced from
#: :class:`MCEq.solvers.backends.host.HostBackend`,
#: :class:`MCEq.solvers.backends.cuda.CudaOperator`,
#: :func:`MCEq.solvers.etd2.etd2_driver` and
#: :func:`MCEq.solvers.etd2.solve_etd2`.
_PRECISION_CONTRACT = """\
The diagonals (``d_int``, ``d_dec``, and ``lam_j D0_i`` on the sec(theta)
corner) and the phi factors computed from them (``eD``, ``h phi1``,
``h phi2``) are evaluated in FP64 whatever the requested precision, and
cast once, on the way out, to the state dtype. The state, the scratch
buffers and the off-diagonal operator are stored in the requested dtype,
and every stage that touches them runs there. The step size and
``rho_inv`` therefore reach the factor stage in FP64; ``rho_inv`` reaches
the SpMM in the state dtype.

The phi quotients cancel near their Taylor-switch radii, which are
calibrated for FP64 rounding; the contract binds the inputs of the factor
stage, not only its arithmetic.

At FP32 the driver solves for ``s phi`` with ``s`` an exact power of two
(:func:`MCEq.solvers.etd2._fp32_scale`) and divides it out of the result:
fluxes falling like E^-3 reach 1e-40 near 1e10 GeV, below the FP32 normal
range, and the products inside the SpMM fall below it first.
"""

#: State dtype per ``fp_precision``; the one place 32/64 becomes a dtype.
_FP_DTYPE = {32: np.float32, 64: np.float64}


def _state_dtype(fp_precision):
    """The state dtype of ``fp_precision`` (32 or 64)."""
    try:
        return _FP_DTYPE[int(fp_precision)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"fp_precision must be 32 or 64, got {fp_precision!r}"
        ) from None


class ScipyApplyOff:
    """``out = int_off x + ri dec_off x`` through scipy CSR SpMM.

    The operator is stored in ``dtype`` so the SpMM runs at the state
    precision (see :data:`_PRECISION_CONTRACT`)."""

    name = "numpy"

    def __init__(self, int_off, dec_off, dtype=np.float64):
        self.int_off, self.dec_off = (
            m.astype(dtype, copy=False) for m in (int_off, dec_off)
        )

    def bind(self, dim, K, nsteps):
        pass

    def __call__(self, x, out, ri):
        np.copyto(out, self.int_off.dot(x))
        ri_dec = self.dec_off.dot(x)
        ri_dec *= ri  # scalar, or (K,) broadcast over the lane axis
        np.add(out, ri_dec, out=out)

    def close(self):
        pass
