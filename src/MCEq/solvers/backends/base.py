"""Precision rule, state dtype and scipy off-diagonal SpMM shared by the ETD2 backends."""

import numpy as np

# Precision rule of every backend. The diagonals (d_int, d_dec, and lam_j D0_i
# on the sec(theta) corner) and the phi factors (eD, h phi1, h phi2) are
# evaluated in fp64 and cast once to the state dtype; the step size and
# rho_inv reach the factor stage in fp64 (the phi Taylor-switch radii are
# calibrated for fp64). The state, the scratch buffers and the off-diagonal
# operator are stored in the requested dtype, and rho_inv reaches the SpMM in
# it. At fp32 the driver solves for s*phi with s an exact power of two
# (MCEq.solvers.etd2._fp32_scale) to keep steep spectra above the fp32 normal
# range.

#: State dtype per ``fp_precision``.
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

    The operator is stored in ``dtype``, the state precision."""

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
