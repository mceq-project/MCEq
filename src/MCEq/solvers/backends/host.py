"""The host backend: numpy elementwise kernels and BLAS GEMMs.

:class:`HostBackend` executes every stage of
:func:`MCEq.solvers.etd2.etd2_driver` on host arrays and delegates only the
off-diagonal SpMM to an ``apply_off`` binding -- scipy in
:mod:`MCEq.solvers.backends.base`, MKL in
:mod:`MCEq.solvers.backends.mkl`, Apple Accelerate in
:mod:`MCEq.solvers.backends.accelerate`. The optional fused fp64 predictor /
corrector of :mod:`MCEq.etd2_kernels` is bound here too.

Stage code shared by all three host bindings belongs here; a binding's own
ctypes plumbing belongs in that binding's module.
"""

from ctypes import POINTER, c_double

import numpy as np

from MCEq.solvers.backends.base import ScipyApplyOff, _state_dtype
from MCEq.solvers.numerics import (
    _etd_compute_diag_factors,
    _etd_compute_diag_factors_multipath,
    _etd_step_buffers,
    _etd_step_buffers_multipath,
    _secant_left_matmul,
    _secant_phi_factors,
)

_C_DOUBLE_P = POINTER(c_double)

#: Below this many state elements the fused C post-apply does not pay for
#: its ctypes call (4 ufunc passes ≈ 7 µs vs 19 µs at dim = 4182, K = 1;
#: 0.18 vs 0.10 ms at dim = 171360, K = 1; 4.8 vs 1.9 ms at K = 8).
_FUSED_MIN_ELEMENTS = 1 << 16


def _dptr(arr):
    return arr.ctypes.data_as(_C_DOUBLE_P)


def _rowmajor_post_apply():
    """The fused fp64 predictor / corrector of ``MCEq.etd2_kernels`` for the
    row-major ``(dim, K)`` state, or ``None`` when the extension is missing
    (the host backend then falls back to numpy ufuncs)."""
    try:
        from MCEq.etd2_kernels import (
            etd2_post_apply1_rowmajor,
            etd2_post_apply2_rowmajor,
        )
    except ImportError:
        return None
    return etd2_post_apply1_rowmajor, etd2_post_apply2_rowmajor


class HostBackend:
    """Stage execution on host arrays for
    :func:`MCEq.solvers.etd2.etd2_driver`.

    numpy elementwise kernels and BLAS GEMMs throughout; the SpMM is the
    ``apply_off`` binding of the sparse library (scipy, MKL or Apple
    Accelerate). ``op`` is
    the :class:`~MCEq.operator_assembly.CompiledOperator` the binding was
    built from — it carries the layout and the coupling operators.

    ``dtype`` is the state precision, float64 or float32: the state, the
    scratch buffers and ``apply_off``'s operator are stored in it, while
    the diagonals and the phi factors are computed in fp64 and cast once,
    in :meth:`diag_factors`. See
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`.
    """

    xp = np
    left_matmul = staticmethod(_secant_left_matmul)

    def __init__(self, op, apply_off, dtype=np.float64):
        self.op = op
        self.name = apply_off.name
        self.dtype = np.dtype(dtype).type
        self.d_int, self.d_dec = op.d_int, op.d_dec
        self._apply_off = apply_off
        self._post = _rowmajor_post_apply()
        self._coupling = None

    def bind(self, dim, K, per_lane, nsteps):
        if self.op.dim != dim:
            raise ValueError(
                f"HostBackend: operator dim {self.op.dim} != state dim {dim}"
            )
        fp64 = self.dtype is np.float64
        self._per_lane = per_lane
        self._bufs = (
            _etd_step_buffers_multipath(dim, K) if per_lane else _etd_step_buffers(dim)
        )
        self._scratch = np.empty((dim, K), dtype=self.dtype)
        self._block = None
        # The fused C post-apply is fp64-only; fp32 takes the ufunc passes.
        self._fused = fp64 and self._post is not None and dim * K >= _FUSED_MIN_ELEMENTS
        self._h1 = np.empty(1, dtype=np.float64)
        # Landing buffers for the one cast of the fp64 factors, or None at
        # fp64 where the factor buffers are already the state dtype.
        shape = (dim, K) if per_lane else (dim,)
        self._factors = (
            None if fp64 else tuple(np.empty(shape, dtype=self.dtype) for _ in range(3))
        )
        self._apply_off.bind(dim, K, nsteps)

    def coupling(self):
        """``T_P, T_PP, V, Vi`` in the state dtype (they act on the state);
        ``lam`` in fp64 (it forms the phi arguments of the exact slot)."""
        if self._coupling is None:
            c = self.op.coupling
            self._coupling = tuple(
                np.asarray(m, dtype=self.dtype) for m in (c.T_P, c.T_PP, c.V, c.Vi)
            ) + (np.asarray(c.lam, dtype=np.float64),)
        return self._coupling

    def state_buffers(self, dim, K):
        return tuple(np.zeros((dim, K), dtype=self.dtype) for _ in range(4))

    def apply_off(self, x, out, ri):
        self._apply_off(x, out, ri)

    def _cast_factors(self, *factors):
        """The one cast of the fp64 factors to the state dtype; identity at
        fp64, where the factor buffers already are the state dtype."""
        if self._factors is None:
            return factors
        for src, dst in zip(factors, self._factors):
            np.copyto(dst, src)
        return self._factors

    def diag_factors(self, h, ri):
        """``eD, phi1, phi2`` of ``h (d_int + ri d_dec)``, broadcastable to
        (dim, K). ``h`` and ``ri`` come in fp64 and the evaluation runs
        there; only the result is cast."""
        b = self._bufs
        if self._per_lane:
            _etd_compute_diag_factors_multipath(h, ri, self.d_int, self.d_dec, b)
            return self._cast_factors(b["eD"], b["phi1"], b["phi2"])
        _etd_compute_diag_factors(h, ri, self.d_int, self.d_dec, b)
        eD, phi1, phi2 = self._cast_factors(b["eD"], b["phi1"], b["phi2"])
        return eD[:, None], phi1[:, None], phi2[:, None]

    def block_factors(self, ZB):
        if self._block is None or self._block[0].shape != ZB.shape:
            self._block = tuple(np.empty(ZB.shape) for _ in range(4)) + (
                np.empty(ZB.shape, dtype=bool),
            )
        eDB, phi1B, phi2B, scratch, large = self._block
        _secant_phi_factors(ZB, out=(eDB, phi1B, phi2B), work=(scratch, large))
        return eDB, phi1B, phi2B

    def _fused_args(self, factor, h, out):
        """Strides of the fused C post-apply for ``(dim,)`` vs ``(dim, K)``
        factors and scalar vs ``(K,)`` step sizes (see ``etd2_kernels.c``)."""
        dim, K = out.shape
        per_lane = factor.ndim == 2 and factor.shape[1] == K and K > 1
        f_row, f_col = (K, 1) if per_lane else (1, 0)
        if np.ndim(h) == 0:
            self._h1[0] = h
            h_arr, h_stride = self._h1, 0
        else:
            h_arr, h_stride = np.ascontiguousarray(h, dtype=np.float64), 1
        return dim, K, _dptr(h_arr), h_stride, f_row, f_col, h_arr

    def predictor(self, eD, x, phi1, F, h, out):
        """``out = eD x + h phi1 F`` — one fused pass, or four ufunc passes."""
        if self._fused:
            dim, K, h_p, h_s, f_row, f_col, _keep = self._fused_args(eD, h, out)
            self._post[0](
                dim,
                K,
                h_p,
                h_s,
                _dptr(eD),
                _dptr(phi1),
                f_row,
                f_col,
                _dptr(x),
                _dptr(F),
                _dptr(out),
            )
            return
        s = self._scratch
        np.multiply(eD, x, out=out)
        np.multiply(phi1, F, out=s)
        s *= h
        np.add(out, s, out=out)

    def corrector(self, a, F_a, F, phi2, h, out):
        """``out = a + h phi2 (F_a - F)`` — one fused pass, or four ufunc passes."""
        if self._fused:
            dim, K, h_p, h_s, f_row, f_col, _keep = self._fused_args(phi2, h, out)
            self._post[1](
                dim,
                K,
                h_p,
                h_s,
                _dptr(phi2),
                f_row,
                f_col,
                _dptr(a),
                _dptr(F_a),
                _dptr(F),
                _dptr(out),
            )
            return
        s = self._scratch
        np.subtract(F_a, F, out=s)
        s *= h
        np.multiply(s, phi2, out=s)
        np.add(a, s, out=out)

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=self.dtype if dtype is None else dtype)

    def to_host(self, a):
        return np.asarray(a, dtype=np.float64)

    def synchronize(self):
        pass

    def close(self):
        self._apply_off.close()


def numpy_backend(op, fp_precision=64):
    """Host backend on scipy CSR SpMM."""
    dtype = _state_dtype(fp_precision)
    return HostBackend(op, ScipyApplyOff(op.int_off, op.dec_off, dtype), dtype)
