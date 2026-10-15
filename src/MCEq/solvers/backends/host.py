"""The host backend: numpy elementwise kernels, C step stages, BLAS GEMMs.

:class:`HostBackend` executes every stage of
:func:`MCEq.solvers.etd2.etd2_driver` on host arrays and delegates the
off-diagonal SpMM to an ``apply_off`` binding: scipy (:class:`ScipyApplyOff`),
MKL (:mod:`MCEq.solvers.backends.mkl`) or Apple Accelerate
(:mod:`MCEq.solvers.backends.accelerate`). The predictor and the corrector run
in C (:mod:`MCEq.solvers._kernels.etd2`), with the numpy lowering of
:mod:`MCEq.solvers.numerics` as the fallback.

Precision rule of every backend: the diagonals and the phi factors are
evaluated in fp64 and cast once to the state dtype; the state, the scratch
buffers and the off-diagonal operator are stored in the state dtype. At fp32
the driver solves for ``s * phi`` with ``s`` a power of two
(:func:`MCEq.solvers.etd2._fp32_scale`).
"""

from ctypes import POINTER, c_double, c_float

import numpy as np

from MCEq.operators.compiled import coupled_corner
from MCEq.solvers import numerics
from MCEq.solvers.numerics import (
    diagonal_factors,
    left_matmul,
    step_buffers,
)

__all__ = [
    "CtypesApplyOff",
    "HostBackend",
    "ScipyApplyOff",
    "numpy_backend",
    "state_dtype",
]

#: ctypes pointer type per state dtype, for the fused C stages.
_C_POINTER = {np.float64: POINTER(c_double), np.float32: POINTER(c_float)}

#: ctypes scalar per state dtype, for the sparse-library pointer arguments.
_C_SCALAR = {np.dtype(np.float64): c_double, np.dtype(np.float32): c_float}


def state_dtype(dtype):
    """``dtype`` as float64 or float32; anything else raises ``ValueError``."""
    dtype = np.dtype(dtype)
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"state dtype must be float32 or float64, got {dtype}")
    return dtype


class ScipyApplyOff:
    """``out = int_off x + ri dec_off x`` through scipy CSR SpMM, with the
    operator stored in the state ``dtype``."""

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


class CtypesApplyOff:
    """``out = int_off x + ri dec_off x`` through the sparse handles of a
    ctypes library, on the C-contiguous ``(dim, K)`` buffers of the driver.

    A scalar ``ri`` rides the alpha of the decay product; a ``(K,)`` lane row
    scales a separate accumulator. ``owns`` closes the handles with the
    binding. Subclasses issue one product in :meth:`_spmm`.
    """

    def __init__(self, int_handle, dec_handle, owns=False, dtype=np.float64):
        self.int_off, self.dec_off = (
            m if m is not None and m.nnz else None for m in (int_handle, dec_handle)
        )
        self.handles = [m for m in (self.int_off, self.dec_off) if m is not None]
        self.owns = owns
        self.dtype = np.dtype(dtype)

    def bind(self, dim, K, nsteps):
        self.K = K
        # Dropped first (the memo pins its arrays), so two generations of
        # staging never coexist.
        self._dec_buf = None
        self._ptrs = {}
        self._ptr_type = POINTER(_C_SCALAR[self.dtype])
        self._dec_buf = np.empty((dim, K), dtype=self.dtype)

    def _ptr(self, arr):
        p = self._ptrs.get(id(arr))
        if p is None:
            p = self._ptrs[id(arr)] = arr.ctypes.data_as(self._ptr_type)
        return p

    def _spmm(self, m, alpha, x, out, overwrite):
        """``out = alpha m x`` with ``overwrite``, else ``out += alpha m x``."""
        raise NotImplementedError

    def __call__(self, x, out, ri):
        if self.int_off is None:
            out.fill(0.0)
        else:
            self._spmm(self.int_off, 1.0, x, out, overwrite=True)
        if self.dec_off is not None:
            if np.ndim(ri) == 0:
                self._spmm(self.dec_off, float(ri), x, out, overwrite=False)
            else:
                self._spmm(self.dec_off, 1.0, x, self._dec_buf, overwrite=True)
                np.multiply(self._dec_buf, ri, out=self._dec_buf)
                np.add(out, self._dec_buf, out=out)

    def close(self):
        self._dec_buf = None
        self._ptrs = {}
        if self.owns:
            for m in self.handles:
                m.close()


def _fused_stages(dtype):
    """The C predictor / corrector of :mod:`MCEq.solvers._kernels.etd2` at
    ``dtype``, or ``None`` when the extension is not built."""
    try:
        from MCEq.solvers._kernels.etd2 import (
            etd2_corrector_f32,
            etd2_corrector_f64,
            etd2_predictor_f32,
            etd2_predictor_f64,
        )
    except ImportError:
        return None

    if dtype is np.float64:
        return etd2_predictor_f64, etd2_corrector_f64
    return etd2_predictor_f32, etd2_corrector_f32


class HostBackend:
    """Stage execution on host arrays for
    :func:`MCEq.solvers.etd2.etd2_driver`.

    ``op`` is the :class:`~MCEq.operators.compiled.CompiledOperator`,
    ``apply_off`` the SpMM binding of the sparse library and ``dtype`` the
    state precision (float64 or float32): the state, the scratch buffers and
    the operator are stored in it; the diagonals and the phi factors are
    computed in fp64 and cast once.
    """

    xp = np
    left_matmul = staticmethod(left_matmul)

    def __init__(self, op, apply_off, dtype=np.float64):
        self.op = op
        self.name = apply_off.name
        self.dtype = np.dtype(dtype).type
        self.d_int, self.d_dec = op.d_int, op.d_dec
        self._apply_off = apply_off
        self._band = None
        self._coupling = None
        self._ptr_cache = {}

    def bind(self, dim, K, per_lane, nsteps):
        if self.op.dim != dim:
            raise ValueError(
                f"HostBackend: operator dim {self.op.dim} != state dim {dim}"
            )
        fp64 = self.dtype is np.float64
        self._dim, self._K, self._per_lane = dim, K, per_lane
        shape = (dim, K) if per_lane else (dim,)
        # dropped first, so two generations of scratch never coexist
        self._bufs = self._corner = self._factors = self._work = None
        self._ptr_cache = {}
        self._bufs = step_buffers(shape)
        lay = self.op.layout
        if lay.coupled:
            # factor buffers and diagonals of the corner, flat over (mode, column)
            n_c = lay.n_P * lay.n_g
            self._corner = step_buffers((n_c, K) if per_lane else (n_c,))
            self._corner_diag = tuple(d.ravel() for d in self.op.exact_slot_diagonals)
        stages = _fused_stages(self.dtype)
        self._predict, self._correct = stages or (None, None)
        self._ptr = _C_POINTER[self.dtype]
        # scratch for the numpy lowering
        self._work = None if stages else np.empty((dim, K), dtype=self.dtype)
        # landing buffers for the cast of the fp64 factors (None at fp64)
        self._factors = (
            None if fp64 else tuple(np.empty(shape, dtype=self.dtype) for _ in range(3))
        )
        self._apply_off.bind(dim, K, nsteps)

    def coupling(self):
        """``W, V, Vi`` of the compiled operator in the state dtype."""
        if self._coupling is None:
            c = self.op.coupling
            self._coupling = tuple(
                np.asarray(m, dtype=self.dtype) for m in (c.W, c.V, c.Vi)
            )
        return self._coupling

    def state_buffers(self, dim, K):
        return tuple(np.zeros((dim, K), dtype=self.dtype) for _ in range(4))

    def apply_off(self, x, out, ri):
        self._apply_off(x, out, ri)

    def apply_band(self, x, out):
        """``out += band_off x``: the continuous-loss off-diagonal of a
        Legendre operator (scipy SpMM on every host backend)."""
        if self._band is None:
            self._band = self.op.band_off.astype(self.dtype)
        out += self._band @ x

    def _cast_factors(self, *factors):
        """Cast the fp64 factors to the state dtype; identity at fp64."""
        if self._factors is None:
            return factors
        for src, dst in zip(factors, self._factors):
            np.copyto(dst, src)
        return self._factors

    def diag_factors(self, h, ri):
        """``eD, h phi1, h phi2`` of ``h (d_int + ri d_dec)`` in the state
        dtype, ``(dim,)`` or ``(dim, K)``; on the sec(theta) corner the
        factors of ``h lam_j D0_i``. ``h`` and ``ri`` come in fp64 and the
        evaluation runs there."""
        b = self._bufs
        diagonal_factors(h, ri, self.d_int, self.d_dec, b)
        if self._corner is not None:
            c = self._corner
            diagonal_factors(h, ri, *self._corner_diag, c)
            for name in ("eD", "phi1", "phi2"):
                dst = coupled_corner(self.op.layout, b[name])
                np.copyto(dst, c[name].reshape(dst.shape))
        return self._cast_factors(b["eD"], b["phi1"], b["phi2"])

    def _ptrs(self, arrays):
        """ctypes pointers to ``arrays``, memoized by identity for the solve
        (``data_as`` costs more than a kernel at K = 1). Each entry keeps its
        array so a recycled ``id`` cannot hit."""
        cache = self._ptr_cache
        out = []
        for arr in arrays:
            entry = cache.get(id(arr))
            if entry is None or entry[0] is not arr:
                entry = (arr, arr.ctypes.data_as(self._ptr))
                cache[id(arr)] = entry
            out.append(entry[1])
        return out

    def _bcast(self, factor):
        """A shared-path factor as a column, broadcast over the lane axis
        (the numpy lowering only)."""
        return factor if self._per_lane or factor.ndim == 2 else factor[:, None]

    def predictor(self, eD, x, hphi1, F, out):
        """``a = eD x + hphi1 F``: the C kernel, or the numpy lowering."""
        if self._predict is None:
            numerics.predictor(
                self._bcast(eD), x, self._bcast(hphi1), F, out, self._work
            )
            return
        self._predict(
            self._dim,
            self._K,
            self._per_lane,
            *self._ptrs((eD, hphi1, x, F, out)),
        )

    def corrector(self, a, F_a, F, hphi2, out):
        """``x = a + hphi2 (F_a - F)``: the C kernel, or the numpy lowering."""
        if self._correct is None:
            numerics.corrector(a, self._bcast(hphi2), F_a, F, out, self._work)
            return
        self._correct(
            self._dim,
            self._K,
            self._per_lane,
            *self._ptrs((hphi2, a, F_a, F, out)),
        )

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=self.dtype if dtype is None else dtype)

    def to_host(self, a):
        return np.asarray(a, dtype=np.float64)

    def synchronize(self):
        pass

    def close(self):
        """Release the buffers, the coupling and the pointer memo (which pins
        the driver's state planes). Safe before a bind, idempotent after."""
        self._bufs = self._corner = self._factors = None
        self._work = self._coupling = None
        self._ptr_cache = {}
        self._apply_off.close()


def numpy_backend(op, dtype=np.float64):
    """Host backend on scipy CSR SpMM."""
    dtype = state_dtype(dtype)
    return HostBackend(op, ScipyApplyOff(op.int_off, op.dec_off, dtype), dtype)
