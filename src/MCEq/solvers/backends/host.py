"""The host backend: numpy elementwise kernels, C step stages, BLAS GEMMs.

:class:`HostBackend` executes every stage of
:func:`MCEq.solvers.etd2.etd2_driver` on host arrays and delegates only the
off-diagonal SpMM to an ``apply_off`` binding -- scipy in
:mod:`MCEq.solvers.backends.base`, MKL in
:mod:`MCEq.solvers.backends.mkl`, Apple Accelerate in
:mod:`MCEq.solvers.backends.accelerate`. The factor stages are the numpy
lowering of :mod:`MCEq.solvers.numerics`; the predictor and the corrector are
its C lowering, :mod:`MCEq.etd2_kernels`, at both precisions and every
problem size.

Stage code shared by all three host bindings belongs here; a binding's own
ctypes plumbing belongs in that binding's module.
"""

from ctypes import POINTER, c_double, c_float

import numpy as np

from MCEq.solvers import numerics
from MCEq.solvers.backends.base import ScipyApplyOff, _state_dtype
from MCEq.solvers.numerics import (
    diagonal_factors,
    left_matmul,
    phi_factors,
    phi_work,
    step_buffers,
)

#: ctypes pointer type per state dtype, for the fused C stages.
_C_POINTER = {np.float64: POINTER(c_double), np.float32: POINTER(c_float)}


def _fused_stages(dtype):
    """The C predictor / corrector of :mod:`MCEq.etd2_kernels` at ``dtype``,
    or ``None`` when the extension is not built.

    Imported on demand rather than at module level: importing
    :mod:`MCEq.solvers` must not dlopen anything (see
    ``tests/test_solvers_import.py``). An unbuilt source tree, or a platform
    where the extension does not compile, falls back to the numpy lowering of
    the same table; the two agree to the bit, which
    ``test_c_stages_match_numpy_lowering`` pins.
    """
    try:
        from MCEq.etd2_kernels import (
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

    numpy elementwise kernels, the fused C step stages and BLAS GEMMs; the
    SpMM is the ``apply_off`` binding of the sparse library (scipy, MKL or
    Apple Accelerate). ``op`` is
    the :class:`~MCEq.operator_assembly.CompiledOperator` the binding was
    built from — it carries the layout and the coupling operators.

    ``dtype`` is the state precision, float64 or float32: the state, the
    scratch buffers and ``apply_off``'s operator are stored in it, while
    the diagonals and the phi factors are computed in fp64 and cast once,
    in :meth:`diag_factors`. See
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`.
    """

    xp = np
    left_matmul = staticmethod(left_matmul)

    def __init__(self, op, apply_off, dtype=np.float64):
        self.op = op
        self.name = apply_off.name
        self.dtype = np.dtype(dtype).type
        self.d_int, self.d_dec = op.d_int, op.d_dec
        self._apply_off = apply_off
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
        # Dropped first, so two generations of scratch never coexist.
        self._bufs = self._block = self._factors = self._work = None
        self._ptr_cache = {}
        self._bufs = step_buffers(shape)
        stages = _fused_stages(self.dtype)
        self._predict, self._correct = stages or (None, None)
        self._ptr = _C_POINTER[self.dtype]
        # Scratch for the numpy lowering, allocated only when it is what runs.
        self._work = None if stages else np.empty((dim, K), dtype=self.dtype)
        # Landing buffers for the one cast of the fp64 factors, or None at
        # fp64 where the factor buffers are already the state dtype.
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
        """``eD, h phi1, h phi2`` of ``h (d_int + ri d_dec)``.

        ``h`` and ``ri`` come in fp64 and the evaluation runs there; only the
        result is cast. The arrays are ``(dim,)`` for a shared integration
        path and ``(dim, K)`` for one path per lane — the C stages take the
        lane stride from ``per_lane``, not from the shape, so these are the
        backend's own buffers rather than views of them."""
        b = self._bufs
        diagonal_factors(h, ri, self.d_int, self.d_dec, b)
        return self._cast_factors(b["eD"], b["phi1"], b["phi2"])

    def block_factors(self, ZB):
        """``eDB, phi1B, phi2B`` of the coupled plane's argument, in fp64.

        No step size folded in: the exact slot applies ``h`` after the
        eigenbasis GEMMs, where it scales the coupled corner as a whole."""
        if self._block is None or self._block[0].shape != ZB.shape:
            self._block = tuple(np.empty(ZB.shape) for _ in range(3)) + phi_work(
                ZB.shape
            )
        eDB, phi1B, phi2B = self._block[:3]
        phi_factors(ZB, eDB, phi1B, phi2B, self._block[3:])
        return eDB, phi1B, phi2B

    def _ptrs(self, arrays):
        """ctypes pointers to `arrays`, memoized for the life of the solve.

        Every array the two stages touch is allocated once, in :meth:`bind`
        or :meth:`state_buffers`, and handed back unchanged on every step.
        ``ndarray.ctypes.data_as`` costs ~2.5 us per array, which is more
        than the kernel itself at K = 1, so the pointers are built once. The
        key is ``id(arr)``, unique only among live objects, so the entry
        carries its array and the identity check refuses a hit that a
        recycled id would otherwise satisfy. The entry is not what keeps the
        array alive: ``data_as`` pins it through the pointer's own ``._arr``,
        which is why :meth:`close` has to drop the cache to release it.
        """
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
        """A shared-path factor as a column, so numpy broadcasts it over the
        lane axis. The C stages take it flat and index it under ``per_lane``
        instead, so only the numpy lowering needs this."""
        return factor if self._per_lane or factor.ndim == 2 else factor[:, None]

    def predictor(self, eD, x, hphi1, F, out):
        """``a = eD x + hphi1 F`` — one fused pass of the C kernel, or the
        numpy lowering of the same expression when it is not built."""
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
        """``x = a + hphi2 (F_a - F)`` — one fused pass of the C kernel, or
        the numpy lowering of the same expression when it is not built."""
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
        """Release the step buffers, the lazily built coupling corner and the
        pointer memo -- which is what holds the driver's state planes, since
        ``data_as`` pins through the pointer. Assignment only, so it is safe
        before a bind and idempotent after one."""
        self._bufs = self._block = self._factors = None
        self._work = self._coupling = None
        self._ptr_cache = {}
        self._apply_off.close()


def numpy_backend(op, fp_precision=64):
    """Host backend on scipy CSR SpMM."""
    dtype = _state_dtype(fp_precision)
    return HostBackend(op, ScipyApplyOff(op.int_off, op.dec_off, dtype), dtype)
