"""CUDA backend: cuSPARSE SpMM and cupy ElementwiseKernels.

:class:`CudaOperator` uploads a compiled operator's split to the device
once, :class:`CudaBackend` runs the stages of
:func:`MCEq.solvers.etd2.etd2_driver` against it, and :func:`cuda_backend`
builds the pair. The fused elementwise stages are the kernel set of
:func:`_build_cuda_etd2_kernels`, compiled on first use.

Everything that needs a device belongs here. cupy is imported inside the
functions that use it and the CUDA runtime libraries are dlopened there
too, so importing this module on a machine with no GPU costs nothing.
"""

from types import SimpleNamespace

import numpy as np

from MCEq.solvers.backends.base import _state_dtype
from MCEq.solvers.numerics import _INV_6, _INV_24, _PHI1_SMALL, _PHI2_SMALL


def _preload_nvidia_pip_libs():
    """Dlopen the nvidia-* pip-package CUDA libs so cupy 13 can find them.

    CuPy 14 auto-discovers the ``nvidia/<lib>/lib`` directories from
    site-packages, but CuPy 13 does not — and on systems where the system
    CUDA toolkit is a different major (e.g. CUDA 13 toolkit + cupy-cuda12x),
    that mismatch leaves the cupy loader unable to find ``libnvrtc.so.12``
    etc. This function looks for the standard pip wheels
    (``nvidia-cuda-nvrtc-cu12``, ``nvidia-cuda-runtime-cu12``,
    ``nvidia-cusparse-cu12``, ``nvidia-cublas-cu12``) and dlopens whichever
    are present. Missing packages are silently ignored — if a library is
    actually needed, cupy will still report the missing-symbol error.
    """
    import ctypes
    import importlib
    import os

    # Order matters: cublas depends on cudart, nvrtc depends on
    # nvrtc-builtins (which lives next to nvrtc itself). Load runtimes first.
    package_libs = [
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.cublas", "libcublasLt.so.12"),
        ("nvidia.cublas", "libcublas.so.12"),
        ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
        ("nvidia.cusparse", "libcusparse.so.12"),
    ]
    for pkg_name, libname in package_libs:
        try:
            mod = importlib.import_module(pkg_name)
        except ImportError:
            continue
        # The nvidia.* wheels are PEP-420 namespace packages under some
        # installers (uv): ``__file__`` is None and the package may span
        # several site-packages dirs — walk ``__path__`` instead.
        if getattr(mod, "__file__", None) is not None:
            pkg_dirs = [os.path.dirname(mod.__file__)]
        else:
            pkg_dirs = list(getattr(mod, "__path__", []))
        for pkg_dir in pkg_dirs:
            candidate = os.path.join(pkg_dir, "lib", libname)
            if not os.path.isfile(candidate):
                continue
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Best effort — if the dlopen fails, cupy will still try
                # its own discovery and report a more specific error if
                # it's actually missing.
                pass
            break


# --------------------------------------------------------------------
# cupy ElementwiseKernels of the CUDA backend
#
# The SpMM is eager cuSPARSE through ``cupyx.scipy.sparse.csr_matrix @
# dense_2d``. No CUDA Graph capture: cupy 14 explicitly blocks cuSPARSE
# during ``stream.begin_capture()`` and PriNCe found (and we confirmed)
# the eager SpMM is already amortised at K ≥ 32, so the graph win for
# multi-RHS is marginal. The fused elementwise stages broadcast the
# per-step factors across the K axis.
# --------------------------------------------------------------------
_CUDA_ETD2_KERNELS = None


def _build_cuda_etd2_kernels(cp):
    """Build the cupy ElementwiseKernel set of the CUDA backend.

    Transplanted from PriNCe's etd2.py (lines 57–131). The kernels broadcast
    the (dim,) per-step factors over the (dim, K) state via cupy's
    ElementwiseKernel broadcasting (pass ``factor[:, None]`` at the call
    site). Kept dtype-agnostic via the ``T`` template — cupy compiles a
    specialisation per (input dtype combination) on first launch.

    ``post_apply1`` / ``post_apply2`` also serve the per-RHS-path
    (multipath) variant: pass ``h`` as a ``(1, K)`` broadcasted buffer
    instead of a Python scalar — the kernel signature is unchanged
    because ``T`` accepts both scalars and arrays.

    The two diag-factor kernels take fp64 inputs and do their arithmetic
    in double whatever ``T`` is, writing the factors in the state dtype.
    That cast is the one place the state precision enters stage 1; see
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT` for why the
    arithmetic stays in double.
    """
    phi_compute = cp.ElementwiseKernel(
        "float64 hD, float64 eD_in",
        "T eD_out, T phi1, T phi2",
        f"""
        double abs_hd = (hD >= 0.0) ? hD : -hD;
        double p1, p2;
        eD_out = (T)eD_in;
        if (abs_hd > {_PHI1_SMALL!r}) {{
            p1 = (eD_in - 1.0) / hD;
        }} else {{
            p1 = 1.0 + hD * (0.5 + hD * {_INV_6!r});
        }}
        if (abs_hd > {_PHI2_SMALL!r}) {{
            p2 = (eD_in - 1.0 - hD) / (hD * hD);
        }} else {{
            p2 = 0.5 + hD * ({_INV_6!r} + hD * {_INV_24!r});
        }}
        phi1 = (T)p1;
        phi2 = (T)p2;
        """,
        "mceq_etd2_phi_compute",
    )
    # Per-RHS-path diag factor kernel: D = d_int + ri * d_dec ; hD = h * D ;
    # eD = exp(hD) ; phi1, phi2 via the same analytic/Taylor branch as
    # ``phi_compute``. Single fused kernel — saves the intermediate (dim, K)
    # hD/eD buffers vs the staged numpy implementation. Pass
    # ``d_int[:, None], d_dec[:, None], h_K[None, :], ri_K[None, :]`` to
    # broadcast onto the (dim, K) output shape.
    phi_compute_multipath = cp.ElementwiseKernel(
        "float64 d_int, float64 d_dec, float64 h, float64 ri",
        "T eD, T phi1, T phi2",
        f"""
        double D = d_int + ri * d_dec;
        double hd = h * D;
        double e = exp(hd);
        double abs_hd = (hd >= 0.0) ? hd : -hd;
        double p1, p2;
        eD = (T)e;
        if (abs_hd > {_PHI1_SMALL!r}) {{
            p1 = (e - 1.0) / hd;
        }} else {{
            p1 = 1.0 + hd * (0.5 + hd * {_INV_6!r});
        }}
        if (abs_hd > {_PHI2_SMALL!r}) {{
            p2 = (e - 1.0 - hd) / (hd * hd);
        }} else {{
            p2 = 0.5 + hd * ({_INV_6!r} + hd * {_INV_24!r});
        }}
        phi1 = (T)p1;
        phi2 = (T)p2;
        """,
        "mceq_etd2_phi_compute_multipath",
    )
    post_apply1 = cp.ElementwiseKernel(
        "T eD, T state, T phi1, T F_phi, T h",
        "T a",
        "a = eD * state + h * phi1 * F_phi;",
        "mceq_etd2_post_apply1",
    )
    post_apply2 = cp.ElementwiseKernel(
        "T a, T F_a, T F_phi, T phi2, T h",
        "T state",
        "state = a + h * phi2 * (F_a - F_phi);",
        "mceq_etd2_post_apply2",
    )
    return SimpleNamespace(
        phi_compute=phi_compute,
        phi_compute_multipath=phi_compute_multipath,
        post_apply1=post_apply1,
        post_apply2=post_apply2,
    )


def _cuda_etd2_kernels():
    """Lazy singleton — cupy ElementwiseKernels for the multi-RHS path."""
    global _CUDA_ETD2_KERNELS
    if _CUDA_ETD2_KERNELS is None:
        import cupy as cp

        _CUDA_ETD2_KERNELS = _build_cuda_etd2_kernels(cp)
    return _CUDA_ETD2_KERNELS


def _cuda_secant_phi_factors(cp, ZB):
    """cupy form of :func:`MCEq.solvers.numerics._secant_phi_factors`
    (cupy ufuncs take no ``where``).

    ``ZB`` is fp64 on both backends, so the block factors are too."""
    safe = cp.where(ZB == 0.0, 1.0, ZB)
    e1 = cp.expm1(ZB)
    phi1B = cp.where(
        cp.abs(ZB) > _PHI1_SMALL, e1 / safe, 1.0 + ZB * (0.5 + ZB * _INV_6)
    )
    phi2B = cp.where(
        cp.abs(ZB) > _PHI2_SMALL,
        (e1 - ZB) / (safe * safe),
        0.5 + ZB * (_INV_6 + ZB * _INV_24),
    )
    return cp.exp(ZB), phi1B, phi2B


class CudaOperator:
    """Device copy of a compiled operator's split for the CUDA backend.

    Owns the cuSPARSE CSR copies of ``int_off`` / ``dec_off`` (``None``
    when empty — an empty CSR is ill-defined for some cuSPARSE versions)
    in the state dtype of ``fp_precision`` (32 or 64), and the diagonals
    in fp64 whatever the precision — see
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`. The
    state and scratch buffers are allocated per solve by
    :class:`CudaBackend`; one device operator serves every K.
    """

    def __init__(self, int_off, dec_off, d_int, d_dec, device_id, fp_precision):
        # CuPy 13.x does not auto-discover the nvidia-* pip packages that
        # ship the CUDA 12 runtime libs; dlopen them before the first JIT
        # (a no-op where the libs are already on the loader path).
        _preload_nvidia_pip_libs()
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cusp
        except ImportError as e:
            raise RuntimeError(
                "CudaOperator: CuPy is not available. Install a build of "
                "cupy matching your CUDA runtime."
            ) from e
        fl_pr = _state_dtype(fp_precision)
        self.cp = cp
        self.fl_pr = fl_pr
        self.fp_precision = int(fp_precision)
        self.device_id = int(device_id)
        cp.cuda.Device(self.device_id).use()
        self.dim = int(d_int.shape[0])
        self.cu_int_off = cusp.csr_matrix(int_off, dtype=fl_pr) if int_off.nnz else None
        self.cu_dec_off = cusp.csr_matrix(dec_off, dtype=fl_pr) if dec_off.nnz else None
        self.cu_d_int = cp.asarray(d_int, dtype=cp.float64)
        self.cu_d_dec = cp.asarray(d_dec, dtype=cp.float64)


class CudaBackend:
    """Stage execution on the device for
    :func:`MCEq.solvers.etd2.etd2_driver`.

    cuSPARSE SpMM through cupyx, cublas GEMMs, and the fused
    ElementwiseKernels of :func:`_cuda_etd2_kernels`. ``dev`` is the
    :class:`CudaOperator` of ``op``'s split, and carries the state dtype.
    At ``fp_precision=32`` everything runs in fp32 except the diagonals
    and the phi factors: those come out of fp64 inputs, are evaluated in
    fp64 inside the kernels, and are cast on the kernels' way out — the
    same stages in the same order as
    :class:`MCEq.solvers.backends.host.HostBackend`. See
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`.
    """

    def __init__(self, dev, op):
        if dev.dim != op.dim:
            raise ValueError(f"CudaBackend: device operator dim {dev.dim} != {op.dim}")
        self.op = op
        self.name = "cuda"
        self.dev = dev
        self.cp = self.xp = dev.cp
        self.dtype = dev.fl_pr
        self.d_int = dev.cu_d_int
        self.d_dec = dev.cu_d_dec
        self._kernels = _cuda_etd2_kernels()
        self._coupling = None

    def bind(self, dim, K, per_lane, nsteps):
        cp, dtype = self.cp, self.dtype
        cp.cuda.Device(self.dev.device_id).use()
        self._per_lane = per_lane
        self._state = tuple(cp.empty((dim, K), dtype=dtype) for _ in range(4))
        self._dec_tmp = (
            None if self.dev.cu_dec_off is None else cp.empty((dim, K), dtype=dtype)
        )
        shape = (dim, K) if per_lane else (dim,)
        self._factors = tuple(cp.empty(shape, dtype=dtype) for _ in range(3))
        # D, hD and exp(hD) of the shared-path stage, in fp64.
        self._D = (
            None
            if per_lane
            else tuple(cp.empty(dim, dtype=cp.float64) for _ in range(3))
        )

    def coupling(self):
        """``T_P, T_PP, V, Vi`` in the state dtype (they act on the state);
        ``lam`` in fp64 (it forms the phi arguments of the exact slot)."""
        if self._coupling is None:
            c = self.op.coupling
            cp = self.cp
            self._coupling = tuple(
                cp.asarray(m, dtype=self.dtype) for m in (c.T_P, c.T_PP, c.V, c.Vi)
            ) + (cp.asarray(c.lam, dtype=cp.float64),)
        return self._coupling

    def state_buffers(self, dim, K):
        return self._state

    def apply_off(self, x, out, ri):
        dev, cp = self.dev, self.cp
        if dev.cu_int_off is None:
            out.fill(0)
        else:
            cp.copyto(out, dev.cu_int_off @ x)
        if dev.cu_dec_off is not None:
            tmp = self._dec_tmp
            cp.copyto(tmp, dev.cu_dec_off @ x)
            cp.multiply(tmp, ri, out=tmp)
            out += tmp

    def left_matmul(self, matrix, plane, out):
        plane_2d = plane.reshape(plane.shape[0], -1)
        self.cp.matmul(matrix, plane_2d, out=out.reshape(matrix.shape[0], -1))
        return out

    def diag_factors(self, h, ri):
        """``eD, phi1, phi2`` of ``h (d_int + ri d_dec)``, broadcastable to
        (dim, K). ``h`` and ``ri`` come in fp64 and the diagonals are fp64;
        the kernels evaluate there and write the factors in the state
        dtype, so the cast happens once, at the kernel's output."""
        cp, Kset = self.cp, self._kernels
        d_int, d_dec = self.d_int, self.d_dec
        eD, phi1, phi2 = self._factors
        if self._per_lane:
            Kset.phi_compute_multipath(
                d_int[:, None], d_dec[:, None], h[None, :], ri[None, :], eD, phi1, phi2
            )
            return eD, phi1, phi2
        D, hD, eD64 = self._D
        cp.multiply(d_dec, ri, out=D)
        cp.add(D, d_int, out=D)
        cp.multiply(D, h, out=hD)
        cp.exp(hD, out=eD64)
        Kset.phi_compute(hD, eD64, eD, phi1, phi2)
        return eD[:, None], phi1[:, None], phi2[:, None]

    def block_factors(self, ZB):
        return _cuda_secant_phi_factors(self.cp, ZB)

    def predictor(self, eD, x, phi1, F, h, out):
        self._kernels.post_apply1(eD, x, phi1, F, h, out)

    def corrector(self, a, F_a, F, phi2, h, out):
        self._kernels.post_apply2(a, F_a, F, phi2, h, out)

    def asarray(self, a, dtype=None):
        return self.cp.asarray(a, dtype=self.dtype if dtype is None else dtype)

    def to_host(self, a):
        return self.cp.asnumpy(a).astype(np.float64, copy=False)

    def synchronize(self):
        self.cp.cuda.Stream.null.synchronize()

    def close(self):
        self._state = self._factors = self._D = self._dec_tmp = self._coupling = None


def cuda_backend(op, device_id=0, fp_precision=64):
    """Device backend; uploads the compiled operator's split once."""
    dev = CudaOperator(
        op.int_off, op.dec_off, op.d_int, op.d_dec, device_id, fp_precision
    )
    return CudaBackend(dev, op)
