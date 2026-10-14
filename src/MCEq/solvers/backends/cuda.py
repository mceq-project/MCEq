"""CUDA backend: CSR SpMM RawKernels and cupy ElementwiseKernels.

:class:`CudaOperator` uploads a compiled operator's split to the device
once, :class:`CudaBackend` runs the stages of
:func:`MCEq.solvers.etd2.etd2_driver` against it, and :func:`cuda_backend`
builds the pair. The off-diagonal SpMM and the Legendre band are the CSR
kernels of :data:`_CSR_SPMM_SRC`; the fused elementwise stages are the
kernel set of :func:`_build_cuda_etd2_kernels`. All compile on first use.

Everything that needs a device belongs here. cupy is imported inside the
functions that use it and the CUDA runtime libraries are dlopened there
too, so importing this module on a machine with no GPU costs nothing.
"""

from types import SimpleNamespace

import numpy as np

from MCEq.operators.compiled import coupled_corner
from MCEq.solvers.backends.base import _state_dtype
from MCEq.solvers.numerics import CORRECTOR_EXPR, PHI_C_BODY, PREDICTOR_EXPR


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
# The off-diagonal SpMM ``out = int_off x + ri dec_off x`` as one CSR
# kernel that streams both operators once whatever K. One row per group of
# ``S * C`` lanes of a warp: slice ``s`` strides the row's nonzeros, column
# group ``c`` owns ``K / C`` contiguous columns of the ``(dim, K)`` C-ordered
# state, so a group reads a state row as one coalesced segment; the slices
# are summed with warp shuffles; :func:`_lane_map` picks ``(S, C)``.
# The operators are ``n_blk`` diagonal blocks with one sparsity pattern each
# (the modes of a 2D operator, else ``n_blk = 1``), stored once.
# Groups walk tiles of ``_SPMM_TILE`` pattern rows through every block in
# turn, so a tile's indices are read from cache after the first block.
# ``csr_spmm_acc`` adds one operator's product to ``out`` on the same lane
# layout (the Legendre continuous-loss band).
# --------------------------------------------------------------------
_CSR_SPMM_SRC = r"""
template <typename T, int KC>
__device__ __forceinline__ void load_chunk(const T* __restrict__ p, T (&v)[KC])
{
    if constexpr (sizeof(T) == 8 && KC % 2 == 0) {
        #pragma unroll
        for (int i = 0; i < KC; i += 2) {
            const double2 q = *reinterpret_cast<const double2*>(p + i);
            v[i] = q.x; v[i + 1] = q.y;
        }
    } else if constexpr (sizeof(T) == 4 && KC % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < KC; i += 4) {
            const float4 q = *reinterpret_cast<const float4*>(p + i);
            v[i] = q.x; v[i + 1] = q.y; v[i + 2] = q.z; v[i + 3] = q.w;
        }
    } else if constexpr (sizeof(T) == 4 && KC % 2 == 0) {
        #pragma unroll
        for (int i = 0; i < KC; i += 2) {
            const float2 q = *reinterpret_cast<const float2*>(p + i);
            v[i] = q.x; v[i + 1] = q.y;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < KC; ++i) v[i] = p[i];
    }
}

// acc[i] = sum over the row's nonzeros of data[j] * x[indices[j], c*KC + i],
// summed over the S slices into the lanes with s == 0 (lane < C). The values
// are read as V and widened to the state type T before the product.
template <typename T, typename V, int K, int S, int C>
__device__ __forceinline__ void row_dot(const int* __restrict__ indptr,
                                        const int* __restrict__ indices,
                                        const V* __restrict__ data,
                                        const T* __restrict__ x,
                                        const int row, const bool active,
                                        const int lane, T (&acc)[K / C])
{
    constexpr int LPR = S * C, KC = K / C;
    const int s = lane / C, c = lane % C;
    #pragma unroll
    for (int i = 0; i < KC; ++i) acc[i] = T(0);
    const int start = active ? indptr[row] : 0;
    const int end = active ? indptr[row + 1] : 0;
    T v[KC];
    for (int j = start + s; j < end; j += S) {
        load_chunk<T, KC>(x + (size_t)indices[j] * K + c * KC, v);
        const T a = T(data[j]);
        #pragma unroll
        for (int i = 0; i < KC; ++i) acc[i] += a * v[i];
    }
    #pragma unroll
    for (int off = LPR / 2; off >= C; off >>= 1) {
        #pragma unroll
        for (int i = 0; i < KC; ++i)
            acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off, LPR);
    }
}

// y = A x + ri B x over (n_blk * N, K) C-ordered x and y. A and B are n_blk
// diagonal blocks of N rows with one pattern (ip, idx) each; block b's values
// start at b * nnz. ri is read as ri[k * ri_stride]: stride 0 for one shared
// value, 1 for one per lane. Every lane of a warp reaches the shuffles
// (inactive rows sum nothing).
template <typename T, typename V, int K, int S, int C>
__global__ void csr_spmm_off(const int N, const int n_blk, const int tile,
                             const int* __restrict__ ipA,
                             const int* __restrict__ idxA,
                             const V* __restrict__ dA, const int nnzA,
                             const int* __restrict__ ipB,
                             const int* __restrict__ idxB,
                             const V* __restrict__ dB, const int nnzB,
                             const T* __restrict__ x,
                             const T* __restrict__ ri, const int ri_stride,
                             T* __restrict__ y)
{
    constexpr int LPR = S * C, KC = K / C;
    const int lane = threadIdx.x & (LPR - 1);
    const int g = (blockIdx.x * blockDim.x + threadIdx.x) / LPR;
    // group g -> tile t, block b, pattern row i
    const int per_tile = tile * n_blk;
    const int t = g / per_tile, b = (g % per_tile) / tile, i = t * tile + g % tile;
    const bool active = i < N;
    const T* xb = x + (size_t)b * N * K;
    T accA[KC], accB[KC];
    row_dot<T, V, K, S, C>(ipA, idxA, dA + (size_t)b * nnzA, xb, i, active, lane, accA);
    row_dot<T, V, K, S, C>(ipB, idxB, dB + (size_t)b * nnzB, xb, i, active, lane, accB);
    if (active && lane < C) {
        const size_t row = (size_t)b * N + i;
        #pragma unroll
        for (int q = 0; q < KC; ++q) {
            const int k = lane * KC + q;
            y[row * K + k] = accA[q] + ri[k * ri_stride] * accB[q];
        }
    }
}

// y += A x: the continuous-loss band of a Legendre operator.
template <typename T, typename V, int K, int S, int C>
__global__ void csr_spmm_acc(const int n_rows,
                             const int* __restrict__ ipA,
                             const int* __restrict__ idxA,
                             const V* __restrict__ dA,
                             const T* __restrict__ x,
                             T* __restrict__ y)
{
    constexpr int LPR = S * C, KC = K / C;
    const int lane = threadIdx.x & (LPR - 1);
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / LPR;
    const bool active = row < n_rows;
    T acc[KC];
    row_dot<T, V, K, S, C>(ipA, idxA, dA, x, row, active, lane, acc);
    if (active && lane < C) {
        #pragma unroll
        for (int i = 0; i < KC; ++i) y[(size_t)row * K + lane * KC + i] += acc[i];
    }
}
"""

#: Threads per block of the SpMM kernel; ``S * C`` divides it.
_SPMM_BLOCK = 256

#: Pattern rows per tile of the SpMM kernel's row order.
_SPMM_TILE = 256

#: ``(S, C)`` of the SpMM kernel per state itemsize and K (powers of two),
#: tuned on an RTX 3090 with the 2D production operator (about 800 nonzeros
#: per row); :func:`_lane_map` narrows S for shorter rows.
_LANE_MAPS = {
    8: {
        1: (32, 1),
        2: (16, 2),
        4: (8, 4),
        8: (4, 8),
        16: (4, 8),
        32: (2, 16),
        64: (2, 16),
    },
    4: {
        1: (32, 1),
        2: (32, 1),
        4: (32, 1),
        8: (16, 2),
        16: (8, 4),
        32: (4, 8),
        64: (4, 8),
    },
}


def _lane_map(K, itemsize, row_nnz=None, kernel="csr_spmm_off"):
    """``(S, C)`` for ``K`` state columns: the :data:`_LANE_MAPS` entry, else
    ``(16, 1)``; ``S`` is halved while ``16 S`` exceeds ``row_nnz``, the
    nonzeros per row of both operators together. The band kernel
    ``csr_spmm_acc`` takes ``S = 1`` (about two nonzeros per row; within 3 %
    of the best S on the 2D Legendre band)."""
    S, C = _LANE_MAPS[itemsize].get(K, (16, 1))
    if kernel == "csr_spmm_acc":
        return 1, C
    while row_nnz is not None and S > 1 and 16 * S > row_nnz:
        S //= 2
    return S, C


_SPMM_KERNELS = {}


def _spmm_kernel(cp, dtype, K, row_nnz=None, kernel="csr_spmm_off", value_dtype=None):
    """The compiled ``kernel<T, V, K, S, C>`` of :data:`_CSR_SPMM_SRC` (state
    ``dtype``, operator values ``value_dtype``, default the state's) and its
    lanes per row; cached per process."""
    dtype = np.dtype(dtype)
    value_dtype = dtype if value_dtype is None else np.dtype(value_dtype)
    S, C = _lane_map(K, dtype.itemsize, row_nnz, kernel)
    key = (dtype, value_dtype, int(K), kernel, S, C)
    if key not in _SPMM_KERNELS:
        T, V = ({8: "double", 4: "float"}[d.itemsize] for d in (dtype, value_dtype))
        name = f"{kernel}<{T},{V},{K},{S},{C}>"
        mod = cp.RawModule(
            code=_CSR_SPMM_SRC, options=("-std=c++17",), name_expressions=(name,)
        )
        _SPMM_KERNELS[key] = (mod.get_function(name), S * C)
    return _SPMM_KERNELS[key]


#: Largest growth of the shared pattern over the fullest block, see
#: :func:`_block_pattern`.
_PATTERN_SLACK = 0.05


def _block_pattern(csr, n_blk):
    """``(indptr, indices, data)`` of one sparsity pattern shared by the
    ``n_blk`` diagonal blocks of ``csr``, with every block's values on it
    (block ``b`` at ``data[b * nnz:]``), or None.

    The pattern is the union of the blocks' patterns, explicit zeros where a
    block lacks an entry: the 2D modes nearly share one (the Legendre
    operators differ by a few residual entries at l = 0). None when ``csr``
    has entries off the diagonal blocks or the union exceeds the fullest
    block by more than :data:`_PATTERN_SLACK`; the full CSR is then kept.
    """
    dim = csr.shape[0]
    N = dim // n_blk
    if n_blk < 2 or N * n_blk != dim:
        return None
    blocks = [
        csr[b * N : (b + 1) * N, b * N : (b + 1) * N].tocsr() for b in range(n_blk)
    ]
    if sum(blk.nnz for blk in blocks) != csr.nnz:
        return None
    keys = [
        np.repeat(np.arange(N, dtype=np.int64), np.diff(blk.indptr)) * N + blk.indices
        for blk in blocks
    ]
    union = np.unique(np.concatenate(keys))
    if union.size > (1 + _PATTERN_SLACK) * max(blk.nnz for blk in blocks):
        return None
    data = np.zeros((n_blk, union.size), dtype=csr.dtype)
    for b, (blk, key) in enumerate(zip(blocks, keys)):
        data[b, np.searchsorted(union, key)] = blk.data
    indptr = np.searchsorted(union // N, np.arange(N + 1)).astype(np.int64)
    return indptr, union % N, data.ravel()


class _DeviceCsr:
    """The three arrays of a CSR matrix on the device, ``data`` in the
    operator value dtype; with a ``pattern`` of :func:`_block_pattern` the shared pattern
    once and every block's values on it. ``nnz`` is per block."""

    def __init__(self, cp, csr, dtype, pattern=None):
        indptr, indices, data = (
            (csr.indptr, csr.indices, csr.data) if pattern is None else pattern
        )
        self.nnz = int(indices.size)
        self.indptr = cp.asarray(indptr, dtype=cp.int32)
        self.indices = cp.asarray(indices, dtype=cp.int32)
        self.data = cp.asarray(data, dtype=dtype)


# --------------------------------------------------------------------
# cupy ElementwiseKernels of the CUDA backend
# --------------------------------------------------------------------
_CUDA_ETD2_KERNELS = None


def _build_cuda_etd2_kernels(cp):
    """Build the cupy ElementwiseKernel set of the CUDA backend.

    One kernel per driver stage, named for the stage, and every body
    generated from the formula table of :mod:`MCEq.solvers.numerics` --
    :data:`~MCEq.solvers.numerics.PHI_C_BODY` for the factor stage,
    :data:`~MCEq.solvers.numerics.PREDICTOR_EXPR` and
    :data:`~MCEq.solvers.numerics.CORRECTOR_EXPR` for the two state stages --
    so a change to a formula reaches this backend and the NumPy path
    together. The C kernels duplicate the same formulas and require matching
    edits; tests compare the implementations against each other.

    The state stages are dtype-agnostic through the ``T`` template; cupy
    compiles a specialisation per input dtype combination on first launch.
    They broadcast a ``(dim, 1)`` shared-path factor over the ``(dim, K)``
    state and take a ``(dim, K)`` per-lane factor as it is.

    ``diag_factors`` takes fp64 inputs and does its arithmetic in double
    whatever ``T`` is, writing the factors in the state dtype. That cast is
    the one place the state precision enters stage 1; see
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT` for why the
    arithmetic stays in double. Pass ``d_int[:, None], d_dec[:, None]``
    against scalar ``h`` / ``ri`` for a shared path, or against
    ``h_K[None, :], ri_K[None, :]`` for one path per lane; the broadcast
    shape is the output's.
    """
    diag_factors = cp.ElementwiseKernel(
        "float64 d_int, float64 d_dec, float64 h, float64 ri",
        "T eD, T hphi1, T hphi2",
        f"""
        const double z = h * (d_int + ri * d_dec);
        {PHI_C_BODY}
        eD = (T)e;
        hphi1 = (T)(h * p1);
        hphi2 = (T)(h * p2);
        """,
        "mceq_etd2_diag_factors",
    )
    predictor = cp.ElementwiseKernel(
        "T eD, T x, T hphi1, T F",
        "T a",
        f"a = {PREDICTOR_EXPR};",
        "mceq_etd2_predictor",
    )
    corrector = cp.ElementwiseKernel(
        "T a, T F_a, T F, T hphi2",
        "T x",
        f"x = {CORRECTOR_EXPR};",
        "mceq_etd2_corrector",
    )
    return SimpleNamespace(
        diag_factors=diag_factors,
        predictor=predictor,
        corrector=corrector,
    )


def _cuda_etd2_kernels():
    """Lazy singleton — cupy ElementwiseKernels for the multi-RHS path."""
    global _CUDA_ETD2_KERNELS
    if _CUDA_ETD2_KERNELS is None:
        import cupy as cp

        _CUDA_ETD2_KERNELS = _build_cuda_etd2_kernels(cp)
    return _CUDA_ETD2_KERNELS


class CudaOperator:
    """Device copy of a compiled operator's split for the CUDA backend.

    Owns the device CSR arrays of ``int_off`` / ``dec_off`` and, for a
    Legendre operator, ``band_off`` (:class:`_DeviceCsr`) in the state dtype
    of ``fp_precision`` (32 or 64), and the diagonals in fp64 whatever the
    precision (see
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`). When both
    off-diagonal matrices are ``n_k`` diagonal blocks whose patterns (nearly)
    agree, one shared pattern is kept (``n_blk = n_k``, :func:`_block_pattern`). The state and scratch
    buffers are allocated per solve by :class:`CudaBackend`; one device
    operator serves every K.
    """

    def __init__(
        self,
        int_off,
        dec_off,
        d_int,
        d_dec,
        device_id,
        fp_precision,
        n_k=1,
        band_off=None,
        operator_precision=None,
    ):
        # CuPy 13.x does not auto-discover the nvidia-* pip packages that
        # ship the CUDA 12 runtime libs; dlopen them before the first JIT
        # (a no-op where the libs are already on the loader path).
        _preload_nvidia_pip_libs()
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "CudaOperator: CuPy is not available. Install a build of "
                "cupy matching your CUDA runtime."
            ) from e
        fl_pr = _state_dtype(fp_precision)
        # the operator values: the state dtype, or fp32 under fp64 states
        # (mixed precision: half the bytes the SpMM streams per step)
        val = fl_pr if operator_precision is None else _state_dtype(operator_precision)
        if np.dtype(val).itemsize > np.dtype(fl_pr).itemsize:
            raise ValueError("operator_precision above the state precision")
        self.cp = cp
        self.fl_pr = fl_pr
        self.value_dtype = np.dtype(val)
        self.fp_precision = int(fp_precision)
        self.device_id = int(device_id)
        cp.cuda.Device(self.device_id).use()
        self.dim = int(d_int.shape[0])
        int_off, dec_off = int_off.tocsr(), dec_off.tocsr()
        patterns = [_block_pattern(m, n_k) for m in (int_off, dec_off)]
        if None in patterns:
            patterns = [None, None]
        self.n_blk = 1 if patterns[0] is None else int(n_k)
        self.cu_int_off = _DeviceCsr(cp, int_off, val, patterns[0])
        self.cu_dec_off = _DeviceCsr(cp, dec_off, val, patterns[1])
        self.cu_band_off = None if band_off is None else _DeviceCsr(cp, band_off, val)
        self.cu_d_int = cp.asarray(d_int, dtype=cp.float64)
        self.cu_d_dec = cp.asarray(d_dec, dtype=cp.float64)


class CudaBackend:
    """Stage execution on the device for
    :func:`MCEq.solvers.etd2.etd2_driver`.

    The CSR SpMM kernel of :func:`_spmm_kernel`, cublas GEMMs, and the fused
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
        self._band = None

    def bind(self, dim, K, per_lane, nsteps):
        cp, dtype = self.cp, self.dtype
        cp.cuda.Device(self.dev.device_id).use()
        self._per_lane = per_lane
        self._state = tuple(cp.empty((dim, K), dtype=dtype) for _ in range(4))
        A, B, n_blk = self.dev.cu_int_off, self.dev.cu_dec_off, self.dev.n_blk
        N = dim // n_blk
        vdt = self.dev.value_dtype
        self._spmm, lanes = _spmm_kernel(
            cp, dtype, K, (A.nnz + B.nnz) / N, value_dtype=vdt
        )
        n_tiles = (N + _SPMM_TILE - 1) // _SPMM_TILE
        groups = n_tiles * _SPMM_TILE * n_blk
        self._spmm_grid = ((groups * lanes + _SPMM_BLOCK - 1) // _SPMM_BLOCK,)
        self._spmm_args = (np.int32(N), np.int32(n_blk), np.int32(_SPMM_TILE)) + tuple(
            v for m in (A, B) for v in (m.indptr, m.indices, m.data, np.int32(m.nnz))
        )
        band = self.dev.cu_band_off
        if band is not None:
            kern, lanes = _spmm_kernel(
                cp, dtype, K, kernel="csr_spmm_acc", value_dtype=vdt
            )
            grid = ((dim * lanes + _SPMM_BLOCK - 1) // _SPMM_BLOCK,)
            self._band = (kern, grid, (band.indptr, band.indices, band.data))
        # (dim, 1) for one shared integration path, (dim, K) for one per lane;
        # either broadcasts over the state in the two elementwise stages.
        self._factors = tuple(
            cp.empty((dim, K if per_lane else 1), dtype=dtype) for _ in range(3)
        )
        self._diag = (self.d_int[:, None], self.d_dec[:, None])
        # the corner's diagonals lam_j D0_i, broadcast over the lane axis
        self._corner_diag = None
        if self.op.layout.coupled:
            self._corner_diag = tuple(
                cp.asarray(d, dtype=cp.float64)[:, :, None]
                for d in self.op.exact_slot_diagonals
            )

    def coupling(self):
        """``W, V, Vi`` of the compiled operator on the device, in the state
        dtype."""
        if self._coupling is None:
            c = self.op.coupling
            self._coupling = tuple(
                self.cp.asarray(m, dtype=self.dtype) for m in (c.W, c.V, c.Vi)
            )
        return self._coupling

    def state_buffers(self, dim, K):
        return self._state

    def apply_band(self, x, out):
        """``out += band_off x``: the continuous-loss off-diagonal of a
        Legendre operator, from the device copy uploaded with the operator."""
        kern, grid, args = self._band
        kern(grid, (_SPMM_BLOCK,), (np.int32(x.shape[0]), *args, x, out))

    def apply_off(self, x, out, ri):
        """``out = int_off x + ri dec_off x`` in one launch. ``ri`` is a
        device array in the state dtype, 0-d for a shared path or ``(K,)``
        per lane."""
        stride = np.int32(0 if ri.ndim == 0 else 1)
        self._spmm(
            self._spmm_grid, (_SPMM_BLOCK,), (*self._spmm_args, x, ri, stride, out)
        )

    def left_matmul(self, matrix, plane, out=None):
        """The device form of :func:`MCEq.solvers.numerics.left_matmul`."""
        plane_2d = plane.reshape(plane.shape[0], -1)
        if out is None:
            return (matrix @ plane_2d).reshape((matrix.shape[0],) + plane.shape[1:])
        self.cp.matmul(matrix, plane_2d, out=out.reshape(matrix.shape[0], -1))
        return out

    def diag_factors(self, h, ri):
        """``eD, h phi1, h phi2`` of ``h (d_int + ri d_dec)``, broadcastable
        to (dim, K); on the sec(theta) corner the factors of ``h lam_j D0_i``.
        ``h`` and ``ri`` come in fp64 and the diagonals are fp64; the kernel
        evaluates there and writes the factors in the state dtype, so the
        cast happens once, at the kernel's output."""
        d_int, d_dec = self._diag
        if self._per_lane:
            h, ri = h[None, :], ri[None, :]
        self._kernels.diag_factors(d_int, d_dec, h, ri, *self._factors)
        if self._corner_diag is not None:
            corners = (coupled_corner(self.op.layout, f) for f in self._factors)
            self._kernels.diag_factors(*self._corner_diag, h, ri, *corners)
        return self._factors

    def predictor(self, eD, x, hphi1, F, out):
        self._kernels.predictor(eD, x, hphi1, F, out)

    def corrector(self, a, F_a, F, hphi2, out):
        self._kernels.corrector(a, F_a, F, hphi2, out)

    def asarray(self, a, dtype=None):
        return self.cp.asarray(a, dtype=self.dtype if dtype is None else dtype)

    def to_host(self, a):
        return self.cp.asnumpy(a).astype(np.float64, copy=False)

    def synchronize(self):
        self.cp.cuda.Stream.null.synchronize()

    def close(self):
        self._state = self._factors = self._diag = self._corner_diag = None
        self._spmm = self._spmm_args = self._coupling = self._band = None


def cuda_backend(op, device_id=0, fp_precision=64, operator_precision=None):
    """Device backend; uploads the compiled operator's split once.
    ``operator_precision=32`` stores the off-diagonal values in fp32 under an
    fp64 state (mixed precision); None keeps them in the state precision."""
    dev = CudaOperator(
        op.int_off,
        op.dec_off,
        op.d_int,
        op.d_dec,
        device_id,
        fp_precision,
        op.layout.n_k,
        band_off=op.band_off,
        operator_precision=operator_precision,
    )
    return CudaBackend(dev, op)
