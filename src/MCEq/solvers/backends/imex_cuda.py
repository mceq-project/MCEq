"""CUDA kernels of the IMEX level sweep (:class:`MCEq.solvers.imex.LevelSweep`).

Per level two launches: ``imex_gather`` sums the products of every row's entries with the solved operand in chunks
of ``CHUNK`` entries, one thread block per chunk and tile of columns (``threadIdx.x`` over modes x columns,
``threadIdx.y`` interleaved lanes); ``imex_finalize`` adds a row's chunk sums in a fixed order and applies the mode mixing, the diagonal and,
for a state in the sec(theta) corner, its dense mode block. The launches of a whole sweep are captured once per
column count as a CUDA graph; the step's ``c`` and ``ri`` are read from device memory.

The sweep is GMRES's preconditioner, so it runs in fp32 by default: on GPUs with few fp64 units the dense mode
algebra of the corner states dominates in fp64, and GMRES (which keeps the preconditioned vectors) corrects the
rounding against the fp64 operator.
"""

import numpy as np

#: Entries per gather block.
CHUNK = 32

_SOURCE = r"""
typedef REAL real;
// A block covers one chunk (gather) or one state (finalize) and a tile of Kt columns: threadIdx.x = t runs over
// (mode l, column kc) with the column fastest (coalesced in the (N, nk, K) state), threadIdx.y over entry lanes.
extern "C" __global__ void imex_gather(
    const int c0, const int nk, const int K, const int Kt,
    const int* __restrict__ ch_lo, const int* __restrict__ ch_hi, const int* __restrict__ ch_band,
    const int* __restrict__ Mi, const real* __restrict__ Ma, const real* __restrict__ Md,
    const int* __restrict__ Bi, const real* __restrict__ Bv,
    const real* __restrict__ par, const real* __restrict__ Xs, real* part)
{
    extern __shared__ real red[];  // [LANES][TX]
    const int TX = blockDim.x, LANES = blockDim.y, t = threadIdx.x, y = threadIdx.y;
    const int ch = c0 + blockIdx.x, kc = t % Kt, l = t / Kt, col = blockIdx.y * Kt + kc;
    const bool ok = (l < nk) && (col < K);
    const real ri = par[1];
    real s = 0.0;
    if (ok) {
        const int lo = ch_lo[ch], hi = ch_hi[ch];
        if (ch_band[ch] == 0) {
            for (int e = lo + y; e < hi; e += LANES) {
                const size_t v = (size_t)e * nk + l;
                s += (Ma[v] + ri * Md[v]) * Xs[((size_t)Mi[e] * nk + l) * K + col];
            }
        } else {
            for (int e = lo + y; e < hi; e += LANES)
                s += Bv[(size_t)e * nk + l] * Xs[((size_t)Bi[e] * nk + l) * K + col];
        }
    }
    red[y * TX + t] = s;
    __syncthreads();
    if (y == 0 && ok) {
        real u = 0.0;
        for (int k = 0; k < LANES; ++k) u += red[k * TX + t];
        part[((size_t)ch * nk + l) * K + col] = u;
    }
}

extern "C" __global__ void imex_finalize(
    const int a, const int nk, const int nP, const int nD, const int K, const int Kt,
    const int* __restrict__ rc, const int* __restrict__ bs, const real* __restrict__ part,
    const real* __restrict__ mixT, const real* __restrict__ TPT, const real* __restrict__ kinvT,
    const int* __restrict__ slot, const real* __restrict__ rhs, const real* __restrict__ den,
    const real* __restrict__ w, const real* __restrict__ par, real* X, real* Xs)
{
    // Chunks [rc[r], bs[r]) are the mixed group, [bs[r], rc[r + 1]) the band; per-column vectors are [nk][Kt].
    extern __shared__ real sh[];
    const int TX = blockDim.x, LANES = blockDim.y, t = threadIdx.x, y = threadIdx.y;
    real* red = sh;                       // [LANES][TX]
    real* s_p = sh + LANES * TX;          // produced, mixed group
    real* s_b = s_p + nk * Kt;            // band
    real* s_z = s_b + nk * Kt;            // diagonal solve
    real* s_u = s_z + nk * Kt;            // T_P z, [nP][Kt]
    const int r = a + blockIdx.x, kc = t % Kt, l = t / Kt, col = blockIdx.y * Kt + kc;
    const bool ok = (l < nk) && (col < K), lead = ok && (y == 0);
    const int m = l * Kt + kc;
    const real c = par[0];
    const size_t d = (size_t)r * nk + l, q = d * K + col;
    const int s = slot[r];  // uniform in the block

    if (lead) {
        real pm = 0, pb = 0;
        for (int h = rc[r]; h < bs[r]; ++h) pm += part[((size_t)h * nk + l) * K + col];
        for (int h = bs[r]; h < rc[r + 1]; ++h) pb += part[((size_t)h * nk + l) * K + col];
        s_p[m] = pm;
        s_b[m] = pb;
    }
    __syncthreads();
    // backward-removal mixing of the produced particles, then the diagonal
    real v = 0;
    if (ok && l < nD)
        for (int k = y; k < nD; k += LANES) v += mixT[k * nD + l] * s_p[k * Kt + kc];
    red[y * TX + t] = v;
    __syncthreads();
    if (lead) {
        real p = s_p[m];
        if (l < nD) {
            p = 0;
            for (int k = 0; k < LANES; ++k) p += red[k * TX + t];
        }
        s_z[m] = (rhs[q] + c * (p + s_b[m])) / den[d];
    }
    __syncthreads();
    if (s < 0) {
        if (lead) { X[q] = s_z[m]; Xs[q] = s_z[m]; }
        return;
    }
    // corner state: u = T_P z, v = K^-1 u, x = z + c w v, Sx = x + v
    v = 0;
    if (ok && l < nP)
        for (int k = y; k < nk; k += LANES) v += TPT[k * nP + l] * s_z[k * Kt + kc];
    red[y * TX + t] = v;
    __syncthreads();
    if (lead && l < nP) {
        real u = 0;
        for (int k = 0; k < LANES; ++k) u += red[k * TX + t];
        s_u[m] = u;
    }
    __syncthreads();
    v = 0;
    if (ok && l < nP) {
        const real* Ks = kinvT + (size_t)s * nP * nP;
        for (int j = y; j < nP; j += LANES) v += Ks[j * nP + l] * s_u[j * Kt + kc];
    }
    red[y * TX + t] = v;
    __syncthreads();
    if (lead) {
        real x = s_z[m], xs = x;
        if (l < nP) {
            real u = 0;
            for (int k = 0; k < LANES; ++k) u += red[k * TX + t];
            x += c * w[d] * u;
            xs = x + u;
        }
        X[q] = x;
        Xs[q] = xs;
    }
}
"""


def _chunks(indptr, n_rows):
    """``(row, lo, hi)`` of every ``CHUNK``-entry piece of the rows of a CSR ``indptr``."""
    counts = -(-np.diff(indptr[: n_rows + 1]) // CHUNK)
    rows = np.repeat(np.arange(n_rows), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    lo = indptr[rows] + CHUNK * (np.arange(len(rows)) - first)
    hi = np.minimum(lo + CHUNK, indptr[rows + 1])
    return rows, lo, hi


class CudaLevelKernel:
    """The sweep of a cupy :class:`MCEq.solvers.imex.LevelSweep` as a CUDA graph of per-level kernels."""

    def __init__(self, sweep, dtype=np.float32):
        import cupy as cp

        self.cp = cp
        op = sweep.op
        N = op.N

        def i32(a):
            return cp.asarray(a, dtype=cp.int32)

        self.dtype = dtype = np.dtype(dtype)

        def f64(a):  # values in the sweep's precision
            return cp.ascontiguousarray(cp.asarray(a, dtype=dtype))

        m_ptr, m_cols, va, vd = sweep._mixed
        b_ptr, b_cols, vb = sweep._band
        self.entries = (i32(m_cols), f64(va), f64(vd), i32(b_cols), f64(vb))

        # chunks ordered by row, the mixed group before the band in every row
        m_row, m_lo, m_hi = _chunks(m_ptr, N)
        b_row, b_lo, b_hi = _chunks(b_ptr, N)
        row = np.r_[m_row, b_row]
        band = np.r_[np.zeros(len(m_row), int), np.ones(len(b_row), int)]
        order = np.lexsort((band, row))
        self.chunks = (
            i32(np.r_[m_lo, b_lo][order]),
            i32(np.r_[m_hi, b_hi][order]),
            i32(band[order]),
        )
        self.n_chunks = len(order)
        row_chunks = np.searchsorted(row[order], np.arange(N + 1))
        self.rc = i32(row_chunks)
        self.bs = i32(row_chunks[:-1] + np.bincount(m_row, minlength=N))
        self.level_chunks = [
            (int(row_chunks[a]), int(row_chunks[b])) for a, b in sweep.levels
        ]

        dummy = cp.zeros(1, dtype=dtype)
        self.n_k = op.n_k
        self.n_D = 0 if op.mix is None else op.mix.shape[0]
        self.mix = (
            dummy if op.mix is None else f64(op.mix.T)
        )  # transposed: coalesced reads
        self.n_P = 0 if op.T_P is None else op.T_P.shape[0]
        self.T_P = dummy if op.T_P is None else f64(op.T_P.T)
        n_corner = 0 if sweep.corner is None else len(sweep.corner)
        slot = np.full(N, -1)
        if n_corner:
            slot[sweep.corner] = np.arange(n_corner)
        self.slot = i32(slot)
        self.levels = sweep.levels
        self.N = N
        real = "float" if dtype == np.float32 else "double"
        module = cp.RawModule(code=_SOURCE, options=(f"-DREAL={real}",))
        self.gather = module.get_function("imex_gather")
        self.finalize = module.get_function("imex_finalize")

        # the graph reads the step's data from these fixed buffers
        self.par = cp.zeros(2, dtype=dtype)
        self.den = cp.zeros((N, op.n_k), dtype=dtype)
        self.w = cp.zeros((N, op.n_k), dtype=dtype)
        self.kinv = (
            cp.zeros((n_corner, self.n_P, self.n_P), dtype=dtype) if n_corner else dummy
        )
        self._graphs = {}

    def _buffers(self, K):
        """Input, output and scratch arrays and the captured sweep for ``K`` columns."""
        if K not in self._graphs:
            cp = self.cp
            shape = (self.N, self.n_k, K)
            buf = {
                "b": cp.zeros(shape, dtype=self.dtype),
                "x": cp.zeros(shape, dtype=self.dtype),
                "xs": cp.zeros(shape, dtype=self.dtype),
                "part": cp.zeros(
                    (max(self.n_chunks, 1), self.n_k, K), dtype=self.dtype
                ),
                "stream": cp.cuda.Stream(non_blocking=True),
            }
            kt = max(1, min(K, 128 // self.n_k))  # columns per block
            tx = 32 * -(-(self.n_k * kt) // 32)
            lanes = max(1, min(8, 1024 // tx))
            item = self.dtype.itemsize
            buf["tile"] = (kt, -(-K // kt), (tx, lanes))
            buf["shared"] = (
                tx * lanes * item,
                (tx * lanes + (3 * self.n_k + self.n_P) * kt) * item,
            )
            with buf["stream"]:
                buf["stream"].begin_capture()
                self._launch(buf, K)
                buf["graph"] = buf["stream"].end_capture()
            self._graphs[K] = buf
        return self._graphs[K]

    def _launch(self, buf, K):
        """Enqueue the launches of one sweep."""
        kt, n_tiles, block = buf["tile"]
        nk, nP, nD, k32, kt32 = (
            np.int32(v) for v in (self.n_k, self.n_P, self.n_D, K, kt)
        )
        for (a, e), (c0, c1) in zip(self.levels, self.level_chunks):
            if c1 > c0:
                self.gather(
                    (c1 - c0, n_tiles),
                    block,
                    (
                        np.int32(c0),
                        nk,
                        k32,
                        kt32,
                        *self.chunks,
                        *self.entries,
                        self.par,
                        buf["xs"],
                        buf["part"],
                    ),
                    shared_mem=buf["shared"][0],
                )
            self.finalize(
                (e - a, n_tiles),
                block,
                (
                    np.int32(a),
                    nk,
                    nP,
                    nD,
                    k32,
                    kt32,
                    self.rc,
                    self.bs,
                    buf["part"],
                    self.mix,
                    self.T_P,
                    self.kinv,
                    self.slot,
                    buf["b"],
                    self.den,
                    self.w,
                    self.par,
                    buf["x"],
                    buf["xs"],
                ),
                shared_mem=buf["shared"][1],
            )

    def set_step(self, prepared):
        """Load the ``prepare`` tuple ``(c, ri, den, w, kinv)`` of a step into the graph's buffers."""
        cp = self.cp
        c, ri, den, w, kinv = prepared
        self.par[0], self.par[1] = c, ri
        cp.copyto(self.den, den)
        cp.copyto(self.w, w)
        if kinv is not None:
            cp.copyto(self.kinv, kinv.transpose(0, 2, 1))

    def __call__(self, b):
        """Solve for ``b`` ``(N, n_k, K)`` in level order with the step set by :meth:`set_step`."""
        buf = self._buffers(b.shape[2])
        self.cp.copyto(buf["b"], b)
        self.cp.cuda.get_current_stream().synchronize()
        buf["graph"].launch(stream=buf["stream"])
        buf["stream"].synchronize()
        return buf["x"].astype(b.dtype)
