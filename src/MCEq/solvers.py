import numpy as np
import scipy.sparse as sp

from MCEq import config
from MCEq.misc import info

#: Documented contract for the EM-row blowup at extreme zenith. Referenced
#: from each ETD2 kernel; see ``docs/mceq_v1.x_v2_diff.md`` "EM cascade
#: caveat" for the full derivation. Until a validated EM database ships,
#: ``config.adv_set["disabled_particles"]`` defaults to ``[11, -11]`` so
#: this branch is never entered for production runs.
_EM_BLOWUP_CAVEAT = """\
At extreme zenith the e± semi-Lagrangian L/R rows produce ``inf`` in
``F_phi`` / ``F_a`` (no diagonal damping). The blowup is contained to
those rows: e±/γ do not feed back into hadrons via ``int_m`` / ``dec_m``,
so muons and neutrinos are unaffected. Each ETD2 kernel wraps its loop
with ``np.errstate(over='ignore', invalid='ignore')`` to suppress the
resulting overflow / NaN warnings. To exclude the EM block entirely, set
``config.adv_set['disabled_particles'] = [11, -11]`` (the default).
"""


def etd2_nonuniform_path(
    density_model,
    *,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
    int_grid=None,
):
    """Build a non-uniform integration path for ETD2 kernels.

    Step sizes follow ``h_k = min(dX_max, eps / |d ln rho_inv / dX|(X_k))``
    with a ``max(.., dX_min)`` floor. ``rho_inv`` for the kernel is the
    integral mean of ``density_model.r_X2rho`` over each step (via
    ``scipy.integrate.quad``), not a point sample — this is essential at
    the very first step which crosses the spline-saturation cap.

    See ``docs/mceq_v1.x_v2_diff.md`` ("Step-size control and the resonance
    approximation") for the design.

    Args:
      density_model: object with ``r_X2rho(X)`` and ``max_X``.
      X_start (float | None): starting depth in g/cm^2; ``None`` →
        ``config.X_start``.
      eps (float | None): within-step ``rho_inv`` variation tolerance;
        ``None`` → ``config.etd2_path["eps"]``.
      dX_max (float | None): cap on step size (off-diagonal stability
        cliff); ``None`` → ``config.etd2_path["dX_max"]``.
      dX_min (float | None): floor on step size; ``None`` →
        ``config.etd2_path["dX_min"]``.
      fd_span (float | None): forward-FD probe span; ``None`` →
        ``config.etd2_path["fd_span"]``.
      int_grid (np.ndarray | None): X values at which to record snapshots.
        Steps are truncated to land exactly on each ``int_grid`` entry.

    Returns:
      (nsteps, dX, rho_inv, grid_idcs): tuple compatible with the
      kernel-dispatch contract used by ``MCEqRun.integration_path``.
    """
    if X_start is None:
        X_start = config.X_start
    p = config.etd2_path
    if eps is None:
        eps = p["eps"]
    if dX_max is None:
        dX_max = p["dX_max"]
    if dX_min is None:
        dX_min = p["dX_min"]
    if fd_span is None:
        fd_span = p["fd_span"]

    ri = density_model.r_X2rho
    max_X = density_model.max_X
    n_int = int(np.size(int_grid)) if int_grid is not None else 0

    if n_int and float(np.min(int_grid)) < float(X_start):
        raise ValueError(
            "Steps in int_grid must be larger than or equal to X_start "
            f"(got min(int_grid)={float(np.min(int_grid)):.6g}, "
            f"X_start={float(X_start):.6g})."
        )

    Xs, dXs, grid_idcs = [], [], []
    grid_step = 0
    X = float(X_start)
    step = 0
    while X < max_X:
        rate = abs(np.log(float(ri(X + fd_span))) - np.log(float(ri(X)))) / fd_span
        h = min(dX_max, eps / rate) if rate > 0 else dX_max
        h = max(h, dX_min)
        h = min(h, max_X - X)
        # Truncate the step to land exactly on the next user-requested
        # snapshot point. The truncation can drive h below dX_min — that's
        # the user's intent, not a stability issue (smaller h is always
        # more stable for ETD2). It can chain across many short steps if
        # the user's grid is finer than the natural schedule.
        if n_int and grid_step < n_int and X + h >= int_grid[grid_step]:
            h = float(int_grid[grid_step]) - X
            grid_idcs.append(step)
            grid_step += 1
        Xs.append(X)
        dXs.append(h)
        X += h
        step += 1

    Xs = np.asarray(Xs, dtype=np.float64)
    dXs = np.asarray(dXs, dtype=np.float64)
    rho_inv = np.empty(len(dXs), dtype=np.float64)

    # Compute per-step integral means of ``r_X2rho`` via differences on
    # a cumulative-trapezoid antiderivative built once on a hybrid
    # log+linear sample.
    #
    # Why a hybrid sample (and not a uniform one): ``r_X2rho`` is
    # ``1/rho``, and atmosphere density splines deliberately saturate at
    # the top of atmosphere so that the path-builder's FD probe stays
    # well-defined. The saturation produces a step-function-like spike
    # near ``X = 0``: ``ri(0) ~ 1e9 cm^3/g`` falls to ``~1e7`` by
    # ``X = 0.01``. A uniform-grid quadrature smears that spike across
    # samples and over-estimates the mean by ~2-20x for the first few
    # steps. Sampling logarithmically near ``X = 0`` (concentrated where
    # ``ri`` varies fast) and linearly in the bulk recovers
    # ``quad(epsrel=1e-6)``-class accuracy at ``O(1)`` per step.
    #
    # Why a spline of the *cumulative* and not of ``r_X2rho`` directly:
    # the cumulative is smooth and strictly monotone, so a cubic spline
    # through it is well-behaved even though ``r_X2rho`` spans 5+ decades.
    # A direct fit overshoots and yields non-physical (negative) means
    # near the top of atmosphere.
    #
    # Falls back to ``quad`` only if ``ri`` rejects array input.
    use_cum = False
    cum_spline = None
    if len(dXs) > 0:
        try:
            from scipy.integrate import cumulative_trapezoid
            from scipy.interpolate import UnivariateSpline

            X_min = float(Xs[0])
            X_max = float(Xs[-1] + dXs[-1])
            # Cap at the model's max depth (numerical drift on the last step).
            X_max = min(X_max, float(max_X))
            if X_max > X_min:
                # Concentrated log-sample for the saturated top, dense
                # linear sample for the bulk. ~8e3 total points is enough
                # to hit ~1e-6 rel error against quad on SIBYLL21 paths.
                X_log_lo = max(1e-7, X_min if X_min > 0 else 1e-7)
                X_log_hi = min(1.0, X_max)
                if X_log_hi > X_log_lo:
                    X_top = np.geomspace(X_log_lo, X_log_hi, 4001)
                else:
                    X_top = np.empty(0)
                X_bulk = np.linspace(max(X_log_hi, X_min), X_max, 4001)
                sample_X = np.unique(np.r_[X_min, X_top, X_bulk])
                sample_X.sort()
                sample_ri = np.asarray(ri(sample_X), dtype=np.float64)
                if (
                    sample_ri.shape == sample_X.shape
                    and np.all(np.isfinite(sample_ri))
                    and np.all(sample_ri > 0.0)
                ):
                    cum = cumulative_trapezoid(sample_ri, sample_X, initial=0.0)
                    cum_spline = UnivariateSpline(sample_X, cum, k=3, s=0.0)
                    use_cum = True
        except Exception:
            use_cum = False

    if not use_cum:
        from scipy.integrate import quad

    for i in range(len(dXs)):
        # A zero-length step occurs when ``int_grid`` requests a snapshot
        # at X_start: the truncation drives the first dX to 0 and the kernel
        # records the initial state. Point-sample ri there to avoid /0.
        if dXs[i] == 0.0:
            rho_inv[i] = float(ri(Xs[i]))
        elif use_cum:
            a = Xs[i]
            b = a + dXs[i]
            rho_inv[i] = float(cum_spline(b) - cum_spline(a)) / dXs[i]
        else:
            rho_inv[i] = (
                quad(ri, Xs[i], Xs[i] + dXs[i], limit=50, epsrel=1e-6)[0] / dXs[i]
            )
    return len(dXs), dXs, rho_inv, grid_idcs


def _etd_split_cache(int_m, dec_m):
    """Pre-compute the diagonal/off-diagonal split used by ETD kernels.

    Returns (d_int, d_dec, int_off, dec_off) where d_* are 1-D arrays holding
    the diagonals and *_off are sparse matrices with their diagonals zeroed
    out. Both pieces are constant in X — only `rho_inv` modulates how they
    combine per step — so we cache them across the integration loop.
    """
    d_int = int_m.diagonal()
    d_dec = dec_m.diagonal()
    int_off = int_m - sp.diags(d_int, format=int_m.format)
    dec_off = dec_m - sp.diags(d_dec, format=dec_m.format)
    int_off.eliminate_zeros()
    dec_off.eliminate_zeros()
    return d_int, d_dec, int_off, dec_off


def _etd_off_to_bsr(off_csr, blocksize):
    """Convert a CSR off-diagonal to BSR(``blocksize``), padding if needed.

    Returns ``(off_bsr, n_padded)``. Pads to the next multiple of
    ``blocksize`` by extending ``indptr`` with copies of its tail (zero new
    entries); the trailing rows / cols are then zero. Empty input
    (``nnz == 0``) is returned unchanged with the original dimension.
    """
    n_orig = off_csr.shape[0]
    if off_csr.nnz == 0:
        return off_csr, n_orig
    pad = (-n_orig) % blocksize
    if pad:
        indptr = np.concatenate(
            [
                off_csr.indptr,
                np.full(pad, off_csr.indptr[-1], dtype=off_csr.indptr.dtype),
            ]
        )
        off_csr = sp.csr_matrix(
            (off_csr.data, off_csr.indices, indptr),
            shape=(n_orig + pad, n_orig + pad),
        )
    return off_csr.tobsr(blocksize=(blocksize, blocksize)), n_orig + pad


def _etd_get_split_for_numpy(int_m, dec_m, blocksize):
    """Return (d_int, d_dec, int_off, dec_off, n_padded) for the numpy
    ETD2 kernel, with a per-matrix-instance memoised BSR conversion.

    The cache lives as a private attribute on ``int_m``. It auto-clears
    when the matrix is garbage-collected (cache attribute goes with it),
    and gets invalidated on a ``dec_m`` identity / blocksize change.

    Set ``blocksize=None`` for plain CSR (matches scipy's default
    behaviour, slowest of the available paths but safest).
    """
    cache = getattr(int_m, "_etd_split_cache_v2", None)
    if (
        cache is not None
        and cache["dec_m_id"] == id(dec_m)
        and cache["blocksize"] == blocksize
    ):
        c = cache
        return c["d_int"], c["d_dec"], c["int_off"], c["dec_off"], c["n_padded"]

    d_int, d_dec, int_off, dec_off = _etd_split_cache(int_m, dec_m)
    if blocksize is None:
        n_padded = int_m.shape[0]
    else:
        if not sp.isspmatrix_csr(int_off):
            int_off = int_off.tocsr()
        if not sp.isspmatrix_csr(dec_off):
            dec_off = dec_off.tocsr()
        int_off, n_padded = _etd_off_to_bsr(int_off, blocksize)
        dec_off, n_padded_d = _etd_off_to_bsr(dec_off, blocksize)
        # Both off-diags share the original int_m / dec_m dim, so n_padded
        # values must agree (or one of them was empty and kept its original).
        n_padded = max(n_padded, n_padded_d)

    try:
        int_m._etd_split_cache_v2 = {
            "dec_m_id": id(dec_m),
            "blocksize": blocksize,
            "d_int": d_int,
            "d_dec": d_dec,
            "int_off": int_off,
            "dec_off": dec_off,
            "n_padded": n_padded,
        }
    except (AttributeError, TypeError):
        # scipy slot-only matrix variant — fall through without caching.
        pass
    return d_int, d_dec, int_off, dec_off, n_padded


# phi1(z) = (e^z - 1) / z              (limit 1   as z -> 0)
# phi2(z) = (e^z - 1 - z) / z**2       (limit 1/2 as z -> 0)
# Below the analytic-formula cutoffs we patch with the order-2 Taylor
# expansion via Horner. phi2 cancels at a wider radius than phi1 because
# its numerator has a leading z² term.
_PHI1_SMALL = 1e-6
_PHI2_SMALL = 1e-3
_INV_6 = 1.0 / 6.0
_INV_24 = 1.0 / 24.0


def _etd_step_buffers(dim):
    """Allocate the per-step scratch arrays the ETD kernels need.

    Centralized here so both the numpy and the spacc kernels share an
    identical layout — they're hot loops, and any allocation inside them
    dominates the SpMVs once those are running on Accelerate / a tuned BLAS.
    """
    return {
        "D": np.empty(dim, dtype=np.float64),
        "hD": np.empty(dim, dtype=np.float64),
        "eD": np.empty(dim, dtype=np.float64),
        "phi1": np.empty(dim, dtype=np.float64),
        "phi2": np.empty(dim, dtype=np.float64),
        "scratch": np.empty(dim, dtype=np.float64),
        "abs_hD": np.empty(dim, dtype=np.float64),
        "mask1": np.empty(dim, dtype=bool),
        "mask2": np.empty(dim, dtype=bool),
    }


def _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs):
    """Fill ``bufs['eD']`` / ``bufs['phi1']`` / ``bufs['phi2']`` in place.

    Computes the per-step diagonal of ``A + ri * B``, exponentiates it, and
    evaluates the two phi-functions of ``h*D``. All work is done in
    preallocated buffers — no temporaries — and the small-|hD| Taylor
    branch is patched in only on the rows that need it (instead of being
    evaluated eagerly across the whole array as ``np.where`` would).
    """
    D = bufs["D"]
    hD = bufs["hD"]
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]
    abs_hD = bufs["abs_hD"]
    mask1 = bufs["mask1"]
    mask2 = bufs["mask2"]

    # D = d_int + ri * d_dec
    np.multiply(d_dec, ri, out=D)
    np.add(D, d_int, out=D)
    # hD = h * D ; eD = exp(hD)
    np.multiply(D, h, out=hD)
    np.exp(hD, out=eD)

    # Branch masks: True ⇒ analytic form is safe.
    np.abs(hD, out=abs_hD)
    np.greater(abs_hD, _PHI1_SMALL, out=mask1)
    np.greater(abs_hD, _PHI2_SMALL, out=mask2)

    # phi1: analytic (eD - 1) / hD where mask1, Taylor 1 + hD/2 + hD²/6 elsewhere.
    np.subtract(eD, 1.0, out=phi1)
    np.divide(phi1, hD, out=phi1, where=mask1)
    # Horner Taylor for phi1: ((hD/6) + 1/2)*hD + 1
    np.multiply(hD, _INV_6, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 1.0, out=scratch)
    np.invert(mask1, out=mask1)  # mask1 now: small |hD| rows
    np.copyto(phi1, scratch, where=mask1)

    # phi2: analytic (eD - 1 - hD) / hD² where mask2, Taylor 1/2 + hD/6 + hD²/24 elsewhere.
    np.subtract(eD, 1.0, out=phi2)
    np.subtract(phi2, hD, out=phi2)
    np.multiply(hD, hD, out=scratch)  # hD² in scratch
    np.divide(phi2, scratch, out=phi2, where=mask2)
    # Horner Taylor for phi2: ((hD/24) + 1/6)*hD + 1/2
    np.multiply(hD, _INV_24, out=scratch)
    np.add(scratch, _INV_6, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.invert(mask2, out=mask2)  # mask2 now: small |hD| rows
    np.copyto(phi2, scratch, where=mask2)


def solv_numpy_etd2(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    """ETD2RK (Cox-Matthews exponential Runge-Kutta, single-stage, 2nd order).

    Solves dPhi/dX = (A + rho_inv(X) * B) Phi by treating the diagonal part
    of (A + rho_inv * B) exactly via an integrating factor and the off-
    diagonal part with a two-stage explicit RK2 in exponential form. The
    diagonal carries all of MCEq's stiffness, so the explicit-stability
    constraint that bounds forward-Euler step size does not apply; the
    remaining limit is the explicit-RK stability of the off-diagonal block.

    Update (D and N frozen at start of step):
        F(state) = A_off @ state + rho_inv * B_off @ state
        a   = exp(h*D) * Phi + h * phi1(h*D) * F(Phi)
        Phi <- a + h * phi2(h*D) * ( F(a) - F(Phi) )
    where
        phi1(z) = (e^z - 1) / z              (limit 1   as z -> 0)
        phi2(z) = (e^z - 1 - z) / z**2       (limit 1/2 as z -> 0)
    are evaluated elementwise on the diagonal vector h*D, with stable
    Taylor series near 0.

    Per-step cost: 4 SpMVs (two F evaluations against int_off and dec_off
    each) plus a handful of elementwise vector ops on length-N arrays.
    Globally O(h**2).

    Off-diagonals are converted to BSR with ``config.numpy_bsr_blocksize``
    (default ``11``) for ~2x faster scipy SpMV; the converted matrices are
    memoised on ``int_m`` so repeated ``solve()`` calls amortise the
    conversion. Set ``config.numpy_bsr_blocksize = None`` to fall back to
    plain CSR.

    Args:
      nsteps (int): number of integration steps
      dX (np.ndarray[nsteps]): step sizes Delta X_i in g/cm**2
      rho_inv (np.ndarray[nsteps]): 1/rho(X_i)
      int_m (scipy.sparse): interaction matrix A
      dec_m (scipy.sparse): decay matrix B
      phi (np.ndarray): initial state Phi(X_0)
      grid_idcs (list[int]): step indices at which to record snapshots

    Returns:
      (np.ndarray, np.ndarray): final state and stacked snapshots.
    """
    blocksize = getattr(config, "numpy_bsr_blocksize", None)
    d_int, d_dec, int_off, dec_off, n_padded = _etd_get_split_for_numpy(
        int_m, dec_m, blocksize
    )

    dim = phi.shape[0]
    # Buffers at n_padded so scipy BSR `.dot()` returns into them directly.
    # Padding slots stay zero throughout (matrix has zero rows/cols there).
    phc = np.zeros(n_padded, dtype=np.float64)
    phc[:dim] = phi
    F_phi = np.empty(n_padded, dtype=np.float64)
    F_a = np.empty(n_padded, dtype=np.float64)
    a = np.empty(n_padded, dtype=np.float64)
    bufs = _etd_step_buffers(dim)
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]

    # Live views into the unpadded prefix; per-step elementwise math
    # touches only these, leaving the padding slots at their initial 0.
    phc_v = phc[:dim]
    F_phi_v = F_phi[:dim]
    F_a_v = F_a[:dim]
    a_v = a[:dim]

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    # See module-level :data:`_EM_BLOWUP_CAVEAT`: e± semi-Lagrangian rows
    # can blow up at extreme zenith, and the kernel suppresses the
    # downstream overflow/NaN warnings. The MKL/spacc/CUDA kernels share
    # this contract.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = dX[k]
            ri = rho_inv[k]

            _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs)

            # F_phi = int_off @ phc + ri * dec_off @ phc
            # scipy SpMV allocates the result internally; copy into preallocated F_phi.
            np.copyto(F_phi, int_off.dot(phc))
            ri_dec = dec_off.dot(phc)
            ri_dec *= ri
            np.add(F_phi, ri_dec, out=F_phi)

            # a = eD * phc + h * phi1 * F_phi  (unpadded slice)
            np.multiply(eD, phc_v, out=a_v)
            np.multiply(phi1, F_phi_v, out=scratch)
            scratch *= h
            np.add(a_v, scratch, out=a_v)

            # F_a = int_off @ a + ri * dec_off @ a
            np.copyto(F_a, int_off.dot(a))
            ri_dec = dec_off.dot(a)
            ri_dec *= ri
            np.add(F_a, ri_dec, out=F_a)

            # phc = a + h * phi2 * (F_a - F_phi)
            np.subtract(F_a_v, F_phi_v, out=scratch)
            scratch *= h
            np.multiply(scratch, phi2, out=scratch)
            np.add(a_v, scratch, out=phc_v)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc_v))
                grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc_v.copy(), np.array(grid_sol)


# ---------------------------------------------------------------------------
# MKL ETD2 kernel
# ---------------------------------------------------------------------------
class MklSparseMatrix:
    """Thin RAII wrapper around an Intel MKL sparse-matrix handle.

    Holds either a CSR or BSR view of the same off-diagonal block. MKL keeps
    raw pointers into the backing arrays, so the Python objects must outlive
    the handle — we keep references on the wrapper.
    ``mkl_sparse_set_mv_hint`` + ``mkl_sparse_optimize`` are called once at
    construction so the per-solve loop reuses the optimised layout.

    For BSR mode, MKL requires the matrix dimension to be a multiple of
    ``blocksize``. The wrapper pads rows/cols with zeros to satisfy that;
    ``n_padded`` reports the padded dimension and ``n_orig`` the original
    one. The padding rows/cols are zero, so SpMV against a length-``n_padded``
    vector with zeros in the trailing slots is equivalent to the unpadded
    SpMV. Callers (``solv_mkl_etd2``) allocate padded scratch buffers once
    and slice ``[:n_orig]`` for the per-step elementwise math.

    Why ``blocksize=6`` is the default: empirically the fastest block size
    for the SIBYLL21 off-diagonals on Intel MKL — ~1.5× faster than CSR.
    MKL appears to have a hand-tuned BSR microkernel for ``b ∈ [2, 7]``;
    ``b ≥ 8`` falls into a generic path and is slower than CSR for these
    matrices. See ``docs/mceq_v1.x_v2_diff.md`` §8.4.

    Args:
      csr (scipy.sparse.csr_matrix): float64 CSR matrix with int32 indices.
      expected_calls (int): SpMV count hint for MKL planning.
      blocksize (int | None): If ``None``, store as CSR. If int, store as
        BSR with that block size (auto-padding the matrix as needed).
    """

    def __init__(self, csr, expected_calls=200, blocksize=None):
        from ctypes import POINTER, Structure, byref, c_int, c_void_p
        from ctypes import c_double as fl_pr

        if config.mkl is None:
            raise RuntimeError(
                "MklSparseMatrix: MKL library is not loaded. "
                "Call config.set_mkl_threads(...) first."
            )
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrix expects a CSR matrix, got {type(csr).__name__}"
            )
        if csr.dtype != np.float64:
            raise TypeError(f"MklSparseMatrix expects float64 data, got {csr.dtype}")

        n_orig = csr.shape[0]
        self.n_orig = n_orig
        self.blocksize = blocksize

        mkl = config.mkl
        self._mkl = mkl

        if blocksize is None:
            # ----- CSR path -----
            indices = csr.indices.astype(np.int32, copy=False)
            indptr = csr.indptr.astype(np.int32, copy=False)
            data = csr.data
            self._data = data
            self._indices = indices
            self._indptr = indptr
            self.nnz = csr.nnz
            self.n_padded = n_orig

            handle = c_void_p()
            data_p = data.ctypes.data_as(POINTER(fl_pr))
            ci_p = indices.ctypes.data_as(POINTER(c_int))
            pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
            pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))

            st = mkl.mkl_sparse_d_create_csr(
                byref(handle),
                c_int(0),
                c_int(n_orig),
                c_int(n_orig),
                pb_p,
                pe_p,
                ci_p,
                data_p,
            )
            if st != 0:
                raise RuntimeError(f"mkl_sparse_d_create_csr failed with status {st}")
        else:
            # ----- BSR path -----
            if not isinstance(blocksize, int) or blocksize < 2:
                raise ValueError(f"blocksize must be int >= 2, got {blocksize!r}")
            pad = (-n_orig) % blocksize
            if pad > 0:
                # Append `pad` zero rows / cols at the end. CSR-pad: extend
                # indptr with copies of the last value (no new entries).
                indptr_padded = np.concatenate(
                    [csr.indptr, np.full(pad, csr.indptr[-1], dtype=csr.indptr.dtype)]
                )
                csr = sp.csr_matrix(
                    (csr.data, csr.indices, indptr_padded),
                    shape=(n_orig + pad, n_orig + pad),
                ).tocsr()
            n_padded = csr.shape[0]
            self.n_padded = n_padded

            B = csr.tobsr(blocksize=(blocksize, blocksize))
            data = np.ascontiguousarray(B.data, dtype=np.float64)
            indices = B.indices.astype(np.int32, copy=False)
            indptr = B.indptr.astype(np.int32, copy=False)
            self._data = data
            self._indices = indices
            self._indptr = indptr
            # BSR `nnz` is the total number of stored entries, not just
            # explicit nonzeros (each block stores blocksize**2 entries).
            self.nnz = data.size

            handle = c_void_p()
            n_blocks = c_int(n_padded // blocksize)
            data_p = data.ctypes.data_as(POINTER(fl_pr))
            ci_p = indices.ctypes.data_as(POINTER(c_int))
            pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
            pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))
            # SPARSE_LAYOUT_ROW_MAJOR = 101 — scipy stores BSR blocks
            # row-major within each block.
            st = mkl.mkl_sparse_d_create_bsr(
                byref(handle),
                c_int(0),
                c_int(101),
                n_blocks,
                n_blocks,
                c_int(blocksize),
                pb_p,
                pe_p,
                ci_p,
                data_p,
            )
            if st != 0:
                raise RuntimeError(f"mkl_sparse_d_create_bsr failed with status {st}")
        self._handle = handle

        class _MatrixDescr(Structure):
            _fields_ = [
                ("type", c_int),
                ("mode", c_int),
                ("diag", c_int),
            ]

        descr = _MatrixDescr()
        descr.type = c_int(20)  # SPARSE_MATRIX_TYPE_GENERAL
        descr.mode = c_int(121)
        descr.diag = c_int(131)
        self._descr = descr
        self._operation = c_int(10)  # SPARSE_OPERATION_NON_TRANSPOSE

        st = mkl.mkl_sparse_set_mv_hint(
            handle, self._operation, descr, c_int(int(expected_calls))
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mv_hint failed with status {st}")
        st = mkl.mkl_sparse_optimize(handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def gemv_ctargs(self, alpha, x_p, beta, y_p):
        """``y = alpha * A * x + beta * y`` via raw c_double pointers.

        Mirrors :meth:`MCEq.spacc.SpaccMatrix.gemv_ctargs` so the ETD2
        kernels can be written backend-agnostic up to the gemv binding.
        """
        from ctypes import c_double as fl_pr

        st = self._mkl.mkl_sparse_d_mv(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            x_p,
            fl_pr(beta),
            y_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_d_mv failed with status {st}")

    def close(self):
        """Free the underlying MKL sparse handle.

        Idempotent — safe to call repeatedly. Prefer this over
        ``del`` or relying on refcount-driven ``__del__`` when caches
        in ``MCEqRun._build_kernel_dispatch`` are rotated; the call
        below the C boundary returns the MKL-internal optimised layout
        memory, not just the Python wrapper.
        """
        handle = getattr(self, "_handle", None)
        mkl = getattr(self, "_mkl", None)
        if handle is None or mkl is None:
            return
        try:
            mkl.mkl_sparse_destroy(handle)
        except Exception:
            pass
        self._handle = None

    def __del__(self):
        # Defer to ``close()``; both are idempotent. Guards in ``close()``
        # cover partially-initialised instances (constructor raised before
        # the handle was created).
        try:
            self.close()
        except Exception:
            pass


def solv_mkl_etd2(
    nsteps,
    dX,
    rho_inv,
    mkl_int_off,
    mkl_dec_off,
    d_int,
    d_dec,
    phi,
    grid_idcs,
):
    """ETD2RK on Intel MKL sparse BLAS.

    Pre-split kernel: takes the off-diagonal matrices already wrapped as
    :class:`MklSparseMatrix` instances (CSR or BSR backed; see the
    wrapper docstring for the BSR padding contract) and the diagonal
    vectors as plain numpy arrays. The diagonal/off-diagonal split is
    constant in X so the caller (``MCEqRun.solve``) builds it once per
    ``solve()`` call.

    Per step (mirrors :func:`solv_numpy_etd2`):

      F_phi = int_off @ phc + ri * dec_off @ phc           (2 SpMVs)
      a     = exp(h*D) * phc + h * phi1(h*D) * F_phi
      F_a   = int_off @ a   + ri * dec_off @ a             (2 SpMVs)
      phc   = a + h * phi2(h*D) * (F_a - F_phi)

    Implementation notes:

    * ``MklSparseMatrix.gemv_ctargs`` calls ``mkl_sparse_d_mv`` via raw
      ctypes pointers, so the buffers backing those pointers must keep
      the same address across the whole loop. We pre-allocate ``phc`` /
      ``F_phi`` / ``F_a`` / ``a`` once and update them in place — never
      rebind those names inside the loop.
    * For BSR-backed handles the working buffers are sized to
      ``n_padded`` rather than ``dim``; the trailing padding slots stay
      zero throughout (the matrix has zero rows/cols there, so SpMV
      preserves that). Per-step elementwise math operates on the
      ``[:dim]`` slice — eD / phi1 / phi2 remain length ``dim``.

    Args:
      nsteps (int): number of integration steps
      dX (np.ndarray[nsteps]): step sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (np.ndarray[nsteps]): :math:`\\rho^{-1}(X_i)` per step
      mkl_int_off (MklSparseMatrix | None): off-diagonal of A = int_m,
        ``None`` if it has zero nnz (kernel skips the SpMV).
      mkl_dec_off (MklSparseMatrix | None): off-diagonal of B = dec_m,
        ``None`` if empty.
      d_int (np.ndarray): diagonal of A (length ``dim``)
      d_dec (np.ndarray): diagonal of B (length ``dim``)
      phi (np.ndarray): initial state :math:`\\Phi(X_0)` (length ``dim``)
      grid_idcs (list[int]): step indices at which to record snapshots

    Returns:
      (np.ndarray, np.ndarray): final state and stacked snapshots, both
      sliced back to length ``dim``.
    """
    from ctypes import POINTER, c_double

    dim = phi.shape[0]
    # If either wrapper used BSR padding, allocate buffers at the padded
    # length. Both wrappers come from the same matrix dim so n_padded must
    # agree; defaulting to ``dim`` covers the all-CSR / both-empty case.
    n_padded = dim
    for m in (mkl_int_off, mkl_dec_off):
        if m is not None:
            n_padded = max(n_padded, m.n_padded)

    # Persistent buffers — ctypes pointers must remain valid across the loop,
    # so every per-step update writes into these in place (never rebinds).
    # Padding slots stay zero throughout: the matrix has zero rows/cols
    # there, so SpMV-against-zero-suffix preserves that invariant.
    phc = np.zeros(n_padded, dtype=np.float64)
    phc[:dim] = phi
    F_phi = np.zeros(n_padded, dtype=np.float64)
    F_a = np.zeros(n_padded, dtype=np.float64)
    a = np.zeros(n_padded, dtype=np.float64)
    bufs = _etd_step_buffers(dim)
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]

    # Live views into the unpadded prefix; the elementwise math touches
    # only these, leaving the padding slots untouched.
    phc_v = phc[:dim]
    F_phi_v = F_phi[:dim]
    F_a_v = F_a[:dim]
    a_v = a[:dim]

    phc_p = phc.ctypes.data_as(POINTER(c_double))
    F_phi_p = F_phi.ctypes.data_as(POINTER(c_double))
    F_a_p = F_a.ctypes.data_as(POINTER(c_double))
    a_p = a.ctypes.data_as(POINTER(c_double))

    int_off_empty = mkl_int_off is None or mkl_int_off.nnz == 0
    dec_off_empty = mkl_dec_off is None or mkl_dec_off.nnz == 0

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    # See module-level :data:`_EM_BLOWUP_CAVEAT`.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = dX[k]
            ri = rho_inv[k]

            _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs)

            # F_phi = int_off @ phc + ri * dec_off @ phc
            # gemv: y = alpha * A * x + beta * y. beta=0 zeros y, beta=1 accumulates.
            if not int_off_empty:
                mkl_int_off.gemv_ctargs(1.0, phc_p, 0.0, F_phi_p)
            else:
                F_phi_v.fill(0.0)
            if not dec_off_empty:
                mkl_dec_off.gemv_ctargs(ri, phc_p, 1.0, F_phi_p)

            # a = eD * phc + h * phi1 * F_phi  (unpadded slice)
            np.multiply(eD, phc_v, out=a_v)
            np.multiply(phi1, F_phi_v, out=scratch)
            scratch *= h
            np.add(a_v, scratch, out=a_v)

            # F_a = int_off @ a + ri * dec_off @ a
            if not int_off_empty:
                mkl_int_off.gemv_ctargs(1.0, a_p, 0.0, F_a_p)
            else:
                F_a_v.fill(0.0)
            if not dec_off_empty:
                mkl_dec_off.gemv_ctargs(ri, a_p, 1.0, F_a_p)

            # phc = a + h * phi2 * (F_a - F_phi)  (in-place into phc[:dim])
            np.subtract(F_a_v, F_phi_v, out=scratch)
            scratch *= h
            np.multiply(scratch, phi2, out=scratch)
            np.add(a_v, scratch, out=phc_v)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc_v))
                grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc_v.copy(), np.array(grid_sol)


# ---------------------------------------------------------------------------
# CUDA ETD2 kernel
# ---------------------------------------------------------------------------
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
        libdir = os.path.join(os.path.dirname(mod.__file__), "lib")
        candidate = os.path.join(libdir, libname)
        if not os.path.isfile(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Best effort — if the dlopen fails, cupy will still try its own
            # discovery and report a more specific error if it's actually
            # missing.
            pass


class CudaEtd2Context:
    """GPU-resident state for :func:`solv_cuda_etd2`.

    Owns the cuSPARSE CSR copies of ``int_off`` / ``dec_off``, the diagonal
    arrays ``d_int`` / ``d_dec``, and pre-allocated scratch buffers
    (``phc`` / ``F_phi`` / ``F_a`` / ``a`` / ``eD`` / ``phi1`` / ``phi2`` /
    ``scratch``). The caller (``MCEqRun``) creates one of these per
    interaction-matrix rebuild and reuses it across ``solve()`` calls;
    :func:`solv_cuda_etd2` then only has to upload the boundary state and
    pull the final state back at the end.

    Args:
      int_off (scipy.sparse): off-diagonal of A
      dec_off (scipy.sparse): off-diagonal of B
      d_int (np.ndarray): diagonal of A
      d_dec (np.ndarray): diagonal of B
      device_id (int): CUDA device index
      fp_precision (int): 32 (single) or 64 (double). 64 is the default;
        single is exposed for memory-bound use cases but ETD2's accuracy
        budget will typically prefer double.
    """

    def __init__(self, int_off, dec_off, d_int, d_dec, device_id, fp_precision):
        # CuPy 13.x does not auto-discover the nvidia-* pip packages that ship
        # the CUDA 12 runtime libs (libnvrtc.so.12, libcudart.so.12,
        # libcusparse.so.12, ...). When the system CUDA toolkit is a different
        # major (e.g. CUDA 13.0), those system libs don't satisfy the cupy
        # loader. The clean fix is to dlopen the pip-shipped libs before the
        # first kernel JIT — that's a no-op on systems where the libs are
        # already on the dynamic-loader path.
        _preload_nvidia_pip_libs()
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cusp
        except ImportError as e:
            raise RuntimeError(
                "CudaEtd2Context: CuPy is not available. Install a build of "
                "cupy matching your CUDA runtime."
            ) from e

        if fp_precision == 32:
            fl_pr = cp.float32
        elif fp_precision == 64:
            fl_pr = cp.float64
        else:
            raise ValueError(
                f"CudaEtd2Context: fp_precision must be 32 or 64, got {fp_precision}"
            )

        self.cp = cp
        self.fl_pr = fl_pr
        self.device_id = int(device_id)
        cp.cuda.Device(self.device_id).use()

        dim = int(d_int.shape[0])
        self.dim = dim

        # cuSPARSE CSR copies. None when the off-diagonal is empty — the
        # kernel skips those SpMVs (an empty CSR can be ill-defined for
        # cuSPARSE handles on some versions).
        self.cu_int_off = cusp.csr_matrix(int_off, dtype=fl_pr) if int_off.nnz else None
        self.cu_dec_off = cusp.csr_matrix(dec_off, dtype=fl_pr) if dec_off.nnz else None
        self.cu_d_int = cp.asarray(d_int, dtype=fl_pr)
        self.cu_d_dec = cp.asarray(d_dec, dtype=fl_pr)

        # Persistent scratch — same layout as the host buffers in
        # _etd_step_buffers; per-step writes go into these in place.
        self.cu_phc = cp.empty(dim, dtype=fl_pr)
        self.cu_F_phi = cp.empty(dim, dtype=fl_pr)
        self.cu_F_a = cp.empty(dim, dtype=fl_pr)
        self.cu_a = cp.empty(dim, dtype=fl_pr)
        self.cu_D = cp.empty(dim, dtype=fl_pr)
        self.cu_hD = cp.empty(dim, dtype=fl_pr)
        self.cu_eD = cp.empty(dim, dtype=fl_pr)
        self.cu_phi1 = cp.empty(dim, dtype=fl_pr)
        self.cu_phi2 = cp.empty(dim, dtype=fl_pr)
        self.cu_scratch = cp.empty(dim, dtype=fl_pr)


def _cuda_compute_diag_factors(ctx, h, ri):
    """GPU analogue of :func:`_etd_compute_diag_factors`.

    Fills ``ctx.cu_eD`` / ``ctx.cu_phi1`` / ``ctx.cu_phi2`` in place. Uses
    ``cp.where`` for the small-|hD| Taylor switch — the GPU does not gain
    from the masked-store optimization we use on host, since the warp
    executes both branches anyway.
    """
    cp = ctx.cp
    D = ctx.cu_D
    hD = ctx.cu_hD
    eD = ctx.cu_eD

    cp.multiply(ctx.cu_d_dec, ri, out=D)
    cp.add(D, ctx.cu_d_int, out=D)
    cp.multiply(D, h, out=hD)
    cp.exp(hD, out=eD)

    abs_hD = cp.abs(hD)
    # phi1
    safe_hD = cp.where(hD != 0, hD, 1.0)
    phi1_anal = (eD - 1.0) / safe_hD
    phi1_taylor = 1.0 + 0.5 * hD + (1.0 / 6.0) * hD * hD
    ctx.cu_phi1 = cp.where(abs_hD > _PHI1_SMALL, phi1_anal, phi1_taylor)
    # phi2
    safe_hD2 = cp.where(hD != 0, hD * hD, 1.0)
    phi2_anal = (eD - 1.0 - hD) / safe_hD2
    phi2_taylor = 0.5 + (1.0 / 6.0) * hD + (1.0 / 24.0) * hD * hD
    ctx.cu_phi2 = cp.where(abs_hD > _PHI2_SMALL, phi2_anal, phi2_taylor)


def solv_cuda_etd2(nsteps, dX, rho_inv, ctx, phi, grid_idcs):
    """ETD2RK on NVIDIA cuSPARSE via cupy.

    Same Cox–Matthews update as :func:`solv_numpy_etd2`, run end-to-end on
    the GPU. ``ctx`` holds the cuSPARSE CSR copies of ``int_off`` /
    ``dec_off`` and the diagonal arrays — they're constant in X, so the
    caller (``MCEqRun.solve``) materialises them once per matrix-rebuild
    and reuses them across ``solve()`` calls. The per-step path arrays
    ``dX`` / ``rho_inv`` stay on host: ``rho_inv[k]`` is read into a
    scalar each step, which is cheap and avoids syncing a GPU array.

    Args:
      nsteps (int): number of integration steps
      dX (np.ndarray[nsteps]): step sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (np.ndarray[nsteps]): :math:`\\rho^{-1}(X_i)` per step
      ctx (CudaEtd2Context): GPU state (matrices, diagonals, scratch)
      phi (np.ndarray): initial state :math:`\\Phi(X_0)` on host
      grid_idcs (list[int]): step indices at which to record snapshots

    Returns:
      (np.ndarray, np.ndarray): final state and stacked snapshots, both
      on host (downloaded from GPU before return).
    """
    cp = ctx.cp
    fl_pr = ctx.fl_pr
    cu_phc = ctx.cu_phc
    cu_F_phi = ctx.cu_F_phi
    cu_F_a = ctx.cu_F_a
    cu_a = ctx.cu_a
    cu_scratch = ctx.cu_scratch

    # Switch to the configured device for this kernel — multi-GPU users
    # might have other contexts active between solves.
    cp.cuda.Device(ctx.device_id).use()

    # Upload boundary state (in place: cu_phc is preallocated).
    cu_phc[:] = cp.asarray(phi, dtype=fl_pr)

    grid_sol_gpu = []
    grid_step = 0

    int_off_empty = ctx.cu_int_off is None
    dec_off_empty = ctx.cu_dec_off is None

    from time import time

    start = time()

    for k in range(nsteps):
        h = float(dX[k])
        ri = float(rho_inv[k])

        _cuda_compute_diag_factors(ctx, h, ri)
        eD = ctx.cu_eD
        phi1 = ctx.cu_phi1
        phi2 = ctx.cu_phi2

        # F_phi = int_off @ phc + ri * dec_off @ phc
        if not int_off_empty:
            cu_F_phi[:] = ctx.cu_int_off @ cu_phc
        else:
            cu_F_phi.fill(0)
        if not dec_off_empty:
            cu_F_phi += ri * (ctx.cu_dec_off @ cu_phc)

        # a = eD * phc + h * phi1 * F_phi
        cp.multiply(eD, cu_phc, out=cu_a)
        cp.multiply(phi1, cu_F_phi, out=cu_scratch)
        cu_scratch *= h
        cp.add(cu_a, cu_scratch, out=cu_a)

        # F_a = int_off @ a + ri * dec_off @ a
        if not int_off_empty:
            cu_F_a[:] = ctx.cu_int_off @ cu_a
        else:
            cu_F_a.fill(0)
        if not dec_off_empty:
            cu_F_a += ri * (ctx.cu_dec_off @ cu_a)

        # phc = a + h * phi2 * (F_a - F_phi)
        cp.subtract(cu_F_a, cu_F_phi, out=cu_scratch)
        cu_scratch *= h
        cp.multiply(cu_scratch, phi2, out=cu_scratch)
        cp.add(cu_a, cu_scratch, out=cu_phc)

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
            grid_sol_gpu.append(cu_phc.copy())
            grid_step += 1

    # Implicit sync via asnumpy — needed before timing to be honest, but the
    # cost dominates the loop's last few SpMVs anyway and is amortised over
    # nsteps in the per-iteration print.
    phc_host = cp.asnumpy(cu_phc).astype(np.float64, copy=False)
    if grid_sol_gpu:
        grid_arr = cp.asnumpy(cp.stack(grid_sol_gpu)).astype(np.float64, copy=False)
    else:
        grid_arr = np.array([])

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc_host, grid_arr


def solv_spacc_etd2(
    nsteps,
    dX,
    rho_inv,
    spacc_int_off,
    spacc_dec_off,
    d_int,
    d_dec,
    phi,
    grid_idcs,
):
    """ETD2RK on Apple Accelerate via the spacc bindings.

    Pre-split kernel: takes the off-diagonal matrices already wrapped as
    :class:`MCEq.spacc.SpaccMatrix` instances and the diagonal vectors as
    plain numpy arrays. The diagonal/off-diagonal split is constant in X
    so the caller (``MCEqRun.solve``) builds it once per ``solve()`` call.

    Per step (mirrors ``solv_numpy_etd2``):

      F_phi = int_off @ phc + ri * dec_off @ phc           (2 SpMVs)
      a     = exp(h*D) * phc + h * phi1(h*D) * F_phi
      F_a   = int_off @ a   + ri * dec_off @ a             (2 SpMVs)
      phc   = a + h * phi2(h*D) * (F_a - F_phi)

    Implementation note: ``SpaccMatrix.gemv_ctargs`` performs ``y = α·M·x + y``
    via raw ctypes pointers. The buffers backing those pointers must stay
    at the same address across the whole loop, so we pre-allocate ``phc``,
    ``F_phi``, ``F_a``, ``a`` once and use ``np.copyto`` / ``ndarray.fill``
    to update them in place. Rebinding any of those names (e.g. ``a = ...``)
    would silently break — the ctypes pointer would still point at the old
    buffer.

    Args:
      nsteps (int): number of integration steps
      dX (np.ndarray[nsteps]): step sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (np.ndarray[nsteps]): :math:`\\rho^{-1}(X_i)` per step
      spacc_int_off (SpaccMatrix): off-diagonal of A = int_m
      spacc_dec_off (SpaccMatrix): off-diagonal of B = dec_m
      d_int (np.ndarray): diagonal of A
      d_dec (np.ndarray): diagonal of B
      phi (np.ndarray): initial state :math:`\\Phi(X_0)`
      grid_idcs (list[int]): step indices at which to record snapshots

    Returns:
      (np.ndarray, np.ndarray): final state and stacked snapshots.
    """
    from ctypes import POINTER, c_double

    dim = phi.shape[0]
    # Persistent buffers — ctypes pointers must remain valid across the loop,
    # so every per-step update writes into these in place (never rebinds).
    phc = np.copy(phi).astype(np.float64, copy=False)
    F_phi = np.zeros(dim, dtype=np.float64)
    F_a = np.zeros(dim, dtype=np.float64)
    a = np.empty(dim, dtype=np.float64)
    bufs = _etd_step_buffers(dim)
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]

    phc_p = phc.ctypes.data_as(POINTER(c_double))
    F_phi_p = F_phi.ctypes.data_as(POINTER(c_double))
    F_a_p = F_a.ctypes.data_as(POINTER(c_double))
    a_p = a.ctypes.data_as(POINTER(c_double))

    int_off_empty = (spacc_int_off is None) or (spacc_int_off.nnz == 0)
    dec_off_empty = (spacc_dec_off is None) or (spacc_dec_off.nnz == 0)

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    # See module-level :data:`_EM_BLOWUP_CAVEAT`.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = dX[k]
            ri = rho_inv[k]

            _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs)

            # F_phi = int_off @ phc + ri * dec_off @ phc
            F_phi.fill(0.0)
            if not int_off_empty:
                spacc_int_off.gemv_ctargs(1.0, phc_p, F_phi_p)
            if not dec_off_empty:
                spacc_dec_off.gemv_ctargs(ri, phc_p, F_phi_p)

            # a = eD * phc + h * phi1 * F_phi
            np.multiply(eD, phc, out=a)
            np.multiply(phi1, F_phi, out=scratch)
            scratch *= h
            np.add(a, scratch, out=a)

            # F_a = int_off @ a + ri * dec_off @ a
            F_a.fill(0.0)
            if not int_off_empty:
                spacc_int_off.gemv_ctargs(1.0, a_p, F_a_p)
            if not dec_off_empty:
                spacc_dec_off.gemv_ctargs(ri, a_p, F_a_p)

            # phc = a + h * phi2 * (F_a - F_phi)  (in-place into phc)
            np.subtract(F_a, F_phi, out=scratch)
            scratch *= h
            np.multiply(scratch, phi2, out=scratch)
            np.add(a, scratch, out=phc)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc))
                grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc, np.array(grid_sol)
