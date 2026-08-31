from ctypes import POINTER, c_double
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from MCEq import config
from MCEq.misc import info
from MCEq.operator_assembly import (  # noqa: F401  (re-exported)
    CompiledOperator,
    compile_operator,
    secant_layout,
    split_diagonal,
)

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
                # Concentrated log-sample plus a dense linear sample for
                # the bulk. The log part must span the WHOLE domain, not
                # stop at X = 1: on near-horizontal trajectories
                # (max_X ~ 3e4) the interval X = 1..50 g/cm^2 still sits
                # at 80..35 km altitude where 1/rho varies over decades,
                # and a linear bulk sample alone (max_X/4000 ~ 7 g/cm^2
                # per point) under-resolves it. The mis-integrated
                # per-step means (x0.4..x1.6) imprinted a coherent
                # 10-25% bump-dip on ALL species at X ~ 5-40 g/cm^2 for
                # theta >~ 85 deg (found via nu3d q-table wiggles at
                # ~40 km altitude, 2026-07-10). Log sampling costs
                # nothing in the bulk, where 1/rho is flat per log-X;
                # the linear sample is kept so vertical-path accuracy
                # is unchanged.
                X_log_lo = max(1e-7, X_min if X_min > 0 else 1e-7)
                if X_max > X_log_lo:
                    X_top = np.geomspace(X_log_lo, X_max, 6001)
                else:
                    X_top = np.empty(0)
                X_bulk = np.linspace(max(X_log_lo, X_min), X_max, 4001)
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


#: The diagonal / off-diagonal split lives in :mod:`MCEq.operator_assembly`.
_etd_split_cache = split_diagonal


def _etd_off_to_bsr(off_csr, blocksize):
    """Convert a CSR off-diagonal to BSR(``blocksize``), padding if needed.

    Returns ``(off_bsr, n_padded)``. Pads to the next multiple of
    ``blocksize`` by extending ``indptr`` with copies of its tail (zero new
    entries); the trailing rows / cols are then zero. Empty input
    (``nnz == 0``) is returned as a zero BSR at the SAME padded dimension —
    NOT at ``n_orig``: when one off-diagonal is empty and the other is not
    (e.g. ``dec_off`` empty because all decays are disabled, as in a pure
    e±/γ EM-cascade solve), an unpadded empty block mismatches the padded
    ``int_off`` and state vector and crashes the per-step SpMV.
    """
    n_orig = off_csr.shape[0]
    pad = (-n_orig) % blocksize
    if off_csr.nnz == 0:
        z = sp.csr_matrix((n_orig + pad, n_orig + pad))
        return z.tobsr(blocksize=(blocksize, blocksize)), n_orig + pad
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

# Default K-tile size for the Accelerate Sparse BLAS SpMM kernel in
# :func:`solv_spacc_etd2_multirhs`. The K-to-1000 bench shows
# ``sparse_matrix_product_dense_double`` peaks at K ≈ 32–64 on the M3 Pro
# (3.0–3.2× /RHS) then drops to ≈ 1.4× at K ≥ 128. Splitting larger K
# requests into 64-column tiles restores the peak operating point at all K.
_SPACC_SPMM_TILE = 64


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


def _etd_step_buffers_multipath(dim, K):
    """Scratch buffers for the per-RHS-path multi-RHS kernel.

    Stage-3 lifts ``D`` / ``eD`` / ``φ₁`` / ``φ₂`` from ``(dim,)`` to
    ``(dim, K)`` because both ``h`` and ``ri`` vary across columns (each
    column carries its own atmosphere path). Memory cost at the full-sky
    operating point dim=7986, K=3072, fp64: ≈ 600 MB for the three
    (dim, K) diag buffers + (dim, K) scratch. See
    wiki/methods/multi-rhs-etd2-design.md Stage 3 section.
    """
    return {
        "D": np.empty((dim, K), dtype=np.float64),
        "hD": np.empty((dim, K), dtype=np.float64),
        "eD": np.empty((dim, K), dtype=np.float64),
        "phi1": np.empty((dim, K), dtype=np.float64),
        "phi2": np.empty((dim, K), dtype=np.float64),
        "scratch": np.empty((dim, K), dtype=np.float64),
        "abs_hD": np.empty((dim, K), dtype=np.float64),
        "mask1": np.empty((dim, K), dtype=bool),
        "mask2": np.empty((dim, K), dtype=bool),
    }


def _etd_compute_diag_factors_multipath(h_K, ri_K, d_int, d_dec, bufs):
    """Per-RHS-path analogue of :func:`_etd_compute_diag_factors`.

    Computes per-column ``D[i, k] = d_int[i] + ri_K[k] · d_dec[i]``,
    ``hD = h_K[k] · D``, ``eD = exp(hD)``, and the two φ-functions of
    ``hD`` elementwise over the ``(dim, K)`` plane. Branches around the
    small-|hD| Taylor patch are computed via ``where=`` masks; numpy's
    ufunc ``where=`` works on 2-D arrays so the per-cell branch is
    cheap. cupy 14 rejects ``where=`` on most arithmetic ufuncs (verified
    on the PriNCe port); when porting this to cupy, switch to the
    ``copyto(where=)`` pattern used in PriNCe's etd2.py.

    Frozen-column semantics: when ``h_K[k] == 0`` the column is "done"
    (its own path has fewer steps than max). The math degenerates to
    ``eD = 1, φ₁ = 1, φ₂ = 1/2`` for that column (Taylor branches at
    hD = 0), and the downstream ETD2 update collapses to
    ``state ← state``. No explicit masking needed.
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

    # D[i, k] = d_int[i] + ri_K[k] · d_dec[i]  -- (dim, K), no extra alloc.
    np.multiply(d_dec[:, None], ri_K[None, :], out=D)
    np.add(D, d_int[:, None], out=D)
    # hD = h_K[None, :] * D ; eD = exp(hD).
    np.multiply(D, h_K[None, :], out=hD)
    np.exp(hD, out=eD)

    np.abs(hD, out=abs_hD)
    np.greater(abs_hD, _PHI1_SMALL, out=mask1)
    np.greater(abs_hD, _PHI2_SMALL, out=mask2)

    # phi1: analytic (eD - 1) / hD where mask1, Taylor elsewhere.
    np.subtract(eD, 1.0, out=phi1)
    np.divide(phi1, hD, out=phi1, where=mask1)
    np.multiply(hD, _INV_6, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 1.0, out=scratch)
    np.invert(mask1, out=mask1)
    np.copyto(phi1, scratch, where=mask1)

    # phi2: analytic (eD - 1 - hD) / hD² where mask2, Taylor elsewhere.
    np.subtract(eD, 1.0, out=phi2)
    np.subtract(phi2, hD, out=phi2)
    np.multiply(hD, hD, out=scratch)
    np.divide(phi2, scratch, out=phi2, where=mask2)
    np.multiply(hD, _INV_24, out=scratch)
    np.add(scratch, _INV_6, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.invert(mask2, out=mask2)
    np.copyto(phi2, scratch, where=mask2)


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


# ---------------------------------------------------------------------------
# The ETD2RK driver and its backends.
#
# One step loop serves every route — paraxial and sec(theta)-coupled, single
# axis, shared-path multi-RHS and the LPT carousel — on numpy, MKL and CUDA.
# :func:`MCEq.operator_assembly.compile_operator` prepares the operator; a
# backend object places it on its library / device and executes the stages
# of :func:`etd2_driver` there. Nothing else differs between backends.
# ---------------------------------------------------------------------------


def secant_split(int_m, dec_m, sec_ops):
    """``(d_int, d_dec, int_off, dec_off)`` of A, B in the secant layout
    (see :func:`MCEq.operator_assembly.compile_operator`)."""
    return compile_operator(int_m, dec_m, sec_ops).split


def _secant_phi_factors(ZB, out=None, work=None):
    """Elementwise exp/phi1/phi2 with Taylor patches for a block argument.

    ``out=(exp, phi1, phi2)`` and ``work=(scratch, large)`` let the hot
    loop reuse its block-sized arrays instead of allocating six
    temporaries per step. The no-argument form keeps the small standalone
    helper convenient for tests and callers outside the driver.
    """
    if out is not None:
        eDB, phi1, phi2 = out
        scratch, large = work
        np.expm1(ZB, out=scratch)

        # Taylor branches, evaluated for the whole block and overwritten
        # by the quotient on entries outside the small-argument patch.
        np.multiply(ZB, _INV_6, out=phi1)
        np.add(phi1, 0.5, out=phi1)
        np.multiply(ZB, phi1, out=phi1)
        np.add(phi1, 1.0, out=phi1)
        np.greater(np.abs(ZB, out=eDB), _PHI1_SMALL, out=large)
        np.divide(scratch, ZB, out=phi1, where=large)

        np.multiply(ZB, _INV_24, out=phi2)
        np.add(phi2, _INV_6, out=phi2)
        np.multiply(ZB, phi2, out=phi2)
        np.add(phi2, 0.5, out=phi2)
        np.greater(eDB, _PHI2_SMALL, out=large)
        np.subtract(scratch, ZB, out=eDB)
        np.multiply(ZB, ZB, out=scratch)
        np.divide(eDB, scratch, out=phi2, where=large)
        np.exp(ZB, out=eDB)
        return out

    safe = np.where(ZB == 0.0, 1.0, ZB)
    e1 = np.expm1(ZB)
    phi1 = np.where(np.abs(ZB) > _PHI1_SMALL, e1 / safe, 1.0 + ZB * (0.5 + ZB * _INV_6))
    phi2 = np.where(
        np.abs(ZB) > _PHI2_SMALL,
        (e1 - ZB) / (safe * safe),
        0.5 + ZB * (_INV_6 + ZB * _INV_24),
    )
    return np.exp(ZB), phi1, phi2


def _cuda_secant_phi_factors(cp, fl_pr, ZB):
    """cupy form of :func:`_secant_phi_factors` (cupy ufuncs take no ``where``)."""
    safe = cp.where(ZB == 0.0, fl_pr(1.0), ZB)
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


def _secant_left_matmul(matrix, plane, out=None):
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


class ScipyApplyOff:
    """``out = int_off x + ri dec_off x`` through scipy CSR SpMM."""

    name = "numpy"

    def __init__(self, int_off, dec_off):
        self.int_off, self.dec_off = int_off, dec_off

    def bind(self, dim, K, nsteps):
        pass

    def __call__(self, x, out, ri):
        np.copyto(out, self.int_off.dot(x))
        ri_dec = self.dec_off.dot(x)
        ri_dec *= ri  # scalar, or (K,) broadcast over the lane axis
        np.add(out, ri_dec, out=out)

    def close(self):
        pass


class MklApplyOff:
    """``out = int_off x + ri dec_off x`` through MKL sparse BLAS.

    The driver's ``(dim, K)`` buffers are C-contiguous, so one row-major
    ``mkl_sparse_d_mm`` per stage covers all lanes (SpMV at K = 1). A
    scalar ``ri`` is fused into the dec SpMM's alpha; a ``(K,)`` lane row
    scales a separate accumulator. BSR handles pad the operator to a
    multiple of the block size; the operand is then staged through
    padded copies. ``owns`` closes the handles with the binding.
    """

    name = "mkl"

    def __init__(self, mkl_int_off, mkl_dec_off, owns=False):
        self.int_off, self.dec_off = (
            m if m is not None and m.nnz else None for m in (mkl_int_off, mkl_dec_off)
        )
        self.handles = [m for m in (self.int_off, self.dec_off) if m is not None]
        self.owns = owns

    def bind(self, dim, K, nsteps):
        from ctypes import POINTER, c_double

        self.K = K
        self.n_padded = max([dim] + [m.n_padded for m in self.handles])
        self.dim = dim
        if K > 1:
            for m in self.handles:
                m.set_mm_hint(K, expected_calls=2 * nsteps, layout=101)
        self._dec_buf = np.empty((self.n_padded, K), dtype=np.float64)
        self._pad = self.n_padded != dim
        if self._pad:
            self._x_pad = np.zeros((self.n_padded, K), dtype=np.float64)
            self._out_pad = np.empty((self.n_padded, K), dtype=np.float64)
        self._ptrs = {}
        self._ptr_type = POINTER(c_double)

    def _ptr(self, arr):
        p = self._ptrs.get(id(arr))
        if p is None:
            p = self._ptrs[id(arr)] = arr.ctypes.data_as(self._ptr_type)
        return p

    def _spmm(self, m, alpha, x, beta, out):
        if self.K == 1:
            m.gemv_ctargs(alpha, self._ptr(x), beta, self._ptr(out))
        else:
            K = self.K
            m.gemm_ctargs(
                alpha, K, self._ptr(x), K, self._ptr(out), K, beta=beta, layout=101
            )

    def __call__(self, x, out, ri):
        if self._pad:
            self._x_pad[: self.dim] = x
            x, out_full = self._x_pad, out
            out = self._out_pad
        if self.int_off is None:
            out.fill(0.0)
        else:
            self._spmm(self.int_off, 1.0, x, 0.0, out)
        if self.dec_off is not None:
            if np.ndim(ri) == 0:
                self._spmm(self.dec_off, float(ri), x, 1.0, out)
            else:
                self._spmm(self.dec_off, 1.0, x, 0.0, self._dec_buf)
                np.multiply(self._dec_buf, ri, out=self._dec_buf)
                np.add(out, self._dec_buf, out=out)
        if self._pad:
            np.copyto(out_full, out[: self.dim])

    def close(self):
        if self.owns:
            for m in self.handles:
                m.close()


# --- backends ------------------------------------------------------------


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
    """Stage execution on host arrays for :func:`etd2_driver`.

    numpy elementwise kernels and BLAS GEMMs throughout; the SpMM is the
    ``apply_off`` binding of the sparse library (scipy or MKL). ``op`` is
    the :class:`~MCEq.operator_assembly.CompiledOperator` the binding was
    built from — it carries the layout and the coupling operators.
    """

    xp = np
    dtype = np.float64
    left_matmul = staticmethod(_secant_left_matmul)

    def __init__(self, op, apply_off):
        self.op = op
        self.name = apply_off.name
        self.d_int, self.d_dec = op.d_int, op.d_dec
        self._apply_off = apply_off
        self._post = _rowmajor_post_apply()

    def bind(self, dim, K, per_lane, nsteps):
        if self.op.dim != dim:
            raise ValueError(
                f"HostBackend: operator dim {self.op.dim} != state dim {dim}"
            )
        self._per_lane = per_lane
        self._bufs = (
            _etd_step_buffers_multipath(dim, K) if per_lane else _etd_step_buffers(dim)
        )
        self._scratch = np.empty((dim, K), dtype=np.float64)
        self._block = None
        self._fused = self._post is not None and dim * K >= _FUSED_MIN_ELEMENTS
        self._h1 = np.empty(1, dtype=np.float64)
        self._apply_off.bind(dim, K, nsteps)

    def coupling(self):
        c = self.op.coupling
        return c.T_P, c.T_PP, c.V, c.Vi, c.lam

    def state_buffers(self, dim, K):
        return tuple(np.zeros((dim, K), dtype=np.float64) for _ in range(4))

    def apply_off(self, x, out, ri):
        self._apply_off(x, out, ri)

    def diag_factors(self, h, ri):
        """``eD, phi1, phi2`` of ``h (d_int + ri d_dec)``, broadcastable to (dim, K)."""
        b = self._bufs
        if self._per_lane:
            _etd_compute_diag_factors_multipath(h, ri, self.d_int, self.d_dec, b)
            return b["eD"], b["phi1"], b["phi2"]
        _etd_compute_diag_factors(h, ri, self.d_int, self.d_dec, b)
        return b["eD"][:, None], b["phi1"][:, None], b["phi2"][:, None]

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

    def asarray(self, a):
        return np.asarray(a, dtype=np.float64)

    def to_host(self, a):
        return np.asarray(a, dtype=np.float64)

    def synchronize(self):
        pass

    def close(self):
        self._apply_off.close()


def numpy_backend(op):
    """Host backend on scipy CSR SpMM."""
    return HostBackend(op, ScipyApplyOff(op.int_off, op.dec_off))


def mkl_backend(op, expected_calls=2000):
    """Host backend on MKL sparse BLAS; owns the CSR handles it creates."""
    handles = tuple(
        MklSparseMatrix(off, expected_calls=expected_calls) if off.nnz else None
        for off in (op.int_off, op.dec_off)
    )
    return HostBackend(op, MklApplyOff(*handles, owns=True))


class CudaOperator:
    """Device copy of a compiled operator's split for the CUDA backend.

    Owns the cuSPARSE CSR copies of ``int_off`` / ``dec_off`` (``None``
    when empty — an empty CSR is ill-defined for some cuSPARSE versions)
    and the diagonals, in ``fp_precision`` (32 or 64). The state and
    scratch buffers are allocated per solve by :class:`CudaBackend`; one
    device operator serves every K.
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
        if fp_precision == 32:
            fl_pr = cp.float32
        elif fp_precision == 64:
            fl_pr = cp.float64
        else:
            raise ValueError(
                f"CudaOperator: fp_precision must be 32 or 64, got {fp_precision}"
            )
        self.cp = cp
        self.fl_pr = fl_pr
        self.fp_precision = int(fp_precision)
        self.device_id = int(device_id)
        cp.cuda.Device(self.device_id).use()
        self.dim = int(d_int.shape[0])
        self.cu_int_off = cusp.csr_matrix(int_off, dtype=fl_pr) if int_off.nnz else None
        self.cu_dec_off = cusp.csr_matrix(dec_off, dtype=fl_pr) if dec_off.nnz else None
        self.cu_d_int = cp.asarray(d_int, dtype=fl_pr)
        self.cu_d_dec = cp.asarray(d_dec, dtype=fl_pr)


#: Name the CUDA operator carried before the backends were unified.
CudaEtd2Context = CudaOperator


class CudaEtd2MultiRHSContext(CudaOperator):
    """:class:`CudaOperator` that also records the batch width it was
    created for (kept for callers of the former multi-RHS context)."""

    def __init__(self, int_off, dec_off, d_int, d_dec, K, device_id, fp_precision):
        super().__init__(int_off, dec_off, d_int, d_dec, device_id, fp_precision)
        self.K = int(K)


class CudaBackend:
    """Stage execution on the device for :func:`etd2_driver`.

    cuSPARSE SpMM through cupyx, cublas GEMMs, and the fused
    ElementwiseKernels of :func:`_cuda_etd2_kernels`. ``dev`` is the
    :class:`CudaOperator` of ``op``'s split. With ``fp_precision=32``
    everything runs in fp32 except the diagonal factors, which the fused
    kernel evaluates in fp64.
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
        self._D = (
            None if per_lane else tuple(cp.empty(dim, dtype=dtype) for _ in range(2))
        )

    def coupling(self):
        c = self.op.coupling
        if self._coupling is None:
            self._coupling = tuple(
                self.cp.asarray(m, dtype=self.dtype)
                for m in (c.T_P, c.T_PP, c.V, c.Vi, c.lam)
            )
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
        cp, Kset = self.cp, self._kernels
        d_int, d_dec = self.d_int, self.d_dec
        eD, phi1, phi2 = self._factors
        fp32 = self.dtype is cp.float32
        if self._per_lane:
            kernel = (
                Kset.phi_compute_multipath_f64diag
                if fp32
                else Kset.phi_compute_multipath
            )
            kernel(
                d_int[:, None], d_dec[:, None], h[None, :], ri[None, :], eD, phi1, phi2
            )
            return eD, phi1, phi2
        if fp32:
            Kset.phi_compute_multipath_f64diag(d_int, d_dec, h, ri, eD, phi1, phi2)
        else:
            D, hD = self._D
            cp.multiply(d_dec, ri, out=D)
            cp.add(D, d_int, out=D)
            cp.multiply(D, h, out=hD)
            cp.exp(hD, out=eD)
            Kset.phi_compute(hD, eD, eD, phi1, phi2)
        return eD[:, None], phi1[:, None], phi2[:, None]

    def block_factors(self, ZB):
        return _cuda_secant_phi_factors(self.cp, self.dtype, ZB)

    def predictor(self, eD, x, phi1, F, h, out):
        self._kernels.post_apply1(eD, x, phi1, F, h, out)

    def corrector(self, a, F_a, F, phi2, h, out):
        self._kernels.post_apply2(a, F_a, F, phi2, h, out)

    def asarray(self, a):
        return self.cp.asarray(a, dtype=self.dtype)

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


# --- the driver ------------------------------------------------------------


def etd2_driver(
    nsteps, dX, rho_inv, be, phi, grid_idcs, schedule=None, phi0_per_pixel=None
):
    """ETD2RK step loop — every route, every backend.

    Integrates ``dPhi/dX = (A + ri B) S Phi`` with the Cox–Matthews
    exponential RK2: the diagonal ``D`` of ``A + ri B`` is treated exactly
    through an integrating factor, the off-diagonal remainder explicitly.
    ``S = I`` is the paraxial transport. With the sec(theta) transport
    ``S = I + T Pi``, where ``T`` is the constant Hankel-space representation
    of multiplication by ``min(sec theta, sec theta_cap)`` (see
    :mod:`MCEq.secant`) and ``Pi`` the projector onto the state columns with
    ``E_kin < config.secant_theta_e_max``. The operator is then split, per
    state column i in the support of Pi and the coupled mode subspace P
    (S_P = (I+T)[P,P]):

      exact slot   D0_i S_P      D0_i = diagonal of A + ri B at kappa = 0
                                 (the k-independent part)
      remainder    everything else: off-diagonal production acting on
                   w = S Phi, the k-dependent diagonal spread on the
                   coupled rows, and the one-way cross coupling from the
                   uncoupled modes.

    The exact slot is evaluated in the eigenbasis ``S_P = V diag(lam)
    V^-1`` (constant, shared by every column), where exp/phi1/phi2 of
    ``h D0_i S_P`` are elementwise. Unconditionally stable at any
    stiffness; stitching the coupling into the CSR instead puts stiff
    coupled loss terms in the explicit part and diverges.

    The operator behind ``be`` is a :class:`~MCEq.operator_assembly.
    CompiledOperator`; with coupling, the state lives in its low-E-first
    layout (``phi`` and the results are in the original layout), the
    coupled plane is the corner block ``x.reshape(n_k, N, K)[:n_P, :n_g]``
    and the operand of ``T_P`` the low-E block ``[:, :n_g]`` — strided
    views.

    Stages per step (state x = Phi_n, corner C(.) of a full-state array;
    the corner terms are absent for the paraxial transport):

      1. factors      eD, phi1, phi2 = f(h D), D = d_int + ri d_dec;
                      block factors f(h D0_i lam_j) on the corner
      2. operand      x_c = C(x); Y = T_P x[:, G];  C(x) <- x_c + Y
                      (x now holds w = S x)
      3. remainder    F = A_off w + ri B_off w  (SpMM);
                      C(F) += Df (x_c + Y) - D0 (x_c + T_PP x_c)
      4. predictor    a = eD x + h phi1 F on the full state;
                      C(a) = V eDB Vi x_c + h V phi1B Vi C(F)
      5. operand and remainder (2-3) at a: a_c = C(a), Y_a, F_a
      6. corrector    x = a + h phi2 (F_a - F) on the full state;
                      C(x) = a_c + h V phi2B Vi (C(F_a) - C(F))
      7. harvest      carousel harvest/reset, or int_grid snapshot

    In 4 and 6 the full-state formula also writes the corner, using the
    operand w there instead of the state; that block is discarded and
    replaced by the exact-slot update, so no copy of the state is needed
    to form w. Batching: ``phi`` is ``(dim,)`` or ``(dim, K)``; ``dX`` /
    ``rho_inv`` are ``(nsteps,)`` (one shared path, the multi-RHS route)
    or ``(nsteps, K)`` (per-lane paths; lanes with ``h == 0`` are pinned
    to exact identity). A :class:`CarouselSchedule` with
    ``phi0_per_pixel`` turns the per-lane form into the LPT carousel and
    the return value into the ``(dim, K_total)`` per-pixel solution.
    Single-axis is K = 1 without a schedule.
    """
    xp = be.xp
    dtype = be.dtype
    lay = be.op.layout
    coupled = lay.coupled
    phi = np.asarray(phi)
    batched = phi.ndim == 2
    per_lane = np.ndim(dX) == 2
    dim = phi.shape[0]
    K = phi.shape[1] if batched else 1
    if per_lane and not batched:
        raise ValueError("etd2_driver: (nsteps, K) dX requires a (dim, K) state")
    if schedule is not None:
        if not per_lane:
            raise ValueError(
                "etd2_driver: a carousel schedule requires (T, K) dX / rho_inv"
            )
        if grid_idcs:
            raise ValueError(
                "etd2_driver: carousel runs do not support int_grid snapshots"
            )
        if phi0_per_pixel.shape != (dim, schedule.K_total):
            raise ValueError(
                "etd2_driver: phi0_per_pixel must be (dim, K_total) "
                f"= ({dim}, {schedule.K_total}); got {phi0_per_pixel.shape}"
            )

    be.bind(dim, K, per_lane, nsteps)
    phc, F_phi, F_a, a = be.state_buffers(dim, K)

    def to_layout(x):
        return x if lay.perm is None else x[xp.asarray(lay.perm)]

    def from_layout(x):
        return x if lay.inv_perm is None else x[xp.asarray(lay.inv_perm)]

    phc[:] = to_layout(be.asarray(phi.reshape(dim, K)))

    if per_lane:
        dX_b = be.asarray(dX)
        ri_b = be.asarray(rho_inv)

    if coupled:
        n_k, N, n_P, n_g = lay.n_k, lay.N, lay.n_P, lay.n_g
        T_P, T_PP, V, Vi, lam = be.coupling()
        lmm = be.left_matmul

        def corner(x):
            return x.reshape(n_k, N, K)[:n_P, :n_g]

        def low_e(x):
            return x.reshape(n_k, N, K)[:, :n_g]

        # Constant diagonals of the coupled plane and of the kappa = 0 mode.
        d_int_c, d_dec_c = (d.reshape(n_k, N)[:n_P, :n_g] for d in (be.d_int, be.d_dec))
        d_int_0, d_dec_0 = (d.reshape(n_k, N)[0, :n_g] for d in (be.d_int, be.d_dec))

        plane = (n_P, n_g, K)
        Y = xp.empty(plane, dtype=dtype)
        x_c = xp.empty(plane, dtype=dtype)
        a_c = xp.empty(plane, dtype=dtype)
        F_c = xp.empty(plane, dtype=dtype)
        tmp = xp.empty(plane, dtype=dtype)
        mode_tmp = xp.empty(plane, dtype=dtype)

        def block_action(factors, source, out):
            """``V diag(factors) V^-1 source`` on a coupled plane."""
            lmm(Vi, source, out=mode_tmp)
            xp.multiply(factors, mode_tmp, out=mode_tmp)
            lmm(V, mode_tmp, out=out)

        def eval_F(x, x_c, F, ri, Df, D0):
            """Stages 2-3 at the state x whose corner is held in x_c: F <- the
            remainder at x; x itself becomes the operand w = S x."""
            lmm(T_P, low_e(x), out=Y)
            xp.add(x_c, Y, out=corner(x))
            be.apply_off(x, F, ri)
            lmm(T_PP, x_c, out=tmp)
            xp.add(x_c, tmp, out=tmp)
            xp.multiply(Df, corner(x), out=Y)
            xp.multiply(D0, tmp, out=tmp)
            xp.subtract(Y, tmp, out=Y)
            xp.add(corner(F), Y, out=corner(F))

    if schedule is not None:
        sol_pixel = xp.empty((dim, schedule.K_total), dtype=dtype)
        phi0_pp = to_layout(be.asarray(phi0_per_pixel))
        rs, cs = schedule.reset_t_starts, schedule.record_t_starts
        rj, rp = (xp.asarray(schedule.reset_j), xp.asarray(schedule.reset_pixel))
        cj, cpix = (xp.asarray(schedule.record_j), xp.asarray(schedule.record_pixel))

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    # See module-level :data:`_EM_BLOWUP_CAVEAT` for the errstate contract.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            # 1. diagonal factors of the full state and of the exact slot
            if per_lane:
                h, ri = dX_b[k], ri_b[k]  # (K,) lane rows
                h_b = h[None, :]
            else:
                h, ri = dtype(dX[k]), dtype(rho_inv[k])
                h_b = h
            eD, phi1, phi2 = be.diag_factors(h, ri)
            if coupled:
                if per_lane:
                    h_c = h[None, None, :]
                    frozen = (h == 0.0)[None, None, :]
                    Df = d_dec_c[:, :, None] * ri + d_int_c[:, :, None]
                    D0 = d_dec_0[:, None] * ri + d_int_0[:, None]
                    ZB = lam[:, None, None] * (D0 * h)
                    D0_b = D0[None]
                    eDB, phi1B, phi2B = be.block_factors(ZB)
                else:
                    h_c = h
                    Df = (d_dec_c * ri + d_int_c)[:, :, None]
                    D0 = d_dec_0 * ri + d_int_0
                    ZB = lam[:, None] * (D0 * h)
                    D0_b = D0[None, :, None]
                    eDB, phi1B, phi2B = (f[:, :, None] for f in be.block_factors(ZB))

            # 2-3. operand and remainder at the state
            if coupled:
                xp.copyto(x_c, corner(phc))
                eval_F(phc, x_c, F_phi, ri, Df, D0_b)
                xp.copyto(F_c, corner(F_phi))
            else:
                be.apply_off(phc, F_phi, ri)

            # 4. predictor a = eD x + h phi1 F, exact slot on the corner
            be.predictor(eD, phc, phi1, F_phi, h_b, out=a)
            if coupled:
                block_action(eDB, x_c, a_c)
                block_action(phi1B, F_c, tmp)
                xp.multiply(tmp, h_c, out=tmp)
                xp.add(a_c, tmp, out=a_c)
                if per_lane:
                    xp.copyto(a_c, x_c, where=frozen)
                xp.copyto(corner(a), a_c)

            # 5. operand and remainder at the predictor
            if coupled:
                eval_F(a, a_c, F_a, ri, Df, D0_b)
            else:
                be.apply_off(a, F_a, ri)

            # 6. corrector x = a + h phi2 (F_a - F), exact slot on the corner
            be.corrector(a, F_a, F_phi, phi2, h_b, out=phc)
            if coupled:
                xp.subtract(corner(F_a), F_c, out=tmp)
                block_action(phi2B, tmp, tmp)
                xp.multiply(tmp, h_c, out=tmp)
                xp.add(a_c, tmp, out=tmp)
                if per_lane:
                    xp.copyto(tmp, x_c, where=frozen)
                xp.copyto(corner(phc), tmp)

            # 7. harvest finished lanes BEFORE the reset reloads them
            if schedule is not None:
                lo, hi = int(cs[k]), int(cs[k + 1])
                if hi > lo:
                    sol_pixel[:, cpix[lo:hi]] = phc[:, cj[lo:hi]]
                lo, hi = int(rs[k]), int(rs[k + 1])
                if hi > lo:
                    phc[:, rj[lo:hi]] = phi0_pp[:, rp[lo:hi]]
            elif grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(phc.copy())
                grid_step += 1

    be.synchronize()
    elapsed = time() - start
    info(
        2,
        f"Performance ({be.name} K={K}): "
        f"{1e3 * elapsed / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * elapsed / float(nsteps) / float(K):6.3f}ms/iteration/RHS)",
    )

    if schedule is not None:
        return be.to_host(from_layout(sol_pixel))
    sol = be.to_host(from_layout(phc))
    grid = np.array([])
    if grid_sol:
        grid = xp.stack(grid_sol)
        grid = be.to_host(
            grid if lay.inv_perm is None else grid[:, xp.asarray(lay.inv_perm)]
        )
    if not batched:
        sol = sol[:, 0]
        if grid.size:
            grid = grid[:, :, 0]
    return sol, grid


# --- entry points by backend and route -------------------------------------
#
# Every route is ``etd2_driver`` on one backend; these names bind the
# (matrices | handles | device operator) a caller holds to that backend.
# ``MCEqRun`` builds its backends through ``operator_assembly`` and calls
# the driver directly.


def _host_op(d_int, d_dec, sec_ops=None):
    return CompiledOperator.from_split(d_int, d_dec, sec_ops=sec_ops)


def _multirhs(single, phi_pos):
    """The multi-RHS name of a route: the same driver call, ``(dim, K)``
    state required (``phi`` is positional argument ``phi_pos``)."""

    def kernel(*args):
        phi = args[phi_pos]
        if np.ndim(phi) != 2:
            raise ValueError(
                f"{kernel.__name__}: phi must be 2-D (dim, K), got shape {np.shape(phi)}"
            )
        return single(*args)

    kernel.__name__ = single.__name__ + "_multirhs"
    kernel.__doc__ = f"Shared-path multi-RHS lift of :func:`{single.__name__}`."
    return kernel


def solv_numpy_etd2(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    """ETD2RK on scipy sparse; ``phi`` may be ``(dim,)`` or ``(dim, K)``."""
    be = numpy_backend(compile_operator(int_m, dec_m))
    return etd2_driver(nsteps, dX, rho_inv, be, phi, grid_idcs)


solv_numpy_etd2_multirhs = _multirhs(solv_numpy_etd2, phi_pos=5)


def solv_numpy_etd2_carousel(
    int_m, dec_m, dX, rho_inv, phi_initial, schedule, phi0_per_pixel
):
    """ETD2RK LPT carousel on scipy sparse (see :func:`compile_carousel_schedule`)."""
    be = numpy_backend(compile_operator(int_m, dec_m))
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        be,
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


def solv_numpy_etd2_secant(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs, sec_ops):
    """ETD2RK with the sec(theta) mode coupling on scipy sparse."""
    be = numpy_backend(compile_operator(int_m, dec_m, sec_ops))
    return etd2_driver(nsteps, dX, rho_inv, be, phi, grid_idcs)


solv_numpy_etd2_secant_multirhs = _multirhs(solv_numpy_etd2_secant, phi_pos=5)


def solv_numpy_etd2_secant_carousel(
    int_m, dec_m, dX, rho_inv, phi_initial, schedule, phi0_per_pixel, sec_ops
):
    be = numpy_backend(compile_operator(int_m, dec_m, sec_ops))
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        be,
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


def solv_mkl_etd2(
    nsteps, dX, rho_inv, mkl_int_off, mkl_dec_off, d_int, d_dec, phi, grid_idcs
):
    """ETD2RK on MKL sparse BLAS handles of the off-diagonals (CSR or BSR)."""
    be = HostBackend(_host_op(d_int, d_dec), MklApplyOff(mkl_int_off, mkl_dec_off))
    return etd2_driver(nsteps, dX, rho_inv, be, phi, grid_idcs)


solv_mkl_etd2_multirhs = _multirhs(solv_mkl_etd2, phi_pos=7)


def solv_mkl_etd2_carousel(
    mkl_int_off,
    mkl_dec_off,
    d_int,
    d_dec,
    dX,
    rho_inv,
    phi_initial,
    schedule,
    phi0_per_pixel,
):
    be = HostBackend(_host_op(d_int, d_dec), MklApplyOff(mkl_int_off, mkl_dec_off))
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        be,
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


def solv_mkl_etd2_secant(
    nsteps, dX, rho_inv, mkl_int_off, mkl_dec_off, d_int, d_dec, phi, grid_idcs, sec_ops
):
    """ETD2RK with the sec(theta) coupling on MKL handles of the
    :func:`secant_split` off-diagonals."""
    be = HostBackend(
        _host_op(d_int, d_dec, sec_ops), MklApplyOff(mkl_int_off, mkl_dec_off)
    )
    return etd2_driver(nsteps, dX, rho_inv, be, phi, grid_idcs)


solv_mkl_etd2_secant_multirhs = _multirhs(solv_mkl_etd2_secant, phi_pos=7)


def solv_mkl_etd2_secant_carousel(
    mkl_int_off,
    mkl_dec_off,
    d_int,
    d_dec,
    dX,
    rho_inv,
    phi_initial,
    schedule,
    phi0_per_pixel,
    sec_ops,
):
    be = HostBackend(
        _host_op(d_int, d_dec, sec_ops), MklApplyOff(mkl_int_off, mkl_dec_off)
    )
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        be,
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


def _cuda_be(ctx, sec_ops=None):
    d_int, d_dec = (
        ctx.cp.asnumpy(d).astype(np.float64) for d in (ctx.cu_d_int, ctx.cu_d_dec)
    )
    return CudaBackend(ctx, _host_op(d_int, d_dec, sec_ops))


def solv_cuda_etd2(nsteps, dX, rho_inv, ctx, phi, grid_idcs):
    """ETD2RK on cuSPARSE via cupy; ``ctx`` a :class:`CudaOperator`."""
    return etd2_driver(nsteps, dX, rho_inv, _cuda_be(ctx), phi, grid_idcs)


solv_cuda_etd2_multirhs = _multirhs(solv_cuda_etd2, phi_pos=4)


def solv_cuda_etd2_carousel(ctx, dX, rho_inv, phi_initial, schedule, phi0_per_pixel):
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        _cuda_be(ctx),
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


def solv_cuda_etd2_secant(nsteps, dX, rho_inv, ctx, phi, grid_idcs, sec_ops):
    """ETD2RK with the sec(theta) coupling on cuSPARSE; ``ctx`` holds the
    :func:`secant_split` operators."""
    return etd2_driver(nsteps, dX, rho_inv, _cuda_be(ctx, sec_ops), phi, grid_idcs)


solv_cuda_etd2_secant_multirhs = _multirhs(solv_cuda_etd2_secant, phi_pos=4)


def solv_cuda_etd2_secant_carousel(
    ctx, dX, rho_inv, phi_initial, schedule, phi0_per_pixel, sec_ops
):
    return etd2_driver(
        schedule.T,
        dX,
        rho_inv,
        _cuda_be(ctx, sec_ops),
        phi_initial,
        [],
        schedule=schedule,
        phi0_per_pixel=phi0_per_pixel,
    )


# ---------------------------------------------------------------------------
# LPT carousel schedule
#
# ``K_total`` pixels stream through a fixed-width ``K`` pipeline; when a
# slot finishes its current pixel, the next pixel's phi0 is loaded into
# that slot's column on the same step. The build phase below
# (``schedule_lpt`` + ``compile_carousel_schedule``) is pure NumPy and
# backend-agnostic; :func:`etd2_driver` consumes the schedule as stage 7
# (harvest before reset) of its step loop.
#
# Design: ../mceq-em-integration/wiki/methods/multi-rhs-lpt-carousel.md
# ---------------------------------------------------------------------------
from collections import namedtuple  # noqa: E402  (section-local, see banner)

CarouselSchedule = namedtuple(
    "CarouselSchedule",
    [
        "T",  # int — makespan (outer loop iters)
        "K",  # int — pipeline width (slots)
        "K_total",  # int — total pixels packed
        "slot_assignments",  # list[list[int]] — per-slot pixel ids in run order
        "reset_t_starts",  # (T+1,) int32 — CSR ptrs into reset_j / reset_pixel
        "reset_j",  # (R,) int32 — slot id of each reset event
        "reset_pixel",  # (R,) int32 — pixel id whose phi0 to load
        "record_t_starts",  # (T+1,) int32 — CSR ptrs into record_j / record_pixel
        "record_j",  # (K_total,) int32 — slot id of each harvest event
        "record_pixel",  # (K_total,) int32 — pixel id to record into
    ],
)


def schedule_lpt(nsteps_per_pixel, K):
    """LPT (longest-processing-time-first) multiway-makespan assignment.

    Sorts pixels by ``nsteps`` descending and greedily appends each to the
    slot with the currently smallest running length sum. LPT is guaranteed
    to be within 4/3 of optimal; in our regime (no single pixel
    dominates) it typically achieves ``T ≈ ⌈Σ/K⌉``.

    Args:
        nsteps_per_pixel: array-like of int, length K_total.
        K: int — desired pipeline width. Clamped to ``min(K, K_total)``.

    Returns:
        slot_assignments: list of K lists; slot j → ordered pixel ids.
        T: int — makespan = max over slots of total assigned nsteps.

    Notes:
        Pixel order within a slot does not affect the makespan; we keep
        the natural LPT order (longest first) for determinism.
    """
    import heapq

    ns = np.asarray(nsteps_per_pixel, dtype=np.int64)
    K_total = int(ns.size)
    K_eff = int(min(K, K_total))
    if K_eff < 1:
        raise ValueError(f"schedule_lpt: K must be >= 1 (got {K})")

    order = np.argsort(ns, kind="stable")[::-1]  # longest first

    # Min-heap keyed on (current slot length, slot id). The list of pixel
    # ids per slot lives outside the heap to keep heap entries small.
    heap = [(0, j) for j in range(K_eff)]
    heapq.heapify(heap)
    slot_assignments = [[] for _ in range(K_eff)]
    for pid in order:
        pid_i = int(pid)
        L_j, j = heapq.heappop(heap)
        slot_assignments[j].append(pid_i)
        heapq.heappush(heap, (L_j + int(ns[pid_i]), j))

    T = max(int(ns[s].sum() if s else 0) for s in slot_assignments) if True else 0
    # Recompute T cleanly from the heap residuals:
    T = max(L_j for L_j, _ in heap)
    return slot_assignments, T


def compile_carousel_schedule(paths, slot_assignments, T, dim, phi0_per_pixel):
    """Build the (T, K) path tensors and sparse reset/record events.

    Concatenates each slot's pixel paths end-to-end into columns of
    ``dX_2d`` / ``rho_inv_2d``. Records the per-pixel harvest step (last
    step of that pixel's slice within its slot) and the per-pixel reset
    step (right after the prior pixel's harvest, except for the first
    pixel in a slot which is loaded directly into ``phi_initial``).

    Args:
        paths: list of ``(nsteps, dX_k, rho_inv_k, _grid_idcs)`` tuples,
            indexed by pixel id.
        slot_assignments: from :func:`schedule_lpt`.
        T: makespan from :func:`schedule_lpt`.
        dim: state dimension.
        phi0_per_pixel: ``(dim, K_total)`` array — per-pixel initial phi.

    Returns:
        dX_carousel: ``(T, K)`` f64 — slot-concatenated step sizes,
            zero-padded after each slot's total length.
        rho_inv_carousel: ``(T, K)`` f64 — slot-concatenated densities.
        phi_initial: ``(dim, K)`` f64 — first pixel's phi0 per slot.
        schedule: :class:`CarouselSchedule`.
    """
    K = len(slot_assignments)
    K_total = sum(len(s) for s in slot_assignments)

    dX_2d = np.zeros((T, K), dtype=np.float64)
    rho_inv_2d = np.zeros((T, K), dtype=np.float64)
    phi_initial = np.zeros((dim, K), dtype=np.float64)

    reset_per_t = [[] for _ in range(T)]
    record_per_t = [[] for _ in range(T)]

    for j, pixels in enumerate(slot_assignments):
        if not pixels:
            continue
        phi_initial[:, j] = phi0_per_pixel[:, pixels[0]]
        t_cursor = 0
        for i, pid in enumerate(pixels):
            ns_p, dX_p, ri_p, _ = paths[pid]
            if int(ns_p) != len(dX_p) or int(ns_p) != len(ri_p):
                raise ValueError(
                    f"compile_carousel_schedule: pixel {pid} path "
                    f"length mismatch (nsteps={ns_p}, len(dX)={len(dX_p)}, "
                    f"len(rho_inv)={len(ri_p)})"
                )
            dX_2d[t_cursor : t_cursor + ns_p, j] = dX_p
            rho_inv_2d[t_cursor : t_cursor + ns_p, j] = ri_p
            t_finish = t_cursor + ns_p - 1
            record_per_t[t_finish].append((j, pid))
            t_cursor += ns_p
            if i + 1 < len(pixels):
                reset_per_t[t_finish].append((j, pixels[i + 1]))

    reset_t_starts = np.zeros(T + 1, dtype=np.int32)
    record_t_starts = np.zeros(T + 1, dtype=np.int32)
    for t in range(T):
        reset_t_starts[t + 1] = reset_t_starts[t] + len(reset_per_t[t])
        record_t_starts[t + 1] = record_t_starts[t] + len(record_per_t[t])
    R = int(reset_t_starts[T])
    Rec = int(record_t_starts[T])
    reset_j = np.empty(R, dtype=np.int32)
    reset_pixel = np.empty(R, dtype=np.int32)
    record_j = np.empty(Rec, dtype=np.int32)
    record_pixel = np.empty(Rec, dtype=np.int32)
    r_c = 0
    rec_c = 0
    for t in range(T):
        for j, pid in reset_per_t[t]:
            reset_j[r_c] = j
            reset_pixel[r_c] = pid
            r_c += 1
        for j, pid in record_per_t[t]:
            record_j[rec_c] = j
            record_pixel[rec_c] = pid
            rec_c += 1

    schedule = CarouselSchedule(
        T=T,
        K=K,
        K_total=K_total,
        slot_assignments=slot_assignments,
        reset_t_starts=reset_t_starts,
        reset_j=reset_j,
        reset_pixel=reset_pixel,
        record_t_starts=record_t_starts,
        record_j=record_j,
        record_pixel=record_pixel,
    )
    return dX_2d, rho_inv_2d, phi_initial, schedule


# ---------------------------------------------------------------------------
# ρ-stack variant: per-step log-linear interpolation between two int_m slices
# ---------------------------------------------------------------------------
def _build_step_blend_indices(rho_inv, rho_grid):
    """Map each step's effective density to (lo_idx, weight) into ``rho_grid``.

    ``ρ_eff[k] = 1 / rho_inv[k]`` and the blend is log-linear:
    blended ≈ (1−w) · slice[lo] + w · slice[lo+1].

    Clamps to the stack endpoints when ρ_eff falls outside the grid range.
    """
    rho_eff = 1.0 / np.asarray(rho_inv, dtype=float)
    log_rho_eff = np.log10(np.clip(rho_eff, 1e-30, None))
    log_grid = np.log10(np.asarray(rho_grid, dtype=float))
    if not np.all(np.diff(log_grid) > 0):
        order = np.argsort(log_grid)
        log_grid = log_grid[order]
        # Caller must apply the same permutation to int_m_stack.
        return None  # signal — handled by caller
    n = len(log_grid)
    lo = np.searchsorted(log_grid, log_rho_eff, side="right") - 1
    lo = np.clip(lo, 0, n - 2)
    span = log_grid[lo + 1] - log_grid[lo]
    weight = np.where(span > 0, (log_rho_eff - log_grid[lo]) / span, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    return lo.astype(np.int64), weight.astype(np.float64)


def solv_numpy_etd2_rho_stack(
    nsteps, dX, rho_inv, int_m_stack, rho_grid, dec_m, phi, grid_idcs
):
    """ETD2RK with per-step log-linear blending of ``int_m`` slices.

    Variant of :func:`solv_numpy_etd2` for the LPM-density-stratified air
    pipeline.  At each step the effective interaction matrix is
    ``int_m_eff = (1-w) · int_m_stack[lo] + w · int_m_stack[lo+1]``
    where ``w`` is the log-linear weight from ρ_eff = 1/rho_inv to the
    bracketing ρ-grid slices.

    Memory: ``N_ρ × int_m`` (each slice carries its own (d_int, int_off,
    BSR-conv).  Per-step cost: 2× the standard kernel's off-diagonal SpMVs
    plus a 1-D blend on the diagonal.  Decay branch is ρ-invariant.

    Args:
      int_m_stack: list/tuple of N_ρ scipy sparse matrices, one per
        entry of ``rho_grid``.  All must share the same shape and
        sparsity-compatible structure.
      rho_grid: 1-D array of densities (g/cm³) corresponding to
        ``int_m_stack`` slices.  Must be strictly monotonic; the
        function clamps ρ_eff to the grid endpoints.
    """
    blocksize = getattr(config, "numpy_bsr_blocksize", None)
    n_slices = len(int_m_stack)
    if n_slices < 2:
        raise ValueError("rho_stack solver requires at least 2 slices.")

    # Per-slice splits + BSR conversion (memoised on the matrices themselves).
    slice_splits = [
        _etd_get_split_for_numpy(int_m_stack[i], dec_m, blocksize)
        for i in range(n_slices)
    ]
    # Each split: (d_int, d_dec, int_off, dec_off, n_padded).
    # d_dec, dec_off, n_padded are shared across slices (decay branch is
    # density-invariant); take them from slice 0.
    d_dec = slice_splits[0][1]
    dec_off = slice_splits[0][3]
    n_padded = slice_splits[0][4]
    d_ints = [sp[0] for sp in slice_splits]
    int_offs = [sp[2] for sp in slice_splits]

    # Pre-compute step → (lo_idx, weight) blend.
    log_rho = np.log10(np.clip(1.0 / np.asarray(rho_inv, dtype=float), 1e-30, None))
    log_grid = np.log10(np.asarray(rho_grid, dtype=float))
    if not np.all(np.diff(log_grid) > 0):
        order = np.argsort(log_grid)
        log_grid = log_grid[order]
        d_ints = [d_ints[i] for i in order]
        int_offs = [int_offs[i] for i in order]
    lo_idx = np.clip(
        np.searchsorted(log_grid, log_rho, side="right") - 1, 0, n_slices - 2
    )
    span = log_grid[lo_idx + 1] - log_grid[lo_idx]
    w_step = np.where(span > 0, (log_rho - log_grid[lo_idx]) / span, 0.0)
    w_step = np.clip(w_step, 0.0, 1.0)

    dim = phi.shape[0]
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

    phc_v = phc[:dim]
    F_phi_v = F_phi[:dim]
    F_a_v = F_a[:dim]
    a_v = a[:dim]

    # Scratch for the second-slice SpMV (off-diagonal blend).
    aux_off = np.empty(n_padded, dtype=np.float64)

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = dX[k]
            ri = rho_inv[k]
            i_lo = int(lo_idx[k])
            i_hi = i_lo + 1
            w = float(w_step[k])
            one_minus_w = 1.0 - w

            # Per-step diagonal blend
            d_int_eff = one_minus_w * d_ints[i_lo] + w * d_ints[i_hi]
            int_off_lo = int_offs[i_lo]
            int_off_hi = int_offs[i_hi]

            _etd_compute_diag_factors(h, ri, d_int_eff, d_dec, bufs)

            # F_phi = (1-w) * int_off_lo @ phc + w * int_off_hi @ phc + ri * dec_off @ phc
            np.copyto(F_phi, int_off_lo.dot(phc))
            F_phi *= one_minus_w
            np.copyto(aux_off, int_off_hi.dot(phc))
            aux_off *= w
            np.add(F_phi, aux_off, out=F_phi)
            ri_dec = dec_off.dot(phc)
            ri_dec *= ri
            np.add(F_phi, ri_dec, out=F_phi)

            # a = eD * phc + h * phi1 * F_phi
            np.multiply(eD, phc_v, out=a_v)
            np.multiply(phi1, F_phi_v, out=scratch)
            scratch *= h
            np.add(a_v, scratch, out=a_v)

            # F_a = blended off-diag @ a + ri * dec_off @ a
            np.copyto(F_a, int_off_lo.dot(a))
            F_a *= one_minus_w
            np.copyto(aux_off, int_off_hi.dot(a))
            aux_off *= w
            np.add(F_a, aux_off, out=F_a)
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
        f"Performance (ρ-stack): "
        f"{1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc_v.copy(), np.array(grid_sol)


def solv_numpy_etd2_rho_stack_multirhs(
    nsteps, dX, rho_inv, int_m_stack, rho_grid, dec_m, phi, grid_idcs
):
    """Multi-RHS variant of :func:`solv_numpy_etd2_rho_stack`.

    Same per-step log-linear blend of ``int_m_stack`` slices on the
    shared ``rho_inv`` array, but state is ``(dim, K)`` Fortran-flexible
    (scipy SpMM doesn't care about column ordering). Each column of
    ``phi`` carries an independent initial spectrum; the operator and
    its per-step diagonal/off-diagonal blend are shared across all K
    columns — because all K columns share the same atmosphere path
    (``rho_inv[k]``) in v1.

    Per-step cost: **two SpMMs per ``apply_F``** (one per bracketing
    slice), versus the non-stack kernel's single SpMM. So the ρ-stack
    multi-RHS path is roughly 2× slower per-step than the non-stack
    multi-RHS path, while still amortising over K columns.

    The Stage-3 extension (per-RHS atmosphere paths) lifts the
    ``rho_inv`` shape from ``(nsteps,)`` to ``(nsteps, K)`` and the
    blend index/weight from ``(nsteps,)`` to ``(nsteps, K)``. Then
    ``d_int_eff`` becomes ``(dim, K)`` and the per-step diagonal
    factors ``(eD, φ₁, φ₂)`` also become ``(dim, K)``. The K-column
    broadcast pattern in the present kernel makes that extension
    additive — see ``wiki/methods/multi-rhs-etd2-design.md`` Stage-3
    section for the full plan.

    Args:
      nsteps, dX, rho_inv, int_m_stack, rho_grid, dec_m:
        same as :func:`solv_numpy_etd2_rho_stack`.
      phi (np.ndarray[dim, K]): initial states; one column per RHS.
      grid_idcs (list[int]): step indices at which to record snapshots.

    Returns:
      (np.ndarray[dim, K], np.ndarray[len(grid_idcs), dim, K]): final
      state matrix and stacked snapshots.
    """
    if phi.ndim != 2:
        raise ValueError(
            "solv_numpy_etd2_rho_stack_multirhs: phi must be 2-D (dim, K), "
            f"got shape {phi.shape}"
        )
    dim, K = phi.shape
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    n_slices = len(int_m_stack)
    if n_slices < 2:
        raise ValueError("rho_stack solver requires at least 2 slices.")

    # CSR off-diagonals (see :func:`solv_numpy_etd2_multirhs` for the
    # CSR-vs-BSR-on-2D-RHS rationale: BSR `@ X` is a sequential
    # block-SpMV loop with no K-axis vectorisation; CSR `@ X` is a true
    # SpMM and wins at K ≥ 8). The single-RHS ρ-stack memoises BSR-form
    # splits on each slice via ``_etd_get_split_for_numpy``; here we go
    # through ``_etd_split_cache`` directly to get CSR splits, then
    # cache them on the slice instances under a different attribute so
    # the multi-RHS and single-RHS caches don't fight.
    splits = []
    for i, slice_m in enumerate(int_m_stack):
        cache = getattr(slice_m, "_etd_split_cache_csr_multirhs", None)
        if cache is not None and cache["dec_m_id"] == id(dec_m):
            splits.append(cache["split"])
            continue
        d_int_i, d_dec_i, int_off_i, dec_off_i = _etd_split_cache(slice_m, dec_m)
        if not sp.isspmatrix_csr(int_off_i):
            int_off_i = int_off_i.tocsr()
        if not sp.isspmatrix_csr(dec_off_i):
            dec_off_i = dec_off_i.tocsr()
        split = (d_int_i, d_dec_i, int_off_i, dec_off_i)
        splits.append(split)
        try:
            slice_m._etd_split_cache_csr_multirhs = {
                "dec_m_id": id(dec_m),
                "split": split,
            }
        except (AttributeError, TypeError):
            pass

    # Decay branch is ρ-invariant; take its split from slice 0.
    d_dec = splits[0][1]
    dec_off = splits[0][3]
    d_ints = [s[0] for s in splits]
    int_offs = [s[2] for s in splits]

    # Pre-compute step → (lo_idx, weight) blend. Same logic as
    # the single-RHS ρ-stack kernel; the blend is per-step (shared
    # across K columns) because rho_inv is shared in v1.
    log_rho = np.log10(np.clip(1.0 / np.asarray(rho_inv, dtype=float), 1e-30, None))
    log_grid = np.log10(np.asarray(rho_grid, dtype=float))
    if not np.all(np.diff(log_grid) > 0):
        order = np.argsort(log_grid)
        log_grid = log_grid[order]
        d_ints = [d_ints[i] for i in order]
        int_offs = [int_offs[i] for i in order]
    lo_idx = np.clip(
        np.searchsorted(log_grid, log_rho, side="right") - 1, 0, n_slices - 2
    )
    span = log_grid[lo_idx + 1] - log_grid[lo_idx]
    w_step = np.where(span > 0, (log_rho - log_grid[lo_idx]) / span, 0.0)
    w_step = np.clip(w_step, 0.0, 1.0)

    # (dim, K) state-shape buffers.
    phc = np.array(phi, dtype=np.float64, copy=True)
    F_phi = np.empty((dim, K), dtype=np.float64)
    F_a = np.empty((dim, K), dtype=np.float64)
    a = np.empty((dim, K), dtype=np.float64)
    aux_off = np.empty((dim, K), dtype=np.float64)
    bufs = _etd_step_buffers(dim)
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch_NK = np.empty((dim, K), dtype=np.float64)

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = dX[k]
            ri = rho_inv[k]
            i_lo = int(lo_idx[k])
            i_hi = i_lo + 1
            w = float(w_step[k])
            one_minus_w = 1.0 - w

            # Per-step diagonal blend — same arithmetic as the single-RHS path.
            d_int_eff = one_minus_w * d_ints[i_lo] + w * d_ints[i_hi]
            int_off_lo = int_offs[i_lo]
            int_off_hi = int_offs[i_hi]

            _etd_compute_diag_factors(h, ri, d_int_eff, d_dec, bufs)

            # F_phi = (1-w) * int_off_lo @ phc + w * int_off_hi @ phc
            #         + ri * dec_off @ phc
            np.copyto(F_phi, int_off_lo.dot(phc))
            F_phi *= one_minus_w
            np.copyto(aux_off, int_off_hi.dot(phc))
            aux_off *= w
            np.add(F_phi, aux_off, out=F_phi)
            ri_dec = dec_off.dot(phc)
            ri_dec *= ri
            np.add(F_phi, ri_dec, out=F_phi)

            # a = eD[:, None] * phc + h * phi1[:, None] * F_phi
            np.multiply(eD[:, None], phc, out=a)
            np.multiply(phi1[:, None], F_phi, out=scratch_NK)
            scratch_NK *= h
            np.add(a, scratch_NK, out=a)

            # F_a = blended off-diag @ a + ri * dec_off @ a
            np.copyto(F_a, int_off_lo.dot(a))
            F_a *= one_minus_w
            np.copyto(aux_off, int_off_hi.dot(a))
            aux_off *= w
            np.add(F_a, aux_off, out=F_a)
            ri_dec = dec_off.dot(a)
            ri_dec *= ri
            np.add(F_a, ri_dec, out=F_a)

            # phc = a + h * phi2[:, None] * (F_a - F_phi)
            np.subtract(F_a, F_phi, out=scratch_NK)
            scratch_NK *= h
            np.multiply(phi2[:, None], scratch_NK, out=scratch_NK)
            np.add(a, scratch_NK, out=phc)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc))
                grid_step += 1

    info(
        2,
        f"Performance (ρ-stack multirhs K={K}): "
        f"{1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * (time() - start) / float(nsteps) / float(K):6.2f}ms/iteration/RHS)",
    )

    return phc.copy(), np.array(grid_sol)


# ---------------------------------------------------------------------------
# MKL ETD2 kernel
# ---------------------------------------------------------------------------
class _MklMatrixDescr:
    """Shared ``struct matrix_descr`` ctypes wrapper for MKL sparse calls.

    Has to be a module-level class so the argtypes-registered version and
    the per-wrapper instance reference the same Python type — ctypes does
    strict isinstance() checking on Structure argtypes.
    """


def _build_mkl_descr_class():
    from ctypes import Structure, c_int

    class _MatrixDescr(Structure):
        _fields_ = [("type", c_int), ("mode", c_int), ("diag", c_int)]

    return _MatrixDescr


_MklMatrixDescr = _build_mkl_descr_class()


_MKL_ARGTYPES_SET = False


def _set_mkl_argtypes(mkl):
    """Register ctypes argtypes/restype on the MKL functions we call.

    Without explicit argtypes, Python ctypes inspects the value passed at
    each call site to decide how to marshal it. That breaks when callers
    pass ``c_double.from_address(addr)`` where ``POINTER(c_double)`` is
    expected — ctypes marshals the c_double by value (not as a pointer)
    and MKL rejects the call with ``SPARSE_STATUS_INVALID_VALUE``.
    Setting argtypes once after the library load fixes the marshalling.
    """
    global _MKL_ARGTYPES_SET
    if _MKL_ARGTYPES_SET:
        return
    from ctypes import POINTER, c_int, c_void_p
    from ctypes import c_double as fl64
    from ctypes import c_float as fl32

    # fp64
    mkl.mkl_sparse_d_mv.argtypes = [
        c_int,
        fl64,
        c_void_p,
        _MklMatrixDescr,
        POINTER(fl64),
        fl64,
        POINTER(fl64),
    ]
    mkl.mkl_sparse_d_mv.restype = c_int
    mkl.mkl_sparse_d_mm.argtypes = [
        c_int,
        fl64,
        c_void_p,
        _MklMatrixDescr,
        c_int,
        POINTER(fl64),
        c_int,
        c_int,
        fl64,
        POINTER(fl64),
        c_int,
    ]
    mkl.mkl_sparse_d_mm.restype = c_int
    # fp32
    if hasattr(mkl, "mkl_sparse_s_mv"):
        mkl.mkl_sparse_s_mv.argtypes = [
            c_int,
            fl32,
            c_void_p,
            _MklMatrixDescr,
            POINTER(fl32),
            fl32,
            POINTER(fl32),
        ]
        mkl.mkl_sparse_s_mv.restype = c_int
        mkl.mkl_sparse_s_mm.argtypes = [
            c_int,
            fl32,
            c_void_p,
            _MklMatrixDescr,
            c_int,
            POINTER(fl32),
            c_int,
            c_int,
            fl32,
            POINTER(fl32),
            c_int,
        ]
        mkl.mkl_sparse_s_mm.restype = c_int

    _MKL_ARGTYPES_SET = True


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
        from ctypes import POINTER, byref, c_int, c_void_p
        from ctypes import c_double as fl_pr

        config._load_mkl()
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrix expects a CSR matrix, got {type(csr).__name__}"
            )
        if csr.dtype != np.float64:
            raise TypeError(f"MklSparseMatrix expects float64 data, got {csr.dtype}")

        n_orig = csr.shape[0]
        self.n_orig = n_orig
        self.n_cols = csr.shape[1]
        self.blocksize = blocksize

        mkl = config.mkl
        self._mkl = mkl
        _set_mkl_argtypes(mkl)

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
                c_int(self.n_cols),
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
            if csr.shape[0] != csr.shape[1]:
                raise ValueError(
                    "BSR mode requires a square matrix; rectangular "
                    "operators are CSR-only (blocksize=None)"
                )
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

        descr = _MklMatrixDescr()
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

    def gemm_ctargs(self, alpha, nrhs, B_p, ldb, C_p, ldc, beta=1.0, layout=102):
        """``C = alpha * A * B + beta * C`` via raw c_double pointers.

        Wraps ``mkl_sparse_d_mm``. The default column-major layout
        (``layout=102``) serves the (dim, K) Fortran-contiguous buffers
        of the multi-RHS / multipath kernels without transpose;
        ``layout=101`` (row-major, ``ldb``/``ldc`` = K) serves the
        C-contiguous buffers of the secant batch driver. ``ldb`` and
        ``ldc`` are the leading dimensions; per-tile callers offset the
        pointer instead.

        Default ``beta = 1.0`` matches :class:`MCEq.spacc.SpaccMatrix.gemm_ctargs`
        (accumulating SpMM). Caller is responsible for zeroing ``C`` before
        the first call in an accumulator chain.
        """
        from ctypes import c_double as fl_pr
        from ctypes import c_int

        # SPARSE_LAYOUT_COLUMN_MAJOR = 102, SPARSE_LAYOUT_ROW_MAJOR = 101.
        # Operation enum (10 = non-transpose) comes from self._operation,
        # set in __init__.
        st = self._mkl.mkl_sparse_d_mm(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            c_int(int(layout)),
            B_p,
            c_int(int(nrhs)),
            c_int(int(ldb)),
            fl_pr(beta),
            C_p,
            c_int(int(ldc)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_d_mm failed with status {st}")

    def set_mm_hint(self, nrhs, expected_calls=200, layout=102):
        """Tell MKL the SpMM-specific shape so it can re-plan.

        ``mkl_sparse_set_mm_hint`` accepts the layout, op, descr, ncols, and
        expected call count; if the layout/ncols differs from the SpMV hint
        registered at construction, the optimiser can pick a different
        kernel. Followed by another ``mkl_sparse_optimize``. Optional —
        callers can skip if the SpMV hint is already adequate.
        """
        from ctypes import c_int

        # SPARSE_LAYOUT_COLUMN_MAJOR = 102, SPARSE_LAYOUT_ROW_MAJOR = 101.
        st = self._mkl.mkl_sparse_set_mm_hint(
            self._handle,
            self._operation,
            self._descr,
            c_int(int(layout)),
            c_int(int(nrhs)),
            c_int(int(expected_calls)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mm_hint failed with status {st}")
        st = self._mkl.mkl_sparse_optimize(self._handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def close(self):
        """Free the underlying MKL sparse handle.

        Idempotent — safe to call repeatedly. Prefer this over
        ``del`` or relying on refcount-driven ``__del__`` when the
        backend cache of ``MCEqRun`` rotates; the call below the C
        boundary returns the MKL-internal optimised layout memory, not
        just the Python wrapper.
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


class MklSparseMatrixF32:
    """fp32 sibling of :class:`MklSparseMatrix` for the multi-RHS f32 path.

    Wraps an Intel MKL fp32 sparse-matrix handle. The fp32 ETD2 multirhs
    kernel needs the SpMM to produce fp32 directly (avoids the (dim, K) cast
    that would otherwise dominate at K ≥ 64); MKL's ``mkl_sparse_s_*``
    functions are the standard route.

    Currently CSR-only — BSR was a 1.5× win at fp64 on SIBYLL21 but the BSR
    block-microkernels MKL ships are fp64-only on most builds; CSR is the
    safe default. Revisit if benches motivate it.

    Args:
      csr (scipy.sparse.csr_matrix): float32 OR float64 CSR matrix. fp64
        input is cast down once at construction; downstream the SpMM runs
        purely in fp32.
      expected_calls (int): SpMV count hint for MKL planning.
    """

    def __init__(self, csr, expected_calls=200):
        from ctypes import POINTER, byref, c_int, c_void_p
        from ctypes import c_float as fl_pr

        config._load_mkl()
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrixF32 expects a CSR matrix, got {type(csr).__name__}"
            )

        n_orig = csr.shape[0]
        self.n_orig = n_orig
        self.n_padded = n_orig
        self.blocksize = None

        mkl = config.mkl
        self._mkl = mkl
        _set_mkl_argtypes(mkl)

        indices = csr.indices.astype(np.int32, copy=False)
        indptr = csr.indptr.astype(np.int32, copy=False)
        data = csr.data.astype(np.float32, copy=False)
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self.nnz = csr.nnz

        handle = c_void_p()
        data_p = data.ctypes.data_as(POINTER(fl_pr))
        ci_p = indices.ctypes.data_as(POINTER(c_int))
        pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
        pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))

        st = mkl.mkl_sparse_s_create_csr(
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
            raise RuntimeError(f"mkl_sparse_s_create_csr failed with status {st}")
        self._handle = handle

        descr = _MklMatrixDescr()
        descr.type = c_int(20)
        descr.mode = c_int(121)
        descr.diag = c_int(131)
        self._descr = descr
        self._operation = c_int(10)

        st = mkl.mkl_sparse_set_mv_hint(
            handle, self._operation, descr, c_int(int(expected_calls))
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mv_hint failed with status {st}")
        st = mkl.mkl_sparse_optimize(handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def gemv_ctargs(self, alpha, x_p, beta, y_p):
        from ctypes import c_float as fl_pr

        st = self._mkl.mkl_sparse_s_mv(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            x_p,
            fl_pr(beta),
            y_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_s_mv failed with status {st}")

    def gemm_ctargs(self, alpha, nrhs, B_p, ldb, C_p, ldc, beta=1.0):
        """``C = alpha * A * B + beta * C`` via raw c_float pointers."""
        from ctypes import c_float as fl_pr
        from ctypes import c_int

        st = self._mkl.mkl_sparse_s_mm(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            c_int(102),
            B_p,
            c_int(int(nrhs)),
            c_int(int(ldb)),
            fl_pr(beta),
            C_p,
            c_int(int(ldc)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_s_mm failed with status {st}")

    def set_mm_hint(self, nrhs, expected_calls=200):
        from ctypes import c_int

        st = self._mkl.mkl_sparse_set_mm_hint(
            self._handle,
            self._operation,
            self._descr,
            c_int(102),
            c_int(int(nrhs)),
            c_int(int(expected_calls)),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_set_mm_hint failed with status {st}")
        st = self._mkl.mkl_sparse_optimize(self._handle)
        if st != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")

    def close(self):
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
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CUDA runtime helpers
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
    """
    phi_compute = cp.ElementwiseKernel(
        "T hD, T eD_in",
        "T eD_out, T phi1, T phi2",
        f"""
        T abs_hd = (hD >= T(0)) ? hD : -hD;
        eD_out = eD_in;
        if (abs_hd > T({_PHI1_SMALL!r})) {{
            phi1 = (eD_in - T(1)) / hD;
        }} else {{
            phi1 = T(1) + hD * (T(0.5) + hD * T({_INV_6!r}));
        }}
        if (abs_hd > T({_PHI2_SMALL!r})) {{
            phi2 = (eD_in - T(1) - hD) / (hD * hD);
        }} else {{
            phi2 = T(0.5) + hD * (T({_INV_6!r}) + hD * T({_INV_24!r}));
        }}
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
        "T d_int, T d_dec, T h, T ri",
        "T eD, T phi1, T phi2",
        f"""
        T D = d_int + ri * d_dec;
        T hd = h * D;
        T e = exp(hd);
        eD = e;
        T abs_hd = (hd >= T(0)) ? hd : -hd;
        if (abs_hd > T({_PHI1_SMALL!r})) {{
            phi1 = (e - T(1)) / hd;
        }} else {{
            phi1 = T(1) + hd * (T(0.5) + hd * T({_INV_6!r}));
        }}
        if (abs_hd > T({_PHI2_SMALL!r})) {{
            phi2 = (e - T(1) - hd) / (hd * hd);
        }} else {{
            phi2 = T(0.5) + hd * (T({_INV_6!r}) + hd * T({_INV_24!r}));
        }}
        """,
        "mceq_etd2_phi_compute_multipath",
    )
    # fp32-pipeline diag-factor kernel: float32 buffers in/out, double
    # arithmetic inside. The phi1/phi2 cancellations ((e-1)/hd and
    # (e-1-hd)/hd^2) lose 3-7 digits in fp32 around the Taylor-switch
    # thresholds (which are calibrated for fp64 rounding: fp32 phi1 err
    # up to ~8e-3, phi2 up to ~1e-1 on real MCEq diagonals); promoting
    # the kernel-internal math to double brings all three factors to
    # fp32 roundoff (~6e-8) for ~5% step cost on an RTX 3090 — the
    # SpMMs dominate the step. Serves both the shared-path (scalar
    # h/ri) and the carousel ((1, K) row) call sites via broadcasting.
    phi_compute_multipath_f64diag = cp.ElementwiseKernel(
        "float32 d_int, float32 d_dec, float32 h, float32 ri",
        "float32 eD, float32 phi1, float32 phi2",
        f"""
        double D = (double)d_int + (double)ri * (double)d_dec;
        double hd = (double)h * D;
        double e = exp(hd);
        eD = (float)e;
        double abs_hd = (hd >= 0.0) ? hd : -hd;
        double p1, p2;
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
        phi1 = (float)p1;
        phi2 = (float)p2;
        """,
        "mceq_etd2_phi_compute_multipath_f64diag",
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
        phi_compute_multipath_f64diag=phi_compute_multipath_f64diag,
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


# --------------------------------------------------------------------
# MKL Sparse BLAS fp32 multi-RHS kernel
#
# The fp64 MKL routes run on :func:`etd2_driver` with :class:`MklApplyOff`.
# This fp32 kernel (column-major, tiled SpMM, ``MCEq.etd2_kernels`` C
# post-apply) stays until the host backend carries an fp32 state.
# --------------------------------------------------------------------

# Default K-tile for the MKL Sparse BLAS SpMM call. MKL's
# ``mkl_sparse_d_mm`` should scale better than Accelerate's beyond
# K = 64 (the EPYC AVX2/AVX-512 microkernels stay cache-friendly for
# larger K), but we keep the same tiling shape for parity with the
# Accelerate kernel — micro-bench in :doc:`runs/2026-05-23_multirhs-satori-gpu`
# can tune this if needed.
_MKL_SPMM_TILE = 64


def solv_mkl_etd2_multirhs_f32(
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
    """fp32 sibling of :func:`solv_mkl_etd2_multirhs`.

    State + SpMM in fp32 via :class:`MklSparseMatrixF32`. The diagonal
    pipeline still runs fp64 (exp(h·D) saturates fp32 quickly at high
    zenith); eD/phi1/phi2 are cast down once per step for the post-apply.
    """
    from ctypes import POINTER, c_float, sizeof

    if phi.ndim != 2:
        raise ValueError(
            f"solv_mkl_etd2_multirhs_f32: phi must be 2-D (dim, K), "
            f"got shape {phi.shape}"
        )
    dim, K = phi.shape
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    n_padded = dim
    for m in (mkl_int_off, mkl_dec_off):
        if m is not None:
            n_padded = max(n_padded, m.n_padded)

    tile = _MKL_SPMM_TILE
    tile = max(1, min(int(tile), K))

    phc = np.zeros((n_padded, K), dtype=np.float32, order="F")
    phc[:dim, :] = phi.astype(np.float32, copy=False)
    F_phi = np.zeros((n_padded, K), dtype=np.float32, order="F")
    F_a = np.zeros((n_padded, K), dtype=np.float32, order="F")
    a = np.zeros((n_padded, K), dtype=np.float32, order="F")

    bufs = _etd_step_buffers(dim)
    eD_f64 = bufs["eD"]
    phi1_f64 = bufs["phi1"]
    phi2_f64 = bufs["phi2"]
    eD_f32 = np.zeros(n_padded, dtype=np.float32)
    phi1_f32 = np.zeros(n_padded, dtype=np.float32)
    phi2_f32 = np.zeros(n_padded, dtype=np.float32)

    flt = sizeof(c_float)
    tile_starts = list(range(0, K, tile))
    tile_widths = [min(tile, K - c0) for c0 in tile_starts]
    n_tiles = len(tile_starts)

    phc_addr = phc.ctypes.data
    F_phi_addr = F_phi.ctypes.data
    F_a_addr = F_a.ctypes.data
    a_addr = a.ctypes.data

    def _ptrs_at(addr, c0):
        return c_float.from_address(addr + c0 * n_padded * flt)

    phc_tile_ptrs = [_ptrs_at(phc_addr, c0) for c0 in tile_starts]
    F_phi_tile_ptrs = [_ptrs_at(F_phi_addr, c0) for c0 in tile_starts]
    F_a_tile_ptrs = [_ptrs_at(F_a_addr, c0) for c0 in tile_starts]
    a_tile_ptrs = [_ptrs_at(a_addr, c0) for c0 in tile_starts]

    phc_p_full = phc.ctypes.data_as(POINTER(c_float))
    F_phi_p_full = F_phi.ctypes.data_as(POINTER(c_float))
    F_a_p_full = F_a.ctypes.data_as(POINTER(c_float))
    a_p_full = a.ctypes.data_as(POINTER(c_float))
    eD_p = eD_f32.ctypes.data_as(POINTER(c_float))
    phi1_p = phi1_f32.ctypes.data_as(POINTER(c_float))
    phi2_p = phi2_f32.ctypes.data_as(POINTER(c_float))

    from MCEq.etd2_kernels import etd2_post_apply1_multirhs_f32 as _post1
    from MCEq.etd2_kernels import etd2_post_apply2_multirhs_f32 as _post2

    int_off_empty = (mkl_int_off is None) or (mkl_int_off.nnz == 0)
    dec_off_empty = (mkl_dec_off is None) or (mkl_dec_off.nnz == 0)

    primary_nrhs = tile_widths[0]
    if not int_off_empty:
        mkl_int_off.set_mm_hint(primary_nrhs, expected_calls=2 * nsteps * n_tiles)
    if not dec_off_empty:
        mkl_dec_off.set_mm_hint(primary_nrhs, expected_calls=2 * nsteps * n_tiles)

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = float(dX[k])
            ri = float(rho_inv[k])

            _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs)
            eD_f32[:dim] = eD_f64
            phi1_f32[:dim] = phi1_f64
            phi2_f32[:dim] = phi2_f64

            F_phi.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    mkl_int_off.gemm_ctargs(
                        1.0,
                        nrhs,
                        phc_tile_ptrs[t],
                        n_padded,
                        F_phi_tile_ptrs[t],
                        n_padded,
                        beta=1.0,
                    )
                if not dec_off_empty:
                    mkl_dec_off.gemm_ctargs(
                        ri,
                        nrhs,
                        phc_tile_ptrs[t],
                        n_padded,
                        F_phi_tile_ptrs[t],
                        n_padded,
                        beta=1.0,
                    )

            _post1(n_padded, K, h, eD_p, phi1_p, phc_p_full, F_phi_p_full, a_p_full)

            F_a.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    mkl_int_off.gemm_ctargs(
                        1.0,
                        nrhs,
                        a_tile_ptrs[t],
                        n_padded,
                        F_a_tile_ptrs[t],
                        n_padded,
                        beta=1.0,
                    )
                if not dec_off_empty:
                    mkl_dec_off.gemm_ctargs(
                        ri,
                        nrhs,
                        a_tile_ptrs[t],
                        n_padded,
                        F_a_tile_ptrs[t],
                        n_padded,
                        beta=1.0,
                    )

            _post2(
                n_padded, K, h, phi2_p, a_p_full, F_a_p_full, F_phi_p_full, phc_p_full
            )

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc[:dim, :]))
                grid_step += 1

    elapsed = time() - start
    info(
        2,
        f"Performance (mkl multirhs f32 K={K}): "
        f"{1e3 * elapsed / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * elapsed / float(nsteps) / float(K):6.2f}ms/iteration/RHS)",
    )

    return phc[:dim, :].copy(order="F"), np.array(grid_sol)


def solv_spacc_etd2_multirhs(
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
    """ETD2RK on Apple Accelerate Sparse BLAS — multi-RHS variant.

    Same Cox–Matthews update as :func:`solv_spacc_etd2` and
    :func:`solv_numpy_etd2_multirhs`, but the four per-step SpMVs are
    promoted to SpMMs through Accelerate's
    ``sparse_matrix_product_dense_double`` (wrapped as
    :meth:`MCEq.spacc.SpaccMatrix.gemm_ctargs`).

    State layout is column-major ``(dim, K)`` Fortran-contiguous; Apple
    Accelerate's SpMM API expects column-major dense buffers and walks
    columns with leading dimension ``ldB = ldC = dim``.

    The Accelerate sparse handle re-optimises the matrix layout on
    construction (see :class:`MCEq.spacc.SpaccMatrix`); reusing the same
    handle across all SpMVs/SpMMs in a solve is what amortises that cost.
    Pre-existing handles are reused via ``MCEqRun``'s backend cache.

    Args:
      nsteps, dX, rho_inv: same as :func:`solv_spacc_etd2`.
      spacc_int_off, spacc_dec_off: :class:`SpaccMatrix` wrappers; may be
        ``None`` if the underlying off-diagonal has zero nnz.
      d_int, d_dec (np.ndarray): diagonals (length ``dim``).
      phi (np.ndarray[dim, K]): initial states, column-major (Fortran-
        contiguous). The function ``np.asfortranarray``s the input if
        needed.
      grid_idcs (list[int]): step indices at which to record snapshots.

    Returns:
      (np.ndarray[dim, K], np.ndarray[len(grid_idcs), dim, K]): final
      state matrix and stacked snapshots; the final state is a copy
      (column-major).
    """
    from ctypes import POINTER, c_double, sizeof

    if phi.ndim != 2:
        raise ValueError(
            f"solv_spacc_etd2_multirhs: phi must be 2-D (dim, K), got shape {phi.shape}"
        )
    dim, K = phi.shape
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    # K-tile size for the Accelerate SpMM call. The bench at
    # runs/2026-05-21_multi-rhs-etd2-prototype shows per-RHS throughput peaking
    # at K ≈ 32–64 (3.0–3.2× /RHS) then falling off a cliff at K ≥ 128 (≈ 1.4×).
    # Hypothesis: Accelerate's internal SpMM tile sizing for
    # ``sparse_matrix_product_dense_double`` does not stay cache-friendly past
    # ~64 columns. Tiling the call into chunks of ``_SPACC_SPMM_TILE`` columns
    # restores the peak operating point for any caller-requested K. Per-step
    # buffers stay (dim, K); only the SpMM call site is tiled. Setting tile
    # ≥ K disables tiling (single call, original behaviour).
    tile = _SPACC_SPMM_TILE
    tile = max(1, min(int(tile), K))

    # Persistent column-major buffers — gemm reads/writes through raw
    # ctypes pointers, so the backing storage must keep its address
    # across the loop. Fortran-order arrays are column-major with leading
    # dimension == number of rows.
    phc = np.asfortranarray(phi.astype(np.float64, copy=True))
    F_phi = np.zeros((dim, K), dtype=np.float64, order="F")
    F_a = np.zeros((dim, K), dtype=np.float64, order="F")
    a = np.empty((dim, K), dtype=np.float64, order="F")
    bufs = _etd_step_buffers(dim)
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    # scratch_NK was the elementwise scratch for the un-fused ufunc post-apply
    # chains; the fused C kernels write directly into ``a`` / ``phc``, so we
    # no longer need a separate (dim, K) scratch buffer here.

    # Precompute per-tile pointer offsets. Fortran-contiguous (dim, K) has
    # column-major layout — column c starts at byte ``c * dim * sizeof(double)``
    # from the buffer base. We materialise the per-tile pointer-arithmetic
    # offsets once outside the hot loop so the per-step inner loop is just
    # ctypes addition + SpMM dispatch.
    dbl = sizeof(c_double)
    tile_starts = list(range(0, K, tile))
    tile_widths = [min(tile, K - c0) for c0 in tile_starts]
    n_tiles = len(tile_starts)

    phc_addr = phc.ctypes.data
    F_phi_addr = F_phi.ctypes.data
    F_a_addr = F_a.ctypes.data
    a_addr = a.ctypes.data

    def _ptrs_at(addr, c0):
        return c_double.from_address(addr + c0 * dim * dbl)

    # Pre-built per-tile pointer pairs. ``c_double.from_address(addr)`` returns
    # a ctypes scalar referencing the byte at ``addr``; passing it where the
    # binding expects ``POINTER(c_double)`` lets ctypes auto-box it (verified
    # against the existing gemm binding in MCEq.spacc).
    phc_tile_ptrs = [_ptrs_at(phc_addr, c0) for c0 in tile_starts]
    F_phi_tile_ptrs = [_ptrs_at(F_phi_addr, c0) for c0 in tile_starts]
    F_a_tile_ptrs = [_ptrs_at(F_a_addr, c0) for c0 in tile_starts]
    a_tile_ptrs = [_ptrs_at(a_addr, c0) for c0 in tile_starts]

    # Whole-buffer pointers for the fused post-apply C kernels — those
    # operate on the whole (dim, K) block, not per-tile.
    phc_p_full = phc.ctypes.data_as(POINTER(c_double))
    F_phi_p_full = F_phi.ctypes.data_as(POINTER(c_double))
    F_a_p_full = F_a.ctypes.data_as(POINTER(c_double))
    a_p_full = a.ctypes.data_as(POINTER(c_double))
    eD_p = eD.ctypes.data_as(POINTER(c_double))
    phi1_p = phi1.ctypes.data_as(POINTER(c_double))
    phi2_p = phi2.ctypes.data_as(POINTER(c_double))

    # Fused post-apply C kernels. Replace the 4-ufunc post_apply chains
    # with one stride-1 pass per stage — see :mod:`MCEq.spacc.spacc.c`
    # ``etd2_post_apply{1,2}_multirhs``.
    from MCEq.spacc import etd2_post_apply1_multirhs as _post1
    from MCEq.spacc import etd2_post_apply2_multirhs as _post2

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

            # F_phi = int_off @ phc + ri * dec_off @ phc  (accumulating SpMM)
            # K-tile: each call processes a contiguous slice of ``tile``
            # columns. The accumulating semantics of gemm(C += α·A·B) work
            # per-tile against the same fresh-zero F_phi buffer.
            F_phi.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, phc_tile_ptrs[t], dim, F_phi_tile_ptrs[t], dim
                    )
                if not dec_off_empty:
                    spacc_dec_off.gemm_ctargs(
                        ri, nrhs, phc_tile_ptrs[t], dim, F_phi_tile_ptrs[t], dim
                    )

            # a = eD[:, None] * phc + h * phi1[:, None] * F_phi  (fused)
            _post1(dim, K, h, eD_p, phi1_p, phc_p_full, F_phi_p_full, a_p_full)

            # F_a = int_off @ a + ri * dec_off @ a
            F_a.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, a_tile_ptrs[t], dim, F_a_tile_ptrs[t], dim
                    )
                if not dec_off_empty:
                    spacc_dec_off.gemm_ctargs(
                        ri, nrhs, a_tile_ptrs[t], dim, F_a_tile_ptrs[t], dim
                    )

            # phc = a + h * phi2[:, None] * (F_a - F_phi)  (fused)
            _post2(dim, K, h, phi2_p, a_p_full, F_a_p_full, F_phi_p_full, phc_p_full)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc))
                grid_step += 1

    info(
        2,
        f"Performance (spacc multirhs K={K}): "
        f"{1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * (time() - start) / float(nsteps) / float(K):6.2f}ms/iteration/RHS)",
    )

    return phc.copy(), np.array(grid_sol)


def solv_spacc_etd2_multirhs_f32(
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
    """fp32 sibling of :func:`solv_spacc_etd2_multirhs`.

    Same ETD2 Cox–Matthews update; all state and per-step buffers live
    in float32. The sparse off-diagonals come in as
    :class:`MCEq.spacc.SpaccMatrixF32` wrappers (the caller is
    responsible for constructing fp32 handles via
    ``sparse_matrix_create_float``). The diagonal vectors ``d_int`` /
    ``d_dec`` are still computed in fp64 by the caller (the diag-factor
    pipeline needs fp64 because ``exp(h·D)`` saturates fp32 fast at
    high zenith); we cast them to fp32 only for the final multiplication
    against the fp32 state buffers.

    Precision budget on real SIBYLL21 vs the fp64 reference is verified
    in ``test_solv_spacc_etd2_multirhs_f32_stability``: per-particle
    relative error stays below 1e-4 across all species.
    """
    from ctypes import POINTER, c_float, sizeof

    if phi.ndim != 2:
        raise ValueError(
            f"solv_spacc_etd2_multirhs_f32: phi must be 2-D (dim, K), "
            f"got shape {phi.shape}"
        )
    dim, K = phi.shape
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    tile = _SPACC_SPMM_TILE
    tile = max(1, min(int(tile), K))

    # fp32 column-major buffers.
    phc = np.asfortranarray(phi.astype(np.float32, copy=True))
    F_phi = np.zeros((dim, K), dtype=np.float32, order="F")
    F_a = np.zeros((dim, K), dtype=np.float32, order="F")
    a = np.empty((dim, K), dtype=np.float32, order="F")

    # Diag-factor pipeline runs in fp64 against the fp64 ``d_int``/``d_dec``;
    # we cast eD/phi1/phi2 to fp32 once per step for the fused post-apply.
    bufs = _etd_step_buffers(dim)
    eD_f64 = bufs["eD"]
    phi1_f64 = bufs["phi1"]
    phi2_f64 = bufs["phi2"]
    eD_f32 = np.empty(dim, dtype=np.float32)
    phi1_f32 = np.empty(dim, dtype=np.float32)
    phi2_f32 = np.empty(dim, dtype=np.float32)

    flt = sizeof(c_float)
    tile_starts = list(range(0, K, tile))
    tile_widths = [min(tile, K - c0) for c0 in tile_starts]
    n_tiles = len(tile_starts)

    phc_addr = phc.ctypes.data
    F_phi_addr = F_phi.ctypes.data
    F_a_addr = F_a.ctypes.data
    a_addr = a.ctypes.data

    def _ptrs_at(addr, c0):
        return c_float.from_address(addr + c0 * dim * flt)

    phc_tile_ptrs = [_ptrs_at(phc_addr, c0) for c0 in tile_starts]
    F_phi_tile_ptrs = [_ptrs_at(F_phi_addr, c0) for c0 in tile_starts]
    F_a_tile_ptrs = [_ptrs_at(F_a_addr, c0) for c0 in tile_starts]
    a_tile_ptrs = [_ptrs_at(a_addr, c0) for c0 in tile_starts]

    phc_p_full = phc.ctypes.data_as(POINTER(c_float))
    F_phi_p_full = F_phi.ctypes.data_as(POINTER(c_float))
    F_a_p_full = F_a.ctypes.data_as(POINTER(c_float))
    a_p_full = a.ctypes.data_as(POINTER(c_float))
    eD_p = eD_f32.ctypes.data_as(POINTER(c_float))
    phi1_p = phi1_f32.ctypes.data_as(POINTER(c_float))
    phi2_p = phi2_f32.ctypes.data_as(POINTER(c_float))

    from MCEq.spacc import etd2_post_apply1_multirhs_f32 as _post1
    from MCEq.spacc import etd2_post_apply2_multirhs_f32 as _post2

    int_off_empty = (spacc_int_off is None) or (spacc_int_off.nnz == 0)
    dec_off_empty = (spacc_dec_off is None) or (spacc_dec_off.nnz == 0)

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            h = float(dX[k])
            ri = float(rho_inv[k])

            _etd_compute_diag_factors(h, ri, d_int, d_dec, bufs)
            eD_f32[:] = eD_f64
            phi1_f32[:] = phi1_f64
            phi2_f32[:] = phi2_f64

            # F_phi = int_off @ phc + ri * dec_off @ phc (accumulating SpMM f32)
            F_phi.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, phc_tile_ptrs[t], dim, F_phi_tile_ptrs[t], dim
                    )
                if not dec_off_empty:
                    spacc_dec_off.gemm_ctargs(
                        ri, nrhs, phc_tile_ptrs[t], dim, F_phi_tile_ptrs[t], dim
                    )

            _post1(dim, K, h, eD_p, phi1_p, phc_p_full, F_phi_p_full, a_p_full)

            F_a.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, a_tile_ptrs[t], dim, F_a_tile_ptrs[t], dim
                    )
                if not dec_off_empty:
                    spacc_dec_off.gemm_ctargs(
                        ri, nrhs, a_tile_ptrs[t], dim, F_a_tile_ptrs[t], dim
                    )

            _post2(dim, K, h, phi2_p, a_p_full, F_a_p_full, F_phi_p_full, phc_p_full)

            if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(np.copy(phc))
                grid_step += 1

    info(
        2,
        f"Performance (spacc multirhs f32 K={K}): "
        f"{1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * (time() - start) / float(nsteps) / float(K):6.2f}ms/iteration/RHS)",
    )

    return phc.copy(), np.array(grid_sol)


def solv_spacc_etd2_carousel(
    spacc_int_off,
    spacc_dec_off,
    d_int,
    d_dec,
    dX,
    rho_inv,
    phi_initial,
    schedule,
    phi0_per_pixel,
):
    """ETD2RK carousel on Apple Accelerate Sparse BLAS.

    Step body identical to :func:`solv_spacc_etd2_multipath`. After
    each step boundary, harvest pixels that just finished, then reset
    those slots to the next pixel's phi0. See
    :func:`solv_numpy_etd2_carousel` for the algorithm; the only
    backend-specific bit is the SpMM dispatch and the (dim, K)
    column-major buffer layout that Accelerate expects.
    """
    from ctypes import POINTER, c_double, sizeof

    T = schedule.T
    K = schedule.K
    K_total = schedule.K_total
    dim = phi_initial.shape[0]
    if dX.shape != (T, K) or rho_inv.shape != (T, K):
        raise ValueError(
            f"solv_spacc_etd2_carousel: dX/rho_inv must be (T,K)={T, K}; "
            f"got dX={dX.shape}, rho_inv={rho_inv.shape}"
        )
    if phi_initial.shape != (dim, K):
        raise ValueError(
            f"solv_spacc_etd2_carousel: phi_initial must be (dim,K)="
            f"({dim},{K}); got {phi_initial.shape}"
        )
    if phi0_per_pixel.shape != (dim, K_total):
        raise ValueError(
            f"solv_spacc_etd2_carousel: phi0_per_pixel must be "
            f"(dim,K_total)=({dim},{K_total}); got {phi0_per_pixel.shape}"
        )

    tile = _SPACC_SPMM_TILE
    tile = max(1, min(int(tile), K))

    phc = np.asfortranarray(phi_initial.astype(np.float64, copy=True))
    F_phi = np.zeros((dim, K), dtype=np.float64, order="F")
    F_a = np.zeros((dim, K), dtype=np.float64, order="F")
    a = np.empty((dim, K), dtype=np.float64, order="F")
    dec_phc = np.zeros((dim, K), dtype=np.float64, order="F")
    dec_a = np.zeros((dim, K), dtype=np.float64, order="F")

    diag = {}
    for key in ("D", "hD", "eD", "phi1", "phi2", "scratch", "abs_hD"):
        diag[key] = np.empty((dim, K), dtype=np.float64, order="F")
    for key in ("mask1", "mask2"):
        diag[key] = np.empty((dim, K), dtype=bool, order="F")

    eD = diag["eD"]
    phi1 = diag["phi1"]
    phi2 = diag["phi2"]

    dbl = sizeof(c_double)
    tile_starts = list(range(0, K, tile))
    tile_widths = [min(tile, K - c0) for c0 in tile_starts]
    n_tiles = len(tile_starts)

    phc_addr = phc.ctypes.data
    F_phi_addr = F_phi.ctypes.data
    F_a_addr = F_a.ctypes.data
    a_addr = a.ctypes.data
    dec_phc_addr = dec_phc.ctypes.data
    dec_a_addr = dec_a.ctypes.data

    def _ptrs_at(addr, c0):
        return c_double.from_address(addr + c0 * dim * dbl)

    phc_tile_ptrs = [_ptrs_at(phc_addr, c0) for c0 in tile_starts]
    F_phi_tile_ptrs = [_ptrs_at(F_phi_addr, c0) for c0 in tile_starts]
    F_a_tile_ptrs = [_ptrs_at(F_a_addr, c0) for c0 in tile_starts]
    a_tile_ptrs = [_ptrs_at(a_addr, c0) for c0 in tile_starts]
    dec_phc_tile_ptrs = [_ptrs_at(dec_phc_addr, c0) for c0 in tile_starts]
    dec_a_tile_ptrs = [_ptrs_at(dec_a_addr, c0) for c0 in tile_starts]

    phc_p_full = phc.ctypes.data_as(POINTER(c_double))
    F_phi_p_full = F_phi.ctypes.data_as(POINTER(c_double))
    F_a_p_full = F_a.ctypes.data_as(POINTER(c_double))
    a_p_full = a.ctypes.data_as(POINTER(c_double))
    eD_p = eD.ctypes.data_as(POINTER(c_double))
    phi1_p = phi1.ctypes.data_as(POINTER(c_double))
    phi2_p = phi2.ctypes.data_as(POINTER(c_double))

    from MCEq.spacc import etd2_post_apply1_multipath as _post1
    from MCEq.spacc import etd2_post_apply2_multipath as _post2

    int_off_empty = (spacc_int_off is None) or (spacc_int_off.nnz == 0)
    dec_off_empty = (spacc_dec_off is None) or (spacc_dec_off.nnz == 0)

    sol_pixel = np.empty((dim, K_total), dtype=np.float64)

    rs = schedule.reset_t_starts
    rj = schedule.reset_j
    rp = schedule.reset_pixel
    cs = schedule.record_t_starts
    cj = schedule.record_j
    cp = schedule.record_pixel

    from time import time

    start = time()

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(T):
            h_K = dX[k]
            ri_K = rho_inv[k]

            _etd_compute_diag_factors_multipath(h_K, ri_K, d_int, d_dec, diag)

            F_phi.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, phc_tile_ptrs[t], dim, F_phi_tile_ptrs[t], dim
                    )
            if not dec_off_empty:
                dec_phc.fill(0.0)
                for t in range(n_tiles):
                    nrhs = tile_widths[t]
                    spacc_dec_off.gemm_ctargs(
                        1.0, nrhs, phc_tile_ptrs[t], dim, dec_phc_tile_ptrs[t], dim
                    )
                dec_phc *= ri_K[None, :]
                np.add(F_phi, dec_phc, out=F_phi)

            h_K_p = h_K.ctypes.data_as(POINTER(c_double))
            _post1(dim, K, h_K_p, eD_p, phi1_p, phc_p_full, F_phi_p_full, a_p_full)

            F_a.fill(0.0)
            for t in range(n_tiles):
                nrhs = tile_widths[t]
                if not int_off_empty:
                    spacc_int_off.gemm_ctargs(
                        1.0, nrhs, a_tile_ptrs[t], dim, F_a_tile_ptrs[t], dim
                    )
            if not dec_off_empty:
                dec_a.fill(0.0)
                for t in range(n_tiles):
                    nrhs = tile_widths[t]
                    spacc_dec_off.gemm_ctargs(
                        1.0, nrhs, a_tile_ptrs[t], dim, dec_a_tile_ptrs[t], dim
                    )
                dec_a *= ri_K[None, :]
                np.add(F_a, dec_a, out=F_a)

            _post2(
                dim, K, h_K_p, phi2_p, a_p_full, F_a_p_full, F_phi_p_full, phc_p_full
            )

            for r in range(cs[k], cs[k + 1]):
                sol_pixel[:, cp[r]] = phc[:, cj[r]]
            for r in range(rs[k], rs[k + 1]):
                phc[:, rj[r]] = phi0_per_pixel[:, rp[r]]

    elapsed = time() - start
    info(
        2,
        f"Performance (spacc carousel K={K}, K_total={K_total}, T={T}): "
        f"{1e3 * elapsed / float(T):6.2f}ms/iteration "
        f"({1e3 * elapsed / float(T) / float(K):6.2f}ms/iter/slot)",
    )

    return sol_pixel


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
