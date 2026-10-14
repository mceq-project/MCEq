"""The non-uniform integration path of the ETD2 step loop and its stiffness caps."""

import numpy as np
import scipy.sparse as sp

from MCEq import config
from MCEq.misc import info


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
    integral mean of ``density_model.r_X2rho`` over each step, estimated from
    a cumulative trapezoid integral on a hybrid log/linear sample grid;
    ``scipy.integrate.quad`` is used per step only when ``ri`` does not accept
    array input. The step mean, not a point sample, is needed at the first
    step, which crosses the spline-saturation cap.

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
        eps = 0.01 if p["eps"] is None else p["eps"]
    if dX_max is None:
        dX_max = 2.0 if p["dX_max"] is None else p["dX_max"]
    if dX_min is None:
        dX_min = p["dX_min"]
    if fd_span is None:
        fd_span = p["fd_span"]

    values = (X_start, eps, dX_max, dX_min, fd_span)
    if (
        not np.isfinite(values).all()
        or X_start < 0
        or dX_min < 0
        or min(eps, dX_max, fd_span) <= 0
    ):
        raise ValueError(
            "ETD2 path settings must be finite and positive (X_start and dX_min >= 0)"
        )
    if dX_min > dX_max:
        raise ValueError("ETD2 dX_min must not exceed dX_max")

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
        # snapshot point; this can drive h below dX_min.
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

    # Per-step integral means of ``r_X2rho`` from a cubic spline of the
    # cumulative trapezoid integral on a hybrid log/linear sample: the log part
    # resolves the saturation spike of 1/rho near X = 0, and the cumulative is
    # monotone where a direct fit of r_X2rho would overshoot. Falls back to
    # ``quad`` if ``ri`` rejects array input.
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
                # Log sample over the whole domain (1/rho varies over decades
                # at high altitude, also on near-horizontal paths) plus a
                # linear sample for the bulk.
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


def em_cascade_dx_cap(step_scale):
    """EM step cap ``config.em_step_safety / r_EM``, or inf when disabled or ``r_EM`` is not > 0."""
    if not config.em_adaptive_step:
        return np.inf
    if not (step_scale > 0.0):
        return np.inf
    cap = config.em_step_safety / step_scale
    info(
        2,
        "EM-adaptive step: r_EM={:.4g} 1/(g/cm^2) -> dX_max <= {:.4g} g/cm^2".format(
            step_scale, cap
        ),
    )
    return cap


#: The continuous-loss cap is LOSS_STEP_SAFETY / rate (:ref:`solver-math`).
LOSS_STEP_SAFETY = 0.5


def continuous_loss_rate(bands, secant_eigenvalues=None, upwind_rows=None):
    """The inverse-depth rate of the explicit part of the loss bands.

    Args:
      bands: the assembled loss stencils, one dense band per species.
      secant_eigenvalues: eigenvalues of the sec(theta) coupling; the rate
        is scaled by their maximum (at least 1). ``None``: paraxial.
      upwind_rows: per band, the number of leading rows that are its
        monotone upwind closure and do not enter the rate. ``None`` counts
        every row.

    Empty or diagonal bands impose no restriction; nonfinite coefficients
    raise ``ValueError``.
    """
    rate = 0.0
    if upwind_rows is None:
        upwind_rows = (0 for _ in bands)
    for band, n_up in zip(bands, upwind_rows, strict=True):
        band = np.asarray(band, dtype=np.float64)
        if not np.isfinite(band).all():
            raise ValueError("Nonfinite continuous-loss stencil coefficients")
        off = band.copy()
        np.fill_diagonal(off, 0.0)
        row_sums = np.sum(np.abs(off), axis=1)
        rate = max(rate, float(np.max(row_sums[int(n_up) :], initial=0.0)))
    if secant_eigenvalues is not None:
        lam = np.asarray(secant_eigenvalues)
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise ValueError("Secant eigenvalues must be finite and positive")
        rate *= max(1.0, float(np.max(lam, initial=1.0)))
    return rate


def em_step_scale(int_m, particles, dense_eig_max=4000):
    """Spectral radius of the off-diagonal e+/-/gamma block of ``int_m``, in cm^2/g.

    The EM off-diagonal is strongly non-normal, so its spectral radius, not a
    norm, sets the step scale r_EM. ``particles`` supply ``is_em`` and the
    ``lidx``/``uidx`` of one ``dim_states`` block (mode l = 0 on a 2D
    operator). Returns 0.0 if no species is EM.
    """
    col_ranges = [
        np.arange(p.lidx, p.uidx) for p in particles if getattr(p, "is_em", False)
    ]
    if not col_ranges:
        return 0.0
    idx = np.concatenate(col_ranges)
    diag = int_m.diagonal()
    int_off = (int_m - sp.diags(diag, format="csr")).tocsc()
    block = int_off[idx][:, idx].tocsc().astype(np.float64)
    # Dense eigvals up to dense_eig_max (ARPACK often fails to converge on
    # this non-normal block); ARPACK beyond.
    n = block.shape[0]
    try:
        if n <= int(dense_eig_max):
            ev = np.linalg.eigvals(block.toarray())
        else:
            from scipy.sparse.linalg import eigs

            ev = eigs(block, k=1, which="LM", return_eigenvectors=False, maxiter=10000)
        return float(np.max(np.abs(ev)))
    except Exception:
        # The spectral radius is bounded by both induced norms; take the smaller.
        n1 = float(np.abs(block).sum(axis=0).max())
        ninf = float(np.abs(block).sum(axis=1).max())
        return min(n1, ninf)
