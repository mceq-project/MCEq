"""The integration path the ETD2 kernels step along.

:func:`etd2_nonuniform_path` turns a density model into the
``(nsteps, dX, rho_inv, grid_idcs)`` tuple of the kernel-dispatch contract:
step sizes from the local variation of ``1/rho``, and ``rho_inv`` as the
per-step integral mean rather than a point sample. The module also carries
the documented EM-row caveat the kernels tolerate.

Step-size policy and the depth grid belong here. The step formulas are in
:mod:`MCEq.solvers.numerics` and the loop that consumes the path is
:func:`MCEq.solvers.etd2.etd2_driver`.
"""

import numpy as np

from MCEq.misc import info


def _live_config():
    """The :mod:`MCEq.config` module, fetched dynamically.

    Same device as :func:`MCEq.misc._views`: the remaining
    ``None -> global`` fallbacks (a standalone ``etd2_nonuniform_path``
    call, a user-built ``EarthGeometry()``) must keep reading the live
    module, but this layer must not carry a static import edge to it
    (contract C5). Milestones M3/M5 shrink these fallbacks to the two
    documented process-level exceptions.
    """
    from importlib import import_module

    return import_module("MCEq.config")


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
    step=None,
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
        ``step.X_start``.
      eps (float | None): within-step ``rho_inv`` variation tolerance;
        ``None`` → ``step.etd2_path["eps"]``.
      dX_max (float | None): cap on step size (off-diagonal stability
        cliff); ``None`` → ``step.etd2_path["dX_max"]``.
      dX_min (float | None): floor on step size; ``None`` →
        ``step.etd2_path["dX_min"]``.
      fd_span (float | None): forward-FD probe span; ``None`` →
        ``step.etd2_path["fd_span"]``.
      int_grid (np.ndarray | None): X values at which to record snapshots.
        Steps are truncated to land exactly on each ``int_grid`` entry.
      step: the ``solver`` setting group (``X_start``, ``etd2_path``);
        ``None`` → ``MCEq.config.solver``. Read here, when the path is
        planned — a live instance's next ``solve()`` does not see a write
        made after the path was built, because
        ``MCEqRun._calculate_integration_path`` caches on the unresolved
        keyword arguments.

    Returns:
      (nsteps, dX, rho_inv, grid_idcs): tuple compatible with the
      kernel-dispatch contract used by ``MCEqRun.integration_path``.
    """
    if step is None:
        step = _live_config().solver
    if X_start is None:
        X_start = step.X_start
    p = step.etd2_path
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


# ---------------------------------------------------------------------------
# Path machinery owned by the driver-facing facade (Phase 6 commit 5, plan
# D26). These functions take the ``MCEqRun`` facade as ``run`` and drive it
# exactly as the ``MCEqRun`` methods did: the EM stiffness memo stays on the
# driver (``run._em_cascade_step_scale``), the path cache lives in
# ``run.integration_path`` / ``run._cached_etd2_path_params``, and direction
# or atmosphere changes go through the facade methods, so a subclass
# override of ``set_density_model`` / ``set_zenith_azimuth`` /
# ``_calculate_integration_path`` still reaches this code. ``MCEqRun``
# keeps thin delegators for all three.
# ---------------------------------------------------------------------------


def em_cascade_dx_cap(run):
    """Cure-B effective dX cap from the EM-cascade stiffness, or np.inf.

    ``np.inf`` (no cap) when the ``em`` group's ``adaptive_step`` is off
    or the EM cascade is inactive, so the legacy schedule is reproduced
    exactly. Settings come from ``run.config.em`` (the run's snapshot or
    the live group), never from a module-level config import (C5).
    """
    em = run.config.em
    if not em.adaptive_step:
        return np.inf
    r_em = run._em_cascade_step_scale()
    if not (r_em > 0.0):
        return np.inf
    cap = em.step_safety / r_em
    info(
        2,
        "EM-adaptive step (cure B): r_EM={:.4g} 1/(g/cm^2) "
        "-> dX_max <= {:.4g} g/cm^2".format(r_em, cap),
    )
    return cap


def calculate_integration_path(
    run,
    int_grid,
    grid_var,
    force=False,
    *,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
):
    # ETD2 is the only path builder. Step sizes follow the
    # atmosphere-aware non-uniform schedule keyed off the local
    # |d ln rho_inv / dX|; see ``MCEq.solvers.etd2_nonuniform_path``.
    # Cure B: additionally cap dX_max by the explicit-stepping stiffness
    # of the EM block of int_m (no-op when the em group's adaptive_step
    # is off). int_m is X-constant, so this is a single global cap that the
    # density-gradient schedule never relaxes above. Settings read the
    # run's ``solver`` group (snapshot or live view — C5).
    solver = run.config.solver
    em_cap = run._em_cascade_dx_cap()
    if np.isfinite(em_cap):
        base_dX_max = dX_max if dX_max is not None else solver.etd2_path["dX_max"]
        dX_max = min(base_dX_max, em_cap)
    etd2_params = (X_start, eps, dX_max, dX_min, fd_span)
    cached_etd2_params = getattr(run, "_cached_etd2_path_params", None)

    if (
        run.integration_path
        and np.all(int_grid == run.int_grid)
        and np.all(run.grid_var == grid_var)
        and cached_etd2_params == etd2_params
        and not force
    ):
        info(5, "skipping calculation.")
        return

    run._cached_etd2_path_params = etd2_params
    run.int_grid, run.grid_var = int_grid, grid_var
    if grid_var != "X":
        raise NotImplementedError(
            "Grid variables other than the depth X not supported."
        )

    info(
        2,
        "ETD2 non-uniform path (eps={}, dX_max={}, dX_min={}, "
        "fd_span={}, X_start={})".format(
            eps if eps is not None else solver.etd2_path["eps"],
            dX_max if dX_max is not None else solver.etd2_path["dX_max"],
            dX_min if dX_min is not None else solver.etd2_path["dX_min"],
            fd_span if fd_span is not None else solver.etd2_path["fd_span"],
            X_start if X_start is not None else solver.X_start,
        ),
    )
    run.integration_path = etd2_nonuniform_path(
        run.density_model,
        X_start=X_start,
        eps=eps,
        dX_max=dX_max,
        dX_min=dX_min,
        fd_span=fd_span,
        int_grid=int_grid,
        step=solver,
    )


# Module-level worker state for the optional process-pool path build
# inside :func:`build_condition_paths`. Workers fork from the parent
# and inherit ``_PATH_WORKER_MCEQ`` via copy-on-write — the MCEqRun
# instance itself never has to be picklable. Each worker process gets
# its own CoW copy of the density model, so per-worker
# ``set_zenith_azimuth`` mutations stay process-local. Only used when
# ``solve_fullsky(path_workers=N>0)`` is requested *and* the atmosphere
# is not azimuth-symmetric (MSIS location-centered case). Every atmosphere
# is fork-reproducible; the paths a worker returns are bitwise equal to the
# serial ones.
_PATH_WORKER_MCEQ = None


def _path_worker_one(args):
    """Build one (zenith, azimuth) integration path inside a forked worker."""
    flat_idx, zen, az, kwargs = args
    if az is None:
        _PATH_WORKER_MCEQ.set_zenith_azimuth(zen)
    else:
        _PATH_WORKER_MCEQ.set_zenith_azimuth(zen, az)
    _PATH_WORKER_MCEQ._calculate_integration_path(None, "X", **kwargs)
    return flat_idx, _PATH_WORKER_MCEQ.integration_path


def build_condition_paths(
    run,
    conditions,
    *,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
    path_workers=0,
):
    """Build one ETD2 integration path per batch condition.

    Each condition is a dict with optional keys ``zenith_deg``,
    ``azimuth_deg`` and ``density_model`` (config tuple or density-
    model instance); missing keys fall back to the instance's current
    setting. Conditions that resolve to the same physical path — the
    same density model and zenith, and the same azimuth when the
    model's ``depends_on_azimuth`` is True — share one path tuple, so
    duplicates (e.g. azimuth pixels of an azimuth-independent
    atmosphere) cost nothing. Restores the active density model and
    angles before returning.

    Args:
      conditions (list[dict]): one dict per batch member.
      X_start, eps, dX_max, dX_min, fd_span: ETD2 path knobs, see
        :meth:`solve`.
      path_workers (int): fork-pool size for a parallel path build.
        Only allowed without ``density_model`` overrides. A worker's
        paths are bitwise equal to the serial ones on every
        atmosphere, MSIS00 included. Ignored where fork is
        unavailable (Windows), which builds serially.

    Returns:
      list: ``(nsteps, dX, rho_inv, grid_idcs)`` tuples, one per
      condition (duplicate conditions share the same tuple object —
      callers can detect a fully shared batch with ``is``).
    """
    allowed_keys = {"zenith_deg", "azimuth_deg", "density_model"}
    norm = []
    for i, c in enumerate(conditions):
        if c is None:
            c = {}
        if not isinstance(c, dict):
            raise TypeError(
                f"_build_condition_paths: condition {i} must be a dict "
                f"with keys in {sorted(allowed_keys)}, got "
                f"{type(c).__name__}"
            )
        unknown = set(c) - allowed_keys
        if unknown:
            raise ValueError(
                f"_build_condition_paths: condition {i} has unknown "
                f"keys {sorted(unknown)}; allowed: {sorted(allowed_keys)}"
            )
        norm.append(dict(c))

    has_dm_override = any(c.get("density_model") is not None for c in norm)
    n_workers = int(path_workers) if path_workers else 0
    if n_workers > 1:
        if has_dm_override:
            raise ValueError(
                "path_workers > 1 supports only zenith/azimuth batches "
                "on the active atmosphere; build density_model "
                "overrides serially (path_workers=0)."
            )
        import multiprocessing as _mp

        # Windows and any spawn-only platform: the pool needs fork to
        # share the atmosphere, so build serially instead. Same paths.
        if "fork" not in _mp.get_all_start_methods():
            n_workers = 0

    def dm_key_of(c):
        dm_spec = c.get("density_model")
        if dm_spec is None:
            return ("current",)
        if isinstance(dm_spec, (tuple, list)):
            return ("cfg", repr(tuple(dm_spec)))
        return ("obj", id(dm_spec))

    # Save the *current* direction from the density model, which is the
    # only place the azimuth lives. ``MCEqRun.theta_deg`` does track the
    # zenith since B13 was fixed, but it carries no azimuth, so restoring
    # from the atmosphere keeps both halves of the direction together.
    saved_dm = run.density_model
    saved_zen = getattr(saved_dm, "theta_deg", None)
    saved_az = getattr(saved_dm, "_current_azimuth_deg", None)

    kwargs = dict(
        X_start=X_start, eps=eps, dX_max=dX_max, dX_min=dX_min, fd_span=fd_span
    )
    try:
        # Pass 1: resolve unique density models (instantiate config
        # tuples exactly once).
        dm_instances = {}
        for c in norm:
            key = dm_key_of(c)
            if key in dm_instances:
                continue
            dm_spec = c.get("density_model")
            if dm_spec is None:
                dm_instances[key] = saved_dm
            elif isinstance(dm_spec, (tuple, list)):
                run.set_density_model(tuple(dm_spec))
                dm_instances[key] = run.density_model
            else:
                dm_instances[key] = dm_spec

        # Pass 2: dedup conditions into unique path-build jobs. The
        # azimuth only enters the key when the density model actually
        # depends on it, so azimuth pixels of symmetric atmospheres
        # collapse onto one job per zenith.
        job_of_key = {}
        cond_keys = []
        for c in norm:
            dm_key = dm_key_of(c)
            dm = dm_instances[dm_key]
            zen = c.get("zenith_deg")
            if zen is None:
                zen = saved_zen
            if zen is None:
                raise ValueError(
                    "_build_condition_paths: condition without "
                    "'zenith_deg' and no zenith set on the instance"
                )
            zen = float(zen)
            az = c.get("azimuth_deg")
            az_dep = getattr(dm, "depends_on_azimuth", False)
            az_eff = float(az) if (az is not None and az_dep) else None
            pkey = (dm_key, zen, az_eff)
            cond_keys.append(pkey)
            if pkey not in job_of_key:
                job_of_key[pkey] = (dm_key, zen, az_eff)

        # Group jobs by density model so per-model state (splines,
        # caches) is not rebuilt more often than necessary.
        jobs = sorted(job_of_key.items(), key=lambda item: repr(item[0]))
        unique_paths = {}
        if n_workers > 1 and len(jobs) > 1:
            # Fork-based worker pool. Pickling MCEqRun would be
            # fragile; instead set a module-level global and rely on
            # fork() to share via CoW. Only zenith/azimuth vary here
            # (density_model overrides were rejected above).
            import multiprocessing as _mp

            global _PATH_WORKER_MCEQ
            _PATH_WORKER_MCEQ = run  # inherited by forked children
            try:
                ctx = _mp.get_context("fork")
                worker_args = [
                    (idx, job[1], job[2], kwargs) for idx, (_, job) in enumerate(jobs)
                ]
                chunksize = max(1, len(jobs) // (n_workers * 8))
                with ctx.Pool(n_workers) as pool:
                    for flat_idx, path in pool.imap_unordered(
                        _path_worker_one, worker_args, chunksize=chunksize
                    ):
                        unique_paths[jobs[flat_idx][0]] = path
            finally:
                _PATH_WORKER_MCEQ = None
        else:
            for pkey, (dm_key, zen, az_eff) in jobs:
                dm = dm_instances[dm_key]
                if run.density_model is not dm:
                    run.set_density_model(dm)
                if az_eff is None:
                    run.set_zenith_azimuth(zen)
                else:
                    run.set_zenith_azimuth(zen, az_eff)
                run._calculate_integration_path(None, "X", **kwargs)
                unique_paths[pkey] = run.integration_path

        return [unique_paths[k] for k in cond_keys]
    finally:
        if run.density_model is not saved_dm:
            run.set_density_model(saved_dm)
        if saved_zen is not None:
            run.set_zenith_azimuth(saved_zen, saved_az)
        run.integration_path = None
