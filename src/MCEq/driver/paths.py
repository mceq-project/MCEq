"""Run-level integration-path construction, caching, and batch preparation.

These functions operate on an ``MCEqRun`` object: they write
``run.integration_path`` / ``run._cached_etd2_path_params``, switch the
run's zenith / atmosphere, and the fork pool holds the run in a module
global, so they belong to the driver rather than the solver layer.
Numerical path construction is implemented in
:func:`MCEq.solvers.path.etd2_nonuniform_path`; the EM stiffness memo
stays on ``MCEqRun._em_cascade_step_scale`` (the cap function here
consumes it and passes the value down pure).

``MCEqRun._calculate_integration_path`` and
``MCEqRun._em_cascade_dx_cap`` delegate here, so subclass overrides of
``set_density_model`` / ``set_zenith_azimuth`` still reach this code.
"""

import numpy as np

from MCEq.misc import info
from MCEq.operators.stiffness import LOSS_STEP_SAFETY, continuous_loss_rate
from MCEq.solvers.path import em_cascade_dx_cap as path_em_cascade_dx_cap
from MCEq.solvers.path import etd2_nonuniform_path


def em_cascade_dx_cap(run, r_em):
    """Effective EM stiffness-based step cap, or np.inf.

    Combine the run's EM settings with the supplied stiffness estimate to
    determine the additional step cap. Settings come from
    ``run.config.em`` (the run's snapshot or the live group), never from a
    module-level config import. *r_em* is the caller's memoised
    stiffness, so the memo stays on the run. The arithmetic lives in the
    pure core :func:`MCEq.solvers.path.em_cascade_dx_cap`.
    """
    return path_em_cascade_dx_cap(r_em, run.config.em)


def continuous_loss_dx_cap(run):
    """Memoize on matrix/secant identity; a rebuild invalidates the cap."""
    if getattr(run, "int_m", None) is None:
        return np.inf  # atmosphere-only path planning before assembly
    builder = run.matrix_builder
    sec = run._resolve_secant()
    cached = getattr(run, "_loss_step_cache", None)
    if cached is None or cached[0] is not run.int_m or cached[1] is not sec:
        rate = continuous_loss_rate(
            builder._contloss_bands.values(),
            None if sec is None else sec["lam"],
        )
        cap = LOSS_STEP_SAFETY / rate if rate else np.inf
        run._loss_step_cache = (run.int_m, sec, cap)
    return run._loss_step_cache[2]


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
    # Apply atmosphere accuracy settings plus caps from the assembled loss
    # stencil (including secant elongation) and the optional EM estimator.
    # Resolve before caching so live config changes cannot reuse stale paths.
    solver = run.config.solver
    p = solver.etd2_path
    X_start = solver.X_start if X_start is None else X_start
    eps = p["eps"] if eps is None else eps
    dX_max = p["dX_max"] if dX_max is None else dX_max
    dX_min = p["dX_min"] if dX_min is None else dX_min
    fd_span = p["fd_span"] if fd_span is None else fd_span
    cap = min(
        em_cascade_dx_cap(run, run._em_cascade_step_scale()),
        continuous_loss_dx_cap(run),
    )
    dX_max = min(dX_max, cap)
    # A user floor must never undo the stiffness ceiling.
    dX_min = min(dX_min, dX_max)
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
# ``set_zenith_azimuth`` mutations stay process-local. The pool is used
# when multiple distinct paths are requested, ``path_workers > 1``, and no
# per-condition density-model overrides require serial handling. Every
# atmosphere is fork-reproducible; the paths a worker returns are bitwise
# equal to the serial ones.
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
    model instance). Missing zenith and density-model fields use current
    values; missing azimuth is ``None``. Conditions that resolve to the
    same physical path — the
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

    # Save both angles from the density model, which owns the active
    # azimuth setting.
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
