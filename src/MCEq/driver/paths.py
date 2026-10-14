"""Integration-path construction, including per-condition batched paths."""

import numpy as np

from MCEq.misc import info
from MCEq.operators.stiffness import LOSS_STEP_SAFETY, continuous_loss_rate
from MCEq.solvers.path import em_cascade_dx_cap as path_em_cascade_dx_cap
from MCEq.solvers.path import etd2_nonuniform_path


def em_cascade_dx_cap(run, r_em):
    """EM step cap for the stiffness ``r_em`` and ``run.config.em``, or np.inf."""
    return path_em_cascade_dx_cap(r_em, run.config.em)


def continuous_loss_dx_cap(run):
    """Step cap ``LOSS_STEP_SAFETY / rate`` from the off-diagonal row sums of
    the continuous-loss bands (scaled by sec(theta) if secant is on).

    Cached per identity of ``int_m`` and the secant operators; np.inf before
    the matrices exist.
    """
    if getattr(run, "int_m", None) is None:
        return np.inf  # atmosphere-only path planning before assembly
    builder = run.matrix_builder
    sec = run._resolve_secant()
    cached = getattr(run, "_loss_step_cache", None)
    if cached is None or cached[0] is not run.int_m or cached[1] is not sec:
        bands = builder._contloss_bands
        rate = continuous_loss_rate(
            bands.values(),
            None if sec is None else sec["lam"],
            [builder._contloss_upwind_rows[key] for key in bands],
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
    """Build the ETD2 path on ``run.integration_path`` unless the cached one
    matches the resolved parameters.

    ``dX_max`` is capped by the continuous-loss and EM stiffness limits.
    """
    solver = run.config.solver
    p = solver.etd2_path
    X_start = solver.X_start if X_start is None else X_start
    eps = p["eps"] if eps is None else eps
    dX_max = p["dX_max"] if dX_max is None else dX_max
    if eps is None or dX_max is None:
        conservative = run._mceq_db.is_2d or any(
            abs(particle.pdg_id[0]) == 11 for particle in run.pman.all_particles
        )
        if eps is None:
            eps = 0.01 if conservative else 0.03
        if dX_max is None:
            dX_max = 2.0 if conservative else 5.0
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
        "fd_span={}, X_start={})".format(eps, dX_max, dX_min, fd_span, X_start),
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


# Run instance shared with forked path workers (inherited, not pickled).
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
    values; missing azimuth is ``None``. Conditions with the same density
    model and zenith, and the same azimuth when the model depends on it,
    share one path tuple. Restores the active density model and angles
    before returning.

    Args:
      conditions (list[dict]): one dict per batch member.
      X_start, eps, dX_max, dX_min, fd_span: ETD2 path knobs, see
        :meth:`solve`.
      path_workers (int): fork-pool size for a parallel path build.
        Only allowed without ``density_model`` overrides. Builds
        serially where fork is unavailable (Windows).

    Returns:
      list: ``(nsteps, dX, rho_inv, grid_idcs)`` tuples, one per
      condition; duplicate conditions share the same tuple object.
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

        # The pool needs fork; build serially on spawn-only platforms.
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
