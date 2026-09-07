"""Batch orchestration: K independent solves, sky grids, multi-RHS.

Moved verbatim from ``MCEq/core.py`` in Phase 6 commit 4 (plan §16). The
functions take the :class:`MCEqRun` facade as their first argument
(``run``) and use it as the orchestrator used to: resolve the secant
operator set, build paths (``run._calculate_integration_path`` /
``run._build_condition_paths`` -- the fork pool and the path builder
follow the planner to ``solvers/path.py`` in commit 5), run the ETD2
routes, and wrap results in :class:`MCEq.driver.results.MCEqBatchResult`.

Two deltas against the pre-split code, both recorded on the delegators in
``core.py``: the moved bodies call ``run.solve_batch`` etc. through the
facade (monkey-patching the facade method still reaches this code), and
``solve_fullsky``'s 2-D-phi0 cutoff warning raises ``stacklevel=3`` so it
still points at the user's call through the delegator.
"""

from time import time

import numpy as np

from MCEq.driver.results import MCEqBatchResult
from MCEq.misc import info


def solve_batch(
    run,
    phi0=None,
    conditions=None,
    int_grid=None,
    grid_var="X",
    *,
    dtype=np.float64,
    carousel_K=None,
    path_workers=0,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
):
    """Solve K independent cascade problems in one batched call.

    This is the general entry point for anything that would
    otherwise be a loop of :meth:`solve` calls with a shared
    operator: many primary spectra or single-primary energies at one
    direction, many zenith/azimuth directions, atmospheres for
    different days/seasons — or any mix of these. The batch members
    are defined by the columns of ``phi0`` and/or the entries of
    ``conditions``:

    * ``conditions=None`` (default): all columns share the current
      ``(zenith, atmosphere)``. K is the number of ``phi0`` columns.
      Supports ``int_grid`` snapshots and fp32 (non-numpy backends).
    * ``conditions=[...]``: one dict per batch member with optional
      keys ``zenith_deg``, ``azimuth_deg`` and ``density_model``
      (config tuple or instance); missing keys fall back to the
      instance's current setting. One integration path is built per
      *distinct* condition (duplicates — including azimuth pixels of
      azimuth-independent atmospheres — share a path), and the batch
      runs on the LPT carousel so total kernel work is
      ``Σ nsteps × ms/RHS`` rather than ``max(nsteps) × K``. If all
      conditions resolve to the same path, the shared-path fast
      route is used automatically.

    Does NOT mutate ``run._phi0`` / ``run._solution`` /
    ``run.grid_sol``; the active density model and angles are
    restored after the path build.

    For 2D databases the sec(theta) transport correction resolves
    through ``config.secant_mode(is_2d)`` exactly as in
    :meth:`solve`: one constant operator set (the cap is not zenith
    dependent) is shared by every column, on every backend and at
    either ``dtype``.

    Examples::

        # K single primaries (response matrix) at the current angle
        phi0 = np.stack(
            [mceq.initial_state({"E": E, "pdg_id": 2212}) for E in E_primaries],
            axis=1,
        )
        res = mceq.solve_batch(phi0)

        # one spectrum through 12 months x 3 zeniths
        conditions = [
            {"zenith_deg": z, "density_model": ("MSIS21", ("SouthPole", month))}
            for month in months
            for z in (0.0, 30.0, 60.0)
        ]
        res = mceq.solve_batch(conditions=conditions)
        res.get_solution("conv_numu", k=5, mag=3)

    Args:
      phi0 (np.ndarray | None): initial state. ``None`` uses the
        instance initial condition (from ``set_primary_model`` etc.);
        1-D ``(dim_states,)`` is broadcast to all batch members; 2-D
        ``(dim_states, K)`` carries one column per member (compose
        columns with :meth:`initial_state` /
        :meth:`get_initial_state`).
      conditions (list[dict] | None): per-member direction and
        atmosphere, see above.
      int_grid (list | None): X values at which to record snapshots.
        Only supported for shared-path batches (``conditions=None``).
      grid_var (str): only ``"X"`` is supported.
      dtype (np.float32 | np.float64): precision of the state
        buffers, on every backend and every route. The diagonal-factor
        pipeline (``exp(h·D)``, φ₁, φ₂) is always computed in fp64
        (fp32 φ-functions suffer catastrophic cancellation) and cast
        once on the way out, so relative error vs fp64 is ≤ 1e-4 for
        the production particle set on all fp32 routes. See
        ``MCEq.solvers.backends.base._PRECISION_CONTRACT``.
      carousel_K (int | None): pipeline width for the LPT scheduler
        (heterogeneous batches only). ``None`` → ``min(K, 128)``.
      path_workers (int): fork-pool size for a parallel path build
        (heterogeneous batches only). Must be 0 with ``density_model``
        overrides; every atmosphere is fork-reproducible.
      X_start, eps, dX_max, dX_min, fd_span (float | None): ETD2
        non-uniform path knobs forwarded to
        :meth:`_calculate_integration_path`; same semantics as
        :meth:`solve`.

    Returns:
      :class:`MCEqBatchResult`: final states plus per-column
      named-spectrum extraction. Also unpacks as the legacy
      ``(sol, grid_sol)`` pair.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"solve_batch: dtype must be float32 or float64, got {dtype}")

    sec_ops = run._resolve_secant()

    # --- resolve phi0 ------------------------------------------------
    if phi0 is None:
        phi0_arr = run._phi0.copy()
    else:
        phi0_arr = np.asarray(phi0, dtype=np.float64)
    if phi0_arr.ndim == 1:
        if phi0_arr.size != run.dim_states:
            raise ValueError(
                f"solve_batch: phi0 has length {phi0_arr.size}, "
                f"expected first axis = dim_states = {run.dim_states}"
            )
        n_cols = None
    elif phi0_arr.ndim == 2:
        if phi0_arr.shape[0] != run.dim_states:
            raise ValueError(
                f"solve_batch: phi0 has shape {phi0_arr.shape}, "
                f"expected first axis = dim_states = {run.dim_states}"
            )
        n_cols = phi0_arr.shape[1]
    else:
        raise ValueError(
            f"solve_batch: phi0 must be 1-D or 2-D, got shape {phi0_arr.shape}"
        )

    if conditions is not None:
        K = len(conditions)
        if K < 1:
            raise ValueError("solve_batch: conditions must not be empty")
        if n_cols is not None and n_cols != K:
            raise ValueError(
                f"solve_batch: phi0 has shape {phi0_arr.shape}, expected "
                f"second axis = K = {K} (len(conditions))"
            )
    else:
        K = n_cols if n_cols is not None else 1

    if phi0_arr.ndim == 1:
        phi0_mat = np.ascontiguousarray(
            np.broadcast_to(phi0_arr[:, None], (run.dim_states, K))
        )
    else:
        phi0_mat = np.ascontiguousarray(phi0_arr)

    # 2D databases: the stitched solver state is n_k * dim_states per
    # column. phi0 stays user-facing at dim_states (per mode); a delta
    # angular initial condition populates every Hankel mode with the
    # same amplitude, so tile the rows across the n_k blocks — the
    # exact analogue of the np.tile in solve().
    if run._mceq_db.is_2d:
        phi0_mat = np.ascontiguousarray(np.tile(phi0_mat, (run._mceq_db.n_k, 1)))

    # --- build integration paths -------------------------------------
    path_kwargs = dict(
        X_start=X_start, eps=eps, dX_max=dX_max, dX_min=dX_min, fd_span=fd_span
    )
    if conditions is None:
        if int_grid is not None and np.any(np.diff(int_grid) < 0):
            raise Exception(
                "The X values in int_grid are required to be strictly increasing."
            )
        run._calculate_integration_path(int_grid, grid_var, **path_kwargs)
        paths = [run.integration_path] * K
    else:
        if int_grid is not None:
            raise NotImplementedError(
                "solve_batch: int_grid snapshots are only supported for "
                "shared-path batches (conditions=None). Set the "
                "direction/atmosphere on the instance and vary only "
                "phi0 columns to record snapshots."
            )
        if grid_var != "X":
            raise NotImplementedError("solve_batch: only grid_var='X' is supported.")
        paths = run._build_condition_paths(
            conditions, path_workers=path_workers, **path_kwargs
        )

    nsteps_per_col = np.array([p[0] for p in paths], dtype=np.int32)
    shared = all(p is paths[0] for p in paths)

    start = time()
    if shared:
        nsteps, dX, rho_inv, grid_idcs = paths[0]
        info(
            2,
            f"solve_batch: shared-path multi-RHS route, K={K}, "
            f"kernel={run.config.backend.kernel_config}, nsteps={nsteps}",
        )
        # ``dtype`` controls the state-buffer precision; the diagonals
        # ``d_int`` / ``d_dec`` remain fp64 in the diag-factor pipeline
        # for the fp32 path (exp(h·D) saturates fp32 fast at high
        # zenith).
        phi0_typed = phi0_mat.astype(dtype, copy=True)
        sol, grid_sol = run._run_etd2(
            nsteps,
            dX,
            rho_inv,
            phi0_typed,
            grid_idcs,
            dtype=dtype,
            sec_ops=sec_ops,
        )
        legacy = (sol, grid_sol)
    else:
        from MCEq.solvers import compile_carousel_schedule, schedule_lpt

        if carousel_K is None:
            carousel_K = min(K, 128)
        K_pipe = max(1, min(int(carousel_K), K))
        slots, T = schedule_lpt(nsteps_per_col, K_pipe)
        dX_c, ri_c, phi_init, sched = compile_carousel_schedule(
            paths, slots, T, phi0_mat.shape[0], phi0_mat
        )
        sum_ns = int(nsteps_per_col.sum())
        waste = 1.0 - sum_ns / float(T * K_pipe) if (T * K_pipe) else 0.0
        info(
            2,
            f"solve_batch: carousel route K={K} K_pipe={K_pipe} T={T} "
            f"sum_nsteps={sum_ns} waste={waste * 100:.2f}%",
        )
        sol = run._run_etd2_carousel(
            dX_c,
            ri_c,
            phi_init,
            sched,
            phi0_mat,
            dtype=dtype,
            sec_ops=sec_ops,
        )
        grid_sol = None
        legacy = (sol, nsteps_per_col)

    info(2, f"solve_batch: total wall {time() - start:.2f}s")

    return MCEqBatchResult(
        run,
        sol,
        grid_sol=grid_sol,
        int_grid=int_grid,
        nsteps_per_col=nsteps_per_col,
        conditions=conditions,
        legacy_tuple=legacy,
    )


def solve_multirhs(
    run,
    phi0_matrix,
    int_grid=None,
    grid_var="X",
    *,
    dtype=np.float64,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
):
    """Propagate K independent initial conditions through one shared
    ETD2 operator.

    .. deprecated::
        Thin wrapper around :meth:`solve_batch` with
        ``conditions=None``. Prefer :meth:`solve_batch`, which
        additionally handles per-column directions/atmospheres and
        returns an :class:`MCEqBatchResult` with named-spectrum
        extraction. Kept for backwards compatibility.

    Args:
      phi0_matrix (np.ndarray[dim_states, K]): initial state matrix.
        Each column carries one independent initial spectrum.
      int_grid, grid_var, dtype, X_start, eps, dX_max, dX_min,
      fd_span: see :meth:`solve_batch`.

    Returns:
      (np.ndarray[dim_states, K], np.ndarray[len(int_grid), dim_states, K]):
      final state matrix and stacked snapshots.
    """
    phi0_matrix = np.asarray(phi0_matrix)
    if phi0_matrix.ndim != 2:
        raise ValueError(
            f"solve_multirhs: phi0_matrix must be 2-D (dim_states, K), "
            f"got shape {phi0_matrix.shape}"
        )
    res = run.solve_batch(
        phi0_matrix,
        None,
        int_grid,
        grid_var,
        dtype=dtype,
        X_start=X_start,
        eps=eps,
        dX_max=dX_max,
        dX_min=dX_min,
        fd_span=fd_span,
    )
    return res.sol, res.grid_sol


def solve_fullsky(
    run,
    zenith_grid,
    azimuth_grid=None,
    phi0=None,
    *,
    carousel_K=None,
    dtype=np.float64,
    X_start=None,
    eps=None,
    dX_max=None,
    dX_min=None,
    fd_span=None,
    return_pixel_index=False,
    path_workers=0,
    geomagnetic_cutoff=None,
    cutoff_kwargs=None,
):
    """Propagate phi0 through every (zenith, azimuth) pixel of a sky grid.

    Builds a per-pixel integration path (zenith- and azimuth-dependent
    ``dX``/``rho_inv``/``nsteps``) and runs a Stage-5 LPT static
    carousel: pixels are scheduled into ``K_pipe`` pipeline slots so
    the total kernel work is ``Σ nsteps × ms/RHS`` rather than
    ``max(nsteps) × K``. Wired for all four backends
    (``numpy_etd2``, ``cuda_etd2``, ``mkl_etd2``, ``accelerate_etd2``).

    Args:
        zenith_grid: 1-D zenith angles in degrees.
        azimuth_grid: 1-D azimuth angles in degrees, or ``None`` for
            zenith-only.
        phi0: initial spectrum, either ``(dim_states,)`` (broadcast
            across pixels) or ``(dim_states, K)`` (per-pixel — column
            order matches the ``(i_zen, i_az)`` flattening with
            azimuth as the inner axis). ``None`` reuses
            ``run._phi0``.
        carousel_K: pipeline width for the LPT scheduler. ``None``
            picks the default ``min(K, 128)``, which is the sweet
            spot across the full-sky benchmarks for K ≤ 2664.
        X_start, eps, dX_max, dX_min, fd_span: per-pixel path-builder
            knobs.
        return_pixel_index: also return the ``(K, 2)`` mapping
            ``(i_zen, i_az)`` for reshaping back to grid.
        path_workers: fork-pool size for parallel path build. Any
            value >= 1 is fine on every atmosphere.
        geomagnetic_cutoff: override the constructor-level toggle for
            this call. ``None`` means "use ``run.geomagnetic_cutoff``"
            (auto-detect from the atmosphere if also ``None``).
        cutoff_kwargs: optional dict forwarded to
            :func:`MCEq.geometry.gtracr_cutoff.get_cutoff_map`
            (e.g. ``{"iter_num": 30000, "bfield_type": "igrf"}``).

    Returns:
        :class:`MCEqBatchResult` — final state per pixel
        ``(dim_states, K)`` plus the sky grid, with per-pixel
        named-spectrum extraction (:meth:`MCEqBatchResult.get_solution`,
        :meth:`MCEqBatchResult.skymap`). Also unpacks as the legacy
        ``(sol, nsteps_per_col[, pixel_index])`` tuple.
    """
    info(2, f"solve_fullsky: kernel={run.config.backend.kernel_config}")
    start = time()

    # Resolve geomagnetic-cutoff toggle. Per-call argument has
    # priority; otherwise fall back to instance default; otherwise
    # auto-detect from the atmosphere.
    cutoff_flag = geomagnetic_cutoff
    if cutoff_flag is None:
        cutoff_flag = run.geomagnetic_cutoff
    if cutoff_flag is None:
        cutoff_flag = run._is_geomag_eligible_atmosphere()

    zenith_grid = np.asarray(zenith_grid, dtype=np.float64).reshape(-1)
    if azimuth_grid is not None:
        azimuth_grid = np.asarray(azimuth_grid, dtype=np.float64).reshape(-1)
    n_zen = zenith_grid.size
    n_az = azimuth_grid.size if azimuth_grid is not None else 1
    K = n_zen * n_az
    if K < 1:
        raise ValueError("solve_fullsky: empty (zenith, azimuth) grid")

    if phi0 is None:
        phi0_arr = run._phi0.copy()
        phi0_is_2d = False
    else:
        phi0_arr = np.asarray(phi0, dtype=np.float64)
        if phi0_arr.ndim == 1:
            if phi0_arr.size != run.dim_states:
                raise ValueError(
                    f"solve_fullsky: phi0 has length {phi0_arr.size}, "
                    f"expected {run.dim_states}"
                )
            phi0_is_2d = False
        elif phi0_arr.ndim == 2:
            if phi0_arr.shape[0] != run.dim_states:
                raise ValueError(
                    f"solve_fullsky: phi0 has shape {phi0_arr.shape}, "
                    f"expected first axis = dim_states = {run.dim_states}"
                )
            phi0_is_2d = True
        else:
            raise ValueError(
                f"solve_fullsky: phi0 must be 1-D or 2-D, got shape {phi0_arr.shape}"
            )

    if phi0_is_2d and phi0_arr.shape[1] != K:
        raise ValueError(
            f"solve_fullsky: phi0 has shape {phi0_arr.shape}, expected "
            f"second axis = K = {K}"
        )

    # Apply geomagnetic rigidity cutoff per pixel if requested. Skip
    # when phi0 was already supplied as 2-D (caller is in charge of
    # the per-pixel primary spectrum then) — warn if the cutoff was
    # requested explicitly in that case, since silently combining
    # both would double-apply per-pixel physics.
    if phi0_is_2d and (
        geomagnetic_cutoff is True
        or (geomagnetic_cutoff is None and run.geomagnetic_cutoff is True)
    ):
        import warnings

        warnings.warn(
            "solve_fullsky: 2-D phi0 supplied — the geomagnetic cutoff "
            "is NOT applied on top (the caller owns the per-pixel "
            "primary spectrum). Bake the cutoff into phi0 with "
            "MCEq.geometry.gtracr_cutoff.build_phi0_with_cutoff, or "
            "pass geomagnetic_cutoff=False to silence this warning.",
            stacklevel=3,
        )
    if cutoff_flag and not phi0_is_2d:
        from MCEq.environment.geomagnetic.cutoff import (
            build_phi0_with_cutoff,
            get_cutoff_map,
        )

        az_centres = azimuth_grid if azimuth_grid is not None else np.array([0.0])
        primary = getattr(run, "pmodel", None)
        if primary is None:
            info(
                1,
                "solve_fullsky: geomagnetic_cutoff requested but no "
                "primary_model is set — skipping cutoff.",
            )
        else:
            ck = dict(cutoff_kwargs or {})
            rc_grid = get_cutoff_map(
                run.density_model,
                zenith_grid,
                az_centres,
                **ck,
            )
            # Pixel order: (i_zen, i_az) flattened with az inner.
            rc_flat = rc_grid.flatten(order="C")
            if rc_flat.size != K:
                raise RuntimeError(
                    f"solve_fullsky: cutoff map size {rc_flat.size} "
                    f"does not match K={K}"
                )
            phi0_arr = build_phi0_with_cutoff(run, primary, rc_flat)
            phi0_is_2d = True
            info(
                2,
                f"solve_fullsky: applied per-pixel R_c cutoff "
                f"(R_c range [{rc_grid.min():.2f}, {rc_grid.max():.2f}] GV)",
            )

    # Expand the sky grid into per-pixel batch conditions (azimuth
    # as the inner axis) and delegate to the general batched solver.
    conditions = []
    pixel_index = np.empty((K, 2), dtype=np.int32)
    k = 0
    for i_zen, zen in enumerate(zenith_grid):
        for i_az in range(n_az):
            cond = {"zenith_deg": float(zen)}
            if azimuth_grid is not None:
                cond["azimuth_deg"] = float(azimuth_grid[i_az])
            conditions.append(cond)
            pixel_index[k] = (i_zen, i_az)
            k += 1

    res = run.solve_batch(
        phi0_arr,
        conditions,
        dtype=dtype,
        carousel_K=carousel_K,
        path_workers=path_workers,
        X_start=X_start,
        eps=eps,
        dX_max=dX_max,
        dX_min=dX_min,
        fd_span=fd_span,
    )

    # Decorate the batch result with the sky-grid metadata and the
    # legacy solve_fullsky return tuple.
    res.pixel_index = pixel_index
    res.zenith_grid = zenith_grid
    res.azimuth_grid = azimuth_grid
    if return_pixel_index:
        res._legacy = (res.sol, res.nsteps_per_col, pixel_index)
    else:
        res._legacy = (res.sol, res.nsteps_per_col)

    info(2, f"solve_fullsky: total wall {time() - start:.2f}s")
    return res


# --- the ETD2RK step loop: operator assembly, backends, routes -----------
#
# ``compile_operator`` turns ``int_m`` / ``dec_m`` (and the secant
# operator set) into one CompiledOperator; a backend of
# ``config.kernel_config`` binds it; ``etd2_driver`` runs every route on
# it. Both objects are cached here against the identity of their inputs.
# fp32 is a dtype the backends carry, not a kernel family of its own.
