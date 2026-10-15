"""Batched solves of :class:`MCEq.core.MCEqRun`: many initial states and/or directions in one call."""

import multiprocessing
import warnings
from time import time

import numpy as np

from MCEq import config
from MCEq.geometry.cutoff import get_cutoff_map
from MCEq.initial_state import fill_nucleons
from MCEq.misc import info
from MCEq.solvers import compile_carousel_schedule, schedule_lpt

#: Mass groups ``(crflux id, Z, A)`` of the primary models for the cutoff.
_DEFAULT_MASS_GROUPS = (
    (14, 1, 1),  # p
    (402, 2, 4),  # He
    (1206, 6, 12),  # C-N-O
    (2814, 14, 28),  # Si
    (5426, 26, 56),  # Fe-56; the id is crflux's group key, not A*100+Z
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


def build_phi0_with_cutoff(
    mceq, primary_model, rc_GV_per_pixel, mass_groups=_DEFAULT_MASS_GROUPS
):
    """``(dim_states, K)`` initial states of ``primary_model`` with the
    rigidity cutoff ``rc_GV_per_pixel`` of each column.

    A mass group ``(Z, A)`` contributes where ``A * E_nuc > Z * R_c``
    (ultrarelativistic, ``R ≈ E / Z``) and is split into nucleons by
    superposition as in :meth:`MCEqRun.set_primary_model`.
    """
    K = int(rc_GV_per_pixel.size)
    e_grid = mceq._energy_grid.c
    p_mass = mceq.pman[(2212, 0)].mass
    n_mass = mceq.pman[(2112, 0)].mass
    e_tot_nucleon = e_grid + 0.5 * (p_mass + n_mass)
    min_idx = int(np.argmin(np.abs(e_tot_nucleon - config.minimal_primary_energy)))

    p_flux = np.zeros((e_grid.size, K))
    n_flux = np.zeros((e_grid.size, K))
    for corsika_id, Z, A in mass_groups:
        if corsika_id not in primary_model.nucleus_ids:
            continue
        E_nucleus_tot = e_tot_nucleon * A
        phi_nuc = np.asarray(
            primary_model.nucleus_flux(corsika_id, E_nucleus_tot), dtype=np.float64
        )
        mask = (E_nucleus_tot[:, None] > Z * rc_GV_per_pixel[None, :]).astype(
            np.float64
        )
        p_flux += Z * A * (phi_nuc[:, None] * mask)
        n_flux += (A - Z) * A * (phi_nuc[:, None] * mask)

    phi0 = np.zeros((mceq.dim_states, K))
    fill_nucleons(phi0, mceq.pman, min_idx, p_flux[min_idx:], n_flux[min_idx:])
    return phi0


def resolve_dtype(dtype=None, where="solve"):
    """``dtype`` if given, else float32 when the kernel is CUDA and
    ``config.cuda_fp_precision`` is 32, else float64."""
    if dtype is None:
        on_cuda = config.kernel_config.lower() in ("cuda", "cuda_etd2")
        fp = int(config.cuda_fp_precision) if on_cuda else 64
        if fp not in (32, 64):
            raise ValueError(f"config.cuda_fp_precision must be 32 or 64, got {fp}")
        dtype = np.float32 if fp == 32 else np.float64
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"{where}: dtype must be float32 or float64, got {dtype}")
    return dtype


class MCEqBatchResult:
    """Result of :meth:`MCEqRun.solve_batch` and :meth:`MCEqRun.solve_fullsky`.

    Attributes:
      sol (np.ndarray[n_k * dim_states, K]): final state per batch member
        (``n_k = 1`` in 1D runs).
      grid_sol (np.ndarray[len(int_grid), n_k * dim_states, K] | None): depth
        snapshots (shared-path batches with ``int_grid`` only).
      nsteps_per_col (np.ndarray[K] | None): integration steps per column.
      shared_path (bool): every column was solved along one path.
      conditions (list | None): the per-column conditions of the solve.
      pixel_index (np.ndarray[K, 2] | None): ``(i_zen, i_az)`` per column of
        a full-sky solve.
      zenith_grid, azimuth_grid (np.ndarray | None): the sky grid of a
        full-sky solve; azimuth is the inner axis of the column order.

    The result reads particle names from the run that produced it; changing
    that run's particle list invalidates :meth:`get_solution`.
    """

    def __init__(
        self,
        mceq,
        sol,
        grid_sol=None,
        int_grid=None,
        nsteps_per_col=None,
        conditions=None,
        pixel_index=None,
        zenith_grid=None,
        azimuth_grid=None,
        shared_path=None,
    ):
        self._mceq = mceq
        self.sol = sol
        self.grid_sol = grid_sol
        self.int_grid = int_grid
        self.nsteps_per_col = nsteps_per_col
        self.conditions = conditions
        self.pixel_index = pixel_index
        self.zenith_grid = zenith_grid
        self.azimuth_grid = azimuth_grid
        self.shared_path = None if shared_path is None else bool(shared_path)

    @property
    def K(self):
        """Number of batch members (columns)."""
        return self.sol.shape[1]

    @property
    def n_azimuth(self):
        """Length of the azimuth (inner) axis of the column order."""
        return self.azimuth_grid.size if self.azimuth_grid is not None else 1

    def column_index(self, k=None, pixel=None, zenith=None, azimuth=None):
        """Column of a batch member, selected by one of ``k``, ``pixel``
        (``(i_zen, i_az)``) or ``zenith`` (+ ``azimuth``) in degrees; none is
        needed for ``K == 1``."""
        n_selected = sum(x is not None for x in (k, pixel, zenith))
        if n_selected > 1:
            raise ValueError("column_index: provide only one of k, pixel, or zenith")
        if k is not None:
            k = int(k)
            if not -self.K <= k < self.K:
                raise IndexError(f"column_index: k={k} out of range for K={self.K}")
            return k % self.K

        if pixel is not None or zenith is not None:
            if self.zenith_grid is None:
                raise ValueError(
                    "column_index: pixel/zenith selection requires a "
                    "solve_fullsky result (no sky grid attached)"
                )
        if pixel is not None:
            i_zen, i_az = pixel
            n_zen = self.zenith_grid.size
            if not (0 <= i_zen < n_zen and 0 <= i_az < self.n_azimuth):
                raise IndexError(
                    f"column_index: pixel {pixel} outside grid "
                    f"({n_zen} x {self.n_azimuth})"
                )
            return int(i_zen) * self.n_azimuth + int(i_az)
        if zenith is not None:
            match = np.flatnonzero(np.isclose(self.zenith_grid, zenith))
            if match.size == 0:
                raise ValueError(
                    f"column_index: zenith {zenith} not in grid {self.zenith_grid}"
                )
            i_zen = int(match[0])
            if azimuth is None:
                if self.n_azimuth > 1:
                    raise ValueError(
                        "column_index: azimuth required (grid has "
                        f"{self.n_azimuth} azimuth pixels)"
                    )
                i_az = 0
            else:
                match_az = np.flatnonzero(np.isclose(self.azimuth_grid, azimuth))
                if match_az.size == 0:
                    raise ValueError(
                        f"column_index: azimuth {azimuth} not in grid "
                        f"{self.azimuth_grid}"
                    )
                i_az = int(match_az[0])
            return i_zen * self.n_azimuth + i_az

        if self.K == 1:
            return 0
        raise ValueError(
            f"column_index: batch has K={self.K} members - select one "
            "with k=, pixel=, or zenith="
        )

    def get_solution(
        self,
        particle_name,
        k=None,
        *,
        pixel=None,
        zenith=None,
        azimuth=None,
        mag=0.0,
        grid_idx=None,
        integrate=False,
        return_as=None,
        dont_sum_helicities=False,
    ):
        """Spectrum of ``particle_name`` for one batch member (see
        :meth:`column_index`), as :meth:`MCEqRun.get_solution`."""
        col = self.column_index(k=k, pixel=pixel, zenith=zenith, azimuth=azimuth)
        if grid_idx is None:
            state = self.sol[:, col]
        else:
            if self.grid_sol is None or len(self.grid_sol) == 0:
                raise Exception(
                    "Solution has not been computed on a grid. Re-run with int_grid."
                )
            state = self.grid_sol[min(grid_idx, len(self.grid_sol) - 1)][:, col]
        return self._mceq._get_solution_from_state(
            state,
            particle_name,
            mag=mag,
            integrate=integrate,
            return_as=return_as,
            dont_sum_helicities=dont_sum_helicities,
        )

    def skymap(self, particle_name, kin_energy, mag=0.0):
        """``(n_zen, n_az)`` flux map of a full-sky solve at ``kin_energy``,
        interpolated linearly in log E between the grid points."""
        if self.zenith_grid is None:
            raise ValueError("skymap() is only available on solve_fullsky results")
        e_grid = self._mceq.e_grid
        if not e_grid[0] <= kin_energy <= e_grid[-1]:
            raise ValueError(
                f"skymap: kin_energy {kin_energy} outside energy grid "
                f"[{e_grid[0]:.3g}, {e_grid[-1]:.3g}] GeV"
            )
        i_hi = int(np.clip(np.searchsorted(e_grid, kin_energy), 1, e_grid.size - 1))
        i_lo = i_hi - 1
        w = (np.log(kin_energy) - np.log(e_grid[i_lo])) / (
            np.log(e_grid[i_hi]) - np.log(e_grid[i_lo])
        )
        flux = np.empty((self.zenith_grid.size, self.n_azimuth))
        for k in range(self.K):
            f = self.get_solution(
                particle_name, k=k, mag=mag, return_as="kinetic energy"
            )
            i_zen, i_az = divmod(k, self.n_azimuth)
            flux[i_zen, i_az] = (1.0 - w) * f[i_lo] + w * f[i_hi]
        return flux

    def __repr__(self):
        parts = [f"K={self.sol.shape[1]}"]
        if self.zenith_grid is not None:
            parts.append(f"sky_grid={self.zenith_grid.size}x{self.n_azimuth}")
        if self.grid_sol is not None and len(self.grid_sol):
            parts.append(f"n_snapshots={len(self.grid_sol)}")
        return f"MCEqBatchResult({', '.join(parts)})"


class BatchSolver:
    """The batched solves of :class:`MCEq.core.MCEqRun`."""

    def _warn_low_energy_cuda_fp32(self, dtype, sec_ops):
        """Warn on the CUDA float32 low-energy 2D secant precision regime."""
        if (
            dtype == np.dtype(np.float32)
            and config.kernel_config.lower() in ("cuda", "cuda_etd2")
            and self._mceq_db.is_2d
            and sec_ops is not None
            and self.e_bins[0] <= 0.1 * (1.0 + 1e-12)
        ):
            warnings.warn(
                "CUDA FP32 with low-energy 2D secant transport can lose percent-level "
                "angular accuracy; compare with dtype=np.float64 on the same "
                "integration path.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _phi0_columns(self, phi0, K, where):
        """``phi0`` (``None``: the run's, ``(dim_states,)`` or
        ``(dim_states, K)``) as a float64 array, checked against ``K``
        (``None``: any column count)."""
        arr = self._phi0.copy() if phi0 is None else np.asarray(phi0, np.float64)
        if arr.ndim not in (1, 2):
            raise ValueError(f"{where}: phi0 must be 1-D or 2-D, got shape {arr.shape}")
        if arr.shape[0] != self.dim_states:
            raise ValueError(
                f"{where}: phi0 has shape {arr.shape}, "
                f"expected first axis = dim_states = {self.dim_states}"
            )
        if arr.ndim == 2 and K is not None and arr.shape[1] != K:
            raise ValueError(
                f"{where}: phi0 has shape {arr.shape}, expected second axis = K = {K}"
            )
        return arr

    def solve_batch(
        self,
        phi0=None,
        conditions=None,
        int_grid=None,
        grid_var="X",
        *,
        dtype=None,
        carousel_K=None,
        path_workers=0,
        X_start=None,
        eps=None,
        dX_max=None,
        dX_min=None,
        fd_span=None,
    ):
        """Solve K independent cascade problems with one operator.

        The batch members are the columns of ``phi0`` and/or the entries of
        ``conditions``:

        * ``conditions=None``: all columns share the current direction and
          atmosphere and one integration path; ``int_grid`` snapshots are
          supported.
        * ``conditions=[...]``: one dict per member with optional keys
          ``zenith_deg``, ``azimuth_deg`` and ``density_model`` (config tuple
          or instance); missing values use the current settings, a missing
          azimuth means azimuth-averaged. One path is built per distinct
          condition and the batch runs on the LPT carousel, so the work is
          ``sum(nsteps)`` instead of ``max(nsteps) * K``.

        The run's initial state, solution and direction are left unchanged.

        Examples::

            # one spectrum through 12 months x 3 zeniths
            conditions = [
                {"zenith_deg": z, "density_model": ("MSIS21", ("SouthPole", month))}
                for month in months
                for z in (0.0, 30.0, 60.0)
            ]
            res = mceq.solve_batch(conditions=conditions)
            res.get_solution("conv_numu", k=5, mag=3)

        Args:
          phi0 (np.ndarray | None): ``None`` uses the run's initial state;
            ``(dim_states,)`` is broadcast, ``(dim_states, K)`` gives one
            column per member.
          conditions (list[dict] | None): per-member direction and atmosphere.
          int_grid (list | None): X values of the snapshots (``conditions=None`` only).
          grid_var (str): only ``"X"`` is supported.
          dtype (np.float32 | np.float64 | None): precision of the state
            buffers, default as in :meth:`solve`.
          carousel_K (int | None): pipeline width, default ``min(K, 128)``.
          path_workers (int): fork-pool size for building the paths; not with
            ``density_model`` conditions.
          X_start, eps, dX_max, dX_min, fd_span: path parameters as in :meth:`solve`.

        Returns:
          :class:`MCEqBatchResult`
        """
        dtype = resolve_dtype(dtype, "solve_batch")
        sec_ops = self._resolve_secant()
        self._warn_low_energy_cuda_fp32(dtype, sec_ops)

        if conditions is not None and len(conditions) < 1:
            raise ValueError("solve_batch: conditions must not be empty")
        phi0_arr = self._phi0_columns(
            phi0, None if conditions is None else len(conditions), "solve_batch"
        )
        if conditions is not None:
            K = len(conditions)
        else:
            K = phi0_arr.shape[1] if phi0_arr.ndim == 2 else 1
        phi0_mat = np.ascontiguousarray(
            np.broadcast_to(phi0_arr.reshape(self.dim_states, -1), (self.dim_states, K))
        )
        phi0_mat = self._stack_modes(phi0_mat)

        path_kwargs = dict(
            X_start=X_start, eps=eps, dX_max=dX_max, dX_min=dX_min, fd_span=fd_span
        )
        if conditions is None:
            if int_grid is not None and np.any(np.diff(int_grid) < 0):
                raise Exception(
                    "The X values in int_grid are required to be strictly increasing."
                )
            self._calculate_integration_path(int_grid, grid_var, **path_kwargs)
            paths = [self.integration_path] * K
        else:
            if int_grid is not None:
                raise NotImplementedError(
                    "solve_batch: int_grid snapshots are only supported for "
                    "shared-path batches (conditions=None)."
                )
            if grid_var != "X":
                raise NotImplementedError(
                    "solve_batch: only grid_var='X' is supported."
                )
            paths = self._build_condition_paths(
                conditions, path_workers=path_workers, **path_kwargs
            )

        nsteps_per_col = np.array([p[0] for p in paths], dtype=np.int32)
        shared = all(p is paths[0] for p in paths)

        start = time()
        if shared:
            nsteps, dX, rho_inv, grid_idcs = paths[0]
            info(2, f"solve_batch: shared path, K={K}, nsteps={nsteps}")
            sol, grid_sol = self._run_etd2(
                nsteps,
                dX,
                rho_inv,
                phi0_mat.astype(dtype, copy=True),
                grid_idcs,
                dtype=dtype,
                sec_ops=sec_ops,
            )
        else:
            K_pipe = max(
                1, min(int(min(K, 128) if carousel_K is None else carousel_K), K)
            )
            slots, T = schedule_lpt(nsteps_per_col, K_pipe)
            dX_c, ri_c, phi_init, sched = compile_carousel_schedule(
                paths, slots, T, phi0_mat.shape[0], phi0_mat
            )
            sum_ns = int(nsteps_per_col.sum())
            waste = 1.0 - sum_ns / float(T * K_pipe) if (T * K_pipe) else 0.0
            info(
                2,
                f"solve_batch: carousel K={K} K_pipe={K_pipe} T={T} "
                f"sum_nsteps={sum_ns} waste={waste * 100:.2f}%",
            )
            sol = self._run_etd2_carousel(
                dX_c, ri_c, phi_init, sched, phi0_mat, dtype=dtype, sec_ops=sec_ops
            )
            grid_sol = None
        info(2, f"solve_batch: total wall {time() - start:.2f}s")

        return MCEqBatchResult(
            self,
            sol,
            grid_sol=grid_sol,
            int_grid=int_grid,
            nsteps_per_col=nsteps_per_col,
            conditions=conditions,
            shared_path=shared,
        )

    def solve_fullsky(
        self,
        zenith_grid,
        azimuth_grid=None,
        phi0=None,
        *,
        carousel_K=None,
        dtype=None,
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
        """Solve every (zenith, azimuth) pixel of a sky grid with :meth:`solve_batch`.

        Args:
            zenith_grid: zenith angles in degrees.
            azimuth_grid: azimuth angles in degrees, or ``None`` for zenith only.
            phi0: ``(dim_states,)`` (all pixels) or ``(dim_states, K)``, columns
              in ``(i_zen, i_az)`` order with azimuth inner; ``None`` uses the
              run's initial state.
            carousel_K, dtype, X_start, eps, dX_max, dX_min, fd_span, path_workers:
              as in :meth:`solve_batch`.
            return_pixel_index: also return the ``(K, 2)`` ``(i_zen, i_az)``
              of the columns.
            geomagnetic_cutoff: apply the rigidity cutoff per pixel; ``None``
              uses the run setting, which by default is on for MSIS and
              location-tagged atmospheres.
            cutoff_kwargs: keywords of
              :func:`MCEq.geometry.cutoff.get_cutoff_map`.

        Returns:
            :class:`MCEqBatchResult`, or ``(result, pixel_index)``.
        """
        info(2, f"solve_fullsky: kernel={config.kernel_config}")
        start = time()

        cutoff_flag = geomagnetic_cutoff
        if cutoff_flag is None:
            cutoff_flag = self.geomagnetic_cutoff
        if cutoff_flag is None:
            cutoff_flag = self._is_geomag_eligible_atmosphere()

        zenith_grid = np.asarray(zenith_grid, dtype=np.float64).reshape(-1)
        if azimuth_grid is not None:
            azimuth_grid = np.asarray(azimuth_grid, dtype=np.float64).reshape(-1)
        n_zen = zenith_grid.size
        n_az = azimuth_grid.size if azimuth_grid is not None else 1
        K = n_zen * n_az
        if K < 1:
            raise ValueError("solve_fullsky: empty (zenith, azimuth) grid")

        phi0_arr = self._phi0_columns(phi0, K, "solve_fullsky")
        phi0_is_2d = phi0_arr.ndim == 2
        if phi0_is_2d and (
            geomagnetic_cutoff is True
            or (geomagnetic_cutoff is None and self.geomagnetic_cutoff is True)
        ):
            warnings.warn(
                "solve_fullsky: the geomagnetic cutoff is not applied to a 2-D "
                "phi0; build it with MCEq.batch.build_phi0_with_cutoff, or pass "
                "geomagnetic_cutoff=False to silence this warning.",
                stacklevel=2,
            )
        if cutoff_flag and not phi0_is_2d:
            if self.pmodel is None:
                info(1, "solve_fullsky: no primary model, skipping the cutoff.")
            else:
                rc_grid = get_cutoff_map(
                    self.density_model,
                    zenith_grid,
                    azimuth_grid if azimuth_grid is not None else np.array([0.0]),
                    **dict(cutoff_kwargs or {}),
                )
                rc_flat = rc_grid.flatten(order="C")  # azimuth inner
                if rc_flat.size != K:
                    raise RuntimeError(
                        f"solve_fullsky: cutoff map size {rc_flat.size} "
                        f"does not match K={K}"
                    )
                phi0_arr = build_phi0_with_cutoff(self, self.pmodel, rc_flat)
                info(
                    2,
                    f"solve_fullsky: per-pixel cutoff, R_c in "
                    f"[{rc_grid.min():.2f}, {rc_grid.max():.2f}] GV",
                )

        conditions = []
        pixel_index = np.empty((K, 2), dtype=np.int32)
        for k, (i_zen, i_az) in enumerate(np.ndindex(n_zen, n_az)):
            cond = {"zenith_deg": float(zenith_grid[i_zen])}
            if azimuth_grid is not None:
                cond["azimuth_deg"] = float(azimuth_grid[i_az])
            conditions.append(cond)
            pixel_index[k] = (i_zen, i_az)

        res = self.solve_batch(
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
        res.pixel_index = pixel_index
        res.zenith_grid = zenith_grid
        res.azimuth_grid = azimuth_grid

        info(2, f"solve_fullsky: total wall {time() - start:.2f}s")
        if return_pixel_index:
            return res, pixel_index
        return res

    def _build_condition_paths(
        self,
        conditions,
        *,
        X_start=None,
        eps=None,
        dX_max=None,
        dX_min=None,
        fd_span=None,
        path_workers=0,
    ):
        """One integration path per condition (see :meth:`solve_batch`).

        Conditions with the same density model and zenith, and the same
        azimuth where the model depends on it, share one path tuple. The
        active density model and angles are restored.
        """
        allowed_keys = {"zenith_deg", "azimuth_deg", "density_model"}
        norm = []
        for i, c in enumerate(conditions):
            c = {} if c is None else c
            if not isinstance(c, dict):
                raise TypeError(
                    f"_build_condition_paths: condition {i} must be a dict "
                    f"with keys in {sorted(allowed_keys)}, got {type(c).__name__}"
                )
            unknown = set(c) - allowed_keys
            if unknown:
                raise ValueError(
                    f"_build_condition_paths: condition {i} has unknown "
                    f"keys {sorted(unknown)}; allowed: {sorted(allowed_keys)}"
                )
            norm.append(dict(c))

        n_workers = int(path_workers) if path_workers else 0
        if n_workers > 1:
            if any(c.get("density_model") is not None for c in norm):
                raise ValueError(
                    "path_workers > 1 supports only zenith/azimuth batches "
                    "on the active atmosphere; build density_model "
                    "overrides serially (path_workers=0)."
                )
            if "fork" not in multiprocessing.get_all_start_methods():
                n_workers = 0

        def dm_key_of(c):
            dm_spec = c.get("density_model")
            if dm_spec is None:
                return ("current",)
            if isinstance(dm_spec, (tuple, list)):
                return ("cfg", repr(tuple(dm_spec)))
            return ("obj", id(dm_spec))

        saved_dm = self.density_model
        saved_zen = getattr(saved_dm, "theta_deg", None)
        saved_az = getattr(saved_dm, "_current_azimuth_deg", None)
        kwargs = dict(
            X_start=X_start, eps=eps, dX_max=dX_max, dX_min=dX_min, fd_span=fd_span
        )
        try:
            # Each distinct density model is built once.
            dm_instances = {}
            for c in norm:
                key = dm_key_of(c)
                if key in dm_instances:
                    continue
                dm_spec = c.get("density_model")
                if dm_spec is None:
                    dm_instances[key] = saved_dm
                elif isinstance(dm_spec, (tuple, list)):
                    self.set_density_model(tuple(dm_spec))
                    dm_instances[key] = self.density_model
                else:
                    dm_instances[key] = dm_spec

            # Azimuth enters the key only where the model depends on it.
            cond_keys = []
            for c in norm:
                dm_key = dm_key_of(c)
                zen = c.get("zenith_deg")
                if zen is None:
                    zen = saved_zen
                if zen is None:
                    raise ValueError(
                        "_build_condition_paths: condition without "
                        "'zenith_deg' and no zenith set on the instance"
                    )
                az = c.get("azimuth_deg")
                az_dep = getattr(dm_instances[dm_key], "depends_on_azimuth", False)
                az_eff = float(az) if (az is not None and az_dep) else None
                cond_keys.append((dm_key, float(zen), az_eff))

            # Jobs grouped by density model, so per-model state is reused.
            jobs = sorted(dict.fromkeys(cond_keys), key=repr)
            unique_paths = {}
            if n_workers > 1 and len(jobs) > 1:
                global _PATH_WORKER_MCEQ
                _PATH_WORKER_MCEQ = self  # inherited by forked children
                try:
                    ctx = multiprocessing.get_context("fork")
                    worker_args = [
                        (idx, zen, az, kwargs) for idx, (_, zen, az) in enumerate(jobs)
                    ]
                    chunksize = max(1, len(jobs) // (n_workers * 8))
                    with ctx.Pool(n_workers) as pool:
                        for flat_idx, path in pool.imap_unordered(
                            _path_worker_one, worker_args, chunksize=chunksize
                        ):
                            unique_paths[jobs[flat_idx]] = path
                finally:
                    _PATH_WORKER_MCEQ = None
            else:
                for pkey in jobs:
                    dm_key, zen, az_eff = pkey
                    if self.density_model is not dm_instances[dm_key]:
                        self.set_density_model(dm_instances[dm_key])
                    if az_eff is None:
                        self.set_zenith_azimuth(zen)
                    else:
                        self.set_zenith_azimuth(zen, az_eff)
                    self._calculate_integration_path(None, "X", **kwargs)
                    unique_paths[pkey] = self.integration_path

            return [unique_paths[k] for k in cond_keys]
        finally:
            if self.density_model is not saved_dm:
                self.set_density_model(saved_dm)
            if saved_zen is not None:
                self.set_zenith_azimuth(saved_zen, saved_az)
            self.integration_path = None
