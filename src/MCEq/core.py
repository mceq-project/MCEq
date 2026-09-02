from itertools import product
from time import time

import numpy as np
import scipy.sparse as sp

import MCEq.data
from MCEq import config
from MCEq.download import ensure_db_available
from MCEq.misc import info, normalize_hadronic_model_name
from MCEq.operator_assembly import compile_operator
from MCEq.particlemanager import ParticleManager

# trapz was finally removed with numpy 2.4
if hasattr(np, "trapezoid"):
    trapz = np.trapezoid
else:
    trapz = np.trapz


# Module-level worker state for the optional process-pool path build
# inside :meth:`MCEqRun._build_condition_paths`. Workers fork from the parent
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


class MCEqBatchResult:
    """Result of a batched (multi-RHS) solve.

    Returned by :meth:`MCEqRun.solve_batch` and
    :meth:`MCEqRun.solve_fullsky`. Wraps the raw ``(dim_states, K)``
    final-state matrix together with the batch metadata and provides
    named per-column spectrum extraction through :meth:`get_solution`,
    with exactly the same particle-name semantics as
    :meth:`MCEqRun.get_solution`.

    For backwards compatibility the object also unpacks like the legacy
    return tuple of the method that produced it, e.g.::

        sol, nsteps_per_col = mceq.solve_fullsky(zen_grid, az_grid)
        sol, grid_sol = mceq.solve_batch(phi0_matrix)

    Attributes:
      sol (np.ndarray[dim_states, K]): final state, one column per batch
        member.
      grid_sol (np.ndarray[len(int_grid), dim_states, K] | None): stacked
        snapshots when ``int_grid`` was requested (shared-path batches
        only), else ``None``.
      nsteps_per_col (np.ndarray[K] | None): integration steps per column.
      conditions (list | None): the per-column conditions passed to
        :meth:`MCEqRun.solve_batch`, if any.
      pixel_index (np.ndarray[K, 2] | None): ``(i_zen, i_az)`` per column
        for :meth:`MCEqRun.solve_fullsky` results.
      zenith_grid, azimuth_grid (np.ndarray | None): the sky grid for
        :meth:`MCEqRun.solve_fullsky` results (azimuth is the inner axis
        of the column order).

    Note:
      The result holds a reference to the producing :class:`MCEqRun`
      instance for particle-name lookups. Changing that instance's
      interaction model or particle list after the solve invalidates
      :meth:`get_solution` on this result.
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
        legacy_tuple=None,
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
        self._legacy = tuple(legacy_tuple) if legacy_tuple is not None else (sol,)

    @property
    def K(self):
        """Number of batch members (columns)."""
        return self.sol.shape[1]

    @property
    def n_azimuth(self):
        """Length of the azimuth (inner) axis of the column order."""
        return self.azimuth_grid.size if self.azimuth_grid is not None else 1

    def column_index(self, k=None, pixel=None, zenith=None, azimuth=None):
        """Resolve a batch member to its column index.

        Exactly one selector should be provided: ``k`` (direct column
        index), ``pixel`` (``(i_zen, i_az)`` grid coordinates, sky-grid
        results only) or ``zenith`` (+ optional ``azimuth``) in degrees,
        matched against the sky grid with ``np.isclose``. With ``K == 1``
        all selectors may be omitted.
        """
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
        """Retrieve a named spectrum for one batch member.

        Same ``particle_name`` semantics, helicity summation, ``mag``,
        ``integrate`` and ``return_as`` behaviour as
        :meth:`MCEqRun.get_solution`; the column is selected with ``k``,
        ``pixel`` or ``zenith``/``azimuth`` (see :meth:`column_index`).

        Args:
          particle_name (str): e.g. ``conv_numu``, ``total_mu+``.
          k (int, optional): column index.
          pixel (tuple, optional): ``(i_zen, i_az)`` sky-grid coordinates.
          zenith, azimuth (float, optional): angles in degrees, matched
            against the sky grid.
          grid_idx (int, optional): snapshot index when the batch was
            solved with ``int_grid`` (shared-path batches only).
          mag, integrate, return_as, dont_sum_helicities: see
            :meth:`MCEqRun.get_solution`.

        Returns:
          (np.ndarray): flux on the energy grid.
        """
        col = self.column_index(k=k, pixel=pixel, zenith=zenith, azimuth=azimuth)
        if grid_idx is None:
            state = self.sol[:, col]
        else:
            if self.grid_sol is None or len(self.grid_sol) == 0:
                raise Exception(
                    "Solution has not been computed on a grid. Re-run with int_grid."
                )
            if grid_idx >= len(self.grid_sol):
                state = self.grid_sol[-1][:, col]
            else:
                state = self.grid_sol[grid_idx][:, col]
        return self._mceq._get_solution_from_state(
            state,
            particle_name,
            mag=mag,
            integrate=integrate,
            return_as=return_as,
            dont_sum_helicities=dont_sum_helicities,
        )

    def skymap(self, particle_name, kin_energy, mag=0.0):
        """Return the ``(n_zen, n_az)`` flux map at one kinetic energy.

        Extracts ``particle_name`` for every pixel and interpolates
        linearly in log(E_kin) between the two bracketing grid points.
        Only available on :meth:`MCEqRun.solve_fullsky` results.

        Args:
          particle_name (str): see :meth:`get_solution`.
          kin_energy (float): kinetic energy in GeV; must lie within the
            energy grid.
          mag (float): energy magnification exponent, as in
            :meth:`get_solution`.

        Returns:
          (np.ndarray[n_zen, n_az]): flux map, kinetic-energy units.
        """
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
        n_zen = self.zenith_grid.size
        flux = np.empty((n_zen, self.n_azimuth))
        for k in range(self.K):
            f = self.get_solution(
                particle_name, k=k, mag=mag, return_as="kinetic energy"
            )
            i_zen, i_az = divmod(k, self.n_azimuth)
            flux[i_zen, i_az] = (1.0 - w) * f[i_lo] + w * f[i_hi]
        return flux

    # Legacy tuple compatibility -------------------------------------
    def __iter__(self):
        return iter(self._legacy)

    def __len__(self):
        return len(self._legacy)

    def __getitem__(self, item):
        return self._legacy[item]

    def __repr__(self):
        parts = [f"K={self.sol.shape[1]}"]
        if self.zenith_grid is not None:
            parts.append(f"sky_grid={self.zenith_grid.size}x{self.n_azimuth}")
        if self.grid_sol is not None and len(self.grid_sol):
            parts.append(f"n_snapshots={len(self.grid_sol)}")
        return f"MCEqBatchResult({', '.join(parts)})"


class MCEqRun:
    """Main class for handling the calculation.

    This class is the main user interface for the caclulation. It will
    handle initialization and various error/configuration checks. The
    setup has to be accomplished before invoking the integration routine
    is :func:`MCeqRun.solve`. Changes of configuration, such as:

    - interaction model in :meth:`MCEqRun.set_interaction_model`,
    - primary flux in :func:`MCEqRun.set_primary_model`,
    - zenith angle in :func:`MCEqRun.set_theta_deg`,
    - density profile in :func:`MCEqRun.set_density_model`,
    - member particles of the special ``obs_`` group
        in :func:`MCEqRun.set_obs_particles`,

    can be made on an active instance of this class, while calling
    :func:`MCEqRun.solve` subsequently to calculate the solution
    corresponding to the settings.

    The result can be retrieved by calling :func:`MCEqRun.get_solution`.


    Args:
      interaction_model (string): interaction model name, e.g. SIBYLL2.3E
      primary_model (class, param_tuple): classes derived from
        :class:`crflux.models.PrimaryFlux` and its parameters as tuple
      theta_deg (float): zenith angle :math:`\\theta` in degrees,
        measured positively from vertical direction
      medium (string, optional): "air", "water", "rock", "co2", "hydrogen", "iron"
      density_model (instance or tuple): Instance of initialized density model or
        tuple of strings, such as ('CORSIKA', ('BK_USStd', None))
      particle_list (list, optional): Construct a system for only these partices
        including their decay products.
    """

    def __init__(self, interaction_model, primary_model, theta_deg, **kwargs):
        ensure_db_available()
        if config.enable_em and config.muon_helicity_dependence:
            # Helicity L/R muon variants add semi-Lagrangian rows without
            # diagonal damping that destabilize the EM system
            # (_EM_BLOWUP_CAVEAT). Forced off for enable_em runs.
            info(
                1,
                "enable_em: forcing muon_helicity_dependence=False "
                "(helicity rows destabilize the EM cascade).",
            )
            config.muon_helicity_dependence = False
        self.medium = kwargs.pop("medium", config.interaction_medium)
        le_config = config.low_energy_extension
        low_energy_model = kwargs.pop("low_energy_model", le_config.get("model"))
        he_le_transition = kwargs.pop(
            "he_le_transition", le_config.get("he_le_transition", 80.0)
        )
        he_le_trwidth = kwargs.pop("he_le_trwidth", le_config.get("he_le_trwidth", 0.3))
        # Group views, not snapshots: a config write after construction is seen.
        self._mceq_db = MCEq.data.HDF5Backend(
            medium=self.medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
            paths=config.paths,
            grid=config.grid,
            physics=config.physics,
            em=config.em,
        )

        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_model = kwargs.pop("density_model", config.density_model)
        self.theta_deg = theta_deg

        #: Interface to interaction tables of the HDF5 database
        self._interactions = MCEq.data.Interactions(
            mceq_hdf_db=self._mceq_db, physics=config.physics
        )

        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self._int_cs = MCEq.data.InteractionCrossSections(
            mceq_hdf_db=self._mceq_db, interaction_model=interaction_model
        )

        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self._cont_losses = MCEq.data.ContinuousLosses(
            mceq_hdf_db=self._mceq_db, physics=config.physics
        )

        #: Interface to decay tables of the HDF5 database
        self._decays = MCEq.data.Decays(
            mceq_hdf_db=self._mceq_db, physics=config.physics
        )

        #: Particle manager (initialized/updated in set_interaction_model)
        self.pman = None

        # Particle list to keep track of previously initialized particles
        self._particle_list = None

        # General Matrix dimensions and shortcuts, controlled by
        # grid of yield matrices
        self._energy_grid = self._mceq_db.energy_grid

        # Initialize solution vector
        self._solution = np.zeros(1)
        # Initialize empty state (particle density) vector
        self._phi0 = np.zeros(1)
        # Initialize matrix builder (initialized in set_interaction_model)
        self.matrix_builder = None
        # Save initial condition (primary flux) to restore after dimensional resizing
        self._restore_initial_condition = []

        # Set interaction model and compute grids and matrices
        self.set_interaction_model(
            interaction_model,
            particle_list=kwargs.pop("particle_list", None),
            build_matrices=kwargs.pop("build_matrices", True),
        )

        # Default GPU device id for CUDA
        self._cuda_device = kwargs.pop("cuda_gpu_id", config.cuda_gpu_id)

        # Geomagnetic rigidity cutoff toggle. ``None`` (default) auto-detects
        # from the density model — MSIS-based atmospheres and location-tagged
        # CORSIKA atmospheres get the cutoff on by default, everything else
        # off. Set explicitly True / False to override.
        self.geomagnetic_cutoff = kwargs.pop("geomagnetic_cutoff", None)

        # Print particle list after tracking particles have been initialized
        self.pman.print_particle_tables(2)

        # Set atmosphere and geometry
        self.integration_path, self.int_grid, self.grid_var = None, None, None
        self.set_density_model(self.density_model)

        # Set initial flux condition
        if primary_model is not None:
            try:
                self.set_primary_model(*primary_model)
            except TypeError:
                self.set_primary_model(primary_model)

    @property
    def e_grid(self):
        """Energy grid (bin centers)"""
        return self._energy_grid.c

    @property
    def e_bins(self):
        """Energy grid (bin edges)"""
        return self._energy_grid.b

    @property
    def e_widths(self):
        """Energy grid (bin widths)"""
        return self._energy_grid.w

    @property
    def dim(self):
        """Energy grid (dimension)"""
        return self._energy_grid.d

    @property
    def dim_states(self):
        """Number of cascade particles times dimension of grid
        (dimension of the equation system)"""
        return self.pman.dim_states

    def ptot_grid(self, particle_name, return_bins=False):
        """Computes and returns the total momentum grid.

        If `return_bins` `True`, return bins, centers, otherwise
        just the bin centers.
        """

        ptot_bins = np.sqrt(
            (self.e_bins + self.pman[particle_name].mass) ** 2
            - self.pman[particle_name].mass ** 2
        )
        ptot_grid = np.sqrt(ptot_bins[1:] * ptot_bins[:-1])

        if return_bins:
            return ptot_bins, ptot_grid
        return ptot_grid

    def etot_grid(self, particle_name, return_bins=False):
        """Computes and returns the total energy grid.

        If `return_bins = True` return bins and centers, otherwise
        just the bin centers.
        """

        etot_bins = self.e_bins + self.pman[particle_name].mass
        etot_grid = np.sqrt(etot_bins[1:] * etot_bins[:-1])

        if return_bins:
            return etot_bins, etot_grid
        return etot_grid

    def xgrid(self, particle_name, return_as, return_bins=False):
        """Uniform access to the spectrum variable, depending on the
        same `return_as` argument as in get_solution."""

        if return_as == "kinetic energy":
            return (self.e_bins, self.e_grid) if return_bins else self.e_grid
        if return_as == "total energy":
            return self.etot_grid(particle_name, return_bins)
        if return_as == "total momentum":
            return self.ptot_grid(particle_name, return_bins)
        raise Exception("Unknown grid type requested.")

    def closest_energy(self, kin_energy):
        """Convenience function to obtain the nearest grid energy
        to the `energy` argument, provided as kinetik energy in lab. frame."""
        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        return self._energy_grid.c[eidx]

    def _get_state_vector(self, grid_idx=None):
        """Returns state vector"""
        if not hasattr(self, "_solution") and grid_idx is None:
            raise Exception("State vector not initialized. Run solve() first.")
        if not hasattr(self, "grid_sol") and grid_idx is not None:
            raise Exception("Solution not on grid. Re-run solve() with a grid.")

        if grid_idx is None:
            state_vec = np.copy(self._solution)
        elif grid_idx < len(self.grid_sol):
            state_vec = self.grid_sol[grid_idx, :]
        else:
            raise Exception("Invalid grid index", grid_idx)

        order = [(p.mceqidx, p.name) for p in self.pman.cascade_particles]

        return order, state_vec

    def _set_state_vector(self, order_i, state_vec, only_available=False):
        """Sets the initial to that supplied as state vector."""

        order = [(p.mceqidx, p.name) for p in self.pman.cascade_particles]
        if order_i != order and not only_available:
            raise Exception(
                f"The orders of the state vecs don't match {order_i}!={order}"
            )
        if order_i != order and only_available:
            particles_requested = [o[1] for o in order_i]
            for pidx, pname in order:
                if pname in self.pman.pname2pref:
                    p = self.pman.pname2pref[pname]
                    self._phi0[p.lidx : p.uidx] *= 0.0
                    if pname in particles_requested:
                        try:
                            self._phi0[p.lidx : p.uidx] = state_vec[
                                pidx * self.dim : (pidx + 1) * self.dim
                            ]
                        except ValueError:
                            raise Exception("Error when setting state for", p.name)

        else:
            self._phi0[:] = state_vec[:]

    def get_solution(
        self,
        particle_name,
        mag=0.0,
        grid_idx=None,
        integrate=False,
        return_as=config.return_as,
        dont_sum_helicities=False,
    ):
        """Retrieves solution of the calculation on the energy grid.

        Some special prefixes are accepted for lepton names:

        - the total flux of muons, muon neutrinos etc. from all sources/mothers
          can be retrieved without a prefix ``mu+`` or with the prefix ``total_mu+``,
          ``total_numu``
        - the conventional flux of muons, muon neutrinos etc. from all sources
          can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
        - the prompt flux of muons, muon neutrinos etc. from all sources
          can be retrieved by the prefix ``pr_``, i.e. ``pr_numu``
        - correspondigly, the flux of leptons which originated from the decay
          of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``

        Args:
          particle_name (str): The name of the particle such, e.g.
            ``total_mu+`` for the total flux spectrum of positive muons or
            ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
          mag (float, optional): 'magnification factor': the solution is
            multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
          grid_idx (int, optional): if the integrator has been configured to save
            intermediate solutions on a depth grid, then ``grid_idx`` specifies
            the index of the depth grid for which the solution is retrieved. If
            not specified the flux at the surface is returned
          integrate (bool, optional): return averge particle number instead of
          flux (multiply by bin width)
          return_as (str, optional): the flux can be returned as ``total energy``, ``kinetic energy``,
            or ``total momentum`` flux. This defaults to ``kinetic energy`` and is in general taken from
            ``MCEq.config.return_as``
          dont_sum_helicities (bool, optional): Per default the lepton flux is summed over the available helicities,
            e.g. ``total_mu+`` is the muon flux from (-1, 0, +1) helicity for mu+.

        Returns:
          (:func: numpy.array): flux of particles on energy grid :attr:`e_grid`
        """

        if grid_idx is not None and len(self.grid_sol) == 0:
            raise Exception("Solution not has not been computed on grid. Check input.")
        if grid_idx is None:
            sol = np.copy(self._solution)
        elif grid_idx >= len(self.grid_sol):
            sol = self.grid_sol[-1, :]
        else:
            sol = self.grid_sol[grid_idx, :]

        return self._get_solution_from_state(
            sol,
            particle_name,
            mag=mag,
            integrate=integrate,
            return_as=return_as,
            dont_sum_helicities=dont_sum_helicities,
        )

    def _get_solution_from_state(
        self,
        sol,
        particle_name,
        mag=0.0,
        integrate=False,
        return_as=None,
        dont_sum_helicities=False,
    ):
        """Extract a named spectrum from an explicit state vector.

        Same particle-name/prefix semantics, helicity summation and
        ``return_as`` conversions as :meth:`get_solution`, but operates
        on the state vector ``sol`` passed by the caller instead of
        ``self._solution``. This is the shared extraction backend for
        :meth:`get_solution` and :meth:`MCEqBatchResult.get_solution`
        (per-column retrieval from multi-RHS solves).

        Args:
          sol (np.ndarray[dim_states]): state vector to extract from.
          particle_name (str): see :meth:`get_solution`.
          mag, integrate, return_as, dont_sum_helicities: see
            :meth:`get_solution`. ``return_as=None`` resolves to
            ``config.return_as``.

        Returns:
          (np.ndarray): flux of particles on energy grid :attr:`e_grid`
        """
        if return_as is None:
            return_as = config.return_as

        res = np.zeros(self._energy_grid.d)
        ref = self.pman.pname2pref

        def sum_lr(lep_str, prefix):
            result = np.zeros(self.dim)
            nsuccess = 0

            if dont_sum_helicities:
                sum_over = [lep_str]
            else:
                sum_over = [lep_str, lep_str + "_l", lep_str + "_r"]

            for ls in sum_over:
                if prefix + ls not in ref:
                    info(
                        15,
                        "No separate left and right handed particles,",
                        f"or, unavailable particle prefix {prefix + ls}.",
                    )
                    continue
                result += sol[ref[prefix + ls].lidx : ref[prefix + ls].uidx]
                nsuccess += 1
            if nsuccess == 0 and config.excpt_on_missing_particle:
                raise Exception(f"Requested particle {particle_name} not found.")
            return result

        lep_str = particle_name.split("_")[1] if "_" in particle_name else particle_name

        default_tracking_prefixes = [
            "conv_",
            "pr_",
            "pi_",
            "k_",
            "K0_",
            "mulr_",
            "mu_h0_",
            "prcas_",
            "prres_",
        ]
        if not config.enable_default_tracking:
            for track_pref in default_tracking_prefixes:
                if particle_name.startswith(track_pref):
                    raise Exception(
                        "Tracking category requested but "
                        + "enable_default_tracking is off in config."
                    )

        if particle_name.startswith("total_"):
            # Note: This has changed from previous MCEq versions,
            # since pi_ and k_ prefixes are mere tracking counters
            # and no full particle species anymore

            res = sum_lr(lep_str, prefix="")

        elif particle_name.startswith("conv_"):
            # Note: This changed from previous MCEq versions,
            # conventional is defined as total - prompt
            res = self._get_solution_from_state(
                sol,
                "total_" + lep_str,
                mag=0,
                integrate=False,
                return_as="kinetic energy",
            ) - self._get_solution_from_state(
                sol,
                "pr_" + lep_str,
                mag=0,
                integrate=False,
                return_as="kinetic energy",
            )

        elif particle_name.startswith("pr_"):
            if "prcas_" + lep_str in ref:
                res += sum_lr(lep_str, prefix="prcas_")
            if "prres_" + lep_str in ref:
                res += sum_lr(lep_str, prefix="prres_")
            if "em_" + lep_str in ref:
                res += sum_lr(lep_str, prefix="em_")
        else:
            try:
                res = sum_lr(particle_name, prefix="")
            except KeyError:
                if config.excpt_on_missing_particle:
                    raise Exception(f"Requested particle {particle_name} not found.")
                else:
                    info(1, f"Requested particle {particle_name} not found.")

        # When returning in Etot, interpolate on different grid
        if return_as == "total energy":
            etot_grid = self.etot_grid(lep_str)
            if not integrate:
                return res * etot_grid**mag
            return res * etot_grid**mag * self.e_widths

        if return_as == "kinetic energy":
            if not integrate:
                return res * self._energy_grid.c**mag
            return res * self._energy_grid.c**mag * self.e_widths

        if return_as == "total momentum":
            ptot_bins, ptot_grid = self.ptot_grid(lep_str, return_bins=True)
            dEkindp = np.diff(ptot_bins) / self.e_widths
            if not integrate:
                return dEkindp * res * ptot_grid**mag
            return dEkindp * res * ptot_grid**mag * np.diff(ptot_bins)

        raise Exception(
            "Unknown 'return_as' variable choice.",
            'the options are "kinetic energy", "total energy", "total momentum"',
        )

    def set_interaction_model(
        self,
        interaction_model,
        particle_list=None,
        update_particle_list=True,
        force=False,
        build_matrices=True,
    ):
        """Sets interaction model and/or an external charm model for calculation.

        Decay and interaction matrix will be regenerated automatically
        after performing this call.

        Args:
          interaction_model (str): name of interaction model
          charm_model (str, optional): name of charm model
          force (bool): force loading interaction model
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

        info(1, interaction_model)

        if (
            not force
            and (self._interactions.iam == interaction_model)
            and particle_list != self._particle_list
        ):
            info(2, "Skip, since current model identical to", interaction_model + ".")
            return

        self._int_cs.load(interaction_model)

        # TODO: simplify this, stuff not needed anymore
        if not update_particle_list and self._particle_list is not None:
            info(10, "Re-using particle list.")
            self._interactions.load(interaction_model, parent_list=self._particle_list)
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)

        elif self._particle_list is None:
            info(10, "New initialization of particle list.")
            # First initialization
            if particle_list is None:
                self._interactions.load(interaction_model)
            else:
                self._interactions.load(interaction_model, parent_list=particle_list)

            self._decays.load(parent_list=self._interactions.particles)
            self._particle_list = self._interactions.particles + self._decays.particles
            # Create particle database
            self.pman = ParticleManager(
                self._particle_list,
                self._energy_grid,
                self._int_cs,
                self.medium,
                physics=config.physics,
            )
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)
            self.matrix_builder = MatrixBuilder(self.pman, self._mceq_db)

        elif update_particle_list and particle_list != self._particle_list:
            info(10, "Updating particle list.")
            # Updated particle list received
            if particle_list is None:
                self._interactions.load(interaction_model)
            else:
                self._interactions.load(interaction_model, parent_list=particle_list)
            self._decays.load(parent_list=self._interactions.particles)
            self._particle_list = self._interactions.particles + self._decays.particles
            self.pman.set_interaction_model(
                self._int_cs,
                self._interactions,
                updated_parent_list=self._particle_list,
            )
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)

        else:
            raise Exception("Should not happen in practice.")

        self._resize_vectors_and_restore()

        # initialize matrices
        if not build_matrices:
            return
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=False
        )

    def _resize_vectors_and_restore(self):
        """Update solution and grid vectors if the number of particle species
        or the interaction models change. The previous state, such as the
        initial spectrum, are restored."""

        # Update dimensions if particle dimensions changed
        self._phi0 = np.zeros(self.dim_states)
        self._solution = np.zeros(self.dim_states)

        # Restore initial condition if present.
        # Entries are tuples of (method_name_str, *args). We store method
        # *names* — not bound methods — so that this list does not pin
        # ``self`` via a Python-level reference cycle. See PR #163: bound
        # methods kept old MCEqRun instances alive, which on the macOS
        # Accelerate backend overflowed the fixed-size sparse-matrix store
        # (SIZE_MSTORE=10) after ~5 instances.
        if len(self._restore_initial_condition) > 0:
            for con in self._restore_initial_condition:
                getattr(self, con[0])(*con[1:])

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model.

        This functions is quick and does not require re-generation of
        matrices.

        Args:
          interaction_model (:class:`CRFluxModel.PrimaryFlux`): reference
          to primary model **class**
          tag (tuple): positional argument list for model class
        """

        assert not isinstance(model_class_or_object, tuple), (
            "Primary model can not be supplied as tuples"
        )

        # Check if classs or object supplied
        if not isinstance(model_class_or_object, type):
            assert any(
                [
                    "PrimaryFlux" in b.__name__
                    for b in model_class_or_object.__class__.__bases__
                ]
            ), "model_class_or_object is not derived from crflux.models.PrimaryFlux"
            info(5, "Primary model supplied as object")
            self.pmodel = model_class_or_object
        else:
            # Initialize primary model object
            info(5, "Primary model supplied as class")
            self.pmodel = model_class_or_object(tag)

        info(1, f"Primary model set to {self.pmodel.name}")

        # Save primary flux model for restoration after interaction model
        # changes. Store the method *name*, not a bound method — see the
        # comment in ``_resize_vectors_and_restore`` (PR #163).
        self._restore_initial_condition = [("set_primary_model", self.pmodel)]
        # TODO: Maybe needs to catch the removal of the np.vectorize
        # self.get_nucleon_spectrum = np.vectorize(self.pmodel.p_and_n_flux)
        self.get_nucleon_spectrum = self.pmodel.p_and_n_flux

        try:
            self.dim_states
        except AttributeError:
            self.finalize_pmodel = True

        # Set initial condition
        minimal_energy = config.minimal_primary_energy
        if (2212, 0) in self.pman and (2112, 0) in self.pman:
            e_tot = self._energy_grid.c + 0.5 * (
                self.pman[(2212, 0)].mass + self.pman[(2112, 0)].mass
            )
        else:
            raise Exception(
                "No nucleons in eqn system, primary flux model can not be used."
            )

        min_idx = np.argmin(np.abs(e_tot - minimal_energy))
        self._phi0 *= 0
        p_top, n_top = self.get_nucleon_spectrum(e_tot[min_idx:])[1:]
        if (2212, 0) in self.pman:
            self._phi0[
                min_idx + self.pman[(2212, 0)].lidx : self.pman[(2212, 0)].uidx
            ] = 1e-4 * p_top
        else:
            info(
                1,
                "Protons not in equation system, can not set primary flux.",
            )

        if (2112, 0) in self.pman and not self.pman[(2112, 0)].is_resonance:
            self._phi0[
                min_idx + self.pman[(2112, 0)].lidx : self.pman[(2112, 0)].uidx
            ] = 1e-4 * n_top
        elif (2212, 0) in self.pman:
            info(
                2,
                "Neutrons not part of equation system,",
                "substituting initial flux with protons.",
            )
            self._phi0[
                min_idx + self.pman[(2212, 0)].lidx : self.pman[(2212, 0)].uidx
            ] += 1e-4 * n_top

    def set_single_primary_particle(
        self, E, corsika_id=None, pdg_id=None, append=False
    ):
        """Set type and kinetic energy of a single primary nucleus to
        calculation of particle yields.

        The functions uses the superposition theorem, where the flux of
        a nucleus with mass A and charge Z is modeled by using Z protons
        and A-Z neutrons at energy :math:`E_{nucleon}= E_{nucleus} / A`
        The nucleus type is defined via :math:`\\text{CORSIKA ID} = A*100 + Z`. For
        example iron has the CORSIKA ID 5226.

        Single leptons or hadrons can be defined by specifiying `pdg_id` instead of
        `corsika_id`.

        The `append` argument can be used to compose an initial state with
        multiple particles. If it is `False` the initial condition is reset to zero
        before adding the particle.

        A continuous input energy range is allowed between
        :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.

        Args:
          E (float): kinetic energy of a nucleus in GeV
          corsika_id (int): ID of a nucleus (see text)
          pdg_id (int): PDG ID of a particle
          append (bool): If True, keep previous state and append a new particle.
        """
        import warnings

        from scipy.linalg import solve

        from MCEq.misc import getAZN, getAZN_corsika

        if corsika_id and pdg_id:
            raise Exception("Provide either corsika or PDG ID")

        info(2, f"CORSIKA ID {corsika_id}, PDG ID {pdg_id}, energy {E:5.3g} GeV")

        if corsika_id:
            n_nucleons, n_protons, n_neutrons = getAZN_corsika(corsika_id)
        elif pdg_id:
            n_nucleons, n_protons, n_neutrons = getAZN(pdg_id)

        En = E / float(n_nucleons) if n_nucleons > 0 else E

        if En < np.min(self._energy_grid.c):
            raise Exception("energy per nucleon too low for primary " + str(corsika_id))

        if append is False:
            # Store ``False`` explicitly so the replay does not silently
            # default to overwriting on the first call of an append chain.
            self._restore_initial_condition = [
                ("set_single_primary_particle", E, corsika_id, pdg_id, False)
            ]
            self._phi0 *= 0.0
        else:
            self._restore_initial_condition.append(
                ("set_single_primary_particle", E, corsika_id, pdg_id, True)
            )
        egrid = self._energy_grid.c
        ebins = self._energy_grid.b
        ewidths = self._energy_grid.w

        info(
            3,
            (
                f"superposition: n_protons={n_protons}, n_neutrons={n_neutrons}, "
                + f"energy per nucleon={En:5.3g} GeV"
            ),
        )

        cenbin = np.argwhere(En < ebins)[0][0] - 1

        # Equalize the first three moments for 3 normalizations around the central
        # bin
        emat = np.vstack(
            (
                ewidths[cenbin - 1 : cenbin + 2],
                ewidths[cenbin - 1 : cenbin + 2] * egrid[cenbin - 1 : cenbin + 2],
                ewidths[cenbin - 1 : cenbin + 2] * egrid[cenbin - 1 : cenbin + 2] ** 2,
            )
        )

        if n_nucleons == 0:
            # This case handles other exotic projectiles
            b_particle = np.array([1.0, En, En**2])
            lidx = self.pman[pdg_id].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._phi0[lidx + cenbin - 1 : lidx + cenbin + 2] += solve(
                    emat, b_particle
                )
            return

        if n_protons > 0:
            b_protons = np.array([n_protons, En * n_protons, En**2 * n_protons])
            p_lidx = self.pman[2212].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._phi0[p_lidx + cenbin - 1 : p_lidx + cenbin + 2] += solve(
                    emat, b_protons
                )
        if n_neutrons > 0:
            b_neutrons = np.array([n_neutrons, En * n_neutrons, En**2 * n_neutrons])
            n_lidx = self.pman[2112].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._phi0[n_lidx + cenbin - 1 : n_lidx + cenbin + 2] += solve(
                    emat, b_neutrons
                )

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user-defined spectrum for an arbitrary species as initial condition.

        This function is an equivalent to :func:`set_single_primary_particle`. It
        allows to define an arbitrary spectrum for each available particle species
        as initial condition for the integration. Set the `append`
        argument to `True` for subsequent species to define initial
        spectra combined from different particles.

        The (differential) spectrum has to be distributed on the energy
        grid as dN/dptot, i.e. divided by the bin widths and with the
        total momentum units in GeV(/c).

        Args:
          spectrum (np.array): spectrum dN/dptot
          pdg_id (int): PDG ID in case of a particle
        """

        info(2, f"PDG ID {pdg_id}")

        if not append:
            self._restore_initial_condition = [
                ("set_initial_spectrum", spectrum, pdg_id, append)
            ]
            self._phi0 *= 0
        else:
            self._restore_initial_condition.append(
                ("set_initial_spectrum", spectrum, pdg_id, append)
            )
        if len(spectrum) != self.dim:
            raise Exception("Lengths of spectrum and energy grid do not match.")

        self._phi0[self.pman[pdg_id].lidx : self.pman[pdg_id].uidx] += spectrum

    def get_initial_state(self):
        """Return a copy of the current initial-condition vector ``phi0``.

        This is the state vector composed by the most recent
        :meth:`set_primary_model` / :meth:`set_single_primary_particle` /
        :meth:`set_initial_spectrum` calls — the same vector
        :meth:`solve` propagates. Use it to assemble columns for
        :meth:`solve_batch` without touching private attributes.

        Returns:
          (np.ndarray[dim_states]): copy of the initial state vector.
        """
        return self._phi0.copy()

    def initial_state(self, components):
        """Compose and return an initial-state column without mutating
        the instance.

        Builds a ``(dim_states,)`` vector from one or more components
        using the same machinery as :meth:`set_single_primary_particle`
        and :meth:`set_initial_spectrum`, then restores the previous
        initial condition. The intended use is assembling the columns of
        a :meth:`solve_batch` initial-state matrix::

            # response matrix: one column per primary energy
            phi0 = np.stack(
                [mceq.initial_state({"E": E, "pdg_id": 2212}) for E in E_primaries],
                axis=1,
            )
            res = mceq.solve_batch(phi0)

        Args:
          components (dict | list[dict]): one component dict or a list of
            component dicts (summed). Each component is either

            - a single primary: ``{"E": <GeV>, "corsika_id": <A*100+Z>}``
              or ``{"E": <GeV>, "pdg_id": <PDG>}`` (forwarded to
              :meth:`set_single_primary_particle`), or
            - a user spectrum: ``{"spectrum": <array dN/dptot>,
              "pdg_id": <PDG>}`` (forwarded to
              :meth:`set_initial_spectrum`).

        Returns:
          (np.ndarray[dim_states]): the composed initial-state column.
        """
        if isinstance(components, dict):
            components = [components]
        if not components:
            raise ValueError("initial_state: components must not be empty")

        saved_phi0 = self._phi0.copy()
        saved_restore = list(self._restore_initial_condition)
        try:
            for i, comp in enumerate(components):
                comp = dict(comp)
                append = i > 0
                if "spectrum" in comp:
                    spectrum = comp.pop("spectrum")
                    pdg_id = comp.pop("pdg_id")
                    if comp:
                        raise ValueError(
                            f"initial_state: unknown keys {sorted(comp)} in "
                            f"spectrum component"
                        )
                    self.set_initial_spectrum(spectrum, pdg_id, append=append)
                elif "E" in comp:
                    E = comp.pop("E")
                    unknown = set(comp) - {"corsika_id", "pdg_id"}
                    if unknown:
                        raise ValueError(
                            f"initial_state: unknown keys {sorted(unknown)} in "
                            f"single-primary component"
                        )
                    self.set_single_primary_particle(E, append=append, **comp)
                else:
                    raise ValueError(
                        "initial_state: each component needs either 'E' "
                        "(single primary) or 'spectrum' + 'pdg_id' "
                        f"(user spectrum); got keys {sorted(comp)}"
                    )
            return self._phi0.copy()
        finally:
            self._phi0 = saved_phi0
            self._restore_initial_condition = saved_restore

    def set_density_model(self, density_model_or_config):
        """Sets model of the atmosphere.

        To choose, for example, a CORSIKA parametrization for the Southpole in January,
        do the following::

            mceq_instance.set_density_model(("CORSIKA", ("PL_SouthPole", "January")))

        More details about the choices can be found in
        :mod:`MCEq.geometry.density_profiles`.Calling this method will
        issue a recalculation of the interpolation and the integration path.

        From version 1.2 and above, the `density_model_or_config`
        parameter can be a reference to an instance of a density class
        directly. The class has to be derived either from
        :class:`MCEq.geometry.density_profiles.EarthsAtmosphere` or
        :class:`MCEq.geometry.density_profiles.GeneralizedTarget`.

        Args:
          density_model_or_config (obj or tuple of strings):
            (parametrization type, arguments)
        """
        import MCEq.geometry.density_profiles as dprof

        # Check if string arguments or an instance of the density class is provided
        if not isinstance(
            density_model_or_config, (dprof.EarthsAtmosphere, dprof.GeneralizedTarget)
        ):
            base_model, model_config = density_model_or_config

            available_models = [
                "MSIS00",
                "MSIS00_IC",
                "MSIS21",
                "MSIS21_IC",
                "MSIS21_KM3NeT",
                "CORSIKA",
                "AIRS",
                "Isothermal",
                "GeneralizedTarget",
            ]

            if base_model not in available_models:
                info(
                    0,
                    "Unknown density model. Available choices are:\n",
                    "\n".join(available_models),
                )
                raise ValueError("Choose a different profile.")

            info(1, "Setting density profile to", base_model, model_config)

            if base_model == "MSIS00":
                self.density_model = dprof.MSIS00Atmosphere(*model_config)
            elif base_model == "MSIS00_IC":
                self.density_model = dprof.MSIS00IceCubeCentered(*model_config)
            elif base_model == "MSIS21":
                self.density_model = dprof.MSIS21Atmosphere(*model_config)
            elif base_model == "MSIS21_IC":
                self.density_model = dprof.MSIS21IceCubeCentered(*model_config)
            elif base_model == "MSIS21_KM3NeT":
                self.density_model = dprof.MSIS21KM3NeTCentered(*model_config)
            elif base_model == "CORSIKA":
                self.density_model = dprof.CorsikaAtmosphere(*model_config)
            elif base_model == "AIRS":
                self.density_model = dprof.AIRSAtmosphere(*model_config)
            elif base_model == "Isothermal":
                self.density_model = dprof.IsothermalAtmosphere(*model_config)
            elif base_model == "GeneralizedTarget":
                self.density_model = dprof.GeneralizedTarget()
        else:
            self.density_model = density_model_or_config

        if self.theta_deg is not None and isinstance(
            self.density_model, dprof.EarthsAtmosphere
        ):
            if self.theta_deg is None:
                info(1, "Using default zenith angle theta=0.")
                self.set_zenith_azimuth(0)
            else:
                self.set_zenith_azimuth(self.theta_deg)
        elif isinstance(self.density_model, dprof.GeneralizedTarget):
            self.integration_path = None
        else:
            raise ValueError(f"Density model {self.density_model} not supported.")

        # TODO: Make the pman aware of that density might have changed and
        # indices as well
        # self.pmod._gen_list_of_particles()

    def set_zenith_azimuth(self, zenith_deg, azimuth_deg=None):
        """Set the zenith and (optionally) azimuth angles for the shower.

        This is the primary API for configuring the shower direction.
        :meth:`set_theta_deg` is a deprecated alias for this method.

        **Azimuth convention**: 0° = geographic North, 90° = East
        (clockwise from North, meteorological convention).

        **Zenith convention**: 0° = directly above the detector (vertical
        downgoing shower), 90° = horizontal, > 90° = upgoing shower whose
        source is below the horizon.  Upgoing angles require a density
        model with ``max_theta = 180`` (e.g.
        :class:`~MCEq.geometry.density_profiles.MSIS00LocationCentered`
        subclasses).

        When *azimuth_deg* is ``None`` and the active density model is an
        instance of
        :class:`~MCEq.geometry.density_profiles.MSIS00LocationCentered`,
        the atmospheric density profile is averaged over all azimuth
        directions for the given zenith angle.  For models without azimuth
        awareness the argument is silently ignored.

        Args:
            zenith_deg (float): Zenith angle at the detector in degrees.
            azimuth_deg (float, optional): Azimuth angle in degrees.
                ``None`` (default) triggers azimuth-averaging for capable
                models.
        """
        import MCEq.geometry.density_profiles as dprof

        info(
            2,
            f"Zenith angle {zenith_deg:6.2f}"
            + (f", azimuth {azimuth_deg:6.2f}" if azimuth_deg is not None else ""),
        )

        if isinstance(self.density_model, dprof.GeneralizedTarget):
            raise Exception("GeneralizedTarget does not support angles.")

        # Cache check: skip if nothing has changed
        cached_theta = self.density_model.theta_deg
        cached_azi = getattr(self.density_model, "_current_azimuth_deg", None)
        if cached_theta == zenith_deg and cached_azi == azimuth_deg:
            info(2, "Angle selection corresponds to cached value, skipping calc.")
            return

        # Dispatch to set_theta with or without azimuth_deg depending on
        # the density model's ``depends_on_azimuth`` attribute. Both
        # MSIS00LocationCentered and MSIS21LocationCentered accept the
        # extra azimuth_deg argument; everything else ignores azimuth.
        if getattr(self.density_model, "depends_on_azimuth", False):
            self.density_model.set_theta(zenith_deg, azimuth_deg=azimuth_deg)
        else:
            self.density_model.set_theta(zenith_deg)
        self.integration_path = None

    def set_theta_deg(self, theta_deg):
        """Sets zenith angle :math:`\\theta` as seen from a detector.

        .. deprecated::
            Use :meth:`set_zenith_azimuth` instead.  This method will be
            removed in a future release.

        Args:
          theta_deg (float): zenith angle in degrees
        """
        import warnings

        warnings.warn(
            "set_theta_deg() is deprecated; use set_zenith_azimuth() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_zenith_azimuth(theta_deg)

    def inject_ddm(self, ddm):
        """Set a DDM object as interaction model.

        The argument requires a DDM model object. Calling `set_interaction_model`
        overwrites DDM with a different model.
        """

        from .ddm import isospin_partners, isospin_symmetries

        injected = []
        for (prim, sec), mati in ddm.ddm_matrices(self).items():
            info(5, f"Injecting DDM {prim} --> {sec}")
            iso_part = isospin_partners[prim]

            self.pman[prim].hadr_yields[self.pman[sec]] = np.asarray(mati)
            info(5, "Injecting isopart", iso_part, isospin_symmetries[iso_part][sec])
            self.pman[iso_part].hadr_yields[
                self.pman[isospin_symmetries[iso_part][sec]]
            ] = np.asarray(mati)
            injected.append(
                ((prim, sec), (iso_part, isospin_symmetries[iso_part][sec]))
            )

        if config.debug_level > 2:
            s = "DDM matrices injected into MCEq:\n"
            for (prim, sec), (iprim, isec) in injected:
                s += f"\t{prim}-->{sec}, isospin: {iprim} --> {isec}\n"
            print(s)

        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=False
        )

    def set_mod_pprod(self, prim_pdg, sec_pdg, x_func, x_func_args, delay_init=False):
        """Sets combination of projectile/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of
        ``prim_pdg`` is modified according to the function passed to
        :func:`InteractionYields.init_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
          x_func (object): reference to function
          x_func_args (tuple): arguments passed to ``x_func``
          delay_init (bool): Prevent init of mceq matrices if you are
                             planning to add more modifications
        """
        info(1, f"{prim_pdg}/{sec_pdg}, {sec_pdg}, {x_func.__name__}, {x_func_args!s}")

        init = self._interactions._set_mod_pprod(prim_pdg, sec_pdg, x_func, x_func_args)

        # Need to regenerate matrices completely
        return int(init)

    def unset_mod_pprod(self, dont_fill=False):
        """Removes modifications from :func:`MCEqRun.set_mod_pprod`.

        Args:
          skip_fill (bool): If `true` do not regenerate matrices
          (has to be done at a later step by hand)
        """
        from collections import defaultdict

        info(1, "Particle production modifications reset to defaults.")

        self._interactions.mod_pprod = defaultdict(dict)
        # Need to regenerate matrices completely
        if not dont_fill:
            self.regenerate_matrices()

    def regenerate_matrices(self, skip_decay_matrix=False):
        """Call this function after applying particle prod. modifications aka
        Barr parameters"""

        # TODO: Not all particles need to be reset and there is some performance loss
        # This can be optmized by refreshing only the particles that change or through
        # lazy evaluation, i.e. hadronic channels dict. calls data.int..get_matrix
        # on demand
        self.pman.set_interaction_model(self._int_cs, self._interactions, force=True)
        self._resize_vectors_and_restore()
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=skip_decay_matrix
        )

    def solve(
        self,
        int_grid=None,
        grid_var="X",
        *,
        X_start=None,
        eps=None,
        dX_max=None,
        dX_min=None,
        fd_span=None,
        **kwargs,
    ):
        """Launches the solver.

        The setting `kernel_config` in the config file decides which solver
        to launch.

        Args:
          int_grid (list): list of depths at which results are recorded
          grid_var (str): Can be depth `X` or something else (currently
            only `X` supported)
          X_start (float | None): starting depth in g/cm^2 used for
            ETD2 path construction. ``None`` → ``config.X_start`` (= 0).
          eps (float | None): within-step ``rho_inv`` variation tolerance
            for the ETD2 non-uniform schedule. ``None`` →
            ``config.etd2_path["eps"]``.
          dX_max (float | None): cap on step size (off-diagonal stability
            cliff) for ETD2. ``None`` → ``config.etd2_path["dX_max"]``.
          dX_min (float | None): floor on step size for ETD2. ``None`` →
            ``config.etd2_path["dX_min"]``.
          fd_span (float | None): forward-FD probe span for the ETD2
            schedule's local rate estimate. ``None`` →
            ``config.etd2_path["fd_span"]``.
          kwargs (dict): Reserved for advanced flags. Currently only
            ``skip_integration_path=True`` is recognised — it bypasses
            the call to ``_calculate_integration_path`` and reuses the
            cached ``self.integration_path`` (used by tests / experts who
            want to keep the path fixed across multiple ``solve`` calls).

        """
        info(2, f"Launching {config.kernel_config} solver")

        if not kwargs.pop("skip_integration_path", False):
            if int_grid is not None and np.any(np.diff(int_grid) < 0):
                raise Exception(
                    "The X values in int_grid are required to be strictly",
                    "increasing.",
                )

            # Calculate integration path if not yet happened
            self._calculate_integration_path(
                int_grid,
                grid_var,
                X_start=X_start,
                eps=eps,
                dX_max=dX_max,
                dX_min=dX_min,
                fd_span=fd_span,
            )
        else:
            info(2, "Warning: integration path calculation skipped.")

        # If the initial angular density is a delta function (e.g. a single
        # cosmic-ray shower incident at some angle theta), all Hankel modes
        # k are populated with the same amplitude as the 1D initial
        # condition. The stitched ETD2 path operates on a length
        # ``n_k * dim_states`` state vector — broadcast ``_phi0`` across
        # the n_k blocks via ``np.tile`` so block-k of the stacked state
        # is a copy of ``_phi0``. For 1D databases this reduces to a
        # plain copy.
        if self._mceq_db.is_2d:
            phi0 = np.tile(self._phi0, self._mceq_db.n_k)
        else:
            phi0 = np.copy(self._phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, f"for {nsteps} integration steps.")

        start = time()

        self._solution, self.grid_sol = self._run_etd2(
            nsteps,
            dX,
            rho_inv,
            phi0,
            grid_idcs,
            sec_ops=self._resolve_secant(),
        )

        if isinstance(self.grid_sol, list):
            self.grid_sol = np.asarray(self.grid_sol)

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def convert_to_theta_space(
        self,
        hankel_transf,
        pdg_id,
        hel,
        oversample_res=10,
        theta_res=1200,
        log_theta=False,
    ):
        """Convert Hankel-space amplitudes from the 2D MCEq solver to real
        (angular) space.

        The transform itself is :func:`MCEq.hankel.inverse_hankel_legacy`;
        this method adds the state-vector indexing and the theta grid.

        Args:
            hankel_transf (list of np.arrays): list of Hankel space solutions at
                the requested slant depths (i.e. the output of ``self.grid_sol``)
            pdg_id (int): PDG ID of the particle whose angular density is being
                requested (e.g. 14 for NuMu)
            hel (int): helicity of the particle whose angular density is being
                requested (e.g. 0 or ±1 for polarized muons)
            oversample_res (int): resolution of the Hankel grid oversampling
                (used to approximate the continuous inverse Hankel transform)
            theta_res (int): resolution of the angular (theta) grid where the
                inverse Hankel transform will output the densities
            log_theta (bool): whether to return logarithmic angular grid
                (True=logarithmic, False=linear)

        Returns:
            tuple: ``(k_oversampled, oversampled_amps, theta_range,
            f_theta)`` with the list entries indexed ``[depth][eidx]``.
        """
        from MCEq.hankel import inverse_hankel_legacy

        if log_theta:
            theta_range = np.logspace(-5, np.log10(np.pi / 2), theta_res)
        else:
            theta_range = np.linspace(0, np.pi / 2, theta_res)
        k_grid = self._mceq_db.k_grid
        n_e = len(self.e_grid)
        ref = self.pman.pdg2mceqidx[(pdg_id, hel)] * n_e
        oversample_pts = int(np.max(k_grid) * oversample_res)
        oversampled_k_arr = np.linspace(np.min(k_grid), np.max(k_grid), oversample_pts)

        store_oversampled_hankel_amps = []
        store_inverse_hankel_transfs = []
        for sol in hankel_transf:
            _, F_ov, f_theta = inverse_hankel_legacy(
                sol[:, ref : ref + n_e].T,
                k_grid,
                theta_range,
                oversample_res=oversample_res,
                return_oversampled=True,
            )
            store_oversampled_hankel_amps.append(list(F_ov))
            store_inverse_hankel_transfs.append(list(f_theta))

        return (
            oversampled_k_arr,
            store_oversampled_hankel_amps,
            theta_range,
            store_inverse_hankel_transfs,
        )

    def solve_from_integration_path(self, nsteps, dX, rho_inv, grid_idcs):
        """Launches the solver directly for parameters of the integration path.


        The helper function is useful if you want to skip the calculation of
        the integration path every time. This function is intended for expert
        use and is not required for normal operation.

        The parameters can be obtained after calling _calculate_integration_path
        with correct settings for density and angle parameters::

            nsteps, dX, rho_inv, grid_idcs = self.integration_path

        Args:
          phi0 (np.array): initial condition
          nsteps (int): number of integration steps
          dX (list): the delta_X's
          rho_inv (list): the inverse of the density at each step
          grid_idcs (list): list of steps at which the solution
          is dumped into `grid_sol`
        """

        info(2, f"Launching {config.kernel_config} solver")
        info(2, f"for {nsteps} integration steps.")

        start = time()

        # See ``MCEqRun.solve`` for the rationale: stitched ETD2 needs a
        # length ``n_k * dim_states`` initial state for 2D databases.
        if self._mceq_db.is_2d:
            phi0 = np.tile(self._phi0, self._mceq_db.n_k)
        else:
            phi0 = np.copy(self._phi0)

        self._solution, self.grid_sol = self._run_etd2(
            nsteps,
            dX,
            rho_inv,
            phi0,
            grid_idcs,
            sec_ops=self._resolve_secant(),
        )

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def solve_batch(
        self,
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

        Does NOT mutate ``self._phi0`` / ``self._solution`` /
        ``self.grid_sol``; the active density model and angles are
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
            raise ValueError(
                f"solve_batch: dtype must be float32 or float64, got {dtype}"
            )

        sec_ops = self._resolve_secant()

        # --- resolve phi0 ------------------------------------------------
        if phi0 is None:
            phi0_arr = self._phi0.copy()
        else:
            phi0_arr = np.asarray(phi0, dtype=np.float64)
        if phi0_arr.ndim == 1:
            if phi0_arr.size != self.dim_states:
                raise ValueError(
                    f"solve_batch: phi0 has length {phi0_arr.size}, "
                    f"expected first axis = dim_states = {self.dim_states}"
                )
            n_cols = None
        elif phi0_arr.ndim == 2:
            if phi0_arr.shape[0] != self.dim_states:
                raise ValueError(
                    f"solve_batch: phi0 has shape {phi0_arr.shape}, "
                    f"expected first axis = dim_states = {self.dim_states}"
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
                np.broadcast_to(phi0_arr[:, None], (self.dim_states, K))
            )
        else:
            phi0_mat = np.ascontiguousarray(phi0_arr)

        # 2D databases: the stitched solver state is n_k * dim_states per
        # column. phi0 stays user-facing at dim_states (per mode); a delta
        # angular initial condition populates every Hankel mode with the
        # same amplitude, so tile the rows across the n_k blocks — the
        # exact analogue of the np.tile in solve().
        if self._mceq_db.is_2d:
            phi0_mat = np.ascontiguousarray(np.tile(phi0_mat, (self._mceq_db.n_k, 1)))

        # --- build integration paths -------------------------------------
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
                    "shared-path batches (conditions=None). Set the "
                    "direction/atmosphere on the instance and vary only "
                    "phi0 columns to record snapshots."
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
            info(
                2,
                f"solve_batch: shared-path multi-RHS route, K={K}, "
                f"kernel={config.kernel_config}, nsteps={nsteps}",
            )
            # ``dtype`` controls the state-buffer precision; the diagonals
            # ``d_int`` / ``d_dec`` remain fp64 in the diag-factor pipeline
            # for the fp32 path (exp(h·D) saturates fp32 fast at high
            # zenith).
            phi0_typed = phi0_mat.astype(dtype, copy=True)
            sol, grid_sol = self._run_etd2(
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
            sol = self._run_etd2_carousel(
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
            self,
            sol,
            grid_sol=grid_sol,
            int_grid=int_grid,
            nsteps_per_col=nsteps_per_col,
            conditions=conditions,
            legacy_tuple=legacy,
        )

    def solve_multirhs(
        self,
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
        res = self.solve_batch(
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
            atmosphere, MSIS00 included.

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

        def dm_key_of(c):
            dm_spec = c.get("density_model")
            if dm_spec is None:
                return ("current",)
            if isinstance(dm_spec, (tuple, list)):
                return ("cfg", repr(tuple(dm_spec)))
            return ("obj", id(dm_spec))

        # Save the *current* direction from the density model — the
        # MCEqRun-level ``theta_deg`` attribute only reflects the
        # constructor argument and is not updated by set_zenith_azimuth.
        saved_dm = self.density_model
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
                    self.set_density_model(tuple(dm_spec))
                    dm_instances[key] = self.density_model
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
                _PATH_WORKER_MCEQ = self  # inherited by forked children
                try:
                    ctx = _mp.get_context("fork")
                    worker_args = [
                        (idx, job[1], job[2], kwargs)
                        for idx, (_, job) in enumerate(jobs)
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
                    if self.density_model is not dm:
                        self.set_density_model(dm)
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

    def _is_geomag_eligible_atmosphere(self):
        """True if the active atmosphere has a meaningful geographic location.

        The :meth:`solve_fullsky` auto-cutoff fires only when this returns
        True and ``self.geomagnetic_cutoff`` is True or None. Eligible
        atmospheres are MSIS*-derived classes (both MSIS00 and MSIS21
        hierarchies) and any other atmosphere whose ``self.location``
        appears in :data:`atmosphere_parameters.LOCATIONS`.
        """
        import MCEq.geometry.density_profiles as dprof
        from MCEq.geometry.atmosphere_parameters import LOCATIONS

        dm = self.density_model
        if dm is None:
            return False
        if isinstance(dm, dprof.MSIS00Atmosphere):
            return True
        # MSIS21 is a parallel class tree (not MSIS00 subclass).
        if hasattr(dprof, "MSIS21Atmosphere") and isinstance(
            dm, dprof.MSIS21Atmosphere
        ):
            return True
        loc = getattr(dm, "location", None)
        return isinstance(loc, str) and loc in LOCATIONS

    def solve_fullsky(
        self,
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
                ``self._phi0``.
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
                this call. ``None`` means "use ``self.geomagnetic_cutoff``"
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
        info(2, f"solve_fullsky: kernel={config.kernel_config}")
        start = time()

        # Resolve geomagnetic-cutoff toggle. Per-call argument has
        # priority; otherwise fall back to instance default; otherwise
        # auto-detect from the atmosphere.
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

        if phi0 is None:
            phi0_arr = self._phi0.copy()
            phi0_is_2d = False
        else:
            phi0_arr = np.asarray(phi0, dtype=np.float64)
            if phi0_arr.ndim == 1:
                if phi0_arr.size != self.dim_states:
                    raise ValueError(
                        f"solve_fullsky: phi0 has length {phi0_arr.size}, "
                        f"expected {self.dim_states}"
                    )
                phi0_is_2d = False
            elif phi0_arr.ndim == 2:
                if phi0_arr.shape[0] != self.dim_states:
                    raise ValueError(
                        f"solve_fullsky: phi0 has shape {phi0_arr.shape}, "
                        f"expected first axis = dim_states = {self.dim_states}"
                    )
                phi0_is_2d = True
            else:
                raise ValueError(
                    f"solve_fullsky: phi0 must be 1-D or 2-D, "
                    f"got shape {phi0_arr.shape}"
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
            or (geomagnetic_cutoff is None and self.geomagnetic_cutoff is True)
        ):
            import warnings

            warnings.warn(
                "solve_fullsky: 2-D phi0 supplied — the geomagnetic cutoff "
                "is NOT applied on top (the caller owns the per-pixel "
                "primary spectrum). Bake the cutoff into phi0 with "
                "MCEq.geometry.gtracr_cutoff.build_phi0_with_cutoff, or "
                "pass geomagnetic_cutoff=False to silence this warning.",
                stacklevel=2,
            )
        if cutoff_flag and not phi0_is_2d:
            from MCEq.geometry.gtracr_cutoff import (
                build_phi0_with_cutoff,
                get_cutoff_map,
            )

            az_centres = azimuth_grid if azimuth_grid is not None else np.array([0.0])
            primary = getattr(self, "pmodel", None)
            if primary is None:
                info(
                    1,
                    "solve_fullsky: geomagnetic_cutoff requested but no "
                    "primary_model is set — skipping cutoff.",
                )
            else:
                ck = dict(cutoff_kwargs or {})
                rc_grid = get_cutoff_map(
                    self.density_model,
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
                phi0_arr = build_phi0_with_cutoff(self, primary, rc_flat)
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

    def _resolve_secant(self):
        """The sec(theta) operator set for a solve, or ``None`` for the
        paraxial transport.

        One constant set serves every column of every batch (the cap is
        not zenith dependent). The coupled route is driver code plus the
        ``apply_off`` binding, so every kernel carries it at every
        precision; only ``config.secant_mode`` decides.
        """
        if config.secant_mode(self._mceq_db.is_2d) == "off":
            return None
        return self._build_secant_ops()

    def _build_secant_ops(self):
        """The constant sec(theta) operator set for the current geometry
        (see :mod:`MCEq.secant`), built once per (geometry, ``secant_*``
        parameters) and reused across solves — the operator and backend
        caches key on its identity."""
        from MCEq.secant import build_secant_kernel_ops

        k_grid = np.asarray(self._mceq_db.k_grid)
        e_centers = np.asarray(self._energy_grid.c)
        n_species = self.dim_states // self.dim
        key = (
            hash(k_grid.tobytes()),
            hash(e_centers.tobytes()),
            n_species,
            config.secant_theta_cap(),
            config.secant_theta_row_kmax,
            config.secant_theta_lam_rel,
            config.secant_theta_w_flat,
            getattr(config, "secant_theta_e_max", None),
        )
        cached = getattr(self, "_secant_ops_cache", None)
        if cached is None or cached[0] != key:
            ops = build_secant_kernel_ops(
                k_grid,
                e_centers,
                n_species,
                config.secant,
                theta_cap_deg=config.secant_theta_cap(),
                paths=config.paths,
            )
            self._secant_ops_cache = cached = (key, ops)
        return cached[1]

    def _compiled_operator(self, sec_ops=None):
        """:func:`~MCEq.operator_assembly.compile_operator` of the current
        matrices, cached against the identity of ``int_m`` / ``dec_m`` and
        of the operator set (a paraxial and a coupled entry coexist)."""
        cache = self.__dict__.setdefault("_operator_cache", {})
        coupled = sec_ops is not None
        entry = cache.get(coupled)
        if (
            entry is None
            or entry[0] is not self.int_m
            or entry[1] is not self.dec_m
            or entry[2] is not sec_ops
        ):
            op = compile_operator(self.int_m, self.dec_m, sec_ops)
            cache[coupled] = entry = (self.int_m, self.dec_m, sec_ops, op)
        return entry[3]

    def _etd2_backend(self, sec_ops=None, dtype=np.float64):
        """The step-loop backend of ``config.kernel_config`` bound to the
        compiled operator: scipy CSR, MKL or Accelerate handles, or the
        device upload. Cached per (kernel, precision, coupling); a backend
        whose operator or device rotated is closed and rebuilt.
        """
        import MCEq.solvers as solvers

        kc = config.kernel_config.lower()
        on_cuda = kc in ("cuda", "cuda_etd2")
        fp = 32 if np.dtype(dtype) == np.float32 else 64
        op = self._compiled_operator(sec_ops)
        cache = self.__dict__.setdefault("_backend_cache", {})
        key = (kc, fp, op.coupled)
        be = cache.get(key)
        if be is not None and (
            be.op is not op or (on_cuda and be.dev.device_id != self._cuda_device)
        ):
            be.close()
            del cache[key]
            be = None
        if be is None:
            if kc == "numpy_etd2":
                be = solvers.numpy_backend(op, fp_precision=fp)
            elif kc in ("accelerate", "accelerate_etd2"):
                be = solvers.accelerate_backend(op, fp_precision=fp)
            elif kc in ("mkl", "mkl_etd2"):
                be = solvers.mkl_backend(op, fp_precision=fp)
            elif on_cuda:
                be = solvers.cuda_backend(
                    op, device_id=self._cuda_device, fp_precision=fp
                )
            else:
                raise Exception(
                    f"Unsupported integrator setting '{config.kernel_config}'. "
                    "Choose one of: numpy_etd2, accelerate_etd2, mkl_etd2, cuda_etd2."
                )
            cache[key] = be
        return be

    def _run_etd2(
        self, nsteps, dX, rho_inv, phi0, grid_idcs, dtype=np.float64, sec_ops=None
    ):
        """One integration path for a ``(dim,)`` or ``(dim, K)`` state —
        the single-axis and the shared-path multi-RHS route."""
        import MCEq.solvers as solvers

        dtype = np.dtype(dtype)
        be = self._etd2_backend(sec_ops, dtype)
        sol, grid_sol = solvers.etd2_driver(nsteps, dX, rho_inv, be, phi0, grid_idcs)
        if dtype != np.float64:
            sol = sol.astype(dtype)
            grid_sol = grid_sol.astype(dtype)
        return sol, grid_sol

    def _run_etd2_carousel(
        self,
        dX_c,
        rho_inv_c,
        phi_initial,
        schedule,
        phi0_per_pixel,
        dtype=np.float64,
        sec_ops=None,
    ):
        """One LPT carousel launch (per-lane paths); returns the
        ``(dim, K_total)`` final states in pixel order."""
        import MCEq.solvers as solvers

        be = self._etd2_backend(sec_ops, dtype)
        sol = solvers.etd2_driver(
            schedule.T,
            dX_c,
            rho_inv_c,
            be,
            phi_initial,
            [],
            schedule=schedule,
            phi0_per_pixel=phi0_per_pixel,
        )
        return np.asarray(sol, dtype=np.dtype(dtype))

    def close(self):
        """Release the backend resources held by this MCEqRun — MKL sparse
        handles, Accelerate slots, GPU buffers. Idempotent; the next solve
        rebuilds the operator and backend caches lazily, so this is also a
        "drop and reset" knob in long-running scripts."""
        for be in self.__dict__.pop("_backend_cache", {}).values():
            try:
                be.close()
            except Exception:
                pass
        self.__dict__.pop("_operator_cache", None)

    def __del__(self):
        # Best-effort cleanup; never raise from __del__.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _em_cascade_step_scale(self):
        """Cure-B stiffness scale r_EM of the e+/-/gamma block of ``int_m``.

        Returns ``spec(int_off_EM)``: the spectral radius of the *off-diagonal*
        part of ``int_m`` restricted to the e+/-/gamma rows and columns, in
        1/(g/cm^2). This is the same quantity (``spec(int_off)``) that fixes
        the legacy ``config.etd2_path['dX_max']`` for the hadronic matrix
        (~0.094), evaluated on the much stiffer EM block (~0.13). The spectral
        radius — not a matrix norm — is the right scale: the EM off-diagonal is
        a near-singular, highly non-normal difference+production operator whose
        1-norm is dominated by the near-cancelling continuous-loss bands and is
        meaningless as a step scale. ``int_m`` is X-constant, so r_EM is a
        single global scalar; it is cached and invalidated when ``int_m`` is
        rebuilt. Returns 0.0 when no e+/-/gamma are loaded (EM cascade inactive)
        or the matrices are not yet built.
        """
        int_m = getattr(self, "int_m", None)
        if int_m is None:
            return 0.0
        cache = getattr(self, "_em_step_scale_cache", None)
        if cache is not None and cache[0] is int_m:
            return cache[1]
        col_ranges = [
            np.arange(p.lidx, p.uidx)
            for p in self.pman.all_particles
            if getattr(p, "is_em", False)
        ]
        if not col_ranges:
            self._em_step_scale_cache = (int_m, 0.0)
            return 0.0
        idx = np.concatenate(col_ranges)
        diag = int_m.diagonal()
        int_off = (int_m - sp.diags(diag, format="csr")).tocsc()
        block = int_off[idx][:, idx].tocsc().astype(np.float64)
        # Spectral radius of the EM off-diagonal. The block is a small
        # sub-system (a few e+/-/gamma species x dim_e, typically < ~2000) and
        # strongly NON-NORMAL (||A||_2 / rho ~ 3). Sparse ``eigs(k=1, 'LM')``
        # routinely FAILS to converge on it (ARPACK DNAUPD finds no eigenvalue
        # to tolerance), and the old except-branch then silently substituted a
        # matrix norm — 2-3x larger than the true spectral radius AND
        # nondeterministic — capping dX far tighter than this method's own
        # docstring promises ("spectral radius, not a matrix norm"). Dense
        # ``eigvals`` is exact, deterministic and cheap at this size; ARPACK +
        # norm survive only as a fallback for an unexpectedly huge block.
        n = block.shape[0]
        try:
            if n <= int(getattr(config, "em_step_dense_eig_max", 4000)):
                ev = np.linalg.eigvals(block.toarray())
            else:
                from scipy.sparse.linalg import eigs

                ev = eigs(
                    block, k=1, which="LM", return_eigenvectors=False, maxiter=10000
                )
            r_em = float(np.max(np.abs(ev)))
        except Exception:
            # Last-resort fall-back (huge block + ARPACK failure): the spectral
            # radius is bounded above by both induced norms; take the smaller
            # so we never under-cap.
            n1 = float(np.abs(block).sum(axis=0).max())
            ninf = float(np.abs(block).sum(axis=1).max())
            r_em = min(n1, ninf)
        self._em_step_scale_cache = (int_m, r_em)
        return r_em

    def _em_cascade_dx_cap(self):
        """Cure-B effective dX cap from the EM-cascade stiffness, or np.inf.

        ``np.inf`` (no cap) when ``config.em_adaptive_step`` is off or the EM
        cascade is inactive, so the legacy schedule is reproduced exactly.
        """
        if not config.em_adaptive_step:
            return np.inf
        r_em = self._em_cascade_step_scale()
        if not (r_em > 0.0):
            return np.inf
        cap = config.em_step_safety / r_em
        info(
            2,
            "EM-adaptive step (cure B): r_EM={:.4g} 1/(g/cm^2) "
            "-> dX_max <= {:.4g} g/cm^2".format(r_em, cap),
        )
        return cap

    def _calculate_integration_path(
        self,
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
        # of the EM block of int_m (no-op when config.em_adaptive_step is
        # off). int_m is X-constant, so this is a single global cap that the
        # density-gradient schedule never relaxes above.
        em_cap = self._em_cascade_dx_cap()
        if np.isfinite(em_cap):
            base_dX_max = dX_max if dX_max is not None else config.etd2_path["dX_max"]
            dX_max = min(base_dX_max, em_cap)
        etd2_params = (X_start, eps, dX_max, dX_min, fd_span)
        cached_etd2_params = getattr(self, "_cached_etd2_path_params", None)

        if (
            self.integration_path
            and np.all(int_grid == self.int_grid)
            and np.all(self.grid_var == grid_var)
            and cached_etd2_params == etd2_params
            and not force
        ):
            info(5, "skipping calculation.")
            return

        self._cached_etd2_path_params = etd2_params
        self.int_grid, self.grid_var = int_grid, grid_var
        if grid_var != "X":
            raise NotImplementedError(
                "Grid variables other than the depth X not supported."
            )

        from MCEq.solvers import etd2_nonuniform_path

        info(
            2,
            "ETD2 non-uniform path (eps={}, dX_max={}, dX_min={}, "
            "fd_span={}, X_start={})".format(
                eps if eps is not None else config.etd2_path["eps"],
                dX_max if dX_max is not None else config.etd2_path["dX_max"],
                dX_min if dX_min is not None else config.etd2_path["dX_min"],
                fd_span if fd_span is not None else config.etd2_path["fd_span"],
                X_start if X_start is not None else config.X_start,
            ),
        )
        self.integration_path = etd2_nonuniform_path(
            self.density_model,
            X_start=X_start,
            eps=eps,
            dX_max=dX_max,
            dX_min=dX_min,
            fd_span=fd_span,
            int_grid=int_grid,
        )

    def n_particles(self, label, grid_idx=None, min_energy_cutoff=1e-1):
        """Returns number of particles of type `label` at a grid step above
        an energy threshold for counting.

        Args:
            label (str): Particle name
            grid_idx (int): Depth grid index (for profiles)
            min_energy_cutoff (float): Energy threshold > mceq_config.e_min
        """
        ie_min = np.argmin(
            np.abs(self.e_bins - self.e_bins[self.e_bins >= min_energy_cutoff][0])
        )
        _e = self.e_bins[ie_min]
        _e_n = self.e_bins[ie_min + 1]
        _e_m = self.e_grid[ie_min]
        info(
            10,
            f"Energy cutoff for particle number calculation {_e:4.3e} GeV",
        )
        info(
            15,
            f"First bin is between {_e:3.2e} and {_e_n:3.2e} with midpoint {_e_m:3.2e}",
        )
        return np.sum(
            self.get_solution(label, mag=0, integrate=True, grid_idx=grid_idx)[ie_min:]
        )

    def n_mu(self, grid_idx=None, min_energy_cutoff=1e-1):
        """Returns the number of positive and negative muons at a grid step above
        `min_energy_cutoff`.

        Args:
            grid_idx (int): Depth grid index (for profiles)
            min_energy_cutoff (float): Energy threshold > mceq_config.e_min

        """
        return self.n_particles(
            "total_mu+", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        ) + self.n_particles(
            "total_mu-", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        )

    def n_e(self, grid_idx=None, min_energy_cutoff=1e-1):
        """Returns the number of electrons plus positrons at a grid step above
        `min_energy_cutoff`.

        Args:
            grid_idx (int): Depth grid index (for profiles)
            min_energy_cutoff (float): Energy threshold > mceq_config.e_min
        """
        return self.n_particles(
            "e+", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        ) + self.n_particles(
            "e-", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        )

    def z_factor(
        self,
        projectile_pdg,
        secondary_pdg,
        definition="primary_e",
        min_energy=0.3,
        use_cs_scaling=True,
    ):
        """Energy dependent Z-factor according to Thunman et al. (1996)"""

        proj = self.pman[projectile_pdg]
        sec = self.pman[secondary_pdg]

        if not proj.is_projectile:
            raise Exception(f"{proj.name} is not a projectile particle.")
        info(10, f"Computing e-dependent Zfactor for {proj.name} -> {sec.name}")
        if not proj.is_secondary(sec):
            raise Exception(f"{sec.name} is not a secondary particle of {proj.name}.")

        if proj == 2112:
            nuc_flux = self.pmodel.p_and_n_flux(self.e_grid)[2]
        else:
            nuc_flux = self.pmodel.p_and_n_flux(self.e_grid)[1]
        zfac = np.zeros(self.dim)

        smat = proj.hadr_yields[sec]
        proj_cs = proj.prod_cross_section()
        zfac = np.zeros_like(self.e_grid)

        if config.has_cuda:
            import cupy

            smat = cupy.asnumpy(smat)
            proj_cs = cupy.asnumpy(proj_cs)
        # Definition wrt CR energy (different from Thunman) on x-axis
        min_idx = 0
        if definition == "primary_e":
            for p_eidx, e in enumerate(self.e_grid):
                if e < min_energy:
                    min_idx = p_eidx
                    continue
                nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
                assert use_cs_scaling is False, (
                    f"cs_scaling has when definition = {definition}"
                )
                cs_fac = 1.0
                zfac[p_eidx] = np.sum(
                    smat[min_idx : p_eidx + 1, p_eidx] * nuc_fac * cs_fac
                )
            return zfac
        else:
            # Like in Thunman et al. 1996
            for p_eidx, e in enumerate(self.e_grid):
                if e < min_energy:
                    continue
                min_idx = p_eidx
                nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
                if use_cs_scaling:
                    cs_fac = np.zeros(p_eidx - min_idx + 1)
                    old_settings = np.seterr(all="ignore")
                    res = proj_cs[p_eidx] / proj_cs[min_idx : p_eidx + 1]
                    np.seterr(**old_settings)
                    cs_fac[(res > 0) & np.isfinite(res)] = res[
                        (res > 0) & np.isfinite(res)
                    ]
                else:
                    cs_fac = 1.0
                zfac[p_eidx] = np.sum(smat[p_eidx, p_eidx:] * nuc_fac * cs_fac)
            return zfac

    def decay_z_factor(self, parent_pdg, child_pdg):
        """Energy dependent Z-factor according to Lipari (1993)."""

        proj = self.pman[parent_pdg]
        sec = self.pman[child_pdg]

        if proj.is_stable:
            raise Exception(f"{proj.name} does not decay.")
        info(
            10,
            f"Computing e-dependent decay Zfactor for {proj.name} -> {sec.name}",
        )
        if not proj.is_child(sec):
            raise Exception(f"{sec.name} is not a a child particle of {proj.name}.")

        cr_gamma = self.pmodel.nucleon_gamma(self.e_grid)
        zfac = np.zeros(self.dim)

        zfac = np.zeros_like(self.e_grid)
        for p_eidx, e in enumerate(self.e_grid):
            # if e < min_energy:
            #     min_idx = p_eidx + 1
            #     continue
            xlab, xdist = proj.dNdec_dxlab(e, sec)
            zfac[p_eidx] = trapz(xlab ** (-cr_gamma[p_eidx] - 2.0) * xdist, x=xlab)
        return zfac


class MatrixBuilder:
    """This class constructs the interaction and decay matrices."""

    def __init__(self, particle_manager, backend):
        self._pman = particle_manager
        self._backend = backend
        self.is_2d = backend.is_2d
        self.n_k = backend.n_k
        self.k_grid = backend.k_grid
        self._energy_grid = self._pman._energy_grid
        self.int_m = None
        self.dec_m = None
        self._construct_differential_operator()

    def construct_matrices(self, skip_decay_matrix=False):
        r"""Constructs the matrices for calculation.

        These are:

        - :math:`\boldsymbol{M}_{int} = (-\boldsymbol{1} +
            \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}`,
        - :math:`\boldsymbol{M}_{dec} = (-\boldsymbol{1} +
            \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}`.

        For debug_levels >= 2 some general information about matrix
        shape and the number of non-zero elements is printed. The
        intermediate matrices :math:`\boldsymbol{C}` and
        :math:`\boldsymbol{D}` are deleted afterwards to save memory.

        Set the ``skip_decay_matrix`` flag to avoid recreating the decay
        matrix. This is not necessary if, for example, particle production
        is modified, or the interaction model is changed.

        Args:
          skip_decay_matrix (bool): Omit re-creating D matrix

        """

        info(
            3,
            f"Start filling matrices. Skip_decay_matrix = {skip_decay_matrix}",
        )

        self._fill_matrices(skip_decay_matrix=skip_decay_matrix)

        cparts = self._pman.cascade_particles

        # interaction part
        # -I + C
        # In first interaction mode it is just C
        self.max_lint = 0.0

        for parent, child in product(cparts, cparts):
            idx = (child.mceqidx, parent.mceqidx)
            # Main diagonal
            if child.mceqidx == parent.mceqidx and parent.can_interact:
                # Subtract unity from the main diagonals
                info(10, "subtracting main C diagonal from", child.name, parent.name)
                if self.is_2d:
                    self.C_blocks[idx][
                        :, np.diag_indices(self.dim)[0], np.diag_indices(self.dim)[1]
                    ] -= 1.0
                else:
                    self.C_blocks[idx][np.diag_indices(self.dim)] -= 1.0

            if idx in self.C_blocks:
                # Multiply with Lambda_int and keep track the maximal
                # interaction length for the calculation of integration steps
                self.max_lint = np.max(
                    [self.max_lint, np.max(parent.inverse_interaction_length())]
                )
                self.C_blocks[idx] *= np.asarray(
                    parent.inverse_interaction_length(), dtype=config.floatlen
                )

            if child.mceqidx == parent.mceqidx and parent.has_contloss:
                pid = abs(parent.pdg_id[0])
                if config.enable_energy_loss:
                    if (
                        pid == 13
                        or (config.enable_em_ion and pid == 11)
                        or (config.generic_losses_all_charged and pid != 11)
                    ):
                        info(5, "Cont. loss for", parent.name)
                        if config.enable_cont_rad_loss:
                            if self.is_2d:
                                self.C_blocks[idx] += self.cont_loss_operator(
                                    parent.pdg_id
                                )[None, :, :]
                            else:
                                self.C_blocks[idx] += self.cont_loss_operator(
                                    parent.pdg_id
                                )

        self.int_m = self._csr_from_blocks(self.C_blocks, apply_muon_scattering=True)
        # -I + D

        if not skip_decay_matrix or self.dec_m is None:
            self.max_ldec = 0.0
            for parent, child in product(cparts, cparts):
                idx = (child.mceqidx, parent.mceqidx)
                # Main diagonal
                if child.mceqidx == parent.mceqidx and not parent.is_stable:
                    # Subtract unity from the main diagonals
                    info(
                        10, "subtracting main D diagonal from", child.name, parent.name
                    )
                    if self.is_2d:
                        self.D_blocks[idx][
                            :,
                            np.diag_indices(self.dim)[0],
                            np.diag_indices(self.dim)[1],
                        ] -= 1.0
                    else:
                        self.D_blocks[idx][np.diag_indices(self.dim)] -= 1.0
                if idx not in self.D_blocks:
                    info(25, parent.pdg_id[0], child.pdg_id, "not in D_blocks")
                    continue
                # Multiply with Lambda_dec and keep track of the
                # maximal decay length for the calculation of integration steps
                self.max_ldec = max(
                    [self.max_ldec, np.max(parent.inverse_decay_length())]
                )

                self.D_blocks[idx] *= np.asarray(
                    parent.inverse_decay_length(), dtype=config.floatlen
                )

            self.dec_m = self._csr_from_blocks(self.D_blocks)

        for mname, mat in [("C", self.int_m), ("D", self.dec_m)]:
            mat_density = float(mat.nnz) / float(np.prod(mat.shape))
            info(5, f"{mname} Matrix info:")
            info(5, f"    density    : {mat_density:3.2%}")
            info(5, "    shape      : {0} x {1}".format(*mat.shape))
            info(5, f"    nnz        : {mat.nnz}")
            info(10, "    sum        :", mat.sum())

        info(3, "Done filling matrices.")

        return self.int_m, self.dec_m

    def _average_operator(self, op_mat):
        """Averages the continuous loss operator by performing
        1/max_step explicit euler steps"""

        n_steps = int(1.0 / config.loss_step_for_average)
        info(
            10,
            f"Averaging continuous loss using {n_steps} intermediate steps.",
        )

        op_step = np.eye(self._energy_grid.d) + op_mat * config.loss_step_for_average
        return np.linalg.matrix_power(op_step, n_steps) - np.eye(self._energy_grid.d)

    def cont_loss_operator(self, pdg_id):
        """Returns continuous loss operator that can be summed with appropriate
        position in the C matrix."""
        op_mat = -np.diag(1 / self._energy_grid.c).dot(
            self.op_matrix.dot(np.diag(self._pman[pdg_id].dEdX))
        )

        if config.average_loss_operator:
            return self._average_operator(op_mat)
        return op_mat

    @property
    def dim(self):
        """Energy grid (dimension)"""
        return int(self._pman.dim)

    @property
    def dim_states(self):
        """Number of cascade particles times dimension of grid
        (dimension of the equation system)"""
        return int(self._pman.dim_states)

    def _zero_mat(self):
        """Returns a new square zero valued matrix with dimensions of grid.

        For 2D databases the per-channel block carries an extra leading
        ``n_k`` axis (one slab per Hankel mode), so the returned shape is
        ``(n_k, dim, dim)`` instead of ``(dim, dim)``.
        """
        if self.is_2d:
            return np.zeros(
                (self.n_k, self._pman.dim, self._pman.dim),
                dtype=config.floatlen,
            )
        return np.zeros((self._pman.dim, self._pman.dim), dtype=config.floatlen)

    def _muon_scattering_damping(self):
        """Per-energy Gaussian multiple-scattering damping data for muons.

        Returns ``(muon_lidcs, theta_s_sq)`` — the state-vector offsets of
        all muon species present (PDG ±13, helicities 0, ±1) and the
        squared scattering angle per unit depth — or ``None`` when muon
        multiple scattering does not apply. The per-mode diagonal
        contribution is ``-kappa^2 * theta_s^2(E) / 4``; it sits on the
        diagonal D so ETD2RK's ``e^{h*D}`` integrates it exactly, without
        a per-step operator split.
        """
        if not (self.is_2d and getattr(config, "muon_multiple_scattering", False)):
            return None
        # Constants of the Gauss approximation used by CORSIKA (Heck &
        # Pierog handbook p.12): Gaussian core only, no Moliere tail.
        lambda_s = 37.7  # g/cm^2
        E_s = 0.021  # GeV
        e_kin = self._energy_grid.c
        # Prefer pman's (13, 0) mass; fall back to the PDG value.
        try:
            mu_mass = float(self._pman[(13, 0)].mass)
        except (KeyError, AttributeError):
            mu_mass = 0.10566  # GeV
        E_lab = e_kin + mu_mass
        p2 = E_lab**2 - mu_mass**2
        p2 = np.where(p2 > 0, p2, 1e-30)
        beta = np.sqrt(p2) / E_lab
        theta_s_sq = (1.0 / lambda_s) * (E_s / (E_lab * beta**2)) ** 2
        muon_lidcs = []
        for pdg in (13, -13):
            for hel in (0, 1, -1):
                key = (pdg, hel)
                if key in self._pman.pdg2pref:
                    p = self._pman.pdg2pref[key]
                    if hasattr(p, "lidx") and getattr(p, "mceqidx", -1) >= 0:
                        muon_lidcs.append(p.lidx)
        if not muon_lidcs:
            return None
        return muon_lidcs, theta_s_sq

    def _csr_from_blocks(self, blocks, apply_muon_scattering=False):
        """Assemble the per-channel blocks into the global CSR operator.

        For 1D databases each block is a dense ``(dim, dim)`` channel
        matrix placed at its (child, parent) offsets in a single
        ``(dim_states, dim_states)`` sparse matrix.

        For 2D databases each block carries a leading ``n_k`` axis (one
        slab per Hankel mode, shape ``(n_k, dim, dim)``). The Hankel modes
        are mutually decoupled, so the operator is block-diagonal in
        kappa: per mode, the nonzero entries of every channel slab are
        scattered (with their global row/column offsets) into one COO
        triplet set and converted to CSR in one shot — no dense
        ``(dim_states, dim_states)`` intermediate. ``scipy.sparse.
        block_diag`` then stitches the ``n_k`` mode matrices into the
        final ``(n_k * dim_states, n_k * dim_states)`` CSR that the
        dimension-agnostic ETD2RK kernels consume.

        When ``apply_muon_scattering`` is True (the interaction-matrix
        path) and ``config.muon_multiple_scattering`` is on, muon-row
        diagonals receive the per-mode Gaussian multiple-scattering
        damping ``-kappa^2 * theta_s^2(E) / 4`` (see
        :meth:`_muon_scattering_damping`), added as extra COO entries
        (duplicates are summed on CSR conversion).

        ``config.secant_theta_transport`` does not alter these matrices;
        the sec(theta) mode coupling is applied inside the ETD2RK secant
        kernels (see :mod:`MCEq.secant` for why it is not stitched into
        the CSR).
        """
        from scipy.sparse import coo_matrix, csr_matrix

        if self.is_2d:
            mu_damp = None
            if apply_muon_scattering:
                mu_damp = self._muon_scattering_damping()
            n_e = self.dim
            shape = (self.dim_states, self.dim_states)
            per_mode_csr = []
            for k in range(self.n_k):
                rows, cols, vals = [], [], []
                for (c, p), d in iter(blocks.items()):
                    rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
                    slab = d[k]
                    if slab.shape != (rc.uidx - rc.lidx, rp.uidx - rp.lidx):
                        _d = self.dim_states
                        raise Exception(
                            "Dimension mismatch: matrix "
                            + f"{_d}x{_d}, p={rp.name}:({rp.lidx},{rp.uidx}),"
                            + f" c={rc.name}:({rc.lidx},{rc.uidx})"
                        )
                    r, cc = np.nonzero(slab)
                    rows.append(r + rc.lidx)
                    cols.append(cc + rp.lidx)
                    vals.append(slab[r, cc])
                kappa = self.k_grid[k]
                if mu_damp is not None and kappa != 0:
                    muon_lidcs, theta_s_sq = mu_damp
                    damping = -(kappa**2) * theta_s_sq / 4.0
                    for lidx in muon_lidcs:
                        diag = np.arange(lidx, lidx + n_e)
                        rows.append(diag)
                        cols.append(diag)
                        vals.append(damping)
                if rows:
                    m = coo_matrix(
                        (
                            np.concatenate(vals).astype(config.floatlen),
                            (np.concatenate(rows), np.concatenate(cols)),
                        ),
                        shape=shape,
                    ).tocsr()
                else:
                    m = csr_matrix(shape, dtype=config.floatlen)
                m.eliminate_zeros()
                m.sort_indices()
                per_mode_csr.append(m)
            stitched = sp.block_diag(per_mode_csr, format="csr")
            stitched.eliminate_zeros()
            stitched.sort_indices()
            return stitched

        new_mat = np.zeros((self.dim_states, self.dim_states), dtype=config.floatlen)
        for (c, p), d in iter(blocks.items()):
            rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
            try:
                new_mat[rc.lidx : rc.uidx, rp.lidx : rp.uidx] = d
            except ValueError:
                _d = self.dim_states
                _n = rp.name
                _l = rp.lidx
                _u = rp.uidx
                _nc = rc.name
                _lc = rc.lidx
                _uc = rc.uidx
                raise Exception(
                    "Dimension mismatch: matrix "
                    + f"{_d}x{_d}, p={_n}:({_l},{_u}), c={_nc}:({_lc},{_uc})"
                )
        return csr_matrix(new_mat)

    def _follow_chains(self, p, pprod_mat, p_orig, propmat, reclev=0):
        """Recursively project ``p_orig``'s production through resonance
        children of ``p`` into ``propmat``.

        For each child ``d`` of ``p``:

        * If ``d`` is *not* a resonance, ``d`` has its own state-vector slot,
          so we add a direct contribution ``propmat[d, p_orig] += d's
          production matrix · pprod_mat`` and stop.
        * If ``d`` *is* a resonance (set via ``adv_set["force_resonance"]``),
          ``d`` has no slot of its own, so we fold its production into
          ``p_orig``'s row by multiplying through and recursing into ``d``'s
          own children.
        """
        info(40, reclev * "\t", "entering with", p.name)
        for d in p.children:
            info(40, reclev * "\t", "following to", d.name)
            if not d.is_resonance:
                dprop = self._zero_mat()
                p._assign_decay_dist(d, dprop)
                propmat[(d.mceqidx, p_orig.mceqidx)] += dprop.dot(pprod_mat)
                info(20, reclev * "\t", "\t terminating at", d.name)
            else:
                dres = self._zero_mat()
                p._assign_decay_dist(d, dres)
                self._follow_chains(d, dres.dot(pprod_mat), p_orig, propmat, reclev + 1)

    def _fill_matrices(self, skip_decay_matrix=False):
        """Generates the interaction and decay matrices from scratch."""
        from collections import defaultdict

        # Fill decay matrix blocks
        if not skip_decay_matrix or self.dec_m is None:
            # Initialize empty D matrix
            self.D_blocks = defaultdict(lambda: self._zero_mat())
            for p in self._pman.cascade_particles:
                # Fill parts of the D matrix related to p as mother
                if not p.is_stable and bool(p.children) and not p.is_tracking:
                    self._follow_chains(
                        p,
                        np.diag(np.ones(self.dim)).astype(config.floatlen),
                        p,
                        self.D_blocks,
                        reclev=0,
                    )
                else:
                    info(20, p.name, "stable or not added to D matrix")

        # Initialize empty C blocks
        self.C_blocks = defaultdict(lambda: self._zero_mat())
        for p in self._pman.cascade_particles:
            # if p doesn't interact, skip interaction matrices
            if not p.is_projectile:
                if p.is_hadron:
                    info(1, f"No interactions by {p.name} ({p.pdg_id}).")
                continue
            for s in p.hadr_secondaries:
                cmat = self._zero_mat()
                p._assign_hadr_dist(s, cmat)
                if not s.is_resonance:
                    # s has its own state-vector slot — direct entry.
                    self.C_blocks[(s.mceqidx, p.mceqidx)] += cmat
                else:
                    # s is folded — recurse into its children.
                    self._follow_chains(s, cmat, p, self.C_blocks, reclev=1)

    def _construct_differential_operator(self):
        """Constructs a derivative operator for the continuous losses.

        Builds a (dim_e x dim_e) banded matrix that approximates d/du with
        u = ln E on the (log-uniform) energy grid. The interior 7-point
        stencil is selected by :data:`MCEq.config.loss_stencil_method`:

        - ``"expfit_low_upwind2"`` (default) / ``"expfit_low_upwind"``:
          expfit interior with the low-energy boundary layer
          (:data:`MCEq.config.loss_stencil_low_upwind_rows` rows) replaced
          by monotone second-/first-order upwind rows — removes the
          low-energy boundary cliff of the pure expfit operator.
        - ``"expfit"``: exponentially-fitted 7-point stencil anchored
          at :data:`MCEq.config.loss_stencil_alpha0`. Near-exact for power-law
          spectra E^{-alpha} with alpha ~ alpha0 on a coarse log grid.
        - ``"centered"``: symmetric 6th-order centered FD.
        - ``"biased"``: legacy 7-point biased "6th-order" stencil.

        The non-upwind options share the same one-sided polynomial-fit
        stencils on the boundary rows (0, 1, 2 and last-2, last-1, last); see
        ``docs/mceq_v1.x_v2_diff.md`` for the boundary-cliff caveat.
        """
        # First rows of operator matrix (values are truncated at the edges
        # of a matrix.)
        diags_leftmost = [0, 1, 2, 3]
        coeffs_leftmost = [-11, 18, -9, 2]
        denom_leftmost = 6
        diags_left_1 = [-1, 0, 1, 2, 3]
        coeffs_left_1 = [-3, -10, 18, -6, 1]
        denom_left_1 = 12
        diags_left_2 = [-2, -1, 0, 1, 2, 3]
        coeffs_left_2 = [3, -30, -20, 60, -15, 2]
        denom_left_2 = 60

        # Last rows at the right of operator matrix
        diags_right_2 = [-d for d in diags_left_2[::-1]]
        coeffs_right_2 = [-d for d in coeffs_left_2[::-1]]
        denom_right_2 = denom_left_2
        diags_right_1 = [-d for d in diags_left_1[::-1]]
        coeffs_right_1 = [-d for d in coeffs_left_1[::-1]]
        denom_right_1 = denom_left_1
        diags_rightmost = [-d for d in diags_leftmost[::-1]]
        coeffs_rightmost = [-d for d in coeffs_leftmost[::-1]]
        denom_rightmost = denom_leftmost

        h = np.log(self._energy_grid.b[1:] / self._energy_grid.b[:-1])
        dim_e = int(self._energy_grid.d)
        last = dim_e - 1

        # Interior stencil selection. All options are 7-point and span at
        # most [-3, +3], so the row range range(3, dim_e - 3) is uniform.
        method = getattr(config, "loss_stencil_method", "expfit_low_upwind2")
        low_boundary = None
        if method in ("expfit_low_upwind", "expfit_low_upwind2"):
            low_boundary = method.rsplit("_", 1)[-1]
            method = "expfit"

        # Simple monotone UPWIND stencils (full-matrix, no high-order boundary
        # rows → avoids the expfit "boundary cliff"). Energy loss advects the
        # spectrum toward lower E, so the upwind direction is toward HIGHER E
        # (forward in u = ln E), matching the one-sided orientation of the
        # low-E boundary row. Added for the fine-grid Nmax convergence study
        # (runs/2026-06-06_em-grid-exact-nmax): unconditionally stable, diffusive
        # at O(h) (upwind) / O(h²) (upwind2), converges as the grid refines.
        if method in ("upwind", "upwind2"):
            op_matrix = np.zeros((dim_e, dim_e), dtype=config.floatlen)
            if method == "upwind":  # 1st-order forward difference
                for row in range(dim_e - 1):
                    op_matrix[row, row] = -1.0 / h[row]
                    op_matrix[row, row + 1] = 1.0 / h[row]
                # top edge: 1st-order backward (forward out of range)
                op_matrix[last, last - 1] = -1.0 / h[last - 1]
                op_matrix[last, last] = 1.0 / h[last - 1]
            else:  # "upwind2": 2nd-order forward-biased
                for row in range(dim_e - 2):
                    op_matrix[row, row] = -1.5 / h[row]
                    op_matrix[row, row + 1] = 2.0 / h[row]
                    op_matrix[row, row + 2] = -0.5 / h[row]
                # top two edges: 2nd-order backward-biased
                for row in (dim_e - 2, last):
                    hh = h[row - 2]
                    op_matrix[row, row - 2] = 0.5 / hh
                    op_matrix[row, row - 1] = -2.0 / hh
                    op_matrix[row, row] = 1.5 / hh
            self.op_matrix = op_matrix
            return

        if method == "biased":
            diags_int = np.asarray(diags_left_2)
            coeffs_int = np.asarray(coeffs_left_2, dtype=np.float64) / 60.0
        elif method == "centered":
            diags_int = np.asarray([-3, -2, -1, 1, 2, 3])
            coeffs_int = np.asarray([-1, 9, -45, 45, -9, 1], dtype=np.float64) / 60.0
        elif method == "expfit":
            alpha0 = float(getattr(config, "loss_stencil_alpha0", 3.0))
            diags_int = np.arange(-3, 4)
            # Use the mean log-spacing for a single fit (grid is log-uniform).
            h_avg = float(np.mean(h))
            deltas = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            a = -alpha0 + deltas
            Aexp = np.exp(np.outer(a, diags_int) * h_avg)
            rhs = a * h_avg
            coeffs_int = np.linalg.solve(Aexp, rhs)
        else:
            raise ValueError(
                f"Unknown loss_stencil_method: {method!r}. "
                "Expected 'expfit', 'centered', 'biased', 'upwind', 'upwind2', "
                "'expfit_low_upwind', or 'expfit_low_upwind2'."
            )

        op_matrix = np.zeros((dim_e, dim_e), dtype=config.floatlen)
        op_matrix[0, np.asarray(diags_leftmost)] = np.asarray(coeffs_leftmost) / (
            denom_leftmost * h[0]
        )
        op_matrix[1, 1 + np.asarray(diags_left_1)] = np.asarray(coeffs_left_1) / (
            denom_left_1 * h[1]
        )
        op_matrix[2, 2 + np.asarray(diags_left_2)] = np.asarray(coeffs_left_2) / (
            denom_left_2 * h[2]
        )
        op_matrix[last, last + np.asarray(diags_rightmost)] = np.asarray(
            coeffs_rightmost
        ) / (denom_rightmost * h[last])
        op_matrix[last - 1, last - 1 + np.asarray(diags_right_1)] = np.asarray(
            coeffs_right_1
        ) / (denom_right_1 * h[last - 1])
        op_matrix[last - 2, last - 2 + np.asarray(diags_right_2)] = np.asarray(
            coeffs_right_2
        ) / (denom_right_2 * h[last - 2])
        for row in range(3, dim_e - 3):
            op_matrix[row, row + diags_int] = coeffs_int / h[row]

        if low_boundary is not None:
            n_low = min(
                max(3, int(getattr(config, "loss_stencil_low_upwind_rows", 8))),
                dim_e - 2,
            )
            op_matrix[:n_low, :] = 0.0
            if low_boundary == "upwind":
                for row in range(n_low):
                    op_matrix[row, row] = -1.0 / h[row]
                    op_matrix[row, row + 1] = 1.0 / h[row]
            elif low_boundary == "upwind2":
                for row in range(n_low):
                    op_matrix[row, row] = -1.5 / h[row]
                    op_matrix[row, row + 1] = 2.0 / h[row]
                    op_matrix[row, row + 2] = -0.5 / h[row]

        self.op_matrix = op_matrix
