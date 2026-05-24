from time import time

import numpy as np
import scipy.sparse as sp
import six

import MCEq.data
from MCEq import config
from MCEq.misc import info, normalize_hadronic_model_name
from MCEq.particlemanager import ParticleManager

# trapz was finally removed with numpy 2.4
if hasattr(np, "trapezoid"):
    trapz = np.trapezoid
else:
    trapz = np.trapz


# Module-level worker state for the optional process-pool path build
# inside :meth:`MCEqRun._build_pixel_paths`. Workers fork from the parent
# and inherit ``_PATH_WORKER_MCEQ`` via copy-on-write — the MCEqRun
# instance itself never has to be picklable. Each worker process gets
# its own CoW copy of the density model, so per-worker
# ``set_zenith_azimuth`` mutations stay process-local. Only used when
# ``solve_fullsky(path_workers=N>0)`` is requested *and* the atmosphere
# is not azimuth-symmetric (MSIS location-centered case).
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
        config.ensure_db_available()
        self.medium = kwargs.pop("medium", config.interaction_medium)
        self._mceq_db = MCEq.data.HDF5Backend(medium=self.medium)

        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_model = kwargs.pop("density_model", config.density_model)
        self.theta_deg = theta_deg

        #: Interface to interaction tables of the HDF5 database
        self._interactions = MCEq.data.Interactions(mceq_hdf_db=self._mceq_db)

        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self._int_cs = MCEq.data.InteractionCrossSections(
            mceq_hdf_db=self._mceq_db, interaction_model=interaction_model
        )

        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self._cont_losses = MCEq.data.ContinuousLosses(mceq_hdf_db=self._mceq_db)

        #: Interface to decay tables of the HDF5 database
        self._decays = MCEq.data.Decays(mceq_hdf_db=self._mceq_db)

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
        else:
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
        else:
            return etot_grid

    def xgrid(self, particle_name, return_as, return_bins=False):
        """Uniform access to the spectrum variable, depending on the
        same `return_as` argument as in get_solution."""

        if return_as == "kinetic energy":
            return (self.e_bins, self.e_grid) if return_bins else self.e_grid
        elif return_as == "total energy":
            return self.etot_grid(particle_name, return_bins)
        elif return_as == "total momentum":
            return self.ptot_grid(particle_name, return_bins)
        else:
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
                "The orders of the state vecs don't match {0}!={1}".format(
                    order_i, order
                )
            )
        elif order_i != order and only_available:
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

        res = np.zeros(self._energy_grid.d)
        ref = self.pman.pname2pref
        sol = None
        if grid_idx is not None and len(self.grid_sol) == 0:
            raise Exception("Solution not has not been computed on grid. Check input.")
        if grid_idx is None:
            sol = np.copy(self._solution)
        elif grid_idx >= len(self.grid_sol):
            sol = self.grid_sol[-1, :]
        else:
            sol = self.grid_sol[grid_idx, :]

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
            res = self.get_solution(
                "total_" + lep_str,
                mag=0,
                grid_idx=grid_idx,
                integrate=False,
                return_as="kinetic energy",
            ) - self.get_solution(
                "pr_" + lep_str,
                mag=0,
                grid_idx=grid_idx,
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
            else:
                return res * etot_grid**mag * self.e_widths

        elif return_as == "kinetic energy":
            if not integrate:
                return res * self._energy_grid.c**mag
            else:
                return res * self._energy_grid.c**mag * self.e_widths

        elif return_as == "total momentum":
            ptot_bins, ptot_grid = self.ptot_grid(lep_str, return_bins=True)
            dEkindp = np.diff(ptot_bins) / self.e_widths
            if not integrate:
                return dEkindp * res * ptot_grid**mag
            else:
                return dEkindp * res * ptot_grid**mag * np.diff(ptot_bins)

        else:
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
                self._particle_list, self._energy_grid, self._int_cs, self.medium
            )
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)
            self.matrix_builder = MatrixBuilder(self.pman)

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

    def enable_em_density_interpolation(self, rho_grid=None):
        """Build an int_m stack indexed by air density ρ for LPM-realistic runs.

        Reads the ρ-stack written by mceq-maintenance-tools's
        ``5_assemble_em_db --air-density-grid`` from the active EM HDF5 file
        and rebuilds the interaction matrix once per slice.  The solver
        kernel (currently only ``numpy_etd2``) will log-linear-interpolate
        between bracketing slices at each integration step using ρ(X).

        Default behaviour (no call) keeps the single-density legacy slice —
        which for air showers below ~10 EeV is what the project's
        validated CORSIKA closures use, so this method is opt-in.

        Args:
          rho_grid: 1-D array of densities (g/cm³).  Defaults to the
            ``rho_grid`` dataset present in the EM DB.

        Raises:
          RuntimeError: when ``config.enable_em`` is False, the EM DB has
            no ρ-stack, or the medium is not air.
        """
        if not config.enable_em:
            raise RuntimeError("EM module disabled (config.enable_em is False).")
        if self.medium != "air":
            raise RuntimeError(
                f"ρ-stratified EM is only available for the air medium "
                f"(self.medium={self.medium!r})."
            )
        if rho_grid is None:
            rho_grid = self._mceq_db.em_rho_grid(self.medium)
        if rho_grid is None or len(rho_grid) < 2:
            raise RuntimeError(
                "No ρ-stack in the active EM database — build one with "
                "5_assemble_em_db --air-density-grid=lo,hi,N."
            )

        info(
            1,
            f"Building int_m stack for {len(rho_grid)} ρ slices "
            f"({float(rho_grid[0]):.2e} – {float(rho_grid[-1]):.2e} g/cm³)...",
        )
        prev_density = getattr(config, "em_air_density", None)
        int_m_stack = []
        try:
            for rho in rho_grid:
                config.em_air_density = float(rho)
                # Force-reload EM cross sections and yield matrices for this slice.
                self._int_cs.load(self._interactions.iam)
                self._interactions.load(
                    self._interactions.iam, parent_list=self._particle_list
                )
                self.pman.set_interaction_model(self._int_cs, self._interactions)
                int_m_slice, _ = self.matrix_builder.construct_matrices(
                    skip_decay_matrix=True
                )
                int_m_stack.append(int_m_slice)
        finally:
            config.em_air_density = prev_density
            # Restore the working int_m to the previously active density slice.
            self._int_cs.load(self._interactions.iam)
            self._interactions.load(
                self._interactions.iam, parent_list=self._particle_list
            )
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
                skip_decay_matrix=False
            )

        self._int_m_stack = int_m_stack
        self._em_rho_grid = np.asarray(rho_grid, dtype=float)
        info(1, f"int_m stack ready ({len(int_m_stack)} slices).")

    def disable_em_density_interpolation(self):
        """Drop the ρ-stack; subsequent solves use the single int_m."""
        if hasattr(self, "_int_m_stack"):
            del self._int_m_stack
        if hasattr(self, "_em_rho_grid"):
            del self._em_rho_grid

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
        # the density model's set_theta signature. Both
        # MSIS00LocationCentered and MSIS21LocationCentered accept the
        # extra azimuth_deg argument; everything else ignores azimuth.
        import inspect as _inspect

        _az_aware = "azimuth_deg" in _inspect.signature(
            self.density_model.set_theta
        ).parameters
        if _az_aware:
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
          kwargs (dict): Arguments are passed directly to the solver
            methods. ``X_start`` is honoured by all kernels (defaults to
            ``config.X_start = 0``). ``eps`` / ``dX_max`` / ``dX_min`` /
            ``fd_span`` control the ETD2 non-uniform schedule; pass them
            here to override the defaults in ``config.etd2_path``.

        """
        info(2, f"Launching {config.kernel_config} solver")

        if not kwargs.pop("skip_integration_path", False):
            if int_grid is not None and np.any(np.diff(int_grid) < 0):
                raise Exception(
                    "The X values in int_grid are required to be strickly",
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

        phi0 = np.copy(self._phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, f"for {nsteps} integration steps.")

        start = time()

        kernel, args = self._build_kernel_dispatch(nsteps, dX, rho_inv, phi0, grid_idcs)

        self._solution, self.grid_sol = kernel(*args)

        if isinstance(self.grid_sol, list):
            self.grid_sol = np.asarray(self.grid_sol)

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

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

        phi0 = np.copy(self._phi0)

        kernel, args = self._build_kernel_dispatch(nsteps, dX, rho_inv, phi0, grid_idcs)

        self._solution, self.grid_sol = kernel(*args)

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def _dispatch_mkl_multirhs(self, nsteps, dX, rho_inv, grid_idcs, phi0, dtype):
        """Pick the MKL multirhs kernel by ``dtype`` and reuse a per-dtype
        sparse-handle cache. The MKL handle owns the optimised internal
        layout (after ``mkl_sparse_optimize``) — reusing the handle across
        a multi-RHS solve is what amortises that cost. fp64 lives in
        ``self._mkl_etd2_cache_multirhs``; fp32 in
        ``self._mkl_etd2_cache_multirhs_f32``.
        """
        import MCEq.solvers
        from MCEq.solvers import _etd_split_cache

        if dtype == np.float64:
            cache_attr = "_mkl_etd2_cache_multirhs"
            matrix_cls = MCEq.solvers.MklSparseMatrix
            solver = MCEq.solvers.solv_mkl_etd2_multirhs
        else:
            cache_attr = "_mkl_etd2_cache_multirhs_f32"
            matrix_cls = MCEq.solvers.MklSparseMatrixF32
            solver = MCEq.solvers.solv_mkl_etd2_multirhs_f32

        cached = getattr(self, cache_attr, None)
        if (
            cached is None
            or cached["int_m"] is not self.int_m
            or cached["dec_m"] is not self.dec_m
        ):
            d_int, d_dec, int_off, dec_off = _etd_split_cache(self.int_m, self.dec_m)
            new_cached = {
                "int_m": self.int_m,
                "dec_m": self.dec_m,
                "mkl_int_off": matrix_cls(int_off) if int_off.nnz else None,
                "mkl_dec_off": matrix_cls(dec_off) if dec_off.nnz else None,
                "d_int": d_int,
                "d_dec": d_dec,
            }
            old_cached = cached
            setattr(self, cache_attr, new_cached)
            if old_cached is not None:
                for key in ("mkl_int_off", "mkl_dec_off"):
                    old = old_cached.get(key)
                    if old is not None:
                        old.close()
        c = getattr(self, cache_attr)
        return solver(
            nsteps,
            dX,
            rho_inv,
            c["mkl_int_off"],
            c["mkl_dec_off"],
            c["d_int"],
            c["d_dec"],
            phi0,
            grid_idcs,
        )

    def _dispatch_cuda_multirhs(self, nsteps, dX, rho_inv, grid_idcs, phi0, dtype):
        """Pick the cupy multirhs kernel and reuse a per-(dtype, K) context
        cache. The context owns the cuSPARSE CSR copies of the off-diagonals
        and the (dim, K) state/scratch buffers, so reconstructing them costs
        a non-trivial number of allocations + CSR builds. Cached in
        ``self._cuda_etd2_multirhs_cache`` keyed on (dtype, K) and tied
        to the current ``int_m`` / ``dec_m`` identity.
        """
        import MCEq.solvers
        from MCEq.solvers import _etd_split_cache

        fp_precision = 32 if dtype == np.float32 else 64
        dim, K = phi0.shape
        cache_key = (fp_precision, K)
        cache = getattr(self, "_cuda_etd2_multirhs_cache", None)
        if cache is None:
            cache = {}
            self._cuda_etd2_multirhs_cache = cache

        entry = cache.get(cache_key)
        if (
            entry is None
            or entry["int_m"] is not self.int_m
            or entry["dec_m"] is not self.dec_m
        ):
            d_int, d_dec, int_off, dec_off = _etd_split_cache(
                self.int_m, self.dec_m
            )
            device_id = int(getattr(config, "cuda_device_id", 0))
            ctx = MCEq.solvers.CudaEtd2MultiRHSContext(
                int_off,
                dec_off,
                d_int,
                d_dec,
                K=K,
                device_id=device_id,
                fp_precision=fp_precision,
            )
            cache[cache_key] = {
                "int_m": self.int_m,
                "dec_m": self.dec_m,
                "ctx": ctx,
            }
            entry = cache[cache_key]
        ctx = entry["ctx"]
        return MCEq.solvers.solv_cuda_etd2_multirhs(
            nsteps,
            dX,
            rho_inv,
            ctx,
            phi0,
            grid_idcs,
        )

    def _dispatch_spacc_multirhs(self, nsteps, dX, rho_inv, grid_idcs, phi0, dtype):
        """Pick the spacc multirhs kernel by ``dtype`` and reuse a per-dtype
        sparse-handle cache so the Sparse BLAS optimisation cost is paid
        once per ``MCEqRun`` instance per dtype. fp64 lives in
        ``self._spacc_etd2_cache`` (shared with the single-RHS spacc path);
        fp32 lives in ``self._spacc_etd2_cache_f32``.
        """
        import MCEq.spacc as spacc
        from MCEq.solvers import _etd_split_cache

        if dtype == np.float64:
            cache_attr = "_spacc_etd2_cache"
            matrix_cls = spacc.SpaccMatrix
            solver = MCEq.solvers.solv_spacc_etd2_multirhs
        else:  # float32
            cache_attr = "_spacc_etd2_cache_f32"
            matrix_cls = spacc.SpaccMatrixF32
            solver = MCEq.solvers.solv_spacc_etd2_multirhs_f32

        cached = getattr(self, cache_attr, None)
        if (
            cached is None
            or cached["int_m"] is not self.int_m
            or cached["dec_m"] is not self.dec_m
        ):
            d_int, d_dec, int_off, dec_off = _etd_split_cache(self.int_m, self.dec_m)
            new_cached = {
                "int_m": self.int_m,
                "dec_m": self.dec_m,
                "spacc_int_off": matrix_cls(int_off) if int_off.nnz else None,
                "spacc_dec_off": matrix_cls(dec_off) if dec_off.nnz else None,
                "d_int": d_int,
                "d_dec": d_dec,
            }
            old_cached = cached
            setattr(self, cache_attr, new_cached)
            if old_cached is not None:
                for key in ("spacc_int_off", "spacc_dec_off"):
                    old = old_cached.get(key)
                    if old is not None:
                        old.close()
        c = getattr(self, cache_attr)
        return solver(
            nsteps,
            dX,
            rho_inv,
            c["spacc_int_off"],
            c["spacc_dec_off"],
            c["d_int"],
            c["d_dec"],
            phi0,
            grid_idcs,
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

        Mirrors :meth:`solve` but operates on a ``(dim_states, K)`` initial
        state matrix instead of the single-RHS ``self._phi0``. Each
        column ``phi0_matrix[:, k]`` is an independent initial spectrum,
        layout identical to ``self._phi0`` (i.e. composed in the same way
        you would prepare ``set_initial_spectrum`` / ``set_single_primary_particle``
        calls). Per-step work that depends only on ``(X, ρ⁻¹(X))`` —
        the diagonal split, ``exp(h·D)``, ``φ₁(h·D)``, ``φ₂(h·D)`` — is
        computed once per step and broadcast over the K column axis.

        Selection rules:

        * ``kernel_config == "numpy_etd2"`` →
          :func:`MCEq.solvers.solv_numpy_etd2_multirhs` (CSR-SpMM through
          scipy ``@`` on a 2-D RHS).
        * ``kernel_config == "accelerate_etd2"`` →
          :func:`MCEq.solvers.solv_spacc_etd2_multirhs`
          (Accelerate Sparse BLAS ``sparse_matrix_product_dense_double``).
        * Other kernels raise ``NotImplementedError`` (MKL Sparse BLAS
          and cuSPARSE multi-RHS variants are not yet wired).

        Does NOT mutate ``self._phi0`` / ``self._solution`` /
        ``self.grid_sol``. The returned arrays are the entire state of
        the multi-RHS solve; the caller indexes columns to retrieve
        per-RHS final spectra or grid snapshots.

        Args:
          phi0_matrix (np.ndarray[dim_states, K]): initial state matrix.
            Each column carries one independent initial spectrum.
          int_grid (list | None): X values at which to record snapshots,
            shared across all K columns. Same semantics as :meth:`solve`.
          grid_var (str): currently only ``"X"`` is supported.
          dtype (np.float32 | np.float64): precision of the state buffers
            and the SpMM. Default ``np.float64``. When ``np.float32`` and
            ``kernel_config == "accelerate_etd2"``, the solver dispatches
            to :func:`MCEq.solvers.solv_spacc_etd2_multirhs_f32` with a
            cached :class:`MCEq.spacc.SpaccMatrixF32` handle; ~1.10–1.14×
            faster per-RHS on Mac M3 Pro at K ≥ 64, with per-particle
            relative error ≤ 1e-5 vs the fp64 reference (verified by
            ``runs/2026-05-21_multi-rhs-etd2-prototype/inputs/test_etd2_fp32.py``).
            ``np.float32`` is not yet wired for ``numpy_etd2`` — the
            scipy CSR @ X path would need fp32 versions of
            ``int_m``/``dec_m`` and the numpy kernel; defer until needed.
          X_start, eps, dX_max, dX_min, fd_span (float | None): ETD2
            non-uniform path knobs forwarded to
            :meth:`_calculate_integration_path`; same semantics as
            :meth:`solve`.

        Returns:
          (np.ndarray[dim_states, K], np.ndarray[len(int_grid), dim_states, K]):
          final state matrix and stacked snapshots. The trailing K axis
          is preserved; index by column for per-RHS retrieval. The
          ``sol`` dtype matches the ``dtype`` argument.
        """
        import MCEq.solvers

        if phi0_matrix.ndim != 2:
            raise ValueError(
                f"solve_multirhs: phi0_matrix must be 2-D (dim_states, K), "
                f"got shape {phi0_matrix.shape}"
            )
        if phi0_matrix.shape[0] != self.dim_states:
            raise ValueError(
                f"solve_multirhs: phi0_matrix.shape[0] ({phi0_matrix.shape[0]}) "
                f"must equal self.dim_states ({self.dim_states})"
            )

        info(
            2,
            f"Launching {config.kernel_config} multi-RHS solver "
            f"(K={phi0_matrix.shape[1]})",
        )

        if int_grid is not None and np.any(np.diff(int_grid) < 0):
            raise Exception(
                "The X values in int_grid are required to be strictly increasing."
            )
        self._calculate_integration_path(
            int_grid,
            grid_var,
            X_start=X_start,
            eps=eps,
            dX_max=dX_max,
            dX_min=dX_min,
            fd_span=fd_span,
        )
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        kc = config.kernel_config.lower()
        # ``dtype`` controls the state-buffer precision; the diagonals
        # ``d_int`` / ``d_dec`` remain fp64 in the diag-factor pipeline
        # for the fp32 path (exp(h·D) saturates fp32 fast at high zenith).
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                f"solve_multirhs: dtype must be float32 or float64, got {dtype}"
            )
        phi0 = np.asarray(phi0_matrix, dtype=dtype).copy()

        start = time()

        if kc == "numpy_etd2":
            if dtype == np.float32:
                raise NotImplementedError(
                    "solve_multirhs(dtype=float32) is currently wired only for "
                    "kernel_config in {'accelerate_etd2', 'cuda_etd2', "
                    "'mkl_etd2'}. A scipy fp32 path would need fp32 versions "
                    "of int_m / dec_m and the numpy multirhs kernel — defer "
                    "until needed."
                )
            # If a ρ-stack has been built (via enable_em_density_interpolation),
            # route to the ρ-aware multi-RHS kernel so per-step log-linear
            # blending of the air block kicks in for all K columns.
            int_m_stack = getattr(self, "_int_m_stack", None)
            em_rho_grid = getattr(self, "_em_rho_grid", None)
            if int_m_stack is not None and em_rho_grid is not None:
                sol, grid_sol = MCEq.solvers.solv_numpy_etd2_rho_stack_multirhs(
                    nsteps,
                    dX,
                    rho_inv,
                    int_m_stack,
                    em_rho_grid,
                    self.dec_m,
                    phi0,
                    grid_idcs,
                )
            else:
                sol, grid_sol = MCEq.solvers.solv_numpy_etd2_multirhs(
                    nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs
                )
        elif kc == "accelerate_etd2":
            sol, grid_sol = self._dispatch_spacc_multirhs(
                nsteps, dX, rho_inv, grid_idcs, phi0, dtype
            )
        elif kc == "cuda_etd2":
            sol, grid_sol = self._dispatch_cuda_multirhs(
                nsteps, dX, rho_inv, grid_idcs, phi0, dtype
            )
        elif kc == "mkl_etd2":
            sol, grid_sol = self._dispatch_mkl_multirhs(
                nsteps, dX, rho_inv, grid_idcs, phi0, dtype
            )
        else:
            raise NotImplementedError(
                f"solve_multirhs is not yet wired for kernel_config={kc!r}. "
                f"Supported: 'numpy_etd2', 'accelerate_etd2', 'cuda_etd2', "
                f"'mkl_etd2'."
            )

        info(2, f"time elapsed during multi-RHS integration: {time() - start:5.2f}sec")
        return sol, grid_sol

    def _build_pixel_paths(
        self,
        zenith_grid,
        azimuth_grid=None,
        *,
        X_start=None,
        eps=None,
        dX_max=None,
        dX_min=None,
        fd_span=None,
        path_workers=0,
    ):
        """Build per-pixel ETD2 integration paths for a (zenith × azimuth) grid.

        Returns ``(paths, pixel_index, K)`` where ``paths`` is a list of
        ``(nsteps, dX, rho_inv, grid_idcs)`` tuples (one per pixel,
        flattened with azimuth as the inner axis) and ``pixel_index`` is
        a ``(K, 2)`` int array mapping each column back to its
        ``(i_zen, i_az)`` grid coordinates. Restores the active
        ``(zenith, azimuth)`` to whatever it was before the call.

        Used by both :meth:`solve_fullsky` (single dispatch) and the
        Stage-4 bucketed path (one dispatch per nsteps-bucket).
        """
        zenith_grid = np.asarray(zenith_grid, dtype=np.float64).reshape(-1)
        if azimuth_grid is not None:
            azimuth_grid = np.asarray(azimuth_grid, dtype=np.float64).reshape(-1)
        n_zen = zenith_grid.size
        n_az = azimuth_grid.size if azimuth_grid is not None else 1
        K = n_zen * n_az
        if K < 1:
            raise ValueError("_build_pixel_paths: empty (zenith, azimuth) grid")

        # Auto-detect azimuth symmetry from the density model. Only
        # ``MSIS00LocationCentered`` and subclasses bind the impact point
        # to (zenith, azimuth) — their ``set_theta`` accepts ``azimuth_deg``.
        # Every other atmosphere (CORSIKA, Isothermal, plain MSIS00,
        # AIRS) ignores azimuth, so the path at fixed zenith is identical
        # across all azimuth pixels and only needs to be built once.
        import inspect as _inspect
        az_symmetric = (
            "azimuth_deg" not in _inspect.signature(
                self.density_model.set_theta
            ).parameters
        )

        saved_zen = getattr(self, "theta_deg", None)
        saved_az = getattr(self, "azimuth_deg", None)
        paths = []
        pixel_index = np.empty((K, 2), dtype=np.int32)
        try:
            if az_symmetric and azimuth_grid is not None and n_az > 1:
                # Build one path per unique zenith, then duplicate the
                # tuple ``n_az`` times — same (nsteps, dX, rho_inv, grid_idcs)
                # for every azimuth at that zenith.
                k = 0
                for i_zen, zen in enumerate(zenith_grid):
                    self.set_zenith_azimuth(float(zen))
                    self._calculate_integration_path(
                        None, "X",
                        X_start=X_start, eps=eps,
                        dX_max=dX_max, dX_min=dX_min, fd_span=fd_span,
                    )
                    shared_path = self.integration_path
                    for i_az in range(n_az):
                        paths.append(shared_path)
                        pixel_index[k] = (i_zen, i_az)
                        k += 1
            else:
                # Per-pixel paths. Optional fork-based worker pool —
                # used only for atmospheres where each (zen, az) is
                # distinct (MSIS location-centered). Pickling MCEqRun
                # would be fragile; instead set a module-level global
                # and rely on fork() to share via CoW.
                jobs = []
                k = 0
                for i_zen, zen in enumerate(zenith_grid):
                    for i_az in range(n_az):
                        az = (
                            float(azimuth_grid[i_az])
                            if azimuth_grid is not None
                            else None
                        )
                        jobs.append((k, float(zen), az, i_zen, i_az))
                        k += 1
                paths = [None] * K
                kwargs = dict(
                    X_start=X_start, eps=eps,
                    dX_max=dX_max, dX_min=dX_min, fd_span=fd_span,
                )
                if path_workers and int(path_workers) > 1:
                    import multiprocessing as _mp

                    # MSIS is not safe across forked workers: the
                    # underlying nrlmsise-00 Fortran library has
                    # state that does not properly isolate via fork
                    # CoW. Empirically the paths drift by ~fp32-ε
                    # relative — small but non-reproducible. Refuse
                    # the worker pool for MSIS rather than silently
                    # producing non-bit-exact runs.
                    from MCEq.geometry.density_profiles import (
                        MSIS00Atmosphere,
                    )

                    if isinstance(self.density_model, MSIS00Atmosphere):
                        raise ValueError(
                            "path_workers > 1 is not safe with MSIS-based "
                            "atmospheres (nrlmsise-00 is not fork-safe; "
                            "paths drift by ~1e-7 relative and are not "
                            "reproducible). Use path_workers=0 for MSIS."
                        )

                    global _PATH_WORKER_MCEQ
                    n_workers = int(path_workers)
                    _PATH_WORKER_MCEQ = self  # inherited by forked children
                    try:
                        ctx = _mp.get_context("fork")
                        worker_args = [
                            (j[0], j[1], j[2], kwargs) for j in jobs
                        ]
                        chunksize = max(1, K // (n_workers * 8))
                        with ctx.Pool(n_workers) as pool:
                            for flat_idx, path in pool.imap_unordered(
                                _path_worker_one, worker_args, chunksize=chunksize
                            ):
                                paths[flat_idx] = path
                    finally:
                        _PATH_WORKER_MCEQ = None
                    for j in jobs:
                        pixel_index[j[0]] = (j[3], j[4])
                else:
                    for j in jobs:
                        flat_idx, zen, az, i_zen, i_az = j
                        if az is None:
                            self.set_zenith_azimuth(zen)
                        else:
                            self.set_zenith_azimuth(zen, az)
                        self._calculate_integration_path(
                            None, "X", **kwargs
                        )
                        paths[flat_idx] = self.integration_path
                        pixel_index[flat_idx] = (i_zen, i_az)
        finally:
            if saved_zen is not None:
                self.set_zenith_azimuth(saved_zen, saved_az)
            self.integration_path = None
        return paths, pixel_index, K

    def _dispatch_multipath(self, nsteps_max, dX_2d, rho_inv_2d, phi0_multi, dtype=np.float64):
        """Dispatch a single multipath kernel call. Picks the kernel from
        ``config.kernel_config``; reuses the spacc/cuda handle cache. Returns
        the final ``(dim, K)`` solution.
        """
        import MCEq.solvers

        kc = config.kernel_config.lower()
        if kc == "numpy_etd2":
            sol, _ = MCEq.solvers.solv_numpy_etd2_multipath(
                nsteps_max,
                dX_2d,
                rho_inv_2d,
                self.int_m,
                self.dec_m,
                phi0_multi,
                [],
            )
            return sol
        if kc == "accelerate_etd2":
            import MCEq.spacc as spacc
            from MCEq.solvers import _etd_split_cache

            cached = getattr(self, "_spacc_etd2_cache", None)
            if (
                cached is None
                or cached["int_m"] is not self.int_m
                or cached["dec_m"] is not self.dec_m
            ):
                d_int, d_dec, int_off, dec_off = _etd_split_cache(
                    self.int_m, self.dec_m
                )
                new_cached = {
                    "int_m": self.int_m,
                    "dec_m": self.dec_m,
                    "spacc_int_off": (
                        spacc.SpaccMatrix(int_off) if int_off.nnz else None
                    ),
                    "spacc_dec_off": (
                        spacc.SpaccMatrix(dec_off) if dec_off.nnz else None
                    ),
                    "d_int": d_int,
                    "d_dec": d_dec,
                }
                old_cached = cached
                self._spacc_etd2_cache = new_cached
                if old_cached is not None:
                    for key in ("spacc_int_off", "spacc_dec_off"):
                        old = old_cached.get(key)
                        if old is not None:
                            old.close()
            c = self._spacc_etd2_cache
            sol, _ = MCEq.solvers.solv_spacc_etd2_multipath(
                nsteps_max,
                dX_2d,
                rho_inv_2d,
                c["spacc_int_off"],
                c["spacc_dec_off"],
                c["d_int"],
                c["d_dec"],
                phi0_multi,
                [],
            )
            return sol
        if kc == "cuda_etd2":
            sol, _ = self._dispatch_cuda_multipath(
                nsteps_max, dX_2d, rho_inv_2d, phi0_multi, dtype=dtype
            )
            return sol
        if kc == "mkl_etd2":
            sol, _ = self._dispatch_mkl_multipath(
                nsteps_max, dX_2d, rho_inv_2d, phi0_multi, dtype=dtype
            )
            return sol
        raise NotImplementedError(
            f"solve_fullsky is not yet wired for kernel_config={kc!r}. "
            f"Supported: 'numpy_etd2', 'accelerate_etd2', 'cuda_etd2', "
            f"'mkl_etd2'."
        )

    def _dispatch_carousel(
        self,
        dX_c,
        rho_inv_c,
        phi_initial,
        schedule,
        phi0_per_pixel,
        dtype=np.float64,
    ):
        """Dispatch one carousel solve. Currently wired for numpy_etd2 only;
        cuda_etd2 lift is next. Returns ``(dim, K_total)`` pixel-order final
        states.
        """
        import MCEq.solvers

        kc = config.kernel_config.lower()
        if kc == "numpy_etd2":
            sol = MCEq.solvers.solv_numpy_etd2_carousel(
                self.int_m,
                self.dec_m,
                dX_c,
                rho_inv_c,
                phi_initial,
                schedule,
                phi0_per_pixel,
            )
            return np.asarray(sol, dtype=np.dtype(dtype))
        if kc == "cuda_etd2":
            # Reuse the multi-RHS cupy context cache (keyed on (dtype, K))
            # — the carousel uses ctx.K = K_pipe (pipeline width), not
            # K_total. Different K_pipe values get separate ctx slots.
            from MCEq.solvers import _etd_split_cache

            K_pipe = schedule.K
            dtype = np.dtype(dtype)
            if dtype not in (np.float32, np.float64):
                raise ValueError(
                    f"_dispatch_carousel: cuda dtype must be float32/64, got {dtype}"
                )
            fp_precision = 32 if dtype == np.float32 else 64
            cache_key = (fp_precision, K_pipe)
            cache = getattr(self, "_cuda_etd2_multirhs_cache", None)
            if cache is None:
                cache = {}
                self._cuda_etd2_multirhs_cache = cache
            entry = cache.get(cache_key)
            if (
                entry is None
                or entry["int_m"] is not self.int_m
                or entry["dec_m"] is not self.dec_m
            ):
                d_int, d_dec, int_off, dec_off = _etd_split_cache(
                    self.int_m, self.dec_m
                )
                device_id = int(getattr(config, "cuda_device_id", 0))
                ctx = MCEq.solvers.CudaEtd2MultiRHSContext(
                    int_off, dec_off, d_int, d_dec,
                    K=K_pipe, device_id=device_id,
                    fp_precision=fp_precision,
                )
                cache[cache_key] = {"int_m": self.int_m, "dec_m": self.dec_m, "ctx": ctx}
                entry = cache[cache_key]
            ctx = entry["ctx"]
            phi_init_typed = np.asarray(phi_initial, dtype=dtype)
            phi0_typed = np.asarray(phi0_per_pixel, dtype=dtype)
            dX_typed = np.asarray(dX_c, dtype=dtype)
            ri_typed = np.asarray(rho_inv_c, dtype=dtype)
            sol = MCEq.solvers.solv_cuda_etd2_carousel(
                ctx, dX_typed, ri_typed, phi_init_typed, schedule, phi0_typed
            )
            return sol
        raise NotImplementedError(
            f"_dispatch_carousel: kernel_config={kc!r} not yet wired. "
            f"Supported: 'numpy_etd2', 'cuda_etd2'."
        )

    def _dispatch_mkl_multipath(
        self, nsteps_max, dX_2d, rho_inv_2d, phi0_multi, dtype=np.float64
    ):
        """Dispatch the MKL multipath kernel. fp64 only for now —
        the multipath path uses ``_etd_compute_diag_factors_multipath``
        which is fp64-internal; a fully-fp32 multipath would need an fp32
        analogue of that helper. Cast back if the caller requested fp32.
        """
        import MCEq.solvers
        from MCEq.solvers import _etd_split_cache

        if np.dtype(dtype) == np.float32:
            raise NotImplementedError(
                "_dispatch_mkl_multipath: fp32 multipath not yet wired "
                "(diag-factor pipeline runs fp64). Use kernel_config="
                "'cuda_etd2' for fp32 full-sky."
            )

        cache_attr = "_mkl_etd2_cache_multirhs"
        cached = getattr(self, cache_attr, None)
        if (
            cached is None
            or cached["int_m"] is not self.int_m
            or cached["dec_m"] is not self.dec_m
        ):
            d_int, d_dec, int_off, dec_off = _etd_split_cache(self.int_m, self.dec_m)
            new_cached = {
                "int_m": self.int_m,
                "dec_m": self.dec_m,
                "mkl_int_off": (
                    MCEq.solvers.MklSparseMatrix(int_off) if int_off.nnz else None
                ),
                "mkl_dec_off": (
                    MCEq.solvers.MklSparseMatrix(dec_off) if dec_off.nnz else None
                ),
                "d_int": d_int,
                "d_dec": d_dec,
            }
            old_cached = cached
            setattr(self, cache_attr, new_cached)
            if old_cached is not None:
                for key in ("mkl_int_off", "mkl_dec_off"):
                    old = old_cached.get(key)
                    if old is not None:
                        old.close()
        c = getattr(self, cache_attr)
        sol, _ = MCEq.solvers.solv_mkl_etd2_multipath(
            nsteps_max,
            dX_2d,
            rho_inv_2d,
            c["mkl_int_off"],
            c["mkl_dec_off"],
            c["d_int"],
            c["d_dec"],
            phi0_multi,
            [],
        )
        return sol, None

    def _dispatch_cuda_multipath(
        self, nsteps_max, dX_2d, rho_inv_2d, phi0_multi, dtype=np.float64
    ):
        """Dispatch the cupy multipath kernel. Reuses the multi-RHS context
        cache (keyed on (dtype, K)); multipath uses the same (dim, K) state
        buffers plus the lazily-allocated (dim, K) diag-factor buffers.

        ``dtype`` (np.float32 or np.float64) controls state + SpMM precision.
        On Ampere (RTX 3090) without fp64 tensor cores, fp32 buys ~6× per-step
        at this dim/sparsity. fp32 stability budget against the fp64 reference
        is 1e-4 rel-L2 (verified on the toy test).
        """
        import MCEq.solvers
        from MCEq.solvers import _etd_split_cache

        dim, K = phi0_multi.shape
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                f"_dispatch_cuda_multipath: dtype must be float32 or float64, got {dtype}"
            )
        fp_precision = 32 if dtype == np.float32 else 64
        cache_key = (fp_precision, K)
        cache = getattr(self, "_cuda_etd2_multirhs_cache", None)
        if cache is None:
            cache = {}
            self._cuda_etd2_multirhs_cache = cache
        entry = cache.get(cache_key)
        if (
            entry is None
            or entry["int_m"] is not self.int_m
            or entry["dec_m"] is not self.dec_m
        ):
            d_int, d_dec, int_off, dec_off = _etd_split_cache(
                self.int_m, self.dec_m
            )
            device_id = int(getattr(config, "cuda_device_id", 0))
            ctx = MCEq.solvers.CudaEtd2MultiRHSContext(
                int_off,
                dec_off,
                d_int,
                d_dec,
                K=K,
                device_id=device_id,
                fp_precision=fp_precision,
            )
            cache[cache_key] = {
                "int_m": self.int_m,
                "dec_m": self.dec_m,
                "ctx": ctx,
            }
            entry = cache[cache_key]
        ctx = entry["ctx"]
        phi0_typed = np.asarray(phi0_multi, dtype=dtype)
        sol, _ = MCEq.solvers.solv_cuda_etd2_multipath(
            nsteps_max, dX_2d, rho_inv_2d, ctx, phi0_typed, []
        )
        return sol, None

    @staticmethod
    def _stack_paths(paths, nsteps_max):
        """Stack a list of per-pixel paths into (nsteps_max, K) tensors,
        zero-padding columns shorter than ``nsteps_max``."""
        K = len(paths)
        dX_2d = np.zeros((nsteps_max, K), dtype=np.float64)
        rho_inv_2d = np.zeros((nsteps_max, K), dtype=np.float64)
        for j, (ns, dX_k, ri_k, _) in enumerate(paths):
            dX_2d[:ns, j] = dX_k
            rho_inv_2d[:ns, j] = ri_k
        return dX_2d, rho_inv_2d

    def solve_fullsky(
        self,
        zenith_grid,
        azimuth_grid=None,
        phi0=None,
        *,
        bucket_count=None,
        carousel_K=None,
        dtype=np.float64,
        X_start=None,
        eps=None,
        dX_max=None,
        dX_min=None,
        fd_span=None,
        return_pixel_index=False,
        path_workers=0,
    ):
        """Propagate one initial spectrum through every (zenith, azimuth)
        pixel of a sky grid in a (Stage-4 bucketed) multi-RHS solve.

        For each pixel, builds an independent ETD2 integration path
        (zenith-dependent ρ(X), own ``dX`` / ``rho_inv`` / ``nsteps``)
        via the standard :func:`MCEq.solvers.etd2_nonuniform_path`
        scheduler. With ``bucket_count = 1`` (Stage-3 behaviour) the
        kernel runs a single multipath call with ``(nsteps_max, K)``
        path tensors and zero-pads shorter columns past their own
        ``nsteps[k]`` (the math freezes those columns automatically:
        ``h = 0 ⇒ eD = 1, φ₁ = 1, φ₂ = 1/2 ⇒ state ← state``).

        With ``bucket_count > 1`` (Stage 4): pixels are sorted by
        ``nsteps`` and partitioned into ``bucket_count`` equal-size
        buckets; one multipath call per bucket runs with a
        bucket-local ``nsteps_max_b`` ≈ ``max(nsteps_in_bucket)``,
        which is dramatically smaller than the global ``nsteps_max``
        when the per-pixel ``nsteps`` distribution is long-tailed
        (e.g. uniform-cos sweeps to ~85° zenith, where the
        ``dX_min`` floor at thin top-of-atmosphere pushes the steepest
        paths to 10× the overhead path's nsteps). Wasted work drops
        from ``nsteps_max · K`` to ``sum_b (nsteps_max_b · K_b)``,
        which is 3–5× smaller at realistic full-sky grids.

        Args:
          zenith_grid (np.ndarray): 1-D zenith angles in degrees.
          azimuth_grid (np.ndarray | None): 1-D azimuth angles in
            degrees. If ``None``, the calculation reduces to the
            ``len(zenith_grid)``-pixel zenith-only sky.
          phi0 (np.ndarray | None): initial spectrum. Two shapes accepted:
            * ``(dim_states,)`` — single primary, replicated across all K
              pixel columns. Propagates one spectrum through K different
              atmospheres.
            * ``(dim_states, K)`` — per-pixel initial spectrum. Column
              order matches the ``(i_zen, i_az)``-flattened pixel order
              (azimuth is the inner axis; same convention as
              :meth:`_build_pixel_paths`). Use this to apply per-pixel
              modifications (e.g. geomagnetic rigidity cutoff masks on
              the primary).
            If ``None``, uses the currently set ``self._phi0`` and
            replicates as in the 1-D case.
          bucket_count (int | None): number of nsteps-buckets to
            partition pixels into. ``None`` ⇒ heuristic default
            (1 if K ≤ 4 else min(K, 8)). ``1`` ⇒ Stage-3 single
            dispatch. ``K`` ⇒ degenerates to serial (each bucket is
            one column).
          X_start, eps, dX_max, dX_min, fd_span: ETD2 path-builder
            knobs forwarded for every pixel.
          return_pixel_index (bool): if True, also return the K × 2
            mapping ``(i_zen, i_az)`` so callers can reshape the
            output back onto a ``(n_zen, n_az)`` grid.

        Returns:
          (np.ndarray[dim_states, K], np.ndarray[K] | None): final
          state per pixel and the per-pixel ``nsteps`` array
          (preserved in original ``(i_zen, i_az)`` order). With
          ``return_pixel_index=True`` also returns the ``(K, 2)``
          pixel grid mapping.

        Notes:
          * Currently wired for ``kernel_config = "numpy_etd2"`` and
            ``"accelerate_etd2"``; MKL / CUDA paths raise
            ``NotImplementedError``.
          * Bucketing is orthogonal to the K-tile (``_SPACC_SPMM_TILE``)
            on the Accelerate backend — that tile applies inside each
            bucket's SpMM call.
        """
        info(
            2,
            f"solve_fullsky: kernel={config.kernel_config}",
        )
        start = time()

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
                    f"solve_fullsky: phi0 must be 1-D (dim_states,) or 2-D "
                    f"(dim_states, K); got shape {phi0_arr.shape}"
                )

        paths, pixel_index, K = self._build_pixel_paths(
            zenith_grid,
            azimuth_grid,
            X_start=X_start,
            eps=eps,
            dX_max=dX_max,
            dX_min=dX_min,
            fd_span=fd_span,
            path_workers=path_workers,
        )
        nsteps_per_col = np.array([p[0] for p in paths], dtype=np.int32)
        ns_min = int(nsteps_per_col.min())
        ns_max = int(nsteps_per_col.max())
        ns_mean = float(nsteps_per_col.mean())
        info(
            2,
            f"solve_fullsky: K={K}, nsteps range [{ns_min}, {ns_max}], "
            f"mean={ns_mean:.1f}",
        )

        if phi0_is_2d and phi0_arr.shape[1] != K:
            raise ValueError(
                f"solve_fullsky: phi0 has shape {phi0_arr.shape}, expected "
                f"second axis = K = {K} pixels from the (zenith, azimuth) grid"
            )

        if carousel_K is not None:
            # Stage 5 — LPT static carousel. Supersedes bucket_count.
            from MCEq.solvers import schedule_lpt, compile_carousel_schedule

            K_pipe = max(1, min(int(carousel_K), K))
            slots, T = schedule_lpt(nsteps_per_col, K_pipe)
            if phi0_is_2d:
                phi0_per_pixel = np.ascontiguousarray(phi0_arr)
            else:
                phi0_per_pixel = np.broadcast_to(
                    phi0_arr[:, None], (self.dim_states, K)
                ).copy()
            dX_c, ri_c, phi_init, sched = compile_carousel_schedule(
                paths, slots, T, self.dim_states, phi0_per_pixel
            )
            sum_ns = int(nsteps_per_col.sum())
            waste = 1.0 - sum_ns / float(T * K_pipe) if (T * K_pipe) else 0.0
            info(
                2,
                f"solve_fullsky: carousel K_pipe={K_pipe} T={T} "
                f"sum_nsteps={sum_ns} waste={waste*100:.2f}%",
            )
            sol = self._dispatch_carousel(
                dX_c, ri_c, phi_init, sched, phi0_per_pixel, dtype=dtype
            )
            info(2, f"solve_fullsky: total wall {time() - start:.2f}s")
            if return_pixel_index:
                return sol, nsteps_per_col, pixel_index
            return sol, nsteps_per_col

        if bucket_count is None:
            bucket_count = 1 if K <= 4 else min(K, 8)
        bucket_count = max(1, min(int(bucket_count), K))

        sol = np.empty((self.dim_states, K), dtype=np.dtype(dtype))

        if bucket_count == 1:
            dX_2d, rho_inv_2d = self._stack_paths(paths, ns_max)
            if phi0_is_2d:
                phi0_multi = np.ascontiguousarray(phi0_arr)
            else:
                phi0_multi = np.broadcast_to(
                    phi0_arr[:, None], (self.dim_states, K)
                ).copy()
            sol = self._dispatch_multipath(
                ns_max, dX_2d, rho_inv_2d, phi0_multi, dtype=dtype
            )
        else:
            # Sort columns by nsteps; partition into equal-K buckets.
            # Within-bucket nsteps spread ≪ global spread, so each
            # bucket's nsteps_max_b is close to the bucket mean and
            # very little SpMM work is wasted on frozen columns.
            order = np.argsort(nsteps_per_col)
            edges = np.linspace(0, K, bucket_count + 1, dtype=int)
            for b in range(bucket_count):
                cols = order[edges[b] : edges[b + 1]]
                if cols.size == 0:
                    continue
                K_b = cols.size
                ns_max_b = int(nsteps_per_col[cols].max())
                bucket_paths = [paths[i] for i in cols]
                dX_b, rho_inv_b = self._stack_paths(bucket_paths, ns_max_b)
                if phi0_is_2d:
                    phi0_b = np.ascontiguousarray(phi0_arr[:, cols])
                else:
                    phi0_b = np.broadcast_to(
                        phi0_arr[:, None], (self.dim_states, K_b)
                    ).copy()
                sol_b = self._dispatch_multipath(
                    ns_max_b, dX_b, rho_inv_b, phi0_b, dtype=dtype
                )
                # Scatter the bucket's solution back into the original
                # column positions so the caller sees pixels in their
                # original ``(i_zen, i_az)``-flattened order.
                sol[:, cols] = sol_b
            info(
                2,
                f"solve_fullsky: bucket_count={bucket_count}, "
                f"per-bucket nsteps_max ranged "
                f"[{int(nsteps_per_col[order[edges[1] - 1]])}, {ns_max}]",
            )

        info(2, f"solve_fullsky: total wall {time() - start:.2f}s")
        if return_pixel_index:
            return sol, nsteps_per_col, pixel_index
        return sol, nsteps_per_col

    def _build_kernel_dispatch(self, nsteps, dX, rho_inv, phi0, grid_idcs):
        """Resolve ``config.kernel_config`` to ``(kernel, args)``.

        Recognised kernels: ``numpy_etd2`` (always available),
        ``accelerate_etd2`` (macOS), ``mkl_etd2`` (Linux/Windows when
        ``libmkl_rt`` is present), and ``cuda_etd2`` (cuSPARSE via cupy).
        The legacy short names (``numpy``/``mkl``/``cuda``/``accelerate``)
        are no longer accepted — the corresponding forward-Euler kernels
        were retired in v2 (see ``changes/+remove-euler-resonance.api.md``).
        """
        import MCEq.solvers

        kc = config.kernel_config.lower()

        if kc == "numpy_etd2":
            # If an EM ρ-stack has been built (via
            # enable_em_density_interpolation), route to the ρ-aware kernel
            # so per-step log-linear blending of the air block kicks in.
            int_m_stack = getattr(self, "_int_m_stack", None)
            em_rho_grid = getattr(self, "_em_rho_grid", None)
            if int_m_stack is not None and em_rho_grid is not None:
                return MCEq.solvers.solv_numpy_etd2_rho_stack, (
                    nsteps,
                    dX,
                    rho_inv,
                    int_m_stack,
                    em_rho_grid,
                    self.dec_m,
                    phi0,
                    grid_idcs,
                )
            return MCEq.solvers.solv_numpy_etd2, (
                nsteps,
                dX,
                rho_inv,
                self.int_m,
                self.dec_m,
                phi0,
                grid_idcs,
            )

        if kc == "accelerate_etd2":
            import MCEq.spacc as spacc
            from MCEq.solvers import _etd_split_cache

            # Cache the diagonal/off-diagonal split AND its SpaccMatrix
            # wrappers, keyed against ``int_m`` / ``dec_m`` identity. When
            # either matrix is rebuilt (e.g. ``set_density_model`` →
            # ``construct_matrices``) we deterministically free the old
            # SpaccMatrix slots so the global Accelerate matrix store
            # (fixed ``SIZE_MSTORE``) does not fill up.
            cached = getattr(self, "_spacc_etd2_cache", None)
            if (
                cached is None
                or cached["int_m"] is not self.int_m
                or cached["dec_m"] is not self.dec_m
            ):
                # Build the new cache fully *before* freeing the old one.
                # If construction fails partway (e.g. memory pressure), the
                # previous cache stays valid and the next solve() call will
                # retry the rebuild without leaking either side.
                d_int, d_dec, int_off, dec_off = _etd_split_cache(
                    self.int_m, self.dec_m
                )
                spacc_int_off = spacc.SpaccMatrix(int_off) if int_off.nnz else None
                spacc_dec_off = spacc.SpaccMatrix(dec_off) if dec_off.nnz else None
                new_cached = {
                    "int_m": self.int_m,
                    "dec_m": self.dec_m,
                    "spacc_int_off": spacc_int_off,
                    "spacc_dec_off": spacc_dec_off,
                    "d_int": d_int,
                    "d_dec": d_dec,
                }
                old_cached = cached
                self._spacc_etd2_cache = new_cached
                if old_cached is not None:
                    for key in ("spacc_int_off", "spacc_dec_off"):
                        old = old_cached.get(key)
                        if old is not None:
                            old.close()  # idempotent
            c = self._spacc_etd2_cache
            return MCEq.solvers.solv_spacc_etd2, (
                nsteps,
                dX,
                rho_inv,
                c["spacc_int_off"],
                c["spacc_dec_off"],
                c["d_int"],
                c["d_dec"],
                phi0,
                grid_idcs,
            )

        if kc in ("mkl", "mkl_etd2"):
            from MCEq.solvers import MklSparseMatrix, _etd_split_cache

            # Cache the diagonal/off-diagonal split AND its MKL handle
            # wrappers, keyed against ``int_m`` / ``dec_m`` identity. When
            # either matrix is rebuilt we deterministically free the old
            # MKL handles so they don't accumulate (each handle owns
            # MKL-internal optimised-layout memory beyond the Python ref).
            cached = getattr(self, "_mkl_etd2_cache", None)
            if (
                cached is None
                or cached["int_m"] is not self.int_m
                or cached["dec_m"] is not self.dec_m
            ):
                # Build new before freeing old — see the spacc branch above
                # for the rationale.
                d_int, d_dec, int_off, dec_off = _etd_split_cache(
                    self.int_m, self.dec_m
                )
                # MKL requires CSR; the split inherits the input format.
                if not sp.isspmatrix_csr(int_off):
                    int_off = int_off.tocsr()
                if not sp.isspmatrix_csr(dec_off):
                    dec_off = dec_off.tocsr()
                bs = config.mkl_bsr_blocksize
                mkl_int_off = (
                    MklSparseMatrix(int_off, blocksize=bs) if int_off.nnz else None
                )
                mkl_dec_off = (
                    MklSparseMatrix(dec_off, blocksize=bs) if dec_off.nnz else None
                )
                new_cached = {
                    "int_m": self.int_m,
                    "dec_m": self.dec_m,
                    "mkl_int_off": mkl_int_off,
                    "mkl_dec_off": mkl_dec_off,
                    "d_int": d_int,
                    "d_dec": d_dec,
                }
                old_cached = cached
                self._mkl_etd2_cache = new_cached
                if old_cached is not None:
                    for key in ("mkl_int_off", "mkl_dec_off"):
                        old = old_cached.get(key)
                        if old is not None:
                            old.close()  # idempotent
            c = self._mkl_etd2_cache
            return MCEq.solvers.solv_mkl_etd2, (
                nsteps,
                dX,
                rho_inv,
                c["mkl_int_off"],
                c["mkl_dec_off"],
                c["d_int"],
                c["d_dec"],
                phi0,
                grid_idcs,
            )

        if kc in ("cuda", "cuda_etd2"):
            from MCEq.solvers import CudaEtd2Context, _etd_split_cache

            cached = getattr(self, "_cuda_etd2_cache", None)
            if (
                cached is None
                or cached["int_m"] is not self.int_m
                or cached["dec_m"] is not self.dec_m
                or cached["device_id"] != self._cuda_device
                or cached["fp_precision"] != config.cuda_fp_precision
            ):
                # The previous context's GPU buffers / cusparse handles drop
                # automatically when the dict is replaced (cupy frees them
                # on garbage collection).
                d_int, d_dec, int_off, dec_off = _etd_split_cache(
                    self.int_m, self.dec_m
                )
                if not sp.isspmatrix_csr(int_off):
                    int_off = int_off.tocsr()
                if not sp.isspmatrix_csr(dec_off):
                    dec_off = dec_off.tocsr()
                ctx = CudaEtd2Context(
                    int_off,
                    dec_off,
                    d_int,
                    d_dec,
                    device_id=self._cuda_device,
                    fp_precision=config.cuda_fp_precision,
                )
                self._cuda_etd2_cache = {
                    "int_m": self.int_m,
                    "dec_m": self.dec_m,
                    "device_id": self._cuda_device,
                    "fp_precision": config.cuda_fp_precision,
                    "ctx": ctx,
                }
            ctx = self._cuda_etd2_cache["ctx"]
            return MCEq.solvers.solv_cuda_etd2, (
                nsteps,
                dX,
                rho_inv,
                ctx,
                phi0,
                grid_idcs,
            )

        raise Exception(
            f"Unsupported integrator setting '{config.kernel_config}'. "
            "Choose one of: numpy_etd2, accelerate_etd2, mkl_etd2, cuda_etd2."
        )

    def close(self):
        """Release all backend solver resources held by this MCEqRun.

        Frees Accelerate slots, MKL sparse handles, and the cuSPARSE
        context (cupy GPU buffers). Idempotent — safe to call repeatedly
        and safe to call before falling out of scope. Calling the
        instance again after ``close()`` will lazily rebuild the caches
        on the next ``solve()``, so this is also a "drop and reset"
        knob during long-running scripts.
        """
        # spacc and MKL wrappers expose explicit close(); cupy GPU memory
        # is reclaimed by cupy's allocator when the cache dict drops.
        for cache_attr, wrapper_keys in (
            ("_spacc_etd2_cache", ("spacc_int_off", "spacc_dec_off")),
            ("_spacc_etd2_cache_f32", ("spacc_int_off", "spacc_dec_off")),
            ("_mkl_etd2_cache", ("mkl_int_off", "mkl_dec_off")),
        ):
            cached = getattr(self, cache_attr, None)
            if cached is None:
                continue
            for k in wrapper_keys:
                w = cached.get(k)
                if w is not None:
                    try:
                        w.close()
                    except Exception:
                        pass
            try:
                delattr(self, cache_attr)
            except AttributeError:
                pass
        # CUDA context: drop the dict; cupy's GC reclaims GPU memory.
        try:
            delattr(self, "_cuda_etd2_cache")
        except AttributeError:
            pass

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

    def __init__(self, particle_manager):
        self._pman = particle_manager
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

        from itertools import product

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
                        self.C_blocks[idx] += self.cont_loss_operator(parent.pdg_id)

        self.int_m = self._csr_from_blocks(self.C_blocks)
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
        else:
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
        """Returns a new square zero valued matrix with dimensions of grid."""
        return np.zeros((self._pman.dim, self._pman.dim), dtype=config.floatlen)

    def _csr_from_blocks(self, blocks):
        """Construct a csr matrix from a dictionary of submatrices (blocks)

        Note::

            It's super pain the a** to construct a properly indexed sparse matrix
            directly from the blocks, since bmat totally messes up the order.
        """
        from scipy.sparse import csr_matrix

        new_mat = np.zeros((self.dim_states, self.dim_states), dtype=config.floatlen)

        for (c, p), d in six.iteritems(blocks):
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

        - ``"expfit"`` (default): exponentially-fitted 7-point stencil anchored
          at :data:`MCEq.config.loss_stencil_alpha0`. Near-exact for power-law
          spectra E^{-alpha} with alpha ~ alpha0 on a coarse log grid.
        - ``"centered"``: symmetric 6th-order centered FD.
        - ``"biased"``: legacy 7-point biased "6th-order" stencil.

        All three options share the same one-sided polynomial-fit stencils
        on the boundary rows (0, 1, 2 and last-2, last-1, last); see
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
        method = getattr(config, "loss_stencil_method", "expfit")
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
                "Expected 'expfit', 'centered', or 'biased'."
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

        self.op_matrix = op_matrix
