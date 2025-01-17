import six
from time import time
import numpy as np
from MCEq import asarray, zeros, csr_matrix, eye, linalg, diag, ones
from MCEq.misc import normalize_hadronic_model_name, info
from MCEq.particlemanager import ParticleManager
import MCEq.data
import mceq_config as config


class MCEqRun(object):
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
      interaction_model (string): interaction model name, e.g. SIBYLL-2.3D
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
        self.medium = kwargs.pop("medium", config.interaction_medium)
        self._mceq_db = MCEq.data.HDF5Backend(medium=self.medium)

        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_model = kwargs.pop("density_model", config.density_model)
        self.theta_deg = theta_deg

        #: Interface to interaction tables of the HDF5 database
        self._interactions = MCEq.data.Interactions(mceq_hdf_db=self._mceq_db)

        #: handler for cross-section data of type :class:`MCEq.data.HadAirCrossSections`
        self._int_cs = MCEq.data.InteractionCrossSections(mceq_hdf_db=self._mceq_db)

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
          can be retrieved by the prefix ``total_``, i.e. ``total_numu``
        - the conventional flux of muons, muon neutrinos etc. from all sources
          can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
        - correspondigly, the flux of leptons which originated from the decay
          of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``
        - conventional leptons originating neither from pion nor from kaon
          decay are collected in a category without any prefix, e.g. ``numu`` or
          ``mu+``

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

        Returns:
          (numpy.array): flux of particles on energy grid :attr:`e_grid`
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
                        "or, unavailable particle prefix {0}.".format(prefix + ls),
                    )
                    continue
                result += sol[ref[prefix + ls].lidx : ref[prefix + ls].uidx]
                nsuccess += 1
            if nsuccess == 0 and config.excpt_on_missing_particle:
                raise Exception(
                    "Requested particle {0} not found.".format(particle_name)
                )
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
                    raise Exception(
                        "Requested particle {0} not found.".format(particle_name)
                    )
                else:
                    info(1, "Requested particle {0} not found.".format(particle_name))

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

    def _resize_vectors_and_restore(self):
        """Update solution and grid vectors if the number of particle species
        or the interaction models change. The previous state, such as the
        initial spectrum, are restored."""

        # Update dimensions if particle dimensions changed
        self._phi0 = np.zeros(self.dim_states)
        self._solution = np.zeros(self.dim_states)

        # Restore insital condition if present
        if len(self._restore_initial_condition) > 0:
            for con in self._restore_initial_condition:
                con[0](*con[1:])

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model.

        This functions is quick and does not require re-generation of
        matrices.

        Args:
          interaction_model (:class:`CRFluxModel.PrimaryFlux`): reference
          to primary model **class**
          tag (tuple): positional argument list for model class
        """

        assert not isinstance(
            model_class_or_object, tuple
        ), "Primary model can not be supplied as tuples"

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

        info(1, "Primary model set to {0}".format(self.pmodel.name))

        # Save primary flux model for restauration after interaction model changes
        self._restore_initial_condition = [(self.set_primary_model, self.pmodel)]
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
            ] = (1e-4 * p_top)
        else:
            info(
                1,
                "Protons not in equation system, can not set primary flux.",
            )

        if (2112, 0) in self.pman and not self.pman[(2112, 0)].is_resonance:
            self._phi0[
                min_idx + self.pman[(2112, 0)].lidx : self.pman[(2112, 0)].uidx
            ] = (1e-4 * n_top)
        elif (2212, 0) in self.pman:
            info(
                2,
                "Neutrons not part of equation system,",
                "substituting initial flux with protons.",
            )
            self._phi0[
                min_idx + self.pman[(2212, 0)].lidx : self.pman[(2212, 0)].uidx
            ] += (1e-4 * n_top)

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
        from MCEq.misc import getAZN_corsika, getAZN

        if corsika_id and pdg_id:
            raise Exception("Provide either corsika or PDG ID")

        info(
            2,
            "CORSIKA ID {0}, PDG ID {1}, energy {2:5.3g} GeV".format(
                corsika_id, pdg_id, E
            ),
        )

        if append is False:
            self._restore_initial_condition = [
                (self.set_single_primary_particle, E, corsika_id, pdg_id)
            ]
            self._phi0 *= 0.0
        else:
            self._restore_initial_condition.append(
                (self.set_single_primary_particle, E, corsika_id, pdg_id)
            )
        egrid = self._energy_grid.c
        ebins = self._energy_grid.b
        ewidths = self._energy_grid.w

        if corsika_id:
            n_nucleons, n_protons, n_neutrons = getAZN_corsika(corsika_id)
        elif pdg_id:
            n_nucleons, n_protons, n_neutrons = getAZN(pdg_id)

        En = E / float(n_nucleons) if n_nucleons > 0 else E

        if En < np.min(self._energy_grid.c):
            raise Exception("energy per nucleon too low for primary " + str(corsika_id))

        info(
            3,
            (
                "superposition: n_protons={0}, n_neutrons={1}, "
                + "energy per nucleon={2:5.3g} GeV"
            ).format(n_protons, n_neutrons, En),
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

    def set_initial_spectrum(self, spectrum, pdg_id=None, append=False):
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

        info(2, "PDG ID {0}".format(pdg_id))

        if not append:
            self._restore_initial_condition = [
                (self.set_initial_spectrum, pdg_id, append)
            ]
            self._phi0 *= 0
        else:
            self._restore_initial_condition.append(
                (self.set_initial_spectrum, pdg_id, append)
            )
        if len(spectrum) != self.dim:
            raise Exception("Lengths of spectrum and energy grid do not match.")

        self._phi0[self.pman[pdg_id].lidx : self.pman[pdg_id].uidx] += spectrum

    def set_density_model(self, density_model_or_config):
        """Sets model of the atmosphere.

        To choose, for example, a CORSIKA parametrization for the Southpole in January,
        do the following::

            mceq_instance.set_density_model(('CORSIKA', ('PL_SouthPole', 'January')))

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
                "CORSIKA",
                "AIRS",
                "ERA5Atmosphere",
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
            elif base_model == "CORSIKA":
                self.density_model = dprof.CorsikaAtmosphere(*model_config)
            elif base_model == "AIRS":
                self.density_model = dprof.AIRSAtmosphere(*model_config)
            elif base_model == "ERA5Atmosphere":
                self.density_model = dprof.ERA5Atmosphere(*model_config)
            elif base_model == "Isothermal":
                self.density_model = dprof.IsothermalAtmosphere(*model_config)
            elif base_model == "GeneralizedTarget":
                self.density_model = dprof.GeneralizedTarget()
            else:
                raise ValueError("Unknown atmospheric base model.")
        else:
            self.density_model = density_model_or_config

        if self.theta_deg is not None and isinstance(
            self.density_model, dprof.EarthsAtmosphere
        ):
            if self.theta_deg is None:
                info(1, "Using default zenith angle theta=0.")
                self.set_theta_deg(0)
            else:
                self.set_theta_deg(self.theta_deg)
        elif isinstance(self.density_model, dprof.GeneralizedTarget):
            self.integration_path = None
        else:
            raise ValueError(f"Density model {self.density_model} not supported.")

        # TODO: Make the pman aware of that density might have changed and
        # indices as well
        # self.pmod._gen_list_of_particles()

    def set_theta_deg(self, theta_deg):
        """Sets zenith angle :math:`\\theta` as seen from a detector.

        Currently only 'down-going' angles (0-90 degrees) are supported.

        Args:
          theta_deg (float): zenith angle in the range 0-90 degrees
        """
        import MCEq.geometry.density_profiles as dprof

        info(2, "Zenith angle {0:6.2f}".format(theta_deg))

        if isinstance(self.density_model, dprof.GeneralizedTarget):
            raise Exception("GeneralizedTarget does not support angles.")

        if self.density_model.theta_deg == theta_deg:
            info(2, "Theta selection correponds to cached value, skipping calc.")
            return

        self.density_model.set_theta(theta_deg)
        self.integration_path = None

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

            self.pman[prim].hadr_yields[self.pman[sec]] = asarray(mati)
            info(5, "Injecting isopart", iso_part, isospin_symmetries[iso_part][sec])
            self.pman[iso_part].hadr_yields[
                self.pman[isospin_symmetries[iso_part][sec]]
            ] = asarray(mati)
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
        info(
            1,
            "{0}/{1}, {2}, {3}".format(
                prim_pdg, sec_pdg, x_func.__name__, str(x_func_args)
            ),
        )

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

        self._interactions.mod_pprod = defaultdict(lambda: {})
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
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=skip_decay_matrix
        )

    def solve(self, int_grid=None, grid_var="X", **kwargs):
        """Launches the solver.

        The setting `integrator` in the config file decides which solver
        to launch.

        Args:
          int_grid (list): list of depths at which results are recorded
          grid_var (str): Can be depth `X` or something else (currently
            only `X` supported)
          kwargs (dict): Arguments are passed directly to the solver methods.

        """
        info(2, "Launching {0} solver".format(config.integrator))

        if not kwargs.pop("skip_integration_path", False):
            if int_grid is not None and np.any(np.diff(int_grid) < 0):
                raise Exception(
                    "The X values in int_grid are required to be strickly",
                    "increasing.",
                )

            # Calculate integration path if not yet happened
            self._calculate_integration_path(int_grid, grid_var)
        else:
            info(2, "Warning: integration path calculation skipped.")

        phi0 = np.copy(self._phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, "for {0} integration steps.".format(nsteps))

        import MCEq.solvers

        start = time()

        if config.kernel_config.lower() == "numpy":
            kernel = MCEq.solvers.solv_numpy
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs)

        elif config.kernel_config.lower() == "accelerate":
            kernel = MCEq.solvers.solv_spacc_sparse
            import MCEq.spacc as spacc

            try:
                if not np.array_equal(self._spacc_dec_m.data, self.dec_m.data):
                    self._spacc_dec_m = spacc.SpaccMatrix(self.dec_m)
                if not np.array_equal(self._spacc_int_m.data, self.int_m.data):
                    self._spacc_int_m = spacc.SpaccMatrix(self.int_m)
            except AttributeError:
                info(10, "Matrices not yet in Accelerate format")
                self._spacc_int_m = spacc.SpaccMatrix(self.int_m)
                self._spacc_dec_m = spacc.SpaccMatrix(self.dec_m)

            args = (
                nsteps,
                dX,
                rho_inv,
                self._spacc_int_m,
                self._spacc_dec_m,
                phi0,
                grid_idcs,
            )

        elif config.kernel_config.lower() == "cuda":
            kernel = MCEq.solvers.solv_CUDA_sparse
            try:
                self.cuda_context.set_matrices(self.int_m, self.dec_m)
            except AttributeError:
                from MCEq.solvers import CUDASparseContext

                self.cuda_context = CUDASparseContext(
                    self.int_m, self.dec_m, device_id=self._cuda_device
                )
            args = (nsteps, dX, rho_inv, self.cuda_context, phi0, grid_idcs)

        elif config.kernel_config.lower() == "mkl":
            kernel = MCEq.solvers.solv_MKL_sparse
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs)

        else:
            raise Exception(
                "Unsupported integrator setting '{0}'.".format(config.kernel_config)
            )

        self._solution, self.grid_sol = kernel(*args)

        info(2, "time elapsed during integration: {0:5.2f}sec".format(time() - start))

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

        info(2, "Launching {0} solver".format(config.integrator))
        info(2, "for {0} integration steps.".format(nsteps))

        import MCEq.solvers

        start = time()

        phi0 = np.copy(self._phi0)

        if config.kernel_config.lower() == "numpy":
            kernel = MCEq.solvers.solv_numpy
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs)

        elif config.kernel_config.lower() == "accelerate":
            kernel = MCEq.solvers.solv_spacc_sparse
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs)

        elif config.kernel_config.lower() == "cuda":
            kernel = MCEq.solvers.solv_CUDA_sparse
            try:
                self.cuda_context.set_matrices(self.int_m, self.dec_m)
            except AttributeError:
                from MCEq.solvers import CUDASparseContext

                self.cuda_context = CUDASparseContext(
                    self.int_m, self.dec_m, device_id=self._cuda_device
                )
            args = (nsteps, dX, rho_inv, self.cuda_context, phi0, grid_idcs)

        elif config.kernel_config.lower() == "mkl":
            kernel = MCEq.solvers.solv_MKL_sparse
            args = (nsteps, dX, rho_inv, self.int_m, self.dec_m, phi0, grid_idcs)

        else:
            raise Exception(
                "Unsupported integrator setting '{0}'.".format(config.kernel_config)
            )

        self._solution, self.grid_sol = kernel(*args)

        info(2, "time elapsed during integration: {0:5.2f}sec".format(time() - start))

    def _calculate_integration_path(self, int_grid, grid_var, force=False):
        if (
            self.integration_path
            and np.alltrue(int_grid == self.int_grid)
            and np.alltrue(self.grid_var == grid_var)
            and not force
        ):
            info(5, "skipping calculation.")
            return

        self.int_grid, self.grid_var = int_grid, grid_var
        if grid_var != "X":
            raise NotImplementedError(
                "Grid variables other than the depth X not supported."
            )

        max_X = self.density_model.max_X
        ri = self.density_model.r_X2rho
        max_lint = self.matrix_builder.max_lint
        max_ldec = self.matrix_builder.max_ldec
        dXmax = config.dXmax

        info(2, "X_surface = {0:7.2f}g/cm2".format(max_X))

        dX_vec = []
        rho_inv_vec = []

        X = config.X_start
        if int_grid is not None and min(int_grid) < X:
            raise ValueError(
                "Steps in int_grid must be larger than mceq_config.X_start."
            )
        step = 0
        grid_step = 0
        grid_idcs = []
        assert config.leading_process in [
            "auto",
            "decays",
            "interactions",
        ], "Set leading process to auto, decays or interactions"
        if config.leading_process == "decays":
            info(3, "using decays as leading eigenvalues")

            def delta_X(X, inv_rho):
                return min(config.stability_margin / (max_ldec * inv_rho), dXmax)

        elif config.leading_process == "interactions":
            info(2, "using interactions as leading eigenvalues")

            def delta_X(X, inv_rho):
                return min(config.stability_margin / max_lint, dXmax)

        else:
            # This is the case for auto setting.
            # If no particles in eqn system force decays
            # as leading eigenvalues

            if np.allclose(max_lint, 0.0):

                def delta_X(X, inv_rho):
                    return min(config.stability_margin / (max_ldec * inv_rho), dXmax)

            else:

                def delta_X(X, inv_rho):
                    dX = min(
                        config.stability_margin / (max_ldec * inv_rho),
                        config.stability_margin / max_lint,
                        dXmax,
                    )
                    # if dX / self.density_model.max_X < 1e-7:
                    #     raise Exception(
                    #         "Stiffness warning: dX <= 1e-7. Check configuration or"
                    #         + "manually call MCEqRun._calculate_integration_path("
                    #         + 'int_grid, "X", force=True).'
                    #     )
                    return dX

        enable_int_grid = np.any(int_grid)
        len_int_grid = len(int_grid) if enable_int_grid else 0
        while X < max_X:
            inv_rho = ri(X)
            dX = delta_X(X, inv_rho)
            if (
                enable_int_grid
                and (grid_step < len_int_grid)
                and (X + dX >= int_grid[grid_step])
            ):
                dX = int_grid[grid_step] - X
                grid_idcs.append(step)
                grid_step += 1
            dX_vec.append(dX)
            rho_inv_vec.append(inv_rho)
            X += dX
            step += 1

        # Integrate
        dX_vec = np.array(dX_vec, dtype=config.floatlen)
        rho_inv_vec = np.array(rho_inv_vec, dtype=config.floatlen)

        self.integration_path = len(dX_vec), dX_vec, rho_inv_vec, grid_idcs

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
        info(
            10,
            "Energy cutoff for particle number calculation {0:4.3e} GeV".format(
                self.e_bins[ie_min]
            ),
        )
        info(
            15,
            "First bin is between {0:3.2e} and {1:3.2e} with midpoint {2:3.2e}".format(
                self.e_bins[ie_min], self.e_bins[ie_min + 1], self.e_grid[ie_min]
            ),
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
            raise Exception("{0} is not a projectile particle.".format(proj.name))
        info(
            10,
            "Computing e-dependent Zfactor for {0} -> {1}".format(proj.name, sec.name),
        )
        if not proj.is_secondary(sec):
            raise Exception(
                "{0} is not a secondary particle of {1}.".format(sec.name, proj.name)
            )

        if proj == 2112:
            nuc_flux = self.pmodel.p_and_n_flux(self.e_grid)[2]
        else:
            nuc_flux = self.pmodel.p_and_n_flux(self.e_grid)[1]
        zfac = np.zeros(self.dim)

        smat = proj.hadr_yields[sec]
        proj_cs = proj.inel_cross_section()
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
                assert (
                    use_cs_scaling is False
                ), f"cs_scaling has when definition = {definition}"
                cs_fac = 1.0
                zfac[p_eidx] = np.sum(
                    smat[min_idx : p_eidx + 1, p_eidx] * nuc_fac * cs_fac
                )
            return zfac
        else:
            # Like in Thunman et al. 1996
            for p_eidx, e in enumerate(self.e_grid):
                if e < min_energy:
                    min_idx = p_eidx
                    continue
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
            raise Exception("{0} does not decay.".format(proj.name))
        info(
            10,
            "Computing e-dependent decay Zfactor for {0} -> {1}".format(
                proj.name, sec.name
            ),
        )
        if not proj.is_child(sec):
            raise Exception(
                "{0} is not a a child particle of {1}.".format(sec.name, proj.name)
            )

        cr_gamma = self.pmodel.nucleon_gamma(self.e_grid)
        zfac = np.zeros(self.dim)

        zfac = np.zeros_like(self.e_grid)
        for p_eidx, e in enumerate(self.e_grid):
            # if e < min_energy:
            #     min_idx = p_eidx + 1
            #     continue
            xlab, xdist = proj.dNdec_dxlab(e, sec)
            zfac[p_eidx] = np.trapz(xlab ** (-cr_gamma[p_eidx] - 2.0) * xdist, x=xlab)
        return zfac


class MatrixBuilder(object):
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
            "Start filling matrices. Skip_decay_matrix = {0}".format(skip_decay_matrix),
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
                self.C_blocks[idx] *= asarray(
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

                self.D_blocks[idx] *= asarray(
                    parent.inverse_decay_length(), dtype=config.floatlen
                )

            self.dec_m = self._csr_from_blocks(self.D_blocks)

        for mname, mat in [("C", self.int_m), ("D", self.dec_m)]:
            mat_density = float(mat.nnz) / float(np.prod(mat.shape))
            info(5, "{0} Matrix info:".format(mname))
            info(5, "    density    : {0:3.2%}".format(mat_density))
            info(5, "    shape      : {0} x {1}".format(*mat.shape))
            info(5, "    nnz        : {0}".format(mat.nnz))
            info(10, "    sum        :", mat.sum())

        info(3, "Done filling matrices.")

        return self.int_m, self.dec_m

    def _average_operator(self, op_mat):
        """Averages the continuous loss operator by performing
        1/max_step explicit euler steps"""

        n_steps = int(1.0 / config.loss_step_for_average)
        info(
            10,
            "Averaging continuous loss using {0} intermediate steps.".format(n_steps),
        )

        op_step = eye(self._energy_grid.d) + op_mat * config.loss_step_for_average
        return linalg.matrix_power(op_step, n_steps) - eye(self._energy_grid.d)

    def cont_loss_operator(self, pdg_id):
        """Returns continuous loss operator that can be summed with appropriate
        position in the C matrix."""
        op_mat = -diag(1 / self._energy_grid.c).dot(
            self.op_matrix.dot(diag(self._pman[pdg_id].dEdX))
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
        return zeros((self._pman.dim, self._pman.dim), dtype=config.floatlen)

    def _csr_from_blocks(self, blocks):
        """Construct a csr matrix from a dictionary of submatrices (blocks)

        Note::

            It's super pain the a** to construct a properly indexed sparse matrix
            directly from the blocks, since bmat totally messes up the order.
        """
        new_mat = zeros((self.dim_states, self.dim_states), dtype=config.floatlen)

        for (c, p), d in six.iteritems(blocks):
            rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
            try:
                new_mat[rc.lidx : rc.uidx, rp.lidx : rp.uidx] = d
            except ValueError:
                raise Exception(
                    "Dimension mismatch: matrix "
                    + "{0}x{1}, p={2}:({3},{4}), c={5}:({6},{7})".format(
                        self.dim_states,
                        self.dim_states,
                        rp.name,
                        rp.lidx,
                        rp.uidx,
                        rc.name,
                        rc.lidx,
                        rc.uidx,
                    )
                )
        return csr_matrix(new_mat)

    def _follow_chains(self, p, pprod_mat, p_orig, idcs, propmat, reclev=0):
        """Some recursive magic."""
        info(40, reclev * "\t", "entering with", p.name)
        # print 'orig, p', p_orig.pdg_id, p.pdg_id
        for d in p.children:
            info(40, reclev * "\t", "following to", d.name)
            if not d.is_resonance:
                # print 'adding stuff', p_orig.pdg_id, p.pdg_id, d.pdg_id
                dprop = self._zero_mat()
                p._assign_decay_idx(d, idcs, d.hadridx, dprop)
                propmat[(d.mceqidx, p_orig.mceqidx)] += dprop.dot(pprod_mat)

            if config.debug_level >= 20:
                pstr = "res"
                dstr = "Mchain"
                if idcs == p.hadridx:
                    pstr = "prop"
                    dstr = "Mprop"
                info(
                    40,
                    reclev * "\t",
                    "setting {0}[({1},{3})->({2},{4})]".format(
                        dstr, p_orig.name, d.name, pstr, "prop"
                    ),
                )

            if d.is_mixed or d.is_resonance:
                dres = self._zero_mat()
                p._assign_decay_idx(d, idcs, d.residx, dres)
                reclev += 1
                self._follow_chains(
                    d, dres.dot(pprod_mat), p_orig, d.residx, propmat, reclev
                )
            else:
                info(20, reclev * "\t", "\t terminating at", d.name)

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
                        diag(ones(self.dim)).astype(config.floatlen),
                        p,
                        p.hadridx,
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
                    info(1, "No interactions by {0} ({1}).".format(p.name, p.pdg_id))
                continue
            for s in p.hadr_secondaries:
                # if s not in self.pman.cascade_particles:
                #     print 'Doing nothing with', p.pdg_id, s.pdg_id
                #     continue

                if not s.is_resonance:
                    cmat = self._zero_mat()
                    p._assign_hadr_dist_idx(s, p.hadridx, s.hadridx, cmat)
                    self.C_blocks[(s.mceqidx, p.mceqidx)] += cmat

                cmat = self._zero_mat()
                p._assign_hadr_dist_idx(s, p.hadridx, s.residx, cmat)
                self._follow_chains(s, cmat, p, s.residx, self.C_blocks, reclev=1)

    def _construct_differential_operator(self):
        """Constructs a derivative operator for the contiuous losses.

        This implmentation uses a 6th-order finite differences operator,
        only depends on the energy grid. This is an operator for a sub-matrix
        of dimension (energy grid, energy grid) for a single particle. It
        can be likewise applied to all particle species. The dEdX values are
        applied later in ...
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

        # Centered diagonals
        # diags = [-3, -2, -1, 1, 2, 3]
        # coeffs = [-1, 9, -45, 45, -9, 1]
        # denom = 60.
        diags = diags_left_2
        coeffs = coeffs_left_2
        denom = 60.0

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

        op_matrix = zeros((dim_e, dim_e), dtype=config.floatlen)
        op_matrix[0, asarray(diags_leftmost)] = asarray(coeffs_leftmost) / (
            denom_leftmost * h[0]
        )
        op_matrix[1, 1 + asarray(diags_left_1)] = asarray(coeffs_left_1) / (
            denom_left_1 * h[1]
        )
        op_matrix[2, 2 + asarray(diags_left_2)] = asarray(coeffs_left_2) / (
            denom_left_2 * h[2]
        )
        op_matrix[last, last + asarray(diags_rightmost)] = asarray(coeffs_rightmost) / (
            denom_rightmost * h[last]
        )
        op_matrix[last - 1, last - 1 + asarray(diags_right_1)] = asarray(
            coeffs_right_1
        ) / (denom_right_1 * h[last - 1])
        op_matrix[last - 2, last - 2 + asarray(diags_right_2)] = asarray(
            coeffs_right_2
        ) / (denom_right_2 * h[last - 2])
        for row in range(3, dim_e - 3):
            op_matrix[row, row + asarray(diags)] = asarray(coeffs) / (denom * h[row])

        self.op_matrix = op_matrix
