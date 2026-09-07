from time import time

import numpy as np

from MCEq import config
from MCEq.data.download import ensure_db_available
from MCEq.data.model_names import normalize_hadronic_model_name
from MCEq.driver import batch, paths
from MCEq.driver.initial_state import InitialState
from MCEq.driver.system import CascadeSystem
from MCEq.misc import info
from MCEq.operators.compiled import compile_operator, em_step_scale

# trapz was finally removed with numpy 2.4
if hasattr(np, "trapezoid"):
    trapz = np.trapezoid
else:
    trapz = np.trapz


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

    #: D3: per-run configuration; the class default is the live module (the
    #: tier-2 read-through semantics), ``config=RunConfig.snapshot()`` replaces
    #: it with a detached copy. The default lives here rather than only in
    #: ``__init__`` so unbound calls from test stubs keep working unchanged.
    _cfg = config

    def __init__(self, interaction_model, primary_model, theta_deg, **kwargs):
        # D3: `config=RunConfig.snapshot()` freezes the settings at
        # construction; the default None keeps the live read-through
        # (post-construction config writes are seen at the next solve()).
        from MCEq.config.run import RunConfig

        snapshot = kwargs.pop("config", None)
        if snapshot is not None and not isinstance(snapshot, RunConfig):
            raise TypeError(
                "MCEqRun(config=...) expects a MCEq.config.run.RunConfig "
                "(build one with RunConfig.snapshot()); got "
                f"{type(snapshot).__name__}."
            )
        self._cfg = snapshot if snapshot is not None else config
        ensure_db_available(self._cfg)
        # The enable_em/helicity incompatibility is a validator on the physics
        # group (config.run_physics, plan §9): CascadeSystem holds the forced
        # copy, the flat default is untouched. (The old ctor force-wrote the
        # flat flag globally here, which leaked into every later non-EM run
        # in the same process.)
        self.medium = kwargs.pop("medium", self._cfg.physics.interaction_medium)
        le_config = self._cfg.physics.low_energy
        low_energy_model = kwargs.pop("low_energy_model", le_config.get("model"))
        he_le_transition = kwargs.pop(
            "he_le_transition", le_config.get("he_le_transition", 80.0)
        )
        he_le_trwidth = kwargs.pop("he_le_trwidth", le_config.get("he_le_trwidth", 0.3))
        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_model = kwargs.pop(
            "density_model", self._cfg.environment.density_model
        )
        self.theta_deg = theta_deg

        # The physics system (database, tables, pman, builder, matrices)
        # lives in driver/system.py since Phase 6; the keep-list names are
        # properties below delegating into it.
        self._system = CascadeSystem(
            interaction_model,
            medium=self.medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
            cfg=self._cfg,
        )

        # Initialize solution vector
        self._solution = np.zeros(1)
        # Initial condition (phi0 + restore list + pmodel) lives in
        # driver/initial_state.py since Phase 6 commit 3
        self._initial_state = InitialState(self._system)

        # Set interaction model and compute grids and matrices
        self.set_interaction_model(
            interaction_model,
            particle_list=kwargs.pop("particle_list", None),
            build_matrices=kwargs.pop("build_matrices", True),
        )

        # Default GPU device id for CUDA
        self._cuda_device = kwargs.pop("cuda_gpu_id", self._cfg.backend.cuda_gpu_id)

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
            return_as = self._cfg.output.return_as

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
            if nsuccess == 0 and self._cfg.output.excpt_on_missing_particle:
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
        if not self._cfg.physics.enable_default_tracking:
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
                if self._cfg.output.excpt_on_missing_particle:
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

    # --- Phase 6 delegation into CascadeSystem (driver/system.py) ---------
    # The §8.7 keep-list names the physics system owns are properties here,
    # so every attribute access written against the single-class original
    # still resolves. ``int_m``/``dec_m`` are read-write: the batch sweep
    # harness in tests/golden/_operator_sweep.py swaps matrices directly.

    @property
    def config(self):
        """The configuration this run reads: the live module, or the detached
        snapshot passed as ``config=RunConfig.snapshot()`` (D3)."""
        return self._cfg

    @property
    def _mceq_db(self):
        return self._system._mceq_db

    @property
    def _interactions(self):
        return self._system._interactions

    @property
    def _int_cs(self):
        return self._system._int_cs

    @property
    def _decays(self):
        return self._system._decays

    @property
    def _cont_losses(self):
        return self._system._cont_losses

    @property
    def pman(self):
        return self._system.pman

    @property
    def _particle_list(self):
        return self._system._particle_list

    @property
    def _energy_grid(self):
        return self._system._energy_grid

    @property
    def matrix_builder(self):
        return self._system.matrix_builder

    @property
    def _phi0(self):
        return self._initial_state.phi0

    @_phi0.setter
    def _phi0(self, vec):
        self._initial_state.phi0 = vec

    @property
    def pmodel(self):
        return self._initial_state.pmodel

    @pmodel.setter
    def pmodel(self, model):
        self._initial_state.pmodel = model

    @property
    def _restore_initial_condition(self):
        return self._initial_state._restore_initial_condition

    @_restore_initial_condition.setter
    def _restore_initial_condition(self, lst):
        self._initial_state._restore_initial_condition = lst

    @property
    def get_nucleon_spectrum(self):
        return self._initial_state.get_nucleon_spectrum

    @property
    def int_m(self):
        return self._system.int_m

    @int_m.setter
    def int_m(self, mat):
        self._system.int_m = mat

    @property
    def dec_m(self):
        return self._system.dec_m

    @dec_m.setter
    def dec_m(self, mat):
        self._system.dec_m = mat

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

        The load itself is the system's branch ladder
        (:meth:`CascadeSystem.reload_for_model`); the facade keeps the
        original skip short-circuit before it and the resize/rebuild after
        it — the resize replays the facade's initial-condition methods by
        name (PR #163), so only the facade can drive the load -> resize ->
        build order.

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

        # R2-B: a genuine model change drops the injected channels (the
        # docstring above: "Calling `set_interaction_model` overwrites DDM
        # with a different model"); a same-model reload keeps them, since
        # surviving the reload is the whole point of the override hook.
        if self._interactions.iam != interaction_model:
            self._interactions.clear_channel_overrides()

        self._system.reload_for_model(
            interaction_model, particle_list, update_particle_list
        )

        self._resize_vectors_and_restore()

        # initialize matrices
        if not build_matrices:
            return
        self._system.build_matrices()

    def _resize_vectors_and_restore(self):
        """Update solution and grid vectors if the number of particle species
        or the interaction models change. The previous state, such as the
        initial spectrum, are restored.

        The ``phi0`` reallocation and the restore replay live on
        :class:`MCEq.driver.initial_state.InitialState`; this wrapper keeps
        the original name and order (solution first, then initial state)
        and supplies the replay, which must resolve against the facade
        because the stored entries are facade method *names* (PR #163).
        """
        self._solution = np.zeros(self.dim_states)
        self._initial_state.resize(
            self._restore_initial_condition,
            lambda name, args: getattr(self, name)(*args),
        )

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model (delegates to InitialState).

        See :meth:`MCEq.driver.initial_state.InitialState.set_primary_model`
        for the original docstring; behaviour unchanged.
        """
        return self._initial_state.set_primary_model(model_class_or_object, tag)

    def set_single_primary_particle(
        self, E, corsika_id=None, pdg_id=None, append=False
    ):
        """Set a single primary particle (delegates to InitialState).

        See :meth:`MCEq.driver.initial_state.InitialState.
        set_single_primary_particle` for the original docstring; behaviour
        unchanged.
        """
        return self._initial_state.set_single_primary_particle(
            E, corsika_id=corsika_id, pdg_id=pdg_id, append=append
        )

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user spectrum (delegates to InitialState).

        See :meth:`MCEq.driver.initial_state.InitialState.
        set_initial_spectrum` for the original docstring; behaviour
        unchanged.
        """
        return self._initial_state.set_initial_spectrum(spectrum, pdg_id, append=append)

    def get_initial_state(self):
        """Return a copy of the current initial-condition vector ``phi0``.

        Delegates to
        :meth:`MCEq.driver.initial_state.InitialState.get_initial_state`.
        """
        return self._initial_state.get_initial_state()

    def initial_state(self, components):
        """Compose an initial-state column without mutating the instance.

        Delegates to :meth:`MCEq.driver.initial_state.InitialState.
        initial_state`; behaviour unchanged.
        """
        return self._initial_state.initial_state(components)

    def set_density_model(self, density_model_or_config):
        """Sets model of the atmosphere.

        To choose, for example, a CORSIKA parametrization for the Southpole in January,
        do the following::

            mceq_instance.set_density_model(("CORSIKA", ("PL_SouthPole", "January")))

        More details about the choices can be found in
        :mod:`MCEq.geometry.density_profiles`.Calling this method will
        issue a recalculation of the interpolation and the integration path.

        From v1.2, `density_model_or_config` may also be an instance of
        :class:`~MCEq.environment.base.EarthsAtmosphere` or
        :class:`~MCEq.environment.target.GeneralizedTarget`.

        Args:
          density_model_or_config (obj or tuple of strings):
            (parametrization type, arguments)
        """
        from MCEq.environment.base import EarthsAtmosphere
        from MCEq.environment.target import GeneralizedTarget

        # Check if string arguments or an instance of the density class is provided
        if not isinstance(
            density_model_or_config, (EarthsAtmosphere, GeneralizedTarget)
        ):
            from MCEq.environment import registry

            base_model, model_config = density_model_or_config

            if base_model not in registry.DENSITY_MODELS:
                info(
                    0,
                    "Unknown density model. Available choices are:\n",
                    "\n".join(registry.available_models()),
                )
                raise ValueError("Choose a different profile.")

            info(1, "Setting density profile to", base_model, model_config)

            self.density_model = registry.build(
                base_model, model_config, environment=self._cfg.environment
            )
        else:
            self.density_model = density_model_or_config

        if self.theta_deg is not None and isinstance(
            self.density_model, EarthsAtmosphere
        ):
            self.set_zenith_azimuth(self.theta_deg)
        elif isinstance(self.density_model, GeneralizedTarget):
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
        from MCEq.environment.target import GeneralizedTarget

        info(
            2,
            f"Zenith angle {zenith_deg:6.2f}"
            + (f", azimuth {azimuth_deg:6.2f}" if azimuth_deg is not None else ""),
        )

        if isinstance(self.density_model, GeneralizedTarget):
            raise Exception("GeneralizedTarget does not support angles.")

        # Cache check: skip if nothing has changed
        cached_theta = self.density_model.theta_deg
        cached_azi = getattr(self.density_model, "_current_azimuth_deg", None)
        if cached_theta == zenith_deg and cached_azi == azimuth_deg:
            info(2, "Angle selection corresponds to cached value, skipping calc.")
            # Track the angle here too (bug B13): the attribute has to be
            # right even when the recomputation is skipped, and even if the
            # atmosphere was driven directly, behind this method's back. Safe
            # on this branch because a cache hit means an earlier ``set_theta``
            # already accepted the angle.
            self.theta_deg = zenith_deg
            return

        # Dispatch to set_theta with or without azimuth_deg depending on
        # the density model's ``depends_on_azimuth`` attribute. Both
        # MSIS00LocationCentered and MSIS21LocationCentered accept the
        # extra azimuth_deg argument; everything else ignores azimuth.
        if getattr(self.density_model, "depends_on_azimuth", False):
            self.density_model.set_theta(zenith_deg, azimuth_deg=azimuth_deg)
        else:
            self.density_model.set_theta(zenith_deg)

        # Track the angle actually applied (bug B13). It used to keep its
        # constructor value forever, so ``MCEqRun.theta_deg`` disagreed with
        # ``density_model.theta_deg`` after the first change of angle -- which
        # is why the paths golden labels its rows by the latter.
        #
        # Assigned only once ``set_theta`` has accepted the angle. Assigning
        # it earlier is observable: an atmosphere with ``max_theta = 90``
        # rejects 120 deg, and ``set_density_model`` replays ``self.theta_deg``
        # into the next model (l.1250), so a caller who caught the range error
        # and then swapped atmospheres would silently solve at an angle that
        # was never accepted.
        self.theta_deg = zenith_deg
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

        R2-B: the matrices are registered as channel overrides on the
        interaction table (``Interactions.set_channel_override``) and
        reach ``hadr_yields``/``int_m`` through the ordinary
        ``get_matrix`` wiring, so the injection now survives a model
        reload and ``regenerate_matrices`` (which previously wiped it
        silently — the matrix only lived in a particle dict).
        ``set_interaction_model`` to a different hadronic model clears
        the overrides first, matching the docstring above.
        """

        from MCEq.models.ddm.ddm import isospin_partners, isospin_symmetries

        injected = []
        for (prim, sec), mati in ddm.ddm_matrices(self).items():
            info(5, f"Injecting DDM {prim} --> {sec}")
            iso_part = isospin_partners[prim]

            self._interactions.set_channel_override(prim, sec, mati)
            info(5, "Injecting isopart", iso_part, isospin_symmetries[iso_part][sec])
            self._interactions.set_channel_override(
                iso_part, isospin_symmetries[iso_part][sec], mati
            )
            injected.append(
                ((prim, sec), (iso_part, isospin_symmetries[iso_part][sec]))
            )

        if self._cfg.debug.level > 2:
            s = "DDM matrices injected into MCEq:\n"
            for (prim, sec), (iprim, isec) in injected:
                s += f"\t{prim}-->{sec}, isospin: {iprim} --> {isec}\n"
            print(s)

        self.regenerate_matrices()

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
        self._system.regenerate(skip_decay_matrix=skip_decay_matrix)
        self._resize_vectors_and_restore()
        self._system.build_matrices(skip_decay_matrix=skip_decay_matrix)

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
        info(2, f"Launching {self._cfg.backend.kernel_config} solver")

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

        info(2, f"Launching {self._cfg.backend.kernel_config} solver")
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

    def _is_geomag_eligible_atmosphere(self):
        """True if the active atmosphere has a meaningful geographic location.

        The :meth:`solve_fullsky` auto-cutoff fires only when this returns
        True and ``self.geomagnetic_cutoff`` is True or None. Eligible
        atmospheres are MSIS*-derived classes (both MSIS00 and MSIS21
        hierarchies) and any other atmosphere whose ``self.location``
        appears in :data:`atmosphere_parameters.LOCATIONS`.
        """
        from MCEq.environment.msis00 import MSIS00Atmosphere
        from MCEq.environment.msis21 import MSIS21Atmosphere
        from MCEq.environment.parameters import LOCATIONS

        dm = self.density_model
        if dm is None:
            return False
        if isinstance(dm, MSIS00Atmosphere):
            return True
        # MSIS21 is a parallel class tree (not MSIS00 subclass). Importing the
        # module is cheap -- it defers `nrlmsis` to the constructor -- so the
        # `hasattr` guard the PEP-562 re-export needed is gone.
        if isinstance(dm, MSIS21Atmosphere):
            return True
        loc = getattr(dm, "location", None)
        return isinstance(loc, str) and loc in LOCATIONS

    # Batch surface: direct bindings of the module functions (M7). The
    # Phase-6 delegators are gone -- a plain ``solve_batch = batch.solve_batch``
    # keeps the public method surface identical (docs automodapi, monkeypatch
    # via the class, subclass calls all unchanged) while dropping ~120 lines
    # of pass-through prose; the full contract lives in the module docstrings.
    # The first parameter of each function is named ``run`` precisely so that
    # this binding reads as the method.
    solve_batch = batch.solve_batch
    solve_multirhs = batch.solve_multirhs
    solve_fullsky = batch.solve_fullsky
    _build_condition_paths = paths.build_condition_paths

    def _resolve_secant(self):
        """The sec(theta) operator set for a solve, or ``None`` for the
        paraxial transport.

        One constant set serves every column of every batch (the cap is
        not zenith dependent). The coupled route is driver code plus the
        ``apply_off`` binding, so every kernel carries it at every
        precision; only ``config.secant_mode`` decides.
        """
        # getattr fallback: the tri-state test calls this unbound against a
        # SimpleNamespace stub, as before D3.
        cfg = getattr(self, "_cfg", config)
        if cfg.secant_mode(self._mceq_db.is_2d) == "off":
            return None
        return self._build_secant_ops()

    def _build_secant_ops(self):
        """The constant sec(theta) operator set for the current geometry (see
        :mod:`MCEq.operators.secant`), built once per (geometry, ``secant_*``
        parameters) and reused across solves — the operator and backend
        caches key on its identity."""
        from MCEq.operators.secant import build_secant_kernel_ops

        k_grid = np.asarray(self._mceq_db.k_grid)
        e_centers = np.asarray(self._energy_grid.c)
        n_species = self.dim_states // self.dim
        key = (
            hash(k_grid.tobytes()),
            hash(e_centers.tobytes()),
            n_species,
            self._cfg.secant_theta_cap(),
            self._cfg.secant.row_kmax,
            self._cfg.secant.lam_rel,
            self._cfg.secant.w_flat,
            self._cfg.secant.e_max,
        )
        cached = getattr(self, "_secant_ops_cache", None)
        if cached is None or cached[0] != key:
            ops = build_secant_kernel_ops(
                k_grid,
                e_centers,
                n_species,
                self._cfg.secant,
                theta_cap_deg=self._cfg.secant_theta_cap(),
                paths=self._cfg.paths,
            )
            self._secant_ops_cache = cached = (key, ops)
        return cached[1]

    def _compiled_operator(self, sec_ops=None):
        """:func:`~MCEq.operators.compiled.compile_operator` of the current
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

        kc = self._cfg.backend.kernel_config.lower()
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
                    f"Unsupported integrator setting '{self._cfg.backend.kernel_config}'. "
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
        """Cure-B stiffness scale r_EM of the current ``int_m``, memoised.

        The scale itself is a property of the operator and is computed by
        :func:`MCEq.operators.compiled.em_step_scale`; this method holds the
        memo. ``int_m`` is X-constant, so one spectral radius serves every step
        of every solve, and the entry is keyed on the *identity* of ``int_m``,
        which invalidates it whenever the matrices are rebuilt. Returns 0.0
        before the matrices exist, and 0.0 when no e+/-/gamma are loaded.
        """
        int_m = getattr(self, "int_m", None)
        if int_m is None:
            return 0.0
        cache = getattr(self, "_em_step_scale_cache", None)
        if cache is not None and cache[0] is int_m:
            return cache[1]
        r_em = em_step_scale(
            int_m,
            self.pman.all_particles,
            dense_eig_max=getattr(config, "em_step_dense_eig_max", 4000),
        )
        self._em_step_scale_cache = (int_m, r_em)
        return r_em

    def _em_cascade_dx_cap(self):
        """Cure-B effective dX cap from the EM-cascade stiffness, or np.inf.

        Moved verbatim to :func:`MCEq.solvers.path.em_cascade_dx_cap`
        (Phase 6 commit 5, D26), then to :func:`MCEq.driver.paths.em_cascade_dx_cap`
        at M6 (D-4); the memoised stiffness it consumes stays on this class
        (:meth:`_em_cascade_step_scale`).
        """
        return paths.em_cascade_dx_cap(self, self._em_cascade_step_scale())

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
        """Build (or reuse the cache of) the ETD2 integration path.

        Moved verbatim to :func:`MCEq.driver.paths.calculate_integration_path`
        (Phase 6 commit 5, D26; M6 / D-4 moved the home from solvers to
        driver); the cache and ``force`` live on this instance, which the
        function reads and writes as before.
        """
        return paths.calculate_integration_path(
            self,
            int_grid,
            grid_var,
            force,
            X_start=X_start,
            eps=eps,
            dX_max=dX_max,
            dX_min=dX_min,
            fd_span=fd_span,
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

        if self._cfg.has_cuda:
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
