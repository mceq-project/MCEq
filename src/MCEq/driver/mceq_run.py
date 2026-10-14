from time import time

import numpy as np

from MCEq import config
from MCEq.data.download import ensure_db_available
from MCEq.data.model_names import normalize_hadronic_model_name
from MCEq.driver import angular, batch, observables, paths, results
from MCEq.driver.initial_state import InitialState
from MCEq.driver.system import CascadeSystem
from MCEq.misc import info
from MCEq.operators.compiled import compile_operator, em_step_scale


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
        ensure_db_available(config)
        self.medium = kwargs.pop("medium", config.interaction_medium)
        le_config = config.low_energy_extension
        low_energy_model = kwargs.pop("low_energy_model", le_config.get("model"))
        he_le_transition = kwargs.pop(
            "he_le_transition", le_config.get("he_le_transition", 80.0)
        )
        he_le_trwidth = kwargs.pop("he_le_trwidth", le_config.get("he_le_trwidth", 0.3))
        interaction_model = normalize_hadronic_model_name(interaction_model)

        # Save atmospheric parameters
        self.density_model = kwargs.pop("density_model", config.density_model)
        self.theta_deg = theta_deg

        self._system = CascadeSystem(
            interaction_model,
            medium=self.medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
        )

        # Initialize solution vector
        self._solution = np.zeros(1)
        self._initial_state = InitialState(self._system)

        # Set interaction model and compute grids and matrices
        self.set_interaction_model(
            interaction_model,
            particle_list=kwargs.pop("particle_list", None),
            build_matrices=kwargs.pop("build_matrices", True),
        )

        # Default GPU device id for CUDA
        self._cuda_device = kwargs.pop("cuda_gpu_id", config.cuda_gpu_id)

        # Geomagnetic rigidity cutoff: None auto-enables it for MSIS and
        # location-tagged atmospheres; True/False override.
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
        """Select the interaction model and rebuild the matrices.

        The initial condition is restored on the new particle layout.

        Args:
          interaction_model (str): name of interaction model
          particle_list (list, optional): parent particles of the system;
            ``None`` takes all particles of the model and their decay products
          update_particle_list (bool): re-derive the particle list from
            ``particle_list``; False reuses the current one
          force (bool): force loading interaction model
          build_matrices (bool): False skips the matrix construction
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

        info(1, interaction_model)

        if (
            not force
            and self._interactions.iam == interaction_model
            and particle_list in (None, self._system.parent_list)
        ):
            info(2, "Skip, since current model identical to", interaction_model + ".")
            return

        # A model change clears injected channel overrides; a same-model
        # reload keeps them.
        if self._interactions.iam != interaction_model:
            self._interactions.clear_channel_overrides()

        with self._mceq_db._read_session():
            self._system.reload_for_model(
                interaction_model, particle_list, update_particle_list
            )

            self._resize_vectors_and_restore()

            if not build_matrices:
                return
            self._system.build_matrices()

    def _resize_vectors_and_restore(self):
        """Reallocate the state vectors and replay the initial condition."""
        self._solution = np.zeros(self.dim_states)
        self._initial_state.resize(
            self._restore_initial_condition,
            lambda name, args: getattr(self, name)(*args),
        )

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model.

        Args:
          model_class_or_object: a :class:`crflux.models.PrimaryFlux`
            subclass, or an instance of one
          tag: argument passed to the class when constructing the model
        """
        return self._initial_state.set_primary_model(model_class_or_object, tag)

    def set_single_primary_particle(
        self, E, corsika_id=None, pdg_id=None, append=False
    ):
        """Set a single primary nucleus or particle as initial condition.

        Nuclei are superpositions of Z protons and A-Z neutrons at
        :math:`E/A`; see
        :meth:`MCEq.driver.initial_state.InitialState.set_single_primary_particle`.

        Args:
          E (float): kinetic energy of a nucleus in GeV
          corsika_id (int): CORSIKA ID ``A*100 + Z`` of a nucleus
          pdg_id (int): PDG ID of a particle
          append (bool): If True, keep previous state and append a new particle.
        """
        return self._initial_state.set_single_primary_particle(
            E, corsika_id=corsika_id, pdg_id=pdg_id, append=append
        )

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user spectrum of one species as initial condition.

        Args:
          spectrum (np.array): differential spectrum on the kinetic-energy grid
          pdg_id (int): PDG ID of the particle
          append (bool): If True, add to the previous initial condition.
        """
        return self._initial_state.set_initial_spectrum(spectrum, pdg_id, append=append)

    def get_initial_state(self):
        """Return a copy of the current initial-condition vector ``phi0``."""
        return self._initial_state.get_initial_state()

    def initial_state(self, components):
        """Compose an initial-state column without mutating the instance.

        See :meth:`MCEq.driver.initial_state.InitialState.initial_state` for
        the component format.
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

        `density_model_or_config` may also be an instance of
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

            self.density_model = registry.build(base_model, model_config)
        else:
            self.density_model = density_model_or_config

        self.integration_path = None  # the angle cache cannot see a model switch
        if self.theta_deg is not None and isinstance(
            self.density_model, EarthsAtmosphere
        ):
            self.set_zenith_azimuth(self.theta_deg)
        elif not isinstance(self.density_model, GeneralizedTarget):
            raise ValueError(f"Density model {self.density_model} not supported.")

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
            self.theta_deg = zenith_deg
            return

        # Only azimuth-dependent atmospheres accept azimuth_deg.
        if getattr(self.density_model, "depends_on_azimuth", False):
            self.density_model.set_theta(zenith_deg, azimuth_deg=azimuth_deg)
        else:
            self.density_model.set_theta(zenith_deg)

        # Store the angle only after set_theta accepted it.
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

        The DDM matrices are installed as channel overrides of the
        interaction tables and the matrices are rebuilt. The overrides survive
        same-model reloads and ``regenerate_matrices``; selecting a
        different interaction model clears them.
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

        if config.debug_level > 2:
            s = "DDM matrices injected into MCEq:\n"
            for (prim, sec), (iprim, isec) in injected:
                s += f"\t{prim}-->{sec}, isospin: {iprim} --> {isec}\n"
            print(s)

        self.regenerate_matrices()

    def set_mod_pprod(self, prim_pdg, sec_pdg, x_func, x_func_args, delay_init=False):
        """Sets combination of projectile/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of
        ``prim_pdg`` is modified according to the function passed to
        :meth:`MCEq.data.interaction_tables.Interactions._gen_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
          x_func (object): reference to function
          x_func_args (tuple): arguments passed to ``x_func``
          delay_init (bool): accepted but currently unused; call
                             ``regenerate_matrices`` after adding all
                             modifications
        """
        info(1, f"{prim_pdg}/{sec_pdg}, {sec_pdg}, {x_func.__name__}, {x_func_args!s}")

        init = self._interactions._set_mod_pprod(prim_pdg, sec_pdg, x_func, x_func_args)

        # Need to regenerate matrices completely
        return int(init)

    def unset_mod_pprod(self, dont_fill=False):
        """Removes modifications from :func:`MCEqRun.set_mod_pprod`.

        Args:
          dont_fill (bool): if True, defer matrix regeneration; the caller
          runs ``regenerate_matrices`` later by hand
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

        self._system.regenerate()
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
        dtype=None,
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
            ``config.etd2_path["eps"]``; ``None`` there selects a
            system-aware default.
          dX_max (float | None): cap on step size (off-diagonal stability
            cliff) for ETD2. ``None`` → ``config.etd2_path["dX_max"]``;
            ``None`` there selects a system-aware default.
          dX_min (float | None): floor on step size for ETD2. ``None`` →
            ``config.etd2_path["dX_min"]``.
          fd_span (float | None): forward-FD probe span for the ETD2
            schedule's local rate estimate. ``None`` →
            ``config.etd2_path["fd_span"]``.
          dtype (np.float32 | np.float64 | None): state precision.
            ``None``: float32 on CUDA with ``config.cuda_fp_precision = 32``,
            else float64. The solution is returned in float64 either way.
          kwargs (dict): ``skip_integration_path=True`` reuses the cached
            ``self.integration_path`` instead of rebuilding it.

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

        # 2D: a delta in angle has equal amplitude in every Legendre mode
        # (P_l(1) = 1), so phi0 is repeated over the n_k mode blocks.
        if self._mceq_db.is_2d:
            phi0 = np.tile(self._phi0, self._mceq_db.n_k)
        else:
            phi0 = np.copy(self._phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, f"for {nsteps} integration steps.")

        start = time()

        self._solution, self.grid_sol = self._run_etd2_resolved(
            nsteps, dX, rho_inv, phi0, grid_idcs, dtype
        )

        if isinstance(self.grid_sol, list):
            self.grid_sol = np.asarray(self.grid_sol)

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def _run_etd2_resolved(self, nsteps, dX, rho_inv, phi0, grid_idcs, dtype):
        """:meth:`_run_etd2` at the resolved precision; results in float64."""
        dtype = batch.resolve_dtype(self, dtype)
        sec_ops = self._resolve_secant()
        batch._warn_low_energy_cuda_fp32(self, dtype, sec_ops)
        sol, grid_sol = self._run_etd2(
            nsteps, dX, rho_inv, phi0, grid_idcs, dtype=dtype, sec_ops=sec_ops
        )
        return np.asarray(sol, dtype=np.float64), np.asarray(grid_sol, dtype=np.float64)

    def solve_from_integration_path(
        self, nsteps, dX, rho_inv, grid_idcs, *, dtype=None
    ):
        """Launches the solver directly for parameters of the integration path.

        The helper function is useful if you want to skip the calculation of
        the integration path every time. This function is intended for expert
        use and is not required for normal operation.

        The parameters can be obtained after calling _calculate_integration_path
        with correct settings for density and angle parameters::

            nsteps, dX, rho_inv, grid_idcs = self.integration_path

        Args:
          nsteps (int): number of integration steps
          dX (list): the delta_X's
          rho_inv (list): the inverse of the density at each step
          grid_idcs (list): list of steps at which the solution
          is dumped into `grid_sol`
        """

        info(2, f"Launching {config.kernel_config} solver")
        info(2, f"for {nsteps} integration steps.")

        start = time()

        # 2D: phi0 repeated over the n_k mode blocks, as in solve().
        if self._mceq_db.is_2d:
            phi0 = np.tile(self._phi0, self._mceq_db.n_k)
        else:
            phi0 = np.copy(self._phi0)

        self._solution, self.grid_sol = self._run_etd2_resolved(
            nsteps, dX, rho_inv, phi0, grid_idcs, dtype
        )

        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def _is_geomag_eligible_atmosphere(self):
        """True for MSIS atmospheres and atmospheres with a known location.

        Decides the :meth:`solve_fullsky` geomagnetic cutoff when
        ``geomagnetic_cutoff`` is None.
        """
        from MCEq.environment.msis00 import MSIS00Atmosphere
        from MCEq.environment.msis21 import MSIS21Atmosphere
        from MCEq.environment.parameters import LOCATIONS

        dm = self.density_model
        if dm is None:
            return False
        if isinstance(dm, MSIS00Atmosphere):
            return True
        if isinstance(dm, MSIS21Atmosphere):
            return True
        loc = getattr(dm, "location", None)
        return isinstance(loc, str) and loc in LOCATIONS

    solve_batch = batch.solve_batch
    solve_fullsky = batch.solve_fullsky
    _build_condition_paths = paths.build_condition_paths
    get_solution = results.get_solution
    _get_solution_from_state = results._get_solution_from_state
    convert_to_theta_space = results.convert_to_theta_space
    n_particles = observables.n_particles
    n_mu = observables.n_mu
    n_e = observables.n_e
    z_factor = observables.z_factor
    decay_z_factor = observables.decay_z_factor

    def _resolve_secant(self):
        """sec(theta) operators for the solve, or None if secant mode is off."""
        if not config.secant_enabled(self._mceq_db.is_2d):
            return None
        return self._build_secant_ops()

    def _build_secant_ops(self):
        """sec(theta) operator set (:mod:`MCEq.operators.secant`), cached per
        grid, species count and secant parameters."""
        from MCEq.operators.secant import build_secant_kernel_ops

        k_grid = np.asarray(self._mceq_db.k_grid)
        e_centers = np.asarray(self._energy_grid.c)
        n_species = self.dim_states // self.dim
        key = (
            hash(k_grid.tobytes()),
            hash(e_centers.tobytes()),
            n_species,
            config.secant_theta_cap(),
            config.secant_theta_e_max,
        )
        cached = getattr(self, "_secant_ops_cache", None)
        if cached is None or cached[0] != key:
            ops = build_secant_kernel_ops(
                k_grid,
                e_centers,
                n_species,
                config.secant_theta_cap(),
                e_max=config.secant_theta_e_max,
            )
            self._secant_ops_cache = cached = (key, ops)
        return cached[1]

    def _compiled_operator(self, sec_ops=None):
        """Compiled operator of the current matrices, cached per identity of
        ``int_m``, ``dec_m`` and ``sec_ops`` (paraxial and coupled entries)."""
        cache = self.__dict__.setdefault("_operator_cache", {})
        coupled = sec_ops is not None
        entry = cache.get(coupled)
        if (
            entry is None
            or entry[0] is not self.int_m
            or entry[1] is not self.dec_m
            or entry[2] is not sec_ops
        ):
            op = compile_operator(
                *angular.inputs(self), sec_ops, **angular.extras(self)
            )
            cache[coupled] = entry = (self.int_m, self.dec_m, sec_ops, op)
        return entry[3]

    def _etd2_backend(self, sec_ops=None, dtype=np.float64):
        """Solver backend of ``kernel_config`` for the compiled operator.

        Cached per kernel, precision and coupling; rebuilt when the operator
        or the CUDA device changes.
        """
        import MCEq.solvers as solvers

        kc = config.kernel_config.lower()
        on_cuda = kc in ("cuda", "cuda_etd2")
        fp = 32 if np.dtype(dtype) == np.float32 else 64
        op = self._compiled_operator(sec_ops)
        cache = self.__dict__.setdefault("_backend_cache", {})
        op_fp = config.cuda_operator_precision if on_cuda else None
        key = (kc, fp, op.coupled, op_fp)
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
                    op,
                    device_id=self._cuda_device,
                    fp_precision=fp,
                    operator_precision=op_fp,
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
        """Integrate a ``(dim,)`` or ``(dim, K)`` state along one path."""
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
        """Integrate K states along per-column paths with the LPT carousel;
        returns the ``(dim, K)`` final states in column order."""
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
        """Release backend resources (MKL handles, Accelerate slots, GPU buffers).

        Idempotent; the next solve rebuilds the caches.
        """
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
        """Spectral radius of the EM block of ``int_m`` (cached per matrix).

        0.0 before the matrices exist or without e+/e-/gamma.
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
            dense_eig_max=config.em_step_dense_eig_max,
        )
        self._em_step_scale_cache = (int_m, r_em)
        return r_em

    def _em_cascade_dx_cap(self):
        """EM stiffness step cap, or np.inf."""
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
        """Build the ETD2 integration path, or reuse the cached one."""
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
