"""MCEqRun, the user interface for the cascade calculation."""

import warnings
from collections import defaultdict
from time import time
from types import SimpleNamespace

import numpy as np

import MCEq.data
import MCEq.solvers as solvers
from MCEq import config
from MCEq.batch import BatchSolver, resolve_dtype
from MCEq.config import detect
from MCEq.data.download import ensure_db_available
from MCEq.geometry import density_profiles as dp
from MCEq.geometry import msis21
from MCEq.initial_state import deposit_single_primary, fill_nucleons
from MCEq.misc import info, normalize_hadronic_model_name
from MCEq.operators import secant_table
from MCEq.operators.compiled import compile_operator
from MCEq.operators.legendre import backward_removal, dense_count, theta_density
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.operators.secant import build_secant_kernel_ops
from MCEq.particlemanager import ParticleManager
from MCEq.solvers.path import (
    LOSS_STEP_SAFETY,
    continuous_loss_rate,
    em_cascade_dx_cap,
    em_step_scale,
    etd2_nonuniform_path,
)

__all__ = ["MCEqRun", "MatrixBuilder"]

#: The atmospheres of :meth:`MCEqRun.set_density_model` by name.
DENSITY_MODELS = {
    "MSIS00": dp.MSIS00Atmosphere,
    "MSIS00_IC": dp.MSIS00IceCubeCentered,
    "MSIS00_KM3NeT": dp.MSIS00KM3NeTCentered,
    "MSIS21": msis21.MSIS21Atmosphere,
    "MSIS21_IC": msis21.MSIS21IceCubeCentered,
    "MSIS21_KM3NeT": msis21.MSIS21KM3NeTCentered,
    "CORSIKA": dp.CorsikaAtmosphere,
    "AIRS": dp.AIRSAtmosphere,
    "Isothermal": dp.IsothermalAtmosphere,
    "GeneralizedTarget": dp.GeneralizedTarget,
}


class MCEqRun(BatchSolver):
    """Main class for handling the calculation.

    Set up the interaction model (:meth:`set_interaction_model`), the primary
    flux (:meth:`set_primary_model`), the zenith angle
    (:meth:`set_zenith_azimuth`) and the atmosphere (:meth:`set_density_model`),
    call :meth:`solve` and read the result with :meth:`get_solution`. Any of
    the settings can be changed on an active instance before the next solve.

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
      low_energy_model (string or None, optional): model blended in below
        ``he_le_transition``; default ``config.low_energy_extension["model"]``
        (FLUKA20251), ``None`` switches the blending off
      he_le_transition (float, optional): transition energy in GeV (80)
      he_le_trwidth (float, optional): 10--90% width of the transition in
        log10(E/GeV) decades (0.3); 0 is a hard switch
    """

    def __init__(self, interaction_model, primary_model, theta_deg, **kwargs):
        ensure_db_available(config)
        self.medium = kwargs.pop("medium", config.interaction_medium)
        le_config = config.low_energy_extension
        interaction_model = normalize_hadronic_model_name(interaction_model)
        self.density_model = kwargs.pop("density_model", config.density_model)
        self.theta_deg = theta_deg

        if config.enable_em and config.muon_helicity_dependence:
            info(1, "enable_em: helicity-dependent muon decays are switched off.")
        le_default = "low_energy_model" not in kwargs
        self._mceq_db = MCEq.data.HDF5Backend(
            medium=self.medium,
            low_energy_model=kwargs.pop("low_energy_model", le_config.get("model")),
            he_le_transition=kwargs.pop(
                "he_le_transition", le_config.get("he_le_transition", 80.0)
            ),
            he_le_trwidth=kwargs.pop(
                "he_le_trwidth", le_config.get("he_le_trwidth", 0.3)
            ),
        )
        with self._mceq_db._read_session():
            le_model = self._mceq_db.low_energy_model
            if (
                le_default
                and le_model is not None
                and not self._mceq_db.has_interaction_model(le_model)
            ):
                # The default must not borrow air tables for another medium.
                info(
                    1,
                    f"No {le_model} tables for medium {self.medium!r}: "
                    "running without a low-energy model.",
                )
                self._mceq_db.low_energy_model = None
            self._interactions = MCEq.data.Interactions(self._mceq_db)
            self._int_cs = MCEq.data.InteractionCrossSections(
                self._mceq_db, interaction_model
            )
            self._cont_losses = MCEq.data.ContinuousLosses(self._mceq_db)
            self._decays = MCEq.data.Decays(self._mceq_db)
        self._energy_grid = self._mceq_db.energy_grid

        #: :class:`MCEq.particlemanager.ParticleManager` of the current species
        self.pman = None
        self.matrix_builder = None
        # Parent list requested by the caller (None: all particles of the model)
        self._parent_list = None
        self.pmodel = None
        self.get_nucleon_spectrum = None
        # Initial condition as (kind, *args) entries, re-deposited on every
        # new species layout
        self._initial_condition = []
        self._phi0 = np.zeros(1)
        self._solution = np.zeros(1)

        self.set_interaction_model(
            interaction_model,
            particle_list=kwargs.pop("particle_list", None),
            build_matrices=kwargs.pop("build_matrices", True),
        )

        self._cuda_device = kwargs.pop("cuda_gpu_id", config.cuda_gpu_id)
        # Geomagnetic rigidity cutoff of solve_fullsky (off by default)
        self.geomagnetic_cutoff = kwargs.pop("geomagnetic_cutoff", False)

        self.pman.print_particle_tables(2)

        self.integration_path, self.int_grid, self.grid_var = None, None, None
        self.set_density_model(self.density_model)

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
        """Total momentum grid (centers, or bins and centers) of a particle."""
        mass = self.pman[particle_name].mass
        ptot_bins = np.sqrt((self.e_bins + mass) ** 2 - mass**2)
        ptot_grid = np.sqrt(ptot_bins[1:] * ptot_bins[:-1])
        if return_bins:
            return ptot_bins, ptot_grid
        return ptot_grid

    def etot_grid(self, particle_name, return_bins=False):
        """Total energy grid (centers, or bins and centers) of a particle."""
        etot_bins = self.e_bins + self.pman[particle_name].mass
        etot_grid = np.sqrt(etot_bins[1:] * etot_bins[:-1])
        if return_bins:
            return etot_bins, etot_grid
        return etot_grid

    def xgrid(self, particle_name, return_as, return_bins=False):
        """The grid of the spectrum variable ``return_as`` of :meth:`get_solution`."""
        if return_as == "kinetic energy":
            return (self.e_bins, self.e_grid) if return_bins else self.e_grid
        if return_as == "total energy":
            return self.etot_grid(particle_name, return_bins)
        if return_as == "total momentum":
            return self.ptot_grid(particle_name, return_bins)
        raise Exception("Unknown grid type requested.")

    def closest_energy(self, kin_energy):
        """Nearest grid energy to the kinetic energy ``kin_energy``."""
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

    def set_interaction_model(
        self,
        interaction_model,
        particle_list=None,
        update_particle_list=True,
        force=False,
        build_matrices=True,
    ):
        """Select the interaction model and rebuild the matrices.

        Args:
          interaction_model (str): name of interaction model
          particle_list (list, optional): parent particles of the system;
            ``None`` takes all particles of the model and their decay products
          update_particle_list (bool): re-derive the species from
            ``particle_list``; False keeps the current species
          force (bool): reload also an unchanged model
          build_matrices (bool): False skips the matrix construction
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)
        info(1, interaction_model)

        if (
            not force
            and self._interactions.iam == interaction_model
            and particle_list in (None, self._parent_list)
        ):
            info(2, "Skip, since current model identical to", interaction_model + ".")
            return

        # A model change clears injected channel overrides; a same-model
        # reload keeps them.
        if self._interactions.iam != interaction_model:
            self._interactions.clear_channel_overrides()

        with self._mceq_db._read_session():
            self._int_cs.load(interaction_model)
            if self.pman is not None and not update_particle_list:
                info(10, "Re-using particle list.")
                self._interactions.load(
                    interaction_model, parent_list=self.pman.pdg_id_list
                )
            else:
                self._parent_list = particle_list
                self._interactions.load(interaction_model, parent_list=particle_list)
                self._decays.load(parent_list=self._interactions.particles)
                previous = self.pman
                self.pman = ParticleManager(
                    self._interactions.particles + self._decays.particles,
                    self._energy_grid,
                    self._int_cs,
                    self.medium,
                )
                if previous is not None:
                    self.pman._tracking_requested = list(previous._tracking_requested)
                self.matrix_builder = MatrixBuilder(self.pman, self._mceq_db)
            self.pman.set_interaction_model(
                self._int_cs, self._interactions, force=True
            )
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)

            self._resize_vectors_and_restore()
            if build_matrices:
                self.int_m, self.dec_m = self.matrix_builder.construct_matrices()

    def _resize_vectors_and_restore(self):
        """Reallocate the state vectors and deposit the initial condition again."""
        self._solution = np.zeros(self.dim_states)
        self._phi0 = np.zeros(self.dim_states)
        for entry in self._initial_condition:
            self._deposit(entry)

    def _deposit(self, entry):
        """Add the initial-condition entry ``(kind, *args)`` to ``phi0``."""
        kind, *args = entry
        if kind == "primary":
            pman = self.pman
            if (2212, 0) not in pman or (2112, 0) not in pman:
                raise Exception(
                    "No nucleons in eqn system, primary flux model can not be used."
                )
            proton, neutron = pman[(2212, 0)], pman[(2112, 0)]
            e_tot = self._energy_grid.c + 0.5 * (proton.mass + neutron.mass)
            min_idx = np.argmin(np.abs(e_tot - config.minimal_primary_energy))
            p_top, n_top = self.get_nucleon_spectrum(e_tot[min_idx:])[1:]
            fill_nucleons(self._phi0, pman, min_idx, p_top, n_top)
        elif kind == "single":
            E, corsika_id, pdg_id = args
            single = np.zeros_like(self._phi0)
            deposit_single_primary(
                single, self.pman, self._energy_grid, E, corsika_id, pdg_id
            )
            self._phi0 += single
        else:
            spectrum, pdg_id = args
            if len(spectrum) != self._energy_grid.d:
                raise Exception("Lengths of spectrum and energy grid do not match.")
            p = self.pman[pdg_id]
            self._phi0[p.lidx : p.uidx] += spectrum

    def _set_initial(self, entry, append):
        """Make ``entry`` the initial condition, or add it with ``append``."""
        if append:
            self._deposit(entry)
            self._initial_condition.append(entry)
        else:
            previous = self._phi0
            self._phi0 = np.zeros_like(previous)
            try:
                self._deposit(entry)
            except Exception:
                self._phi0 = previous
                raise
            self._initial_condition = [entry]

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model.

        Args:
          model_class_or_object: a :class:`crflux.models.PrimaryFlux`
            subclass, or an instance of one
          tag: argument passed to the class when constructing the model
        """
        assert not isinstance(model_class_or_object, tuple), (
            "Primary model can not be supplied as tuples"
        )
        if not isinstance(model_class_or_object, type):
            assert any(
                "PrimaryFlux" in b.__name__
                for b in model_class_or_object.__class__.__bases__
            ), "model_class_or_object is not derived from crflux.models.PrimaryFlux"
            info(5, "Primary model supplied as object")
            self.pmodel = model_class_or_object
        else:
            info(5, "Primary model supplied as class")
            self.pmodel = model_class_or_object(tag)
        info(1, f"Primary model set to {self.pmodel.name}")
        self.get_nucleon_spectrum = self.pmodel.p_and_n_flux
        self._set_initial(("primary", self.pmodel), append=False)

    def set_single_primary_particle(
        self, E, corsika_id=None, pdg_id=None, append=False
    ):
        """Set a single primary nucleus or particle as initial condition.

        A nucleus (CORSIKA ID ``A*100 + Z``, iron is 5626) is ``Z`` protons and
        ``A - Z`` neutrons at :math:`E/A`; other particles are given by PDG ID.
        The energy per nucleon must lie on the energy grid
        (:func:`MCEq.initial_state.deposit_single_primary`).

        Args:
          E (float): kinetic energy of a nucleus in GeV
          corsika_id (int): CORSIKA ID of a nucleus
          pdg_id (int): PDG ID of a particle
          append (bool): If True, keep previous state and append a new particle.
        """
        info(2, f"CORSIKA ID {corsika_id}, PDG ID {pdg_id}, energy {E:5.3g} GeV")
        self._set_initial(("single", E, corsika_id, pdg_id), append)

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user spectrum of one species as initial condition.

        Args:
          spectrum (np.array): differential spectrum on the kinetic-energy grid
          pdg_id (int): PDG ID of the particle
          append (bool): If True, add to the previous initial condition.
        """
        info(2, f"PDG ID {pdg_id}")
        self._set_initial(("spectrum", spectrum, pdg_id), append)

    def get_initial_state(self):
        """Copy of the initial-condition vector ``phi0`` that :meth:`solve` propagates."""
        return self._phi0.copy()

    def set_density_model(self, density_model_or_config):
        """Set the atmosphere or target and recompute its integration path.

        ``density_model_or_config`` is a ``(model, config)`` tuple, e.g.
        ``("CORSIKA", ("PL_SouthPole", "January"))`` (choices in
        :data:`DENSITY_MODELS`), or an instance of
        :class:`~MCEq.geometry.density_profiles.EarthsAtmosphere` or
        :class:`~MCEq.geometry.target.GeneralizedTarget`.
        """
        if not isinstance(
            density_model_or_config, (dp.EarthsAtmosphere, dp.GeneralizedTarget)
        ):
            base_model, model_config = density_model_or_config
            if base_model not in DENSITY_MODELS:
                info(
                    0,
                    "Unknown density model. Available choices are:\n",
                    "\n".join(DENSITY_MODELS),
                )
                raise ValueError("Choose a different profile.")
            info(1, "Setting density profile to", base_model, model_config)
            if base_model == "GeneralizedTarget":
                self.density_model = dp.GeneralizedTarget()
            else:
                self.density_model = DENSITY_MODELS[base_model](*model_config)
        else:
            self.density_model = density_model_or_config

        self.integration_path = None  # the angle cache cannot see a model switch
        if self.theta_deg is not None and isinstance(
            self.density_model, dp.EarthsAtmosphere
        ):
            self.set_zenith_azimuth(self.theta_deg)
        elif not isinstance(self.density_model, dp.GeneralizedTarget):
            raise ValueError(f"Density model {self.density_model} not supported.")

    def set_zenith_azimuth(self, zenith_deg, azimuth_deg=None):
        """Set the zenith and, optionally, the azimuth of the shower.

        Zenith 0° is vertically downgoing and 90° horizontal; angles above 90°
        are upgoing and need a density model with ``max_theta = 180`` (the
        location-centred MSIS models). Azimuth 0° is geographic North, 90°
        East. With ``azimuth_deg=None`` azimuth-aware atmospheres are averaged
        over azimuth; other models ignore the azimuth.

        Args:
            zenith_deg (float): Zenith angle at the detector in degrees.
            azimuth_deg (float, optional): Azimuth angle in degrees.
        """
        info(
            2,
            f"Zenith angle {zenith_deg:6.2f}"
            + (f", azimuth {azimuth_deg:6.2f}" if azimuth_deg is not None else ""),
        )
        if isinstance(self.density_model, dp.GeneralizedTarget):
            raise Exception("GeneralizedTarget does not support angles.")

        cached_theta = self.density_model.theta_deg
        cached_azi = getattr(self.density_model, "_current_azimuth_deg", None)
        if cached_theta == zenith_deg and cached_azi == azimuth_deg:
            info(2, "Angle selection corresponds to cached value, skipping calc.")
            self.theta_deg = zenith_deg
            return

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
            Use :meth:`set_zenith_azimuth` instead.

        Args:
          theta_deg (float): zenith angle in degrees
        """
        warnings.warn(
            "set_theta_deg() is deprecated; use set_zenith_azimuth() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_zenith_azimuth(theta_deg)

    def inject_ddm(self, ddm):
        """Install the matrices of the :class:`MCEq.ddm.DataDrivenModel` ``ddm``.

        They become channel overrides of the interaction tables, which survive
        same-model reloads and ``regenerate_matrices``; selecting a different
        interaction model clears them.
        """
        for (prim, sec), mati in ddm.ddm_matrices(self).items():
            info(5, f"Injecting DDM {prim} --> {sec}")
            self._interactions.set_channel_override(prim, sec, mati)
        self.regenerate_matrices()

    def set_mod_pprod(self, prim_pdg, sec_pdg, x_func, x_func_args, delay_init=False):
        """Modify the production of ``sec_pdg`` in ``prim_pdg`` interactions by
        ``x_func(x, E, *x_func_args)`` (:meth:`MCEq.data.Interactions._gen_mod_matrix`).

        Returns 1 if the modification is new; call :meth:`regenerate_matrices`
        after adding all modifications. ``delay_init`` is unused.
        """
        info(1, f"{prim_pdg}/{sec_pdg}, {sec_pdg}, {x_func.__name__}, {x_func_args!s}")
        return int(
            self._interactions._set_mod_pprod(prim_pdg, sec_pdg, x_func, x_func_args)
        )

    def unset_mod_pprod(self, dont_fill=False):
        """Remove the :meth:`set_mod_pprod` modifications; ``dont_fill`` defers
        :meth:`regenerate_matrices` to the caller."""
        info(1, "Particle production modifications reset to defaults.")
        self._interactions.mod_pprod = defaultdict(dict)
        if not dont_fill:
            self.regenerate_matrices()

    def regenerate_matrices(self, skip_decay_matrix=False):
        """Rebuild the matrices, e.g. after production modifications or a DDM."""
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
        dtype=None,
    ):
        """Propagate the initial state with the kernel ``config.kernel_config``.

        Args:
          int_grid (list): depths X at which the solution is recorded
          grid_var (str): only the depth ``X`` is supported
          X_start (float | None): starting depth in g/cm²; ``None`` →
            ``config.X_start``
          eps (float | None): tolerated ``rho_inv`` variation within a step;
            ``None`` → ``config.etd2_path["eps"]``, ``None`` there a default
            for the system
          dX_max (float | None): step cap; ``None`` → ``config.etd2_path["dX_max"]``,
            ``None`` there a default for the system
          dX_min (float | None): step floor; ``None`` → ``config.etd2_path["dX_min"]``
          fd_span (float | None): span of the finite difference that estimates
            the local rate; ``None`` → ``config.etd2_path["fd_span"]``
          dtype (np.float32 | np.float64 | None): state precision; ``None`` is
            float32 on CUDA with ``config.cuda_fp_precision = 32``, else
            float64. The solution is float64 either way.
        """
        if int_grid is not None and np.any(np.diff(int_grid) < 0):
            raise Exception(
                "The X values in int_grid are required to be strictly", "increasing."
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
        self.solve_from_integration_path(*self.integration_path, dtype=dtype)

    def solve_from_integration_path(
        self, nsteps, dX, rho_inv, grid_idcs, *, dtype=None
    ):
        """Propagate the initial state along a given path, e.g. a stored
        ``self.integration_path`` (``nsteps, dX, rho_inv, grid_idcs``)."""
        info(2, f"Launching {config.kernel_config} solver for {nsteps} steps.")
        start = time()
        dtype = resolve_dtype(dtype)
        sec_ops = self._resolve_secant()
        self._warn_low_energy_cuda_fp32(dtype, sec_ops)
        if secant_table.table_for(self) is not None:
            X_start = getattr(self, "_cached_etd2_path_params", (config.X_start,))[0]
            sol, grid_sol = secant_table.solve_segments(
                self, nsteps, dX, rho_inv, grid_idcs, X_start, dtype
            )
        else:
            sol, grid_sol = self._run_etd2(
                nsteps,
                dX,
                rho_inv,
                self._stack_modes(self._phi0),
                grid_idcs,
                dtype=dtype,
                sec_ops=sec_ops,
            )
        self._solution = np.asarray(sol, dtype=np.float64)
        self.grid_sol = np.asarray(grid_sol, dtype=np.float64)
        info(2, f"time elapsed during integration: {time() - start:5.2f}sec")

    def _stack_modes(self, phi0):
        """``phi0`` repeated over the Legendre modes of a 2D run (a delta in
        angle has equal amplitude in every mode, P_l(1) = 1)."""
        if not self._mceq_db.is_2d:
            return np.copy(phi0)
        reps = (self._mceq_db.n_k,) + (1,) * (np.ndim(phi0) - 1)
        return np.ascontiguousarray(np.tile(phi0, reps))

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
        """Build the ETD2 path on ``integration_path`` unless the cached one
        matches; ``dX_max`` is capped by the continuous-loss and EM stiffness."""
        p = config.etd2_path
        X_start = config.X_start if X_start is None else X_start
        eps = p["eps"] if eps is None else eps
        dX_max = p["dX_max"] if dX_max is None else dX_max
        if eps is None or dX_max is None:
            conservative = self._mceq_db.is_2d or any(
                abs(particle.pdg_id[0]) == 11 for particle in self.pman.all_particles
            )
            if eps is None:
                eps = 0.01 if conservative else 0.03
            if dX_max is None:
                dX_max = 2.0 if conservative else 5.0
        dX_min = p["dX_min"] if dX_min is None else dX_min
        fd_span = p["fd_span"] if fd_span is None else fd_span
        cap = min(
            em_cascade_dx_cap(self._em_cascade_step_scale()),
            self._continuous_loss_dx_cap(),
        )
        dX_max = min(dX_max, cap)
        # A user floor must never undo the stiffness ceiling.
        dX_min = min(dX_min, dX_max)
        etd2_params = (X_start, eps, dX_max, dX_min, fd_span)

        if (
            self.integration_path
            and np.array_equal(int_grid, self.int_grid)
            and self.grid_var == grid_var
            and getattr(self, "_cached_etd2_path_params", None) == etd2_params
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
        info(
            2,
            f"ETD2 non-uniform path (eps={eps}, dX_max={dX_max}, dX_min={dX_min}, "
            f"fd_span={fd_span}, X_start={X_start})",
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

    def _continuous_loss_dx_cap(self):
        """Step cap ``LOSS_STEP_SAFETY / rate`` of the continuous-loss bands
        (scaled by sec(theta) with secant transport), cached per matrix; inf
        before the matrices exist."""
        if getattr(self, "int_m", None) is None:
            return np.inf  # atmosphere-only path planning before assembly
        builder = self.matrix_builder
        sec = self._resolve_secant()
        cached = getattr(self, "_loss_step_cache", None)
        if cached is None or cached[0] is not self.int_m or cached[1] is not sec:
            bands = builder._contloss_bands
            rate = continuous_loss_rate(
                bands.values(),
                None if sec is None else sec["lam"],
                [builder._contloss_upwind_rows[key] for key in bands],
            )
            cap = LOSS_STEP_SAFETY / rate if rate else np.inf
            self._loss_step_cache = (self.int_m, sec, cap)
        return self._loss_step_cache[2]

    def _em_cascade_step_scale(self):
        """Spectral radius of the EM block of ``int_m`` (cached per matrix);
        0.0 before the matrices exist or without e+/e-/gamma."""
        int_m = getattr(self, "int_m", None)
        if int_m is None:
            return 0.0
        cache = getattr(self, "_em_step_scale_cache", None)
        if cache is not None and cache[0] is int_m:
            return cache[1]
        r_em = em_step_scale(
            int_m, self.pman.all_particles, dense_eig_max=config.em_step_dense_eig_max
        )
        self._em_step_scale_cache = (int_m, r_em)
        return r_em

    def get_solution(
        self,
        particle_name,
        mag=0.0,
        grid_idx=None,
        integrate=False,
        return_as=None,
        dont_sum_helicities=False,
    ):
        """Spectrum of a particle on the energy grid :attr:`e_grid`.

        Lepton names take the prefixes ``total_`` (all origins, same as no
        prefix), ``conv_`` (conventional), ``pr_`` (prompt), and the origin
        tags of the tracking particles, e.g. ``pi_`` and ``k_`` for pion and
        kaon decays.

        Args:
          particle_name (str): e.g. ``total_mu+`` or ``pr_antinumu``
          mag (float, optional): the flux is multiplied by :math:`E^{mag}`
          grid_idx (int, optional): index of the depth in ``int_grid``; the
            default is the final depth
          integrate (bool, optional): particle number per bin instead of flux
          return_as (str, optional): ``total energy``, ``kinetic energy`` or
            ``total momentum``; ``None`` is ``config.return_as``
          dont_sum_helicities (bool, optional): leptons are summed over the
            helicities unless set, e.g. ``total_mu+`` sums (-1, 0, +1)

        Returns:
          (numpy.array): flux of particles on energy grid :attr:`e_grid`
        """
        if grid_idx is not None and len(self.grid_sol) == 0:
            raise Exception("Solution not has not been computed on grid. Check input.")
        if grid_idx is None:
            sol = np.copy(self._solution)
        else:
            sol = self.grid_sol[min(grid_idx, len(self.grid_sol) - 1), :]
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
        """:meth:`get_solution` applied to the state vector ``sol``."""
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
                    info(15, f"No particle {prefix + ls}.")
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
        if not config.enable_default_tracking and particle_name.startswith(
            tuple(default_tracking_prefixes)
        ):
            raise Exception(
                "Tracking category requested but "
                + "enable_default_tracking is off in config."
            )

        if particle_name.startswith("total_"):
            res = sum_lr(lep_str, prefix="")
        elif particle_name.startswith("conv_"):
            # conventional = total - prompt
            res = self._get_solution_from_state(
                sol, "total_" + lep_str, return_as="kinetic energy"
            ) - self._get_solution_from_state(
                sol, "pr_" + lep_str, return_as="kinetic energy"
            )
        elif particle_name.startswith("pr_"):
            for prefix in ("prcas_", "prres_", "em_"):
                if prefix + lep_str in ref:
                    res += sum_lr(lep_str, prefix=prefix)
        else:
            try:
                res = sum_lr(particle_name, prefix="")
            except KeyError:
                if config.excpt_on_missing_particle:
                    raise Exception(f"Requested particle {particle_name} not found.")
                info(1, f"Requested particle {particle_name} not found.")

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

    def convert_to_theta_space(
        self, modes, pdg_id, hel, theta_res=1200, log_theta=False
    ):
        """Angular densities of one species from the Legendre modes of a 2D solve.

        The moments are interpolated in ``l`` between the sparse modes and
        summed with Jackson damping (:func:`MCEq.operators.legendre.theta_density`).

        Args:
            modes (list of np.arrays): mode amplitudes at the requested depths,
                each reshaped to ``(n_k, dim_states)``
            pdg_id (int): PDG ID of the particle
            hel (int): helicity of the particle
            theta_res (int): number of points of the theta grid
            log_theta (bool): logarithmic instead of linear theta grid

        Returns:
            tuple: ``(l_grid, amps, theta_range, f_theta)``; ``amps[depth][eidx]``
            are the moments on ``l_grid``, ``f_theta[depth][eidx]`` the density
            in the measure ``theta dtheta``, normalised to
            ``int_0^pi f theta dtheta = a_0``.
        """
        if log_theta:
            theta_range = np.logspace(-5, np.log10(np.pi / 2), theta_res)
        else:
            theta_range = np.linspace(0, np.pi / 2, theta_res)
        l_grid = np.asarray(self._mceq_db.k_grid)
        n_e = len(self.e_grid)
        ref = self.pman.pdg2mceqidx[(pdg_id, hel)] * n_e
        amps = [list(s[:, ref : ref + n_e].T) for s in modes]
        f = [list(theta_density(np.asarray(a), l_grid, theta_range)) for a in amps]
        return l_grid, amps, theta_range, f

    def n_particles(self, label, grid_idx=None, min_energy_cutoff=1e-1):
        """Number of particles ``label`` above ``min_energy_cutoff`` (GeV) at
        depth index ``grid_idx``."""
        ie_min = np.argmin(
            np.abs(self.e_bins - self.e_bins[self.e_bins >= min_energy_cutoff][0])
        )
        info(10, f"Energy cutoff for particle number {self.e_bins[ie_min]:4.3e} GeV")
        return np.sum(
            self.get_solution(label, mag=0, integrate=True, grid_idx=grid_idx)[ie_min:]
        )

    def n_mu(self, grid_idx=None, min_energy_cutoff=1e-1):
        """Number of muons of both charges above ``min_energy_cutoff`` (GeV)."""
        return self.n_particles(
            "total_mu+", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        ) + self.n_particles(
            "total_mu-", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
        )

    def n_e(self, grid_idx=None, min_energy_cutoff=1e-1):
        """Number of electrons and positrons above ``min_energy_cutoff`` (GeV)."""
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
        r"""Energy-dependent Z-factor of ``projectile -> secondary`` interactions.

        ``definition="thunman"`` (any value but ``"primary_e"``) is the moment
        of Thunman, Ingelman and Gondolo (1996) at the secondary energy,

        .. math::

            Z(E) = \int_E^\infty dE'\,\frac{\phi_N(E')}{\phi_N(E)}
                \frac{\lambda_N(E)}{\lambda_N(E')}\frac{dn(E', E)}{dE},

        as a matrix row :math:`Z_i = \sum_j C_{ij}\,\sigma_j\phi_j /
        (\sigma_i\phi_i)`, with :math:`\phi` the proton (for a neutron
        projectile the neutron) flux of the primary model and :math:`\sigma`
        the production cross section, which ``use_cs_scaling=False`` drops.
        Under Feynman scaling and a power law :math:`E^{-(\gamma+1)}` it is
        :math:`\int_0^1 x^{\gamma-1}F(x)dx`.

        ``definition="primary_e"`` is the column at the projectile energy
        :math:`E_j`, weighted by :math:`\phi_N(E_j)/\phi_N(E_i)` over the
        secondaries above ``min_energy`` (:meth:`MCEqParticle.parent_column`
        for stencil-averaged tables). Values below ``min_energy`` are 0.

        Returns:
          (numpy.ndarray): Z on ``e_grid``
        """
        proj = self.pman[projectile_pdg]
        sec = self.pman[secondary_pdg]
        if not proj.is_projectile:
            raise Exception(f"{proj.name} is not a projectile particle.")
        info(10, f"Computing e-dependent Zfactor for {proj.name} -> {sec.name}")
        if sec not in proj.hadr_yields:
            raise Exception(f"{sec.name} is not a secondary particle of {proj.name}.")

        p_flux, n_flux = self.pmodel.p_and_n_flux(self.e_grid)[1:]
        nuc_flux = np.asarray(
            n_flux if proj.pdg_id[0] == 2112 else p_flux, dtype=np.float64
        )
        smat = np.asarray(proj.hadr_yields[sec], dtype=np.float64)
        proj_cs = np.asarray(proj.prod_cross_section(), dtype=np.float64)
        e = np.asarray(self.e_grid, dtype=np.float64)
        zfac = np.zeros_like(e)

        if definition == "primary_e":
            below = np.flatnonzero(e < min_energy)
            min_idx = int(below[-1]) if len(below) else 0
            for p_eidx in np.flatnonzero(e >= min_energy):
                col = proj.parent_column(smat, p_eidx)
                nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
                zfac[p_eidx] = np.sum(col[min_idx : p_eidx + 1] * nuc_fac)
            return zfac

        # Thunman et al. 1996: a row of the table against the projectile flux
        weight = nuc_flux * proj_cs if use_cs_scaling else nuc_flux.copy()
        source = smat @ weight
        ok = (e >= min_energy) & (weight > 0)
        zfac[ok] = source[ok] / weight[ok]
        return zfac

    def decay_z_factor(self, parent_pdg, child_pdg, gamma=None):
        r"""Energy-dependent decay Z-factor (Lipari 1993) at the daughter energy.

        The matrix row :math:`Z_i = \sum_j D_{ij} S_j / S_i` against a parent
        spectrum :math:`S(E) \propto E^{\gamma}`; ``gamma`` defaults to the
        nucleon index of the primary model at the daughter energy. Under scaling
        it is :math:`\int_0^1 x^{-\gamma-1} dN/dx\,dx`.

        Returns:
          (numpy.ndarray): Z on ``e_grid``
        """
        proj = self.pman[parent_pdg]
        sec = self.pman[child_pdg]
        if proj.is_stable:
            raise Exception(f"{proj.name} does not decay.")
        info(10, f"Computing e-dependent decay Zfactor for {proj.name} -> {sec.name}")
        if sec not in proj.decay_dists:
            raise Exception(f"{sec.name} is not a a child particle of {proj.name}.")

        e = np.asarray(self.e_grid, dtype=np.float64)
        if gamma is None:
            gamma = self.pmodel.nucleon_gamma(e)
        gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), e.shape)
        dmat = np.asarray(proj.decay_dists[sec], dtype=np.float64)
        zfac = np.zeros_like(e)
        for i in range(len(e)):
            zfac[i] = dmat[i] @ ((e / e[i]) ** gamma[i])
        return zfac

    def _resolve_secant(self):
        """sec(theta) operators for the solve, or None if secant mode is off."""
        if not config.secant_enabled(self._mceq_db.is_2d):
            return None
        return self._build_secant_ops()

    def _build_secant_ops(self):
        """sec(theta) operators, cached per grid, species count and secant parameters."""
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
        ``int_m``, ``dec_m`` and ``sec_ops``.

        On 2D runs with ``config.legendre_backward_removal`` the hadronic part
        and the loss band are compiled separately, so that the removal of
        backward production acts on produced particles only.
        """
        cache = self.__dict__.setdefault("_operator_cache", {})
        coupled = sec_ops is not None
        entry = cache.get(coupled)
        if (
            entry is None
            or entry[0] is not self.int_m
            or entry[1] is not self.dec_m
            or entry[2] is not sec_ops
        ):
            if self._mceq_db.is_2d and config.legendre_backward_removal:
                n_d = dense_count(self._mceq_db.k_grid)
                mix = SimpleNamespace(
                    M=backward_removal(n_d - 1), n_D=n_d, N=self.dim_states
                )
                op = compile_operator(
                    self.matrix_builder.int_m_hadr,
                    self.dec_m,
                    sec_ops,
                    band=self.matrix_builder.dEdx_band,
                    mode_mix=mix,
                )
            else:
                op = compile_operator(self.int_m, self.dec_m, sec_ops)
            cache[coupled] = entry = (self.int_m, self.dec_m, sec_ops, op)
        return entry[3]

    def _etd2_backend(self, sec_ops=None, dtype=np.float64):
        """Solver backend of ``kernel_config`` for the compiled operator, cached
        per kernel, precision and coupling; rebuilt when the operator or the
        CUDA device changes."""
        kernel = config.kernel_config.lower()
        on_cuda = detect.ALIASES.get(kernel, kernel) == "cuda_etd2"
        op = self._compiled_operator(sec_ops)
        op_dtype = None
        if on_cuda and config.cuda_operator_precision is not None:
            op_dtype = {32: np.float32, 64: np.float64}[
                int(config.cuda_operator_precision)
            ]
        key = (kernel, np.dtype(dtype), op.coupled, op_dtype)
        cache = self.__dict__.setdefault("_backend_cache", {})
        be = cache.get(key)
        if be is not None and (
            be.op is not op or (on_cuda and be.device_id != self._cuda_device)
        ):
            be.close()
            del cache[key]
            be = None
        if be is None:
            be = cache[key] = solvers.make_backend(
                op, kernel, dtype, self._cuda_device, op_dtype
            )
        return be

    def _run_etd2(
        self, nsteps, dX, rho_inv, phi0, grid_idcs, dtype=np.float64, sec_ops=None
    ):
        """Integrate a ``(dim,)`` or ``(dim, K)`` state along one path."""
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
        """Release backend resources (MKL handles, Accelerate slots, GPU
        buffers); the next solve rebuilds them."""
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
