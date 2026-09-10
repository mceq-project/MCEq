"""The cascade system: database handle, yield/decay tables, particle manager.

``CascadeSystem`` owns the physics-side state used by :class:`MCEqRun`: the
HDF5 backend, the four table interfaces, the cross-section set, the particle
manager, the matrix builder, and the two assembled matrices ``int_m`` /
``dec_m``. The solution vectors, initial condition and its restore-replay,
geometry, and solve orchestration stay on ``MCEqRun``, which controls the
load, resize, and build sequence. The split of model loading across the
skip-check and the resize/rebuild tail is documented on
:meth:`reload_for_model`.

This module never refers back to ``MCEqRun``: the restore-replay list stores
method *names*, not bound methods, so the resize cannot be replayed from
here. ``MCEqRun`` drives load -> resize -> build through
:meth:`reload_for_model` and :meth:`build_matrices`.
"""

import MCEq.data
from MCEq import config
from MCEq.data.model_names import normalize_hadronic_model_name
from MCEq.misc import info
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.species.manager import ParticleManager


def run_physics_view(cfg=None):
    """Return the run's physics settings, applying the EM compatibility rule.

    ``enable_em`` and muon-helicity rows are incompatible: the L/R semi-
    Lagrangian muon rows have no diagonal damping and destabilize the EM
    cascade (EM-blowup caveat, ``MCEq.solvers.path``). The rule is enforced
    here — at the single point where the driver hands a physics group to a
    run — as a validator: warn, and return a read-only copy of the view with
    muon helicity dependence disabled. The flat
    ``muon_helicity_dependence`` default is never
    written, so a later EM-off run in the same process is unaffected; every
    other reader of ``config.physics`` sees the setting exactly as the user
    set it. Lives in the driver rather than in ``config`` because the
    warning needs ``misc.info`` and util sits below config (C1).
    """
    physics = config.physics if cfg is None else cfg.physics
    if physics.enable_em and physics.muon_helicity_dependence:
        info(
            1,
            "enable_em: forcing muon_helicity_dependence=False "
            "(helicity rows destabilize the EM cascade).",
        )
        return physics.with_overrides(muon_helicity_dependence=False)
    return physics


class CascadeSystem:
    """Database handle, loaded tables, particle manager, matrix builder.

    Built by :class:`MCEqRun`. The constructor creates the database handle,
    the four table interfaces (cross-sections constructed with the supplied
    model name but not yet loaded), the empty particle-manager slots, and
    takes the energy grid from the backend. Loading the model and building
    the particle manager happen in the first :meth:`reload_for_model`,
    driven by :meth:`MCEqRun.set_interaction_model`.

    Args:
      interaction_model (str): model-name string; the tables are loaded
        later by :meth:`reload_for_model`.
      medium (str): interaction medium (air/water).
      low_energy_model, he_le_transition, he_le_trwidth: resolved values of
        the low-energy-extension arguments.
    """

    def __init__(
        self,
        interaction_model,
        medium,
        low_energy_model,
        he_le_transition,
        he_le_trwidth,
        cfg=None,
    ):
        self._cfg = config if cfg is None else cfg
        self.medium = medium
        # Use the supplied live configuration or snapshot. The physics view
        # goes through ``run_physics_view``: for an EM run it is a read-only
        # copy with muon_helicity_dependence disabled, and this one instance
        # is what every table and the particle manager read.
        self._physics = run_physics_view(cfg)
        self._mceq_db = MCEq.data.HDF5Backend(
            medium=medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
            paths=self._cfg.paths,
            grid=self._cfg.grid,
            physics=self._physics,
            em=self._cfg.em,
        )

        interaction_model = normalize_hadronic_model_name(interaction_model)

        #: Interface to interaction tables of the HDF5 database
        self._interactions = MCEq.data.Interactions(
            mceq_hdf_db=self._mceq_db, physics=self._physics
        )

        #: Cross-section data managed by :class:`MCEq.data.InteractionCrossSections`
        self._int_cs = MCEq.data.InteractionCrossSections(
            mceq_hdf_db=self._mceq_db, interaction_model=interaction_model
        )

        #: Continuous-energy-loss data managed by :class:`MCEq.data.ContinuousLosses`
        self._cont_losses = MCEq.data.ContinuousLosses(
            mceq_hdf_db=self._mceq_db, physics=self._physics
        )

        #: Interface to decay tables of the HDF5 database
        self._decays = MCEq.data.Decays(
            mceq_hdf_db=self._mceq_db, physics=self._physics
        )

        #: Particle manager (initialized/updated in reload_for_model)
        self.pman = None

        # Particle list to keep track of previously initialized particles
        self._particle_list = None

        # General Matrix dimensions and shortcuts, controlled by
        # grid of yield matrices
        self._energy_grid = self._mceq_db.energy_grid

        # Initialize matrix builder (initialized in reload_for_model)
        self.matrix_builder = None

    def reload_for_model(self, interaction_model, particle_list, update_particle_list):
        """Load tables and (re)build the pman/matrix_builder for one model.

        Handles the model-loading branches from the normalize-and-load point
        down. ``MCEqRun.set_interaction_model`` decides whether loading is
        required, then coordinates loading, state resizing, and matrix
        construction; the resize replays the initial-condition methods by
        name, so only ``MCEqRun`` can drive the order.

        Args:
          interaction_model (str): normalized model name
          particle_list (list | None): parent list to load against
          update_particle_list (bool): re-derive the parent list or reuse
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

        self._int_cs.load(interaction_model)

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
                physics=self._physics,
            )
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)
            self.matrix_builder = MatrixBuilder(
                self.pman,
                self._mceq_db,
                grid=self._cfg.grid,
                losses=self._cfg.losses,
                physics=self._physics,
            )

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

    def build_matrices(self, skip_decay_matrix=False):
        """Assemble ``int_m``/``dec_m`` from the current particle manager.

        A separate step so ``MCEqRun`` can update its state layout between
        loading and matrix construction.
        """
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=skip_decay_matrix
        )

    def regenerate(self, skip_decay_matrix=False):
        """Reconnect particle channels to the current interaction tables.

        Used after production modifications. The caller resizes the state
        layout and rebuilds matrices separately through :meth:`build_matrices`.
        """
        self.pman.set_interaction_model(self._int_cs, self._interactions, force=True)
