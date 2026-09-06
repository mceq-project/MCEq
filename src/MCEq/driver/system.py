"""The cascade system: database handle, yield/decay tables, particle manager.

``CascadeSystem`` is the object the :class:`MCEqRun` facade delegates its
whole "what is the physics system" question to: the HDF5 backend, the four
table interfaces, the cross-section set, the particle manager, the matrix
builder, and the two assembled matrices ``int_m`` / ``dec_m``. Everything
physics-side lives here; the solution vectors, the initial condition and its
restore-replay, geometry and solve orchestration stay on the facade. The
bodies here are the originals from ``MCEq/core.py``, moved verbatim in
Phase 6 commit 2 (plan §16); the two splits (skip-check and resize/rebuild
tail of the model-loading ladder) are documented on
:meth:`reload_for_model`.

No reference from here back to the facade, on purpose: the
restore-replay list stores facade method *names*, not bound methods (PR
#163 — bound methods pinned whole MCEqRun instances and overflowed the
macOS Accelerate sparse-matrix store), so the resize cannot be replayed
from here. The facade drives load -> resize -> build through
:meth:`reload_for_model` and :meth:`build_matrices`.
"""

import MCEq.data
from MCEq import config
from MCEq.data.model_names import normalize_hadronic_model_name
from MCEq.misc import info
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.species.manager import ParticleManager


class CascadeSystem:
    """Database handle, loaded tables, particle manager, matrix builder.

    Built by :class:`MCEqRun`. The ctor runs the original constructor
    segment verbatim: database, the four table interfaces (cross-sections
    constructed with the normalized model name but not yet loaded), the
    empty particle-manager slots, the energy grid taken from the backend.
    Loading the model and building the pman happens in the first
    :meth:`reload_for_model`, driven by the facade's
    ``set_interaction_model`` exactly as the single-class original did.

    Args:
      interaction_model (str): normalized model name (or falsy to defer).
      medium (str): interaction medium (air/water).
      low_energy_model, he_le_transition, he_le_trwidth: resolved values of
        the facade's low-energy-extension kwargs.
    """

    def __init__(
        self,
        interaction_model,
        medium,
        low_energy_model,
        he_le_transition,
        he_le_trwidth,
    ):
        self.medium = medium
        # Group views, not snapshots: a config write after construction is seen.
        self._mceq_db = MCEq.data.HDF5Backend(
            medium=medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
            paths=config.paths,
            grid=config.grid,
            physics=config.physics,
            em=config.em,
        )

        interaction_model = normalize_hadronic_model_name(interaction_model)

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

        The original ``set_interaction_model`` branch ladder, verbatim from
        the normalize-and-load point down. Three splits moved to the facade
        wrapper: ``info(1)`` + the skip short-circuit (before the load) and
        the resize/rebuild tail (after it) — the resize replays the facade's
        initial-condition methods by name (PR #163), so only the facade can
        drive the order. ``force`` is likewise decided by the wrapper.

        Args:
          interaction_model (str): normalized model name
          particle_list (list | None): parent list to load against
          update_particle_list (bool): re-derive the parent list or reuse
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

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
            self.matrix_builder = MatrixBuilder(
                self.pman,
                self._mceq_db,
                grid=config.grid,
                losses=config.losses,
                physics=config.physics,
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

        The original tail of ``set_interaction_model``, verbatim, as its own
        step so the facade can slot the resize between load and build.
        """
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=skip_decay_matrix
        )

    def regenerate(self, skip_decay_matrix=False):
        """Force a yields reload and rebuild, for production modifications.

        The original ``MCEqRun.regenerate_matrices`` first line, verbatim;
        the resize stays in the facade wrapper (module docstring) and the
        build is the shared :meth:`build_matrices`.
        """
        self.pman.set_interaction_model(self._int_cs, self._interactions, force=True)
