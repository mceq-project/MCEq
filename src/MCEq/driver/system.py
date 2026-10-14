"""Database handle, data tables, particle manager and matrix builder of a run."""

import MCEq.data
from MCEq import config
from MCEq.misc import info
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.particlemanager import ParticleManager


class CascadeSystem:
    """Tables, particle manager and matrices for one interaction model.

    The tables are loaded and the particle manager is built by
    :meth:`reload_for_model`.

    Args:
      interaction_model (str): normalized model name.
      medium (str): interaction medium (air/water).
      low_energy_model, he_le_transition, he_le_trwidth: low-energy
        extension settings.
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
        if config.enable_em and config.muon_helicity_dependence:
            info(1, "enable_em: helicity-dependent muon decays are switched off.")
        self._mceq_db = MCEq.data.HDF5Backend(
            medium=medium,
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
        )

        with self._mceq_db._read_session():
            #: Interface to interaction tables of the HDF5 database
            self._interactions = MCEq.data.Interactions(mceq_hdf_db=self._mceq_db)

            #: Cross-section data managed by :class:`MCEq.data.InteractionCrossSections`
            self._int_cs = MCEq.data.InteractionCrossSections(
                mceq_hdf_db=self._mceq_db, interaction_model=interaction_model
            )

            #: Continuous-energy-loss data managed by :class:`MCEq.data.ContinuousLosses`
            self._cont_losses = MCEq.data.ContinuousLosses(mceq_hdf_db=self._mceq_db)

            #: Interface to decay tables of the HDF5 database
            self._decays = MCEq.data.Decays(mceq_hdf_db=self._mceq_db)

        #: Particle manager, built by reload_for_model
        self.pman = None
        # Parent list requested by the caller (None: all particles of the model)
        self.parent_list = None

        # General Matrix dimensions and shortcuts, controlled by
        # grid of yield matrices
        self._energy_grid = self._mceq_db.energy_grid

        # Initialize matrix builder (initialized in reload_for_model)
        self.matrix_builder = None

    def reload_for_model(self, interaction_model, particle_list, update_particle_list):
        """Load ``interaction_model``; a new particle list gets a new particle manager.

        Args:
          interaction_model (str): normalized model name
          particle_list (list | None): parent list to load against
          update_particle_list (bool): re-derive the species from
            ``particle_list``; False keeps the current species
        """
        self._int_cs.load(interaction_model)
        if self.pman is not None and not update_particle_list:
            info(10, "Re-using particle list.")
            self._interactions.load(
                interaction_model, parent_list=self.pman.pdg_id_list
            )
            self._wire_particles()
            return

        self.parent_list = particle_list
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
        self._wire_particles()
        self.matrix_builder = MatrixBuilder(self.pman, self._mceq_db)

    def _wire_particles(self):
        """Give the particles their cross sections, channels and losses."""
        self.pman.set_interaction_model(self._int_cs, self._interactions, force=True)
        self.pman.set_decay_channels(self._decays)
        self.pman.set_continuous_losses(self._cont_losses)

    def build_matrices(self, skip_decay_matrix=False):
        """Assemble ``int_m``/``dec_m`` from the current particle manager."""
        self.int_m, self.dec_m = self.matrix_builder.construct_matrices(
            skip_decay_matrix=skip_decay_matrix
        )

    def regenerate(self):
        """Reconnect particle channels to the current interaction tables."""
        self.pman.set_interaction_model(self._int_cs, self._interactions, force=True)
