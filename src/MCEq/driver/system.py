"""Database handle, data tables, particle manager and matrix builder of a run."""

import MCEq.data
from MCEq import config
from MCEq.misc import info, strip_em_species
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.species.manager import ParticleManager


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

        #: Particle manager (initialized/updated in reload_for_model)
        self.pman = None

        # Particle list to keep track of previously initialized particles
        self._particle_list = None
        # Parent list requested by the caller (None: all particles of the model)
        self.parent_list = None

        # General Matrix dimensions and shortcuts, controlled by
        # grid of yield matrices
        self._energy_grid = self._mceq_db.energy_grid

        # Initialize matrix builder (initialized in reload_for_model)
        self.matrix_builder = None

    def reload_for_model(
        self, interaction_model, particle_list, update_particle_list
    ):  # anfe: this function exposes a weakness in the design and is awkward, has to take care of special cases, and shows that ownership is not correctly maintained. Why should my controller own and push data such as decays...?
        """Load ``interaction_model`` and rewire the particle manager.

        Args:
          interaction_model (str): normalized model name
          particle_list (list | None): parent list to load against
          update_particle_list (bool): re-derive the parent list or reuse
        """
        self._int_cs.load(interaction_model)

        if not update_particle_list and self._particle_list is not None:
            info(10, "Re-using particle list.")
            self._interactions.load(interaction_model, parent_list=self._particle_list)
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)

        elif self._particle_list is None:
            info(10, "New initialization of particle list.")
            self.parent_list = particle_list
            if particle_list is None:
                self._interactions.load(interaction_model)
            else:
                self._interactions.load(interaction_model, parent_list=particle_list)

            self._decays.load(parent_list=self._interactions.particles)
            self._particle_list = strip_em_species(
                self._interactions.particles + self._decays.particles
            )
            # Create particle database
            self.pman = ParticleManager(
                self._particle_list,
                self._energy_grid,
                self._int_cs,
                self.medium,
            )
            self.pman.set_interaction_model(self._int_cs, self._interactions)
            self.pman.set_decay_channels(self._decays)
            self.pman.set_continuous_losses(self._cont_losses)
            self.matrix_builder = MatrixBuilder(self.pman, self._mceq_db)

        else:
            info(10, "Updating particle list.")
            self.parent_list = particle_list
            if particle_list is None:
                self._interactions.load(interaction_model)
            else:
                self._interactions.load(interaction_model, parent_list=particle_list)
            self._decays.load(parent_list=self._interactions.particles)
            self._particle_list = strip_em_species(
                self._interactions.particles + self._decays.particles
            )
            self.pman.set_interaction_model(
                self._int_cs,
                self._interactions,
                updated_parent_list=self._particle_list,
            )
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
