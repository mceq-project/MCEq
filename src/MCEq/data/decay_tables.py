"""Decay yield matrices: the ``Decays`` consumer.

Plan section 1's ``data/decay_tables.py``, §13.2 item 4. The class moved here
verbatim from ``data/__init__.py``, which becomes a pure re-export surface; no
name in the class body needed changing (the backend is injected, the config
fallback stays function-local inside ``__init__``).
"""

from MCEq.misc import info


class Decays:
    """Class for managing the dictionary of decay yield matrices.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db, override_decay_db_name=None, physics=None):
        from MCEq import config

        physics = config.physics if physics is None else physics
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: (list) List of particles in the decay matrices
        self.parent_list = []
        self._default_decay_dset = override_decay_db_name

        if self._default_decay_dset is None:
            if physics.muon_helicity_dependence:
                self._default_decay_dset = "polarized"
            else:
                self._default_decay_dset = "unpolarized"

    def load(self, parent_list=None, decay_dset=None):
        # Load tables and index from file
        if decay_dset is None:
            decay_dset = self._default_decay_dset

        index = self.mceq_db.decay_db(decay_dset)

        self.parents = index["parents"]
        self.particles = index["particles"]
        self.relations = index["relations"]
        self.index_d = index["index_d"]
        self.description = index["description"]
        # Advanced options
        regenerate_index = False
        if parent_list:
            # Take only the parents provided by the list
            # Add the decay products, which can become new parents
            def _follow_decay_chain(p, plist):
                if p in self.relations:
                    plist.append(p)
                    for d in self.relations[p]:
                        _follow_decay_chain(d, plist)
                else:
                    return plist

            plist = []
            for p in parent_list:
                _follow_decay_chain(p, plist)

            self.parents = sorted(list(set(plist)))
            regenerate_index = True

        if regenerate_index:
            self.particles = []
            for p in list(self.relations):
                if p not in self.parents:
                    _ = self.relations.pop(p, None)
                    continue
                self.particles.append(p)
                self.particles += self.relations[p]
            self.particles = sorted(list(set(self.particles)))

    def __getitem__(self, key):
        return self.get_matrix(*key)

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def children(self, parent_pdg):
        if parent_pdg not in self.relations:
            raise Exception(f"Parent {parent_pdg} not in decay database.")

        return self.relations[parent_pdg]

    def get_matrix(self, parent, child):
        """Returns a ``DIM x DIM`` decay matrix.

        Args:
          parent (int): PDG ID of parent particle
          child (int): PDG ID of final state child particle
        Returns:
          numpy.array: decay matrix
        """
        info(20, "entering with", parent, child)
        if child not in self.relations[parent]:
            raise Exception(f"trying to get empty matrix {parent} -> {child}")

        return self.index_d[(parent, child)]
