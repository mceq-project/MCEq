"""Hadron-air cross-sections: the ``InteractionCrossSections`` consumer.

Plan section 1's ``data/cross_sections.py``, §13.2 item 4. The class moved here
verbatim from ``data/__init__.py``, which becomes a pure re-export surface; the
unit constants and the energy-grid reference are unchanged.
"""

import numpy as np

from MCEq.data.model_names import normalize_hadronic_model_name
from MCEq.misc import info


class InteractionCrossSections:
    """Class for managing the dictionary of hadron-air cross-sections.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
      interaction_model (str): name of the interaction model
    """

    #: unit - :math:`\text{GeV} \cdot \text{fm}`
    GeVfm = 0.19732696312541853
    #: unit - :math:`\text{GeV} \cdot \text{cm}`
    GeVcm = GeVfm * 1e-13
    #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
    GeV2mbarn = 10.0 * GeVfm**2
    #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
    mbarn2cm2 = GeVcm**2 / GeV2mbarn

    def __init__(self, mceq_hdf_db, interaction_model="DPMJETIII193"):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: reference to energy grid
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Dictionary containing the distribuiton matrices
        self.index_d = None
        #: (str) Interaction Model name
        self.iam = normalize_hadronic_model_name(interaction_model)

    def __getitem__(self, parent):
        """Return the cross section in :math:`\\text{cm}^2` as a dictionary
        lookup."""
        return self.get_cs(parent)

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def load(self, interaction_model):
        #: (str) Interaction Model name
        self.iam = normalize_hadronic_model_name(interaction_model)
        # Load tables and index from file
        index = self.mceq_db.cs_db(self.iam)

        self.parents = index["parents"]
        self.index_d = index["index_d"]

    def get_cs(self, parent, mbarn=False):
        """Returns production ``parent``-air cross-section
        :math:`\\sigma_{prod}^{proj-Air}(E)` as vector spanned over
        the energy grid.

        Args:
          parent (int): PDG ID of parent particle
          mbarn (bool,optional): if ``True``, the units of the cross-section
                                 will be :math:`mbarn`,
                                 else :math:`\\text{cm}^2`

        Returns:
          numpy.array: cross-section in :math:`mbarn` or :math:`\\text{cm}^2`
        """

        message_templ = "replacing {0} with {1} cross section"
        if isinstance(parent, tuple):
            parent = parent[0]

        def _family_cs(base_pdg, label):
            """Family representative, sign-preserving: a negative parent
            uses the antiparticle column when the model provides one
            (annihilation channels for antibaryons; dedicated K-/pi-
            columns) and falls back to the particle column otherwise."""
            if parent < 0 and -base_pdg in self.index_d:
                info(15, message_templ.format(parent, f"anti-{label}"))
                return self.index_d[-base_pdg]
            info(15, message_templ.format(parent, label))
            return self.index_d[base_pdg]

        if parent in list(self.index_d):
            cs = self.index_d[parent]
        elif abs(parent) in list(self.index_d):
            cs = self.index_d[abs(parent)]
        elif 100 < abs(parent) < 300 and abs(parent) != 130:
            cs = _family_cs(211, "pi+-")
        elif 300 < abs(parent) < 1000 or abs(parent) in [130, 10313, 10323]:
            cs = _family_cs(321, "K+-")
        elif abs(parent) > 1000 and abs(parent) < 5000:
            cs = _family_cs(2212, "nucleon")
        elif 5 < abs(parent) < 23:
            info(15, "returning 0 cross-section for lepton", parent)
            return np.zeros(self.energy_grid.d)
        else:
            info(1, "Strange case for parent, using zero cross section.")
            cs = 0.0

        if not mbarn:
            return self.mbarn2cm2 * cs
        return cs
