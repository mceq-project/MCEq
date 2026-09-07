"""Continuous ionisation losses: the ``ContinuousLosses`` consumer.

Plan section 1's ``data/continuous_losses.py``, §13.2 item 4. The class moved
here verbatim from ``data/__init__.py``, which becomes a pure re-export
surface; the generic-loss spline check in ``load_db`` is unchanged.
"""

import numpy as np


class ContinuousLosses:
    """Class for managing the dictionary of hadron-air cross-sections.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
      material (str): name of the material (not fully implemented)
    """

    def __init__(self, mceq_hdf_db, *, physics):
        self._physics = physics
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: reference to energy grid
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Dictionary containing the distribuiton matrices
        self.index_d = None
        # Load defaults
        self.load_db()

    def __getitem__(self, parent):
        """Return the cross section in :math:`\\text{cm}^2` as
        a dictionary lookup."""
        return self.index_d[parent]

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def load_db(self):
        from scipy.interpolate import InterpolatedUnivariateSpline

        # Load tables and index from file
        index = self.mceq_db.continuous_loss_db()
        self.parents = index["parents"]
        self.index_d = index["index_d"]
        if "generic" not in index and self._physics.generic_losses_all_charged:
            raise Exception("New data file needed to support generic losses.")
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(index["generic"][0]), np.log(index["generic"][1]), k=1
        )
