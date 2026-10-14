"""Continuous energy-loss tables and the generic hadron dE/dX spline.

The generic (proton) curve is ``ionization/hadron`` when ``enable_em`` is on and
the group exists, otherwise ``total/hadron``.
"""

import numpy as np

from MCEq import config


class ContinuousLosses:
    """Manage stopping-power arrays loaded from ``mceq_hdf_db``.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: Energy grid of the backend
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Stopping-power arrays indexed by particle
        self.index_d = None
        #: Raw ``(boost, dEdX)`` table behind :attr:`generic_spl`, used to
        #: clamp arguments to the tabulated range.
        self.generic_table = None
        # Load defaults
        self.load_db()

    def __getitem__(self, parent):
        """Return the particle's stopping-power array in GeV/(g/cm²) as
        a dictionary lookup."""
        return self.index_d[parent]

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def _ionization_hadron_curve(self):
        """The ``ionization/hadron`` generic curve, or None if absent."""
        import h5py

        path = f"continuous_losses/{self.mceq_db.medium}/ionization/hadron"
        try:
            with h5py.File(self.mceq_db.had_fname, "r") as f:
                if path not in f:
                    return None
                return np.asarray(f[path], dtype=np.float64)
        except OSError:
            return None

    def load_db(self):
        from scipy.interpolate import InterpolatedUnivariateSpline

        # Load tables and index from file
        index = self.mceq_db.continuous_loss_db()
        self.parents = index["parents"]
        self.index_d = index["index_d"]
        if "generic" not in index and config.generic_losses_all_charged:
            raise Exception("New data file needed to support generic losses.")
        generic = index["generic"]
        if "generic" in index and config.enable_em:
            ionization = self._ionization_hadron_curve()
            if ionization is not None:
                generic = ionization
        self.generic_table = np.asarray(generic, dtype=np.float64)
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(generic[0]), np.log(generic[1]), k=1
        )
