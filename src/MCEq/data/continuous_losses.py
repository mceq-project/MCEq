"""Continuous ionisation losses: the ``ContinuousLosses`` consumer.

Plan section 1's ``data/continuous_losses.py``, §13.2 item 4. The class moved
here verbatim from ``data/__init__.py``, which becomes a pure re-export
surface; the generic-loss spline check in ``load_db`` is unchanged.

The generic (proton-curve) source follows the lepton branch of the backend:
with ``enable_em`` on it is ``ionization/hadron``, otherwise ``total/hadron``
(the backend still reads the latter, so ``load_db`` re-reads the ionization
group itself when the flag is on and the group exists). The raw table pair is
kept on :attr:`generic_table` so consumers can clamp their boost argument to
the tabulated interval instead of extrapolating the k=1 log-log spline off
either end.
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
        #: Raw ``(boost, dEdX)`` pair behind :attr:`generic_spl`, as read
        #: (before the log transform); see :meth:`load_db`.
        self.generic_table = None
        # Load defaults
        self.load_db()

    def __getitem__(self, parent):
        """Return the cross section in :math:`\\text{cm}^2` as
        a dictionary lookup."""
        return self.index_d[parent]

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def _ionization_hadron_curve(self):
        """The ``ionization/hadron`` generic curve, or None if absent.

        Mirrors the backend's lepton branch (``ionization`` when ``enable_em``)
        for the generic proton curve. Read directly rather than through
        ``continuous_loss_db`` because the backend pairs the generic table with
        whichever loss case its leptons got and has no switch for the hadron
        group.
        """
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
        if "generic" not in index and self._physics.generic_losses_all_charged:
            raise Exception("New data file needed to support generic losses.")
        generic = index["generic"]
        if "generic" in index and self._physics.enable_em:
            ionization = self._ionization_hadron_curve()
            if ionization is not None:
                generic = ionization
        self.generic_table = np.asarray(generic, dtype=np.float64)
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(generic[0]), np.log(generic[1]), k=1
        )
