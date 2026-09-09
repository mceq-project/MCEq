"""The energy grid: the grid tuple, the cut evaluation and the x_lab matrix.

``EnergyGrid`` aliases the ``energy_grid`` namedtuple. :mod:`MCEq.misc`
forwards the compatibility names.

:func:`gen_xmat` caches by matrix shape: grids of equal ``d`` share one
array and the return value *is* the cache, so mutating it affects later
calls. See ``tests/test_data_bug_pins.py``.
"""

from collections import namedtuple

import numpy as np

#: Energy grid: centers, bin edges, bin widths, bin count
energy_grid = namedtuple("energy_grid", ("c", "b", "w", "d"))

#: Alias of the same namedtuple type; its type name remains ``energy_grid``.
EnergyGrid = energy_grid

#: Matrix with x_lab=E_child/E_parent values
_xmat = None


def _eval_energy_cuts(e_centers, e_min=None, e_max=None):
    """Evaluate the energy cuts and return the corresponding indices and slice.

    Args:
        e_centers: numpy.ndarray
            Array of energy grid centers.
        e_min: float, optional
            Minimum energy value. Default is None.
        e_max: float, optional
            Maximum energy value. Default is None.

    Returns:
        min_idx: int
            Index of the bin nearest to ``e_min``.
        max_idx: int
            Exclusive upper index for the energy-grid slice.
        energy_slice: slice
            Slice corresponding to the energy range.

    """
    min_idx, max_idx = 0, len(e_centers)
    energy_slice = slice(None)
    if e_min is not None:
        min_idx = np.argmin(np.abs(e_centers - e_min))
        energy_slice = slice(min_idx, None)
    if e_max is not None:
        max_idx = np.argmin(np.abs(e_centers - e_max)) + 1
        energy_slice = slice(min_idx, max_idx)
    return min_idx, max_idx, energy_slice


def gen_xmat(energy_grid):
    """Return the cached child-to-parent energy-ratio matrix.

    Rebuilt only when the grid dimension changes: changed centers with the
    same length reuse the cached array.
    """
    global _xmat
    dims = (energy_grid.d, energy_grid.d)
    if _xmat is None or _xmat.shape != dims:
        _xmat = np.zeros(dims)
        for eidx in range(energy_grid.d):
            xvec = energy_grid.c[: eidx + 1] / energy_grid.c[eidx]
            _xmat[: eidx + 1, eidx] = xvec
    return _xmat
