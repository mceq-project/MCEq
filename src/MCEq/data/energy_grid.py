"""The energy grid: the grid tuple, the cut evaluation and the x_lab matrix.

:mod:`MCEq.misc` (C1's bottom layer, the wrong home for a data-layer
table) keeps a read/write forwarding shim for the three names so user
code and the bug pins keep resolving them there.

Two deliberate non-changes, both of which the plan asks for eventually and
neither of which belongs in a move commit:

* ``energy_grid`` stays a :func:`~collections.namedtuple`. Plan section 5 wants a
  frozen dataclass with ``cuts()`` and ``xmat()`` methods; a namedtuple has
  ``__slots__ = ()`` and no per-instance ``__dict__``, unpacks as a tuple and
  compares equal to one, and the tests hand ``SimpleNamespace`` stand-ins to the
  same call sites -- so the conversion is a type change, not a move.
  ``EnergyGrid`` is bound here as the plan's name for it so the eventual
  conversion is a change of one assignment.
* :func:`gen_xmat` keeps its module-global cache, keyed on ``(d, d)`` alone.
  Two grids of equal ``d`` therefore share one array and the return value *is*
  the cache -- ``tests/test_data_bug_pins.py`` pins exactly that. The per-grid
  cached method that replaces it is a separate commit, and the pins are the
  signal that it landed.
"""

from collections import namedtuple

import numpy as np

#: Energy grid (centers, bind widths, dimension)
energy_grid = namedtuple("energy_grid", ("c", "b", "w", "d"))

#: Plan section 1's name for the same type. An alias, not a second class: the
#: namedtuple's ``__name__`` stays ``energy_grid`` so no repr moves.
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
            Index corresponding to the minimum energy value.
        max_idx: int
            Index corresponding to the maximum energy value.
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
    """Generates x_lab matrix for a given energy grid"""
    global _xmat
    dims = (energy_grid.d, energy_grid.d)
    if _xmat is None or _xmat.shape != dims:
        _xmat = np.zeros(dims)
        for eidx in range(energy_grid.d):
            xvec = energy_grid.c[: eidx + 1] / energy_grid.c[eidx]
            _xmat[: eidx + 1, eidx] = xvec
    return _xmat
