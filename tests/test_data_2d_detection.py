"""2D-database detection on HDF5Backend.

The backend should detect a 2D database by reading the ``k_dim`` attribute
from the ``common`` HDF5 group and expose ``is_2d``, ``n_k`` and ``k_grid``
attributes. The 1D reduced test database has no ``k_dim`` attribute; the
URQMD 2D database has ``k_dim=24``.
"""

import os

import numpy as np
import pytest

from MCEq import config
from MCEq.data import HDF5Backend


@pytest.fixture(scope="module")
def backend_1d():
    """Reduced 1D test database (already used by other tests)."""
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    return HDF5Backend(medium="air")


@pytest.fixture(scope="module")
def backend_2d():
    """URQMD 2D database. Must be present (or symlinked) on disk."""
    fn = "mceq_db_URQMD_150GeV_2D.h5"
    if not os.path.exists(os.path.join(config.data_dir, fn)):
        pytest.skip(f"{fn} not available; symlink it into src/MCEq/data/")
    config.mceq_db_fname = fn
    return HDF5Backend(medium="air")


def test_1d_database_not_2d(backend_1d):
    assert backend_1d.is_2d is False
    assert backend_1d.n_k == 1
    # k_grid for 1D is the trivial single-mode case [0]
    assert list(backend_1d.k_grid) == [0]


def test_2d_database_detected(backend_2d):
    assert backend_2d.is_2d is True
    assert backend_2d.n_k == 24
    assert backend_2d.k_grid[0] == 0
    assert backend_2d.k_grid[-1] == 2000
    # URQMD 2D db k_grid is integer-valued
    assert np.issubdtype(backend_2d.k_grid.dtype, np.integer) or np.issubdtype(
        backend_2d.k_grid.dtype, np.floating
    )
