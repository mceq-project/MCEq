"""Cold-cache readout construction stays bounded on production-sized grids."""

import numpy as np
import scipy.special

from MCEq.operators.secant import _readout_matrix


def test_readout_chunk_bounds_workspace_without_changing_values(monkeypatch):
    k_grid = np.geomspace(0.1, 2000.0, 48)
    theta = np.linspace(0.0, np.pi / 2, 7)
    expected = _readout_matrix(k_grid, theta, chunk=1)
    j0 = scipy.special.j0
    widths = []

    def tracked_j0(values):
        widths.append(values.shape[1])
        return j0(values)

    monkeypatch.setattr(scipy.special, "j0", tracked_j0)
    actual = _readout_matrix(k_grid, theta)
    assert len(widths) > 1
    assert max(widths) * 20000 * 48 * 8 * 3 <= 64 * 1024**2
    np.testing.assert_array_equal(actual, expected)
