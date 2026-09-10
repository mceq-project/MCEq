"""Energy-grid accessors: centers, bins, widths, closest_energy.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)

testdata_ecenters = [
    8.91250938e2,
    1.12201845e3,
    1.41253754e3,
    1.77827941e3,
    2.23872114e3,
    2.81838293e3,
    3.54813389e3,
    4.46683592e3,
    5.62341325e3,
    7.07945784e3,
    8.91250938e3,
    1.12201845e4,
    1.41253754e4,
    1.77827941e4,
    2.23872114e4,
    2.81838293e4,
    3.54813389e4,
    4.46683592e4,
    5.62341325e4,
    7.07945784e4,
    8.91250938e4,
    1.12201845e5,
    1.41253754e5,
    1.77827941e5,
    2.23872114e5,
    2.81838293e5,
    3.54813389e5,
    4.46683592e5,
    5.62341325e5,
    7.07945784e5,
    8.91250938e5,
]
testdata_eedges = [
    7.94328235e2,
    1.00000000e3,
    1.25892541e3,
    1.58489319e3,
    1.99526231e3,
    2.51188643e3,
    3.16227766e3,
    3.98107171e3,
    5.01187234e3,
    6.30957344e3,
    7.94328235e3,
    1.00000000e4,
    1.25892541e4,
    1.58489319e4,
    1.99526231e4,
    2.51188643e4,
    3.16227766e4,
    3.98107171e4,
    5.01187234e4,
    6.30957344e4,
    7.94328235e4,
    1.00000000e5,
    1.25892541e5,
    1.58489319e5,
    1.99526231e5,
    2.51188643e5,
    3.16227766e5,
    3.98107171e5,
    5.01187234e5,
    6.30957344e5,
    7.94328235e5,
    1.00000000e6,
]
testdata_ewidths = [
    2.05671765e2,
    2.58925412e2,
    3.25967781e2,
    4.10369123e2,
    5.16624117e2,
    6.50391229e2,
    8.18794045e2,
    1.03080063e3,
    1.29770111e3,
    1.63370890e3,
    2.05671765e3,
    2.58925412e3,
    3.25967781e3,
    4.10369123e3,
    5.16624117e3,
    6.50391229e3,
    8.18794045e3,
    1.03080063e4,
    1.29770111e4,
    1.63370890e4,
    2.05671765e4,
    2.58925412e4,
    3.25967781e4,
    4.10369123e4,
    5.16624117e4,
    6.50391229e4,
    8.18794045e4,
    1.03080063e5,
    1.29770111e5,
    1.63370890e5,
    2.05671765e5,
]


def test_energy_grid_access(mceq_sib21):
    print(mceq_sib21.e_grid)
    print(mceq_sib21.e_bins)
    print(mceq_sib21.e_widths)
    assert np.allclose(mceq_sib21.e_grid, testdata_ecenters, rtol=1e-8, atol=0)
    assert np.allclose(mceq_sib21.e_bins, testdata_eedges, rtol=1e-8, atol=0)
    assert np.allclose(mceq_sib21.e_widths, testdata_ewidths, rtol=1e-8, atol=0)


def test_closest_energy(mceq_sib21):
    closest_energy = np.array(
        [mceq_sib21.closest_energy(energy) for energy in testdata_ecenters]
    )
    assert np.allclose(closest_energy, testdata_ecenters, rtol=0, atol=1e1)
