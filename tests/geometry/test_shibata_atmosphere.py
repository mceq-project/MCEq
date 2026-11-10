"""Shibata fit of the US Standard Atmosphere."""

import numpy as np
import pytest

from MCEq.geometry.density_profiles import ShibataAtmosphere


def gms_he(h_m):
    """Vertical depth of the reference Fortran function GMS_HE (ZNTH = 1)."""
    h = h_m * 1000.0
    if h > 2.50916e7:
        c = 4.863e7 - h
        return np.exp((8.275e6 - np.sqrt(6.8476e13 - 1.196e6 * c)) / 5.98e5)
    if h > 1.10171e7:
        return np.exp((4.55e7 - h) / 6.34e6)
    return np.exp(np.log((4.434e7 - h) / 1.186e7) / 0.19)


H_CM = np.array([0.0, 5e5, 1.1e6, 1.2e6, 2.0e6, 2.6e6, 4.0e6, 8.0e6, 1.128e7])


def test_overburden_matches_fortran():
    atm = ShibataAtmosphere()
    ref = [gms_he(h / 100.0) for h in H_CM]
    assert np.allclose(atm.get_mass_overburden(H_CM), ref, rtol=1e-12)


def test_density_is_height_derivative():
    atm = ShibataAtmosphere()
    d = 10.0
    num = -(atm.get_mass_overburden(H_CM + d) - atm.get_mass_overburden(H_CM - d))
    assert np.allclose(atm.get_density(H_CM), num / (2 * d), rtol=1e-6)


def test_depth2height_roundtrip():
    atm = ShibataAtmosphere()
    assert np.allclose(atm.depth2height(atm.get_mass_overburden(H_CM)), H_CM)


@pytest.mark.parametrize("theta", [0.0, 60.0])
def test_slant_column(theta):
    atm = ShibataAtmosphere("USStd", None)
    atm.set_theta(theta)
    X0 = atm.get_mass_overburden(0.0)
    assert X0 == pytest.approx(1033.4, abs=0.1)
    if theta == 0.0:
        assert atm.max_X == pytest.approx(X0, rel=2e-3)
    else:
        assert 1.9 * X0 < atm.max_X < 2.0 * X0
