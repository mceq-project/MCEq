"""The browser atmosphere reproduces the compiled CORSIKA formulas at boundaries."""

import numpy as np
import pytest

from MCEq.geometry import corsika_numpy
from MCEq.geometry import corsikaatm
from MCEq.geometry.atmosphere_parameters import (
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)


@pytest.mark.parametrize("location,season", list_available_corsika_atmospheres())
def test_python_atmosphere_matches_c(location, season):
    parameters = np.array(get_atmosphere_parameters(location, season))
    heights = np.unique(
        np.concatenate(
            [
                np.linspace(0, 1.1e7, 81),
                parameters[4],
                parameters[4] - 1e-5,
                parameters[4] + 1e-5,
            ]
        )
    )
    for name in ("corsika_get_density", "corsika_get_m_overburden"):
        expected = [getattr(corsikaatm, name)(h, *parameters) for h in heights]
        actual = [getattr(corsika_numpy, name)(h, *parameters) for h in heights]
        np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-15)
    depths = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-4, 1500, 81),
                parameters[3],
                parameters[3] - 1e-8,
                parameters[3] + 1e-8,
            ]
        )
    )
    for cosine in (1.0, 0.5, 0.1):
        expected = [corsikaatm.planar_rho_inv(x, cosine, *parameters) for x in depths]
        actual = [corsika_numpy.planar_rho_inv(x, cosine, *parameters) for x in depths]
        np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-15)
