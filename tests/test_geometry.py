import numpy as np
import pytest

from MCEq.geometry.geometry import EarthGeometry


@pytest.fixture(scope="module")
def earth_geometry() -> EarthGeometry:
    return EarthGeometry()


def test_check_angles(earth_geometry: EarthGeometry) -> None:
    # Test exception when zenith angle is above max
    with pytest.raises(Exception):
        earth_geometry._check_angles(np.pi / 2 + 0.1)

    # Test that no exception is raised when zenith angle is below max
    earth_geometry._check_angles(np.pi / 2 - 0.1)


def test_a_1(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    expected_result = earth_geometry.r_obs * np.cos(theta)
    assert np.isclose(expected_result, earth_geometry._a_1(theta))


def test_a_2(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    expected_result = earth_geometry.r_obs * np.sin(theta)
    assert np.isclose(expected_result, earth_geometry._a_2(theta))


def test_pl(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    expected_result = np.sqrt(
        earth_geometry.r_top**2 - earth_geometry._a_2(theta) ** 2
    ) - earth_geometry._a_1(theta)
    assert np.isclose(expected_result, earth_geometry.pl(theta))


def test_cos_th_star(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    expected_result = (
        earth_geometry._a_1(theta) + earth_geometry.pl(theta)
    ) / earth_geometry.r_top
    assert np.isclose(expected_result, earth_geometry.cos_th_star(theta))


def test_h(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    dl = 100
    expected_result = (
        np.sqrt(
            earth_geometry._a_2(theta) ** 2
            + (earth_geometry._a_1(theta) + earth_geometry.pl(theta) - dl) ** 2
        )
        - earth_geometry.r_e
    )
    assert np.isclose(expected_result, earth_geometry.h(dl, theta))


def test_delta_l(earth_geometry: EarthGeometry) -> None:
    theta = np.pi / 4
    h = 100
    expected_result = (
        earth_geometry._a_1(theta)
        + earth_geometry.pl(theta)
        - np.sqrt((h + earth_geometry.r_e) ** 2 - earth_geometry._a_2(theta) ** 2)
    )
    assert np.isclose(expected_result, earth_geometry.delta_l(h, theta))
