import numpy as np
import pytest

testdata_geom = {
    "a1": [
        6.37131500e08,
        6.27452041e08,
        5.98707769e08,
        5.51772065e08,
        4.88071045e08,
        4.09540234e08,
        3.18565750e08,
        2.17911807e08,
        1.10636724e08,
        3.90130526e-08,
    ],
    "a2": [
        0.00000000e00,
        1.10636724e08,
        2.17911807e08,
        3.18565750e08,
        4.09540234e08,
        4.88071045e08,
        5.51772065e08,
        5.98707769e08,
        6.27452041e08,
        6.37131500e08,
    ],
    "pl": [
        1.12800000e07,
        1.14509163e07,
        1.19901250e07,
        1.29874739e07,
        1.46358975e07,
        1.73369478e07,
        2.20000470e07,
        3.10590360e07,
        5.28912851e07,
        1.20419787e08,
    ],
    "cos_star": [
        1.0,
        0.98533564,
        0.94183693,
        0.87098939,
        0.77528999,
        0.65834301,
        0.52523096,
        0.38397043,
        0.25219789,
        0.18571507,
    ],
    "delta_l": [
        11280000,
        10178287.081463099,
        9323278.090486884,
        8650036.903688192,
        8109344.7668501735,
        7654262.725001514,
        7214413.011852622,
        6603227.476002127,
        5042586.7104329765,
        0.0,
    ],
}


def test_earth_geometry():
    from MCEq.geometry.geometry import EarthGeometry

    geom = EarthGeometry()
    theta = np.deg2rad(np.linspace(0, 90, 10))
    h = np.linspace(0, geom.h_atm, 10)

    a1 = geom._A_1(theta)
    assert a1 == pytest.approx(testdata_geom["a1"], rel=1e-6)

    a2 = geom._A_2(theta)
    assert a2 == pytest.approx(testdata_geom["a2"], rel=1e-6)

    pl = geom.path_len(theta)
    assert pl == pytest.approx(testdata_geom["pl"], rel=1e-6)

    cos_star = geom.cos_th_star(theta)
    assert cos_star == pytest.approx(testdata_geom["cos_star"], rel=1e-6)

    delta_l = geom.delta_l(h, theta)
    assert delta_l == pytest.approx(testdata_geom["delta_l"], rel=1e-6, abs=1e-4)

    h_ret = geom.h(delta_l, theta)
    assert h_ret == pytest.approx(h, rel=1e-6, abs=1e-4)


def test_earth_geometry_set_h_obs():
    from MCEq.geometry.geometry import EarthGeometry

    geom = EarthGeometry()
    h_new = 2834.0 * 1e2  # IceCube depth in cm
    geom.set_h_obs(h_new)

    assert geom.h_obs == h_new
    assert geom.r_obs == pytest.approx(geom.r_E + h_new)
    expected_theta_max = np.rad2deg(
        max(np.pi / 2.0, np.pi - np.arcsin(geom.r_E / geom.r_obs))
    )
    assert geom.theta_max_deg == pytest.approx(expected_theta_max)
    assert geom.theta_max_rad == pytest.approx(np.deg2rad(expected_theta_max))


def test_earth_geometry_init_invalid_h_obs(monkeypatch):
    import MCEq.config as config
    from MCEq.geometry.geometry import EarthGeometry

    # h_obs below zero
    monkeypatch.setattr(config, "h_obs", -100.0)
    with pytest.raises(ValueError, match="Observation height"):
        EarthGeometry()

    # h_obs above h_atm
    monkeypatch.setattr(config, "h_obs", config.h_atm + 1.0)
    with pytest.raises(ValueError, match="Observation height"):
        EarthGeometry()


def test_earth_geometry_init_invalid_h_atm(monkeypatch):
    import MCEq.config as config
    from MCEq.geometry.geometry import EarthGeometry

    # h_atm equal to h_obs triggers the second guard (h_atm <= h_obs)
    monkeypatch.setattr(config, "h_obs", config.h_atm)
    with pytest.raises(ValueError, match="Top of atmosphere"):
        EarthGeometry()


def test_earth_geometry_set_h_obs_invalid():
    from MCEq.geometry.geometry import EarthGeometry

    geom = EarthGeometry()

    with pytest.raises(ValueError, match="Observation height"):
        geom.set_h_obs(-1.0)

    with pytest.raises(ValueError, match="Observation height"):
        geom.set_h_obs(geom.h_atm + 1.0)

    # equal to h_atm hits the second guard
    with pytest.raises(ValueError, match="Top of atmosphere"):
        geom.set_h_obs(geom.h_atm)


def test_chirkin_cos_theta_star():
    from MCEq.geometry.geometry import chirkin_cos_theta_star

    theta = np.deg2rad(np.linspace(0, 90, 10))
    cos_theta = np.cos(theta)

    chirkin = chirkin_cos_theta_star(cos_theta)

    # zero element is 1
    assert not chirkin[1:] == pytest.approx(cos_theta[1:], rel=1e-8, abs=1e-12)
