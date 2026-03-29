import numpy as np
import pytest

testdata_geom = {
    "a1": [
        6.39100000e08,
        6.29390635e08,
        6.00557554e08,
        5.53476836e08,
        4.89579004e08,
        4.10805561e08,
        3.19550000e08,
        2.18585074e08,
        1.10978550e08,
        3.91335885e-08,
    ],
    "a2": [
        0.00000000e00,
        1.10978550e08,
        2.18585074e08,
        3.19550000e08,
        4.10805561e08,
        4.89579004e08,
        5.53476836e08,
        6.00557554e08,
        6.29390635e08,
        6.39100000e08,
    ],
    "pl": [
        1.12800000e07,
        1.14509256e07,
        1.19901666e07,
        1.29875869e07,
        1.46361639e07,
        1.73375732e07,
        2.20016607e07,
        3.10642180e07,
        5.29164558e07,
        1.20604040e08,
    ],
    "cos_star": [
        1.0,
        0.98533405,
        0.9418305,
        0.87097454,
        0.77526241,
        0.65829689,
        0.52515708,
        0.38385143,
        0.25199884,
        0.18543627,
    ],
    "delta_l": [
        11280000,
        10178296.32746565,
        9323317.64271486,
        8650136.95563626,
        8109556.85318005,
        7654687.97116119,
        7215277.80642521,
        6605097.18068355,
        5046468.16941541,
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
