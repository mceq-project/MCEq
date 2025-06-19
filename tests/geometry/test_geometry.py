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
