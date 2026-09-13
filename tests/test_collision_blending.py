import numpy as np

from MCEq.data.blending import blend_collision_yields


def test_collision_operator_and_unique_channels():
    p = (2212, 0)
    a = (211, 0)
    b = (111, 0)
    hs = np.array([0.0, 100.0, 200.0])
    ls = np.array([50.0, 60.0, 70.0])
    w = np.array([0.0, 0.4, 1.0])
    h = np.arange(9.0).reshape(3, 3)
    low = np.ones((3, 3)) * 2
    r = blend_collision_yields(
        {"index_d": {(p, a): h}},
        {"index_d": {(p, a): low, (p, b): low}},
        "SIBYLL23E",
        "FLUKA20251",
        w,
        {"index_d": {2212: hs}},
        {"index_d": {2212: ls}},
    )
    total = w * hs + (1 - w) * ls
    np.testing.assert_allclose(
        r["index_d"][(p, a)] * total, w * hs * h + (1 - w) * ls * low
    )
    np.testing.assert_allclose(r["index_d"][(p, b)] * total, (1 - w) * ls * low)
    np.testing.assert_allclose(r["index_d"][(p, a)][:, 0], low[:, 0])
