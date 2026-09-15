"""Scalar CORSIKA atmosphere formulas for WebAssembly.

Same layers, boundary comparisons and expressions as
geometry/corsikaatm/corsikaatm.c; no compiled-library loading.
"""

from math import exp


def _height_layer(height, boundaries):
    layer = 0
    for i in range(5):
        if not height <= boundaries[i]:
            layer = i
    return layer


def corsika_get_density(h_cm, a, b, c, t, hl):
    layer = _height_layer(h_cm, hl)
    if layer == 4:
        return b[4] / c[4]
    return b[layer] / c[layer] * exp(-h_cm / c[layer])


def corsika_get_m_overburden(h_cm, a, b, c, t, hl):
    layer = _height_layer(h_cm, hl)
    if layer == 4:
        return a[4] - b[4] / c[4] * h_cm
    return a[layer] + b[layer] * exp(-h_cm / c[layer])


def planar_rho_inv(X, cos_theta, a, b, c, t, hl):
    x_v = X * cos_theta
    layer = 0
    for i in range(5):
        if not x_v >= t[i]:
            layer = i
    if layer == 4:
        return c[4] / b[4]
    return c[layer] / (x_v - a[layer])
