from ctypes import (cdll, Structure, c_int, c_double, POINTER)
import os
base = os.path.dirname(os.path.abspath(__file__))

for fn in os.listdir(base):
    if 'libcorsikaatm' in fn and (fn.endswith('.so') or fn.endswith('.dll')
                                  or fn.endswith('.dylib') or fn.endswith('.pyd')):
        corsika_acc = cdll.LoadLibrary(os.path.join(base, fn))
        break

for func in [
        corsika_acc.corsika_get_density, corsika_acc.planar_rho_inv,
        corsika_acc.corsika_get_m_overburden
]:
    func.restype = c_double


def corsika_get_density(h_cm, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.corsika_get_density(
        c_double(h_cm), a.ctypes, b.ctypes, c.ctypes, t.ctypes, hl.ctypes)


def planar_rho_inv(X, cos_theta, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.planar_rho_inv(
        c_double(X), c_double(cos_theta), a.ctypes, b.ctypes, c.ctypes,
        t.ctypes, hl.ctypes)


def corsika_get_m_overburden(h_cm, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.corsika_get_m_overburden(
        c_double(h_cm), a.ctypes, b.ctypes, c.ctypes, t.ctypes, hl.ctypes)
