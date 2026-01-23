import os
import sysconfig
from ctypes import c_double, cdll

base = os.path.dirname(os.path.abspath(__file__))
suffix = sysconfig.get_config_var("EXT_SUFFIX")
# Some Python 2.7 versions don't define EXT_SUFFIX
if suffix is None and "SO" in sysconfig.get_config_vars():
    suffix = sysconfig.get_config_var("SO")

assert suffix is not None, "Shared lib suffix was not identified."

for fn in os.listdir(base):
    if "libcorsikaatm" in fn and fn.endswith(suffix):
        corsika_acc = cdll.LoadLibrary(os.path.join(base, fn))
        break

for func in [
    corsika_acc.corsika_get_density,
    corsika_acc.planar_rho_inv,
    corsika_acc.corsika_get_m_overburden,
]:
    func.restype = c_double


def corsika_get_density(h_cm, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.corsika_get_density(
        c_double(h_cm), a.ctypes, b.ctypes, c.ctypes, t.ctypes, hl.ctypes
    )


def planar_rho_inv(X, cos_theta, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.planar_rho_inv(
        c_double(X),
        c_double(cos_theta),
        a.ctypes,
        b.ctypes,
        c.ctypes,
        t.ctypes,
        hl.ctypes,
    )


def corsika_get_m_overburden(h_cm, a, b, c, t, hl):
    """Wrap arguments for ctypes function"""
    return corsika_acc.corsika_get_m_overburden(
        c_double(h_cm), a.ctypes, b.ctypes, c.ctypes, t.ctypes, hl.ctypes
    )
