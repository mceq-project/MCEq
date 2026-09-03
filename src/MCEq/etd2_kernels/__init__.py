"""The fused ETD2 predictor and corrector, compiled.

Loads the shared library built from ``etd2_kernels.c`` and exposes its four
symbols: the two state stages of :func:`MCEq.solvers.etd2.etd2_driver` at
fp64 and at fp32. Both precisions come from one macro body over the
row-major ``(dim, K)`` state, and the formulas are the table in
:mod:`MCEq.solvers.numerics`.

The kernels have no sparse-backend dependency, so the same compiled module
serves the scipy, MKL and Accelerate bindings of
:class:`MCEq.solvers.backends.host.HostBackend` on Mac, Linux and Windows.

Every kernel takes ``(dim, K, per_lane, *pointers)``: ``per_lane`` says
whether the factor arrays are ``(dim, K)``, one integration path per lane, or
``(dim,)``, one path shared by all of them.
"""

import os
import sysconfig
from ctypes import POINTER, c_double, c_float, c_int, cdll

_base = os.path.dirname(os.path.abspath(__file__))
_suffix = sysconfig.get_config_var("EXT_SUFFIX")
if _suffix is None and "SO" in sysconfig.get_config_vars():
    _suffix = sysconfig.get_config_var("SO")
assert _suffix is not None, "Shared lib suffix was not identified."

_lib = None
for _fn in os.listdir(_base):
    if "libetd2_kernels" in _fn and _fn.endswith(_suffix):
        _lib = cdll.LoadLibrary(os.path.join(_base, _fn))
        break
if _lib is None:
    raise ImportError(
        "MCEq.etd2_kernels: failed to find compiled libetd2_kernels"
        f"{_suffix} in {_base}. The build step did not produce the "
        "shared library; re-run ``pip install -e .`` or check the "
        "etd2_kernels CMake target."
    )


def _bind(name, pointer):
    """Bind one kernel: ``(dim, K, per_lane)`` and five arrays of one dtype."""
    fn = getattr(_lib, name)
    fn.restype = None
    fn.argtypes = [c_int, c_int, c_int] + [pointer] * 5
    return fn


etd2_predictor_f64 = _bind("etd2_predictor_f64", POINTER(c_double))
etd2_corrector_f64 = _bind("etd2_corrector_f64", POINTER(c_double))
etd2_predictor_f32 = _bind("etd2_predictor_f32", POINTER(c_float))
etd2_corrector_f32 = _bind("etd2_corrector_f32", POINTER(c_float))

__all__ = [
    "etd2_corrector_f32",
    "etd2_corrector_f64",
    "etd2_predictor_f32",
    "etd2_predictor_f64",
]
