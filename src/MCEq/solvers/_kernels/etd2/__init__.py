"""ctypes bindings of the compiled ETD2 predictor and corrector (``etd2_kernels.c``), fp64 and fp32.

The formulas are those of :mod:`MCEq.solvers.numerics`, over the row-major
``(dim, K)`` state. Every kernel takes ``(dim, K, per_lane, *pointers)``: ``per_lane`` says
whether the factor arrays are ``(dim, K)``, one integration path per lane, or
``(dim,)``, one path shared by all of them.
"""

import os
import sysconfig
from ctypes import POINTER, c_double, c_float, c_int, cdll

_base = os.path.dirname(os.path.abspath(__file__))
_suffix = sysconfig.get_config_var("EXT_SUFFIX")
assert _suffix is not None, "Shared lib suffix was not identified."

_lib = None
for _fn in os.listdir(_base):
    if "libetd2_kernels" in _fn and _fn.endswith(_suffix):
        _lib = cdll.LoadLibrary(os.path.join(_base, _fn))
        break
if _lib is None:
    raise ImportError(
        "MCEq.solvers._kernels.etd2: failed to find compiled libetd2_kernels"
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
