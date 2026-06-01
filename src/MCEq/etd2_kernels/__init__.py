"""Platform-neutral fused ETD2 post-apply C kernels.

Loads the shared library built from ``etd2_kernels.c`` and exposes the four
post-apply functions used by the multi-RHS / multipath solvers. The kernels
themselves have no sparse-backend dependencies (pure stride-1 loops over
column-major (dim, K) buffers), so the same compiled module works on Mac,
Linux, and Windows. Used by the MKL multi-RHS path on Linux; the Accelerate
path on Mac keeps using its in-tree copies in ``MCEq.spacc`` for now.

Symbol names + ABI match ``MCEq.spacc``'s post-apply bindings exactly, so
callers can swap one import for the other without touching the call sites.
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

# fp64 bindings (signatures match spacc.c).
_lib.etd2_post_apply1_multirhs.restype = None
_lib.etd2_post_apply1_multirhs.argtypes = [
    c_int,
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
_lib.etd2_post_apply2_multirhs.restype = None
_lib.etd2_post_apply2_multirhs.argtypes = [
    c_int,
    c_int,
    c_double,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
_lib.etd2_post_apply1_multipath.restype = None
_lib.etd2_post_apply1_multipath.argtypes = [
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
_lib.etd2_post_apply2_multipath.restype = None
_lib.etd2_post_apply2_multipath.argtypes = [
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]

# fp32 bindings.
_lib.etd2_post_apply1_multirhs_f32.restype = None
_lib.etd2_post_apply1_multirhs_f32.argtypes = [
    c_int,
    c_int,
    c_float,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
_lib.etd2_post_apply2_multirhs_f32.restype = None
_lib.etd2_post_apply2_multirhs_f32.argtypes = [
    c_int,
    c_int,
    c_float,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
_lib.etd2_post_apply1_multipath_f32.restype = None
_lib.etd2_post_apply1_multipath_f32.argtypes = [
    c_int,
    c_int,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]
_lib.etd2_post_apply2_multipath_f32.restype = None
_lib.etd2_post_apply2_multipath_f32.argtypes = [
    c_int,
    c_int,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
]

etd2_post_apply1_multirhs = _lib.etd2_post_apply1_multirhs
etd2_post_apply2_multirhs = _lib.etd2_post_apply2_multirhs
etd2_post_apply1_multipath = _lib.etd2_post_apply1_multipath
etd2_post_apply2_multipath = _lib.etd2_post_apply2_multipath

etd2_post_apply1_multirhs_f32 = _lib.etd2_post_apply1_multirhs_f32
etd2_post_apply2_multirhs_f32 = _lib.etd2_post_apply2_multirhs_f32
etd2_post_apply1_multipath_f32 = _lib.etd2_post_apply1_multipath_f32
etd2_post_apply2_multipath_f32 = _lib.etd2_post_apply2_multipath_f32

__all__ = [
    "etd2_post_apply1_multirhs",
    "etd2_post_apply2_multirhs",
    "etd2_post_apply1_multipath",
    "etd2_post_apply2_multipath",
    "etd2_post_apply1_multirhs_f32",
    "etd2_post_apply2_multirhs_f32",
    "etd2_post_apply1_multipath_f32",
    "etd2_post_apply2_multipath_f32",
]
