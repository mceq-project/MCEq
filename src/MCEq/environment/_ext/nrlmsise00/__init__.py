"""
Ctypes interface for struct-based interface to the C-version of NRLMSISE-00.
This C version of NRLMSISE-00 is written by Dominik Brodowski
"""

import os
import sysconfig
from ctypes import POINTER, Structure, c_double, c_int, cdll

base = os.path.dirname(os.path.abspath(__file__))
suffix = sysconfig.get_config_var("EXT_SUFFIX")
# Some Python 2.7 versions don't define EXT_SUFFIX
if suffix is None and "SO" in sysconfig.get_config_vars():
    suffix = sysconfig.get_config_var("SO")

assert suffix is not None, "Shared lib suffix was not identified."

for fn in os.listdir(base):
    if fn.startswith("_libnrlmsis") and fn.endswith(suffix):
        msis = cdll.LoadLibrary(os.path.join(base, fn))
        break
else:
    raise ImportError(
        f"No _libnrlmsis*{suffix} found in {base}. Install the compiled "
        "NRLMSISE-00 library beside this module (`pip install -e .`)."
    )


class nrlmsise_flags(Structure):
    """C-struct containing NRLMSISE related switches"""

    _fields_ = [("switches", c_int * 24), ("sw", c_double * 24), ("swc", c_double * 24)]


class ap_array(Structure):
    """C-struct holding the seven geomagnetic Ap inputs."""

    _fields_ = [("a", c_double * 7)]


class nrlmsise_input(Structure):
    """Input variables for NRLMSISE.

    ``_field_`` (not ``_fields_``) means ctypes does not lay out a struct: the
    entries are plain Python attributes holding ctypes scalars, read via
    ``.value`` by ``cNRLMSISE00``. The struct is never passed to the C library.
    """

    _field_ = [
        ("year", c_int),
        ("doy", c_int),
        ("sec", c_double),
        ("alt", c_double),
        ("g_lat", c_double),
        ("g_long", c_double),
        ("lst", c_double),
        ("f107A", c_double),
        ("f107", c_double),
        ("ap", c_double),
        ("ap_a", POINTER(ap_array)),
    ]


class nrlmsise_output(Structure):
    """The C-struct contains output variables for NRLMSISE."""

    _fields_ = [("d", c_double * 9), ("t", c_double * 2)]
