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
    # Without this, the module imports cleanly with no `msis` attribute and the
    # failure surfaces much later as `AttributeError: module has no attribute
    # 'msis'` from the first density call. The compiled NRLMSISE-00 library
    # must be installed beside this module; it is gitignored, so a fresh
    # checkout needs a build (`pip install -e .`).
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

    This object supplies Python attributes to the scalar ``gtd7_py`` call; it
    is not passed to the native library as an input structure.

    ``_field_`` is a typo for ``_fields_`` and has been one since the struct
    was written, so ctypes never lays the struct out: ``sizeof()`` is 0,
    ``_fields_`` is ``None``, and every ``inp.<name>`` is an ordinary Python
    attribute holding whatever object was assigned. That is harmless only
    because ``gtd7_py`` takes scalars; ``nrlmsise_flags`` and
    ``nrlmsise_output``, which are passed ``byref``, declare ``_fields_``
    correctly.

    Do not "fix" the name without adapting the consumers: Python code reads
    these as ctypes scalar objects (``self.inp.sec.value`` in
    :class:`MCEq.environment.msis00_backend.cNRLMSISE00`), and a real
    ``_fields_`` would hand back plain floats and break every ``.value``.
    ``inp.doy`` already carries two different runtime types depending on
    whether ``set_season`` (bare int) or ``set_doy`` (``c_int``) ran last.
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
