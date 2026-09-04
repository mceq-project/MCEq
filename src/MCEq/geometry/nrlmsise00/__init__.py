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


class nrlmsise_flags(Structure):
    """C-struct containing NRLMSISE related switches"""

    _fields_ = [("switches", c_int * 24), ("sw", c_double * 24), ("swc", c_double * 24)]


class ap_array(Structure):
    """C-struct containing NRLMSISE related switches"""

    _fields_ = [("a", c_double * 7)]


class nrlmsise_input(Structure):
    """The C-struct contains input variables for NRLMSISE.

    ``_field_`` is a typo for ``_fields_`` and has been one since the struct
    was written. The consequence is that ctypes never lays the struct out:
    ``sizeof()`` is 0, ``_fields_`` is ``None``, and every ``inp.<name>`` is an
    ordinary Python attribute holding whatever object was assigned. This is
    harmless only because ``gtd7_py`` is a scalar shim, so this struct is never
    passed ``byref`` -- ``nrlmsise_flags`` and ``nrlmsise_output``, which are,
    both declare ``_fields_`` correctly.

    Do not "fix" the name without auditing the callers: Python code reads these
    as ctypes objects (``self.inp.sec.value`` in
    :meth:`MCEq.geometry.nrlmsise00_mceq.cNRLMSISE00._update_lst`), and a real
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
