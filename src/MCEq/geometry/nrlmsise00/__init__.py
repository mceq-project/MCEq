"""ctypes interface of the C version of NRLMSISE-00 by Dominik Brodowski."""

import os
import sysconfig
from ctypes import Structure, c_double, c_int, cdll

base = os.path.dirname(os.path.abspath(__file__))
suffix = sysconfig.get_config_var("EXT_SUFFIX")

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


class nrlmsise_input:
    """The inputs of ``gtd7_py`` as ctypes scalars (``year, doy, sec, alt,
    g_lat, g_long, lst, f107A, f107, ap``) and the ``ap_a`` array pointer."""


class nrlmsise_output(Structure):
    """The C-struct contains output variables for NRLMSISE."""

    _fields_ = [("d", c_double * 9), ("t", c_double * 2)]
