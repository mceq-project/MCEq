'''
Ctypes interface for struct-based interface to the C-version of NRLMSISE-00.
This C version of NRLMSISE-00 is written by Dominik Brodowski 
'''

from ctypes import (cdll, Structure, c_int, c_double, pointer, byref, POINTER)
import os
base = os.path.dirname(os.path.abspath(__file__))

for fn in os.listdir(base):
    if 'libnrlmsis' in fn and (fn.endswith('.so') or
                               fn.endswith('.dll') or 
                               fn.endswith('.dylib') or
                               fn.endswith('.pyd')):
        msis = cdll.LoadLibrary(os.path.join(base, fn))
        break


class nrlmsise_flags(Structure):
    """C-struct containing NRLMSISE related switches"""
    _fields_ = [("switches", c_int * 24), ("sw", c_double * 24),
                ("swc", c_double * 24)]


class ap_array(Structure):
    """C-struct containing NRLMSISE related switches"""
    _fields_ = [("a", c_double * 7)]


class nrlmsise_input(Structure):
    """The C-struct contains input variables for NRLMSISE."""
    _field_ = [("year", c_int), ("doy", c_int), ("sec", c_double),
               ("alt", c_double), ("g_lat", c_double), ("g_long", c_double),
               ("lst", c_double), ("f107A", c_double), ("f107", c_double),
               ("ap", c_double), ("ap_a", POINTER(ap_array))]


class nrlmsise_output(Structure):
    """The C-struct contains output variables for NRLMSISE."""
    _fields_ = [("d", c_double * 9), ("t", c_double * 2)]
