'''
Created on Feb 18, 2015

@author: afedynitch
'''

from ctypes import cdll, Structure, c_long, c_int, \
                   c_double, pointer, byref, \
                   POINTER
import os.path as path
base = path.dirname(path.abspath(__file__))

try:
    msis = cdll.LoadLibrary("./msis-00.so")
except OSError:
    msis = cdll.LoadLibrary(path.join(base, "msis-00.so"))


#===============================================================================
# nrlmsise_flags
#===============================================================================
class nrlmsise_flags(Structure):
    """C-struct containing NRLMSISE related switches"""
    _fields_ = [("switches", c_int * 24), ("sw", c_double * 24),
                ("swc", c_double * 24)]


#===============================================================================
# ap_array
#===============================================================================
class ap_array(Structure):
    """C-struct containing NRLMSISE related switches"""
    _fields_ = [("a", c_double * 7)]


#===============================================================================
# nrlmsise_input
#===============================================================================
class nrlmsise_input(Structure):
    """The C-struct contains input variables for NRLMSISE."""
    _field_ = [("year", c_int), ("doy", c_int), ("sec", c_double),
               ("alt", c_double), ("g_lat", c_double), ("g_long", c_double),
               ("lst", c_double), ("f107A", c_double), ("f107", c_double),
               ("ap", c_double), ("ap_a", POINTER(ap_array))]


#===============================================================================
# nrlmsise_output
#===============================================================================
class nrlmsise_output(Structure):
    """The C-struct contains output variables for NRLMSISE."""
    _fields_ = [("d", c_double * 9), ("t", c_double * 2)]
