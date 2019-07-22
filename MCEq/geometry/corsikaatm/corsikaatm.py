from ctypes import (cdll, Structure, c_int, c_double, POINTER)
import os
base = os.path.dirname(os.path.abspath(__file__))

for fn in os.listdir(base):
    if 'libcorsikaatm' in fn and (fn.endswith('.so') or
                                  fn.endswith('.dll') or 
                                  fn.endswith('.dylib')):
        corsika_acc = cdll.LoadLibrary(os.path.join(base, fn))
        break