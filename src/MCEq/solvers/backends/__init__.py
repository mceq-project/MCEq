"""Backend bindings of the ETD2RK step loop, one module per sparse library or device.

No module here loads a shared library, imports cupy or probes hardware at
import time.
"""

from MCEq.solvers.backends.accelerate import accelerate_backend
from MCEq.solvers.backends.cuda import cuda_backend
from MCEq.solvers.backends.host import numpy_backend
from MCEq.solvers.backends.mkl import mkl_backend

__all__ = [
    "accelerate_backend",
    "cuda_backend",
    "mkl_backend",
    "numpy_backend",
]
