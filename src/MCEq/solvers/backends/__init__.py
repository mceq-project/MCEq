"""Backend bindings of the ETD2RK step loop, one module per library.

A backend places a :class:`~MCEq.operator_assembly.CompiledOperator` on its
sparse library or device and executes the stages of
:func:`MCEq.solvers.etd2.etd2_driver` there. The four constructors
re-exported here are what :data:`MCEq.solvers.etd2._BACKENDS` binds by name.

Importing this package stays cheap on a machine with no MKL, no GPU and no
Accelerate: no module below it may load a shared library, import cupy or
probe hardware at import time. Every such call belongs inside the function
or constructor that needs it.
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
