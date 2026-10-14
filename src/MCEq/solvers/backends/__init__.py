"""Backends of the ETD2RK step loop, one module per sparse library or device.

No module here loads a shared library, imports cupy or probes hardware at
import time.
"""

import numpy as np

from MCEq.config import detect
from MCEq.solvers.backends.accelerate import accelerate_backend
from MCEq.solvers.backends.cuda import CudaBackend
from MCEq.solvers.backends.host import numpy_backend
from MCEq.solvers.backends.mkl import mkl_backend

__all__ = [
    "CudaBackend",
    "accelerate_backend",
    "make_backend",
    "mkl_backend",
    "numpy_backend",
]


def make_backend(op, kernel, dtype=np.float64, device_id=0, operator_dtype=None):
    """Backend of the compiled operator ``op`` for ``kernel``, a name of
    :data:`MCEq.config.detect.KERNELS` or its alias; ``device_id`` and
    ``operator_dtype`` apply to CUDA only."""
    name = detect.ALIASES.get(kernel.lower(), kernel.lower())
    if name == "numpy_etd2":
        return numpy_backend(op, dtype)
    if name == "mkl_etd2":
        return mkl_backend(op, dtype)
    if name == "accelerate_etd2":
        return accelerate_backend(op, dtype)
    if name == "cuda_etd2":
        return CudaBackend(op, device_id, dtype, operator_dtype)
    raise ValueError(
        f"Unsupported kernel {kernel!r}; choose one of {', '.join(detect.KERNELS)}."
    )
