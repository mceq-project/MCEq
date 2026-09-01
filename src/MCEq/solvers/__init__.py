"""ETD2RK cascade solvers: the step loop, its backends and its inputs.

The public surface of the solver stack, re-exported from the modules that
own each piece:

* :mod:`MCEq.solvers.numerics` -- the scalar phi-function maths and the
  scratch-buffer layouts every backend shares.
* :mod:`MCEq.solvers.etd2` -- the ETD2RK step loop
  (:func:`~MCEq.solvers.etd2.etd2_driver`) and the one entry point that
  compiles an operator, binds a backend and runs it
  (:func:`~MCEq.solvers.etd2.solve_etd2`).
* :mod:`MCEq.solvers.backends` -- one module per sparse library or device:
  scipy, MKL, Apple Accelerate, CUDA.
* :mod:`MCEq.solvers.path` -- the non-uniform integration path.
* :mod:`MCEq.solvers.schedule` -- the LPT carousel that packs many pixels
  through a fixed-width multi-RHS pipeline.

A name is re-exported here when a caller outside the package uses it;
everything else stays in the module that owns it. The operator-assembly
names below are re-exported because callers reach them through this module.
"""

from MCEq.operator_assembly import (
    CompiledOperator,
    compile_operator,
    secant_layout,
    split_diagonal,
)
from MCEq.solvers.backends import (
    accelerate_backend,
    cuda_backend,
    mkl_backend,
    numpy_backend,
)
from MCEq.solvers.backends.accelerate import SpaccApplyOff
from MCEq.solvers.backends.base import ScipyApplyOff
from MCEq.solvers.backends.cuda import CudaBackend, CudaOperator
from MCEq.solvers.backends.host import HostBackend
from MCEq.solvers.backends.mkl import MklApplyOff, MklSparseMatrix
from MCEq.solvers.etd2 import etd2_driver, solve_etd2
from MCEq.solvers.path import etd2_nonuniform_path
from MCEq.solvers.schedule import (
    CarouselSchedule,
    compile_carousel_schedule,
    schedule_lpt,
)

__all__ = [
    "CarouselSchedule",
    "CompiledOperator",
    "CudaBackend",
    "CudaOperator",
    "HostBackend",
    "MklApplyOff",
    "MklSparseMatrix",
    "ScipyApplyOff",
    "SpaccApplyOff",
    "accelerate_backend",
    "compile_carousel_schedule",
    "compile_operator",
    "cuda_backend",
    "etd2_driver",
    "etd2_nonuniform_path",
    "mkl_backend",
    "numpy_backend",
    "schedule_lpt",
    "secant_layout",
    "secant_split",
    "solve_etd2",
    "split_diagonal",
]


def secant_split(int_m, dec_m, sec_ops):
    """``(d_int, d_dec, int_off, dec_off)`` of A, B in the secant layout
    (see :func:`MCEq.operator_assembly.compile_operator`)."""
    return compile_operator(int_m, dec_m, sec_ops).split
