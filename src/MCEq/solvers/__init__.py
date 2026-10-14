"""ETD2RK cascade solvers and their sparse/device backends."""

from MCEq.operators.compiled import (
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
    "solve_etd2",
    "split_diagonal",
]
