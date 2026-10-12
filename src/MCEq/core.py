"""Compatibility import path for :mod:`MCEq.driver.mceq_run`.

Exports :class:`MCEqRun`, :class:`MCEqBatchResult`, and
:class:`MatrixBuilder` through the ``MCEq.core`` import path; the
implementations live in :mod:`MCEq.driver.mceq_run`,
:mod:`MCEq.driver.results`, and :mod:`MCEq.operators.matrix_builder`.
""" #anfe: prose verbose. Only the first line is enough, paragraph says nothing.

from MCEq.driver.mceq_run import MCEqRun
from MCEq.driver.results import MCEqBatchResult  # noqa: F401
from MCEq.operators.matrix_builder import MatrixBuilder  # noqa: F401

__all__ = ["MCEqRun", "MCEqBatchResult", "MatrixBuilder"] #anfe: Batch result and MatrixBuilder were never in pre v2, so no need to expose them here.
