"""Compatibility facade for :mod:`MCEq.driver.mceq_run` (plan §8.6).

``MCEq.core`` is a *permanent* facade: the v1.4.1 import path and every name
it has always exposed keep resolving here. The implementation moved to
:mod:`MCEq.driver.mceq_run` in Phase 6 commit 6; this module re-exports it,
plus the compat names documented on the re-export lines below.
"""

from MCEq.driver.mceq_run import MCEqRun
from MCEq.driver.results import MCEqBatchResult  # noqa: F401
from MCEq.operators.matrix_builder import MatrixBuilder  # noqa: F401

__all__ = ["MCEqRun", "MCEqBatchResult", "MatrixBuilder"]
