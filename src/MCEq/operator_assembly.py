"""Compatibility re-export of :mod:`MCEq.operators.compiled`.

Operator assembly lives in the ``operators`` layer of the package. The names
below are the surface this module defined before the move, so code importing
them from here keeps working: silent in this release, a
``DeprecationWarning`` in the next, removed after (refactoring plan, D14).
"""

from MCEq.operators.compiled import (
    CompiledOperator,
    compile_operator,
    identity_layout,
    secant_coupling,
    secant_layout,
    split_diagonal,
)

__all__ = [
    "CompiledOperator",
    "compile_operator",
    "identity_layout",
    "secant_coupling",
    "secant_layout",
    "split_diagonal",
]
