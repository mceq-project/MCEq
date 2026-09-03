"""Compatibility re-export of :mod:`MCEq.operators.secant`.

The sec(theta) coupling operator lives in the ``operators`` layer of the
package. The names below are the surface this module defined before the move,
so code importing them from here keeps working: silent in this release, a
``DeprecationWarning`` in the next, removed after (refactoring plan, D14).
"""

from MCEq.operators.secant import build_secant_kernel_ops, secant_coupling_matrix

__all__ = [
    "build_secant_kernel_ops",
    "secant_coupling_matrix",
]
