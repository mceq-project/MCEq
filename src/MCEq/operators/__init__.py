"""Physics and matrices to a compiled cascade operator.

The layer between the species definition and the solvers, host-only and
backend-agnostic:

* :mod:`MCEq.operators.compiled` -- :func:`~MCEq.operators.compiled.compile_operator`
  turns the constant sparse matrices ``A = int_m`` and ``B = dec_m`` into one
  immutable :class:`~MCEq.operators.compiled.CompiledOperator`: the diagonal /
  off-diagonal split in the state layout the ETD2RK step loop wants.
* :mod:`MCEq.operators.secant` -- the constant Hankel-space representation of
  the sec(theta) path elongation, which that layout couples exactly.

A backend of :mod:`MCEq.solvers` places the compiled operator on its library
handles or device and executes the step loop against it.

This module re-exports nothing, so ``import MCEq.operators`` pulls in neither
numpy nor scipy; a caller imports the module that owns what it needs.
"""
