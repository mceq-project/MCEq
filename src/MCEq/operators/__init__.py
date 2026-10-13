"""Physics and matrices to a compiled cascade operator.

The layer between the species definition and the solvers, host-only and
backend-agnostic:

* :mod:`MCEq.operators.matrix_builder` --
  :class:`~MCEq.operators.matrix_builder.MatrixBuilder` fills the interaction
  and decay matrices ``int_m`` and ``dec_m``, per (child, parent) channel, from
  the tabulated yields a particle manager carries.
* :mod:`MCEq.operators.loss_stencil` -- the finite-difference families for
  ``d/du``, ``u = ln E``, behind the continuous-loss band, and the exactness
  condition each one imposes.
* :mod:`MCEq.operators.legendre` -- the Legendre angular modes of the 2D
  transport: moments, the couplings ``M[g]``, backward removal and readout.
* :mod:`MCEq.operators.scattering` -- the per-mode multiple Coulomb
  scattering rates of muons in the 2D transport.
* :mod:`MCEq.operators.compiled` -- :func:`~MCEq.operators.compiled.compile_operator`
  turns the constant sparse matrices ``A = int_m`` and ``B = dec_m`` into one
  :class:`~MCEq.operators.compiled.CompiledOperator`, to be treated as
  read-only while cached by a run: the diagonal /
  off-diagonal split in the state layout the ETD2RK step loop wants.
* :mod:`MCEq.operators.secant` -- the constant Legendre-mode representation
  of the sec(theta) path elongation, which that layout couples exactly.

A backend of :mod:`MCEq.solvers` places the compiled operator on its library
handles or device and executes the step loop against it.

This module re-exports nothing, so ``import MCEq.operators`` pulls in neither
numpy nor scipy; a caller imports the module that owns what it needs.
"""
