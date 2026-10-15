.. _operators:

************************************************
operators (:mod:`MCEq.operators`)
************************************************
.. currentmodule:: MCEq.operators


The layer between the species definition and the solvers:
:class:`MCEq.core.MatrixBuilder` (module ``MCEq.operators.matrix_builder``,
documented with :mod:`MCEq.core`) fills the interaction and decay matrices,
:mod:`MCEq.operators.compiled` assembles them into the operator the ETD2RK
step loop runs against, :mod:`MCEq.operators.legendre` holds the Legendre
angular modes of the 2D transport, :mod:`MCEq.operators.secant` builds the
sec(θ) mode coupling, :mod:`MCEq.operators.loss_stencil` the
continuous-loss stencils and :mod:`MCEq.operators.scattering` the muon
scattering rates. The step caps are in :mod:`MCEq.solvers.path`. The math is
on :ref:`solver-math`.

Reference/API
=============
.. automodapi:: MCEq.operators.compiled
  :include-all-objects:
  :no-inheritance-diagram:
  :skip: SimpleNamespace

.. automodapi:: MCEq.operators.legendre
  :no-inheritance-diagram:

.. automodapi:: MCEq.operators.secant
  :no-inheritance-diagram:

.. automodapi:: MCEq.operators.loss_stencil
  :no-inheritance-diagram:

.. automodapi:: MCEq.operators.scattering
  :no-inheritance-diagram:
