.. _operators:

************************************************
operators (:mod:`MCEq.operators`)
************************************************
.. currentmodule:: MCEq.operators


The layer between the species definition and the solvers:
:class:`~MCEq.operators.matrix_builder.MatrixBuilder` fills the interaction
and decay matrices, :mod:`MCEq.operators.compiled` assembles them into the
operator the ETD2RK step loop runs against, :mod:`MCEq.operators.secant`
builds the sec(θ) mode coupling and :mod:`MCEq.operators.stiffness` the
continuous-loss step cap. The mathematics is on :ref:`solver-mathematics`.

Reference/API
=============
.. automodapi:: MCEq.operators.compiled
  :include-all-objects:
  :no-inheritance-diagram:
  :skip: SimpleNamespace

.. automodapi:: MCEq.operators.secant
  :no-inheritance-diagram:
  :skip: info

.. automodapi:: MCEq.operators.stiffness
  :include-all-objects:
  :no-inheritance-diagram:

.. automodapi:: MCEq.operators.loss_stencil
  :no-inheritance-diagram:

.. automodapi:: MCEq.operators.scattering
  :no-inheritance-diagram:
