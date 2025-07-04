.. _data:
************************************************
solvers (:mod:`MCEq.solvers`)
************************************************
.. currentmodule:: MCEq.solvers


The module contains functions which are called by :func:`MCEq.core.MCEqRun.solve()` method.

The implementation is a simple Forward-Euler stepper. The stability is under control
since the smallest Eigenvalues are known a priory. The step size is "adaptive", but it
is deterministic and known before the integration starts.

The steps that each solver routine does are:

.. math::

  \Phi_{i + 1} = \Delta X_i\boldsymbol{M}_{int} \cdot \Phi_i  + \frac{\Delta X_i}{\rho(X_i)}\cdot\boldsymbol{M}_{dec} \cdot \Phi_i)

with

.. math::
  \boldsymbol{M}_{int} = (-\boldsymbol{1} + \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}
  :label: int_matrix

and

.. math::
  \boldsymbol{M}_{dec} = (-\boldsymbol{1} + \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}.
  :label: dec_matrix

As one can easily see, each step can be represented by two sparse `gemv` calls and one vector addition.
This is what happens in the MKL and CUDA functions below.

The fastest solver is using NVidia's cuSparse library provided via `the cupy matrix library <https://cupy.chainer.org>`_. 
Intel MKL is recommended for Intel CPUs, in particular since MKL is using AVX instructions.
The plain numpy solver is for compatibility and hacking, but not recommended for general use.

Reference/API
=============
.. automodapi:: MCEq.solvers
  :inherited-members:
