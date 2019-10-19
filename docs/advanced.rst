
Advanced documentation
----------------------

The "advanced documentation" is the almost complete documentation of all modules. 

.. contents::
   :local:
   :depth: 2

:mod:`mceq_config` -- default configuration options
...................................................

These are all options MCEq accepts. Usually there is no need to change, except for advanced
scenarios. Check out the file for a better formatted description and some advanced settings
not contained in the list below.

.. automodule:: mceq_config
   :members:

:mod:`MCEq.core` -- Core module
...............................

This module contains the main program features. Instantiating :class:`MCEq.core.MCEqRun`
will initialize the data structures and particle tables, create and fill the
interaction and decay matrix and check if all information for the calculation
of inclusive fluxes in the atmosphere is available.

.. automodule:: MCEq.core
   :members: 

:mod:`MCEq.particlemanager` -- Particle manager
...............................................

The :class:`MCEq.particlemanager.ParticleManager` handles the bookkeeping of
:class:`MCEq.particlemanager.MCEqParticle`s. It feeds the parameterizations of interactions
and decays from :mod:`MCEq.data` into the corresponding variables and validates certain relations.
The construction of the interaction and decay matrices proceeds by iterating over the particles
in :class:`MCEq.particlemanager.ParticleManager`, querying the interaction and decay yields
for child particles. Therefore, there is usually no need to directly access any of the
classes in :mod:`MCEq.data`.

.. automodule:: MCEq.particlemanager
   :members:

:mod:`MCEq.data` -- Data handling
.................................

The tabulated data in MCEq is handled by :class:`HDF5Backend`. The HDF5 file densly packed
data, where matrices are stored as vectors of a sparse CSR data structure. Index dictionaries
and other metadata are stored as attributes. The other classes pf this module know how to
interact with the backend and provide an intermediate step to the :class:`ParticleManager`
that propagates data further to the :class:`MCEqParticle` objects.

.. automodule:: MCEq.data
   :members:

:mod:`MCEq.solvers` -- ODE solver implementations
.................................................

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


.. automodule:: MCEq.solvers
   :members:

Miscellaneous
.............

Different helper functions.

.. automodule:: MCEq.misc
   :members: