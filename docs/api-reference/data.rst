.. _data-package:
************************************************
data (:mod:`MCEq.data`)
************************************************
.. currentmodule:: MCEq.data


The tabulated data in MCEq is handled by :class:`MCEq.data.HDF5Backend`. 
The HDF5 file densly packed data, where matrices are stored as vectors of a sparse CSR data structure. 
Index dictionaries and other metadata are stored as attributes. 
The other classes pf this module know how to interact with the backend
and provide an intermediate step to the :class:`MCEq.particlemanager.ParticleManager` that propagates data further to the :class:`MCEq.particlemanager.MCEqParticle` objects.

Reference/API
=============
.. automodapi:: MCEq.data
  :inherited-members:
  :skip: EnergyGrid

.. autoclass:: MCEq.data.energy_grid.EnergyGrid
   :show-inheritance:

   ``EnergyGrid`` is a namedtuple (fields ``c``, ``b``, ``w``, ``d``) —
   documented explicitly because automodsumm would otherwise document
   the namedtuple fields and inherited tuple methods as if they were
   class attributes.
