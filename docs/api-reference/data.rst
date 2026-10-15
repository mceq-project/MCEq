.. _data-package:

************************************************
data (:mod:`MCEq.data`)
************************************************
.. currentmodule:: MCEq.data


The tabulated data in MCEq is handled by :class:`MCEq.data.HDF5Backend`.
The HDF5 file contains densely packed data, where matrices are stored as vectors of a sparse CSR data structure.
Index dictionaries and other metadata are stored as attributes.
The other classes of this module know how to interact with the backend
and provide an intermediate step to the :class:`MCEq.particlemanager.ParticleManager` that propagates data further to the :class:`MCEq.particlemanager.MCEqParticle` objects.
:mod:`MCEq.data.download` fetches the database files listed in the data manifest
(see :doc:`../data_installation`).

Reference/API
=============
.. automodapi:: MCEq.data
  :inherited-members:

.. automodule:: MCEq.data.download
   :members: ensure_db_available, ensure_package, ensure_manifest, local_manifest, load_manifest, PackageFetchError
