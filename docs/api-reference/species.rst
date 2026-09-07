.. _species:

************************************************
species (:mod:`MCEq.species`)
************************************************
.. currentmodule:: MCEq.species


The species layer: particle identities, channel bookkeeping, and the
tracking registry. In v1 this was the flat ``MCEq.particlemanager``
module, which remains importable as a re-export shim until 2.0 (see
:doc:`/migration_v2_layout`). ``constants`` holds the shared
``PYTHIAParticleData`` singleton.

Reference/API
=============
.. automodule:: MCEq.species.manager
   :members:
.. automodule:: MCEq.species.particle
   :members:
.. automodule:: MCEq.species.constants
   :members:
