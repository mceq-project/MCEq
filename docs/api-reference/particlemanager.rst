.. _data:
************************************************
particlemanager (:mod:`MCEq.particlemanager`)
************************************************
.. currentmodule:: MCEq.particlemanager


The :class:`MCEq.particlemanager.ParticleManager` handles the bookkeeping of :class:`MCEq.particlemanager.MCEqParticle`'s. 
It feeds the parameterizations of interactions and 
decays from :mod:`MCEq.data` into the corresponding variables and validates certain relations. 
The construction of the interaction and decay matrices proceeds by iterating over the particles 
in :class:`MCEq.particlemanager.ParticleManager`, querying the interaction and decay yields for child particles.
Therefore, there is usually no need to directly access any of the classes in :mod:`MCEq.data`.


Reference/API
=============
.. automodapi:: MCEq.particlemanager
  :inherited-members:
