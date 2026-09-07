"""The species layer: particle identities, channels, and the registry.

The layer the flat ``MCEq.particlemanager`` shim re-exports (until Phase 7).
``constants`` holds the shared ``PYTHIAParticleData`` singleton and the mass
constant, ``particle`` the :class:`MCEqParticle` bookkeeping class, ``manager``
the :class:`ParticleManager` registry. The planned ``spectra.py`` /
``tracking.py`` method splits are deferred: extracting methods from the two
classes cannot be done without changing them.
"""
