"""The species layer: particle identities, channels, and the registry.

Plan section 1's ``species/``, landed by §13.2 item 6 from the flat
``MCEq.particlemanager`` (which stays as a re-export shim until Phase 7).
``constants`` holds the shared ``PYTHIAParticleData`` singleton and the mass
constant, ``particle`` the :class:`MCEqParticle` bookkeeping class, ``manager``
the :class:`ParticleManager` registry. The planned ``spectra.py`` /
``tracking.py`` method splits are deferred: extracting methods from the two
classes cannot be done verbatim, and Phase 4 is verbatim moves only.
"""
