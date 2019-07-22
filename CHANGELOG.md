Changes in first release version of MCEq from release candidate 1:

Version 1.0.0:

General remark::
    This is a major rewrite of the MCEq core code. Mostly obsolete stuff is removed.
    The interfaces to the various data sources are handled in a different way via the
    container 'ParticleManager'. Each MCEqParticle object knows its properties, for
    instance if it can decay or interact, and if requires to be included in continuous
    losses. The matrix generation could be simplified a lot and the class MCEqRun became
    what it was meant to be - a user interface. Several experimental and less successful
    solver implementations have been removed. The splitting between energy and time solvers
    is not necessary anymore since the only successful energy (derivative) solver is the
    one based on DifferentialOperators that are not part of the "C Matrix". With the new
    structure it became simpler to exchange physical models or track particle decays of
    any kind.

