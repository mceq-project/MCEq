"""Particle classes and particle data from :mod:`MCEq.species`."""

# anfe: check if this entire module can be removed. I assume that only the BSM example showed how to use the particle manager.
from MCEq.misc import average_A_target, getAZN, info, print_in_rows  # noqa: F401
from MCEq.species.constants import (  # noqa: F401
    ATOMIC_MASS_UNIT_G,
    _pdata,
    _pname,
    backward_compatible_namestr,
)
from MCEq.species.manager import ParticleManager  # noqa: F401
from MCEq.species.particle import MCEqParticle  # noqa: F401

#: Public names; the private ones above are importable but not listed.
__all__ = [
    "ATOMIC_MASS_UNIT_G",
    "MCEqParticle",
    "ParticleManager",
    "backward_compatible_namestr",
]
