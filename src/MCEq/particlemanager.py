"""Compatibility exports from :mod:`MCEq.species`.

The species layer lives in ``species/constants.py``, ``species/particle.py``
and ``species/manager.py``. ``_pdata`` is the same particle-table object used
by the species and DDM implementations; see
``test_ddm_and_particlemanager_share_pdata``.
"""

from MCEq.misc import average_A_target, getAZN, info, print_in_rows  # noqa: F401
from MCEq.species.constants import (  # noqa: F401
    ATOMIC_MASS_UNIT_G,
    _pdata,
    _pname,
    backward_compatible_namestr,
)
from MCEq.species.manager import ParticleManager  # noqa: F401
from MCEq.species.particle import MCEqParticle  # noqa: F401

#: automodapi drops names whose ``__module__`` is not the page's module unless
#: ``__all__`` claims them; the shim must keep rendering the same members on
#: ``docs/api-reference/particlemanager.rst`` as before the move. ``__all__``
#: preserves the public objects shown by API documentation; private names are
#: re-exported above for their named importers but stay out of the list;
#: pinned by
#: ``test_particlemanager_shim_dunder_all_is_the_recorded_surface``.
__all__ = [
    "ATOMIC_MASS_UNIT_G",
    "MCEqParticle",
    "ParticleManager",
    "backward_compatible_namestr",
]
