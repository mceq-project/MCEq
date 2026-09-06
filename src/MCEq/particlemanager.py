"""Compat shim: the species layer moved to :mod:`MCEq.species`.

Phase 4 (§13.2 item 6) split the module into ``species/constants.py``
(``_pdata``, ``ATOMIC_MASS_UNIT_G``, ``_pname``), ``species/particle.py`` and
``species/manager.py``. This module re-exports the pinned surface (D14, Phase 7
deletes it). ``_pdata`` re-binds the *same* table object the species modules
use, so the single-``PYTHIAParticleData`` invariant stays observable from the
flat path (``test_ddm_and_particlemanager_share_pdata``).
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
#: ``docs/api-reference/particlemanager.rst`` as before the move. Exactly the
#: public surface the Phase 4 contract records (privates are re-exported above
#: for their named importers but stay out of the page, as before); pinned by
#: ``test_particlemanager_shim_dunder_all_is_the_recorded_surface``.
__all__ = [
    "ATOMIC_MASS_UNIT_G",
    "MCEqParticle",
    "ParticleManager",
    "backward_compatible_namestr",
]
