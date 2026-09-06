"""Compat shim: the DDM moved to :mod:`MCEq.models.ddm.ddm`.

Phase 4 (§13.2 item 7) moved it under ``models``. ``MCEq.ddm`` stays as the
entry point the surface test pins and ``MCEq.core`` reaches through, and it is
also how existing user code imports the model. Phase 7 deletes the shim (D14).

``_pdata`` / ``_DDMEntry`` / ``_DDMChannel`` / ``_DEFAULT_DDM_FILE`` are the
private module bindings ``tests/conftest.py``, the B8 pin and
``test_ddm_and_particlemanager_share_pdata`` reach through the flat name; the
re-export keeps them bound to the *same* objects the moved module uses, so the
single-``PYTHIAParticleData`` and shared-spline-cache invariants stay observable
from the old path.
"""

from MCEq.models.ddm.ddm import (  # noqa: F401
    _DEFAULT_DDM_FILE,
    DataDrivenModel,
    DDMSplineDB,
    _DDMChannel,
    _DDMEntry,
    _pdata,
    isospin_partners,
    isospin_symmetries,
)
