"""Compatibility exports from :mod:`MCEq.models.ddm.ddm`.

``MCEq.ddm`` stays as the import path used by existing user code and pinned by
the surface test.

``_pdata`` / ``_DDMEntry`` / ``_DDMChannel`` / ``_DEFAULT_DDM_FILE`` are
private bindings imported by ``tests/conftest.py``, the B8 pin, and
``test_ddm_and_particlemanager_share_pdata``; the re-export keeps them bound to
the *same* objects the implementation module uses, so the single-
``PYTHIAParticleData`` invariant stays observable from the old path.
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
