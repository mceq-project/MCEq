"""Compat shim: the DDM helpers moved to :mod:`MCEq.models.ddm.ddm_utils`.

Phase 4 (§13.2 item 7). ``MCEq.ddm_utils`` stays for tests that patch or reach
these helpers by the flat path. Phase 7 deletes the shim (D14). Private helpers
are re-exported here because `tests/test_ddm.py` and the B8 pin import them by
name; `fmteb` and `gen_matrix_variations` are the pinned surface names.
"""

from MCEq.models.ddm.ddm_utils import (  # noqa: F401
    _eval_spline,
    _gen_averaged_dndx,
    _gen_dndx,
    _generate_DDM_matrix,
    _spline_min_max_at_knot,
    fmteb,
    gen_matrix_variations,
)
