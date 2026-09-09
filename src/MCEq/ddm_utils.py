"""Compatibility exports from :mod:`MCEq.models.ddm.ddm_utils`.

``MCEq.ddm_utils`` stays for tests that patch or reach these helpers by the
flat path. Private helpers remain available for named imports in
``tests/test_ddm.py``; `fmteb` and `gen_matrix_variations` are the pinned
surface names.
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
