"""Regression test for GH-149: segfault at interpreter exit when the
Accelerate (spacc) solver is used on macOS.

The crash only manifests when the process exits, so it cannot be caught by a
normal pytest assertion.  We run a minimal MCEq script in a fresh subprocess
and assert it exits with code 0.  On non-macOS platforms the spacc module is
never loaded and the test simply verifies a clean exit with the default solver.
"""

import subprocess
import sys

import pytest

from MCEq import config

_SCRIPT = """
import MCEq.config as config
config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"

import crflux.models as pm
from MCEq.core import MCEqRun

mceq = MCEqRun(
    interaction_model="SIBYLL21",
    theta_deg=0.0,
    primary_model=(pm.HillasGaisser2012, "H3a"),
)
mceq.solve()
"""

# Variant that forces the accelerate ETD2 (spacc) kernel explicitly.
_SCRIPT_ACCELERATE = """
import MCEq.config as config
config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
config.kernel_config = "accelerate_etd2"

import crflux.models as pm
from MCEq.core import MCEqRun

mceq = MCEqRun(
    interaction_model="SIBYLL21",
    theta_deg=0.0,
    primary_model=(pm.HillasGaisser2012, "H3a"),
)
mceq.solve()
"""

# Multiple solve() calls reuse / recreate SpaccMatrix objects — exercises the
# re-allocation path and the __del__ guard when matrices are replaced.
_SCRIPT_MULTI_SOLVE = """
import MCEq.config as config
config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
config.kernel_config = "accelerate_etd2"

import crflux.models as pm
from MCEq.core import MCEqRun

mceq = MCEqRun(
    interaction_model="SIBYLL21",
    theta_deg=0.0,
    primary_model=(pm.HillasGaisser2012, "H3a"),
)
for angle in (0.0, 30.0, 60.0):
    mceq.set_theta_deg(angle)
    mceq.solve()
"""

# Explicitly create and delete SpaccMatrix objects to exercise __del__ and the
# double-free guard without going through MCEqRun.
_SCRIPT_SPACC_MATRIX_DEL = """
import MCEq.config as config
config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"

import numpy as np
from scipy.sparse import eye
import MCEq.spacc as spacc

matrices = [spacc.SpaccMatrix(eye(5, format="coo")) for _ in range(5)]
# Explicit deletion; the GC will also call __del__ on gc collect/exit
for m in matrices:
    del m

# Create and destroy one more to make sure store slots were freed
m2 = spacc.SpaccMatrix(eye(5, format="coo"))
del m2
"""


# Tests that filling all SIZE_MSTORE slots and then trying one more raises.
# Must run in a subprocess so the matrix store starts completely empty.
_SCRIPT_SPACC_STORE_FULL = """
import sys
from scipy.sparse import eye
import MCEq.spacc as spacc

matrices = []
for _ in range(10):
    matrices.append(spacc.SpaccMatrix(eye(3, format="coo")))

try:
    extra = spacc.SpaccMatrix(eye(3, format="coo"))
    print("ERROR: expected exception not raised", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    if "Matrix creation failed" not in str(e):
        print(f"ERROR: unexpected exception message: {e}", file=sys.stderr)
        sys.exit(1)
# Expected exception raised; exit 0.
"""


def _run(script):
    """Run *script* in a fresh interpreter and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def _assert_clean(result):
    assert result.returncode == 0, (
        f"Process exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_clean_exit():
    """Process must exit with code 0 (no segfault or unhandled exception)."""
    _assert_clean(_run(_SCRIPT))


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_accelerate_explicit_exit():
    """Forcing kernel_config='accelerate' must not crash on exit (GH-149)."""
    _assert_clean(_run(_SCRIPT_ACCELERATE))


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_accelerate_multi_solve_exit():
    """Multiple solve() calls (matrix reinit) must not crash on exit."""
    _assert_clean(_run(_SCRIPT_MULTI_SOLVE))


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_matrix_explicit_del_exit():
    """Explicit deletion of SpaccMatrix objects must not cause a double-free crash."""
    _assert_clean(_run(_SCRIPT_SPACC_MATRIX_DEL))


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_spacc_matrix_store_full():
    """Overflowing SIZE_MSTORE (10) must raise; subprocess ensures a clean store."""
    _assert_clean(_run(_SCRIPT_SPACC_STORE_FULL))
