"""Regression test for GH-149: segfault at interpreter exit when the
Accelerate (spacc) solver is used on macOS.

The crash only manifests when the process exits, so it cannot be caught by a
normal pytest assertion.  We run a minimal MCEq script in a fresh subprocess
and assert it exits with code 0.  On non-macOS platforms the spacc module is
never loaded and the test simply verifies a clean exit with the default solver.
"""

import subprocess
import sys

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


def test_clean_exit():
    """Process must exit with code 0 (no segfault or unhandled exception)."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Process exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
