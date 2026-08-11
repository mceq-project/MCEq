"""Production defaults for 2D databases: FLUKA-only model selection and
the "auto" sec(theta) transport tri-state.

The model restriction is tested at the HDF5Backend level (no solve); the
tri-state resolution is a pure function of (config flag, is_2d) and is
tested unbound with a stub. End-to-end secant solves are covered by the
2026-08-10 secant runs, not the unit suite (the operator build is
minutes-scale without its disk cache).
"""

import os
from types import SimpleNamespace

import pytest

from MCEq import config
from MCEq.core import MCEqRun
from MCEq.data import HDF5Backend

FN_2D = "mceq_db_URQMD_150GeV_2D.h5"


@pytest.fixture()
def backend_2d():
    if not os.path.exists(os.path.join(config.data_dir, FN_2D)):
        pytest.skip(f"{FN_2D} not available; symlink it into src/MCEq/data/")
    saved = (config.mceq_db_fname, config.restrict_2d_to_fluka)
    config.mceq_db_fname = FN_2D
    config.restrict_2d_to_fluka = True
    try:
        yield HDF5Backend(medium="air")
    finally:
        config.mceq_db_fname, config.restrict_2d_to_fluka = saved


def test_non_fluka_model_on_2d_db_refused(backend_2d):
    with pytest.raises(NotImplementedError, match="FLUKA"):
        backend_2d.interaction_db("SIBYLL23D")


def test_le_blending_on_2d_db_refused():
    if not os.path.exists(os.path.join(config.data_dir, FN_2D)):
        pytest.skip(f"{FN_2D} not available; symlink it into src/MCEq/data/")
    saved = (config.mceq_db_fname, config.restrict_2d_to_fluka)
    config.mceq_db_fname = FN_2D
    config.restrict_2d_to_fluka = True
    try:
        backend = HDF5Backend(medium="air", low_energy_model="FLUKA20251")
        # Both guards fire before any HDF5 group is touched, so the model
        # names need not exist in the validation DB.
        with pytest.raises(NotImplementedError, match="blending"):
            backend.interaction_db("FLUKA2011")
    finally:
        config.mceq_db_fname, config.restrict_2d_to_fluka = saved


def _mode_for(flag, is_2d):
    stub = SimpleNamespace(_mceq_db=SimpleNamespace(is_2d=is_2d))
    saved = config.secant_theta_transport
    config.secant_theta_transport = flag
    try:
        return MCEqRun._secant_mode(stub)
    finally:
        config.secant_theta_transport = saved


def test_secant_mode_resolution():
    # Default "auto": on for 2D, no-op for 1D.
    assert _mode_for("auto", True) == "auto"
    assert _mode_for("auto", False) == "off"
    # Explicit True: required (unsupported paths raise); still off in 1D.
    assert _mode_for(True, True) == "require"
    assert _mode_for(True, False) == "off"
    # Explicit False: off everywhere.
    assert _mode_for(False, True) == "off"
    assert _mode_for(False, False) == "off"
