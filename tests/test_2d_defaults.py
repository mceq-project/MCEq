"""Production defaults for 2D databases: the "auto" sec(theta) transport
tri-state.

The tri-state resolution is a pure function of (config flag, is_2d) and is
tested unbound with a stub. End-to-end secant solves are covered by the
2026-08-10 secant runs, not the unit suite (the operator build is
minutes-scale without its disk cache).

(The former FLUKA-only 2D model restriction and its refusal tests were
removed when runtime per-kappa HE/LE blending on multi-model 2D databases
became the production design, superseding the postponed 1D-HE/2D-LE
kappa-window hybrid.)
"""

from types import SimpleNamespace

import pytest

from MCEq import config
from MCEq.core import MCEqRun


def _mode_for(flag, is_2d):
    stub = SimpleNamespace(_mceq_db=SimpleNamespace(is_2d=is_2d))
    saved = config.secant_theta_transport
    config.secant_theta_transport = flag
    try:
        return MCEqRun._secant_mode(stub)
    finally:
        config.secant_theta_transport = saved


def test_secant_cap_validation():
    """The cap is a plain float in [50, 90): sec(theta) diverges at 90
    deg, and below ~45-50 deg the S_P eigenbasis is numerically defective
    (near-nilpotent coupling), so out-of-range values are rejected."""
    stub = SimpleNamespace()

    def cap_for(value):
        config.secant_theta_cap_deg = value
        return MCEqRun._secant_theta_cap_deg(stub)

    saved = config.secant_theta_cap_deg
    try:
        assert cap_for(75.0) == 75.0  # the default
        assert cap_for(62.5) == 62.5  # any float in range passes through
        assert cap_for(50.0) == 50.0  # lower edge included
        for bad in (45.0, 90.0, 120.0):
            with pytest.raises(ValueError):
                cap_for(bad)
    finally:
        config.secant_theta_cap_deg = saved


def test_secant_cap_default_is_75():
    assert config.secant_theta_cap_deg == 75.0


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
