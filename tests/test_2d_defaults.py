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


def test_secant_auto_cap_clips_at_50():
    """The 'auto' cap is 90 - zenith + 5 snapped to 5-deg steps, clipped
    to [50, 75]: below ~45 deg the S_P eigenbasis is numerically
    defective (near-nilpotent coupling), so inclined zeniths run at 50."""
    def cap_for(theta_z):
        stub = SimpleNamespace(
            density_model=SimpleNamespace(theta_deg=theta_z))
        return MCEqRun._secant_theta_cap_deg(stub)

    saved = config.secant_theta_cap_deg
    config.secant_theta_cap_deg = "auto"
    try:
        assert cap_for(0.0) == 75.0
        assert cap_for(30.0) == 65.0
        assert cap_for(45.0) == 50.0
        assert cap_for(60.0) == 50.0   # not 35 — defective below ~45
        assert cap_for(85.0) == 50.0
        config.secant_theta_cap_deg = 62.5
        assert cap_for(60.0) == 62.5   # explicit float passes through
    finally:
        config.secant_theta_cap_deg = saved


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
