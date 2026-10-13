"""Production defaults for 2D databases: the "auto" sec(theta) transport
tri-state.

The tri-state resolution is a pure function of (config flag, is_2d) and is
tested unbound with a stub. End-to-end secant solves on the 2D CI database
are in ``tests/test_solvers_2d.py``.

(The former FLUKA-only 2D model restriction and its refusal tests were
removed when runtime per-kappa HE/LE blending on multi-model 2D databases
became the production design, superseding the postponed 1D-HE/2D-LE
kappa-window hybrid.)
"""

import pytest

from MCEq import config


def _mode_for(flag, is_2d):
    saved = config.secant_theta_transport
    config.secant_theta_transport = flag
    try:
        return config.secant_mode(is_2d)
    finally:
        config.secant_theta_transport = saved


@pytest.mark.parametrize("snapshot", [False, True], ids=["module", "RunConfig"])
def test_secant_cap_validation(snapshot):
    """The cap is a plain float in [0, 90): sec(theta) diverges at 90 deg,
    and the Legendre coupling is well defined down to 0 (no elongation).
    The module helper and a RunConfig snapshot apply the same rule."""
    from MCEq.config.run import RunConfig

    def cap_for(value):
        config.secant_theta_cap_deg = value
        if snapshot:
            return RunConfig.snapshot().secant_theta_cap()
        return config.secant_theta_cap()

    saved = config.secant_theta_cap_deg
    try:
        assert cap_for(75.0) == 75.0  # the default
        assert cap_for(62.5) == 62.5  # any float in range passes through
        for small in (0.0, 1.0, 10.0, 45.0):  # lower edge included
            assert cap_for(small) == small
        for bad in (-1.0, 90.0, 120.0):
            with pytest.raises(ValueError, match=r"\[0, 90\)"):
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
