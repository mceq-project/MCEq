"""``MCEqRun.theta_deg`` must follow the angle actually set (bug B13).

It used to keep its constructor value forever: ``set_zenith_azimuth`` drove
``density_model.set_theta`` and never touched the attribute, so any code
reading ``MCEqRun.theta_deg`` after the first change of angle got the wrong
number. The golden ``paths`` section had to label its rows by
``density_model.theta_deg`` and say why, and an external user reading the
documented attribute would have been wrong.

``src/`` *was* misled, contrary to what this file first claimed. The guard at
``core.py:1243`` only tests the attribute for ``None``, but the next line
passes it by value -- ``self.set_zenith_azimuth(self.theta_deg)`` -- so before
the fix, swapping atmosphere or season silently reset the direction to the
constructor angle. That behaviour change is pinned by
:func:`test_swapping_the_atmosphere_keeps_the_current_zenith`.

Kept out of ``tests/test_core.py``, which the refactor otherwise leaves alone:
it has moved only for config-leak repair and formatting (b9f40ec, d6d12bf).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xdist_group("spacc")


def test_theta_deg_tracks_set_zenith_azimuth(mceq_sib21):
    mceq_sib21.set_zenith_azimuth(0.0)
    assert mceq_sib21.theta_deg == 0.0

    for zenith in (30.0, 60.0, 0.0, 85.0):
        mceq_sib21.set_zenith_azimuth(zenith)
        assert mceq_sib21.theta_deg == zenith, (
            f"MCEqRun.theta_deg is {mceq_sib21.theta_deg}, not the {zenith} just "
            "set -- bug B13 is back"
        )
        assert mceq_sib21.theta_deg == mceq_sib21.density_model.theta_deg


def test_theta_deg_tracks_the_deprecated_alias(mceq_sib21):
    with pytest.warns(DeprecationWarning):
        mceq_sib21.set_theta_deg(45.0)
    assert mceq_sib21.theta_deg == 45.0


def test_theta_deg_is_right_even_when_the_recomputation_is_skipped(mceq_sib21):
    """The cache short-circuit returns early; the attribute is set on it too.

    Driving the atmosphere directly puts the two out of step, which is exactly
    the state that made the early return able to leave a stale value behind.
    """
    mceq_sib21.set_zenith_azimuth(60.0)
    mceq_sib21.density_model.set_theta(60.0)  # behind MCEqRun's back
    mceq_sib21.theta_deg = 12345.0  # pretend a stale value survived

    mceq_sib21.set_zenith_azimuth(60.0)  # takes the early return
    assert mceq_sib21.theta_deg == 60.0


def test_a_rejected_zenith_is_not_recorded(mceq_sib21):
    """The attribute must not remember an angle the atmosphere refused.

    The first version of the B13 fix assigned ``theta_deg`` before dispatching
    to ``set_theta``, so an out-of-range angle was still committed. That is
    worse than the bug it replaced: ``set_density_model`` replays
    ``self.theta_deg`` into the next atmosphere, so catching the range error
    and then swapping to a ``max_theta = 180`` model silently solved at an
    angle that was never accepted.
    """
    mceq_sib21.set_zenith_azimuth(30.0)
    assert mceq_sib21.density_model.max_theta == 90.0, (
        "fixture atmosphere must reject upgoing angles for this test to bite"
    )

    with pytest.raises(Exception, match="not in allowed range"):
        mceq_sib21.set_zenith_azimuth(120.0)

    assert mceq_sib21.theta_deg == 30.0, (
        f"MCEqRun.theta_deg is {mceq_sib21.theta_deg}: the rejected 120 deg "
        "was recorded anyway"
    )
    assert mceq_sib21.theta_deg == mceq_sib21.density_model.theta_deg

    mceq_sib21.set_zenith_azimuth(0.0)


def test_swapping_the_atmosphere_keeps_the_current_zenith(mceq_sib21):
    """A consequence of the B13 fix, pinned because it is user-visible.

    ``set_density_model`` ends with ``set_zenith_azimuth(self.theta_deg)``.
    Before B13 that attribute still held the *constructor* angle, so changing
    season or atmosphere silently reset the direction to 0 deg; now it carries
    the angle actually in force. This is the intended behaviour, not a
    side effect -- if it ever reverts, the reset comes back.
    """
    mceq_sib21.set_zenith_azimuth(60.0)
    mceq_sib21.set_density_model(("CORSIKA", ("PL_SouthPole", "January")))

    assert mceq_sib21.theta_deg == 60.0
    assert mceq_sib21.density_model.theta_deg == 60.0

    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))
    mceq_sib21.set_zenith_azimuth(0.0)
