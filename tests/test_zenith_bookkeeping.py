"""``MCEqRun.theta_deg`` must follow the angle actually set (bug B13).

It used to keep its constructor value forever: ``set_zenith_azimuth`` drove
``density_model.set_theta`` and never touched the attribute, so any code
reading ``MCEqRun.theta_deg`` after the first change of angle got the wrong
number. Nothing in ``src/`` was misled -- the one consumer, the guard in
``set_density_model``, only tests for ``None`` -- but the golden ``paths``
section had to label its rows by ``density_model.theta_deg`` and say why, and
an external user reading the documented attribute would have been wrong.

Kept out of ``tests/test_core.py``, which is held byte-unchanged across the
refactor.
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
    """The cache short-circuit returns early; the attribute is set before it.

    Driving the atmosphere directly puts the two out of step, which is exactly
    the state that made the early return able to leave a stale value behind.
    """
    mceq_sib21.set_zenith_azimuth(60.0)
    mceq_sib21.density_model.set_theta(60.0)  # behind MCEqRun's back
    mceq_sib21.theta_deg = 12345.0  # pretend a stale value survived

    mceq_sib21.set_zenith_azimuth(60.0)  # takes the early return
    assert mceq_sib21.theta_deg == 60.0
