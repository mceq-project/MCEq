"""set_zenith_azimuth / set_theta_deg deprecation and the MSIS21 opt-in smoke test.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)

import MCEq.geometry.density_profiles as dprof

# ---------------------------------------------------------------------------
# set_zenith_azimuth / set_theta_deg deprecation tests
# ---------------------------------------------------------------------------


def test_set_zenith_azimuth(mceq_sib21):
    """set_zenith_azimuth should set zenith and keep integration_path invalidated."""
    # Ensure an EarthsAtmosphere model is active; a previous test may have left
    # a GeneralizedTarget which does not support angle settings.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))
    mceq_sib21.set_zenith_azimuth(30.0)
    assert mceq_sib21.density_model.theta_deg == 30.0

    # Calling again with same angle should skip recalculation (cached)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mceq_sib21.set_zenith_azimuth(30.0)  # should be no-op
    assert mceq_sib21.density_model.theta_deg == 30.0


def test_set_theta_deg_deprecation(mceq_sib21):
    """set_theta_deg must emit a DeprecationWarning."""
    import warnings

    # Ensure an EarthsAtmosphere model is active; a previous test may have left
    # a GeneralizedTarget which does not support angle settings.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mceq_sib21.set_theta_deg(45.0)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "set_zenith_azimuth" in str(w[0].message)
    assert mceq_sib21.density_model.theta_deg == 45.0


def test_set_zenith_azimuth_with_km3net(mceq_sib21):
    """set_zenith_azimuth passes azimuth to location-coupled MSIS models."""
    km3net_atm = dprof.MSIS00KM3NeTCentered("ORCA", season="January")
    mceq_sib21.set_density_model(km3net_atm)

    mceq_sib21.set_zenith_azimuth(60.0, azimuth_deg=0.0)
    assert mceq_sib21.density_model.theta_deg == 60.0
    assert mceq_sib21.density_model.current_impact_latitude is not None
    assert mceq_sib21.density_model.current_impact_longitude is not None

    # Without azimuth → azimuth-averaging
    mceq_sib21.set_zenith_azimuth(60.0)
    assert mceq_sib21.density_model.theta_deg == 60.0
    assert mceq_sib21.density_model.current_impact_latitude is None

    # Restore the session fixture to a neutral atmosphere so other tests are
    # not affected by the KM3NeT density model we set above.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))


# ---------------------------------------------------------------------------
# MSIS21 (NRLMSIS 2.1) opt-in atmosphere — smoke test
# ---------------------------------------------------------------------------


def test_msis21_atmosphere_smoke():
    """MSIS21 is opt-in (needs the pure-Python 'nrlmsis' package).

    Skips cleanly where nrlmsis isn't installed; in CI it is in the test group,
    so this exercises the otherwise-default-off MSIS21 path: build the model and
    check it returns a physically sane, monotonically decreasing density.
    """
    pytest.importorskip("nrlmsis")
    atm = dprof.MSIS21Atmosphere("SouthPole", season="January")
    rho_low = atm.get_density(2.0e5)  # 2 km in cm
    rho_high = atm.get_density(3.0e6)  # 30 km in cm
    assert rho_low > 0.0
    assert rho_high > 0.0
    assert rho_high < rho_low  # density falls with altitude
