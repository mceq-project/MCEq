"""Columns end at the ground of the atmosphere model, never below it."""

import pytest

import MCEq.geometry.density_profiles as dp
from MCEq import config

SOUTH_POLE_CM = 2834.0e2


@pytest.fixture
def h_obs_config():
    saved = config.h_obs
    yield
    config.h_obs = saved


@pytest.mark.parametrize(
    ("location", "season"),
    [("SouthPole", "June"), ("PL_SouthPole", "January"), ("SDR_SouthPole", "October")],
)
def test_south_pole_sets_end_at_the_ice(location, season):
    atm = dp.CorsikaAtmosphere(location, season)
    assert atm.h_ground == atm.geom.h_obs == SOUTH_POLE_CM
    atm.set_theta(0.0)
    assert 680.0 < atm.max_X < 710.0


def test_sea_level_sets_keep_sea_level():
    atm = dp.CorsikaAtmosphere("USStd", None)
    assert atm.h_ground == atm.geom.h_obs == 0.0


def test_observation_level_below_the_ground_raises():
    atm = dp.CorsikaAtmosphere("SDR_SouthPole", "October")
    with pytest.raises(ValueError, match="below the ground"):
        atm.set_h_obs(1000.0e2)
    atm.set_h_obs(3000.0e2)  # above the ice is allowed
    assert atm.geom.h_obs == 3000.0e2


def test_explicit_config_level_below_the_ground_raises(h_obs_config):
    config.h_obs = 0.0
    assert dp.CorsikaAtmosphere("USStd", None).geom.h_obs == 0.0
    with pytest.raises(ValueError, match="below the ground"):
        dp.CorsikaAtmosphere("PL_SouthPole", "January")


def test_default_level_follows_the_ground_on_a_switch():
    atm = dp.CorsikaAtmosphere("SDR_SouthPole", "June")
    atm.set_theta(0.0)
    atm.init_parameters("USStd", None)
    assert atm.geom.h_obs == 0.0
    atm.init_parameters("SDR_SouthPole", "June")
    assert atm.geom.h_obs == SOUTH_POLE_CM


def test_explicit_level_survives_a_switch():
    atm = dp.CorsikaAtmosphere("USStd", None)
    atm.set_h_obs(3000.0e2)
    atm.init_parameters("SDR_SouthPole", "June")
    assert atm.geom.h_obs == 3000.0e2


def test_raised_level_keeps_columns_downgoing():
    """Splines need heights that fall along the column, so the dip below the
    horizon of a raised level is refused instead of failing in the fit."""
    atm = dp.CorsikaAtmosphere("SDR_SouthPole", "June")
    assert atm.max_theta == 90.0
    with pytest.raises(Exception, match="not in allowed range"):
        atm.set_theta(91.0)


def test_set_h_obs_at_vertical_rebuilds_the_splines():
    atm = dp.CorsikaAtmosphere("USStd", None)
    atm.set_theta(0.0)
    sea_level = atm.max_X
    atm.set_h_obs(2000.0e2)
    assert atm.max_X < 0.9 * sea_level


def test_icecube_ground_is_the_glacier_surface():
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")
    assert atm.h_ground == dp.ICECUBE["surface_elevation_m"] * 1e2
    with pytest.raises(ValueError, match="below the ground"):
        atm.set_h_obs(0.0)
