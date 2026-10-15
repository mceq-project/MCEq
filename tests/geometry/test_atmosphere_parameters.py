import numpy as np


def test_corsika_atmosphere_param_shapes():
    from MCEq.geometry.atmosphere_parameters import _cosika_atmosphere_params

    required_keys = {"_aatm", "_batm", "_catm", "_thickl", "_hlay"}

    for key, param in _cosika_atmosphere_params.items():
        # key is (location, season)
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(param, dict)

        # must contain all required keys
        assert set(param.keys()) == required_keys

        # each param must be an ndarray of shape (5,)
        for subkey in required_keys:
            val = param[subkey]
            assert isinstance(val, np.ndarray), f"{key} → {subkey} is not array"
            assert val.shape == (5,), f"{key} → {subkey} has shape {val.shape}"


def test_month_to_day_of_year():
    from MCEq.geometry.atmosphere_parameters import MONTH_TO_DAY_OF_YEAR

    assert len(MONTH_TO_DAY_OF_YEAR) == 12

    # Basic month sanity
    assert MONTH_TO_DAY_OF_YEAR["January"] == 1
    assert MONTH_TO_DAY_OF_YEAR["February"] > MONTH_TO_DAY_OF_YEAR["January"]
    assert MONTH_TO_DAY_OF_YEAR["December"] == 335

    # Monotonic increasing
    vals = list(MONTH_TO_DAY_OF_YEAR.values())
    assert vals == sorted(vals)


def test_locations():
    from MCEq.geometry.atmosphere_parameters import LOCATIONS

    for loc, (lon, lat, h_cm) in LOCATIONS.items():
        assert isinstance(lon, float)
        assert isinstance(lat, float)
        assert isinstance(h_cm, float)
        assert -180.0 <= lon <= 180.0, f"{loc} has invalid longitude"
        assert -90.0 <= lat <= 90.0, f"{loc} has invalid latitude"
        assert h_cm > 0.0, f"{loc} has non-positive altitude"


def test_default_geophysical_constants():
    from MCEq.geometry.atmosphere_parameters import get_nrlmsise00_defaults

    f107a, f107, ap = get_nrlmsise00_defaults()

    assert f107a == 150.0
    assert f107 == 150.0
    assert ap == 4.0


def test_get_location_data():
    from MCEq.geometry.atmosphere_parameters import get_location_data

    loc = get_location_data("SouthPole")
    assert isinstance(loc, tuple)
    assert len(loc) == 3
    lon, lat, h_cm = loc
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90
    assert h_cm > 0

    assert get_location_data("NotARealPlace") is None


def test_get_month_day_of_year():
    from MCEq.geometry.atmosphere_parameters import get_month_day_of_year

    assert get_month_day_of_year("January") == 1
    assert get_month_day_of_year("December") == 335
    assert get_month_day_of_year("NotAMonth") is None


def test_get_day_time_seconds():
    from MCEq.geometry.atmosphere_parameters import get_day_time_seconds

    assert get_day_time_seconds("day") == 43200.0
    assert get_day_time_seconds("night") == 0.0
    assert get_day_time_seconds("dusk") is None
