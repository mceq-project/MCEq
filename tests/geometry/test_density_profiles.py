import inspect

import numpy as np
import pytest

import MCEq.geometry.density_profiles as dp

corsika_expected = [
    ("USStd", None, (1036.099233683902, 0.00015623258808300557)),
    ("BK_USStd", None, (1033.8094962133184, 0.00015782685585891685)),
    ("Karlsruhe", None, (1055.861981113731, 0.00016209949387937668)),
    ("ANTARES/KM3NeT-ORCA", "Summer", (986.9593811082788, 0.00015529574727367941)),
    ("ANTARES/KM3NeT-ORCA", "Winter", (988.4293864278521, 0.0001589317236294479)),
    ("KM3NeT-ARCA", "Summer", (1032.7184058861765, 0.00016954131888323744)),
    ("KM3NeT-ARCA", "Winter", (1039.3697214845179, 0.00016202068935405075)),
    ("KM3NeT", None, (1018.1547240905948, 0.0001609490344992944)),
    ("SouthPole", "December", (1011.4568036341923, 0.00014626903051217024)),
    ("SouthPole", "June", (1020.3505579524912, 0.00018246219074986874)),
    ("PL_SouthPole", "January", (1019.974568696789, 0.0001464549375212421)),
    ("PL_SouthPole", "August", (1019.9764946890782, 0.0001685608228906579)),
    ("SDR_SouthPole", "January", (1034.0143423913353, 0.00014632473385006882)),
    ("SDR_SouthPole", "February", (1035.9936617195242, 0.00014918253835319762)),
    ("SDR_SouthPole", "March", (1038.9308255875999, 0.00015298022770783904)),
    ("SDR_SouthPole", "April", (1041.9777521676683, 0.0001601901839449753)),
    ("SDR_SouthPole", "May", (1039.907580402069, 0.00016807635328451188)),
    ("SDR_SouthPole", "June", (1037.2666634880595, 0.00017542702214644906)),
    ("SDR_SouthPole", "July", (1035.7659887947702, 0.00018145735342421507)),
    ("SDR_SouthPole", "August", (1034.5825095745786, 0.0001865567510599077)),
    ("SDR_SouthPole", "September", (1033.8370290916944, 0.00018184168516617518)),
    ("SDR_SouthPole", "October", (1300.9045332765688, 0.0001729949911605945)),
    ("SDR_SouthPole", "November", (1021.2313399252452, 0.00016452181236376558)),
    ("SDR_SouthPole", "December", (1028.3233680860317, 0.00015434578972428948)),
]


# Test that all corsika atmospheres are tested
def test_cka_atm_completeness():
    from MCEq.geometry.atmosphere_parameters import list_available_corsika_atmospheres

    missing = []
    expected_entries = {(loc, season) for loc, season, _ in corsika_expected}
    for loc, season in list_available_corsika_atmospheres():
        if (loc, season) not in expected_entries:
            missing.append((loc, season))
    if missing:
        for i, (loc, season) in enumerate(missing):
            # Create reference data
            from MCEq.geometry.density_profiles import CorsikaAtmosphere

            cka_obj = CorsikaAtmosphere(loc, season)
            ref = (float(cka_obj.max_X), float(1.0 / cka_obj.r_X2rho(100.0)))
            missing[i] = (loc, season, ref)

    assert len(missing) == 0, f"Missing tests for {missing}."


ids = [f"{loc}-{season or 'None'}" for loc, season, _ in corsika_expected]


@pytest.mark.parametrize(("loc", "season", "expected"), corsika_expected, ids=ids)
def test_corsika_atm(loc, season, expected):
    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    cka_obj = CorsikaAtmosphere(loc, season)
    assert np.allclose([cka_obj.max_X, 1.0 / cka_obj.r_X2rho(100.0)], expected)


msis00_expected = [
    ("SouthPole", "January", (1022.6914983678925, 0.00014380042112573175)),
    ("Karlsruhe", "January", (1041.2180457811605, 0.00016046129606232836)),
    ("Geneva", "January", (1044.6608866969684, 0.00016063221634835724)),
    ("Tokyo", "January", (1046.427667371285, 0.00016041531186210874)),
    ("GranSasso", "January", (1048.6505423154006, 0.00016107650347480857)),
    ("TelAviv", "January", (1050.6431802896034, 0.00016342084740033518)),
    ("KSC", "January", (1050.2145039327452, 0.00016375664772178006)),
    ("SoudanMine", "January", (1033.3640270683418, 0.00015614485659072835)),
    ("Tsukuba", "January", (1045.785578319159, 0.00015970449150213374)),
    ("LynnLake", "January", (1019.9475650272982, 0.000153212909250962)),
    ("PeaceRiver", "January", (1020.3640351872195, 0.00015221038616604717)),
    ("FtSumner", "January", (1047.964376368261, 0.00016218804771381842)),
    ("SouthPole", "July", (1022.1737895082897, 0.00017812023753792838)),
]

ids = [f"{loc}-{season or 'None'}" for loc, season, _ in msis00_expected]


@pytest.mark.parametrize(("loc", "season", "expected"), msis00_expected, ids=ids)
def test_msis_atm(loc, season, expected):
    from MCEq.geometry.density_profiles import MSIS00Atmosphere

    msis_obj = MSIS00Atmosphere(loc, season)
    if expected is None:
        ref = (float(msis_obj.max_X), float(1.0 / msis_obj.r_X2rho(100.0)))
        msg = f"MSIS-00 reference data for {loc} in {season} not available."
        msg += f" Creating a new one. {ref}"
        pytest.fail(msg)
    assert np.allclose([msis_obj.max_X, 1.0 / msis_obj.r_X2rho(100.0)], expected)


@pytest.mark.parametrize(
    "cls,args",
    [
        (dp.CorsikaAtmosphere, ("USStd", None)),
        (dp.MSIS00Atmosphere, ("SouthPole", "January")),
        (dp.IsothermalAtmosphere, ("Nowhere", None)),
    ],
)
def test_common_atmosphere_interface(cls, args):
    atm = cls(*args)
    atm.set_theta(0.0)

    X_test = np.linspace(1, atm.max_X * 0.99, 10)
    h_test = np.linspace(0, atm.geom.h_atm, 10)

    # r_X2rho should give positive finite values
    inv_rho = atm.r_X2rho(X_test)
    assert np.all(np.isfinite(inv_rho))
    assert np.all(inv_rho > 0)

    # X2rho should be consistent with inverse
    rho = atm.X2rho(X_test)
    assert np.allclose(1 / rho, inv_rho, rtol=1e-3)

    # h2X and X2h roundtrip
    for h in h_test:
        X = atm.h2X(h)
        h_back = atm.X2h(X)
        assert np.isclose(h, h_back, rtol=1e-2)

    # misc functions
    for h in h_test:
        mol = atm.moliere_air(h)
        nrel = atm.nref_rel_air(h)
        theta_c = atm.theta_cherenkov_air(h)
        gamma_c = atm.gamma_cherenkov_air(h)
        assert mol > 0
        assert 0 < nrel < 1e-3
        assert 0 < theta_c < 90
        assert gamma_c > 1


def test_corsika_depth_and_inverse_functions():
    atm = dp.CorsikaAtmosphere("USStd", None)
    atm.set_theta(0.0)

    # depth2height should roughly invert get_mass_overburden
    h_cm = np.linspace(1e5, 6e5, 10)
    X = [atm.get_mass_overburden(h) for h in h_cm]
    h_back = [atm.depth2height(x) for x in X]
    assert np.allclose(h_cm, h_back, rtol=0.1)

    # rho_inv gives inverse density
    rho = [atm.get_density(h) for h in h_cm]
    rho_inv = [atm.rho_inv(atm.get_mass_overburden(h), np.cos(0.0)) for h in h_cm]
    assert np.allclose([1.0 / r for r in rho], rho_inv, rtol=1e-2)

    # calc_thickl returns 5 values
    thickl = atm.calc_thickl()
    assert isinstance(thickl, list)
    assert len(thickl) == 5


def test_isothermal_mass_overburden():
    atm = dp.IsothermalAtmosphere("Unknown", None)
    h = np.linspace(0, 5e5, 10)
    overburden = atm.get_mass_overburden(h)
    assert np.all(overburden > 0)
    assert np.all(overburden <= atm.X0)


def test_msis_setters_and_cache_clear():
    atm = dp.MSIS00Atmosphere("SouthPole", "January")
    atm.theta_deg = 42
    atm._clear_cache()
    assert atm.theta_deg is None

    atm.set_location("Karlsruhe")
    atm.set_location_coord(10.0, 48.0)
    atm.set_season("February")
    atm.set_doy(42)
    assert atm.theta_deg is None  # should still be cleared

    h = 2e5
    T = atm.get_temperature(h)
    assert 100 < T < 1000


def test_msis00_icecube_centered():
    atm = dp.MSIS00IceCubeCentered(
        "Karlsruhe", "January"
    )  # should override to SouthPole
    # just check if works
    assert atm.get_density(1e5) > 0

    # test latitude at 0 and 90 deg
    lat_0 = atm._latitude(0.0)
    lat_90 = atm._latitude(90.0)
    assert -90.0 <= lat_0 <= 0.0
    assert -90.0 <= lat_90 <= 0.0
    assert lat_0 < lat_90  # as zenith increases, impact moves away from vertical

    # test set_theta for downgoing and upgoing
    atm.set_theta(45.0)
    assert atm.theta_deg == 45.0

    atm.set_theta(135.0)
    assert atm.theta_deg == 135.0


@pytest.mark.xfail(reason="AIRSAtmosphere requires unavailable data files")
def test_airs_instantiation():
    dp.AIRSAtmosphere("SouthPole", "January")


@pytest.mark.parametrize("X", [1.0, 10.0, 100.0])
def test_generalized_target(X):
    tgt = dp.GeneralizedTarget(len_target=100.0, env_density=2.0, env_name="default")

    # reset + basic props
    assert tgt.max_den == 2.0
    assert np.isclose(tgt.max_X, 100.0 * 2.0)
    assert np.isclose(tgt.get_density_X(X), 2.0)
    assert np.isclose(tgt.r_X2rho(X), 0.5)
    assert np.isclose(tgt.get_density(tgt.s_X2h(X)), 2.0)

    # set_length ok
    tgt.set_length(150.0)
    assert tgt.len_target == 150.0
    assert np.isclose(tgt.max_X, 2.0 * 150.0)

    # add material (replace default)
    tgt.reset()
    tgt.add_material(0.0, 3.0, "iron")
    assert len(tgt.mat_list) == 1
    assert np.isclose(tgt.get_density(0), 3.0)

    # add second layer
    tgt.reset()
    tgt.add_material(60.0, 4.0, "lead")
    assert len(tgt.mat_list) == 2
    assert np.isclose(tgt.get_density(70), 4.0)

    # set_length too small
    tgt.reset()
    tgt.add_material(50.0, 3.0, "x")
    with pytest.raises(Exception):
        tgt.set_length(40.0)

    # add invalid materials
    with pytest.raises(Exception):
        tgt.add_material(200.0, 1.0, "fail")
    with pytest.raises(Exception):
        tgt.add_material(30.0, 2.0, "bad")  # not monotonic

    # set_theta must raise
    with pytest.raises(NotImplementedError):
        tgt.set_theta(0.0)

    # spline roundtrip
    X_vals = np.linspace(1, tgt.max_X - 1, 10)
    h_vals = tgt.s_X2h(X_vals)
    assert np.allclose(tgt.s_h2X(h_vals), X_vals)

    # density errors
    with pytest.raises(Exception):
        tgt.get_density([0, 200])
    with pytest.raises(Exception):
        tgt.get_density_X(tgt.max_X * 1.01)

    # reset again
    tgt.add_material(60.0, 5.0, "z")
    tgt.reset()
    assert len(tgt.mat_list) == 1
    assert np.isclose(tgt.get_density(10), 2.0)


# ---------------------------------------------------------------------------
# MSIS00LocationCentered tests
# ---------------------------------------------------------------------------


def test_impact_point_at_southpole_matches_icecube_formula():
    """MSIS00LocationCentered at South Pole must reproduce the legacy _latitude formula."""
    atm = dp.MSIS00LocationCentered(
        detector_coord=(0.0, -90.0),
        depth_m=1948.0,
        season="January",
    )
    # Replicate the original IceCube formula for reference
    r = atm.geom.r_E / 1e2  # cm → m
    d = 1948.0

    for theta_deg in [0.0, 30.0, 45.0, 60.0, 90.0]:
        lat_ecef, _ = atm._impact_point(theta_deg, 0.0)

        theta_rad = np.deg2rad(theta_deg)
        x = np.sqrt(2.0 * r * d + ((r - d) * np.cos(theta_rad)) ** 2 - d**2) - (
            r - d
        ) * np.cos(theta_rad)
        lat_ref = (
            -90.0
            + np.arctan2(x * np.sin(theta_rad), r - d + x * np.cos(theta_rad))
            / np.pi
            * 180.0
        )
        assert np.isclose(lat_ecef, lat_ref, atol=1e-6), (
            f"theta={theta_deg}: ECEF={lat_ecef:.6f}, ref={lat_ref:.6f}"
        )


def test_impact_point_vertical_returns_detector_location():
    """For theta=0 (vertical), impact point must equal detector lat/lon."""
    for lon, lat in [(6.033, 42.803), (15.4, 36.264), (139.0, 35.0), (0.0, -90.0)]:
        atm = dp.MSIS00LocationCentered(
            detector_coord=(lon, lat),
            depth_m=1000.0,
            season="January",
        )
        for azi in [0.0, 90.0, 180.0, 270.0]:
            lat_imp, lon_imp = atm._impact_point(0.0, azi)
            assert np.isclose(lat_imp, lat, atol=1e-4), (
                f"lat mismatch for ({lon},{lat}) azi={azi}: got {lat_imp}"
            )
            assert np.isclose(lon_imp, lon, atol=1e-4), (
                f"lon mismatch for ({lon},{lat}) azi={azi}: got {lon_imp}"
            )


def test_location_centered_single_azimuth():
    """Single azimuth mode: impact coordinates must be set and density > 0."""
    atm = dp.MSIS00LocationCentered(
        detector_coord=(6.033, 42.803),
        depth_m=2450.0,
        season="January",
    )
    atm.set_theta(30.0, azimuth_deg=0.0)  # North
    assert atm.theta_deg == 30.0
    assert atm.current_impact_latitude is not None
    assert atm.current_impact_longitude is not None
    assert -90.0 <= atm.current_impact_latitude <= 90.0
    assert -180.0 <= atm.current_impact_longitude <= 180.0
    assert atm.get_density(1e5) > 0.0

    # Northward shot should push impact latitude above detector latitude
    atm.set_theta(60.0, azimuth_deg=0.0)
    assert atm.current_impact_latitude >= 42.803


def test_location_centered_azimuth_averaging():
    """No-azimuth mode: impact coords are None (averaged), density > 0."""
    atm = dp.MSIS00LocationCentered(
        detector_coord=(6.033, 42.803),
        depth_m=2450.0,
        season="January",
    )
    atm.set_theta(30.0)  # no azimuth → average
    assert atm.theta_deg == 30.0
    assert atm.current_impact_latitude is None
    assert atm.current_impact_longitude is None
    assert atm.get_density(1e5) > 0.0


def test_location_centered_upgoing():
    """Upgoing angles are correctly handled when max_theta=180."""
    atm = dp.MSIS00LocationCentered(
        detector_coord=(6.033, 42.803),
        depth_m=2450.0,
        season="January",
        max_theta=180.0,
    )
    atm.set_theta(135.0)
    assert atm.theta_deg == 135.0
    assert atm.get_density(1e5) > 0.0

    # With azimuth
    atm.set_theta(150.0, azimuth_deg=90.0)
    assert atm.theta_deg == 150.0
    assert atm.current_impact_latitude is not None


def test_km3net_orca_and_arca_instantiation():
    """ORCA and ARCA detectors can be instantiated and return positive density."""
    for det in ["ORCA", "ARCA"]:
        atm = dp.MSIS00KM3NeTCentered(det, season="January")
        assert atm.get_density(1e5) > 0.0
        assert atm.max_theta == 180.0

    with pytest.raises(ValueError, match="Unknown KM3NeT detector"):
        dp.MSIS00KM3NeTCentered("INVALID")


def test_km3net_upgoing():
    """KM3NeT can handle upgoing neutrino angles."""
    atm = dp.MSIS00KM3NeTCentered("ORCA", season="January")
    atm.set_theta(135.0)
    assert atm.theta_deg == 135.0
    assert atm.get_density(1e5) > 0.0


def test_km3net_single_azimuth():
    """KM3NeT with explicit azimuth returns a specific impact point."""
    atm = dp.MSIS00KM3NeTCentered("ORCA", season="January")
    atm.set_theta(60.0, azimuth_deg=0.0)
    assert atm.current_impact_latitude is not None
    assert atm.current_impact_longitude is not None
    assert atm.get_density(1e5) > 0.0


def test_km3net_arca_set_theta():
    """ARCA model with azimuth averaging works for a typical zenith."""
    atm = dp.MSIS00KM3NeTCentered("ARCA", season="July")
    atm.set_theta(45.0)
    assert atm.theta_deg == 45.0
    assert atm.get_density(1e5) > 0.0


def test_base_class_impact_properties_return_none():
    """Non-location-coupled models return None for impact coordinates."""
    atm = dp.CorsikaAtmosphere("USStd")
    assert atm.current_impact_latitude is None
    assert atm.current_impact_longitude is None


def test_icecube_latitude_wrapper():
    """MSIS00IceCubeCentered._latitude backward-compat wrapper returns correct values."""
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")
    # Detector is 1948 m below the glacier top at 2835 m elevation
    r = atm.geom.r_E / 1e2 + 2835.0
    d = 1948.0
    for theta_deg in [0.0, 30.0, 60.0, 90.0]:
        lat_wrapper = atm._latitude(theta_deg)
        theta_rad = np.deg2rad(theta_deg)
        x = np.sqrt(2.0 * r * d + ((r - d) * np.cos(theta_rad)) ** 2 - d**2) - (
            r - d
        ) * np.cos(theta_rad)
        lat_ref = (
            -90.0
            + np.arctan2(x * np.sin(theta_rad), r - d + x * np.cos(theta_rad))
            / np.pi
            * 180.0
        )
        assert np.isclose(lat_wrapper, lat_ref, atol=1e-6), (
            f"theta={theta_deg}: _latitude={lat_wrapper:.6f}, ref={lat_ref:.6f}"
        )


# ---------------------------------------------------------------------------
# Detector-depth zenith correction and far-side geometry
# ---------------------------------------------------------------------------


def test_local_zenith_correction():
    """The column angle is the local zenith at the surface crossing:
    sin(theta_s) = (r_det/r_surf) sin(theta_det)."""
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")
    r_det = atm.geom.r_E + (2835.0 - 1948.0) * 1e2

    # Negligible at small angles
    atm.set_theta(30.0)
    assert np.isclose(np.rad2deg(atm.thrad), 30.0, atol=0.02)

    # Exact impact-parameter formula at the horizon: ~1.4 deg correction
    atm.set_theta(90.0)
    expected = np.rad2deg(np.arcsin(r_det / atm.geom.r_obs))
    assert np.isclose(np.rad2deg(atm.thrad), expected, atol=1e-9)
    assert np.rad2deg(atm.thrad) < 88.7  # ~88.58 for 1948 m below the surface


def test_icecube_observation_level_follows_surface():
    """Downgoing/grazing columns end at the glacier top (2835 m), upgoing
    columns at sea level (far side is the ocean)."""
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")
    atm.set_theta(0.0)
    assert np.isclose(atm.geom.h_obs, 2835.0e2)
    # Vertical column above 2835 m is much thinner than to sea level
    assert atm.max_X < 800.0

    atm.set_theta(180.0)
    assert np.isclose(atm.geom.h_obs, 0.0)
    assert atm.max_X > 950.0  # full sea-level column on the far side

    # switching back restores the surface level
    atm.set_theta(45.0)
    assert np.isclose(atm.geom.h_obs, 2835.0e2)


def test_grazing_window_is_near_side():
    """Angles in (90, 90 + dip] exit through the near-side surface and are
    treated as downgoing grazing columns, not far-side upgoing ones."""
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")
    atm.set_theta(91.0, azimuth_deg=0.0)
    # Impact point is a few hundred km from the pole, not on the far side
    assert atm.current_impact_latitude < -85.0
    assert np.isclose(atm.geom.h_obs, 2835.0e2)
    assert np.rad2deg(atm.thrad) < 90.0


def test_upgoing_impact_point_is_far_side():
    """For upgoing angles the MSIS anchor is the far-side surface crossing,
    where the shower actually developed."""
    atm = dp.MSIS00IceCubeCentered("SouthPole", "January")

    atm.set_theta(180.0, azimuth_deg=0.0)
    assert atm.current_impact_latitude > 89.9  # North Pole, not South

    atm.set_theta(120.0, azimuth_deg=0.0)
    # Chord from the South Pole at 120 deg exits near latitude -30
    assert np.isclose(atm.current_impact_latitude, -30.0, atol=0.1)
    # Local zenith at the far-side crossing is ~(180 - 120) deg
    assert np.isclose(np.rad2deg(atm.thrad), 60.0, atol=0.1)


def test_arca_site_coordinates():
    """ARCA (KM3NeT-It) is offshore Capo Passero at 36deg16'N 16deg06'E."""
    atm = dp.MSIS00KM3NeTCentered("ARCA", season="January")
    lat, lon = atm._impact_point(0.0, 0.0)
    assert np.isclose(lat, 36.267, atol=1e-3)
    assert np.isclose(lon, 16.100, atol=1e-3)


# ---------------------------------------------------------------------------
# MSIS21 <-> MSIS00 geometry parity
#
# MSIS21 is deliberately a **separate class tree** from MSIS00 -- not a
# subclass hierarchy and not a shared implementation -- but it must expose
# the **same interface** and, where the maths is the same, produce the same
# numbers.  That design decision is what these tests enforce.
#
# The risk it guards: MSIS21LocationCentered re-implements the
# MSIS00LocationCentered geometry (impact-point projection, local-zenith
# correction, grazing window, far-side anchor, site coordinates), and two
# copies of the same maths drift silently -- git cannot report a conflict
# across two files.  That has already happened once: the PR #164 geometry
# review updated MSIS00 while the MSIS21 copy kept the pre-review version.
#
# Two layers, deliberately split by what they need:
#   * interface conformance -- pure class introspection, no backend, so it
#     runs everywhere including CI;
#   * numerical parity -- needs the optional 'nrlmsis' package (MSIS21 is
#     opt-in), so it skips where that is absent, including CI.
# ---------------------------------------------------------------------------

MSIS_TREE_PAIRS = [
    ("MSIS00Atmosphere", "MSIS21Atmosphere"),
    ("MSIS00LocationCentered", "MSIS21LocationCentered"),
    ("MSIS00IceCubeCentered", "MSIS21IceCubeCentered"),
    ("MSIS00KM3NeTCentered", "MSIS21KM3NeTCentered"),
]


@pytest.mark.parametrize("name00,name21", MSIS_TREE_PAIRS)
def test_msis21_public_interface_matches_msis00(name00, name21):
    """Each MSIS21 class must be interface-compatible with its MSIS00 peer.

    Introspection only -- no atmosphere is instantiated, so this needs
    neither 'nrlmsis' nor any MSIS backend and therefore runs in CI, which
    the numerical parity tests cannot.  MSIS21 may *add* public API; it may
    not drop or rename anything MSIS00 exposes, and shared keyword
    arguments must keep the same defaults so the two are drop-in
    substitutable.
    """
    cls00, cls21 = getattr(dp, name00), getattr(dp, name21)

    public00 = {n for n in dir(cls00) if not n.startswith("_")}
    public21 = {n for n in dir(cls21) if not n.startswith("_")}
    missing = sorted(public00 - public21)
    assert not missing, (
        f"{name21} is missing public API that {name00} exposes: {missing}"
    )

    params00 = inspect.signature(cls00.__init__).parameters
    params21 = inspect.signature(cls21.__init__).parameters
    missing_args = [n for n in params00 if n not in params21]
    assert not missing_args, (
        f"{name21}.__init__ does not accept {missing_args}, which "
        f"{name00}.__init__ does -- the two are not drop-in substitutable"
    )

    for name, p00 in params00.items():
        if p00.default is inspect.Parameter.empty:
            continue
        assert params21[name].default == p00.default, (
            f"{name21}.__init__ default for '{name}' is "
            f"{params21[name].default!r}, but {name00} uses {p00.default!r}"
        )

MSIS21_PAIRS = [
    ("IceCube", lambda: dp.MSIS00IceCubeCentered("SouthPole", "January"),
     lambda: dp.MSIS21IceCubeCentered("SouthPole", "January")),
    ("ARCA", lambda: dp.MSIS00KM3NeTCentered("ARCA", season="January"),
     lambda: dp.MSIS21KM3NeTCentered("ARCA", season="January")),
    ("ORCA", lambda: dp.MSIS00KM3NeTCentered("ORCA", season="January"),
     lambda: dp.MSIS21KM3NeTCentered("ORCA", season="January")),
]


@pytest.mark.parametrize("label,make00,make21", MSIS21_PAIRS)
def test_msis21_impact_point_matches_msis00(label, make00, make21):
    """The two families must project to the same impact point everywhere."""
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    a00, a21 = make00(), make21()

    assert a00._detector_depth_m == a21._detector_depth_m, "depth differs"
    assert a00._surface_elevation_m == a21._surface_elevation_m, "elevation differs"
    assert a00._detector_latitude == a21._detector_latitude
    assert a00._detector_longitude == a21._detector_longitude

    for theta in (0.0, 30.0, 60.0, 80.0, 89.0, 90.0, 90.5, 100.0, 140.0, 179.0):
        if theta > min(a00.max_theta, a21.max_theta):
            continue
        # set_theta moves the observation level, which _impact_point reads,
        # so drive both families through it in step before comparing.
        a00.set_theta(theta, azimuth_deg=0.0)
        a21.set_theta(theta, azimuth_deg=0.0)
        assert np.isclose(
            a00._effective_theta_deg, a21._effective_theta_deg, rtol=0, atol=1e-10
        ), (
            f"{label}: local zenith differs at theta={theta}: "
            f"{a00._effective_theta_deg} vs {a21._effective_theta_deg}"
        )
        assert a00.geom.h_obs == a21.geom.h_obs, (
            f"{label}: observation level differs at theta={theta}"
        )
        for azi in (0.0, 45.0, 90.0, 180.0, 270.0):
            lat0, lon0 = a00._impact_point(theta, azi)
            lat1, lon1 = a21._impact_point(theta, azi)
            assert np.isclose(lat0, lat1, rtol=0, atol=1e-9), (
                f"{label}: impact latitude differs at theta={theta}, azi={azi}"
            )
            assert np.isclose(lon0, lon1, rtol=0, atol=1e-9), (
                f"{label}: impact longitude differs at theta={theta}, azi={azi}"
            )


def test_msis21_shares_km3net_site_table():
    """Site coordinates must come from one table, not a second copy."""
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    from MCEq.geometry import msis21_atmosphere

    assert msis21_atmosphere._KM3NET_DETECTORS is dp._KM3NET_DETECTORS
