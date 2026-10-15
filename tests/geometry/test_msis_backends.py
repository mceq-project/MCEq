"""Backend behaviour of the NRLMSIS atmospheres: negative heights and the noon convention.

Near the observation level the sampled height can be negative by one ULP of
``r_E``. NRLMSISE-00 receives that height; NRLMSIS 2.1 clamps ``z`` at 0 km.
MSIS00 holds local noon, MSIS21 UT noon.
"""

from __future__ import annotations

import numpy as np
import pytest

import MCEq.geometry.msis21 as m21
from MCEq.geometry import density_profiles as dp
from MCEq.geometry.geometry import EarthGeometry

#: Sample count of every ``calculate_density_spline`` call, and the grid the
#: divergences below are measured on.
N_STEPS = 2000

#: Zenith grid for the array-vs-scalar sweep, in degrees. One degree, not the
#: 0.1 deg of the golden section's own measurement: 91 x 2000 scalar ``h``
#: calls is 0.4 s and finds the divergence at ~20 angles.
ZENITH_GRID_DEG = np.linspace(0.0, 90.0, 91)

#: Zeniths scanned for a negative sampled height. A geometry at ``h_obs = 0``
#: ends its path at ``h = 0`` exactly, and the last sample lands one ULP below
#: it at a good fraction of angles.
CLAMP_SCAN_DEG = (2.0, 4.0, 7.0, 10.0, 11.0, 13.0, 16.0, 23.0, 30.0, 35.0)


def _h_both_ways(geom, thrad, n_steps=N_STEPS):
    """``h(dl, thrad)`` computed on the vector and sample by sample."""
    dl = np.linspace(0.0, geom.path_len(thrad), n_steps)
    as_array = geom.h(dl, thrad)
    per_scalar = np.array([geom.h(sample, thrad) for sample in dl])
    return as_array, per_scalar


# --------------------------------------------------------------------------
# the z >= 0 clamp
# --------------------------------------------------------------------------


def test_the_sampled_height_really_goes_negative():
    """The clamp is live, not defensive: ``min(h) < 0`` at ``h_obs = 0``.

    The path ends at the observation level, so the last sample is ``h_obs``
    exactly in exact arithmetic and one ULP of ``r_E`` below it in float64.
    Measured on this host: -1.1920929e-07 cm at one of the 2000 samples, at 22
    of 96 sampled zeniths for every atmosphere at sea level.
    """
    geom = EarthGeometry()
    assert geom.h_obs == 0.0
    hits = {}
    for theta in CLAMP_SCAN_DEG:
        as_array, per_scalar = _h_both_ways(geom, np.deg2rad(theta))
        if as_array.min() < 0.0 or per_scalar.min() < 0.0:
            hits[theta] = (float(as_array.min()), float(per_scalar.min()))

    assert hits, (
        f"no sampled height is negative over {CLAMP_SCAN_DEG} deg, so the "
        f"np.maximum(h/1e5, 0.0) clamp in the two MSIS21 tails is now dead "
        f"code and the two tails without it are no longer exposed"
    )
    ulp = float(np.spacing(geom.r_E))
    for theta, (a, s) in hits.items():
        assert min(a, s) >= -2 * ulp, (
            f"theta {theta}: min(h) = {min(a, s):.3e} cm, deeper than two ULP "
            f"of r_E ({-2 * ulp:.3e}). That is a geometry error, not the "
            f"endpoint cancellation this pins."
        )


def test_the_base_tail_hands_the_negative_height_to_get_density():
    """``EarthsAtmosphere.calculate_density_spline`` does not clamp.

    The tail wraps ``self.get_density(self.geom.h(dl, thrad))`` in
    ``np.vectorize`` with no floor, so a negative sample reaches the density
    model. Pinned by watching the argument.
    """
    atm = dp.CorsikaAtmosphere("BK_USStd", None)
    seen: list[float] = []
    real = atm.get_density
    atm.get_density = lambda h, _r=real, _s=seen: (_s.append(float(h)), _r(h))[1]
    try:
        negative = [
            theta
            for theta in CLAMP_SCAN_DEG
            if _set_theta_and_min(atm, theta, seen) < 0
        ]
    finally:
        del atm.get_density

    assert negative, (
        "no negative height reached CorsikaAtmosphere.get_density over "
        f"{CLAMP_SCAN_DEG} deg. Either the base tail acquired a clamp — say so "
        "and update the environment golden — or the geometry no longer "
        "undershoots the observation level."
    )


def test_the_averaged_msis00_tail_hands_the_negative_height_to_the_backend():
    """``MSIS00LocationCentered.calculate_density_spline`` does not clamp either.

    It is a separate tail from the base one — azimuth-major, its own
    ``h_vec`` — and it too passes the raw ``h_cm`` into
    ``np.vectorize(self._msis.get_density)``. ``n_azimuth=2`` keeps the
    height-major ctypes loop to 2 x 50 calls per zenith.
    """
    atm = dp.MSIS00KM3NeTCentered("ORCA", "January", n_azimuth=2)
    seen: list[float] = []
    real = atm._msis.get_density
    atm._msis.get_density = lambda h, _r=real, _s=seen: (_s.append(float(h)), _r(h))[1]
    try:
        negative = []
        for theta in CLAMP_SCAN_DEG:
            atm.set_theta(theta)  # averaged mode: no azimuth
            seen.clear()
            atm.calculate_density_spline(n_steps=50)
            assert atm._azimuth_averaging
            if seen and min(seen) < 0.0:
                negative.append(theta)
    finally:
        del atm._msis.get_density

    assert negative, (
        "no negative height reached the MSIS00 backend from the "
        "azimuth-averaged tail; it has acquired a clamp the base tail still "
        "lacks, or the geometry changed"
    )


@pytest.mark.parametrize(
    "maker",
    [
        pytest.param(
            lambda: m21.MSIS21Atmosphere("SouthPole", "January"), id="MSIS21Atmosphere"
        ),
        pytest.param(
            lambda: m21.MSIS21KM3NeTCentered("ORCA", "January", n_azimuth=2),
            id="MSIS21KM3NeTCentered",
        ),
    ],
)
def test_the_msis21_tails_clamp_the_negative_height_to_zero(maker):
    """Both MSIS21 tails floor ``z`` at 0 km before calling the backend.

    So at a zenith where the base and averaged-MSIS00 tails pass a negative
    height through, these ask for ``z = 0.0`` instead. The two sides of the
    same divergence, in the same file.
    """
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    atm = maker()

    seen: list[np.ndarray] = []
    real = atm._msis._model.calc_altitude_array
    atm._msis._model.calc_altitude_array = lambda _r=real, _s=seen, **kw: (
        _s.append(np.array(kw["z"], copy=True)),
        _r(**kw),
    )[1]
    try:
        clamped = []
        for theta in CLAMP_SCAN_DEG:
            atm.set_theta(theta)
            seen.clear()
            atm.calculate_density_spline(n_steps=50)
            dl = np.linspace(0.0, atm.geom.path_len(atm.thrad), 50)
            raw = atm.geom.h(dl, atm.thrad)
            if raw.min() < 0.0:
                assert seen, "the tail issued no calc_altitude_array call"
                for z in seen:
                    assert z.min() == 0.0, (
                        f"theta {theta}: raw min(h) = {raw.min():.3e} cm and the "
                        f"backend was asked for z = {z.min():.3e} km. The "
                        f"np.maximum(h/1e5, 0.0) clamp is gone."
                    )
                clamped.append(theta)
    finally:
        del atm._msis._model.calc_altitude_array

    assert clamped, (
        f"no zenith in {CLAMP_SCAN_DEG} produced a negative raw height at "
        f"n_steps=50, so the clamp was never exercised here"
    )


# --------------------------------------------------------------------------
# scalar and batched NRLMSIS 2.1
# --------------------------------------------------------------------------


def _capture(sink, result):
    sink.append(np.array(result.densities[0], copy=True))
    return result


def test_scalar_and_batched_msis21_agree_within_numerical_precision():
    """``get_density`` and the spline tail also differ in *which* backend call.

    One is scalar ``calc`` per height, the other one batched
    ``calc_altitude_array``, and that difference is an order larger than the
    scaling order above — 7.2e-15 to 1.4e-14 against 1e-15 on this host,
    measured over theta = 0/30/60/85/90 at 200 and 2000 samples. So a phase
    that unifies only the scaling order has not made the two paths agree, and
    this is what says so.

    Linux/Python 3.14 CI measures 6.3e-11. Bound the numerical agreement
    at one part per billion across builds; bitwise inequality is not an
    API requirement.
    """
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    atm = m21.MSIS21Atmosphere("SouthPole", "January")
    atm.set_theta(60.0)

    dl = np.linspace(0.0, atm.geom.path_len(atm.thrad), 200)
    h = atm.geom.h(dl, atm.thrad)
    batched = (
        atm._msis._model.calc_altitude_array(
            day=atm._msis._doy,
            utsec=atm._msis._sec,
            z=np.maximum(h / 1.0e5, 0.0),
            lat=atm._msis._lat,
            lon=atm._msis._lon,
            sfluxavg=atm._msis._f107a,
            sflux=atm._msis._f107,
            ap=atm._msis._ap,
        ).densities[0]
        * 1.0e-3
    )
    scalar = np.array([atm.get_density(float(x)) for x in h])

    maxrel = float(np.max(np.abs(batched - scalar) / np.abs(scalar)))
    assert maxrel < 1e-9, (
        f"scalar calc and calc_altitude_array differ by {maxrel:.3e} relative, "
        "exceeding the cross-build numerical agreement bound"
    )


# --------------------------------------------------------------------------
# the local-solar-time convention (Anatoli's ruling, 2026-09-04)
# --------------------------------------------------------------------------


def test_msis00_holds_every_site_at_local_noon():
    """``lst`` is 12.0 everywhere, and ``sec`` is derived to match.

    NRLMSISE-00 wants ``lst = sec/3600 + g_long/15``. There is no setter for
    the time of day, so one of the two has to be derived; MCEq holds the
    *local* time fixed at noon (a representative daytime atmosphere per site)
    and derives ``sec``, which is UT.

    The previous two states both fail this: the original bug left ``sec`` at
    43200 while pinning ``lst`` at 12 (relation violated), and the first fix
    recomputed ``lst`` from the fixed ``sec`` (relation satisfied, but every
    non-Greenwich site at 12:00 UT -- Tokyo came out at lst 21.3, a night
    atmosphere).
    """
    from MCEq.geometry.atmosphere_parameters import LOCATIONS
    from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

    for site, (lon, _lat, _alt) in LOCATIONS.items():
        msis = cNRLMSISE00()
        msis.set_location(site)
        lst, sec, g_long = msis.inp.lst.value, msis.inp.sec.value, msis.inp.g_long.value

        assert lst == pytest.approx(12.0), f"{site} is not at local noon: lst={lst}"
        assert lst == pytest.approx(sec / 3600.0 + g_long / 15.0, abs=1e-9), (
            f"{site} violates the NRLMSISE-00 consistency relation"
        )
        assert 0.0 <= sec <= 86400.0, f"{site} derived sec={sec} outside one day"
        assert g_long == pytest.approx(lon)


def test_msis00_local_noon_survives_a_coordinate_change():
    """``set_location_coord`` must re-derive, not keep the previous site's UT."""
    from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

    msis = cNRLMSISE00()
    for lon in (0.0, 139.0, -117.2, 180.0, -180.0):
        msis.set_location_coord(lon, 20.0)
        assert msis.inp.lst.value == pytest.approx(12.0)
        assert msis.inp.lst.value == pytest.approx(
            msis.inp.sec.value / 3600.0 + lon / 15.0, abs=1e-9
        )
        assert 0.0 <= msis.inp.sec.value <= 86400.0


def test_msis00_is_local_noon_and_msis21_is_ut_noon():
    """The two backends deliberately sample different instants. Pin the gap.

    MSIS 2.1 derives ``lst = utsec/3600 + lon/15`` from a fixed ``utsec``
    (``nrlmsis/globe.py``), i.e. 12:00 UT, while MCEq's MSIS00 wrapper holds
    local noon. At a non-zero longitude the two therefore describe different
    times of day, and a density comparison between them mixes that in with the
    genuine model difference.

    This is a consequence of the ruling, not an accident, so it is pinned
    rather than fixed: if MSIS21 is ever moved to local noon as well, this test
    is what says so.
    """
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    from MCEq.geometry.atmosphere_parameters import DAY_TIMES_SEC
    from MCEq.geometry.msis21 import MSIS21Atmosphere
    from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

    tokyo_lon = 139.0

    msis00 = cNRLMSISE00()
    msis00.set_location("Tokyo")
    assert msis00.inp.lst.value == pytest.approx(12.0)

    atm21 = MSIS21Atmosphere("Tokyo", "January")
    assert atm21._msis._sec == pytest.approx(DAY_TIMES_SEC["day"]), (
        "MSIS21 no longer uses a fixed UT second; re-derive this pin"
    )
    msis21_lst = (atm21._msis._sec / 3600.0 + tokyo_lon / 15.0) % 24.0
    assert msis21_lst == pytest.approx(21.266, abs=1e-3)

    assert abs(msis21_lst - msis00.inp.lst.value) == pytest.approx(9.266, abs=1e-3)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _have_nrlmsis() -> bool:
    try:
        import nrlmsis  # noqa: F401
    except ImportError:
        return False
    return True


def _set_theta_and_min(atm, theta_deg, seen):
    """Respline at ``n_steps=50`` and return the smallest recorded height."""
    atm.set_theta(theta_deg)
    seen.clear()
    atm.calculate_density_spline(n_steps=50)
    return min(seen) if seen else 0.0
