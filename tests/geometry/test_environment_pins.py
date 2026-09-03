"""Three divergences among the four density-spline tails, pinned as behaviour.

``EarthsAtmosphere.calculate_density_spline`` and the three overrides
(``MSIS00LocationCentered`` for azimuth-averaging mode,
``MSIS21Atmosphere``, ``MSIS21LocationCentered``) compute the same
:math:`\\rho(X)` from the same geometry, and they do not agree bitwise. Phase 3
merges them into one ``environment/atmosphere`` tail, so each divergence is
recorded here first: every test below FAILS once the tails are unified, which
is the point — the unification has to come with a statement of which form it
kept, not a silent choice.

The three:

1. **``geom.h`` on an array against ``geom.h`` per scalar.** Two tails call it
   once on the whole ``dl`` vector, two loop over the samples, and the results
   differ by one ULP of ``r_E``.
2. **The ``z >= 0`` clamp.** The two MSIS21 tails clamp with
   ``np.maximum(h / 1e5, 0.0)``; the base tail and the azimuth-averaged MSIS00
   tail hand the raw height to the backend, and the height *is* negative — by
   one ULP of ``r_E`` — at one or two of the 2000 samples for a good fraction
   of the zenith range.
3. **The kg -> g scaling order in the azimuth average.**
   ``MSIS21LocationCentered.get_density`` multiplies each direction's density
   by 1e-3 and then sums; its ``calculate_density_spline`` sums and then
   multiplies once.

Sizes are asserted with bounds, not equalities: the exact counts are host
libm's, and they live in the ``environment`` golden section
(``geom_h/forms``, ``extra.n_geometries_where_h_forms_differ``), which is
``golden_host`` for exactly this reason. What is asserted here is the
structure — that the divergence exists, which code path takes which side, and
that the magnitude is the one-ULP cancellation and not something larger.
"""

from __future__ import annotations

import numpy as np
import pytest

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
# 1. geom.h: array against scalar
# --------------------------------------------------------------------------


def test_geom_h_disagrees_between_an_array_call_and_a_scalar_loop():
    """One ULP of ``r_E``, at a fifth of the zenith range.

    ``h`` returns ``sqrt(A_2^2 + (A_1 + l - dl)^2) - r_E``. numpy lowers
    ``arr ** 2`` to a multiply and sends a float64 *scalar* ``** 2`` through
    libm ``pow``, which is 1 ULP off at some arguments; the ``- r_E`` then
    rewrites that as 1 ULP of 6.37e8 cm on a height of order 1e6.

    Measured on this host at 0.1 deg resolution: 258 of 901 zeniths, always
    one or two of the 2000 samples, always exactly 1.1920929e-07 cm. The
    counts are the golden section's (``geom_h/forms``); here only the
    structure is asserted.
    """
    geom = EarthGeometry()
    differing, worst = [], 0.0
    for theta in ZENITH_GRID_DEG:
        as_array, per_scalar = _h_both_ways(geom, np.deg2rad(theta))
        delta = np.abs(as_array - per_scalar)
        if np.any(delta):
            differing.append(float(theta))
            worst = max(worst, float(delta.max()))

    assert differing, (
        "geom.h now agrees on an array and per scalar over the whole zenith "
        "grid. If the tails were unified, say which form was kept and update "
        "the environment golden's geom_h/forms; if libm changed, this test "
        "and that array both move."
    )
    ulp = float(np.spacing(geom.r_E))
    assert worst <= 2 * ulp, (
        f"array-vs-scalar geom.h differs by {worst:.3e} cm, more than two ULP "
        f"of r_E ({2 * ulp:.3e} cm). That is no longer the pow-against-multiply "
        f"cancellation this pins."
    )


def test_the_scalar_square_is_libm_pow_and_the_array_square_is_a_multiply():
    """The mechanism behind the test above, without the geometry.

    A witness is searched for rather than hard-coded, because which argument
    libm rounds the other way is the host's business. What is fixed is that a
    witness exists, that the array loop agrees with the multiply, and that the
    scalar ``** 2`` is the one that does not.
    """
    geom = EarthGeometry()
    thrad = np.deg2rad(0.1)
    xs = (
        geom._A_1(thrad)
        + geom.path_len(thrad)
        - np.linspace(0.0, geom.path_len(thrad), N_STEPS)
    )

    squared_as_array = xs**2
    assert np.array_equal(squared_as_array, xs * xs), (
        "numpy no longer lowers `arr ** 2` to a multiply, so the array tails "
        "and the scalar ones may have swapped sides"
    )

    witnesses = np.flatnonzero(
        np.array([float(x) ** 2 != float(x) * float(x) for x in xs])
    )
    assert witnesses.size, (
        "no float64 scalar in this sample squares differently through `** 2` "
        "than through a multiply; libm pow has become exact here and the "
        "array-vs-scalar divergence is gone with it"
    )
    x = xs[witnesses[0]]
    assert squared_as_array[witnesses[0]] == x * x


def test_the_environment_golden_labels_each_tail_with_the_form_it_uses():
    """``gen_environment.h_form`` against a count of the ``geom.h`` calls.

    The golden section stores a per-cell ``h_form``, and a claim in a golden is
    worth what verifies it. This counts the calls the tail actually makes: one
    for an array tail, ``O(n_steps)`` for a scalar one.

    ``calculate_density_spline`` is called directly at ``n_steps=50`` after
    ``set_theta`` has established the mode, so the count is cheap and the two
    azimuth modes of the detector-centred models are both reachable.
    """
    from tests.golden.gen_environment import h_form

    cases = (
        ("CorsikaAtmosphere", dp.CorsikaAtmosphere("BK_USStd", None), (60.0,)),
        (
            "IsothermalAtmosphere",
            dp.IsothermalAtmosphere("SouthPole", "January"),
            (60.0,),
        ),
        ("MSIS00Atmosphere", dp.MSIS00Atmosphere("SouthPole", "January"), (60.0,)),
        (
            "MSIS00KM3NeTCentered",
            dp.MSIS00KM3NeTCentered("ORCA", "January", n_azimuth=2),
            (60.0, None),
        ),
    )
    if _have_nrlmsis():
        cases += (
            ("MSIS21Atmosphere", dp.MSIS21Atmosphere("SouthPole", "January"), (60.0,)),
            (
                "MSIS21KM3NeTCentered",
                dp.MSIS21KM3NeTCentered("ORCA", "January", n_azimuth=2),
                (60.0, None),
            ),
        )

    for cls_name, atm, azimuths in cases:
        for azimuth in azimuths:
            if azimuth is None:
                atm.set_theta(60.0)
            elif atm.depends_on_azimuth:
                atm.set_theta(60.0, azimuth)
            else:
                atm.set_theta(60.0)

            calls = []
            real = atm.geom.h
            atm.geom.h = lambda dl, thrad, _r=real, _c=calls: (
                _c.append(np.size(dl)),
                _r(dl, thrad),
            )[1]
            try:
                atm.calculate_density_spline(n_steps=50)
            finally:
                del atm.geom.h

            measured = "array" if max(calls) > 1 else "scalar"
            assert measured == h_form(cls_name), (
                f"{cls_name} (azimuth {azimuth}) calls geom.h in {len(calls)} "
                f"call(s) of width {sorted(set(calls))}, which reads as "
                f"{measured!r}, while gen_environment.h_form says "
                f"{h_form(cls_name)!r}. The golden section's h_form array and "
                f"the tails have drifted apart."
            )


# --------------------------------------------------------------------------
# 2. the z >= 0 clamp
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
            lambda: dp.MSIS21Atmosphere("SouthPole", "January"), id="MSIS21Atmosphere"
        ),
        pytest.param(
            lambda: dp.MSIS21KM3NeTCentered("ORCA", "January", n_azimuth=2),
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
    real = atm._model.calc_altitude_array
    atm._model.calc_altitude_array = lambda _r=real, _s=seen, **kw: (
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
        del atm._model.calc_altitude_array

    assert clamped, (
        f"no zenith in {CLAMP_SCAN_DEG} produced a negative raw height at "
        f"n_steps=50, so the clamp was never exercised here"
    )


# --------------------------------------------------------------------------
# 3. the kg -> g scaling order in the azimuth average
# --------------------------------------------------------------------------


def test_the_azimuth_average_scales_before_summing_in_one_tail_and_after_in_the_other():
    """``get_density`` converts per direction; the spline tail converts once.

    ``MSIS21LocationCentered.get_density`` accumulates
    ``densities[0] * 1.0e-3`` and divides by ``n``;
    ``calculate_density_spline`` accumulates ``densities[0]``, divides, then
    multiplies by ``1.0e-3``. Floating-point multiplication does not commute
    with a sum, so the two orders disagree on most samples.

    Measured on this host over the tail's own 36 per-direction arrays at
    zenith 60: 1497 of 2000 samples differ for IceCube and 1548 for ORCA, at
    9.2e-16 and 1.1e-15 relative. Asserted here as "they differ, by less than
    an ULP-scale bound".
    """
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    atm = dp.MSIS21IceCubeCentered("SouthPole", "January")

    captured: list[np.ndarray] = []
    real = atm._model.calc_altitude_array
    atm._model.calc_altitude_array = lambda _r=real, _c=captured, **kw: _capture(
        _c, _r(**kw)
    )
    try:
        atm.set_theta(60.0)  # averaged over n_azimuth directions
    finally:
        del atm._model.calc_altitude_array

    assert atm._azimuth_averaging
    assert len(captured) == atm._n_azimuth, (
        f"the averaged tail issued {len(captured)} backend calls for "
        f"{atm._n_azimuth} directions"
    )

    n = len(captured)
    sum_then_scale = np.zeros_like(captured[0])
    scale_then_sum = np.zeros_like(captured[0])
    for rho_kg in captured:
        sum_then_scale += rho_kg
        scale_then_sum += rho_kg * 1.0e-3
    sum_then_scale = sum_then_scale / n * 1.0e-3
    scale_then_sum = scale_then_sum / n

    assert not np.array_equal(sum_then_scale, scale_then_sum), (
        "the two kg->g orders now agree bitwise, so either "
        "MSIS21LocationCentered.get_density and its calculate_density_spline "
        "have been unified — say which order was kept — or the arrays are "
        "degenerate"
    )
    n_diff = int(np.count_nonzero(sum_then_scale != scale_then_sum))
    maxrel = float(
        np.max(np.abs(sum_then_scale - scale_then_sum) / np.abs(scale_then_sum))
    )
    assert 0 < maxrel < 1e-14, (
        f"the scaling order moves the averaged density by {maxrel:.3e} "
        f"relative at {n_diff} of {sum_then_scale.size} samples. Above an "
        f"ULP-scale bound that is not a reassociation any more."
    )


def _capture(sink, result):
    sink.append(np.array(result.densities[0], copy=True))
    return result


def test_the_two_msis21_backends_disagree_more_than_the_scaling_order_does():
    """``get_density`` and the spline tail also differ in *which* backend call.

    One is scalar ``calc`` per height, the other one batched
    ``calc_altitude_array``, and that difference is two orders larger than the
    scaling order above — 1.9e-13 against 1e-15 on this host. So a phase that
    unifies only the scaling order has not made the two paths agree, and this
    is what says so.
    """
    pytest.importorskip("nrlmsis", reason="MSIS21 is opt-in")
    atm = dp.MSIS21Atmosphere("SouthPole", "January")
    atm.set_theta(60.0)

    dl = np.linspace(0.0, atm.geom.path_len(atm.thrad), 200)
    h = atm.geom.h(dl, atm.thrad)
    batched = (
        atm._model.calc_altitude_array(
            day=atm._doy,
            utsec=atm._sec,
            z=np.maximum(h / 1.0e5, 0.0),
            lat=atm._lat,
            lon=atm._lon,
            sfluxavg=atm._f107a,
            sflux=atm._f107,
            ap=atm._ap,
        ).densities[0]
        * 1.0e-3
    )
    scalar = np.array([atm.get_density(float(x)) for x in h])

    maxrel = float(np.max(np.abs(batched - scalar) / np.abs(scalar)))
    assert maxrel > 0.0, (
        "the scalar and batched nrlmsis paths now agree bitwise; the "
        "get_density / calculate_density_spline split no longer costs anything"
    )
    assert maxrel < 1e-11, (
        f"scalar calc and calc_altitude_array differ by {maxrel:.3e} relative, "
        f"far above the 1.9e-13 this pins — a backend regression, not a "
        f"reassociation"
    )


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
