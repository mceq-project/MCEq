"""Golden section ``environment``: atmospheres, their raw splines, and geometry.

Sweeps (atmosphere class x zenith x azimuth mode x observation level) and pins,
per cell, the three numbers :meth:`calculate_density_spline` leaves behind
(``max_X``, ``max_den``, ``max_theta``), the FITPACK knot and
coefficient arrays of ``s_h2X`` / ``s_X2rho`` / ``s_lX2h``, the public spline
wrappers on a per-cell probe grid, the :class:`EarthGeometry` state, and the
impact-point geography of the detector-centred models. Beside the sweep it pins
:class:`GeneralizedTarget` and the three location-free helpers of
``gtracr_cutoff``.

Why the section exists
----------------------
``paths`` covers 3 of the 11 concrete atmosphere classes at 4 zeniths, pins no
raw spline, and owns only the ``etd2_nonuniform_path`` outputs
``(nsteps, dX, rho_inv, grid_idcs)``. That leaves unpinned:

* the two azimuth-averaged ``LocationCentered`` spline tails
  (``density_profiles.py:925-982``, ``msis21_atmosphere.py:465-545``) — roughly
  130 of the ~330 lines the Phase-3 MSIS collapse merges;
* ``max_den``, and every raw spline: ``s_h2X``, ``s_X2rho``, ``s_lX2h``, and
  the ``h2X`` / ``X2h`` wrappers over them;
* every ``h_obs`` mutation through ``EarthsAtmosphere.set_h_obs``, and
  ``use_loc_altitudes``;
* MSIS21 and MSIS00 beyond a single site;
* :class:`GeneralizedTarget`, whose ``s_h2X`` / ``s_X2h`` are a different
  degree-1 pair;
* anything at all in ``gtracr_cutoff.py``.

Four spline tails compute the same ``rho(X)`` differently, and the raw
coefficients are what tells them apart: the ``paths`` section sees only
``r_X2rho`` sampled on the integration steps, which the differences below do
not reach. See ``tests/geometry/test_environment_pins.py`` for the three
divergences among the tails that this section's numbers freeze and that test
states as behaviour.

What the sweep buys
-------------------
84 cells over 10 of the 11 concrete classes — every one but
:class:`AIRSAtmosphere`, whose reason is stored in ``meta/uncovered`` — and no
two of them stored the same numbers, which is the section's own consistency
check (:func:`_check_invariants`). The collision structure of the 250 knot and
250 coefficient digests is the second:

* **195 distinct knot digests of 250.** With ``s = 0.0`` FITPACK interpolates,
  so ``s_h2X``'s knots are fixed by the ``h_intp`` samples alone -- not equal
  to them, since at ``k = 2`` the interior knots fall between the data points
  and only the two boundary knots coincide -- hence they depend on
  ``(thrad, h_obs, h_form)`` alone. 18 geometries are shared by several cells,
  and there the CORSIKA, isothermal, MSIS00 and MSIS21 knots coincide bitwise
  across completely different density models. That is the invariant, not a
  defect, and the generator asserts it in both directions.
* **247 distinct coefficient digests of 250.** The three collisions are one
  cell: ``hobs/corsika_usstd/theta0_h200000``, whose three splines equal
  ``corsika_usstd/theta0``'s because ``set_h_obs`` guards its respline with
  ``if self.theta_deg:`` and ``0.0`` is falsy. Pinned as
  ``respline_fires = 0``.
* **3 of 27 geometries where the two ``geom.h`` forms disagree**, by
  1.1920929e-07 cm — one ULP of ``r_E`` — in ``geom_h/forms``.

Cost, and why it is DB-free
---------------------------
No :class:`~MCEq.core.MCEqRun`, no HDF5 database and no fork. An atmosphere
object needs only its own parameter tables plus, for the MSIS families, the
ctypes ``nrlmsise00`` shared library (MSIS00, built with the package) or the
optional ``nrlmsis`` Fortran extension (MSIS21). Only the MSIS21 cells are
gated on that package, the way the rest of the tree gates it with
``importorskip``; ``config.mceq_db_fname`` is pinned so
``make_provenance`` records a known database digest, but the file need not
exist and the section does not assert that it does.

84 cells at 15 s of measured phases (17 s wall including imports and the
provenance digests): 3.5 s for the azimuth-independent atmospheres, 11.0 s for
the detector-centred ones, 0.25 s for the observation-level cells. The one
expensive cell family is the azimuth-averaged MSIS00 tail — 36 directions x
2000 heights through the scalar ctypes wrapper, 0.89 s each — against 0.19 s
for the batched MSIS21 equivalent, which is why
:data:`AVERAGED_ZENITHS_MSIS00` is shorter than
:data:`AVERAGED_ZENITHS_MSIS21` and two of the four MSIS00 sites run
:data:`AVERAGED_ZENITHS_MSIS00_ONE`.

``get_density`` in azimuth-averaging mode is the other Python-loop cost, and
:data:`PROBE_HEIGHTS_AVERAGED_CM` is the answer to it: six probe heights there
cost 3.7 s to say what two say.

Comparison mode
---------------
Bitwise, and ``golden_host``. The knot and coefficient arrays are stored as
sha256 digests — 84 cells x 3 splines x ~2000 float64 knots is 3 MB of npz
otherwise — and a digest admits no tolerance at all, which is the same reason
``operators1d`` is host-pinned.

That is the *whole* reason, and the thread measurement is what says so rather
than the default. Rebuilding the section with ``OMP``/``OPENBLAS``/``MKL``/
``NUMEXPR``/``VECLIB`` thread counts at 1, 2, 4 and 8 reproduces all 2574 keys
bitwise: FITPACK's band solve is serial Fortran and not a BLAS call at all,
``cumulative_trapezoid`` and the ``nrlmsis`` vectorised path are pure numpy,
and the one BLAS-shaped operation in the builder is the 3x3 ``T @ d_ENU`` of
``_impact_point``. So threads are left ambient and recorded in
``extra.blas_threads`` rather than pinned with ``config.set_mkl_threads``,
which would leave a process-wide limiter behind.

Thread invariance is not build invariance, though. A quadratic interpolating
spline fitted through 2000 MSIS densities is at least as sensitive to a
numpy/scipy bump as the solves are — those move by 1e-14 relative L2 between
numpy 2.2/scipy 1.15 and numpy 2.5/scipy 1.18 — and here the sensitive
quantity is a hash. Hence ``golden_host``.

Nothing in this section is sparse, so it calls
:func:`._harness.assert_canonical_csr` nowhere: the environment layer produces
splines, floats and geography, and the only matrix anywhere near it is the 3x3
ENU->ECEF rotation of ``_impact_point``. The canonical-CSR guard belongs to the
operator sections, which is where the assembled operators are.
"""

from __future__ import annotations

import time

import numpy as np

from ._harness import array_digest, blas_threads, make_provenance

SECTION = "environment"

#: Config globals this section fixes. The atmospheres read ``r_E``, ``h_atm``,
#: ``h_obs`` and ``max_density`` out of ``config.environment`` at construction,
#: so all four are pinned or the section describes whatever the host had set.
#: ``mceq_db_fname`` is pinned for ``make_provenance``'s database digest only.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": "mceq_db_v140reduced_compact.h5",
    "r_E": 6371.315e3,
    "h_obs": 0.0,
    "h_atm": 112.8e3,
    "max_density": 0.001225,
    "len_target": 1000.0,
    "env_density": 0.001225,
    "env_name": "air",
    # Read by build_phi0_with_cutoff through `config.physics`.
    "minimal_primary_energy": 3.0,
}

#: Sample count of every ``calculate_density_spline`` call. The default of all
#: four tails; named here because the cell values are a function of it.
N_STEPS = 2000

#: Zeniths for the azimuth-independent classes. 0 is the vertical special case,
#: 60 the generic slant, 85 the near-horizontal knee and 90 the horizon, which
#: is ``max_theta`` for every one of them.
ZENITHS = (0.0, 60.0, 85.0, 90.0)

#: Zeniths for a second site of the same class — enough to say the site reaches
#: the numbers, not a second full sweep.
ZENITHS_SITE2 = (0.0, 60.0)

#: Single-azimuth zeniths for the detector-centred models, with the azimuth.
#: 0/90 deg at one zenith says the azimuth reaches ``_impact_point``; 120 deg
#: zenith is upgoing, where ``set_theta`` moves the observation level to sea
#: level and the impact point to the far-side crossing.
SINGLE_AZIMUTH_CELLS = ((60.0, 0.0), (60.0, 90.0), (120.0, 0.0))

#: Azimuth-averaged zeniths for the MSIS21 detector-centred models. 91 deg is
#: inside the grazing window of a buried detector (theta <= 90 + dip stays a
#: near-side column) and is where the ``z >= 0`` clamp is live; 120 deg is
#: upgoing, where the column stands on the far-side crossing at sea level.
AVERAGED_ZENITHS_MSIS21 = (0.0, 60.0, 85.0, 91.0, 120.0)

#: The same for MSIS00, at two sites only and two zeniths. Its averaged tail is
#: the height-major scalar ctypes loop — 36 directions x 2000 heights, 0.89 s
#: per cell against 0.19 s for the batched MSIS21 one — and it is one tail, so
#: a third and fourth site would pay 1.8 s to state at a new latitude what the
#: first two already state. :data:`AVERAGED_ZENITHS_MSIS00_ONE` is what the
#: other two sites run.
AVERAGED_ZENITHS_MSIS00 = (60.0, 91.0)
AVERAGED_ZENITHS_MSIS00_ONE = (60.0,)

#: Fractions of ``[min_X, max_X]`` at which the depth-keyed wrappers are
#: probed, where ``min_X`` is the first ``s_X2rho`` knot. Every point is inside
#: the fitted domain by construction, including the two endpoints, so the
#: section never pins an extrapolation it did not mean to.
X_FRACTIONS = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.999, 1.0)

#: Heights in cm at which ``h2X`` and ``get_density`` are probed. Fixed rather
#: than per-cell so the same physical altitude is comparable across cells;
#: 112.8e5 is the top of the atmosphere.
PROBE_HEIGHTS_CM = (0.0, 1.0e5, 5.0e5, 20.0e5, 50.0e5, 112.8e5)

#: ``get_density`` heights in azimuth-averaging mode. Two, not six: there the
#: method is a Python loop over ``n_azimuth`` backend calls per height — 36
#: scalar ``nrlmsis`` evaluations at 1.25 ms each, or 36 ctypes calls for
#: MSIS00 — and it costs 3.7 s of the section to say at six heights what it
#: says at two. ``h2X`` still uses the full :data:`PROBE_HEIGHTS_CM`: that one
#: is a spline evaluation and free.
PROBE_HEIGHTS_AVERAGED_CM = (0.0, 20.0e5)

#: ``(label, class, args, kwargs)`` per atmosphere. The classes are named, not
#: imported, so this module stays MCEq-free until :func:`build` runs; the
#: MSIS21 names are re-exported from ``density_profiles`` through its PEP-562
#: ``__getattr__``. ``needs_nrlmsis`` is carried by the label prefix.
ATMOSPHERES = (
    ("corsika_usstd", "CorsikaAtmosphere", ("BK_USStd", None), {}),
    ("corsika_sp_jun", "CorsikaAtmosphere", ("SouthPole", "June"), {}),
    ("isothermal", "IsothermalAtmosphere", ("SouthPole", "January"), {}),
    ("msis00_sp_jan", "MSIS00Atmosphere", ("SouthPole", "January"), {}),
    ("msis00_ka_jun", "MSIS00Atmosphere", ("Karlsruhe", "June"), {}),
    (
        "msis00_sp_localt",
        "MSIS00Atmosphere",
        ("SouthPole", "January"),
        {"use_loc_altitudes": True},
    ),
    ("msis21_sp_jan", "MSIS21Atmosphere", ("SouthPole", "January"), {}),
    ("msis21_ka_jun", "MSIS21Atmosphere", ("Karlsruhe", "June"), {}),
    (
        "msis21_sp_localt",
        "MSIS21Atmosphere",
        ("SouthPole", "January"),
        {"use_loc_altitudes": True},
    ),
)

#: Detector geometry for the two ``*LocationCentered`` cells that exercise the
#: base class directly. All three knobs are non-zero and none matches either
#: subclass (IceCube is depth 1948 / elevation 2835 at the pole, KM3NeT is a
#: deep-sea depth at zero elevation), so the cell says the base class works at
#: a configuration neither subclass reaches.
#:
#: A ``depth_m`` of 0 would make the cell useless: ``_impact_point`` solves
#: ``|P_det + x d|^2 = r^2`` and with ``r_det == r_obs`` the larger root is
#: ``x = -A + |A| = 0`` for any downgoing direction, so the impact point is the
#: detector itself and the azimuth is a measured no-op — the two az cells came
#: out bitwise identical, splines and all, on the first build of this section.
GENERIC_SITE = {
    "detector_coord": (-70.0, -30.0),
    "depth_m": 1000.0,
    "surface_elevation_m": 500.0,
    "max_theta": 180.0,
}

#: ``(label, class, args, kwargs, averaged_zeniths)``. The detector-centred
#: models, which take ``set_theta(theta, azimuth)``. ``*LocationCentered``
#: itself is covered directly, not only through its two subclasses: it is the
#: class the Phase-3 collapse merges, and its ``max_theta`` /
#: ``surface_elevation_m`` defaults differ from both.
CENTERED = (
    (
        "msis00_generic",
        "MSIS00LocationCentered",
        (GENERIC_SITE["detector_coord"], GENERIC_SITE["depth_m"]),
        {
            "season": "January",
            "max_theta": GENERIC_SITE["max_theta"],
            "surface_elevation_m": GENERIC_SITE["surface_elevation_m"],
        },
        AVERAGED_ZENITHS_MSIS00,
    ),
    (
        "msis00_icecube",
        "MSIS00IceCubeCentered",
        ("SouthPole", "January"),
        {},
        AVERAGED_ZENITHS_MSIS00,
    ),
    (
        "msis00_orca",
        "MSIS00KM3NeTCentered",
        ("ORCA",),
        {"season": "January"},
        AVERAGED_ZENITHS_MSIS00_ONE,
    ),
    (
        "msis00_arca",
        "MSIS00KM3NeTCentered",
        ("ARCA",),
        {"season": "January"},
        AVERAGED_ZENITHS_MSIS00_ONE,
    ),
    (
        "msis21_generic",
        "MSIS21LocationCentered",
        (GENERIC_SITE["detector_coord"], GENERIC_SITE["depth_m"]),
        {
            "season": "January",
            "max_theta": GENERIC_SITE["max_theta"],
            "surface_elevation_m": GENERIC_SITE["surface_elevation_m"],
        },
        AVERAGED_ZENITHS_MSIS21,
    ),
    (
        "msis21_icecube",
        "MSIS21IceCubeCentered",
        ("SouthPole", "January"),
        {},
        AVERAGED_ZENITHS_MSIS21,
    ),
    (
        "msis21_orca",
        "MSIS21KM3NeTCentered",
        ("ORCA",),
        {"season": "January"},
        AVERAGED_ZENITHS_MSIS21,
    ),
    (
        "msis21_arca",
        "MSIS21KM3NeTCentered",
        ("ARCA",),
        {"season": "January"},
        AVERAGED_ZENITHS_MSIS21,
    ),
)

#: ``(label, class, args, kwargs, theta_deg, h_obs_cm)`` for the observation-
#: level cells. ``EarthsAtmosphere.set_h_obs`` is reached by nothing else in
#: the harness. The two zeniths are the point of the pair: the method guards
#: its respline with ``if self.theta_deg:``, so at ``theta_deg == 0.0`` — a
#: falsy float — the geometry moves and the spline does not.
H_OBS_CELLS = (
    ("corsika_usstd", "CorsikaAtmosphere", ("BK_USStd", None), {}, 60.0, 2.0e5),
    ("corsika_usstd", "CorsikaAtmosphere", ("BK_USStd", None), {}, 0.0, 2.0e5),
    ("msis00_sp_jan", "MSIS00Atmosphere", ("SouthPole", "January"), {}, 60.0, 1.0e5),
    ("msis21_sp_jan", "MSIS21Atmosphere", ("SouthPole", "January"), {}, 60.0, 1.0e5),
)

#: Class names that need the optional ``nrlmsis`` package.
MSIS21_CLASSES = frozenset(
    {
        "MSIS21Atmosphere",
        "MSIS21LocationCentered",
        "MSIS21IceCubeCentered",
        "MSIS21KM3NeTCentered",
    }
)

#: Atmosphere classes the section cannot construct, with the reason. Reported
#: as a golden array so the gap is a compared fact rather than a claim in prose.
UNCOVERED = (
    "AIRSAtmosphere: init_parameters reads tabulated AIRS files from a "
    "hard-coded ~/OneDrive path unless a table_path kwarg supplies them, and "
    "imports matplotlib.dates, which is not a runtime dependency "
    "(tests/geometry/test_density_profiles.py::test_airs_instantiation is "
    "xfail for the same reason)",
)

#: ``(label, key_arguments)`` for :func:`MCEq.geometry.gtracr_cutoff._cache_key`.
GTRACR_KEY_CELLS = (
    ("igrf_default", ("IceCube", "igrf", "2020-01-01", 100.0, 30000, 0.1, 55.0, 0.5)),
    ("table_orca", ("ORCA", "table", "2026-09-03", 80.0, 1000, 0.5, 20.0, 0.25)),
)

#: Rigidity cutoffs in GV for the :func:`build_phi0_with_cutoff` cell. Spans
#: the mass-group thresholds: 0 admits everything, 55 is the top of the default
#: gtracr scan, and the middle values cut each ``Z * R_c`` in turn.
PHI0_CUTOFFS_GV = (0.0, 1.0, 5.0, 17.0, 55.0)

NOTE = """\
Atmospheres, their raw FITPACK splines and the Earth geometry, over
(atmosphere class x zenith x azimuth mode x observation level). DB-free,
fork-free, MCEqRun-free: an atmosphere needs only its own parameter tables plus
the ctypes nrlmsise00 library (MSIS00) or the optional nrlmsis extension
(MSIS21, and the only thing this section gates on).

Per cell: max_X, max_den, max_theta, theta_deg, thrad; the sha256 of
the knot and coefficient arrays of s_h2X / s_X2rho / s_lX2h with their degree
and lengths; r_X2rho / X2h at eight fractions of [min_X, max_X] -- min_X being
the first s_X2rho knot, i.e. the fitted lower bound -- and h2X at
six fixed heights; get_density at those six heights, or at two of them in
azimuth-averaging mode, where the method is a Python loop over n_azimuth
backend calls per height; the seven EarthGeometry scalars; and for the
detector-centred models the azimuth mode, the impact latitude/longitude and
the full (n_azimuth, 2) impact-coordinate table that pins _impact_point.

84 cells, all mutually distinct. 195 of the 250 knot digests are distinct
because s_h2X's knots are fixed by the h_intp samples, hence geometry: 18
geometries are shared by several cells and the knots coincide there across
every density model. 247 of the 250 coefficient digests are distinct, the
three collisions being the falsy-theta_deg set_h_obs cell below.

build_phi0_with_cutoff is pinned against a stub run supplying exactly the five
things it reads -- _energy_grid.c, dim_states, and the (lidx, uidx, mass,
is_resonance) of the proton and neutron pman entries -- so the golden is exact
for the function while the section stays free of the HDF5 database an MCEqRun
would open. get_cutoff_map is NOT pinned: it imports gtracr before it reaches
the cache branch, so its interpolation is unreachable without the package.

Coverage: 11 concrete EarthsAtmosphere subclasses exist and 10 are covered.
AIRSAtmosphere is not -- it reads tabulated files from a hard-coded ~/OneDrive
path and imports matplotlib.dates -- and the reason is stored in
meta/uncovered. GeneralizedTarget is covered beside the sweep: it is not an
EarthsAtmosphere (no zenith, k=1 splines, s_X2h rather than s_lX2h) and had no
golden coverage either.

Pinned behaviours beyond the numbers:

- set_h_obs guards its respline with `if self.theta_deg:`. theta_deg == 0.0 is
  falsy, so the h_obs_theta0 cell moves r_obs and leaves max_X and every
  spline at the pre-move values, while the theta60 cell resplines. Both are
  cells, and hobs/respline_fires records the difference directly.
- _preserve_max_theta: set_h_obs overwrites max_theta with geom.theta_max_deg
  on the plain atmospheres and must not on the detector-centred ones, which
  allow upgoing angles. hobs/max_theta_{before,after} pins both directions.
- The zenith guard of the LocationCentered classes: set_theta above max_theta
  raises ValueError, where the base classes raise a bare Exception (paths pins
  that one). Stored as the exception type name.
- gtracr_cutoff._gtracr_location_from_atmosphere names ORCA and ARCA from the
  _detector_name the KM3NeT subclasses set beside the coordinates, IceCube
  from the pole, and anything else from a coord stub. The coordinates are
  MCEq's in every branch, and get_cutoff_map samples at exactly those -- it
  hands gtracr an explicit Location and never a bare site name, so gtracr's
  own table cannot move the sampling point away from the column MCEq
  integrates. gtracr/cache_version pins the CACHE_VERSION that states it.
- The two geom.h call forms. Two of the four spline tails call geom.h once on
  the dl vector and two loop over the samples, and the two do not agree
  bitwise: numpy lowers arr ** 2 to a multiply while a float64 scalar goes
  through libm pow, and the - r_E cancellation turns 1 ULP of ~4e17 into 1 ULP
  of 6.37e8 cm. Every cell carries the form its tail uses in h_form, and
  geom_h/forms records (thrad, h_obs, n_differing, max_abs_diff_cm) at all 27
  geometries the sweep visits: 3 of them differ, at 1.1920929e-07 cm.
  tests/geometry/test_environment_pins.py states the same thing as behaviour,
  and also pins the z >= 0 clamp (present in the two MSIS21 tails, absent from
  the base and azimuth-averaged-MSIS00 ones, and live) and the kg->g scaling
  order of the MSIS21 azimuth average.

BLAS threads are ambient and recorded in extra.blas_threads: rebuilding with
the thread env vars at 1, 2, 4 and 8 reproduces all 2574 keys bitwise. The
section is golden_host anyway -- the spline knots and coefficients are sha256
digests, which admit no tolerance at all.
"""


# --------------------------------------------------------------------------
# recorders
# --------------------------------------------------------------------------


def _record_spline(arrays, prefix, spline):
    """Degree, lengths, endpoint knots and the digests of one FITPACK spline.

    ``get_knots`` and ``get_coeffs`` are the fitted ``t`` and ``c`` less the
    ``k`` repeated boundary knots and the trailing zero coefficients, so
    together with ``k`` they carry the whole spline. They are digested rather
    than stored: 84 cells x 3 splines x ~2000 float64 knots is 3 MB of npz,
    against 64 bytes of hex each.

    The endpoint knots are stored raw beside the digests so a mismatch says
    which fitted domain moved instead of only that two hashes differ.

    The degree comes off ``_eval_args``, the ``(t, c, k)`` triple the evaluator
    binds; ``UnivariateSpline`` exposes no public ``k``.
    """
    knots = np.asarray(spline.get_knots(), dtype=np.float64)
    coeffs = np.asarray(spline.get_coeffs(), dtype=np.float64)
    arrays[f"{prefix}/k"] = np.int64(spline._eval_args[2])
    arrays[f"{prefix}/n_knots"] = np.int64(knots.size)
    arrays[f"{prefix}/n_coeffs"] = np.int64(coeffs.size)
    arrays[f"{prefix}/knot_range"] = np.array([knots[0], knots[-1]], dtype=np.float64)
    arrays[f"{prefix}/knots"] = np.asarray(array_digest(knots))
    arrays[f"{prefix}/coeffs"] = np.asarray(array_digest(coeffs))


def _record_geometry(arrays, prefix, geom):
    """The seven cached scalars of an :class:`EarthGeometry`.

    ``r_top``, ``r_obs`` and the two ``theta_max`` values are derived state
    that ``set_h_obs`` recomputes, so storing them says whether a move of the
    observation level propagated.
    """
    arrays[f"{prefix}/geom"] = np.array(
        [
            geom.h_obs,
            geom.h_atm,
            geom.r_E,
            geom.r_top,
            geom.r_obs,
            geom.theta_max_rad,
            geom.theta_max_deg,
        ],
        dtype=np.float64,
    )


def _record_atmosphere(arrays, prefix, atm, h_form):
    """One configured atmosphere: scalars, splines, probes, geometry.

    Called after ``set_theta``, so every spline and every ``_max_*`` attribute
    is the one that call produced.

    ``h_form`` is ``"array"`` when this cell's ``calculate_density_spline``
    calls ``geom.h`` once on the whole ``dl`` vector and ``"scalar"`` when it
    calls it per sample. It is a stored, compared fact because the two forms do
    not agree bitwise — see :func:`_check_invariants` and
    ``tests/geometry/test_environment_pins.py`` — so a phase that unifies the
    four tails has to change this array and say which form it chose.

    ``min_X`` is the lower end of the ``s_X2rho`` fitted domain, read off the
    spline's first knot rather than from an attribute: with ``s = 0.0``
    FITPACK interpolates, so that knot *is* the first cumulative-trapezoid
    depth the tail integrated. It bounds the probe grid only — it is not
    pinned as a scalar, because no MCEq code reads it and the knot digests
    below already fix it.
    """
    arrays[f"{prefix}/h_form"] = np.asarray(h_form)
    min_X = float(atm.s_X2rho.get_knots()[0])
    max_X = float(atm.max_X)
    arrays[f"{prefix}/scalars"] = np.array(
        [
            float(atm.theta_deg),
            float(atm.thrad),
            float(atm.max_theta),
            max_X,
            float(atm.max_den),
        ],
        dtype=np.float64,
    )

    for name in ("s_h2X", "s_X2rho", "s_lX2h"):
        _record_spline(arrays, f"{prefix}/{name}", getattr(atm, name))

    X_probe = np.array(
        [min_X + f * (max_X - min_X) for f in X_FRACTIONS], dtype=np.float64
    )
    h_probe = np.asarray(PROBE_HEIGHTS_CM, dtype=np.float64)
    rho_probe = np.asarray(
        PROBE_HEIGHTS_AVERAGED_CM
        if getattr(atm, "_azimuth_averaging", False)
        else PROBE_HEIGHTS_CM,
        dtype=np.float64,
    )
    arrays[f"{prefix}/probe/X"] = X_probe
    arrays[f"{prefix}/probe/r_X2rho"] = np.asarray(atm.r_X2rho(X_probe), np.float64)
    arrays[f"{prefix}/probe/X2h"] = np.asarray(atm.X2h(X_probe), np.float64)
    arrays[f"{prefix}/probe/h2X"] = np.asarray(atm.h2X(h_probe), np.float64)
    arrays[f"{prefix}/probe/rho_h"] = rho_probe
    arrays[f"{prefix}/probe/get_density"] = np.array(
        [float(atm.get_density(float(h))) for h in rho_probe], dtype=np.float64
    )
    _record_geometry(arrays, f"{prefix}", atm.geom)


def _record_impact(arrays, prefix, atm):
    """The geography a detector-centred model resolved for the current angle.

    ``_azimuth_avg_coords`` is the whole ``_impact_point`` output for the
    averaged mode — ``n_azimuth`` latitude/longitude pairs — and is the only
    thing that pins that projection per direction rather than in aggregate.
    ``nan`` stands for the ``None`` the single-point properties carry in
    averaged mode, so the array keeps a float dtype.
    """
    arrays[f"{prefix}/azimuth_mode"] = np.asarray(
        "averaged" if atm._azimuth_averaging else "single"
    )
    arrays[f"{prefix}/effective_theta_deg"] = np.float64(atm._effective_theta_deg)
    arrays[f"{prefix}/azimuth_deg"] = np.float64(
        np.nan if atm._current_azimuth_deg is None else atm._current_azimuth_deg
    )
    arrays[f"{prefix}/impact"] = np.array(
        [
            np.nan
            if atm.current_impact_latitude is None
            else atm.current_impact_latitude,
            np.nan
            if atm.current_impact_longitude is None
            else atm.current_impact_longitude,
        ],
        dtype=np.float64,
    )
    arrays[f"{prefix}/impact_coords"] = np.asarray(
        atm._azimuth_avg_coords, dtype=np.float64
    ).reshape(-1, 2)


# --------------------------------------------------------------------------
# cell families
# --------------------------------------------------------------------------


def _sweep_plain(arrays, dprof, have21, labels):
    """The azimuth-independent atmospheres over :data:`ZENITHS`.

    The second-site and ``use_loc_altitudes`` variants run the shorter
    :data:`ZENITHS_SITE2`: what they have to say is that the site and the
    observation level reach the numbers, which two zeniths state as well as
    four.
    """
    for label, cls_name, args, kwargs in ATMOSPHERES:
        if cls_name in MSIS21_CLASSES and not have21:
            continue
        atm = getattr(dprof, cls_name)(*args, **kwargs)
        zeniths = (
            ZENITHS_SITE2 if label.endswith(("_ka_jun", "_sp_localt")) else ZENITHS
        )
        arrays[f"cells/{label}/class"] = np.asarray(cls_name)
        for theta in zeniths:
            cell = f"{label}/theta{theta:g}"
            atm.set_theta(theta)
            _record_atmosphere(arrays, f"cells/{cell}", atm, h_form(cls_name))
            labels.append(cell)


def _sweep_centered(arrays, dprof, have21, labels):
    """The detector-centred models over single-azimuth and averaged modes."""
    for label, cls_name, args, kwargs, averaged in CENTERED:
        if cls_name in MSIS21_CLASSES and not have21:
            continue
        atm = getattr(dprof, cls_name)(*args, **kwargs)
        arrays[f"cells/{label}/class"] = np.asarray(cls_name)
        arrays[f"cells/{label}/depends_on_azimuth"] = np.int64(atm.depends_on_azimuth)
        arrays[f"cells/{label}/preserve_max_theta"] = np.int64(atm._preserve_max_theta)

        for theta, azimuth in SINGLE_AZIMUTH_CELLS:
            if theta > atm.max_theta:
                continue
            cell = f"{label}/theta{theta:g}_az{azimuth:g}"
            atm.set_theta(theta, azimuth)
            _record_atmosphere(arrays, f"cells/{cell}", atm, h_form(cls_name))
            _record_impact(arrays, f"cells/{cell}", atm)
            labels.append(cell)

        for theta in averaged:
            if theta > atm.max_theta:
                continue
            cell = f"{label}/theta{theta:g}_avg"
            atm.set_theta(theta)
            _record_atmosphere(arrays, f"cells/{cell}", atm, h_form(cls_name))
            _record_impact(arrays, f"cells/{cell}", atm)
            labels.append(cell)

        # Zenith guard. The LocationCentered classes raise ValueError where the
        # base classes raise a bare Exception, which `paths` pins for those.
        arrays[f"cells/{label}/zenith_guard"] = np.asarray(
            _guard_exception(atm, atm.max_theta + 1e-4)
        )


def h_form(cls_name: str) -> str:
    """Which ``geom.h`` call form this class's spline tail uses.

    Four tails compute ``rho(X)``. ``EarthsAtmosphere`` (the CORSIKA,
    isothermal and named-MSIS00 route, and the MSIS00 detector-centred models
    in single-azimuth mode, which delegate to it) and the azimuth-averaged
    ``MSIS00LocationCentered`` tail both call ``geom.h`` once per ``dl``
    sample; the two MSIS21 tails call it once on the whole vector. So the split
    is exactly MSIS21 against everything else.

    It matters because the two forms do not agree bitwise: ``h`` squares
    ``A_1 + l - dl``, and numpy lowers ``arr ** 2`` to a multiply while a
    float64 scalar goes through libm ``pow``, which is 1 ULP off at some
    arguments. The ``- r_E`` at the end turns that into 1 ULP of 6.37e8 cm.
    """
    return "array" if cls_name in MSIS21_CLASSES else "scalar"


def _guard_exception(atm, theta_deg):
    """Name of the exception type ``set_theta`` raises past ``max_theta``."""
    try:
        atm.set_theta(theta_deg)
    except Exception as exc:
        return type(exc).__name__
    return "<none>"


def _sweep_h_obs(arrays, dprof, have21, labels):
    """``set_h_obs`` on a configured atmosphere, before and after.

    Two statements per cell. First, whether the respline fires: the method
    guards it with ``if self.theta_deg:`` and ``0.0`` is falsy, so the
    ``theta0`` cell moves ``r_obs`` while ``max_X`` and every spline keep the
    pre-move value. Second, ``max_theta``: it is overwritten with
    ``geom.theta_max_deg`` unless ``_preserve_max_theta``.
    """
    for label, cls_name, args, kwargs, theta, h_obs in H_OBS_CELLS:
        if cls_name in MSIS21_CLASSES and not have21:
            continue
        atm = getattr(dprof, cls_name)(*args, **kwargs)
        atm.set_theta(theta)
        before_max_X = float(atm.max_X)
        before_h2X = np.asarray(atm.h2X(np.asarray(PROBE_HEIGHTS_CM)), np.float64)
        before_max_theta = float(atm.max_theta)

        cell = f"hobs/{label}/theta{theta:g}_h{h_obs:g}"
        atm.set_h_obs(h_obs)
        _record_atmosphere(arrays, f"cells/{cell}", atm, h_form(cls_name))
        arrays[f"cells/{cell}/max_theta_before"] = np.float64(before_max_theta)
        arrays[f"cells/{cell}/respline_fires"] = np.int64(
            not (
                float(atm.max_X) == before_max_X
                and np.array_equal(
                    np.asarray(atm.h2X(np.asarray(PROBE_HEIGHTS_CM)), np.float64),
                    before_h2X,
                )
            )
        )
        labels.append(cell)

    # The preserving side of the same flag, on a detector-centred model. Its
    # own set_theta rewrites h_obs, so the flag is read straight after the
    # set_h_obs call rather than through a respline.
    for label, cls_name, args, kwargs in (
        ("msis00_icecube", "MSIS00IceCubeCentered", ("SouthPole", "January"), {}),
    ):
        atm = getattr(dprof, cls_name)(*args, **kwargs)
        before = float(atm.max_theta)
        atm.set_h_obs(1.0e5)
        arrays[f"cells/hobs/{label}/preserved_max_theta"] = np.array(
            [before, float(atm.max_theta), float(atm.geom.theta_max_deg)],
            dtype=np.float64,
        )


def _record_generalized_target(arrays, dprof, labels):
    """:class:`GeneralizedTarget`: the piecewise-constant, zenith-free target.

    Not an :class:`EarthsAtmosphere` — no ``set_theta``, degree-1 splines and
    ``s_X2h`` in place of ``s_lX2h`` — and covered by no other section. Two
    cells: the single-material default, and a three-layer build through
    ``add_material``, which is what exercises ``_integrate``.
    """
    for cell, materials in (
        ("gtarget/default", ()),
        (
            "gtarget/three_layer",
            ((20.0e2, 1.0, "water"), (50.0e2, 2.65, "rock")),
        ),
    ):
        tgt = dprof.GeneralizedTarget(
            len_target=100.0e2, env_density=0.001225, env_name="air"
        )
        for start, density, name in materials:
            tgt.add_material(start, density, name)

        prefix = f"cells/{cell}"
        arrays[f"{prefix}/scalars"] = np.array(
            [float(tgt.max_X), float(tgt.max_den), float(tgt.len_target)],
            dtype=np.float64,
        )
        # `tgt.knots` are the material boundaries in cm, not a spline knot
        # vector; stored under a name that does not collide with the
        # `<spline>/knots` digest keys the summary counts.
        arrays[f"{prefix}/material_bounds"] = np.asarray(tgt.knots, dtype=np.float64)
        arrays[f"{prefix}/X_int"] = np.asarray(tgt.X_int, dtype=np.float64)
        arrays[f"{prefix}/densities"] = np.asarray(tgt.densities, dtype=np.float64)
        for name in ("s_h2X", "s_X2h"):
            _record_spline(arrays, f"{prefix}/{name}", getattr(tgt, name))
        X_probe = np.array([f * tgt.max_X for f in X_FRACTIONS], dtype=np.float64)
        l_probe = np.array([f * tgt.len_target for f in X_FRACTIONS], dtype=np.float64)
        arrays[f"{prefix}/probe/X"] = X_probe
        arrays[f"{prefix}/probe/r_X2rho"] = np.array(
            [float(tgt.r_X2rho(float(X))) for X in X_probe], dtype=np.float64
        )
        arrays[f"{prefix}/probe/get_density_X"] = np.array(
            [float(tgt.get_density_X(float(X))) for X in X_probe], dtype=np.float64
        )
        # `rho_h` is the abscissa of `probe/get_density` in every cell; here it
        # is a position along the target in cm rather than an altitude, since
        # `GeneralizedTarget.get_density` takes `l_cm`.
        arrays[f"{prefix}/probe/rho_h"] = l_probe
        arrays[f"{prefix}/probe/get_density"] = np.asarray(
            tgt.get_density(l_probe), dtype=np.float64
        )
        labels.append(cell)


# --------------------------------------------------------------------------
# gtracr_cutoff
# --------------------------------------------------------------------------


class _StubRun:
    """The five attributes :func:`build_phi0_with_cutoff` reads off an MCEqRun.

    The function's output is a pure function of ``_energy_grid.c``,
    ``dim_states`` and the ``(lidx, uidx, mass, is_resonance)`` of the proton
    and neutron entries of ``pman``, plus the primary model and the cutoff map.
    Supplying exactly those makes the golden exact for the function under test
    while keeping the section free of the HDF5 database an
    :class:`~MCEq.core.MCEqRun` would open.
    """

    class _Particle:
        def __init__(self, mass, lidx, uidx, is_resonance=False):
            self.mass = mass
            self.lidx = lidx
            self.uidx = uidx
            self.is_resonance = is_resonance

    class _Grid:
        def __init__(self, c):
            self.c = c

    def __init__(self, e_grid, n_species=4):
        dim = e_grid.size
        self._energy_grid = self._Grid(e_grid)
        self.dim_states = dim * n_species
        # Proton in block 0, neutron in block 2, so a phi0 that wrote the two
        # into the same rows would show up.
        self._pman = {
            (2212, 0): self._Particle(0.93827, 0, dim),
            (2112, 0): self._Particle(0.93957, 2 * dim, 3 * dim),
        }

    @property
    def pman(self):
        return self._pman


def _record_gtracr(arrays):
    """The three parts of ``gtracr_cutoff`` that need neither gtracr nor a DB.

    ``get_cutoff_map`` is not among them: it imports ``gtracr`` before it
    reaches the cache branch, so there is no way to exercise its interpolation
    without the package installed. What is here is the cache key, the location
    resolution and the rigidity-cutoff arithmetic.
    """
    import crflux.models as pm

    from MCEq.geometry import density_profiles as dprof
    from MCEq.geometry import gtracr_cutoff as gc

    for label, args in GTRACR_KEY_CELLS:
        key_str, key_hash = gc._cache_key(*args)
        arrays[f"gtracr/cache_key/{label}"] = np.array([key_str, key_hash])
    arrays["gtracr/cache_version"] = np.int64(gc.CACHE_VERSION)
    arrays["gtracr/native_grid"] = np.array(
        [gc._NATIVE_N_ZEN, gc._NATIVE_N_AZ], dtype=np.int64
    )
    arrays["gtracr/default_mass_groups"] = np.asarray(
        gc._DEFAULT_MASS_GROUPS, dtype=np.int64
    )

    # Location resolution. The lat/lon columns are the load-bearing ones:
    # get_cutoff_map samples gtracr at exactly them for every row, so they are
    # what a cutoff map belongs to. The name column only labels them.
    rows = []
    for label, maker in (
        (
            "MSIS00IceCubeCentered",
            lambda: dprof.MSIS00IceCubeCentered("SouthPole", "January"),
        ),
        (
            "MSIS00KM3NeTCentered/ORCA",
            lambda: dprof.MSIS00KM3NeTCentered("ORCA", "January"),
        ),
        (
            "MSIS00KM3NeTCentered/ARCA",
            lambda: dprof.MSIS00KM3NeTCentered("ARCA", "January"),
        ),
        (
            "MSIS00Atmosphere/Karlsruhe",
            lambda: dprof.MSIS00Atmosphere("Karlsruhe", "June"),
        ),
        (
            "CorsikaAtmosphere/BK_USStd",
            lambda: dprof.CorsikaAtmosphere("BK_USStd", None),
        ),
        (
            "IsothermalAtmosphere",
            lambda: dprof.IsothermalAtmosphere("SouthPole", "January"),
        ),
    ):
        try:
            name, lat, lon = gc._gtracr_location_from_atmosphere(maker())
            rows.append((label, name, f"{lat:.6f}", f"{lon:.6f}"))
        except ValueError:
            rows.append((label, "<ValueError>", "nan", "nan"))
    arrays["gtracr/locations"] = np.array(rows, dtype="U64")

    # build_phi0_with_cutoff on the stub run. A 12-bin decade grid keeps the
    # array small while spanning every mass-group threshold in PHI0_CUTOFFS_GV.
    e_grid = np.logspace(0.0, 5.0, 12)
    run = _StubRun(e_grid)
    model = pm.HillasGaisser2012("H3a")
    rc = np.asarray(PHI0_CUTOFFS_GV, dtype=np.float64)
    phi0 = gc.build_phi0_with_cutoff(run, model, rc)
    arrays["gtracr/phi0/e_grid"] = e_grid
    arrays["gtracr/phi0/rc_GV"] = rc
    arrays["gtracr/phi0/shape"] = np.asarray(phi0.shape, dtype=np.int64)
    arrays["gtracr/phi0/values"] = np.asarray(phi0, dtype=np.float64)
    arrays["gtracr/phi0/nonzero_rows"] = np.asarray(
        np.flatnonzero(np.any(phi0 != 0.0, axis=1)), dtype=np.int64
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _cell_signature(arrays, cell):
    """Everything the section stored under one cell, as a comparable tuple."""
    prefix = f"cells/{cell}/"
    out = []
    for key in sorted(arrays):
        if not key.startswith(prefix):
            continue
        value = np.asarray(arrays[key])
        out.append(
            (
                key[len(prefix) :],
                value.tolist() if value.dtype.kind in "US" else value.tobytes(),
            )
        )
    return tuple(out)


def _record_geom_h_forms(arrays, geometries) -> list:
    """``h_intp`` per scalar against ``h_intp`` on the vector, per geometry.

    The four spline tails build the same ``h_intp = h(dl[2:], thrad)`` reversed,
    two of them by looping ``geom.h`` over the samples and two by one array
    call, and the two do not agree bitwise: ``h`` squares ``A_1 + l - dl``, and
    numpy lowers ``arr ** 2`` to a multiply while a float64 scalar goes through
    libm ``pow``, which is 1 ULP off at some arguments. Subtracting ``r_E``
    then turns that 1 ULP of ~4e17 into 1 ULP of 6.37e8 cm — 1.192e-7 cm — on
    the height itself.

    Recorded as ``(thrad, h_obs, n_differing, max_abs_diff_cm)`` per row, at
    exactly the geometries the sweep visits, so any unification of the tails
    moves this array and has to say which form it kept. It is pure geometry —
    no atmosphere, no density model — hence re-derived from an
    :class:`EarthGeometry` here rather than read out of the cells.
    """
    from MCEq.geometry.geometry import EarthGeometry

    rows = []
    for thrad, h_obs in geometries:
        geom = EarthGeometry()
        geom.set_h_obs(h_obs)
        dl = np.linspace(0.0, geom.path_len(thrad), N_STEPS)
        as_array = geom.h(dl, thrad)[2:][::-1]
        per_scalar = np.array([geom.h(sample, thrad) for sample in dl[2:][::-1]])
        delta = np.abs(as_array - per_scalar)
        rows.append((thrad, h_obs, float(np.count_nonzero(delta)), float(delta.max())))
    arrays["geom_h/forms"] = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    return rows


def _check_invariants(arrays, cells) -> dict:
    """Assert what the sweep makes free, and return the summary it measures.

    Three statements, each a property of the atmospheres rather than of a
    number, and each one the section would stop making if a later phase quietly
    dropped an argument:

    * every cell is a distinct configuration — no two of the 80-odd cells
      stored the same numbers, so none of the sweep axes is a no-op;
    * ``s_h2X`` knots are *geometry*: with ``s = 0.0`` FITPACK interpolates, so
      the knots are the ``h_intp`` samples and depend on ``(thrad, h_obs)`` and
      the ``geom.h`` call form alone. Cells sharing all three therefore share
      the knots bitwise across completely different density models, and it is
      the *coefficients* that discriminate the atmosphere. Both halves are
      asserted, because a change that made the knots density-dependent and one
      that made the coefficients geometry-only are opposite defects with the
      same symptom here.

      Sharing the geometry alone is *not* enough, and the first build of this
      section is what said so: ``msis00_arca/theta120_az0`` and
      ``msis21_arca/theta120_az0`` run identical ``(thrad, h_obs)`` and
      disagree on the knots, because one tail calls ``geom.h`` per scalar and
      the other on the whole array. So the grouping carries
      :func:`h_form`, and the cross-form residual is *measured* into
      ``h_form_knot_delta_cm`` rather than asserted away;
    * the azimuth reaches the spline at every detector-centred site: the two
      single-azimuth cells of a zenith differ in ``s_X2rho`` coefficients.
    """
    signatures = {cell: _cell_signature(arrays, cell) for cell in cells}
    collisions = sorted(
        (a, b)
        for i, a in enumerate(cells)
        for b in cells[i + 1 :]
        if signatures[a] == signatures[b]
    )
    assert not collisions, (
        f"cells stored identical numbers: {collisions}. One of the sweep axes "
        f"(atmosphere, zenith, azimuth mode, observation level) is a no-op."
    )

    # s_h2X knots against the geometry and the geom.h call form behind them.
    by_geometry: dict = {}
    for cell in cells:
        geom_key = f"cells/{cell}/geom"
        if geom_key not in arrays or f"cells/{cell}/s_h2X/knots" not in arrays:
            continue
        scalars = arrays[f"cells/{cell}/scalars"]
        key = (
            float(scalars[1]),
            float(arrays[geom_key][0]),
            str(arrays[f"cells/{cell}/h_form"]),
        )
        by_geometry.setdefault(key, []).append(cell)

    shared = 0
    for key, group in sorted(by_geometry.items()):
        knots = {str(arrays[f"cells/{c}/s_h2X/knots"]) for c in group}
        assert len(knots) == 1, (
            f"cells at (thrad, h_obs, h_form) = {key} disagree on the s_h2X "
            f"knots: {sorted(group)}. With s=0.0 the knots are the h_intp "
            f"samples, which depend on the geometry and the geom.h call form "
            f"alone — unless a tail changed how it builds h_intp."
        )
        if len(group) > 1:
            shared += 1
            coeffs = {str(arrays[f"cells/{c}/s_h2X/coeffs"]) for c in group}
            assert len(coeffs) == len(group), (
                f"cells at (thrad, h_obs, h_form) = {key} share s_h2X "
                f"coefficients as well as knots: {sorted(group)}. The "
                f"coefficients are log(X) along the column and must separate "
                f"the density models."
            )

    # The two geom.h forms, at every geometry the sweep actually visits. Pure
    # geometry, so it is re-derived from an EarthGeometry rather than inferred
    # from the digests: the same `h_intp` the tails build, once per scalar and
    # once on the vector, compared element-wise.
    cross = _record_geom_h_forms(arrays, sorted({key[:2] for key in by_geometry}))

    # The azimuth has to reach the density.
    for label, _cls, _args, _kwargs, _avg in CENTERED:
        pairs = [
            (t, a)
            for t, a in SINGLE_AZIMUTH_CELLS
            if f"cells/{label}/theta{t:g}_az{a:g}/s_X2rho/coeffs" in arrays
        ]
        for theta in sorted({t for t, _ in pairs}):
            azimuths = [a for t, a in pairs if t == theta]
            if len(azimuths) < 2:
                continue
            digests = {
                str(arrays[f"cells/{label}/theta{theta:g}_az{a:g}/s_X2rho/coeffs"])
                for a in azimuths
            }
            assert len(digests) == len(azimuths), (
                f"{label} at theta {theta}: azimuths {azimuths} give the same "
                f"s_X2rho. _impact_point returned the same column for both, "
                f"which is what a zero detector depth does (r_det == r_obs "
                f"makes the larger root x = 0 for any downgoing direction)."
            )

    knot_keys = sorted(
        k for k in arrays if k.endswith("/knots") and arrays[k].dtype.kind in "US"
    )
    coeff_keys = sorted(k for k in arrays if k.endswith("/coeffs"))
    return {
        "n_cells": len(cells),
        "n_distinct_cells": len(set(signatures.values())),
        "n_knot_digests": len(knot_keys),
        "n_distinct_knot_digests": len({str(arrays[k]) for k in knot_keys}),
        "n_coeff_digests": len(coeff_keys),
        "n_distinct_coeff_digests": len({str(arrays[k]) for k in coeff_keys}),
        "n_geometries_shared_by_several_cells": shared,
        "n_geometries": len(cross),
        "n_geometries_where_h_forms_differ": sum(1 for row in cross if row[2]),
        "max_h_form_delta_cm": max((row[3] for row in cross), default=0.0),
    }


def _concrete_atmosphere_classes() -> list[str]:
    """Every concrete :class:`EarthsAtmosphere` subclass, from the source.

    Read off the two modules rather than listed, so a class a later phase adds
    appears in the coverage table without a second edit here — and so the
    ``environment`` count and the ``meta/uncovered`` reason stay honest.
    """
    import inspect

    from MCEq.geometry import density_profiles as dprof
    from MCEq.geometry import msis21_atmosphere as m21
    from MCEq.geometry.density_profiles import EarthsAtmosphere

    found = set()
    for module in (dprof, m21):
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, EarthsAtmosphere)
                and obj is not EarthsAtmosphere
                and not inspect.isabstract(obj)
            ):
                found.add(obj.__name__)
    return sorted(found)


def build() -> tuple[dict, dict]:
    """Produce (arrays, provenance) for this section."""
    from MCEq import config
    from MCEq.geometry import density_profiles as dprof

    try:
        import nrlmsis  # noqa: F401

        have21 = True
    except ImportError:
        have21 = False

    arrays: dict = {}
    labels: list[str] = []
    seconds: dict = {}

    saved = {key: getattr(config, key) for key in CONFIG_PINS}
    missing = sorted(key for key in CONFIG_PINS if not hasattr(config, key))
    assert not missing, f"config globals absent, pin block is stale: {missing}"
    try:
        for key, value in CONFIG_PINS.items():
            setattr(config, key, value)

        started = time.perf_counter()
        _sweep_plain(arrays, dprof, have21, labels)
        seconds["plain"] = time.perf_counter() - started

        started = time.perf_counter()
        _sweep_centered(arrays, dprof, have21, labels)
        seconds["centered"] = time.perf_counter() - started

        started = time.perf_counter()
        _sweep_h_obs(arrays, dprof, have21, labels)
        seconds["h_obs"] = time.perf_counter() - started

        started = time.perf_counter()
        _record_generalized_target(arrays, dprof, labels)
        _record_gtracr(arrays)
        seconds["extras"] = time.perf_counter() - started

        covered = sorted(
            {cls for _, cls, *_ in ATMOSPHERES if cls not in MSIS21_CLASSES or have21}
            | {cls for _, cls, *_ in CENTERED if cls not in MSIS21_CLASSES or have21}
        )
        arrays["meta/cells"] = np.array(labels, dtype="U64")
        arrays["meta/covered_classes"] = np.array(covered, dtype="U32")
        arrays["meta/concrete_classes"] = np.array(
            _concrete_atmosphere_classes(), dtype="U32"
        )
        arrays["meta/uncovered"] = np.array(UNCOVERED, dtype="U512")
        arrays["meta/n_steps"] = np.int64(N_STEPS)
        arrays["meta/have_nrlmsis"] = np.int64(have21)

        seconds["total"] = sum(seconds.values())
        summary = _check_invariants(arrays, labels)
        provenance = make_provenance(
            SECTION,
            note=NOTE,
            tolerances={},
            extra={
                "cells": labels,
                "covered_classes": covered,
                "concrete_classes": _concrete_atmosphere_classes(),
                **summary,
                "zeniths": list(ZENITHS),
                "zeniths_site2": list(ZENITHS_SITE2),
                "single_azimuth_cells": [list(c) for c in SINGLE_AZIMUTH_CELLS],
                "averaged_zeniths_msis21": list(AVERAGED_ZENITHS_MSIS21),
                "averaged_zeniths_msis00": list(AVERAGED_ZENITHS_MSIS00),
                "averaged_zeniths_msis00_one": list(AVERAGED_ZENITHS_MSIS00_ONE),
                "generic_site": GENERIC_SITE,
                "x_fractions": list(X_FRACTIONS),
                "probe_heights_cm": list(PROBE_HEIGHTS_CM),
                "probe_heights_averaged_cm": list(PROBE_HEIGHTS_AVERAGED_CM),
                "n_steps": N_STEPS,
                "have_nrlmsis": have21,
                "blas_threads": blas_threads(),
                "seconds": seconds,
            },
        )
    finally:
        for key, value in saved.items():
            setattr(config, key, value)

    return arrays, provenance
