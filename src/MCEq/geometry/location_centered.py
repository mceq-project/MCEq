"""Detector-centred atmosphere geometry, shared by both MSIS backends.

Where the detector sits, where a shower of a given zenith and azimuth crosses
the surface, and what local zenith the column is then integrated at -- none of
that depends on which MSIS implementation supplies the densities. Both MSIS
families nevertheless carried their own copy of all of it. This is the single
copy.

The copies agreed to within floating-point reassociation, not exactly: MSIS00
spelled the impact-point discriminant ``A**2`` and MSIS21 ``A*A``. ``A`` is a
float64 *scalar*, and numpy sends ``scalar**2`` through libm ``pow`` while
``A*A`` is a multiply, so the two differ by 1 ULP at some arguments -- the same
hazard already recorded at ``tests/golden/gen_environment.py``. This module
keeps the MSIS00 spelling, so ``MSIS21LocationCentered``'s impact points moved.

How far is sample-dependent and small: over 200k random (zenith, azimuth) pairs
per shipped site on this host (x86-64, glibc, numpy 2.0.2), 0.01--0.03 % of
directions differ at all, and the largest latitude shift was 2.6e-12 degrees at
IceCube -- amplified there because ``arcsin(z/r)`` is ill-conditioned at
latitude -89.98. Longitude stays at ~1e-14 degrees. That is far below the 1e-11
D18 bound in relative density, and a direct diff of ``environment.npz`` across
the collapse shows no MSIS21 array moved: the golden's sampled angles do not
hit it. Treat the numbers as the order of magnitude, not a bound -- a different
sweep finds a different worst case.

The ``max_theta`` rejection agrees in type (``ValueError``). Its *message* did
not: MSIS00 said "not in allowed range [0, X]." and MSIS21 "not in [0, X]."
Both now use the MSIS00 wording, as does the ``info()`` line. Nothing in the
suite matches either text -- only the exception type is pinned.

Otherwise, measured before the collapse, MSIS00 against MSIS21 at the same
site: the impact coordinates agree to 0.000e+00 degrees over 40 (zenith,
azimuth) pairs including 90, 91 and 179 degrees; the full 36-direction averaged
tables agree bitwise at four zeniths; and ``theta_deg``, ``thrad``, the azimuth
mode and the impact latitude/longitude all agree.

The mixin imports nothing from :mod:`MCEq.geometry.density_profiles`, which is
what keeps the geometry siblings acyclic (contract C7): the density backend
calls into the mixin, never the reverse. It does not reach the base initialiser
through ``super()`` for an unrelated reason -- the next class in the MRO is the
backend atmosphere, whose ``__init__`` takes ``(location, season, doy, ...)``,
not detector arguments. The concrete class runs ``EarthsAtmosphere.__init__``
itself and then calls
:meth:`LocationCenteredMixin._init_detector_geometry`.
"""

from __future__ import annotations

import numpy as np

from MCEq.misc import info


class LocationCenteredMixin:
    """Geometry for an atmosphere centred on a detector at an arbitrary site.

    A concrete class must supply:

    * ``_set_backend_location(longitude, latitude)`` -- point the density
      backend at a coordinate. MSIS00 hands it to the C library, MSIS21 stores
      it for the next batched call, which is the one thing in
      :meth:`set_theta` that was ever genuinely per-backend.
    * ``calculate_density_spline`` and ``get_density``, which are per-backend by
      design -- the scalar-loop and batched forms differ at the 1e-14 level
      (7.2e-15 to 1.4e-14 measured over theta = 0/30/60/85/90 on this host) and
      that divergence is pinned by ``tests/geometry/test_environment_pins.py``
      at < 1e-11, not a defect.
    * ``self.geom``, i.e. ``EarthsAtmosphere.__init__`` must have run.
    """

    def _set_backend_location(self, longitude, latitude):
        raise NotImplementedError("concrete class must point its backend at (lon, lat)")

    def _init_detector_geometry(
        self,
        longitude,
        latitude,
        depth_m,
        season,
        n_azimuth,
        max_theta,
        surface_elevation_m,
    ):
        """Store the detector geometry. Call *after* ``EarthsAtmosphere.__init__``.

        That order is the one change the collapse makes to either family: both
        used to set these attributes first and initialise the base class
        afterwards. Nothing in the base initialiser reads them and nothing here
        reads what it sets except ``self.geom``, so only ``__dict__`` insertion
        order moves -- and the three values the base sets that matter here
        (``max_theta``, ``location``, ``season``) are overwritten below exactly
        as they were before.
        """
        self._detector_longitude = longitude
        self._detector_latitude = latitude
        self._detector_depth_m = depth_m
        self._surface_elevation_m = surface_elevation_m
        self._n_azimuth = n_azimuth
        self._azimuth_averaging = False
        self._effective_theta_deg = 0.0
        self._current_azimuth_deg = None
        self._azimuth_avg_coords = []
        self._current_impact_latitude = None
        self._current_impact_longitude = None

        if surface_elevation_m != 0.0:
            self.geom.set_h_obs(surface_elevation_m * 1e2)
        self.max_theta = max_theta
        self.location = f"({longitude:.3f}°E, {latitude:.3f}°N)"
        self.season = season

    def _impact_point(self, zenith_deg, azimuth_deg):
        """Return the geographic coordinates of the shower impact point.

        Finds the crossing of the shower axis (a straight line through the
        detector, followed toward the incoming shower source) with the
        surface sphere at the current observation level (``geom.r_obs``)
        using full 3-D ECEF (Earth-Centred, Earth-Fixed) Cartesian
        geometry.  The surface crossing on the source side is where the
        atmospheric shower column stands.

        Azimuth convention: 0° = North, 90° = East (clockwise from North).
        The full zenith range [0°, 180°] is supported: for downgoing
        showers the crossing is near the detector; for upgoing showers
        (zenith > 90°) the toward-source ray traverses the Earth and the
        crossing is on the far side, where the shower actually developed.
        The original (theta, azimuth) at the detector is passed in — no
        mirroring of either angle is needed.

        At the South Pole and zenith ≤ 90° this reduces to the 2-D formula the
        IceCube-centred model used before the ECEF rewrite (see the git history
        of :class:`MSIS00IceCubeCentered`, which is a live class registered as
        ``"MSIS00_IC"`` and holds no formula of its own today). That
        equivalence is why the ``msis00_icecube`` golden cell did not move when
        the 2-D form was replaced.

        Args:
            zenith_deg (float): Zenith angle at the detector in degrees
                (0°–180°).
            azimuth_deg (float): Azimuth angle in degrees (0° = North,
                90° = East).

        Returns:
            tuple: ``(latitude_deg, longitude_deg)`` of the impact point.
        """
        r = self.geom.r_obs / 1e2  # cm → m
        r_det = self.geom.r_E / 1e2 + self._surface_elevation_m - self._detector_depth_m

        theta = np.deg2rad(zenith_deg)
        alpha = np.deg2rad(azimuth_deg)
        lat0 = np.deg2rad(self._detector_latitude)
        lon0 = np.deg2rad(self._detector_longitude)

        # Detector position in ECEF (metres)
        P_det = np.array(
            [
                r_det * np.cos(lat0) * np.cos(lon0),
                r_det * np.cos(lat0) * np.sin(lon0),
                r_det * np.sin(lat0),
            ]
        )

        # Shower direction in ENU frame (pointing toward incoming source)
        # azimuth 0° → North, 90° → East
        d_ENU = np.array(
            [
                np.sin(theta) * np.sin(alpha),  # East component
                np.sin(theta) * np.cos(alpha),  # North component
                np.cos(theta),  # Up component
            ]
        )

        # ENU → ECEF rotation matrix (columns: East, North, Up basis vectors)
        T = np.array(
            [
                [
                    -np.sin(lon0),
                    -np.sin(lat0) * np.cos(lon0),
                    np.cos(lat0) * np.cos(lon0),
                ],
                [
                    np.cos(lon0),
                    -np.sin(lat0) * np.sin(lon0),
                    np.cos(lat0) * np.sin(lon0),
                ],
                [0.0, np.cos(lat0), np.sin(lat0)],
            ]
        )
        d_ECEF = T @ d_ENU

        # Crossing with the surface sphere |P_det + x * d_ECEF|² = r².
        # The larger root is the surface crossing on the source side; it is
        # valid for detectors below (r_det < r) and above (r_det > r) the
        # sphere, and for the full zenith range 0°–180°.
        A = np.dot(d_ECEF, P_det)
        x = -A + np.sqrt(A**2 + (r**2 - r_det**2))

        P_impact = P_det + x * d_ECEF
        lat_imp = np.rad2deg(np.arcsin(np.clip(P_impact[2] / r, -1.0, 1.0)))
        lon_imp = np.rad2deg(np.arctan2(P_impact[1], P_impact[0]))
        return lat_imp, lon_imp

    def set_theta(self, theta_deg, azimuth_deg=None):
        """Configure the zenith (and optionally azimuth) angle.

        *theta_deg* is the zenith angle **at the detector**.  The
        atmospheric column is integrated at the local zenith angle where
        the shower axis crosses the surface, obtained from impact-parameter
        conservation along the straight axis,
        :math:`\\sin\\theta_s = (r_{det}/r_{surf})\\sin\\theta`.  This one
        formula covers all regimes: downgoing (tiny correction below ~80°,
        ~1.4° at the horizon for a detector 2 km below the surface), the
        grazing window just below the geometric horizon (90° < theta ≤
        90° + dip, which stays a *near-side* downgoing column), and upgoing
        showers (whose column stands at the far-side surface crossing,
        found by following the axis toward the source through the Earth).
        For upgoing angles the observation level is set to sea level, since
        the far-side surface is generically the ocean, and restored to
        *surface_elevation_m* for downgoing/grazing angles.

        Args:
            theta_deg (float): Zenith angle at the detector in degrees
                [0, max_theta].
            azimuth_deg (float, optional): Azimuth angle in degrees
                (0° = North, 90° = East).  When ``None`` (default) the
                density profile is averaged over all azimuth directions.
        """
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise ValueError(
                f"Zenith angle {theta_deg} not in allowed range [0, {self.max_theta}]."
            )

        # Below-horizon dip of the local surface seen from the detector:
        # rays with theta <= 90 + dip still exit through the near-side
        # surface (grazing); larger angles traverse the Earth (upgoing).
        r_E = self.geom.r_E  # cm
        elev_cm = self._surface_elevation_m * 1e2
        r_det = r_E + elev_cm - self._detector_depth_m * 1e2
        dip_deg = np.rad2deg(np.arccos(min(r_det / (r_E + elev_cm), 1.0)))
        far_side = theta_deg > 90.0 + dip_deg

        # Near side: column ends at the local surface elevation.
        # Far side: column ends at sea level (generic far-side surface).
        h_obs_cm = 0.0 if far_side else elev_cm
        if self.geom.h_obs != h_obs_cm:
            self.geom.set_h_obs(h_obs_cm)

        # Local zenith angle of the shower axis at the surface crossing
        # (the impact parameter r*sin(theta) is conserved along the axis).
        sin_loc = np.clip(
            r_det / self.geom.r_obs * np.sin(np.deg2rad(theta_deg)), 0.0, 1.0
        )
        effective_theta = np.rad2deg(np.arcsin(sin_loc))

        if azimuth_deg is not None:
            lat, lon = self._impact_point(theta_deg, azimuth_deg)
            self._set_backend_location(lon, lat)
            self._current_impact_latitude = lat
            self._current_impact_longitude = lon
            self._azimuth_averaging = False
            self._azimuth_avg_coords = []
            # ``info`` prefixes the *runtime* class name, so the line still
            # names MSIS00LocationCentered or MSIS21LocationCentered without
            # the mixin having to carry a backend tag. MSIS21's old message
            # said "MSIS21:" redundantly and omitted the local zenith; this is
            # the MSIS00 wording, which both now share.
            info(
                1,
                f"zenith={theta_deg:.1f}°, azimuth={azimuth_deg:.1f}°"
                f" → impact lat={lat:.2f}°, lon={lon:.2f}°,"
                f" local zenith={effective_theta:.2f}°",
            )
        else:
            # Pre-compute all impact points once; they depend only on zenith,
            # not on height, so they are constant for this set_theta call.
            # Use the original theta_deg (transparent-Earth projection — see
            # the comment above for the single-azimuth branch).
            azi_grid = np.linspace(0.0, 360.0, self._n_azimuth, endpoint=False)
            self._azimuth_avg_coords = [
                self._impact_point(theta_deg, azi) for azi in azi_grid
            ]
            self._azimuth_averaging = True
            self._current_impact_latitude = None
            self._current_impact_longitude = None

        self._effective_theta_deg = effective_theta
        self._current_azimuth_deg = azimuth_deg
        self.thrad = np.deg2rad(effective_theta)
        self.theta_deg = theta_deg  # keep original; may be > 90 for upgoing
        self.calculate_density_spline()

    @property
    def current_impact_latitude(self):
        """Latitude of the shower impact point for the current angle (degrees).

        ``None`` when azimuth-averaging mode is active (no single impact
        point is defined in that case) or before :meth:`set_theta` has
        been called.
        """
        return self._current_impact_latitude

    @property
    def current_impact_longitude(self):
        """Longitude of the shower impact point for the current angle (degrees).

        ``None`` when azimuth-averaging mode is active or before
        :meth:`set_theta` has been called.
        """
        return self._current_impact_longitude
