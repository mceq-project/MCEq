"""Detector-centred atmosphere geometry, shared by both MSIS backends.

Where the detector sits, where a shower of a given zenith and azimuth crosses
the surface, and what local zenith the column is then integrated at -- none of
that depends on which MSIS implementation supplies the densities. This module
is the single copy shared by both MSIS families.

Keep the scalar expressions in the coordinate formulas exactly as written. The
impact-point discriminant is spelled ``A**2`` on a float64 scalar: numpy sends
``scalar**2`` through libm ``pow`` while ``A*A`` is a multiply, so the two
spellings differ by ~1 ULP at some arguments, and that difference propagates
into the spline inputs. The ``max_theta`` rejection raises ``ValueError``; only
the exception type is pinned by the tests, not the message text.

The mixin imports nothing from :mod:`MCEq.environment.base`, which keeps the
geometry modules free of import cycles: the density backend calls into the
mixin, never the reverse. It does not reach the base initialiser through
``super()`` because the next class in the MRO is the backend atmosphere, whose
``__init__`` takes ``(location, season, doy, ...)``, not detector arguments. The
concrete class runs ``EarthsAtmosphere.__init__`` itself and then calls
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
      it for the next batched call; the density evaluation itself is provided
      per-backend, as is ``calculate_density_spline``.
    * ``calculate_density_spline`` and ``get_density``. The scalar-loop and
      batched forms differ at the ~1e-14 level;
      ``tests/geometry/test_environment_pins.py`` bounds that difference at
      < 1e-11 in relative density. It is expected, not a defect.
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

        This method reads ``self.geom``, which only the base initialiser
        creates.
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
        The full (zenith, azimuth) pair at the detector is used directly —
        no mirroring of either angle is needed.

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
            info(
                1,
                f"zenith={theta_deg:.1f}°, azimuth={azimuth_deg:.1f}°"
                f" → impact lat={lat:.2f}°, lon={lon:.2f}°,"
                f" local zenith={effective_theta:.2f}°",
            )
        else:
            # Pre-compute all impact points once; they depend only on zenith,
            # not on height, so they are constant for this set_theta call.
            # Use the detector zenith directly to compute the source-side
            # impact points (transparent-Earth projection).
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
