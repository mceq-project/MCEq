"""NRLMSIS 2.1 atmosphere models for MCEq.

Drop-in replacements for the MSIS00 atmosphere hierarchy backed by the
NRLMSIS 2.1 empirical model (Emmert et al. 2021).  The 2.1 model is a
whole-atmosphere refit with newer satellite data; it differs from
NRLMSISE-00 by a few percent in the 50–100 km range.

Key implementation note: ``calculate_density_spline`` is overridden to
issue a single batched call to :meth:`nrlmsis.NRLMSIS21.calc_altitude_array`
instead of looping ``get_density`` per altitude.  On the all-sky pipeline
this reduces the per-pixel MSIS share to ~5 ms vs ~25 ms for the C
MSISE-00 ctypes wrapper, in pure numpy.

The class hierarchy mirrors :mod:`MCEq.geometry.density_profiles`:
- :class:`MSIS21Atmosphere`        — named-location atmosphere
- :class:`MSIS21LocationCentered`  — arbitrary (lon, lat) + detector depth
- :class:`MSIS21IceCubeCentered`   — South Pole, depth 1948 m
- :class:`MSIS21KM3NeTCentered`    — ORCA / ARCA

Requires ``nrlmsis`` (https://github.com/afedynitch/nrlmsis2.1) installed.
The vectorised below-ZETA_B path is on master as of 2026-05-24.
"""

from __future__ import annotations

import numpy as np

from MCEq.geometry.atmosphere_parameters import (
    DAY_TIMES_SEC,
    DEFAULT_AP,
    DEFAULT_F107,
    DEFAULT_F107A,
    LOCATIONS,
    MONTH_TO_DAY_OF_YEAR,
)
from MCEq.geometry.density_profiles import _KM3NET_DETECTORS, EarthsAtmosphere
from MCEq.misc import info

# Cached top-of-atmosphere constant in km (h_atm in geometry.py is 112.8 km).
# NRLMSIS 2.1 is valid 0–2000 km; we never query above ~113 km.

# Site coordinates are shared with the MSIS00 hierarchy via
# ``_KM3NET_DETECTORS`` above — do not duplicate them here, the two
# families must always describe the same detectors.


class MSIS21Atmosphere(EarthsAtmosphere):
    """NRLMSIS 2.1 atmosphere at a named location.

    Drop-in replacement for :class:`MCEq.geometry.density_profiles.MSIS00Atmosphere`.
    The public surface (``get_density``, ``get_temperature``, ``set_location``,
    ``set_location_coord``, ``set_season``, ``set_doy``) is identical so MCEq
    code can swap MSIS00 → MSIS21 without touching call sites.

    Args:
        location (str): A key of
            :data:`MCEq.geometry.atmosphere_parameters.LOCATIONS`
            (e.g. ``"SouthPole"``, ``"Karlsruhe"``).
        season (str, optional): Month name; mapped via
            :data:`MONTH_TO_DAY_OF_YEAR`.
        doy (int, optional): Day of year [1, 365].  Takes precedence over
            ``season`` if both are supplied.
        use_loc_altitudes (bool): Use the named location's surface altitude
            from :data:`LOCATIONS` as the observation level.  Mirrors the
            argument of :class:`MSIS00Atmosphere` -- the two families are
            separate class trees but expose the same interface.
    """

    def __init__(self, location, season=None, doy=None, use_loc_altitudes=False):
        try:
            from nrlmsis import NRLMSIS21
        except ImportError as e:
            raise ImportError(
                "MSIS21Atmosphere requires the 'nrlmsis' package. "
                "Install via: pip install "
                "'git+https://github.com/afedynitch/nrlmsis2.1'"
            ) from e

        self._model = NRLMSIS21()
        # Base class first: it creates self.geom, which init_parameters needs
        # when ``use_loc_altitudes`` moves the observation level.
        EarthsAtmosphere.__init__(self)
        self.init_parameters(location, season, doy, use_loc_altitudes)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def init_parameters(self, location, season, doy, use_loc_altitudes=False):
        """Set location, day-of-year, and default geophysical parameters."""
        if location not in LOCATIONS:
            raise ValueError(
                f"Location '{location}' not in MCEq's LOCATIONS table. "
                f"Available: {list(LOCATIONS.keys())}"
            )
        lon, lat, alt_cm = LOCATIONS[location]
        self._lon = float(lon)
        self._lat = float(lat)

        if use_loc_altitudes:
            info(0, "Using loc altitude", alt_cm, "cm")
            self.geom.set_h_obs(alt_cm)

        if doy is not None:
            self._doy = int(doy)
        elif season is not None:
            if season not in MONTH_TO_DAY_OF_YEAR:
                raise ValueError(f"Unknown month '{season}'")
            self._doy = MONTH_TO_DAY_OF_YEAR[season]
        else:
            self._doy = MONTH_TO_DAY_OF_YEAR["June"]

        self._sec = float(DAY_TIMES_SEC["day"])
        self._f107a = float(DEFAULT_F107A)
        self._f107 = float(DEFAULT_F107)
        # nrlmsis 2.1 wants a 7-element ap vector; replicate the MSISE-00
        # scalar default the same way MCEq's MSIS00 wrapper does.
        self._ap = np.full(7, float(DEFAULT_AP))

        self.location = location
        self.season = season
        self.theta_deg = None  # forces re-computation on next set_theta

    def _clear_cache(self):
        """Invalidate the cached density spline."""
        self.theta_deg = None

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_location(self, location):
        """Re-target by named location (entry in LOCATIONS)."""
        if location not in LOCATIONS:
            raise ValueError(f"Unknown location '{location}'")
        lon, lat, _ = LOCATIONS[location]
        self._lon = float(lon)
        self._lat = float(lat)
        self.location = location
        self._clear_cache()

    def set_location_coord(self, longitude, latitude):
        """Re-target by explicit (longitude, latitude) in degrees."""
        if abs(latitude) > 90 or abs(longitude) > 180:
            raise ValueError("longitude/latitude out of range")
        self._lon = float(longitude)
        self._lat = float(latitude)
        self._clear_cache()

    def set_season(self, month):
        """Set day-of-year via month name."""
        if month not in MONTH_TO_DAY_OF_YEAR:
            raise ValueError(f"Unknown month '{month}'")
        self._doy = MONTH_TO_DAY_OF_YEAR[month]
        self.season = month
        self._clear_cache()

    def set_doy(self, day_of_year):
        """Set day-of-year directly (1..365)."""
        if not (1 <= day_of_year <= 366):
            raise ValueError("doy out of range")
        self._doy = int(day_of_year)
        self._clear_cache()

    def update_parameters(self, **kwargs):
        """Bulk-update parameters (matches MSIS00Atmosphere API).

        Recognised kwargs: ``location_coord=(lon, lat)``, ``season``, ``doy``.
        ``doy`` takes precedence over ``season`` if both are set.
        """
        self._clear_cache()
        if not kwargs:
            return
        if "location_coord" in kwargs:
            self.set_location_coord(*kwargs["location_coord"])
        if "season" in kwargs:
            self.set_season(kwargs["season"])
        if "doy" in kwargs:
            self.set_doy(kwargs["doy"])
            if "season" in kwargs:
                info(
                    2,
                    "Both 'season' and 'doy' supplied; 'doy' takes precedence.",
                )

    # ------------------------------------------------------------------
    # Scalar density / temperature lookups (mirrors MSIS00 API)
    # ------------------------------------------------------------------

    def _calc_one(self, h_cm):
        z_km = max(float(h_cm) / 1e5, 0.0)
        return self._model.calc(
            day=self._doy,
            utsec=self._sec,
            z=z_km,
            lat=self._lat,
            lon=self._lon,
            sfluxavg=self._f107a,
            sflux=self._f107,
            ap=self._ap,
        )

    def get_density(self, h_cm):
        """Air mass density at height *h_cm* (cm) in g/cm³.

        Note: nrlmsis 2.1 returns kg/m³ internally; we convert to g/cm³
        (factor 1e-3) to match MCEq's existing MSIS00 convention.
        """
        return self._calc_one(h_cm).densities[0] * 1.0e-3

    def get_temperature(self, h_cm):
        """Air temperature at height *h_cm* (cm) in K."""
        return self._calc_one(h_cm).temperature

    # ------------------------------------------------------------------
    # Vectorised spline build — the speed win
    # ------------------------------------------------------------------

    def calculate_density_spline(self, n_steps=2000):
        """Build the rho(X) spline using one batched nrlmsis call.

        Overrides the base-class height-major Python loop.  Replaces
        N_steps scalar ``get_density`` calls (~25 µs each in MSISE-00 C,
        ~80 µs each in scalar nrlmsis2.1 Python) with a single vectorised
        ``calc_altitude_array`` (~4 ms total for n_steps=2000).
        """
        from time import time

        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import UnivariateSpline

        if self.theta_deg is None:
            raise Exception("zenith angle not set")

        info(
            5,
            f"MSIS21: rho(X) spline for zenith {self.theta_deg:4.1f}°",
        )

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0.0, path_length, n_steps)
        # geom.h is pure-numpy and accepts an array dl — call once.
        h_vec_cm = self.geom.h(dl_vec, thrad)
        # nrlmsis is valid 0..2000 km; clamp negatives (can occur for
        # detector-depth geometries on the way to obs level)
        z_km_vec = np.maximum(h_vec_cm / 1.0e5, 0.0)

        now = time()
        result = self._model.calc_altitude_array(
            day=self._doy,
            utsec=self._sec,
            z=z_km_vec,
            lat=self._lat,
            lon=self._lon,
            sfluxavg=self._f107a,
            sflux=self._f107,
            ap=self._ap,
        )
        rho_vec = result.densities[0] * 1.0e-3  # kg/m³ → g/cm³

        info(5, f".. nrlmsis2.1 vectorised call took {time() - now:1.3f}s")

        X_int = cumulative_trapezoid(rho_vec, dl_vec)  # (n_steps-1,)

        self._max_X = X_int[-1]
        self._min_X = X_int[0]
        self._max_den = float(rho_vec[0])

        # Base-class spline fit: h_intp = reversed(geom.h(dl_vec[2:], thrad)),
        # X_intp = reversed(X_int[1:]). h_vec_cm[i] = geom.h(dl_vec[i], thrad),
        # so dl_vec[2:] → h_vec_cm[2:].
        h_intp = h_vec_cm[2:][::-1]
        X_intp = X_int[1:][::-1]
        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1], k=2, s=0.0)

    # ------------------------------------------------------------------
    # set_theta (base-class behavior; no azimuth concept here)
    # ------------------------------------------------------------------

    def set_theta(self, theta_deg):
        """Configure zenith angle and trigger spline build."""
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise Exception("Zenith angle not in allowed range.")
        self.thrad = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        self.calculate_density_spline()


class MSIS21LocationCentered(MSIS21Atmosphere):
    """NRLMSIS 2.1 atmosphere coupled to an arbitrary detector location.

    Drop-in replacement for
    :class:`MCEq.geometry.density_profiles.MSIS00LocationCentered`.
    Geometry (impact-point projection, azimuth-averaging) is identical;
    only the MSIS backend differs.

    Args:
        detector_coord (tuple): ``(longitude, latitude)`` of the detector
            in degrees.
        depth_m (float): Detector depth below the local surface in metres.
        season (str, optional): Month name.
        doy (int, optional): Day of year (1–365).
        n_azimuth (int): Azimuth samples for averaging mode (default 36).
        max_theta (float): Max zenith in degrees (90 = downgoing only,
            180 = also grazing and upgoing).
        surface_elevation_m (float): Elevation of the local surface above
            sea level in metres (default 0).  See
            :class:`MCEq.geometry.density_profiles.MSIS00LocationCentered`
            for the depth/elevation and local-zenith conventions.
    """

    _preserve_max_theta: bool = True

    #: Density profile depends on azimuth (impact point bound to the
    #: detector location); ``set_theta`` accepts ``azimuth_deg``.
    depends_on_azimuth: bool = True

    def __init__(
        self,
        detector_coord,
        depth_m,
        season=None,
        doy=None,
        n_azimuth=36,
        max_theta=90.0,
        surface_elevation_m=0.0,
    ):
        try:
            from nrlmsis import NRLMSIS21
        except ImportError as e:
            raise ImportError(
                "MSIS21LocationCentered requires the 'nrlmsis' package. "
                "Install via: pip install "
                "'git+https://github.com/afedynitch/nrlmsis2.1'"
            ) from e

        longitude, latitude = detector_coord

        # Backend model + parameters (bypass MSIS21Atmosphere.__init__ to skip
        # the LOCATIONS lookup which only knows named sites).
        self._model = NRLMSIS21()
        self._lon = float(longitude)
        self._lat = float(latitude)

        if doy is not None:
            self._doy = int(doy)
        elif season is not None:
            self._doy = MONTH_TO_DAY_OF_YEAR[season]
        else:
            self._doy = MONTH_TO_DAY_OF_YEAR["June"]
        self._sec = float(DAY_TIMES_SEC["day"])
        self._f107a = float(DEFAULT_F107A)
        self._f107 = float(DEFAULT_F107)
        self._ap = np.full(7, float(DEFAULT_AP))

        # Detector geometry
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
        self.theta_deg = None

        EarthsAtmosphere.__init__(self)
        if surface_elevation_m != 0.0:
            self.geom.set_h_obs(surface_elevation_m * 1e2)
        self.max_theta = max_theta
        self.location = f"({longitude:.3f}°E, {latitude:.3f}°N)"
        self.season = season

    # ------------------------------------------------------------------
    # Geometry — identical to MSIS00LocationCentered._impact_point
    # ------------------------------------------------------------------

    def _impact_point(self, zenith_deg, azimuth_deg):
        """Project the detector + shower direction to the surface impact point.

        Uses 3-D ECEF geometry, transparent-Earth convention for upgoing
        zeniths (passes the antipodal-hemisphere crossing for theta > 90°).
        Azimuth: 0° = North, 90° = East.  The surface sphere is taken at
        the current observation level (``geom.r_obs``), so the detector may
        sit below or above it.
        """
        r = self.geom.r_obs / 1e2  # cm → m
        r_det = self.geom.r_E / 1e2 + self._surface_elevation_m - self._detector_depth_m

        theta = np.deg2rad(zenith_deg)
        alpha = np.deg2rad(azimuth_deg)
        lat0 = np.deg2rad(self._detector_latitude)
        lon0 = np.deg2rad(self._detector_longitude)

        P_det = np.array(
            [
                r_det * np.cos(lat0) * np.cos(lon0),
                r_det * np.cos(lat0) * np.sin(lon0),
                r_det * np.sin(lat0),
            ]
        )
        d_ENU = np.array(
            [
                np.sin(theta) * np.sin(alpha),  # East
                np.sin(theta) * np.cos(alpha),  # North
                np.cos(theta),  # Up
            ]
        )
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

        # Larger root = surface crossing on the source side; valid for
        # detectors below (r_det < r) and above (r_det > r) the sphere and
        # for the full zenith range 0°–180°.
        A = np.dot(d_ECEF, P_det)
        x = -A + np.sqrt(A * A + (r**2 - r_det**2))
        P_imp = P_det + x * d_ECEF
        lat_imp = np.rad2deg(np.arcsin(np.clip(P_imp[2] / r, -1.0, 1.0)))
        lon_imp = np.rad2deg(np.arctan2(P_imp[1], P_imp[0]))
        return lat_imp, lon_imp

    # ------------------------------------------------------------------
    # Density — single-azimuth or azimuth-averaged
    # ------------------------------------------------------------------

    def get_density(self, h_cm):
        """Single-altitude density in g/cm³.

        In azimuth-averaging mode iterates over the pre-cached impact
        points; in single-azimuth mode goes straight to the backend.
        """
        if self._azimuth_averaging:
            total = 0.0
            for lat, lon in self._azimuth_avg_coords:
                self._lat = lat
                self._lon = lon
                total += self._calc_one(h_cm).densities[0] * 1.0e-3
            return total / len(self._azimuth_avg_coords)
        return self._calc_one(h_cm).densities[0] * 1.0e-3

    # ------------------------------------------------------------------
    # Vectorised spline build (single or azimuth-averaged)
    # ------------------------------------------------------------------

    def calculate_density_spline(self, n_steps=2000):
        """Batched rho(X) spline using calc_altitude_array.

        Single-azimuth: one vectorised call.
        Azimuth-averaging: n_azimuth vectorised calls (one per direction),
        averaged before the spline fit — far better than the height-major
        loop in the legacy MSIS00 path.
        """
        from time import time

        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import UnivariateSpline

        if self.theta_deg is None:
            raise Exception("zenith angle not set")

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0.0, path_length, n_steps)
        # geom.h is pure-numpy, broadcast-friendly — single call replaces
        # the n_steps Python loop.
        h_vec_cm = self.geom.h(dl_vec, thrad)
        z_km_vec = np.maximum(h_vec_cm / 1.0e5, 0.0)

        now = time()
        if self._azimuth_averaging:
            info(
                5,
                f"MSIS21: azimuth-averaged spline for zenith "
                f"{self.theta_deg:4.1f}° "
                f"({len(self._azimuth_avg_coords)} directions)",
            )
            rho_sum = np.zeros(n_steps)
            for lat, lon in self._azimuth_avg_coords:
                res = self._model.calc_altitude_array(
                    day=self._doy,
                    utsec=self._sec,
                    z=z_km_vec,
                    lat=float(lat),
                    lon=float(lon),
                    sfluxavg=self._f107a,
                    sflux=self._f107,
                    ap=self._ap,
                )
                rho_sum += res.densities[0]
            rho_vec = rho_sum / len(self._azimuth_avg_coords) * 1.0e-3
        else:
            info(
                5,
                f"MSIS21: rho(X) spline for zenith "
                f"{self.theta_deg:4.1f}° at "
                f"(lat={self._lat:.2f}, lon={self._lon:.2f})",
            )
            res = self._model.calc_altitude_array(
                day=self._doy,
                utsec=self._sec,
                z=z_km_vec,
                lat=self._lat,
                lon=self._lon,
                sfluxavg=self._f107a,
                sflux=self._f107,
                ap=self._ap,
            )
            rho_vec = res.densities[0] * 1.0e-3

        info(5, f".. spline build took {time() - now:1.3f}s")

        X_int = cumulative_trapezoid(rho_vec, dl_vec)  # (n_steps-1,)

        self._max_X = X_int[-1]
        self._min_X = X_int[0]
        self._max_den = float(rho_vec[0])

        # Same indexing as the MSIS00 base-class spline contract:
        # h_intp = reversed(geom.h(dl_vec[2:], thrad))
        # X_intp = reversed(X_int[1:])
        h_intp = h_vec_cm[2:][::-1]
        X_intp = X_int[1:][::-1]
        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1], k=2, s=0.0)

    # ------------------------------------------------------------------
    # set_theta — location-centered version with azimuth support
    # ------------------------------------------------------------------

    def set_theta(self, theta_deg, azimuth_deg=None):
        """Configure zenith and optional azimuth; rebuild density spline.

        Mirrors :meth:`MSIS00LocationCentered.set_theta` exactly — see that
        method for the detector-frame zenith, local-zenith correction,
        grazing window and far-side conventions.
        """
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise ValueError(f"Zenith angle {theta_deg} not in [0, {self.max_theta}].")

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
            self._lat = lat
            self._lon = lon
            self._current_impact_latitude = lat
            self._current_impact_longitude = lon
            self._azimuth_averaging = False
            self._azimuth_avg_coords = []
            info(
                1,
                f"MSIS21: zenith={theta_deg:.1f}°, "
                f"azimuth={azimuth_deg:.1f}° → "
                f"impact lat={lat:.2f}°, lon={lon:.2f}°",
            )
        else:
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
        self.theta_deg = theta_deg
        self.calculate_density_spline()

    # ------------------------------------------------------------------
    # Impact-point properties
    # ------------------------------------------------------------------

    @property
    def current_impact_latitude(self):
        return self._current_impact_latitude

    @property
    def current_impact_longitude(self):
        return self._current_impact_longitude


class MSIS21IceCubeCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere centred on the IceCube detector at South Pole.

    Drop-in replacement for
    :class:`MCEq.geometry.density_profiles.MSIS00IceCubeCentered`.
    """

    def __init__(self, location, season):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
        super().__init__(
            detector_coord=(0.0, -90.0),
            depth_m=1948.0,
            season=season,
            max_theta=180.0,
            surface_elevation_m=2835.0,
        )

    def _latitude(self, det_zenith_deg):
        lat, _ = self._impact_point(det_zenith_deg, 0.0)
        return lat


class MSIS21KM3NeTCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere coupled to a KM3NeT detector.

    Drop-in replacement for
    :class:`MCEq.geometry.density_profiles.MSIS00KM3NeTCentered`.
    """

    def __init__(self, detector, season=None, doy=None, n_azimuth=36):
        if detector not in _KM3NET_DETECTORS:
            raise ValueError(
                f"Unknown KM3NeT detector '{detector}'. "
                f"Choose from {list(_KM3NET_DETECTORS.keys())}."
            )
        det = _KM3NET_DETECTORS[detector]
        super().__init__(
            detector_coord=(det["longitude"], det["latitude"]),
            depth_m=det["depth_m"],
            season=season,
            doy=doy,
            n_azimuth=n_azimuth,
            max_theta=180.0,
        )
        self._detector_name = detector
