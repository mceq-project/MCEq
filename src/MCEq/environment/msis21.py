"""NRLMSIS 2.1 atmospheres (Emmert et al. 2021; requires the nrlmsis package)."""

from __future__ import annotations

import numpy as np

from MCEq.environment.base import EarthsAtmosphere
from MCEq.environment.column import fit_column_splines
from MCEq.environment.location_centered import LocationCenteredMixin
from MCEq.environment.parameters import (
    DAY_TIMES_SEC,
    DEFAULT_AP,
    DEFAULT_F107,
    DEFAULT_F107A,
    KM3NET_DETECTORS,
    LOCATIONS,
    MONTH_TO_DAY_OF_YEAR,
)
from MCEq.misc import info


class MSIS21Atmosphere(EarthsAtmosphere):
    """NRLMSIS 2.1 atmosphere at a named location (interface of MSIS00Atmosphere).

    Args:
        location (str): A key of
            :data:`MCEq.environment.parameters.LOCATIONS`
            (e.g. ``"SouthPole"``, ``"Karlsruhe"``).
        season (str, optional): Month name; mapped via
            :data:`MONTH_TO_DAY_OF_YEAR`.
        doy (int, optional): Day of year [1, 365].  Takes precedence over
            ``season`` if both are supplied.
        use_loc_altitudes (bool): Use the named location's surface altitude
            from :data:`LOCATIONS` as the observation level.
    """

    def __init__(
        self, location, season=None, doy=None, use_loc_altitudes=False, environment=None
    ):
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
        EarthsAtmosphere.__init__(self, environment=environment)
        self.init_parameters(location, season, doy, use_loc_altitudes)

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
        # nrlmsis 2.1 takes a 7-element ap vector; fill it with the scalar default.
        self._ap = np.full(7, float(DEFAULT_AP))

        self.location = location
        self.season = season
        self.theta_deg = None  # forces re-computation on next set_theta

    def _clear_cache(self):
        """Invalidate the cached density spline."""
        self.theta_deg = None

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
        """Update several parameters at once.

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
        """Air mass density at height *h_cm* (cm) in g/cm³ (nrlmsis returns kg/m³)."""
        return self._calc_one(h_cm).densities[0] * 1.0e-3

    def get_temperature(self, h_cm):
        """Air temperature at height *h_cm* (cm) in K."""
        return self._calc_one(h_cm).temperature

    def calculate_density_spline(self, n_steps=2000):
        """Build the rho(X) spline from one batched nrlmsis call."""
        from time import time

        if self.theta_deg is None:
            raise Exception("zenith angle not set")

        info(
            5,
            f"MSIS21: rho(X) spline for zenith {self.theta_deg:4.1f}°",
        )

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0.0, path_length, n_steps)
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

        max_den = float(rho_vec[0])

        fit_column_splines(self, rho_vec, dl_vec, h_vec_cm[2:][::-1], max_den=max_den)

    def set_theta(self, theta_deg):
        """Configure zenith angle and trigger spline build."""
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise Exception("Zenith angle not in allowed range.")
        self.thrad = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        self.calculate_density_spline()


class MSIS21LocationCentered(LocationCenteredMixin, MSIS21Atmosphere):
    """NRLMSIS 2.1 atmosphere centred on a detector at an arbitrary site.

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
            :class:`MCEq.environment.msis00.MSIS00LocationCentered`
            for the depth/elevation and local-zenith conventions.
    """

    _preserve_max_theta: bool = True

    #: Density profile depends on azimuth (impact point bound to the
    #: detector location); ``set_theta`` accepts ``azimuth_deg``.
    depends_on_azimuth: bool = True

    def _set_backend_location(self, longitude, latitude):
        """nrlmsis 2.1 takes the position per call, so just store it."""
        self._lon = longitude
        self._lat = latitude

    def __init__(
        self,
        detector_coord,
        depth_m,
        season=None,
        doy=None,
        n_azimuth=36,
        max_theta=90.0,
        surface_elevation_m=0.0,
        environment=None,
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

        # Bypass MSIS21Atmosphere.__init__, which requires a named location.
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

        EarthsAtmosphere.__init__(self, environment=environment)
        self._init_detector_geometry(
            longitude,
            latitude,
            depth_m,
            season,
            n_azimuth,
            max_theta,
            surface_elevation_m,
        )

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

    def calculate_density_spline(self, n_steps=2000):
        """Build the rho(X) spline from batched nrlmsis calls.

        In azimuth-averaging mode, one call per impact point; the densities
        are averaged before the spline fit.
        """
        from time import time

        if self.theta_deg is None:
            raise Exception("zenith angle not set")

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0.0, path_length, n_steps)
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

        max_den = float(rho_vec[0])

        fit_column_splines(self, rho_vec, dl_vec, h_vec_cm[2:][::-1], max_den=max_den)


class MSIS21IceCubeCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere centred on the IceCube detector at South Pole."""

    def __init__(self, location, season, environment=None):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
        super().__init__(
            detector_coord=(0.0, -90.0),
            depth_m=1948.0,
            season=season,
            max_theta=180.0,
            surface_elevation_m=2835.0,
            environment=environment,
        )

    def _latitude(self, det_zenith_deg):
        lat, _ = self._impact_point(det_zenith_deg, 0.0)
        return lat


class MSIS21KM3NeTCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere centred on a KM3NeT detector (ORCA or ARCA)."""

    def __init__(self, detector, season=None, doy=None, n_azimuth=36, environment=None):
        if detector not in KM3NET_DETECTORS:
            raise ValueError(
                f"Unknown KM3NeT detector '{detector}'. "
                f"Choose from {list(KM3NET_DETECTORS.keys())}."
            )
        det = KM3NET_DETECTORS[detector]
        super().__init__(
            detector_coord=(det["longitude"], det["latitude"]),
            depth_m=det["depth_m"],
            season=season,
            doy=doy,
            n_azimuth=n_azimuth,
            max_theta=180.0,
            environment=environment,
        )
        self._detector_name = detector
