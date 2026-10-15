"""NRLMSIS 2.1 atmospheres (Emmert et al. 2021; requires the nrlmsis package)."""

import numpy as np

from MCEq.geometry.atmosphere_parameters import (
    DAY_TIMES_SEC,
    DEFAULT_AP,
    DEFAULT_F107,
    DEFAULT_F107A,
    LOCATIONS,
    MONTH_TO_DAY_OF_YEAR,
    location_key,
)
from MCEq.geometry.density_profiles import (
    ICECUBE,
    MSISAtmosphere,
    MSISLocationCentered,
    km3net,
)
from MCEq.misc import info


class Msis21:
    """NRLMSIS 2.1 at a site and date, with the interface of
    :class:`MCEq.geometry.nrlmsise00_mceq.cNRLMSISE00`; by default the South
    Pole in June at noon."""

    def __init__(self):
        try:
            from nrlmsis import NRLMSIS21
        except ImportError as e:
            raise ImportError(
                "The MSIS21 atmospheres require the 'nrlmsis' package. Install via: "
                "pip install 'git+https://github.com/afedynitch/nrlmsis2.1'"
            ) from e
        self._model = NRLMSIS21()
        self.set_location("SouthPole")
        self._doy = MONTH_TO_DAY_OF_YEAR["June"]
        self._sec = float(DAY_TIMES_SEC["day"])
        self._f107a = float(DEFAULT_F107A)
        self._f107 = float(DEFAULT_F107)
        self._ap = np.full(7, float(DEFAULT_AP))  # the 7-element ap vector

    def set_location(self, location):
        location = location_key(location)
        if location not in LOCATIONS:
            raise ValueError(
                f"Location '{location}' not in MCEq's LOCATIONS table. "
                f"Available: {list(LOCATIONS.keys())}"
            )
        lon, lat, alt_cm = LOCATIONS[location]
        self._lon, self._lat = float(lon), float(lat)
        self.current_location = location
        self.alt_surface = alt_cm

    def set_location_coord(self, longitude, latitude):
        if abs(latitude) > 90 or abs(longitude) > 180:
            raise ValueError("longitude/latitude out of range")
        self._lon, self._lat = float(longitude), float(latitude)

    def set_season(self, month):
        if month not in MONTH_TO_DAY_OF_YEAR:
            raise ValueError(f"Unknown month '{month}'")
        self._doy = MONTH_TO_DAY_OF_YEAR[month]

    def set_doy(self, day_of_year):
        if not (1 <= day_of_year <= 366):
            raise ValueError("doy out of range")
        self._doy = int(day_of_year)

    def _calc(self, method, z_km):
        return method(
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
        """Density in g/cm^3 (nrlmsis returns kg/m^3) at ``h_cm``."""
        return (
            self._calc(self._model.calc, max(float(h_cm) / 1e5, 0.0)).densities[0]
            * 1e-3
        )

    def get_temperature(self, h_cm):
        """Temperature in K at ``h_cm``."""
        return self._calc(self._model.calc, max(float(h_cm) / 1e5, 0.0)).temperature

    def density_profile(self, h_vec):
        """Density in g/cm^3 at the heights ``h_vec`` in cm, in one call."""
        z_km = np.maximum(np.asarray(h_vec) / 1.0e5, 0.0)
        return self._calc(self._model.calc_altitude_array, z_km).densities[0] * 1.0e-3


class MSIS21Atmosphere(MSISAtmosphere):
    """NRLMSIS 2.1 atmosphere at a named site; arguments as in
    :class:`MCEq.geometry.density_profiles.MSISAtmosphere`."""

    def __init__(self, location, season=None, doy=None, use_loc_altitudes=False):
        super().__init__(Msis21(), location, season, doy, use_loc_altitudes)


class MSIS21LocationCentered(MSISLocationCentered):
    """NRLMSIS 2.1 atmosphere centred on a detector; arguments as in
    :class:`MCEq.geometry.density_profiles.MSISLocationCentered` without the backend."""

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
        super().__init__(
            Msis21(),
            detector_coord,
            depth_m,
            season,
            doy,
            n_azimuth,
            max_theta,
            surface_elevation_m,
        )


class MSIS21IceCubeCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere centred on IceCube; ``location`` is ignored."""

    def __init__(self, location, season):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
        super().__init__(season=season, **ICECUBE)


class MSIS21KM3NeTCentered(MSIS21LocationCentered):
    """NRLMSIS 2.1 atmosphere centred on the KM3NeT detector ``"ORCA"`` or ``"ARCA"``."""

    def __init__(self, detector, season=None, doy=None, n_azimuth=36):
        super().__init__(
            season=season, doy=doy, n_azimuth=n_azimuth, **km3net(detector)
        )
        self._detector_name = detector
