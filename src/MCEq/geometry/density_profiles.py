"""Atmosphere models: CORSIKA, isothermal, NRLMSISE-00 and AIRS, and the material target."""

from abc import ABCMeta, abstractmethod
from os.path import join
from time import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline

from MCEq import config
from MCEq.geometry.atmosphere_parameters import (
    KM3NET_DETECTORS,
    MONTH_TO_DAY_OF_YEAR,
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)
from MCEq.geometry.geometry import EarthGeometry
from MCEq.geometry.target import GeneralizedTarget
from MCEq.misc import info

# The imported helpers stay in the star export of v1.4, which notebooks rely on.
_V14_STAR_EXPORTS = [
    "ABCMeta",
    "MONTH_TO_DAY_OF_YEAR",
    "abstractmethod",
    "get_atmosphere_parameters",
    "info",
    "join",
    "list_available_corsika_atmospheres",
    "np",
]

__all__ = _V14_STAR_EXPORTS + [
    "AIRSAtmosphere",
    "CorsikaAtmosphere",
    "EarthsAtmosphere",
    "GeneralizedTarget",
    "IsothermalAtmosphere",
    "MSIS00Atmosphere",
    "MSIS00IceCubeCentered",
    "MSIS00KM3NeTCentered",
    "MSIS00LocationCentered",
    "MSISAtmosphere",
    "MSISLocationCentered",
]


class EarthsAtmosphere(metaclass=ABCMeta):
    """Base class of the atmospheres: the slant depth along the shower axis and
    the splines of the density and height versus depth. A subclass implements
    :meth:`get_density`.

    Attributes:
      thrad (float): current zenith angle :math:`\\theta` in radians
      theta_deg (float): current zenith angle :math:`\\theta` in degrees
      geom (EarthGeometry): the geometry; ``geometry=`` replaces the default
    """

    #: Detector-centred models that allow upgoing angles keep their
    #: :attr:`max_theta` when the observation level changes.
    _preserve_max_theta: bool = False

    #: The density depends on the azimuth and :meth:`set_theta` accepts
    #: ``azimuth_deg``; batched solves share paths across azimuths otherwise.
    depends_on_azimuth: bool = False

    def __init__(self, *args, **kwargs):
        geometry = kwargs.pop("geometry", None)
        if geometry is None:
            geometry = EarthGeometry(
                r_E=config.r_E, h_atm=config.h_atm, h_obs=config.h_obs
            )
        self.geom = geometry
        self.thrad = None
        self.theta_deg = None
        # Replaced by the density at the atmosphere top on ``set_theta``.
        self._max_den = config.max_density
        self.max_theta = 90.0
        self.location = None
        self.season = None

    @abstractmethod
    def get_density(self, h_cm):
        """Density in g/cm**3 at the height ``h_cm`` in cm."""
        raise NotImplementedError("Base class called.")

    def _density_profile(self, h_vec):
        """Density in g/cm**3 at the heights ``h_vec``."""
        return np.vectorize(self.get_density)(h_vec)

    def calculate_density_spline(self, n_steps=2000):
        """Sample the density along the path at the current zenith angle and
        fit the splines of :meth:`_fit_splines`.

        Raises:
            Exception: if :func:`set_theta` was not called before.
        """
        if self.theta_deg is None:
            raise Exception("zenith angle not set")
        info(
            5, f"Calculating spline of rho(X) for zenith {self.theta_deg:4.1f} degrees."
        )
        dl_vec = np.linspace(0, self.geom.path_len(self.thrad), n_steps)
        h_vec = self.geom.h(dl_vec, self.thrad)
        now = time()
        rho_vec = self._density_profile(h_vec)
        info(5, f".. took {time() - now:1.2f}s")
        self._fit_splines(rho_vec, dl_vec, h_vec)

    def _fit_splines(self, rho_vec, dl_vec, h_vec):
        """Integrate the slant column and fit the quadratic splines h -> log X,
        X -> rho and log X -> h from the densities at the path lengths
        ``dl_vec`` (from the atmosphere top) and heights ``h_vec``. The heights
        must decrease along the path, so upgoing columns above a raised
        observation level are not supported."""
        X_int = cumulative_trapezoid(rho_vec, dl_vec)
        self._max_X = X_int[-1]
        self._max_den = rho_vec[0]  # the atmosphere top is the path minimum
        h_intp = np.asarray(h_vec[2:][::-1])
        log_X_intp = np.log(X_int[1:])[::-1]
        self._s_h2X = UnivariateSpline(h_intp, log_X_intp, k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(log_X_intp[::-1], h_intp[::-1], k=2, s=0.0)

    @property
    def max_X(self):
        """Slant depth at the observation level."""
        if not hasattr(self, "_max_X"):
            self.set_theta(0)
        return self._max_X

    @property
    def max_den(self):
        """Density at the atmosphere top (``config.max_density`` before ``set_theta``)."""
        if not hasattr(self, "_max_den"):
            self.set_theta(0)
        return self._max_den

    @property
    def s_h2X(self):
        """Spline for conversion from altitude to log depth."""
        if not hasattr(self, "_s_h2X"):
            self.set_theta(0)
        return self._s_h2X

    @property
    def s_X2rho(self):
        """Spline for conversion from depth to density."""
        if not hasattr(self, "_s_X2rho"):
            self.set_theta(0)
        return self._s_X2rho

    @property
    def s_lX2h(self):
        """Spline for conversion from log depth to altitude."""
        if not hasattr(self, "_s_lX2h"):
            self.set_theta(0)
        return self._s_lX2h

    def set_theta(self, theta_deg):
        """Set the zenith angle at the detector in degrees and rebuild the splines."""
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise Exception("Zenith angle not in allowed range.")
        self.thrad = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        self.calculate_density_spline()

    def set_h_obs(self, h_obs):
        """Set the elevation of the observation (detector) level in cm."""
        self.geom.set_h_obs(h_obs)
        if not self._preserve_max_theta:
            self.max_theta = self.geom.theta_max_deg
        if self.theta_deg:
            self.calculate_density_spline()

    def r_X2rho(self, X):
        """Inverse density :math:`1/\\rho` in cm**3/g at the slant depth ``X`` in g/cm**2."""
        return 1.0 / self.s_X2rho(X)

    def h2X(self, h):
        """Slant depth in g/cm**2 at the height ``h`` in cm."""
        return np.exp(self.s_h2X(h))

    def X2h(self, X):
        """Height in cm at the slant depth ``X`` in g/cm**2."""
        return self.s_lX2h(np.log(X))

    def X2rho(self, X):
        """Density in g/cm**3 at the slant depth ``X`` in g/cm**2."""
        return self.s_X2rho(X)

    def moliere_air(self, h_cm):
        """Returns the Moliere unit of air for US standard atmosphere."""
        return 9.3 / (self.get_density(h_cm) * 100.0)

    def nref_rel_air(self, h_cm):
        """Refractive index - 1 of air (density parametrization as in CORSIKA)."""
        return 0.000283 * self.get_density(h_cm) / self.get_density(0)

    def gamma_cherenkov_air(self, h_cm):
        """Lorentz factor at the Cherenkov threshold in air."""
        nrel = self.nref_rel_air(h_cm)
        return (1.0 + nrel) / np.sqrt(2.0 * nrel + nrel**2)

    def theta_cherenkov_air(self, h_cm):
        """Cherenkov angle in air in degrees."""
        return np.arccos(1.0 / (1.0 + self.nref_rel_air(h_cm))) * 180.0 / np.pi

    @property
    def current_impact_latitude(self):
        """Latitude in degrees of the shower impact point, for atmospheres
        bound to a detector position; None otherwise."""
        return None

    @property
    def current_impact_longitude(self):
        """Longitude in degrees of the shower impact point, or None."""
        return None


class CorsikaAtmosphere(EarthsAtmosphere):
    """Linsley's five-layer parameterisation, as in the air-shower Monte Carlo
    `CORSIKA <https://web.ikp.kit.edu/corsika/>`_.

    Args:
      location (str), season (str, optional): a parameter set of
        :func:`MCEq.geometry.atmosphere_parameters.list_available_corsika_atmospheres`
    """

    def __init__(self, location, season=None):
        available_atmospheres = list_available_corsika_atmospheres()
        if (location, season) not in available_atmospheres:
            raise ValueError(
                f"Atmosphere '{location}/{season}' not available. "
                f"Available atmospheres: {available_atmospheres}"
            )
        EarthsAtmosphere.__init__(self)
        import MCEq.geometry.corsikaatm as corsika_acc

        self.corsika_acc = corsika_acc
        self.init_parameters(location, season)

    def init_parameters(self, location, season):
        """Set the parameters ``_aatm, _batm, _catm, _thickl, _hlay`` of the
        five layers (:attr:`_atm_param`) for ``(location, season)``."""
        self._atm_param = np.array(get_atmosphere_parameters(location, season))
        self.location, self.season = location, season
        self.theta_deg = None

    def depth2height(self, x_v):
        """Height in cm at the vertical depth ``x_v`` in g/cm**2."""
        _aatm, _batm, _catm, _thickl, _hlay = self._atm_param
        for i in range(4):
            if x_v >= _thickl[i + 1]:
                return _catm[i] * np.log(_batm[i] / (x_v - _aatm[i]))
        return (_aatm[4] - x_v) * _catm[4]

    def get_density(self, h_cm):
        """Density in g/cm**3 at the height ``h_cm`` in cm."""
        return self.corsika_acc.corsika_get_density(h_cm, *self._atm_param)

    def get_mass_overburden(self, h_cm):
        """Vertical mass overburden in g/cm**2 at the height ``h_cm`` in cm."""
        return self.corsika_acc.corsika_get_m_overburden(h_cm, *self._atm_param)

    def rho_inv(self, X, cos_theta):
        """Inverse density in cm**3/g at the slant depth ``X`` in the planar
        approximation, valid below 70 degrees."""
        return self.corsika_acc.planar_rho_inv(X, cos_theta, *self._atm_param)

    def calc_thickl(self):
        """Vertical depths of the layer boundaries ``_thickl``, for a new
        parameter set; returned and logged as strings."""
        from scipy.integrate import quad

        thickl = []
        for h in self._atm_param[4]:
            thickl.append(f"{quad(self.get_density, h, 112.8e5, epsrel=1e-4)[0]:4.6f}")
        info(5, "_thickl = np.array([" + ", ".join(thickl) + "])")
        return thickl


class IsothermalAtmosphere(EarthsAtmosphere):
    """Isothermal atmosphere of scale height ``hiso_km`` and ground overburden
    ``X0`` in g/cm**2; the defaults are those of Thunman, Ingelman and Gondolo,
    Astropart. Phys. 5 (1996) 309. ``location`` and ``season`` have no effect."""

    def __init__(self, location, season, hiso_km=6.3, X0=1300.0):
        EarthsAtmosphere.__init__(self)
        self.hiso_cm = hiso_km * 1e5
        self.X0 = X0
        self.location = location
        self.season = season

    def get_density(self, h_cm):
        """Density in g/cm**3 at the height ``h_cm`` in cm."""
        return self.X0 / self.hiso_cm * np.exp(-h_cm / self.hiso_cm)

    def get_mass_overburden(self, h_cm):
        """Vertical mass overburden in g/cm**2 at the height ``h_cm`` in cm."""
        return self.X0 * np.exp(-h_cm / self.hiso_cm)


class MSISAtmosphere(EarthsAtmosphere):
    """Atmosphere of an NRLMSIS model at a named site and date.

    ``backend`` evaluates the model (NRLMSISE-00 in
    :class:`MCEq.geometry.nrlmsise00_mceq.cNRLMSISE00`, NRLMSIS 2.1 in
    :class:`MCEq.geometry.msis21.Msis21`). ``location`` is a key of
    :data:`MCEq.geometry.atmosphere_parameters.LOCATIONS`; the date is the
    month ``season`` or the day of year ``doy``, which wins if both are given
    (default: June);
    ``use_loc_altitudes`` puts the observation level at the site's altitude.
    """

    def __init__(
        self, backend, location, season=None, doy=None, use_loc_altitudes=False
    ):
        self._msis = backend
        EarthsAtmosphere.__init__(self)
        self.init_parameters(location, season, doy, use_loc_altitudes)

    def init_parameters(self, location, season, doy, use_loc_altitudes):
        """Set the site and the date of the model."""
        self.set_location(location)
        self._set_date(season, doy)
        if use_loc_altitudes:
            info(0, "Using loc altitude", self._msis.alt_surface, "cm")
            self.geom.set_h_obs(self._msis.alt_surface)

    def _clear_cache(self):
        """Rebuild the splines on the next ``set_theta``."""
        self.theta_deg = None

    def _set_date(self, season, doy):
        """Set the date to ``doy`` if given, else to the month ``season``."""
        if doy is not None:
            if season is not None:
                info(2, "Both 'season' and 'doy' are set; 'doy' takes precedence.")
            self.set_doy(doy)
        elif season is not None:
            self.set_season(season)

    def update_parameters(self, **kwargs):
        """Set ``location_coord=(longitude, latitude)``, ``season`` and/or
        ``doy``; ``doy`` takes precedence over ``season``."""
        self._clear_cache()
        if "location_coord" in kwargs:
            self.set_location_coord(*kwargs["location_coord"])
        self._set_date(kwargs.get("season"), kwargs.get("doy"))

    def get_density(self, h_cm):
        """Density in g/cm**3 at the height ``h_cm`` in cm."""
        return self._msis.get_density(h_cm)

    def get_temperature(self, h_cm):
        """Temperature in K at the height ``h_cm`` in cm."""
        return self._msis.get_temperature(h_cm)

    def _density_profile(self, h_vec):
        return self._msis.density_profile(h_vec)

    def set_location(self, location):
        """Move the model to a named site."""
        self._msis.set_location(location)
        self.location = self._msis.current_location
        self._clear_cache()

    def set_location_coord(self, longitude, latitude):
        """Move the model to ``(longitude, latitude)`` in degrees."""
        self._msis.set_location_coord(longitude, latitude)
        self._clear_cache()

    def set_season(self, month):
        """Set the date to the month ``month``."""
        self._msis.set_season(month)
        self.season = month
        self._clear_cache()

    def set_doy(self, day_of_year):
        """Set the date to the day of year ``day_of_year``."""
        self._msis.set_doy(day_of_year)
        self._clear_cache()


class MSISLocationCentered(MSISAtmosphere):
    """NRLMSIS atmosphere centred on a detector at an arbitrary site.

    The shower impact point for a zenith and azimuth is found on a spherical
    Earth and the model is evaluated there; without an azimuth the density is
    averaged over ``n_azimuth`` directions. Azimuth 0° is North, 90° East.
    Zenith is measured at the detector; above 90° is upgoing (``max_theta=180``).
    The column is integrated at the local zenith of the surface crossing,
    :math:`\\sin\\theta_s = (r_{det}/r_{surf})\\sin\\theta`.

    Args:
        backend: NRLMSIS backend, as in :class:`MSISAtmosphere`.
        detector_coord (tuple): ``(longitude, latitude)`` of the detector in degrees.
        depth_m (float): depth of the detector below the local surface in m.
        season (str, optional), doy (int, optional): the date.
        n_azimuth (int): azimuth samples of the averaging.
        max_theta (float): maximum zenith angle in degrees.
        surface_elevation_m (float): elevation of the local surface in m (an ice
            sheet, for example). The observation level is there for downgoing
            showers and at sea level for upgoing ones.
    """

    _preserve_max_theta: bool = True
    depends_on_azimuth: bool = True

    def __init__(
        self,
        backend,
        detector_coord,
        depth_m,
        season=None,
        doy=None,
        n_azimuth=36,
        max_theta=90.0,
        surface_elevation_m=0.0,
    ):
        longitude, latitude = detector_coord
        self._msis = backend
        EarthsAtmosphere.__init__(self)
        backend.set_location_coord(longitude, latitude)
        self._set_date(season, doy)

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

    def _impact_point(self, zenith_deg, azimuth_deg):
        """``(latitude, longitude)`` in degrees where the shower axis, followed
        from the detector toward the source, crosses the surface sphere at the
        observation level; for upgoing showers that is on the far side."""
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
        # Direction toward the source: East, North, Up
        d_ENU = np.array(
            [
                np.sin(theta) * np.sin(alpha),
                np.sin(theta) * np.cos(alpha),
                np.cos(theta),
            ]
        )
        # ENU -> ECEF rotation (columns: East, North, Up)
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
        # Larger root of |P_det + x d|^2 = r^2: the source-side crossing
        A = np.dot(d_ECEF, P_det)
        x = -A + np.sqrt(A**2 + (r**2 - r_det**2))
        P_impact = P_det + x * d_ECEF
        lat_imp = np.rad2deg(np.arcsin(np.clip(P_impact[2] / r, -1.0, 1.0)))
        lon_imp = np.rad2deg(np.arctan2(P_impact[1], P_impact[0]))
        return lat_imp, lon_imp

    def _latitude(self, det_zenith_deg):
        """Latitude of the impact point at azimuth 0."""
        return self._impact_point(det_zenith_deg, 0.0)[0]

    def set_theta(self, theta_deg, azimuth_deg=None):
        """Set the zenith angle at the detector and, optionally, the azimuth.

        Rays within the dip of the local surface below the horizon still exit
        through the near-side surface; steeper ones cross the Earth, and their
        column ends at sea level on the far side.
        """
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise ValueError(
                f"Zenith angle {theta_deg} not in allowed range [0, {self.max_theta}]."
            )
        r_E = self.geom.r_E  # cm
        elev_cm = self._surface_elevation_m * 1e2
        r_det = r_E + elev_cm - self._detector_depth_m * 1e2
        dip_deg = np.rad2deg(np.arccos(min(r_det / (r_E + elev_cm), 1.0)))
        h_obs_cm = 0.0 if theta_deg > 90.0 + dip_deg else elev_cm
        if self.geom.h_obs != h_obs_cm:
            self.geom.set_h_obs(h_obs_cm)

        # The impact parameter r sin(theta) is conserved along the axis.
        sin_loc = np.clip(
            r_det / self.geom.r_obs * np.sin(np.deg2rad(theta_deg)), 0.0, 1.0
        )
        effective_theta = np.rad2deg(np.arcsin(sin_loc))

        if azimuth_deg is not None:
            lat, lon = self._impact_point(theta_deg, azimuth_deg)
            self._msis.set_location_coord(lon, lat)
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
            azi_grid = np.linspace(0.0, 360.0, self._n_azimuth, endpoint=False)
            self._azimuth_avg_coords = [
                self._impact_point(theta_deg, a) for a in azi_grid
            ]
            self._azimuth_averaging = True
            self._current_impact_latitude = None
            self._current_impact_longitude = None

        self._effective_theta_deg = effective_theta
        self._current_azimuth_deg = azimuth_deg
        self.thrad = np.deg2rad(effective_theta)
        self.theta_deg = theta_deg  # may be > 90 for upgoing
        self.calculate_density_spline()

    def get_density(self, h_cm):
        """Density in g/cm**3 at ``h_cm``, averaged over the azimuths when no
        azimuth is set."""
        if not self._azimuth_averaging:
            return self._msis.get_density(h_cm)
        total = 0.0
        for lat, lon in self._azimuth_avg_coords:
            self._msis.set_location_coord(lon, lat)
            total += self._msis.get_density(h_cm)
        return total / len(self._azimuth_avg_coords)

    def _density_profile(self, h_vec):
        if not self._azimuth_averaging:
            return self._msis.density_profile(h_vec)
        rho_sum = np.zeros(len(h_vec))
        for lat, lon in self._azimuth_avg_coords:
            self._msis.set_location_coord(lon, lat)
            rho_sum += self._msis.density_profile(h_vec)
        return rho_sum / len(self._azimuth_avg_coords)

    @property
    def current_impact_latitude(self):
        """Latitude of the impact point; None while averaging over azimuth."""
        return self._current_impact_latitude

    @property
    def current_impact_longitude(self):
        """Longitude of the impact point; None while averaging over azimuth."""
        return self._current_impact_longitude


#: Detector-centred setup of IceCube: 1948 m below the ice surface at 2835 m.
ICECUBE = dict(
    detector_coord=(0.0, -90.0),
    depth_m=1948.0,
    max_theta=180.0,
    surface_elevation_m=2835.0,
)


def km3net(detector):
    """Detector-centred setup of the KM3NeT site ``"ORCA"`` or ``"ARCA"``."""
    if detector not in KM3NET_DETECTORS:
        raise ValueError(
            f"Unknown KM3NeT detector '{detector}'. "
            f"Choose from {list(KM3NET_DETECTORS.keys())}."
        )
    det = KM3NET_DETECTORS[detector]
    return dict(
        detector_coord=(det["longitude"], det["latitude"]),
        depth_m=det["depth_m"],
        max_theta=180.0,
    )


def _cnrlmsise00():
    from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

    return cNRLMSISE00()


class MSIS00Atmosphere(MSISAtmosphere):
    """NRLMSISE-00 atmosphere at a named site; arguments as in :class:`MSISAtmosphere`."""

    def __init__(self, location, season=None, doy=None, use_loc_altitudes=False):
        super().__init__(_cnrlmsise00(), location, season, doy, use_loc_altitudes)


class MSIS00LocationCentered(MSISLocationCentered):
    """NRLMSISE-00 atmosphere centred on a detector; arguments as in
    :class:`MSISLocationCentered` without the backend."""

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
            _cnrlmsise00(),
            detector_coord,
            depth_m,
            season,
            doy,
            n_azimuth,
            max_theta,
            surface_elevation_m,
        )


class MSIS00IceCubeCentered(MSIS00LocationCentered):
    """NRLMSISE-00 atmosphere centred on IceCube; ``location`` is ignored."""

    def __init__(self, location, season):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
        super().__init__(season=season, **ICECUBE)


class MSIS00KM3NeTCentered(MSIS00LocationCentered):
    """NRLMSISE-00 atmosphere centred on the KM3NeT detector ``"ORCA"`` or ``"ARCA"``."""

    def __init__(self, detector, season=None, doy=None, n_azimuth=36):
        super().__init__(
            season=season, doy=doy, n_azimuth=n_azimuth, **km3net(detector)
        )
        self._detector_name = detector


class AIRSAtmosphere(EarthsAtmosphere):
    """Atmosphere interpolated from daily AIRS satellite tables at the South
    Pole, extended above the tables with NRLMSISE-00 if ``extrapolate``.

    ``table_path`` is the directory holding ``tables/airs_amsu_*_180_daily.txt``.
    """

    def __init__(self, location, season, extrapolate=True, table_path=None):
        if location != "SouthPole":
            raise Exception(
                f"{self.__class__.__name__}(): Only South Pole location supported. {location}"
            )
        if table_path is None:
            raise ValueError("AIRSAtmosphere needs the table_path of the AIRS tables")
        EarthsAtmosphere.__init__(self)
        self.extrapolate = extrapolate
        self.month2doy = dict(MONTH_TO_DAY_OF_YEAR)
        self.season = season
        self.init_parameters(location, table_path)

    def init_parameters(self, location, table_path):
        """Read the tables and prepare the interpolation."""
        from matplotlib.dates import datestr2num, num2date

        def bytespdate2num(b):
            return datestr2num(b.decode("utf-8"))

        files = [
            ("dens", "airs_amsu_dens_180_daily.txt"),
            ("temp", "airs_amsu_temp_180_daily.txt"),
            ("alti", "airs_amsu_alti_180_daily.txt"),
        ]
        data_collection = {}
        min_press_idx = 4  # limit the South Pole pressure to <= 600
        IC79_idx_1 = IC79_idx_2 = None
        for d_key, fname in files:
            tab = np.loadtxt(
                join(table_path, "tables", fname),
                converters={0: bytespdate2num},
                usecols=[0] + list(range(2, 27)),
            )
            dates = num2date(tab[:, 0])
            for di, date in enumerate(dates):
                if date.month == 6 and date.day == 1:
                    if date.year == 2010:
                        IC79_idx_1 = di
                    elif date.year == 2011:
                        IC79_idx_2 = di
            data_collection[d_key] = (dates, tab[:, 1], tab[:, min_press_idx + 2 :])

        self.interp_tab_d = {}
        self.interp_tab_t = {}
        self.dates = {}
        dates = data_collection["alti"][0]

        msis = MSIS00Atmosphere(location, "January")
        for didx, date in enumerate(dates):
            h_vec = np.array(data_collection["alti"][2][didx, :] * 1e2)
            d_vec = np.array(data_collection["dens"][2][didx, :])
            t_vec = np.array(data_collection["temp"][2][didx, :])
            if self.extrapolate:
                h_extra = np.linspace(h_vec[-1], self.geom.h_atm * 1e2, 250)
                msis._msis.set_doy(self._get_y_doy(date)[1] - 1)
                msis_extra_d = np.array([msis.get_density(h) for h in h_extra])
                msis_extra_t = np.array([msis.get_temperature(h) for h in h_extra])
                # Blend the last few altitude bins into MSIS
                ninterp = 5
                for ni in range(ninterp):
                    cl = 1 - np.exp(-ninterp + ni + 1)
                    ch = 1 - np.exp(-ni)
                    norm = 1.0 / (cl + ch)
                    d_vec[-ni - 1] = (
                        d_vec[-ni - 1] * cl * norm
                        + msis.get_density(h_vec[-ni - 1]) * ch * norm
                    )
                    t_vec[-ni - 1] = (
                        t_vec[-ni - 1] * cl * norm
                        + msis.get_temperature(h_vec[-ni - 1]) * ch * norm
                    )
                h_vec = np.hstack([h_vec[:-1], h_extra])
                d_vec = np.hstack([d_vec[:-1], msis_extra_d])
                t_vec = np.hstack([t_vec[:-1], msis_extra_t])
            key = self._get_y_doy(date)
            self.interp_tab_d[key] = (h_vec, d_vec)
            self.interp_tab_t[key] = (h_vec, t_vec)
            self.dates[key] = date

        self.IC79_start = self._get_y_doy(dates[IC79_idx_1])
        self.IC79_end = self._get_y_doy(dates[IC79_idx_2])
        self.IC79_days = (dates[IC79_idx_2] - dates[IC79_idx_1]).days
        self.location = location
        if self.season is None:
            self.set_IC79_day(0)
        else:
            self.set_season(self.season)
        self.theta_deg = None

    def set_date(self, year, doy):
        self.h, self.dens = self.interp_tab_d[(year, doy)]
        _, self.temp = self.interp_tab_t[(year, doy)]
        self.date = self.dates[(year, doy)]
        self.season = self.date

    def _set_doy(self, doy, year=2010):
        self.h, self.dens = self.interp_tab_d[(year, doy)]
        _, self.temp = self.interp_tab_t[(year, doy)]
        self.date = self.dates[(year, doy)]

    def set_season(self, month):
        self.season = month
        self._set_doy(self.month2doy[month])

    def set_IC79_day(self, IC79_day):
        import datetime

        if IC79_day > self.IC79_days:
            raise Exception(
                self.__class__.__name__ + "::set_IC79_day(): IC79_day above range."
            )
        target_day = self._get_y_doy(
            self.dates[self.IC79_start] + datetime.timedelta(days=IC79_day)
        )
        info(2, "setting IC79_day", IC79_day)
        self.h, self.dens = self.interp_tab_d[target_day]
        _, self.temp = self.interp_tab_t[target_day]
        self.date = self.dates[target_day]
        self.season = self.date

    def _get_y_doy(self, date):
        return date.timetuple().tm_year, date.timetuple().tm_yday

    def _interp(self, h_cm, table):
        ret = np.exp(np.interp(h_cm, self.h, np.log(table)))
        try:
            ret[h_cm > self.h[-1]] = np.nan
        except TypeError:
            if h_cm > self.h[-1]:
                return np.nan
        return ret

    def get_density(self, h_cm):
        """Density in g/cm**3 at ``h_cm``, interpolated in the table of the set day."""
        return self._interp(h_cm, self.dens)

    def get_temperature(self, h_cm):
        """Temperature in K at ``h_cm``, interpolated in the table of the set day."""
        return self._interp(h_cm, self.temp)
