from abc import ABCMeta, abstractmethod
from os.path import join

import numpy as np
from six import with_metaclass

from MCEq import config

# Import the new atmosphere data module
from MCEq.geometry.atmosphere_parameters import (
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)
from MCEq.misc import info, theta_rad


class EarthsAtmosphere(with_metaclass(ABCMeta)):
    """
    Abstract class containing common methods on atmosphere.
    You have to inherit from this class and implement the virtual method
    :func:`get_density`.

    Note:
      Do not instantiate this class directly.

    Attributes:
      thrad (float): current zenith angle :math:`\\theta` in radiants
      theta_deg (float): current zenith angle :math:`\\theta` in degrees
      max_X (float): Slant depth at the surface according to the geometry
                     defined in the :mod:`MCEq.geometry`
      geometry (object): Can be a custom instance of EarthGeometry

    """

    def __init__(self, *args, **kwargs):
        from MCEq.geometry.geometry import EarthGeometry

        self.geom = kwargs.pop("geometry", EarthGeometry())
        self.thrad = None
        self.theta_deg = None
        self._max_den = config.max_density
        self.max_theta = 90.0
        self.location = None
        self.season = None

    @abstractmethod
    def get_density(self, h_cm):
        """Abstract method which implementation  should return the density in g/cm**3.

        Args:
           h_cm (float):  height in cm

        Returns:
           float: density in g/cm**3

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError("Base class called.")

    def calculate_density_spline(self, n_steps=2000):
        """Calculates and stores a spline of :math:`\\rho(X)`.

        Args:
          n_steps (int, optional): number of :math:`X` values
                                   to use for interpolation

        Raises:
            Exception: if :func:`set_theta` was not called before.
        """
        from time import time

        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import UnivariateSpline

        if self.theta_deg is None:
            raise Exception("zenith angle not set")
        info(
            5,
            f"Calculating spline of rho(X) for zenith {self.theta_deg:4.1f} degrees.",
        )

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        vec_rho_l = np.vectorize(
            lambda delta_l: self.get_density(self.geom.h(delta_l, thrad))
        )
        dl_vec = np.linspace(0, path_length, n_steps)

        now = time()

        # Calculate integral for each depth point
        X_int = cumulative_trapezoid(vec_rho_l(dl_vec), dl_vec)
        dl_vec = dl_vec[1:]

        info(5, f".. took {time() - now:1.2f}s")

        # Save depth value at h_obs
        self._max_X = X_int[-1]
        self._max_den = self.get_density(self.geom.h(0, thrad))

        # Interpolate with bi-splines without smoothing
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[1:])]
        X_intp = [X for X in reversed(X_int[1:])]

        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, vec_rho_l(dl_vec), k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1], k=2, s=0.0)

    @property
    def max_X(self):
        """Depth at altitude 0."""
        if not hasattr(self, "_max_X"):
            self.set_theta(0)
        return self._max_X

    @property
    def max_den(self):
        """Density at altitude 0."""
        if not hasattr(self, "_max_den"):
            self.set_theta(0)
        return self._max_den

    @property
    def s_h2X(self):
        """Spline for conversion from altitude to depth."""
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
        """Spline for conversion from depth to altitude."""
        if not hasattr(self, "_s_lX2h"):
            self.set_theta(0)
        return self._s_lX2h

    def set_theta(self, theta_deg):
        """Configures geometry and initiates spline calculation for
        :math:`\\rho(X)`.

        If the option 'use_atm_cache' is enabled in the config, the
        function will check, if a corresponding spline is available
        in the cache and use it. Otherwise it will call
        :func:`calculate_density_spline`,  make the function
        :func:`r_X2rho` available to the core code and store the spline
        in the cache.

        Args:
          theta_deg (float): zenith angle :math:`\\theta` at detector
        """
        if theta_deg < 0.0 or theta_deg > self.max_theta:
            raise Exception("Zenith angle not in allowed range.")

        self.thrad = theta_rad(theta_deg)
        self.theta_deg = theta_deg
        self.calculate_density_spline()

    def r_X2rho(self, X):
        """Returns the inverse density :math:`\\frac{1}{\\rho}(X)`.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`1/\\rho` in cm**3/g

        """
        return 1.0 / self.s_X2rho(X)

    def h2X(self, h):
        """Returns the depth along path as function of height above
        surface.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           h (float):  vertical height above surface in cm

        Returns:
           float: X  slant depth in g/cm**2

        """
        return np.exp(self.s_h2X(h))

    def X2h(self, X):
        """Returns the height above surface as a function of slant depth
        for currently selected zenith angle.

        The spline `s_lX2h` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float h:  height above surface in cm

        """
        return self.s_lX2h(np.log(X))

    def X2rho(self, X):
        """Returns the density :math:`\\rho(X)`.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`\\rho` in cm**3/g

        """
        return self.s_X2rho(X)

    def moliere_air(self, h_cm):
        """Returns the Moliere unit of air for US standard atmosphere."""

        return 9.3 / (self.get_density(h_cm) * 100.0)

    def nref_rel_air(self, h_cm):
        """Returns the refractive index - 1 in air (density parametrization
        as in CORSIKA).
        """

        return 0.000283 * self.get_density(h_cm) / self.get_density(0)

    def gamma_cherenkov_air(self, h_cm):
        """Returns the Lorentz factor gamma of Cherenkov threshold in air (MeV)."""

        nrel = self.nref_rel_air(h_cm)
        return (1.0 + nrel) / np.sqrt(2.0 * nrel + nrel**2)

    def theta_cherenkov_air(self, h_cm):
        """Returns the Cherenkov angle in air (degrees)."""

        return np.arccos(1.0 / (1.0 + self.nref_rel_air(h_cm))) * 180.0 / np.pi


class CorsikaAtmosphere(EarthsAtmosphere):
    """Class, holding the parameters of a Linsley type parameterization
    similar to the Air-Shower Monte Carlo
    `CORSIKA <https://web.ikp.kit.edu/corsika/>`_.

    The parameters pre-defined parameters are taken from the CORSIKA
    manual. If new sets of parameters are added to :func:`init_parameters`,
    the array _thickl can be calculated using :func:`calc_thickl` .

    Attributes:
      _atm_param (numpy.array): (5x5) Stores 5 atmospheric parameters
                                _aatm, _batm, _catm, _thickl, _hlay
                                for each of the 5 layers

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    _atm_param = None

    def __init__(self, location, season=None):
        # Check if the atmosphere is available
        # Use the renamed list_available_atmospheres function
        available_atmospheres = list_available_corsika_atmospheres()
        if (location, season) not in available_atmospheres:
            raise ValueError(
                f"Atmosphere '{location}/{season}' not available. "
                f"Available atmospheres: {available_atmospheres}"
            )

        self.init_parameters(location, season)
        import MCEq.geometry.corsikaatm.corsikaatm as corsika_acc

        self.corsika_acc = corsika_acc  # Assuming corsika_acc is defined elsewhere or needs to be imported
        EarthsAtmosphere.__init__(self)

    def init_parameters(self, location, season):
        """Initializes :attr:`_atm_param` by fetching them from the
        `atmosphere_parameters` module.

        Args:
          location (str): The location identifier.
          season (str, optional): The season identifier.

        Raises:
          Exception: if parameter set not available (via get_atmosphere_parameters)
        """
        # Use the renamed get_atmosphere_parameters function
        _aatm, _batm, _catm, _thickl, _hlay = get_atmosphere_parameters(
            location, season
        )

        self._atm_param = np.array([_aatm, _batm, _catm, _thickl, _hlay])

        self.location, self.season = location, season
        # Clear cached theta value to force spline recalculation
        self.theta_deg = None

    def depth2height(self, x_v):
        """Converts column/vertical depth to height.

        Args:
          x_v (float): column depth :math:`X_v` in g/cm**2

        Returns:
          float: height in cm
        """
        _aatm, _batm, _catm, _thickl, _hlay = self._atm_param

        if x_v >= _thickl[1]:
            height = _catm[0] * np.log(_batm[0] / (x_v - _aatm[0]))
        elif x_v >= _thickl[2]:
            height = _catm[1] * np.log(_batm[1] / (x_v - _aatm[1]))
        elif x_v >= _thickl[3]:
            height = _catm[2] * np.log(_batm[2] / (x_v - _aatm[2]))
        elif x_v >= _thickl[4]:
            height = _catm[3] * np.log(_batm[3] / (x_v - _aatm[3]))
        else:
            height = (_aatm[4] - x_v) * _catm[4]

        return height

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Uses the optimized module function :func:`corsika_get_density_jit`.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self.corsika_acc.corsika_get_density(h_cm, *self._atm_param)
        # return corsika_get_density_jit(h_cm, self._atm_param)

    def get_mass_overburden(self, h_cm):
        """Returns the mass overburden in atmosphere in g/cm**2.

        Uses the optimized module function :func:`corsika_get_m_overburden_jit`

        Args:
          h_cm (float): height in cm

        Returns:
          float: column depth :math:`T(h_{cm})` in g/cm**2
        """
        return self.corsika_acc.corsika_get_m_overburden(h_cm, *self._atm_param)
        # return corsika_get_m_overburden_jit(h_cm, self._atm_param)

    def rho_inv(self, X, cos_theta):
        """Returns reciprocal density in cm**3/g using planar approximation.

        This function uses the optimized function :func:`planar_rho_inv_jit`

        Args:
          h_cm (float): height in cm

        Returns:
          float: :math:`\\frac{1}{\\rho}(X,\\cos{\\theta})` cm**3/g
        """
        return self.corsika_acc.planar_rho_inv(X, cos_theta, *self._atm_param)
        # return planar_rho_inv_jit(X, cos_theta, self._atm_param)

    def calc_thickl(self):
        """Calculates thickness layers for :func:`depth2height`

        The analytical inversion of the CORSIKA parameterization
        relies on the knowledge about the depth :math:`X`, where
        trasitions between layers/exponentials occur.

        Example:
          Create a new set of parameters in :func:`init_parameters`
          inserting arbitrary values in the _thikl array::

          $ cor_atm = CorsikaAtmosphere(new_location, new_season)
          $ cor_atm.calc_thickl()

          Replace _thickl values with printout.

        """
        from scipy.integrate import quad

        thickl = []
        for h in self._atm_param[4]:
            thickl.append(f"{quad(self.get_density, h, 112.8e5, epsrel=1e-4)[0]:4.6f}")
        info(5, "_thickl = np.array([" + ", ".join(thickl) + "])")
        return thickl


class IsothermalAtmosphere(EarthsAtmosphere):
    """Isothermal model of the atmosphere.

    This model is widely used in semi-analytical calculations. The isothermal
    approximation is valid in a certain range of altitudes and usually
    one adjust the parameters to match a more realistic density profile
    at altitudes between 10 - 30 km, where the high energy muon production
    rate peaks. Such parametrizations are given in the book "Cosmic Rays and
    Particle Physics", Gaisser, Engel and Resconi (2016). The default values
    are from M. Thunman, G. Ingelman, and P. Gondolo, Astropart. Physics 5,
    309 (1996).

    Args:
      location (str): no effect
      season (str): no effect
      hiso_km (float): isothermal scale height in km
      X0 (float): Ground level overburden
    """

    def __init__(self, location, season, hiso_km=6.3, X0=1300.0):
        self.hiso_cm = hiso_km * 1e5
        self.X0 = X0
        self.location = location
        self.season = season

        EarthsAtmosphere.__init__(self)

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self.X0 / self.hiso_cm * np.exp(-h_cm / self.hiso_cm)

    def get_mass_overburden(self, h_cm):
        """Returns the mass overburden in atmosphere in g/cm**2.

        Args:
          h_cm (float): height in cm

        Returns:
          float: column depth :math:`T(h_{cm})` in g/cm**2
        """
        return self.X0 * np.exp(-h_cm / self.hiso_cm)


class MSIS00Atmosphere(EarthsAtmosphere):
    """Wrapper class for a python interface to the NRLMSISE-00 model.

    `NRLMSISE-00 <http://ccmc.gsfc.nasa.gov/modelweb/atmos/nrlmsise00.html>`_
    is an empirical model of the Earth's atmosphere. It is available as
    a FORTRAN 77 code or as a verson traslated into
    `C by Dominik Borodowski <http://www.brodo.de/english/pub/nrlmsise/>`_.
    Here a PYTHON wrapper has been used.

    Attributes:
      _msis : NRLMSISE-00 python wrapper object handler

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    def __init__(self, location, season=None, doy=None, use_loc_altitudes=False):
        from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

        msis_atmospheres = [
            "SouthPole",
            "Karlsruhe",
            "Geneva",
            "Tokyo",
            "SanGrasso",
            "TelAviv",
            "KSC",
            "SoudanMine",
            "Tsukuba",
            "LynnLake",
            "PeaceRiver",
            "FtSumner",
        ]
        assert (
            location in msis_atmospheres
        ), f"{location} not available for MSIS00Atmosphere"

        self._msis = cNRLMSISE00()

        self.init_parameters(location, season, doy, use_loc_altitudes)

        EarthsAtmosphere.__init__(self)

    def init_parameters(self, location, season, doy, use_loc_altitudes):
        """Sets location and season in :class:`NRLMSISE-00`.

        Translates location and season into day of year
        and geo coordinates.

        Args:
          location (str): Supported are "SouthPole" and "Karlsruhe"
          season (str): months of the year: January, February, etc.
          use_loc_altitudes (bool): If to use default altitudes from location
        """
        self._msis.set_location(location)
        if season is not None:
            self._msis.set_season(season)
        else:
            self._msis.set_doy(doy)
        self.location, self.season = location, season
        # Clear cached value to force spline recalculation
        self.theta_deg = None
        if use_loc_altitudes:
            info(0, "Using loc altitude", self._msis.alt_surface, "cm")
            self.geom.h_obs = self._msis.alt_surface

    def _clear_cache(self):
        """Clears the density model cache so that density profiles can be recalculated

        It is a private method to wrap the logic of cache cleaning
        """
        self.theta_deg = None

    def update_parameters(self, **kwargs):
        """Updates parameters of the density model

        Args:
          location_coord (tuple of str): (longitude, latitude)
          season (str): months of the year: January, February, etc.
          doy (int): day of the year. 'doy' takes precedence over 'season' if both are set
        """

        self._clear_cache()
        if not kwargs:
            return

        if "location_coord" in kwargs:
            self.set_location_coord(*kwargs.get("location_coord"))

        if "season" in kwargs:
            self.set_season(kwargs.get("season"))

        if "doy" in kwargs:
            self.set_doy(kwargs.get("doy"))
            if "season" in kwargs:
                info(
                    2,
                    "Both 'season' and 'doy' are set in parameter list.\n'doy' takes precedence over 'season'",
                )

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self._msis.get_density(h_cm)

    def set_location(self, location):
        """Changes MSIS location by strings defined in _msis_wrapper.

        Args:
          location (str): location as defined in :class:`NRLMSISE-00.`

        """
        self._msis.set_location(location)
        self._clear_cache()

    def set_location_coord(self, longitude, latitude):
        """Changes MSIS location by longitude, latitude in _msis_wrapper

        Args:
          longitude (float): longitude of the location with abs(longitude) <= 180
          latitude (float): latitude of the location with abs(latitude) <= 90

        """
        self._msis.set_location_coord(longitude, latitude)
        self._clear_cache()

    def set_season(self, month):
        """Changes MSIS location by month strings defined in _msis_wrapper.

        Args:
          location (str): month as defined in :class:`NRLMSISE-00.`

        """
        self._msis.set_season(month)
        self._clear_cache()

    def set_doy(self, day_of_year):
        """Changes MSIS season by day of year.

        Args:
          day_of_year (int): 1. Jan.=0, 1.Feb=32

        """
        self._msis.set_doy(day_of_year)
        self._clear_cache()

    def get_temperature(self, h_cm):
        """Returns the temperature of air in K.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`T(h_{cm})` in K
        """
        return self._msis.get_temperature(h_cm)


class AIRSAtmosphere(EarthsAtmosphere):
    """Interpolation class for tabulated atmospheres.

    This class is intended to read preprocessed AIRS Satellite data.

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    def __init__(self, location, season, extrapolate=True, *args, **kwargs):
        if location != "SouthPole":
            raise Exception(
                self.__class__.__name__
                + "(): Only South Pole location supported. "
                + location
            )

        self.extrapolate = extrapolate

        self.month2doy = {
            "January": 1,
            "February": 32,
            "March": 60,
            "April": 91,
            "May": 121,
            "June": 152,
            "July": 182,
            "August": 213,
            "September": 244,
            "October": 274,
            "November": 305,
            "December": 335,
        }

        self.season = season
        self.init_parameters(location, **kwargs)
        EarthsAtmosphere.__init__(self)

    def init_parameters(self, location, **kwargs):
        """Loads tables and prepares interpolation.

        Args:
          location (str): supported is only "SouthPole"
          doy (int): Day Of Year
        """
        # from time import strptime
        from os import path

        from matplotlib.dates import datestr2num, num2date

        def bytespdate2num(b):
            return datestr2num(b.decode("utf-8"))

        data_path = join(
            path.expanduser("~"), "OneDrive/Dokumente/projects/atmospheric_variations/"
        )

        if "table_path" in kwargs:
            data_path = kwargs["table_path"]

        files = [
            ("dens", "airs_amsu_dens_180_daily.txt"),
            ("temp", "airs_amsu_temp_180_daily.txt"),
            ("alti", "airs_amsu_alti_180_daily.txt"),
        ]

        data_collection = {}

        # limit SouthPole pressure to <= 600
        min_press_idx = 4

        IC79_idx_1 = None
        IC79_idx_2 = None

        for d_key, fname in files:
            fname = data_path + "tables/" + fname
            # tabf = open(fname).read()

            tab = np.loadtxt(
                fname, converters={0: bytespdate2num}, usecols=[0] + list(range(2, 27))
            )
            # with open(fname, 'r') as f:
            #     comline = f.readline()
            # p_levels = [
            #     float(s.strip()) for s in comline.split(' ')[3:] if s != ''
            # ][min_press_idx:]
            dates = num2date(tab[:, 0])
            for di, date in enumerate(dates):
                if date.month == 6 and date.day == 1:
                    if date.year == 2010:
                        IC79_idx_1 = di
                    elif date.year == 2011:
                        IC79_idx_2 = di
            surf_val = tab[:, 1]
            cols = tab[:, min_press_idx + 2 :]
            data_collection[d_key] = (dates, surf_val, cols)

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
                # Extrapolate using msis
                h_extra = np.linspace(h_vec[-1], self.geom.h_atm * 1e2, 250)
                msis._msis.set_doy(self._get_y_doy(date)[1] - 1)
                msis_extra_d = np.array([msis.get_density(h) for h in h_extra])
                msis_extra_t = np.array([msis.get_temperature(h) for h in h_extra])

                # Interpolate last few altitude bins
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

                # Merge the two datasets
                h_vec = np.hstack([h_vec[:-1], h_extra])
                d_vec = np.hstack([d_vec[:-1], msis_extra_d])
                t_vec = np.hstack([t_vec[:-1], msis_extra_t])

            self.interp_tab_d[self._get_y_doy(date)] = (h_vec, d_vec)
            self.interp_tab_t[self._get_y_doy(date)] = (h_vec, t_vec)

            self.dates[self._get_y_doy(date)] = date

        self.IC79_start = self._get_y_doy(dates[IC79_idx_1])
        self.IC79_end = self._get_y_doy(dates[IC79_idx_2])
        self.IC79_days = (dates[IC79_idx_2] - dates[IC79_idx_1]).days
        self.location = location
        if self.season is None:
            self.set_IC79_day(0)
        else:
            self.set_season(self.season)
        # Clear cached value to force spline recalculation
        self.theta_deg = None

    def set_date(self, year, doy):
        self.h, self.dens = self.interp_tab_d[(year, doy)]
        _, self.temp = self.interp_tab_t[(year, doy)]
        self.date = self.dates[(year, doy)]
        # Compatibility with caching
        self.season = self.date

    def _set_doy(self, doy, year=2010):
        self.h, self.dens = self.interp_tab_d[(year, doy)]
        _, self.temp = self.interp_tab_t[(year, doy)]
        self.date = self.dates[(year, doy)]

    def set_season(self, month):
        self.season = month
        self._set_doy(self.month2doy[month])
        self.season = month

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
        # Compatibility with caching
        self.season = self.date

    def _get_y_doy(self, date):
        return date.timetuple().tm_year, date.timetuple().tm_yday

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Interpolates table at requested value for previously set
        year and day of year (doy).

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        ret = np.exp(np.interp(h_cm, self.h, np.log(self.dens)))
        try:
            ret[h_cm > self.h[-1]] = np.nan
        except TypeError:
            if h_cm > self.h[-1]:
                return np.nan
        return ret

    def get_temperature(self, h_cm):
        """Returns the temperature in K.

        Interpolates table at requested value for previously set
        year and day of year (doy).

        Args:
          h_cm (float): height in cm

        Returns:
          float: temperature :math:`T(h_{cm})` in K
        """
        ret = np.exp(np.interp(h_cm, self.h, np.log(self.temp)))
        try:
            ret[h_cm > self.h[-1]] = np.nan
        except TypeError:
            if h_cm > self.h[-1]:
                return np.nan
        return ret


class MSIS00IceCubeCentered(MSIS00Atmosphere):
    """Extension of :class:`MSIS00Atmosphere` which couples the latitude
    setting with the zenith angle of the detector.

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    def __init__(self, location, season):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
            location = "SouthPole"

        MSIS00Atmosphere.__init__(self, location, season)

        # Allow for upgoing zenith angles
        self.max_theta = 180.0

    def latitude(self, det_zenith_deg):
        """Returns the geographic latitude of the shower impact point.

        Assumes a spherical earth. The detector is 1948m under the
        surface.

        Credits: geometry fomulae by Jakob van Santen, DESY Zeuthen.

        Args:
          det_zenith_deg (float): zenith angle at detector in degrees

        Returns:
          float: latitude of the impact point in degrees
        """
        r = self.geom.r_E
        d = 1948  # m

        theta_rad = det_zenith_deg / 180.0 * np.pi

        x = np.sqrt(2.0 * r * d + ((r - d) * np.cos(theta_rad)) ** 2 - d**2) - (
            r - d
        ) * np.cos(theta_rad)

        return (
            -90.0
            + np.arctan2(x * np.sin(theta_rad), r - d + x * np.cos(theta_rad))
            / np.pi
            * 180.0
        )

    def set_theta(self, theta_deg):

        self._msis.set_location_coord(longitude=0.0, latitude=self.latitude(theta_deg))
        info(
            1,
            f"latitude = {self.latitude(theta_deg):5.2f} for zenith angle = {theta_deg:5.2f}",
        )
        downgoing_theta_deg = theta_deg
        if theta_deg > 90.0:
            downgoing_theta_deg = 180.0 - theta_deg
            info(
                1,
                f"theta = {theta_deg:5.2f} below horizon. using theta = {downgoing_theta_deg:5.2f}",
            )
        MSIS00Atmosphere.set_theta(self, downgoing_theta_deg)

        self.theta_deg = theta_deg


class GeneralizedTarget:
    """This class provides a way to run MCEq on piece-wise constant
    one-dimenional density profiles.

    The default values for the average density are taken from
    config file variables `len_target`, `env_density` and `env_name`.
    The density profile has to be built by calling subsequently
    :func:`add_material`. The current composition of the target
    can be checked with :func:`draw_materials` or :func:`print_table`.

    Note:
      If the target is not air or hydrogen, the result is approximate,
      since seconday particle yields are provided for nucleon-air or
      proton-proton collisions. Depending on this choice one has to
      adjust the nuclear mass in :mod:`mceq_config`.

    Args:
      len_target (float): total length of the target in meters
      env_density (float): density of the default material in g/cm**3
      env_name (str): title for this environment
    """

    def __init__(
        self,
        len_target=config.len_target * 1e2,  # cm
        env_density=config.env_density,  # g/cm3
        env_name=config.env_name,
    ):

        self.len_target = len_target
        self.env_density = env_density
        self.env_name = env_name
        self.reset()

    @property
    def max_den(self):
        return self._max_den

    def reset(self):
        """Resets material list to defaults."""
        self.mat_list = [[0.0, self.len_target, self.env_density, self.env_name]]
        self._update_variables()

    def _update_variables(self):
        """Updates internal variables. Not needed to call by user."""

        self.start_bounds, self.end_bounds, self.densities = list(zip(*self.mat_list))[
            :-1
        ]
        self.densities = np.array(self.densities)
        self.start_bounds = np.array(self.start_bounds)
        self.end_bounds = np.array(self.end_bounds)
        self._max_den = np.max(self.densities)
        self._integrate()

    def set_length(self, new_length_cm):
        """Updates the total length of the target.

        Usually the length is set
        """
        if new_length_cm < self.mat_list[-1][0]:
            raise Exception(
                "GeneralizedTarget::set_length(): "
                + "can not set length below lower boundary of last "
                + "material."
            )
        self.len_target = new_length_cm
        self.mat_list[-1][1] = new_length_cm
        self._update_variables()

    def add_material(self, start_position_cm, density, name):
        """Adds one additional material to a composite target.

        Args:
           start_position_cm (float):  position where the material starts
                                       counted from target origin l|X = 0 in cm
           density (float):  density of material in g/cm**3
           name (str):  any user defined name

        Raises:
            Exception: If requested start_position_cm is not properly defined.
        """

        if start_position_cm < 0.0 or start_position_cm > self.len_target:
            raise Exception(
                "GeneralizedTarget::add_material(): "
                + "distance exceeds target dimensions."
            )
        if (
            start_position_cm == self.mat_list[-1][0]
            and self.mat_list[-1][-1] == self.env_name
        ):
            self.mat_list[-1] = [start_position_cm, self.len_target, density, name]

        elif start_position_cm <= self.mat_list[-1][0]:
            raise Exception(
                "GeneralizedTarget::add_material(): "
                + "start_position_cm is ahead of previous material."
            )

        else:
            self.mat_list[-1][1] = start_position_cm
            self.mat_list.append([start_position_cm, self.len_target, density, name])

        info(
            2,
            (
                "{0}::add_material(): Material '{1}' added. "
                + "location on path {2} to {3} m"
            ).format(
                self.__class__.__name__,
                name,
                self.mat_list[-1][0],
                self.mat_list[-1][1],
            ),
        )

        self._update_variables()

    def set_theta(self, *args):
        """This method is not defined for the generalized target. The purpose
        is to catch usage errors.

        Raises:
            NotImplementedError: always
        """

        raise NotImplementedError(
            "GeneralizedTarget::set_theta(): Method"
            + "not defined for this target class."
        )

    def _integrate(self):
        """Walks through material list and computes the depth along the
        position (path). Computes the spline for the position-depth relation
        and determines the maximum depth for the material selection.

        Method does not need to be called by the user, instead the class
        calls it when necessary.
        """

        from scipy.interpolate import UnivariateSpline

        self.density_depth = None
        self.knots = [0.0]
        self.X_int = [0.0]

        for start, end, density, _ in self.mat_list:
            self.knots.append(end)
            self.X_int.append(density * (end - start) + self.X_int[-1])

        self._s_X2h = UnivariateSpline(self.X_int, self.knots, k=1, s=0.0)
        self._s_h2X = UnivariateSpline(self.knots, self.X_int, k=1, s=0.0)
        self._max_X = self.X_int[-1]

    @property
    def s_X2h(self):
        """Spline for depth at distance."""
        if not hasattr(self, "_s_X2h"):
            self._integrate()
        return self._s_X2h

    @property
    def s_h2X(self):
        """Spline for distance at depth."""
        if not hasattr(self, "_s_h2X"):
            self._integrate()
        return self._s_h2X

    @property
    def max_X(self):
        """Maximal depth of target."""
        if not hasattr(self, "_max_X"):
            self._integrate()
        return self._max_X

    def get_density_X(self, X):
        """Returns the density in g/cm**3 as a function of depth X.

        Args:
           X (float):  depth in g/cm**2

        Returns:
           float: density in g/cm**3

        Raises:
            Exception: If requested depth exceeds target.
        """
        X = np.atleast_1d(X)
        # allow for some small constant extrapolation for odepack solvers
        if X[-1] > self.max_X and X[-1] < self.max_X * 1.003:
            X[-1] = self.max_X
        if np.min(X) < 0.0 or np.max(X) > self.max_X:
            # return self.get_density(self.s_X2h(self.max_X))
            info(
                0,
                f"Depth {np.max(X):4.3f} exceeds target dimensions {self.max_X:4.3f}",
            )
            raise Exception("Invalid input")

        return self.get_density(self.s_X2h(X))

    def r_X2rho(self, X):
        """Returns the inverse density :math:`\\frac{1}{\\rho}(X)`.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`1/\\rho` in cm**3/g

        """
        return 1.0 / self.get_density_X(X)

    def get_density(self, l_cm):
        """Returns the density in g/cm**3 as a function of position l in cm.

        Args:
           l (float):  position in target in cm

        Returns:
           float: density in g/cm**3

        Raises:
            Exception: If requested position exceeds target length.
        """
        l_cm = np.atleast_1d(l_cm)
        res = np.zeros_like(l_cm)

        if np.min(l_cm) < 0 or np.max(l_cm) > self.len_target:
            raise Exception(
                "GeneralizedTarget::get_density(): "
                + "requested position exceeds target legth."
            )
        for i, li in enumerate(l_cm):
            bi = 0
            while not (li >= self.start_bounds[bi] and li <= self.end_bounds[bi]):
                bi += 1
            res[i] = self.densities[bi]
        return res

    def draw_materials(self, axes=None, logx=False):
        """Makes a plot of depth and density profile as a function
        of the target length. The list of materials is printed out, too.

        Args:
           axes (plt.axes, optional):  handle for matplotlib axes
        """
        import matplotlib.pyplot as plt

        if not axes:
            plt.figure(figsize=(5, 2.5))
            axes = plt.gca()
        ymax = np.max(self.X_int) * 1.01
        for _, mat in enumerate(self.mat_list):
            xstart = mat[0]
            xend = mat[1]
            alpha = 0.188 * mat[2] / max(self.densities) + 0.248
            if alpha > 1:
                alpha = 1.0
            elif alpha < 0.0:
                alpha = 0.0
            axes.fill_between(
                (xstart, xend),
                (ymax, ymax),
                (0.0, 0.0),
                label=mat[2],
                facecolor="grey",
                alpha=alpha,
            )
            # axes.text(0.5e-2 * (xstart + xend), 0.5 * ymax, str(nm))

        axes.plot([xl for xl in self.knots], self.X_int, lw=1.7, color="r")

        if logx:
            axes.set_xscale("log", nonposx="clip")

        axes.set_ylim(0.0, ymax)
        axes.set_xlabel("distance in target (cm)")
        axes.set_ylabel(r"depth X (g/cm$^2)$")

        self.print_table(min_dbg_lev=2)

    def print_table(self, min_dbg_lev=0):
        """Prints table of materials to standard output."""

        templ = "{0:^3} | {1:15} | {2:^9.3g}  | {3:^9.3g} | {4:^8.5g}"
        info(
            min_dbg_lev,
            "********************* List of materials ***********************",
            no_caller=True,
        )
        head = "{0:3} | {1:15} | {2:9} | {3:9} | {4:9}".format(
            "no", "name", "start [cm]", "end [cm]", "density [g/cm**3]"
        )
        info(min_dbg_lev, "-" * len(head), no_caller=True)
        info(min_dbg_lev, head, no_caller=True)
        info(min_dbg_lev, "-" * len(head), no_caller=True)
        for nm, mat in enumerate(self.mat_list):
            info(
                min_dbg_lev,
                templ.format(nm, mat[3], mat[0], mat[1], mat[2]),
                no_caller=True,
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.title("CORSIKA atmospheres")
    cka_atmospheres = [
        ("USStd", None),
        ("BK_USStd", None),
        ("Karlsruhe", None),
        ("ANTARES/KM3NeT-ORCA", "Summer"),
        ("ANTARES/KM3NeT-ORCA", "Winter"),
        ("KM3NeT-ARCA", "Summer"),
        ("KM3NeT-ARCA", "Winter"),
        ("KM3NeT", None),
        ("SouthPole", "December"),
        ("PL_SouthPole", "January"),
        ("PL_SouthPole", "August"),
        ("SDR_SouthPole", "April"),
    ]
    cka_surf_100 = []
    for loc, season in cka_atmospheres:
        cka_obj = CorsikaAtmosphere(loc, season)
        cka_obj.set_theta(0.0)
        x_vec = np.linspace(0, cka_obj.max_X, 5000)
        plt.plot(
            x_vec,
            1 / cka_obj.r_X2rho(x_vec),
            lw=1.5,
            label=(f"{loc}/{season}" if season is not None else f"{loc}"),
        )
        cka_surf_100.append((cka_obj.max_X, 1.0 / cka_obj.r_X2rho(100.0)))
    print(cka_surf_100)
    plt.ylabel(r"Density $\rho$ (g/cm$^3$)")
    plt.xlabel(r"Depth (g/cm$^2$)")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.figure(figsize=(5, 4))
    plt.title("NRLMSISE-00 atmospheres")
    msis_atmospheres = [
        ("SouthPole", "January"),
        ("Karlsruhe", "January"),
        ("Geneva", "January"),
        ("Tokyo", "January"),
        ("SanGrasso", "January"),
        ("TelAviv", "January"),
        ("KSC", "January"),
        ("SoudanMine", "January"),
        ("Tsukuba", "January"),
        ("LynnLake", "January"),
        ("PeaceRiver", "January"),
        ("FtSumner", "January"),
    ]
    msis_surf_100 = []
    for loc, season in msis_atmospheres:
        msis_obj = MSIS00Atmosphere(loc, season)
        msis_obj.set_theta(0.0)
        x_vec = np.linspace(0, msis_obj.max_X, 5000)
        plt.plot(x_vec, 1 / msis_obj.r_X2rho(x_vec), lw=1.5, label=f"{loc}")
        msis_surf_100.append((msis_obj.max_X, 1.0 / msis_obj.r_X2rho(100.0)))
    print(msis_surf_100)
    plt.ylabel(r"Density $\rho$ (g/cm$^3$)")
    plt.xlabel(r"Depth (g/cm$^2$)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
