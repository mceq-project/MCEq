from abc import ABCMeta, abstractmethod

import numpy as np
from six import with_metaclass

from MCEq import config

# Import the new atmosphere data module
from MCEq.geometry.atmosphere_parameters import (
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)
from MCEq.misc import info


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

    #: If True, subclasses manage their own :attr:`max_theta` and
    #: :meth:`set_h_obs` must not overwrite it when the observation level
    #: changes.  Set this on any detector-centred model that allows
    #: upgoing angles (> 90°) so that updating the observation level does
    #: not silently reset the allowed zenith-angle range.
    _preserve_max_theta: bool = False

    #: If True, the density profile depends on the azimuth angle and
    #: :meth:`set_theta` accepts an ``azimuth_deg`` argument (detector-
    #: centred models that bind the shower impact point to a geographic
    #: location).  Batched solvers use this flag to share integration
    #: paths across azimuth pixels at fixed zenith when it is False.
    depends_on_azimuth: bool = False

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

    def _assert_finite_density(self, rho_vec, h_vec_cm):
        """Raises if any sampled density is not finite.

        A backend that has no data for part of the integration path (a
        tabulated profile that stops below the top of the atmosphere, say)
        returns ``nan`` there.  Integrating that silently produces ``nan``
        splines much later, so the failure is reported here, where the
        offending heights are still known.

        Args:
          rho_vec (numpy.ndarray): sampled densities in g/cm**3
          h_vec_cm (numpy.ndarray): the heights they were sampled at, in cm

        Raises:
            ValueError: if any entry of *rho_vec* is not finite.
        """
        bad = ~np.isfinite(rho_vec)
        if not np.any(bad):
            return
        h_bad = np.atleast_1d(h_vec_cm)[bad]
        raise ValueError(
            f"{self.__class__.__name__}::calculate_density_spline(): "
            f"{np.count_nonzero(bad)} of {len(rho_vec)} density samples are "
            f"not finite at zenith {self.theta_deg:.1f} deg, for heights "
            f"{h_bad.min():.4e} - {h_bad.max():.4e} cm. The density model does "
            "not cover the whole integration path."
        )

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

        # Compute density at every step once to avoid calling vec_rho_l twice
        rho_vec = vec_rho_l(dl_vec)
        self._assert_finite_density(rho_vec, self.geom.h(dl_vec, thrad))

        # Calculate integral for each depth point
        X_int = cumulative_trapezoid(rho_vec, dl_vec)
        dl_vec = dl_vec[1:]

        info(5, f".. took {time() - now:1.2f}s")

        # Save depth value at h_obs
        self._max_X = X_int[-1]
        self._max_den = self.get_density(self.geom.h(0, thrad))

        # Store minimum valid slant depth for the integration path.  The
        # spline below is only fitted for X >= X_int[0]; starting the
        # numerical integration from X_int[0] avoids evaluating r_X2rho
        # outside the fitted domain, which can return non-physical (zero or
        # negative) values due to quadratic spline extrapolation and cause an
        # infinite loop in _calculate_integration_path.
        self._min_X = X_int[0]

        # Interpolate with bi-splines without smoothing
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[1:])]
        X_intp = [X for X in reversed(X_int[1:])]
        # This is an incomplete workaround for non-monothonic elevations for
        # upgoing trajectories.
        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
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

        Normalised to the density at the observation level.  Models whose
        profile does not reach sea level (an elevated site, or a tabulated
        atmosphere that starts at the surface) have no density at h=0, so
        anchoring this on the observation level is what keeps the value finite.
        For the default h_obs=0 this is the sea-level normalisation it has
        always been.
        """

        return 0.000283 * self.get_density(h_cm) / self.get_density(self.geom.h_obs)

    def gamma_cherenkov_air(self, h_cm):
        """Returns the Lorentz factor gamma of Cherenkov threshold in air (MeV)."""

        nrel = self.nref_rel_air(h_cm)
        return (1.0 + nrel) / np.sqrt(2.0 * nrel + nrel**2)

    def theta_cherenkov_air(self, h_cm):
        """Returns the Cherenkov angle in air (degrees)."""

        return np.arccos(1.0 / (1.0 + self.nref_rel_air(h_cm))) * 180.0 / np.pi

    @property
    def current_impact_latitude(self):
        """Geographic latitude of the shower impact point in degrees.

        Returns ``None`` for models that do not couple the atmosphere to a
        detector position.  Subclasses such as :class:`MSIS00LocationCentered`
        override this to return the actual impact latitude for the currently
        configured zenith/azimuth angle (or ``None`` when azimuth-averaging
        is active).  Downstream code (e.g. geomagnetic cutoff calculations)
        can query this on any atmosphere object without knowing its type.
        """
        return None

    @property
    def current_impact_longitude(self):
        """Geographic longitude of the shower impact point in degrees.

        Returns ``None`` for models that do not couple the atmosphere to a
        detector position.  See :attr:`current_impact_latitude` for details.
        """
        return None


class CorsikaAtmosphere(EarthsAtmosphere):
    """Class, holding the parameters of a Linsley type parameterization
    similar to the Air-Shower Monte Carlo
    `CORSIKA <https://web.ikp.kit.edu/corsika/>`_.

    The parameters pre-defined parameters are taken from the CORSIKA
    manual. If new sets of parameters are added to :func:`init_parameters`,
    the array _thickl can be calculated using :func:`calc_thickl` .

    Attributes:
      _atm_param (:func:`numpy.array`): (5x5) Stores 5 atmospheric parameters
                                _aatm, _batm, _catm, _thickl, _hlay
                                for each of the 5 layers

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

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
        import MCEq.geometry.corsikaatm as corsika_acc

        # Assuming corsika_acc is defined elsewhere or needs to be imported
        self.corsika_acc = corsika_acc
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
      season (str,optional): months (January, February, etc.)
      doy (int, optional): day of year
    """

    def __init__(self, location, season=None, doy=None, use_loc_altitudes=False):
        from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

        msis_atmospheres = [
            "SouthPole",
            "Karlsruhe",
            "Geneva",
            "Tokyo",
            "GranSasso",
            "TelAviv",
            "KSC",
            "SoudanMine",
            "Tsukuba",
            "LynnLake",
            "PeaceRiver",
            "FtSumner",
        ]
        assert location in msis_atmospheres, (
            f"{location} not available for MSIS00Atmosphere"
        )

        self._msis = cNRLMSISE00()

        # Base class first: it creates self.geom, which init_parameters needs
        # when ``use_loc_altitudes`` moves the observation level.  Calling it
        # afterwards made ``use_loc_altitudes=True`` raise AttributeError.
        EarthsAtmosphere.__init__(self)

        self.init_parameters(location, season, doy, use_loc_altitudes)

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
            # set_h_obs, not a bare attribute write: the geometry caches
            # r_obs and the maximal zenith angle off the observation level.
            self.geom.set_h_obs(self._msis.alt_surface)

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


class MSIS00LocationCentered(MSIS00Atmosphere):
    """MSIS atmosphere model coupled to an arbitrary detector location.

    This is the general base class for detector-centred atmosphere models.
    It computes the geographic coordinates (latitude, longitude) of the
    shower impact point on Earth's surface for a given zenith and azimuth
    angle using a spherical-Earth ECEF (Earth-Centred, Earth-Fixed) geometry,
    then feeds those coordinates to the NRLMSISE-00 empirical atmosphere model
    so that the density profile reflects the actual atmosphere above the
    shower column.

    Azimuth convention: 0° = geographic North, 90° = East (meteorological /
    clockwise from North).  Zenith: 0° = directly above detector, 90° =
    horizontal, > 90° = upgoing (shower source below horizon) provided
    *max_theta* is set to 180.

    When *set_theta* is called **without** an azimuth angle the model
    computes an azimuth-averaged density profile by sampling *n_azimuth*
    equally-spaced directions around the compass and averaging the MSIS
    density at every height step.  This provides a single representative
    profile that accounts for the latitudinal variation around the detector.

    Subclasses (e.g. :class:`MSIS00IceCubeCentered`,
    :class:`MSIS00KM3NeTCentered`) specialise this for concrete detector
    sites.

    The zenith angle is interpreted as the angle *at the detector*.  Since
    the detector sits below (or above, for elevated sites) the local
    surface, the local zenith angle where the shower axis crosses the
    surface differs from the detector-frame angle.  For a straight line on
    a spherical Earth the impact parameter :math:`r\\,\\sin\\theta` is
    conserved, so the surface-frame angle used for the atmospheric column
    is :math:`\\theta_s = \\arcsin((r_{det}/r_{surf})\\sin\\theta)`.  The
    difference is negligible below ~80° but reaches ~1.4° at the horizon
    for a detector 2 km below the surface (a ~60% slant-depth effect), and
    it maps the grazing window just below the geometric horizon
    (:math:`90° < \\theta \\le 90° + \\arccos(r_{det}/r_{surf})`) onto
    near-side downgoing columns instead of mis-classifying it as upgoing.

    Args:
        detector_coord (tuple): ``(longitude, latitude)`` of the detector
            in degrees.  Longitude in (−180, 180], latitude in [−90, 90].
        depth_m (float): Depth of the detector below the local surface in
            metres (positive value = below surface).
        season (str, optional): Month name (e.g. ``"January"``).  If both
            *season* and *doy* are ``None`` the MSIS default day is used.
        doy (int, optional): Day of year (1–365).  Takes precedence over
            *season* when both are supplied.
        n_azimuth (int): Number of azimuth samples used for azimuth-
            averaging (default 36, i.e. every 10°).
        max_theta (float): Maximum allowed zenith angle in degrees.
            Use 90.0 (default) for downgoing-only models, 180.0 to also
            accept grazing and upgoing angles.
        surface_elevation_m (float): Elevation of the local surface above
            sea level in metres (default 0).  For detectors buried in an
            elevated ice sheet (e.g. IceCube: glacier top at ~2835 m,
            detector 1948 m below it) the shower impacts the glacier
            surface, and the atmospheric column ends there.  The
            observation level of the geometry is set to this elevation for
            downgoing/grazing showers and to sea level for upgoing showers
            (whose column ends on the far side of the Earth).
    """

    #: Preserve max_theta across set_h_obs calls (see EarthsAtmosphere).
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
        from MCEq.geometry.nrlmsise00_mceq import cNRLMSISE00

        longitude, latitude = detector_coord

        # Bypass MSIS00Atmosphere.__init__ (which requires a named location
        # string) and set up the C library object directly via coordinates.
        self._msis = cNRLMSISE00()
        self._msis.set_location_coord(longitude, latitude)
        if season is not None:
            self._msis.set_season(season)
        elif doy is not None:
            self._msis.set_doy(doy)

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

        # Initialise the base class (sets geom, thrad, theta_deg, …)
        EarthsAtmosphere.__init__(self)
        if surface_elevation_m != 0.0:
            self.geom.set_h_obs(surface_elevation_m * 1e2)
        self.max_theta = max_theta
        self.location = f"({longitude:.3f}\u00b0E, {latitude:.3f}\u00b0N)"
        self.season = season

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

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

        At South Pole and zenith ≤ 90° this formula is algebraically
        equivalent to the original 2-D formula in the legacy
        :class:`MSIS00IceCubeCentered`.

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

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    def get_density(self, h_cm):
        """Return air density in g/cm³ at height *h_cm*.

        In single-azimuth mode (azimuth was passed to :meth:`set_theta`)
        delegates directly to the MSIS C library whose location is already
        set to the impact point.

        In azimuth-averaging mode (no azimuth passed) computes the mean
        density over the pre-computed :attr:`_azimuth_avg_coords` impact
        points.  The impact points are fixed for a given zenith angle and
        are cached by :meth:`set_theta`, so this method only iterates over
        the cached coordinates without recomputing them.
        """
        if self._azimuth_averaging:
            total = 0.0
            for lat, lon in self._azimuth_avg_coords:
                self._msis.set_location_coord(lon, lat)
                total += self._msis.get_density(h_cm)
            return total / len(self._azimuth_avg_coords)
        return self._msis.get_density(h_cm)

    # ------------------------------------------------------------------
    # Density spline (azimuth-averaging optimisation)
    # ------------------------------------------------------------------

    def calculate_density_spline(self, n_steps=2000):
        """Calculate and store a spline of :math:`\\rho(X)`.

        In single-azimuth mode delegates to the base-class implementation.

        In azimuth-averaging mode uses an azimuth-major loop for efficiency:
        for each of the :attr:`_n_azimuth` pre-computed impact points the
        MSIS location is set **once** and the full height profile is sampled
        in a single vectorised call, so the C library benefits from staying
        at the same latitude/longitude across all height steps.  The 36
        per-azimuth profiles are averaged before spline fitting.
        """
        if not self._azimuth_averaging:
            super().calculate_density_spline(n_steps)
            return

        from time import time

        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import UnivariateSpline

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0, path_length, n_steps)
        h_vec = np.array([self.geom.h(dl, thrad) for dl in dl_vec])

        info(
            5,
            f"Calculating azimuth-averaged spline for zenith {self.theta_deg:4.1f}\u00b0"
            f" ({len(self._azimuth_avg_coords)} directions).",
        )
        now = time()

        # Azimuth-major loop: set MSIS location once per direction, then
        # vectorise over all heights.  This is far more efficient than the
        # height-major order because the C library stays at the same
        # lat/lon across all altitude queries.
        rho_sum = np.zeros(n_steps)
        msis_vec = np.vectorize(self._msis.get_density)
        for lat, lon in self._azimuth_avg_coords:
            self._msis.set_location_coord(lon, lat)
            rho_sum += msis_vec(h_vec)
        rho_vec = rho_sum / len(self._azimuth_avg_coords)

        info(5, f".. took {time() - now:1.2f}s")

        X_int = cumulative_trapezoid(rho_vec, dl_vec)
        dl_vec = dl_vec[1:]

        self._max_X = X_int[-1]
        self._min_X = X_int[0]
        self._max_den = float(rho_vec[0])

        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[1:])]
        X_intp = [X for X in reversed(X_int[1:])]
        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1], k=2, s=0.0)

    # ------------------------------------------------------------------
    # Angle setting
    # ------------------------------------------------------------------

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
            self._msis.set_location_coord(lon, lat)
            self._current_impact_latitude = lat
            self._current_impact_longitude = lon
            self._azimuth_averaging = False
            self._azimuth_avg_coords = []
            info(
                1,
                f"zenith={theta_deg:.1f}\u00b0, azimuth={azimuth_deg:.1f}\u00b0"
                f" \u2192 impact lat={lat:.2f}\u00b0, lon={lon:.2f}\u00b0,"
                f" local zenith={effective_theta:.2f}\u00b0",
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

    # ------------------------------------------------------------------
    # Impact coordinate properties
    # ------------------------------------------------------------------

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


class MSIS00IceCubeCentered(MSIS00LocationCentered):
    """Atmosphere model centred on the IceCube detector at South Pole.

    Specialisation of :class:`MSIS00LocationCentered` for IceCube.  The
    detector sits 1948 m below the top of the Antarctic ice sheet, whose
    surface at the South Pole is at ~2835 m elevation — the detector
    centre is therefore ~887 m *above* sea level, and downgoing showers
    impact the glacier surface at 2835 m.  Upgoing angles up to 180° are
    supported.

    The public interface is identical to the original implementation so
    that existing code continues to work unchanged.  The ``location``
    argument is accepted for backward compatibility but is always
    overridden to ``"SouthPole"``.

    Args:
      location (str): Ignored (kept for backward compatibility).
      season (str): Month name, e.g. ``"January"``.
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
        """Return the geographic latitude of the shower impact point.

        Backward-compatible wrapper around :meth:`_impact_point`.
        The azimuth is irrelevant at the South Pole (all directions are
        equivalent), so 0° is passed.

        Args:
          det_zenith_deg (float): zenith angle at detector in degrees

        Returns:
          float: latitude of the impact point in degrees
        """
        lat, _lon = self._impact_point(det_zenith_deg, 0.0)
        return lat


# ---------------------------------------------------------------------------
# KM3NeT detector coordinates (approximate positions)
# ---------------------------------------------------------------------------
_KM3NET_DETECTORS = {
    # ORCA: offshore Toulon (France), ~2450 m depth
    "ORCA": {"longitude": 6.033, "latitude": 42.803, "depth_m": 2450.0},
    # ARCA: offshore Capo Passero (Sicily, Italy), 36°16'N 16°06'E, ~3500 m depth
    "ARCA": {"longitude": 16.1, "latitude": 36.267, "depth_m": 3500.0},
}


class MSIS00KM3NeTCentered(MSIS00LocationCentered):
    """MSIS atmosphere model coupled to a KM3NeT detector location.

    Convenience subclass of :class:`MSIS00LocationCentered` for the two
    KM3NeT deep-sea neutrino telescope sites in the Mediterranean:

    * **ORCA** (Oscillation Research with Cosmics in the Abyss) —
      offshore Toulon, France (6.033°E, 42.803°N, ~2450 m depth).
    * **ARCA** (Astroparticle Research with Cosmics in the Abyss) —
      offshore Capo Passero, Sicily (16.1°E, 36.267°N, ~3500 m depth).

    Both sites are at non-polar latitudes so the azimuth angle matters:
    the same zenith angle can probe very different atmospheric columns
    depending on direction.  Upgoing neutrino angles up to 180° are
    supported (``max_theta=180``).

    When *set_theta* is called without an azimuth, the density profile is
    averaged over *n_azimuth* equally-spaced azimuth directions (default
    36), providing an isotropically-averaged column for the given zenith.

    Args:
        detector (str): ``"ORCA"`` or ``"ARCA"``.
        season (str, optional): Month name (e.g. ``"January"``).
        doy (int, optional): Day of year (1–365).
        n_azimuth (int): Azimuth steps for averaging (default 36).
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


#: Molar mass of dry air in g/mol.
_M_AIR = 28.964
#: Ideal gas constant in hPa cm**3 / (K mol).
_R_GAS = 8.314e4
#: Padding added to both ends of a tabulated profile in cm.  Purely to absorb
#: floating-point error at the exact integration limits (``h_obs`` and
#: ``h_atm``); 1 m is physically negligible.
_TABLE_PAD_CM = 100.0
#: Number of table bins blended into MSIS00 by ``top_extension="msis00"``.
_MSIS_BLEND_BINS = 5


def load_atmosphere_table(filename):
    """Reads a tabulated atmosphere from a CSV file.

    The format is deliberately minimal so that any data source -- ERA5, AIRS,
    GDAS, radiosondes -- can be converted to it with a few lines of code.  See
    :class:`TabulatedAtmosphere` for the full specification and
    ``docs/examples/ERA5_density_spline.ipynb`` for a worked converter.

    Args:
      filename (str): path to the CSV file

    Returns:
      dict: ``{"h_cm": ndarray, "rho_gcm3": ndarray}``, plus ``"T_K"`` when the
      file provides a temperature column.  Rows are sorted by height and rows
      with missing height or density are dropped.
    """
    # Strip full-line comments and blanks first: numpy's ``names=True`` always
    # takes the *first* line as the header, comment marker or not.
    with open(filename) as table_file:
        rows = [
            line
            for line in table_file
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if len(rows) < 2:
        raise ValueError(
            f"load_atmosphere_table(): {filename} holds {max(len(rows) - 1, 0)} "
            "data row(s). A header line naming the columns plus at least two "
            "data rows are required."
        )

    raw = np.atleast_1d(np.genfromtxt(rows, delimiter=",", names=True, dtype=float))
    if raw.dtype.names is None:
        raise ValueError(
            f"load_atmosphere_table(): {filename} has no column header row. "
            "The first non-comment line must name the columns."
        )
    columns = {name.lower(): raw[name].astype(np.float64) for name in raw.dtype.names}

    if "h_cm" not in columns:
        raise ValueError(
            f"load_atmosphere_table(): {filename} is missing the required "
            f"'h_cm' column. Found: {sorted(columns)}."
        )
    h_cm = columns["h_cm"]

    if "rho_gcm3" in columns:
        dens = columns["rho_gcm3"]
    elif "t_k" in columns and "p_hpa" in columns:
        # Dry-air ideal gas law.  ERA5 provides specific humidity on the
        # model-level path; ignoring it overestimates the near-surface density
        # by at most ~1%.
        dens = columns["p_hpa"] / columns["t_k"] * _M_AIR / _R_GAS
    else:
        raise ValueError(
            f"load_atmosphere_table(): {filename} must provide either a "
            f"'rho_gcm3' column or both 'T_K' and 'p_hPa'. Found: {sorted(columns)}."
        )

    valid = np.isfinite(h_cm) & np.isfinite(dens) & (dens > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            f"load_atmosphere_table(): {filename} has "
            f"{np.count_nonzero(valid)} usable row(s); at least 2 rows with a "
            "finite height and a positive density are required."
        )

    order = np.argsort(h_cm[valid])
    table = {"h_cm": h_cm[valid][order], "rho_gcm3": dens[valid][order]}
    if "t_k" in columns:
        table["T_K"] = columns["t_k"][valid][order]

    info(
        2,
        f"loaded {len(table['h_cm'])} rows from {filename}, "
        f"h = {table['h_cm'][0]:.3e} - {table['h_cm'][-1]:.3e} cm",
    )
    return table


class TabulatedAtmosphere(EarthsAtmosphere):
    """Atmosphere interpolated from a tabulated vertical density profile.

    The table is a **CSV file describing one vertical column above one site at
    one point in time**.  It replaces the former ``AIRSAtmosphere`` and is the
    supported way to drive MCEq with measured or reanalysed atmospheric data;
    ``docs/examples/ERA5_density_spline.ipynb`` shows how to turn an ERA5
    download into such a table.

    **Table format (v1)**::

        # MCEq tabulated atmosphere v1 -- comment lines are ignored
        # South Pole, ERA5 pressure levels, 2012-01-03 12:00 UTC
        h_cm,T_K,p_hPa
        283400.0,247.1,681.2
        510000.0,231.4,500.0
        900000.0,,300.0

    * ``h_cm`` (required) -- height above sea level in cm.
    * Density, either directly as ``rho_gcm3`` in g/cm**3, or derived from
      ``T_K`` (K) and ``p_hPa`` (hPa) via the dry-air ideal gas law.
    * ``T_K`` also feeds :func:`get_temperature`.
    * Column order is irrelevant and unknown columns are ignored, so a table may
      carry extra data products alongside the ones MCEq reads.  Rows may be in
      any order; they are sorted by height on load.
    * Missing values are written as an empty field or ``nan`` and are dropped
      row-wise.

    .. note::
       **This model uses a single vertical column for every direction.**  The
       shower impact point moves away from the detector as the zenith angle
       grows (~100 km at 80 deg), and the atmosphere it develops in varies with
       that position -- which a one-column table cannot express.  The zenith
       range is therefore limited to downgoing angles, and the azimuth angle is
       ignored.  For a direction-resolved atmosphere use
       :class:`MSIS00LocationCentered` or :class:`MSIS21LocationCentered`.
       Extending the table format with ``lat_deg``/``lon_deg`` columns so the
       profile can be sampled at the impact point (and upgoing angles supported)
       is planned; see the MCEq issue tracker.

    Args:
      table (str or dict): path to a CSV file, or a table already parsed by
        :func:`load_atmosphere_table`.
      surface_elevation_m (float, optional): observation level in metres above
        sea level.  ``None`` (default) uses the lowest height in the table,
        clipped at sea level -- reanalysis products commonly extrapolate below
        ground and report negative heights.
      top_extension (str): how to continue the profile above the topmost
        tabulated point up to the top of the atmosphere.  ``"isothermal"``
        (default) appends an exponential tail whose scale height is fitted to
        the two topmost points; it is self-contained and deterministic, and
        although it drifts from a real atmosphere by a factor of a few far
        above the table, the column left up there is small (~1 g/cm**2 above
        47 km, i.e. ~0.1% of a vertical column).  ``"msis00"`` blends into
        :class:`MSIS00Atmosphere` instead, which requires *location* to be one
        of its named sites; prefer it when the table stops low or when the top
        of the profile matters.  ``"none"`` leaves the profile short, so
        :func:`get_density` returns ``nan`` above the table and building the
        density spline raises.
      location (str, optional): name of the site, for bookkeeping and for the
        ``"msis00"`` extension.
      season (str, optional): month name, for bookkeeping and for the
        ``"msis00"`` extension.
    """

    def __init__(
        self,
        table,
        surface_elevation_m=None,
        top_extension="isothermal",
        location=None,
        season=None,
    ):
        if top_extension not in ("isothermal", "msis00", "none"):
            raise ValueError(
                f"{self.__class__.__name__}(): unknown top_extension "
                f"'{top_extension}'. Choose 'isothermal', 'msis00' or 'none'."
            )

        # Base class first: it creates self.geom, which the height bookkeeping
        # below reads for h_atm and h_obs.
        EarthsAtmosphere.__init__(self)

        parsed = table if isinstance(table, dict) else load_atmosphere_table(table)
        # Copy: the profile is completed in place below, and a caller-supplied
        # table must not be modified by building an atmosphere from it.
        self.h = np.array(parsed["h_cm"], dtype=np.float64)
        self.dens = np.array(parsed["rho_gcm3"], dtype=np.float64)
        self.temp = (
            np.array(parsed["T_K"], dtype=np.float64) if "T_K" in parsed else None
        )

        self.location = location
        self.season = season
        self.top_extension = top_extension

        if surface_elevation_m is None:
            h_obs_cm = max(float(self.h[0]), 0.0)
        else:
            h_obs_cm = surface_elevation_m * 1e2

        self._extend_below(h_obs_cm - _TABLE_PAD_CM)
        self._extend_above(self.geom.h_atm + _TABLE_PAD_CM)
        self._log_dens = np.log(self.dens)

        # Cleared before set_h_obs so that it does not try to build a spline.
        self.theta_deg = None
        self.set_h_obs(h_obs_cm)

    # ------------------------------------------------------------------
    # Profile completion
    # ------------------------------------------------------------------

    def _log_linear_point(self, h_target, i_near, i_far):
        """Density at *h_target* from a log-linear fit through two table rows."""
        h_near, h_far = self.h[i_near], self.h[i_far]
        ln_near, ln_far = np.log(self.dens[i_near]), np.log(self.dens[i_far])
        slope = (ln_far - ln_near) / (h_far - h_near)
        return float(np.exp(ln_near + slope * (h_target - h_near)))

    def _extend_below(self, h_bottom):
        """Extends the profile down to *h_bottom* with a log-linear tail."""
        if self.h[0] <= h_bottom:
            return
        info(
            2,
            f"{self.__class__.__name__}: table starts at {self.h[0]:.3e} cm, "
            f"extrapolating down to {h_bottom:.3e} cm.",
        )
        rho_bottom = self._log_linear_point(h_bottom, 0, 1)
        self.h = np.hstack([h_bottom, self.h])
        self.dens = np.hstack([rho_bottom, self.dens])
        if self.temp is not None:
            self.temp = np.hstack([self.temp[0], self.temp])

    def _extend_above(self, h_top):
        """Extends the profile up to *h_top* according to *top_extension*."""
        if self.top_extension == "none" or self.h[-1] >= h_top:
            return
        if self.top_extension == "isothermal":
            # np.interp works on log(rho), so an exponential tail is exactly a
            # straight line: a single extra point reproduces it everywhere.
            if not self.dens[-2] > self.dens[-1]:
                raise ValueError(
                    f"{self.__class__.__name__}(): the two topmost table rows "
                    "do not decrease in density, so no isothermal scale height "
                    "can be fitted. Use top_extension='msis00' or 'none'."
                )
            rho_top = self._log_linear_point(h_top, -1, -2)
            self.h = np.hstack([self.h, h_top])
            self.dens = np.hstack([self.dens, rho_top])
            if self.temp is not None:
                self.temp = np.hstack([self.temp, self.temp[-1]])
            return

        # top_extension == "msis00": blend the top of the table into MSIS00 so
        # that the seam does not show up as a density step in the 40-80 km
        # region, where inclined showers develop.
        msis = MSIS00Atmosphere(self.location, self.season)
        h_extra = np.linspace(self.h[-1], h_top, 100)
        msis_dens = np.array([msis.get_density(h) for h in h_extra])
        msis_temp = np.array([msis.get_temperature(h) for h in h_extra])

        n_blend = min(_MSIS_BLEND_BINS, len(self.h) - 1)
        for i in range(n_blend):
            w_tab = 1.0 - np.exp(-n_blend + i + 1)
            w_msis = 1.0 - np.exp(-i)
            norm = 1.0 / (w_tab + w_msis)
            self.dens[-i - 1] = (
                self.dens[-i - 1] * w_tab + msis.get_density(self.h[-i - 1]) * w_msis
            ) * norm
            if self.temp is not None:
                self.temp[-i - 1] = (
                    self.temp[-i - 1] * w_tab
                    + msis.get_temperature(self.h[-i - 1]) * w_msis
                ) * norm

        self.h = np.hstack([self.h[:-1], h_extra])
        self.dens = np.hstack([self.dens[:-1], msis_dens])
        if self.temp is not None:
            self.temp = np.hstack([self.temp[:-1], msis_temp])

    # ------------------------------------------------------------------
    # Density and temperature
    # ------------------------------------------------------------------

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Log-linear interpolation of the tabulated profile.  Heights outside the
        tabulated range return ``nan``; with the default *top_extension* the
        range covers the whole integration path.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return np.exp(
            np.interp(h_cm, self.h, self._log_dens, left=np.nan, right=np.nan)
        )

    def get_temperature(self, h_cm):
        """Returns the temperature in K.

        Args:
          h_cm (float): height in cm

        Returns:
          float: temperature :math:`T(h_{cm})` in K
        """
        if self.temp is None:
            raise ValueError(
                f"{self.__class__.__name__}::get_temperature(): the table has "
                "no 'T_K' column."
            )
        return np.interp(h_cm, self.h, self.temp, left=np.nan, right=np.nan)

    # ------------------------------------------------------------------
    # Density spline (vectorised)
    # ------------------------------------------------------------------

    def calculate_density_spline(self, n_steps=2000):
        """Calculates and stores a spline of :math:`\\rho(X)`.

        Overrides the base implementation to evaluate the whole altitude grid in
        one interpolation instead of looping :func:`get_density` through
        ``np.vectorize``.  The spline node bookkeeping is identical.

        Args:
          n_steps (int, optional): number of :math:`X` values to use for
            interpolation
        """
        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import UnivariateSpline

        if self.theta_deg is None:
            raise Exception("zenith angle not set")

        thrad = self.thrad
        dl_vec = np.linspace(0.0, self.geom.path_len(thrad), n_steps)
        h_vec_cm = self.geom.h(dl_vec, thrad)
        rho_vec = self.get_density(h_vec_cm)
        self._assert_finite_density(rho_vec, h_vec_cm)

        X_int = cumulative_trapezoid(rho_vec, dl_vec)

        self._max_X = X_int[-1]
        self._min_X = X_int[0]
        self._max_den = float(rho_vec[0])

        # Same node contract as the base class: X_int[i] pairs with dl_vec[i+1],
        # and the splines are fitted on the reversed (descending-height) arrays.
        h_intp = h_vec_cm[2:][::-1]
        X_intp = X_int[1:][::-1]
        self._s_h2X = UnivariateSpline(h_intp, np.log(X_intp), k=2, s=0.0)
        self._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
        self._s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1], k=2, s=0.0)


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
        """Updates the total length of the target."""
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
                "GeneralizedTarget::set_length(): "
                + "can not set length below lower boundary of last "
                + "material."
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
            ("Material '{0}' added between {1:4.1f} and {2:4.1f} m").format(
                name, self.mat_list[-1][0] / 1e2, self.mat_list[-1][1] / 1e2
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
            Exception: If requested position exceeds target.
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
            Exception: If requested depth exceeds target.
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

        res = res.item() if res.size == 1 else res

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
        ("GranSasso", "January"),
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
    plt.ylabel(r"Density $\rho$ (g/cm$^3$)")
    plt.xlabel(r"Depth (g/cm$^2$)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NRLMSIS 2.1 atmosphere models (see geometry/msis21_atmosphere.py).
# Re-exported here so users can do
#     from MCEq.geometry.density_profiles import MSIS21Atmosphere, ...
# alongside MSIS00Atmosphere and friends.  The classes live in a separate
# module to keep this file from growing further; their public API mirrors
# the MSIS00 hierarchy 1:1.
#
# The re-export is lazy (PEP 562): msis21_atmosphere imports names from
# *this* module, so an eager import here makes the cycle order-dependent —
# importing MCEq.geometry.msis21_atmosphere first would fail on a
# partially initialised module.  Resolving on first attribute access keeps
# both import orders working.
# ---------------------------------------------------------------------------
_MSIS21_EXPORTS = (
    "MSIS21Atmosphere",
    "MSIS21LocationCentered",
    "MSIS21IceCubeCentered",
    "MSIS21KM3NeTCentered",
)


def __getattr__(name):
    if name in _MSIS21_EXPORTS:
        from MCEq.geometry import msis21_atmosphere

        return getattr(msis21_atmosphere, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_MSIS21_EXPORTS])
