"""The abstract atmosphere: column depth, splines and the zenith contract.

:class:`EarthsAtmosphere` is what every density model in this package derives
from. It owns the path integration that turns a density profile into the
depth/density splines the solver reads, and delegates the actual air density to
a subclass through :meth:`get_density`.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from MCEq.environment.column import fit_column_splines
from MCEq.misc import info


def _default_environment_group():
    """The live ``config.environment`` view, fetched dynamically.

    The no-argument construction path (a user-built
    ``CorsikaAtmosphere()``, the tutorial's bare model) keeps reading
    process-wide settings, but this layer must not carry a static import
    edge to :mod:`MCEq.config` (contract C5). The device is the one
    :func:`MCEq.misc._views` uses. Driver-built atmospheres never come
    here: ``set_density_model`` injects the run's environment group.
    """
    from importlib import import_module

    return import_module("MCEq.config").environment


class EarthsAtmosphere(metaclass=ABCMeta):
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
                     defined in the :mod:`MCEq.environment.geometry`
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
        environment = kwargs.pop("environment", None)
        if environment is None:
            environment = _default_environment_group()
        geometry = kwargs.pop("geometry", None)
        if geometry is None:
            from MCEq.environment.geometry import EarthGeometry

            geometry = EarthGeometry(
                r_E=environment.r_E,
                h_atm=environment.h_atm,
                h_obs=environment.h_obs,
            )
        self.geom = geometry
        self.thrad = None
        self.theta_deg = None
        # Configured surface density, so ``max_den`` reads before the first
        # ``set_theta``; that call overwrites it with ``get_density(geom.h(0,
        # thrad))``, the atmosphere top -- the path minimum, not the maximum.
        self._max_den = environment.max_density
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

        info(5, f".. took {time() - now:1.2f}s")

        # ``dl = 0`` is the atmosphere top, so this is the path minimum.
        max_den = self.get_density(self.geom.h(0, thrad))

        # One scalar ``geom.h`` call per sample. The MSIS21 overrides pass the
        # heights of a single array call instead, and the two spellings differ
        # by one ULP of ``r_E`` at 258 of 901 zeniths -- see
        # ``tests/geometry/test_environment_pins.py``. Which one a given
        # atmosphere uses is therefore its own business, not the fit's.
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[2:])]
        fit_column_splines(self, rho_vec, dl_vec, h_intp, max_den=max_den)

    @property
    def max_X(self):
        """Depth at altitude 0."""
        if not hasattr(self, "_max_X"):
            self.set_theta(0)
        return self._max_X

    @property
    def max_den(self):
        """Atmosphere-top density -- the path minimum, despite the name (B29).

        Two meanings across its lifetime; see the note in :func:`__init__`.
        """
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

        Calls :func:`calculate_density_spline`, which makes :func:`r_X2rho`
        available to the core code. There is no spline cache and no
        ``use_atm_cache`` config option -- this docstring described both for
        years -- only the ``theta_deg`` short-circuit below.

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
        """

        return 0.000283 * self.get_density(h_cm) / self.get_density(0)

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
