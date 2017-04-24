# -*- coding: utf-8 -*-
"""
:mod:`MCEq.density_profiles` - models of the Earth's atmosphere
===============================================================

This module includes classes and functions modeling the Earth's atmosphere.
Currently, two different types models are supported:

- Linsley-type/CORSIKA-style parameterization
- Numerical atmosphere via external routine (NRLMSISE-00)

Both implementations have to inherit from the abstract class
:class:`EarthAtmosphere`, which provides the functions for other parts of
the program. In particular the function :func:`EarthAtmosphere.get_density`

Typical interaction::

      $ atm_object = CorsikaAtmosphere("BK_USStd")
      $ atm_object.set_theta(90)
      $ print 'density at X=100', atm_object.X2rho(100.)

The class :class:`MCEqRun` will only the following routines::
    - :func:`EarthAtmosphere.set_theta`,
    - :func:`EarthAtmosphere.r_X2rho`.

If you are extending this module make sure to provide these
functions without breaking compatibility.

Example:
  An example can be run by executing the module::

      $ python MCEq/atmospheres.py
"""

import numpy as np
import geometry
from numba import jit, double  # @UnresolvedImport
from os.path import join
from abc import ABCMeta, abstractmethod
from mceq_config import dbg, config


def _load_cache():
    """Loads atmosphere cache from file.

    If file does not exist, function returns
    a new empty dictionary.

    Returns:
        dict: Dictionary containing splines.

    """
    import cPickle as pickle
    if dbg > 0:
        print "atmospheres::_load_cache(): loading cache."
    fname = join(config['data_dir'],
                 config['atm_cache_file'])

    try:
        return pickle.load(open(fname, 'rb'))
    except IOError:
        print "density_profiles::_load_cache(): creating new cache.."
        return {}


def _dump_cache(cache):
    """Stores atmosphere cache to file.

    Args:
        (dict) current cache
    Raises:
        IOError:
    """
    import cPickle as pickle

    if dbg > 0:
        print "density_profiles::_dump_cache() dumping cache."
    fname = join(config['data_dir'],
                 config['atm_cache_file'])
    print fname
    try:
        pickle.dump(cache, open(fname, 'wb'), protocol=-1)
    except IOError:
        raise IOError("density_profiles::_dump_cache(): " +
                      'could not (re-)create cache. Wrong working directory?')


class GeneralizedTarget():

    len_target = config['len_target'] * 1e2  # cm
    env_density = config['env_density']  # g/cm3
    env_name = config['env_name']

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets material list to defaults."""
        self.mat_list = [[0., self.len_target,
                          self.env_density,
                          self.env_name]]
        self._update_variables()

    def _update_variables(self):
        """Updates internal variables. Not needed to call by user."""

        self.start_bounds, self.end_bounds, \
            self.densities = zip(*self.mat_list)[:-1]
        self.densities = np.array(self.densities)
        self.start_bounds = np.array(self.start_bounds)
        self.end_bounds = np.array(self.end_bounds)
        self.max_den = np.max(self.densities)
        self._integrate()

    def set_length(self, new_length_cm):
        if new_length_cm < self.mat_list[-1][0]:
            raise Exception("GeneralizedTarget::set_length(): " +
                            "can not set length below lower boundary of last " +
                            "material.")
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

        if (start_position_cm < 0. or start_position_cm > self.len_target):
            raise Exception("GeneralizedTarget::add_material(): " +
                            "distance exceeds target dimensions.")
        elif start_position_cm < self.mat_list[-1][0]:
            raise Exception("GeneralizedTarget::add_material(): " +
                            "start_position_cm is ahead of previous material.")

        self.mat_list[-1][1] = start_position_cm
        self.mat_list.append([start_position_cm,
                              self.len_target, density, name])

        if dbg > 0:
            ("{0}::add_material(): Material '{1}' added. " +
             "location on path {2} to {3} m").format(
                self.__class__.__name__, name,
                self.mat_list[-1][0], self.mat_list[-1][1])

        self._update_variables()

    def set_theta(self, *args):
        """This method is not defined for the generalized target. The purpose
        is to catch usage errors.

        Raises:
            NotImplementedError: always
        """

        raise NotImplementedError('GeneralizedTarget::set_theta(): Method'
                                  + 'not defined for this target class.')

    def _integrate(self):
        """Walks through material list and computes the depth along the
        position (path). Computes the spline for the position-depth relation
        and determines the maximum depth for the material selection.

        Method does not need to be called by the user, instead the class
        calls it when necessary.
        """

        from scipy.interpolate import UnivariateSpline
        self.density_depth = None
        self.knots = [0.]
        self.X_int = [0.]

        for start, end, density, name in self.mat_list:
            self.knots.append(end)
            self.X_int.append(density*(end-start) + self.X_int[-1])

        self.s_X2h = UnivariateSpline(self.X_int, self.knots, k=1, s=0.)
        self.s_h2X = UnivariateSpline(self.knots, self.X_int, k=1, s=0.)
        self.max_X = self.X_int[-1]

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
        #allow for some small constant extrapolation for odepack solvers
        if X[-1] > self.max_X and X[-1] < self.max_X*1.003:
            X[-1] = self.max_X
        if np.min(X) < 0. or np.max(X) > self.max_X:
            raise Exception(("GeneralizedTarget::get_density_X(): " +
                             "requested depth {0:4.3f} " +
                             "exceeds target.").format(np.max(X)))

        return self.get_density(self.s_X2h(X))

    def r_X2rho(self, X):
        """Returns the inverse density :math:`\\frac{1}{\\rho}(X)`.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`1/\\rho` in cm**3/g

        """
        return 1./self.get_density_X(X)

    def get_density(self, l_cm):
        """Returns the density in g/cm**3 as a function of position l in cm.

        Args:
           l (float):  position in target in cm

        Returns:
           float: density in g/cm**3

        Raises:
            Exception: If requested position exceeds target length.
        """
        l = np.atleast_1d(l_cm)
        res = np.zeros_like(l)

        if np.min(l) < 0 or np.max(l) > self.len_target:
            raise Exception("GeneralizedTarget::get_density(): " +
                            "requested position exceeds target legth.")
        for i in range(len(l)):
            bi = 0
            while not (l[i] >= self.start_bounds[bi] and
                       l[i] <= self.end_bounds[bi]):
                bi += 1
            res[i] = self.densities[bi]
        return res

    def draw_materials(self, axes=None):
        """Makes a plot of depth and density profile as a function
        of the target length. The list of materials is printed out, too.

        Args:
           axes (plt.axes, optional):  handle for matplotlib axes
        """
        import matplotlib.pyplot as plt

        if not axes:
            plt.figure(figsize=(5, 2.5))
            axes = plt.gca()
        ymax = np.max(self.X_int)*1.01
        for nm, mat in enumerate(self.mat_list):
            xstart = mat[0]
            xend = mat[1]
            alpha = 0.188*mat[2] + 0.248
            if alpha > 1:
                alpha = 1.
            elif alpha < 0.:
                alpha = 0.
            axes.fill_between((xstart/1e2, xend/1e2), (ymax, ymax),
                              (0., 0.), label=mat[2], facecolor='grey',
                              alpha=alpha)
            axes.text(0.5e-2*(xstart + xend), 0.5*ymax, str(nm))
        plt.plot([xl/1e2 for xl in self.knots], self.X_int, lw=1.7, color='r')
        axes.set_ylim(0., ymax)
        axes.set_xlabel('distance in target [m]')
        axes.set_ylabel(r'depth [g/cm$^2$]')
        self.print_table()

    def print_table(self):
        """Prints table of materials to standard output.
        """

        templ = '{0:^3} | {1:15} | {2:^9.3f} | {3:^9.3f} | {4:^8.5f}'
        print '********************* List of materials *************************'
        head = '{0:3} | {1:15} | {2:9} | {3:9} | {4:9}'.format(
            'no', 'name', 'start [m]', 'end [m]', 'density [g/cm**3]')
        print '-' * len(head)
        print head
        print '-' * len(head)
        for nm, mat in enumerate(self.mat_list):
            print templ.format(nm, mat[3], mat[0]/1e2, mat[1]/1e2, mat[2])


class EarthAtmosphere():
    """Abstract class containing common methods on atmosphere.
    You have to inherit from this class and implement the virtual method
    :func:`get_density`.

    Note:
      Do not instantiate this class directly.

    Attributes:
      thrad (float): current zenith angle :math:`\\theta` in radiants
      theta_deg (float): current zenith angle :math:`\\theta` in degrees
      max_X (float): Slant depth at the surface according to the geometry
                     defined in the :mod:`MCEq.geometry`

    """

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self.geom = geometry.EarthGeometry()
        self.thrad = None
        self.theta_deg = None
        self.max_X = None
        self.max_den = 1.240e-03

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
        raise NotImplementedError("EarthAtmosphere::get_density(): " +
                                  "Base class called.")

    def calculate_density_spline(self, n_steps=1000):
        """Calculates and stores a spline of :math:`\\rho(X)`.

        Args:
          n_steps (int, optional): number of :math:`X` values
                                   to use for interpolation

        Raises:
            Exception: if :func:`set_theta` was not called before.
        """
        from scipy.integrate import quad
        from time import time
        from scipy.interpolate import UnivariateSpline

        if self.theta_deg is None:
            raise Exception('{0}::calculate_density_spline(): ' +
                            'zenith angle not set'.format(
                                self.__class__.__name__))
        else:
            print ('{0}::calculate_density_spline(): ' +
                   'Calculating spline of rho(X) for zenith ' +
                   '{1} degrees.').format(self.__class__.__name__,
                                          self.theta_deg)

        thrad = self.thrad
        path_length = self.geom.l(thrad)
        vec_rho_l = np.vectorize(
            lambda delta_l: self.get_density(self.geom.h(delta_l, thrad)))
        dl_vec = np.linspace(0, path_length, n_steps)

        now = time()
        # Calculate integral for each depth point
        # functionality could be more efficient :)
        X_int = np.zeros_like(dl_vec, dtype='float64')

        X_int[0] = 0.
        for i in range(1, len(dl_vec)):
            X_int[i] = X_int[i - 1] + quad(vec_rho_l, 
                            dl_vec[i-1], dl_vec[i], 
                            epsrel=0.01)[0]

        print '.. took {0:1.2f}s'.format(time() - now)

        # Save depth value at h_obs
        self.max_X = X_int[-1]
        self.max_den = self.get_density(self.geom.h(0, thrad))

        # Interpolate with bi-splines without smoothing
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[1:])]
        X_intp = [X for X in reversed(X_int[1:])]

#        print  splrep(np.array(h_intp),
#                      np.log(X_intp),
#                      k=2, s=0.0)
        self.s_h2X = UnivariateSpline(h_intp, np.log(X_intp),
                                      k=2, s=0.0)
        self.s_X2rho = UnivariateSpline(X_int, vec_rho_l(dl_vec),
                                        k=2, s=0.0)
        # print np.log(X_intp), h_intp
        self.s_lX2h = UnivariateSpline(np.log(X_intp)[::-1], h_intp[::-1],
                                        k=2, s=0.0)

        print 'Average spline error:', np.std(vec_rho_l(dl_vec) /
                                              self.s_X2rho(X_int))

    def set_theta(self, theta_deg, force_spline_calc=False):
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
          force_spline_calc (bool): forces (re-)calculation of the
                                    spline for each call
        """
        def calculate_and_store(key, cache):
            self.thrad = self.geom._theta_rad(theta_deg)
            self.theta_deg = theta_deg
            self.calculate_density_spline()
            cache[key][theta_deg] = (self.max_X, self.s_h2X, self.s_X2rho, self.s_lX2h)
            _dump_cache(cache)

        if self.theta_deg == theta_deg and not force_spline_calc:
            print (self.__class__.__name__ +
                   '::set_theta(): Using previous' +
                   'density spline.')
            return

        elif config['use_atm_cache'] and not force_spline_calc:
            from MCEq.misc import _get_closest
            cache = _load_cache()
            key = (self.__class__.__name__, self.location, self.season)
            if cache and key in cache.keys():
                try:
                    closest = _get_closest(theta_deg, cache[key].keys())[1]
                    if abs(closest - theta_deg) < 1.:
                        self.thrad = self.geom._theta_rad(closest)
                        self.theta_deg = closest
                        self.max_X, self.s_h2X, self.s_X2rho, self.s_lX2h = cache[key][closest]
                    else:
                        calculate_and_store(key, cache)
                except:
                    cache[key] = {}
                    calculate_and_store(key, cache)

            else:
                cache[key] = {}
                calculate_and_store(key, cache)

        else:
            self.thrad = self.geom._theta_rad(theta_deg)
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
        return 1. / self.s_X2rho(X)

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
        """Returns the Moliere unit of air for US standard atmosphere. """

        return 9.3 / (self.get_density(h_cm) * 100.)

    def nref_rel_air(self, h_cm):
        """Returns the refractive index - 1 in air (density parametrization
        as in CORSIKA).
        """

        return 0.000283 * self.get_density(h_cm) / self.get_density(0)

    def gamma_cherenkov_air(self, h_cm):
        """Returns the Lorentz factor gamma of Cherenkov threshold in air (MeV).
        """

        nrel = self.nref_rel_air(h_cm)
        return (1. + nrel) / np.sqrt(2. * nrel + nrel ** 2)

    def theta_cherenkov_air(self, h_cm):
        """Returns the Cherenkov angle in air (degrees).
        """

        return np.arccos(1. / (1. + self.nref_rel_air(h_cm))) * 180. / np.pi


#=========================================================================
# CorsikaAtmosphere
#=========================================================================
class CorsikaAtmosphere(EarthAtmosphere):
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
        self.init_parameters(location, season)
        EarthAtmosphere.__init__(self)

    def init_parameters(self, location, season):
        """Initializes :attr:`_atm_param`.

        +--------------+-------------------+------------------------------+
        | location     | CORSIKA Table     | Description/season           |
        +==============+===================+==============================+
        | "USStd"      |         1         |  US Standard atmosphere      |
        +--------------+-------------------+------------------------------+
        | "BK_USStd"   |         31        |  Bianca Keilhauer's USStd    |
        +--------------+-------------------+------------------------------+
        | "Karlsruhe"  |         18        |  AT115 / Karlsruhe           |
        +--------------+-------------------+------------------------------+
        | "SouthPole"  |      26 and 28    |  MSIS-90-E for Dec and June  |
        +--------------+-------------------+------------------------------+
        |"PL_SouthPole"|      29 and 30    |  P. Lipari's  Jan and Aug    |
        +--------------+-------------------+------------------------------+


        Args:
          location (str): see table
          season (str, optional): choice of season for supported locations

        Raises:
          Exception: if parameter set not available
        """
        _aatm, _batm, _catm, _thickl, _hlay = None, None, None, None, None
        self.max_X = None
        if location == "USStd":
            _aatm = np.array([-186.5562, -94.919, 0.61289, 0.0, 0.01128292])
            _batm = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
            _catm = np.array([994186.38, 878153.55, 636143.04, 772170., 1.0e9])
            _thickl = np.array(
                [1036.102549, 631.100309, 271.700230, 3.039494, 0.001280])
            _hlay = np.array([0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
        elif location == "BK_USStd":
            _aatm = np.array(
                [-149.801663, -57.932486, 0.63631894, 4.3545369e-4, 0.01128292])
            _batm = np.array([1183.6071, 1143.0425, 1322.9748, 655.69307, 1.0])
            _catm = np.array(
                [954248.34, 800005.34, 629568.93, 737521.77, 1.0e9])
            _thickl = np.array(
                [1033.804941, 418.557770, 216.981635, 4.344861, 0.001280])
            _hlay = np.array([0.0, 7.0e5, 1.14e6, 3.7e6, 1.0e7])
        elif location == "Karlsruhe":
                _aatm = np.array(
                    [-118.1277, -154.258, 0.4191499, 5.4094056e-4, 0.01128292])
                _batm = np.array(
                    [1173.9861, 1205.7625, 1386.7807, 555.8935, 1.0])
                _catm = np.array(
                    [919546., 963267.92, 614315., 739059.6, 1.0e9])
                _thickl = np.array(
                    [1055.858707, 641.755364, 272.720974, 2.480633, 0.001280])
                _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
        elif location == 'SouthPole':
            if season == 'December':
                _aatm = np.array(
                    [-128.601, -39.5548, 1.13088, -0.00264960, 0.00192534])
                _batm = np.array([1139.99, 1073.82, 1052.96, 492.503, 1.0])
                _catm = np.array(
                    [861913., 744955., 675928., 829627., 5.8587010e9])
                _thickl = np.array(
                    [1011.398804, 588.128367, 240.955360, 3.964546, 0.000218])
                _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
            elif season == "June":
                _aatm = np.array(
                    [-163.331, -65.3713, 0.402903, -0.000479198, 0.00188667])
                _batm = np.array([1183.70, 1108.06, 1424.02, 207.595, 1.0])
                _catm = np.array(
                    [875221., 753213., 545846., 793043., 5.9787908e9])
                _thickl = np.array(
                    [1020.370363, 586.143464, 228.374393, 1.338258, 0.000214])
                _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
            else:
                raise Exception('CorsikaAtmosphere(): Season "' + season +
                                '" not parameterized for location SouthPole.')
        elif location == 'PL_SouthPole':
            if season == 'January':
                _aatm = np.array(
                    [-113.139, -7930635, -54.3888, -0.0, 0.00421033])
                _batm = np.array([1133.10, 1101.20, 1085.00, 1098.00, 1.0])
                _catm = np.array(
                    [861730., 826340., 790950., 682800., 2.6798156e9])
                _thickl = np.array(
                    [1019.966898, 718.071682, 498.659703, 340.222344, 0.000478])
                _hlay = np.array([0.0, 2.67e5, 5.33e5, 8.0e5, 1.0e7])
            elif season == "August":
                _aatm = np.array(
                    [-59.0293, -21.5794, -7.14839, 0.0, 0.000190175])
                _batm = np.array([1079.0, 1071.90, 1182.0, 1647.1, 1.0])
                _catm = np.array(
                    [764170., 699910., 635650., 551010., 59.329575e9])
                _thickl = np.array(
                    [1019.946057, 391.739652, 138.023515, 43.687992, 0.000022])
                _hlay = np.array([0.0, 6.67e5, 13.33e5, 2.0e6, 1.0e7])
            else:
                raise Exception('CorsikaAtmosphere(): Season "' + season +
                                '" not parameterized for location SouthPole.')
        else:
            raise Exception("CorsikaAtmosphere:init_parameters(): Location " +
                            str(location) + " not parameterized.")

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
        """ Returns the density of air in g/cm**3.

        Uses the optimized module function :func:`corsika_get_density_jit`.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return corsika_get_density_jit(h_cm, self._atm_param)

    def get_mass_overburden(self, h_cm):
        """ Returns the mass overburden in atmosphere in g/cm**2.

        Uses the optimized module function :func:`corsika_get_m_overburden_jit`.

        Args:
          h_cm (float): height in cm

        Returns:
          float: column depth :math:`\\T(h_{cm})` in g/cm**2
        """
        return corsika_get_m_overburden_jit(h_cm, self._atm_param)

    def rho_inv(self, X, cos_theta):
        """Returns reciprocal density in cm**3/g using planar approximation.

        This function uses the optimized function :func:`planar_rho_inv_jit`

        Args:
          h_cm (float): height in cm

        Returns:
          float: :math:`\\frac{1}{\\rho}(X,\\cos{\\theta})` cm**3/g
        """
        return planar_rho_inv_jit(X, cos_theta, self._atm_param)

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
            thickl.append('{0:4.6f}'.format(quad(self.get_density, h,
                                                 112.8e5, epsrel=1e-4)[0]))
        print '_thickl = np.array([' + ', '.join(thickl) + '])'


@jit(double(double, double, double[:, :]), target='cpu')
def planar_rho_inv_jit(X, cos_theta, param):
    """Optimized calculation of :math:`1/\\rho(X,\\theta)` in
    planar approximation.

    This function can be used for calculations where
    :math:`\\theta < 70^\\circ`.

    Args:
      X (float): slant depth in g/cm**2
      cos_theta (float): :math:`\\cos(\\theta)`

    Returns:
      float: :math:`1/\\rho(X,\\theta)` in cm**3/g
    """
    a = param[0]
    b = param[1]
    c = param[2]
    t = param[3]
    res = 0.0
    x_v = X * cos_theta
    layer = 0
    for i in xrange(t.size):
        if not (x_v >= t[i]):
            layer = i
    if layer == 4:
        res = c[4] / b[4]
    else:
        l = layer
        res = c[l] / (x_v - a[l])
    return res


@jit(double(double, double[:, :]), target='cpu')
def corsika_get_density_jit(h_cm, param):
    """Optimized calculation of :math:`\\rho(h)` in
    according to CORSIKA type parameterization.

    Args:
      h_cm (float): height above surface in cm
      param (numpy.array): 5x5 parameter array from
                        :class:`CorsikaAtmosphere`

    Returns:
      float: :math:`\\rho(h)` in g/cm**3
    """
    b = param[1]
    c = param[2]
    hl = param[4]
    res = 0.0
    layer = 0
    for i in xrange(hl.size):
        if not (h_cm <= hl[i]):
            layer = i
    if layer == 4:
        res = b[4] / c[4]
    else:
        l = layer
        res = b[l] / c[l] * np.exp(-h_cm / c[l])

    return res

@jit(double(double, double[:, :]), target='cpu')
def corsika_get_m_overburden_jit(h_cm, param):
    """Optimized calculation of :math:`\\T(h)` in
    according to CORSIKA type parameterization.

    Args:
      h_cm (float): height above surface in cm
      param (numpy.array): 5x5 parameter array from
                        :class:`CorsikaAtmosphere`

    Returns:
      float: :math:`\\rho(h)` in g/cm**3
    """
    a = param[0]
    b = param[1]
    c = param[2]
    hl = param[4]
    res = 0.0
    layer = 0

    for i in xrange(hl.size):
        if not (h_cm <= hl[i]):
            layer = i

    if layer == 4:
        res = a[4] - b[4] / c[4] * h_cm
    else:
        l = layer
        res = a[l] + b[l] * np.exp(-h_cm / c[l])

    return res

class IsothermalAtmosphere(EarthAtmosphere):

    """Isothermal model of the atmosphere.

    This model is widely used in semi-analytical calculations. The isothermal
    approximation is valid in a certain range of altitudes and usually
    one adjust the parameters to match a more realistic density profile
    at altitudes between 10 - 30 km, where the high energy muon production
    rate peaks. Such parametrizations are given in the book "Cosmic Rays and
    Particle Physics", Gaisser, Engel and Resconi (2016). The default values are
    from M. Thunman, G. Ingelman, and P. Gondolo, Astropart. Physics 5, 309 (1996).
       
    Args:
      location (str): no effect
      season (str): no effect
      hiso_km (float): isothermal scale height in km
      X0 (float): Ground level overburden
    """

    def __init__(self, location, season, hiso_km = 6.3, X0 = 1300.):
        self.hiso_cm = hiso_km * 1e5
        self.X0 = X0
        self.location = location
        self.season = season
        
        EarthAtmosphere.__init__(self)

    def get_density(self, h_cm):
        """ Returns the density of air in g/cm**3.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self.X0/self.hiso_cm*np.exp(-h_cm/self.hiso_cm)

class MSIS00Atmosphere(EarthAtmosphere):

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

    def __init__(self, location, season):
        from msis_wrapper import cNRLMSISE00, pyNRLMSISE00
        if config['msis_python'] == 'ctypes':
            self._msis = cNRLMSISE00()
        else:
            self._msis = pyNRLMSISE00()

        self.init_parameters(location, season)

        EarthAtmosphere.__init__(self)

    def init_parameters(self, location, season):
        """Sets location and season in :class:`NRLMSISE-00`.

        Translates location and season into day of year
        and geo coordinates.

        Args:
          location (str): Supported are "SouthPole" and "Karlsruhe"
          season (str): months of the year: January, February, etc.
        """
        self._msis.set_location(location)
        self._msis.set_season(season)

        self.location, self.season = location, season
        # Clear cached value to force spline recalculation
        self.theta_deg = None

    def get_density(self, h_cm):
        """ Returns the density of air in g/cm**3.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self._msis.get_density(h_cm)


class AIRSAtmosphere(EarthAtmosphere):

    """Interpolation class for tabulated atmospheres.

    This class is intended to read preprocessed AIRS Satellite data.

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    def __init__(self, location, season, *args, **kwargs):
        if location != 'SouthPole':
            raise Exception(self.__class__.__name__ + 
                "(): Only South Pole location supported. " + location)

        self.month2doy = {'January':1,
                          'February':32,
                          'March':60,
                          'April':91,
                          'May':121,
                          'June':152,
                          'July':182,
                          'August':213,
                          'September':244,
                          'October':274,
                          'November':305,
                          'December':335}
        self.season = season
        self.init_parameters(location, **kwargs)
        EarthAtmosphere.__init__(self)

    def init_parameters(self, location, **kwargs):
        """Loads tables and prepares interpolation.

        Args:
          location (str): supported is only "SouthPole"
          doy (int): Day Of Year
        """
        from matplotlib.dates import strpdate2num, UTC, num2date
        from os import path
        
        data_path = (join(path.expanduser('~'),
            'work/projects/atmospheric_variations/'))

        if 'table_path' in kwargs:
            data_path = kwargs['table_path']

        files = [
            ('dens','airs_amsu_dens_180_daily.txt'),
            ('temp','airs_amsu_temp_180_daily.txt'),
            ('alti','airs_amsu_alti_180_daily.txt')]

        data_collection = {}

        #limit SouthPole pressure to <= 600
        min_press_idx = 4
        
        IC79_idx_1 = None
        IC79_idx_2 = None

        for d_key, fname in files:
            fname = data_path + 'tables/' + fname
            tab = np.loadtxt(fname,
                             converters={0:strpdate2num('%Y/%m/%d')}, 
                             usecols=[0] + range(2,27))
            with open(fname,'r') as f:
                comline = f.readline()
            # print comline
            p_levels = [float(s.strip()) for s in 
                comline.split(' ')[3:] if s != ''][min_press_idx:]
            dates = num2date(tab[:,0])
            for di, date in enumerate(dates):
                if (date.month==6 and date.day==1):
                    if date.year==2010: IC79_idx_1=di
                    elif date.year==2011: IC79_idx_2=di
            surf_val = tab[:,1]
            cols = tab[:, min_press_idx+2:]
            data_collection[d_key] = (dates,surf_val,cols)

        self.interp_tab = {}
        self.dates = {}
        dates = data_collection['alti'][0]

        msis = MSIS00Atmosphere(location,'January')

        for didx, date in enumerate(dates):
            h_vec = np.array(data_collection['alti'][2][didx,:]*1e2)
            d_vec = np.array(data_collection['dens'][2][didx,:])

            #Extrapolate using msis
            h_extra = np.linspace(h_vec[-1],config['h_atm']*1e2,25)
            msis._msis.set_doy(self._get_y_doy(date)[1]-1)
            msis_extra = np.array([msis.get_density(h) for h in h_extra])

            # Interpolate last few altitude bins
            ninterp = 5

            for ni in range(ninterp):
                cl = (1 - np.exp(-ninterp+ni + 1))
                ch = (1 - np.exp(-ni))
                norm = 1./(cl + ch)
                d_vec[-ni-1] = (d_vec[-ni-1]*cl*norm + 
                                msis.get_density(h_vec[-ni-1])*ch*norm)

            # Merge the two datasets
            h_vec = np.hstack([h_vec[:-1], h_extra])
            d_vec = np.hstack([d_vec[:-1], msis_extra])

            self.interp_tab[self._get_y_doy(date)] = (h_vec, d_vec)

            self.dates[self._get_y_doy(date)] = date

        self.IC79_start = self._get_y_doy(dates[IC79_idx_1])
        self.IC79_end = self._get_y_doy(dates[IC79_idx_2])
        self.IC79_days = (dates[IC79_idx_2] - dates[IC79_idx_1]).days
        self.location = location
        if self.season == None:
            self.set_IC79_day(0)
        else:
            self.set_season(self.season)
        # Clear cached value to force spline recalculation
        self.theta_deg = None

    def set_date(self, year,doy):
        self.h, self.dens = self.interp_tab[(year,doy)]
        self.date = self.dates[(year,doy)]
        # Compatibility with caching
        self.season = self.date

    def _set_doy(self, doy, year=2010):
        self.h, self.dens = self.interp_tab[(year,doy)]
        self.date = self.dates[(year,doy)]

    def set_season(self, month):
        self.season = month
        self._set_doy(self.month2doy[month])
        self.season = month

    def set_IC79_day(self, IC79_day):
        import datetime
        if IC79_day > self.IC79_days:
            raise Exception(self.__class__.__name__ + 
                "::set_IC79_day(): IC79_day above range.")
        target_day = self._get_y_doy(self.dates[self.IC79_start] + 
                      datetime.timedelta(days=IC79_day))
        print 'setting IC79_day', IC79_day
        self.h, self.dens = self.interp_tab[target_day]
        self.date = self.dates[target_day]
        # Compatibility with caching
        self.season = self.date

    def _get_y_doy(self, date):
        return date.timetuple().tm_year, date.timetuple().tm_yday 

    def get_density(self, h_cm):
        """ Returns the density of air in g/cm**3.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return np.exp(np.interp(h_cm, self.h, np.log(self.dens)))


class MSIS00IceCubeCentered(MSIS00Atmosphere):

    """Extension of :class:`MSIS00Atmosphere` which couples the latitude
    setting with the zenith angle of the detector.

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """
    def __init__(self, location, season):
        if location != 'SouthPole':
            if dbg > 0: print ('{0} location forced to SouthPole in' + 
                            ' class').format(self.__class__.__name__)
            location = 'SouthPole'
        MSIS00Atmosphere.__init__(self, location, season)


    def latitude(self, det_zenith_deg):
        """ Returns the geographic latitude of the shower impact point.

        Assumes a spherical earth. The detector is 1948m under the 
        surface.

        Credits: geometry fomulae by Jakob van Santen, DESY Zeuthen.

        Args:
          det_zenith_deg (float): zenith angle at detector in degrees

        Returns:
          float: latitude of the impact point in degrees
        """
        r = config['r_E']
        d = 1948  # m

        theta_rad = det_zenith_deg/180.*np.pi

        x = (np.sqrt(2.*r*d + ((r - d)*np.cos(theta_rad))**2 - d**2)
             - (r - d)*np.cos(theta_rad))

        return -90. + np.arctan2(x * np.sin(theta_rad),
                                 r - d + x * np.cos(theta_rad))/np.pi*180.

    def set_theta(self, theta_deg):

        self._msis.set_location_coord(longitude=0.,
                                      latitude=self.latitude(theta_deg))
        if dbg > 0:
            print ('{0}::set_theta(): latitude = {1} for ' + 
                'zenith angle = {2}').format(self.__class__.__name__,
                                             self.latitude(theta_deg),
                                             theta_deg)
        if theta_deg > 90.:
            if dbg > 0:
                print ('{0}::set_theta(): theta = {1} below horizon.' + 
                    'using theta = {2}').format(self.__class__.__name__,
                                                 theta_deg,
                                                 180. - theta_deg)
            theta_deg = 180. - theta_deg
        MSIS00Atmosphere.set_theta(self, theta_deg,
                                   force_spline_calc=True)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    atm_obj = CorsikaAtmosphere('PL_SouthPole', 'January')

    atm_obj.set_theta(0.0)
    x_vec = np.linspace(0, atm_obj.max_X, 10000)
    plt.plot(x_vec, 1 / atm_obj.r_X2rho(x_vec), lw=1.5,
             label="PL_SouthPole/January")

    atm_obj.init_parameters('PL_SouthPole', 'August')
    atm_obj.set_theta(0.0)
    x_vec = np.linspace(0, atm_obj.max_X, 10000)
    plt.plot(x_vec, 1 / atm_obj.r_X2rho(x_vec), lw=1.5,
             label="PL_SouthPole/August")
    plt.legend()
    plt.tight_layout()
    plt.show()
