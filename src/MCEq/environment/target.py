"""A generic homogeneous target, e.g. a fixed-target or beam-dump setup.

:class:`GeneralizedTarget` is not an :class:`~MCEq.environment.base.
EarthsAtmosphere`: it has no zenith angle and its ``set_theta`` raises. That
raise is the protocol boundary between "an atmosphere" and "a medium".
"""

import numpy as np

from MCEq.misc import info


class GeneralizedTarget:
    """This class provides a way to run MCEq on piece-wise constant
    one-dimenional density profiles.

    The default values for the average density mirror the config file
    variables `len_target`, `env_density` and `env_name` as of import.
    The density profile has to be built by calling subsequently
    :func:`add_material`. The current composition of the target
    can be checked with :func:`draw_materials` or :func:`print_table`.

    Note:
      If the target is not air or hydrogen, the result is approximate,
      since seconday particle yields are provided for nucleon-air or
      proton-proton collisions. Depending on this choice one has to
      adjust the nuclear mass in :mod:`mceq_config`.

    Args:
      len_target (float): total length of the target in cm
      env_density (float): density of the default material in g/cm**3
      env_name (str): title for this environment
    """

    def __init__(
        self,
        len_target=None,  # cm
        env_density=None,  # g/cm3
        env_name=None,
        environment=None,
    ):
        # Resolved here rather than as signature defaults: a default binds when
        # this module is first imported, which happens lazily from
        # MCEqRun.set_density_model, so whether a caller's `config.len_target`
        # was seen depended on whether it was written before that import.
        if len_target is None or env_density is None or env_name is None:
            if environment is None:
                from importlib import import_module

                environment = import_module("MCEq.config").environment
            env = environment
            if len_target is None:
                len_target = env.len_target * 1e2
            if env_density is None:
                env_density = env.env_density
            if env_name is None:
                env_name = env.env_name

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
