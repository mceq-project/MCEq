import numpy as np

from MCEq import config


class EarthGeometry:
    r"""A model of the Earth's geometry, approximating it
       by a sphere. The figure below illustrates the meaning of the parameters.

    .. figure:: ../_static/graphics/geometry.*
        :scale: 50 %
        :alt: picture of geometry


        Curved geometry as it is used in the code (not to scale!).

    Example:
      The plots below will be produced by executing the module::

          $ python geometry.py

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from MCEq.geometry.geometry import *

        earth = EarthGeometry()

        theta_list = np.linspace(0, 90, 500)
        h_vec = np.linspace(0, earth.h_atm, 500)
        th_list_rad = np.deg2rad(theta_list)
        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(theta_list, earth.path_len(th_list_rad) / 1e5, lw=2)
        plt.xlabel(r"zenith $\theta$ at detector")
        plt.ylabel(r"path length $l(\theta)$ in km")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(
            theta_list, np.arccos(earth.cos_th_star(th_list_rad)) / np.pi * 180.0, lw=2
        )
        plt.xlabel(r"zenith $\theta$ at detector")
        plt.ylabel(r"$\theta^*$ at top of the atm.")
        plt.ylim([0, 90])
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(h_vec / 1e5, earth.delta_l(h_vec, np.deg2rad(85.0)) / 1e5, lw=2)
        plt.ylabel(r"Path length $\Delta l(h)$ in km")
        plt.xlabel(r"atm. height $h_{atm}$ in km")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        for theta in [30.0, 60.0, 70.0, 80.0, 85.0, 90.0]:
            theta_path = np.deg2rad(theta)
            delta_l_vec = np.linspace(0, earth.path_len(theta_path), 1000)
            plt.plot(
                delta_l_vec / 1e5,
                earth.h(delta_l_vec, theta_path) / 1e5,
                label=rf"${theta}^o$",
                lw=2,
            )
        plt.legend()
        plt.xlabel(r"path length $\Delta l$ [km]")
        plt.ylabel(r"atm. height $h_{atm}(\Delta l, \theta)$ [km]")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        plt.show()

    Attributes:
      h_obs (float): observation level height [cm]
      h_atm (float): top of the atmosphere [cm]
      r_E (float): radius Earth [cm]
      r_top (float): radius at top of the atmosphere  [cm]
      r_obs (float): radius at observation level [cm]

    """

    def __init__(self, r_e=config.r_E, h_obs=config.h_obs, h_atm=config.h_atm):
        self.r_e = r_e
        self.h_atm = h_atm
        self.r_top = self.r_e + self.h_atm
        self.set_h_obs(h_obs)

    def set_h_obs(self, h_obs):
        """
        Set the elevation of the observation level.

        Args:
            h_obs (float): Elevation of the observation level in cm.
        """
        if h_obs >= self.r_top:
            raise ValueError("Observation level cannot be above atmospheric boundary.")
        self.h_obs = h_obs
        self.r_obs = self.r_e + self.h_obs
        self.theta_max_rad = max(np.pi / 2.0, np.pi - np.arcsin(self.r_e / self.r_obs))
        self.theta_max_deg = np.rad2deg(self.theta_max_rad)

    def _check_angles(self, theta):
        """
        Check if the angles are within the valid range.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Raises:
            ValueError: If the zenith angle is above the maximum allowed.
        """
        if np.any(np.atleast_1d(theta) > self.theta_max_rad):
            raise ValueError(
                f"Zenith angle above maximum {self.theta_max_deg:.1f} degrees."
            )

    def _a_1(self, theta):
        """
        Calculate segment length A1(theta) in cm.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Segment length A1(theta) in cm.
        """
        return self.r_obs * np.cos(theta)

    def _a_2(self, theta):
        """
        Calculate segment length A2(theta) in cm.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Segment length A2(theta) in cm.
        """
        return self.r_obs * np.sin(theta)

    def path_len(self, theta):
        r"""Returns path length in [cm] for given zenith
        angle :math:`\theta` [rad].
        """
        Calculate the path length in cm for a given zenith angle.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Path length in cm.
        """
        self._check_angles(theta)
        return np.sqrt(self.r_top**2 - self._a_2(theta) ** 2) - self._a_1(theta)

    def cos_th_star(self, theta):
        r"""Returns the zenith angle at atmospheric boarder
        :math:`\cos(\theta^*)` in [rad] as a function of zenith at detector.
        """
        Calculate the zenith angle at the atmospheric border as a function of
        zenith angle at the detector.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Zenith angle at the atmospheric border in radians.
        """
        self._check_angles(theta)
        return (self._a_1(theta) + self.path_len(theta)) / self.r_top

    def h(self, dl, theta):
        r"""Height above surface at distance :math:`dl` counted from the beginning
        of path :math:`l(\theta)` in cm.
        """
        return (
            np.sqrt(
                self._a_2(theta) ** 2 + (self._a_1(theta) + self.path_len(theta) - dl) ** 2
            )
            - self.r_e
        )

    def delta_l(self, h, theta):
        r"""Distance :math:`dl` covered along path :math:`l(\theta)`
        as a function of current height. Inverse to :func:`h`.
        """
        return (
            self._a_1(theta)
            + self.path_len(theta)
            - np.sqrt((h + self.r_e) ** 2 - self._a_2(theta) ** 2)
        )


def chirkin_cos_theta_star(costheta):
    r""":math:`\cos(\theta^*)` parameterization.

    This function returns the equivalent zenith angle for
    for very inclined showers. It is based on a CORSIKA study by
    `D. Chirkin, hep-ph/0407078v1, 2004
    <http://arxiv.org/abs/hep-ph/0407078v1>`_.

    Args:
        costheta (float): :math:`\cos(\theta)` in [rad]

    Returns:
        float: :math:`\cos(\theta*)` in [rad]
    """

    p1 = 0.102573
    p2 = -0.068287
    p3 = 0.958633
    p4 = 0.0407253
    p5 = 0.817285
    x = costheta
    return np.sqrt((x**2 + p1**2 + p2 * x**p3 + p4 * x**p5) / (1 + p1**2 + p2 + p4))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    earth = EarthGeometry()

    theta_list = np.linspace(0, 90, 500)
    h_vec = np.linspace(0, earth.h_atm, 500)
    th_list_rad = np.deg2rad(theta_list)
    fig = plt.figure(figsize=(5, 4))
    fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
    plt.plot(theta_list, earth.path_len(th_list_rad) / 1e5, lw=2)
    plt.xlabel(r"zenith $\theta$ at detector")
    plt.ylabel(r"path length $l(\theta)$ in km")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig = plt.figure(figsize=(5, 4))
    fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
    plt.plot(
        theta_list, np.arccos(earth.cos_th_star(th_list_rad)) / np.pi * 180.0, lw=2
    )
    plt.xlabel(r"zenith $\theta$ at detector")
    plt.ylabel(r"$\theta^*$ at top of the atm.")
    plt.ylim([0, 90])
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig = plt.figure(figsize=(5, 4))
    fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
    plt.plot(h_vec / 1e5, earth.delta_l(h_vec, np.deg2rad(85.0)) / 1e5, lw=2)
    plt.ylabel(r"Path length $\Delta l(h)$ in km")
    plt.xlabel(r"atm. height $h_{atm}$ in km")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig = plt.figure(figsize=(5, 4))
    fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
    for theta in [30.0, 60.0, 70.0, 80.0, 85.0, 90.0]:
        theta_path = np.deg2rad(theta)
        delta_l_vec = np.linspace(0, earth.path_len(theta_path), 1000)
        plt.plot(
            delta_l_vec / 1e5,
            earth.h(delta_l_vec, theta_path) / 1e5,
            label=rf"${theta}^o$",
            lw=2,
        )
    plt.legend()
    plt.xlabel(r"path length $\Delta l$ [km]")
    plt.ylabel(r"atm. height $h_{atm}(\Delta l, \theta)$ [km]")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.show()
