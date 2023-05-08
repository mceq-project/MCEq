import numpy as np
import mceq_config as config


class EarthGeometry:
    r"""
    A model of the Earth's geometry, approximating it by a sphere.

    .. figure:: graphics/geometry.*
        :scale: 30 %
        :alt: picture of geometry

    Args:
        r_E (float): Radius of the Earth in cm.
        h_obs (float): Observation level height in cm.
        h_atm (float): Top of the atmosphere height in cm.
    """

    def __init__(self, r_E=config.r_E, h_obs=config.h_obs, h_atm=config.h_atm):
        self.r_E = r_E
        self.h_atm = h_atm
        self.r_top = self.r_E + self.h_atm
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
        self.r_obs = self.r_E + self.h_obs
        self.theta_max_rad = max(np.pi / 2.0, np.pi - np.arcsin(self.r_E / self.r_obs))
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

    def _A_1(self, theta):
        """
        Calculate segment length A1(theta) in cm.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Segment length A1(theta) in cm.
        """
        return self.r_obs * np.cos(theta)

    def _A_2(self, theta):
        """
        Calculate segment length A2(theta) in cm.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Segment length A2(theta) in cm.
        """
        return self.r_obs * np.sin(theta)

    def pl(self, theta):
        """
        Calculate the path length in cm for a given zenith angle.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Path length in cm.
        """
        self._check_angles(theta)
        return np.sqrt(self.r_top**2 - self._A_2(theta) ** 2) - self._A_1(theta)

    def cos_th_star(self, theta):
        """
        Calculate the zenith angle at the atmospheric border as a function of
        zenith angle at the detector.

        Args:
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Zenith angle at the atmospheric border in radians.
        """
        self._check_angles(theta)
        return (self._A_1(theta) + self.pl(theta)) / self.r_top

    def h(self, dl, theta):
        """
        Calculate the height above the surface at a distance dl along the
        path for a given zenith angle.

        Args:
            dl (float or numpy.ndarray): Distance along the path in cm.


            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Height above the surface in cm.
        """
        self._check_angles(theta)
        return (
            np.sqrt(
                self._A_2(theta) ** 2 + (self._A_1(theta) + self.pl(theta) - dl) ** 2
            )
            - self.r_E
        )

    def delta_l(self, h, theta):
        """
        Calculate the distance dl covered along the path as a function of
        the current height.

        Args:
            h (float or numpy.ndarray): Height above the surface in cm.
            theta (float or numpy.ndarray): Zenith angle in radians.

        Returns:
            float or numpy.ndarray: Distance along the path in cm.
        """
        self._check_angles(theta)
        return (
            self._A_1(theta)
            + self.pl(theta)
            - np.sqrt((h + self.r_E) ** 2 - self._A_2(theta) ** 2)
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
    return np.sqrt(
        (x**2 + p1**2 + p2 * x**p3 + p4 * x**p5) / (1 + p1**2 + p2 + p4)
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    earth = EarthGeometry()

    theta_list = np.linspace(0, 90, 500)
    h_vec = np.linspace(0, earth.h_atm, 500)
    th_list_rad = np.radians(theta_list)
    fig = plt.figure(figsize=(5, 4))
    fig.set_layout_engine("tight")
    plt.plot(theta_list, earth.pl(th_list_rad) / 1e5, lw=2)
    plt.xlabel(r"zenith $\theta$ at detector")
    plt.ylabel(r"path length $l(\theta)$ in km")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig = plt.figure(figsize=(5, 4))
    fig.set_layout_engine("tight")
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
    fig.set_layout_engine("tight")
    plt.plot(h_vec / 1e5, earth.delta_l(h_vec, np.radians(85.0)) / 1e5, lw=2)
    plt.ylabel(r"Path length Delta l(h)$ in km (theta=85 deg.)")
    plt.xlabel(r"atm. height $h_{atm}$ in km")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig = plt.figure(figsize=(5, 4))
    fig.set_layout_engine("tight")
    for theta in [30.0, 60.0, 70.0, 80.0, 85.0, 90.0]:
        theta_path = np.radians(theta)
        delta_l_vec = np.linspace(0, earth.pl(theta_path), 1000)
        plt.plot(
            delta_l_vec / 1e5,
            earth.h(delta_l_vec, theta_path) / 1e5,
            label=r"${0}^o$".format(theta),
            lw=2,
        )
    plt.legend()
    plt.xlabel(r"path length Delta l$ [km]")
    plt.ylabel(r"atm. height $h_{atm}(\Delta l, \theta)$ [km]")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.show()
