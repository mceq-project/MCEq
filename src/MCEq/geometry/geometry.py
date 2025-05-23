import numpy as np

from MCEq import config
from MCEq.misc import theta_rad


class EarthGeometry:
    r"""A model of the Earth's geometry, approximating it
       by a sphere. The figure below illustrates the meaning of the parameters.

    .. figure:: graphics/geometry.*
        :scale: 30 %
        :alt: picture of geometry


        Curved geometry as it is used in the code (not to scale!).

    Example:
      The plots below will be produced by executing the module::

          $ python geometry.py

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from MCEq.geometry.geometry import *
        from MCEq.misc import theta_rad

        g = EarthGeometry()
        theta_list = np.linspace(0, 90, 500)
        h_vec = np.linspace(0, g.h_atm, 500)
        th_list_rad = theta_rad(theta_list)
        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(theta_list, g.l(th_list_rad) / 1e5,
                 lw=2)
        plt.xlabel(r'zenith $\theta$ at detector')
        plt.ylabel(r'path length $l(\theta)$ in km')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(theta_list,
            np.arccos(g.cos_th_star(th_list_rad)) / np.pi * 180.,
            lw=2)
        plt.xlabel(r'zenith $\theta$ at detector')
        plt.ylabel(r'$\theta^*$ at top of the atm.')
        plt.ylim([0, 90])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        plt.plot(h_vec / 1e5, g.delta_l(h_vec, theta_rad(85.)) / 1e5,
                 lw=2)
        plt.ylabel(r'Path length $\Delta l(h)$ in km')
        plt.xlabel(r'atm. height $h_{atm}$ in km')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        fig = plt.figure(figsize=(5, 4))
        fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
        for theta in [30., 60., 70., 80., 85., 90.]:
            theta_path = theta_rad(theta)
            delta_l_vec = np.linspace(0, g.l(theta_path), 1000)
            plt.plot(delta_l_vec / 1e5, g.h(delta_l_vec, theta_path) / 1e5,
                     label=r'${0}^o$'.format(theta), lw=2)
        plt.legend()
        plt.xlabel(r'path length $\Delta l$ [km]')
        plt.ylabel(r'atm. height $h_{atm}(\Delta l, \theta)$ [km]')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.show()

    Attributes:
      h_obs (float): observation level height [cm]
      h_atm (float): top of the atmosphere [cm]
      r_E (float): radius Earth [cm]
      r_top (float): radius at top of the atmosphere  [cm]
      r_obs (float): radius at observation level [cm]

    """

    def __init__(self):
        self.h_obs = config.h_obs * 1e2  # cm
        self.h_atm = config.h_atm * 1e2  # cm
        self.r_E = config.r_E * 1e2  # cm
        self.r_top = self.r_E + self.h_atm
        self.r_obs = self.r_E + self.h_obs

    def _A_1(self, theta):
        r"""Segment length :math:`A1(\theta)` in cm."""
        return self.r_obs * np.cos(theta)

    def _A_2(self, theta):
        r"""Segment length :math:`A2(\theta)` in cm."""
        return self.r_obs * np.sin(theta)

    def path_len(self, theta):
        r"""Returns path length in [cm] for given zenith
        angle :math:`\theta` [rad].
        """
        return np.sqrt(self.r_top**2 - self._A_2(theta) ** 2) - self._A_1(theta)

    def cos_th_star(self, theta):
        r"""Returns the zenith angle at atmospheric boarder
        :math:`\cos(\theta^*)` in [rad] as a function of zenith at detector.
        """
        return (self._A_1(theta) + self.path_len(theta)) / self.r_top

    def h(self, dl, theta):
        r"""Height above surface at distance :math:`dl` counted from the beginning
        of path :math:`l(\theta)` in cm.
        """
        return (
            np.sqrt(
                self._A_2(theta) ** 2
                + (self._A_1(theta) + self.path_len(theta) - dl) ** 2
            )
            - self.r_E
        )

    def delta_l(self, h, theta):
        r"""Distance :math:`dl` covered along path :math:`l(\theta)`
        as a function of current height. Inverse to :func:`h`.
        """
        return (
            self._A_1(theta)
            + self.path_len(theta)
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
    return np.sqrt((x**2 + p1**2 + p2 * x**p3 + p4 * x**p5) / (1 + p1**2 + p2 + p4))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    earth = EarthGeometry()

    theta_list = np.linspace(0, 90, 500)
    h_vec = np.linspace(0, earth.h_atm, 500)
    th_list_rad = theta_rad(theta_list)
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
    plt.plot(h_vec / 1e5, earth.delta_l(h_vec, theta_rad(85.0)) / 1e5, lw=2)
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
        theta_path = theta_rad(theta)
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
