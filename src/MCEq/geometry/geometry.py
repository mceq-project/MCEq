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

    def __init__(self):
        self.h_obs = config.h_obs * 1e2  # cm
        self.h_atm = config.h_atm * 1e2  # cm
        if self.h_obs < 0 or self.h_obs > self.h_atm:
            raise ValueError(
                f"Observation height must be between 0 and {self.h_atm:.2e} cm."
            )
        if self.h_atm <= self.h_obs:
            raise ValueError(
                f"Top of atmosphere must be above observation height ({self.h_atm:.2e} cm > {self.h_obs:.2e} cm)."
            )
        self.r_E = config.r_E * 1e2  # cm
        self.r_top = self.r_E + self.h_atm
        self.r_obs = self.r_E + self.h_obs
        self.theta_max_rad = max(np.pi / 2.0, np.pi - np.arcsin(self.r_E / self.r_obs))
        self.theta_max_deg = np.rad2deg(self.theta_max_rad)

    def set_h_obs(self, h_obs):
        """Set the elevation of the observation (detector) level in cm."""
        if h_obs < 0 or h_obs > self.h_atm:
            raise ValueError(
                f"Observation height must be between 0 and {self.h_atm:.2e} cm."
            )
        if self.h_atm <= h_obs:
            raise ValueError(
                f"Top of atmosphere must be above observation height ({self.h_atm:.2e} cm > {h_obs:.2e} cm)."
            )

        self.h_obs = h_obs
        self.r_obs = self.r_E + self.h_obs
        self.theta_max_rad = max(np.pi / 2.0, np.pi - np.arcsin(self.r_E / self.r_obs))
        self.theta_max_deg = np.rad2deg(self.theta_max_rad)

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


def surface_dip_deg(r_det_cm, r_surf_cm):
    r"""Angular dip of the local surface horizon seen from the detector.

    A detector below the surface sees the horizon slightly *below* 90 deg:
    rays with :math:`\theta \le 90^\circ + \mathrm{dip}` still leave through
    the near-side surface, larger angles traverse the Earth.

    Args:
        r_det_cm (float): distance of the detector from the Earth's centre [cm]
        r_surf_cm (float): radius of the local surface sphere [cm]

    Returns:
        float: dip angle in degrees, zero for a detector on the surface
    """
    return np.rad2deg(np.arccos(min(r_det_cm / r_surf_cm, 1.0)))


def local_zenith_deg(r_det_cm, r_surf_cm, zenith_deg):
    r"""Zenith angle of a straight shower axis where it crosses the surface.

    The impact parameter :math:`r \sin\theta` is conserved along a straight
    line, so the surface-frame angle follows from the detector-frame one as
    :math:`\sin\theta_s = (r_{det}/r_{surf})\,\sin\theta`.  The correction is
    negligible below ~80 deg but reaches ~1.4 deg at the horizon for a detector
    2 km below the surface.

    Args:
        r_det_cm (float): distance of the detector from the Earth's centre [cm]
        r_surf_cm (float): radius of the local surface sphere [cm]
        zenith_deg (float): zenith angle at the detector [deg]

    Returns:
        float: local zenith angle at the surface crossing in degrees
    """
    sin_loc = np.clip(r_det_cm / r_surf_cm * np.sin(np.deg2rad(zenith_deg)), 0.0, 1.0)
    return np.rad2deg(np.arcsin(sin_loc))


def impact_point(
    r_surf_cm, r_det_cm, det_lon_deg, det_lat_deg, zenith_deg, azimuth_deg
):
    """Geographic coordinates where a shower axis crosses the surface sphere.

    Follows the shower axis from the detector toward the incoming source and
    intersects it with the sphere of radius *r_surf_cm*, in full 3-D ECEF
    (Earth-Centred, Earth-Fixed) Cartesian geometry.  That crossing is where the
    atmospheric column the shower developed in stands.

    Azimuth convention: 0 deg = geographic North, 90 deg = East (clockwise from
    North).  The full zenith range [0, 180] deg is supported: for downgoing
    showers the crossing is near the detector, for upgoing ones the ray
    traverses the Earth and the crossing is on the far side.

    Args:
        r_surf_cm (float): radius of the surface sphere to intersect [cm]
        r_det_cm (float): distance of the detector from the Earth's centre [cm]
        det_lon_deg (float): detector longitude [deg]
        det_lat_deg (float): detector latitude [deg]
        zenith_deg (float): zenith angle at the detector [deg]
        azimuth_deg (float): azimuth angle [deg], 0 = North, 90 = East

    Returns:
        tuple: ``(latitude_deg, longitude_deg)`` of the impact point
    """
    theta = np.deg2rad(zenith_deg)
    alpha = np.deg2rad(azimuth_deg)
    lat0 = np.deg2rad(det_lat_deg)
    lon0 = np.deg2rad(det_lon_deg)

    # Detector position in ECEF
    p_det = np.array(
        [
            r_det_cm * np.cos(lat0) * np.cos(lon0),
            r_det_cm * np.cos(lat0) * np.sin(lon0),
            r_det_cm * np.sin(lat0),
        ]
    )

    # Shower direction in the local ENU frame, pointing toward the source
    d_enu = np.array(
        [
            np.sin(theta) * np.sin(alpha),  # East
            np.sin(theta) * np.cos(alpha),  # North
            np.cos(theta),  # Up
        ]
    )

    # ENU -> ECEF rotation (columns: East, North, Up basis vectors)
    rot = np.array(
        [
            [-np.sin(lon0), -np.sin(lat0) * np.cos(lon0), np.cos(lat0) * np.cos(lon0)],
            [np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0) * np.sin(lon0)],
            [0.0, np.cos(lat0), np.sin(lat0)],
        ]
    )
    d_ecef = rot @ d_enu

    # Crossing with |p_det + x * d_ecef|^2 = r_surf^2.  The larger root is the
    # crossing on the source side; it is valid for detectors below and above
    # the sphere and over the full zenith range.
    a = np.dot(d_ecef, p_det)
    x = -a + np.sqrt(a**2 + (r_surf_cm**2 - r_det_cm**2))

    p_impact = p_det + x * d_ecef
    lat_imp = np.rad2deg(np.arcsin(np.clip(p_impact[2] / r_surf_cm, -1.0, 1.0)))
    lon_imp = np.rad2deg(np.arctan2(p_impact[1], p_impact[0]))
    return lat_imp, lon_imp


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
