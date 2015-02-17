# -*- coding: utf-8 -*-
"""
:mod:`MCEq.geometry` --- Extensive-Air-Shower geometry
======================================================

This module includes classes and functions modeling 
the Earth's geometry. The drawing on the right illustrates
the meaning of the parameters.

.. figure:: graphics/geometry.pdf
    :scale: 35 %
    :alt: picture of geometry
    :align: right
    
    Curved geometry as it is used in the code (not to scale!).

The module currently contains only module level functions. To
implement a different geometry, e.g. in an astrophysical content,
you can create a new module providing similar functions or
place the current content into a class.

Example:
  An example can be run by executing the module::

      $ python geometry.py

Attributes:
  h_obs (float): observation level height [cm]
  h_atm (float): top of the atmosphere [cm]
  r_E (float): radius Earth [cm]
  r_top (float): radius at top of the atmosphere  [cm]
  r_obs (float): radius at observation level [cm]
  
"""
import numpy as np
from mceq_config import config

h_obs = config['h_obs'] * 1e2  # cm
h_atm = config['h_atm'] * 1e2  # cm
r_E = config['r_E'] * 1e2  # cm
r_top = r_E + h_atm
r_obs = r_E + h_obs


def _theta_deg(cos_theta):
    """Converts :math:`\\cos{\\theta}` to :math:`\\theta` in degrees. 
    """
    return np.arccos(cos_theta) * 180. / np.pi


def _theta_rad(theta):
    """Converts :math:`\\theta` from rad to degrees.
    """
    return theta / 180. * np.pi


def _A_1(theta):
    """Segment length :math:`A1(\\theta)` in cm.
    """
    return r_obs * np.cos(theta)


def _A_2(theta):
    """Segment length :math:`A2(\\theta)` in cm.
    """
    return r_obs * np.sin(theta)


def l(theta):
    """Returns path length in [cm] for given zenith 
    angle :math:`\\theta` [rad].
    """
    return np.sqrt(r_top ** 2 - _A_2(theta) ** 2) - _A_1(theta)

def cos_th_star(theta):
    """Returns the zenith angle at atmospheric boarder 
    :math:`\\cos(\\theta^*)` in [rad] as a function of zenith at detector. 
    """
    return (_A_1(theta) + l(theta)) / r_top

def h(dl, theta):
    """Height above surface at distance :math:`dl` counted from the beginning
    of path :math:`l(\\theta)` in cm.  
    """
    return np.sqrt(_A_2(theta) ** 2 + (_A_1(theta) + l(theta) - dl) ** 2) - r_E

def delta_l(h, theta):
    """Distance :math:`dl` covered along path :math:`l(\\theta)` 
    as a function of current height. Inverse to :func:`h`. 
    """
    return _A_1(theta) + l(theta) - np.sqrt((h + r_E) ** 2 - _A_2(theta) ** 2)

def chirkin_cos_theta_star(costheta):
    """:math:`\\cos(\\theta^*)` parameterization.
    
    This function returns the equivalent zenith angle for
    for very inclined showers. It is based on a CORSIKA study by
    `D. Chirkin, hep-ph/0407078v1, 2004 <http://arxiv.org/abs/hep-ph/0407078v1>`_.
    
    Args:
      costheta (float): :math:`\\cos(\\theta)` in [rad]
    
    Returns:
      float: :math:`\\cos(\\theta*)` in [rad]
    """
    
    p1 = 0.102573
    p2 = -0.068287
    p3 = 0.958633
    p4 = 0.0407253
    p5 = 0.817285
    x = costheta
    return np.sqrt((x ** 2 + p1 ** 2 + p2 * x ** p3 + p4 * x ** p5)
                   / (1 + p1 ** 2 + p2 + p4))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    theta_list = np.linspace(0, 90, 500)
    h_vec = np.linspace(0, h_atm, 500)
    th_list_rad = _theta_rad(theta_list)
    fig = plt.figure(figsize=(5, 4))
    fig.set_tight_layout(dict(rect=[0.00, 0.00, 1, 1]))
    plt.plot(theta_list, l(th_list_rad) / 1e5,
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
    plt.plot(theta_list, np.arccos(cos_th_star(th_list_rad)) / np.pi * 180.,
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
    plt.plot(h_vec / 1e5, delta_l(h_vec, _theta_rad(85.)) / 1e5,
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
        theta_path = _theta_rad(theta)
        delta_l_vec = np.linspace(0, l(theta_path), 1000)
        plt.plot(delta_l_vec / 1e5, h(delta_l_vec, theta_path) / 1e5,
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
