"""A single-scale-height isothermal atmosphere, mostly for cross-checks."""

import numpy as np

from MCEq.environment.base import EarthsAtmosphere


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

    def __init__(self, location, season, hiso_km=6.3, X0=1300.0, environment=None):
        self.hiso_cm = hiso_km * 1e5
        self.X0 = X0
        self.location = location
        self.season = season

        EarthsAtmosphere.__init__(self, environment=environment)

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
