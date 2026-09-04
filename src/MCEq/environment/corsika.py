"""CORSIKA's Linsley-parameterised atmospheres."""

import numpy as np

from MCEq.environment.base import EarthsAtmosphere
from MCEq.environment.parameters import (
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)
from MCEq.misc import info


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
        import MCEq.environment._ext.corsikaatm as corsika_acc

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
