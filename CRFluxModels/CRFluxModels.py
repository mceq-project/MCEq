'''
:mod:`CRFluxModels` --- models of the high-energy cosmic ray flux
=================================================================

This module is a collection of models of the high-energy primary cosmic
ray flux, found in the literature of the last decades. The base class
:class:`PrimaryFlux` contains various methods which can be used
in all types of lepton flux calculations, such as semi-analytic,
numerical or Monte Carlo methods.

.. note::
    You might also consider using this module as a starting point
    for a new fitting project of the all-nucleon flux (with errors).
    Let me know if you need data tables.

The numbering scheme for nuclei is adopted from the Cosmic Ray
Air-Shower Monte Carlo `CORSIKA <https://web.ikp.kit.edu/corsika/>`_.
Protons have the ID 14. The ID for nuclei is composed using the
formula :math:`ID=100 \\cdot A + Z`, where :math:`A` is the mass
number and :math:`Z` the charge. Using this scheme one can
easily obtain charge and mass from the ID and vice-versa (see
:func:`PrimaryFlux.Z_A`).

The physics of each model can be found following the references
in this documentation.

.. note::

    As always, if you use this for your work, please cite the corresponding
    publications.

Example:
  To generate the plots from below, just run::

      $ python CRFluxModels.py

.. plot::

    from matplotlib import pyplot as plt
    from CRFluxModels import *
    pmodels = [(GaisserStanevTilav, "3-gen", "GST 3-gen"),
               (GaisserStanevTilav, "4-gen", "GST 4-gen"),
               (HillasGaisser2012, "H3a", "H3a"),
               (HillasGaisser2012, "H4a", "H4a"),
               (PolyGonato, False, "poly-gonato"),
               (Thunman, None, "TIG"),
               (ZatsepinSokolskaya, 'default', 'ZS'),
               (ZatsepinSokolskaya, 'pamela', 'ZSP'),
               (GaisserHonda, None, 'GH')]

    nfrac = {}
    evec = np.logspace(0, 10, 1000)
    plt.figure(figsize=(7.5,5))
    plt.title('Cosmic ray nucleon flux (proton + neutron)')
    for mclass, moptions, mtitle in pmodels:
        pmod = mclass(moptions)
        pfrac, p, n = pmod.p_and_n_flux(evec)
        plt.plot(evec, (p + n) * evec ** 2.5, ls='-', lw=1.5, label=mtitle)
        nfrac[mtitle] = (1 - pfrac)

    plt.loglog()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.ylim([10, 2e4])
    plt.tight_layout()

    plt.figure(figsize=(7.5,5))
    plt.title('Cosmic ray particle flux (all-nuclei).')

    for mclass, moptions, mtitle in pmodels:
        pmod = mclass(moptions)

        flux = pmod.total_flux(evec)
        plt.plot(evec, flux * evec ** 2.5, ls='-', lw=1.5, label=mtitle)

    plt.loglog()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.ylim([10, 2e4])
    plt.tight_layout()

    plt.figure(figsize=(7.5,5))
    plt.title('Fraction of neutrons relative to protons.')
    for mclass, moptions, mtitle in pmodels:
        plt.plot(evec, nfrac[mtitle], ls='-', lw=1.5, label=mtitle)

    plt.semilogx()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel("Neutron fraction")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.tight_layout()
    plt.show()
'''
import numpy as np
from abc import ABCMeta, abstractmethod


def _get_closest(value, in_list):
    """Compares the parameters ``value`` with values of the
    iterable ``in_list`` and returns the index and value of closest
    element.

    Args:
      value (int,float): value to be looked for in the list
      in_list (list,tuple): list of values with which to compare
    Returns:
      tuple (int,float): (index, value) of closest value in the list
    """
    minindex = np.argmin(np.abs(in_list - value * np.ones(len(in_list))))
    return minindex, in_list[minindex]


class PrimaryFlux():

    """Base class for primary cosmic ray flux models.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def nucleus_flux(self, corsika_id, E):
        """Returns the flux of nuclei corresponding to
        the ``corsika_id`` at energy ``E``.

        Args:
          corsika_id (int): see :mod:`CRFluxModels` for description.
          E (float): laboratory energy of nucleus in GeV
        Returns:
          (float): flux of single nucleus type :math:`\\Phi_{nucleus}`
          in :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        raise NotImplementedError(
            self.__class__.__name__ +
            '::nucleus_flux(): Base class method nucleus_flux called.')

    def total_flux(self, E):
        """Returns total flux of nuclei, the "all-particle-flux".

        Args:
          E (float): laboratory energy of particles in GeV
        Returns:
          (float): particle flux in :math:`\\Phi_{particles}` in
          :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """

        nuc_flux = np.vectorize(self.nucleus_flux)
        return sum([nuc_flux(corsika_id, E)
                    for corsika_id in self.nucleus_ids])

    def tot_nucleon_flux(self, E):
        """Returns total flux of nucleons, the "all-nucleon-flux".

        Args:
          E (float): laboratory energy of nucleons in GeV
        Returns:
          (float): nucleon flux :math:`\\Phi_{nucleons}` in
          :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        nuc_flux = np.vectorize(self.nucleus_flux)
        return sum([self.Z_A(corsika_id)[1] ** 2.0 *
                    nuc_flux(corsika_id, E * self.Z_A(corsika_id)[1])
                    for corsika_id in self.nucleus_ids])

    def nucleon_gamma(self, E, rel_delta=0.01):
        """Returns differential spectral index of all-nucleon-flux
        obtained from a numerical derivative.

        Args:
          E (float): laboratory energy of nucleons in GeV
          rel_delta (float): range of derivative relative to log10(E)
        Returns:
          (float): spectral index :math:`\\gamma` of nucleons
        """
        delta = rel_delta * E
        fl = self.tot_nucleon_flux
        return (np.log10(fl(E + delta) / fl(E - delta)) /
                np.log10((E + delta) / (E - delta)))

    def nucleus_gamma(self, E, corsika_id, rel_delta=0.01):
        """Returns differential spectral index of nuclei
        obtained from a numerical derivative.

        Args:
          E (float): laboratory energy of nuclei in GeV
          corsika_id (int): corsika id of nucleus/mass group
          rel_delta (float): range of derivative relative to log10(E)
        Returns:
          (float): spectral index :math:`\\gamma` of nuclei
        """
        delta = rel_delta * E
        fl = np.vectorize(self.nucleus_flux)
        return (np.log10(fl(corsika_id, E + delta) /
                         fl(corsika_id, E - delta)) /
                np.log10((E + delta) / (E - delta)))

    def delta_0(self, E):
        """Returns proton excess.

        The proton excess is defined as
        :math:`\\delta_0 = \\frac{\\Phi_p - \\Phi_n}{\\Phi_p + \\Phi_n}`.

        Args:
          E (float): laboratory energy of nucleons in GeV
        Returns:
          (float): proton excess :math:`\\delta_0`
        """
        p_0 = 0.0
        n_0 = 0.0

        p_0 += self.nucleus_flux(14, E)

        p_0 += 2.**2 * self.nucleus_flux(402, E * 4.)
        n_0 += 2.**2 * self.nucleus_flux(402, E * 4.)

        p_0 += 6.**2 * self.nucleus_flux(1206, E * 12.)
        n_0 += 6.**2 * self.nucleus_flux(1206, E * 12.)

        p_0 += 14.**2 * self.nucleus_flux(2814, E * 28.)
        n_0 += 14.**2 * self.nucleus_flux(2814, E * 28.)

        p_0 += 26.**2 * self.nucleus_flux(5226, E * 52.)
        n_0 += 26.**2 * self.nucleus_flux(5226, E * 52.)

        return (p_0 - n_0) / (p_0 + n_0)

    def p_and_n_flux(self, E):
        """Returns tuple with proton fraction, proton flux and neutron flux.

        The proton fraction is defined as :math:`\\frac{\\Phi_p}{\\Phi_p + \\Phi_n}`.
        The calculation assumes that half of nuclear isotope consist of protons,
        the other half of neutrons.

        Args:
          E (float): laboratory energy of nucleons in GeV
        Returns:
          (float,float,float): proton fraction, proton flux, neutron flux
        """
        nuc_flux = np.vectorize(self.nucleus_flux)
        za = self.Z_A
        p_flux = 0.0
        # protons
        p_flux += self.nucleus_flux(14, E)

        n_flux = 0.5 * sum([za(corsika_id)[1] ** 2.0 * nuc_flux(
            corsika_id, E * za(corsika_id)[1])
            for corsika_id in self.nucleus_ids if corsika_id != 14])

        # protons from other nuclei
        p_flux += n_flux

        return p_flux / (p_flux + n_flux), p_flux, n_flux

    def _find_nearby_id(self, corsika_id, delta_A=3):
        """Looks in :attr:`self.params` for a nucleus with same ``corsika_id`` and returns
        the corsika_id if these parameters exist. If not, it will look for nuclei of
        mass number +- delta_A around the requested nucleus and return its corsika_id
        if it exists.

        Args:
          corsika_id (int): corsika id of nucleus/mass group
        Returns:
          (int): corsika_id of requested or similar nucleus
        Raises:
          Exception: if no nucleus with mass number closer than delta_A can be found
           in parameters
        """
        if corsika_id in self.nucleus_ids:
            return corsika_id

        A_in = (corsika_id - corsika_id % 100) / 100
        closest_id = _get_closest(corsika_id, self.nucleus_ids)[1]
        A_close = (closest_id - closest_id % 100) / 100
        if np.abs(A_in - A_close) > 3:
            e = ('{0}::_find_nearby_id(): No similar nucleus found with ' +
                 'delta_A <= {1} for A_in = {2}. Closest is {3}.')
            raise Exception(
                e.format(self.__class__.__name__, delta_A, A_in, A_close))
        else:
            return A_close

    def Z_A(self, corsika_id):
        """Returns mass number :math:`A` and charge :math:`Z` corresponding
        to ``corsika_id``

        Args:
          corsika_id (int): corsika id of nucleus/mass group
        Returns:
          (int,int): (Z,A) tuple
        """
        Z, A = 1, 1
        if corsika_id > 14:
            Z = corsika_id % 100
            A = (corsika_id - Z) / 100
        return Z, A


class PolyGonato(PrimaryFlux):

    """J. R. Hoerandel, Astroparticle Physics 19, 193 (2003).
        """

    def __init__(self, constdelta=False):
        PrimaryFlux.__init__(self)

        self.name = 'poly-gonato'
        self.sname = 'pg'
        self.constdelta = constdelta
        self.E_p = 4.51e6
        self.gamma_c = -4.68
        self.epsilon_c = 1.87
        self.delta_gamma = 2.10
        if self.constdelta:
            self.epsilon_c = 1.90
            self.E_p = 4.49e6

        self.params = {}

        self.params[14] = (8.73e-2, 2.71, 1)  # H
        self.params[402] = (5.71e-2, 2.64, 2)  # He
        self.params[1206] = (1.06e-2, 2.66, 6)  # C
        self.params[1407] = (2.35e-3, 2.72, 7)  # N
        self.params[1608] = (1.57e-2, 2.68, 8)  # O
        self.params[2412] = (8.01e-3, 2.64, 12)  # Mg
        self.params[2613] = (1.15e-3, 2.66, 13)  # Al
        self.params[2814] = (7.96e-3, 2.75, 14)  # Si
        self.params[5025] = (1.35e-3, 2.46, 25)  # Mn
        self.params[5226] = (2.04e-2, 2.59, 26)  # Fe
        self.params[5427] = (7.51e-5, 2.72, 27)  # Co

        self.nucleus_ids = self.params.keys()

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        return self.PolyGonatoFormula(corsika_id, E)

    def PolyGonatoFormula(self, corsika_id, E):
        p = self.params[corsika_id]
        gam = (self.gamma_c + p[1]) if self.constdelta else -self.delta_gamma

        return (p[0] / 1000.0 * (E / 1000.0) ** (-p[1]) *
                (1 + (E / p[2] / self.E_p) ** self.epsilon_c) **
                (gam / self.epsilon_c))


class _BenzviMontaruli(PrimaryFlux):

    """http://wiki.icecube.wisc.edu/index.php/Composition
    Project started by Segev BenZvi and T. Montaruli but now it
    is not maintained and rather canceled. Feb. 2014
        """

    def __init__(self, model=None):
        self.name = 'Benzvi-Montaruli'
        self.sname = 'BM'
        self.params = {}
        # dictionary[corsika_id] = (E_0, A, gamma1, Eb, gamma2)
        self.params[14] = (31.623, 1.145, -2.786, 262.628, -2.695)  # H
        self.params[402] = (100.0, 0.030, -2.712, 484.059, -2.600)  # He
        self.params[1206] = (240.0, 6.014e-4, -2.741, 2400.0, -2.503)  # C
        self.params[1608] = (320.0, 3.828e-5, -2.741, 3200.0, -2.503)  # O
        self.params[2412] = (400.0, 4.458e-5, -2.741, 4800.0, -2.503)  # Mg
        self.params[2814] = (560.0, 4.314e-5, -2.741, 5600.0, -2.503)  # Si
        self.params[5226] = (
            1120.0, 1.659e-5, -2.741, 11200.0, -2.503)  # Fe

        self.nucleus_ids = self.params.keys()

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        return self.BenzviFlux(corsika_id, E)

    def BenzviFlux(self, corsika_id, E):

        param = self.params[corsika_id]
        try:
            if E < param[3]:
                return param[1] * (E / param[0]) ** param[2]
            else:
                return (param[1] * (param[3] / param[0]) **
                        (param[2] - param[4]) * (E / param[0]) ** param[4])
        except:
            return np.hstack(
                [param[1] * (E[E < param[3]] / param[0]) ** param[2],
                 param[1] * (param[3] / param[0]) ** (param[2] - param[4])
                 * (E[E >= param[3]] / param[0]) ** param[4]])


class HillasGaisser2012(PrimaryFlux):

    """Gaisser, T.K., Astroparticle Physics 35, 801 (2012).

    Model is based on Hillas ideas and eye-ball fits by T.K. Gaisser.
    H3a is a 3-'peters cycle' 5 mass group model with mixed
    composition above the ankle. H4a has protons only in the
    3. component.

    Args:
      model (str): can be either H3a or H4a.
    """

    def __init__(self, model="H4a"):

        self.name = 'Hillas-Gaisser (' + model + ')'
        self.sname = model
        self.model = model
        self.params = {}
        self.rid_cutoff = {}

        mass_comp = [14, 402, 1206, 2814, 5226]
        for mcomp in mass_comp:
            self.params[mcomp] = {}

        self.rid_cutoff[1] = 4e6
        self.rid_cutoff[2] = 30e6
        self.rid_cutoff[3] = 2e9
        self.params[14][1] = (7860, 1.66, 1)  # H
        self.params[402][1] = (3550, 1.58, 2)  # He
        self.params[1206][1] = (2200, 1.63, 6)  # CNO
        self.params[2814][1] = (1430, 1.67, 14)  # MgAlSi
        self.params[5226][1] = (2120, 1.63, 26)  # Fe

        self.params[14][2] = (20, 1.4, 1)  # H
        self.params[402][2] = (20, 1.4, 2)  # He
        self.params[1206][2] = (13.4, 1.4, 6)  # CNO
        self.params[2814][2] = (13.4, 1.4, 14)  # MgAlSi
        self.params[5226][2] = (13.4, 1.4, 26)  # Fe

        if self.model == "H3a":
            self.rid_cutoff[3] = 2e9
            self.params[14][3] = (1.7, 1.4, 1)  # H
            self.params[402][3] = (1.7, 1.4, 2)  # He
            self.params[1206][3] = (1.14, 1.4, 6)  # CNO
            self.params[2814][3] = (1.14, 1.4, 14)  # MgAlSi
            self.params[5226][3] = (1.14, 1.4, 26)  # Fe
        elif self.model == "H4a":
            self.rid_cutoff[3] = 60e9
            self.params[14][3] = (200., 1.6, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[2814][3] = (0, 1.4, 14)  # MgAlSi
            self.params[5226][3] = (0, 1.4, 26)  # Fe
        else:
            raise Exception(
                'HillasGaisser2012(): Unknown model version requested.')

        self.nucleus_ids = self.params.keys()

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0
        for i in range(1, 4):
            p = self.params[corsika_id][i]
            flux += p[0] * E ** (-p[1] - 1.0) * \
                np.exp(-E / p[2] / self.rid_cutoff[i])
        return flux


class GaisserStanevTilav(PrimaryFlux):

    """T. K. Gaisser, T. Stanev, and S. Tilav, arXiv:1303.3565, (2013).

    Args:
      model (str): 3-gen or 4-gen

    Raises:
      Exception: if ``model`` not properly specified.
    """

    def __init__(self, model="3-gen"):
        PrimaryFlux.__init__(self)

        self.name = 'GST (' + model + ')'
        self.sname = 'GST' + model[0]
        self.model = model
        self.params = {}
        self.rid_cutoff = {}

        self.rid_cutoff[1] = 120e3
        self.rid_cutoff[2] = 4e6

        mass_comp = [14, 402, 1206, 1608, 5226]
        for mcomp in mass_comp:
            self.params[mcomp] = {}

        self.params[14][1] = (7000, 1.66, 1)  # H
        self.params[402][1] = (3200, 1.58, 2)  # He
        self.params[1206][1] = (100, 1.4, 6)  # C
        self.params[1608][1] = (130, 1.4, 8)  # O
        self.params[5226][1] = (60, 1.3, 26)  # Fe

        self.params[14][2] = (150, 1.4, 1)  # H
        self.params[402][2] = (65, 1.3, 2)  # He
        self.params[1206][2] = (6, 1.3, 6)  # C
        self.params[1608][2] = (7, 1.3, 8)  # O

        if self.model == "3-gen":
            self.params[5226][2] = (2.3, 1.2, 26)  # Fe
            self.rid_cutoff[3] = 1.3e9

            self.params[14][3] = (14, 1.4, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[1608][3] = (0, 1.3, 8)  # O
            self.params[5226][3] = (0.025, 1.2, 26)  # Fe

        elif self.model == "4-gen":
            self.params[5226][2] = (2.1, 1.2, 26)  # Fe

            self.rid_cutoff[3] = 1.5e9
            self.params[14][3] = (12., 1.4, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[1608][3] = (0, 1.3, 8)  # O
            self.params[5226][3] = (0.011, 1.2, 26)  # Fe

            self.rid_cutoff[4] = 40e9
            self.params[14][4] = (1.2, 1.4, 1)  # H
            self.params[402][4] = (0, 0, 2)  # He
            self.params[1206][4] = (0, 0, 6)  # CNO
            self.params[1608][4] = (0, 0, 8)  # O
            self.params[5226][4] = (0, 0, 26)  # Fe
        else:
            raise Exception(
                'GaisserStanevTilav(): Unknown model version.')

        self.nucleus_ids = self.params.keys()

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0
        ngen = 0

        if self.model == '3-gen':
            ngen = 4
        elif self.model == '4-gen':
            ngen = 5
        else:
            raise Exception('GaisserStanevTilav(): Unknown model type.')

        for i in range(1, ngen):
            p = self.params[corsika_id][i]
            flux += p[0] * E ** (-p[1] - 1.0) * \
                np.exp(-E / p[2] / self.rid_cutoff[i])
        return flux


class CombinedGHandHG(PrimaryFlux):

    """A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 (2012).

    In the paper the high energy models were called cHGm for GH+H3a and cHGp for GH+H4a.
    The names have been change to use the quite unintuitive names H3a and H4a in
    ongoing literature.

    """

    def __init__(self, model="H3a"):
        self.name = 'comb. GH and ' + model
        self.sname = 'c' + model
        self.params = {}
        self.leModel = GaisserHonda()
        self.heModel = HillasGaisser2012(model)
        self.heCutOff = 1e5
        cid_list = [14, 402, 1206, 2814, 5226]

        # Store low- to high-energy model transitions in params
        for cid in cid_list:
            self.params[cid] = self.FindTransition(cid)
        self.nucleus_ids = self.params.keys()

    def FindTransition(self, corsika_id):
        from scipy.optimize import fsolve

        def func(logE):
            return (self.leModel.nucleus_flux(corsika_id, 10 ** logE) -
                    self.heModel.nucleus_flux(corsika_id, 10 ** logE))

        result = fsolve(func, 3.1)
        # print 'CombinedSpectrum(): low E to high E model transition for',
        # corsika_id, 10 ** result[0]
        return 10 ** result[0]

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        try:
            E = np.array(E)
            le = E < self.params[corsika_id]
            he = E >= self.params[corsika_id]
            return np.hstack((self.leModel.nucleus_flux(corsika_id, E[le]),
                              self.heModel.nucleus_flux(corsika_id, E[he])))
        except IndexError:
            if E < self.params[corsika_id]:
                return self.leModel.nucleus_flux(corsika_id, E)
            else:
                return self.heModel.nucleus_flux(corsika_id, E)


class ZatsepinSokolskaya(PrimaryFlux):

    """The model was first released in V. I. Zatsepin and N. V. Sokolskaya,
    Astronomy and Astrophysics 458, 1 (2006).

    Later, the PAMELA experiment has fitted the parameters of this model
    to their data in PAMELA Collaboration, O. Adriani et al., Science 332, 69 (2011).
    Both versions of parameters can be accessed here.

    The model does not describe the flux above the knee. Therefore, the highest energies
    should not exceed 1-10 PeV.

    Args:
      model (str): 'default' for original or 'pamela' for PAMELA parameters
    """

    def __init__(self, model='pamela'):
        if model == 'pamela':
            self.name = 'Zatsepin-Sokolskaya/Pamela'
            self.sname = 'ZSP'
            self.R_0 = 5.5
            self.alpha = (2.3, 2.1, 2.57)
            self.R_max = (8e4, 4e6, 2e2)
            self.gamma = (2.63, 2.43, 2.9)
            self.gamma_k = (8, 4.5, 4.5)
            self.f_norm = {}
            self.f_norm[14] = (7.1e3, 6.25e3, 3.0, 74., 1)
            self.f_norm[402] = (9.5e3, 8.5e3, 0.74, 18., 2)
            self.f_norm[1206] = (6.75e3, 1.8e3, 30, 5.8, 7)
            self.f_norm[2814] = (5.5e3, 1.5e3, 110, 3.5, 12)
            self.f_norm[5226] = (3.5e3, 1.2e3, 750, 2.4, 20)
            self.m_p = 0.983
        elif model == 'default':
            self.name = 'Zatsepin-Sokolskaya'
            self.sname = 'ZS'
            self.R_0 = 5.5
            self.alpha = (2.3, 2.1, 2.57)
            self.R_max = (8e4, 4e6, 2e2)
            self.gamma = (2.63, 2.43, 2.9)
            self.gamma_k = (8, 4.5, 4.5)
            self.f_norm = {}
            self.f_norm[14] = (1.36e4, 6.25e3, 2.1, 74., 1)
            self.f_norm[402] = (8.75e3, 8.5e3, 3.0, 18., 2)
            self.f_norm[1206] = (6.75e3, 1.8e3, 30, 5.8, 7)
            self.f_norm[2814] = (5.5e3, 1.5e3, 110, 3.5, 12)
            self.f_norm[5226] = (3.5e3, 1.2e3, 750, 2.4, 20)
            self.m_p = 0.983
        else:
            raise Exception("{0}():: Unknown model selection '{1}'.".format(
                self.__class__.__name__, model))
        self.nucleus_ids = self.f_norm.keys()

    def lamba_esc(self, R):
        return (4.2 * (R / self.R_0) ** (-1. / 3.) *
                (1 + (R / self.R_0) ** (-2. / 3.)))

    def Q(self, R, gen):
        return R ** (-self.alpha[gen]) * self.phi(R, gen)

    def phi(self, R, gen):

        return ((1 + (R / self.R_max[gen]) ** 2) **
                ((self.gamma[gen] - self.gamma_k[gen]) / 2.))

    def dR_dE(self, E, corsika_id):
        Z, A = self.Z_A(corsika_id)
        return (1. / Z * (E + self.m_p * A) /
                np.sqrt(E ** 2 + 2 * self.m_p * A * E))

    def R(self, E, corsika_id):
        Z, A = self.Z_A(corsika_id)
        return 1. / Z * np.sqrt(E ** 2 + 2 * self.m_p * A * E)

    def f_mod(self, E, corsika_id):
        Z = self.Z_A(corsika_id)[0]

        P = ((E ** 2 + 2 * self.m_p * E) /
             ((E + Z * 0.511e-3 * 0.6) ** 2 +
              2 * self.m_p * (E + Z * 0.511e-3 * 0.6)))
        return self.flux(E + Z * 0.511e-3 * 0.6, corsika_id) * P

    def dN_dR(self, R, corsika_id, gen):
        return (self.Q(R, gen) * self.lamba_esc(R) /
                (1 + self.lamba_esc(R) / self.f_norm[corsika_id][3]))

    def dN_dE(self, E, corsika_id, gen):
        return (self.dR_dE(E, corsika_id) *
                self.dN_dR(self.R(E, corsika_id), corsika_id, gen))

    def flux(self, E, corsika_id):
        flux = 0.0
        for gen in range(3):
            flux += (self.f_norm[corsika_id][gen] * 1e4 ** (-2.75) /
                     self.dN_dE(1e4, corsika_id, gen) *
                     self.dN_dE(E, corsika_id, gen))
        return flux

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        try:
            le = E < 300
            he = E >= 300
            return np.hstack((self.f_mod(E[le], corsika_id),
                              self.flux(E[he], corsika_id)))

        except:
            if E < 300:
                return self.f_mod(E, corsika_id)
            else:
                return self.flux(E, corsika_id)


class GaisserHonda(PrimaryFlux):

    """5 mass group single power-law model from T.K. Gaisser and M. Honda,
    Annual Review of Nuclear and Particle Science 52, 153 (2002)

    This model was tuned to lower energy baloon data. It fails to
    describe the flux at and above the knee. A safe range for using
    it is < 100 TeV/nucleon.
    """

    def __init__(self, opt=None):

        self.name = 'Gaisser-Honda'
        self.sname = 'GH'
        self.params = {}
        self.params[14] = (2.74, 14900, 2.15, 0.21)
        self.params[402] = (2.64, 600, 1.25, 0.14)
        self.params[1206] = (2.60, 33.2, 0.97, 0.01)
        self.params[2814] = (2.79 + 0.08, 34.2 - 6.0, 2.14, 0.01)
        self.params[5226] = (2.68, 4.45, 3.07, 0.41)
        self.nucleus_ids = self.params.keys()

    def nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        A = self.Z_A(corsika_id)[1]

        alpha = self.params[corsika_id][0]
        K = self.params[corsika_id][1]
        b = self.params[corsika_id][2]
        c = self.params[corsika_id][3]

        return K / A * (E / A + b * np.exp(-c * np.sqrt(E / A))) ** -alpha


class Thunman(PrimaryFlux):

    """Popular broken power-law flux model.

    The parameters of this model are taken from the prompt flux calculation
    paper by M. Thunman, G. Ingelman, and P. Gondolo, Astroparticle Physics 5, 309 (1996).
    The model contians only protons with a power-law index of -2.7 below the knee,
    located at 5 PeV, and -3.0 for energies higher than that.
    """

    name = "Thunman et al. ('96)"
    sname = 'TIG'

    def __init__(self, opt=None):
        self.params = {}
        self.params["low_e"] = (1e4 * 1.7, -2.7)
        self.params["high_e"] = (1e4 * 174, -3.0)
        self.params["trans"] = 5e6
        self.nucleus_flux = np.vectorize(self.nucleus_flux)

        self.nucleus_ids = [14]

    def nucleus_flux(self, corsika_id, E):
        if corsika_id != 14:
            return 0.0
        if E < self.params["trans"]:
            return self.params['low_e'][0] * E ** (self.params['low_e'][1])
        else:
            return self.params['high_e'][0] * E ** (self.params['high_e'][1])


class SimplePowerlaw27(PrimaryFlux):

    """Simple E**-2.7 parametrization based on values from
    :class:`Thunman` below knee.
    """

    name = r"$E^{-2.7}$"
    sname = r"$E^{-2.7}$"

    def __init__(self, opt=None):
        self.params = (1e4 * 1.7, -2.7)
        self.nucleus_flux = np.vectorize(self.nucleus_flux)
        self.nucleus_ids = [14]

    def nucleus_flux(self, corsika_id, E):
        if corsika_id != 14:
            return 0.0

        return self.params[0] * E ** (self.params[1])


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    pmodels = [(GaisserStanevTilav, "3-gen", "GST 3-gen"),
               (GaisserStanevTilav, "4-gen", "GST 4-gen"),
               (CombinedGHandHG, "H3a", "cH3a"),
               (CombinedGHandHG, "H4a", "cH4a"),
               (HillasGaisser2012, "H3a", "H3a"),
               (HillasGaisser2012, "H4a", "H4a"),
               (PolyGonato, False, "poly-gonato"),
               (Thunman, None, "TIG"),
               (ZatsepinSokolskaya, 'default', 'ZS'),
               (ZatsepinSokolskaya, 'pamela', 'ZSP'),
               (GaisserHonda, None, 'GH')]

    nfrac = {}
    evec = np.logspace(0, 10, 1000)
    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray nucleon flux (proton + neutron)')
    for mclass, moptions, mtitle in pmodels:
        pmod = mclass(moptions)
        pfrac, p, n = pmod.p_and_n_flux(evec)
        plt.plot(evec, (p + n) * evec ** 2.5, ls='-', lw=1.5, label=mtitle)
        nfrac[mtitle] = (1 - pfrac)

    plt.loglog()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.ylim([10, 2e4])
    plt.tight_layout()

    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray particle flux (all-nuclei).')

    for mclass, moptions, mtitle in pmodels:
        pmod = mclass(moptions)

        flux = pmod.total_flux(evec)
        plt.plot(evec, flux * evec ** 2.5, ls='-', lw=1.5, label=mtitle)

    plt.loglog()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.ylim([10, 2e4])
    plt.tight_layout()

    plt.figure(figsize=(7.5, 5))
    plt.title('Fraction of neutrons relative to protons.')
    for mclass, moptions, mtitle in pmodels:
        plt.plot(evec, nfrac[mtitle], ls='-', lw=1.5, label=mtitle)

    plt.semilogx()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel("Neutron fraction")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e10])
    plt.tight_layout()
    plt.show()
