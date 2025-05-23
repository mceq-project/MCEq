"""
:mod:`MCEq.charm_models` --- charmed particle production
========================================================

This module includes classes for custom charmed particle
production. Currently only the MRS model is implemented
as the class :class:`MRS_charm`. The abstract class
:class:`CharmModel` guides the implementation of custom
classes.

The :class:`Yields` instantiates derived classes of
:class:`CharmModel` and calls :func:`CharmModel.get_yield_matrix`
when overwriting a model yield file in
:func:`Yields.set_custom_charm_model`.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from six import with_metaclass

from MCEq.misc import info


class CharmModel(with_metaclass(ABCMeta)):
    """Abstract class, from which implemeted charm models can inherit.

    Note:
      Do not instantiate this class directly.

    """

    @abstractmethod
    def get_yield_matrix(self, proj, sec):
        """The implementation of this abstract method returns
        the yield matrix spanned over the energy grid of the calculation.

        Args:
           proj (int): PDG ID of the interacting particle (projectile)
           sec (int): PDG ID of the final state charmed meson (secondary)

        Returns:
           np.array: yield matrix

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(
            "CharmModel::get_yield_matrix(): " + "Base class called."
        )


class MRS_charm(CharmModel):
    """Martin-Ryskin-Stasto charm model.

    The model is described in A. D. Martin, M. G. Ryskin,
    and A. M. Stasto, Acta Physica Polonica B 34, 3273 (2003).
    The parameterization of the inclusive :math:`c\\bar{c}`
    cross-section is given in the appendix of the paper.
    This formula provides the behavior of the cross-section,
    while fragmentation functions and certain scales are
    needed to obtain meson and baryon fluxes as a function
    of the kinematic variable :math:`x_F`. At high energies
    and :math:`x_F > 0.05`, where this model is valid,
    :math:`x_F \\approx x=E_c/E_{proj}`.
    Here, these fragmentation functions are used:

        - :math:`D`-mesons :math:`\\frac{4}{3} x`
        - :math:`\\Lambda`-baryons :math:`\\frac{1}{1.47} x`

    The production ratios between the different types of
    :math:`D`-mesons are stored in the attribute :attr:`cs_scales`
    and :attr:`D0_scale`, where :attr:`D0_scale` is the
    :math:`c\\bar{c}` to :math:`D^0` ratio and :attr:`cs_scales`
    stores the production ratios of :math:`D^\\pm/D^0`,
    :math:`D_s/D^0` and :math:`\\Lambda_c/D^0`.

    Since the model employs only perturbartive production of
    charm, the charge conjugates are symmetric, i.e.
    :math:`\\sigma_{D^+} = \\sigma_{D^-}` etc.

    Args:
      e_grid (np.array): energy grid as it is defined in
                         :class:`MCEqRun`.
      csm (np.array): inelastic cross-sections as used in
                      :class:`MCEqRun`.
    """

    #: fractions of cross-section wrt to D0 cross-section
    cs_scales = {421: 1.0, 411: 0.5, 431: 0.15, 4122: 0.45}
    #: D0 cross-section wrt to the ccbar cross-section
    D0_scale = 1.0 / 2.1

    #: hadron projectiles, which are allowed to produce charm
    allowed_proj = [2212, -2212, 2112, -2112, 211, -211, 321, -321]

    #: charm secondaries, which are predicted by this model
    allowed_sec = [411, 421, 431, 4122]

    def __init__(self, e_grid, csm):

        # inverted fragmentation functions
        self.lambda_c_frag = lambda xhad: 1 / 1.47 * xhad
        self.d_frag = lambda xhad: 4.0 / 3.0 * xhad

        self.e_grid = e_grid
        self.d = e_grid.size
        self.no_prod = np.zeros(self.d**2).reshape(self.d, self.d)
        self.siginel = csm.get_cs(2212, mbarn=True)

    def sigma_cc(self, E):
        """Returns the integrated ccbar cross-section in mb.

        Note:
          Integration is not going over complete phase-space due to
          limitations of the parameterization.
        """
        from scipy.integrate import quad

        E = np.asarray(E)
        if E.size > 1:
            return 2 * np.array([quad(self.dsig_dx, 0.05, 0.6, args=Ei)[0] for Ei in E])
        return 2 * quad(self.dsig_dx, 0.05, 0.6, args=E)[0]

    def dsig_dx(self, x, E):
        """Returns the Feynman-:math:`x_F` distribution
        of :math:`\\sigma_{c\\bar{c}}` in mb

        Args:
          x (float or np.array): :math:`x_F`
          E (float): center-of-mass energy in GeV

        Returns:
          float: :math:`\\sigma_{c\\bar{c}}` in mb
        """

        x = np.asarray(x)
        E = np.asarray(E)
        beta = 0.05 - 0.016 * np.log(E / 10e4)
        n, A = None, None
        if E < 1e4:
            return 0.0
        if E >= 1e4 and E < 1e8:
            n = 7.6 + 0.025 * np.log(E / 1e4)
            A = 140 + (11.0 * np.log(E / 1e2)) ** 1.65
        elif E >= 1e8 and E <= 1e11:
            n = 7.6 + 0.012 * np.log(E / 1e4)
            A = 4100.0 + 245.0 * np.log(E / 1e8)
        else:
            raise Exception("MRS_charm()::out of range")
        res = np.zeros_like(x)
        ran = (x > 0.01) & (x < 0.7)
        res[ran] = np.array(A * x[ran] ** (beta - 1.0) * (1 - x[ran] ** 1.2) ** n / 1e3)
        return res

    def D_dist(self, x, E, mes):
        """Returns the Feynman-:math:`x_F` distribution
        of :math:`\\sigma_{D-mesons}` in mb

        Args:
          x (float or np.array): :math:`x_F`
          E (float): center-of-mass energy in GeV
          mes (int): PDG ID of D-meson: :math:`\\pm421, \\pm431, \\pm411`

        Returns:
          float: :math:`\\sigma_{D-mesons}` in mb
        """
        xc = self.d_frag(x)
        return self.dsig_dx(xc, E) * self.D0_scale * self.cs_scales[mes]

    def LambdaC_dist(self, x, E):
        """Returns the Feynman-:math:`x_F` distribution
        of :math:`\\sigma_{\\Lambda_C}` in mb

        Args:
          x (float or np.array): :math:`x_F`
          E (float): center-of-mass energy in GeV
          mes (int): PDG ID of D-meson: :math:`\\pm421, \\pm431, \\pm411`

        Returns:
          float: :math:`\\sigma_{D-mesons}` in mb
        """
        xc = self.lambda_c_frag(x)
        return self.dsig_dx(xc, E) * self.D0_scale * self.cs_scales[4122]

    def get_yield_matrix(self, proj, sec):
        """Returns the yield matrix in proper format for :class:`MCEqRun`.

        Args:
          proj (int): projectile PDG ID :math:`\\pm` [2212, 211, 321]
          sec (int): charmed particle PDG ID :math:`\\pm` [411, 421, 431, 4122]

        Returns:
          np.array: yield matrix if (proj,sec) combination allowed,
                    else zero matrix
        """
        # TODO: Make this function a member of the base class!

        if (proj not in self.allowed_proj) or (abs(sec) not in self.allowed_sec):
            return self.no_prod

        self.xdist = None

        if abs(sec) == 4122 and ((np.sign(proj) != np.sign(sec)) or abs(proj) < 1000):
            return self.no_prod
        self.xdist = lambda e: self.LambdaC_dist(self.e_grid / e, e) / e
        if abs(sec) != 4122:
            self.xdist = lambda e: self.D_dist(self.e_grid / e, e, abs(sec)) / e

        m_out = np.zeros_like(self.no_prod)

        # convert x distribution to E_sec distribution and distribute on the grid
        for i, e in enumerate(self.e_grid):
            m_out[:, i] = self.xdist(e) / self.siginel[i]

        info(3, f"returning matrix for ({proj},{sec})")

        return m_out

    def test(self):
        """Plots the meson, baryon and charm quark distribution as shown in
        the plot below.

        .. figure:: graphics/MRS_test.png
            :scale: 50 %
            :alt: output of test function

        """
        import matplotlib.pyplot as plt

        xvec = np.linspace(0.0001, 1.0, 20)

        # Energy for plotting inclusive cross-sections
        eprobe = 1e7

        plt.figure(figsize=(8.5, 4))
        plt.subplot(121)
        plt.semilogy(
            xvec, xvec * self.dsig_dx(xvec, eprobe), lw=1.5, label=r"$c$-quark"
        )
        plt.semilogy(
            xvec, xvec * self.D_dist(xvec, eprobe, 421), lw=1.5, label=r"$D^0$"
        )
        plt.semilogy(
            xvec, xvec * self.D_dist(xvec, eprobe, 411), lw=1.5, label=r"$D^+$"
        )
        plt.semilogy(
            xvec, xvec * self.D_dist(xvec, eprobe, 431), lw=1.5, label=r"$Ds^+$"
        )
        plt.semilogy(
            xvec, xvec * self.LambdaC_dist(xvec, 1e4), lw=1.5, label=r"$\Lambda_C^+$"
        )
        plt.legend()
        plt.xlabel(r"$x_F$")
        plt.ylabel(r"inclusive $\sigma$ [mb]")

        plt.subplot(122)
        evec = np.logspace(4, 11, 100)
        plt.loglog(
            np.sqrt(evec), self.sigma_cc(evec), lw=1.5, label=r"$\sigma_{c\bar{c}}$"
        )
        plt.legend()
        plt.xlabel(r"$\sqrt{s}$ [GeV]")
        plt.ylabel(r"$\sigma_{c\bar{c}}$ [mb]")
        plt.tight_layout()


class WHR_charm(MRS_charm):
    """Logan Wille, Francis Halzen, Hall Reno.

    The approach is the same as in  A. D. Martin, M. G. Ryskin,
    and A. M. Stasto, Acta Physica Polonica B 34, 3273 (2003).
    The parameterization of the inclusive :math:`c\\bar{c}`
    cross-section is replaced by interpolated tables from the
    calculation. Fragmentation functions and certain scales are
    needed to obtain meson and baryon fluxes as a function
    of the kinematic variable :math:`x_F`. At high energies
    and :math:`x_F > 0.05`, where this model is valid,
    :math:`x_F \\approx x=E_c/E_{proj}`.
    Here, these fragmentation functions are used:

        - :math:`D`-mesons :math:`\\frac{4}{3} x`
        - :math:`\\Lambda`-baryons :math:`\\frac{1}{1.47} x`

    The production ratios between the different types of
    :math:`D`-mesons are stored in the attribute :attr:`cs_scales`
    and :attr:`D0_scale`, where :attr:`D0_scale` is the
    :math:`c\\bar{c}` to :math:`D^0` ratio and :attr:`cs_scales`
    stores the production ratios of :math:`D^\\pm/D^0`,
    :math:`D_s/D^0` and :math:`\\Lambda_c/D^0`.

    Since the model employs only perturbartive production of
    charm, the charge conjugates are symmetric, i.e.
    :math:`\\sigma_{D^+} = \\sigma_{D^-}` etc.

    Args:
      e_grid (np.array): energy grid as it is defined in
                         :class:`MCEqRun`.
      csm (np.array): inelastic cross-sections as used in
                      :class:`MCEqRun`.
    """

    def __init__(self, e_grid, csm):
        import pickle

        self.sig_table = pickle.load(open("references/logan_charm.ppl", "rb"))
        self.e_idcs = {}
        for i, e in enumerate(e_grid):
            self.e_idcs[e] = i

        MRS_charm.__init__(self, e_grid, csm)

    def dsig_dx(self, x, E):
        """Returns the Feynman-:math:`x_F` distribution
        of :math:`\\sigma_{c\\bar{c}}` in mb

        Args:
          x (float or np.array): :math:`x_F`
          E (float): center-of-mass energy in GeV

        Returns:
          float: :math:`\\sigma_{c\\bar{c}}` in mb
        """
        res = self.sig_table[self.e_idcs[E]](x) * 1e-3  # mub -> mb
        res[res < 0] = 0.0
        return res
