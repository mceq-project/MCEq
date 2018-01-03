# -*- coding: utf-8 -*-
"""
:mod:`MCEq.misc` - other useful things
======================================

Some helper functions and plotting features are collected in this module.

"""

import numpy as np
from mceq_config import dbg


def normalize_hadronic_model_name(name):
    """Converts hadronic model name into standard form"""
    return name.translate(None, ".-").upper()


def theta_deg(cos_theta):
    """Converts :math:`\\cos{\\theta}` to :math:`\\theta` in degrees.
    """
    return np.arccos(cos_theta) * 180. / np.pi


def theta_rad(theta):
    """Converts :math:`\\theta` from rad to degrees.
    """
    return theta / 180. * np.pi


def print_in_rows(str_list, n_cols=8):
    """Prints contents of a list in rows `n_cols`
    entries per row.
    """
    l = len(str_list)
    n_full_length = int(l / n_cols)
    n_rest = l % n_cols
    print_str = '\n'
    for i in range(n_full_length):
        print_str += ('"{:}", ' * n_cols
                      ).format(*str_list[i * n_cols:(i + 1) * n_cols]) + '\n'
    print_str += ('"{:}", ' * n_rest).format(*str_list[-n_rest:])

    print print_str.strip()[:-1]


def set_ticks(which, n_divs=5, ax=None):
    """Helps to control the number of ticks on x and y axes.

    Args:
        which (str): can be `x` or `y` or `both`
    """
    from matplotlib.pyplot import gca
    from matplotlib.ticker import AutoMinorLocator
    if ax is None:
        ax = gca()
    if which not in ['x', 'y', 'both']:
        print 'Warning: undefined axis', which, 'when adjusting ticks.'
    if which in ['x', 'both']:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_divs))
    if which in ['y', 'both']:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_divs))


def cornertext(text, loc=2, color=None, frameon=False, axes=None, **kwargs):
    """
    Conveniently places text in a corner of a plot.

    Parameters
    ----------
    text: string or tuple of strings
      Text to be placed in the plot. May be a tuple of strings to get
      several lines of text.
    loc: integer or string
      Location of text, same as in legend(...).
    frameon: boolean (optional)
      Whether to draw a border around the text. Default is False.
    axes: Axes (optional, default: None)
      Axes object which houses the text (defaults to the current axes).
    fontproperties: matplotlib.font_manager.FontProperties object
      Change the font style.

    Other keyword arguments are forwarded to the text instance.

    Authors
    -------
    Hans Dembinski <hans.dembinski@kit.edu>
    """

    from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, TextArea
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    import warnings

    if axes is None:
        from matplotlib import pyplot as plt
        axes = plt.gca()

    locTranslate = {
        'upper right': 1,
        'upper left': 2,
        'lower left': 3,
        'lower right': 4,
        'right': 5,
        'center left': 6,
        'center right': 7,
        'lower center': 8,
        'upper center': 9,
        'center': 10
    }

    if isinstance(loc, str):
        if loc in locTranslate:
            loc = locTranslate[loc]
        else:
            message = ('Unrecognized location "{0:s}". Falling back on ' +
                       '"upper left"; valid locations are\n\t{1:s}').format(
                           loc, '\n\t'.join(locTranslate.keys()))
            warnings.warn(message)
            loc = 2

    if "borderpad" in kwargs:
        borderpad = kwargs["borderpad"]
    else:
        borderpad = rcParams["legend.borderpad"]

    if "borderaxespad" in kwargs:
        borderaxespad = kwargs["borderaxespad"]
    else:
        borderaxespad = rcParams["legend.borderaxespad"]

    if "handletextpad" in kwargs:
        handletextpad = kwargs["handletextpad"]
    else:
        handletextpad = rcParams["legend.handletextpad"]

    if "fontproperties" in kwargs:
        fontproperties = kwargs["fontproperties"]  # @UnusedVariable
        del kwargs["fontproperties"]
    else:
        if "size" in kwargs:
            size = kwargs["size"]
            del kwargs["size"]
        elif "fontsize" in kwargs:
            size = kwargs["fontsize"]
            del kwargs["fontsize"]
        else:
            size = rcParams["legend.fontsize"]
    fontproperties = FontProperties(size=size)

    texts = [text] if isinstance(text, str) else text

    colors = [color for t in texts] if (isinstance(color, str) or
                                        color is None) else color

    tas = []
    for t, c in zip(texts, colors):
        ta = TextArea(
            t,
            textprops={"color": c,
                       "fontproperties": fontproperties},
            multilinebaseline=True,
            minimumdescent=True,
            **kwargs)
        tas.append(ta)

    vpack = VPacker(children=tas, pad=0, sep=handletextpad)

    aob = AnchoredOffsetbox(
        loc,
        child=vpack,
        pad=borderpad,
        borderpad=borderaxespad,
        frameon=frameon)

    axes.add_artist(aob)
    return aob


def plot_hist(xedges, ws, axes=None, facecolor=None, **kwargs):
    """Plots histogram data in ROOT style.

    Parameters
    ----------
    xedge: lower bin boundaries + upper boundary of last bin
    w: content of the bins
    """

    if axes is None:
        from matplotlib import pyplot as plt
        axes = plt.gca()

    import numpy as np

    m = len(ws)
    n = 2 * m + 2

    xs = np.zeros(n)
    ys = np.zeros(n)

    xs[0] = xedges[0]
    xs[-1] = xedges[-1]

    for i in xrange(m):
        xs[1 + 2 * i] = xedges[i]
        ys[1 + 2 * i] = ws[i]
        xs[1 + 2 * i + 1] = xedges[i + 1]
        ys[1 + 2 * i + 1] = ws[i]

    if not facecolor is None:
        return axes.fill(xs, ys, facecolor=facecolor, **kwargs)
    else:
        return axes.plot(xs, ys, **kwargs)


def _get_closest(value, in_list):
    """Returns the closes value to 'value' from given list."""

    minindex = np.argmin(np.abs(in_list - value * np.ones(len(in_list))))
    return minindex, in_list[minindex]


def get_bins_and_width_from_centers(vector):
    """Returns bins and bin widths given given bin centers."""

    vector_log = np.log10(vector)
    steps = vector_log[1] - vector_log[0]
    bins_log = vector_log - 0.5 * steps
    bins_log = np.resize(bins_log, vector_log.size + 1)
    bins_log[-1] = vector_log[-1] + 0.5 * steps
    bins = 10**bins_log
    widths = bins[1:] - bins[:-1]
    return bins, widths

class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
    """

    def __init__(self, lower, upper, bins_dec):
        import numpy as np
        self.bins = np.logspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(1, 'Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
            self.bins[0], self.bins[-1], self.grid.size))
            
class EdepZFactors():
    """Handles calculation of energy dependent Z factors.

    Was not recently checked and results could be wrong."""

    def __init__(self, interaction_model, primary_flux_model):
        from MCEq.data import InteractionYields, HadAirCrossSections
        from ParticleDataTool import SibyllParticleTable
        from misc import get_bins_and_width_from_centers

        self.y = InteractionYields(interaction_model)
        self.cs = HadAirCrossSections(interaction_model)

        self.pm = primary_flux_model
        self.e_bins, self.e_widths = get_bins_and_width_from_centers(
            self.y.e_grid)
        self.e_vec = self.y.e_grid
        self.iamod = interaction_model
        self.sibtab = SibyllParticleTable()
        self._gen_integrator()

    def get_zfactor(self, proj, sec_hadr, logx=False, use_cs=True):
        proj_cs_vec = self.cs.get_cs(proj)
        nuc_flux = self.pm.tot_nucleon_flux(self.e_vec)
        zfac = np.zeros(self.y.dim)
        sec_hadr = sec_hadr
        if self.y.is_yield(proj, sec_hadr):
            if dbg > 1:
                print(("EdepZFactors::get_zfactor(): " +
                       "calculating zfactor Z({0},{1})").format())
            y_mat = self.y.get_y_matrix(proj, sec_hadr)

            self.calculate_zfac(self.e_vec, self.e_widths, nuc_flux,
                                proj_cs_vec, y_mat, zfac, use_cs)

        if logx:
            return np.log10(self.e_vec), zfac
        return self.e_vec, zfac

    def _gen_integrator(self):
        try:
            from numba import jit, double, boolean, void

            @jit(
                void(double[:], double[:], double[:], double[:], double[:, :],
                     double[:], boolean),
                target='cpu')
            def calculate_zfac(e_vec, e_widths, nuc_flux, proj_cs, y, zfac,
                               use_cs):
                for h, E_h in enumerate(e_vec):
                    for k in range(len(e_vec)):
                        E_k = e_vec[k]
                        # dE_k = e_widths[k]
                        if E_k < E_h:
                            continue
                        csfac = proj_cs[k] / proj_cs[h] if use_cs else 1.

                        zfac[h] += nuc_flux[k] / nuc_flux[h] * csfac * \
                            y[:, k][h]  # * dE_k
        except ImportError:
            print("Warning! Numba not in PYTHONPATH. ZFactor " +
                  "calculation won't work.")

        self.calculate_zfac = calculate_zfac
