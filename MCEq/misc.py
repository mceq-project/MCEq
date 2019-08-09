
from __future__ import print_function
from collections import namedtuple
import numpy as np
import mceq_config as config

#: Energy grid (centers, bind widths, dimension)
energy_grid = namedtuple("energy_grid", ("c", "b", "w", "d"))

#: Matrix with x_lab=E_child/E_parent values
_xmat = None

def normalize_hadronic_model_name(name):
    import re
    """Converts hadronic model name into standard form"""
    return re.sub('[-.]', '', name).upper()


def theta_deg(cos_theta):
    """Converts :math:`\\cos{\\theta}` to :math:`\\theta` in degrees.
    """
    return np.rad2deg(np.arccos(cos_theta))


def theta_rad(theta):
    """Converts :math:`\\theta` from rad to degrees.
    """
    return np.deg2rad(theta)


def gen_xmat(energy_grid):
    """Generates x_lab matrix for a given energy grid"""
    global _xmat
    dims = (energy_grid.d, energy_grid.d)
    if _xmat is None or _xmat.shape != dims:
        _xmat = np.zeros(dims)
        for eidx in range(energy_grid.d):
            xvec = energy_grid.c[:eidx + 1] / energy_grid.c[eidx]
            _xmat[:eidx + 1, eidx] = xvec
    return _xmat


def print_in_rows(min_dbg_level, str_list, n_cols=5):
    """Prints contents of a list in rows `n_cols`
    entries per row.
    """
    if min_dbg_level > config.debug_level:
        return

    ls = len(str_list)
    n_full_length = int(ls / n_cols)
    n_rest = ls % n_cols
    print_str = '\n'
    for i in range(n_full_length):
        print_str += ('"{:}", ' * n_cols).format(*str_list[i * n_cols:(i + 1) *
                                                           n_cols]) + '\n'
    print_str += ('"{:}", ' * n_rest).format(*str_list[-n_rest:])

    print(print_str.strip()[:-1])


def is_charm_pdgid(pdgid):
    """Returns True if particle ID belongs to a heavy (charm) hadron."""

    return ((abs(pdgid) > 400 and abs(pdgid) < 500)
            or (abs(pdgid) > 4000 and abs(pdgid) < 5000))


def _get_closest(value, in_list):
    """Returns the closes value to 'value' from given list."""

    minindex = np.argmin(np.abs(in_list - value * np.ones(len(in_list))))
    return minindex, in_list[minindex]


def getAZN(pdg_id):
    """Returns mass number :math:`A`, charge :math:`Z` and neutron
    number :math:`N` of ``pdg_id``.

    Note::

        PDG ID for nuclei is coded according to 10LZZZAAAI.
        For iron-52 it is 1000260520.

    Args:
        pdgid (int): PDG ID of nucleus/mass group
    Returns:
        (int,int,int): (Z,A) tuple
    """
    Z, A = 1, 1
    if pdg_id < 2000:
        return 0, 0, 0
    elif pdg_id == 2112:
        return 1, 0, 1
    elif pdg_id == 2212:
        return 1, 1, 0
    elif pdg_id > 1000000000:
        A = pdg_id % 1000 / 10
        Z = pdg_id % 1000000 / 10000
        return A, Z, A - Z
    else:
        return 1, 0, 0


def getAZN_corsika(corsikaid):
    """Returns mass number :math:`A`, charge :math:`Z` and neutron
    number :math:`N` of ``corsikaid``.

    Args:
        corsikaid (int): corsika id of nucleus/mass group
    Returns:
        (int,int,int): (Z,A) tuple
    """
    Z, A = 1, 1

    if corsikaid == 14:
        return getAZN(2212)
    if corsikaid >= 100:
        Z = corsikaid % 100
        A = (corsikaid - Z) / 100
    else:
        Z, A = 0, 0

    return A, Z, A - Z


def corsikaid2pdg(corsika_id):
    """Conversion of CORSIKA nuclear code to PDG nuclear code"""
    if corsika_id in [101, 14]:
        return 2212
    elif corsika_id in [100, 13]:
        return 2112
    else:
        A, Z, _ = getAZN_corsika(corsika_id)
        # 10LZZZAAAI
        pdg_id = 1000000000
        pdg_id += 10 * A
        pdg_id += 10000 * Z
        return pdg_id


def pdg2corsikaid(pdg_id):
    """Conversion from nuclear PDG ID to CORSIKA ID.

    Note::

        PDG ID for nuclei is coded according to 10LZZZAAAI.
        For iron-52 it is 1000260520.
    """
    if pdg_id == 2212:
        return 14

    A = pdg_id % 1000 / 10
    Z = pdg_id % 1000000 / 10000

    return A * 100 + Z


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
    An empty string is returned if skipped levels exceed stack height.abs

    From https://gist.github.com/techtonik/2151727
    """
    import inspect

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ''

    parentframe = stack[start][0]

    name = []

    if config.print_module:
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        if module:
            name.append(module.__name__ + '.')

    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # there seems to be no way to detect static method call - it will
        # be just a function call

        name.append(parentframe.f_locals['self'].__class__.__name__ + '::')

    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename + '(): ')  # function or a method
    else:
        name.append(': ')  # If called from module scope

    del parentframe
    return "".join(name)


def info(min_dbg_level, *message, **kwargs):
    """Print to console if `min_debug_level <= config.debug_level`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
        condition (bool): Print only if condition is True
        blank_caller (bool): blank the caller name (for multiline output)
        no_caller (bool): don't print the name of the caller

    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """
    condition = kwargs.pop('condition', True)
    blank_caller = kwargs.pop('blank_caller', False)
    no_caller = kwargs.pop('no_caller', False)
    if config.override_debug_fcn and min_dbg_level < config.override_max_level:
        fcn_name = caller_name(skip=2).split('::')[-1].split('():')[0]
        if fcn_name in config.override_debug_fcn:
            min_dbg_level = 0

    if condition and min_dbg_level <= config.debug_level:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ''
        if blank_caller:
            cname = len(cname) * ' '
        print(cname + " ".join(message))
