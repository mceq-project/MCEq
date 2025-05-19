from collections import namedtuple

import numpy as np

from MCEq import config

#: Energy grid (centers, bind widths, dimension)
energy_grid = namedtuple("energy_grid", ("c", "b", "w", "d"))

#: Matrix with x_lab=E_child/E_parent values
_xmat = None

_target_masses = {
    # <A> = 14.6568 (source https://en.wikipedia.org/wiki/Atmosphere_of_Earth)
    "air": sum([f[0] * f[1] for f in [(0.78084, 14), (0.20946, 16), (0.00934, 40)]]),
    "water": 1.0 / 3.0 * (2.0 + 16.0),
    "ice": 1.0 / 3.0 * (2.0 + 16.0),
    "co2": 1.0 / 3.0 * (12.0 + 2.0 * 16.0),
    "rock": 22.0,
    "hydrogen": 1.0,
    "iron": 26.0,
}


def _eval_energy_cuts(e_centers, e_min=None, e_max=None):
    """Evaluate the energy cuts and return the corresponding indices and slice.

    Args:
        e_centers: numpy.ndarray
            Array of energy grid centers.
        e_min: float, optional
            Minimum energy value. Default is None.
        e_max: float, optional
            Maximum energy value. Default is None.

    Returns:
        min_idx: int
            Index corresponding to the minimum energy value.
        max_idx: int
            Index corresponding to the maximum energy value.
        energy_slice: slice
            Slice corresponding to the energy range.

    """
    min_idx, max_idx = 0, len(e_centers)
    energy_slice = slice(None)
    if e_min is not None:
        min_idx = np.argmin(np.abs(e_centers - e_min))
        energy_slice = slice(min_idx, None)
    if e_max is not None:
        max_idx = np.argmin(np.abs(e_centers - e_max)) + 1
        energy_slice = slice(min_idx, max_idx)
    return min_idx, max_idx, energy_slice


def normalize_hadronic_model_name(name):
    """Converts a hadronic model name into a standard form.

    Args:
        name: str
            Hadronic model name.

    Returns:
        str
            Normalized hadronic model name.

    """
    import re

    return re.sub("[-.]", "", name).upper()


def average_A_target(mat="auto"):
    """Average target mass number.

    For air <A> = 14.6568 (using mass fractions from
    https://en.wikipedia.org/wiki/Atmosphere_of_Earth)
    Other media supported are co2, rock, ice, water, and hydrogen.

    Args:
        mat: str or float, optional
            Interaction medium or custom target mass number. Default is "auto".

    Returns:
        float
            Average target mass number.

    Raises:
        ValueError:
            If `mat` is not a valid option.

    """
    if isinstance(mat, str) and mat.lower() == "auto":
        return _target_masses[config.interaction_medium.lower()]
    if isinstance(mat, str) and mat.lower() in _target_masses:
        return _target_masses[mat.lower()]
    if isinstance(mat, float) or isinstance(mat, int):
        return float(mat)
    raise ValueError(
        "mceq_config.A_target is expected to be a "
        + 'number or one of {0} or "auto"'.format(", ".join(_target_masses.keys()))
    )


def gen_xmat(kinetic_energy_grid):
    """Generates the x_lab (kinetic) matrix for a given energy grid.

    Args:
        energy_grid: namedtuple
            Kinetic energy grid containing the grid centers, bin widths, and dimension.

    Returns:
        numpy.ndarray
            The x_lab matrix.

    """
    global _xmat
    dims = (kinetic_energy_grid.d, kinetic_energy_grid.d)
    if _xmat is None or _xmat.shape != dims:
        _xmat = np.zeros(dims)
        for eidx in range(kinetic_energy_grid.d):
            xvec = kinetic_energy_grid.c[: eidx + 1] / kinetic_energy_grid.c[eidx]
            _xmat[: eidx + 1, eidx] = xvec
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
    print_str = "\n"
    for i in range(n_full_length):
        print_str += ('"{:}", ' * n_cols).format(
            *str_list[i * n_cols : (i + 1) * n_cols]
        ) + "\n"
    print_str += ('"{:}", ' * n_rest).format(*str_list[-n_rest:])

    print(print_str.strip()[:-1])


def is_charm_pdgid(pdgid):
    """Returns True if particle ID belongs to a heavy (charm) hadron."""

    return (abs(pdgid) > 400 and abs(pdgid) < 500) or (
        abs(pdgid) > 4000 and abs(pdgid) < 5000
    )


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
    if pdg_id == 2112:
        return 1, 0, 1
    if pdg_id == 2212:
        return 1, 1, 0
    if pdg_id > 1000000000:
        A = (pdg_id % 1000) // 10
        Z = (pdg_id % 1000000) // 10000
        return A, Z, A - Z
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
        A = (corsikaid - Z) // 100
    else:
        Z, A = 0, 0

    return A, Z, A - Z


def corsikaid2pdg(corsika_id):
    """Conversion of CORSIKA nuclear code to PDG nuclear code"""
    if corsika_id in [101, 14]:
        return 2212
    if corsika_id in [100, 13]:
        return 2112
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
        return ""

    parentframe = stack[start][0]

    name = []

    if config.print_module:
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        if module:
            name.append(module.__name__ + ".")

    # detect classname
    if "self" in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # there seems to be no way to detect static method call - it will
        # be just a function call

        name.append(parentframe.f_locals["self"].__class__.__name__ + "::")

    codename = parentframe.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename + "(): ")  # function or a method
    else:
        name.append(": ")  # If called from module scope

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
    condition = kwargs.pop("condition", min_dbg_level <= config.debug_level)
    # Dont' process the if the function if nothing will happen
    if not (condition or config.override_debug_fcn):
        return

    blank_caller = kwargs.pop("blank_caller", False)
    no_caller = kwargs.pop("no_caller", False)
    if config.override_debug_fcn and min_dbg_level < config.override_max_level:
        fcn_name = caller_name(skip=2).split("::")[-1].split("():")[0]
        if fcn_name in config.override_debug_fcn:
            min_dbg_level = 0

    if condition and min_dbg_level <= config.debug_level:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ""
        if blank_caller:
            cname = len(cname) * " "
        print(cname + " ".join(message))
