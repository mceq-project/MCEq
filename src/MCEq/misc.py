"""Shared helpers: energy grid, model names, target masses and logging."""

import re
import sys
from collections import namedtuple

import numpy as np

from MCEq import config

#: Energy grid: centers, bin edges, bin widths, bin count
energy_grid = namedtuple("energy_grid", ("c", "b", "w", "d"))

#: Matrix with x_lab=E_child/E_parent values
_xmat = None

#: Dry air: volume fraction, mass number and atoms per molecule of N2, O2, Ar.
_AIR = [(0.78084, 14.0, 2), (0.20946, 16.0, 2), (0.00934, 40.0, 1)]

#: Mean target mass number per atom. Averaging over atoms, not molecules, makes
#: N_A/<A> * <sigma> with the per-atom cross section the interaction rate per
#: grammage; for air <A> = 14.5431, CORSIKA's AVERAW.
_target_masses = {
    "air": sum(n * f * a for f, a, n in _AIR) / sum(n * f for f, _, n in _AIR),
    "water": 1.0 / 3.0 * (2.0 + 16.0),
    "ice": 1.0 / 3.0 * (2.0 + 16.0),
    "co2": 1.0 / 3.0 * (12.0 + 2.0 * 16.0),
    "rock": 22.0,
    "hydrogen": 1.0,
    "iron": 55.845,
}


def average_A_target(mat="auto"):
    """Mean target mass number per atom of medium ``mat``.

    ``mat`` is a name from ``_target_masses``, a mass number, or "auto" for
    ``config.A_target``, which may itself be "auto" for
    ``config.interaction_medium``.
    """
    if str(mat).lower() == "auto":
        mat = config.A_target
    if str(mat).lower() == "auto":
        mat = config.interaction_medium
    if isinstance(mat, (int, float)):
        return float(mat)
    if isinstance(mat, str) and mat.lower() in _target_masses:
        return _target_masses[mat.lower()]
    raise ValueError(
        f"config.A_target must be a number, 'auto' or one of {list(_target_masses)}"
    )


def _eval_energy_cuts(e_centers, e_min=None, e_max=None):
    """``(min_idx, max_idx, slice)`` of the bins nearest to ``e_min`` and ``e_max``."""
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
    """Upper-case ``name`` without dashes and dots ("Sibyll-2.3d" -> "SIBYLL23D")."""
    return re.sub("[-.]", "", name).upper()


def gen_xmat(energy_grid):
    """Matrix of x_lab = E_child/E_parent, cached and reused while ``energy_grid.d`` is unchanged."""
    global _xmat
    dims = (energy_grid.d, energy_grid.d)
    if _xmat is None or _xmat.shape != dims:
        _xmat = np.zeros(dims)
        for eidx in range(energy_grid.d):
            xvec = energy_grid.c[: eidx + 1] / energy_grid.c[eidx]
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


#: Photons, electrons and pi0 -- the species that exist only as part of the
#: electromagnetic cascade. pi0 decays into photons and nothing else.
EM_PDG_IDS = (
    22,
    11,
    -11,
    111,
)  # anfe: funny to find this in the middle of the misc module


def strip_em_species(
    particle_list,
):  # anfe: this function is used twice, but it should be handled rather not by stripping the EM species later. The list should be build without EM species if EM is not switched on. So remove this module and work carefully the disabled em logic into particle manager.
    """Leave the electromagnetic species out unless the cascade is on.

    Without EM interactions photons and electrons have no transport, and pi0
    has nothing to decay into; energy going into them leaves the cascade. The
    databases carry the channels either way; this decides only what the
    solver propagates.
    """
    if config.enable_em:
        return particle_list
    return [p for p in particle_list if p[0] not in EM_PDG_IDS]


def getAZN(pdg_id):
    """Mass number A, charge Z and neutron number N of a nucleon or nucleus.

    Nuclei are coded 10LZZZAAAI (iron-52 is 1000260520); any other particle
    gives (0, 0, 0).
    """
    if pdg_id == 2112:
        return 1, 0, 1
    if pdg_id == 2212:
        return 1, 1, 0
    if pdg_id > 1000000000:
        A = (pdg_id % 1000) // 10
        Z = (pdg_id % 1000000) // 10000
        return A, Z, A - Z
    return 0, 0, 0


def caller_name(skip=2):
    """Name of a caller as ``module.Class::method(): ``.

    `skip` counts stack levels above this function: skip=1 is "who calls me",
    skip=2 "who calls my caller". Returns "" if the stack is shallower.

    From https://gist.github.com/techtonik/2151727
    """
    try:
        frame = sys._getframe(skip)
    except ValueError:  # stack shallower than `skip`
        return ""

    name = []

    if config.print_module:
        modname = frame.f_globals.get("__name__")
        if modname:
            name.append(modname + ".")

    # A frame with a local named `self` is a method call; there is no way to
    # tell a static method from a plain function this way.
    if "self" in frame.f_locals:
        name.append(frame.f_locals["self"].__class__.__name__ + "::")

    codename = frame.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename + "(): ")  # function or a method
    else:
        name.append(": ")  # If called from module scope

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
    condition = kwargs.pop("condition", True)
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
