import sys
from types import ModuleType

from MCEq import config

#: Compatibility names that moved into the :mod:`MCEq.data` package and stay
#: reachable here, each mapped to the submodule that owns it now. `misc` is
#: C1's bottom layer, so a top-of-file re-export would invert `util -> data`;
#: the names are forwarded lazily instead, and C1's ledger carries the edges
#: that creates.
_MOVED_TO_DATA = {
    "EnergyGrid": "MCEq.data.energy_grid",
    "energy_grid": "MCEq.data.energy_grid",
    "_eval_energy_cuts": "MCEq.data.energy_grid",
    "gen_xmat": "MCEq.data.energy_grid",
    "_xmat": "MCEq.data.energy_grid",
    "normalize_hadronic_model_name": "MCEq.data.model_names",
}
# These symbols were never in misc, so we don't need any compatibility shim in this module?

#: Memo of the resolved submodules, so that only the FIRST access to a moved
#: name executes an import. See :func:`_moved_to_data` on why `sys.modules` is
#: not enough for this.
_RESOLVED = {}


def _moved_to_data(name):  # anfe: looks totally boilerplate to me.
    """Resolve a moved name from its implementation module and retain the module.

    Read out of `sys.modules` rather than bound by the import statements:
    `from MCEq.data import energy_grid` resolves the *attribute* of the
    package, which a later re-export of the namedtuple under its lowercase
    name would silently turn into the tuple type. The statements are still
    written out so that static import analysis (`grimp`) sees the two
    `MCEq.misc -> MCEq.data.*` edges.

    Only the FIRST access imports; later ones come out of `_RESOLVED`. The
    memo matters at interpreter shutdown, when an `import` can fail and
    `sys.modules` itself is already cleared: with the memo warm, a late read
    of a moved name succeeds. A *cold* first access during finalization
    still raises, because the import genuinely has to run once; nothing
    under `src/` does that, and `tests/test_exit.py` guards the warm path.
    """
    target = _MOVED_TO_DATA[name]
    module = _RESOLVED.get(target)
    if module is None:
        import MCEq.data.energy_grid  # noqa: F401
        import MCEq.data.model_names  # noqa: F401

        module = _RESOLVED[target] = sys.modules[target]
    return module


class _MiscCompatModule(ModuleType):
    """Forward reads, writes, deletes, and `dir()` for the names in `_MOVED_TO_DATA`.

    Forwarding writes is necessary because `__getattr__` alone would let an
    assignment such as `MCEq.misc._xmat = ...` put a real attribute in this
    namespace that shadows the hook from then on, leaving two caches where
    `gen_xmat` has one. Modules have no `__setattr__` hook, so the shim is
    the pre-562 idiom PEP 562 itself names -- a `ModuleType` subclass
    installed over this module -- and `misc._xmat` stays an alias of the live
    object (`tests/test_data_bug_pins.py::clean_xmat_cache` sets and restores
    it, and one pin asserts identity against the returned array). All four
    hooks keep the shim invisible: `__delattr__` preserves the previously
    legal `del MCEq.misc._xmat`, and `__dir__` keeps `dir()`,
    `inspect.getmembers()`, `help()` and tab completion listing the moved
    names.
    """

    def __getattr__(self, name):
        if name in _MOVED_TO_DATA:
            return getattr(_moved_to_data(name), name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _MOVED_TO_DATA:
            setattr(_moved_to_data(name), name, value)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _MOVED_TO_DATA:
            delattr(_moved_to_data(name), name)
            return
        super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_MOVED_TO_DATA))


sys.modules[__name__].__class__ = _MiscCompatModule

# anfe: IMO, everything above is unnecessary, and is some sort of misunderstanding.

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
