import sys
from types import ModuleType

import numpy as np

#: Settings views, installed by :mod:`MCEq.config` when it finishes loading
#: (see the tail of its ``__init__``), and lazily on first use otherwise.
#: ``info``/``print_in_rows``/``caller_name`` read the ``debug`` group,
#: ``average_A_target`` the ``physics`` one. ``misc`` never imports
#: ``MCEq.config`` statically -- util sits below config in the layering, and
#: the C5 ledger line for this module dies with this mechanism. The lazy
#: path goes through :func:`importlib.import_module`, mirroring the
#: ``sys.modules`` indirection ``config/groups.py`` already uses.
_DEBUG = None
_PHYSICS = None


def _views():
    global _DEBUG, _PHYSICS
    if _DEBUG is None:
        from importlib import import_module

        cfg = import_module("MCEq.config")
        _DEBUG, _PHYSICS = cfg.debug, cfg.physics
    return _DEBUG, _PHYSICS


#: Names that moved into the :mod:`MCEq.data` package and stay reachable here,
#: each mapped to the submodule that owns it now.
#: A D14 compat shim (plan section 8.6): `misc` is C1's bottom layer, so a
#: top-of-file re-export would invert `util -> data`; the names are fetched on
#: demand instead, and C1's ledger carries the edges that creates. Phase 7
#: deletes all of it.
_MOVED_TO_DATA = {
    "EnergyGrid": "MCEq.data.energy_grid",
    "energy_grid": "MCEq.data.energy_grid",
    "_eval_energy_cuts": "MCEq.data.energy_grid",
    "gen_xmat": "MCEq.data.energy_grid",
    "_xmat": "MCEq.data.energy_grid",
    "normalize_hadronic_model_name": "MCEq.data.model_names",
}


#: Memo of the resolved submodules, so that only the FIRST access to a moved
#: name executes an import. See :func:`_moved_to_data` on why `sys.modules` is
#: not enough for this.
_RESOLVED = {}


def _moved_to_data(name):
    """The :mod:`MCEq.data` submodule that owns the moved `name`.

    Read out of `sys.modules` rather than bound by the import statements:
    `from MCEq.data import energy_grid` resolves the *attribute* of the package,
    which a later re-export of the namedtuple under its lowercase name would
    silently turn into the tuple type. The statements are still written out so
    that `grimp` -- and `tests/golden/gen_structure.py`, which parses the same
    text -- sees the two `MCEq.misc -> MCEq.data.*` edges.

    Only the FIRST access imports; later ones come out of `_RESOLVED`. That
    matters at interpreter shutdown: once `sys.meta_path` has been torn down an
    `import` raises `ImportError: sys.meta_path is None`, so a `__del__`
    reading one of these names -- a plain global lookup before the move --
    would have started failing. Measured: with the memo warm the shutdown read
    succeeds; without it, it raises.

    `sys.modules.get(target)` is NOT sufficient in its place, which is why the
    memo is a separate dict: `sys.modules` is itself cleared before late
    `__del__`s run, so the lookup misses and falls through to the import. That
    was measured too -- the first attempt at this fix used `sys.modules` and
    still raised.

    **The remaining hole, stated rather than papered over:** a *cold* first
    access during finalization still raises, because the import genuinely has
    to run once. Nothing in the tree does that (the three `__del__`s under
    `src/` all swallow exceptions and none touches a moved name), and this
    repository carries `tests/test_exit.py` because exit-time behaviour has
    bitten it before, so the warm path is worth having and the cold one is
    worth naming.
    """
    target = _MOVED_TO_DATA[name]
    module = _RESOLVED.get(target)
    if module is None:
        import MCEq.data.energy_grid  # noqa: F401
        import MCEq.data.model_names  # noqa: F401

        module = _RESOLVED[target] = sys.modules[target]
    return module


class _MiscCompatModule(ModuleType):
    """`MCEq.misc` forwarding reads, writes, deletes and `dir()` for the
    names in `_MOVED_TO_DATA`.

    A module-level PEP-562 `__getattr__` answers reads and nothing else, and
    `_xmat` is mutable module state: a write to `MCEq.misc._xmat` would put a
    real attribute in this namespace that shadows the hook from then on, leaving
    two caches where `gen_xmat` has one. Modules have no `__setattr__` hook, so
    the shim is the pre-562 idiom PEP 562 itself names -- a `ModuleType`
    subclass installed over this module -- and `misc._xmat` stays an alias of
    the live object (`tests/test_data_bug_pins.py::clean_xmat_cache` sets and
    restores it, and one pin asserts identity against the returned array).

    All four hooks are needed to make the shim invisible, and only the first
    two were here originally:

    * `__getattr__`/`__setattr__` -- reads and writes, as above.
    * `__delattr__` -- `del MCEq.misc._xmat` was legal before the move (it
      removed the module global) and raised `AttributeError` on the subclass,
      because deletion is not covered by the other two hooks.
    * `__dir__` -- without it `dir()`, `inspect.getmembers()`, `help()` and
      tab completion all lose the six moved names, since they read the
      instance dictionary rather than going through `__getattr__`. PEP 562
      specifies a module-level `__dir__` for exactly this.
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

_target_masses = {
    # <A> must be the ATOM-number-weighted mean mass so that the
    # interaction rate per grammage N_A/<A> * <sigma> (with <sigma> the
    # per-atom air average) equals N_A * sum_i w_i/A_i * sigma_i.
    # Dry-air volume fractions N2/O2/Ar = 0.78084/0.20946/0.00934
    # (https://en.wikipedia.org/wiki/Atmosphere_of_Earth) contribute
    # 2/2/1 atoms per molecule -> <A> = 14.5431, identical to CORSIKA's
    # AVERAW. (The pre-2026 value 14.6568 weighted by MOLECULE fractions,
    # double-counting argon relative to the diatomics: +0.78% in every
    # interaction length.)
    "air": (
        sum(
            n * f * a
            for f, a, n in [(0.78084, 14.0, 2), (0.20946, 16.0, 2), (0.00934, 40.0, 1)]
        )
        / sum(
            n * f
            for f, _, n in [(0.78084, 14.0, 2), (0.20946, 16.0, 2), (0.00934, 40.0, 1)]
        )
    ),
    "water": 1.0 / 3.0 * (2.0 + 16.0),
    "ice": 1.0 / 3.0 * (2.0 + 16.0),
    "co2": 1.0 / 3.0 * (12.0 + 2.0 * 16.0),
    "rock": 22.0,
    "hydrogen": 1.0,
    # A of iron, not Z (was 26.0 = Z until 2026)
    "iron": 55.845,
}


def average_A_target(mat="auto", physics=None):
    """Average target mass number.

    For air <A> = 14.5431: the atom-number-weighted mean mass of dry
    air (N2/O2/Ar volume fractions with 2/2/1 atoms per molecule),
    matching CORSIKA's AVERAW. Pair it with per-atom-averaged air
    cross sections. Other media supported are co2, rock, ice, water,
    hydrogen, and iron.

    Args:
        mat: str or float, optional
            Interaction medium or custom target mass number. Default is "auto".
        physics: physics settings group, optional
            Where ``mat="auto"`` reads ``A_target`` from. Pass a
            ``RunConfig().physics`` snapshot for isolation; the default
            reads the live process setting (M5, D3).

    Returns:
        float
            Average target mass number.

    Raises:
        ValueError:
            If `mat` is not a valid option.

    """
    if isinstance(mat, str) and mat.lower() == "auto":
        if physics is None:
            physics = _views()[1]
        mat = physics.A_target
        if isinstance(mat, str) and mat.lower() == "auto":
            mat = physics.interaction_medium
    if isinstance(mat, str) and mat.lower() in _target_masses:
        return _target_masses[mat.lower()]
    if isinstance(mat, float) or isinstance(mat, int):
        return float(mat)
    raise ValueError(
        "mceq_config.A_target is expected to be a "
        + 'number or one of {0} or "auto"'.format(", ".join(_target_masses.keys()))
    )


def theta_deg(cos_theta):
    """Converts :math:`\\cos{\\theta}` to :math:`\\theta` in degrees."""
    return np.rad2deg(np.arccos(cos_theta))


def print_in_rows(min_dbg_level, str_list, n_cols=5):
    """Prints contents of a list in rows `n_cols`
    entries per row.
    """
    if min_dbg_level > _views()[0].level:
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
    """Name of a caller as ``module.Class::method(): ``.

    `skip` counts stack levels above this function: skip=1 is "who calls me",
    skip=2 "who calls my caller". An empty string is returned when the stack is
    shallower than that.

    The frame is fetched with :func:`sys._getframe` rather than
    :func:`inspect.stack`, which materialises `FrameInfo` records with source
    context for the whole stack: 0.08 us against 192 us on a typical stack, and
    :func:`info` reaches this on every call while an override list is active.

    From https://gist.github.com/techtonik/2151727
    """
    import sys

    try:
        frame = sys._getframe(skip)
    except ValueError:  # stack shallower than `skip`
        return ""

    name = []

    if _views()[0].print_module:
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
    dbg = _views()[0]
    if dbg.override_fcn and min_dbg_level < dbg.override_max_level:
        fcn_name = caller_name(skip=2).split("::")[-1].split("():")[0]
        if fcn_name in dbg.override_fcn:
            min_dbg_level = 0

    if condition and min_dbg_level <= _views()[0].level:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ""
        if blank_caller:
            cname = len(cname) * " "
        print(cname + " ".join(message))


def reset_plt(ticksize, fontsize):
    # Lazy-import: matplotlib is not a required dependency.
    import matplotlib.pyplot as plt

    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["axes.formatter.limits"] = (-1, 3)
    plt.rcParams["axes.linewidth"] = 2.25


def put_ticks(this_fig, this_ax):
    # Lazy-import: matplotlib is not a required dependency.
    import matplotlib

    this_ax.xaxis.set_tick_params(
        which="major", direction="in", width=2.5, length=12, zorder=1, top=True
    )
    this_ax.yaxis.set_tick_params(
        which="major", direction="in", width=2.5, length=12, zorder=1, right=True
    )
    this_ax.xaxis.set_tick_params(
        which="minor", direction="in", width=1.5, length=6, zorder=1, top=True
    )
    this_ax.yaxis.set_tick_params(
        which="minor", direction="in", width=1.5, length=6, zorder=1, right=True
    )
    dx = -3 / 72
    dy = -3 / 72
    y_offset = matplotlib.transforms.ScaledTranslation(0, dy, this_fig.dpi_scale_trans)
    x_offset = matplotlib.transforms.ScaledTranslation(dx, 0, this_fig.dpi_scale_trans)

    for label in this_ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + y_offset)

    for label in this_ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + x_offset)
