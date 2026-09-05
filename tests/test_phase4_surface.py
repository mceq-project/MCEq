"""The public surface of the five flat modules Phase 4 moves, pinned before the move.

Phase 4 splits `MCEq.data` into `data/` (`hdf5_store`, `interaction_tables`,
`decay_tables`, `cross_sections`, `continuous_losses`, `equivalences`),
`MCEq.particlemanager` into `species/`, `MCEq.ddm` + `MCEq.ddm_utils` into
`models/ddm`, and `MCEq.download` under `data/`. Each flat module stays behind
as a silent re-export shim for this release (D14: silent now, warn next, remove
after). While there is still exactly one import path, "the shim re-exports the
right things" cannot be tested by comparing the two paths -- so this file writes
the surface down as literals instead, and the Phase 4 shims have to satisfy it.

Every name below was checked against HEAD (c81d182) before it was written down.
The assignment's list needed **no corrections**: all 22 names resolve, and all
of them are in `vars()` of the module that is supposed to carry them.

Two facts about today that shape what is asserted here:

* None of the five modules defines `__all__` (verified at c81d182). So this
  file pins membership in `vars(module)`, not membership in `__all__`. That is
  the weaker of the two and it is deliberate: `sphinx-automodapi` only turns
  off its "was this defined locally?" filter when `__all__` is present, so a
  `from .interaction_tables import *` shim with no `__all__` would silently
  drop the re-exported *functions* from the rendered page -- the Phase 3 trap
  that cost `MCEq.geometry.atmosphere_parameters` six documented functions
  (see `tests/geometry/test_geometry_shims.py`). `docs/api-reference/data.rst`
  and `docs/api-reference/particlemanager.rst` both run `.. automodapi::`, so
  the Phase 4 shims for those two need an explicit `__all__` to stay rendered.
* Each name is still *defined* in the flat module. `test_the_flat_module_still_
  owns_every_name` records that; it is the one test here that is **expected to
  flip** when Phase 4 lands, and it flipping is the signal that the code moved
  rather than being copied.

DB-free: nothing here constructs an `MCEqRun` or touches the HDF5 database. The
two `DDMSplineDB` tests do load the packaged `data/DDM_1.0.npy` (0.01 s
measured) and put the class-level spline cache back the way they found it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pathlib
from types import ModuleType

import pytest

#: Flat module -> the names it must keep exposing, verbatim, after Phase 4.
#: Private names are included where the tree already imports them; incidental
#: module-level imports (`np`, `pathlib`, `info`, ...) and the private helpers
#: nothing else reaches (`_pname`, `_DDMEntry`, `_DDMChannel`,
#: `_generate_DDM_matrix`, `_download_file`, ...) are deliberately absent --
#: they are not part of the pinned contract. (`_select_em_rho_slice` left the
#: package for `MCEq.data.em_tables.select_em_rho_slice` in Phase 4.)
SURFACE: dict[str, tuple[str, ...]] = {
    "MCEq.data": (
        "ContinuousLosses",
        "Decays",
        "HDF5Backend",
        "InteractionCrossSections",
        "Interactions",
        "equivalences",
    ),
    "MCEq.particlemanager": (
        "ATOMIC_MASS_UNIT_G",
        "MCEqParticle",
        "ParticleManager",
        # Private, and therefore never carried by `import *` nor picked up by a
        # hand-written `__all__`. `src/MCEq/ddm.py:14` and `tests/test_ddm.py:5`
        # both import it by name; see `test_ddm_and_particlemanager_share_pdata`.
        "_pdata",
        "backward_compatible_namestr",
    ),
    "MCEq.ddm": (
        "DDMSplineDB",
        "DataDrivenModel",
        "isospin_partners",
        "isospin_symmetries",
    ),
    "MCEq.ddm_utils": (
        "fmteb",
        "gen_matrix_variations",
    ),
    "MCEq.download": (
        "FileIntegrityCheck",
        "base_url",
        "ensure_db_available",
        "file_checksum",
        "release_tag",
    ),
}

#: The `(module, name)` pairs of SURFACE, flattened for parametrisation.
SURFACE_PAIRS = [(mod, name) for mod, names in SURFACE.items() for name in names]

#: The dotted paths `src/MCEq/core.py` *constructs*, exactly. Written as
#: literals rather than read back through `MCEq.data.<name>`: an assertion of
#: the shape `MCEq.data.HDF5Backend is MCEq.data.HDF5Backend` compares a value
#: with itself through the same dotted path and holds however wrong the
#: re-export is. Parsing core.py records *where core reaches*, which is the
#: property Phase 4 can break -- core.py:5 is a bare `import MCEq.data`, so
#: these five call sites resolve through the shim's module namespace at call
#: time, not through a name bound at import time.
CORE_CONSTRUCTS = {
    "MCEq.data.ContinuousLosses",
    "MCEq.data.Decays",
    "MCEq.data.HDF5Backend",
    "MCEq.data.InteractionCrossSections",
    "MCEq.data.Interactions",
}


def _dotted(node: ast.expr) -> str | None:
    """`MCEq.data.Decays` for an Attribute chain rooted at a plain Name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _from_module(node: ast.ImportFrom, package: str = "MCEq") -> str:
    """Absolute module an `ImportFrom` in `<package>/<module>.py` reads from.

    `from MCEq.data import X` and `from .data import X` both answer
    `MCEq.data`; `from . import data` answers `MCEq`, since it binds the
    submodule rather than names out of it. `level` is only ever 0 or 1 in a
    top-level module of the package -- a deeper one would climb out of it and
    fail at import -- so anything deeper is reported as unresolved.
    """
    if node.level == 0:
        return node.module or ""
    if node.level > 1:
        return ""
    return f"{package}.{node.module}" if node.module else package


@pytest.mark.parametrize("module_name", sorted(SURFACE))
def test_the_flat_path_is_a_real_module(module_name):
    """Not a lazy proxy object: `automodapi` and `vars()` both need a module."""
    assert isinstance(importlib.import_module(module_name), ModuleType)


@pytest.mark.parametrize(("module_name", "name"), SURFACE_PAIRS)
def test_pinned_name_is_in_the_module_namespace(module_name, name):
    """In `vars()`, not merely `getattr`-able.

    A module `__getattr__` (PEP 562) answers `getattr` but leaves `vars()`
    empty, and `sphinx-automodapi` walks the namespace. Phase 3 showed the
    same distinction from the other side: the lazy MSIS21 names are reachable
    on `MCEq.geometry.density_profiles` but never enter a star-import.
    """
    module = importlib.import_module(module_name)
    assert name in vars(module), (
        f"{module_name}.{name} is not in the module namespace; a `getattr` "
        "hook satisfies callers but not the docs build"
    )


@pytest.mark.parametrize(("module_name", "name"), SURFACE_PAIRS)
def test_the_flat_module_still_owns_every_name(module_name, name):
    """Today the flat module *defines* these; Phase 4 must flip this test.

    Only the classes and functions in SURFACE are checked; the constants,
    dicts and the `_pdata` singleton are skipped. `__module__` is no use for
    those -- an *instance* inherits it from its class, so `_pdata.__module__`
    is `particletools.tables`, which says nothing about where MCEq built it.
    When this starts failing with `__module__` pointing at
    `MCEq.data.interaction_tables`, `MCEq.species.particle`, ... the move
    happened. If it keeps passing after Phase 4, the code was copied rather
    than relocated and there are now two definitions in the tree.
    """
    obj = getattr(importlib.import_module(module_name), name)
    if not (inspect.isclass(obj) or inspect.isroutine(obj)):
        pytest.skip(f"{name} is a value, not a class or function definition")
    assert obj.__module__ == module_name


def test_ddm_and_particlemanager_share_pdata():
    """One `PYTHIAParticleData`, not two.

    `ddm.py` does `from .particlemanager import _pdata` and uses it for the
    `x_min` cut (`ddm.py:137`, `ddm.py:834`); `tests/test_ddm.py:25` asserts
    `ddm_entry.x_min == _pdata.mass(211) / 2` against the *particlemanager*
    object. If the Phase 4 shims each construct their own table those two
    stay equal by luck, so pin the identity while it is still trivially true.
    """
    import MCEq.ddm
    import MCEq.particlemanager

    assert MCEq.ddm._pdata is MCEq.particlemanager._pdata


def test_ddm_spline_cache_is_a_class_attribute_on_the_class():
    """`DDMSplineDB._ddm_splines` must stay reachable through the flat name.

    `tests/golden/gen_species.py` saves it (`dict(DDMSplineDB._ddm_splines)`,
    line 548) and restores it in place (`.clear()` / `.update()`, lines
    665-666) around its build, because the first instance in a session
    populates a cache shared by every later one. A shim that re-exported a
    *wrapper* instead of the class, or a subclass of it, would give that
    save/restore a different dict and leak channel entries across the session
    -- silently, since the golden would still be produced.

    Structure only. The sharing itself is
    `test_ddm_spline_cache_is_shared_by_every_instance`; a class can satisfy
    everything below and still hand each instance its own dict.
    """
    from MCEq.ddm import DDMSplineDB

    assert isinstance(DDMSplineDB, type), "DDMSplineDB must be the class itself"
    assert "_ddm_splines" in vars(DDMSplineDB), (
        "_ddm_splines is not defined on DDMSplineDB itself; gen_species.py "
        "would save and restore some other object's cache"
    )
    assert isinstance(DDMSplineDB._ddm_splines, dict)


def test_ddm_spline_cache_is_shared_by_every_instance():
    """Constructing the *first* `DDMSplineDB` fills the class dict. A leak, pinned.

    This is the behaviour `gen_species.py` works around and the reason the
    structural test above is not enough: a class can keep `_ddm_splines` in
    its own `vars()` and still hand each instance a private dict, and then the
    save/restore at `gen_species.py:548,665-666` would quietly guard nothing.
    `add_entry` writes `self._ddm_splines[channel] = ...` (`ddm.py:566-569`)
    and `__init__` never rebinds the attribute, so on a fresh class the write
    lands on the class and every later instance sees it.

    The class cache is emptied first and restored in `finally`: with a
    non-empty cache `_load_from_file` takes a different branch
    (`test_a_second_ddm_spline_db_shadows_the_class_cache`), so without that
    the outcome would depend on which tests ran before this one in the
    session. Reads `DDM_1.0.npy` from the package data directory (0.01 s
    measured); no HDF5 database is touched.
    """
    from MCEq.ddm import DDMSplineDB

    saved = dict(DDMSplineDB._ddm_splines)
    try:
        DDMSplineDB._ddm_splines.clear()
        first = DDMSplineDB()
        assert "_ddm_splines" not in vars(first), (
            "the first instance shadows the class attribute; gen_species.py's "
            "save/restore of DDMSplineDB._ddm_splines would no longer see what "
            "a DataDrivenModel wrote"
        )
        assert DDMSplineDB._ddm_splines, "loading wrote nothing onto the class"
        assert first._ddm_splines is DDMSplineDB._ddm_splines

        later = DDMSplineDB.__new__(DDMSplineDB)
        assert later._ddm_splines.keys() == DDMSplineDB._ddm_splines.keys()
    finally:
        DDMSplineDB._ddm_splines.clear()
        DDMSplineDB._ddm_splines.update(saved)


def test_a_second_ddm_spline_db_shadows_the_class_cache():
    """The second instance in a process gets a private dict. Today's bug, pinned.

    `_load_from_file` clears "if `_load` is called multiple times" by
    *rebinding* rather than emptying -- `if self._ddm_splines:
    self._ddm_splines = {}` (`ddm.py:510-511`) -- and an assignment through
    `self` creates an instance attribute. So the entries the second instance
    loads never reach `DDMSplineDB._ddm_splines`, which keeps whatever the
    first one left there. That asymmetry is why the test above has to empty
    the cache before it measures anything, and why `gen_species.py` restores
    the class dict by hand.

    Recording the current behaviour, not endorsing it: if a later commit makes
    `_load_from_file` clear the dict in place, this pin flips and the flip is
    the intended signal.
    """
    from MCEq.ddm import DDMSplineDB

    saved = dict(DDMSplineDB._ddm_splines)
    try:
        DDMSplineDB._ddm_splines.clear()
        DDMSplineDB()
        class_cache = DDMSplineDB._ddm_splines
        sentinel = object()
        class_cache["__pin_probe__"] = sentinel

        second = DDMSplineDB()
        assert "_ddm_splines" in vars(second), (
            "the second instance no longer shadows; _load_from_file has "
            "started clearing the class dict in place"
        )
        assert second._ddm_splines is not class_cache
        assert "__pin_probe__" not in second._ddm_splines
        assert class_cache["__pin_probe__"] is sentinel, (
            "the class cache was emptied by the second load"
        )
    finally:
        DDMSplineDB._ddm_splines.clear()
        DDMSplineDB._ddm_splines.update(saved)


def test_core_reaches_the_data_tables_by_these_dotted_paths():
    """`src/MCEq/core.py` constructs exactly CORE_CONSTRUCTS -- no more, no less.

    Parsed from source rather than probed at runtime, so the pin records the
    call sites (core.py:355, 373, 378, 383, 388) and not just that a name
    happens to resolve.
    """
    origin = importlib.util.find_spec("MCEq.core").origin
    tree = ast.parse(pathlib.Path(origin).read_text())

    constructed = {
        dotted
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for dotted in [_dotted(node.func)]
        if dotted is not None and dotted.split(".")[0] == "MCEq"
    }
    assert constructed == CORE_CONSTRUCTS


def test_core_imports_the_data_module_not_its_names():
    """The dotted paths above only work because core.py binds the *module*.

    `import MCEq.data` (core.py:5) makes every `MCEq.data.X` a namespace
    lookup at call time; a Phase 4 shim that is a module keeps working. If
    core.py ever switches to `from MCEq.data import HDF5Backend` the five
    call sites stop going through the shim namespace and
    `test_core_reaches_the_data_tables_by_these_dotted_paths` empties out
    without anything else noticing.

    The from-import half resolves `level` before it compares: core.py sits in
    the `MCEq` package, so the form a refactor inside the package actually
    reaches for is the *relative* `from .data import Interactions`, which an
    `ImportFrom.module == "MCEq.data"` test never sees (measured: that node
    carries `module="data"`, `level=1`). `from . import data` is deliberately
    not counted -- it binds the module, like the import on line 5.

    The positive half is form-specific on purpose: it records line 5 as it is
    written today. An equivalent rewrite (`import MCEq.data as mdata`,
    `from MCEq import data`) flips it, and that flip is a review prompt rather
    than a false alarm, since the dotted-path pin above is written against the
    `MCEq.data.X` spelling those rewrites abandon.
    """
    origin = importlib.util.find_spec("MCEq.core").origin
    tree = ast.parse(pathlib.Path(origin).read_text())

    plain_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.asname is None
    }
    assert "MCEq.data" in plain_imports

    from_data = [
        f"{_from_module(node)}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and _from_module(node) == "MCEq.data"
        for alias in node.names
    ]
    assert from_data == []


# The `MCEq.misc` compat shim. Phase 4 moved six names out of `misc` into
# `MCEq.data`, and `misc` keeps forwarding them (D14, plan section 8.6) through
# a `ModuleType` subclass rather than a PEP-562 `__getattr__`, because `_xmat`
# is mutable module state that a bare hook cannot forward writes for. Reads and
# writes were covered from the start; the three below were not, and an
# adversarial audit found all three.


def test_the_misc_shim_forwards_dir_not_only_getattr():
    """`dir()` reads the instance dict, so `__getattr__` alone loses the names.

    Without `__dir__`, `dir(MCEq.misc)`, `inspect.getmembers()`, `help()` and
    tab completion all silently stop listing the six moved names, which the
    module did list before the move. PEP 562 specifies a module-level
    `__dir__` for exactly this.
    """
    import inspect

    import MCEq.misc

    moved = set(MCEq.misc._MOVED_TO_DATA)
    assert moved, "the shim table is empty; this test has nothing to check"
    assert moved <= set(dir(MCEq.misc))
    assert moved <= {name for name, _ in inspect.getmembers(MCEq.misc)}


def test_the_misc_shim_forwards_deletion():
    """`del MCEq.misc._xmat` was legal before the move.

    It removed the module global, and a read afterwards raised
    `AttributeError`. `__getattr__`/`__setattr__` do not cover deletion, so
    without `__delattr__` the *deletion itself* raised on the subclass. With
    it, the delete reaches the owning module and the read raises as before --
    faithful, and equally a foot-gun in both versions, since `gen_xmat`'s
    `global _xmat` then hits a `NameError`. Restored in a `finally` so the rest
    of the session is unaffected. Nothing in the tree deletes these attributes;
    this keeps the shim honest rather than reporting a live bug.
    """
    import numpy as np

    import MCEq.misc
    from MCEq.data.energy_grid import energy_grid, gen_xmat

    grid = energy_grid(c=np.logspace(0.0, 2.0, 3), b=None, w=None, d=3)
    gen_xmat(grid)
    assert MCEq.misc._xmat is not None

    try:
        del MCEq.misc._xmat
        # The delete reached the owning module, not this namespace.
        with pytest.raises(AttributeError, match="MCEq.data.energy_grid"):
            MCEq.misc._xmat
    finally:
        MCEq.misc._xmat = None


def test_the_misc_shim_survives_a_read_during_finalization():
    """A warm read of a moved name must not import at interpreter shutdown.

    `_moved_to_data` memoizes the resolved submodule so only the first access
    imports. Once `sys.meta_path` is torn down an `import` raises
    `ImportError: sys.meta_path is None`, and `sys.modules` is cleared before
    late `__del__`s run, so a `sys.modules.get` fallback is not enough either.
    Run in a subprocess because the property is about this interpreter dying.
    """
    import subprocess
    import sys

    program = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, MCEq.misc as m\n"
        "from MCEq.data.energy_grid import energy_grid, gen_xmat\n"
        "gen_xmat(energy_grid(c=np.logspace(0.,2.,3), b=None, w=None, d=3))\n"
        "_ = m._xmat\n"
        "class Late:\n"
        "    def __del__(self):\n"
        "        try:\n"
        "            print('OK', type(m._xmat).__name__)\n"
        "        except BaseException as exc:\n"
        "            print('FAILED', type(exc).__name__)\n"
        "import builtins; builtins._keep = Late()\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=180
    )
    assert "OK ndarray" in out.stdout, (out.stdout, out.stderr)


#: Every public (non-underscore), non-module name in `vars(MCEq.data)` today.
#: Incidental imports (`defaultdict`, `isfile`, `join`, `info`, ...) are
#: listed because they ARE in the namespace -- this is a record of what is
#: exposed, not an endorsement. Adding a public name to `MCEq.data` is a
#: deliberate act that updates this list in the same commit.
DATA_PUBLIC_SURFACE = frozenset(
    {
        "ContinuousLosses",
        "Decays",
        "EnergyGrid",
        "HDF5Backend",
        "InteractionCrossSections",
        "Interactions",
        "apply_equivalences",
        "blend_cross_sections",
        "blend_yields",
        "defaultdict",
        "equivalences",
        "family_of",
        "he_le_weight",
        "info",
        "isfile",
        "join",
        "normalize_hadronic_model_name",
        "reverse_equivalences",
    }
)


def test_data_public_surface_is_exactly_this_set():
    """Upper bound: `MCEq.data` exposes exactly DATA_PUBLIC_SURFACE, no more.

    `SURFACE["MCEq.data"]` above is a lower bound (names that MUST exist). A
    commit once widened the public surface with a name that should have stayed
    private, and nothing caught it. Submodules bound onto the package by the
    import machinery (`blending`, `energy_grid`, `model_names`, `hdf5_store`,
    `np`) are excluded: they are `ModuleType` objects, not part of the API
    contract.
    A new public name is a deliberate act that updates DATA_PUBLIC_SURFACE in
    the same commit; a mismatch in either direction is a failure.
    """
    import MCEq.data

    public = {
        name
        for name, value in vars(MCEq.data).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    }
    assert public == DATA_PUBLIC_SURFACE, (
        f"unexpected: {sorted(public - DATA_PUBLIC_SURFACE)}; "
        f"missing: {sorted(DATA_PUBLIC_SURFACE - public)}"
    )
    # Consistency between the two literals in this file (the lower bound must
    # be inside the upper bound), not a pin on `MCEq.data` itself.
    assert set(SURFACE["MCEq.data"]) <= DATA_PUBLIC_SURFACE


def test_data_equivalences_attribute_is_the_dict_not_the_submodule():
    """`MCEq.data.equivalences` is the DICT; the submodule of the same name is shadowed.

    `src/MCEq/data/__init__.py` imports the submodule `MCEq.data.equivalences`
    (which binds the module onto the package) and then rebinds the name to the
    dict with `from MCEq.data.equivalences import equivalences`. The dict is
    the pinned public object; the module is reachable only through
    `sys.modules`. See the module docstring of `src/MCEq/data/equivalences.py`.
    """
    import sys

    import MCEq.data

    assert isinstance(MCEq.data.equivalences, dict)
    # Literal-vs-literal consistency check (the name is in the upper bound
    # above), not a pin on `MCEq.data`; the pins are the isinstance/is lines.
    assert "equivalences" in DATA_PUBLIC_SURFACE
    assert isinstance(sys.modules["MCEq.data.equivalences"], ModuleType)
    assert sys.modules["MCEq.data.equivalences"].equivalences is MCEq.data.equivalences
