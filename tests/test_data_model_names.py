"""`MCEq.data.model_names`: the name -> family half of the equivalence lookup.

`family_of` replaced two inline implementations that walked the same eight family
substrings in two different orders -- the `elif` chain of
`HDF5Backend._interaction_db_single` (`SIBYLL21` first) and the
`equivalences.items()` loop of `HDF5Backend._mapped_cross_section` (`SIBYLL23`
first). This file pins that the replacement is exactly what both did, and pins
the one input class where the two orders could ever have differed.

It also pins the *non*-unification: the two PDG -> family-representative
fallbacks are untouched and still disagree. That is deliberate (they change
blended cross sections), so the disagreement is written down rather than fixed
-- and it is pinned by driving the two REAL callables, not transcriptions of
them. An earlier version of that test compared two local closures against each
other, which could not fail for any change to the code it named.

No shipped database is opened: the model-name universe below is a literal,
measured off the databases symlinked into `src/MCEq/data` on the machine that
wrote it (ten links, of which five resolved). It is a convenience sample of
real spellings, not a property of the package. The one test that needs a
file builds a synthetic one into `tmp_path` with
`tests/data/make_hdf5_fixtures.build_database`.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq.data import HDF5Backend, equivalences

# `MCEq.data.equivalences` the attribute is the DICT, not the submodule (the
# package rebinds it on import), so the functions have to come through
# `from ... import`. `import MCEq.data.equivalences as x` would bind the dict.
from MCEq.data.equivalences import family_fallback, mapped_cross_section
from MCEq.data.model_names import (
    HADRONIC_MODEL_FAMILIES,
    family_of,
    normalize_hadronic_model_name,
)

#: `tests/data/make_hdf5_fixtures.py`, loaded by path -- `tests/data` is not a
#: package. Same idiom as `tests/test_data_hdf5_decode.py`.
_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures_model_names",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

#: Every hadronic-model name appearing under `hadronic_interactions/*` or
#: `cross_sections/*` in the shipped databases, plus two un-normalized spellings
#: and the empty string.
MODEL_NAMES = (
    "DPMJETIII191",
    "DPMJETIII193",
    "DPMJETIII306",
    "EPOSLHC",
    "EPOSLHCR",
    "FLUKA20251",
    "QGSJET01C",
    "QGSJETII03",
    "QGSJETII04",
    "QGSJETIII",
    "SIBYLL21",
    "SIBYLL23",
    "SIBYLL23C",
    "SIBYLL23D",
    "SIBYLL23E",
    "SIBYLL23ESTARBAR",
    "SIBYLL23ESTARMIXED",
    "SIBYLL23ESTARRHO",
    "SIBYLL23ESTARSTRANGE",
    "URQMD34",
    "SIBYLL-2.3d",
    "dpmjet-III-19.1",
    "",
)

#: The `elif` chain of `_interaction_db_single`, transcribed.
_ELIF_ORDER = (
    "SIBYLL21",
    "SIBYLL23",
    "QGSJET01",
    "QGSJETII",
    "DPMJET",
    "EPOSLHC",
    "PYTHIA8",
    "FLUKA",
)


def _former_elif_chain(mname):
    for family in _ELIF_ORDER:
        if family in mname:
            return family
    return None


#: The insertion order of the `equivalences` literal at the time
#: `_mapped_cross_section`'s loop was deleted. Frozen, not read live: the loop
#: is gone, so this is a historical record. Walking `equivalences` itself would
#: turn a cosmetic reorder of a production dict literal -- which by
#: `test_no_family_is_a_substring_of_another` cannot change any behaviour --
#: into a red test.
_DICT_ORDER = (
    "SIBYLL23",
    "SIBYLL21",
    "QGSJET01",
    "QGSJETII",
    "DPMJET",
    "EPOSLHC",
    "PYTHIA8",
    "FLUKA",
)


def _former_dict_loop(model_name):
    """The `for family, mapping in equivalences.items()` loop, transcribed."""
    for family in _DICT_ORDER:
        if family in model_name:
            return family
    return None


def test_the_frozen_dict_order_is_still_a_permutation_of_the_table():
    """`_DICT_ORDER` is frozen, so this is what catches it going stale.

    A family added to or removed from `equivalences` must be reflected here,
    or `_former_dict_loop` stops being a transcription of anything. The
    *order* is deliberately not compared -- that is the point of freezing it.
    """
    assert set(_DICT_ORDER) == set(equivalences)


def test_families_are_exactly_the_equivalence_table_keys():
    """`family_of` indexes `equivalences`, so a drift here is a `KeyError`."""
    assert set(HADRONIC_MODEL_FAMILIES) == set(equivalences)


def test_no_family_is_a_substring_of_another():
    """Why order only matters for names carrying two family substrings at once."""
    assert [
        (a, b) for a, b in itertools.permutations(HADRONIC_MODEL_FAMILIES, 2) if b in a
    ] == []


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_family_of_reproduces_both_former_implementations(name):
    mname = normalize_hadronic_model_name(name)
    expected = _former_elif_chain(mname)
    assert _former_dict_loop(mname) == expected
    assert family_of(mname) == expected


def _backend_on(path):
    """An `HDF5Backend` on one synthetic file, all four config groups injected.

    The idiom of `tests/test_data_hdf5_decode.py::backend`: with
    `paths`/`grid`/`physics`/`em` all given, `HDF5Backend.__init__` takes no
    setting from the process-wide `MCEq.config` (which it would otherwise read
    `config.em` from, unused while `enable_em` is off); only the `info` logger
    reads it (`debug_level`, `override_debug_fcn`/`override_max_level`,
    `print_module`), and no `MCEqRun` is built.
    """
    paths = SimpleNamespace(
        data_dir=path.parent,
        mceq_db_fname=path.name,
        # No EM database exists for the fixtures; enable_em is False.
        em_db_fname="no_such_em_db.h5",
    )
    grid = SimpleNamespace(e_min=None, e_max=None, dtype=None, em_standalone_grid=False)
    physics = SimpleNamespace(
        filters={
            "disabled_particles": [],
            "forced_int_cs": None,
            "replace_meson_cross_sections_with": None,
        },
        assume_nucleon_interactions_for_exotics=True,
        enable_em=False,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium=fixtures.MEDIUM,
        muon_helicity_dependence=False,
    )
    em = SimpleNamespace(air_density=None)
    return HDF5Backend(
        medium=fixtures.MEDIUM, paths=paths, grid=grid, physics=physics, em=em
    )


def test_unknown_family_is_none():
    """URQMD34 is the shipped model in no family; the empty string is in none."""
    assert family_of("URQMD34") is None
    assert family_of("") is None


def test_the_two_reactions_to_an_unknown_family_stay_different(tmp_path, monkeypatch):
    """One caller raises, the other degrades: both real reactions, driven.

    `HDF5Backend._interaction_db_single` raises `ValueError("Unknown
    equivalence table for", mname)` -- but only after
    `_check_subgroup_exists` found the model in the file, so the synthetic
    database is rebuilt here with `URQMD34` as its model name
    (`build_database` reads the module constant `MODEL` at call time).
    `mapped_cross_section` never raises: with no family it skips the
    equivalence mapping and falls through to `family_fallback`, so a Lambda
    (3122) whose column is absent from `index_d` gets the proton column.
    Both are the behaviour today, recorded so that unifying them is a
    visible change and not an accident.
    """
    monkeypatch.setattr(fixtures, "MODEL", "URQMD34")
    path = fixtures.build_database(tmp_path / "urqmd34_1d.h5")
    backend = _backend_on(path)

    with pytest.raises(ValueError, match="Unknown equivalence table"):
        backend._interaction_db_single("URQMD34")

    proton_column = np.array([1.0, 2.0, 3.0])
    neutron_column = np.array([4.0, 5.0, 6.0])
    # 3122 is absent by construction. The neutron column is there so that a
    # family mapping of 3122 -> 2112 (SIBYLL21, QGSJETII and PYTHIA8 all have
    # one) would be visible: with no family the mapping step is skipped and the
    # baryon fallback lands on the proton column; with one it would land on the
    # neutron column and `is proton_column` fails.
    index_d = {2212: proton_column, 2112: neutron_column}
    assert family_fallback(3122) == [2212, 2212]
    assert mapped_cross_section(index_d, 3122, "URQMD34") is proton_column
    # and the fallthrough is the only route: with no proton column there is
    # nothing to fall through to.
    assert mapped_cross_section({}, 3122, "URQMD34") is None


def test_the_two_former_orders_differ_only_where_no_database_goes():
    """The single divergence class, pinned so the choice of order is visible."""
    probe = "SIBYLL21SIBYLL23"
    assert _former_elif_chain(probe) == "SIBYLL21"
    assert _former_dict_loop(probe) == "SIBYLL23"
    # `family_of` keeps the `elif` chain's answer -- the interaction-matrix path,
    # which is the one that raises rather than degrading silently.
    assert family_of(probe) == "SIBYLL21"
    # and no shipped name is in that class.
    for name in MODEL_NAMES:
        mname = normalize_hadronic_model_name(name)
        assert sum(f in mname for f in HADRONIC_MODEL_FAMILIES) <= 1


#: Sentinel columns, one per family representative, so which branch `get_cs`
#: took is readable off its return value.
_CS_COLUMNS = {211: 20.0, 321: 15.0, 2212: 30.0}


def _get_cs_column(parent):
    """Which family column the real `get_cs` resolves `parent` to.

    Returns the base PDG id whose sentinel column came back, or `"zeros"` for
    the lepton branch, or `"scalar"` for the else branch (B6: it returns a
    bare float rather than a vector). Built on `object.__new__` with the three
    attributes `get_cs` reads, the idiom `tests/test_data_bug_pins.py` uses.
    """
    import numpy as np

    from MCEq.data import InteractionCrossSections
    from MCEq.data.energy_grid import EnergyGrid

    db = object.__new__(InteractionCrossSections)
    db.energy_grid = EnergyGrid(c=np.logspace(0.0, 3.0, 5), b=None, w=None, d=5)
    db.index_d = {
        pdg: np.full(db.energy_grid.d, value) for pdg, value in _CS_COLUMNS.items()
    }
    db.mbarn2cm2 = 1.0

    result = db.get_cs(parent, mbarn=True)
    if np.ndim(result) == 0:
        return "scalar"
    if np.all(result == 0.0):
        return "zeros"
    unique = set(np.asarray(result).tolist())
    assert len(unique) == 1, unique
    value = unique.pop()
    return next(pdg for pdg, v in _CS_COLUMNS.items() if v == value)


@pytest.mark.parametrize(
    ("projectile", "expected"),
    [
        (321, [321, 321]),
        (-321, [-321, 321]),
        (130, [321, 321]),
        (310, [321, 321]),
        (311, [321, 321]),
        (333, [321, 321]),
        (211, [211, 211]),
        (-211, [-211, 211]),
        (111, [211, 211]),
        (2212, [2212, 2212]),
        (-2212, [-2212, 2212]),
        (3122, [2212, 2212]),
        (-3122, [-2212, 2212]),
        (13, []),
        (11, []),
        (22, []),
        (10313, []),
        (5122, []),
    ],
)
def test_family_fallback_table(projectile, expected):
    """`family_fallback`'s whole contract, driven directly.

    `9387ab3` rewrote this body from an `elif` chain and pinned it with a
    one-off developer sweep rather than a test, leaving two of its three
    windows with no regression coverage at all.

    Note 130 satisfies BOTH the first window (`apid in (130, 310, 311)`) and
    the second (`100 < apid < 300`); it keeps the kaon representative because
    the branches are tested in order and each returns, not because the windows
    are disjoint.
    """
    assert family_fallback(projectile) == expected


def test_the_two_pdg_policies_are_not_unified():
    """The deliberate non-change, driven through BOTH real callables.

    `family_fallback` (the interaction path) and `InteractionCrossSections.
    get_cs` (the cross-section path) are two of the three PDG -> representative
    policies in the tree. Unifying them moves blended cross sections, so Phase
    4 leaves both in place -- and this test fails if anyone unifies them, which
    is the property its predecessor claimed and did not have.
    """
    # Where they agree: every hadron the databases actually carry.
    for apid in (211, 321, 2112, 2212, 3122):
        assert family_fallback(apid)[1] == _get_cs_column(apid)

    # Disagreement 1 -- the excited kaons. `get_cs` names them explicitly;
    # `family_fallback` has no window for them.
    for apid in (10313, 10323):
        assert family_fallback(apid) == []
        assert _get_cs_column(apid) == 321

    # Disagreement 2 -- leptons. `get_cs` has a dedicated zeros branch.
    for apid in (13, 11, 22):
        assert family_fallback(apid) == []
        assert _get_cs_column(apid) == "zeros"

    # Disagreement 3 -- everything outside all three hadron windows, which is
    # the class the predecessor test omitted entirely. B6: a scalar, not a
    # vector; pinned as today's behaviour in tests/test_data_bug_pins.py.
    for apid in (5122, 5332):
        assert family_fallback(apid) == []
        assert _get_cs_column(apid) == "scalar"


def test_normalize_hadronic_model_name_contract():
    """The whole contract, as literals rather than through a shared path.

    `test_family_of_reproduces_both_former_implementations` normalizes once
    and feeds the same string to all three implementations, so it says nothing
    about the normalizer itself -- any normalizer gives a self-consistent
    answer there.
    """
    assert normalize_hadronic_model_name("SIBYLL-2.3d") == "SIBYLL23D"
    assert normalize_hadronic_model_name("dpmjet-III-19.1") == "DPMJETIII191"
    assert normalize_hadronic_model_name("") == ""
    assert normalize_hadronic_model_name("SIBYLL23D") == "SIBYLL23D"
