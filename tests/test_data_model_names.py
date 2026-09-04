"""`MCEq.data.model_names`: the name -> family half of the equivalence lookup.

`family_of` replaced two inline implementations that walked the same eight family
substrings in two different orders -- the `elif` chain of
`HDF5Backend._interaction_db_single` (`SIBYLL21` first) and the
`equivalences.items()` loop of `HDF5Backend._mapped_cross_section` (`SIBYLL23`
first). This file pins that the replacement is exactly what both did, and pins
the one input class where the two orders could ever have differed.

It also pins the *non*-unification: the two PDG -> family-representative
fallbacks are untouched and still disagree. That is deliberate (they change
blended cross sections), so the disagreement is written down rather than fixed.

DB-free: the model-name universe below is a literal, measured off the ten
shipped `src/MCEq/data/*.h5` files, so the test runs without opening one.
"""

from __future__ import annotations

import itertools

import pytest

from MCEq.data import equivalences
from MCEq.data.model_names import (
    HADRONIC_MODEL_FAMILIES,
    family_of,
    normalize_hadronic_model_name,
)

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


def _former_dict_loop(model_name):
    """The `for family, mapping in equivalences.items()` loop, transcribed."""
    for family in equivalences:
        if family in model_name:
            return family
    return None


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


def test_unknown_family_is_none_and_the_two_reactions_stay_different():
    """URQMD34 is the shipped model in no family.

    `_interaction_db_single` raises `ValueError` on it and `_mapped_cross_section`
    falls through to an unmapped cross section. `family_of` reports the fact and
    leaves both reactions where they are.
    """
    assert family_of("URQMD34") is None
    assert family_of("") is None


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


def test_pdg_family_fallbacks_are_not_unified():
    """The deliberate non-change: the two PDG -> representative policies disagree.

    Transcribed from `HDF5Backend._mapped_cross_section` and
    `InteractionCrossSections.get_cs`. Unifying them moves blended cross
    sections, so Phase 4's naming commit leaves both in place.
    """

    def mapped_cross_section(apid):
        if apid in (130, 310, 311) or 300 < apid < 1000:
            return 321
        if 100 < apid < 300:
            return 211
        if 1000 < apid < 5000:
            return 2212
        return None

    def get_cs(apid):
        if 100 < apid < 300 and apid != 130:
            return 211
        if 300 < apid < 1000 or apid in (130, 10313, 10323):
            return 321
        if 1000 < apid < 5000:
            return 2212
        if 5 < apid < 23:
            return "zeros"
        return "zero-cross-section"

    # They agree on every hadron the databases actually carry.
    for apid in (130, 211, 310, 311, 321, 2112, 2212, 3122, 4122):
        assert mapped_cross_section(apid) == get_cs(apid)
    # And disagree on exactly two classes: the excited kaons and the leptons.
    assert (mapped_cross_section(10313), get_cs(10313)) == (None, 321)
    assert (mapped_cross_section(10323), get_cs(10323)) == (None, 321)
    assert (mapped_cross_section(13), get_cs(13)) == (None, "zeros")
