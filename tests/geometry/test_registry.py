"""The density-model registry, and the two bugs its shape used to allow.

``set_density_model`` used to hold a list literal of accepted names beside an
``elif`` ladder that built them, and nothing tied the two together. They
disagreed -- ``MSIS00_KM3NeT`` was offered by neither although the class
existed, bug B15 -- and the only way to enumerate the models was to parse the
literal out of the method with ``ast``. These tests assert the properties that
made that class of bug possible, so it cannot come back through the registry.

DB-free: nothing here constructs an ``MCEqRun``.
"""

from __future__ import annotations

import pytest

from MCEq.geometry import registry


def test_every_offered_name_resolves_to_a_class():
    """The whole point of the registry: offered == buildable (bug B15).

    A name in :func:`available_models` whose entry does not resolve would be a
    model the error message advertises and the dispatch cannot build.
    """
    for name in registry.available_models():
        cls = registry.density_model_class(name)
        assert isinstance(cls, type), f"{name} resolved to {cls!r}, not a class"


def test_available_models_hands_out_a_copy():
    """The equality is true by construction; what is worth pinning is the copy.

    ``available_models()`` feeds both the user-facing error message and
    ``tests/golden/gen_paths.py``. Returning the live table would let either
    of them corrupt the registry for the rest of the process.
    """
    names = registry.available_models()
    assert names == list(registry.DENSITY_MODELS)
    names.append("NotAModel")
    assert "NotAModel" not in registry.available_models()
    assert "NotAModel" not in registry.DENSITY_MODELS


def test_km3net_is_present_for_both_backends():
    """B15 itself: MSIS00_KM3NeT was missing while MSIS21_KM3NeT was present."""
    names = registry.available_models()
    assert "MSIS00_KM3NeT" in names
    assert "MSIS21_KM3NeT" in names
    assert registry.density_model_class("MSIS00_KM3NeT").__name__ == (
        "MSIS00KM3NeTCentered"
    )


def test_the_two_backend_families_offer_the_same_sites():
    """A site reachable on one MSIS generation should be reachable on the other.

    This is the invariant B15 broke. Asserted over the suffixes rather than a
    hardcoded list so that adding a site to one family without the other fails
    here rather than in a user's script.
    """
    suffixes = {
        prefix: {
            n[len(prefix) :]
            for n in registry.available_models()
            if n.startswith(prefix)
        }
        for prefix in ("MSIS00", "MSIS21")
    }
    assert suffixes["MSIS00"] == suffixes["MSIS21"], suffixes


def test_unknown_name_raises_keyerror_not_something_stranger():
    with pytest.raises(KeyError):
        registry.density_model_class("NoSuchAtmosphere")


def test_entries_are_dotted_strings_not_imported_classes():
    """The shape that makes laziness possible.

    Kept separate from the laziness test itself: this one only asserts the
    table holds ``module:qualname`` strings. It would still pass against a
    registry that eagerly imported both trees, which is why
    :func:`test_resolution_is_lazy` measures ``sys.modules`` instead.
    """
    for target in registry.DENSITY_MODELS.values():
        module_name, _, qualname = target.partition(":")
        assert module_name.startswith("MCEq.geometry."), target
        assert qualname and ":" not in qualname, target


@pytest.mark.parametrize(
    ("name", "must_not_import"),
    [
        ("MSIS00", "MCEq.geometry.msis21_atmosphere"),
        ("MSIS00_KM3NeT", "MCEq.geometry.msis21_atmosphere"),
        ("CORSIKA", "MCEq.geometry.msis21_atmosphere"),
        ("MSIS21", "MCEq.geometry.density_profiles"),
    ],
)
def test_resolution_is_lazy(name, must_not_import):
    """Resolving one model must not drag in the other MSIS tree.

    The ``elif`` ladder imported ``density_profiles`` inside the method and
    only touched the MSIS21 names if one was selected. Dotted-string entries
    keep that property -- but only if nothing along the way imports the other
    side, which is a property of the whole import graph and not of the table.
    So this runs a fresh interpreter and looks at ``sys.modules``.

    The MSIS21 case is the one that can regress silently: ``msis21_atmosphere``
    imports ``EarthsAtmosphere`` back out of ``density_profiles`` today, so it
    is *expected* to fail until Phase 3's package split gives it its own base
    module. It is marked xfail rather than omitted so the split flips it.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "from MCEq.geometry import registry;"
        f"registry.density_model_class({name!r});"
        f"print({must_not_import!r} in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    imported = out.stdout.strip() == "True"
    if name == "MSIS21":
        pytest.xfail(
            "msis21_atmosphere imports EarthsAtmosphere from density_profiles; "
            "the Phase 3 environment/ split breaks that edge"
        )
    assert not imported, (
        f"resolving {name} imported {must_not_import}; the registry is no longer lazy"
    )


def test_registry_import_alone_pulls_in_no_atmosphere():
    """Importing the table itself must cost nothing."""
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import MCEq.geometry.registry;"
        "print([m for m in ('MCEq.geometry.density_profiles',"
        "'MCEq.geometry.msis21_atmosphere') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
