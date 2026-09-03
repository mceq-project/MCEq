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


def test_available_models_is_exactly_the_table():
    """No second list to fall out of step with the first."""
    assert registry.available_models() == list(registry.DENSITY_MODELS)


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


def test_resolution_is_lazy():
    """Importing the registry must not import either MSIS tree.

    The ``elif`` ladder imported ``density_profiles`` inside the method and
    only touched the MSIS21 names if one was selected. Entries are dotted
    strings to keep that property, so selecting an MSIS00 model does not drag
    in the MSIS21 module and vice versa.
    """
    for target in registry.DENSITY_MODELS.values():
        module_name, _, qualname = target.partition(":")
        assert module_name.startswith("MCEq.geometry."), target
        assert qualname and ":" not in qualname, target
