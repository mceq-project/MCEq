"""Hadronic-model names: normalisation and family selection.

Two pure string functions and nothing else -- no HDF5, no numpy, no config --
so the module sits at the bottom of the ``data`` layer, below the equivalence
tables in ``data/equivalences.py``. ``MCEq.misc`` keeps a forwarding name for
:func:`normalize_hadronic_model_name`.

:func:`family_of` maps a normalized name onto its model family. It is **not**
the *PDG -> family representative* fallback: that logic lives separately in
``data.equivalences.family_fallback`` and
``InteractionCrossSections.get_cs``, and the rules differ -- they disagree on
10313/10323 and on leptons, so unifying them would change blended cross
sections. See ``tests/test_data_model_names.py``.
"""

#: Model-name substrings that select an equivalence table, in matching
#: priority order. These are the keys of ``MCEq.data.equivalences``; a name
#: containing two of them at once resolves to the first listed.
HADRONIC_MODEL_FAMILIES = (
    "SIBYLL21",
    "SIBYLL23",
    "QGSJET01",
    "QGSJETII",
    "DPMJET",
    "EPOSLHC",
    "PYTHIA8",
    "FLUKA",
)


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


def family_of(name):
    """The model family a normalized hadronic-model name belongs to.

    Substring containment; the first match in :data:`HADRONIC_MODEL_FAMILIES`
    wins. Returns ``None`` for a name in no family (``URQMD34`` is the shipped
    example); callers decide how to handle an unknown family — one raises
    ``ValueError`` and one falls through to an unmapped cross section.

    Args:
        name: str
            Model name, already through :func:`normalize_hadronic_model_name`.

    Returns:
        str or None
            The matching entry of :data:`HADRONIC_MODEL_FAMILIES`, or ``None``.
    """
    for family in HADRONIC_MODEL_FAMILIES:
        if family in name:
            return family
    return None
