"""Hadronic-model names: normalisation and family selection.

Plan section 1's ``data/model_names.py``. Two pure string functions and nothing
else -- no HDF5, no numpy, no config -- so the module sits at the bottom of the
``data`` layer and the equivalence *tables* can move on top of it later
(``data/equivalences.py``).

:func:`normalize_hadronic_model_name` moved here verbatim from :mod:`MCEq.misc`,
which is C1's bottom layer and therefore the wrong home for a data-layer
concern; ``misc`` keeps a forwarding shim for it.

:func:`family_of` is the *name -> family* half of what the database code did
inline in two places, and only that half. What it is emphatically **not** is the
*PDG -> family representative* fallback, of which this tree carries three
mutually inconsistent copies (``HDF5Backend._mapped_cross_section``'s candidate
list, ``InteractionCrossSections.get_cs``'s ``_family_cs`` chain, and the
``get_cs`` boundaries themselves). They disagree on 10313/10323 and on leptons;
unifying them changes blended cross sections, so they stay where they are until
the commit that owns that policy. See the commit message for the measurement.
"""

#: Model-name substrings that select an equivalence table, in the order the
#: former ``elif`` chain in ``HDF5Backend._interaction_db_single`` tested them.
#: The keys of ``MCEq.data.equivalences``, which the loop in
#: ``_mapped_cross_section`` walked instead -- same set, and the same answer for
#: every model name that does not contain two of these at once.
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

    Substring containment, first match in :data:`HADRONIC_MODEL_FAMILIES` wins,
    exactly as the two former inline implementations did. ``None`` for a name in
    no family (``URQMD34`` is the shipped example); the two call sites keep their
    own -- deliberately different -- reactions to that, one raising
    ``ValueError`` and one falling through to an unmapped cross section.

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
