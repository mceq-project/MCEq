"""Hadronic model name normalisation and model families."""

import re

# anfe: aaaah. It's enough to bundle all model links/setup in a single file with little prose. also equivalences

#: Model-name substrings selecting an equivalence table, in matching order.
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
    return re.sub("[-.]", "", name).upper()


def family_of(name):
    """First entry of :data:`HADRONIC_MODEL_FAMILIES` contained in ``name``, or None.

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
