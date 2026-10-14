"""Hadronic database access: HDF5 backend, tables and model blending."""

from os.path import (  # noqa: F401  (compatibility export)
    isfile,
    join,
)

from MCEq.data.blending import (  # noqa: F401
    blend_cross_sections,
    blend_yields,
    he_le_weight,
)
from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts  # noqa: F401

# `MCEq.data.equivalences` is the dict, which shadows the submodule of that name.
from MCEq.data.equivalences import (  # noqa: F401
    apply_equivalences,
    equivalences,
    reverse_equivalences,
)
from MCEq.data.hdf5_backend import HDF5Backend as HDF5Backend
from MCEq.data.model_names import (  # noqa: F401  (compatibility export)
    family_of,
    normalize_hadronic_model_name,
)
from MCEq.misc import info

__all__ = [
    "ContinuousLosses",
    "Decays",
    "EnergyGrid",
    "HDF5Backend",
    "InteractionCrossSections",
    "Interactions",
    "apply_equivalences",
    "blend_cross_sections",
    "blend_yields",
    "equivalences",
    "family_of",
    "he_le_weight",
    "info",
    "isfile",
    "join",
    "normalize_hadronic_model_name",
    "reverse_equivalences",
]

from MCEq.data.continuous_losses import ContinuousLosses  # noqa: F401
from MCEq.data.cross_sections import InteractionCrossSections  # noqa: F401
from MCEq.data.decay_tables import Decays  # noqa: F401
from MCEq.data.interaction_tables import Interactions  # noqa: F401
