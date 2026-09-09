from os.path import (  # noqa: F401  (compatibility export)
    isfile,
    join,
)

# Public exports for database access and table utilities. The backend class
# lives in `MCEq.data.hdf5_backend`; `CascadeSystem` constructs it as
# `MCEq.data.HDF5Backend`. The bindings below are the package's public surface,
# checked by tests/test_phase4_surface.py.
from MCEq.data.blending import (  # noqa: F401
    blend_cross_sections,
    blend_yields,
    he_le_weight,
)
from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts  # noqa: F401

# `equivalences` is both a submodule of this package and the dict that submodule
# defines. The import machinery binds the *module* onto `MCEq.data` while loading
# it; the `from ... import equivalences` below runs afterwards and rebinds the
# name to the dict. The package attribute `equivalences` is the exported
# dictionary, not the submodule: import submodule members explicitly, or obtain
# the module from `sys.modules`.
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

#: `__all__` defines the public names exposed to star imports and API
#: documentation: automodapi drops names whose `__module__` is not the page's
#: module unless `__all__` claims them, and `import *` binds only these names.
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

# Re-export the four table interfaces constructed by `CascadeSystem`, so
# `MCEq.data.Interactions` etc. keep resolving as public paths.
from MCEq.data.continuous_losses import ContinuousLosses  # noqa: F401
from MCEq.data.cross_sections import InteractionCrossSections  # noqa: F401
from MCEq.data.decay_tables import Decays  # noqa: F401
from MCEq.data.interaction_tables import Interactions  # noqa: F401
