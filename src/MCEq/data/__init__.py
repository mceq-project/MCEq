from os.path import (  # noqa: F401  (recorded surface; used by hdf5_backend now)
    isfile,
    join,
)

# The backend class lives in `MCEq.data.hdf5_backend` and the
# blending/energy-grid helpers are reached by their consumers' own imports.
# The bindings below stay because they are the package's recorded public
# surface (tests/test_phase4_surface.py, both bounds); `HDF5Backend` is the
# name `MCEq.core` constructs through the dotted path.
from MCEq.data.blending import (  # noqa: F401
    blend_cross_sections,
    blend_yields,
    he_le_weight,
)
from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts  # noqa: F401

# `equivalences` is both a submodule of this package and the dict that submodule
# defines. The import machinery binds the *module* onto `MCEq.data` while loading
# it; the `from ... import equivalences` below runs afterwards and rebinds the
# name to the dict, which is the object `MCEq.data.equivalences` has always been
# (pinned by tests/test_phase4_surface.py). Reach the module itself through
# `sys.modules`, never through the package attribute.
from MCEq.data.equivalences import (  # noqa: F401  (the dict is the surface; users moved)
    apply_equivalences,
    equivalences,
    reverse_equivalences,
)
from MCEq.data.hdf5_backend import HDF5Backend as HDF5Backend
from MCEq.data.model_names import (  # noqa: F401  (family_of is surface only)
    family_of,
    normalize_hadronic_model_name,
)
from MCEq.misc import info

#: Since the classes moved out of this file, `automodapi` needs the explicit
#: list to keep rendering them on docs/api-reference/data.rst (an automodapi
#: page drops names whose `__module__` is not the page's module unless
#: `__all__` claims them). Same list the upper-bound test records. Side
#: effect: `import *` no longer binds the incidental module names (`np`,
#: the submodules) -- callers who wanted those always imported them by name.
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

# The four table-consumer classes (§13.2 item 4): one submodule each, re-bound
# here so `MCEq.data.Interactions` etc. stay the recorded public surface and the
# dotted paths `MCEq.core` constructs through keep resolving.
from MCEq.data.continuous_losses import ContinuousLosses  # noqa: F401
from MCEq.data.cross_sections import InteractionCrossSections  # noqa: F401
from MCEq.data.decay_tables import Decays  # noqa: F401
from MCEq.data.interaction_tables import Interactions  # noqa: F401
