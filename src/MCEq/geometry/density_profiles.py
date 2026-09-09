"""Compatibility exports of the atmosphere classes from :mod:`MCEq.environment`.

The atmospheres now live in ``environment/{base,corsika,isothermal,
msis00,tabulated,target}.py``. This module keeps ``from MCEq.geometry.density_profiles import ...``
working, including the ``import *`` form that
``docs/examples/Plot_density_depth_relations.ipynb`` relies on.

``__all__`` is explicit and preserves the exact 19 names the star-import bound
before the split -- ``np``, ``join``, ``ABCMeta``, ``abstractmethod`` and
``info`` included. The notebook calls ``np.linspace`` without importing numpy
itself, so the exported ``np`` name has to stay.
``tests/geometry/test_geometry_shims.py`` checks this export list.
"""

from abc import ABCMeta, abstractmethod  # noqa: F401  (star-import surface)
from os.path import join  # noqa: F401  (star-import surface)

import numpy as np  # noqa: F401  (star-import surface)

from MCEq.environment.base import EarthsAtmosphere
from MCEq.environment.column import fit_column_splines
from MCEq.environment.corsika import CorsikaAtmosphere
from MCEq.environment.isothermal import IsothermalAtmosphere
from MCEq.environment.location_centered import LocationCenteredMixin
from MCEq.environment.msis00 import (
    MSIS00Atmosphere,
    MSIS00IceCubeCentered,
    MSIS00KM3NeTCentered,
    MSIS00LocationCentered,
)
from MCEq.environment.parameters import (  # noqa: F401
    KM3NET_DETECTORS as _KM3NET_DETECTORS,
)
from MCEq.environment.parameters import (
    MONTH_TO_DAY_OF_YEAR,
    get_atmosphere_parameters,
    list_available_corsika_atmospheres,
)
from MCEq.environment.tabulated import AIRSAtmosphere
from MCEq.environment.target import GeneralizedTarget
from MCEq.misc import info  # noqa: F401  (star-import surface)

__all__ = [
    "ABCMeta",
    "AIRSAtmosphere",
    "CorsikaAtmosphere",
    "EarthsAtmosphere",
    "GeneralizedTarget",
    "IsothermalAtmosphere",
    "LocationCenteredMixin",
    "MONTH_TO_DAY_OF_YEAR",
    "MSIS00Atmosphere",
    "MSIS00IceCubeCentered",
    "MSIS00KM3NeTCentered",
    "MSIS00LocationCentered",
    "abstractmethod",
    "fit_column_splines",
    "get_atmosphere_parameters",
    "info",
    "join",
    "list_available_corsika_atmospheres",
    "np",
]

# The MSIS21 names are resolved lazily through PEP 562 so that touching this
# module does not require the optional MSIS21 backend. They are deliberately
# absent from __all__, as they were before the split: a module __getattr__ does
# not extend `import *`.
_MSIS21_EXPORTS = (
    "MSIS21Atmosphere",
    "MSIS21LocationCentered",
    "MSIS21IceCubeCentered",
    "MSIS21KM3NeTCentered",
)


def __getattr__(name):
    if name in _MSIS21_EXPORTS:
        from MCEq.environment import msis21

        return getattr(msis21, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_MSIS21_EXPORTS])
