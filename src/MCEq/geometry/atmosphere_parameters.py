"""Compatibility shim: see :mod:`MCEq.environment.parameters`.

``__all__`` is explicit and lists exactly what this module's namespace held
before the split, plus ``KM3NET_DETECTORS``, which moved here from
``density_profiles`` so that both MSIS backends can reach one site table.

It is not decoration. ``sphinx-automodapi`` only disables its "is this object
local to the module?" filter when ``__all__`` is present; without it, every
re-exported *function* is judged non-local (its ``__module__`` is
``MCEq.environment.parameters``) and silently vanishes from the rendered API
page, while the data objects survive because they have no ``__name__``. A
``from ... import *`` shim therefore loses exactly the six ``get_*`` /
``list_*`` functions this page documents.
"""

from MCEq.environment.parameters import (  # noqa: F401
    DAY_TIMES_SEC,
    DEFAULT_AP,
    DEFAULT_F107,
    DEFAULT_F107A,
    KM3NET_DETECTORS,
    LOCATIONS,
    MONTH_TO_DAY_OF_YEAR,
    _corsika_reference_table,
    _cosika_atmosphere_params,
    get_atmosphere_parameters,
    get_day_time_seconds,
    get_location_data,
    get_month_day_of_year,
    get_nrlmsise00_defaults,
    list_available_corsika_atmospheres,
)

__all__ = [
    "DAY_TIMES_SEC",
    "DEFAULT_AP",
    "DEFAULT_F107",
    "DEFAULT_F107A",
    "KM3NET_DETECTORS",
    "LOCATIONS",
    "MONTH_TO_DAY_OF_YEAR",
    "get_atmosphere_parameters",
    "get_day_time_seconds",
    "get_location_data",
    "get_month_day_of_year",
    "get_nrlmsise00_defaults",
    "list_available_corsika_atmospheres",
]
