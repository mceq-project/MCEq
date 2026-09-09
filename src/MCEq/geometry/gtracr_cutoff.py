"""Compatibility exports from :mod:`MCEq.environment.geomagnetic.cutoff`.

``docs/examples/Full_sky_carousel.ipynb`` documents
``MCEq.geometry.gtracr_cutoff.CACHE_VERSION`` as the way to invalidate a cutoff
cache, so this path has to keep resolving. The private helpers are imported by
``tests/golden/gen_environment.py``.
"""

from MCEq.environment.geomagnetic.cutoff import (  # noqa: F401
    _DEFAULT_MASS_GROUPS,
    _NATIVE_N_AZ,
    _NATIVE_N_ZEN,
    CACHE_VERSION,
    _cache_dir,
    _cache_key,
    _gtracr_location_from_atmosphere,
    build_phi0_with_cutoff,
    get_cutoff_map,
)

__all__ = [
    "CACHE_VERSION",
    "build_phi0_with_cutoff",
    "get_cutoff_map",
]
