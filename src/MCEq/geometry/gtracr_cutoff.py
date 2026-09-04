"""Compatibility shim: see :mod:`MCEq.environment.geomagnetic.cutoff`.

``docs/examples/Full_sky_carousel.ipynb`` documents
``MCEq.geometry.gtracr_cutoff.CACHE_VERSION`` as the way to invalidate a cutoff
cache, so this path has to keep resolving.

The private names come along because ``tests/golden/gen_environment.py`` pins
the cache key, the native grid shape and the default mass groups through them.
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
