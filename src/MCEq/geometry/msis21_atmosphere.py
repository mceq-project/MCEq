"""Compatibility shim: MSIS 2.1 now lives in :mod:`MCEq.environment.msis21`."""

from MCEq.environment.msis21 import (
    MSIS21Atmosphere,
    MSIS21IceCubeCentered,
    MSIS21KM3NeTCentered,
    MSIS21LocationCentered,
)

# Re-exported under its old private name: tests/geometry/test_density_profiles.py
# asserts it is the *same object* as density_profiles._KM3NET_DETECTORS, which is
# the property that says the two MSIS backends read one site table rather than
# two copies. Both now alias environment.parameters.KM3NET_DETECTORS.
from MCEq.environment.parameters import (  # noqa: F401
    KM3NET_DETECTORS as _KM3NET_DETECTORS,
)

__all__ = [
    "MSIS21Atmosphere",
    "MSIS21IceCubeCentered",
    "MSIS21KM3NeTCentered",
    "MSIS21LocationCentered",
]
