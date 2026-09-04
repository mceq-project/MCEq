"""Compatibility shim: see :mod:`MCEq.environment.msis00_backend`."""

from MCEq.environment.msis00_backend import (  # noqa: F401
    NRLMSISE00Base,
    cNRLMSISE00,
    test,
)

# ``test`` is a module-level demo function, not an incidental import, so it was
# part of this module's public surface before the split even though nothing in
# the repo calls it.
__all__ = [
    "NRLMSISE00Base",
    "cNRLMSISE00",
    "test",
]
