"""Compatibility shim: see :mod:`MCEq.environment.msis00_backend`."""

from MCEq.environment.msis00_backend import (  # noqa: F401
    NRLMSISE00Base,
    cNRLMSISE00,
)

__all__ = [
    "NRLMSISE00Base",
    "cNRLMSISE00",
]
