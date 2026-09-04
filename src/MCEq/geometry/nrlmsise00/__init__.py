"""Compatibility shim: see :mod:`MCEq.environment._ext.nrlmsise00`."""

from MCEq.environment._ext.nrlmsise00 import (  # noqa: F401
    ap_array,
    msis,
    nrlmsise_flags,
    nrlmsise_input,
    nrlmsise_output,
)

__all__ = [
    "ap_array",
    "msis",
    "nrlmsise_flags",
    "nrlmsise_input",
    "nrlmsise_output",
]
