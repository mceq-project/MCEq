"""Compatibility re-exports from :mod:`MCEq.environment.geometry`."""

from MCEq.environment.geometry import (  # noqa: F401
    EarthGeometry,
    chirkin_cos_theta_star,
    np,
)

__all__ = [
    "EarthGeometry",
    "chirkin_cos_theta_star",
    "np",
]
