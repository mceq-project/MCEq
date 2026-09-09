"""Compatibility re-exports of the CORSIKA atmosphere extension."""

from MCEq.environment._ext.corsikaatm import (  # noqa: F401
    corsika_acc,
    corsika_get_density,
    corsika_get_m_overburden,
    planar_rho_inv,
)

__all__ = [
    "corsika_acc",
    "corsika_get_density",
    "corsika_get_m_overburden",
    "planar_rho_inv",
]
