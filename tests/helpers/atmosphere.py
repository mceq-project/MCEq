"""Which call form an atmosphere class uses for its density tail."""

from __future__ import annotations

#: Class names that need the optional ``nrlmsis`` package.
MSIS21_CLASSES = frozenset(
    {
        "MSIS21Atmosphere",
        "MSIS21LocationCentered",
        "MSIS21IceCubeCentered",
        "MSIS21KM3NeTCentered",
    }
)

#: Atmosphere classes the section cannot construct, with the reason. Reported
#: as a golden array so the gap is a compared fact rather than a claim in prose.


def h_form(cls_name: str) -> str:
    """Which ``geom.h`` call form this class's spline tail uses.

    Four tails compute ``rho(X)``. ``EarthsAtmosphere`` (the CORSIKA,
    isothermal and named-MSIS00 route, and the MSIS00 detector-centred models
    in single-azimuth mode, which delegate to it) and the azimuth-averaged
    ``MSIS00LocationCentered`` tail both call ``geom.h`` once per ``dl``
    sample; the two MSIS21 tails call it once on the whole vector. So the split
    is exactly MSIS21 against everything else.

    It matters because the two forms do not agree bitwise: ``h`` squares
    ``A_1 + l - dl``, and numpy lowers ``arr ** 2`` to a multiply while a
    float64 scalar goes through libm ``pow``, which is 1 ULP off at some
    arguments. The ``- r_E`` at the end turns that into 1 ULP of 6.37e8 cm.
    """
    return "array" if cls_name in MSIS21_CLASSES else "scalar"
