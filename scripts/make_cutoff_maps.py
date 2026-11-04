"""Tabulate the geomagnetic rigidity cutoffs of the named sites.

Writes ``src/MCEq/geometry/cutoff_maps.json``: for every site of
:func:`MCEq.geometry.atmosphere_parameters.site_coordinates`, the effective
cutoff R_eff in GV of downgoing protons on a 5° grid of zenith (0–90°) and
azimuth (0–355°, arrival direction, 0° is from North, 90° from East). R_eff is
the penumbra-resolved cutoff of Cooke et al. (1991): R_U minus the allowed
rigidity bands below it. Each direction is traced from the point where its line
of sight crosses 100 km altitude (the primary's entry into the atmosphere in
the collinear 1D picture), with gtracr's penumbra mode, IGRF at
:data:`MCEq.geometry.cutoff.CUTOFF_DATE`, the default rigidity-aware escape
criterion and a 0.5 % rigidity ladder.

Usage: python scripts/make_cutoff_maps.py [n_processes]

:func:`trace` needs only gtracr and numpy, so batches of :func:`tasks` can run
on a cluster; :func:`write_table` assembles the results.
"""

import contextlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PARTICLE_ALTITUDE_KM = 100.0
EARTH_RADIUS_KM = 6371.2  # gtracr's spherical Earth
ZENITH_DEG = np.arange(0.0, 90.1, 5.0)
AZIMUTH_DEG = np.arange(0.0, 360.0, 5.0)
R_MIN_GV, R_MAX_GV = 0.05, 60.0
CUTOFF_DATE = "2025-01-01"  # = MCEq.geometry.cutoff.CUTOFF_DATE (checked in tasks())
OUT = Path(__file__).parents[1] / "src/MCEq/geometry/cutoff_maps.json"


def _enu(lat, lon):
    """Unit east, north, up vectors (geocentric) at latitude/longitude in degrees."""
    la, lo = np.radians(lat), np.radians(lon)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return e, n, u


def entry_point(lat, lon, alt_km, zenith, azimuth):
    """Latitude, longitude, local zenith and azimuth (degrees) of the point where
    the line of sight towards (zenith, azimuth) crosses PARTICLE_ALTITUDE_KM."""
    e, n, u = _enu(lat, lon)
    z, a = np.radians(zenith), np.radians(azimuth)
    d = np.sin(z) * np.sin(a) * e + np.sin(z) * np.cos(a) * n + np.cos(z) * u
    det = (EARTH_RADIUS_KM + alt_km) * u
    r_top = EARTH_RADIUS_KM + PARTICLE_ALTITUDE_KM
    b = det @ d
    s = -b + np.sqrt(b * b - det @ det + r_top**2)
    p = det + s * d
    plat = np.degrees(np.arcsin(p[2] / r_top))
    plon = np.degrees(np.arctan2(p[1], p[0]))
    pe, pn, pu = _enu(plat, plon)
    pzen = np.degrees(np.arccos(np.clip(d @ pu, -1.0, 1.0)))
    paz = np.degrees(np.arctan2(d @ pe, d @ pn)) % 360.0
    return plat, plon, pzen, paz


def tasks():
    """Sites and, per (site, zenith, azimuth), the entry point and local direction."""
    from MCEq.geometry import cutoff
    from MCEq.geometry.atmosphere_parameters import (
        KM3NET_DETECTORS,
        LOCATIONS,
        site_coordinates,
    )

    assert cutoff.CUTOFF_DATE == CUTOFF_DATE
    sites = list(LOCATIONS) + list(KM3NET_DETECTORS)
    out = []
    for s in sites:
        _, lon, lat = site_coordinates(s)
        alt = LOCATIONS[s][2] / 1e5 if s in LOCATIONS else 0.0
        for z in ZENITH_DEG:
            for a in AZIMUTH_DEG:
                out.append(entry_point(lat, lon, alt, z, a))
    return sites, out


def trace(args):
    """R_eff (GV) for an entry point (lat, lon) and local direction (zenith, azimuth)."""
    from gtracr.geomagnetic_cutoffs import GMRC
    from gtracr.location import Location

    plat, plon, pzen, paz = args
    with contextlib.redirect_stdout(io.StringIO()):
        gmrc = GMRC(
            location=Location("entry", plat, plon, 0.0),
            particle_altitude=PARTICLE_ALTITUDE_KM,
            date=CUTOFF_DATE,
            solver="rk45",
            method="penumbra",
            min_rigidity=R_MIN_GV,
            max_rigidity=R_MAX_GV,
            n_workers=1,
        )
        gmrc.evaluate_batch(zenith=[pzen], azimuth=[paz])
    d = gmrc.data_dict
    if d["flag"][0] == 2:
        raise RuntimeError(f"{args}: cutoff above {R_MAX_GV} GV")
    return float(d["r_eff"][0])


def write_table(sites, rc):
    """Write OUT from the R_eff of all tasks, in task order."""
    import gtracr

    rc = np.round(np.reshape(rc, (len(sites), ZENITH_DEG.size, AZIMUTH_DEG.size)), 3)
    table = {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "date": CUTOFF_DATE,
        "gtracr_version": gtracr.__version__,
        "particle_altitude_km": PARTICLE_ALTITUDE_KM,
        "zenith_deg": ZENITH_DEG.tolist(),
        "azimuth_deg": AZIMUTH_DEG.tolist(),
        "cutoff_GV": {s: rc[i].tolist() for i, s in enumerate(sites)},
    }
    with open(OUT, "w") as f:
        json.dump(table, f, separators=(",", ":"))
        f.write("\n")
    for s, m in table["cutoff_GV"].items():
        print(f"{s:12s} R_eff {np.min(m):6.3f} - {np.max(m):6.2f} GV")


def main(n_proc=None):
    sites, todo = tasks()
    with Pool(n_proc) as pool:
        rc = pool.map(trace, todo, chunksize=4)
    write_table(sites, rc)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
