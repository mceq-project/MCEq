"""Tabulate the geomagnetic rigidity cutoffs of the named sites.

Writes ``src/MCEq/geometry/cutoff_maps.json``: for every site of
:func:`MCEq.geometry.atmosphere_parameters.site_coordinates`, the cutoff in GV
of downgoing protons on a 5° grid of zenith (0–90°) and azimuth (0–355°, 0° is
North, 90° East). A direction's cutoff is the lowest rigidity, scanned upward
in 0.1 GV steps, whose gtracr trajectory escapes the IGRF field at
:data:`MCEq.geometry.cutoff.CUTOFF_DATE`, started 100 km above the site.

Usage: python scripts/make_cutoff_maps.py [n_processes]
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from MCEq.geometry.atmosphere_parameters import (
    KM3NET_DETECTORS,
    LOCATIONS,
    site_coordinates,
)
from MCEq.geometry.cutoff import CUTOFF_DATE

PARTICLE_ALTITUDE_KM = 100.0
ZENITH_DEG = np.arange(0.0, 90.1, 5.0)
AZIMUTH_DEG = np.arange(0.0, 360.0, 5.0)
RIGIDITY_GV = np.round(np.arange(0.1, 60.0, 0.1), 1)
OUT = Path(__file__).parents[1] / "src/MCEq/geometry/cutoff_maps.json"


def _altitude_km(site):
    return LOCATIONS[site][2] / 1e5 if site in LOCATIONS else 0.0


def cutoff(args):
    """Lowest escaping rigidity of one direction at one site."""
    from gtracr.trajectory import Trajectory

    site, zenith, azimuth = args
    _, lon, lat = site_coordinates(site)
    for rigidity in RIGIDITY_GV:
        traj = Trajectory(
            zenith,
            azimuth,
            rigidity=rigidity,
            particle_altitude=PARTICLE_ALTITUDE_KM,
            latitude=lat,
            longitude=lon,
            detector_altitude=_altitude_km(site),
            date=CUTOFF_DATE,
            solver="rk45",
        )
        traj.get_trajectory(dt=1e-5, max_time=1.0)
        if traj.particle_escaped:
            return float(rigidity)
    raise RuntimeError(f"{site} ({zenith}, {azimuth}): no escape below 60 GV")


def main(n_proc=None):
    import gtracr

    sites = list(LOCATIONS) + list(KM3NET_DETECTORS)
    tasks = [(s, z, a) for s in sites for z in ZENITH_DEG for a in AZIMUTH_DEG]
    with Pool(n_proc) as pool:
        rc = np.array(pool.map(cutoff, tasks, chunksize=8))
    rc = rc.reshape(len(sites), ZENITH_DEG.size, AZIMUTH_DEG.size)
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
        print(f"{s:12s} R_c {np.min(m):5.1f} - {np.max(m):5.1f} GV")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
