#!/usr/bin/env python
"""Reduce the aggregated CORSIKA 7 spectra of the p 1 PeV, 60 deg case to c7_reference.npz.

Input is the aggregated histogram file of the re-scoring pass (one pass over
the 2500 particle files, 5000 showers), either

  c7_spectra_by_level_perpstrip.npz   all strip selections, keys {sp}_L{l}_{sel}{,_w2,_sem}
  c7_spectra_by_level.npz             one selection already extracted, keys {sp}_L{l}{,_w2,_sem}

Usage:
  python make_reference.py <aggregated npz> [--sel w30] [-o c7_reference.npz]

Output layout (all values per primary shower; float32 values and errors,
float64 bin edges):

  e_edges (122,)          kinetic-energy bin edges [GeV] (the MCEq v2 grid,
                          10 bins per decade from 10^-1.1 to 10^11)
  slant_gcm2 (7,)         slant depth of the levels along the axis [g/cm2]
  alt_cm (7,)             OBSLEV altitude [cm]
  spec_<sp> (7, 121)      particles per energy bin and shower (not divided by dE)
  spec_<sp>_err (7, 121)  shower-to-shower standard error of spec_<sp>
  int_gt1_<sp>_err (7,)   shower-to-shower standard error of the E_kin > 1 GeV
                          integral of spec_<sp>
  n_showers               int
  meta                    JSON string (provenance)

<sp> in neutron, proton, pi_charged, K_charged, muon, numu, nue
(particle + antiparticle, e.g. muon = mu+ + mu-, numu = numu + anti-numu).
"""

import argparse
import datetime
import json
import pathlib

import numpy as np

SPECIES = ["neutron", "proton", "pi_charged", "K_charged", "muon", "numu", "nue"]
ALT_CM = [2.0830e6, 1.4960e6, 1.1950e6, 8.8100e5, 7.3600e5, 5.7500e5, 3.0000e4]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("--sel", default="w30", help="strip selection (perpstrip file only)")
    ap.add_argument(
        "-o", "--out", type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("c7_reference.npz"),
    )
    args = ap.parse_args()

    src = np.load(args.source)
    suffix = f"_{args.sel}" if f"neutron_L1_{args.sel}" in src.files else ""
    n_showers = int(src["nshow"])
    levels = [int(level) for level in src["levels"]]
    edges = np.asarray(src["edges"], dtype=np.float64)

    out = {
        "e_edges": edges,
        "slant_gcm2": np.asarray(src["slant"], dtype=np.float64),
        "alt_cm": np.asarray(ALT_CM, dtype=np.float64),
        "n_showers": np.int64(n_showers),
    }
    for sp in SPECIES:
        counts = np.array([src[f"{sp}_L{lv}{suffix}"] for lv in levels]) / n_showers
        sem = np.array([src[f"{sp}_L{lv}{suffix}_sem"] for lv in levels])
        out[f"spec_{sp}"] = counts.astype(np.float32)
        out[f"spec_{sp}_err"] = sem.astype(np.float32)
        out[f"int_gt1_{sp}_err"] = np.asarray(
            src[f"{sp}_integral_gt1_sem{suffix}"], dtype=np.float32
        )

    meta = {
        "case": "p1pev_60deg",
        "corsika_version": "7.8050 (patched: beta-exact PRANGE, isospin-corrected K*+- decays)",
        "binary": "corsika78050Linux_SIBYLL_fluka_thin (suffix _fix2k)",
        "models": "SIBYLL 2.3e above 80 GeV, FLUKA 2025.1 below; EM off (ELMFLG F F)",
        "primary": "proton, 1 PeV, zenith 60 deg, azimuth uniform",
        "n_showers": n_showers,
        "levels_slant_gcm2": [float(x) for x in out["slant_gcm2"]],
        "levels_alt_cm": ALT_CM,
        "selection": args.sel if suffix else "pre-selected (w30 in the published figures)",
        "scoring": (
            "particles on the horizontal OBSLEV planes re-scored onto the plane "
            "perpendicular to the shower axis through the core: keep particles within "
            "+-30 g/cm2 axial grammage of that plane, weight by the inverse azimuthal "
            "acceptance of the strip on the particle's ring about the axis; all angles"
        ),
        "binning": "kinetic energy, MCEq v2 grid edges (10 per decade); energy = sqrt(p^2+m^2)-m",
        "units": "particles per energy bin per primary shower (not divided by bin width)",
        "normalization": "sum of thinning weights / n_showers",
        "errors": "shower-to-shower standard error of the mean (per bin and for the E>1 GeV integral)",
        "date": datetime.date.today().isoformat(),
        "source_files": [args.source.name],
    }
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB), {n_showers} showers")


if __name__ == "__main__":
    main()
