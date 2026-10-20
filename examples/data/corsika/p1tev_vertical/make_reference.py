#!/usr/bin/env python
"""Reduce the aggregated CORSIKA 7 histograms of the p 1 TeV vertical case to c7_reference.npz.

Inputs are the four per-shower histogram files written by the aggregation step
(one pass over all 2500 particle files, 5,000,000 showers):

  c7_lowe_angles.npz           theta histograms, 4 energy bins: p, n, pi+-, mu
  c7_nu_angles.npz             theta histograms, 4 energy bins: nue, numu (and mu)
  c7_hadron_spectra.npz        angle-integrated kinetic-energy spectra of hadrons
  c7_2d_espectra_nob_x100.npz  (E, theta) histograms of mu, numu, nue

Usage:
  python make_reference.py <dir with the four files> [-o c7_reference.npz]

Output layout (all values per primary shower; float32 values and errors,
float64 bin edges):

  theta_edges_deg (46,)        2-degree bins 0..90; the angular arrays carry
                               one extra last column: theta >= 90 deg
  angle_e_bin_edges (4, 2)     kinetic-energy bins [GeV] of the angular histograms
  ang_<sp> (3, 4, 46)          particles per bin, levels 15 km / 5 km / ground
  ang_<sp>_err (3, 4, 46)      Poisson error sqrt(counts) / n_showers
  e_edges (52,)                kinetic-energy edges [GeV] of the spectra
                               (the MCEq grid, 10 bins per decade from 10^-1.1)
  spec_<sp> (3, 51)            particles per energy bin (not divided by dE)
  spec_<sp>_err (3, 51)        Poisson error
  n_showers                    int
  meta                         JSON string (provenance)

<sp> in p, n, pi (pi+ + pi-), mu (mu+ + mu-), numu, nue (nu + anti-nu).
"""

import argparse
import datetime
import json
import pathlib

import numpy as np

LEVELS = [
    {"name": "15 km", "h_cm": 1.5e6, "X_gcm2": 124.14},
    {"name": "5 km", "h_cm": 5.0e5, "X_gcm2": 552.95},
    {"name": "ground", "h_cm": 0.0, "X_gcm2": 1036.10},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agg_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).with_name("c7_reference.npz"))
    args = ap.parse_args()
    d = args.agg_dir
    files = {
        "lowe": "c7_lowe_angles.npz",
        "nu": "c7_nu_angles.npz",
        "had": "c7_hadron_spectra.npz",
        "lep2d": "c7_2d_espectra_nob_x100.npz",
    }
    src = {k: np.load(d / f) for k, f in files.items()}
    nsh = {k: int(v["n_showers"]) for k, v in src.items()}
    assert len(set(nsh.values())) == 1, nsh
    n_showers = nsh["lowe"]

    out = {}
    # angular histograms
    te = src["lowe"]["theta_edges_deg"]
    ebe = src["lowe"]["e_bin_edges"]
    assert np.allclose(te, src["nu"]["theta_edges_deg"])
    assert np.allclose(ebe, src["nu"]["e_bin_edges"])
    out["theta_edges_deg"] = te.astype(np.float64)
    out["angle_e_bin_edges"] = ebe.astype(np.float64)
    for sp, (tag, key) in {
        "p": ("lowe", "p"), "n": ("lowe", "n"), "pi": ("lowe", "pi+-"),
        "mu": ("lowe", "mu"), "numu": ("nu", "numu"), "nue": ("nu", "nue"),
    }.items():
        raw = src[tag][key + "_raw"]
        assert np.allclose(src[tag][key], raw / n_showers)
        out[f"ang_{sp}"] = (raw / n_showers).astype(np.float32)
        out[f"ang_{sp}_err"] = (np.sqrt(raw) / n_showers).astype(np.float32)

    # angle-integrated spectra
    e = src["had"]["e_edges"]
    assert np.allclose(e, src["lep2d"]["e_edges"])
    out["e_edges"] = e.astype(np.float64)
    h = src["had"]
    for sp, arrs in {"p": ["p"], "n": ["n"], "pi": ["pip", "pim"]}.items():
        cnt = sum(h[a + "_cnt"] for a in arrs)
        out[f"spec_{sp}"] = (cnt / n_showers).astype(np.float32)
        out[f"spec_{sp}_err"] = (np.sqrt(cnt) / n_showers).astype(np.float32)
    le = src["lep2d"]
    for sp in ("mu", "numu", "nue"):
        out[f"spec_{sp}"] = le[sp].sum(axis=2).astype(np.float32)
        out[f"spec_{sp}_err"] = np.sqrt((le[sp + "_err"] ** 2).sum(axis=2)).astype(np.float32)

    out["n_showers"] = np.int64(n_showers)
    meta = {
        "case": "p1tev_vertical",
        "description": "proton 1 TeV, vertical, pure FLUKA (HILOW 1500), EM off, unthinned, B ~ 0",
        "corsika": "CORSIKA 7.8050 (patched: beta-exact PRANGE, isospin-corrected K*+- decays), "
                   "SIBYLL 2.3e high-energy model (unused), FLUKA 2025.1 low-energy model",
        "n_showers": n_showers,
        "shards": "2500 x 2000",
        "levels": LEVELS,
        "species": {"p": "p", "n": "n", "pi": "pi+ + pi-", "mu": "mu+ + mu-",
                    "numu": "nu_mu + anti-nu_mu", "nue": "nu_e + anti-nu_e"},
        "energy": "kinetic energy [GeV], E_kin = sqrt(p^2 + m^2) - m",
        "angle": "theta = angle between particle momentum and the primary axis (EVTH), degrees",
        "angular_binning": "2-degree bins 0..90 plus a theta >= 90 column; 4 energy bins "
                           "[0.1,0.1259), [0.3162,0.3981), [0.7943,1.0), [1.995,2.512) GeV",
        "spectra_binning": "logspace(-1.1, 4, 52) GeV; hadrons: all angles; mu/numu/nue: "
                           "theta < 90 deg (summed over 1-degree bins)",
        "normalization": "particles per bin per primary shower (no division by bin width)",
        "errors": "Poisson, sqrt(counts) / n_showers",
        "date": datetime.date.today().isoformat(),
        "sources": sorted(files.values()),
    }
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB), {n_showers} showers")


if __name__ == "__main__":
    main()
