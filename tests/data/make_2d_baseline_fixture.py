"""Regenerate ``tests/data/2d_baseline_solution.npz`` — the 2D regression
fixture used by ``tests/test_2d_baseline.py`` and
``tests/test_2d_baseline_validation.py``.

The fixture is a snapshot of the production 2D configuration: FLUKA 2D
database (48 Hankel modes), sec(theta) transport at the default cap, muon
multiple scattering and helicity-dependent decays on. A single 100 GeV
proton at theta=30 deg is solved with the numpy ETD2 kernel (pinned for
cross-machine determinism) on a tight step schedule, and the Hankel-space
state history plus the theta-space readouts are stored together with the
full provenance needed to reproduce the run.

Usage (from the repo root)::

    .venv/bin/python tests/data/make_2d_baseline_fixture.py
"""

import pathlib
import subprocess

import numpy as np

from MCEq import config

DB_FNAME = "mceq_db_v2_fluka2d_rc7.h5"
INTERACTION_MODEL = "FLUKA20251"
THETA_DEG = 30.0
PRIMARY_ENERGY_GEV = 100.0
PRIMARY_PDG = 2212
SAVE_ALTITUDES_KM = (15.0, 5.0, 0.2, 0.0)
E_MIN, E_MAX = 1e-1, 1e4
EPS, DX_MAX = 0.05, 2.0
OVERSAMPLE_RES, THETA_RES = 5, 600

OUTFILE = pathlib.Path(__file__).parent / "2d_baseline_solution.npz"


def main():
    config.mceq_db_fname = DB_FNAME
    config.e_min = E_MIN
    config.e_max = E_MAX
    config.kernel_config = "numpy_etd2"

    from MCEq.core import MCEqRun

    mceq = MCEqRun(
        interaction_model=INTERACTION_MODEL,
        primary_model=None,
        theta_deg=THETA_DEG,
        density_model=("CORSIKA", ("USStd", None)),
    )
    mceq.set_single_primary_particle(E=PRIMARY_ENERGY_GEV, pdg_id=PRIMARY_PDG)

    save_depths = np.array([mceq.density_model.h2X(h * 1e5) for h in SAVE_ALTITUDES_KM])
    mceq.solve(int_grid=save_depths, eps=EPS, dX_max=DX_MAX)

    n_k = mceq._mceq_db.n_k
    N = mceq.dim_states
    phi_hankel = np.array([snap.reshape(n_k, N) for snap in mceq.grid_sol])
    hankel_history = [snap.reshape(n_k, N) for snap in mceq.grid_sol]

    out = {
        "phi_hankel": phi_hankel,
        "save_depths": save_depths,
        "k_grid": np.asarray(mceq._mceq_db.k_grid),
        "e_grid": np.asarray(mceq.e_grid),
        "db_fname": DB_FNAME,
        "interaction_model": INTERACTION_MODEL,
        "theta_deg": THETA_DEG,
        "primary_energy_gev": PRIMARY_ENERGY_GEV,
        "primary_pdg": PRIMARY_PDG,
        "eps": EPS,
        "dX_max": DX_MAX,
        "kernel_config": "numpy_etd2",
        "secant_theta_transport": str(config.secant_theta_transport),
        "secant_theta_cap_deg": float(config.secant_theta_cap_deg),
        "muon_multiple_scattering": bool(config.muon_multiple_scattering),
        "muon_helicity_dependence": bool(config.muon_helicity_dependence),
        "mceq_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parents[2],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }

    theta_grid = None
    for pdg in (14, 12, 13):
        hels = sorted(
            h
            for (p, h), midx in mceq.pman.pdg2mceqidx.items()
            if p == pdg and midx >= 0
        )
        summed = None
        for hel in hels:
            res = mceq.convert_to_theta_space(
                hankel_history,
                pdg_id=pdg,
                hel=hel,
                oversample_res=OVERSAMPLE_RES,
                theta_res=THETA_RES,
            )
            theta_grid = np.asarray(res[2])
            f_theta = np.asarray(res[3])
            tag = f"m{-hel}" if hel < 0 else (f"p{hel}" if hel > 0 else "0")
            out[f"f_theta_{pdg}_{tag}"] = f_theta
            summed = f_theta if summed is None else summed + f_theta
        out[f"f_theta_pdg{pdg}_summed"] = summed
    out["theta_grid"] = theta_grid
    # canonical entry: (numu, hel=0)
    out["f_theta"] = out["f_theta_14_0"]

    np.savez_compressed(OUTFILE, **out)
    size_mb = OUTFILE.stat().st_size / 1e6
    print(f"wrote {OUTFILE} ({size_mb:.1f} MB)")
    print(f"phi_hankel shape: {phi_hankel.shape}, theta_grid: {theta_grid.shape}")


if __name__ == "__main__":
    main()
