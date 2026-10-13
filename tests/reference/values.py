"""Run the reference calculations and return what they produced.

One place builds the runs, so the comparison and the regeneration cannot
drift apart. Everything is computed on the CI database, whose energy window
is 891 GeV .. 891 TeV.
"""

from __future__ import annotations

import pathlib

import numpy as np

#: The database these numbers are produced on.
DATABASE = "mceq_ci_base_air_1d_v2.h5"
#: Hadronic model and primary flux.
MODEL = "SIBYLL23E"
#: Zenith angles solved, in degrees.
ZENITHS = (0.0, 60.0)
#: Leptons whose spectra are stored.
LEPTONS = ("mu-", "mu+", "numu", "antinumu", "nue", "antinue")

DATA_DIR = pathlib.Path(__file__).parent / "data"
STORE = DATA_DIR / "reference_solutions.npz"


def _run():
    import crflux.models as pm

    from MCEq.core import MCEqRun

    return MCEqRun(
        interaction_model=MODEL,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=ZENITHS[0],
    )


def compute():
    """The reference quantities, as a name -> array mapping.

    Structure first (grid, dimensions, species), then the operators the
    solver is built from, then the solved lepton spectra at each zenith.
    """
    import MCEq.config as config

    saved = config.mceq_db_fname
    config.mceq_db_fname = DATABASE
    try:
        run = _run()
        out = {
            "e_grid": np.asarray(run.e_grid, dtype=np.float64),
            "dim_states": np.asarray(run.dim_states),
            "pdg_ids": np.asarray(
                sorted({p.pdg_id[0] for p in run.pman.all_particles})
            ),
        }
        for name, matrix in (("int_m", run.int_m), ("dec_m", run.dec_m)):
            out[f"{name}/nnz"] = np.asarray(matrix.nnz)
            out[f"{name}/norm"] = np.asarray(
                np.sqrt(float(np.sum(np.abs(matrix.data) ** 2)))
            )
        for theta in ZENITHS:
            run.set_zenith_azimuth(theta)
            run.solve()
            for lepton in LEPTONS:
                key = f"flux/{theta:g}/{lepton}"
                out[key] = np.asarray(run.get_solution(lepton), dtype=np.float64)
        return out
    finally:
        config.mceq_db_fname = saved


def load():
    """The stored reference values."""
    with np.load(STORE) as handle:
        return {key: handle[key] for key in handle.files}


def save(values):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE, **values)


if __name__ == "__main__":
    save(compute())
    print(f"wrote {STORE}")
