"""Validate the 2D solver against the FLUKA regression fixture at
``tests/data/2d_baseline_solution.npz``.

The fixture (see ``tests/data/make_2d_baseline_fixture.py``) is a snapshot
of the production 2D configuration: FLUKA 2D database (48 Hankel modes),
sec(theta) transport at the default cap, muon multiple scattering and
helicity-dependent decays, a single 100 GeV proton at theta=30 deg, solved
with the numpy ETD2 kernel on a tight step schedule (eps=0.05, dX_max=2 —
the production defaults are tuned for cosmic-ray spectra and are too coarse
for a single-energy primary).

This module re-runs the identical configuration and asserts agreement with
the stored solution. Because fixture and test share the code path, kernel
and database, the tolerances are tight regression bounds (they absorb only
BLAS/LAPACK build differences across machines), not physics bounds: any
intentional change to the 2D transport requires regenerating the fixture
and documenting why.
"""

import os
import pathlib

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun

FIXTURE = pathlib.Path(__file__).parent / "data" / "2d_baseline_solution.npz"

# Per-mode rel-L2 regression bounds. The re-run reproduces the fixture
# solve exactly up to floating-point differences in the BLAS/LAPACK builds
# (dense GEMMs in the secant coupling, eig of S_P); 1e-6 leaves orders of
# magnitude of headroom above those while catching any real change.
REL_L2_MAX = 1e-6

_CONFIG_KEYS = (
    "mceq_db_fname",
    "e_min",
    "e_max",
    "kernel_config",
    "muon_helicity_dependence",
    "muon_multiple_scattering",
    "secant_theta_transport",
    "secant_theta_cap_deg",
)


@pytest.fixture(scope="module")
def baseline():
    if not FIXTURE.exists():
        pytest.skip(
            "baseline fixture missing — run tests/data/make_2d_baseline_fixture.py"
        )
    return np.load(FIXTURE, allow_pickle=True)


@pytest.fixture(scope="module")
def mceq_2d(baseline):
    fn = str(baseline["db_fname"])
    if not os.path.exists(os.path.join(config.data_dir, fn)):
        pytest.skip(f"{fn} not available; symlink it into src/MCEq/data/")

    saved = {k: getattr(config, k) for k in _CONFIG_KEYS}
    try:
        # Mirror the fixture-generation configuration exactly.
        config.mceq_db_fname = fn
        config.e_min = 1e-1
        config.e_max = 1e4
        config.kernel_config = str(baseline["kernel_config"])
        config.muon_helicity_dependence = bool(baseline["muon_helicity_dependence"])
        config.muon_multiple_scattering = bool(baseline["muon_multiple_scattering"])
        config.secant_theta_transport = str(baseline["secant_theta_transport"])
        config.secant_theta_cap_deg = float(baseline["secant_theta_cap_deg"])

        mceq = MCEqRun(
            interaction_model=str(baseline["interaction_model"]),
            primary_model=None,
            theta_deg=float(baseline["theta_deg"]),
            density_model=("CORSIKA", ("USStd", None)),
        )
        mceq.set_single_primary_particle(
            E=float(baseline["primary_energy_gev"]),
            pdg_id=int(baseline["primary_pdg"]),
        )
        mceq.solve(
            int_grid=baseline["save_depths"],
            eps=float(baseline["eps"]),
            dX_max=float(baseline["dX_max"]),
        )
        yield mceq
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def test_2d_dim_states_match_baseline(mceq_2d, baseline):
    """Same code + same database must produce the same state-vector layout."""
    assert mceq_2d._mceq_db.n_k == baseline["phi_hankel"].shape[1]
    assert mceq_2d.dim_states == baseline["phi_hankel"].shape[2]


def test_2d_matches_baseline_phi_hankel(mceq_2d, baseline):
    """Per-mode Hankel-space state agrees with the fixture at every stored
    depth within the regression bound."""
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states

    for j, snap in enumerate(mceq_2d.grid_sol):
        snap_now = snap.reshape(n_k, N)
        snap_ref = baseline["phi_hankel"][j]
        for k in range(n_k):
            denom = np.linalg.norm(snap_ref[k])
            if denom < 1e-30:
                continue
            rel_l2 = np.linalg.norm(snap_now[k] - snap_ref[k]) / denom
            assert rel_l2 < REL_L2_MAX, f"depth #{j}, k-mode {k}: rel-L2 = {rel_l2:.3e}"


def test_2d_matches_baseline_numu_theta_space(mceq_2d, baseline):
    """Helicity-summed nu_mu angular density from the inverse Hankel
    readout agrees with the fixture (same quadrature on both sides)."""
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    hankel_history = [snap.reshape(n_k, N) for snap in mceq_2d.grid_sol]

    res = mceq_2d.convert_to_theta_space(
        hankel_history,
        pdg_id=14,
        hel=0,
        oversample_res=5,
        theta_res=600,
    )
    f_theta_now = np.asarray(res[3])
    f_theta_ref = baseline["f_theta_14_0"]
    assert f_theta_now.shape == f_theta_ref.shape

    for j in range(f_theta_ref.shape[0]):
        denom = np.linalg.norm(f_theta_ref[j])
        if denom < 1e-30:
            continue
        rel_l2 = np.linalg.norm(f_theta_now[j] - f_theta_ref[j]) / denom
        assert rel_l2 < REL_L2_MAX, f"depth #{j}: rel-L2 = {rel_l2:.3e}"
