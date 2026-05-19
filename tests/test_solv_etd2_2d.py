"""ETD2 on a stitched 2D operator must match running ETD2 on each
per-mode operator separately, to round-off."""

import os

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun
from MCEq.solvers import solv_numpy_etd2


@pytest.fixture(scope="module")
def mceq_2d():
    fn = "mceq_db_URQMD_150GeV_2D.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; symlink it into src/MCEq/data/")
    config.mceq_db_fname = fn
    config.e_min = 1e-1
    config.e_max = 1e4
    config.muon_helicity_dependence = True
    config.muon_multiple_scattering = False  # keep this test pure to the splitting
    return MCEqRun(
        interaction_model="SIBYLL23D",
        primary_model=None,
        theta_deg=60.0,
        density_model=("CORSIKA", ("USStd", None)),
    )


def test_stitched_etd2_matches_per_mode_loop(mceq_2d):
    """Stitched ETD2 on (n_k*N, n_k*N) is equivalent to n_k separate ETD2 on (N, N)."""
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    int_m = mceq_2d.int_m.tocsr()
    dec_m = mceq_2d.dec_m.tocsr()

    # Random but deterministic initial state
    rng = np.random.default_rng(42)
    phi0 = rng.standard_normal(N)
    phi0_stacked = np.tile(phi0, n_k)

    # Use a short, well-defined integration path
    nsteps = 5
    dX = np.full(nsteps, 5.0)
    rho_inv = np.full(nsteps, 1e-3)
    grid_idcs = []

    # Stitched run
    out_stitched, _ = solv_numpy_etd2(
        nsteps, dX, rho_inv, int_m, dec_m, phi0_stacked.copy(), grid_idcs
    )

    # Per-mode reference
    out_per_mode = np.empty(n_k * N)
    for k in range(n_k):
        int_m_k = int_m[k * N : (k + 1) * N, k * N : (k + 1) * N].tocsr()
        dec_m_k = dec_m[k * N : (k + 1) * N, k * N : (k + 1) * N].tocsr()
        out_k, _ = solv_numpy_etd2(
            nsteps, dX, rho_inv, int_m_k, dec_m_k, phi0.copy(), grid_idcs
        )
        out_per_mode[k * N : (k + 1) * N] = out_k

    np.testing.assert_allclose(out_stitched, out_per_mode, rtol=1e-10, atol=1e-12)


def test_full_solve_2d_runs_end_to_end(mceq_2d):
    """A real MCEqRun.solve() call on a 2D database completes and produces
    a length-(n_k * dim_states) solution."""
    from crflux.models import HillasGaisser2012

    mceq_2d.set_primary_model(HillasGaisser2012, "H3a")
    save_depths = np.array([200.0, 1000.0])
    mceq_2d.solve(int_grid=save_depths)
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    assert mceq_2d._solution.shape == (n_k * N,)
    assert len(mceq_2d.grid_sol) == len(save_depths)
    for snap in mceq_2d.grid_sol:
        assert snap.shape == (n_k * N,)
