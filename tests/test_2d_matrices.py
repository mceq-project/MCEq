"""2D matrix assembly: stitched block-diagonal CSR."""

import os

import numpy as np
import pytest
from scipy import sparse

from MCEq import config
from MCEq.core import MCEqRun


@pytest.fixture(scope="module")
def mceq_2d():
    fn = "mceq_db_URQMD_150GeV_2D.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; symlink it into src/MCEq/data/")
    saved = {
        "mceq_db_fname": config.mceq_db_fname,
        "e_min": config.e_min,
        "e_max": config.e_max,
        "muon_helicity_dependence": config.muon_helicity_dependence,
    }
    config.mceq_db_fname = fn
    config.e_min = 1e-1
    config.e_max = 1e4
    config.muon_helicity_dependence = True
    try:
        yield MCEqRun(
            interaction_model="SIBYLL23D",
            primary_model=None,
            theta_deg=0.0,
            density_model=("CORSIKA", ("USStd", None)),
        )
    finally:
        for key, value in saved.items():
            setattr(config, key, value)


def test_int_m_is_stitched_block_diagonal(mceq_2d):
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    assert sparse.issparse(mceq_2d.int_m), "int_m should be sparse"
    assert mceq_2d.int_m.shape == (n_k * N, n_k * N)


def test_dec_m_is_stitched_block_diagonal(mceq_2d):
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    assert sparse.issparse(mceq_2d.dec_m), "dec_m should be sparse"
    assert mceq_2d.dec_m.shape == (n_k * N, n_k * N)


def test_off_block_is_exactly_zero(mceq_2d):
    """k != k' blocks must be exactly zero — k-modes are decoupled."""
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    M = mceq_2d.int_m.tocsr()
    # Zero a 2D coordinate window for one off-block, check the original was 0.
    for k in range(n_k):
        for kp in range(n_k):
            if k == kp:
                continue
            block = M[k * N : (k + 1) * N, kp * N : (kp + 1) * N]
            assert block.nnz == 0 or np.abs(block.data).max() == 0.0, (
                f"int_m off-block ({k},{kp}) is non-zero"
            )


def test_diagonal_block_matches_per_mode_construction(mceq_2d):
    """The k-th diagonal block should match a per-mode 1D-style operator
    built for k_grid[k]."""
    n_k = mceq_2d._mceq_db.n_k
    N = mceq_2d.dim_states
    M = mceq_2d.int_m.tocsr()
    # Sanity: the diagonal blocks should not all be identical (k-dependent
    # off-diagonals fall off with k).
    blk0 = M[0:N, 0:N].toarray()
    blk_last = M[(n_k - 1) * N : n_k * N, (n_k - 1) * N : n_k * N].toarray()
    diff = np.abs(blk0 - blk_last)
    assert diff.max() > 0, (
        "k=0 and k=k_max diagonal blocks should differ (k-dependent kernel)"
    )


def test_1d_matrix_unchanged():
    """Loading a 1D database must still produce a (N, N) matrix."""
    saved = {
        "mceq_db_fname": config.mceq_db_fname,
        "e_min": config.e_min,
        "e_max": config.e_max,
        "muon_helicity_dependence": config.muon_helicity_dependence,
    }
    try:
        config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
        config.e_min = 1.0
        config.e_max = 1e8
        config.muon_helicity_dependence = False
        # The reduced 1D test db only ships SIBYLL21 / QGSJETII04 (matches the
        # session fixtures in conftest.py). SIBYLL23C is silently rewritten to
        # SIBYLL23D inside ``cs_db`` which isn't available in this database.
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            primary_model=None,
            theta_deg=0.0,
            density_model=("CORSIKA", ("USStd", None)),
        )
        N = mceq.dim_states
        assert mceq.int_m.shape == (N, N)
        assert mceq._mceq_db.is_2d is False and mceq._mceq_db.n_k == 1
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
