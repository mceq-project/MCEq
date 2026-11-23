"""Multiple scattering damping is folded into the diagonal of int_m."""

import os

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun


def _saved_config():
    return {
        "mceq_db_fname": config.mceq_db_fname,
        "e_min": config.e_min,
        "e_max": config.e_max,
        "muon_helicity_dependence": config.muon_helicity_dependence,
        "muon_multiple_scattering": config.muon_multiple_scattering,
    }


def _restore(saved):
    for key, value in saved.items():
        setattr(config, key, value)


@pytest.fixture(scope="module")
def mceq_2d_with_scattering():
    fn = "mceq_ci_base_air_2d_legendre_v2.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; python -m MCEq.data.download --file {fn}")
    saved = _saved_config()
    config.mceq_db_fname = fn
    config.e_min = 1e-1
    config.e_max = 1e2
    config.muon_helicity_dependence = True
    config.muon_multiple_scattering = True
    try:
        yield MCEqRun(
            interaction_model="FLUKA20251",
            primary_model=None,
            theta_deg=0.0,
            density_model=("CORSIKA", ("USStd", None)),
        )
    finally:
        _restore(saved)


@pytest.fixture(scope="module")
def mceq_2d_no_scattering():
    fn = "mceq_ci_base_air_2d_legendre_v2.h5"
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; python -m MCEq.data.download --file {fn}")
    saved = _saved_config()
    config.mceq_db_fname = fn
    config.e_min = 1e-1
    config.e_max = 1e2
    config.muon_helicity_dependence = True
    config.muon_multiple_scattering = False
    try:
        yield MCEqRun(
            interaction_model="FLUKA20251",
            primary_model=None,
            theta_deg=0.0,
            density_model=("CORSIKA", ("USStd", None)),
        )
    finally:
        _restore(saved)


def _muon_lidcs(mceq):
    """Return the set of (lidx, dim) tuples for all muon species in the system."""
    pman = mceq.pman
    out = []
    for pdg in (13, -13):
        for hel in (0, 1, -1):
            key = (pdg, hel)
            if key in pman.pdg2pref:
                p = pman.pdg2pref[key]
                if hasattr(p, "lidx") and getattr(p, "mceqidx", -1) >= 0:
                    out.append((key, p.lidx, len(mceq.e_grid)))
    return out


def test_l_zero_block_unchanged(mceq_2d_with_scattering, mceq_2d_no_scattering):
    """The l = 0 block has a zero rate, so multiple scattering adds nothing.
    Diagonal entries on muon rows must be identical with and without scattering."""
    N = mceq_2d_with_scattering.dim_states
    M_with = mceq_2d_with_scattering.int_m.tocsr()
    M_no = mceq_2d_no_scattering.int_m.tocsr()
    diag_with = M_with.diagonal()[:N]
    diag_no = M_no.diagonal()[:N]
    np.testing.assert_allclose(diag_with, diag_no, atol=1e-30)


def test_non_muon_rows_unchanged(mceq_2d_with_scattering, mceq_2d_no_scattering):
    """Non-muon rows must be identical across all mode blocks."""
    n_k = mceq_2d_with_scattering._mceq_db.n_k
    N = mceq_2d_with_scattering.dim_states
    diag_with = mceq_2d_with_scattering.int_m.tocsr().diagonal()
    diag_no = mceq_2d_no_scattering.int_m.tocsr().diagonal()
    muon_lidcs = set()
    for _pdg_hel, lidx, dim in _muon_lidcs(mceq_2d_with_scattering):
        muon_lidcs.update(range(lidx, lidx + dim))
    for k in range(n_k):
        off = k * N
        for i in range(N):
            if i in muon_lidcs:
                continue
            assert diag_with[off + i] == diag_no[off + i], (
                f"non-muon row k={k} idx={i} differs"
            )


def test_off_diagonal_muon_rows_unchanged(
    mceq_2d_with_scattering, mceq_2d_no_scattering
):
    """Multiple scattering only modifies the diagonal; muon-row off-diagonals
    must be bit-identical between the two configurations."""
    M_with = mceq_2d_with_scattering.int_m.tocsr()
    M_no = mceq_2d_no_scattering.int_m.tocsr()
    n_k = mceq_2d_with_scattering._mceq_db.n_k
    N = mceq_2d_with_scattering.dim_states
    diff = (M_with - M_no).tocsr()
    # Strip the diagonal: any remaining nonzero entry is an off-diagonal change.
    diff_no_diag = diff - sparse_diag_from(diff.diagonal(), diff.shape)
    diff_no_diag.eliminate_zeros()
    assert diff_no_diag.nnz == 0, (
        f"off-diagonal entries changed by scattering: nnz={diff_no_diag.nnz}, "
        f"max|delta|={np.abs(diff_no_diag.data).max() if diff_no_diag.nnz else 0}"
    )
    # Sanity: for n_k > 1 the diagonals MUST differ (we just proved damping is real).
    if n_k > 1:
        assert np.abs(diff.diagonal()).max() > 0, (
            "expected nonzero diagonal difference, got zero"
        )
    # And the difference is confined to l >= 1 blocks (l = 0 is unchanged).
    diag_diff = diff.diagonal()
    assert np.allclose(diag_diff[:N], 0.0, atol=1e-30), (
        "l = 0 diagonal block changed unexpectedly"
    )


def sparse_diag_from(values, shape):
    from scipy.sparse import diags

    return diags(values, 0, shape=shape, format="csr")


def test_muon_rows_carry_the_database_rate(
    mceq_2d_with_scattering, mceq_2d_no_scattering
):
    """The extra muon diagonal is ``-Gamma`` from the database's rate table."""
    db = mceq_2d_with_scattering._mceq_db
    rate = db.scattering_rate_table()[0]
    N = mceq_2d_with_scattering.dim_states
    extra = (
        mceq_2d_with_scattering.int_m.tocsr().diagonal()
        - mceq_2d_no_scattering.int_m.tocsr().diagonal()
    ).reshape(db.n_k, N)
    found = _muon_lidcs(mceq_2d_with_scattering)
    assert found, "no muon species found"
    for _pdg_hel, lidx, dim in found:
        np.testing.assert_allclose(
            extra[:, lidx : lidx + dim], -rate, rtol=1e-6, atol=1e-30
        )
