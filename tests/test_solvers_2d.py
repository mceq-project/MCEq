"""Cross-backend equivalence on the 2D stitched matrix.

Asserts that every ETD2RK backend the host provides produces the same
solution on the stitched (n_k * dim_states) operator built from a 2D
database. v2's existing ``test_solve_etd2_accelerate_matches_numpy_etd2_real``
covers the 1D path; the 2D matrix is just a bigger CSR — the driver is
dimension-agnostic, so equality holds to round-off.

Each backend gates on the capability probe the rest of the suite uses:
``config.has_accelerate`` (macOS), ``config.has_mkl`` (``libmkl_rt`` under the
prefix) and ``config.has_cuda`` (an importable cupy *and* a visible device).
Every test needs the 329 MB FLUKA 2D database and skips without it.
"""

import os

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun

#: Keep this module's tests on one ``--dist loadgroup`` worker; the database
#: costs 329 MB per process. Five other modules open the same file and are not
#: in the group, so this bounds one module rather than the whole suite.
pytestmark = pytest.mark.xdist_group("fluka2d")

#: Globals :func:`_solve` sets that ``conftest._restore_global_config_state``
#: does not snapshot. This module runs against a different database from the
#: rest of the suite, so leaking its grid bounds would change later solves.
_UNSNAPSHOTTED = (
    "e_min",
    "e_max",
    "muon_helicity_dependence",
    "muon_multiple_scattering",
    "secant_theta_transport",
)


@pytest.fixture(scope="module", autouse=True)
def _restore_2d_config():
    """Keep this module's grid bounds and muon/secant switches from leaking
    into later tests, which run against a different database."""
    saved = {name: getattr(config, name) for name in _UNSNAPSHOTTED}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


@pytest.fixture(scope="module")
def base_config():
    """Common 2D-database config used by every backend in this module."""
    return dict(
        mceq_db_fname="mceq_db_v2_fluka2d_rc7.h5",
        e_min=1e-1,
        e_max=1e4,
        muon_helicity_dependence=True,
        muon_multiple_scattering=False,
        theta_deg=60.0,
        interaction_model="FLUKA20251",
        density_model=("CORSIKA", ("USStd", None)),
    )


def _solve(kernel, base):
    """Build an MCEqRun on the 2D DB, set ``kernel`` and solve over a fixed
    grid using a deterministic non-trivial initial state."""
    fn = base["mceq_db_fname"]
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "src", "MCEq", "data", fn)
    ):
        pytest.skip(f"{fn} not available; symlink it into src/MCEq/data/")
    config.mceq_db_fname = fn
    config.e_min = base["e_min"]
    config.e_max = base["e_max"]
    config.muon_helicity_dependence = base["muon_helicity_dependence"]
    config.muon_multiple_scattering = base["muon_multiple_scattering"]
    config.kernel_config = kernel
    # Paraxial cross-backend comparison; the secant coupling is
    # exercised by its own runs.
    config.secant_theta_transport = False
    mceq = MCEqRun(
        interaction_model=base["interaction_model"],
        primary_model=None,
        theta_deg=base["theta_deg"],
        density_model=base["density_model"],
    )
    # Deterministic non-trivial initial state on the 1D-shape ``_phi0``;
    # ``solve()`` tiles it across ``n_k`` modes when a 2D DB is in use.
    N = mceq.dim_states
    rng = np.random.default_rng(0)
    mceq._phi0 = rng.standard_normal(N)
    mceq.solve(int_grid=np.array([200.0]))
    return mceq._solution


@pytest.mark.skipif(not config.has_accelerate, reason="Accelerate only on macOS")
def test_2d_accelerate_matches_numpy(base_config):
    """Accelerate ETD2 on the 2D stitched matrix matches numpy ETD2 to round-off."""
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_acc = _solve("accelerate_etd2", base_config)
    assert sol_numpy.shape == sol_acc.shape
    np.testing.assert_allclose(sol_numpy, sol_acc, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not config.has_mkl, reason="MKL not available")
def test_2d_mkl_matches_numpy(base_config):
    """MKL ETD2 on the 2D stitched matrix matches numpy ETD2 to round-off.

    Measured bit-exact (134 steps, |state| ~ 2e11, max|delta| 0 at 1, 4 and 16
    threads): ``MklApplyOff`` parallelises over rows and sums each row
    serially, as ``ScipyApplyOff`` does, so the thread count does not reorder
    the accumulation. The tolerance is margin for another MKL build rather
    than a measured spread, and ``atol`` is what governs the 17 % of elements
    below 1e-2. Both bounds stay far tighter than the D18 per-species bound,
    which at 2e-7 would make this test vacuous.
    """
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_mkl = _solve("mkl_etd2", base_config)
    assert sol_numpy.shape == sol_mkl.shape
    assert np.isfinite(sol_mkl).all()
    np.testing.assert_allclose(sol_numpy, sol_mkl, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not config.has_cuda, reason="CuPy not available")
def test_2d_cuda_matches_numpy(base_config):
    """CUDA ETD2 on the 2D stitched matrix matches numpy ETD2 to round-off."""
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_cuda = _solve("cuda_etd2", base_config)
    assert np.isfinite(sol_cuda).all()
    # cuSPARSE may reorder partial sums vs scipy CSR; widen tolerance.
    np.testing.assert_allclose(sol_numpy, sol_cuda, rtol=1e-9, atol=1e-10)
