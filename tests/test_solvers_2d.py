"""Cross-backend equivalence on the 2D stitched matrix.

Asserts that all available ETD2RK backends produce identical solutions on
the stitched (n_k * dim_states) operator built from a 2D database. v2's
existing ``test_solv_spacc_etd2_matches_numpy_etd2_real`` covers the 1D
path; the 2D matrix is just a bigger CSR — the kernels are
dimension-agnostic, so equality should hold to round-off.

On macOS only the numpy ↔ Accelerate equivalence runs; the MKL and CUDA
tests skip with a clear reason and will execute as-is on hardware where
``sparse_dot_mkl`` / ``cupy`` are available.
"""

import importlib
import os

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun


def _has_backend(modname):
    try:
        importlib.import_module(modname)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def base_config():
    """Common 2D-database config used by every backend in this module."""
    return dict(
        mceq_db_fname="mceq_db_URQMD_150GeV_2D.h5",
        e_min=1e-1,
        e_max=1e4,
        muon_helicity_dependence=True,
        muon_multiple_scattering=False,
        theta_deg=60.0,
        interaction_model="SIBYLL23D",
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


def test_2d_accelerate_matches_numpy(base_config):
    """Accelerate ETD2 on the 2D stitched matrix matches numpy ETD2 to round-off."""
    if not _has_backend("MCEq.spacc"):
        pytest.skip("Apple Accelerate (spacc) not available")
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_acc = _solve("accelerate_etd2", base_config)
    assert sol_numpy.shape == sol_acc.shape
    np.testing.assert_allclose(sol_numpy, sol_acc, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not _has_backend("sparse_dot_mkl"),
    reason="MKL backend (sparse_dot_mkl) not available",
)
def test_2d_mkl_matches_numpy(base_config):
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_mkl = _solve("mkl_etd2", base_config)
    np.testing.assert_allclose(sol_numpy, sol_mkl, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not _has_backend("cupy"),
    reason="CUDA backend (cupy) not available",
)
def test_2d_cuda_matches_numpy(base_config):
    sol_numpy = _solve("numpy_etd2", base_config)
    sol_cuda = _solve("cuda_etd2", base_config)
    # cuSPARSE may reorder partial sums vs scipy CSR; widen tolerance.
    np.testing.assert_allclose(sol_numpy, sol_cuda, rtol=1e-9, atol=1e-10)
