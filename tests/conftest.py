import crflux.models as pm
import pytest
import pathlib

from MCEq.core import MCEqRun
from MCEq import config


@pytest.fixture(scope="session")
def mceq():
    config.debug_level = 2
    config.kernel_config = "numpy"
    config.cuda_gpu_id = 0
    config.e_min = 1e-1
    config.e_max = 1e11
    if config.has_mkl:
        config.set_mkl_threads(2)

    return MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


@pytest.fixture(scope="session")
def mceq_small():
    config.debug_level = 2
    config.kernel_config = "numpy"
    config.cuda_gpu_id = 0

    config.e_min = 1e9
    config.e_max = 1e10
    if config.has_mkl:
        config.set_mkl_threads(2)

    return MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


@pytest.fixture
def msis_expected_file(request):
    test_dir = pathlib.Path(request.fspath).parent
    path = test_dir / "msis_expected.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output file {path} not found. "
            "Please run the test with the expected output file."
        )
    return path


@pytest.fixture(scope="function")
def toy_solver_problem():
    import numpy as np
    from scipy.sparse import csr_matrix

    nsteps = 10
    size = 5
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    grid_idcs = list(range(nsteps))

    # mimic how self.int_m and self.dec_m are used in solve()
    lam_int = 0.3
    lam_dec = 0.1
    int_m = csr_matrix(-lam_int * np.eye(size))  # simple interaction term
    dec_m = csr_matrix(-lam_dec * np.eye(size))  # simple decay term

    phi0 = np.ones(size)
    return nsteps, dX, rho_inv, int_m, dec_m, phi0, grid_idcs
