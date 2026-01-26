import pathlib

import crflux.models as pm
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from MCEq import config
from MCEq.core import MCEqRun


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
        interaction_model="SIBYLL23E",
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
        interaction_model="SIBYLL23E",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


@pytest.fixture(scope="session")
def mceq_qgs():
    config.debug_level = 2
    config.kernel_config = "numpy"
    config.cuda_gpu_id = 0
    config.e_min = 1e-1
    config.e_max = 1e11
    if config.has_mkl:
        config.set_mkl_threads(2)

    return MCEqRun(
        interaction_model="QGSJETII04",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


@pytest.fixture(scope="function")
def ddm_entry():
    from MCEq import ddm, ddm_utils

    entry = ddm._DDMEntry(
        ebeam=ddm_utils.fmteb(2.0),
        projectile=2212,
        secondary=211,
        x17=False,
        tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        tv=1.0,
        te=0.1,
        spl_idx=1,
    )
    return entry


@pytest.fixture(scope="function")
def ddm_channel():
    from MCEq import ddm, ddm_utils

    ch = ddm._DDMChannel(projectile=2212, secondary=211)
    ch.add_entry(
        ebeam=ddm_utils.fmteb(2.0),
        projectile=2212,
        secondary=211,
        x17=False,
        tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        tv=1.0,
        te=1.0,
    )

    return ch


@pytest.fixture(scope="function")
def ddm_spline_db():
    from MCEq import ddm

    db = ddm.DDMSplineDB(
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
    )
    return db


@pytest.fixture(scope="function")
def data_driven_model():
    from MCEq import ddm

    _ddm = ddm.DataDrivenModel(
        e_min=5.0,
        e_max=500.0,
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
        enable_K0_from_isospin=True,
    )
    return _ddm


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


@pytest.fixture(scope="session")
def toy_solver_setup():
    nsteps = 10
    size = 5
    dX = np.full(nsteps, 0.1)
    rho_inv = np.ones(nsteps)
    grid_idcs = list(range(nsteps))

    lam_int = 0.3
    lam_dec = 0.1
    int_m = csr_matrix(-lam_int * np.eye(size))
    dec_m = csr_matrix(-lam_dec * np.eye(size))

    phi = np.ones(size)

    return nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs
