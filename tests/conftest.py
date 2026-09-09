import copy
import pathlib

import crflux.models as pm
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from MCEq import config
from MCEq.core import MCEqRun

#: Config attributes every test is guaranteed to leave as it found them (plus
#: ``adv_set``, snapshotted and restored separately because it is deep-copied).
_SNAPSHOT_KEYS = (
    "kernel_config",
    "mceq_db_fname",
    "X_start",
    "em_adaptive_step",
    "em_step_safety",
    "muon_helicity_dependence",
    "debug_level",
    "cuda_gpu_id",
    "mkl_threads",
)


def _snapshot_config():
    """Copy the config state that tests and fixtures are known to mutate.

    ``adv_set`` is deep-copied because tests assign into its entries (lists);
    the other keys are plain values and are stored as they are.
    """
    saved = {key: getattr(config, key) for key in _SNAPSHOT_KEYS}
    saved["adv_set"] = copy.deepcopy(config.adv_set)
    return saved


def _restore_config(saved):
    """Put ``config`` back to the state captured by :func:`_snapshot_config`.

    ``adv_set`` is restored in place (``clear()`` + ``update()``) rather than
    rebound: nothing under ``src/`` binds ``config.adv_set`` to a local name
    (every access is ``config.adv_set[...]`` at call time), so a plain
    reassignment would also work today, but restoring in place keeps the
    module attribute the same dict object forever and so stays correct if a
    future caller does hold a reference.

    ``mkl_threads`` is process-wide BLAS state, so it is put back through
    ``config.set_mkl_threads`` -- and only when it actually changed, because
    that call re-registers the threadpoolctl limiter (and dlopens ``libmkl_rt``
    on its first use in the process; ``_load_mkl`` returns early afterwards).
    ``set_mkl_threads`` is safe when MKL is absent (it checks ``mkl is not
    None``). ``config.has_mkl`` is deliberately not read here: it is a lazy
    probe with side effects.
    """
    config.adv_set.clear()
    config.adv_set.update(saved["adv_set"])
    for key in _SNAPSHOT_KEYS:
        if key == "mkl_threads":
            continue
        setattr(config, key, saved[key])
    if config.mkl_threads != saved["mkl_threads"]:
        config.set_mkl_threads(saved["mkl_threads"])


def _apply_session_config():
    """The config the shared ``MCEqRun`` fixtures are built and used with.

    Tests are calibrated with the EM cascade (e±, helicity variants) in the
    system; the production default disables electrons because of the ETD2 EM
    caveat. Re-enable here so the matrix shapes, particle counts, and
    reference values stay consistent with what the tests expect. The helicity
    pin is defensive: it must stay True (matches the config default; the
    unpolarized decay dataset carries a K_mu2-halving defect in DBs <= v150).
    """
    config.debug_level = 2
    config.cuda_gpu_id = 0
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    config.adv_set["disabled_particles"] = []
    config.muon_helicity_dependence = True
    if config.has_mkl:
        config.set_mkl_threads(2)


@pytest.fixture(autouse=True)
def _restore_global_config_state():
    """Every test starts from the ``config`` state it found and leaves it as
    found -- including tests that take the shared ``MCEqRun`` fixtures.

    pytest sets up higher-scoped fixtures before function-scoped ones, and
    function-scoped autouse fixtures before function-scoped requested ones.
    So the snapshot here is taken after the session-scoped ``MCEqRun``
    instances exist (``_build_shared_run`` restores the config it built them
    under before returning, so they hold no mutation while alive) but
    before the function-scoped ``mceq_sib21`` / ``mceq_qgs`` wrappers apply
    the per-test config those instances need. That per-test config is
    therefore undone by this fixture's restore, and tests that do not take
    the wrappers see the pristine module defaults.
    """
    saved = _snapshot_config()
    try:
        yield
    finally:
        _restore_config(saved)


def _build_shared_run(interaction_model):
    """Construct a shared ``MCEqRun`` under the session config, then put
    ``config`` back *before* returning.

    The restore must not wait for session teardown: a session fixture stays
    alive across every test, so a ``yield``-with-``finally`` would keep the
    mutation in force while ``_restore_global_config_state`` takes its
    per-test snapshot, and the leak this file exists to close would be back.
    Restoring here, right after construction, leaves the process pristine
    between tests; the function-scoped wrappers re-apply the config per test.
    """
    saved = _snapshot_config()
    try:
        _apply_session_config()
        return MCEqRun(
            interaction_model=interaction_model,
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
    finally:
        _restore_config(saved)


@pytest.fixture(scope="session")
def _mceq_sib21_instance():
    """Build the shared SIBYLL21 run once; config is left as it was found."""
    return _build_shared_run("SIBYLL21")


@pytest.fixture(scope="session")
def _mceq_qgs_instance():
    """Build the shared QGSJETII04 run once; config is left as it was found."""
    return _build_shared_run("QGSJETII04")


@pytest.fixture
def mceq_sib21(_mceq_sib21_instance):
    """The shared SIBYLL21 run, with the config it was built under applied for
    the duration of this test only (``_restore_global_config_state`` undoes it).
    """
    _apply_session_config()
    return _mceq_sib21_instance


@pytest.fixture
def mceq_qgs(_mceq_qgs_instance):
    """The shared QGSJETII04 run, with the config it was built under applied
    for the duration of this test only (``_restore_global_config_state``
    undoes it).
    """
    _apply_session_config()
    return _mceq_qgs_instance


@pytest.fixture(scope="function")
def ddm_entry():
    from MCEq.models.ddm import ddm, ddm_utils

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
    from MCEq.models.ddm import ddm, ddm_utils

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
    from MCEq.models.ddm import ddm

    db = ddm.DDMSplineDB(
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
    )
    return db


@pytest.fixture(scope="function")
def data_driven_model():
    from MCEq.models.ddm import ddm

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


@pytest.fixture(scope="session")
def hdf5_fixture_dbs(tmp_path_factory):
    """Synthetic MCEq HDF5 databases for ``tests/test_data_hdf5_decode.py``.

    Built once per session into a tmp directory; they are never committed as
    ``.h5``. Returns ``{variant name: path}`` exactly as produced by
    ``tests/data/make_hdf5_fixtures.build_all`` -- see that module for the
    on-disk layout and for which decode branch each variant reaches.

    Imported by file path because ``tests/data`` is not an importable package.
    """
    import importlib.util

    source = pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py"
    spec = importlib.util.spec_from_file_location("make_hdf5_fixtures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_all(tmp_path_factory.mktemp("hdf5_fixtures"))
