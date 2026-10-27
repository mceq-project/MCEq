"""EM in 2D runs: 1D EM channel packs decode as one (n_e, n_e) block that the
matrix builder copies into every mode (collinear production), and e+- get the
screened Coulomb damping with the electron mass and point nuclei."""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq import config
from MCEq.data.hdf5_store import HDF5Store
from MCEq.operators import scattering
from MCEq.operators.matrix_builder import MatrixBuilder

from tests.test_data_hdf5_decode import NEXT_CH, backend
from tests.test_muon_scattering_coulomb import AIR_INTEGER_A, _backend

_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


def test_one_dimensional_pack_decodes_as_one_block_in_a_2d_run(
    monkeypatch, hdf5_fixture_dbs
):
    """A 1D pack (indptr rows of e_dim + 1) read by a 2D backend gives the
    stored (n_e, n_e) block, not a mode stack."""
    two_d = backend(monkeypatch, hdf5_fixture_dbs["two_d_width4"])
    assert two_d.is_2d and two_d.n_k == fixtures.N_K
    pack = HDF5Store(str(hdf5_fixture_dbs["one_d_width4"])).read_channel_pack(
        "decays", "unpolarized"
    )
    index = two_d._gen_db_dictionary(pack)
    parent, child = fixtures.DECAY_CHANNELS[NEXT_CH]
    decoded = index["index_d"][((parent, 0), (child, 0))]
    assert decoded.shape == (fixtures.N_E, fixtures.N_E)
    assert np.array_equal(decoded, fixtures.channel_block(NEXT_CH))


def _em_builder(monkeypatch, db, l_grid, e_kin):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, len(l_grid), l_grid
    builder._layout = _backend(db)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "floatlen", np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=len(e_kin))
    particles = {
        (11, 0): SimpleNamespace(lidx=0, mceqidx=0),
        (-11, 0): SimpleNamespace(lidx=len(e_kin), mceqidx=1),
    }

    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states, pman.dim = particles, 2 * len(e_kin), len(e_kin)
    builder._pman = pman
    return builder


@pytest.fixture
def db(tmp_path):
    import h5py

    path = tmp_path / "composition.h5"
    with h5py.File(path, "w") as f:
        f["material_scattering/air/composition"] = AIR_INTEGER_A
    return path


def test_electron_damping_placement(monkeypatch, db):
    """Both e+- diagonals carry -Gamma_l(E) of the electron-mass, point-nucleus
    Wentzel rate in every mode l > 0, nothing at l = 0."""
    l_grid, e_kin = np.array([0.0, 1.0, 67.0]), np.array([0.002, 0.05, 3.0])
    builder = _em_builder(monkeypatch, db, l_grid, e_kin)
    monkeypatch.setattr(config, "enable_em", True)
    monkeypatch.setattr(config, "em_multiple_scattering", True)
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    rate = scattering.screened_coulomb_rate_legendre(
        e_kin, l_grid, AIR_INTEGER_A, scattering.ELECTRON_MASS, False
    )
    expected = np.concatenate([np.tile(-rate[k], 2) for k in range(len(l_grid))])
    np.testing.assert_allclose(matrix.diagonal(), expected, rtol=1e-15)
    assert np.all(rate[1:] > 0)
    # lighter projectile, same momentum: the electron rate exceeds the muon rate
    mu = scattering.screened_coulomb_rate_legendre(
        e_kin, l_grid, AIR_INTEGER_A, scattering.MUON_MASS, False
    )
    assert np.all(rate[1:] > mu[1:])


@pytest.mark.parametrize("enable_em,em_ms", [(True, False), (False, True)])
def test_electron_damping_off(monkeypatch, db, enable_em, em_ms):
    builder = _em_builder(monkeypatch, db, np.array([0.0, 5.0]), np.array([0.01, 1.0]))
    monkeypatch.setattr(config, "enable_em", enable_em)
    monkeypatch.setattr(config, "em_multiple_scattering", em_ms)
    assert builder._csr_from_blocks({}, apply_muon_scattering=True).nnz == 0


def test_electron_offsets_skip_disabled_species():
    pdg2pref = {
        (11, 0): SimpleNamespace(lidx=30, mceqidx=2),
        (-11, 0): SimpleNamespace(lidx=-1, mceqidx=-1),
        (13, 0): SimpleNamespace(lidx=60, mceqidx=4),
    }
    assert scattering.electron_state_offsets(pdg2pref) == [30]


def test_muon_damping_stops_below_muon_scattering_e_min(monkeypatch, db):
    """MeV muons (lowered e_min, EM runs) get no damping: their decay is stiff
    too, and the shared basis of the sec(theta) coupling integrates the decay
    exactly only without an l-dependent rate."""
    l_grid, e_kin = np.array([0.0, 4.0]), np.array([0.002, 0.03, 0.5])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 2, l_grid
    builder._layout = _backend(db)
    for name, value in dict(
        muon_multiple_scattering=True, muon_scattering_model="screened-coulomb",
        muon_scattering_finite_nuclei=True, muon_scattering_e_min=0.05,
        enable_em=False, floatlen=np.float64,
    ).items():
        monkeypatch.setattr(config, name, value)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=3)
    muon = SimpleNamespace(lidx=0, mceqidx=0, mass=scattering.MUON_MASS)

    class Pman(dict):
        pass

    pman = Pman({(13, 0): muon})
    pman.pdg2pref, pman.dim_states, pman.dim = {(13, 0): muon}, 3, 3
    builder._pman = pman
    _, damping = builder._muon_scattering_damping()
    assert np.all(damping[:, :2] == 0.0) and np.all(damping[1:, 2] < 0.0)
