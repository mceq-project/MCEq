"""EM in 2D runs: e+- get the scattering rates of the 2D EM database, and a
1D EM database is refused."""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq import config
from MCEq.data.hdf5_store import HDF5Store
from MCEq.operators import scattering
from MCEq.operators.matrix_builder import MatrixBuilder
from tests.test_data_hdf5_decode import backend

_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

ELECTRON_MASS = 0.000511


def _em_builder(monkeypatch, l_grid, e_kin, table):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, len(l_grid), l_grid
    stored = None if table is None else (table, l_grid, e_kin, {"mass": ELECTRON_MASS})
    builder._layout = SimpleNamespace(electron_scattering_rate_table=lambda: stored)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "floatlen", np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=len(e_kin))
    particles = {
        (11, 0): SimpleNamespace(lidx=0, mceqidx=0, mass=ELECTRON_MASS),
        (-11, 0): SimpleNamespace(lidx=len(e_kin), mceqidx=1, mass=ELECTRON_MASS),
    }

    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states, pman.dim = particles, 2 * len(e_kin), len(e_kin)
    builder._pman = pman
    return builder


L_GRID, E_KIN = np.array([0.0, 1.0, 67.0]), np.array([0.002, 0.05, 3.0])
TABLE = np.array([[0.0, 0.0, 0.0], [5.0, 2.0, 0.1], [900.0, 300.0, 4.0]])


def test_electron_damping_placement(monkeypatch):
    """Both e+- diagonals carry -Gamma_l(E) of the EM database's table in every
    mode l > 0, nothing at l = 0."""
    builder = _em_builder(monkeypatch, L_GRID, E_KIN, TABLE)
    monkeypatch.setattr(config, "enable_em", True)
    monkeypatch.setattr(config, "em_multiple_scattering", True)
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    expected = np.concatenate([np.tile(-TABLE[k], 2) for k in range(len(L_GRID))])
    np.testing.assert_array_equal(matrix.diagonal(), expected)


def test_missing_electron_table_is_refused(monkeypatch):
    builder = _em_builder(monkeypatch, L_GRID, E_KIN, None)
    monkeypatch.setattr(config, "enable_em", True)
    monkeypatch.setattr(config, "em_multiple_scattering", True)
    with pytest.raises(ValueError, match="em_collinear_2d"):
        builder._csr_from_blocks({}, apply_muon_scattering=True)


@pytest.mark.parametrize("enable_em,em_ms", [(True, False), (False, True)])
def test_electron_damping_off(monkeypatch, enable_em, em_ms):
    builder = _em_builder(monkeypatch, L_GRID, E_KIN, None)
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


def test_one_dimensional_em_database_is_refused_in_a_2d_run(
    monkeypatch, hdf5_fixture_dbs
):
    two_d = backend(monkeypatch, hdf5_fixture_dbs["two_d_width4"])
    two_d._em_store = HDF5Store(str(hdf5_fixture_dbs["one_d_width4"]))
    with pytest.raises(ValueError, match="em_collinear_2d"):
        two_d._check_em_modes()
    two_d._em_store = HDF5Store(str(hdf5_fixture_dbs["two_d_width4"]))
    two_d._check_em_modes()
