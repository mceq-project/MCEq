"""Material loading and operator assembly, using a tiny independent DB."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq.data.hdf5_backend import HDF5Backend
from MCEq.data.hdf5_store import HDF5Store
from MCEq.operators.matrix_builder import MatrixBuilder


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "scattering.h5"
    with h5py.File(path, "w") as db:
        group = db.create_group("material_scattering/water/muon/test-model")
        group.attrs.update(
            schema_version=1,
            model="test-model",
            rate_units="cm2/g",
            energy_units="GeV",
            energy_kind="kinetic",
            kappa_units="rad^-1",
            axes="energy,kappa",
        )
        group["e_kin"] = [0.1, 1.0, 10.0]
        group["kappa"] = [0.0, 10.0]
        group["rate"] = [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0]]
    backend = HDF5Backend.__new__(HDF5Backend)
    backend.had_fname = str(path)
    backend._had = HDF5Store(path)
    backend.medium = "water"
    backend._e_grid_full = np.array([0.1, 1.0, 10.0])
    backend._cuts = slice(1, 3)
    backend.k_grid = np.array([0.0, 10.0])
    return backend


def test_cuts_orientation_and_missing_material(table):
    np.testing.assert_array_equal(
        table.muon_scattering_rate("test-model"), [[0, 0], [2, 1]]
    )
    table.medium = "air"
    with pytest.raises(ValueError, match="absent"):
        table.muon_scattering_rate("test-model")


@pytest.mark.parametrize("corrupt", ["rate", "units", "grid", "zero", "nan"])
def test_reject_incompatible_table(table, corrupt):
    with h5py.File(table.had_fname, "r+") as db:
        group = db["material_scattering/water/muon/test-model"]
        if corrupt == "units":
            group.attrs["rate_units"] = "cm2/atom"
        elif corrupt == "grid":
            group["e_kin"][1] = 2
        elif corrupt == "zero":
            group["rate"][0, 0] = 1
        elif corrupt == "nan":
            group["rate"][0, 1] = np.nan
        else:
            group["rate"][0, 1] = -1
    with pytest.raises(ValueError):
        table.muon_scattering_rate("test-model")


def test_all_muon_charges_helicities_and_operator_placement(table):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 2, table.k_grid
    builder._layout = table
    builder._physics = SimpleNamespace(
        muon_multiple_scattering=True, muon_scattering_model="test-model"
    )
    builder._grid = SimpleNamespace(dtype=np.float64)
    builder._energy_grid = SimpleNamespace(c=np.array([1.0, 10.0]), d=2)
    particles = {
        (pdg, hel): SimpleNamespace(lidx=2 * i, mceqidx=i)
        for i, (pdg, hel) in enumerate(((p, h) for p in (13, -13) for h in (0, -1, 1)))
    }
    builder._pman = SimpleNamespace(pdg2pref=particles, dim_states=14)

    # _muon_scattering_damping accesses pman[(13,0)] for Gaussian mass only.
    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states = particles, 14
    pman.dim = 2
    builder._pman = pman
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    expected = np.zeros(28)
    expected[14:26] = np.tile([-2.0, -1.0], 6)
    np.testing.assert_array_equal(matrix.diagonal(), expected)
    assert matrix.nnz == 12


def test_scattering_model_invalidates_split_assembly():
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d = True
    builder._grid = SimpleNamespace(dtype=None)
    builder._physics = SimpleNamespace(
        muon_multiple_scattering=True, muon_scattering_model="gaussian"
    )
    old_key = builder._current_assembly_key()
    builder._physics.muon_scattering_model = "screened-coulomb-plane-v1"
    assert old_key != builder._current_assembly_key()
