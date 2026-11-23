"""Muon scattering rates come from the database's table, never computed.

The physics is tested in mceq-EM (``mceq_em.coulomb_scattering``); here: the
table is used on the run's grid, refused when it does not match, and placed on
the muon diagonals of every mode.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq import config
from MCEq.operators import scattering
from MCEq.operators.matrix_builder import MatrixBuilder

MUON_MASS = 0.10566
E = np.logspace(-1, 2, 7)
ELL = np.array([0.0, 1.0, 2.0, 5.0, 12.0])
TABLE = np.arange(ELL.size * E.size, dtype=float).reshape(ELL.size, E.size)


def _layout(rates=TABLE, mass=MUON_MASS, ell=ELL, e=E):
    attrs = {"finite_nuclei": True, "muon_mass": mass, "model": "screened-coulomb"}
    return SimpleNamespace(scattering_rate_table=lambda: (rates, ell, e, attrs))


def test_stored_table_is_used():
    got = scattering.mode_rates(E, ELL, _layout(), MUON_MASS)
    assert np.array_equal(got, TABLE)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"mass": 0.1057}, "muon mass"),
        ({"ell": ELL[:-1]}, "modes"),
        ({"e": E * 1.001}, "modes"),
    ],
)
def test_table_for_other_inputs_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        scattering.mode_rates(E, ELL, _layout(**change), MUON_MASS)


def test_missing_table_is_refused():
    layout = SimpleNamespace(scattering_rate_table=lambda: None)
    with pytest.raises(ValueError, match="no muon scattering rates"):
        scattering.mode_rates(E, ELL, layout, MUON_MASS)


def test_backend_cuts_the_stored_table_to_the_energy_window(tmp_path):
    from MCEq.data.hdf5_backend import HDF5Backend
    from MCEq.data.hdf5_store import HDF5Store
    from MCEq.misc import energy_grid

    e_full = np.logspace(-1, 2, 13)
    k_grid = np.array([0, 1, 2, 486])
    table = np.arange(k_grid.size * e_full.size, dtype=float).reshape(k_grid.size, -1)
    path = tmp_path / "table.h5"
    with h5py.File(path, "w") as f:
        f.create_group("common").attrs.update({"e_grid": e_full, "k_grid": k_grid})
        ds = f.create_dataset(
            f"material_scattering/air/{scattering.RATE_TABLE}", data=table
        )
        ds.attrs.update({"finite_nuclei": True, "muon_mass": MUON_MASS})
    backend = HDF5Backend.__new__(HDF5Backend)
    backend._had, backend.medium = HDF5Store(path), "air"
    backend._cuts = slice(3, 9)
    backend._energy_grid = energy_grid(e_full[3:9], None, None, 6)
    reads = []
    read = backend._had.read_optional_table
    backend._had.read_optional_table = lambda *a: reads.append(a) or read(*a)
    rates, l_grid, e_grid, attrs = backend.scattering_rate_table()
    assert backend.scattering_rate_table()[0] is rates  # kept, not reread
    assert len(reads) == 1
    assert np.array_equal(rates, table[:, 3:9])
    assert np.array_equal(l_grid, k_grid) and np.array_equal(e_grid, e_full[3:9])
    got = scattering.mode_rates(e_full[3:9], k_grid, backend, MUON_MASS)
    assert np.array_equal(got, table[:, 3:9])

    backend.medium = "water"
    assert backend.scattering_rate_table() is None


def test_all_muon_charges_helicities_and_operator_placement(monkeypatch):
    k_grid, e_kin = np.array([0.0, 10.0]), np.array([1.0, 10.0])
    table = np.array([[0.0, 0.0], [3.0, 0.5]])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 2, k_grid
    builder._layout = _layout(table, ell=k_grid, e=e_kin)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "floatlen", np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=2)
    particles = {
        (pdg, hel): SimpleNamespace(lidx=2 * i, mceqidx=i, mass=MUON_MASS)
        for i, (pdg, hel) in enumerate(((p, h) for p in (13, -13) for h in (0, -1, 1)))
    }

    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states, pman.dim = particles, 12, 2
    builder._pman = pman
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    expected = np.zeros(24)
    expected[12:24] = np.tile(-table[1], 6)
    np.testing.assert_array_equal(matrix.diagonal(), expected)
    assert matrix.nnz == 12

    monkeypatch.setattr(config, "muon_multiple_scattering", False)
    assert builder._csr_from_blocks({}, apply_muon_scattering=True).nnz == 0
