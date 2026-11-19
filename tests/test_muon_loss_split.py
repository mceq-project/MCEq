"""The muon loss split (``muon_losses/<medium>``) in the backend and the matrix builder."""

import importlib.util
import pathlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq import config
from MCEq.operators.matrix_builder import MatrixBuilder

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

SOFT = -1e-3 * np.ones(fixtures.N_E)
HARD = np.triu(np.full((fixtures.N_E, fixtures.N_E), 1e-6), 1) - 3e-6 * np.eye(
    fixtures.N_E
)


def backend(monkeypatch, tmp_path, *, split=True, enable_em=False, cont_rad=True):
    """A backend on the synthetic DB, with a ``muon_losses`` group if ``split``."""
    from MCEq.data import HDF5Backend

    had = fixtures.build_database(tmp_path / "had.h5")
    if split:
        with h5py.File(had, "a") as f:
            g = f.create_group(f"muon_losses/{fixtures.MEDIUM}")
            g["soft_dedx"], g["hard_rates"] = SOFT, HARD
    em = fixtures.build_em_database(
        tmp_path / "em.h5", {fixtures.MEDIUM: (fixtures.EM_CHANNELS, 1.0)}
    )
    for name, value in dict(
        data_dir=had.parent,
        mceq_db_fname=had.name,
        em_db_fname=em.name,
        e_min=None,
        e_max=None,
        em_standalone_grid=False,
        enable_em=enable_em,
        enable_cont_rad_loss=cont_rad,
        muon_loss_split=True,
        interaction_medium=fixtures.MEDIUM,
    ).items():
        monkeypatch.setattr(config, name, value)
    return HDF5Backend(medium=fixtures.MEDIUM)


def total(pdg):
    """The fixture's stored dE/dX (total = ionization)."""
    return -2e-3 * (1.0 + 0.1 * np.log(fixtures.E_GRID))


def test_split_replaces_the_muon_loss_and_returns_the_rates(monkeypatch, tmp_path):
    db = backend(monkeypatch, tmp_path).continuous_loss_db()
    for pid in (13, -13):
        for hel in (0, 1, -1):
            np.testing.assert_array_equal(db["index_d"][(pid, hel)], SOFT)
    for pid in (11, -11, 15, -15):
        np.testing.assert_array_equal(db["index_d"][(pid, 0)], total(pid))
    np.testing.assert_array_equal(db["muon_hard"], HARD)


def test_energy_window_cuts_both_axes(monkeypatch, tmp_path):
    b = backend(monkeypatch, tmp_path)
    b._cuts = slice(2, 7)
    soft, hard = b._muon_loss_split()
    np.testing.assert_array_equal(soft, SOFT[2:7])
    np.testing.assert_array_equal(hard, HARD[2:7, 2:7])


@pytest.mark.parametrize(
    "kwargs",
    [dict(split=False), dict(enable_em=True), dict(cont_rad=False)],
    ids=["no-group", "em-run", "no-radiative-loss"],
)
def test_no_split(monkeypatch, tmp_path, kwargs):
    db = backend(monkeypatch, tmp_path, **kwargs).continuous_loss_db()
    assert db["muon_hard"] is None
    np.testing.assert_array_equal(db["index_d"][(13, 0)], total(13))


def test_switched_off(monkeypatch, tmp_path):
    b = backend(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "muon_loss_split", False)
    assert b._muon_loss_split() is None


def test_jump_operator_acts_on_the_flux():
    """Counts N = phi * w move with J, so phi moves with diag(1/w) J diag(w)."""
    w = np.array([1.0, 2.0, 5.0])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder._energy_grid = SimpleNamespace(w=w)
    hard = np.array([[-1.0, 0.5, 0.2], [0.0, -2.0, 0.3], [0.0, 0.0, -3.0]])
    op = builder.muon_jump_operator(SimpleNamespace(muon_hard_rates=hard))
    phi = np.array([0.3, 0.7, 1.1])
    np.testing.assert_allclose(op @ phi * w, hard @ (phi * w), rtol=1e-15)
