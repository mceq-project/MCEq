"""The EM DB's restricted muon radiative loss (``mu_soft_dedx``) in ``continuous_loss_db``."""

import importlib.util
import pathlib

import h5py
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures", pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py"
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def backend(monkeypatch, tmp_path, soft=None, enable_em=True):
    """A backend on the synthetic DBs; ``soft`` is written as ``mu_soft_dedx``."""
    from MCEq import config
    from MCEq.data import HDF5Backend

    had = fixtures.build_database(tmp_path / "had.h5")
    em = fixtures.build_em_database(
        tmp_path / "em.h5", {fixtures.MEDIUM: (fixtures.EM_CHANNELS, 1.0)}
    )
    if soft is not None:
        with h5py.File(em, "a") as f:
            f[f"electromagnetic/{fixtures.MEDIUM}"].create_dataset(
                "mu_soft_dedx", data=soft
            )
            if soft.size != fixtures.N_E:
                f.create_group("common").attrs["e_grid"] = np.geomspace(
                    fixtures.E_GRID[0], fixtures.E_GRID[-1], soft.size
                )
    for name, value in dict(
        data_dir=had.parent,
        mceq_db_fname=had.name,
        em_db_fname=em.name,
        e_min=None,
        e_max=None,
        em_standalone_grid=False,
        enable_em=enable_em,
        enable_cont_rad_loss=False,
        interaction_medium=fixtures.MEDIUM,
    ).items():
        monkeypatch.setattr(config, name, value)
    return HDF5Backend(medium=fixtures.MEDIUM)


def ionization(pdg):
    """The fixture's stored ionization dE/dX for ``pdg``."""
    return -2e-3 * (1.0 + 0.1 * np.log(fixtures.E_GRID))


def test_soft_loss_is_added_to_muons_only(monkeypatch, tmp_path):
    soft = -1e-4 * fixtures.E_GRID**0.5
    index_d = backend(monkeypatch, tmp_path, soft).continuous_loss_db()["index_d"]
    for pid in (13, -13):
        for hel in (0, 1, -1):
            assert np.allclose(index_d[(pid, hel)], ionization(pid) + soft, rtol=0)
    for pid in (11, -11, 15, -15):
        assert np.array_equal(index_d[(pid, 0)], ionization(pid))


def test_db_without_soft_loss_is_unchanged(monkeypatch, tmp_path):
    index_d = backend(monkeypatch, tmp_path).continuous_loss_db()["index_d"]
    assert np.array_equal(index_d[(13, 0)], ionization(13))


def test_soft_loss_ignored_with_em_off(monkeypatch, tmp_path):
    soft = np.full(fixtures.N_E, -1e-4)
    index_d = backend(monkeypatch, tmp_path, soft, enable_em=False).continuous_loss_db()
    # EM off: total loss, which the fixture stores equal to ionization.
    assert np.array_equal(index_d["index_d"][(13, 0)], ionization(13))


def test_soft_loss_on_another_grid_is_interpolated(monkeypatch, tmp_path):
    soft = np.full(2 * fixtures.N_E, -1e-4)
    index_d = backend(monkeypatch, tmp_path, soft).continuous_loss_db()["index_d"]
    assert np.allclose(index_d[(13, 0)], ionization(13) - 1e-4, rtol=0, atol=1e-15)
