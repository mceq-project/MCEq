"""The EM cross-section table must carry the target mass MCEq divides it by."""

import importlib.util
import pathlib

import numpy as np
import pytest

from MCEq.misc import average_A_target

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

#: Air <A> the EM assembler assumed before it followed MCEq to the
#: atom-weighted value (molecule-weighted N2/O2/Ar mass numbers).
A_AIR_MOLECULE_WEIGHTED = 0.78084 * 14 + 0.20946 * 16 + 0.00934 * 40


def em_cs_db(monkeypatch, tmp_path, target_mass):
    from MCEq import config
    from MCEq.data import HDF5Backend

    had = fixtures.build_database(tmp_path / "had.h5", medium="air")
    em = fixtures.build_em_database(
        tmp_path / "em.h5", {"air": (fixtures.EM_CHANNELS, 1.0)}, target_mass
    )
    for name, value in dict(
        data_dir=had.parent,
        mceq_db_fname=had.name,
        mceq_db_manifest="mceq_db_manifest_ci_v2.yaml",
        em_db_fname=em.name,
        e_min=None,
        e_max=None,
        floatlen=None,
        em_standalone_grid=False,
        assume_nucleon_interactions_for_exotics=True,
        enable_em=True,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium="air",
        muon_helicity_dependence=False,
    ).items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setitem(config.adv_set, "disabled_particles", [])
    monkeypatch.setitem(config.adv_set, "forced_int_cs", None)
    monkeypatch.setitem(config.adv_set, "replace_meson_cross_sections_with", None)
    return HDF5Backend(medium="air").cs_db(fixtures.MODEL)


def test_stamped_table_loads(monkeypatch, tmp_path):
    db = em_cs_db(monkeypatch, tmp_path, "mceq")
    assert np.array_equal(db["index_d"][22], fixtures.em_cs_table(1.0)[2])


def test_unstamped_table_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="no 'mceq_target_mass'"):
        em_cs_db(monkeypatch, tmp_path, None)


def test_table_for_another_target_mass_is_refused(monkeypatch, tmp_path):
    factor = average_A_target("air") / A_AIR_MOLECULE_WEIGHTED
    assert abs(factor - 0.992246) < 1e-6
    with pytest.raises(ValueError, match=f"factor {factor:.6f}"):
        em_cs_db(monkeypatch, tmp_path, A_AIR_MOLECULE_WEIGHTED)
