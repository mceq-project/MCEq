"""Model names, equivalence tables and the family fallback of the hadronic models."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from MCEq.data import HDF5Backend, equivalences
from MCEq.data.hadronic_models import (
    equivalences_of,
    family_fallback,
    mapped_cross_section,
)
from MCEq.misc import normalize_hadronic_model_name

_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures_model_names",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

#: Model names of the shipped databases, two un-normalised spellings and "".
MODEL_NAMES = (
    "DPMJETIII191",
    "DPMJETIII193",
    "DPMJETIII306",
    "EPOSLHC",
    "EPOSLHCR",
    "FLUKA20251",
    "QGSJET01C",
    "QGSJETII03",
    "QGSJETII04",
    "QGSJETIII",
    "SIBYLL21",
    "SIBYLL23",
    "SIBYLL23C",
    "SIBYLL23D",
    "SIBYLL23E",
    "SIBYLL23ESTARBAR",
    "SIBYLL23ESTARMIXED",
    "SIBYLL23ESTARRHO",
    "SIBYLL23ESTARSTRANGE",
    "URQMD34",
    "SIBYLL-2.3d",
    "dpmjet-III-19.1",
    "",
)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_a_model_name_matches_at_most_one_family(name):
    mname = normalize_hadronic_model_name(name)
    families = [f for f in equivalences if f in mname]
    assert len(families) <= 1
    expected = equivalences[families[0]] if families else None
    assert equivalences_of(mname) is expected


def test_unknown_family_is_none():
    assert equivalences_of("URQMD34") is None
    assert equivalences_of("") is None


def _backend_on(monkeypatch, path):
    """An `HDF5Backend` on one synthetic file, every read setting pinned."""
    from MCEq import config

    for name, value in dict(
        data_dir=path.parent,
        mceq_db_fname=path.name,
        mceq_db_manifest="mceq_db_manifest_ci_v2.yaml",
        # No EM database exists for the fixtures; enable_em is False.
        em_db_fname="no_such_em_db.h5",
        e_min=None,
        e_max=None,
        floatlen=None,
        em_standalone_grid=False,
        assume_nucleon_interactions_for_exotics=True,
        enable_em=False,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium=fixtures.MEDIUM,
        muon_helicity_dependence=False,
    ).items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setitem(config.adv_set, "disabled_particles", [])
    monkeypatch.setitem(config.adv_set, "forced_int_cs", None)
    monkeypatch.setitem(config.adv_set, "replace_meson_cross_sections_with", None)
    return HDF5Backend(medium=fixtures.MEDIUM)


def test_an_unknown_family_raises_for_yields_and_falls_back_for_cross_sections(
    tmp_path, monkeypatch
):
    """Yields need the equivalence table; a cross section falls back to the family."""
    monkeypatch.setattr(fixtures, "MODEL", "URQMD34")
    path = fixtures.build_database(tmp_path / "urqmd34_1d.h5")
    backend = _backend_on(monkeypatch, path)

    with pytest.raises(ValueError, match="Unknown equivalence table"):
        backend._interaction_db_single("URQMD34")

    proton_column = np.array([1.0, 2.0, 3.0])
    # A neutron column would be picked by a family mapping 3122 -> 2112.
    index_d = {2212: proton_column, 2112: np.array([4.0, 5.0, 6.0])}
    assert mapped_cross_section(index_d, 3122, "URQMD34") is proton_column
    assert mapped_cross_section({}, 3122, "URQMD34") is None


@pytest.mark.parametrize(
    ("projectile", "expected"),
    [
        (321, [321, 321]),
        (-321, [-321, 321]),
        (130, [321, 321]),
        (310, [321, 321]),
        (311, [321, 321]),
        (333, [321, 321]),
        (211, [211, 211]),
        (-211, [-211, 211]),
        (111, [211, 211]),
        (2212, [2212, 2212]),
        (-2212, [-2212, 2212]),
        (3122, [2212, 2212]),
        (-3122, [-2212, 2212]),
        (13, []),
        (11, []),
        (22, []),
        (10313, []),
        (5122, []),
    ],
)
def test_family_fallback_table(projectile, expected):
    """130 lies in both the kaon and the pion window; the kaon branch wins."""
    assert family_fallback(projectile) == expected


def test_normalize_hadronic_model_name():
    assert normalize_hadronic_model_name("SIBYLL-2.3d") == "SIBYLL23D"
    assert normalize_hadronic_model_name("dpmjet-III-19.1") == "DPMJETIII191"
    assert normalize_hadronic_model_name("") == ""
    assert normalize_hadronic_model_name("SIBYLL23D") == "SIBYLL23D"
