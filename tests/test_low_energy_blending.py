from types import SimpleNamespace

import numpy as np

from MCEq.data import HDF5Backend


def backend(energy, *, low="FLUKA20251", transition=80.0, width=0.4):
    obj = object.__new__(HDF5Backend)
    obj._energy_grid = SimpleNamespace(c=np.asarray(energy, dtype=float))
    obj.low_energy_model = low
    obj.he_le_transition = transition
    obj.he_le_trwidth = width
    return obj


def interaction_index(index_d):
    relations = {}
    for parent, child in index_d:
        relations.setdefault(parent, []).append(child)
    particles = sorted({p for channel in index_d for p in channel})
    return {
        "parents": sorted(relations),
        "particles": particles,
        "relations": relations,
        "index_d": index_d,
        "description": None,
    }


def test_sigmoid_width_is_ten_to_ninety_percent_in_decades():
    transition = 80.0
    width = 0.4
    energies = transition * 10.0 ** np.array([-width / 2, 0.0, width / 2])
    weight = backend(energies, transition=transition, width=width)._he_le_weight()
    assert np.allclose(weight, [0.1, 0.5, 0.9])


def test_zero_width_is_hard_switch():
    weight = backend([79.0, 80.0, 81.0], width=0.0)._he_le_weight()
    assert np.array_equal(weight, [0.0, 1.0, 1.0])


def test_yields_blend_by_parent_energy_column():
    obj = backend([8.0, 80.0, 800.0], width=0.0)
    shared = ((2212, 0), (211, 0))
    he_only = ((2212, 0), (411, 0))
    he = interaction_index(
        {
            shared: np.full((3, 3), 2.0),
            he_only: np.full((3, 3), 4.0),
        }
    )
    le = interaction_index({shared: np.full((3, 3), 10.0)})
    mixed = obj._blend_interaction_dbs(he, le, "HE", "LE")
    assert np.array_equal(
        mixed["index_d"][shared],
        np.array([[10.0, 2.0, 2.0]] * 3),
    )
    # Historical semantics: an HE-only yield channel remains unchanged.
    assert np.array_equal(
        mixed["index_d"][he_only],
        np.full((3, 3), 4.0),
    )


def test_cross_sections_blend_separately_and_use_equivalences():
    obj = backend([8.0, 80.0, 800.0], width=0.0)
    he = {
        "parents": [111, 2212],
        "index_d": {
            111: np.full(3, 2.0),
            2212: np.full(3, 4.0),
        },
    }
    le = {
        "parents": [211, 2212],
        "index_d": {
            211: np.full(3, 10.0),
            2212: np.full(3, 20.0),
        },
    }
    obj._cs_db_single = lambda name: le if name == "FLUKA20251" else he
    mixed = obj.cs_db("SIBYLL23E")
    # FLUKA/DPMJET equivalence maps pi0 -> charged pion.
    assert np.array_equal(mixed["index_d"][111], [10.0, 2.0, 2.0])
    assert np.array_equal(mixed["index_d"][2212], [20.0, 4.0, 4.0])


def test_disabled_backend_path_loads_only_selected_model():
    obj = backend([1.0], low=None)
    marker = object()
    obj._interaction_db_single = lambda name: marker
    assert obj.interaction_db("SIBYLL23E") is marker
