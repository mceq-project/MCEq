from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from MCEq import config
from MCEq.data import HDF5Backend


def backend(energy, *, low="FLUKA20251", transition=80.0, width=0.4, dtype=float):
    obj = object.__new__(HDF5Backend)
    obj._energy_grid = SimpleNamespace(c=np.asarray(energy, dtype=dtype))
    obj.low_energy_model = low
    obj.he_le_transition = transition
    obj.he_le_trwidth = width
    return obj


@pytest.fixture
def channel_pack():
    """Build a one-channel HDF5 pack and the backend that reads it.

    The factory returns the ``(n_e, n_e)`` matrix ``_gen_db_dictionary``
    reconstructs for the single ``(2212, 0) -> (211, 0)`` channel, plus the
    dictionary itself. The file is in-memory (``driver="core"``), so this stays
    database-free while still going through h5py rather than a stand-in: the
    dtype question under test is a property of the real read.
    """
    handles = []

    def make(matrix, dtype):
        matrix = np.asarray(matrix, dtype=dtype)
        csr = csr_matrix(matrix)
        handle = h5py.File("pack.h5", "w", driver="core", backing_store=False)
        handles.append(handle)
        root = handle.create_dataset(
            "pack", data=np.vstack([csr.data, csr.indices]).astype(dtype)
        )
        root.attrs["description"] = "synthetic single-channel pack"
        root.attrs["len_data"] = np.array([csr.nnz])
        root.attrs["tuple_idcs"] = np.array([[2212, 211]])
        indptrs = handle.create_dataset("indptrs", data=csr.indptr[np.newaxis, :])

        obj = object.__new__(HDF5Backend)
        obj._grid = SimpleNamespace(dtype=config.grid.dtype)
        obj._physics = SimpleNamespace(
            filters={"disabled_particles": []},
            assume_nucleon_interactions_for_exotics=False,
        )
        obj.is_2d = False
        obj.n_k = 1
        obj.dim_full = matrix.shape[0]
        obj.min_idx = 0
        obj.max_idx = matrix.shape[0]
        obj._cuts = slice(0, matrix.shape[0])

        index = obj._gen_db_dictionary(root, indptrs)
        return index, index["index_d"][((2212, 0), (211, 0))]

    yield make
    for handle in handles:
        handle.close()


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
        "parents": [111, 310, 2212],
        "index_d": {
            111: np.full(3, 2.0),
            310: np.full(3, 3.0),
            2212: np.full(3, 4.0),
        },
    }
    le = {
        "parents": [130, 211, 321, 2212],
        "index_d": {
            130: np.full(3, 5.0),
            211: np.full(3, 10.0),
            321: np.full(3, 15.0),
            2212: np.full(3, 20.0),
        },
    }
    obj._cs_db_single = lambda name: le if name == "FLUKA20251" else he
    mixed = obj.cs_db("SIBYLL23E")
    # pi0 -> charged pion. Both routes in `_mapped_cross_section` agree here
    # (the FLUKA table maps 111 -> 211, and so does the |pdg| in (100, 300)
    # family fallback), so this assertion does not by itself show the model
    # equivalence table is consulted -- the K0S one below does.
    assert np.array_equal(mixed["index_d"][111], [10.0, 2.0, 2.0])
    assert np.array_equal(mixed["index_d"][2212], [20.0, 4.0, 4.0])
    # K0S is the discriminating case: the FLUKA table maps 310 -> 130, the
    # kaon-family fallback maps it to 321, and the table wins because its
    # candidates are appended first. Dropping the table would silently pick
    # the 321 column (15.0) instead.
    assert np.array_equal(mixed["index_d"][310], [5.0, 3.0, 3.0])


def test_le_only_projectiles_survive_the_blend_with_mapped_he_sigma():
    obj = backend([8.0, 80.0, 800.0], width=0.0)
    he = {
        "parents": [2212, -2212],
        "index_d": {
            2212: np.full(3, 4.0),
            -2212: np.full(3, 6.0),
        },
    }
    le = {
        "parents": [2112, -2112, 2212, -2212],
        "index_d": {
            2112: np.full(3, 10.0),
            -2112: np.full(3, 30.0),
            2212: np.full(3, 20.0),
            -2212: np.full(3, 40.0),
        },
    }
    obj._cs_db_single = lambda name: le if name == "FLUKA20251" else he
    mixed = obj.cs_db("SIBYLL23E")
    # HE parents keep the historical behaviour.
    assert np.array_equal(mixed["index_d"][2212], [20.0, 4.0, 4.0])
    # An LE-only projectile keeps its own sigma below the transition and
    # switches to the mapped HE equivalent above it (n -> p).
    assert 2112 in mixed["parents"]
    assert np.array_equal(mixed["index_d"][2112], [10.0, 4.0, 4.0])
    # Antibaryons map to the HE pbar column (annihilation channels), not
    # to the proton one.
    assert np.array_equal(mixed["index_d"][-2112], [30.0, 6.0, 6.0])


def test_disabled_backend_path_loads_only_selected_model():
    obj = backend([1.0], low=None)
    marker = object()
    obj._interaction_db_single = lambda name: marker
    assert obj.interaction_db("SIBYLL23E") is marker


def test_requesting_the_low_energy_model_itself_skips_the_blend():
    # The second short circuit in `interaction_db`: with the blend enabled but
    # the LE model itself requested, the db is returned raw rather than blended
    # against a copy of itself. The name is normalised first, so the spelled-out
    # alias hits the same branch.
    obj = backend([1.0], low="FLUKA20251")
    marker = object()
    obj._interaction_db_single = lambda name: marker
    assert obj.interaction_db("fluka20251") is marker


def test_grid_dtype_is_none_by_default():
    # The premise of the three dtype pins below: `config.floatlen` is None, so
    # every `dtype=self._grid.dtype` in data.py is a `dtype=None`.
    assert config.floatlen is None
    assert config.grid.dtype is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_channel_pack_dtype_survives_the_read(channel_pack, dtype):
    """The one `dtype=None` in data.py that preserves anything.

    Ledger: risk 8 ("dtype") of the 2026-08-30 data-layer audit, and the Phase 4
    accept item "dtype test (float32 propagates)".

    `_gen_db_dictionary` reads the pack as `np.asarray(hdf_root[:, :],
    dtype=self._grid.dtype)`, and `np.asarray(x, dtype=None)` keeps `x`'s own
    dtype. A float32 channel pack therefore arrives as a float32 yield matrix;
    only float64 packs exist on disk today (the shipped v140 database stores
    (2, N) float64), so this is the read's latent behaviour, not a live one.
    """
    _, matrix = channel_pack(np.full((3, 3), 2.0), dtype)
    assert matrix.dtype == dtype
    assert np.array_equal(matrix, np.full((3, 3), 2.0))


@pytest.mark.parametrize("width", [0.0, 0.4])
def test_he_le_weight_is_float64_in_both_branches(width):
    """The two `dtype=None` sites in `_he_le_weight` preserve nothing.

    Both branches come back float64 whatever the energy grid is, but neither
    `dtype=None` is what makes them float64, and a pure-function rewrite must
    not "fix" either by accident.

    Hard switch: `.astype(grid.dtype)` runs on a *bool* array, and
    `.astype(None)` forces float64 — preserving would have given bool.

    Sigmoid: the value handed to `np.asarray(..., dtype=None)` is already
    float64, over-determinedly so. Removing only the `np.asarray(...,
    dtype=float)` on the energy grid still leaves float64 (measured), because
    `np.log(9.0)` is an `np.float64` *scalar* and under NEP 50 promotes the
    float32 `arg` to float64 on its own. Both forcings have to go before the
    sigmoid comes back float32.
    """
    weight = backend([8.0, 80.0, 800.0], width=width, dtype=np.float32)._he_le_weight()
    assert weight.dtype == np.float64


def test_float32_yields_upcast_to_float64_through_the_blend():
    """A float32 channel pack does not stay float32 past the blend.

    The float64 weight of the test above wins the promotion, so a blended
    channel is float64 while an HE-only channel — which is passed through
    untouched — keeps the float32 the read gave it. That asymmetry inside one
    returned `index_d` is today's behaviour; pinning it here so the pure
    `blend_yields` cannot change the mix silently.
    """
    obj = backend([8.0, 80.0, 800.0], width=0.0)
    shared = ((2212, 0), (211, 0))
    he_only = ((2212, 0), (411, 0))
    he = interaction_index(
        {
            shared: np.full((3, 3), 2.0, dtype=np.float32),
            he_only: np.full((3, 3), 4.0, dtype=np.float32),
        }
    )
    le = interaction_index({shared: np.full((3, 3), 10.0, dtype=np.float32)})
    mixed = obj._blend_interaction_dbs(he, le, "HE", "LE")
    assert mixed["index_d"][shared].dtype == np.float64
    assert mixed["index_d"][he_only].dtype == np.float32


def test_blend_description_records_transition_and_width():
    # Only the weight is used numerically, but the description is part of the
    # returned index and the only record of how the blend was parameterised.
    obj = backend([8.0, 80.0, 800.0], transition=80.0, width=0.4)
    mixed = obj._blend_interaction_dbs(
        interaction_index({}), interaction_index({}), "SIBYLL23E", "FLUKA20251"
    )
    assert mixed["description"] == (
        "Runtime HE/LE blend: SIBYLL23E + FLUKA20251; "
        "transition=80 GeV, 10-90 width=0.4 decades"
    )


def test_blend_description_uses_g_formatting_for_a_zero_width():
    # `:g` on the floats, so a hard switch reads "width=0", not "width=0.0",
    # and a half-decade transition keeps its decimals.
    obj = backend([8.0, 80.0, 800.0], transition=126.5, width=0.0)
    mixed = obj._blend_interaction_dbs(
        interaction_index({}), interaction_index({}), "HE", "LE"
    )
    assert mixed["description"] == (
        "Runtime HE/LE blend: HE + LE; transition=126.5 GeV, 10-90 width=0 decades"
    )


def transition_window(transition=80.0, width=0.4):
    """The 10/50/90 % energies of the sigmoid, so the weight is [0.1, 0.5, 0.9]."""
    return transition * 10.0 ** np.array([-width / 2, 0.0, width / 2])


def test_weight_broadcasts_along_the_last_axis_of_a_matrix():
    # `w_he[np.newaxis, :]` against an (n_e, n_e) channel: the weight varies
    # with the parent (column) index and is constant down the child rows.
    obj = backend(transition_window(), transition=80.0, width=0.4)
    channel = ((2212, 0), (211, 0))
    mixed = obj._blend_interaction_dbs(
        interaction_index({channel: np.zeros((3, 3))}),
        interaction_index({channel: np.ones((3, 3))}),
        "HE",
        "LE",
    )
    blended = mixed["index_d"][channel]
    assert blended.shape == (3, 3)
    assert np.allclose(blended, np.array([[0.9, 0.5, 0.1]] * 3))


def test_weight_broadcasts_along_the_last_axis_of_a_hankel_tensor():
    # Same weight against an (n_k, n_e, n_e) 2D channel: (1, n_e) broadcasts as
    # (1, 1, n_e), so it still varies along the parent axis and is applied
    # identically to every kappa mode. A rewrite that indexed the first axis
    # would pass the matrix test above and fail here.
    obj = backend(transition_window(), transition=80.0, width=0.4)
    channel = ((2212, 0), (211, 0))
    le_tensor = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 2.0)])
    mixed = obj._blend_interaction_dbs(
        interaction_index({channel: np.zeros((2, 3, 3))}),
        interaction_index({channel: le_tensor}),
        "HE",
        "LE",
    )
    blended = mixed["index_d"][channel]
    assert blended.shape == (2, 3, 3)
    assert np.allclose(blended[0], np.array([[0.9, 0.5, 0.1]] * 3))
    assert np.allclose(blended[1], np.array([[1.8, 1.0, 0.2]] * 3))
