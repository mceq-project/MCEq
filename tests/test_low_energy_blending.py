"""The runtime HE/LE blend, tested on the pure functions of the module.

Every blend property is pinned by calling the free functions of
``MCEq.data.blending`` -- ``he_le_weight``, ``blend_yields`` and
``blend_cross_sections`` -- directly with explicit arguments, no backend at
all. Only two things are about ``HDF5Backend`` itself and go through a REAL
backend: the ``interaction_db`` dispatch (blend on / blend off / LE model
requested) and the channel-pack read dtype. That backend is built on a tiny
synthetic two-model database from ``tests/data/make_hdf5_fixtures.py`` in
``tmp_path``, with injected ``paths``/``grid``/``physics`` config groups, so
nothing here touches the process-wide ``MCEq.config`` state.

The ``HDF5Backend`` shells this file used to fabricate by bypassing
``__init__`` are gone: the blend no longer lives on the backend, so there is
nothing left to stub.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq import config
from MCEq.data import HDF5Backend
from MCEq.data.blending import blend_cross_sections, blend_yields, he_le_weight

_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

#: The second model `build_database(low_energy_model=...)` writes. Any name
#: `family_of` resolves is fine; FLUKA is what the production blend uses.
LE_MODEL = "FLUKA20251"
#: Index into `fixtures.INTERACTION_CHANNELS` / `fixtures.LOW_ENERGY_CHANNELS`
#: of the ``(2212, 0) -> (211, 0)`` channel, which both packs carry.
SHARED = ((2212, 0), (211, 0))
HE_ICH = fixtures.INTERACTION_CHANNELS.index((2212, 211))
LE_ICH = fixtures.LOW_ENERGY_CHANNELS.index((2212, 211))
#: ``(211, 0) -> (211, 0)`` is in the HE pack only.
HE_ONLY = ((211, 0), (211, 0))


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


def transition_window(transition=80.0, width=0.4):
    """The 10/50/90 % energies of the sigmoid, so the weight is [0.1, 0.5, 0.9]."""
    return transition * 10.0 ** np.array([-width / 2, 0.0, width / 2])


def fixture_backend(path, *, low_energy_model=None, width=0.0):
    """A real ``HDF5Backend`` on one fixture file, with injected config groups.

    ``e_min``/``e_max`` are ``None`` so the whole 1 -- 100 GeV fixture grid is
    read and the 80 GeV transition falls inside it (only the last bin, centred
    at 89.1 GeV, lies above it). ``assume_nucleon_interactions_for_exotics``
    is off so the decoded channel set is exactly what the pack stores.
    """
    paths = SimpleNamespace(
        data_dir=path.parent,
        mceq_db_fname=path.name,
        em_db_fname="no_such_em_db.h5",
    )
    grid = SimpleNamespace(e_min=None, e_max=None, dtype=None, em_standalone_grid=False)
    physics = SimpleNamespace(
        filters={
            "disabled_particles": [],
            "forced_int_cs": None,
            "replace_meson_cross_sections_with": None,
        },
        assume_nucleon_interactions_for_exotics=False,
        enable_em=False,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium=fixtures.MEDIUM,
        muon_helicity_dependence=False,
    )
    return HDF5Backend(
        medium=fixtures.MEDIUM,
        low_energy_model=low_energy_model,
        he_le_transition=80.0,
        he_le_trwidth=width,
        paths=paths,
        grid=grid,
        physics=physics,
    )


@pytest.fixture
def two_model_db(tmp_path):
    """A float64 fixture database carrying both `fixtures.MODEL` and `LE_MODEL`."""
    return fixtures.build_database(tmp_path / "two_model.h5", low_energy_model=LE_MODEL)


def test_sigmoid_width_is_ten_to_ninety_percent_in_decades():
    weight = he_le_weight(transition_window(80.0, 0.4), 80.0, 0.4, dtype=None)
    assert np.allclose(weight, [0.1, 0.5, 0.9])


def test_zero_width_is_hard_switch():
    weight = he_le_weight([79.0, 80.0, 81.0], 80.0, 0.0, dtype=None)
    assert np.array_equal(weight, [0.0, 1.0, 1.0])


def test_yields_blend_by_parent_energy_column():
    shared = ((2212, 0), (211, 0))
    he_only = ((2212, 0), (411, 0))
    he = interaction_index(
        {
            shared: np.full((3, 3), 2.0),
            he_only: np.full((3, 3), 4.0),
        }
    )
    le = interaction_index({shared: np.full((3, 3), 10.0)})
    w = he_le_weight([8.0, 80.0, 800.0], 80.0, 0.0, dtype=None)
    mixed = blend_yields(he, le, "HE", "LE", w, 80.0, 0.0)
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
    # The names select the alias table, so they are the normalised model
    # names, not free-form labels.
    w = he_le_weight([8.0, 80.0, 800.0], 80.0, 0.0, dtype=None)
    mixed = blend_cross_sections(he, le, "SIBYLL23E", "FLUKA20251", w)
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
    w = he_le_weight([8.0, 80.0, 800.0], 80.0, 0.0, dtype=None)
    mixed = blend_cross_sections(he, le, "SIBYLL23E", "FLUKA20251", w)
    # HE parents keep the historical behaviour.
    assert np.array_equal(mixed["index_d"][2212], [20.0, 4.0, 4.0])
    # An LE-only projectile keeps its own sigma below the transition and
    # switches to the mapped HE equivalent above it (n -> p).
    assert 2112 in mixed["parents"]
    assert np.array_equal(mixed["index_d"][2112], [10.0, 4.0, 4.0])
    # Antibaryons map to the HE pbar column (annihilation channels), not
    # to the proton one.
    assert np.array_equal(mixed["index_d"][-2112], [30.0, 6.0, 6.0])


def test_disabled_backend_path_loads_only_selected_model(two_model_db):
    """With ``low_energy_model=None``, ``interaction_db`` returns the single
    model's index untouched."""
    index = fixture_backend(two_model_db).interaction_db(fixtures.MODEL)
    # The pack's own description plus the model-name line that loading a
    # single model appends -- not the blend's "Runtime HE/LE blend" description.
    assert index["description"] == (
        f"synthetic {fixtures.MODEL} yields\nInteraction model name: {fixtures.MODEL}"
    )
    # The raw pack, unblended, although the file also carries the LE model.
    assert np.array_equal(index["index_d"][SHARED], fixtures.channel_block(HE_ICH))


def test_enabled_backend_path_blends_the_two_models(two_model_db):
    backend = fixture_backend(two_model_db, low_energy_model=LE_MODEL)
    index = backend.interaction_db(fixtures.MODEL)
    assert index["description"] == (
        f"Runtime HE/LE blend: {fixtures.MODEL} + {LE_MODEL}; "
        "transition=80 GeV, 10-90 width=0 decades"
    )
    energy = backend.energy_grid.c
    above = energy >= 80.0
    assert above.sum() == 1
    # Width 0 so the blend is exact column selection; a float64 pack so exact
    # equality holds. This is the end-to-end form of the pure blend_yields
    # column test above.
    assert np.array_equal(
        index["index_d"][SHARED][:, above],
        fixtures.channel_block(HE_ICH)[:, above],
    )
    assert np.array_equal(
        index["index_d"][SHARED][:, ~above],
        fixtures.channel_block(LE_ICH)[:, ~above],
    )
    assert np.array_equal(
        index["index_d"][HE_ONLY],
        fixtures.channel_block(fixtures.INTERACTION_CHANNELS.index((211, 211))),
    )


def test_requesting_the_low_energy_model_itself_skips_the_blend(two_model_db):
    # The second short circuit in `interaction_db`: with the blend enabled but
    # the LE model itself requested, the db is returned raw rather than blended
    # against a copy of itself. The name is normalised first, so the spelled-out
    # alias hits the same branch.
    backend = fixture_backend(two_model_db, low_energy_model=LE_MODEL)
    index = backend.interaction_db("fluka-2025.1")
    # Raw, not blended: the pack's own description plus the model-name line
    # that loading a single model appends, and only the LE pack's channels.
    assert index["description"] == (
        f"synthetic {LE_MODEL} yields\nInteraction model name: {LE_MODEL}"
    )
    assert set(index["index_d"]) == {
        ((p, 0), (c, 0)) for p, c in fixtures.LOW_ENERGY_CHANNELS
    }
    assert np.array_equal(index["index_d"][SHARED], fixtures.channel_block(LE_ICH))


def test_grid_dtype_is_none_by_default():
    # The premise of the three dtype pins below: `config.floatlen` is None, so
    # every `dtype=self._grid.dtype` in data.py is a `dtype=None`.
    assert config.floatlen is None
    assert config.grid.dtype is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_channel_pack_dtype_survives_the_read(tmp_path, dtype):
    """The one `dtype=None` in data.py that preserves anything.

    Ledger: risk 8 ("dtype") of the 2026-08-30 data-layer audit, and the Phase 4
    accept item "dtype test (float32 propagates)".

    `_gen_db_dictionary` reads the pack as `np.asarray(hdf_root[:, :],
    dtype=self._grid.dtype)`, and `np.asarray(x, dtype=None)` keeps `x`'s own
    dtype. A float32 channel pack therefore arrives as a float32 yield matrix;
    only float64 packs exist on disk today (the shipped v140 database stores
    (2, N) float64), so this is the read's latent behaviour, not a live one.
    """
    path = fixtures.build_database(
        tmp_path / f"pack_{np.dtype(dtype).name}.h5", pack_dtype=dtype
    )
    matrix = fixture_backend(path).interaction_db(fixtures.MODEL)["index_d"][SHARED]
    assert matrix.dtype == dtype
    assert np.array_equal(matrix, fixtures.channel_block(HE_ICH).astype(dtype))


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
    energy = np.asarray([8.0, 80.0, 800.0], dtype=np.float32)
    weight = he_le_weight(energy, 80.0, width, dtype=None)
    assert weight.dtype == np.float64


def test_float32_yields_upcast_to_float64_through_the_blend():
    """A float32 channel pack does not stay float32 past the blend.

    The float64 weight of the test above wins the promotion, so a blended
    channel is float64 while an HE-only channel — which is passed through
    untouched — keeps the float32 the read gave it. That asymmetry inside one
    returned `index_d` is today's behaviour; pinning it here so the pure
    `blend_yields` cannot change the mix silently.
    """
    shared = ((2212, 0), (211, 0))
    he_only = ((2212, 0), (411, 0))
    he = interaction_index(
        {
            shared: np.full((3, 3), 2.0, dtype=np.float32),
            he_only: np.full((3, 3), 4.0, dtype=np.float32),
        }
    )
    le = interaction_index({shared: np.full((3, 3), 10.0, dtype=np.float32)})
    w = he_le_weight([8.0, 80.0, 800.0], 80.0, 0.0, dtype=None)
    mixed = blend_yields(he, le, "HE", "LE", w, 80.0, 0.0)
    assert mixed["index_d"][shared].dtype == np.float64
    assert mixed["index_d"][he_only].dtype == np.float32


def test_float32_pack_asymmetry_survives_the_real_blend(tmp_path):
    """The read hands the blend float32 matrices and the blend upcasts only
    the shared channels; this checks the two compose as claimed in the
    ``MCEq.data.blending`` module docstring."""
    path = fixtures.build_database(
        tmp_path / "pack_float32.h5", pack_dtype=np.float32, low_energy_model=LE_MODEL
    )
    index = fixture_backend(path, low_energy_model=LE_MODEL).interaction_db(
        fixtures.MODEL
    )
    assert index["index_d"][SHARED].dtype == np.float64
    assert index["index_d"][HE_ONLY].dtype == np.float32


def test_blend_description_records_transition_and_width():
    # Only the weight is used numerically, but the description is part of the
    # returned index and the only record of how the blend was parameterised.
    w = he_le_weight([8.0, 80.0, 800.0], 80.0, 0.4, dtype=None)
    description = blend_yields(
        interaction_index({}),
        interaction_index({}),
        "SIBYLL23E",
        "FLUKA20251",
        w,
        80.0,
        0.4,
    )["description"]
    assert description == (
        "Runtime HE/LE blend: SIBYLL23E + FLUKA20251; "
        "transition=80 GeV, 10-90 width=0.4 decades"
    )


def test_blend_description_uses_g_formatting_for_a_zero_width():
    # `:g` on the floats, so a hard switch reads "width=0", not "width=0.0",
    # and a half-decade transition keeps its decimals.
    w = he_le_weight([8.0, 80.0, 800.0], 126.5, 0.0, dtype=None)
    description = blend_yields(
        interaction_index({}),
        interaction_index({}),
        "HE",
        "LE",
        w,
        126.5,
        0.0,
    )["description"]
    assert description == (
        "Runtime HE/LE blend: HE + LE; transition=126.5 GeV, 10-90 width=0 decades"
    )


def test_weight_broadcasts_along_the_last_axis_of_a_matrix():
    # `w_he[np.newaxis, :]` against an (n_e, n_e) channel: the weight varies
    # with the parent (column) index and is constant down the child rows.
    w = he_le_weight(transition_window(), 80.0, 0.4, dtype=None)
    channel = ((2212, 0), (211, 0))
    mixed = blend_yields(
        interaction_index({channel: np.zeros((3, 3))}),
        interaction_index({channel: np.ones((3, 3))}),
        "HE",
        "LE",
        w,
        80.0,
        0.4,
    )
    blended = mixed["index_d"][channel]
    assert blended.shape == (3, 3)
    assert np.allclose(blended, np.array([[0.9, 0.5, 0.1]] * 3))


def test_weight_broadcasts_along_the_last_axis_of_a_hankel_tensor():
    # Same weight against an (n_k, n_e, n_e) 2D channel: (1, n_e) broadcasts as
    # (1, 1, n_e), so it still varies along the parent axis and is applied
    # identically to every kappa mode. A rewrite that indexed the first axis
    # would pass the matrix test above and fail here.
    w = he_le_weight(transition_window(), 80.0, 0.4, dtype=None)
    channel = ((2212, 0), (211, 0))
    le_tensor = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 2.0)])
    mixed = blend_yields(
        interaction_index({channel: np.zeros((2, 3, 3))}),
        interaction_index({channel: le_tensor}),
        "HE",
        "LE",
        w,
        80.0,
        0.4,
    )
    blended = mixed["index_d"][channel]
    assert blended.shape == (2, 3, 3)
    assert np.allclose(blended[0], np.array([[0.9, 0.5, 0.1]] * 3))
    assert np.allclose(blended[1], np.array([[1.8, 1.0, 0.2]] * 3))
