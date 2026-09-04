"""`HDF5Backend._gen_db_dictionary` decode branches, against synthetic files.

The generator is ``tests/data/make_hdf5_fixtures.py`` and the files come from
the session-scoped ``hdf5_fixture_dbs`` fixture in ``conftest.py``. They exist
because two decode branches are unreachable from any shipped database: every
``tuple_idcs`` on disk is 4 columns wide (asserted below on the reduced DB), so
the 2-column branch and the ``"Failed decoding parent-child relation."`` raise
are only reachable from a file built for them. That is the gap the plan's
Phase 4 entry names ("the 2-column ``tuple_idcs`` path is otherwise never
exercised by real data").

Everything here records what the code does **today**, defects included --
``test_width_two_file_loses_the_dedicated_antiproton_channels`` is a pinned
defect, not a wish.

The probes read the fixtures through a real ``HDF5Backend`` with injected
config groups (``paths``/``grid``/``physics``), so no test here touches the
process-wide ``MCEq.config`` state or builds an ``MCEqRun``. Decay packs are
preferred over interaction packs where the equivalence machinery would be
noise: ``decay_db`` passes ``equivalences={}``.
"""

import importlib.util
import pathlib
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq.data import HDF5Backend

_SPEC = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

#: Index into ``fixtures.DECAY_CHANNELS``. Channel 4 is ``mu+ -> e+``, which
#: the default ``disabled_particles`` skips; channel 5 is ``mu+ -> nu_e``,
#: the one immediately behind it in the flat pack.
SKIPPED_CH, NEXT_CH = 4, 5


def backend(path, *, disabled=(), medium=fixtures.MEDIUM):
    """An ``HDF5Backend`` on one fixture file, with injected config groups.

    ``e_min``/``e_max`` are ``None`` so the energy cut is the whole grid and a
    decoded channel matrix is comparable to ``fixtures.channel_block`` as it
    was written.
    """
    paths = SimpleNamespace(
        data_dir=path.parent,
        mceq_db_fname=path.name,
        # No EM database exists for these fixtures; enable_em is False, so the
        # backend never looks for it.
        em_db_fname="no_such_em_db.h5",
    )
    grid = SimpleNamespace(e_min=None, e_max=None, dtype=None, em_standalone_grid=False)
    physics = SimpleNamespace(
        filters={
            "disabled_particles": list(disabled),
            "forced_int_cs": None,
            "replace_meson_cross_sections_with": None,
        },
        assume_nucleon_interactions_for_exotics=True,
        enable_em=False,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium=medium,
        muon_helicity_dependence=False,
    )
    return HDF5Backend(medium=medium, paths=paths, grid=grid, physics=physics)


def decode_decays(path, **kwargs):
    return backend(path, **kwargs).decay_db("unpolarized")


# --------------------------------------------------------------------------
# What the shipped databases actually contain
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "db_fname", ["mceq_db_v140reduced_compact.h5", "mceq_db_v2_fluka2d_rc7.h5"]
)
def test_shipped_databases_are_four_column(db_fname):
    """Justifies the fixtures: no real file reaches the 2-column branch.

    Both the reduced 1D database and the production 2D one store
    ``tuple_idcs`` as ``(n_channels, 4)`` -- ``(parent, helicity, child,
    helicity)``.
    """
    from MCEq import config

    path = pathlib.Path(config.data_dir) / db_fname
    if not path.exists():
        pytest.skip(f"{db_fname} not available; symlink it into src/MCEq/data/")

    with h5py.File(path, "r") as db:
        widths = set()

        def collect(_name, obj):
            if isinstance(obj, h5py.Dataset) and "tuple_idcs" in obj.attrs:
                widths.add(obj.attrs["tuple_idcs"].shape[1])

        db.visititems(collect)

    assert widths == {4}, f"{db_fname} carries tuple_idcs of widths {sorted(widths)}"


def test_fixture_kappa_grid_is_a_subsample_of_the_production_grid():
    """Guards the constant the 2D fixture is built from against DB drift.

    ``make_hdf5_fixtures.K_GRID_PRODUCTION`` is a hand-copy of the rc7
    ``/common`` ``k_grid``; the 8-mode fixture grid is an evenly spaced
    subsample of it. A head truncation instead loads but then fails
    ``build_secant_kernel_ops`` with an error blaming ``theta_cap_deg`` -- see
    the generator docstring.
    """
    from MCEq import config

    path = pathlib.Path(config.data_dir) / "mceq_db_v2_fluka2d_rc7.h5"
    if not path.exists():
        pytest.skip("mceq_db_v2_fluka2d_rc7.h5 not available")

    with h5py.File(path, "r") as db:
        production = np.asarray(db["common"].attrs["k_grid"])

    assert np.array_equal(production, fixtures.K_GRID_PRODUCTION)
    assert set(fixtures.K_GRID) <= set(production)
    assert fixtures.K_GRID[0] == production[0]
    assert fixtures.K_GRID[-1] == production[-1]
    # Not a head truncation: the fixture spans the whole kappa range.
    assert not np.array_equal(fixtures.K_GRID, production[: fixtures.N_K])


def test_fixture_pack_matches_the_shipped_pack_layout(hdf5_fixture_dbs):
    """The synthetic file is built to the layout the reader reads, not to prose.

    Same attribute set, same shapes-in-terms-of-``dim_full``, same dtype split
    (float64 pack carrying integer CSR indices in row 1, integer indptrs) as
    ``decays/unpolarized`` in the reduced database.
    """
    from MCEq import config

    real = pathlib.Path(config.data_dir) / "mceq_db_v140reduced_compact.h5"
    if not real.exists():
        pytest.skip("mceq_db_v140reduced_compact.h5 not available")

    rules = (
        "attrs",
        "packed_length_rule",
        "indptr_width_rule",
        "indptr_integral",
        "common_grid_attrs",
    )

    def describe(path):
        with h5py.File(path, "r") as db:
            pack = db["decays"]["unpolarized"]
            indptrs = db["decays"]["unpolarized_indptrs"]
            e_dim = int(db["common"].attrs["e_dim"])
            n_k = int(db["common"].attrs.get("k_dim", 1))
            len_data = pack.attrs["len_data"]
            return {
                "attrs": {"len_data", "tuple_idcs"} <= set(pack.attrs),
                "pack_dtype": pack.dtype,
                "pack_rows": pack.shape[0],
                "packed_length_rule": pack.shape[1] == int(len_data.sum()) * n_k,
                "indptr_width_rule": indptrs.shape[1] == e_dim * n_k + 1,
                "indptr_integral": np.issubdtype(indptrs.dtype, np.integer),
                "tuple_width": pack.attrs["tuple_idcs"].shape[1],
                "common_grid_attrs": {"e_grid", "e_bins", "widths", "e_dim"}
                <= set(db["common"].attrs),
            }

    reference = describe(real)
    # Anchor the boolean entries before comparing. Half of ``describe`` is
    # predicates, and ``False == False`` compares equal: a mistyped rule would
    # be false on both files and the comparison below would pin nothing.
    assert [reference[rule] for rule in rules] == [True] * len(rules), reference

    assert describe(hdf5_fixture_dbs["one_d_width4"]) == reference


# --------------------------------------------------------------------------
# The two tuple widths and the raise
# --------------------------------------------------------------------------


def test_four_column_tuples_decode_to_helicity_resolved_keys(hdf5_fixture_dbs):
    """The width-4 branch: ``tuple(tup[:2]), tuple(tup[2:])``.

    The PDG half is cast with ``int()``; the helicity half is not, so it stays
    the ``np.int64`` that h5py handed back. Pinned as measured -- the mixed key
    type is what a downstream ``repr`` or a key-ordering golden sees.
    """
    index = decode_decays(hdf5_fixture_dbs["one_d_width4"])
    parent, child = ((211, 0), (-13, 0))

    assert (parent, child) in index["index_d"]
    key = next(k for k in index["index_d"] if k == (parent, child))
    assert [type(component) for component in key[0]] == [int, np.int64]
    assert [type(component) for component in key[1]] == [int, np.int64]


def test_two_column_tuples_decode_to_zero_helicity_keys(hdf5_fixture_dbs):
    """The width-2 branch: ``(tup[0], 0), (tup[1], 0)`` -- no shipped file hits it.

    The helicity here is the Python literal ``0``, so the key components are
    both plain ``int``; on the width-4 file the second component is
    ``np.int64`` instead.
    """
    index = decode_decays(hdf5_fixture_dbs["one_d_width2"])
    parent, child = ((211, 0), (-13, 0))

    assert (parent, child) in index["index_d"]
    key = next(k for k in index["index_d"] if k == (parent, child))
    assert [type(component) for component in key[0]] == [int, int]
    assert [type(component) for component in key[1]] == [int, int]


def test_the_two_widths_produce_interchangeable_keys(hdf5_fixture_dbs):
    """``(2212, np.int64(0))`` and ``(2212, 0)`` hash and compare equal.

    So the type difference above is invisible to every dict lookup; it is
    pinned for what a ``repr``-based golden or a strict type check would see,
    not because it changes a decode.
    """
    width4 = decode_decays(hdf5_fixture_dbs["one_d_width4"])["index_d"]
    width2 = decode_decays(hdf5_fixture_dbs["one_d_width2"])["index_d"]

    assert set(width4) == set(width2)
    for key in width4:
        assert hash(key) == hash(next(k for k in width2 if k == key))


def test_an_unsupported_tuple_width_raises(hdf5_fixture_dbs):
    """The ``else`` arm of the width dispatch, reachable only from a built file.

    A 3-column ``tuple_idcs`` is neither the helicity-resolved nor the bare
    layout, so the decode gives up on the first channel.
    """
    with pytest.raises(Exception, match="Failed decoding parent-child relation."):
        decode_decays(hdf5_fixture_dbs["one_d_width3"])


# --------------------------------------------------------------------------
# The decoded values and the flat cursor
# --------------------------------------------------------------------------


def test_one_dimensional_channel_decodes_to_the_stored_block(hdf5_fixture_dbs):
    """End-to-end: CSR triple -> energy cut -> dense, against the written block."""
    index = decode_decays(hdf5_fixture_dbs["one_d_width4"])
    parent, child = fixtures.DECAY_CHANNELS[NEXT_CH]
    decoded = index["index_d"][((parent, 0), (child, 0))]

    assert decoded.shape == (fixtures.N_E, fixtures.N_E)
    assert np.array_equal(decoded, fixtures.channel_block(NEXT_CH))


def test_two_dimensional_channel_decodes_to_one_block_per_mode(hdf5_fixture_dbs):
    """2D channels come back as ``(n_k, n_e, n_e)``, mode block by mode block.

    The stored matrix is block-diagonal in kappa; the decode slices the
    ``ik``-th diagonal block out of the ``(dim_full, dim_full)`` CSR. Blocks
    differ between modes in the fixture, so a decode that collapsed them would
    not pass.
    """
    index = backend(hdf5_fixture_dbs["two_d_width4"]).decay_db("polarized")
    parent, child = fixtures.DECAY_CHANNELS[NEXT_CH]
    decoded = index["index_d"][((parent, 0), (child, 0))]

    assert decoded.shape == (fixtures.N_K, fixtures.N_E, fixtures.N_E)
    for mode in range(fixtures.N_K):
        assert np.array_equal(decoded[mode], fixtures.channel_block(NEXT_CH, mode))
    assert not np.array_equal(decoded[0], decoded[1])


def test_a_skipped_channel_still_advances_the_flat_cursor(hdf5_fixture_dbs):
    """A channel dropped by ``disabled_particles`` advances ``read_idx`` anyway.

    The pack is one flat run of CSR data; ``read_idx`` is the only thing that
    says where the next channel starts. If the ``continue`` skipped the
    advance, the channel behind a disabled one would silently decode the
    disabled one's numbers. Contract: ``read_idx += len_data[i]`` in 1D.
    """
    path = hdf5_fixture_dbs["one_d_width4"]
    skipped_parent, skipped_child = fixtures.DECAY_CHANNELS[SKIPPED_CH]
    parent, child = fixtures.DECAY_CHANNELS[NEXT_CH]

    # abs() matching in the backend, hence 11 covers e+ as well (ledger B2).
    partial = decode_decays(path, disabled=[11])["index_d"]
    full = decode_decays(path)["index_d"]

    assert ((skipped_parent, 0), (skipped_child, 0)) in full
    assert ((skipped_parent, 0), (skipped_child, 0)) not in partial
    assert np.array_equal(
        partial[((parent, 0), (child, 0))],
        fixtures.channel_block(NEXT_CH),
    )


def test_a_skipped_2d_channel_advances_the_cursor_by_n_k_times_len_data(
    hdf5_fixture_dbs,
):
    """Same contract in 2D, where one channel occupies ``n_k * len_data[i]``.

    ``expand_len`` is ``n_k`` on a 2D database, so a skipped channel must
    advance the cursor over all of its Hankel-mode blocks. Getting this wrong
    by a factor ``n_k`` would still decode to finite matrices.
    """
    path = hdf5_fixture_dbs["two_d_width4"]
    skipped_parent, skipped_child = fixtures.DECAY_CHANNELS[SKIPPED_CH]
    parent, child = fixtures.DECAY_CHANNELS[NEXT_CH]

    partial = backend(path, disabled=[11]).decay_db("polarized")["index_d"]
    full = backend(path).decay_db("polarized")["index_d"]

    # Without this pair the test says nothing about the skip path: if the
    # filter stopped dropping the channel, the block below would decode off
    # the ordinary cursor and still match.
    assert ((skipped_parent, 0), (skipped_child, 0)) in full
    assert ((skipped_parent, 0), (skipped_child, 0)) not in partial

    decoded = partial[((parent, 0), (child, 0))]
    assert decoded.shape == (fixtures.N_K, fixtures.N_E, fixtures.N_E)
    for mode in range(fixtures.N_K):
        assert np.array_equal(decoded[mode], fixtures.channel_block(NEXT_CH, mode))


# --------------------------------------------------------------------------
# Pinned defect: available_parents on a 2-column file
# --------------------------------------------------------------------------


def test_width_four_keeps_a_dedicated_antiproton_channel(hdf5_fixture_dbs):
    """Contrast for the pin below: on a width-4 file the guard works.

    ``available_parents`` is ``sorted(set(tuple_idcs[:, :2]))``, which on a
    width-4 file is exactly the ``(pdg, helicity)`` parents. SIBYLL21's
    equivalence table maps ``-2212 -> 2212``, so with
    ``assume_nucleon_interactions_for_exotics`` the proton's matrices would be
    copied onto the antiproton -- except that ``(-2212, 0)`` is in
    ``available_parents``, so the copy is skipped and the antiproton keeps its
    own, dedicated channels.
    """
    index = backend(hdf5_fixture_dbs["one_d_width4"]).interaction_db(fixtures.MODEL)
    index_d = index["index_d"]

    # Positive control, on this same decode: 3112 also maps to 2212 under
    # SIBYLL21 but is only a *child* in the fixture, so it is not in
    # ``available_parents`` and its channels are copies of the proton's. Without
    # it, a decode that skipped the equivalence block entirely would satisfy
    # every assertion below.
    assert np.array_equal(
        index_d[((3112, 0), (211, 0))], index_d[((2212, 0), (211, 0))]
    )

    assert not np.array_equal(
        index_d[((-2212, 0), (211, 0))], index_d[((2212, 0), (211, 0))]
    )
    assert ((-2212, 0), (2212, 0)) not in index_d
    assert index["relations"][(-2212, 0)] == [
        (-2212, 0),
        (211, 0),
        (-211, 0),
    ]


def test_width_two_file_loses_the_dedicated_antiproton_channels(hdf5_fixture_dbs):
    """PINNED DEFECT (no ledger id -- not in plan section 9 as of c81d182).

    ``available_parents`` is built from ``tuple_idcs[:, :2]``, which is
    ``(pdg, helicity)`` only on a width-4 file. On a width-2 file those two
    columns are ``(parent, child)``, so the list holds channel pairs, and the
    ``if eqv_parent in available_parents`` guard can never fire: it would need
    a literal ``(-2212, 0)`` channel, i.e. a parent decaying to PDG 0.

    Consequence, pinned here as measured: the equivalence copy runs even
    though the antiproton has dedicated simulated channels in the same file.
    ``(-2212, 0) -> (211, 0)`` is overwritten with the proton's matrix, a
    spurious ``(-2212, 0) -> (2212, 0)`` channel appears, and
    ``relations[(-2212, 0)]`` is *replaced* by the proton's child list, so the
    antiproton is recorded as producing a proton rather than an antiproton.

    A fix would slice ``tuple_idcs[:, :2]`` only on the width-4 layout (and
    pair the PDG with helicity 0 on width-2); this test then flips.
    """
    index = backend(hdf5_fixture_dbs["one_d_width2"]).interaction_db(fixtures.MODEL)
    index_d = index["index_d"]

    assert np.array_equal(
        index_d[((-2212, 0), (211, 0))], index_d[((2212, 0), (211, 0))]
    )
    assert ((-2212, 0), (2212, 0)) in index_d
    assert index["relations"][(-2212, 0)] == [
        (2212, 0),
        (211, 0),
        (-211, 0),
    ]


# --------------------------------------------------------------------------
# Object identity in the equivalence machinery (data/equivalences.py)
# --------------------------------------------------------------------------


def test_fluka_and_dpmjet_share_one_equivalence_table():
    """``equivalences["FLUKA"] is equivalences["DPMJET"]`` -- one dict, not two.

    ``data/equivalences.py`` builds the FLUKA entry by aliasing the DPMJET one
    (``equivalences["FLUKA"] = equivalences["DPMJET"]``), so an edit to either
    family is an edit to both. A rewrite that copies -- ``dict(...)``, a
    comprehension, a per-family literal -- keeps every existing test and every
    golden green while changing that, which is why the identity is asserted
    here rather than left to the values agreeing.

    The mutation half is what makes this a real identity test: two dicts with
    equal contents pass an ``==`` comparison forever.
    """
    from MCEq.data import equivalences

    assert equivalences["FLUKA"] is equivalences["DPMJET"]

    probe = "__identity_probe__"
    equivalences["DPMJET"][probe] = probe
    try:
        assert equivalences["FLUKA"].get(probe) == probe
    finally:
        del equivalences["DPMJET"][probe]
    assert probe not in equivalences["FLUKA"]


def test_the_equivalences_package_attribute_is_the_dict_not_the_submodule():
    """``MCEq.data.equivalences`` is the table; the module is in ``sys.modules``.

    The name is both a submodule of ``MCEq.data`` and the dict that submodule
    defines. The import system binds the *module* onto the package while
    loading it, and ``data/__init__.py``'s ``from MCEq.data.equivalences import
    equivalences`` rebinds the name to the dict afterwards. That order is what
    keeps ``MCEq.data.equivalences`` the object it has always been
    (``tests/test_phase4_surface.py`` pins the name; this pins the resolution),
    and it is fragile: dropping the dict from that import list, or reaching the
    module through the package attribute, silently swaps the two.
    """
    import sys

    import MCEq.data

    assert isinstance(MCEq.data.equivalences, dict)
    module = sys.modules["MCEq.data.equivalences"]
    assert isinstance(module, ModuleType)
    assert module.equivalences is MCEq.data.equivalences


def test_equivalence_injection_aliases_rather_than_copies(hdf5_fixture_dbs):
    """The stand-in shares the matrix *object* and the ``relations`` *list*.

    ``apply_equivalences`` does ``index_d[(eqv_parent, child)] =
    index_d[(parent, child)]`` and ``relations[eqv_parent] =
    relations[parent]``. Both are aliases, and the second one is observable
    beyond the call: the alias is installed while decoding the proton's *first*
    channel, and the proton's two later channels append to the very list 3112
    now holds, so 3112 ends up with the proton's complete child list rather
    than with the one child that was known when the alias was made.

    3112 is the one equivalence replacement that fires on this fixture (see
    ``make_hdf5_fixtures.INTERACTION_CHANNELS``): SIBYLL21 maps it to 2212, it
    appears as a child so it is in ``model_particles``, and it is not a parent
    so the ``available_parents`` guard does not skip it.

    ``test_width_four_keeps_a_dedicated_antiproton_channel`` above asserts the
    same copy with ``array_equal``, which a deep copy would satisfy too.
    """
    index = backend(hdf5_fixture_dbs["one_d_width4"]).interaction_db(fixtures.MODEL)
    index_d, relations = index["index_d"], index["relations"]
    parent, stand_in = (2212, 0), (3112, 0)

    children = [(2212, 0), (211, 0), (-211, 0)]
    for child in children:
        assert index_d[(stand_in, child)] is index_d[(parent, child)], (
            f"{child} was copied instead of aliased"
        )

    assert relations[stand_in] is relations[parent]
    assert relations[stand_in] == children


def test_equivalence_aliasing_on_the_shipped_reduced_database():
    """The alias counts on real data, measured: 40 matrix groups, 2 lists.

    The fixture test above reaches one stand-in with three channels. This one
    records the scale on the reduced 1D database with SIBYLL21 and the default
    ``disabled_particles = [11, -11]``: 40 groups of channel matrices that
    share one object (60 keys beyond the first of each group), and exactly two
    shared ``relations`` lists -- 310 sharing 130's, and both 3122 and -3122
    sharing 2112's.

    The numbers are the point: a rewrite that copies instead of aliasing
    changes them to 0 and 0 while every value stays equal.
    """
    from collections import defaultdict

    from MCEq import config

    path = pathlib.Path(config.data_dir) / "mceq_db_v140reduced_compact.h5"
    if not path.exists():
        pytest.skip("reduced database not available; symlink it into src/MCEq/data/")

    index = backend(path, disabled=[11, -11]).interaction_db("SIBYLL2.1")

    def groups(mapping):
        by_id = defaultdict(list)
        for key, value in mapping.items():
            by_id[id(value)].append(key)
        return [keys for keys in by_id.values() if len(keys) > 1]

    matrix_groups = groups(index["index_d"])
    assert len(matrix_groups) == 40
    assert sum(len(keys) - 1 for keys in matrix_groups) == 60

    relation_groups = groups(index["relations"])
    assert sorted(sorted(int(pdg) for pdg, _ in keys) for keys in relation_groups) == [
        [-3122, 2112, 3122],
        [130, 310],
    ]
