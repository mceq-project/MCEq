"""The materialized read seam of ``MCEq.data``: ``hdf5_store``.

Every store method must hand back numpy arrays or plain containers and no
live ``h5py`` node -- the point of the seam is that nothing a caller holds
can outlive the open. Return types are pinned here, not only values; values
are pinned against a direct ``h5py`` read of the same fixture (the store may
not transform what it reads) and, for :func:`unpack_channel`, against the
analytic reference of ``tests/data/make_hdf5_fixtures.channel_block``.

Whole-pipeline decode equality is pinned elsewhere:
``tests/test_data_hdf5_decode.py`` drives a real backend over these
fixtures and the eight golden sections pin the shipped databases.
"""

import importlib.util
import pathlib

import h5py
import numpy as np
import pytest

from MCEq.data.hdf5_store import ChannelPack, HDF5Store, unpack_channel

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def test_return_types_are_numpy_and_plain(hdf5_fixture_dbs):
    """No ``h5py`` object escapes any store method."""
    path = hdf5_fixture_dbs["one_d_width4"]
    store = HDF5Store(path)
    ca = store.attrs("common")
    assert type(ca) is dict
    assert isinstance(ca["e_grid"], np.ndarray)
    assert np.array_equal(ca["e_grid"], fixtures.E_GRID)
    ms = store.members("/")
    assert type(ms) is list and all(type(m) is str for m in ms)
    assert "common" in ms
    assert store.members("common") == []
    assert store.has("common") is True
    assert store.has("no_such_group") is False
    table, attrs = store.read_table("cross_sections/air/SIBYLL21")
    assert isinstance(table, np.ndarray) and type(attrs) is dict
    assert np.array_equal(table, fixtures.cross_section_table(0))
    group = store.read_group_datasets("continuous_losses/air/ionization")
    assert type(group) is dict
    assert all(isinstance(v, np.ndarray) for v in group.values())


def test_read_channel_pack_matches_a_direct_h5py_read(hdf5_fixture_dbs):
    """The store returns exactly what a direct read of the same file gives."""
    path = hdf5_fixture_dbs["one_d_width4"]
    store = HDF5Store(path)
    pack = store.read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    assert isinstance(pack, ChannelPack)
    with h5py.File(path, "r") as f:
        dset = f["hadronic_interactions/air/SIBYLL21"]
        sibling = f["hadronic_interactions/air/SIBYLL21_indptrs"]
        assert np.array_equal(pack.data, dset[:, :])
        assert pack.data.dtype == dset.dtype
        assert np.array_equal(pack.indptrs, sibling[:])
        assert np.array_equal(pack.len_data, dset.attrs["len_data"])
        assert np.array_equal(pack.tuple_idcs, dset.attrs["tuple_idcs"])
        assert pack.description == dset.attrs["description"]
    assert not any("h5py" in type(v).__module__ for v in pack)


def test_read_channel_pack_description_none_when_absent(hdf5_fixture_dbs):
    """The width-3 fixture is the variant written without ``description``."""
    path = hdf5_fixture_dbs["one_d_width3"]
    pack = HDF5Store(path).read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    assert pack.description is None


def test_unpack_channel_1d_matches_the_reference_block(hdf5_fixture_dbs):
    path = hdf5_fixture_dbs["one_d_width4"]
    pack = HDF5Store(path).read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    start = 0
    for ich, n in enumerate(pack.len_data):
        length = int(n)
        got = unpack_channel(
            pack.data,
            pack.indptrs[ich, :],
            start,
            length,
            dim_full=fixtures.N_E,
            is_2d=False,
            n_k=1,
            min_idx=0,
            max_idx=fixtures.N_E,
            cuts=slice(None),
        )
        assert got.shape == (fixtures.N_E, fixtures.N_E)
        assert np.array_equal(got, fixtures.channel_block(ich))
        start += length


def test_unpack_channel_2d_yields_the_kappa_stack(hdf5_fixture_dbs):
    path = hdf5_fixture_dbs["two_d_width4"]
    pack = HDF5Store(path).read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    start = 0
    for ich, n in enumerate(pack.len_data):
        length = fixtures.N_K * int(n)
        got = unpack_channel(
            pack.data,
            pack.indptrs[ich, :],
            start,
            length,
            dim_full=fixtures.N_E * fixtures.N_K,
            is_2d=True,
            n_k=fixtures.N_K,
            min_idx=0,
            max_idx=fixtures.N_E,
            cuts=slice(None),
        )
        assert got.shape == (fixtures.N_K, fixtures.N_E, fixtures.N_E)
        expected = np.stack(
            [fixtures.channel_block(ich, ik) for ik in range(fixtures.N_K)]
        )
        assert np.array_equal(got, expected)
        start += length


def test_unpack_channel_2d_with_one_mode_keeps_the_leading_axis(
    hdf5_fixture_dbs,
):
    """``is_2d`` is explicit, never inferred from ``n_k``.

    A 2D database with ``n_k == 1`` decodes to ``(1, n_e, n_e)``, not
    ``(n_e, n_e)``: the leading axis comes from the layout the backend
    detected on the ``common`` group, not from a count.
    """
    path = hdf5_fixture_dbs["one_d_width4"]
    pack = HDF5Store(path).read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    got = unpack_channel(
        pack.data,
        pack.indptrs[0, :],
        0,
        int(pack.len_data[0]),
        dim_full=fixtures.N_E,
        is_2d=True,
        n_k=1,
        min_idx=0,
        max_idx=fixtures.N_E,
        cuts=slice(None),
    )
    assert got.shape == (1, fixtures.N_E, fixtures.N_E)
    assert np.array_equal(got[0], fixtures.channel_block(0))


def test_energy_cut_shrinks_the_decode(tmp_path):
    """A backend-configured cut reaches ``unpack_channel`` via the caller.

    Built on the 1D fixture: decode channel 0 with ``[2, 18)`` on both
    axes and compare against the reference block sliced the same way.
    """
    path = tmp_path / "cut.h5"
    fixtures.build_database(path, n_k=1, tuple_width=4)
    pack = HDF5Store(path).read_channel_pack("hadronic_interactions/air", "SIBYLL21")
    got = unpack_channel(
        pack.data,
        pack.indptrs[0, :],
        0,
        int(pack.len_data[0]),
        dim_full=fixtures.N_E,
        is_2d=False,
        n_k=1,
        min_idx=2,
        max_idx=18,
        cuts=slice(2, 18),
    )
    assert got.shape == (16, 16)
    assert np.array_equal(got, fixtures.channel_block(0)[2:18, 2:18])


def test_construction_never_touches_the_file(tmp_path):
    """``HDF5Store`` on a missing path constructs fine; only reads raise.

    The backend constructs a store for the EM file even when
    ``enable_em`` is off and that file may not exist.
    """
    missing = tmp_path / "definitely_not_there.h5"
    store = HDF5Store(missing)
    assert store.fname == missing
    with pytest.raises(OSError):
        store.has("common")
