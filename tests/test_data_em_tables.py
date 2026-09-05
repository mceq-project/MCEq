"""Selection policy of :mod:`MCEq.data.em_tables`, on hand-built files.

The rho-stack selection needs a file the shared fixtures do not have (a
``rho_grid`` with slice subgroups), so the probes write small EM-shaped
files with ``h5py`` directly; the pack payloads come from
``tests/data/make_hdf5_fixtures`` loaded by file path, the way
``tests/test_data_hdf5_decode.py`` does it. ``HDF5Store`` itself is pinned
in ``tests/test_data_hdf5_store.py``; here the store is the given, and the
object under test is which group a selection returns.
"""

import importlib.util
import pathlib
from collections import defaultdict

import h5py
import numpy as np

from MCEq.data import em_tables
from MCEq.data.hdf5_store import HDF5Store

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

#: Three densities, a decade apart: the log10-nearest rule picks the middle
#: one for values between its neighbours' geometric means.
RHO_GRID = [1e-3, 1e-2, 1e-1]


def em_file_with_rho_stack(path):
    """``electromagnetic/air`` with a ``rho_grid`` and three slices.

    Slice ``i`` stores the shared channel list rotated by ``i``, so the
    decoded ``(11, 0) -> (11, 0)`` matrix differs per slice: a selection
    that opened the wrong group is visible in the values, not just the path.
    """
    ch = fixtures.EM_CHANNELS
    with h5py.File(path, "w") as db:
        group = db.create_group("electromagnetic/air")
        group.create_dataset("rho_grid", data=np.array(RHO_GRID))
        for idx in range(len(RHO_GRID)):
            rotated = ch[idx:] + ch[:idx]
            slice_group = group.create_group(f"rho_{idx:02d}")
            fixtures._write_pack(slice_group, "emca_mats", rotated, 1, 2, None)
            cs = slice_group.create_dataset("cs", data=fixtures.em_cs_table(1.0 + idx))
            cs.attrs["projectiles"] = np.array(fixtures.EM_CS_PARENTS, np.int64)
    return path


def plain_em_file(path, medium="air"):
    """One medium, pack + cs at the top of its EM group, no rho stack."""
    fixtures.build_em_database(path, {medium: (fixtures.EM_CHANNELS, 1.0)})
    return path


def test_group_paths():
    assert em_tables.em_group_path("air") == "electromagnetic/air"
    assert em_tables.em_group_path("water", 3) == "electromagnetic/water/rho_03"
    assert em_tables.em_group_path("air", 10) == "electromagnetic/air/rho_10"


def test_medium_for_em():
    assert em_tables.medium_for_em("ice") == "water"
    assert em_tables.medium_for_em("air") == "air"
    assert em_tables.medium_for_em("water") == "water"
    assert em_tables.medium_for_em("argon") == "argon"


def test_rho_slice_nearest_in_log10(tmp_path):
    path = em_file_with_rho_stack(tmp_path / "stack.h5")
    store = HDF5Store(path)
    pick = em_tables.select_em_rho_slice
    assert pick(store, "air", 1e-2) == 1
    assert pick(store, "air", 1e-3) == 0
    assert pick(store, "air", 9.9e-2) == 2
    # the geometric mean 10**-2.5 ~ 3.162e-3 is the boundary between slice
    # 0 and slice 1; just below it the nearer edge is still slice 0.
    assert pick(store, "air", 3.16e-3) == 0
    assert pick(store, "air", 3.17e-3) == 1


def test_rho_slice_guards(tmp_path):
    store = HDF5Store(em_file_with_rho_stack(tmp_path / "stack.h5"))
    pick = em_tables.select_em_rho_slice
    assert pick(store, "air", None) is None
    plain = HDF5Store(plain_em_file(tmp_path / "plain.h5"))
    assert pick(plain, "air", 1e-2) is None


def test_rho_slice_is_air_only_even_with_a_water_stack(tmp_path):
    """The `medium != "air"` guard, observable: a water stack is never used."""
    path = tmp_path / "water.h5"
    with h5py.File(path, "w") as db:
        group = db.create_group("electromagnetic/water")
        group.create_dataset("rho_grid", data=np.array(RHO_GRID))
    store = HDF5Store(path)
    assert em_tables.select_em_rho_slice(store, "water", 1e-2) is None


def test_rho_slice_empty_grid(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as db:
        db.create_group("electromagnetic/air").create_dataset(
            "rho_grid", data=np.array([])
        )
    assert em_tables.select_em_rho_slice(HDF5Store(path), "air", 1e-2) is None


def test_path_helpers_round_trip_the_stack(tmp_path):
    path = em_file_with_rho_stack(tmp_path / "stack.h5")
    store = HDF5Store(path)
    assert em_tables.emca_group_path(store, "air", "t", 1e-2) == (
        "electromagnetic/air/rho_01"
    )
    assert em_tables.em_cs_group_path(store, "air", "t", 1e-2) == (
        "electromagnetic/air/rho_01"
    )
    plain = HDF5Store(plain_em_file(tmp_path / "plain.h5"))
    assert em_tables.emca_group_path(plain, "air", "t", 1e-2) == ("electromagnetic/air")
    assert em_tables.em_cs_group_path(plain, "air", "t", None) == (
        "electromagnetic/air"
    )


def test_stack_selection_decodes_the_selected_slice(tmp_path):
    """Not just the path: the matrices come from the group the pick names."""
    path = em_file_with_rho_stack(tmp_path / "stack.h5")
    store = HDF5Store(path)
    group = em_tables.emca_group_path(store, "air", "t", 9.9e-2)
    pack = store.read_channel_pack(group, "emca_mats")
    rot = fixtures.EM_CHANNELS[2:] + fixtures.EM_CHANNELS[:2]
    pos = rot.index((11, 11))
    # the slot under (11,11) belongs to THIS slice's ordering, and the cs
    # scale says which slice the group is; both would break on a wrong pick
    assert tuple(pack.tuple_idcs[pos]) == (11, 11)
    cs, _ = store.read_table(f"{group}/cs")
    assert np.array_equal(cs, fixtures.em_cs_table(3.0))  # slice 2 scale
    from MCEq.data.hdf5_store import unpack_channel

    got = unpack_channel(
        pack.data,
        pack.indptrs[pos, :],
        int(pack.len_data[:pos].sum()),
        int(pack.len_data[pos]),
        dim_full=fixtures.N_E,
        is_2d=False,
        n_k=1,
        min_idx=0,
        max_idx=fixtures.N_E,
        cuts=slice(None),
    )
    assert np.array_equal(got, fixtures.channel_block(pos))


def test_helicity_duplicate():
    # mutable payloads: `is` below then means a real alias, not string interning
    index_d = {
        ((11, 0), (11, 0)): np.array([1.0]),
        ((11, 0), (22, 0)): np.array([2.0]),
        ((-11, 0), (-11, 0)): np.array([3.0]),
        ((-11, 0), (22, 0)): np.array([4.0]),
        ((13, 0), (13, 0)): np.array([5.0]),
        ((13, 0), (22, 0)): np.array([6.0]),
        ((-13, 0), (-13, 0)): np.array([7.0]),
        ((-13, 0), (22, 0)): np.array([8.0]),
    }
    index = {
        "parents": [(11, 0), (-11, 0), (13, 0), (-13, 0)],
        "particles": [],
        "relations": {},
        "index_d": dict(index_d),
        "description": None,
    }
    out = em_tables.helicity_duplicate(index)
    assert out is index
    assert len(out["index_d"]) == 24
    # aliases: the polarised entries share the object, not a copy
    assert out["index_d"][((11, 1), (22, 0))] is out["index_d"][((11, 0), (22, 0))]
    assert (
        out["index_d"][((-13, -1), (-13, -1))] is out["index_d"][((-13, 0), (-13, 0))]
    )
    for pid in (11, -11, 13, -13):
        for h in (-1, 1):
            assert (pid, h) in out["parents"]
    assert isinstance(out["relations"], defaultdict)
    assert set(out["relations"]) == {
        (pid, h) for pid in (11, -11, 13, -13) for h in (0, -1, 1)
    }
    assert out["relations"][(11, 1)] == [(11, 1), (22, 0)]
    assert len(out["particles"]) == 48
    assert (22, 0) in out["particles"]


def test_module_reads_no_config():
    """The seam rule of section 13.1, checked from the source text."""
    import ast

    tree = ast.parse(pathlib.Path(em_tables.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "MCEq.config"
            assert not (node.module or "").startswith("MCEq.config")
        if isinstance(node, ast.Import):
            assert all(a.name != "MCEq.config" for a in node.names)
    names = {
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for a in node.names
    }
    assert "config" not in names
