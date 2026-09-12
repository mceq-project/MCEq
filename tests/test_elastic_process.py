"""Independent elastic gain/removal, recoil, mode dependence and step bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from MCEq.data.elastic import ElasticTables
from MCEq.operators.matrix_builder import MatrixBuilder
from MCEq.operators.stiffness import elastic_rate
from MCEq.species.constants import ATOMIC_MASS_UNIT_G


@pytest.mark.parametrize("n_k", [1, 3])
def test_separate_operator_preserves_production_and_loss_split(n_k):
    b = object.__new__(MatrixBuilder)
    p = SimpleNamespace(mceqidx=0, A_target=14.5431)
    b._pman = {(2112, 0): p}
    b._grid = SimpleNamespace(dtype=None)

    # dim and dim_states use particle-manager attributes.
    class PM(dict):
        pass

    b._pman = PM(b._pman)
    b._pman.dim = 2
    b.is_2d = n_k > 1
    b.n_k = n_k
    b.max_lint = 0.0
    shape = (n_k, 2, 2) if b.is_2d else (2, 2)
    old = np.full(shape, 0.013)
    band = np.eye(2) * 0.2
    b.C_blocks = {(0, 0): old.copy() + band, (1, 0): np.ones(shape)}
    b._preband_blocks = {(0, 0): old.copy()}
    rate = np.array([0.01, 0.02])
    sigma = rate * p.A_target * ATOMIC_MASS_UNIT_G / 1e-27
    # Width ratio 10 converts .25 of the upper-bin particles to a
    # differential-density gain 2.5 in the lower bin.
    kernel = np.array([[1.0, 2.5], [0.0, 0.75]])
    if b.is_2d:
        kernel = np.stack([kernel, kernel * 0.7, kernel * -0.1])
    b.elastic = ElasticTables(
        {(2112, 0): sigma}, {((2112, 0), (2112, 0)): kernel * sigma}
    )
    b._add_elastic()
    expected = (kernel - np.eye(2)) * rate
    np.testing.assert_allclose(b.C_blocks[(0, 0)], old + band + expected)
    np.testing.assert_array_equal(b.C_blocks[(1, 0)], np.ones(shape))
    np.testing.assert_allclose(b._preband_blocks[(0, 0)], old + expected)
    zero = expected[0] if b.is_2d else expected
    np.testing.assert_allclose(np.array([1.0, 10.0]) @ zero, 0.0, atol=1e-16)
    assert elastic_rate(b.elastic_blocks) > 0
    assert elastic_rate(b.elastic_blocks, [1.0, 2.0]) == pytest.approx(
        2 * elastic_rate(b.elastic_blocks)
    )


def test_identity_elastic_has_no_step_penalty():
    assert elastic_rate({(0, 0): np.zeros((3, 2, 2))}, [1.0, 5.0]) == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        elastic_rate({(0, 0): np.array([[np.nan]])})


@pytest.mark.parametrize("is_2d", [False, True])
def test_reader_model_weight_and_coverage(tmp_path, is_2d):
    import h5py
    from scipy.sparse import block_diag, csr_matrix

    from MCEq.data.elastic import load_elastic
    from MCEq.data.hdf5_backend import HDF5Backend
    from MCEq.data.hdf5_store import HDF5Store

    path = tmp_path / "elastic.h5"
    nk = 2 if is_2d else 1
    kernel = np.array([[1.0, 2.0], [0.0, 0.8]])
    matrices = [kernel] if not is_2d else [kernel, kernel * 0.5]
    packed = block_diag([csr_matrix(m) for m in matrices], format="csr")
    with h5py.File(path, "w") as f:
        g = f.create_group("elastic_scattering/air/FLUKA20251")
        g.attrs.update(
            schema_version=1,
            process="elastic",
            cross_section_units="mb",
            energy_kind="kinetic",
            normalization="differential-per-collision",
        )
        g["cross_sections"] = np.array([[10.0], [20.0]])
        g["coverage"] = np.ones((2, 1), dtype=bool)
        g["projectiles"] = np.array([[2112, 0]])
        d = g.create_dataset("yields", data=np.vstack([packed.data, packed.indices]))
        d.attrs["tuple_idcs"] = np.array([[2112, 0, 2112, 0]])
        d.attrs["len_data"] = np.array([3])
        g["yields_indptrs"] = packed.indptr[None, :]
    b = object.__new__(HDF5Backend)
    b._had = HDF5Store(path)
    b.medium = "air"
    b.low_energy_model = "FLUKA20251"
    b._em_standalone = False
    b._e_grid_full = np.array([1.0, 10.0])
    b._energy_grid = SimpleNamespace(d=2)
    b._cuts = slice(0, 2)
    b.min_idx = 0
    b.max_idx = 2
    b.dim_full = 2 * nk
    b.is_2d = is_2d
    b.n_k = nk
    b._grid = SimpleNamespace(dtype=None)
    b._physics = SimpleNamespace(
        filters={"disabled_particles": []},
        assume_nucleon_interactions_for_exotics=False,
    )
    b._he_le_weight = lambda: np.array([0.0, 0.75])
    r = load_elastic(b, "SIBYLL23E")
    np.testing.assert_allclose(r.cross_sections[(2112, 0)], [10.0, 5.0])
    expected = np.array(matrices) if is_2d else kernel
    np.testing.assert_allclose(
        r.production[((2112, 0), (2112, 0))], expected * np.array([10.0, 5.0])
    )
    assert load_elastic(b, "FLUKA20251").cross_sections[(2112, 0)][1] == 20.0
    with h5py.File(path, "r+") as f:
        f["elastic_scattering/air/FLUKA20251/coverage"][0, 0] = False
    with pytest.raises(ValueError, match="coverage missing"):
        load_elastic(b, "FLUKA20251")
    b._he_le_weight = lambda: np.ones(2)
    assert np.all(load_elastic(b, "SIBYLL23E").cross_sections[(2112, 0)] == 0)
    b.low_energy_model = None
    assert not load_elastic(b, "SIBYLL23E").particles


def test_neutral_kaon_conversion_survives_zero_mode():
    from scipy.linalg import expm

    class PM(dict):
        dim = 1

    b = object.__new__(MatrixBuilder)
    b._pman = PM(
        {
            (130, 0): SimpleNamespace(mceqidx=0, A_target=14.5431),
            (310, 0): SimpleNamespace(mceqidx=1, A_target=14.5431),
        }
    )
    b._grid = SimpleNamespace(dtype=None)
    b.is_2d = False
    b.n_k = 1
    b.max_lint = 0.0
    b.C_blocks = {}
    b._preband_blocks = {}
    r = 0.02
    sigma = np.array([r * 14.5431 * ATOMIC_MASS_UNIT_G / 1e-27])
    b.elastic = ElasticTables(
        {p: sigma for p in b._pman},
        {(p, c): sigma[None, :] * 0.5 for p in b._pman for c in b._pman},
    )
    b._add_elastic()
    matrix = np.array([[b.C_blocks[(i, j)][0, 0] for j in range(2)] for i in range(2)])
    state = expm(matrix * 20.0) @ np.array([1.0, 0.0])
    np.testing.assert_allclose(
        state, [0.5 * (1 + np.exp(-r * 20)), 0.5 * (1 - np.exp(-r * 20))]
    )
    assert state[1] > 0
    np.testing.assert_allclose(matrix.sum(axis=0), 0.0, atol=1e-16)


def test_elastic_parent_selection_preserves_conversion_daughters():
    kl, ks, neutron = (130, 0), (310, 0), (2112, 0)
    table = ElasticTables(
        {p: np.ones(1) for p in (kl, ks, neutron)},
        {(kl, ks): np.ones((1, 1)), (neutron, neutron): np.ones((1, 1))},
    )
    restricted = table.restrict_parents([kl])
    assert restricted.particles == [kl, ks]
    assert set(restricted.cross_sections) == {kl}
    assert set(restricted.production) == {(kl, ks)}
    assert len(table.cross_sections) == 3
