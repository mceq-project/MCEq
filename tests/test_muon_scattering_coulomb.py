"""Screened Coulomb muon scattering: rate values, composition reading and
operator placement."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq.data.hdf5_backend import HDF5Backend
from MCEq.data.hdf5_store import HDF5Store
from MCEq.operators import scattering
from MCEq.operators.matrix_builder import MatrixBuilder

#: Air as in the mceq-EM generator that produced the reference values below.
AIR_INTEGER_A = [[7, 14.0, 0.7847874810295787], [8, 16.0, 0.210518910117893],
                 [18, 40.0, 0.004693608852528217]]  # fmt: skip
#: CODATA 2018 muon mass [GeV], used by that generator.
MUON_MASS = 0.1056583755

#: (kinetic energy [GeV], kappa [1/rad], rate [cm2/g]) from the generator.
REFERENCE = [
    (0.011220184543019636, 1, 0.006631083974020151),
    (0.011220184543019636, 43, 7.2701139234472745),
    (0.011220184543019636, 2000, 4817.608988250182),
    (11.220184543019636, 1, 3.750463643107822e-08),
    (11.220184543019636, 43, 5.158614620957443e-05),
    (11.220184543019636, 2000, 0.07237545250018884),
    (11220.18454301964, 1, 5.615368663764808e-14),
    (11220.18454301964, 43, 8.573567172665234e-11),
    (11220.18454301964, 2000, 1.4551770193293214e-07),
]


@pytest.mark.parametrize(("e_kin", "kappa", "expected"), REFERENCE)
def test_rate_matches_reference(e_kin, kappa, expected):
    rate = scattering.screened_coulomb_rate([e_kin], [kappa], AIR_INTEGER_A, MUON_MASS)
    np.testing.assert_allclose(rate[0, 0], expected, rtol=1e-9)


def test_rate_zero_mode_and_monotone():
    e_kin = np.geomspace(1e-2, 1e6, 30)
    kappa = np.concatenate([[0.0], np.geomspace(1e-3, 1e4, 40)])
    rate = scattering.screened_coulomb_rate(e_kin, kappa, AIR_INTEGER_A)
    assert rate.shape == (kappa.size, e_kin.size)
    assert np.all(rate[0] == 0)
    assert np.all(np.diff(rate, axis=0) > 0)
    assert np.all(np.diff(rate[1:], axis=1) < 0)


def _backend(path, medium="air"):
    backend = HDF5Backend.__new__(HDF5Backend)
    backend.had_fname = str(path)
    backend._had = HDF5Store(path)
    backend.medium = medium
    return backend


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "composition.h5"
    with h5py.File(path, "w") as f:
        f["material_scattering/air/composition"] = AIR_INTEGER_A
    return path


def test_composition_read_and_missing_medium(db):
    np.testing.assert_array_equal(_backend(db).scattering_composition(), AIR_INTEGER_A)
    with pytest.raises(ValueError, match="absent"):
        _backend(db, "water").scattering_composition()


@pytest.mark.parametrize("corrupt", ["shape", "fraction", "z", "nan"])
def test_reject_invalid_composition(tmp_path, corrupt):
    comp = np.array(AIR_INTEGER_A)
    if corrupt == "shape":
        comp = comp[:, :2]
    elif corrupt == "fraction":
        comp[0, 2] = 0.5
    elif corrupt == "z":
        comp[0, 0] = 7.5
    else:
        comp[1, 1] = np.nan
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f["material_scattering/air/composition"] = comp
    with pytest.raises(ValueError, match="Invalid"):
        _backend(path).scattering_composition()


def test_all_muon_charges_helicities_and_operator_placement(db):
    k_grid = np.array([0.0, 10.0])
    e_kin = np.array([1.0, 10.0])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 2, k_grid
    builder._layout = _backend(db)
    builder._physics = SimpleNamespace(
        muon_multiple_scattering=True, muon_scattering_model="screened-coulomb"
    )
    builder._grid = SimpleNamespace(dtype=np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=2)
    particles = {
        (pdg, hel): SimpleNamespace(lidx=2 * i, mceqidx=i, mass=MUON_MASS)
        for i, (pdg, hel) in enumerate(((p, h) for p in (13, -13) for h in (0, -1, 1)))
    }

    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states, pman.dim = particles, 12, 2
    builder._pman = pman
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    rate = scattering.screened_coulomb_rate(e_kin, k_grid, AIR_INTEGER_A, MUON_MASS)
    expected = np.zeros(24)
    expected[12:24] = np.tile(-rate[1], 6)
    np.testing.assert_allclose(matrix.diagonal(), expected, rtol=1e-15)
    assert matrix.nnz == 12


def test_unknown_model_raises(db):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.k_grid = True, np.array([0.0, 10.0])
    builder._physics = SimpleNamespace(
        muon_multiple_scattering=True, muon_scattering_model="moliere"
    )
    builder._energy_grid = SimpleNamespace(c=np.array([1.0, 10.0]))

    class Pman(dict):
        pass

    muon = SimpleNamespace(lidx=0, mceqidx=0, mass=MUON_MASS)
    builder._pman = Pman({(13, 0): muon})
    builder._pman.pdg2pref = {(13, 0): muon}
    with pytest.raises(ValueError, match="Unknown muon_scattering_model"):
        builder._muon_scattering_damping()


def test_scattering_model_invalidates_split_assembly():
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d = True
    builder._grid = SimpleNamespace(dtype=None)
    builder._physics = SimpleNamespace(
        muon_multiple_scattering=True, muon_scattering_model="gaussian"
    )
    old_key = builder._current_assembly_key()
    builder._physics.muon_scattering_model = "screened-coulomb"
    assert old_key != builder._current_assembly_key()
