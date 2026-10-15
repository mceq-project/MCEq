"""The precomputed screened Coulomb rates a 2D database may carry.

``material_scattering/<medium>/screened_coulomb_rates`` (written by
mceq-maintenance-tools) replaces the ~l_max^2 evaluation of
``screened_coulomb_rate_legendre`` when the inputs it records match the run.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq.operators import scattering

E = np.logspace(-1, 2, 7)
ELL = np.array([0.0, 1.0, 2.0, 5.0, 12.0])
COMP = np.array([[7, 14.0, 0.78], [8, 16.0, 0.21], [18, 40.0, 0.01]])


def _layout(rates, finite=True, mass=scattering.MUON_MASS, ell=ELL, e=E):
    attrs = {"finite_nuclei": finite, "muon_mass": mass}
    return SimpleNamespace(
        scattering_rate_table=lambda: (rates, ell, e, attrs),
        scattering_composition=lambda: COMP,
    )


def test_stored_table_is_used_when_the_inputs_match():
    stored = np.arange(ELL.size * E.size, dtype=float).reshape(ELL.size, E.size)
    got = scattering.mode_rates("screened-coulomb", E, ELL, _layout(stored))
    assert np.array_equal(got, stored)


@pytest.mark.parametrize(
    "change",
    [{"finite": False}, {"mass": 0.1057}, {"ell": ELL[:-1]}, {"e": E * 1.001}],
)
def test_stored_table_is_ignored_when_an_input_differs(change):
    layout = _layout(np.full((ELL.size, E.size), -1.0), **change)
    got = scattering.mode_rates("screened-coulomb", E, ELL, layout)
    want = scattering.screened_coulomb_rate_legendre(E, ELL, COMP)
    assert np.array_equal(got, want)


def test_computed_rates_are_memoised():
    scattering._screened_coulomb_cached.cache_clear()
    layout = SimpleNamespace(scattering_composition=lambda: COMP)
    a = scattering.mode_rates("screened-coulomb", E, ELL, layout, finite_nuclei=False)
    a[:] = 0.0  # the caller's copy, not the cached array
    b = scattering.mode_rates("screened-coulomb", E, ELL, layout, finite_nuclei=False)
    info = scattering._screened_coulomb_cached.cache_info()
    assert info.hits == 1 and info.misses == 1
    assert np.array_equal(
        b, scattering.screened_coulomb_rate_legendre(E, ELL, COMP, finite_nuclei=False)
    )


def test_backend_cuts_the_stored_table_to_the_energy_window(tmp_path):
    from MCEq.data.hdf5_backend import HDF5Backend
    from MCEq.data.hdf5_store import HDF5Store
    from MCEq.misc import energy_grid

    e_full = np.logspace(-1, 2, 13)
    k_grid = np.array([0, 1, 2, 486])
    table = np.arange(k_grid.size * e_full.size, dtype=float).reshape(k_grid.size, -1)
    path = tmp_path / "table.h5"
    with h5py.File(path, "w") as f:
        f.create_group("common").attrs.update({"e_grid": e_full, "k_grid": k_grid})
        f["material_scattering/air/composition"] = COMP
        ds = f.create_dataset(
            f"material_scattering/air/{scattering.RATE_TABLE}", data=table
        )
        ds.attrs.update({"finite_nuclei": True, "muon_mass": scattering.MUON_MASS})
    backend = HDF5Backend.__new__(HDF5Backend)
    backend._had, backend.medium = HDF5Store(path), "air"
    backend._cuts = slice(3, 9)
    backend._energy_grid = energy_grid(e_full[3:9], None, None, 6)
    reads = []
    read = backend._had.read_optional_table
    backend._had.read_optional_table = lambda *a: reads.append(a) or read(*a)
    rates, l_grid, e_grid, attrs = backend.scattering_rate_table()
    assert backend.scattering_rate_table()[0] is rates  # kept, not reread
    assert len(reads) == 1
    assert np.array_equal(rates, table[:, 3:9])
    assert np.array_equal(l_grid, k_grid) and np.array_equal(e_grid, e_full[3:9])
    assert attrs["finite_nuclei"]
    got = scattering.mode_rates("screened-coulomb", e_full[3:9], k_grid, backend)
    assert np.array_equal(got, table[:, 3:9])

    backend.medium = "water"
    assert backend.scattering_rate_table() is None
