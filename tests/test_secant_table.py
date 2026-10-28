"""1D sec(theta) elongation table: lookup, segmented solve, guards."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from MCEq import config
from MCEq.operators.secant_table import SecantTable, elongation_factors


def _write_table(path, mceq, zeniths, values, X=(1.0, 100.0, 1000.0)):
    """Table with ``values[i](species) -> s`` constant in E and X at zenith i."""
    names = [p.name for p in mceq.pman.all_particles if p.mceqidx >= 0]
    with h5py.File(path, "w") as f:
        f["e_grid"] = mceq.e_grid
        f["zeniths"] = np.asarray(zeniths, float)
        f["species"] = np.array(names, dtype="S")
        for i, value in enumerate(values):
            s = np.array([np.full(len(mceq.e_grid), value(n)) for n in names])
            f[f"X_{i}"] = np.asarray(X, float)
            f[f"s_{i}"] = np.broadcast_to(s, (len(X), *s.shape))
    return str(path)


def _slice(mceq, name):
    j, ne = mceq.pman.pname2pref[name].mceqidx, len(mceq.e_grid)
    return slice(j * ne, (j + 1) * ne)


def test_unit_table_is_plain_solve(mceq_sib21, tmp_path):
    mceq_sib21.solve(int_grid=[50.0, 300.0, 800.0])
    plain, plain_grid = mceq_sib21._solution.copy(), mceq_sib21.grid_sol.copy()
    config.secant_1d_table = _write_table(
        tmp_path / "ones.h5", mceq_sib21, [0, 90], [lambda n: 1.0] * 2
    )
    mceq_sib21.solve(int_grid=[50.0, 300.0, 800.0])
    np.testing.assert_allclose(mceq_sib21._solution, plain, rtol=1e-10, atol=0)
    np.testing.assert_allclose(mceq_sib21.grid_sol, plain_grid, rtol=1e-10, atol=0)


def test_zenith_and_depth_interpolation(mceq_sib21, tmp_path):
    fname = _write_table(
        tmp_path / "t.h5", mceq_sib21, [0, 60], [lambda n: 1.0, lambda n: 2.0]
    )
    table = SecantTable(fname)
    args = (mceq_sib21.pman, mceq_sib21.dim_states, mceq_sib21.e_grid)
    sl = _slice(mceq_sib21, "pi+")
    theta = np.degrees(np.arccos(0.75))  # halfway in cos(theta)
    for X in (0.1, 30.0, 1e5):  # inside and clamped at both ends
        np.testing.assert_allclose(table.factors(*args, theta, X)[sl], 1.5)
    np.testing.assert_allclose(table.factors(*args, 80.0, 30.0)[sl], 2.0)


def test_elongation_attenuates_parent(mceq_sib21, tmp_path):
    mceq_sib21.solve()
    plain = mceq_sib21.get_solution("pi+", mag=0)
    config.secant_1d_table = _write_table(
        tmp_path / "pi.h5",
        mceq_sib21,
        [0, 90],
        [lambda n: 2.0 if n == "pi+" else 1.0] * 2,
    )
    mceq_sib21.solve()
    long = mceq_sib21.get_solution("pi+", mag=0)
    filled = plain > 1e-12 * plain.max()  # skip round-off bins
    assert np.all(long[filled] < 0.5 * plain[filled])


def test_batch_refuses_table(mceq_sib21, tmp_path):
    config.secant_1d_table = _write_table(
        tmp_path / "ones.h5", mceq_sib21, [0, 90], [lambda n: 1.0] * 2
    )
    with pytest.raises(NotImplementedError):
        mceq_sib21.solve_batch(phi0=np.tile(mceq_sib21._phi0[:, None], (1, 2)))


def test_collimated_state_is_not_elongated():
    k_grid = np.concatenate([np.arange(48), np.geomspace(50, 600, 12).round()])
    e_grid = np.geomspace(0.1, 1e3, 5)
    run = SimpleNamespace(
        _mceq_db=SimpleNamespace(k_grid=k_grid, n_k=len(k_grid)),
        grid_sol=np.ones((2, len(k_grid) * 3 * len(e_grid))),
        dim_states=3 * len(e_grid),
        e_grid=e_grid,
    )
    np.testing.assert_allclose(elongation_factors(run), 1.0, atol=2e-3)
