"""1D sec(theta) elongation table: lookup, selection, the column-scaled step."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from MCEq import config
from MCEq.config import detect
from MCEq.operators import secant_table
from MCEq.operators.secant_table import SecantTable, elongation_factors, run_mode

KERNELS = ["numpy_etd2"] + [
    k
    for k, ok in (("mkl_etd2", detect.has_mkl), ("cuda_etd2", detect.has_cuda))
    if ok()
]


def _write_table(path, mceq, zeniths, values, X=(1.0, 100.0, 1000.0), **attrs):
    """Table with ``values[i](species) -> s`` at zenith i, constant in E and X,
    in both groups."""
    names = [p.name for p in mceq.pman.all_particles if p.mceqidx >= 0]
    with h5py.File(path, "w") as f:
        f["e_grid"] = mceq.e_grid
        f["species"] = np.array(names, dtype="S")
        f.attrs["low_energy_model"] = str(mceq._mceq_db.low_energy_model)
        f.attrs.update(attrs)
        for mode in secant_table.MODES:
            g = f.create_group(mode)
            g["zeniths"] = np.asarray(zeniths, float)
            for i, value in enumerate(values):
                s = np.array([np.full(len(mceq.e_grid), value(n)) for n in names])
                g[f"X_{i}"] = np.asarray(X, float)
                g[f"s_{i}"] = np.broadcast_to(s, (len(X), *s.shape))
    return str(path)


def _ones(path, mceq):
    return _write_table(path, mceq, [0, 90], [lambda n: 1.0] * 2)


def _pi_only(path, mceq):
    return _write_table(path, mceq, [0, 90], [lambda n: 2.0 if n == "pi+" else 1.0] * 2)


def test_unit_table_is_plain_solve(mceq_sib21, tmp_path):
    mceq_sib21.solve(int_grid=[50.0, 300.0, 800.0])
    plain, plain_grid = mceq_sib21._solution.copy(), mceq_sib21.grid_sol.copy()
    config.secant_1d_table = _ones(tmp_path / "ones.h5", mceq_sib21)
    mceq_sib21.solve(int_grid=[50.0, 300.0, 800.0])
    np.testing.assert_allclose(mceq_sib21._solution, plain, rtol=1e-12, atol=0)
    np.testing.assert_allclose(mceq_sib21.grid_sol, plain_grid, rtol=1e-12, atol=0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_column_scale_is_scaled_operator(mceq_sib21, kernel):
    """The step with ``col_scale`` equals the plain step on the operator
    with scaled columns, on every backend."""
    config.kernel_config = kernel
    s = 1 + np.random.default_rng(1).random(mceq_sib21.dim_states)
    mceq_sib21._calculate_integration_path(None, "X")
    nsteps, dX, ri, _ = mceq_sib21.integration_path
    half = nsteps // 2
    S, starts = np.stack([s, s[::-1]]), np.array([0, half])
    phi0 = mceq_sib21._phi0.copy()
    scaled, _ = mceq_sib21._run_etd2(nsteps, dX, ri, phi0, [], col_scale=(S, starts))

    int_m, dec_m = mceq_sib21.int_m, mceq_sib21.dec_m
    phi = phi0
    try:
        for a, b, row in ((0, half, S[0]), (half, nsteps, S[1])):
            D = sp.diags(row)
            mceq_sib21.int_m = sp.csr_matrix(int_m @ D)
            mceq_sib21.dec_m = sp.csr_matrix(dec_m @ D)
            phi, _ = mceq_sib21._run_etd2(b - a, dX[a:b], ri[a:b], phi, [])
    finally:
        mceq_sib21.int_m, mceq_sib21.dec_m = int_m, dec_m
    peak = np.abs(phi).max()
    np.testing.assert_allclose(scaled, phi, rtol=1e-9, atol=1e-12 * peak)


def test_interpolation_and_mode(mceq_sib21, tmp_path):
    table = SecantTable(
        _write_table(
            tmp_path / "t.h5", mceq_sib21, [0, 60], [lambda n: 1.0, lambda n: 2.0]
        )
    )
    j = mceq_sib21.pman.pname2pref["pi+"].mceqidx
    ne = len(mceq_sib21.e_grid)
    args = (mceq_sib21.pman, mceq_sib21.dim_states, mceq_sib21.e_grid)
    theta = np.degrees(np.arccos(0.75))  # halfway in cos(theta)
    s = table.factors("inclusive", *args, theta, np.array([0.1, 30.0, 1e5]))
    np.testing.assert_allclose(s[:, j * ne : (j + 1) * ne], 1.5)
    s = table.factors("shower", *args, 80.0, np.array([30.0]))
    np.testing.assert_allclose(s[:, j * ne : (j + 1) * ne], 2.0)

    assert run_mode(mceq_sib21) == "inclusive"
    run = SimpleNamespace(_initial_condition=[("single", 1e3, None, 2212)])
    assert run_mode(run) == "shower"


def test_elongation_attenuates_parent(mceq_sib21, tmp_path):
    mceq_sib21.solve()
    plain = mceq_sib21.get_solution("pi+", mag=0)
    config.secant_1d_table = _pi_only(tmp_path / "pi.h5", mceq_sib21)
    mceq_sib21.solve()
    long = mceq_sib21.get_solution("pi+", mag=0)
    filled = plain > 1e-12 * plain.max()  # skip round-off bins
    assert np.all(long[filled] < 0.5 * plain[filled])


def test_auto_gating(mceq_sib21, monkeypatch):
    config.secant_1d_table = "auto"
    config.e_min = 5.0
    assert secant_table.table_for(mceq_sib21) is None
    config.e_min = 0.1
    config.secant_theta_transport = False
    assert secant_table.table_for(mceq_sib21) is None
    config.secant_theta_transport = "auto"
    monkeypatch.setattr(secant_table, "_manifest_table", lambda m, le: None)
    with pytest.warns(UserWarning, match="No 1D sec"):
        assert secant_table.table_for(mceq_sib21) is None


def test_model_mismatch_and_handover(mceq_sib21, tmp_path, monkeypatch):
    config.secant_1d_table = _write_table(
        tmp_path / "other.h5",
        mceq_sib21,
        [0, 90],
        [lambda n: 1.0] * 2,
        low_energy_model="OTHERMODEL",
    )
    with pytest.raises(ValueError, match="low-energy model"):
        secant_table.table_for(mceq_sib21)
    config.secant_1d_table = _ones(tmp_path / "ones.h5", mceq_sib21)
    monkeypatch.setattr(mceq_sib21._mceq_db, "he_le_transition", 30.0)
    with pytest.warns(UserWarning, match="he_le_transition"):
        assert secant_table.table_for(mceq_sib21) is not None


def test_shared_batch_applies_table(mceq_sib21, tmp_path):
    config.secant_1d_table = _pi_only(tmp_path / "pi.h5", mceq_sib21)
    mceq_sib21.solve()
    res = mceq_sib21.solve_batch(phi0=mceq_sib21._phi0[:, None])
    np.testing.assert_allclose(
        np.asarray(res.sol)[:, 0], mceq_sib21._solution, rtol=1e-10
    )


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
