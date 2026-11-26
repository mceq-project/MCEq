import numpy as np
import pytest

from MCEq.source import Source, bin_on_path


def _proton_source(mceq, x, col=0, energy=1e4):
    """A unit proton at ``energy`` GeV injected at depth ``x`` into column ``col``."""
    mceq.set_single_primary_particle(energy, pdg_id=2212)
    rows = np.flatnonzero(mceq._phi0)
    n = len(rows)
    return Source(np.full(n, col), np.full(n, x), rows, mceq._phi0[rows].copy())


@pytest.fixture(params=["etd2", "imex"])
def integrator(request, monkeypatch):
    from MCEq import config

    monkeypatch.setattr(config, "integrator", request.param)
    monkeypatch.setattr(config, "secant_1d_table", None)
    return request.param


def test_bin_on_path_coalesces_and_drops():
    src = Source(
        col=np.array([0, 0, 1, 0]),
        x=np.array([0.2, 0.4, 1.4, 9.0]),
        row=np.array([3, 3, 2, 3]),
        val=np.array([1.0, 2.0, 4.0, 8.0]),
    )
    starts, rows, cols, vals = bin_on_path(src, 0.0, np.ones(2))  # boundaries 0, 1, 2
    assert list(starts) == [0, 1, 2, 2]
    assert list(zip(rows, cols, vals)) == [(3, 0, 3.0), (2, 1, 4.0)]


def test_source_at_start_equals_initial_state(mceq_sib21, integrator):
    src = _proton_source(mceq_sib21, x=0.0)
    phi0 = np.zeros(mceq_sib21.dim_states)
    phi0[src.row] = src.val
    xg = np.array([100.0, 500.0])
    ref = mceq_sib21.solve_batch(phi0=phi0[:, None], int_grid=xg)
    got = mceq_sib21.solve_batch(
        phi0=np.zeros((mceq_sib21.dim_states, 1)), int_grid=xg, source=src
    )
    np.testing.assert_allclose(got.sol, ref.sol, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(got.grid_sol, ref.grid_sol, rtol=1e-12, atol=0.0)


def test_projected_snapshots_and_late_source(mceq_sib21, integrator):
    m = mceq_sib21
    src = _proton_source(m, x=300.0, col=1)
    xg = np.array([100.0, 290.0, 400.0, 800.0])
    phi0 = np.zeros((m.dim_states, 2))
    phi0[src.row, 0] = src.val
    full = m.solve_batch(phi0=phi0, int_grid=xg, source=src)
    P = np.random.default_rng(0).random((3, m.dim_states))
    proj = m.solve_batch(phi0=phi0, int_grid=xg, source=src, project=P)
    np.testing.assert_allclose(
        proj.grid_sol, np.einsum("od,gdk->gok", P, full.grid_sol), rtol=1e-12
    )
    assert np.all(full.grid_sol[:2, :, 1] == 0.0)  # nothing before the source depth
    assert np.any(full.grid_sol[2:, :, 1] != 0.0)
