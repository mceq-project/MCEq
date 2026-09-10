"""Solve paths: default, integration grid, path reuse, initial spectrum.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_solve_default(mceq_sib21):
    mceq_sib21.solve()
    sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None

    assert np.sum(mceq_sib21._phi0) != np.sum(mceq_sib21._solution)

    muons = mceq_sib21.get_solution("total_mu-", mag=0)

    assert np.sum(muons) != 0


def test_solve_skip_integration_path(mceq_sib21):
    mceq_sib21._calculate_integration_path(int_grid=None, grid_var="X")
    mceq_sib21.solve(skip_integration_path=True)
    sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_other_grid_var(mceq_sib21):
    with pytest.raises(NotImplementedError):
        mceq_sib21.solve(grid_var="Y")


@pytest.mark.parametrize("int_grid", [None, [0, 1]], ids=["no-grid", "with-grid"])
def test_solve_int_grid(mceq_sib21, int_grid):
    mceq_sib21.solve(int_grid)
    if int_grid is None:
        assert mceq_sib21.grid_sol.shape == (0,)
    else:
        assert mceq_sib21.grid_sol.shape == (len(int_grid), mceq_sib21.dim_states)


def test_integration_path_grid_idcs(mceq_sib21):
    int_grid = [0, 1]

    grid_var = "X"
    mceq_sib21._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq_sib21.integration_path
    grid_idcs = integration_path[-1]
    assert len(grid_idcs) == 2
    assert grid_idcs[0] == 0
    assert grid_idcs[0] < grid_idcs[1]


# Reference values recalibrated 2026-07 for the v2 release defaults
# (loss_stencil_method="expfit_low_upwind2", average_loss_operator=False,
# generic_losses_all_charged=True, m_u mass-unit fix in
# inverse_interaction_length). Enabling the charged-hadron stopping power
# lowers all muon/neutrino yields by <0.1% here.
@pytest.mark.parametrize(
    "pdg_id",
    [
        2212,
        2112,
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_set_initial_spectrum(mceq_sib21, pdg_id, append):
    cond = mceq_sib21._restore_initial_condition
    phi0_backup = mceq_sib21._phi0.copy()

    particle = mceq_sib21.pman[pdg_id]
    phi0 = mceq_sib21._phi0[particle.lidx : particle.uidx].copy()

    spectrum = np.ones_like(phi0) * 10
    mceq_sib21.set_initial_spectrum(spectrum, pdg_id, append)

    phi1 = mceq_sib21._phi0[particle.lidx : particle.uidx]

    if append:
        assert np.allclose(phi0, phi1 - spectrum)
    if not append:
        assert np.allclose(spectrum, phi1)

    mceq_sib21._phi0 = phi0_backup
    mceq_sib21._restore_initial_condition = cond
