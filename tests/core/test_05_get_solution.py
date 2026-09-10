"""get_solution surface: grid_idx, error paths, prefixes, return_as, helicities.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest
from pytest import approx

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_get_solution_grid_idx(mceq_sib21):
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.solve()

    with pytest.raises(Exception):
        mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=0)

    mceq_sib21.solve([0, 1])
    sol0 = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=0)
    sol1 = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=1)
    sol_last = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=100)

    assert np.allclose(sol0, 0)
    assert np.all(sol1 != 0)
    assert np.allclose(sol_last, sol1)


def test_get_solution_wrong_particle_name(mceq_sib21, monkeypatch):
    from MCEq import config

    # The production default is False, and conftest does not snapshot this one:
    # leaving it True turns every later missing-particle lookup into a raise.
    monkeypatch.setattr(config, "excpt_on_missing_particle", True)
    mceq_sib21.solve()
    with pytest.raises(Exception):
        mceq_sib21.get_solution("proton", mag=0, integrate=True, grid_idx=0)


def test_get_solution_prefixes(mceq_sib21):
    mceq_sib21.solve()
    total = mceq_sib21.get_solution("total_mu+", mag=0, integrate=True)
    conv = mceq_sib21.get_solution("conv_mu+", mag=0, integrate=True)
    prompt = mceq_sib21.get_solution("pr_mu+", mag=0, integrate=True)

    assert np.allclose(total, conv + prompt)


@pytest.mark.parametrize(
    "return_as",
    ["total energy", "kinetic energy", "total momentum", "none"],
)
@pytest.mark.parametrize("integrate", [True, False])
def test_get_solution_return_as(mceq_sib21, return_as, integrate):
    mceq_sib21.solve()

    if return_as == "none":
        with pytest.raises(Exception):
            result = mceq_sib21.get_solution(
                "mu+", return_as=return_as, integrate=integrate
            )
    else:
        result = mceq_sib21.get_solution(
            "mu+", return_as=return_as, integrate=integrate
        )
        if isinstance(result, tuple):
            xgrid, values = result
            assert len(xgrid) == len(values)
            assert np.all(np.isfinite(values))
        else:
            assert np.all(np.isfinite(result))


def test_get_solution_dont_sum_helicities(mceq_sib21):
    mceq_sib21.solve()

    # Get solution with summed helicities (default)
    solution_summed = mceq_sib21.get_solution("e-", dont_sum_helicities=False)
    solution_left_do = mceq_sib21.get_solution("e-_l", dont_sum_helicities=False)
    solution_right_do = mceq_sib21.get_solution("e-_r", dont_sum_helicities=False)

    # Get individual helicity states
    solution_left = mceq_sib21.get_solution("e-_l", dont_sum_helicities=True)
    solution_right = mceq_sib21.get_solution("e-_r", dont_sum_helicities=True)
    solution_0 = mceq_sib21.get_solution("e-", dont_sum_helicities=True)

    # Manual sum should equal the summed solution
    manual_sum = solution_left + solution_right + solution_0

    assert solution_left == approx(solution_left_do)
    assert solution_right == approx(solution_right_do)
    assert solution_summed == approx(manual_sum)


def test_solve_from_integration_path(mceq_sib21):
    # Normal solve
    mceq_sib21.solve()
    solution_normal = np.copy(mceq_sib21._solution)

    # Get the integration path that was used
    nsteps, dX, rho_inv, grid_idcs = mceq_sib21.integration_path

    # Solve using the same integration path
    mceq_sib21.solve_from_integration_path(nsteps, dX, rho_inv, grid_idcs)
    solution_from_path = mceq_sib21._solution

    # Should give identical results
    assert solution_normal == approx(solution_from_path)
