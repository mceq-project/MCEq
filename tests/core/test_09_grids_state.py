"""Derived grids (ptot/etot/xgrid) and state-vector checkpoint/restore.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import crflux.models as pm
import numpy as np
import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)

import MCEq.geometry.density_profiles as dprof


def test_initial_condition_holds_no_reference_to_the_run(mceq_sib21):
    """A bound method in the stored initial condition would close a reference
    cycle through the run and keep instances alive (PR #163)."""
    for entry in mceq_sib21._initial_condition:
        assert isinstance(entry[0], str)
        assert not any(getattr(x, "__self__", None) is mceq_sib21 for x in entry)


def test_ptot_grid(mceq_sib21):
    # Test without bins
    ptot_centers = mceq_sib21.ptot_grid("mu+", return_bins=False)

    assert len(ptot_centers) == len(mceq_sib21.e_grid)
    assert np.all(ptot_centers > 0)
    assert np.all(np.isfinite(ptot_centers))

    # Test with bins
    ptot_bins, ptot_centers_with_bins = mceq_sib21.ptot_grid("mu+", return_bins=True)

    assert len(ptot_bins) == len(mceq_sib21.e_bins)
    assert len(ptot_centers_with_bins) == len(mceq_sib21.e_grid)
    assert np.allclose(ptot_centers, ptot_centers_with_bins)

    # Check that centers are geometric mean of bins
    expected_centers = np.sqrt(ptot_bins[1:] * ptot_bins[:-1])
    assert np.allclose(ptot_centers_with_bins, expected_centers)


def test_etot_grid(mceq_sib21):
    # Test without bins
    etot_centers = mceq_sib21.etot_grid("mu+", return_bins=False)

    assert len(etot_centers) == len(mceq_sib21.e_grid)
    assert np.all(etot_centers > 0)
    assert np.all(np.isfinite(etot_centers))

    # Test with bins
    etot_bins, etot_centers_with_bins = mceq_sib21.etot_grid("mu+", return_bins=True)

    assert len(etot_bins) == len(mceq_sib21.e_bins)
    assert len(etot_centers_with_bins) == len(mceq_sib21.e_grid)
    assert np.allclose(etot_centers, etot_centers_with_bins)

    # Check that bins are kinetic + mass
    mu_mass = mceq_sib21.pman["mu+"].mass
    expected_bins = mceq_sib21.e_bins + mu_mass
    assert np.allclose(etot_bins, expected_bins)


@pytest.mark.parametrize(
    ["return_as", "expected_method"],
    [
        ["kinetic energy", lambda mceq_sib21: (mceq_sib21.e_bins, mceq_sib21.e_grid)],
        [
            "total energy",
            lambda mceq_sib21: mceq_sib21.etot_grid("mu+", return_bins=True),
        ],
        [
            "total momentum",
            lambda mceq_sib21: mceq_sib21.ptot_grid("mu+", return_bins=True),
        ],
    ],
)
@pytest.mark.parametrize("return_bins", [False, True])
def test_xgrid(mceq_sib21, return_as, expected_method, return_bins):
    result = mceq_sib21.xgrid("mu+", return_as, return_bins=return_bins)

    if return_bins:
        bins, centers = result
        expected_bins, expected_centers = expected_method(mceq_sib21)
        assert np.allclose(bins, expected_bins)
        assert np.allclose(centers, expected_centers)
    else:
        expected_bins, expected_centers = expected_method(mceq_sib21)
        assert np.allclose(result, expected_centers)


def test_xgrid_invalid_return_as(mceq_sib21):
    """Test xgrid raises exception for invalid return_as argument."""
    with pytest.raises(Exception, match="Unknown grid type"):
        mceq_sib21.xgrid("mu+", "invalid_type", return_bins=False)


def test_get_set_state_vector_checkpoint_restore(mceq_sib21):
    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.set_density_model(dprof.CorsikaAtmosphere("BK_USStd"))

    # First solve
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.solve()
    order1, state1 = mceq_sib21._get_state_vector()
    solution1 = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Change something and solve again
    mceq_sib21.set_theta_deg(30.0)
    mceq_sib21.solve()
    order2, state2 = mceq_sib21._get_state_vector()
    solution2 = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Solutions should be different
    assert not np.allclose(solution1, solution2, atol=1e-10)
    assert not np.allclose(state1, state2, atol=1e-14)

    # Restore first state directly (without solving)
    mceq_sib21._solution = np.copy(state1)
    solution1_prime = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Should match original solution
    assert np.allclose(solution1, solution1_prime, atol=1e-10)


def test_int_grid_of_a_new_length_rebuilds_the_path(mceq_sib21):
    mceq_sib21.solve(int_grid=np.linspace(10.0, 900.0, 7))
    mceq_sib21.solve(int_grid=np.linspace(10.0, 900.0, 4))
    assert mceq_sib21.grid_sol.shape == (4, mceq_sib21.dim_states)
