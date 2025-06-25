import numpy as np
import pytest


def test_mrs_charm(mceq_small):
    from MCEq.charm_models import MRS_charm

    e_grid = mceq_small.e_grid
    cms = mceq_small._int_cs
    model = MRS_charm(e_grid, cms)

    # Test sigma_cc
    sigma = model.sigma_cc(e_grid)
    assert sigma.shape == e_grid.shape
    assert np.all(sigma > 0)

    # Test dsig_dx
    x = np.linspace(0.05, 0.6, 10)
    E = 1e7
    dx_vec = model.dsig_dx(x, E)
    dx_scalar = model.dsig_dx(0.1, E)
    assert dx_vec.shape == x.shape
    assert dx_scalar.shape == ()
    assert np.all(dx_vec >= 0)

    # Test D_dist and LambdaC_dist
    D = model.D_dist(x, E, 421)
    L = model.LambdaC_dist(x, E)
    assert D.shape == x.shape
    assert L.shape == x.shape
    assert np.all(D >= 0)
    assert np.all(L >= 0)

    # Check yield matrix for valid pair
    mat = model.get_yield_matrix(2212, 421)
    assert mat.shape == (len(e_grid), len(e_grid))
    assert np.any(mat > 0)

    # Check yield matrix for invalid projectile
    zero_mat = model.get_yield_matrix(11, 421)
    assert np.allclose(zero_mat, 0)

    # Check yield matrix for invalid secondary
    zero_mat2 = model.get_yield_matrix(2212, 13)
    assert np.allclose(zero_mat2, 0)

    # Check sign condition for LambdaC
    zero_mat3 = model.get_yield_matrix(2212, -4122)
    assert np.allclose(zero_mat3, 0)

    # Test all combinations
    for proj in model.allowed_proj:
        for sec in model.allowed_sec:
            _ = model.get_yield_matrix(proj, sec)


@pytest.mark.xfail(reason="Fix issue #79")
def test_whr_charm(mceq_small):
    from MCEq.charm_models import WHR_charm

    e_grid = mceq_small.e_grid
    cms = mceq_small._int_cs
    model = WHR_charm(e_grid, cms)
    assert isinstance(model, WHR_charm)
