import sys

import numpy as np
import pytest
from pytest import approx


if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_solve_default(mceq):
    mceq.solve()
    sol = mceq.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_skip_integration_path(mceq):
    mceq.solve(skip_integration_path=True)
    sol = mceq.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_other_grid_var(mceq):
    with pytest.raises(NotImplementedError):
        mceq.solve(grid_var="Y")


@pytest.mark.parametrize(
    ["int_grid", "grid_shape"],
    [[None, (0,)], [[0, 1], (2, 9922)]],
    ids=["no-grid", "with-grid"],
)
def test_solve_int_grid(mceq, int_grid, grid_shape):
    mceq.solve(int_grid)
    print(mceq.grid_sol.shape)
    assert mceq.grid_sol.shape == grid_shape


@pytest.mark.parametrize(
    ["leading_process", "lenX"],
    [
        ["decays", 594],
        pytest.param(
            "interactions", -1, marks=pytest.mark.xfail(reason="Fix issue #66")
        ),
        pytest.param("auto", -1, marks=pytest.mark.xfail(reason="Fix issue #66")),
    ],
)
def test_integration_path_leading_process(mceq, leading_process, lenX):
    """Fix this test by resolving issue #66"""
    from MCEq import config

    config.leading_process = leading_process
    int_grid = None
    grid_var = "X"
    mceq._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq.integration_path
    assert integration_path[0] == lenX


def test_integration_path_grid_idcs(mceq):
    int_grid = [0, 1]

    grid_var = "X"
    mceq._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq.integration_path
    grid_idcs = integration_path[-1]
    assert len(grid_idcs) == 2
    assert grid_idcs[0] == 0
    assert grid_idcs[0] < grid_idcs[1]


testdata_theta = [
    [0.0, 5.62504370e-3],
    [30.0, 4.20479234e-3],
    [60.0, 1.36630552e-3],
]

ids_theta = [f"{th[0]}" for th in testdata_theta]


@pytest.mark.parametrize(["theta", "nmu"], testdata_theta, ids=ids_theta)
def test_set_theta_deg(mceq, theta, nmu):
    mceq.set_theta_deg(theta)
    mceq.solve()
    nmu_sol = np.sum(
        mceq.get_solution("mu+", mag=0, integrate=True)
        + mceq.get_solution("mu-", mag=0, integrate=True)
    )
    assert nmu_sol == approx(nmu, abs=1e-11)


testdata_model = [
    ["DPMJETIII191", 64],
    ["DPMJETIII306", 64],
    ["QGSJET01C", 58],
    ["QGSJETII03", 44],
    ["QGSJETII04", 44],
    ["SIBYLL21", 48],
    ["SIBYLL23", 62],
    ["SIBYLL23C03", 62],
    ["SIBYLL23C", 62],
    ["SIBYLL23CPP", 62],
]


ids_model = [f"{model[0]}" for model in testdata_model]


@pytest.mark.parametrize(["model", "n"], testdata_model, ids=ids_model)
def test_set_interaction_model(mceq, model, n):
    mceq.set_interaction_model(model)
    n_particles = len(mceq._particle_list)
    assert n_particles == n


testdata_primary = [
    [1e3, 2.03134720e1, 6.80367347e1, 2.36908717e1],
    [1e6, 1.20365838e4, 2.53158948e4, 6.91213253e3],
    [1e9, 7.09254150e6, 1.20884925e7, 2.87396649e6],
    [5e10, 2.63982133e8, 4.14935240e8, 9.27683105e7],
]


ids_primary = [f"energy={primary[0]}" for primary in testdata_primary]


@pytest.mark.parametrize(
    ["energy", "nmu", "nnumu", "nnue"], testdata_primary, ids=ids_primary
)
def test_single_primary(mceq, energy, nmu, nnumu, nnue):
    mceq.set_interaction_model("SIBYLL23C")
    mceq.set_theta_deg(0.0)
    mceq.set_single_primary_particle(E=energy, pdg_id=2212)
    mceq.solve()
    nmu_sol = np.sum(
        mceq.get_solution("mu+", mag=0, integrate=True)
        + mceq.get_solution("mu-", mag=0, integrate=True)
    )
    nnumu_sol = np.sum(
        mceq.get_solution("numu", mag=0, integrate=True)
        + mceq.get_solution("antinumu", mag=0, integrate=True)
    )
    nnue_sol = np.sum(
        mceq.get_solution("nue", mag=0, integrate=True)
        + mceq.get_solution("antinue", mag=0, integrate=True)
    )

    assert nmu_sol == approx(nmu, rel=1e-8, abs=1e-5)
    assert nnumu_sol == approx(nnumu, rel=1e-8, abs=1e-5)
    assert nnue_sol == approx(nnue, rel=1e-8, abs=1e-5)
