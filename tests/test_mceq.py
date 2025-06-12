import sys

import numpy as np
import pytest
from pytest import approx


if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


@pytest.mark.parametrize(
    ["theta", "nmu"],
    [[0.0, 5.62504370e-3], [30.0, 4.20479234e-3], [60.0, 1.36630552e-3]],
)
def test_set_theta_deg(mceq, theta, nmu):
    mceq.set_theta_deg(theta)
    mceq.solve()
    nmu_sol = np.sum(
        mceq.get_solution("mu+", mag=0, integrate=True)
        + mceq.get_solution("mu-", mag=0, integrate=True)
    )
    assert nmu_sol == approx(nmu, abs=1e-11)


@pytest.mark.parametrize(
    ["model", "n"],
    [
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
    ],
)
def test_set_interaction_model(mceq, model, n):
    mceq.set_interaction_model(model)
    n_particles = len(mceq._particle_list)
    assert n_particles == n


@pytest.mark.parametrize(
    ["energy", "nmu", "nnumu", "nnue"],
    [
        [1e3, 2.03134720e1, 6.80367347e1, 2.36908717e1],
        [1e6, 1.20365838e4, 2.53158948e4, 6.91213253e3],
        [1e9, 7.09254150e6, 1.20884925e7, 2.87396649e6],
        [5e10, 2.63982133e8, 4.14935240e8, 9.27683105e7],
    ],
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
