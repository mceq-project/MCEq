"""Interaction-model switching and particle-list bookkeeping.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest
from pytest import approx

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)

testdata_theta = [
    [0.0, 8.836533799620236e-08],
    [30.0, 9.915573533441199e-08],
    [60.0, 1.50709235583021e-07],
]

ids_theta = [f"{th[0]}" for th in testdata_theta]


@pytest.mark.parametrize(["theta", "nmu"], testdata_theta, ids=ids_theta)
def test_set_theta_deg(mceq_sib21, theta, nmu):
    mceq_sib21.set_theta_deg(theta)
    mceq_sib21.solve()
    nmu_sol = np.sum(
        mceq_sib21.get_solution("mu+", mag=0, integrate=True)
        + mceq_sib21.get_solution("mu-", mag=0, integrate=True)
    )
    assert nmu_sol == approx(nmu, abs=1e-8)


testdata_model = [
    ["QGSJETII04", 48],
    ["SIBYLL21", 52],
]
ids_model = [f"{model[0]}" for model in testdata_model]


@pytest.mark.parametrize(["model", "n"], testdata_model, ids=ids_model)
def test_set_interaction_model_model(mceq_sib21, model, n):
    mceq_sib21.set_interaction_model(model)
    n_particles = len(mceq_sib21._particle_list)
    assert n_particles == n


def test_set_interaction_model_update_particle_list(mceq_sib21):
    # Establish a known baseline by re-running update_particle_list under the
    # current (autouse-restored) config; prior parametrized model switches in
    # the session leave _particle_list in an unrelated state otherwise.
    mceq_sib21.set_interaction_model("SIBYLL21", update_particle_list=True)
    n_particles_sib = len(mceq_sib21._particle_list)

    mceq_sib21.set_interaction_model("QGSJETII04", update_particle_list=True)
    n_particles = len(mceq_sib21._particle_list)
    assert n_particles == 48

    mceq_sib21.set_interaction_model("SIBYLL21", update_particle_list=True)
    n_particles_s = len(mceq_sib21._particle_list)

    assert n_particles_s == n_particles_sib
