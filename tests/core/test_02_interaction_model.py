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


def _expected_species(run):
    from MCEq.particlemanager import EM_PDG_IDS

    tables = run._interactions.particles + run._decays.particles
    return {p for p in tables if p[0] not in EM_PDG_IDS}


@pytest.mark.parametrize("model", ["QGSJETII04", "SIBYLL21"])
def test_set_interaction_model_model(mceq_sib21, model):
    """The species are those of the tables, without e+- and photons (no enable_em)."""
    mceq_sib21.set_interaction_model(model)
    assert set(mceq_sib21.pman.pdg_id_list) == _expected_species(mceq_sib21)
    assert len(mceq_sib21.pman.pdg_id_list) == 22


def test_set_interaction_model_update_particle_list(mceq_sib21):
    mceq_sib21.set_interaction_model("SIBYLL21", update_particle_list=True)
    species_sib = list(mceq_sib21.pman.pdg_id_list)

    mceq_sib21.set_interaction_model("QGSJETII04", update_particle_list=True)
    assert set(mceq_sib21.pman.pdg_id_list) == _expected_species(mceq_sib21)

    mceq_sib21.set_interaction_model("SIBYLL21", update_particle_list=True)
    assert mceq_sib21.pman.pdg_id_list == species_sib
