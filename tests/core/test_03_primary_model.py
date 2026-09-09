"""Ctor particle-list handling and single-primary solves (reference values in comments).

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import crflux.models as pm
import numpy as np
import pytest
from pytest import approx

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


@pytest.mark.parametrize(
    ["particle_list", "projectiles"],
    [
        [None, 13],
        [[(2212, 0)], 1],
    ],
    ids=["None", "proton"],
)
def test_mceq_init_particles_list(particle_list, projectiles):
    from MCEq.core import MCEqRun

    mceq_sib21 = MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        particle_list=particle_list,
    )

    assert (
        sum([p.is_projectile for p in mceq_sib21.pman.cascade_particles]) == projectiles
    )


# Recalibrated 2026-07 for the v2 release defaults (see testdata_theta note).
# NB the small negative nnue at 1e3 is a pre-existing low-energy boundary
# artifact of the e+- continuous-loss operator, unchanged by the upwind rows.
# The 1e3 primary sits closest to the grid floor and so is the most sensitive
# to generic_losses_all_charged=True (nmu -0.6%, nnumu -2.6%); the 1e4/1e5
# primaries move by <0.06%. Retuned 2026-09-09 for the generic-spline edge
# clamp (+1.2% on nnumu at 1e3; the boost-7165 pion no longer extrapolates).
testdata_primary = [
    [1e3, 1.2298250970471001e-05, 2.3751092301812317e-07, -4.397660019572766e-08],
    [1e4, 0.09919964595464914, 0.024600024310737535, 0.001460721937230645],
    [1e5, 0.9112967205413417, 0.28325496449179144, 0.020282423131684792],
]
ids_primary = [f"energy={primary[0]}" for primary in testdata_primary]


@pytest.mark.parametrize(
    ["energy", "nmu", "nnumu", "nnue"], testdata_primary, ids=ids_primary
)
def test_single_primary(mceq_sib21, energy, nmu, nnumu, nnue):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.set_single_primary_particle(E=energy, pdg_id=2212)
    mceq_sib21.solve()
    nmu_sol = np.sum(
        mceq_sib21.get_solution("mu+", mag=0, integrate=True)
        + mceq_sib21.get_solution("mu-", mag=0, integrate=True)
    )
    nnumu_sol = np.sum(
        mceq_sib21.get_solution("numu", mag=0, integrate=True)
        + mceq_sib21.get_solution("antinumu", mag=0, integrate=True)
    )
    nnue_sol = np.sum(
        mceq_sib21.get_solution("nue", mag=0, integrate=True)
        + mceq_sib21.get_solution("antinue", mag=0, integrate=True)
    )

    assert nmu_sol == approx(nmu, rel=5e-3)
    assert nnumu_sol == approx(nnumu, rel=5e-3)
    assert nnue_sol == approx(nnue, rel=5e-3)


def test_single_primary_pdg_corsika(mceq_sib21):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.set_single_primary_particle(E=1e5, pdg_id=1000020040)
    mceq_sib21.solve()

    pdg_sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    mceq_sib21.set_single_primary_particle(E=1e5, corsika_id=402)
    mceq_sib21.solve()

    corsika_sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    assert np.allclose(pdg_sol, corsika_sol)


def test_single_primary_e_too_low(mceq_sib21):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    with pytest.raises(Exception):
        mceq_sib21.set_single_primary_particle(E=1e0, pdg_id=2212)
