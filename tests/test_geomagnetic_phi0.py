"""Initial state with a geomagnetic rigidity cutoff."""

import numpy as np

from MCEq.batch import build_phi0_with_cutoff


def test_zero_cutoff_reproduces_the_primary_flux(mceq_sib21):
    phi = build_phi0_with_cutoff(mceq_sib21, mceq_sib21.pmodel, np.zeros(2))
    ref = mceq_sib21._phi0
    assert np.allclose(phi[:, 0], ref, rtol=0, atol=1e-3 * ref.max())
    assert np.array_equal(phi[:, 0], phi[:, 1])


def test_neutron_flux_goes_to_protons_when_neutrons_are_not_propagated(
    mceq_sib21,
):
    ref = build_phi0_with_cutoff(mceq_sib21, mceq_sib21.pmodel, np.zeros(1))
    neutron = mceq_sib21.pman[2112]
    neutron.is_resonance = True
    try:
        phi = build_phi0_with_cutoff(mceq_sib21, mceq_sib21.pmodel, np.zeros(1))
    finally:
        neutron.is_resonance = False
    assert np.isclose(phi.sum(), ref.sum(), rtol=1e-12, atol=0)
    assert not phi[neutron.lidx : neutron.uidx].any()
