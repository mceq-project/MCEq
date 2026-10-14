"""ParticleManager lookups and index maps."""

import pytest


@pytest.mark.parametrize("key", ["numu", "pi+", 211, (211, 0), "pi_numu"])
def test_contains_agrees_with_getitem(mceq_sib21, key):
    pman = mceq_sib21.pman
    assert key in pman
    assert pman[key] is not None


@pytest.mark.parametrize("key", ["no_such_particle", 99999, (211, 7)])
def test_contains_is_false_for_unknown_particles(mceq_sib21, key):
    assert key not in mceq_sib21.pman
