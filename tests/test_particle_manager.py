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


def test_state_index_maps_hold_only_cascade_particles(mceq_sib21):
    """A resonance (mceqidx -1) must not enter the state-index maps."""
    pman = mceq_sib21.pman
    cascade, resonances = pman.cascade_particles, pman.resonances
    try:
        res = cascade[-1]
        pman.cascade_particles = cascade[:-1]
        pman.resonances = [*resonances, res]
        res.mceqidx, res.is_resonance = -1, True
        pman._update_particle_tables()
        assert -1 not in pman.mceqidx2pref
        assert sorted(pman.mceqidx2pref) == list(range(len(cascade) - 1))
    finally:
        res.mceqidx, res.is_resonance = len(cascade) - 1, False
        pman.cascade_particles, pman.resonances = cascade, resonances
        pman._update_particle_tables()
