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


def test_track_leptons_from_keeps_all_leptons_without_exclude_em(mceq_sib21):
    pman = mceq_sib21.pman
    added = []
    real_add = pman.add_tracking_particle
    pman.add_tracking_particle = lambda parents, pdg, *a, **k: added.append(pdg)
    try:
        pman.track_leptons_from([(211, 0)], "x_", exclude_em=False)
    finally:
        pman.add_tracking_particle = real_add
    leptons = {
        p.pdg_id
        for p in pman.all_particles
        if p.is_lepton and not p.is_tracking and p.pdg_id[1] == 0
    }
    assert set(added) == leptons


def test_set_interaction_model_reloads_for_a_new_parent_list():
    from MCEq.core import MCEqRun

    nucleons = [(2212, 0), (2112, 0)]
    run = MCEqRun("SIBYLL23E", None, 0.0)
    assert len(run._interactions.parents) > 2
    run.set_interaction_model("SIBYLL23E", particle_list=nucleons)
    assert sorted(run._interactions.parents) == sorted(nucleons)
    assert sum(p.is_projectile for p in run.pman.cascade_particles) == 2
    # the same request again, None (keep) and force=True do not raise
    run.set_interaction_model("SIBYLL23E", particle_list=nucleons)
    run.set_interaction_model("SIBYLL23E")
    run.set_interaction_model("SIBYLL23E", particle_list=nucleons, force=True)
    assert sorted(run._interactions.parents) == sorted(nucleons)
