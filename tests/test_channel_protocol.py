"""R2 deliverable A: the ``ChannelTable`` protocol and the shared wiring.

Plan §6.1 / §11 Phase 6. ``MCEqParticle.set_hadronic_channels`` and
``set_decay_channels`` collapsed onto one ``_set_channels`` routine that
consumes the table only through the protocol members; these tests pin
(a) that both shipped table classes satisfy the protocol structurally,
(b) the routine's remap-and-fetch semantics, and (c) that the collapse
did not move a channel: a solved run's wiring equals what an
independent reference walk of the tables produces.
"""

import numpy as np
import pytest

from MCEq.data.protocols import ChannelTable
from MCEq.species.particle import MCEqParticle


class FakePman:
    """Particle-manager stand-in with tuple-key ``__getitem__``."""

    def __init__(self, pdg2pref):
        self.pdg2pref = pdg2pref

    def __getitem__(self, key):
        return self.pdg2pref[key]


class Ref:
    """Particle reference stand-in; identity-hashable like ``MCEqParticle``."""

    def __init__(self, pdg_id, name, is_tracking=False):
        self.pdg_id = pdg_id
        self.name = name
        self.is_tracking = is_tracking


class FakeTable:
    """Minimal ChannelTable: parents/relations/get_matrix/__contains__."""

    def __init__(self, relations, matrices, iam="fake"):
        self.relations = dict(relations)
        self.parents = list(self.relations)
        self.iam = iam
        self._matrices = dict(matrices)

    def get_matrix(self, parent, child):
        return self._matrices[(parent, child)]

    def __contains__(self, key):
        return key in self.parents


def _particle(pdg_id=(2212, 0)):
    from MCEq import config

    return MCEqParticle(
        pdg_id[0],
        pdg_id[1],
        init_pdata_defaults=False,
        physics=config.physics,
    )


class TestProtocolMembership:
    def test_fake_table_is_a_channel_table(self):
        assert isinstance(FakeTable({(2212, 0): [(211, 0)]}, {}), ChannelTable)

    def test_cross_sections_are_not_channel_tables(self):
        """The CS table has no child axis (single-parent ``get_cs``); it is
        deliberately outside the protocol."""
        from MCEq.data.cross_sections import InteractionCrossSections

        assert not all(
            hasattr(InteractionCrossSections, name)
            for name in ("relations", "get_matrix")
        )


class TestSetChannelsSemantics:
    def test_remap_order_and_identity(self):
        p = _particle()
        m_pp = np.zeros((3, 3))
        m_pip = np.ones((3, 3))
        table = FakeTable(
            {(2212, 0): [(2212, 0), (211, 0)]},
            {((2212, 0), (2212, 0)): m_pp, ((2212, 0), (211, 0)): m_pip},
        )
        pip = Ref((211, 0), "pi+")
        proton = Ref((2212, 0), "p")
        pman = FakePman({(2212, 0): proton, (211, 0): pip})

        refs, yields = p._set_channels(table, pman, table.relations[(2212, 0)])
        assert refs == [proton, pip]
        assert yields[proton] is m_pp and yields[pip] is m_pip

    def test_tracking_secondaries_are_carried_over(self):
        p = _particle()
        m = np.zeros((3, 3))
        table = FakeTable(
            {(2212, 0): [(2212, 0)]},
            {((2212, 0), (2212, 0)): m, ((2212, 0), (211, 0)): m},
        )
        proton = Ref((2212, 0), "p")
        tracking = Ref((211, 0), "pi+~", is_tracking=True)
        p.hadr_secondaries = [tracking]
        pman = FakePman({(2212, 0): proton, (211, 0): tracking})

        refs, yields = p._set_channels(
            table, pman, table.relations[(2212, 0)], keep_tracking=True
        )
        assert refs == [proton, tracking]
        assert set(yields) == {proton, tracking}


@pytest.fixture(scope="module")
def mceq_sib21_ci_db():
    import crflux.models as pm

    from MCEq import config
    from MCEq.core import MCEqRun

    saved = config.mceq_db_fname
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        yield MCEqRun(
            interaction_model="QGSJET-II-04",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
    finally:
        config.mceq_db_fname = saved


class TestCollapseIsBehaviourNeutral:
    def test_hadronic_wiring_matches_reference_walk(self, mceq_sib21_ci_db):
        run = mceq_sib21_ci_db
        table = run._interactions
        for p in run.pman.all_particles:
            if p.is_tracking:
                continue
            if p.is_projectile:
                expected = [run.pman[k] for k in table.relations[p.pdg_id]]
                assert p.hadr_secondaries[: len(expected)] == expected
                for s in expected:
                    assert p.hadr_yields[s] is table.get_matrix(p.pdg_id, s.pdg_id)
            else:
                assert p.hadr_yields == {} or p.is_tracking

    def test_decay_wiring_matches_reference_walk(self, mceq_sib21_ci_db):
        run = mceq_sib21_ci_db
        table = run._decays
        for p in run.pman.all_particles:
            if p.is_tracking:
                continue  # tracking copies get a hand-appended channel
            if p.is_stable:
                assert p.decay_dists == {}
                continue
            expected = [run.pman[k] for k in table.children(p.pdg_id)]
            assert p.children[: len(expected)] == expected
            for c in expected:
                assert p.decay_dists[c] is table.get_matrix(p.pdg_id, c.pdg_id)

    def test_protocol_membership_on_live_objects(self, mceq_sib21_ci_db):
        assert isinstance(mceq_sib21_ci_db._interactions, ChannelTable)
        assert isinstance(mceq_sib21_ci_db._decays, ChannelTable)
