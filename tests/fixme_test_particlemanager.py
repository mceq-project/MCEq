import pytest
from MCEq.particlemanager import ParticleManager, MCEqParticle


@pytest.fixture
def hdf_db():
    from MCEq.data import HDF5Backend

    return HDF5Backend()


@pytest.fixture
def energy_grid(hdf_db):
    return hdf_db.energy_grid


@pytest.fixture
def cs_db(hdf_db):
    from MCEq.data import InteractionCrossSections

    return InteractionCrossSections(hdf_db, interaction_model="QGSJETII04")


@pytest.fixture
def decay_db(hdf_db):
    from MCEq.data import Decays

    return Decays(hdf_db)


def test_mceqparticle_defaults(energy_grid, cs_db):
    p = MCEqParticle(
        pdg_id=211,
        helicity=0,
        energy_grid=energy_grid,
        cs_db=cs_db,
        init_pdata_defaults=False,
    )
    print(p)
    assert p.pdg_id == (211, 0)
    assert p.helicity == 0
    assert p.name is None
    assert p.mceqidx == -1


def test_mceqparticle_pythia_defaults(energy_grid, cs_db):
    p = MCEqParticle(
        pdg_id=211,
        helicity=0,
        energy_grid=energy_grid,
        cs_db=cs_db,
        init_pdata_defaults=True,
    )
    assert p.pdg_id == (211, 0)
    assert p.helicity == 0
    assert p.name == "pi+"
    assert p.is_hadron is True
    assert p.is_mixed is True
    assert p.E_mix == pytest.approx(8.9125, rel=1e-3)


def test_particle_manager_creation(energy_grid, cs_db):
    pdg_list = [(211, 0), (111, 0)]
    pm = ParticleManager(pdg_list, energy_grid, cs_db)
    print(pm)
    print('1', pm.pdg2pref)
    assert len(pm.cascade_particles) == 2
    assert pm.n_cparticles == 2
    assert pm.dim_states == energy_grid.d * 2
    assert pm[0].pdg_id == (111, 0)
    assert pm[1].pdg_id == (211, 0)


def test_set_cross_sections_db(energy_grid, cs_db):
    pdg_list = [(211, 0), (111, 0)]
    pm = ParticleManager(pdg_list, energy_grid, cs_db)
    pm.set_cross_sections_db(cs_db)
    for p in pm.cascade_particles:
        assert p.current_cross_sections == "QGSJETII04"


def test_set_decay_channels(energy_grid, cs_db, decay_db):
    # Prepare decay data
    decay_db.parents.add((111, 0))
    decay_db.child_map[(111, 0)] = [(211, 0)]

    pdg_list = [(111, 0), (211, 0)]
    pm = ParticleManager(pdg_list, energy_grid, cs_db)
    pm.set_decay_channels(decay_db)
    unstable_particle = pm[(111, 0)]
    assert len(unstable_particle.children) == 1


def test_add_tracking_particle(energy_grid, cs_db):
    pdg_list = [(211, 0), (13, 0)]
    pm = ParticleManager(pdg_list, energy_grid, cs_db)
    pm.add_tracking_particle([(211, 0)], (13, 0), "pi_mu")
    tracking_particle = pm.pname2pref.get("pi_mu", None)
    assert tracking_particle is not None
    assert tracking_particle.is_tracking


def test_inverse_decay_length(energy_grid, cs_db):
    p = MCEqParticle(
        pdg_id=(13, 0),
        helicity=0,
        energy_grid=energy_grid,
        cs_db=cs_db,
        init_pdata_defaults=False,
    )
    p.mass = 0.105
    p.ctau = 100.0
    p.mix_idx = 3

    inv_len = p.inverse_decay_length()
    assert len(inv_len) == p._energy_grid.d
    # Should be zero before mix_idx
    assert all(inv_len[:3] == 0)
