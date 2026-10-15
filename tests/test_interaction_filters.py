"""The projectile filters of ``config.adv_set`` in ``Interactions.load``."""

from MCEq import config
from MCEq.data import HDF5Backend, Interactions

NUCLEONS = {(2212, 0), (2112, 0), (-2212, 0), (-2112, 0)}


def _load(disable_unstable):
    # conftest restores config.adv_set and config.mceq_db_fname afterwards
    config.mceq_db_fname = "mceq_ci_base_air_1d_v2.h5"
    config.adv_set["disable_interactions_of_unstable"] = disable_unstable
    inter = Interactions(HDF5Backend())
    inter.load("SIBYLL21")
    return inter


def test_disable_interactions_of_unstable_keeps_only_the_nucleons():
    baseline = _load(False)
    assert NUCLEONS < set(baseline.parents)
    assert {(211, 0), (321, 0), (3122, 0)} <= set(baseline.parents)

    inter = _load(True)
    assert set(inter.parents) == NUCLEONS
    assert set(inter.relations) == NUCLEONS
    # Pions and kaons are still produced, they just no longer interact.
    assert {(211, 0), (-211, 0), (321, 0)} <= set(inter.particles)
