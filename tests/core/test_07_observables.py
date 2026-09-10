"""Run observables: mod_pprod, regenerate, yields, z-factors, interaction-model forwarding.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_set_mod_pprod(mceq_sib21):
    def weight(xmat, egrid, pname, value):
        return (1 + value) * np.ones_like(xmat)

    ret = mceq_sib21.set_mod_pprod(2212, 211, weight, ("a", 0.15))
    assert ret == 1

    assert mceq_sib21._interactions.mod_pprod[(2212, 211)] is not None
    assert mceq_sib21._interactions.mod_pprod[(2212, -211)] is not None

    mceq_sib21.regenerate_matrices()
    mceq_sib21.solve()

    mceq_sib21.regenerate_matrices(skip_decay_matrix=True)
    mceq_sib21.solve()


def test_unset_mod_pprod(mceq_sib21):
    def weight(xmat, egrid, pname, value):
        return (1 + value) * np.ones_like(xmat)

    mceq_sib21.set_mod_pprod(2212, 211, weight, ("a", 0.15))

    mceq_sib21.regenerate_matrices()
    mceq_sib21.solve()

    mceq_sib21.unset_mod_pprod()
    assert not mceq_sib21._interactions.mod_pprod

    mceq_sib21.set_mod_pprod(2212, 211, weight, ("a", 0.15))

    mceq_sib21.regenerate_matrices()
    mceq_sib21.solve()
    mceq_sib21.unset_mod_pprod(dont_fill=False)
    assert not mceq_sib21._interactions.mod_pprod


def test_regenerate_matrices_after_adding_tracking_particle(mceq_sib21):
    mceq_sib21.pman.add_tracking_particle(
        [(2212, 0)], (13, 0), "p_mu", from_interactions=True
    )
    mceq_sib21.regenerate_matrices()
    mceq_sib21.solve()


def test_n_particles_energy_cutoff_and_grid(mceq_sib21):
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.solve([0, 1])
    n0 = mceq_sib21.n_particles("mu+", grid_idx=0)
    n1 = mceq_sib21.n_particles("mu+", grid_idx=1)
    n_high_cut = mceq_sib21.n_particles("mu+", grid_idx=1, min_energy_cutoff=1e5)

    assert n0 == 0
    assert n1 > n0
    assert n_high_cut < n1


def test_n_mu_energy_cutoff_and_grid(mceq_sib21):
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.solve([0, 1])
    n0 = mceq_sib21.n_mu(grid_idx=0)
    n1 = mceq_sib21.n_mu(grid_idx=1)
    nhigh = mceq_sib21.n_mu(grid_idx=1, min_energy_cutoff=1e5)

    assert n0 == 0
    assert n1 > 0
    assert nhigh < n1


def test_n_e_energy_cutoff_and_grid(mceq_sib21):
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.solve([0, 1])
    n0 = mceq_sib21.n_e(grid_idx=0)
    n1 = mceq_sib21.n_e(grid_idx=1)
    nhigh = mceq_sib21.n_e(grid_idx=1, min_energy_cutoff=1e5)

    assert n0 == 0
    assert n1 > 0
    assert nhigh < n1


@pytest.mark.parametrize(
    ["definition", "use_cs_scaling"],
    [
        ["primary_e", False],
        ["no_name_definition", True],
    ],
)
def test_z_factor(mceq_sib21, definition, use_cs_scaling):
    mceq_sib21.solve([0, 1])
    z = mceq_sib21.z_factor(
        2212, 211, definition=definition, use_cs_scaling=use_cs_scaling
    )

    assert isinstance(z, np.ndarray)
    assert z.shape == mceq_sib21.e_grid.shape
    assert np.all(z >= 0)
    assert np.any(z > 0)


def test_decay_z_factor(mceq_sib21):
    mceq_sib21.solve()
    z = mceq_sib21.decay_z_factor(211, 14)

    assert z.shape == mceq_sib21.e_grid.shape
    assert np.any(z > 0)
    assert not np.any(np.isnan(z))
