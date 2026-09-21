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


def test_depth_profile_requires_grid(mceq_sib21):
    mceq_sib21.solve()
    with pytest.raises(RuntimeError, match="int_grid"):
        mceq_sib21.muon_depth_profile()
    with pytest.raises(RuntimeError, match="int_grid"):
        mceq_sib21.muon_xmax()


def test_depth_profile_is_the_particle_count_per_depth(mceq_sib21):
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 50)
    mceq_sib21.solve(int_grid=X_grid)

    X, n = mceq_sib21.muon_depth_profile()
    np.testing.assert_allclose(X, X_grid)
    expected = [mceq_sib21.n_mu(grid_idx=i) for i in range(len(X_grid))]
    np.testing.assert_allclose(n, expected)
    assert n[0] == 0  # no muons at the top of the atmosphere
    assert np.all(n[1:] > 0)

    _, spectrum = mceq_sib21.depth_profile("total_mu+", per_energy=True)
    assert spectrum.shape == (len(X), mceq_sib21.dim)
    np.testing.assert_allclose(
        spectrum[-1], mceq_sib21.get_solution("total_mu+", grid_idx=len(X) - 1)
    )


def test_depth_profile_sums_names_and_prefixes(mceq_sib21):
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 20)
    mceq_sib21.solve(int_grid=X_grid)

    _, total = mceq_sib21.muon_depth_profile()
    _, plus = mceq_sib21.depth_profile("total_mu+")
    _, minus = mceq_sib21.depth_profile("total_mu-")
    np.testing.assert_allclose(plus + minus, total)
    _, conv = mceq_sib21.muon_depth_profile(prefix="conv_")
    _, prompt = mceq_sib21.muon_depth_profile(prefix="pr_")
    np.testing.assert_allclose(conv + prompt, total, rtol=1e-10, atol=0)


def test_depth_profile_energy_cuts(mceq_sib21):
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 20)
    mceq_sib21.solve(int_grid=X_grid)

    _, total = mceq_sib21.muon_depth_profile()
    _, low = mceq_sib21.muon_depth_profile(max_energy_cutoff=1e4)
    _, high = mceq_sib21.muon_depth_profile(min_energy_cutoff=1e4)
    np.testing.assert_allclose(low + high, total)
    assert np.all(low <= total) and np.any(low < total)
    with pytest.raises(ValueError, match="No energy bin"):
        mceq_sib21.muon_depth_profile(min_energy_cutoff=1e9)


def test_muon_xmax_locates_the_profile_maximum(mceq_sib21):
    """Above the reduced database's ~900 GeV threshold muons survive to the
    ground, so their number peaks deep in the atmosphere, where energy loss
    below the threshold starts to outweigh the dwindling production."""
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 300)
    mceq_sib21.solve(int_grid=X_grid)

    X, n = mceq_sib21.muon_depth_profile()
    X_peak = X[np.argmax(n)]

    xmax = mceq_sib21.muon_xmax()
    assert mceq_sib21.muon_xmax(refine=False) == X_peak
    assert abs(xmax - X_peak) <= X[1] - X[0]
    assert 500.0 < xmax < X_grid[-1]
    assert mceq_sib21.xmax(["total_mu+", "total_mu-"]) == xmax


def test_xmax_on_the_grid_edge(mceq_sib21):
    """A two-point grid cannot bracket a peak: the edge depth comes back
    unrefined."""
    mceq_sib21.solve([0, 1])
    assert mceq_sib21.muon_xmax() == 1.0


def test_xmax_of_an_empty_profile(mceq_sib21):
    """There are no muons at the top of the atmosphere."""
    mceq_sib21.solve([0])
    assert np.isnan(mceq_sib21.muon_xmax())


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
