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


def test_depth_profile_rejects_unknown_definition_and_method(mceq_sib21):
    mceq_sib21.solve([0, 1])
    with pytest.raises(ValueError, match="definition"):
        mceq_sib21.muon_xmax(definition="flux")
    with pytest.raises(ValueError, match="method"):
        mceq_sib21.muon_xmax(method="spline")


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

    for definition in ("number", "production"):
        _, total = mceq_sib21.muon_depth_profile(definition=definition)
        _, plus = mceq_sib21.depth_profile("total_mu+", definition=definition)
        _, minus = mceq_sib21.depth_profile("total_mu-", definition=definition)
        np.testing.assert_allclose(plus + minus, total)
        _, conv = mceq_sib21.muon_depth_profile(definition=definition, prefix="conv_")
        _, prompt = mceq_sib21.muon_depth_profile(definition=definition, prefix="pr_")
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


def test_production_profile_integrates_to_the_muon_number(mceq_sib21):
    """The muon production rate integrated over depth is the number of muons
    produced. Above the reduced database's ~900 GeV threshold muons neither
    decay nor lose a noticeable fraction of their energy in the atmosphere, so
    that number is the muon count at the ground, up to the trapezoid error of
    the depth grid near the sharp production peak."""
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 300)
    mceq_sib21.solve(int_grid=X_grid)

    X, rate = mceq_sib21.muon_depth_profile(definition="production")

    assert X.shape == rate.shape == X_grid.shape
    assert np.all(rate >= 0)
    assert rate[0] < 1e-2 * rate.max()  # no mesons yet at the top
    assert np.trapezoid(rate, X) == pytest.approx(mceq_sib21.n_mu(), rel=0.03)


def test_production_profile_is_the_source_term(mceq_sib21):
    """Muon neutrinos have no losses, so their production rate is the depth
    derivative of their number."""
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 300)
    mceq_sib21.solve(int_grid=X_grid)

    X, rate = mceq_sib21.depth_profile("total_numu", definition="production")
    _, n_numu = mceq_sib21.depth_profile("total_numu")
    i = len(X) // 3
    assert np.gradient(n_numu, X)[i] == pytest.approx(rate[i], rel=1e-2)

    _, spectrum = mceq_sib21.depth_profile(
        "total_numu", definition="production", per_energy=True
    )
    assert spectrum.shape == (len(X), mceq_sib21.dim)
    np.testing.assert_allclose((spectrum * mceq_sib21.e_widths).sum(axis=1), rate)


def test_gaisser_hillas_fit_recovers_the_parameters():
    from MCEq.driver.observables import fit_gaisser_hillas, gaisser_hillas

    X = np.linspace(0, 1500, 300)
    profile = gaisser_hillas(X, 3.0, 560.0, -45.0, 70.0)
    assert np.all(
        gaisser_hillas(np.array([-45.0, -100.0]), 3.0, 560.0, -45.0, 70.0) == 0
    )
    assert np.argmax(profile) == np.argmin(np.abs(X - 560.0))

    np.testing.assert_allclose(
        fit_gaisser_hillas(X, profile), (3.0, 560.0, -45.0, 70.0)
    )
    np.testing.assert_allclose(
        fit_gaisser_hillas(X, profile, x0=None), (3.0, 560.0, -45.0, 70.0), rtol=1e-6
    )
    noisy = profile * (1 + 0.05 * np.random.default_rng(1).standard_normal(X.size))
    assert fit_gaisser_hillas(X, noisy)[1] == pytest.approx(560.0, abs=5.0)


def test_muon_xmax_of_a_single_primary(mceq_sib21):
    """Muons above the reduced database's ~900 GeV threshold from a 100 TeV
    proton come from its first interactions, so the production peaks near
    the top; the Gaisser-Hillas fit is what ``definition="production"``
    returns by default."""
    from MCEq.driver.observables import fit_gaisser_hillas

    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 300)
    mceq_sib21.set_single_primary_particle(1e5, pdg_id=2212)
    mceq_sib21.solve(int_grid=X_grid)

    X, rate = mceq_sib21.muon_depth_profile(definition="production")
    xmax = mceq_sib21.muon_xmax(definition="production")
    assert xmax == fit_gaisser_hillas(X, rate)[1]
    assert xmax == mceq_sib21.muon_xmax(
        definition="production", method="gaisser-hillas"
    )
    assert 0 < xmax < 100.0
    assert 0 < mceq_sib21.muon_xmax(definition="production", method="grid") < 20.0


def test_muon_xmax_of_the_inclusive_flux(mceq_sib21):
    """Above the reduced database's ~900 GeV threshold muons survive to the
    ground, so their number peaks deep in the atmosphere, where energy loss
    below the threshold starts to outweigh the dwindling production; the
    production itself peaks within the first few g/cm^2."""
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    X_grid = np.linspace(0, mceq_sib21.density_model.max_X, 300)
    mceq_sib21.solve(int_grid=X_grid)

    X, n = mceq_sib21.muon_depth_profile()
    X_peak = X[np.argmax(n)]
    xmax = mceq_sib21.muon_xmax()  # parabola for the number profile
    assert mceq_sib21.muon_xmax(method="parabola") == xmax
    assert mceq_sib21.muon_xmax(method="grid") == X_peak
    assert abs(xmax - X_peak) <= X[1] - X[0]
    assert 500.0 < xmax < X_grid[-1]
    assert mceq_sib21.xmax(["total_mu+", "total_mu-"]) == xmax

    assert 0 < mceq_sib21.muon_xmax(definition="production", method="grid") < 20.0


def test_xmax_on_the_grid_edge(mceq_sib21):
    """A two-point grid cannot bracket a peak: the edge depth comes back."""
    mceq_sib21.solve([0, 1])
    assert mceq_sib21.muon_xmax() == 1.0
    assert mceq_sib21.muon_xmax(method="grid") == 1.0


def test_xmax_of_an_empty_profile(mceq_sib21):
    """There are no muons at the top of the atmosphere."""
    mceq_sib21.solve([0])
    assert np.isnan(mceq_sib21.muon_xmax())
    # Neutrinos from muon decays: no muons yet, so nothing is produced.
    assert np.isnan(mceq_sib21.xmax("mu_numu", definition="production"))


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
