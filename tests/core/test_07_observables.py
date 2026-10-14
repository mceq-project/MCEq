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

    if not any(p.name == "e+" for p in mceq_sib21.pman.all_particles):
        # Without enable_em there are no e+- in the system and n_e is zero.
        pytest.skip("no e+- in the hadronic system; n_e needs enable_em")
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
    # The two edge bins of the 31-bin window can go slightly negative (|z| ~ 1e-5
    # against a 4e-2 peak): the stencil tables are not clipped at x -> 1.
    assert np.all(z[2:-2] >= 0)
    assert z.min() > -1e-2 * z.max()
    assert np.any(z > 0)


def test_decay_z_factor(mceq_sib21):
    mceq_sib21.solve()
    z = mceq_sib21.decay_z_factor(211, 14)

    assert z.shape == mceq_sib21.e_grid.shape
    assert np.any(z > 0)
    assert not np.any(np.isnan(z))


M_MU, M_PI = 0.1056583755, 0.13957039


@pytest.mark.parametrize("gamma", [-2.7, -3.7])
def test_decay_z_factor_row_matches_lipari_two_body(mceq_sib21, gamma):
    """pi+ -> nu_mu and pi+ -> mu+ (both helicities) against Lipari's moment
    int x^(-gamma-1) dN/dx dx of the flat two-body spectra, away from the grid
    edges (the row needs parents above the daughter). The CI tables are
    stencil-averaged, so pi -> mu closes to 1e-4 despite its sharp kinematic
    edge at x = (m_mu/m_pi)^2 (bin-centre tables: +2..5 %)."""
    r = (M_MU / M_PI) ** 2
    a = -gamma - 1.0
    z_nu = mceq_sib21.decay_z_factor(211, 14, gamma=gamma)
    z_mu = sum(
        mceq_sib21.decay_z_factor((211, 0), (-13, h), gamma=gamma) for h in (-1, 1)
    )
    inner = slice(3, -12)
    assert np.allclose(z_nu[inner], (1 - r) ** a / (a + 1), rtol=0.01)
    ratio = z_mu[inner] / ((1 - r ** (a + 1)) / ((a + 1) * (1 - r)))
    assert np.allclose(ratio, 1.0, atol=2e-3)


def test_thunman_z_factor_is_a_flux_weighted_row(mceq_sib21):
    """The Thunman Z is the row against the nucleon flux: p -> pi+ between 0.01
    and 0.2 (SIBYLL 2.1, 1-100 TeV), and equal to the explicit row sum."""
    z = mceq_sib21.z_factor(2212, 211, definition="thunman", use_cs_scaling=True)
    e = mceq_sib21.e_grid
    phi = mceq_sib21.pmodel.p_and_n_flux(e)[1]
    p = mceq_sib21.pman[2212]
    c = np.asarray(p.hadr_yields[mceq_sib21.pman[211]], dtype=float)
    w = phi * p.prod_cross_section()
    i = int(np.argmin(abs(e - 1e4)))
    assert z[i] == pytest.approx(c[i] @ w / w[i], rel=1e-12)
    sel = (e > 1e3) & (e < 1e5)
    assert np.all((z[sel] > 0.01) & (z[sel] < 0.2))


def test_neutron_z_factor_uses_the_neutron_flux(mceq_sib21):
    """n -> pi+ is weighted with the neutron flux of the primary model."""
    z = mceq_sib21.z_factor(2112, 211, definition="thunman", use_cs_scaling=False)
    e = mceq_sib21.e_grid
    phi_n = mceq_sib21.pmodel.p_and_n_flux(e)[2]
    n = mceq_sib21.pman[2112]
    c = np.asarray(n.hadr_yields[mceq_sib21.pman[211]], dtype=float)
    i = int(np.argmin(abs(e - 1e4)))
    assert z[i] == pytest.approx(c[i] @ phi_n / phi_n[i], rel=1e-12)
