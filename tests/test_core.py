import sys

import crflux.models as pm
import numpy as np
import pytest
from pytest import approx

import MCEq.geometry.density_profiles as dprof
from MCEq.geometry.atmosphere_parameters import list_available_corsika_atmospheres

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_solve_default(mceq_sib21):
    mceq_sib21.solve()
    sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None

    assert np.sum(mceq_sib21._phi0) != np.sum(mceq_sib21._solution)

    muons = mceq_sib21.get_solution("total_mu-", mag=0)

    assert np.sum(muons) != 0


def test_solve_skip_integration_path(mceq_sib21):
    mceq_sib21._calculate_integration_path(int_grid=None, grid_var="X")
    mceq_sib21.solve(skip_integration_path=True)
    sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_other_grid_var(mceq_sib21):
    with pytest.raises(NotImplementedError):
        mceq_sib21.solve(grid_var="Y")


@pytest.mark.parametrize(
    ["int_grid", "grid_shape"],
    [[None, (0,)], [[0, 1], (2, 2232)]],
    ids=["no-grid", "with-grid"],
)
def test_solve_int_grid(mceq_sib21, int_grid, grid_shape):
    mceq_sib21.solve(int_grid)
    assert mceq_sib21.grid_sol.shape == grid_shape


def test_integration_path_grid_idcs(mceq_sib21):
    int_grid = [0, 1]

    grid_var = "X"
    mceq_sib21._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq_sib21.integration_path
    grid_idcs = integration_path[-1]
    assert len(grid_idcs) == 2
    assert grid_idcs[0] == 0
    assert grid_idcs[0] < grid_idcs[1]


testdata_theta = [
    [0.0, 8.8312635576492481e-08],
    [30.0, 9.9070776732966113e-08],
    [60.0, 1.5039581700049055e-07],
]

ids_theta = [f"{th[0]}" for th in testdata_theta]


@pytest.mark.parametrize(["theta", "nmu"], testdata_theta, ids=ids_theta)
def test_set_theta_deg(mceq_sib21, theta, nmu):
    mceq_sib21.set_theta_deg(theta)
    mceq_sib21.solve()
    nmu_sol = np.sum(
        mceq_sib21.get_solution("mu+", mag=0, integrate=True)
        + mceq_sib21.get_solution("mu-", mag=0, integrate=True)
    )
    assert nmu_sol == approx(nmu, abs=1e-8)


testdata_model = [
    ["QGSJETII04", 48],
    ["SIBYLL21", 52],
]
ids_model = [f"{model[0]}" for model in testdata_model]


@pytest.mark.parametrize(["model", "n"], testdata_model, ids=ids_model)
def test_set_interaction_model_model(mceq_sib21, model, n):
    mceq_sib21.set_interaction_model(model)
    n_particles = len(mceq_sib21._particle_list)
    assert n_particles == n


def test_set_interaction_model_update_particle_list(mceq_sib21):
    n_particles_sib = len(mceq_sib21._particle_list)

    mceq_sib21.set_interaction_model("QGSJETII04", update_particle_list=True)
    n_particles = len(mceq_sib21._particle_list)
    assert n_particles == 48

    mceq_sib21.set_interaction_model("SIBYLL21", update_particle_list=True)
    n_particles_s = len(mceq_sib21._particle_list)

    assert n_particles_s == n_particles_sib


@pytest.mark.parametrize(
    ["particle_list", "projectiles"],
    [
        [None, 13],
        [[(2212, 0)], 1],
    ],
    ids=["None", "proton"],
)
def test_mceq_init_particles_list(particle_list, projectiles):
    import crflux.models as pm

    from MCEq.core import MCEqRun

    mceq_sib21 = MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        particle_list=particle_list,
    )

    assert (
        sum([p.is_projectile for p in mceq_sib21.pman.cascade_particles]) == projectiles
    )


testdata_primary = [
    [1e3, 1.1904680953281074e-05, 2.4098237699529783e-07, -4.401425991268454e-08],
    [1e4, 0.09917221096682655, 0.024609349696095902, 0.001461068061597137],
    [1e5, 0.9113822218370885, 0.2833603479604529, 0.02028732056926894],
]
ids_primary = [f"energy={primary[0]}" for primary in testdata_primary]


@pytest.mark.parametrize(
    ["energy", "nmu", "nnumu", "nnue"], testdata_primary, ids=ids_primary
)
def test_single_primary(mceq_sib21, energy, nmu, nnumu, nnue):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.set_single_primary_particle(E=energy, pdg_id=2212)
    mceq_sib21.solve()
    nmu_sol = np.sum(
        mceq_sib21.get_solution("mu+", mag=0, integrate=True)
        + mceq_sib21.get_solution("mu-", mag=0, integrate=True)
    )
    nnumu_sol = np.sum(
        mceq_sib21.get_solution("numu", mag=0, integrate=True)
        + mceq_sib21.get_solution("antinumu", mag=0, integrate=True)
    )
    nnue_sol = np.sum(
        mceq_sib21.get_solution("nue", mag=0, integrate=True)
        + mceq_sib21.get_solution("antinue", mag=0, integrate=True)
    )

    assert nmu_sol == approx(nmu, rel=5e-3)
    assert nnumu_sol == approx(nnumu, rel=5e-3)
    assert nnue_sol == approx(nnue, rel=5e-3)


def test_single_primary_pdg_corsika(mceq_sib21):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.set_single_primary_particle(E=1e5, pdg_id=1000020040)
    mceq_sib21.solve()

    pdg_sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    mceq_sib21.set_single_primary_particle(E=1e5, corsika_id=402)
    mceq_sib21.solve()

    corsika_sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    assert np.allclose(pdg_sol, corsika_sol)


def test_single_primary_e_too_low(mceq_sib21):
    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    mceq_sib21.set_theta_deg(0.0)
    with pytest.raises(Exception):
        mceq_sib21.set_single_primary_particle(E=1e0, pdg_id=2212)


testdata_ecenters = [
    8.91250938e2,
    1.12201845e3,
    1.41253754e3,
    1.77827941e3,
    2.23872114e3,
    2.81838293e3,
    3.54813389e3,
    4.46683592e3,
    5.62341325e3,
    7.07945784e3,
    8.91250938e3,
    1.12201845e4,
    1.41253754e4,
    1.77827941e4,
    2.23872114e4,
    2.81838293e4,
    3.54813389e4,
    4.46683592e4,
    5.62341325e4,
    7.07945784e4,
    8.91250938e4,
    1.12201845e5,
    1.41253754e5,
    1.77827941e5,
    2.23872114e5,
    2.81838293e5,
    3.54813389e5,
    4.46683592e5,
    5.62341325e5,
    7.07945784e5,
    8.91250938e5,
]
testdata_eedges = [
    7.94328235e2,
    1.00000000e3,
    1.25892541e3,
    1.58489319e3,
    1.99526231e3,
    2.51188643e3,
    3.16227766e3,
    3.98107171e3,
    5.01187234e3,
    6.30957344e3,
    7.94328235e3,
    1.00000000e4,
    1.25892541e4,
    1.58489319e4,
    1.99526231e4,
    2.51188643e4,
    3.16227766e4,
    3.98107171e4,
    5.01187234e4,
    6.30957344e4,
    7.94328235e4,
    1.00000000e5,
    1.25892541e5,
    1.58489319e5,
    1.99526231e5,
    2.51188643e5,
    3.16227766e5,
    3.98107171e5,
    5.01187234e5,
    6.30957344e5,
    7.94328235e5,
    1.00000000e6,
]
testdata_ewidths = [
    2.05671765e2,
    2.58925412e2,
    3.25967781e2,
    4.10369123e2,
    5.16624117e2,
    6.50391229e2,
    8.18794045e2,
    1.03080063e3,
    1.29770111e3,
    1.63370890e3,
    2.05671765e3,
    2.58925412e3,
    3.25967781e3,
    4.10369123e3,
    5.16624117e3,
    6.50391229e3,
    8.18794045e3,
    1.03080063e4,
    1.29770111e4,
    1.63370890e4,
    2.05671765e4,
    2.58925412e4,
    3.25967781e4,
    4.10369123e4,
    5.16624117e4,
    6.50391229e4,
    8.18794045e4,
    1.03080063e5,
    1.29770111e5,
    1.63370890e5,
    2.05671765e5,
]


def test_energy_grid_access(mceq_sib21):
    print(mceq_sib21.e_grid)
    print(mceq_sib21.e_bins)
    print(mceq_sib21.e_widths)
    assert np.allclose(mceq_sib21.e_grid, testdata_ecenters, rtol=1e-8, atol=0)
    assert np.allclose(mceq_sib21.e_bins, testdata_eedges, rtol=1e-8, atol=0)
    assert np.allclose(mceq_sib21.e_widths, testdata_ewidths, rtol=1e-8, atol=0)


def test_closest_energy(mceq_sib21):
    closest_energy = np.array(
        [mceq_sib21.closest_energy(energy) for energy in testdata_ecenters]
    )
    assert np.allclose(closest_energy, testdata_ecenters, rtol=0, atol=1e1)


def test_get_solution_grid_idx(mceq_sib21):
    import crflux.models as pm

    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.solve()

    with pytest.raises(Exception):
        mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=0)

    mceq_sib21.solve([0, 1])
    sol0 = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=0)
    sol1 = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=1)
    sol_last = mceq_sib21.get_solution("mu+", mag=0, integrate=True, grid_idx=100)

    assert np.allclose(sol0, 0)
    assert np.all(sol1 != 0)
    assert np.allclose(sol_last, sol1)


def test_get_solution_wrong_particle_name(mceq_sib21):
    from MCEq import config

    config.excpt_on_missing_particle = True
    mceq_sib21.solve()
    with pytest.raises(Exception):
        mceq_sib21.get_solution("proton", mag=0, integrate=True, grid_idx=0)


def test_get_solution_prefixes(mceq_sib21):
    mceq_sib21.solve()
    total = mceq_sib21.get_solution("total_mu+", mag=0, integrate=True)
    conv = mceq_sib21.get_solution("conv_mu+", mag=0, integrate=True)
    prompt = mceq_sib21.get_solution("pr_mu+", mag=0, integrate=True)

    assert np.allclose(total, conv + prompt)


@pytest.mark.parametrize(
    "return_as",
    ["total energy", "kinetic energy", "total momentum", "none"],
)
@pytest.mark.parametrize("integrate", [True, False])
def test_get_solution_return_as(mceq_sib21, return_as, integrate):
    mceq_sib21.solve()

    if return_as == "none":
        with pytest.raises(Exception):
            result = mceq_sib21.get_solution(
                "mu+", return_as=return_as, integrate=integrate
            )
    else:
        result = mceq_sib21.get_solution(
            "mu+", return_as=return_as, integrate=integrate
        )
        if isinstance(result, tuple):
            xgrid, values = result
            assert len(xgrid) == len(values)
            assert np.all(np.isfinite(values))
        else:
            assert np.all(np.isfinite(result))


def test_get_solution_dont_sum_helicities(mceq_sib21):
    mceq_sib21.solve()

    # Get solution with summed helicities (default)
    solution_summed = mceq_sib21.get_solution("e-", dont_sum_helicities=False)
    solution_left_do = mceq_sib21.get_solution("e-_l", dont_sum_helicities=False)
    solution_right_do = mceq_sib21.get_solution("e-_r", dont_sum_helicities=False)

    # Get individual helicity states
    solution_left = mceq_sib21.get_solution("e-_l", dont_sum_helicities=True)
    solution_right = mceq_sib21.get_solution("e-_r", dont_sum_helicities=True)
    solution_0 = mceq_sib21.get_solution("e-", dont_sum_helicities=True)

    # Manual sum should equal the summed solution
    manual_sum = solution_left + solution_right + solution_0

    assert solution_left == approx(solution_left_do)
    assert solution_right == approx(solution_right_do)
    assert solution_summed == approx(manual_sum)


def test_solve_from_integration_path(mceq_sib21):
    # Normal solve
    mceq_sib21.solve()
    solution_normal = np.copy(mceq_sib21._solution)

    # Get the integration path that was used
    nsteps, dX, rho_inv, grid_idcs = mceq_sib21.integration_path

    # Solve using the same integration path
    mceq_sib21.solve_from_integration_path(nsteps, dX, rho_inv, grid_idcs)
    solution_from_path = mceq_sib21._solution

    # Should give identical results
    assert solution_normal == approx(solution_from_path)


@pytest.mark.parametrize(
    "pdg_id",
    [
        2212,
        2112,
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_set_initial_spectrum(mceq_sib21, pdg_id, append):
    cond = mceq_sib21._restore_initial_condition
    phi0_backup = mceq_sib21._phi0.copy()

    particle = mceq_sib21.pman[pdg_id]
    phi0 = mceq_sib21._phi0[particle.lidx : particle.uidx].copy()

    spectrum = np.ones_like(phi0) * 10
    mceq_sib21.set_initial_spectrum(spectrum, pdg_id, append)

    phi1 = mceq_sib21._phi0[particle.lidx : particle.uidx]

    if append:
        assert np.allclose(phi0, phi1 - spectrum)
    if not append:
        assert np.allclose(spectrum, phi1)

    mceq_sib21._phi0 = phi0_backup
    mceq_sib21._restore_initial_condition = cond


msis_atmospheres = [
    "SouthPole",
    "Karlsruhe",
    "Geneva",
    "Tokyo",
    "GranSasso",
    "TelAviv",
    "KSC",
    "SoudanMine",
    "Tsukuba",
    "LynnLake",
    "PeaceRiver",
    "FtSumner",
]

season = ["January", "July"]

corsika_atmospheres = list_available_corsika_atmospheres()


test_densities_cases = []
# MSIS00: all atmospheres
for atmo in msis_atmospheres:
    for s in ["January", "July"]:
        test_densities_cases.append(
            pytest.param("MSIS00", (atmo, s), id=f"MSIS00-{atmo}-{s}")
        )

# MSIS00_IC and AIRS: only SouthPole
for model in ["MSIS00_IC", "AIRS"]:
    for s in ["January", "July"]:
        if model == "AIRS":
            test_densities_cases.append(
                pytest.param(
                    model,
                    ("SouthPole", s),
                    id=f"{model}-SouthPole-{s}",
                    marks=pytest.mark.xfail(reason="Fix issure #71"),
                )
            )
        else:
            test_densities_cases.append(
                pytest.param(model, ("SouthPole", s), id=f"{model}-SouthPole-{s}")
            )

for density_config in corsika_atmospheres:
    test_densities_cases.append(
        pytest.param(
            "CORSIKA",
            density_config,
            id=f"CORSIKA-{density_config}",
        )
    )

test_densities_cases.append(pytest.param("Isothermal", (None, None), id="Isothermal"))
test_densities_cases.append(
    pytest.param("GeneralizedTarget", (), id="GeneralizedTarget")
)
test_densities_cases.append(
    pytest.param(
        "Unknow",
        (),
        id="Unknown",
        marks=pytest.mark.xfail(reason="Fails for uknown density model"),
    )
)

profiles = {
    "MSIS00": dprof.MSIS00Atmosphere,
    "MSIS00_IC": dprof.MSIS00IceCubeCentered,
    "CORSIKA": dprof.CorsikaAtmosphere,
    "AIRS": dprof.AIRSAtmosphere,
    "Isothermal": dprof.IsothermalAtmosphere,
    "GeneralizedTarget": dprof.GeneralizedTarget,
}


@pytest.mark.parametrize("model, density_config", test_densities_cases)
def test_set_density_profile(mceq_sib21, model, density_config):
    mceq_sib21.set_density_model((model, density_config))
    mceq_sib21.solve()

    # test instances instead of str
    profile = profiles[model](*density_config)
    mceq_sib21.set_density_model(profile)
    mceq_sib21.solve()


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


def test_interaction_model_forwarding():
    """Test that user-provided interaction model is correctly forwarded to InteractionCrossSections."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    # Create MCEqRun with a specific interaction model
    mceq_sib21 = MCEqRun(
        interaction_model="QGSJETII04",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )

    # Verify that the interaction model is correctly set
    assert mceq_sib21._int_cs.iam == "QGSJETII04"
    assert mceq_sib21._interactions.iam == "QGSJETII04"


def test_ptot_grid(mceq_sib21):
    # Test without bins
    ptot_centers = mceq_sib21.ptot_grid("mu+", return_bins=False)

    assert len(ptot_centers) == len(mceq_sib21.e_grid)
    assert np.all(ptot_centers > 0)
    assert np.all(np.isfinite(ptot_centers))

    # Test with bins
    ptot_bins, ptot_centers_with_bins = mceq_sib21.ptot_grid("mu+", return_bins=True)

    assert len(ptot_bins) == len(mceq_sib21.e_bins)
    assert len(ptot_centers_with_bins) == len(mceq_sib21.e_grid)
    assert np.allclose(ptot_centers, ptot_centers_with_bins)

    # Check that centers are geometric mean of bins
    expected_centers = np.sqrt(ptot_bins[1:] * ptot_bins[:-1])
    assert np.allclose(ptot_centers_with_bins, expected_centers)


def test_etot_grid(mceq_sib21):
    # Test without bins
    etot_centers = mceq_sib21.etot_grid("mu+", return_bins=False)

    assert len(etot_centers) == len(mceq_sib21.e_grid)
    assert np.all(etot_centers > 0)
    assert np.all(np.isfinite(etot_centers))

    # Test with bins
    etot_bins, etot_centers_with_bins = mceq_sib21.etot_grid("mu+", return_bins=True)

    assert len(etot_bins) == len(mceq_sib21.e_bins)
    assert len(etot_centers_with_bins) == len(mceq_sib21.e_grid)
    assert np.allclose(etot_centers, etot_centers_with_bins)

    # Check that bins are kinetic + mass
    mu_mass = mceq_sib21.pman["mu+"].mass
    expected_bins = mceq_sib21.e_bins + mu_mass
    assert np.allclose(etot_bins, expected_bins)


@pytest.mark.parametrize(
    ["return_as", "expected_method"],
    [
        ["kinetic energy", lambda mceq_sib21: (mceq_sib21.e_bins, mceq_sib21.e_grid)],
        [
            "total energy",
            lambda mceq_sib21: mceq_sib21.etot_grid("mu+", return_bins=True),
        ],
        [
            "total momentum",
            lambda mceq_sib21: mceq_sib21.ptot_grid("mu+", return_bins=True),
        ],
    ],
)
@pytest.mark.parametrize("return_bins", [False, True])
def test_xgrid(mceq_sib21, return_as, expected_method, return_bins):
    result = mceq_sib21.xgrid("mu+", return_as, return_bins=return_bins)

    if return_bins:
        bins, centers = result
        expected_bins, expected_centers = expected_method(mceq_sib21)
        assert np.allclose(bins, expected_bins)
        assert np.allclose(centers, expected_centers)
    else:
        expected_bins, expected_centers = expected_method(mceq_sib21)
        assert np.allclose(result, expected_centers)


def test_xgrid_invalid_return_as(mceq_sib21):
    """Test xgrid raises exception for invalid return_as argument."""
    with pytest.raises(Exception, match="Unknown grid type"):
        mceq_sib21.xgrid("mu+", "invalid_type", return_bins=False)


def test_get_set_state_vector_checkpoint_restore(mceq_sib21):
    mceq_sib21.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_sib21.set_density_model(dprof.CorsikaAtmosphere("BK_USStd"))

    # First solve
    mceq_sib21.set_theta_deg(0.0)
    mceq_sib21.solve()
    order1, state1 = mceq_sib21._get_state_vector()
    solution1 = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Change something and solve again
    mceq_sib21.set_theta_deg(30.0)
    mceq_sib21.solve()
    order2, state2 = mceq_sib21._get_state_vector()
    solution2 = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Solutions should be different
    assert not np.allclose(solution1, solution2, atol=1e-10)
    assert not np.allclose(state1, state2, atol=1e-14)

    # Restore first state directly (without solving)
    mceq_sib21._solution = np.copy(state1)
    solution1_prime = mceq_sib21.get_solution("mu+", mag=0, integrate=True)

    # Should match original solution
    assert np.allclose(solution1, solution1_prime, atol=1e-10)


# ---------------------------------------------------------------------------
# set_zenith_azimuth / set_theta_deg deprecation tests
# ---------------------------------------------------------------------------


def test_set_zenith_azimuth(mceq_sib21):
    """set_zenith_azimuth should set zenith and keep integration_path invalidated."""
    # Ensure an EarthsAtmosphere model is active; a previous test may have left
    # a GeneralizedTarget which does not support angle settings.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))
    mceq_sib21.set_zenith_azimuth(30.0)
    assert mceq_sib21.density_model.theta_deg == 30.0

    # Calling again with same angle should skip recalculation (cached)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mceq_sib21.set_zenith_azimuth(30.0)  # should be no-op
    assert mceq_sib21.density_model.theta_deg == 30.0


def test_set_theta_deg_deprecation(mceq_sib21):
    """set_theta_deg must emit a DeprecationWarning."""
    import warnings

    # Ensure an EarthsAtmosphere model is active; a previous test may have left
    # a GeneralizedTarget which does not support angle settings.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mceq_sib21.set_theta_deg(45.0)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "set_zenith_azimuth" in str(w[0].message)
    assert mceq_sib21.density_model.theta_deg == 45.0


def test_set_zenith_azimuth_with_km3net(mceq_sib21):
    """set_zenith_azimuth passes azimuth to location-coupled MSIS models."""
    km3net_atm = dprof.MSIS00KM3NeTCentered("ORCA", season="January")
    mceq_sib21.set_density_model(km3net_atm)

    mceq_sib21.set_zenith_azimuth(60.0, azimuth_deg=0.0)
    assert mceq_sib21.density_model.theta_deg == 60.0
    assert mceq_sib21.density_model.current_impact_latitude is not None
    assert mceq_sib21.density_model.current_impact_longitude is not None

    # Without azimuth → azimuth-averaging
    mceq_sib21.set_zenith_azimuth(60.0)
    assert mceq_sib21.density_model.theta_deg == 60.0
    assert mceq_sib21.density_model.current_impact_latitude is None

    # Restore the session fixture to a neutral atmosphere so other tests are
    # not affected by the KM3NeT density model we set above.
    mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))
