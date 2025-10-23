import sys

import numpy as np
import pytest
from pytest import approx
from MCEq.geometry.atmosphere_parameters import list_available_corsika_atmospheres
import MCEq.geometry.density_profiles as dprof


if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


def test_solve_default(mceq_small):
    mceq_small.solve()
    sol = mceq_small.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_skip_integration_path(mceq_small):
    mceq_small.solve(skip_integration_path=True)
    sol = mceq_small.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None


def test_solve_other_grid_var(mceq_small):
    with pytest.raises(NotImplementedError):
        mceq_small.solve(grid_var="Y")


@pytest.mark.parametrize(
    ["int_grid", "grid_shape"],
    [[None, (0,)], [[0, 1], (2, 902)]],
    ids=["no-grid", "with-grid"],
)
def test_solve_int_grid(mceq_small, int_grid, grid_shape):
    mceq_small.solve(int_grid)
    print(mceq_small.grid_sol.shape)
    assert mceq_small.grid_sol.shape == grid_shape


@pytest.mark.parametrize(
    ["leading_process", "lenX"],
    [
        ["decays", 594],
        pytest.param(
            "interactions", -1, marks=pytest.mark.xfail(reason="Fix issue #66")
        ),
        pytest.param("auto", -1, marks=pytest.mark.xfail(reason="Fix issue #66")),
    ],
)
def test_integration_path_leading_process(mceq, leading_process, lenX):
    """Fix this test by resolving issue #66"""
    from MCEq import config

    config.leading_process = leading_process
    int_grid = None
    grid_var = "X"
    mceq._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq.integration_path
    assert integration_path[0] == lenX


def test_integration_path_grid_idcs(mceq):
    int_grid = [0, 1]

    grid_var = "X"
    mceq._calculate_integration_path(int_grid, grid_var)
    integration_path = mceq.integration_path
    grid_idcs = integration_path[-1]
    assert len(grid_idcs) == 2
    assert grid_idcs[0] == 0
    assert grid_idcs[0] < grid_idcs[1]


testdata_theta = [
    [0.0, 5.62504370e-3],
    [30.0, 4.20479234e-3],
    [60.0, 1.36630552e-3],
]

ids_theta = [f"{th[0]}" for th in testdata_theta]


@pytest.mark.parametrize(["theta", "nmu"], testdata_theta, ids=ids_theta)
def test_set_theta_deg(mceq, theta, nmu):
    mceq.set_theta_deg(theta)
    mceq.solve()
    nmu_sol = np.sum(
        mceq.get_solution("mu+", mag=0, integrate=True)
        + mceq.get_solution("mu-", mag=0, integrate=True)
    )
    assert nmu_sol == approx(nmu, abs=1e-11)


testdata_model = [
    ["DPMJETIII191", 64],
    ["DPMJETIII306", 64],
    ["QGSJET01C", 58],
    ["QGSJETII03", 44],
    ["QGSJETII04", 44],
    ["SIBYLL21", 48],
    ["SIBYLL23", 62],
    ["SIBYLL23C03", 62],
    ["SIBYLL23C", 62],
    ["SIBYLL23CPP", 62],
]
ids_model = [f"{model[0]}" for model in testdata_model]


@pytest.mark.parametrize(["model", "n"], testdata_model, ids=ids_model)
def test_set_interaction_model_model(mceq, model, n):
    mceq.set_interaction_model(model)
    n_particles = len(mceq._particle_list)
    assert n_particles == n


def test_set_interaction_model_update_particle_list(mceq):
    mceq.set_interaction_model("DPMJETIII191", update_particle_list=True)
    n_particles = len(mceq._particle_list)
    assert n_particles == 64

    mceq.set_interaction_model("SIBYLL23C", update_particle_list=False)
    n_particles_s = len(mceq._particle_list)

    assert n_particles_s == n_particles


@pytest.mark.parametrize(
    ["particle_list", "projectiles"],
    [
        [None, 19],
        [[(2212, 0)], 1],
    ],
    ids=["None", "proton"],
)
def test_mceq_init_particles_list(particle_list, projectiles):
    from MCEq.core import MCEqRun
    import crflux.models as pm

    mceq = MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        particle_list=particle_list,
    )

    assert sum([p.is_projectile for p in mceq.pman.cascade_particles]) == projectiles


testdata_primary = [
    [1e3, 2.03134720e1, 6.80367347e1, 2.36908717e1],
    [1e6, 1.20365838e4, 2.53158948e4, 6.91213253e3],
    [1e9, 7.09254150e6, 1.20884925e7, 2.87396649e6],
    [5e10, 2.63982133e8, 4.14935240e8, 9.27683105e7],
]
ids_primary = [f"energy={primary[0]}" for primary in testdata_primary]


@pytest.mark.parametrize(
    ["energy", "nmu", "nnumu", "nnue"], testdata_primary, ids=ids_primary
)
def test_single_primary(mceq, energy, nmu, nnumu, nnue):
    mceq.set_interaction_model("SIBYLL23C")
    mceq.set_theta_deg(0.0)
    mceq.set_single_primary_particle(E=energy, pdg_id=2212)
    mceq.solve()
    nmu_sol = np.sum(
        mceq.get_solution("mu+", mag=0, integrate=True)
        + mceq.get_solution("mu-", mag=0, integrate=True)
    )
    nnumu_sol = np.sum(
        mceq.get_solution("numu", mag=0, integrate=True)
        + mceq.get_solution("antinumu", mag=0, integrate=True)
    )
    nnue_sol = np.sum(
        mceq.get_solution("nue", mag=0, integrate=True)
        + mceq.get_solution("antinue", mag=0, integrate=True)
    )

    assert nmu_sol == approx(nmu, rel=1e-8, abs=1e-5)
    assert nnumu_sol == approx(nnumu, rel=1e-8, abs=1e-5)
    assert nnue_sol == approx(nnue, rel=1e-8, abs=1e-5)


testdata_pi0_primary = [
    4.22673568e-14,
    6.07686239e-15,
    -2.62581856e-15,
    2.51119040e-17,
    -2.12746915e-18,
    1.77322773e-20,
    -8.50881199e-22,
    6.61039536e-24,
    -4.09830294e-24,
    3.91022042e-25,
    -3.60445917e-26,
]


def test_single_primary_pdg_corsika(mceq_small):
    mceq_small.set_interaction_model("SIBYLL23C")
    mceq_small.set_theta_deg(0.0)
    mceq_small.set_single_primary_particle(E=1e10, pdg_id=1000020040)
    mceq_small.solve()

    pdg_sol = mceq_small.get_solution("mu+", mag=0, integrate=True)

    mceq_small.set_single_primary_particle(E=1e10, corsika_id=402)
    mceq_small.solve()

    corsika_sol = mceq_small.get_solution("mu+", mag=0, integrate=True)

    assert np.allclose(pdg_sol, corsika_sol)

    # pi0
    mceq_small.set_single_primary_particle(E=1e9, pdg_id=111)
    mceq_small.solve()
    pi0_sol = mceq_small.get_solution("mu+", mag=0, integrate=True)

    assert np.allclose(pi0_sol, testdata_pi0_primary, rtol=1e-6, atol=1e-30)


def test_single_primary_e_too_low(mceq_small):
    mceq_small.set_interaction_model("SIBYLL23C")
    mceq_small.set_theta_deg(0.0)
    with pytest.raises(Exception):
        mceq_small.set_single_primary_particle(E=1e3, pdg_id=2212)


testdata_ecenters = [
    8.91250938e08,
    1.12201845e09,
    1.41253754e09,
    1.77827941e09,
    2.23872114e09,
    2.81838293e09,
    3.54813389e09,
    4.46683592e09,
    5.62341325e09,
    7.07945784e09,
    8.91250938e09,
]
testdata_eedges = [
    7.94328235e08,
    1.00000000e09,
    1.25892541e09,
    1.58489319e09,
    1.99526231e09,
    2.51188643e09,
    3.16227766e09,
    3.98107171e09,
    5.01187234e09,
    6.30957344e09,
    7.94328235e09,
    1.00000000e10,
]
testdata_ewidths = [
    2.05671765e08,
    2.58925412e08,
    3.25967781e08,
    4.10369123e08,
    5.16624117e08,
    6.50391229e08,
    8.18794045e08,
    1.03080063e09,
    1.29770111e09,
    1.63370890e09,
    2.05671765e09,
]


def test_energy_grid_access(mceq_small):
    assert np.allclose(mceq_small.e_grid, testdata_ecenters, rtol=1e-8, atol=0)
    assert np.allclose(mceq_small.e_bins, testdata_eedges, rtol=1e-8, atol=0)
    assert np.allclose(mceq_small.e_widths, testdata_ewidths, rtol=1e-8, atol=0)


def test_closest_energy(mceq_small):
    closest_energy = np.array(
        [mceq_small.closest_energy(energy) for energy in testdata_ecenters]
    )
    assert np.allclose(closest_energy, testdata_ecenters, rtol=0, atol=1e1)


def test_get_solution_grid_idx(mceq_small):
    import crflux.models as pm

    mceq_small.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_small.solve()

    with pytest.raises(Exception):
        mceq_small.get_solution("mu+", mag=0, integrate=True, grid_idx=0)

    mceq_small.solve([0, 1])
    sol0 = mceq_small.get_solution("mu+", mag=0, integrate=True, grid_idx=0)
    sol1 = mceq_small.get_solution("mu+", mag=0, integrate=True, grid_idx=1)
    sol_last = mceq_small.get_solution("mu+", mag=0, integrate=True, grid_idx=100)

    assert np.allclose(sol0, 0)
    assert np.all(sol1 != 0)
    assert np.allclose(sol_last, sol1)


def test_get_solution_wrong_particle_name(mceq_small):
    from MCEq import config

    config.excpt_on_missing_particle = True
    mceq_small.solve()
    with pytest.raises(Exception):
        mceq_small.get_solution("proton", mag=0, integrate=True, grid_idx=0)


def test_get_solution_prefixes(mceq_small):
    mceq_small.solve()
    total = mceq_small.get_solution("total_mu+", mag=0, integrate=True)
    conv = mceq_small.get_solution("conv_mu+", mag=0, integrate=True)
    prompt = mceq_small.get_solution("pr_mu+", mag=0, integrate=True)

    assert np.allclose(total, conv + prompt)


@pytest.mark.parametrize(
    "return_as",
    ["total energy", "kinetic energy", "total momentum", "none"],
)
@pytest.mark.parametrize("integrate", [True, False])
def test_get_solution_return_as(mceq_small, return_as, integrate):
    mceq_small.solve()

    if return_as == "none":
        with pytest.raises(Exception):
            result = mceq_small.get_solution(
                "mu+", return_as=return_as, integrate=integrate
            )
    else:
        result = mceq_small.get_solution(
            "mu+", return_as=return_as, integrate=integrate
        )
        if isinstance(result, tuple):
            xgrid, values = result
            assert len(xgrid) == len(values)
            assert np.all(np.isfinite(values))
        else:
            assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "pdg_id",
    [
        pytest.param(None, marks=pytest.mark.xfail(reason="Fix issue #69")),
        pytest.param(2212, marks=pytest.mark.xfail(reason="Fix issue #69")),
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_set_initial_spectrum(mceq_small, pdg_id, append):
    spectrum = np.ones(42)
    with pytest.raises(Exception):
        mceq_small.set_initial_spectrum(spectrum, pdg_id, append)

    spectrum = np.ones(mceq_small.dim)
    mceq_small.set_initial_spectrum(spectrum, pdg_id, append)

    particle = mceq_small.pman[pdg_id]

    phi0 = mceq_small._phi0[particle.lidx : particle.uidx]
    mceq_small._resize_vectors_and_restore()

    assert np.allclose(phi0, spectrum)


msis_atmospheres = [
    "SouthPole",
    "Karlsruhe",
    "Geneva",
    "Tokyo",
    "SanGrasso",
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
def test_set_density_profile(mceq_small, model, density_config):
    mceq_small.set_density_model((model, density_config))
    mceq_small.solve()

    # test instances instead of str
    profile = profiles[model](*density_config)
    mceq_small.set_density_model(profile)
    mceq_small.solve()


def test_set_mod_pprod(mceq_small):
    def weight(xmat, egrid, pname, value):
        return (1 + value) * np.ones_like(xmat)

    ret = mceq_small.set_mod_pprod(2212, 211, weight, ("a", 0.15))
    assert ret == 1

    assert mceq_small._interactions.mod_pprod[(2212, 211)] is not None
    assert mceq_small._interactions.mod_pprod[(2212, -211)] is not None

    mceq_small.regenerate_matrices()
    mceq_small.solve()

    mceq_small.regenerate_matrices(skip_decay_matrix=True)
    mceq_small.solve()


def test_unset_mod_pprod(mceq_small):
    def weight(xmat, egrid, pname, value):
        return (1 + value) * np.ones_like(xmat)

    mceq_small.set_mod_pprod(2212, 211, weight, ("a", 0.15))

    mceq_small.regenerate_matrices()
    mceq_small.solve()

    mceq_small.unset_mod_pprod()
    assert not mceq_small._interactions.mod_pprod

    mceq_small.set_mod_pprod(2212, 211, weight, ("a", 0.15))

    mceq_small.regenerate_matrices()
    mceq_small.solve()
    mceq_small.unset_mod_pprod(dont_fill=False)
    assert not mceq_small._interactions.mod_pprod


def test_n_particles_energy_cutoff_and_grid(mceq_small):
    import crflux.models as pm

    mceq_small.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_small.solve([0, 1])
    n0 = mceq_small.n_particles("mu+", grid_idx=0)
    n1 = mceq_small.n_particles("mu+", grid_idx=1)
    n_high_cut = mceq_small.n_particles("mu+", grid_idx=1, min_energy_cutoff=1e9)

    assert n0 == 0
    assert n1 > n0
    assert n_high_cut < n1


def test_n_mu_energy_cutoff_and_grid(mceq_small):
    import crflux.models as pm

    mceq_small.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_small.solve([0, 1])
    n0 = mceq_small.n_mu(grid_idx=0)
    n1 = mceq_small.n_mu(grid_idx=1)
    nhigh = mceq_small.n_mu(grid_idx=1, min_energy_cutoff=1e9)

    assert n0 == 0
    assert n1 > 0
    assert nhigh < n1


def test_n_e_energy_cutoff_and_grid(mceq_small):
    import crflux.models as pm

    mceq_small.set_primary_model(pm.HillasGaisser2012, "H3a")
    mceq_small.solve([0, 1])
    n0 = mceq_small.n_e(grid_idx=0)
    n1 = mceq_small.n_e(grid_idx=1)
    nhigh = mceq_small.n_e(grid_idx=1, min_energy_cutoff=1e9)

    assert n0 == 0
    assert n1 > 0
    assert nhigh < n1


@pytest.mark.parametrize(
    "definition",
    [
        pytest.param("primary_e"),
        pytest.param("no_name_definition"),
    ],
)
def test_z_factor(mceq_small, definition):
    mceq_small.solve([0, 1])
    z = mceq_small.z_factor(2212, 211, definition=definition)

    assert isinstance(z, np.ndarray)
    assert z.shape == mceq_small.e_grid.shape
    assert np.all(z >= 0)
    assert np.any(z > 0)


def test_decay_z_factor(mceq_small):
    mceq_small.solve()
    z = mceq_small.decay_z_factor(211, 14)

    assert z.shape == mceq_small.e_grid.shape
    assert np.any(z > 0)
    assert not np.any(np.isnan(z))


def test_interaction_model_forwarding():
    """Test that user-provided interaction model is correctly forwarded to InteractionCrossSections."""
    from MCEq.core import MCEqRun
    import crflux.models as pm

    # Create MCEqRun with a specific interaction model
    mceq = MCEqRun(
        interaction_model="QGSJETII04",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )

    # Verify that the interaction model is correctly set
    assert mceq._int_cs.iam == "QGSJETII04"
    assert mceq._interactions.iam == "QGSJETII04"


def test_interaction_cross_sections_delayed_loading():
    """Test that InteractionCrossSections can be initialized without loading a model."""
    from MCEq.data import HDF5Backend, InteractionCrossSections

    # Create HDF5Backend
    mceq_db = HDF5Backend()

    # Initialize InteractionCrossSections without a model
    int_cs = InteractionCrossSections(mceq_hdf_db=mceq_db, interaction_model=None)

    # Verify that no model is loaded initially
    assert int_cs.iam is None
    assert int_cs.parents is None
    assert int_cs.index_d is None

    # Load a model explicitly
    int_cs.load("SIBYLL23C")

    # Verify that the model is now loaded
    assert int_cs.iam == "SIBYLL23C"
    assert int_cs.parents is not None
    assert int_cs.index_d is not None
