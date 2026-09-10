"""Stability of the continuous-loss band with generic losses on all charged.

Repro and regression trio for the ``generic_losses_all_charged=True`` blowup
documented in ``runs/2026-09-08_nan-generic-losses-repro`` of the
mceq-em-integration harness: with the flag on, a coupled hadronic run on the
wide (1 MeV floor, 10 bins/decade) v13 grid went non-finite at X=440 -- ETD2
exponentiates only the diagonal and treats the loss stencil explicitly, so a
row with one-step stiffness ``dX_max * |dEdX| / (E * dlnE) >> 1`` sits in the
expfit interior's unstable regime and grows from round-off. The fix makes the
monotone upwind layer of the ``expfit_low_upwind*`` composites
species-adaptive in ``MatrixBuilder.cont_loss_operator`` and clamps the
generic proton curve to its tabulated boost range in
``ParticleManager.set_continuous_losses``.

The fixture mirrors the repro driver (p @ 5.6e9 GeV, theta 60, e_min 1e-3,
numpy_etd2, ``disabled_particles`` cleared as in the Xmax-3panel driver whose
observable is the e+- deposit). It is built from local run-artifact databases
in the harness repo, so all three tests skip when those paths are absent.

The solve test carries ``@pytest.mark.slow`` (its name is deliberate, not a
registered marker -- nothing deselects it yet; the full gate runs it, ~30 s)
and the two operator-level tests are the default-on guards. The tests share
one fixture instance through an xdist group, so the parallel gate builds the
run once per worker, not once per test.
"""

import pathlib

import numpy as np
import pytest

HARNESS = pathlib.Path("/ceph/sharedfs/work/SATORI/anatoli/devel/mceq-em-integration")
#: v13 compact DB with the 1 MeV low-energy extension (the grid the blowup
#: was reproduced on; ``generic`` table 0.00146 .. 11.6 boost).
V13_DB = (
    HARNESS
    / "runs/2026-05-25_em-db-v13-1mev/outputs/mceq_db_v13_1mev_compact_lext_DPMJETIII193.h5"
)
#: PPM EM companion DB (enable_em is on, as in the repro driver).
EM_DB = (
    HARNESS / "runs/2026-06-26_ppm-prod-realistic/outputs/mceq_db_real-ppm-10dec_EM.h5"
)

pytestmark = pytest.mark.xdist_group("generic_losses_stability")


@pytest.fixture(scope="module")
def v13_hadronic_run():
    """The repro driver: coupled hadronic p @ 5.6e18 eV, theta 60, numpy_etd2.

    Config exactly as in ``inputs/nan_repro.py`` plus the driver's
    ``disabled_particles = []`` (the shipped default keeps e+- out of the
    tracking set, and the solve test asserts an e+--bearing cascade). No
    ``loss_stencil_low_upwind_rows`` override: this file pins that the
    adaptive layer replaces the manual rows=24 stop-gap.
    """
    if not (V13_DB.is_file() and EM_DB.is_file()):
        pytest.skip("v13 / EM databases are harness-local run artifacts")
    import MCEq.config as config
    from MCEq.core import MCEqRun
    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    saved = {
        key: getattr(config, key)
        for key in (
            "debug_level",
            "mceq_db_fname",
            "em_db_fname",
            "data_dir",
            "kernel_config",
            "e_min",
            "e_max",
            "enable_em",
            "enable_em_ion",
            "enable_energy_loss",
            "enable_default_tracking",
            "generic_losses_all_charged",
            "muon_helicity_dependence",
            "average_loss_operator",
            "loss_stencil_method",
        )
    }
    saved_adv = dict(config.adv_set)
    try:
        config.enable_em = True
        config.debug_level = 0
        config.enable_energy_loss = True
        config.enable_em_ion = True
        config.e_min = 1e-3
        config.e_max = 1.1e14
        config.enable_default_tracking = False
        config.loss_stencil_method = "expfit_low_upwind2"
        config.muon_helicity_dependence = False
        config.average_loss_operator = False
        config.generic_losses_all_charged = True
        config.mceq_db_fname = str(V13_DB)
        config.data_dir = pathlib.Path("")
        config.kernel_config = "numpy_etd2"
        config.em_db_fname = str(EM_DB)
        config.adv_set["disabled_particles"] = []

        density_model = CorsikaAtmosphere("BK_USStd", None)
        density_model.set_theta(60.0)
        run = MCEqRun(
            interaction_model="SIBYLL23E",
            theta_deg=60.0,
            primary_model=None,
            density_model=density_model,
        )
        yield run
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv)


@pytest.mark.slow
def test_generic_losses_all_charged_finite(v13_hadronic_run):
    """The repro, end to end: the whole grid survives the solve.

    Before the fix the p+ block blew geometrically from round-off
    (finite fraction 0.21-0.23, onset X=440-445); with the adaptive upwind
    layer the solve is fully finite.
    """
    run = v13_hadronic_run
    run.set_single_primary_particle(E=5.6e9, pdg_id=2212)
    xg = np.arange(0.0, float(run.density_model.max_X) + 0.1, 5.0)
    run.solve(int_grid=xg, dX_max=1.0)
    assert np.isfinite(np.asarray(run.grid_sol, dtype=np.float64)).all()
    assert np.isfinite(run.get_solution("total_mu+", integrate=True)).all()


def test_generic_dedx_in_range_matches_table(v13_hadronic_run):
    """The clamped generic spline: exact on the table, never above it.

    Inside the tabulated boost interval the p+-from-generic value must
    reproduce the table; outside it the value is the edge value, never the
    35x-extrapolated one the unclamped log-log spline gave at the top of the
    grid.
    """
    run = v13_hadronic_run
    bg, dedx = run._system._cont_losses.generic_table
    p = run.pman["pi+"]
    d = -np.asarray(p.dEdX)
    e_grid = np.asarray(run.e_grid, dtype=np.float64)
    mass = p.mass
    bgp = np.sqrt((e_grid + mass) ** 2 - mass**2) / mass
    inside = (bgp >= bg[0]) & (bgp <= bg[-1])
    assert np.isfinite(d).all() and (d > 0).all()
    assert inside.any() and (~inside).any(), "fixture must straddle the table"
    assert np.allclose(
        d[inside],
        np.exp(np.interp(np.log(bgp[inside]), np.log(bg), np.log(dedx))),
        rtol=1e-6,
    )
    assert d[~inside].max() <= dedx.max()


def test_loss_band_one_step_map_is_contractive(v13_hadronic_run):
    """Cheap proxy of the solve test: the isolated hadron loss block at
    h = dX_max has no eigen-blowup.

    The ETD2 ETD1 proxy ``G = exp(h d) (1 + h phi1 Off/d)`` on the loss band
    alone, per species block; before the fix rho(p+) was 2.467 at h=1 and
    2.662 at h=20. e+- are excluded: their block is stepped under the Cure-B
    cap (``operators/compiled.em_step_scale``), not at ``etd2_path["dX_max"]``,
    and its documented blowup caveat is a separate, contained story.
    """
    run = v13_hadronic_run
    matrix_builder = run.matrix_builder
    matrix_builder.construct_matrices()
    h = float(run._cfg.etd2_path["dX_max"])
    for p in run.pman.cascade_particles:
        if not p.has_contloss or p.is_em:
            continue
        block = matrix_builder.int_m[p.lidx : p.uidx, p.lidx : p.uidx].toarray()
        d = np.diag(block)
        z = h * d
        e_d = np.exp(z)
        phi1 = np.where(np.abs(z) > 1e-8, (e_d - 1) / np.where(z == 0, 1, z), 1.0)
        g = np.diag(e_d) + (h * phi1)[:, None] * (block - np.diag(d))
        rho = float(np.max(np.abs(np.linalg.eigvals(g))))
        assert rho <= 1.0 + 1e-6, f"{p.name}: one-step rho {rho}"
