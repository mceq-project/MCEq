"""Golden section ``operators2d``: the operator sweep on the 2D FLUKA rc7 DB.

Fixture: FLUKA20251 on ``mceq_db_v2_fluka2d_rc7.h5`` under the config of
:mod:`.gen_solve2d` — ``e_max`` 1e2 so the grid is 31 bins,
``disabled_particles = [11, -11]``. 70 cascade species, n_k 48, dim_states
2170, stitched 104160 with 6212352 nonzeros in ``int_m`` and 2432400 in
``dec_m``.

Before this section the 2D operators were pinned by nothing: ``solve2d`` pins
61 keys and carries no ``int_m`` or ``dec_m`` at all, so every 2D assembly
path — the per-mode COO scatter, the ``block_diag`` stitch, the kappa^2 muon
damping — reached the goldens only through a solve.

``golden_slow``: the 329 MB database is not published on the MCEq releases
page and CI does not cache it. ``golden_host``: the CSR buffers are sha256
digests.

Fourteen cells (7 stencils x muon scattering on/off) at 19.4 s — 5.3 s to
construct the run, 0.94 s per cell. All fourteen ``int_m`` digests are
distinct: unlike 1D, ``_muon_scattering_damping`` fires here and adds
``-kappa^2 theta_s^2(E) / 4`` to the muon diagonals of every mode with
kappa != 0. ``dec_m`` still takes one value over all fourteen.

``em_step_scale`` is identically 0 on this fixture and is pinned as such:
``disabled_particles = [11, -11]`` leaves gamma as the only ``is_em`` species,
and with ``enable_em`` off it has no self-production, so the EM off-diagonal
block is empty. ``_em_cascade_step_scale`` also indexes with ``p.lidx`` /
``p.uidx``, which are offsets inside one ``dim_states`` block, so on a 2D
operator it reads mode 0 alone — pinned here as it behaves today. Its rel-L2
entry (:data:`._operator_sweep.EM_SCALE_RTOL`) never bites: an all-zero
reference cannot support a relative bound, so ``compare_key`` demands exact
equality of these fourteen keys. The 1D section is where the value is pinned.
"""

from __future__ import annotations

import pathlib

from ._operator_sweep import build_section
from .make_goldens import SectionUnavailable

SECTION = "operators2d"

#: 329 MB, linked into src/MCEq/data by hand — see :mod:`.gen_solve2d`.
DB_NAME = "mceq_db_v2_fluka2d_rc7.h5"

#: Config globals this section fixes. Same block as gen_solve2d minus the
#: solver, path and secant knobs, which no assembly path reads.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": DB_NAME,
    "density_model": ("CORSIKA", ("USStd", None)),
    "r_E": 6371.315e3,
    "h_obs": 0.0,
    "h_atm": 112.8e3,
    "e_min": 1e-1,
    "e_max": 1e2,
    "floatlen": None,
    "interaction_medium": "air",
    "A_target": "auto",
    "enable_em": False,
    "em_air_density": None,
    "enable_em_ion": True,
    "enable_energy_loss": True,
    "enable_cont_rad_loss": True,
    "generic_losses_all_charged": True,
    "average_loss_operator": False,
    "loss_step_for_average": 1e-1,
    "loss_stencil_method": "expfit_low_upwind2",
    "loss_stencil_low_upwind_rows": 8,
    "loss_stencil_alpha0": 3.0,
    "muon_multiple_scattering": True,
    "muon_helicity_dependence": True,
    "use_isospin_sym": True,
    "assume_nucleon_interactions_for_exotics": True,
    "enable_default_tracking": True,
    "fallback_to_air_cs": True,
    "prompt_ctau": 2.6842,
    "minimal_primary_energy": 3.0,
    "em_step_dense_eig_max": 4000,
    "low_energy_extension": {
        "model": None,
        "he_le_transition": 80,
        "he_le_trwidth": 0.3,
        "use_unknown_cs": True,
    },
}

#: e+/e- stay out of the system, as in gen_solve2d.
ADV_SET_PINS = {
    "disabled_particles": [11, -11],
    "disable_interactions_of_unstable": False,
    "disable_charm_pprod": False,
    "allowed_projectiles": [],
    "disable_direct_leptons": False,
    "disable_leading_mesons": False,
    "disable_decays": [],
    "force_resonance": [],
    "forced_int_cs": None,
    "replace_meson_cross_sections_with": None,
}

RUN_KWARGS = {
    "interaction_model": "FLUKA20251",  # "FLUKA" raises Unknown selections
    "theta_deg": 30.0,
    "primary_model": None,
    "density_model": CONFIG_PINS["density_model"],
}

NOTE = (
    "Solve-free operator sweep on the 2D FLUKA rc7 DB (FLUKA20251,"
    " disabled_particles [11,-11], e_max 1e2 so the grid is 31 bins, n_k 48,"
    " stitched 104160): int_m, dec_m and the differential operator over the"
    " seven loss_stencil_method families x muon_multiple_scattering on/off."
    " One MCEqRun, then per cell MatrixBuilder._construct_differential_"
    "operator() + construct_matrices(skip_decay_matrix=False); measured"
    " identical to a fresh MCEqRun per cell in 14 of 14 cells (19.4 s against"
    " 57.3 s). The explicit differential-operator call is required: core.py"
    " builds op_matrix in MatrixBuilder.__init__ only, so regenerate_matrices()"
    " alone does not pick up a stencil change. All fourteen int_m digests are"
    " distinct — _muon_scattering_damping fires on a 2D database and adds"
    " -kappa^2 theta_s^2 / 4 to the muon diagonals of every mode with kappa"
    " != 0 — against one dec_m digest, which neither knob reaches. BLAS threads"
    " are ambient: rebuilding at 1, 2, 4 and 8 reproduces every key bitwise."
    " em_step_scale is identically 0 here (gamma is the only is_em species and"
    " has no self-production with enable_em off) and is pinned as such;"
    " _em_cascade_step_scale indexes with the dim_states-block offsets"
    " p.lidx/p.uidx, so on a stitched 2D operator it reads mode 0 alone. It"
    " carries the rel-L2 1e-9 entry the 1D section needs, but an all-zero"
    " reference supports no relative bound, so these fourteen keys are compared"
    " exactly like the rest."
)


def _require_db(config):
    """Path of the 2D database, or ``SectionUnavailable`` if it is not here.

    ``MCEqRun.__init__`` calls ``config.ensure_db_available``, which would try
    to download a file the releases page does not carry, so the check has to
    come before the constructor.
    """
    path = pathlib.Path(config.data_dir) / DB_NAME
    if not path.exists():
        raise SectionUnavailable(
            f"{DB_NAME} is not at {path}. The 329 MB 2D FLUKA rc7 database is "
            "not published on the MCEq releases page and is not cached by CI; "
            "link it into src/MCEq/data to build this section."
        )
    return path


def build():
    """Produce (arrays, provenance) for the 2D operator sweep."""
    from MCEq import config

    db_path = _require_db(config)
    return build_section(
        SECTION,
        config_pins=CONFIG_PINS,
        adv_set_pins=ADV_SET_PINS,
        run_kwargs=RUN_KWARGS,
        note=NOTE,
        extra={"db_path": str(db_path)},
    )
