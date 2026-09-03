"""Golden section ``operators1d``: the operator sweep on the 1D reduced DB.

Fixture: SIBYLL21 on ``mceq_db_v140reduced_compact.h5`` with the conftest
``mceq_sib21`` config — ``disabled_particles = []`` so the EM cascade is in the
system, muon helicity dependence on. 72 cascade species, n_k 1, dim 31,
dim_states 2232, ``int_m`` 169786 nonzeros and ``dec_m`` 54795.

This section is NOT ``golden_slow``: CI carries this database, so the seven
stencil families and the assembly path become real CI coverage rather than a
host-only pin. It is ``golden_host`` — the CSR buffers are sha256 digests, and
a digest is bitwise by construction.

Fourteen cells (7 stencils x muon scattering on/off) at 3.1 s. Seven distinct
``int_m`` digests: ``_muon_scattering_damping`` returns ``None`` unless
``is_2d``, so the two scattering cells of a stencil are the same operator here,
and this section is what pins that they are. See :mod:`._operator_sweep` for
the construct-once equivalence measurement and the rest of the rationale.

``em_step_scale`` is the interesting scalar of this section: with e+/e-/gamma
loaded the EM off-diagonal block is 7 species x 31 bins and carries the
continuous-loss bands, so the spectral radius moves with the stencil — seven
distinct values, 0.0454 (``upwind``) to 0.1181 (``centered``) 1/(g/cm^2), and
0.0958 on the default ``expfit_low_upwind2``. The 2D section's counterpart is
identically 0 (``disabled_particles = [11, -11]`` leaves gamma as the only EM
species, and with ``enable_em`` off it has no self-production), which is why
the value is pinned here and only guarded there.

It is also the only key of this section that is not bitwise: see
:data:`._operator_sweep.EM_SCALE_RTOL` for the thread measurement behind its
1e-9 bound.
"""

from __future__ import annotations

from ._operator_sweep import build_section

SECTION = "operators1d"

#: Config globals this section fixes. The autouse fixture in tests/conftest.py
#: restores six of them, so a section built inside a full pytest session has to
#: set — and afterwards restore — the rest itself. ``loss_stencil_method`` and
#: ``muon_multiple_scattering`` are pinned to the production defaults and then
#: swept by :data:`._operator_sweep.CELLS`; they are listed so the sweep starts
#: from a known state and the teardown puts it back.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": "mceq_db_v140reduced_compact.h5",
    "e_min": 0.1,
    "e_max": 1e11,
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

ADV_SET_PINS = {
    "disabled_particles": [],
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
    "interaction_model": "SIBYLL21",
    "theta_deg": 0.0,
    "primary_model": None,
}

NOTE = (
    "Solve-free operator sweep on the 1D SIBYLL21 reduced DB with the conftest"
    " mceq_sib21 config (disabled_particles [], muon helicity on): int_m,"
    " dec_m and the differential operator over the seven loss_stencil_method"
    " families x muon_multiple_scattering on/off. One MCEqRun, then per cell"
    " MatrixBuilder._construct_differential_operator() +"
    " construct_matrices(skip_decay_matrix=False); measured identical to a"
    " fresh MCEqRun per cell in 14 of 14 cells. The explicit differential-"
    "operator call is required: core.py builds op_matrix in"
    " MatrixBuilder.__init__ only, so regenerate_matrices() alone does not pick"
    " up a stencil change (pinned by tests/test_operators_pin.py). Seven"
    " distinct int_m digests over the 14 cells and one dec_m digest:"
    " _muon_scattering_damping returns None unless is_2d, so the scattering"
    " flag is a no-op on a 1D database, and neither knob is on the decay path."
    " em_step_scale takes seven distinct values here, 0.0454 (upwind) to 0.1181"
    " (centered) 1/(g/cm^2), on the 217 x 217 e+/e-/gamma off-diagonal block."
    " BLAS threads are ambient: rebuilding at 1, 2, 4 and 8 reproduces every"
    " key bitwise except em_step_scale, which is max|eigvals| of a dense"
    " non-normal block through LAPACK and moves by up to 6.9e-13 relative"
    " (expfit_low_upwind2). It is therefore the one key not on bitwise, at"
    " rel-L2 1e-9 — 1400x the measured spread, against a 1.5e-3 relative gap"
    " between the two closest stencils."
)


def build():
    """Produce (arrays, provenance) for the 1D operator sweep."""
    return build_section(
        SECTION,
        config_pins=CONFIG_PINS,
        adv_set_pins=ADV_SET_PINS,
        run_kwargs=RUN_KWARGS,
        note=NOTE,
    )
