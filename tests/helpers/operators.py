"""Building the same operator under each loss stencil.

Shared by the operator behaviour tests so they agree on which stencils
exist and on the settings each one is built under.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

#: Config globals set for the operator builds and restored afterwards (the
#: autouse fixture in tests/conftest.py restores only some of them).
#: ``loss_stencil_method`` and ``muon_multiple_scattering`` start at the
#: production defaults and are then varied per build.
CONFIG_PINS = {
    "debug_level": 0,
    "override_debug_fcn": [],
    "print_module": False,
    "mceq_db_fname": "mceq_ci_base_air_1d_v2.h5",
    "mceq_db_manifest": "mceq_db_manifest_ci_v2.yaml",
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


#: Every interior stencil ``MatrixBuilder._construct_differential_operator``
#: accepts (:data:`MCEq.operators.loss_stencil.STENCIL_METHODS`), in the order
#: of the dispatch:
#: the two ``expfit_low_upwind*`` composites, the three high-order families,
#: then the two monotone upwind operators. ``tests/test_operators_pin.py``
#: reads the accepted names out of the ``ValueError`` the method raises and
#: fails if this tuple and the source drift apart.
STENCILS = (
    "expfit_low_upwind2",
    "expfit_low_upwind",
    "expfit",
    "centered",
    "biased",
    "upwind",
    "upwind2",
)

#: ``(label, config.muon_multiple_scattering)`` for the second axis.
SCATTERING = (("ms_on", True), ("ms_off", False))


@contextmanager
def pinned_config(config_pins, adv_set_pins):
    """Set the given config globals, then restore them.

    Fails if a name is no longer a config global.
    """
    from MCEq import config

    missing = sorted(key for key in config_pins if not hasattr(config, key))
    assert not missing, f"config globals absent, pin block is stale: {missing}"

    saved_config = {key: getattr(config, key) for key in config_pins}
    saved_adv_set = copy.deepcopy(config.adv_set)
    try:
        for key, value in config_pins.items():
            setattr(config, key, value)
        config.adv_set.update(adv_set_pins)
        yield config
    finally:
        for key, value in saved_config.items():
            setattr(config, key, value)
        config.adv_set.clear()
        config.adv_set.update(saved_adv_set)


def build_cell_operators(mceq, stencil, scattering, average=False):
    """Assemble ``(int_m, dec_m)`` for one cell on an already-built run.

    Reuses the run instead of constructing a new one per setting;
    ``tests/test_operators_pin.py`` checks that both give the same operator.

    All three globals are set on every cell, ``average_loss_operator``
    included, so the averaged cell cannot leak into the one built after it.

    ``_construct_differential_operator()`` is explicit because it is called
    from ``MatrixBuilder.__init__`` and nowhere else, so
    ``construct_matrices`` alone refills the blocks against the ``op_matrix``
    the constructor left behind. ``skip_decay_matrix`` stays False so ``dec_m``
    is rebuilt and its invariance is measured, not assumed.
    """
    from MCEq import config

    config.loss_stencil_method = stencil
    config.muon_multiple_scattering = scattering
    config.average_loss_operator = average
    builder = mceq.matrix_builder
    builder._construct_differential_operator()
    return builder.construct_matrices(skip_decay_matrix=False)
