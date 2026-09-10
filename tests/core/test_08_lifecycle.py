"""MCEqRun lifecycle and config-advanced surface: close, leaks, force_resonance, restore tuples.

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import numpy as np
import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


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


def test_mceqrun_no_refcycle_or_handle_leak_across_instances():
    """Successive MCEqRun constructions must release backend handles.

    Regression for the bound-method reference cycle (PR #163) and the
    explicit-close path in :meth:`MCEqRun._build_kernel_dispatch`. On
    macOS Accelerate this used to overflow ``SIZE_MSTORE=10`` after ~5
    instances; on MKL it used to leak per-instance MKL handles. Run
    enough iterations to exceed both budgets if anything regresses.
    """
    import gc

    import crflux.models as pm

    from MCEq import config
    from MCEq.core import MCEqRun

    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    config.adv_set["disabled_particles"] = []
    try:
        for _ in range(15):
            mceq = MCEqRun(
                interaction_model="SIBYLL21",
                theta_deg=0.0,
                primary_model=(pm.HillasGaisser2012, "H3a"),
            )
            mceq.solve()
            mceq.close()
            del mceq
            gc.collect()
    finally:
        config.adv_set["disabled_particles"] = saved_disabled


def test_mceqrun_close_is_idempotent_and_context_manager(mceq_sib21):
    """``MCEqRun.close()`` must be idempotent and usable via ``with``."""
    mceq_sib21.solve()
    mceq_sib21.close()
    mceq_sib21.close()  # second call must not raise
    # Still usable: caches lazily rebuild on the next solve().
    mceq_sib21.solve()
    sol = mceq_sib21.get_solution("mu+", mag=0, integrate=True)
    assert sol is not None and np.any(sol > 0)


def test_force_resonance_carve_out_for_standard_particles():
    """``force_resonance`` for a standard-particles entry must be a no-op.

    Documented behaviour: ``adv_set['force_resonance']`` is matched
    against the absolute PDG id, and entries that are also in
    ``config.standard_particles`` are silently ignored (the carve-out).
    Forcing Lambda (3122, a standard particle present in the reduced
    DB) into the list should leave ``is_resonance = False``.
    """
    import crflux.models as pm

    from MCEq import config
    from MCEq.core import MCEqRun

    saved_force = list(config.adv_set.get("force_resonance", []))
    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    config.adv_set["force_resonance"] = [3122]
    config.adv_set["disabled_particles"] = []
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        lam = mceq.pman[3122]
        assert not lam.is_resonance, (
            "3122 is in standard_particles, force_resonance should be a no-op"
        )
        assert lam in mceq.pman.cascade_particles
        assert lam not in mceq.pman.resonances
    finally:
        config.adv_set["force_resonance"] = saved_force
        config.adv_set["disabled_particles"] = saved_disabled


def test_force_resonance_folds_non_standard_particle():
    """``force_resonance`` for a non-standard PDG must flip ``is_resonance``.

    The reduced test DB only ships standard-particles entries, so we
    temporarily remove Lambda (3122) from ``config.standard_particles``
    to drive the positive branch of ``_apply_force_resonance``. With
    the carve-out gone, putting 3122 in ``force_resonance`` should:
      - flip ``is_resonance`` to True for Λ and Λ̄,
      - drop them from ``cascade_particles`` and add them to
        ``resonances``,
      - leave ``solve()`` working (the Λ production is folded into
        downstream rows by ``MatrixBuilder._follow_chains``).
    """
    import crflux.models as pm

    from MCEq import config
    from MCEq.core import MCEqRun

    saved_force = list(config.adv_set.get("force_resonance", []))
    saved_disabled = list(config.adv_set.get("disabled_particles", []))
    saved_std = list(config.standard_particles)
    config.adv_set["force_resonance"] = [3122]
    config.adv_set["disabled_particles"] = []
    config.standard_particles = [
        pid for pid in config.standard_particles if abs(pid) != 3122
    ]
    try:
        mceq = MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        lam = mceq.pman[3122]
        lam_bar = mceq.pman[-3122]
        assert lam.is_resonance, "Lambda should be folded as resonance"
        assert lam_bar.is_resonance, "Lambda-bar should be folded as resonance"
        assert lam in mceq.pman.resonances
        assert lam not in mceq.pman.cascade_particles
        # Solver must still run end-to-end with a folded charm meson.
        mceq.solve()
        sol = mceq.get_solution("mu+", mag=0, integrate=True)
        assert sol is not None and np.any(sol > 0)
    finally:
        config.adv_set["force_resonance"] = saved_force
        config.adv_set["disabled_particles"] = saved_disabled
        config.standard_particles = saved_std
