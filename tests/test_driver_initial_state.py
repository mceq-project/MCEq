"""Contracts for the Phase 6 driver split (driver/initial_state.py).

The behaviour-preservation of the move is carried by the existing suites
(test_core / test_solvers hit the facade API); this file pins the two
properties the split *creates*: the restore replay must run facade methods
by name through the resize, and the facade ``_phi0`` surface must be the
InitialState one (identity, not a copy), so generator-style assignment
still works. Both tests run against the session-shared ``mceq_sib21`` and
restore the run's initial condition afterwards (there is no run-level
autouse fixture; the restore list makes the re-setup cheap).
"""

import numpy as np
import pytest

crflux = pytest.importorskip("crflux.models")


def test_restore_replay_refills_phi0_after_a_model_change(mceq_sib21):
    """The facade resize replays the stored names — phi0 refills.

    A forced interaction-model reload must leave the initial condition in
    place: ``set_interaction_model`` ends in
    ``_resize_vectors_and_restore``, whose phi0 reallocation and replay
    live on ``InitialState`` since commit 3. If the replay were broken,
    phi0 would come back all-zero and this test goes red — the
    behavioural counterpart of the structural names-pin in
    ``test_core.test_restore_initial_condition_uses_method_names``.
    """
    mceq_sib21.set_primary_model(crflux.HillasGaisser2012, "H3a")
    assert mceq_sib21._phi0.sum() > 0.0
    before = mceq_sib21._phi0.copy()

    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    try:
        assert mceq_sib21._phi0.shape == before.shape
        assert np.allclose(mceq_sib21._phi0, before)
    finally:
        mceq_sib21._phi0 = before


def test_facade_phi0_is_the_initial_state_vector(mceq_sib21):
    """``_phi0`` delegates by object, and assignment rebinds the source.

    ``MCEqRun._phi0`` reads ``InitialState.phi0`` — same object — and the
    setter (used by tests/golden generators and test_solvers_2d to
    install a prepared vector) writes it back where readers see it.
    """
    saved = mceq_sib21._phi0
    sentinel = np.zeros(saved.shape)
    try:
        mceq_sib21._phi0 = sentinel
        assert mceq_sib21._initial_state.phi0 is sentinel
        assert mceq_sib21._phi0 is sentinel
    finally:
        mceq_sib21._phi0 = saved
