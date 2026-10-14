"""The initial condition survives a rebuild of the species."""

import numpy as np
import pytest

crflux = pytest.importorskip("crflux.models")


def test_restore_replay_refills_phi0_after_a_model_change(mceq_sib21):
    """A forced reload deposits the same initial state on the new species."""
    mceq_sib21.set_primary_model(crflux.HillasGaisser2012, "H3a")
    assert mceq_sib21._phi0.sum() > 0.0
    before = mceq_sib21._phi0.copy()

    mceq_sib21.set_interaction_model("SIBYLL21", force=True)
    try:
        assert mceq_sib21._phi0.shape == before.shape
        assert np.allclose(mceq_sib21._phi0, before)
    finally:
        mceq_sib21._phi0 = before
