"""Pin for ledger B8 (its duplicate id is B21): the ``DataDrivenModel`` energy
window is dead.

``DataDrivenModel.__init__`` (``src/MCEq/ddm.py``) stores ``self.e_min`` /
``self.e_max``, but ``ddm_matrices`` always calls
``_generate_DDM_matrix(channel=..., mceq=mceq, e_min=mceq.e_bins[0],
e_max=mceq.e_bins[-1])`` -- the full grid off the ``MCEqRun`` -- and never
passes the stored window. The constructor arguments are accepted, stored, and
ignored. Phase 4 ruling R3 pins this behaviour as-is; it is NOT fixed here.
What these tests freeze is the **no-op**: a model built with a strict window
returns exactly the full-grid matrix, bitwise.

Correct behaviour, whichever way the fix lands: either wire ``self.e_min`` /
``self.e_max`` through ``ddm_matrices`` into ``_generate_DDM_matrix``, or
delete the dead attributes and stop accepting the arguments. Both flip
``test_b8_stored_window_is_ignored_bitwise`` -- the first because the windowed
model would then return a windowed (different, smaller) matrix instead of the
full-grid one, the second because constructing a model with ``e_min=`` /
``e_max=`` would no longer be possible. Test 3 is the mutation stand-in that
proves the two behaviours are observably distinct, so the fix (or a pin
change) can never pass silently.

Prior documentation of the defect: the prose note in the module docstring of
``tests/golden/gen_species.py`` ("A third defect, not in the plan's ledger:
``DataDrivenModel.e_min`` / ``e_max`` ... are stored and never read"). This
file makes that note executable.

The window machinery underneath is real: ``_generate_DDM_matrix`` honours its
own ``e_min``/``e_max`` arguments via ``_eval_energy_cuts``. Test 2 pins that
mechanism -- in the spirit of the B20/B5 mechanism tests in
``tests/test_data_bug_pins.py`` -- so the no-op pin cannot be passing merely
because every window collapses onto the full grid.
"""

import numpy as np
import pytest

# The two strict windows both pins probe. Measured against the session
# ``mceq_sib21`` run (reduced DB, grid d = 31, e_bins[0] = 794.3..., e_bins[-1]
# = 1e6): the first lies entirely below the grid -- ``_eval_energy_cuts`` snaps
# each edge to the nearest grid centre, so both cut indices land on the lowest
# bin and the read collapses to a 2-bin block at the bottom of the grid -- the
# second is strictly inside it and yields a genuine 22-bin sub-grid.
WINDOWS = [(10.0, 100.0), (2e3, 2e5)]


@pytest.fixture(scope="module")
def ddm_full():
    """One-channel model with the default (dead) ``-1.0``/``-1.0`` window.

    Built once per module and reused: ``ddm_matrices`` only reads from the
    ``MCEqRun`` it is handed, and the shared session run never mutates, so
    repeated calls on one instance are stable.
    """
    from MCEq import ddm

    return ddm.DataDrivenModel(
        enable_channels=[(2212, 211)],
        enable_K0_from_isospin=False,
    )


def _single_channel(model):
    """The model's one ``_DDMChannel``, checked rather than blindly indexed."""
    channels = list(model.spline_db.channels)
    assert len(channels) == 1
    return channels[0]


# B8/B21 -- stored window ignored by ddm_matrices


@pytest.mark.parametrize("window", WINDOWS, ids=["below-grid", "in-grid"])
def test_b8_stored_window_is_ignored_bitwise(mceq_sib21, ddm_full, window):
    """B8: a windowed model returns the full-grid matrix, bitwise identical.

    This is the no-op pin. For each strict window the model stores it
    (``w.e_min`` / ``w.e_max`` hold ``lo`` / ``hi``) and ``ddm_matrices``
    still answers with the ``(31, 31)`` full-grid matrix, byte-identical to
    the default-window model's -- on the grid-external window *and* on the
    strictly in-grid one.

    A fix that wires the window must make this test fail on both windows (the
    returned matrix would become the truncated / sub-grid one from test 2); a
    fix that deletes the attributes fails at construction. Neither outcome may
    be met with a regenerated expectation here without a ledger update.
    """
    from MCEq import ddm

    lo, hi = window
    full_matrices = ddm_full.ddm_matrices(mceq_sib21)
    assert set(full_matrices) == {(2212, 211)}
    assert full_matrices[(2212, 211)].shape == (31, 31)

    w = ddm.DataDrivenModel(
        e_min=lo,
        e_max=hi,
        enable_channels=[(2212, 211)],
        enable_K0_from_isospin=False,
    )
    assert (w.e_min, w.e_max) == (lo, hi)

    w_matrices = w.ddm_matrices(mceq_sib21)
    assert set(w_matrices) == {(2212, 211)}
    assert np.array_equal(w_matrices[(2212, 211)], full_matrices[(2212, 211)])


def test_b8_direct_generator_does_honour_the_windows(mceq_sib21, ddm_full):
    """B8, mechanism check: the generator below honours the very same windows.

    Not a pin of intended behaviour for ``DataDrivenModel`` -- it calls
    ``_generate_DDM_matrix`` directly, bypassing the dead layer. Without it,
    the no-op pin above would be indistinguishable from "the generator
    ignores windows too", i.e. the whole window concept being fake rather
    than one dead call site. Measured on the session run: the below-grid
    window ``(10, 100)`` snaps both cut indices to the lowest bin and
    collapses to a ``(2, 2)`` block at the bottom of the grid, the in-grid
    window ``(2e3, 2e5)`` gives a genuine ``(22, 22)`` sub-grid.
    Both differ from the ``(31, 31)`` full matrix; the in-grid one is
    strictly smaller.

    This test pins the generator, not B8 -- a B8 fix in ``ddm_matrices``
    leaves it green; only a change to ``_generate_DDM_matrix`` or
    ``_eval_energy_cuts`` can flip it, and that is a different ledger event.
    """
    from MCEq import ddm_utils

    ref = ddm_full.ddm_matrices(mceq_sib21)[(2212, 211)]
    channel = _single_channel(ddm_full)

    below = ddm_utils._generate_DDM_matrix(
        channel=channel, mceq=mceq_sib21, e_min=10.0, e_max=100.0
    )
    assert below.shape == (2, 2)
    assert not np.array_equal(below, ref)

    inside = ddm_utils._generate_DDM_matrix(
        channel=channel, mceq=mceq_sib21, e_min=2e3, e_max=2e5
    )
    assert inside.shape == (22, 22)
    assert inside.shape[0] < ref.shape[0]
    assert not np.array_equal(inside, ref)


def test_b8_fixed_semantics_differ_from_todays_noop(mceq_sib21):
    """B8, mutation stand-in: fixed semantics are observably distinct today.

    Replicates in-test what a *fixed* ``ddm_matrices`` would compute for a
    windowed model -- iterate ``spline_db.channels`` (one channel here) and
    call ``_generate_DDM_matrix`` with the model's own stored ``e_min`` /
    ``e_max`` -- and asserts the result differs from what the model actually
    returns today, using the in-grid window ``(2e3, 2e5)`` (the ``(22, 22)``
    sub-grid vs the ``(31, 31)`` full matrix).

    This is the machine-checkable half of the plan's "a mutation that *fixes*
    the no-op must flip the pin" requirement: it proves the windowed and
    full-grid behaviours are distinct, so wiring the window through
    ``ddm_matrices`` cannot leave every test in this file green.
    """
    from MCEq import ddm, ddm_utils

    w = ddm.DataDrivenModel(
        e_min=2e3,
        e_max=2e5,
        enable_channels=[(2212, 211)],
        enable_K0_from_isospin=False,
    )
    todays = w.ddm_matrices(mceq_sib21)

    channel = _single_channel(w)
    assert (channel.projectile, channel.secondary) == (2212, 211)
    fixed = {
        (channel.projectile, channel.secondary): ddm_utils._generate_DDM_matrix(
            channel=channel, mceq=mceq_sib21, e_min=w.e_min, e_max=w.e_max
        )
    }

    assert set(fixed) == set(todays) == {(2212, 211)}
    assert not np.array_equal(fixed[(2212, 211)], todays[(2212, 211)])
