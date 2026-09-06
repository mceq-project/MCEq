"""Pins for the ``report()`` failure logic of ``scripts/check_module_size.py``.

The gate ratchets in both directions since stage B: an allowlisted entry is a
recorded *current* size, not an upper bound, so both growth and shrinking of a
listed module are problems (the latter demanding a re-pin of the entry). Every
test feeds ``report()`` a synthetic size map and a synthetic ``ALLOW`` table
patched onto the imported module -- no filesystem, no repo state, so the pins
survive any refactor of the tree the real gate walks.
"""

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_module_size",
    pathlib.Path(__file__).parents[1] / "scripts" / "check_module_size.py",
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


@pytest.fixture
def gate_table(monkeypatch):
    """The gate with a synthetic budget; each test overwrites ``ALLOW``."""
    monkeypatch.setattr(gate, "LIMIT", 700)
    monkeypatch.setattr(gate, "ALLOW", {})
    return gate


def test_exact_match_of_recorded_and_measured_is_green(gate_table):
    """Recorded == measured above the limit is the green state (stage B rule).

    An entry that sits exactly on its measured size produces no problem lines:
    the ratchet neither grants headroom nor demands a re-pin.
    """
    gate_table.ALLOW = {"src/MCEq/core.py": 2808}

    assert gate_table.report({"src/MCEq/core.py": 2808, "src/MCEq/other.py": 5}) == []


def test_grown_allowlisted_module_fails_with_the_over_by_amount(gate_table):
    """Measured above recorded: growth over a ratcheted entry fails.

    The failure line names the module and states how far over the recorded
    entry it sits, the same ``+`` rendering as the unallowlisted case.
    """
    gate_table.ALLOW = {"src/MCEq/ddm.py": 870}

    problems = gate_table.report({"src/MCEq/ddm.py": 900})

    assert any("src/MCEq/ddm.py" in line for line in problems)
    assert any("+30 over 870" in line for line in problems)


def test_shrunk_allowlisted_module_demands_a_re_pin(gate_table):
    """Recorded above measured (both over LIMIT): the NEW stage-B direction.

    An entry that only shrunk must be re-pinned to the measured size; the line
    renders as an ALLOW-table edit with a ``re-pin`` marker, distinct from the
    ``delete`` rendering of stale entries.
    """
    gate_table.ALLOW = {"src/MCEq/core.py": 3538}

    problems = gate_table.report({"src/MCEq/core.py": 2808})

    assert problems
    assert any(
        '  - "src/MCEq/core.py": 3538,   # now 2808 lines; re-pin' == line
        for line in problems
    )
    assert not any("delete these lines" in line for line in problems)


def test_allowlisted_module_now_under_limit_is_a_stale_entry(gate_table):
    """Measured at or below LIMIT: the entry is stale and must be deleted.

    Unchanged direction from before stage B: the line renders without the
    ``re-pin`` marker and the advice section says to delete it from ALLOW.
    """
    gate_table.ALLOW = {"src/MCEq/ddm.py": 870}

    problems = gate_table.report({"src/MCEq/ddm.py": 640})

    assert any('"src/MCEq/ddm.py": 870,   # now 640 lines' in line for line in problems)
    assert any("delete these lines from ALLOW" in line for line in problems)
    assert not any("re-pin" in line for line in problems)


def test_allowlist_entry_for_a_missing_file_is_stale(gate_table):
    """Path in ALLOW but absent from the size map: delete the entry."""
    gate_table.ALLOW = {"src/MCEq/gone.py": 900}

    problems = gate_table.report({"src/MCEq/core.py": 900})

    assert any(
        '"src/MCEq/gone.py": 900,   # file no longer exists' in line
        for line in problems
    )
    assert any("delete these lines from ALLOW" in line for line in problems)


def test_unallowlisted_module_over_limit_fails_with_the_over_by(gate_table):
    """A module over LIMIT with no ALLOW entry: unchanged first-direction fail.

    The line names the module and the amount over the plain limit.
    """
    gate_table.ALLOW = {}

    problems = gate_table.report({"src/MCEq/newcomer.py": 742})

    assert any("src/MCEq/newcomer.py" in line for line in problems)
    assert any("+42 over 700" in line for line in problems)


def test_the_old_upper_bound_semantics_is_dead(gate_table):
    """Stage-B regression pin: recorded 730 above measured is a failure.

    This is the pre-stage-B core.py situation (recorded 3538, measured 2808),
    which the old upper-bound reading let pass. Under recorded == measured the
    gate must not be green, so a regression to upper-bound semantics turns
    this red.
    """
    gate_table.ALLOW = {"src/MCEq/core.py": 2808 + 730}

    assert gate_table.report({"src/MCEq/core.py": 2808}) != []
