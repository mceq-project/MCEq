"""Shared scheduling constraint for the ex-test_core files.

The files share one session-scoped MCEqRun (tests/conftest.py) and rely on
collection order, exactly as the single test_core.py did. Under ``-n 8``
--dist loadgroup, without an explicit group each file could land on a
different worker and run out of order, exposing the order-coupling (e.g.
set_mod_pprod before test_single_primary). Marking the whole directory as
one xdist group reproduces the single-file semantics: one worker, serial,
in collection order.
"""

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.xdist_group("mceq-core"))
