"""Pin the isolation property of ``tests/conftest.py``.

``mceq_sib21`` applies the shared-run config for the duration of one test;
``_restore_global_config_state`` undoes it afterwards. Before this pin the
autouse fixture took its snapshot *after* the session fixture had mutated
``config``, so its restore preserved the mutation for every later test in the
same xdist worker (``debug_level = 2``, ``mkl_threads = 2`` where MKL is
present, ...).

Both tests run on one worker in file order (``xdist_group`` + ``--dist
loadgroup``): test A takes the fixture and leaves a flag, test B takes no
MCEq fixture and must see the pristine module defaults. Test B fails on the
old conftest, where it inherits test A's config.
"""

import pytest

from MCEq import config

_A_RAN = False


@pytest.mark.xdist_group("conftest_isolation")
def test_a_session_config_applied_while_fixture_is_held(mceq_sib21):
    """While ``mceq_sib21`` is held the shared-run config is in force."""
    global _A_RAN
    assert config.debug_level == 2
    assert config.muon_helicity_dependence is True
    assert config.mceq_db_fname == "mceq_ci_base_air_1d_v2.h5"
    if config.has_mkl:
        assert config.mkl_threads == 2
    _A_RAN = True


@pytest.mark.xdist_group("conftest_isolation")
def test_b_session_config_undone_for_the_next_test():
    """The test after one that held ``mceq_sib21`` sees the session baseline.

    That baseline is the module defaults with one change: the session-scoped
    ``_ci_database`` fixture points ``mceq_db_fname`` at the test database
    before any snapshot is taken, so it is what a restore returns to. The
    values the shared-run config writes on top -- ``debug_level = 2`` and,
    with MKL, ``mkl_threads = 2`` -- are what must be gone here.
    """
    assert _A_RAN, (
        "ordering assumption broken: both tests must run on one worker in file order"
    )
    assert config.debug_level == 1
    assert config.muon_helicity_dependence is True
    assert config.mceq_db_fname == "mceq_ci_base_air_1d_v2.h5"
    assert config.mkl_threads == config._default_threads()
