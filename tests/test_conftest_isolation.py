"""Pin the isolation property of ``tests/conftest.py``.

``mceq_sib21`` applies the shared-run config for the duration of one test;
``_restore_global_config_state`` undoes it afterwards. Before this pin the
autouse fixture took its snapshot *after* the session fixture had mutated
``config``, so its restore preserved the mutation for every later test in the
same xdist worker (``disabled_particles = []``, ``mkl_threads = 2`` where MKL
is present, the reduced database, ...).

Both tests run on one worker in file order (``xdist_group`` + ``--dist
loadgroup``): test A takes the fixture and leaves a flag, test B takes no
MCEq fixture and must see the pristine module defaults. Test B fails on the
old conftest, where it inherits test A's config.
"""

import os

import pytest

from MCEq import config

_A_RAN = False


@pytest.mark.xdist_group("conftest_isolation")
def test_a_session_config_applied_while_fixture_is_held(mceq_sib21):
    """While ``mceq_sib21`` is held the shared-run config is in force."""
    global _A_RAN
    assert config.adv_set["disabled_particles"] == []
    assert config.muon_helicity_dependence is True
    assert config.mceq_db_fname == "mceq_db_v140reduced_compact.h5"
    if config.has_mkl:
        assert config.mkl_threads == 2
    _A_RAN = True


@pytest.mark.xdist_group("conftest_isolation")
def test_b_session_config_undone_for_the_next_test():
    """The test after one that held ``mceq_sib21`` sees the module defaults.

    The expected values are the literals in ``src/MCEq/config/__init__.py``:

        mceq_db_fname = "mceq_db_lext_dpm193_v140.h5"
        mkl_threads = min(16, os.cpu_count() or 1)
        muon_helicity_dependence = True
        "disabled_particles": [11, -11],  # inside adv_set
    """
    assert _A_RAN, (
        "ordering assumption broken: both tests must run on one worker in file order"
    )
    assert config.adv_set["disabled_particles"] == [11, -11]
    # Pins the module default only: `_apply_session_config` writes the same
    # True, so this line cannot detect the leak. The discriminating assertions
    # are disabled_particles, mceq_db_fname and mkl_threads.
    assert config.muon_helicity_dependence is True
    assert config.mceq_db_fname == "mceq_db_lext_dpm193_v140.h5"
    assert config.mkl_threads == min(16, os.cpu_count() or 1)
