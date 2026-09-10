"""Database identity distinguishes external payloads from generated fixtures."""

import pytest

from ._harness import compare_section, load_section
from .test_goldens import _check_database_identity


def test_external_database_swap_still_fails():
    old = {"databases": {"mceq_db_fname": {"sha256": "old"}}}
    new = {"databases": {"mceq_db_fname": {"sha256": "new"}}}
    with pytest.raises(pytest.fail.Exception, match="different mceq_db_fname"):
        _check_database_identity("solve1d", old, new)


def test_synthetic_container_bytes_are_not_an_identity_gate():
    old = {"databases": {"em_db_fname": {"sha256": "old"}}}
    new = {"databases": {"em_db_fname": {"sha256": "new"}}}
    _check_database_identity("emrho", old, new)


def test_synthetic_operator_change_still_fails():
    arrays, _ = load_section("emrho")
    assert not compare_section("emrho", arrays)
    arrays["yield_operator"] = arrays["yield_operator"].copy()
    arrays["yield_operator"].flat[0] += 1.0
    problems = compare_section("emrho", arrays)
    assert any("yield_operator" in problem for problem in problems)
