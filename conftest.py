"""Root pytest configuration.

`pytest_addoption` is only honoured in the rootdir conftest and in the conftest
files of the directories named on the command line, so the golden-harness
options live here: `pytest --regenerate-goldens` from the repository root has to
work, not just `pytest tests/golden --regenerate-goldens`.

Nothing here imports MCEq — the file is loaded before collection, including in
jobs that only lint.
"""

from __future__ import annotations

MARKERS = (
    "golden: regression against a stored golden section under tests/golden/data",
    "golden_slow: golden section costing minutes or a database CI does not carry"
    " (needs --run-golden-slow)",
    "cuda: requires a usable CUDA device, not merely an importable cupy",
)


def pytest_addoption(parser):
    group = parser.getgroup("golden", "golden regression harness")
    group.addoption(
        "--regenerate-goldens",
        action="store_true",
        default=False,
        help="rewrite tests/golden/data/*.npz from the current tree instead of comparing",
    )
    group.addoption(
        "--run-golden-slow",
        action="store_true",
        default=False,
        help="also run golden sections marked golden_slow (2D FLUKA fixtures, minutes)",
    )
    group.addoption(
        "--golden-section",
        action="append",
        default=[],
        metavar="NAME",
        help="restrict the golden suite to these sections (repeatable)",
    )


def pytest_configure(config):
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Skip `golden_slow` items unless the flag is given.

    Deselecting through `addopts = ["-m", "not golden_slow"]` would silently
    change the meaning of every future `-m` expression, including the one in
    .github/workflows/_run_tests.yml.
    """
    if config.getoption("--run-golden-slow"):
        return
    import pytest

    skip = pytest.mark.skip(reason="needs --run-golden-slow")
    for item in items:
        if "golden_slow" in item.keywords:
            item.add_marker(skip)
