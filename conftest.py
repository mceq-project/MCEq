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
    "golden_host: golden section pinned bitwise to the numpy/BLAS build of the"
    " host that generated it, compared on the reference job only"
    " (needs --run-golden-host)",
    "cuda: requires a usable CUDA device, not merely an importable cupy",
)

#: `(marker, flag)` for every mark this conftest skips unless its flag is
#: given. The marks are independent: an item carrying both needs both flags.
GATED_MARKERS = (
    ("golden_slow", "--run-golden-slow"),
    ("golden_host", "--run-golden-host"),
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
        "--run-golden-host",
        action="store_true",
        default=False,
        help="also run golden sections marked golden_host (bitwise against the"
        " generating host's numpy/BLAS build; the reference job passes it)",
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
    """Skip each `GATED_MARKERS` item unless its flag is given.

    Deselecting through `addopts = ["-m", "not golden_slow"]` would silently
    change the meaning of every future `-m` expression, including the one in
    .github/workflows/_run_tests.yml.
    """
    import pytest

    # `golden_host` says a section's values are only comparable on the host that
    # produced them, which says nothing about producing them: skipping it while
    # regenerating would silently rewrite `structure` alone. `golden_slow` still
    # applies, because it stands for a database the machine may not have.
    regenerating = config.getoption("--regenerate-goldens")

    for marker, flag in GATED_MARKERS:
        if config.getoption(flag) or (regenerating and marker == "golden_host"):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
