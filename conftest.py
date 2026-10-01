"""Root pytest configuration.

`pytest_addoption` is only honoured in the rootdir conftest and in the conftest
files of the directories named on the command line, so the suite's own options
live here and work from the repository root.

Nothing here imports MCEq — the file is loaded before collection, including in
jobs that only lint.

It also sets the BLAS thread count of every test process before numpy loads
its BLAS (OpenBLAS reads ``OPENBLAS_NUM_THREADS`` once, at load): the available
CPUs divided among the xdist workers, capped at 16 (``-n 3`` on a 4-vCPU runner
gives 1 thread each). A value set outside pytest is kept. The marker variable
tells a worker to recompute values it inherited from the controller.
"""

from __future__ import annotations

import os

_THREAD_VARS = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_THREADS_FROM_CONFTEST = "MCEQ_TEST_THREADS_FROM_CONFTEST"


def _thread_budget() -> str:
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    workers = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    return str(max(1, min(16, cpus // workers)))


if os.environ.get(_THREADS_FROM_CONFTEST) or not any(
    os.environ.get(_v) for _v in _THREAD_VARS
):
    for _v in _THREAD_VARS:
        os.environ[_v] = _thread_budget()
    os.environ[_THREADS_FROM_CONFTEST] = "1"

MARKERS = (
    "cuda: requires a usable CUDA device, not merely an importable cupy",
    "slow: a solve costing many minutes on production-size databases (needs"
    " --run-slow); not run in CI",
)

#: `(marker, flag)` for every mark this conftest skips unless its flag is
#: given. The marks are independent: an item carrying both needs both flags.
GATED_MARKERS = (("slow", "--run-slow"),)


def pytest_addoption(parser):
    parser.getgroup("mceq").addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="also run tests marked slow (production-size solves that take"
        " many minutes; e.g. the generic-losses NaN repro at 5.6e18 eV)",
    )


def pytest_configure(config):
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Skip each `GATED_MARKERS` item unless its flag is given.

    Deselecting through `addopts = ["-m", "not slow"]` would silently change
    the meaning of every future `-m` expression, including the one in
    .github/workflows/_run_tests.yml.
    """
    import pytest

    for marker, flag in GATED_MARKERS:
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
