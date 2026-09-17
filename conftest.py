"""Root pytest configuration.

`pytest_addoption` is only honoured in the rootdir conftest and in the conftest
files of the directories named on the command line, so the suite's own options
live here and work from the repository root.

Nothing here imports MCEq — the file is loaded before collection, including in
jobs that only lint.

It also sets the BLAS thread budget of every test process, and it must: this
file is the first thing pytest imports in the controller and in every xdist
worker, before numpy loads its BLAS (OpenBLAS reads ``OPENBLAS_NUM_THREADS``
once, at load). The budget is the CPUs this process may use divided among the
workers (``PYTEST_XDIST_WORKER_COUNT``, 1 in the controller and without xdist),
capped at 16: ``-n 8`` on a 48-core node runs 8 x 6 threads, ``-n 3`` on a
4-vCPU runner 3 x 1. ``MCEq.config._default_threads()`` reads the same
variables, so library and tests agree on one budget. An explicit outer setting
(a scheduler, a CI job) is respected; the marker variable tells a worker that
the values it inherited came from the controller's own computation and must be
redone with the worker count.
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
    " --run-slow); the full gate passes it deliberately, CI never does",
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
