"""Compare each golden section against the stored `.npz`.

One test per section. `--regenerate-goldens` rewrites the section instead of
comparing it; `--run-golden-slow` enables the sections that need a database CI
does not carry.

The sections share no state — each generator constructs and tears down its own
`MCEqRun` — but they do mutate process-global `MCEq.config` while they run, so
the whole suite is pinned to one xdist worker.
"""

from __future__ import annotations

import os

import pytest

from . import SECTIONS, SLOW_SECTIONS
from ._harness import CUDA_RTOL, compare_section, load_section, section_path
from .make_goldens import SectionUnavailable, load_generator, regenerate

pytestmark = [pytest.mark.golden, pytest.mark.xdist_group("golden")]

#: Sections whose values come from a solve and can therefore be re-run on
#: another backend. `structure` is static, `paths` needs no solver.
BACKEND_SECTIONS = ("solve1d",)


def _params():
    for section in SECTIONS:
        marks = [pytest.mark.golden_slow] if section in SLOW_SECTIONS else []
        yield pytest.param(section, marks=marks, id=section)


def cuda_available():
    """A usable device, not merely an importable cupy.

    `config.has_cuda` is `find_spec("cupy") is not None`, so it is True on a
    machine with the wheel installed and no driver.
    """
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _pin_openblas():
    """Pin the OpenBLAS pool for the duration of a golden test.

    The single-axis (K=1) secant route is not covered by
    `solvers._secant_blas_thread_limit`, so its dense coupling GEMMs run at the
    ambient pool, and OpenBLAS switches microkernel between 1 and >=2 threads
    (4.6e-9 max-relative). A runner that exports OPENBLAS_NUM_THREADS=1 would
    otherwise fail a bitwise comparison.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    limit = max(2, int(os.environ.get("OPENBLAS_NUM_THREADS", "2")))
    with threadpool_limits(limits=limit, user_api="blas"):
        yield


def _build(section):
    module = load_generator(section)
    try:
        arrays, provenance = module.build()
    except SectionUnavailable as exc:
        pytest.skip(str(exc))
    return arrays, provenance


def _check_database_identity(section, golden_prov, produced_prov):
    """Fail early and legibly when the section ran against a different database.

    Both stanzas are taken under the section's own config pins, so this compares
    the same files. The `.h5` databases are gitignored symlinks into other
    checkouts; without this a swapped database surfaces as a few hundred
    unexplained rel-L2 lines instead of one sentence.
    """
    stored = {k: v.get("sha256") for k, v in (golden_prov.get("databases") or {}).items()}
    current = {k: v.get("sha256") for k, v in (produced_prov.get("databases") or {}).items()}
    for name, digest in stored.items():
        if digest and current.get(name) and digest != current[name]:
            pytest.fail(
                f"golden section {section!r} was generated against a different "
                f"{name}: golden {digest[:16]}, present {current[name][:16]}",
                pytrace=False,
            )


def _report(section, prov, problems, produced):
    """Build the failure message, preferring a generator's own diff renderer."""
    module = load_generator(section)
    renderer = getattr(module, "diff_report", None)
    if renderer is not None:
        golden, _ = load_section(section)
        detail = renderer(golden, produced) or problems
    else:
        detail = problems
    head = (
        f"golden section {section!r} differs from {str(prov.get('git_commit'))[:12]}\n"
        f"note: {prov.get('note', '')}\n"
    )
    return head + "\n".join(f"  - {line}" for line in detail)


@pytest.mark.parametrize("section", list(_params()))
def test_golden_section(section, request):
    selected = request.config.getoption("--golden-section")
    if selected and section not in selected:
        pytest.skip(f"not in --golden-section {selected}")

    if request.config.getoption("--regenerate-goldens"):
        if regenerate(section) is None:
            pytest.skip(f"inputs for section {section!r} are not available here")
        return

    if not section_path(section).exists():
        pytest.skip(
            f"golden section {section!r} has not been generated; "
            f"run `pytest tests/golden --regenerate-goldens`"
        )

    produced, produced_prov = _build(section)
    _, prov = load_section(section)
    _check_database_identity(section, prov, produced_prov)

    problems = compare_section(section, produced)
    if problems:
        pytest.fail(_report(section, prov, problems, produced), pytrace=False)


@pytest.mark.cuda
@pytest.mark.parametrize("section", BACKEND_SECTIONS)
def test_golden_section_cuda_matches_host(section, request, monkeypatch):
    """The CUDA backend must reproduce the host golden to `CUDA_RTOL`.

    cuSPARSE SpMM does not fix its reduction order, so CUDA is not bitwise even
    between two consecutive calls in one process; measured worst case against
    the host file is 9.8e-14 relative L2, four orders inside the budget.
    """
    if request.config.getoption("--regenerate-goldens"):
        pytest.skip("regeneration writes the host golden only")
    if not cuda_available():
        pytest.skip("no usable CUDA device")
    if not section_path(section).exists():
        pytest.skip(f"golden section {section!r} has not been generated")

    from MCEq import config

    monkeypatch.setattr(config, "kernel_config", "cuda_etd2")
    produced, _ = _build(section)
    problems = compare_section(section, produced, rtol_floor=CUDA_RTOL)
    if problems:
        _, prov = load_section(section)
        pytest.fail(_report(section, prov, problems, produced), pytrace=False)


def test_every_section_has_a_generator():
    """`SECTIONS` and the generator table must not drift apart."""
    from .make_goldens import GENERATORS

    assert set(SECTIONS) == set(GENERATORS), (
        f"SECTIONS {sorted(SECTIONS)} != GENERATORS {sorted(GENERATORS)}"
    )
