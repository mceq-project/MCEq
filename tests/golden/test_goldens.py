"""Compare each golden section against the stored `.npz`.

One test per section. `--regenerate-goldens` rewrites the section instead of
comparing it; `--run-golden-slow` enables the sections that need a database CI
does not carry; `--run-golden-host` enables the sections pinned bitwise to the
generating host's numpy/BLAS build, which only the reference job passes.

The sections share no state — each generator constructs and tears down its own
`MCEqRun` — but they do mutate process-global `MCEq.config` while they run, so
the whole suite is pinned to one xdist worker.
"""

from __future__ import annotations

import pytest

from . import HOST_SECTIONS, SECTIONS, SLOW_SECTIONS
from ._harness import CUDA_RTOL, compare_section, load_section, section_path
from .make_goldens import SectionUnavailable, load_generator, regenerate

pytestmark = [pytest.mark.golden, pytest.mark.xdist_group("golden")]

#: Sections whose values come from a solve and can therefore be re-run on
#: another backend. `structure` is static, `paths` needs no solver.
BACKEND_SECTIONS = ("solve1d",)


def _params():
    for section in SECTIONS:
        marks = []
        if section in SLOW_SECTIONS:
            marks.append(pytest.mark.golden_slow)
        if section in HOST_SECTIONS:
            marks.append(pytest.mark.golden_host)
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
    stored = {
        k: v.get("sha256") for k, v in (golden_prov.get("databases") or {}).items()
    }
    current = {
        k: v.get("sha256") for k, v in (produced_prov.get("databases") or {}).items()
    }
    for name, digest in stored.items():
        if digest and current.get(name) and digest != current[name]:
            pytest.fail(
                f"golden section {section!r} was generated against a different "
                f"{name}: golden {digest[:16]}, present {current[name][:16]}",
                pytrace=False,
            )


def _environment_note(golden_prov, produced_prov):
    """One line naming the build the golden was generated with and this one.

    A diagnostic, never a gate: the sections in `HOST_SECTIONS` are bitwise
    against the generating host's numpy/BLAS, so a 1e-14 mismatch is usually
    explained here, and the line says `same` when it is not. Making an
    environment difference fail on its own would turn a numpy patch bump red.
    """

    def signature(prov):
        env = prov.get("environment") or {}
        return (
            f"numpy {env.get('numpy')} / scipy {env.get('scipy')} / "
            f"{env.get('platform')}"
        )

    golden, current = signature(golden_prov), signature(produced_prov)
    verdict = "same" if golden == current else "differs"
    return f"environment ({verdict}): golden {golden}; here {current}"


def _report(section, prov, problems, produced, produced_prov):
    """Build the failure message: the mismatches, a generator's own diff, the
    environment.

    Concatenated, not either-or: for the flux sections the mismatch lines name
    the keys and the renderer explains them (which species, at which energy),
    and one without the other is half a report.
    """
    module = load_generator(section)
    detail = list(problems)
    renderer = getattr(module, "diff_report", None)
    if renderer is not None:
        golden, _ = load_section(section)
        detail += renderer(golden, produced)
    head = (
        f"golden section {section!r} differs from {str(prov.get('git_commit'))[:12]}\n"
        f"note: {prov.get('note', '')}\n"
    )
    body = "\n".join(f"  - {line}" for line in detail)
    return f"{head}{body}\n{_environment_note(prov, produced_prov)}"


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
        pytest.fail(
            _report(section, prov, problems, produced, produced_prov), pytrace=False
        )


@pytest.mark.cuda
@pytest.mark.parametrize("section", BACKEND_SECTIONS)
def test_golden_section_cuda_matches_host(section, request, monkeypatch):
    """The CUDA backend must reproduce the host golden to `CUDA_RTOL`.

    cuSPARSE SpMM does not fix its reduction order, so CUDA is not bitwise even
    between two consecutive calls in one process; measured worst case against
    the host file is 9.8e-14 relative L2, four orders inside the budget.

    The flux keys are the exception: `compare_section` leaves a key stored on
    `per_species_max` at its own bound, because a per-species maximum over the
    bins that carry flux is the backend-independent statement and an L2 floor
    on top of it would only hide a species the reordering actually moved.
    """
    if request.config.getoption("--regenerate-goldens"):
        pytest.skip("regeneration writes the host golden only")
    if not cuda_available():
        pytest.skip("no usable CUDA device")
    if not section_path(section).exists():
        pytest.skip(f"golden section {section!r} has not been generated")

    from MCEq import config

    monkeypatch.setattr(config, "kernel_config", "cuda_etd2")
    produced, produced_prov = _build(section)
    problems = compare_section(section, produced, rtol_floor=CUDA_RTOL)
    if problems:
        _, prov = load_section(section)
        pytest.fail(
            _report(section, prov, problems, produced, produced_prov), pytrace=False
        )


def test_every_section_has_a_generator():
    """`SECTIONS` and the generator table must not drift apart."""
    from .make_goldens import GENERATORS

    assert set(SECTIONS) == set(GENERATORS), (
        f"SECTIONS {sorted(SECTIONS)} != GENERATORS {sorted(GENERATORS)}"
    )
