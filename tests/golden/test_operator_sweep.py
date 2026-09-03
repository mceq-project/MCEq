"""Structure of the two ``operators`` sections, read off the stored files.

The generators assert their own invariants while they build, which only runs
when a section is regenerated or compared — and ``operators2d`` needs a 329 MB
database CI does not carry. These tests read the committed ``.npz`` instead, so
the discriminating power of the sweep is checked on every platform and in every
job, including the ones that skip both sections.

What they state: the sweep covers the whole cross product; ``dec_m`` sees
neither knob; muon multiple scattering is a no-op in 1D and moves every stencil
in 2D; the seven stencils are mutually distinct; the 28 cells therefore take 21
distinct ``int_m`` values; and each section's tolerance table matches exactly
the keys it holds — ``em_step_scale``, the one key of either section that is
not bitwise, which only ``operators1d`` records. A section that stopped
discriminating — because a refactor made the builder ignore
``config.loss_stencil_method``, say — would still compare green against its own
regenerated self, and these are what catch that.

Two other properties of the sections need a build and are therefore not here:
that the construct-once shortcut equals a fresh ``MCEqRun``, and that the
assembled operators are canonical CSRs. Both live in
``tests/test_operators_pin.py``, on the reduced database.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import SECTIONS
from ._harness import load_section, section_path
from ._operator_sweep import CELLS, SCATTERING, STENCILS

pytestmark = pytest.mark.golden

#: ``(section, is_2d)`` — the two databases the sweep runs on.
OPERATOR_SECTIONS = (("operators1d", False), ("operators2d", True))


def _arrays(section):
    if not section_path(section).exists():
        pytest.skip(f"golden section {section!r} has not been generated")
    arrays, _ = load_section(section)
    return arrays


def _int_digests(arrays):
    return {label: str(arrays[f"cells/{label}/int_m/data"]) for label, _, _ in CELLS}


def test_operator_sections_are_registered():
    """Both sections are in `SECTIONS` and have a generator."""
    from .make_goldens import GENERATORS

    for section, _ in OPERATOR_SECTIONS:
        assert section in SECTIONS
        assert section in GENERATORS


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_sweep_covers_the_cross_product(section, is_2d):
    """Every (stencil, scattering) cell is present, and no other."""
    arrays = _arrays(section)
    found = {
        key.split("/", 1)[1].rsplit("/", 2)[0]
        for key in arrays
        if key.startswith("cells/") and key.endswith("/int_m/data")
    }
    assert found == {label for label, _, _ in CELLS}
    assert len(found) == len(STENCILS) * len(SCATTERING) == 14
    assert int(arrays["fixture/is_2d"]) == int(is_2d)


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_dec_m_is_invariant_under_both_knobs(section, is_2d):
    """One `dec_m` over all 14 cells.

    The stencil reaches `int_m` through `cont_loss_operator` and the muon
    damping through `_csr_from_blocks(apply_muon_scattering=True)`; the decay
    path sets neither, so a `dec_m` that moved would mean the sweep had leaked
    into the decay blocks.
    """
    arrays = _arrays(section)
    digests = {
        label: tuple(
            str(arrays[f"cells/{label}/dec_m/{part}"])
            for part in ("shape", "nnz", "dtype", "data", "indices", "indptr")
        )
        for label, _, _ in CELLS
    }
    assert len(set(digests.values())) == 1, sorted(digests.items())


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_muon_scattering_is_a_no_op_in_1d_only(section, is_2d):
    """`_muon_scattering_damping` returns None unless `is_2d`.

    So in 1D the two cells of a stencil are the same operator, and in 2D they
    differ at every stencil — `-kappa^2 theta_s^2(E) / 4` on the muon diagonals
    of every mode with kappa != 0.
    """
    digests = _int_digests(_arrays(section))
    for stencil in STENCILS:
        on, off = digests[f"{stencil}/ms_on"], digests[f"{stencil}/ms_off"]
        if is_2d:
            assert on != off, stencil
        else:
            assert on == off, stencil


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_every_stencil_gives_its_own_operator(section, is_2d):
    """The seven families are distinct in `op_matrix` and in `int_m`."""
    arrays = _arrays(section)
    matrices = {s: np.asarray(arrays[f"stencil/{s}/op_matrix"]) for s in STENCILS}
    for i, a in enumerate(STENCILS):
        for b in STENCILS[i + 1 :]:
            assert not np.array_equal(matrices[a], matrices[b]), (a, b)

    digests = _int_digests(arrays)
    for ms_label, _ in SCATTERING:
        column = [digests[f"{s}/{ms_label}"] for s in STENCILS]
        assert len(set(column)) == len(STENCILS), (ms_label, column)


def test_the_two_sections_take_21_distinct_int_m_values():
    """7 of 14 in 1D, 14 of 14 in 2D: the count the sweep is worth."""
    counts = {}
    for section, is_2d in OPERATOR_SECTIONS:
        digests = _int_digests(_arrays(section))
        counts[section] = len(set(digests.values()))
        assert counts[section] == len(STENCILS) * (len(SCATTERING) if is_2d else 1)
    assert sum(counts.values()) == 21, counts


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_em_step_scale_is_recorded_only_where_it_has_content(section, is_2d):
    """One key per cell in 1D, none in 2D, and no tolerance entry either way.

    On the 2D fixture ``disabled_particles = [11, -11]`` leaves gamma the only
    ``is_em`` species and ``enable_em`` is off, so the EM off-diagonal block is
    empty and the value was exactly 0.0 in all fourteen cells — a property of
    ``adv_set``, already pinned by the compared ``fixture/em_species`` array,
    rather than of the assembly. The rel-L2 1e-9 entry those keys carried never
    applied: ``compare_key`` short-circuits an all-zero reference to exact
    equality. So a tolerance table matching no key is the defect this also
    guards, in either direction — an entry a reader would credit and the
    harness would never resolve.
    """
    if not section_path(section).exists():
        pytest.skip(f"golden section {section!r} has not been generated")
    arrays, prov = load_section(section)
    expected = {f"cells/{label}/em_step_scale" for label, _, _ in CELLS}
    found = {key for key in arrays if key.endswith("/em_step_scale")}
    if is_2d:
        assert found == set()
        assert list(arrays["fixture/em_species"]) == ["gamma"]
    else:
        assert found == expected
        assert np.any([float(arrays[key]) != 0.0 for key in found])

    table = prov.get("tolerances") or {}
    assert set(table) == found, (
        f"{section}: the tolerance table and the em_step_scale keys disagree; "
        f"an entry matching no key is a bound the harness never resolves"
    )


@pytest.mark.parametrize("section,is_2d", OPERATOR_SECTIONS)
def test_provenance_summary_matches_the_arrays(section, is_2d):
    """The `extra` stanza the generator wrote describes the file it wrote."""
    if not section_path(section).exists():
        pytest.skip(f"golden section {section!r} has not been generated")
    arrays, prov = load_section(section)
    extra = prov["extra"]
    assert extra["stencils"] == list(STENCILS)
    assert extra["cells"] == [label for label, _, _ in CELLS]
    assert extra["n_cells"] == len(CELLS)
    assert extra["n_distinct_dec_m"] == 1
    assert extra["n_distinct_int_m"] == len(set(_int_digests(arrays).values()))
    assert extra["int_m_digests"] == _int_digests(arrays)
