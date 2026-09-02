"""Unit tests for the golden harness comparison layer.

The golden sections themselves are compared by `test_goldens.py`; this module
tests the machinery that decides *how* a key is compared -- `tolerance_for`,
`compare_key`, `compare_section` -- against hand-built arrays, so a change to
the dispatch shows up here rather than as a shifted tolerance on a 400 KB
`.npz`.

`test_dispatch_*` and `test_compare_key_*` pin the two modes that predate the
per-species flux metric (`bitwise`, `rel_l2`) and the `rtol_floor` override the
CUDA comparison uses; `test_per_species_*`, `test_trim_*` and `test_flux_*`
cover the metric of plan D18 on synthetic arrays and on the committed 1D
golden.

The synthetic arrays are a few bins long, so the tests that are about the floor
or the guard pass `trim=0` explicitly and leave the edge trim to the
`test_trim_*` group; a default trim of 3 would otherwise leave them one bin to
score.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from . import _flux_metric, _harness
from ._harness import (
    HOST_RTOL,
    compare_key,
    compare_section,
    save_section,
    tolerance_for,
)

pytestmark = pytest.mark.golden


# --------------------------------------------------------------------------
# tolerance_for
# --------------------------------------------------------------------------


def test_dispatch_default_is_bitwise():
    """A key with no entry is compared bitwise at rtol 0."""
    assert tolerance_for("a/b", {"tolerances": {}}) == ("bitwise", 0.0)
    assert tolerance_for("a/b", {}) == ("bitwise", 0.0)


def test_dispatch_exact_entry_wins():
    prov = {"tolerances": {"a/b": {"mode": "rel_l2", "rtol": 1e-7}}}
    assert tolerance_for("a/b", prov) == ("rel_l2", 1e-7)
    assert tolerance_for("a/bc", prov) == ("bitwise", 0.0)


def test_dispatch_entry_defaults():
    """An entry names the mode and the budget; both have defaults."""
    prov = {
        "tolerances": {"a/b": {}, "c/d": {"rtol": 1e-5}, "e/f": {"mode": "bitwise"}}
    }
    assert tolerance_for("a/b", prov) == ("rel_l2", HOST_RTOL)
    assert tolerance_for("c/d", prov) == ("rel_l2", 1e-5)
    assert tolerance_for("e/f", prov) == ("bitwise", HOST_RTOL)


def test_dispatch_prefix_must_end_in_slash():
    prov = {"tolerances": {"state/": {"rtol": 1e-9}, "spec": {"rtol": 1e-3}}}
    assert tolerance_for("state/single", prov) == ("rel_l2", 1e-9)
    assert tolerance_for("spectrum/x", prov) == ("bitwise", 0.0)


def test_dispatch_longest_prefix_wins():
    prov = {"tolerances": {"a/": {"rtol": 1e-3}, "a/b/": {"rtol": 1e-9}}}
    assert tolerance_for("a/b/c", prov) == ("rel_l2", 1e-9)
    assert tolerance_for("a/z", prov) == ("rel_l2", 1e-3)


def test_dispatch_exact_entry_beats_a_matching_prefix():
    """How a section keeps `state/` on the flux metric and its norms on L2."""
    prov = {
        "tolerances": {
            "state/": {"mode": "per_species_max", "rtol": 3e-8},
            "state/single_mode_l2": {"mode": "rel_l2", "rtol": 1e-11},
        }
    }
    assert tolerance_for("state/single_mode0", prov) == ("per_species_max", 3e-8)
    assert tolerance_for("state/single_mode_l2", prov) == ("rel_l2", 1e-11)


# --------------------------------------------------------------------------
# compare_key: structural checks, shared by every mode
# --------------------------------------------------------------------------


def test_compare_key_shape_before_values():
    problem = compare_key("k", np.zeros(3), np.zeros(4), "bitwise", 0.0)
    assert "shape (4,) != golden (3,)" in problem


def test_compare_key_dtype_kind():
    problem = compare_key("k", np.zeros(3), np.zeros(3, dtype=int), "bitwise", 0.0)
    assert "dtype kind" in problem


def test_compare_key_text_values():
    assert compare_key("k", np.array("a"), np.array("a"), "bitwise", 0.0) is None
    assert "text/object value" in compare_key(
        "k", np.array("a"), np.array("b"), "bitwise", 0.0
    )


# --------------------------------------------------------------------------
# compare_key: bitwise
# --------------------------------------------------------------------------


def test_compare_key_bitwise_equal():
    x = np.array([1.0, 2.0, np.nan])
    assert compare_key("k", x, x.copy(), "bitwise", 0.0) is None


def test_compare_key_bitwise_one_ulp():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, np.nextafter(2.0, 3.0)])
    problem = compare_key("k", x, y, "bitwise", 0.0)
    assert "not bitwise equal" in problem
    assert "rel-L2" in problem and "max elementwise" in problem


def test_compare_key_bitwise_rejects_precision_change():
    x = np.zeros(3, dtype=np.float64)
    problem = compare_key("k", x, x.astype(np.float32), "bitwise", 0.0)
    assert "bitwise mode" in problem


# --------------------------------------------------------------------------
# compare_key: rel_l2
# --------------------------------------------------------------------------


def test_compare_key_rel_l2_inside_and_outside():
    x = np.array([1.0, 1.0])
    y = np.array([1.0, 1.0 + 1e-10])
    assert compare_key("k", x, y, "rel_l2", 1e-9) is None
    problem = compare_key("k", x, y, "rel_l2", 1e-12)
    assert problem.startswith("k: rel-L2")


def test_compare_key_rel_l2_ignores_dtype_width():
    x = np.zeros(3, dtype=np.float64)
    assert compare_key("k", x, x.astype(np.float32), "rel_l2", 1e-9) is None


def test_compare_key_rel_l2_all_zero_golden_requires_exactness():
    zero = np.zeros(3)
    assert compare_key("k", zero, zero.copy(), "rel_l2", 1.0) is None
    problem = compare_key("k", zero, np.array([0.0, 1e-30, 0.0]), "rel_l2", 1.0)
    assert "all-zero" in problem


def test_compare_key_rel_l2_non_finite_pattern():
    x = np.array([1.0, np.nan])
    y = np.array([1.0, 2.0])
    assert "non-finite pattern differs" in compare_key("k", x, y, "rel_l2", 1.0)


def test_compare_key_rel_l2_drops_matching_non_finite():
    x = np.array([1.0, np.inf, np.nan])
    y = np.array([1.0 + 1e-13, np.inf, np.nan])
    assert compare_key("k", x, y, "rel_l2", 1e-12) is None


# --------------------------------------------------------------------------
# compare_section and the rtol_floor override
# --------------------------------------------------------------------------

#: Two species of two bins each, so a state is four entries long. Short enough
#: that the section's tolerance entries have to say `trim_top_bins: 0`.
UNIT_LAYOUT = {
    "meta/species": np.array(["0:a", "1:b"], dtype="U32"),
    "meta/dim": np.asarray(2),
    "meta/dim_states": np.asarray(4),
    "meta/e_grid": np.array([10.0, 100.0]),
    "meta/e_bins": np.array([3.0, 30.0, 300.0]),
}


@pytest.fixture
def unit_section(tmp_path, monkeypatch):
    """A written section under a temporary data dir, with its tolerances.

    Hand-built rather than borrowed from a real golden so the dispatch is
    pinned against a table this file controls, not against whichever
    tolerances `solve1d` happens to carry.
    """

    def write(tolerances, arrays=None, with_layout=True):
        monkeypatch.setattr(_harness, "DATA_DIR", tmp_path)
        payload = dict(UNIT_LAYOUT) if with_layout else {}
        payload.update(
            arrays
            if arrays is not None
            else {
                "state/x": np.array([1.0, 2.0, 3.0, 4.0]),
                "count": np.asarray(7),
            }
        )
        save_section("unit", payload, {"section": "unit", "tolerances": tolerances})
        return payload

    return write


def test_compare_section_identity(unit_section):
    arrays = unit_section({})
    assert compare_section("unit", dict(arrays)) == []


def test_compare_section_reports_key_sets(unit_section):
    arrays = unit_section({})
    produced = dict(arrays)
    produced["extra_key"] = produced.pop("count")
    problems = compare_section("unit", produced)
    assert any("not produced" in line for line in problems)
    assert any("new key(s)" in line for line in problems)


def test_compare_section_rtol_floor_relaxes_a_bitwise_key(unit_section):
    """`rtol_floor` is how the CUDA run is judged against a bitwise golden."""
    arrays = unit_section({})
    produced = dict(arrays)
    produced["state/x"] = arrays["state/x"] * (1.0 + 1e-11)

    assert compare_section("unit", produced) != []
    assert compare_section("unit", produced, rtol_floor=1e-9) == []
    assert compare_section("unit", produced, rtol_floor=1e-13) != []


def test_compare_section_rtol_floor_does_not_tighten(unit_section):
    """A key already looser than the floor keeps its own budget."""
    arrays = unit_section({"state/x": {"mode": "rel_l2", "rtol": 1e-6}})
    produced = dict(arrays)
    produced["state/x"] = arrays["state/x"] * (1.0 + 1e-8)
    assert compare_section("unit", produced, rtol_floor=1e-12) == []


def test_compare_section_rtol_floor_spares_the_flux_metric(unit_section):
    """The CUDA floor does not loosen a key stored on the per-species bound."""
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-12,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 0,
        }
    }
    arrays = unit_section(tolerances)
    produced = dict(arrays)
    produced["state/x"] = arrays["state/x"] * (1.0 + 1e-10)

    assert compare_section("unit", produced) != []
    assert compare_section("unit", produced, rtol_floor=1e-6) != []


def test_compare_section_dispatches_the_flux_metric(unit_section):
    """A `state/` key is split by the section's own `meta/` stanza."""
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-9,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 0,
        }
    }
    arrays = unit_section(tolerances)
    produced = dict(arrays)
    # Species b (entries 2 and 3) moves by 1e-10; species a is untouched.
    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-10), 4.0])
    assert compare_section("unit", produced) == []

    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-8), 4.0])
    problems = compare_section("unit", produced)
    assert len(problems) == 1
    assert "per-species max 1.000e-08 > 1.0e-09 on b" in problems[0]
    assert "top 0 bin(s) trimmed" in problems[0]


def test_compare_section_honours_the_stored_trim(unit_section):
    """The entry's `trim_top_bins` reaches the metric, not just the defaults."""
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-9,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 1,
        }
    }
    arrays = unit_section(tolerances)
    produced = dict(arrays)
    # The move is in bin 1 of species b -- the top bin, which trim 1 drops.
    produced["state/x"] = np.array([1.0, 2.0, 3.0, 4.0 * (1 + 1e-3)])
    assert compare_section("unit", produced) == []

    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-8), 4.0])
    problems = compare_section("unit", produced)
    assert "per-species max 1.000e-08 > 1.0e-09 on b" in problems[0]
    assert "top 1 bin(s) trimmed" in problems[0]


def test_per_species_max_without_a_layout_is_reported(unit_section):
    """A section with no species stanza cannot be scored, and says so."""
    arrays = unit_section(
        {"x": {"mode": "per_species_max", "rtol": 1e-12}},
        arrays={"x": np.array([1.0, 2.0])},
        with_layout=False,
    )
    produced = {"x": arrays["x"] * 1.1}
    problems = compare_section("unit", produced)
    assert "needs a species layout" in problems[0]


# --------------------------------------------------------------------------
# the metric itself, on synthetic arrays
# --------------------------------------------------------------------------


@pytest.fixture
def layout():
    """Three species of four bins on a decade grid."""
    return _flux_metric.Layout(
        table=(("a", 0), ("b", 1), ("c", 2)),
        dim=4,
        dim_states=12,
        e_grid=np.array([10.0, 100.0, 1e3, 1e4]),
        e_bins=np.array([3.0, 30.0, 300.0, 3e3, 3e4]),
    )


def test_species_metric_floor_drops_the_small_bins():
    """A bin below `floor x peak` is not in the maximum."""
    ref = np.array([1.0, 1e-14, 0.5])
    new = np.array([1.0, 1e-14 * 2, 0.5 * (1 + 1e-9)])
    grid = np.array([10.0, 100.0, 1e3])

    loose = _flux_metric.species_metric(ref, new, grid, floor=1e-12, trim=0)
    assert loose.n_kept == 2
    assert loose.max_rel == pytest.approx(1e-9)
    assert loose.at_e == 1e3

    strict = _flux_metric.species_metric(ref, new, grid, floor=1e-16, trim=0)
    assert strict.n_kept == 3
    assert strict.max_rel == pytest.approx(1.0)


def test_species_metric_drops_zero_reference_bins():
    ref = np.array([1.0, 0.0])
    new = np.array([1.0, 7.0])
    score = _flux_metric.species_metric(
        ref, new, np.array([10.0, 100.0]), floor=0.0, trim=0
    )
    assert score.n_kept == 1
    assert score.max_rel == 0.0
    assert not np.isfinite(score.rel_raw[1])


def test_species_metric_non_positive_peak_scores_nan():
    """A reference with no positive bin supports no relative bound."""
    for ref in (np.zeros(3), np.array([-1.0, -2.0, -3.0])):
        score = _flux_metric.species_metric(
            ref, ref + 1.0, np.array([10.0, 100.0, 1e3]), trim=0
        )
        assert score.n_kept == 0
        assert np.isnan(score.max_rel)


def test_guard_excludes_a_species_with_one_negative_bin(layout):
    """The ruling: a species enters the maximum only if it is sign definite."""
    ref = np.array(
        [1.0, 1.0, 1.0, 1.0]  # a: sign definite
        + [1.0, 1.0, -1e-3, 1.0]  # b: one negative bin
        + [1.0, 1.0, 1.0, 1.0]  # c: sign definite
    )
    new = np.array(ref)
    new[4] *= 1 + 1e-6  # b moves by 1e-6
    new[8] *= 1 + 1e-9  # c moves by 1e-9

    guarded = _flux_metric.evaluate_key(ref, new, layout, guard="sign_definite", trim=0)
    assert {entry.species for entry in guarded if entry.guarded} == {"a", "c"}
    assert _flux_metric.worst_entry(guarded).species == "c"
    assert _flux_metric.worst_entry(guarded).score.max_rel == pytest.approx(1e-9)

    unguarded = _flux_metric.evaluate_key(ref, new, layout, guard="none", trim=0)
    assert _flux_metric.worst_entry(unguarded).species == "b"
    assert _flux_metric.worst_entry(unguarded).score.max_rel == pytest.approx(1e-6)


def test_all_zero_species_is_admitted_but_scores_nothing(layout):
    """Sign definite, but no bin above the floor, so it is not the maximum."""
    ref = np.zeros(12)
    ref[0:4] = 1.0
    new = np.array(ref)
    new[4] = 1e-30  # species b was zero and is not any more

    entries = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    by_species = {entry.species: entry for entry in entries}
    assert by_species["b"].guarded
    assert np.isnan(by_species["b"].score.max_rel)
    assert _flux_metric.worst_entry(entries).species == "a"
    assert _flux_metric.worst_entry(entries).score.max_rel == 0.0


def test_unscorable_when_the_guard_admits_nothing(layout):
    """A key the guard empties has no per-species maximum, and says so."""
    ref = np.full(12, -1.0)
    with pytest.raises(_flux_metric.Unscorable, match="no species the sign_definite"):
        _flux_metric.compare("k", ref, ref.copy(), 1e-12, layout=layout, trim=0)


def test_unscorable_falls_back_to_l2_at_the_fallback_rtol(layout):
    """The fallback is a real bound, neither a free pass nor bitwise."""
    ref = np.full(12, -1.0)
    entry = {"mode": "per_species_max", "rtol": 1e-12, "fallback_rtol": 1e-9}

    assert (
        compare_key(
            "k",
            ref,
            ref * (1 + 1e-10),
            "per_species_max",
            1e-12,
            layout=layout,
            entry=entry,
        )
        is None
    )
    problem = compare_key(
        "k",
        ref,
        ref * (1 + 1e-7),
        "per_species_max",
        1e-12,
        layout=layout,
        entry=entry,
    )
    assert problem.startswith("k: rel-L2")
    # without a fallback_rtol the per-species bound is reused
    assert compare_key(
        "k", ref, ref * (1 + 1e-10), "per_species_max", 1e-12, layout=layout
    ).startswith("k: rel-L2")


def test_unscorable_all_zero_reference_still_requires_exactness(layout):
    """The rel_l2 fallback keeps the harness rule for an all-zero golden."""
    ref = np.zeros(12)
    entry = {"fallback_rtol": 1.0}
    assert (
        compare_key(
            "k", ref, ref.copy(), "per_species_max", 1e-12, layout=layout, entry=entry
        )
        is None
    )
    problem = compare_key(
        "k",
        ref,
        np.full(12, 1e-30),
        "per_species_max",
        1e-12,
        layout=layout,
        entry=entry,
    )
    assert "all-zero" in problem


# --------------------------------------------------------------------------
# containment: the species and lanes the maximum does not cover
# --------------------------------------------------------------------------


def test_containment_bounds_a_species_the_guard_rejects(layout):
    """The hole the per-species maximum leaves, and what closes it.

    The maximum is taken over the species the guard admits, so on a key with
    one admitted species a move confined to a rejected one is invisible to it.
    Species `b` here has a negative bin and is rejected; `a` and `c` are not
    and hold the maximum at zero. Containment is what notices.
    """
    ref = np.array(
        [1.0, 1.0, 1.0, 1.0]  # a: sign definite
        + [1.0, 1.0, -1e-3, 1.0]  # b: rejected
        + [1.0, 1.0, 1.0, 1.0]  # c: sign definite
    )
    new = np.array(ref)
    new[4] *= 1 + 1e-6

    # the flux bound sees nothing: every admitted species is untouched
    entries = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    assert _flux_metric.worst_entry(entries).score.max_rel == 0.0

    problem = _flux_metric.compare(
        "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-9
    )
    assert "containment rel-L2" in problem
    assert "> 1.0e-09 over the 1 species the sign_definite guard leaves" in problem
    assert "(b)" in problem
    assert "per-species max" not in problem

    # a real bound, not a hard fail: the same move inside `fallback_rtol` passes
    assert (
        _flux_metric.compare(
            "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-3
        )
        is None
    )


def test_containment_is_per_lane(layout):
    """A rejected species is bounded in the lane it moved in, not on average."""
    ref = np.ones((12, 8))
    ref[6] = -1e-3  # species b, bin 2: rejected in every lane
    new = np.array(ref)
    new[4, 5] *= 1 + 1e-6  # species b, bin 0, lane 5 only

    problem = _flux_metric.compare(
        "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-9
    )
    assert "k[:,5]: containment rel-L2" in problem
    assert "worst of" not in problem  # exactly one lane is over


def test_containment_and_the_flux_bound_are_reported_distinctly(layout):
    """A flux mismatch and a containment mismatch are different statements."""
    ref = np.array([1.0] * 4 + [1.0, 1.0, -1e-3, 1.0] + [1.0] * 4)
    new = np.array(ref)
    new[0] *= 1 + 1e-6  # species a: admitted, so the flux bound
    new[4] *= 1 + 1e-6  # species b: rejected, so containment

    problem = _flux_metric.compare(
        "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-9
    )
    assert "per-species max 1.000e-06 > 1.0e-09 on a" in problem
    assert "containment rel-L2" in problem
    assert problem.count("; ") == 1


def test_containment_covers_a_species_with_nothing_above_the_floor(layout):
    """Admitted but unscored is unbounded too, so it is contained as well."""
    ref = np.zeros(12)
    ref[0:4] = 1.0  # a carries the flux; b and c are identically zero
    new = np.array(ref)
    new[4] = 1e-30  # species b was zero and is not any more

    entries = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    by_species = {entry.species: entry for entry in entries}
    assert by_species["b"].guarded and not _flux_metric.covered(by_species["b"])

    problem = _flux_metric.compare(
        "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1.0
    )
    # an all-zero reference supports no relative bound, so exactness is required
    assert "containment rel-L2 inf" in problem


def test_containment_skips_the_derived_helicity_sums():
    """`total_mu+` is not stored bins, so it carries no containment of its own.

    Its components do. Here `mu+_l` is rejected and `total_mu+` with it, and
    the containment denominator has to be the two component rows -- counting
    the sum as well would double the residual and halve the bound.
    """
    layout = _flux_metric.Layout(
        table=(("mu+_l", 0), ("mu+_r", 1)),
        dim=2,
        dim_states=4,
        e_grid=np.array([10.0, 100.0]),
        e_bins=np.array([3.0, 30.0, 300.0]),
    )
    ref = np.array([1.0, -1.0, 1.0, 1.0])  # mu+_l rejected, mu+_r admitted
    new = np.array(ref)
    new[0] *= 1 + 1e-6

    lanes = _flux_metric.lane_containment(
        ref, new, layout, _flux_metric.evaluate_key(ref, new, layout, trim=0)
    )
    assert len(lanes) == 1
    label, rel, rejected = lanes[0]
    assert (label, rejected) == ("", ["mu+_l"])
    assert rel == pytest.approx(1e-6 / np.sqrt(2.0))


def test_containment_bounds_a_rejected_spectrum_row(layout):
    """A stored spectrum has no species, so the unit of containment is the row."""
    ref = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1e-3, 1.0]])
    new = np.array(ref)
    new[1, 0] *= 1 + 1e-6

    problem = _flux_metric.compare(
        "spectrum/z", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-9
    )
    assert "spectrum/z[1]: containment rel-L2" in problem
    assert "the row the sign_definite guard leaves unscored" in problem


def test_containment_uses_the_entry_fallback_rtol(unit_section):
    """The stored `fallback_rtol`, not `rtol`, is what contains a rejected species."""
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-12,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 0,
            "fallback_rtol": 1e-6,
        }
    }
    # species b (entries 2 and 3) has a negative bin, so the guard rejects it
    arrays = unit_section(
        tolerances, arrays={"state/x": np.array([1.0, 2.0, 3.0, -4.0])}
    )
    produced = dict(arrays)

    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-9), -4.0])
    assert compare_section("unit", produced) == []

    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-3), -4.0])
    problems = compare_section("unit", produced)
    assert len(problems) == 1
    assert "containment rel-L2" in problems[0] and "> 1.0e-06" in problems[0]


def test_containment_respects_the_cuda_rtol_floor(unit_section):
    """A norm moves with the reduction order, so the CUDA floor reaches it.

    The per-species maximum itself does not -- `compare_section` keeps that at
    its stored bound -- but the containment of the species it leaves unscored
    is a relative L2, and judging CUDA's reordering by a 1e-12 norm would
    report the backend rather than the solver.
    """
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-12,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 0,
            "fallback_rtol": 1e-12,
        }
    }
    arrays = unit_section(
        tolerances, arrays={"state/x": np.array([1.0, 2.0, 3.0, -4.0])}
    )
    produced = dict(arrays)
    produced["state/x"] = np.array([1.0, 2.0, 3.0 * (1 + 1e-10), -4.0])

    assert compare_section("unit", produced) != []
    assert compare_section("unit", produced, rtol_floor=1e-9) == []


def test_containment_bounds_a_rejected_species_of_the_1d_golden(solve1d_golden):
    """The regression this closes, on the section that shipped it.

    `emon/theta0/state` admits 68 of its 74 entries; the six the guard rejects
    are `e+-` and their helicities -- the cancellation residuals the sign test
    exists to keep out of the flux bound. A move confined to one of them is
    what the per-species maximum cannot see and containment must.
    """
    key = "emon/theta0/state"
    layout = _flux_metric.layout_for(key, solve1d_golden)
    index = dict(layout.table)["e+_l"]

    reference = solve1d_golden[key]
    entries = _flux_metric.evaluate_key(reference, reference, layout)
    rejected = [entry.species for entry in entries if not _flux_metric.covered(entry)]
    assert rejected == ["e+_l", "e+", "e+_r", "e-_l", "e-", "e-_r"]

    produced = np.array(reference)
    produced[index * layout.dim : (index + 1) * layout.dim] *= 1 + 1e-3

    # the flux bound is blind to it: no admitted species moved
    worst = _flux_metric.worst_entry(
        _flux_metric.evaluate_key(reference, produced, layout)
    )
    assert worst.score.max_rel == 0.0

    problem = _flux_metric.compare(
        key,
        reference,
        produced,
        _flux_metric.RTOL_1D,
        layout=layout,
        fallback_rtol=_flux_metric.RTOL_1D,
    )
    assert "containment rel-L2" in problem
    assert "e+_l" in problem
    assert "per-species max" not in problem


# --------------------------------------------------------------------------
# non-finite actuals
# --------------------------------------------------------------------------


def test_nan_in_a_scored_bin_is_a_mismatch(layout):
    """`nanargmax` steps over a NaN, so the check has to precede the score.

    Not hypothetical: the 1D solve diverges to NaN with `e+-` enabled at high
    zenith, and which entries the divergence reaches is backend-dependent.
    """
    ref = np.ones(12)
    new = np.array(ref)
    new[5] = np.nan  # species b, bin 1 -- above the floor and inside the trim

    score = _flux_metric.species_metric(ref[4:8], new[4:8], layout.e_grid, trim=0)
    assert score.max_rel == 0.0  # what the maximum alone reports

    problem = _flux_metric.compare("k", ref, new, 1e-9, layout=layout, trim=0)
    assert "1 bin(s) of b differ in finiteness from the golden" in problem
    assert "first at E = 100 GeV: actual nan, golden 1.0" in problem
    assert "(1 species/lane(s) affected)" in problem


def test_a_whole_lane_of_nan_is_a_mismatch(layout):
    """The shape that raised out of `nanargmax` instead of reporting anything."""
    ref = np.ones((12, 2))
    new = np.array(ref)
    new[:, 1] = np.nan

    problem = _flux_metric.compare("k", ref, new, 1e-9, layout=layout, trim=0)
    assert "k[:,1]: 4 bin(s) of a differ in finiteness" in problem
    # three species plus the two derived helicity sums are absent here, so
    # every entry of the lane is affected
    assert "(3 species/lane(s) affected)" in problem


def test_infinite_actual_is_a_mismatch(layout):
    """An infinite score is dropped by `worst_entry`, so it needs its own check."""
    for value in (np.inf, -np.inf):
        new = np.ones(12)
        new[0] = value
        entries = _flux_metric.evaluate_key(np.ones(12), new, layout, trim=0)
        by_species = {entry.species: entry for entry in entries}
        assert not np.isfinite(by_species["a"].score.max_rel)
        assert _flux_metric.worst_entry(entries).score.max_rel == 0.0

        problem = _flux_metric.compare(
            "k", np.ones(12), new, 1e-9, layout=layout, trim=0
        )
        assert "1 bin(s) of a differ in finiteness" in problem
        assert f"actual {value!r}" in problem


def test_nan_is_reported_before_the_other_two_bounds(layout):
    """A NaN makes the maximum and the containment meaningless, so it reports alone."""
    ref = np.array([1.0] * 4 + [1.0, 1.0, -1e-3, 1.0] + [1.0] * 4)
    new = np.array(ref)
    new[0] *= 1 + 1e-3  # would be a flux mismatch
    new[4] = np.nan  # and a containment one

    problem = _flux_metric.compare(
        "k", ref, new, 1e-9, layout=layout, trim=0, fallback_rtol=1e-9
    )
    assert "differ in finiteness" in problem
    assert "per-species max" not in problem and "containment" not in problem


def test_nan_in_a_trimmed_bin_is_still_a_mismatch(layout):
    """The trim says which bins carry flux, not which may stop being numbers."""
    ref = np.ones(12)
    new = np.array(ref)
    new[3] = np.nan  # species a, top bin of a four-bin grid

    entries = _flux_metric.evaluate_key(ref, new, layout, trim=1)
    assert _flux_metric.worst_entry(entries).score.max_rel == 0.0
    problem = _flux_metric.compare("k", ref, new, 1e-9, layout=layout, trim=1)
    assert "1 bin(s) of a differ in finiteness" in problem


def test_a_matching_nan_pattern_is_not_a_mismatch(layout):
    """The rule is the sibling modes': the *pattern* has to differ."""
    ref = np.ones(12)
    ref[5] = np.nan
    new = np.array(ref)
    assert _flux_metric.compare("k", ref, new, 1e-9, layout=layout, trim=0) is None


def test_compare_section_reports_a_nan_on_the_flux_metric(unit_section):
    """The dispatch carries it, so the mode is not NaN-blind at section level."""
    tolerances = {
        "state/": {
            "mode": "per_species_max",
            "rtol": 1e-9,
            "floor": 1e-12,
            "guard": "sign_definite",
            "trim_top_bins": 0,
        }
    }
    arrays = unit_section(tolerances)
    produced = dict(arrays)
    produced["state/x"] = np.array([1.0, 2.0, np.nan, 4.0])
    problems = compare_section("unit", produced)
    assert len(problems) == 1
    assert "differ in finiteness from the golden" in problems[0]


def test_helicity_rows_are_summed():
    """`total_mu+` is the flux a result is quoted on, so it is scored."""
    layout = _flux_metric.Layout(
        table=(("mu+", 0), ("mu+_l", 1), ("mu+_r", 2)),
        dim=1,
        dim_states=3,
        e_grid=np.array([10.0]),
        e_bins=np.array([3.0, 30.0]),
    )
    split = _flux_metric.split_species(np.array([1.0, 2.0, 4.0]), layout)
    assert split["total_mu+"] == pytest.approx(7.0)


def test_state_lanes_orientation():
    """Columns for a batched state, rows for a depth-grid stack."""
    columns = _flux_metric.state_lanes(np.zeros((12, 3)), 12)
    assert [label for label, _ in columns] == ["[:,0]", "[:,1]", "[:,2]"]
    rows = _flux_metric.state_lanes(np.zeros((3, 12)), 12)
    assert [label for label, _ in rows] == ["[0]", "[1]", "[2]"]
    assert _flux_metric.state_lanes(np.zeros(5), 12) is None


def test_lanes_are_scored_independently(layout):
    """A batched state is scored per column, and the worst column wins."""
    ref = np.ones((12, 2))
    new = np.array(ref)
    new[0, 1] *= 1 + 1e-7

    problem = _flux_metric.compare("k", ref, new, 1e-9, layout=layout, trim=0)
    assert "k[:,1]: per-species max 1.000e-07" in problem
    assert _flux_metric.compare("k", ref, new, 1e-6, layout=layout, trim=0) is None


def test_spectrum_rows_need_no_species(layout):
    """A stored spectrum is one score per row, on the energy grid it matches."""
    ref = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    new = np.array(ref)
    new[1, 3] *= 1 + 1e-5
    problem = _flux_metric.compare("spectrum/z", ref, new, 1e-7, layout=layout, trim=0)
    assert "spectrum/z[1]: per-species max 1.000e-05" in problem
    assert "at E = 1e+04" in problem


def test_a_row_off_the_energy_grid_is_located_by_index(layout):
    """A sky map is one value per pixel; calling that an energy would mislead."""
    ref = np.array([[1.0, 1.0], [1.0, 1.0]])
    new = np.array(ref)
    new[0, 1] *= 1 + 1e-5
    problem = _flux_metric.compare("spectrum/skymap", ref, new, 1e-7, layout=layout)
    assert "at index 1" in problem
    assert "E =" not in problem


# --------------------------------------------------------------------------
# the edge trim
# --------------------------------------------------------------------------


def test_trim_drops_the_top_bins_from_the_score(layout):
    """A move in a trimmed bin is not the maximum, and is not scored at all."""
    ref = np.ones(12)
    new = np.array(ref)
    new[3] *= 1 + 1e-3  # species a, top bin of a four-bin grid

    entries = _flux_metric.evaluate_key(ref, new, layout, trim=1)
    by_species = {entry.species: entry for entry in entries}
    assert by_species["a"].score.n_kept == 3
    assert by_species["a"].score.n_trimmed == 1
    assert by_species["a"].score.max_rel == 0.0
    assert not by_species["a"].score.keep[3]
    # unfloored, the table still shows how far the edge moved
    assert by_species["a"].score.rel_raw[3] == pytest.approx(1e-3)

    untrimmed = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    assert _flux_metric.worst_entry(untrimmed).score.max_rel == pytest.approx(1e-3)


def test_trim_drops_the_top_bins_from_the_sign_test(layout):
    """The ruling: an untrustworthy reference bin is dropped for both purposes.

    Species `b` is negative only in its top bin -- the 2D boundary artefact in
    miniature -- and the trim is what lets it be scored at all.
    """
    ref = np.array(
        [1.0, 1.0, 1.0, 1.0]  # a
        + [1.0, 1.0, 1.0, -1e-2]  # b: negative in the top bin only
        + [1.0, 1.0, -1e-2, 1.0]  # c: negative one bin deeper
    )
    new = np.array(ref)
    new[4] *= 1 + 1e-6

    admitted = {
        entry.species
        for entry in _flux_metric.evaluate_key(ref, new, layout, trim=1)
        if entry.guarded
    }
    assert admitted == {"a", "b"}
    assert {
        entry.species
        for entry in _flux_metric.evaluate_key(ref, new, layout, trim=0)
        if entry.guarded
    } == {"a"}
    assert {
        entry.species
        for entry in _flux_metric.evaluate_key(ref, new, layout, trim=2)
        if entry.guarded
    } == {"a", "b", "c"}

    worst = _flux_metric.worst_entry(
        _flux_metric.evaluate_key(ref, new, layout, trim=1)
    )
    assert (worst.species, worst.score.max_rel) == ("b", pytest.approx(1e-6))


def test_trim_takes_the_peak_over_the_retained_bins(layout):
    """The floor is relative to the retained peak, not to a trimmed edge spike.

    Species `a` peaks in its top bin; trimmed, the peak drops by 1e6 and the
    bins that were below the floor come back into the score.
    """
    ref = np.array([1.0, 1.0, 1.0, 1e6] + [1.0] * 8)
    new = np.array(ref)
    new[0] *= 1 + 1e-6

    untrimmed = _flux_metric.species_metric(
        ref[:4], new[:4], layout.e_grid, floor=1e-3, trim=0
    )
    assert (untrimmed.peak, untrimmed.n_kept) == (1e6, 1)
    assert untrimmed.max_rel == 0.0

    trimmed = _flux_metric.species_metric(
        ref[:4], new[:4], layout.e_grid, floor=1e-3, trim=1
    )
    assert (trimmed.peak, trimmed.n_kept) == (1.0, 3)
    assert trimmed.max_rel == pytest.approx(1e-6)


def test_trim_is_not_applied_off_the_energy_grid(layout):
    """`trim` names the top bins of an energy grid; a pixel axis has no top."""
    ref = np.array([[1.0, 1.0], [1.0, 1.0]])
    new = np.array(ref)
    new[0, 1] *= 1 + 1e-5

    entries = _flux_metric.evaluate_key(ref, new, layout, trim=1)
    assert [entry.on_e_grid for entry in entries] == [False, False]
    assert all(entry.score.n_trimmed == 0 for entry in entries)
    assert _flux_metric.worst_entry(entries).score.max_rel == pytest.approx(1e-5)


def test_trim_past_the_end_of_the_grid_is_unscorable(layout):
    """Nothing retained is nothing scored, and the message says why."""
    ref = np.ones(12)
    with pytest.raises(_flux_metric.Unscorable, match="top 4 bin"):
        _flux_metric.compare("k", ref, ref.copy(), 1e-12, layout=layout, trim=4)


def test_nearest_bins_flags_off_grid_targets():
    bins = _flux_metric.nearest_bins(
        np.array([10.0, 100.0]), np.array([3.0, 30.0, 300.0]), targets=(90.0, 1e6)
    )
    assert (bins[0].index, bins[0].in_grid) == (1, True)
    assert (bins[1].index, bins[1].e_centre, bins[1].in_grid) == (1, 100.0, False)


# --------------------------------------------------------------------------
# the metric on the committed 1D golden
# --------------------------------------------------------------------------

#: The three numbers measured with the reference implementation of the metric
#: (runs/2026-09-02_phi-metric/inputs/metric.py) on the phi-Taylor retune, over
#: the `<case>/theta*/state` keys of the 1D section, under the ruling's guard:
#: sign definite over the grid less its top `TRIM_TOP_BINS` bins.
#:
#: `DRIFT_SIGN_DEFINITE` is what `RTOL_1D` is set against. Untrimmed the same
#: scan reads 1.165e-13, an order tighter -- but on 135 of 278 entries and 10
#: of the 12 key species instead of 259 and 12, because it disqualifies a
#: species over the top bins of the grid rather than over the flux.
DRIFT_SIGN_DEFINITE = 1.129e-12
DRIFT_KEY_SPECIES = 1.079e-12
DRIFT_UNGUARDED = 1.354e-09

#: The same scan widened to every key the `TOLERANCES` entry actually bounds --
#: the depth-grid stacks as well as the final states. This, not
#: `DRIFT_SIGN_DEFINITE`, is the margin `RTOL_1D` carries.
DRIFT_ALL_STATE_KEYS = 2.915e-12

#: Where the pre-retune 1D golden is looked for. Only one version of a section
#: is committed, so the reference of the drift scan is named by the
#: environment: `MCEQ_GOLDEN_PRE_RETUNE=/path/to/solve1d_pre_v2.npz`.
PRE_RETUNE_ENV = "MCEQ_GOLDEN_PRE_RETUNE"

#: Read directly rather than through `load_section`, whose data dir the
#: `unit_section` fixture monkeypatches.
SOLVE1D_PATH = pathlib.Path(_flux_metric.__file__).parent / "data" / "solve1d.npz"


def _load_npz(path):
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files if k != _harness.PROVENANCE_KEY}


@pytest.fixture(scope="module")
def solve1d_golden():
    return _load_npz(SOLVE1D_PATH)


def test_solve1d_layout_resolves_per_case(solve1d_golden):
    """Each case has its own species table; `emon` adds the six e+/e- rows."""
    emoff = _flux_metric.layout_for("emoff/theta89/state", solve1d_golden)
    emon = _flux_metric.layout_for("emon/theta0/state", solve1d_golden)
    assert (emoff.dim, emoff.dim_states, len(emoff.table)) == (31, 2046, 66)
    assert (emon.dim, emon.dim_states, len(emon.table)) == (31, 2232, 72)
    assert emoff.dim * len(emoff.table) == emoff.dim_states
    assert dict(emoff.table)["antinue"] == 0
    assert "e+_l" in dict(emon.table)
    assert "e+_l" not in dict(emoff.table)


def test_solve1d_split_matches_get_solution(solve1d_golden):
    """The stored `sol/` spectra are the slices the metric scores."""
    layout = _flux_metric.layout_for("emoff/theta0/state", solve1d_golden)
    split = _flux_metric.split_species(solve1d_golden["emoff/theta0/state"], layout)
    assert np.array_equal(split["numu"], solve1d_golden["emoff/theta0/sol/total_numu"])
    assert np.array_equal(
        split["total_mu+"], solve1d_golden["emoff/theta0/sol/total_mu+"]
    )


def test_solve1d_grid_sol_is_read_as_a_stack_of_states(solve1d_golden):
    layout = _flux_metric.layout_for("emoff/grid/grid_sol", solve1d_golden)
    lanes = _flux_metric.state_lanes(
        solve1d_golden["emoff/grid/grid_sol"], layout.dim_states
    )
    assert [label for label, _ in lanes] == ["[0]", "[1]", "[2]", "[3]"]


def test_solve1d_perturbation_is_scored_where_it_was_planted(solve1d_golden):
    """One species, one bin, one lane -- computed independently of the metric."""
    key = "emoff/theta60/state"
    layout = _flux_metric.layout_for(key, solve1d_golden)
    index = dict(layout.table)["numu"]
    bin_ = 12

    reference = solve1d_golden[key]
    produced = np.array(reference)
    produced[index * layout.dim + bin_] *= 1 + 3e-9

    # the planted bin has to be retained and carry flux, or it would not score
    numu = reference[index * layout.dim : (index + 1) * layout.dim]
    retained = numu[: _flux_metric.n_retained(numu.size, _flux_metric.TRIM_TOP_BINS)]
    assert bin_ < retained.size
    assert numu[bin_] >= _flux_metric.FLOOR * retained.max()

    expected_e = layout.e_grid[bin_]
    problem = _flux_metric.compare("k", reference, produced, 1e-12, layout=layout)
    assert "per-species max 3.000e-09" in problem
    assert f"on numu at E = {expected_e:.4g}" in problem
    assert _flux_metric.compare("k", reference, produced, 1e-8, layout=layout) is None

    # every other species is untouched, so numu is the worst of the key
    worst = _flux_metric.worst_entry(
        _flux_metric.evaluate_key(reference, produced, layout)
    )
    assert worst.species == "numu"
    assert worst.score.max_rel == pytest.approx(3e-9, rel=1e-6)


@pytest.fixture(scope="module")
def drift_scan(solve1d_golden):
    """The metric over both cases of the retune drift, guarded and not.

    The reference is the pre-retune 1D golden. Only one version of a section is
    committed, so the file is named by `PRE_RETUNE_ENV` and the scan skips
    without it; the three numbers it checks are the ones the reference
    implementation of the metric produced when the bound was chosen. The trim
    is the metric's, not the guard's, so it applies to the unguarded and
    key-species diagnostics too.
    """
    path = os.environ.get(PRE_RETUNE_ENV)
    if not path or not pathlib.Path(path).exists():
        pytest.skip(f"set {PRE_RETUNE_ENV} to the pre-retune solve1d golden")
    reference = _load_npz(path)

    from .gen_solve1d import STATE_KEYS

    scan = {"sign_definite": [], "none": [], "key_species": [], "all_keys": []}
    for key in STATE_KEYS:
        layout = _flux_metric.layout_for(key, reference)
        worst = _flux_metric.worst_entry(
            _flux_metric.evaluate_key(
                reference[key],
                solve1d_golden[key],
                layout,
                trim=_flux_metric.TRIM_TOP_BINS,
            )
        )
        if worst is not None:
            scan["all_keys"].append((worst.score.max_rel, worst.species, key))

    for key in sorted(k for k in reference if "/theta" in k and k.endswith("/state")):
        layout = _flux_metric.layout_for(key, reference)
        for guard in ("sign_definite", "none"):
            entries = _flux_metric.evaluate_key(
                reference[key],
                solve1d_golden[key],
                layout,
                guard=guard,
                trim=_flux_metric.TRIM_TOP_BINS,
            )
            worst = _flux_metric.worst_entry(entries)
            if worst is not None:
                scan[guard].append((worst.score.max_rel, worst.species, key))
            if guard == "none":
                scan["key_species"] += [
                    (entry.score.max_rel, entry.species, key)
                    for entry in entries
                    if entry.species in _flux_metric.KEY_SPECIES
                    and np.isfinite(entry.score.max_rel)
                ]
    return {name: max(rows) for name, rows in scan.items()}


def test_drift_sign_definite_worst(drift_scan):
    """The guarded worst of the phi-Taylor retune, and where it sits.

    1.129e-12 on the K-decay muon at 3548 GeV, so `RTOL_1D` = 1e-11 carries
    8.9x. Trim 1 or 2 would not do: the artefact still reaches into bin 29
    (`prres_mu-` at 708 TeV, 1.17e-10 -- above the bound) and bin 28
    (`mu+` at 562 TeV, 4.98e-12).
    """
    value, species, key = drift_scan["sign_definite"]
    assert value == pytest.approx(DRIFT_SIGN_DEFINITE, rel=1e-3)
    assert (species, key) == ("k_mu+_l", "emoff/theta89/state")
    assert value < _flux_metric.RTOL_1D


def test_drift_over_every_bounded_key_is_the_real_margin(drift_scan):
    """The entry bounds the depth-grid stacks too, and they move further.

    2.915e-12 on `e+_l` at 447 TeV in the first depth snapshot -- the EM
    cascade at 10 g/cm2 is young enough to still be sign definite, so the
    guard admits the species that at full depth is the 1.35e-9 outlier. 1e-11
    over it is 3.4x, the thinnest margin in the harness and the number to
    quote when this bound is next discussed.
    """
    value, species, key = drift_scan["all_keys"]
    assert value == pytest.approx(DRIFT_ALL_STATE_KEYS, rel=1e-3)
    assert (species, key) == ("e+_l", "emon/grid/grid_sol")
    assert value < _flux_metric.RTOL_1D
    assert _flux_metric.RTOL_1D / value > 3.0
    assert value > DRIFT_SIGN_DEFINITE


def test_drift_key_species_worst(drift_scan):
    """The worst named flux, which the trimmed guard now admits rather than gates.

    Untrimmed it was 4.054e-12 on `total_mu-` at 562 TeV -- bin 28, inside the
    trim -- and gated out; the worst retained named flux is 1.079e-12, and it
    is the same key and bin as the guarded worst.
    """
    value, species, key = drift_scan["key_species"]
    assert value == pytest.approx(DRIFT_KEY_SPECIES, rel=1e-3)
    assert (species, key) == ("total_mu+", "emoff/theta89/state")
    assert value < _flux_metric.RTOL_1D


def test_drift_unguarded_worst_is_a_cancellation_residual(drift_scan):
    """What the guard buys: three orders, on a species with negative bins.

    `e+_l` at 71 TeV is well inside the retained grid, so the trim neither
    creates nor removes this one -- it is the sign test that gates it.
    """
    value, species, _ = drift_scan["none"]
    assert value == pytest.approx(DRIFT_UNGUARDED, rel=1e-3)
    assert species == "e+_l"
    assert value > _flux_metric.RTOL_1D


# --------------------------------------------------------------------------
# the failure report
# --------------------------------------------------------------------------


def test_flux_report_names_the_worst_and_tabulates_it(solve1d_golden):
    from .gen_solve1d import STATE_KEYS, diff_report

    produced = dict(solve1d_golden)
    key = "emoff/theta89/state"
    layout = _flux_metric.layout_for(key, solve1d_golden)
    index = dict(layout.table)["nue"]
    state = np.array(solve1d_golden[key])
    state[index * layout.dim : (index + 1) * layout.dim] *= 1 + 1e-6
    produced[key] = state

    lines = diff_report(solve1d_golden, produced)
    text = "\n".join(lines)
    assert f"over {len(STATE_KEYS)} flux key(s)" in lines[0]
    assert "guard sign_definite" in lines[0]
    assert f"top {_flux_metric.TRIM_TOP_BINS} bin(s) trimmed" in lines[0]
    assert f"{key}: 1.000e-06 on nue" in text
    assert f"fixed-energy table for {key}:" in text
    assert "total_mu+" in text and "antinue" in text
    for target in _flux_metric.FIXED_E_GEV:
        assert f"{target:g}" in text


def test_flux_report_is_empty_without_a_difference(solve1d_golden):
    from .gen_solve1d import diff_report

    lines = diff_report(solve1d_golden, dict(solve1d_golden))
    assert "0.000e+00" in "\n".join(lines)


def test_fixed_energy_table_marks_off_grid_targets(layout):
    ref = np.ones(12)
    new = np.array(ref)
    new[0] *= 1 + 1e-6
    entries = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    lines = _flux_metric.fixed_energy_table(entries, layout, species=("a", "b", "c"))
    header = lines[0]
    # the grid tops out at 10 TeV, so 100 TeV and 1 PeV are off-grid
    assert f"{1e5:g}*" in header and f"{1e6:g}*" in header
    assert f"{1e4:g}*" not in header
    assert lines[-1].startswith("* off-grid target")


def test_fixed_energy_table_parenthesises_floored_bins(layout):
    """A bin the floor drops is shown, in parentheses, not counted."""
    ref = np.array([1.0, 1e-20, 1.0, 1.0] + [1.0] * 8)
    new = np.array(ref)
    new[1] *= 2.0
    entries = _flux_metric.evaluate_key(ref, new, layout, trim=0)
    lines = _flux_metric.fixed_energy_table(entries, layout, species=("a",))
    assert "(1.0e+00)" in lines[1]
    assert _flux_metric.worst_entry(entries).score.max_rel == 0.0


def test_fixed_energy_table_brackets_trimmed_bins(layout):
    """A trimmed bin reads `[...]`, distinct from the floor's `(...)`."""
    ref = np.array([1.0, 1e-20, 1.0, 1.0] + [1.0] * 8)
    new = np.array(ref)
    new[1] *= 2.0  # floored, bin 1
    new[3] *= 2.0  # trimmed, bin 3 -- the top of a four-bin grid
    entries = _flux_metric.evaluate_key(ref, new, layout, trim=1)
    lines = _flux_metric.fixed_energy_table(entries, layout, species=("a",))
    assert "(1.0e+00)" in lines[1]
    assert "[1.0e+00]" in lines[1]
    assert _flux_metric.worst_entry(entries).score.max_rel == 0.0


# --------------------------------------------------------------------------
# the 2D layout: recorded in the provenance, not in the arrays
# --------------------------------------------------------------------------


def test_solve2d_tolerance_table_covers_every_state_key():
    """Norm keys must not fall through to the `state/` per-species entry."""
    from .gen_solve2d import NORM_KEYS, TOLERANCES

    prov = {"tolerances": TOLERANCES}
    for key in NORM_KEYS:
        assert tolerance_for(key, prov)[0] == "rel_l2", key
    for tag in ("single", "multirhs", "carousel", "fullsky"):
        assert tolerance_for(f"state/{tag}_mode0", prov)[0] == "per_species_max"
    assert tolerance_for("spectrum/single_total_numu", prov)[0] == "per_species_max"
    assert tolerance_for("ops/lam_sorted", prov) == ("rel_l2", HOST_RTOL)


def test_tolerance_tables_record_the_guard_and_the_trim():
    """A golden has to say what it gated and how, or the bound is unreadable."""
    from .gen_solve1d import STATE_KEYS
    from .gen_solve1d import TOLERANCES as TOL_1D
    from .gen_solve2d import TOLERANCES as TOL_2D

    for table, keys, rtol in (
        (TOL_1D, STATE_KEYS, _flux_metric.RTOL_1D),
        (TOL_2D, ("state/", "spectrum/"), _flux_metric.RTOL_2D),
    ):
        for key in keys:
            entry = table[key]
            assert entry["mode"] == "per_species_max"
            assert entry["rtol"] == rtol
            assert entry["floor"] == _flux_metric.FLOOR
            assert entry["guard"] == "sign_definite"
            assert entry["trim_top_bins"] == _flux_metric.TRIM_TOP_BINS


def test_the_bounds_clear_the_measured_spreads():
    """The margins the ruling was taken on, so a later edit cannot erase them.

    1D 1e-11 over the 1.129e-12 of the final states, 2D 3e-8 over the 3.74e-9
    the microkernel pairs spread on the monopole blocks -- both ~8x, the 3-8x
    standard `SECANT_RTOL` is set by. Unguarded the 2D scan reads 1.94e-7,
    which no bound of this size could absorb. The 1D depth-grid stacks are the
    tighter case, at 3.4x; `test_drift_over_every_bounded_key_is_the_real_margin`
    holds that one.
    """
    assert _flux_metric.RTOL_1D / DRIFT_SIGN_DEFINITE > 8.0
    assert _flux_metric.RTOL_2D / 3.74e-9 > 8.0
    assert 1.94e-7 > _flux_metric.RTOL_2D
    assert _flux_metric.RTOL_1D / DRIFT_ALL_STATE_KEYS > 3.0


SOLVE2D_PATH = SOLVE1D_PATH.parent / "solve2d.npz"


def test_solve2d_norm_keys_match_the_committed_section():
    """The hand-written norm list is the one the golden actually stores."""
    from .gen_solve2d import NORM_KEYS

    if not SOLVE2D_PATH.exists():
        pytest.skip("solve2d golden has not been generated here")
    stored = {
        key
        for key in _load_npz(SOLVE2D_PATH)
        if key.startswith("state/") and not key.endswith("_mode0")
    }
    assert stored == set(NORM_KEYS)


def test_solve2d_boundary_artefact_sits_inside_the_trim():
    """The measurement the trim was chosen on, pinned against the golden.

    The 2D muon and neutrino spectra carry negative bins -- a percent-level
    artefact of a 31-bin grid ending at 89.13 GeV with the peak at 14.13 GeV,
    not a rounding residual -- and they sit in bins 28-30, inside the top 3.
    Untrimmed the sign test therefore rejects the whole row; trimmed it admits
    it. If the artefact ever reaches deeper than the trim, the muon rows leave
    the per-species bound again and `TRIM_TOP_BINS` has to be revisited.
    """
    if not SOLVE2D_PATH.exists():
        pytest.skip("solve2d golden has not been generated here")
    arrays = _load_npz(SOLVE2D_PATH)
    trim = _flux_metric.TRIM_TOP_BINS

    # the muon row dips to percent level, the numu row only to ppm; an
    # untrimmed sign test rejects both the same, which is why it is the wrong
    # test to put on a spectrum
    for key, dip in (
        ("spectrum/single_total_mu+", -1.360e-2),
        ("spectrum/single_total_numu", -3.638e-6),
    ):
        row = arrays[key]
        assert not _flux_metric.sign_definite(row)
        assert row.min() / row.max() == pytest.approx(dip, rel=1e-2)
        assert _flux_metric.sign_definite(
            row[: _flux_metric.n_retained(row.size, trim)]
        )

    # every zenith of the carousel, and the negatives are the top bins only
    carousel = arrays["spectrum/carousel_total_mu+"]
    for lane in range(carousel.shape[0]):
        row = carousel[lane]
        assert not _flux_metric.sign_definite(row)
        assert np.where(row < 0)[0].min() >= row.size - trim
        assert _flux_metric.sign_definite(
            row[: _flux_metric.n_retained(row.size, trim)]
        )
    # the artefact deepens with zenith: -1.06% of the peak at 0 deg, -12.9% at 72
    assert carousel[0].min() / carousel[0].max() == pytest.approx(-0.0106, abs=1e-4)
    assert carousel[-1].min() / carousel[-1].max() == pytest.approx(-0.1288, abs=1e-4)


def test_solve2d_trim_does_not_make_the_guard_a_no_op():
    """27 of the 124 spectrum rows are still gated, and they are the right ones.

    Six of the eight multi-RHS lanes are primaries at 30 GeV and below, whose
    spectra oscillate about zero halfway down the grid rather than at its edge;
    the three deepest carousel `nue` lanes (68, 70, 72 deg) reach one bin past
    the trim. Those are cancellation residuals, which is what the sign test is
    for -- the trim moves the edge artefact out of its way, it does not disarm
    it.
    """
    if not SOLVE2D_PATH.exists():
        pytest.skip("solve2d golden has not been generated here")
    arrays = _load_npz(SOLVE2D_PATH)
    trim = _flux_metric.TRIM_TOP_BINS

    rejected = {}
    for key in sorted(k for k in arrays if k.startswith("spectrum/")):
        rows = np.atleast_2d(arrays[key])
        if rows.shape[1] != 31:  # the sky maps are one value per pixel
            continue
        keep = _flux_metric.n_retained(rows.shape[1], trim)
        bad = [
            r
            for r in range(rows.shape[0])
            if not _flux_metric.sign_definite(rows[r][:keep])
        ]
        if bad:
            rejected[key] = bad

    assert sum(len(rows) for rows in rejected.values()) == 27
    assert rejected["spectrum/carousel_total_nue"] == [9, 10, 11]
    for species in ("mu+", "mu-", "nue", "numu"):
        assert rejected[f"spectrum/multirhs_total_{species}"] == [1, 2, 3, 4, 5, 6]
    # and the multi-RHS residuals really are deep, not an edge effect
    row = arrays["spectrum/multirhs_total_mu+"][3]
    assert np.where(row < 0)[0].min() < row.size - 10


#: A spectrum-only layout for the 2D golden. The committed file predates the
#: metric and carries no species table, and none is needed to score a stored
#: spectrum: `dim_states` 0 matches no array, so every key is read as a stack
#: of rows, and only the *length* of `e_grid` matters -- it is what decides
#: whether a row is on the energy axis (31 bins) or is a sky map (one value per
#: pixel), and therefore whether the trim applies.
SOLVE2D_SPECTRUM_LAYOUT = _flux_metric.Layout(
    table=(),
    dim=31,
    dim_states=0,
    e_grid=np.arange(1, 32, dtype=float),
    e_bins=np.arange(1, 33, dtype=float),
)


def test_solve2d_trim_admits_the_named_spectra():
    """What the trim buys on the real 2D golden: 12 admitted rows become 97.

    And with them the fall-through goes away: untrimmed, 24 of the 28 spectrum
    keys had no row the guard would score and were bounded by `fallback_rtol`;
    trimmed, none is. On the state blocks, measured the same way with the
    layout a regenerated golden carries, admission goes from 271 to 951 of
    1800 rows.
    """
    if not SOLVE2D_PATH.exists():
        pytest.skip("solve2d golden has not been generated here")
    arrays = _load_npz(SOLVE2D_PATH)
    keys = sorted(k for k in arrays if k.startswith("spectrum/"))

    counts = {}
    for trim in (0, _flux_metric.TRIM_TOP_BINS):
        admitted = total = declined = 0
        for key in keys:
            entries = _flux_metric.evaluate_key(
                arrays[key], arrays[key], SOLVE2D_SPECTRUM_LAYOUT, trim=trim
            )
            total += len(entries)
            admitted += sum(1 for item in entries if item.guarded)
            declined += _flux_metric.worst_entry(entries) is None
        counts[trim] = (admitted, total, declined)

    assert counts[0] == (12, 124, 24)
    assert counts[_flux_metric.TRIM_TOP_BINS] == (97, 124, 0)


def test_layout_from_provenance_round_trip():
    """The stanza gen_solve2d records reads back as a usable layout."""
    stanza = {
        "species": ["0:a", "1:b"],
        "dim": 2,
        "dim_states": 4,
        "e_grid": [10.0, 100.0],
        "e_bins": [3.0, 30.0, 300.0],
    }
    layout = _flux_metric.layout_from_provenance({"extra": {"species_layout": stanza}})
    assert (layout.dim, layout.dim_states, layout.table) == (2, 4, (("a", 0), ("b", 1)))
    assert _flux_metric.layout_from_provenance({"extra": {}}) is None
    assert _flux_metric.layout_from_provenance(None) is None


def test_layout_for_prefers_the_arrays_over_the_provenance():
    arrays = dict(UNIT_LAYOUT)
    prov = {
        "extra": {
            "species_layout": {
                "species": ["0:z"],
                "dim": 1,
                "dim_states": 1,
                "e_grid": [1.0],
                "e_bins": [0.5, 2.0],
            }
        }
    }
    assert _flux_metric.layout_for("state/x", arrays, prov).dim_states == 4
    assert _flux_metric.layout_for("state/x", {}, prov).dim_states == 1
