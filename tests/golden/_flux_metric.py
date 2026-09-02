"""Per-species max-relative flux metric for the golden harness (plan D18).

A relative L2 over a whole state vector hides a factor-two move in a minor
species behind the muon rows that dominate the norm, and an element-wise
relative tolerance is defeated by the components sitting at ~1e-16 of the peak,
where the reduction order of a BLAS microkernel is the whole answer. This
module scores a state the way the maintainer reads it: species by species, on
the bins that carry flux.

For species `s` and energy bin `E`,

    rel_s(E) = |phi(E) - phi_ref(E)| / |phi_ref(E)|,

evaluated only over the *retained* bins -- the grid with its top `trim` bins
dropped -- and there only where

    phi_ref(E) >= floor * max_retained phi_ref   and   phi_ref(E) != 0,

with `floor` = `FLOOR` and `trim` = `TRIM_TOP_BINS`. The score of the species
is `max_E rel_s(E)`; the score of a stored key is the worst score over its
species and its lanes (the columns of a batched state, the rows of a depth-grid
stack).

**Guard** (maintainer ruling, 2026-09-02): a species enters the maximum only if
its reference is *sign definite over the retained bins*,
`(phi_ref[:-trim] >= 0).all()`. Two rulings in one:

*Sign definiteness.* A species with a negative bin is not a flux but a residual
of two cancelling transport terms -- `e+_l` at 71 TeV in the 1D `emon` case
moves 1.35e-9 across the phi-Taylor retune while every admitted species stays
under 1.2e-12 -- and a relative tolerance on a cancellation measures the
cancellation, not the solver.

*The edge trim.* Applied to the whole spectrum the sign test disqualifies
nearly everything, because both solves carry an upper-boundary artefact in the
top few bins: the committed 2D golden's `spectrum/carousel_total_mu+` is
negative in bins 28-30 of 31 on a grid ending at 89.1 GeV whose peak sits at
14.1 GeV, from -1.06 % of the peak at 0 deg to -12.88 % at 72 deg. An
all-or-nothing sign test therefore rejects a species over two edge bins:
untrimmed it admits 12 of the 124 2D spectrum rows and none of `total_mu+-`,
`numu`, `nue`, `pi+-`, `K+-` or `p+`. Trimming the top bins *entirely* -- from
the sign test and from the scored bins alike, because an untrustworthy
reference value is untrustworthy for both -- restores them: 2D spectrum
admission runs 12 (N=0), 40 (N=2), 97 (N=3), 100 (N=4) of 124, so the knee is
at 3. The 1D grid agrees independently: of the species with a negative bin, 88
are negative only in the top bin, 33 down to depth 1 and 3 to depth 2, then a
gap, then genuinely sign-oscillating species at depths 6, 8, 9, 10 and 30. N=3
covers the artefact with one bin of margin, and 1D admission saturates there
(259 of 278 entries, identical at N=4 and N=5).

Bounds from the same ruling: `RTOL_1D` for the 1D sections, `RTOL_2D` for the
2D secant routes, where the dense mode-coupling GEMMs give a wider microkernel
spread. The guard name, the floor and the trim travel in the section's
`__provenance__` tolerance entry, so a golden says which species it gated and
how.

Residual, worth knowing before the bound is ever tightened: the artefact leaks
one bin past the trim. Every worst guarded 2D row sits at 44.67 or 35.48 GeV --
the top retained bins -- and restricting the scan to E <= 30 GeV drops the
microkernel spread 7x, from 3.74e-9 to 5.24e-10.

The species layout of a state comes from the section's own `meta/` stanza
(`species` as `"<mceqidx>:<name>"`, plus `dim`, `dim_states`, `e_grid`,
`e_bins`); the slice `[mceqidx*dim : (mceqidx+1)*dim]` is the lidx/uidx window
`MCEqRun._get_solution_from_state` reads. A section that cannot add `meta/`
arrays without moving its own golden records the same fields under
`__provenance__["extra"]["species_layout"]` instead.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Energies the failure table reports, in GeV.
FIXED_E_GEV = (10.0, 100.0, 1e3, 1e4, 1e5, 1e6)

#: Rows of the failure table: the fluxes a comparison against a Monte Carlo is
#: actually read on, in the names `get_solution` uses.
KEY_SPECIES = (
    "total_mu+",
    "total_mu-",
    "numu",
    "antinumu",
    "nue",
    "antinue",
    "pi+",
    "pi-",
    "K+",
    "K-",
    "p+",
    "n0",
)

#: Bins below `FLOOR * peak` of the reference carry no flux and are dropped.
FLOOR = 1e-12

#: Energy bins dropped from the top of the grid, sign test and score alike.
#: The knee of 2D spectrum admission (12 rows at N=0, 40 at N=2, 97 at N=3, 100
#: at N=4 of 124) and one bin of margin over the deepest 1D artefact (depth 2).
TRIM_TOP_BINS = 3

#: Per-species bounds (maintainer ruling 2026-09-02). Each clears the measured
#: worst case by 3.4-8.9x, the 3-8x standard the harness sets elsewhere.
#:
#: 1D: 1.129e-12 across the phi-Taylor retune on the final states (`k_mu+_l`
#: at 3548 GeV, `emoff/theta89/state`), so 1e-11 carries 8.9x there. Include
#: the depth-grid stacks, which the same entry bounds, and the worst is
#: 2.915e-12 (`e+_l` at 447 TeV, `emon/grid/grid_sol[0]`, where the EM
#: cascade is young enough to still be sign definite) -- 3.4x, the thinnest
#: margin in the harness.
#:
#: 2D: 3.74e-9 across six OpenBLAS-microkernel pairs on one host, on the raw
#: `state/*_mode0` blocks (`carousel` lane 4 / 40 deg, `k_mu-`, 44.67 GeV), so
#: 3e-8 carries 8.0x. The `spectrum/` rows are 44x quieter at 8.4e-11, so the
#: state blocks set the bound; unguarded the same scan reads 1.94e-7. The
#: spread barely depends on the trim -- 3.18e-9 (N=2), 3.74e-9 (N=3), 3.18e-9
#: (N=4), a factor-1.18 plateau -- so neither does the bound.
RTOL_1D = 1e-11
RTOL_2D = 3e-8

#: Guard recorded in a tolerance entry unless it names another.
DEFAULT_GUARD = "sign_definite"


def n_retained(size: int, trim: int) -> int:
    """Bins the metric scores: `size - trim`, clamped at zero."""
    return max(size - max(trim, 0), 0)


def sign_definite(reference) -> bool:
    """True when the reference has no negative bin, i.e. it is a flux.

    Applied to the *retained* bins -- the caller trims the grid edge first --
    because the top bins of both solves carry a boundary artefact whose sign
    says nothing about the species.
    """
    return bool((np.asarray(reference) >= 0.0).all())


def ungated(reference) -> bool:
    """Admit every species, `reference` notwithstanding.

    The unguarded metric, for diagnosis: it is what reports 1.35e-9 on `e+_l`
    where the guarded one reports 1.13e-12.
    """
    return True


#: Guard predicates a tolerance entry may name.
GUARDS = {"sign_definite": sign_definite, "none": ungated}


class Unscorable(Exception):
    """No guarded species of a key keeps a bin above the floor.

    The per-species maximum is then not defined for that key and the caller has
    to fall back to a bound that is: `compare_key` uses relative L2 at the
    entry's `fallback_rtol`.

    Under the edge-trimmed guard no key of either committed section reaches
    here -- untrimmed, 24 of the 28 2D spectrum keys did -- so the fallback is
    now a guard against a key that carries no flux at all rather than a routine
    path. It is still reachable per key, not per lane: the multi-RHS zero
    right-hand side is one identically-zero *lane* of `state/multirhs_mode0`,
    and its seven siblings keep the key scorable.
    """


# --------------------------------------------------------------------------
# species layout
# --------------------------------------------------------------------------


class Layout(NamedTuple):
    """The species layout and energy grid of one section's state vectors."""

    table: tuple  # ((name, mceqidx), ...) in stored order
    dim: int
    dim_states: int
    e_grid: np.ndarray
    e_bins: np.ndarray


def parse_species(entries) -> tuple:
    """`("0:antinue", ...)` -> `(("antinue", 0), ...)`, the `meta/species` form."""
    table = []
    for item in entries:
        index, name = str(item).split(":", 1)
        table.append((name, int(index)))
    return tuple(table)


def layout_from_arrays(arrays: dict, prefix: str = "") -> Layout:
    """Read the `<prefix>meta/` stanza a generator stored in its arrays."""
    return Layout(
        table=parse_species(arrays[prefix + "meta/species"]),
        dim=int(arrays[prefix + "meta/dim"]),
        dim_states=int(arrays[prefix + "meta/dim_states"]),
        e_grid=np.asarray(arrays[prefix + "meta/e_grid"], dtype=np.float64),
        e_bins=np.asarray(arrays[prefix + "meta/e_bins"], dtype=np.float64),
    )


def layout_from_provenance(provenance) -> Layout | None:
    """Read `extra.species_layout`, or `None` when the section has no stanza.

    `compare_section` reports an array key that is not in the golden as a
    mismatch, so a section whose golden predates this metric cannot grow
    `meta/` arrays without moving its own file; it records the layout here,
    which is compared against nothing.
    """
    stanza = ((provenance or {}).get("extra") or {}).get("species_layout")
    if not stanza:
        return None
    return Layout(
        table=parse_species(stanza["species"]),
        dim=int(stanza["dim"]),
        dim_states=int(stanza["dim_states"]),
        e_grid=np.asarray(stanza["e_grid"], dtype=np.float64),
        e_bins=np.asarray(stanza["e_bins"], dtype=np.float64),
    )


def layout_for(key: str, arrays: dict, provenance=None) -> Layout | None:
    """The layout governing `key`: its longest `meta/` prefix, else provenance.

    `emoff/theta89/state` resolves through `emoff/meta/`, so the two cases of
    the 1D section keep their own species tables and dimensions.
    """
    segments = key.split("/")
    for cut in range(len(segments) - 1, -1, -1):
        prefix = "".join(part + "/" for part in segments[:cut])
        if prefix + "meta/species" in arrays:
            return layout_from_arrays(arrays, prefix)
    return layout_from_provenance(provenance)


def split_species(state, layout: Layout) -> dict:
    """`{name: (dim,) spectrum}` for one `(dim_states,)` state vector.

    The helicity-summed rows `total_mu+` and `total_mu-` are appended under the
    names `get_solution` reports, because that sum -- not the polarised
    component -- is the flux a result is quoted on.
    """
    state = np.asarray(state, dtype=np.float64)
    out = {}
    for name, index in layout.table:
        out[name] = state[index * layout.dim : (index + 1) * layout.dim]
    for lepton in ("mu+", "mu-"):
        parts = [out[n] for n in (lepton, lepton + "_l", lepton + "_r") if n in out]
        if parts:
            out["total_" + lepton] = np.sum(parts, axis=0)
    return out


def state_lanes(array, dim_states: int):
    """`[(label, state)]` for the state vectors packed in `array`, else `None`.

    A `(dim_states, K)` block holds one state per column -- `solve_batch`
    stitches its lanes that way -- and an `(n, dim_states)` block one per row,
    which is how `grid_sol` stacks its depth snapshots.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return [("", values)] if values.size == dim_states else None
    if values.ndim == 2 and values.shape[0] == dim_states:
        return [(f"[:,{c}]", values[:, c]) for c in range(values.shape[1])]
    if values.ndim == 2 and values.shape[1] == dim_states:
        return [(f"[{r}]", values[r]) for r in range(values.shape[0])]
    return None


def spectrum_rows(array):
    """`[(label, row)]` for a stored spectrum: the array, or one row each."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return [("", values)]
    if values.ndim == 2:
        return [(f"[{r}]", values[r]) for r in range(values.shape[0])]
    return None


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------


class SpeciesScore(NamedTuple):
    """The score of one species, and the per-bin ratios behind it.

    `rel_raw` and `keep` stay the full length of the stored row -- the trim
    shows up as `keep` being False on the top `n_trimmed` bins -- so the
    fixed-energy table can index them by grid bin.
    """

    max_rel: float  # nan when the floor keeps nothing
    at_e: float
    n_kept: int
    peak: float
    rel: np.ndarray  # nan where the floor or the trim drops the bin
    rel_raw: np.ndarray  # unfloored, untrimmed, for the fixed-energy table
    keep: np.ndarray
    n_trimmed: int = 0


class Entry(NamedTuple):
    """One species of one lane of one key."""

    species: str
    lane: str
    score: SpeciesScore
    guarded: bool  # the guard admits it
    on_e_grid: bool = True  # False for a sky map, whose axis is the pixel


def where(entry: Entry) -> str:
    """Where an entry's maximum sits, as an energy or as a bin index."""
    if entry.on_e_grid:
        return f"E = {entry.score.at_e:.4g} GeV"
    return f"index {int(entry.score.at_e)}"


def species_metric(
    reference,
    actual,
    e_grid,
    floor: float = FLOOR,
    trim: int = TRIM_TOP_BINS,
) -> SpeciesScore:
    """`max_E rel` over the bins the trim and the floor keep, and where it sits.

    The top `trim` bins are dropped before anything else, so `peak` -- and
    therefore the floor -- is taken over the retained bins only. `peak` is the
    signed maximum: a reference with no positive retained bin keeps no bin and
    scores nan, which is how a species that is identically zero, or one that is
    a pure cancellation residual, drops out instead of dividing by a value the
    solver never produced.
    """
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    stop = n_retained(reference.size, trim)
    scored = np.zeros(reference.shape, dtype=bool)
    scored[:stop] = True
    peak = reference[:stop].max() if stop else 0.0
    keep = (
        scored & (reference >= floor * peak)
        if peak > 0
        else np.zeros(reference.shape, dtype=bool)
    )
    keep &= reference != 0.0
    rel = np.full(reference.shape, np.nan)
    rel[keep] = np.abs(actual[keep] - reference[keep]) / np.abs(reference[keep])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_raw = np.abs(actual - reference) / np.abs(reference)
    n_trimmed = reference.size - stop
    if not keep.any():
        return SpeciesScore(
            float("nan"), float("nan"), 0, float(peak), rel, rel_raw, keep, n_trimmed
        )
    index = int(np.nanargmax(rel))
    return SpeciesScore(
        float(rel[index]),
        float(np.asarray(e_grid, dtype=np.float64)[index]),
        int(keep.sum()),
        float(peak),
        rel,
        rel_raw,
        keep,
        n_trimmed,
    )


def evaluate_key(
    expected,
    actual,
    layout: Layout,
    *,
    floor: float = FLOOR,
    guard: str = DEFAULT_GUARD,
    trim: int = TRIM_TOP_BINS,
):
    """`[Entry]` for one key, or `None` when its shape carries no flux.

    A key whose shape contains `dim_states` is split into species; anything
    else is read as a stack of spectra, one score per row, on the energy grid
    when the row length matches it and on the bin index otherwise (the sky maps
    are one value per pixel).

    The guard sees the same retained bins the score does. A row that is not on
    the energy grid is not trimmed at all: `trim` names the top bins of an
    energy grid, and a sky map's axis is the pixel, which has no top.
    """
    admits = GUARDS[guard]
    entries: list[Entry] = []

    lanes = state_lanes(expected, layout.dim_states)
    if lanes is not None:
        produced = dict(state_lanes(actual, layout.dim_states) or [])
        for label, reference in lanes:
            reference_species = split_species(reference, layout)
            actual_species = split_species(produced[label], layout)
            for name, values in reference_species.items():
                entries.append(
                    Entry(
                        name,
                        label,
                        species_metric(
                            values, actual_species[name], layout.e_grid, floor, trim
                        ),
                        admits(values[: n_retained(values.size, trim)]),
                    )
                )
        return entries

    rows = spectrum_rows(expected)
    if rows is None:
        return None
    produced = dict(spectrum_rows(actual) or [])
    for label, reference in rows:
        on_e_grid = reference.size == layout.e_grid.size
        grid = (
            layout.e_grid if on_e_grid else np.arange(reference.size, dtype=np.float64)
        )
        row_trim = trim if on_e_grid else 0
        entries.append(
            Entry(
                "",
                label,
                species_metric(reference, produced[label], grid, floor, row_trim),
                admits(reference[: n_retained(reference.size, row_trim)]),
                on_e_grid,
            )
        )
    return entries


def worst_entry(entries, *, guarded_only: bool = True) -> Entry | None:
    """The entry with the largest finite score, or `None` if there is none."""
    pool = [
        entry
        for entry in entries
        if (entry.guarded or not guarded_only) and np.isfinite(entry.score.max_rel)
    ]
    return max(pool, key=lambda entry: entry.score.max_rel) if pool else None


def worst_by_species(entries) -> dict:
    """`{species: Entry}` -- each species scored on its worst lane."""
    best: dict[str, Entry] = {}
    for entry in entries:
        if not np.isfinite(entry.score.max_rel):
            continue
        previous = best.get(entry.species)
        if previous is None or entry.score.max_rel > previous.score.max_rel:
            best[entry.species] = entry
    return best


def compare(
    key: str,
    expected,
    actual,
    rtol: float,
    *,
    layout: Layout,
    floor: float = FLOOR,
    guard: str = DEFAULT_GUARD,
    trim: int = TRIM_TOP_BINS,
) -> str | None:
    """`None` when every guarded species is within `rtol`, else the reason.

    Raises `Unscorable` when the guard, the trim and the floor leave nothing to
    score; the key then has no per-species maximum and the caller has to bound
    it another way.
    """
    entries = evaluate_key(
        expected, actual, layout, floor=floor, guard=guard, trim=trim
    )
    if entries is None:
        return (
            f"{key}: shape {np.shape(expected)} is neither a state nor a "
            f"spectrum stack (dim_states {layout.dim_states}), so "
            f"per_species_max cannot score it"
        )
    worst = worst_entry(entries)
    if worst is None:
        raise Unscorable(
            f"{key}: no species the {guard} guard admits over the grid less its "
            f"top {trim} bin(s) keeps a bin above {floor:.0e} x peak"
        )
    if worst.score.max_rel <= rtol:
        return None
    return (
        f"{key}{worst.lane}: per-species max {worst.score.max_rel:.3e} > {rtol:.1e}"
        f" on {worst.species or 'the row'} at {where(worst)}"
        f" ({worst.score.n_kept} bins above floor {floor:.0e}, guard {guard},"
        f" top {worst.score.n_trimmed} bin(s) trimmed)"
    )


# --------------------------------------------------------------------------
# failure report
# --------------------------------------------------------------------------


class FixedBin(NamedTuple):
    """One column of the fixed-energy table."""

    target: float
    index: int
    e_centre: float
    in_grid: bool


def nearest_bins(e_grid, e_bins, targets=FIXED_E_GEV) -> list:
    """The bin nearest each target in log E, flagged when the target is off-grid."""
    e_grid = np.asarray(e_grid, dtype=np.float64)
    e_bins = np.asarray(e_bins, dtype=np.float64)
    bins = []
    for target in targets:
        index = int(np.argmin(np.abs(np.log(e_grid) - np.log(target))))
        bins.append(
            FixedBin(
                target,
                index,
                float(e_grid[index]),
                bool(e_bins[0] <= target <= e_bins[-1]),
            )
        )
    return bins


def _cell(score: SpeciesScore, bin_: FixedBin) -> str:
    """The ratio at one fixed energy, bracketed when the bin is not scored.

    `[...]` is a bin the edge trim drops, `(...)` one the floor drops; both are
    shown unfloored, because the maintainer still wants to see how far the edge
    moved even though it is not what the bound is applied to.
    """
    if bin_.index >= score.rel_raw.size:
        return "-"
    value = score.rel_raw[bin_.index]
    if not np.isfinite(value):
        return "(ref=0)"
    text = "0" if value == 0.0 else f"{value:.1e}"
    if bin_.index >= score.rel_raw.size - score.n_trimmed:
        return f"[{text}]"
    return text if score.keep[bin_.index] else f"({text})"


def fixed_energy_table(
    entries, layout: Layout, *, species=KEY_SPECIES, targets=FIXED_E_GEV
) -> list:
    """Table lines: one row per species, one column per fixed energy.

    Every named species is tabulated, guarded in or not: the guard decides what
    the bound is applied to, the table is what the maintainer reads to see
    whether a move is physics, so it must show the muon and neutrino rows even
    where the guard declines to score them.

    A value in square brackets is one of the top bins the edge trim drops; one
    in parentheses is a bin the floor excludes; both are shown unfloored.
    `(ref=0)` is a bin the reference does not populate; `-` is a column past
    the end of the row (a sky map has one value per pixel, not per energy). A
    column marked `*` is a target outside the stored grid, mapped onto the
    nearest grid centre in log E.
    """
    best = worst_by_species(entries)
    if not best:
        return []
    bins = nearest_bins(layout.e_grid, layout.e_bins, targets)
    names = [name for name in species if name in best] or sorted(best)

    header = ["species", "max rel", "E (GeV)"] + [
        f"{b.target:g}" + ("" if b.in_grid else "*") for b in bins
    ]
    rows = [header]
    for name in names:
        entry = best[name]
        rows.append(
            [
                (name or "row") + entry.lane,
                f"{entry.score.max_rel:.2e}",
                f"{entry.score.at_e:.4g}",
            ]
            + [_cell(entry.score, b) for b in bins]
        )

    widths = [max(len(row[c]) for row in rows) for c in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows]
    off_grid = [b for b in bins if not b.in_grid]
    if off_grid:
        lines.append(
            "* off-grid target -> nearest centre: "
            + ", ".join(f"{b.target:g} -> {b.e_centre:.4g}" for b in off_grid)
        )
    return lines


def flux_report(
    golden: dict,
    produced: dict,
    *,
    keys,
    rtol: float,
    provenance=None,
    floor: float = FLOOR,
    guard: str = DEFAULT_GUARD,
    trim: int = TRIM_TOP_BINS,
    species=KEY_SPECIES,
    limit: int = 12,
) -> list:
    """Failure lines for the flux keys of a section.

    The worst guarded species of every gated key, largest first, then the
    fixed-energy table of the worst key -- which is the table the maintainer
    reads to decide whether a move is physics or a reduction order. Keys the
    guard declines entirely are counted, not scored: they are bounded by the
    entry's `fallback_rtol` and their mismatch lines say so.
    """
    scored = []
    declined = []
    for key in keys:
        if key not in golden or key not in produced:
            continue
        layout = layout_for(key, golden, provenance)
        if layout is None:
            continue
        entries = evaluate_key(
            golden[key], produced[key], layout, floor=floor, guard=guard, trim=trim
        )
        if entries is None:
            continue
        worst = worst_entry(entries)
        if worst is None:
            declined.append(key)
            continue
        scored.append((worst.score.max_rel, key, layout, entries, worst))
    if not scored:
        return []

    scored.sort(key=lambda item: -item[0])
    lines = [
        f"per-species max |dphi/phi_ref| over {len(scored)} flux key(s): "
        f"floor {floor:.0e} x peak, guard {guard}, top {trim} bin(s) trimmed, "
        f"bound {rtol:.1e}"
    ]
    if declined:
        lines.append(
            f"  {len(declined)} further key(s) have no species the {guard} guard "
            f"admits and are judged by fallback_rtol instead: "
            f"{declined[:4]}{' ...' if len(declined) > 4 else ''}"
        )
    for max_rel, key, _, entries, worst in scored[:limit]:
        gated = sum(1 for entry in entries if not entry.guarded)
        lines.append(
            f"  {key}{worst.lane}: {max_rel:.3e} on {worst.species or 'the row'}"
            f" at {where(worst)}"
            f" ({'FAIL' if max_rel > rtol else 'ok'}, {gated} species gated out)"
        )
    if len(scored) > limit:
        lines.append(f"  ... {len(scored) - limit} further key(s) not listed")

    _, key, layout, entries, _ = scored[0]
    lines.append(f"fixed-energy table for {key}:")
    lines += [
        "  " + line for line in fixed_energy_table(entries, layout, species=species)
    ]
    return lines
