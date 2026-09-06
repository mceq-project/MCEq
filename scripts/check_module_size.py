#!/usr/bin/env python3
"""Fail when a module exceeds the line budget of the layered architecture.

The target layout (refactoring plan, section 1) has no module longer than
``LIMIT`` lines. The modules still over budget are listed in ``ALLOW`` with the
size measured at the latest re-pin. The rule is recorded == measured for every
allowlisted module, which makes the table a ratchet:

* a module not in ``ALLOW`` may never exceed ``LIMIT``;
* a module in ``ALLOW`` sits exactly at its recorded size: the gate fails with
  a grown line when it exceeds the entry and with a re-pin line when it sits
  below it, so shrinking a module tightens the budget automatically;
* an entry whose module now fits, or whose file is gone, is stale and fails,

so a phase that splits a module has to shrink the table and a phase that only
shrinks one has to re-pin the entry in the same PR -- ``--update`` prints the
replacement -- and the table reaches ``{}`` at the end of Phase 7.

The line count is ``wc -l`` semantics -- newlines, not text lines -- so the
numbers here match the plan and the shell. The empty
``src/MCEq/geometry/__init__.py`` is the case that separates the two.

Stdlib only and no configuration file: the gate runs in a job that neither
builds MCEq nor installs anything, on any Python the repository supports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOTS = ("src/MCEq", "mceq_config")
LIMIT = 700

#: The limit was 600 through Phase 3 (2026-09-06: raised to 700 -- modules in
#: the 500-700 band read as complete units, not arbitrary shingles; the split
#: targets, not the limit, carry the design).
#:
#: Modules still over budget, with the phase that removes each entry. Each
#: number is the size measured at the latest re-pin (2026-09-06), not the
#: Phase-0 size: an extraction commit re-tightens its entry in place with
#: ``--update``, so recorded == measured and no entry ever grants regrowth
#: headroom.
ALLOW = {
    "src/MCEq/core.py": 1653,  # Phase 5 (MatrixBuilder) + Phase 6 (driver/*)
    "src/MCEq/models/ddm/ddm.py": 874,  # B7 fix re-pin 2026-09-06; split by a later item 7 follow-up
}


def count_lines(path: Path) -> int:
    """Number of newlines in `path`, which is what ``wc -l`` reports."""
    return path.read_bytes().count(b"\n")


def collect() -> dict[str, int]:
    """Repo-relative path -> line count for every module the gate covers."""
    return {
        path.relative_to(ROOT).as_posix(): count_lines(path)
        for root in ROOTS
        for path in sorted((ROOT / root).rglob("*.py"))
    }


def report(sizes: dict[str, int]) -> list[str]:
    """Lines of a diff-style report; empty when the tree is within budget.

    ``+`` marks a module that has to shrink, ``-`` an allowlist entry to delete
    or to re-pin to a smaller measured size.
    """
    over_limit, grown, shrunk, now_fits, gone = [], [], [], [], []
    for path, lines in sorted(sizes.items()):
        recorded = ALLOW.get(path)
        if recorded is None:
            if lines > LIMIT:
                over = lines - LIMIT
                over_limit.append(f"  + {path}: {lines} lines (+{over} over {LIMIT})")
        elif lines > recorded:
            grew = lines - recorded
            grown.append(f"  + {path}: {lines} lines (+{grew} over {recorded})")
        elif lines <= LIMIT:
            now_fits.append(f'  - "{path}": {recorded},   # now {lines} lines')
        elif lines < recorded:
            shrunk.append(f'  - "{path}": {recorded},   # now {lines} lines; re-pin')
    for path, recorded in sorted(ALLOW.items()):
        if path not in sizes:
            gone.append(f'  - "{path}": {recorded},   # file no longer exists')

    out = []
    if over_limit:
        out += [f"Over the {LIMIT}-line limit and not in ALLOW:", *over_limit]
        out += ["  split it, or add it to ALLOW with the phase that removes it.", ""]
    if grown:
        out += ["Allowlisted modules that grew:", *grown]
        out += ["  the allowlist is a ratchet; move the code out instead.", ""]
    if shrunk:
        out += ["Allowlisted modules that shrank:", *shrunk]
        out += [
            "  recorded == measured: re-pin these ALLOW lines to the measured size.",
            "",
        ]
    if now_fits or gone:
        out += ["Stale ALLOW entries:", *now_fits, *gone]
        out += ["  delete these lines from ALLOW.", ""]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="print the ALLOW table for the current tree instead of checking it",
    )
    args = parser.parse_args(argv)
    sizes = collect()

    if args.update:
        print("ALLOW = {")
        for path, lines in sorted(sizes.items()):
            if lines > LIMIT:
                print(f'    "{path}": {lines},')
        print("}")
        return 0

    problems = report(sizes)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print("check_module_size: FAILED", file=sys.stderr)
        return 1

    debt = sum(lines - LIMIT for lines in ALLOW.values())
    print(
        f"check_module_size: OK -- {len(sizes)} modules, limit {LIMIT}, "
        f"{len(ALLOW)} allowlisted, {debt} lines over the limit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
