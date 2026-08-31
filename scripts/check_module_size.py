#!/usr/bin/env python3
"""Fail when a module exceeds the line budget of the layered architecture.

The target layout (refactoring plan, section 1) has no module longer than
``LIMIT`` lines. The modules still over budget are listed in ``ALLOW`` with the
size they had at the Phase-0 commit. That table is a ratchet:

* a module not in ``ALLOW`` may never exceed ``LIMIT``;
* a module in ``ALLOW`` may never exceed the size recorded for it;
* an entry whose module now fits, or whose file is gone, is stale and fails,

so a phase that splits a module has to shrink the table in the same PR, and the
table reaches ``{}`` at the end of Phase 7. ``--update`` prints the replacement.

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
LIMIT = 600

#: Modules over budget at the Phase-0 commit (06fdd7e), with the phase that
#: removes each entry. 8 modules, 9062 lines above the limit.
ALLOW = {
    "src/MCEq/config/__init__.py": 669,  # Phase 1 -> config/{schema,defaults,legacy,detect}
    "src/MCEq/core.py": 3667,  # Phase 5 (MatrixBuilder) + Phase 6 (driver/*)
    "src/MCEq/data.py": 1503,  # Phase 4 -> data/*
    "src/MCEq/ddm.py": 870,  # Phase 4 -> models/ddm/*
    "src/MCEq/geometry/density_profiles.py": 1846,  # Phase 3 -> environment/*
    "src/MCEq/geometry/msis21_atmosphere.py": 670,  # Phase 3 -> environment/*
    "src/MCEq/particlemanager.py": 1176,  # Phase 4 -> species/*
    "src/MCEq/solvers.py": 3571,  # Phase 2 -> solvers/* + backends/*
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

    ``+`` marks a module that has to shrink, ``-`` an allowlist entry to delete.
    """
    over_limit, grown, now_fits, gone = [], [], [], []
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
