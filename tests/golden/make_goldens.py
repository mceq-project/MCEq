"""Regenerate golden sections.

    python -m tests.golden.make_goldens                # every section
    python -m tests.golden.make_goldens paths solve1d  # named sections
    python -m tests.golden.make_goldens --list

Each section runs in this process one after the other; the generators restore
the config globals they mutate, so the order does not matter. A section whose
inputs are absent (a database CI does not carry) reports as skipped rather than
failing the run.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import SECTIONS
from ._harness import save_section

#: Section name -> module in this package providing SECTION and build().
GENERATORS = {
    "structure": "gen_structure",
    "paths": "gen_paths",
    "solve1d": "gen_solve1d",
    "species": "gen_species",
    "rhostack": "gen_rhostack",
    "solve2d": "gen_solve2d",
}


class SectionUnavailable(RuntimeError):
    """The inputs for a section are not present on this machine."""


def load_generator(section: str):
    module = importlib.import_module(f".{GENERATORS[section]}", __package__)
    if module.SECTION != section:
        raise RuntimeError(f"{module.__name__}.SECTION is {module.SECTION!r}, expected {section!r}")
    return module


def regenerate(section: str):
    """Build one section and write it. Returns the path, or None if unavailable."""
    module = load_generator(section)
    started = time.perf_counter()
    try:
        arrays, provenance = module.build()
    except SectionUnavailable as exc:
        print(f"  {section}: SKIPPED ({exc})")
        return None
    path = save_section(section, arrays, provenance)
    print(
        f"  {section}: {len(arrays)} keys, {path.stat().st_size / 1024:.0f} KiB, "
        f"{time.perf_counter() - started:.1f} s -> {path}"
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sections", nargs="*", choices=[*SECTIONS, []], default=[])
    parser.add_argument("--list", action="store_true", help="list section names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in SECTIONS:
            print(name)
        return 0

    wanted = args.sections or list(SECTIONS)
    print(f"regenerating {len(wanted)} section(s)")
    for section in wanted:
        regenerate(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
