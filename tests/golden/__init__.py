"""Golden regression harness for the v2 layered-architecture refactor.

Each *section* is one `.npz` under `data/`, produced by a `gen_*.py` module that
exposes `SECTION` and `build() -> (arrays, provenance)`, and compared by
`test_goldens.py`. The sections were generated at commit 2156403 (the tree PR
#179 merges into `2d-on-v2`) and every later refactoring phase must keep them
green, or state in its pull request which numbers changed and why.

Regenerate with `python -m tests.golden.make_goldens [section ...]` or
`pytest tests/golden --regenerate-goldens`.
"""

SECTIONS = (
    "structure",
    "paths",
    "solve1d",
    "species",
    "solve2d",
)

#: Sections that need a database CI does not carry or cost minutes to build.
SLOW_SECTIONS = frozenset({"solve2d"})
