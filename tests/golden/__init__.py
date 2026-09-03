"""Golden regression harness for the v2 layered-architecture refactor.

Each *section* is one `.npz` under `data/`, produced by a `gen_*.py` module that
exposes `SECTION` and `build() -> (arrays, provenance)`, and compared by
`test_goldens.py`. Every refactoring phase must keep them green, or state in
its pull request which numbers changed and why.

Each section carries its own provenance and is regenerated on its own, so there
is no one commit they all belong to: `__provenance__["git_commit"]` is the HEAD
`make_goldens` saw, section by section. That is the *parent* of the commit that
ships the file — generation necessarily precedes the commit that contains it —
so a section records the tree it was built from under the SHA before it.

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

#: Sections pinned bitwise to the numpy/BLAS build of the host that generated
#: them, and therefore compared on one designated reference job instead of
#: across the platform matrix. The discriminator is the build, not the OS: on
#: ubuntu-latest x86-64 with MKL these pass under numpy 2.5 / scipy 1.18 and
#: fail by 1e-14 relative L2 under numpy 2.2 / scipy 1.15 and numpy 2.4 /
#: scipy 1.17, and the sha256 digest keys are bitwise by construction.
#: `structure` is deliberately absent: line counts are portable, so it stays the
#: cheap guard that the harness is still wired up on every platform. `solve2d`
#: is here as well as in SLOW_SECTIONS — the two marks are orthogonal.
HOST_SECTIONS = frozenset({"paths", "solve1d", "species", "solve2d"})
