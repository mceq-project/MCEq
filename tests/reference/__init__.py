"""Stored MCEq results, and the comparison that says they have not moved.

`values.py` runs a small, fixed set of calculations and returns what they
produced; `test_reference_solutions.py` compares that against the numbers in
`data/reference_solutions.npz`, to a tolerance. The point is to notice when a
change to the code changes a flux it should not have changed.

The numbers are regenerated with

    python -m tests.reference.values

against the CI database, which is small enough to ship and to cache.

The much larger set of reference numbers used during the v2 refactor -- full
operator and spline dumps, pinned bitwise to one numpy/BLAS build -- is a
maintenance instrument and lives in mceq-maintenance-tools under `goldens/`.
"""
