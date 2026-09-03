Phase 0 of the layered-architecture refactor: a golden regression harness and
two structural CI gates, with no changes under `src/`.

`tests/golden/` pins six sections at this commit — module inventory and import
graph, integration paths, 1D solve and `get_solution`, the data/species layer,
the numpy EM rho-stack, and the 2D FLUKA secant routes. `scripts/check_module_size.py`
enforces the 600-line module limit against a per-module allowlist, and
`.importlinter` expresses the target layering with an enumerated ignore ledger
that later phases delete from.
