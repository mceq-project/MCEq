**Replaced** the twelve ``solv_<backend>_etd2[_secant][_carousel]`` entry points
and the six ``_multirhs`` names generated from them with a single
``MCEq.solvers.solve_etd2(..., backend="numpy"|"mkl"|"cuda")``, which compiles
the operator, binds the backend and releases the handles it created. None of the
removed names had a caller outside the test suite; ``MCEqRun`` caches its own
operator and backend and calls ``etd2_driver`` directly.

**Removed** BSR storage for the MKL off-diagonals along with
``mkl_backend(blocksize=)``. No solver path ever selected it — the driver has
been CSR-only on every backend since the operator-assembly layer landed — so it
was reachable only from a test that existed to cover it.
