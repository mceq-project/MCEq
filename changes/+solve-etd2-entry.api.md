One entry point, `MCEq.solvers.solve_etd2(nsteps, dX, rho_inv, int_m, dec_m, phi,
grid_idcs, *, backend, sec_ops, schedule, phi0_per_pixel, device_id, fp_precision)`,
**replaces** the twelve thin per-backend names left over from the hand-unrolled
kernels: `solv_{numpy,mkl,cuda}_etd2` and their `_carousel`, `_secant`,
`_secant_carousel` variants, plus the six `_multirhs` names the `_multirhs` closure
factory generated from them. None of the removed names had a caller outside the test
suite. It compiles the operator, binds the named backend (`"numpy"`, `"mkl"`,
`"cuda"`), runs `etd2_driver` and closes the backend, so the library handles and
device buffers a solve allocates are released with it. Route and transport are
keywords now, not separate functions. `MCEqRun` was already on `compile_operator` + a
backend + the driver and is unchanged.

The `_multirhs` lift's only behaviour, a `ValueError` on a 1-D `phi`, is **gone**: it
guarded a name that promised a batch, and `solve_etd2` promises only that the solution
has the rank of `phi`. The user-facing 2-D requirement is unchanged at the API boundary
(`MCEqRun.solve_multirhs`).

**Removed** BSR storage for the MKL off-diagonals along with `mkl_backend(blocksize=)`:
the factory is `mkl_backend(op, expected_calls=2000, fp_precision=64)` and
`MklSparseMatrix` is CSR only. No solver path ever selected BSR — the driver has been
CSR-only on every backend since the operator-assembly layer landed — so it was
reachable only from a test that existed to cover it.
