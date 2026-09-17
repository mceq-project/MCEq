`MCEq.config` no longer overwrites an `MKL_NUM_THREADS` / `OMP_NUM_THREADS` /
`OPENBLAS_NUM_THREADS` already set in the environment: that value becomes the
process-wide BLAS budget, and without one the default is `min(16, cpus)` over
the CPUs the process may actually use (affinity mask) rather than the host's
core count. The test suite's root conftest divides the CPUs among pytest-xdist workers
the same way, so `-n 8` no longer runs eight 16-thread BLAS pools on a shared node.
