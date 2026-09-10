fp32 is a dtype on the shared ETD2 backends rather than a separate kernel
family. ``MklSparseMatrix`` takes ``dtype=``, so ``MklSparseMatrixF32`` and the
``solv_mkl_etd2_multirhs_f32`` kernel are removed, and
``solve_batch(dtype=float32)`` now works on ``numpy_etd2`` as well as MKL and
CUDA.

The precision contract is stated once, as ``solvers._PRECISION_CONTRACT``: the
diagonals and the φ factors computed from them are evaluated in FP64 whatever
the requested precision and cast once on the way out; the state, the scratch
buffers and the off-diagonals live in the requested dtype.

**Bug fix.** ``CudaOperator`` stored its diagonals in the state dtype, so at
``fp_precision=32`` it computed ``exp(h·D)`` from diagonals already rounded to
fp32 — 4.2e-06 to 7.1e-06 away from the fp64 answer where fp32 roundoff is
5.9e-08, and 7e-06 away from what the host computed for the same step. The
diagonals are FP64 on every backend now, and the stage-1 factors are bitwise
identical between CUDA and the host at fp32.

MKL fp32 is also 3.3× faster (1.42 s → 0.44 s for a K=64 solve on the SIBYLL21
reduced fixture), because the shared row-major driver replaces the deleted
column-major tiled kernel.
