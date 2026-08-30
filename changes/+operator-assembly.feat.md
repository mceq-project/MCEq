The ETD2RK solver is layered like the matrix build: `MCEq.operator_assembly.compile_operator`
turns `int_m` / `dec_m` (and the sec(theta) operator set) into one `CompiledOperator` —
the diagonal / off-diagonal split in the kernel's state layout plus the coupling
operators — and a backend (`numpy_backend`, `mkl_backend`, `cuda_backend`) binds it to
scipy CSR, MKL sparse handles or the device. One step loop, `etd2_driver` (numbered
stage list in its docstring), now runs every route — paraxial and secant, single axis,
shared-path multi-RHS and the LPT carousel — on all three backends; the twelve
hand-unrolled paraxial kernels and the two CUDA context classes are gone (the names
remain as thin entry points). `MCEqRun` keeps one operator cache and one backend cache
(`close()` releases everything, including the multi-RHS MKL handles it used to miss);
the batched CUDA routes now honour `cuda_gpu_id` like the single-axis solve. The MKL
routes use row-major SpMM on C-ordered `(dim, K)` buffers, which measured 1.3-2x faster
than the former column-major tiled SpMM; block (BSR) storage is dropped from the driver
(`config.mkl_bsr_blocksize` removed; `numpy_bsr_blocksize` only serves the rho-stack
kernels). The host predictor / corrector run as fused row-major C kernels
(`etd2_post_apply{1,2}_rowmajor`, 2.5x the numpy ufunc chains at 2D K=8) with a ufunc
fallback. Not yet on the driver: the Accelerate kernels and the MKL fp32 multi-RHS kernel.
