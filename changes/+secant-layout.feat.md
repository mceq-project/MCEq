The secant ETD2 kernels work on a low-E-first state layout: `secant_layout`
permutes each Hankel-mode block so the coupled (mode, low-E) plane is a
strided corner view of the state, and `secant_split` permutes the
diagonal/off-diagonal split once (cached). Every gather/scatter of the step
disappears on all backends, and the coupling operand is formed in place
(no full-state copy). One driver (`_etd2_secant_driver`, numbered stage list
in its docstring) now serves numpy, MKL and CUDA through a backend object;
the separate CUDA driver, the compact-coupling template
(`config.secant_compact_coupling`) and the MKL dense-GEMM option
(`config.secant_mkl_gemm`, no longer measurable once the gathers are gone)
are removed. The MKL/CUDA secant kernels take operators built from
`secant_split` (`MCEqRun._secant_kernel_operators`).
