The secant ETD2 route works on a low-E-first state layout: `secant_layout`
permutes each Hankel-mode block so the coupled (mode, low-E) plane is a
strided corner view of the state, and `compile_operator` applies that
permutation to the diagonal/off-diagonal split once (cached on `MCEqRun`).
Every gather/scatter of the step disappears on all backends, and the coupling
operand is formed in place (no full-state copy). One driver (`etd2_driver`,
numbered stage list in its docstring) now serves numpy, MKL and CUDA through a
backend object; the separate CUDA driver, the compact-coupling template
(`config.secant_compact_coupling`) and the MKL dense-GEMM option
(`config.secant_mkl_gemm`, no longer measurable once the gathers are gone)
are removed.
