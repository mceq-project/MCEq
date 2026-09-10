Apple Accelerate is an ``apply_off`` binding of the shared ETD2 host backend,
like scipy and MKL. ``SpaccApplyOff`` and ``accelerate_backend`` join
``ScipyApplyOff`` / ``numpy_backend`` and ``MklApplyOff`` / ``mkl_backend``,
``solve_etd2`` takes ``backend="accelerate"``, and the four Accelerate step
loops — ``solv_spacc_etd2``, ``solv_spacc_etd2_multirhs``,
``solv_spacc_etd2_multirhs_f32`` and ``solv_spacc_etd2_carousel`` — are
removed. ``MCEqRun`` caches, closes and rebuilds the Accelerate backend like
every other one; ``MCEqRun._legacy_handles`` is gone.

``MCEq.spacc.SpaccMatrix`` takes ``dtype=`` and picks the
``sparse_matrix_*_double`` or ``sparse_matrix_*_float`` entry-point family from
it, so ``SpaccMatrixF32`` is removed.

Accelerate therefore gains every route the driver offers: the sec(θ) coupled
transport, the LPT carousel and ``int_grid`` snapshots on the shared step loop.
``_resolve_secant`` no longer lists it as a configuration without a secant
route — the coupled route is driver code plus ``apply_off``, and fp32 state
buffers outside ``cuda_etd2`` are the one remaining blocker.

The 64-column SpMM tiling is kept: Accelerate has no row-major SpMM, so the
binding stages the row-major ``(dim, K)`` state into Fortran-ordered scratch
allocated once per bind and issues one accumulating SpMM per tile. At K = 1 the
two layouts are the same bytes and the driver's buffers go straight to the
SpMV.
