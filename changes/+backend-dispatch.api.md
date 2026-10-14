One backend dispatch: ``MCEq.solvers.make_backend(op, kernel, dtype, device_id,
operator_dtype)`` serves ``MCEqRun`` and ``solve_etd2``, and accepts every kernel
name and alias of ``config.detect``. The backend factories and ``solve_etd2`` take
the state ``dtype`` (``np.float64`` / ``np.float32``) instead of
``fp_precision=64/32``. ``CudaOperator`` is merged into ``CudaBackend(op,
device_id, dtype, operator_dtype)``; ``cuda_backend`` and
``MCEq.solvers.backends.base`` are removed, and ``MCEq.solvers`` no longer
re-exports the operator internals (``CompiledOperator``, ``compile_operator``,
``secant_layout``, ``split_diagonal``; import them from
``MCEq.operators.compiled``) or the apply-off bindings.
