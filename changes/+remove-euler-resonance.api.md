Removed the forward-Euler integrators (``solv_numpy``,
``solv_MKL_sparse``, ``solv_CUDA_sparse``, ``solv_spacc_sparse``,
``CUDASparseContext``). ETD2RK is now the sole integrator, reached
through ``MCEq.solvers.solve_etd2`` on four backends: pure numpy
(always available), Apple Accelerate (macOS), Intel MKL sparse BLAS
(Linux/Windows when ``libmkl_rt`` is found) and NVIDIA cuSPARSE via
cupy. None of them uses block (BSR) off-diagonal storage.
Removed config options ``integrator``, ``ode_params``,
``leading_process``, ``stability_margin``, ``dXmax``,
``hybrid_crossover``, ``adv_set["no_mixing"]``,
``adv_set["exclude_from_mixing"]``. The energy-dependent resonance
approximation is gone; every species is now propagated explicitly.
``adv_set["force_resonance"]`` is retained as an explicit per-PDG
opt-in. ``config.kernel_config = "auto"`` resolves to
``accelerate_etd2`` on macOS, ``mkl_etd2`` on Linux/Windows when MKL
is present, otherwise ``numpy_etd2``.
