Removed the forward-Euler integrators (``solv_numpy``,
``solv_MKL_sparse``, ``solv_CUDA_sparse``, ``solv_spacc_sparse``,
``CUDASparseContext``). ETD2RK is now the sole integrator. The
shipped kernels are ``solv_numpy_etd2`` (pure numpy, always
available), ``solv_spacc_etd2`` (Apple Accelerate, macOS),
``solv_mkl_etd2`` (Intel MKL sparse BLAS, Linux/Windows when
``libmkl_rt`` is found, BSR(6) by default — see
``config.mkl_bsr_blocksize``) and ``solv_cuda_etd2`` (NVIDIA
cuSPARSE via cupy). The numpy kernel also stores its off-diagonals
as BSR (default block size 11, see ``config.numpy_bsr_blocksize``) —
empirically ~2x faster than CSR for the scipy SpMV. Removed config options ``integrator``, ``ode_params``,
``leading_process``, ``stability_margin``, ``dXmax``,
``hybrid_crossover``, ``adv_set["no_mixing"]``,
``adv_set["exclude_from_mixing"]``. The energy-dependent resonance
approximation is gone; every species is now propagated explicitly.
``adv_set["force_resonance"]`` is retained as an explicit per-PDG
opt-in. ``config.kernel_config = "auto"`` resolves to
``accelerate_etd2`` on macOS, ``mkl_etd2`` on Linux/Windows when MKL
is present, otherwise ``numpy_etd2``.
