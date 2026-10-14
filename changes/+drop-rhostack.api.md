**Removed** the EM ρ-stack solver path: ``MCEqRun.enable_em_density_interpolation``,
``MCEqRun.disable_em_density_interpolation``, the ``solv_numpy_etd2_rho_stack``
and ``solv_numpy_etd2_rho_stack_multirhs`` kernels, ``HDF5Backend.em_rho_grid``
and ``config.numpy_bsr_blocksize``. It interpolated the EM interaction matrices
in air density for LPM suppression, was never validated, and no database carries
the ``rho_grid`` it needs, so it was exercised by code and never by data. The
database-side slice selection ``config.em_air_density`` is removed with it.
