`config.set_mkl_threads` now sets every BLAS pool MCEq can reach — OpenBLAS as
well as MKL, through threadpoolctl — once per process, and the ETD2 driver no
longer caps OpenBLAS around the batched secant step loop. `secant_blas_threads`
is removed. The single-axis secant route was outside the old cap and gains
25 % (5.21 s -> 3.92 s on the 2D FLUKA rc7 fixture); the batched route is
unchanged (15.72 s -> 15.24 s at K=8), and an uncapped 48-thread pool remains
14x slower, which is what the setting exists to prevent.
