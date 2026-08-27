Optional MKL backend for the secant drivers' dense mode-coupling GEMMs:
`config.secant_mkl_gemm = True` routes them through `cblas_dgemm` on the
already loaded `mkl_rt` instead of numpy's linked BLAS (MKL secant routes
only; default off preserves bit-identity with the numpy path).
