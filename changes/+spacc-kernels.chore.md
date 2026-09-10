Move the private Apple Accelerate extension into
`MCEq.solvers._kernels.spacc` and remove its unused column-major ETD2
post-apply kernels. The solver backend continues to load it lazily.
