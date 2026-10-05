The CUDA backend computes `int_off x + ri dec_off x` in one CSR SpMM kernel
that reads both matrices once for any number K of state columns. A batched
fp64 solve no longer costs K single solves (2D FLUKA operator on an RTX 3090,
K = 8: 10.5 to 2.3 ms per step). The 2D operators repeat one sparsity pattern
in every Hankel mode; the kernel keeps it once (the column indices of the
production operator shrink from 756 MB to 16 MB), which halves the
single-precision off-diagonal stage at small K.
