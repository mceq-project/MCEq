The CUDA backend computes `int_off x + ri dec_off x` in one CSR SpMM kernel
that reads both matrices once for any number K of state columns. A batched
fp64 solve no longer costs K single solves (2D FLUKA operator on an RTX 3090,
K = 8: 10.5 to 2.3 ms per step).
