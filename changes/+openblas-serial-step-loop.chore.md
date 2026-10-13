On the MKL backend the ETD2 step loop runs OpenBLAS and BLIS at one thread
(threadpoolctl), MKL's pool unchanged. The dense mode-space GEMMs that numpy
sends to OpenBLAS otherwise wait for MKL's spinning SpMM threads on the same
cores. 2D Legendre solve, p 1 TeV, 30 deg, 90 modes, 1139 steps: 299-354 ->
72 ms per step with 8 threads on 8 cores, 279-322 -> 49-52 ms per step with
the default thread setting on 48 cores.
