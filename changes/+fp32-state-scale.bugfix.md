Single-precision solves (`fp_precision=32`) scale the state by a power of two
during integration, so fluxes above ~10^7 GeV stay within the fp32 exponent
range. On the production 2D operator fp32 agrees with fp64 to 1e-5 (0°) to
3e-4 (90°) below 10^10 GeV; without the scaling it was off by order one above
10^8 GeV.
