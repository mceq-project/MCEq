The ETD2RK step formulas have one source, ``MCEq.solvers.numerics``: the phi
functions are written once and the two state stages once, as
``PREDICTOR_EXPR`` / ``CORRECTOR_EXPR``. ``etd2_kernels.c`` carries those two
expressions as its macros and the cupy ElementwiseKernel bodies are generated
from them, so the association order ``h`` -> phi -> remainder is the same on
numpy, MKL, Accelerate and CUDA, at fp64 and fp32, at every problem size.

``h`` is folded into the phi factors at the factor stage, so the predictor and
corrector are elementwise products of three arrays and take no step size. The
65536-element gate that chose between a fused kernel and a numpy ufunc chain is
gone with it -- it made one backend change its floating-point association order
with problem size. Host fp32 now runs the same compiled stages as fp64.

``etd2_kernels`` exports ``etd2_predictor_f64`` / ``etd2_corrector_f64`` and
their ``_f32`` siblings, generated from one macro, in place of the ten
``etd2_post_apply*`` entry points; the eight column-major ones had no caller.

1D host results move by up to 5.6e-14 relative L2 and 2D by 6.6e-15.
