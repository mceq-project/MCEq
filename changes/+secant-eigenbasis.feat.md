The sec(theta) ETD2RK step evolves the mode-coupled part of the state in the
eigenbasis of the coupling block `S_P = V diag(lam) Vi`, where the exactly
integrated operator is diagonal. A step needs four dense mode products instead
of ten. Results are unchanged at fp64 and more accurate at fp32. The `coupling`
namespace of the compiled operator carries `V, Vi, lam, W`.
