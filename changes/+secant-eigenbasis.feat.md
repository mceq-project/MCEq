The sec(theta) ETD2RK step carries the coupled corner of the state in the
eigenbasis of the mode-coupling block ``S_P = V diag(lam) Vi``. The exactly
integrated coupled operator is diagonal there, so the corner runs through the
same factor, predictor and corrector stages as the rest of the state, and one
step needs four dense mode products instead of ten. Solutions agree with the
previous scheme to fp64 roundoff; at fp32 the corner is more accurate because
the state is rotated once per solve rather than three times per step. The
``coupling`` namespace of the compiled operator now carries ``V, Vi, lam, W``.
