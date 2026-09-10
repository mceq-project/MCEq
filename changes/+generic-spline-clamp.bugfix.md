The generic proton stopping-power curve is now evaluated with its boost
argument clamped to the tabulated interval, and, mirroring the lepton branch
of the backend, is taken from ``ionization/hadron`` instead of
``total/hadron`` when ``enable_em`` is on. The log-log k=1 spline extrapolated
its high end: at the top of a wide grid the boost sits four decades past the
table end and the curve overshoots 35x (0.0726 against 0.00205
GeV/(g/cm^2)). The overshoot is irrelevant to the stability fix (dEdX/E there
is 1e-13 per g/cm^2) but the value is simply wrong, so it is clamped. The
affected measurement pins move accordingly: the solve goldens, the FLUKA 2D
regression fixture, the ``test_core`` single-primary table (nnumu at 1e3 GeV,
+1.2%), and two solve2d metric probes in ``test_harness``.
