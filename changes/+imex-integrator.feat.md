New IMEX integrator (`MCEq.solvers.imex`) for stiff runs: ARS(2,2,2) steps with
continuous losses, EM transfers and decays implicit, solved by GMRES with a
level-scheduled sweep as preconditioner, on numpy or CuPy, in 1D and 2D, with
tangent columns. `config.integrator = "auto"` (new default) uses it for 2D EM
showers, where it replaces the 0.008 g/cm² ETD2 stiffness step (about 20× faster),
and keeps ETD2 for every other run.
