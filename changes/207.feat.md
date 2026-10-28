1D solves below 5 GeV reproduce the angle-integrated 2D spectra of sub-GeV
hadrons and muons: with the sec(theta) transport on (default) and
``config.e_min`` < 5 GeV, every operator column is scaled by the species' mean
sec(theta) at the current depth and zenith, read from the data release's table
of 2D solutions of the low-energy model (``config.secant_1d_table = "auto"``,
:mod:`MCEq.operators.secant_table`). The table has an inclusive (primary flux)
and a shower (single primary) group, selected by the initial condition; a
table of another low-energy model is refused, a handover below 50 GeV warns.
The ETD2 step applies the scaling (``col_scale``) on every backend at no
recompilation; per-direction batches and full-sky solves scale each carousel
lane with the ``s`` of its own direction.
