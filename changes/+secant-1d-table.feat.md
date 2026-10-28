Added ``config.secant_1d_table``: 1D solves scale every operator column by the
species' mean sec(theta) at the current depth and zenith, read from a table of
2D solutions (:mod:`MCEq.operators.secant_table`, generated with
:func:`~MCEq.operators.secant_table.generate_secant_table`). This reproduces the
angle-integrated 2D spectra of sub-GeV hadrons and muons without the 2D solver;
neutrino fluxes are unaffected. Batch and full-sky solves refuse a configured table.
