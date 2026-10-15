`MCEq.batch.build_phi0_with_cutoff` and the geomagnetic cutoff of `solve_fullsky`
take the mass groups and their `(Z, A)` from the primary model, as
`set_primary_model` does. The iron group now enters with A = 54, the value crflux
uses, instead of 56, and groups outside a fixed five-group table, such as the
oxygen of GaisserStanevTilav or the extra groups of PolyGonato, are no longer
dropped. Nucleon-flux models such as `GlobalSplineFitBeta` used to give an
all-zero initial state; they now raise `ValueError`. The `mass_groups` argument
is gone.
