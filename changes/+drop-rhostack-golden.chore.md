The EM rho-stack is left out of the golden harness. It exists to interpolate
the EM matrices in air density for LPM suppression, no database on disk carries
the `rho_grid` it needs, and it has never been validated, so it will be
reimplemented with the rest of the EM work rather than carried through the
refactor.
