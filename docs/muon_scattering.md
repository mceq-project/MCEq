# Muon angular scattering in 2D transport

The default remains `config.physics.muon_scattering_model = "gaussian"`.
To select the screened Coulomb model, use a 2D database containing the material
scattering tables, then set before constructing the run:

```python
from MCEq import config
from MCEq.config.run import RunConfig

config.paths.mceq_db_fname = "mceq_2d_muon_scattering_v1.h5"
config.physics.muon_multiple_scattering = True
config.physics.muon_scattering_model = "screened-coulomb-plane-v1"
settings = RunConfig.snapshot()
# Pass config=settings to MCEqRun(...).
```

`muon_multiple_scattering=False` disables either model. One-dimensional runs do
not use angular scattering. Selecting a table model in 2D requires the chosen
interaction material to be present; there is no implicit air or Gaussian fallback.
The legacy Gaussian is an air approximation, not a general material model.

Tables live at `/material_scattering/<material>/muon/<model>/`. `rate` has axes
`(energy,kappa)`, units cm²/g, positive sign, kinetic energy in GeV, and kappa in
inverse radians. The backend checks the schema, units, complete energy and mode
grids, positivity, finiteness and exact zero-mode conservation, then applies the
run's energy cut. The matrix builder adds `-rate` to physical muon diagonals for
both charges and all helicities. Existing secant transport acts on this operator
inside the solver. The uncoupled diagonal is exponentiated by ETD2; this does not
make the full secant-coupled solve exact.

Physics is evaluated offline by `mceq_em.muon_scattering`, packaged by
`python -m database_generator.muon_scattering INPUT OUTPUT` in maintenance tools,
and included unchanged by the 2D release bundler. MCEq has no runtime dependency
on mceq-EM. The table carries elemental composition and generator provenance.

The screened angular-plane model has generator
`Gamma = sum_i R_i [1 - kappa*a_i*K1(kappa*a_i)]`. It includes the Rutherford tail
without a Monte Carlo scattering-count cap. It uses point nuclei, Thomas–Fermi
screening with the Moliere correction, and an approximate atomic-electron Z term
with the same screening as the nuclear Z² term. It neglects recoil, nuclear form
factors, charge-sign and helicity effects. Its unbounded ideal tail has a divergent
second moment: comparisons must use finite angular acceptance or quantiles, not
an untruncated RMS. This is not an exact reproduction of CORSIKA's finite-step
Moliere sampler and is not a precision large-angle transport model.
