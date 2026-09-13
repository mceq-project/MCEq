# Optional hadronic elastic scattering

A database may carry `elastic_scattering/<medium>/<model>` independently of its
inelastic interaction tables. Existing databases without this group retain their
previous behavior. Elastic gain and removal are added to the collision operator;
`prod_cross_section()` continues to return the inelastic/production table value.

The version-1 group attributes are `schema_version=1`, `process="elastic"`,
`cross_section_units="mb"`, `energy_kind="kinetic"`, and
`normalization="differential-per-collision"`. Datasets:

- `projectiles`: integer `(n_parent, 2)` PDG/helicity pairs.
- `cross_sections`: `(n_energy, n_parent)` nonnegative cross sections.
- `coverage`: Boolean array of the same shape. False means unavailable, not zero.
- `yields` and `yields_indptrs`: the existing packed channel format, with
  `tuple_idcs` and `len_data` attributes on `yields`. The shared database energy
  and Hankel grids determine the layout.

Kernels include the parent/child energy-bin width ratio exactly once. A 1D
package stores the zero-mode kernel; a 2D package stores all common modes.
For each parent, the builder adds `P_el * r_el` to the child blocks and subtracts
`r_el` once on the parent's diagonal, with `r_el = sigma_el / (A_target * m_u)`.
The physical inelastic rates and yields are not rescaled. Recoil and loss below
the stored energy range must not be renormalized away.

HE/LE weights multiply the entire elastic operator of each model. A missing
elastic group contributes no modeled elastic process; there is no implicit
material or projectile cross-section fallback. A present group with missing
coverage at nonzero model weight raises an error. In particular, a smooth
transition's small but nonzero tails still require explicit coverage.

Table creators must retain all generated particles needed for compaction. FLUKA
returns flavor-neutral kaons with PDG ±311; retaining only PDG 130/310 silently
loses those particles. Projection and any explicit K_S/K_L parent approximation
belong to database creation and must be recorded in provenance.

In particular, elastic KL→KS and KS→KL channels are off-diagonal production
blocks. They remain active at zero Hankel mode even if energy recoil is
negligible. With equal KL/KS rates and a 50:50 outgoing projection, their
no-recoil operator is `r_el/2 * [[-1, 1], [1, -1]]`: total kaon number is
conserved while the mixture changes. This classical projection convention does
not describe coherent quantum regeneration in matter.

An explicit interaction parent list restricts elastic parents too, retaining
their outgoing daughters. Reusing a fixed particle list raises an error if a
new elastic daughter would require resizing the state.

The additional elastic operator contributes a conservative matrix-norm bound to
the integration step ceiling, including secant amplification. This bound is a
step-control policy, not a substitute for convergence and physical benchmarks.

### Recovered database collision blending

Recovered databases may opt into collision-operator blending using the boolean
`common` attribute `collision_rate_blending`. Conditional inelastic yields are
then `(w sigma_HE Y_HE + (1-w) sigma_LE Y_LE) / sigma_blended`, including channels
unique to either model. The existing table format remains readable.

An optional boolean energy vector `model_coverage/<medium>/<HE model>` sets the
HE weight to zero at uncovered bins. Inelastic gains, inelastic losses and the
separate elastic operator use that same weight. The recovered 1D build records
coverage from the presence of positive original HE inelastic cross sections;
it does not interpolate missing generator bins. This prevents dilution by empty
HE tables in the transition. Coverage-aware fallback can change slope at its
boundary; actual transition cross sections must still be inspected.
