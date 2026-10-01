# Muon angular scattering in 2D transport

In the 2D transport the angular distribution is carried as Hankel modes κ.
Coulomb scattering of muons does not couple modes or energies. It adds a
damping rate Γ(E, κ) to the muon diagonal of each mode:

    dΦ(κ)/dX = … − Γ(E, κ) Φ(κ)

The solver integrates the diagonal exactly, so over a depth X a mode is
multiplied by exp(−Γ X). Γ is the single-scattering rate in mode space, and
multiple scattering follows from the exponential; no separate
multiple-scattering distribution is needed. Γ(E, 0) = 0, so the
angle-integrated flux is unchanged. One-dimensional runs have no angular
scattering.

Two models of Γ are available through `config.muon_scattering_model`:

- `"screened-coulomb"` (default): Γ = Σ_i R_i(E) [1 − κ a_i K₁(κ a_i)], the
  Hankel transform of the screened Rutherford cross section, summed over the
  elements i of the medium. Thomas–Fermi screening angle a_i with the Molière
  correction; scattering on atomic electrons is included as Z(Z + 1). It
  includes the single-scattering tail and neglects nuclear form factors,
  recoil, and charge and helicity dependence. The angular distribution of
  this model has no finite second moment; compare it with other calculations
  within a finite angular acceptance or by quantiles, not by the RMS angle.
- `"gaussian"`: Γ = κ² θ_s²(E) / 4 with the Gauss approximation of CORSIKA,
  θ_s² = (E_s / (E β²))² / λ_s, E_s = 21 MeV, λ_s = 37.7 g/cm². Only the
  Gaussian core; no single-scattering tail.

`config.muon_multiple_scattering = False` switches scattering off for both.

MCEq computes the Coulomb rate when the operator is built
(`MCEq.operators.scattering.screened_coulomb_rate`). It needs the elemental
composition of the medium, read from the database at
`/material_scattering/<medium>/composition`: rows of (Z, A in g/mol, atom
fraction). `mceq_base_air_2d_v2.h5` contains dry air. A database without the
composition of the selected medium raises an error; set
`config.muon_scattering_model = "gaussian"` to use such a database.
