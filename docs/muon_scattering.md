# Muon angular scattering in 2D transport

In the 2D transport the angular distribution is carried as Legendre moments
⟨P_l(cos θ)⟩. Coulomb scattering of muons does not couple modes or energies.
It adds a damping rate Γ(E, l) to the muon diagonal of each mode l:

    dΦ(l)/dX = … − Γ(E, l) Φ(l)

The solver integrates the diagonal exactly, so over a depth X a mode is
multiplied by exp(−Γ X). Γ is the single-scattering rate in mode space, and
multiple scattering follows from the exponential (Goudsmit–Saunderson); no
separate multiple-scattering distribution is needed.
Γ(E, 0) = 0, so the angle-integrated flux is unchanged. One-dimensional runs
have no angular scattering.

Two models of Γ are available through `config.muon_scattering_model`:

- `"screened-coulomb"` (default): the screened Rutherford (Wentzel) cross
  section C / (1 − cos χ + 2s)², summed over the elements i of the medium.
  Thomas–Fermi screening angle a_i = 2√s_i with the Molière correction;
  scattering on atomic electrons is included as Z(Z + 1). The rate is the
  sum of the exact transport cross sections on the sphere,
  Γ_l = Σ_i n_i σ_l,i with σ_l = σ_tot (1 − ⟨P_l(cos χ)⟩), evaluated by
  cancellation-free recurrences
  (`MCEq.operators.scattering.transport_cross_sections`). Its small-angle
  (flat-sky) limit at κ = l + ½ is Σ_i R_i(E) [1 − κ a_i K₁(κ a_i)], 13 %
  high at l = 1 and within 1e-5 from l = 256. It
  includes the single-scattering tail and neglects nuclear form factors,
  recoil, and charge and helicity dependence. The angular distribution of
  this model has no finite second moment; compare it with other calculations
  within a finite angular acceptance or by quantiles, not by the RMS angle.
- `"gaussian"`: Γ = l(l + 1) θ_s²(E) / 4, the diffusion eigenvalue on the
  sphere, with the Gauss approximation
  of CORSIKA, θ_s² = (E_s / (E β²))² / λ_s, E_s = 21 MeV,
  λ_s = 37.7 g/cm². Only the Gaussian core; no single-scattering tail.

`config.muon_multiple_scattering = False` switches scattering off for both.

MCEq computes the Coulomb rate when the operator is built
(`MCEq.operators.scattering.mode_rates`). It needs the elemental
composition of the medium, read from the database at
`/material_scattering/<medium>/composition`: rows of (Z, A in g/mol, atom
fraction). The Legendre 2D databases contain dry air. A database without the
composition of the selected medium raises an error; set
`config.muon_scattering_model = "gaussian"` to use such a database.

With finite nuclei the rate costs about l_max^2: some 150 s for the modes up to
l = 2000 on 31 energy bins. The Legendre 2D release packages therefore also
carry it precomputed, at `/material_scattering/<medium>/screened_coulomb_rates`
(one row per mode, one column per energy bin of the file), written by
mceq-maintenance-tools with this same function. MCEq uses the table, cut to the
run's energy window, when the muon mass, `muon_scattering_finite_nuclei` and
the grids it records match the run; otherwise it computes the rate, once per
process.
