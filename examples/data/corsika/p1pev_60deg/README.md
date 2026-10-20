# CORSIKA 7 reference: proton 1 PeV at 60°, hadron and lepton spectra at seven depths

Reference for the MCEq paper figures `fig:corsika_hadron_spectra` and
`fig:corsika_integral_ratio`, used by
`examples/showers_2d/corsika_p1pev_60deg.py`.

A 1 PeV proton enters a flat atmosphere at 60° zenith. CORSIKA 7 records every
hadron, muon and neutrino at seven slant depths along the axis
(100 – 2000 g/cm²); MCEq solves the same cascade in 2D (Legendre angular
modes) and its angle-integrated (l = 0) spectrum is compared bin by bin.
Electromagnetic showers and the geomagnetic field are switched off in CORSIKA so
that only the hadronic cascade and its decay products are compared.

## Files

| File | Content |
|---|---|
| `steering.inp` | CORSIKA steering card of shard 0, verbatim except `DIRECT` (the output directory, set to `./` here) |
| `c7_reference.npz` | reduced reference spectra (27 KB), layout in `make_reference.py` |
| `make_reference.py` | reduction from the aggregated histogram file |

## CORSIKA run

- **Version:** CORSIKA 7.8050, patched with a β-exact PRANGE fix and
  isospin-corrected K*± decays. Binary `corsika78050Linux_SIBYLL_fluka_thin`
  (local suffix `_fix2k`).
- **Build:** `CORHEMODEL=SIBYLL CORLEMODEL=FLUKA CORDETECTOR=HORIZONTAL
  CORTIMELIB=TIMEAUTO --enable-THIN --enable-NEUTRINO --enable-CHARM
  --enable-KEEPSOURCE --with-pythia` (Pythia handles charm decays). No
  `CURVED`/`UPWARD` option: flat atmosphere and horizontal observation planes.
- **Models:** SIBYLL 2.3e above 80 GeV lab energy, FLUKA 2025.1 below (the card
  has no `HILOW`; the default 80 GeV applies, printed in the run log).
- **Atmosphere:** no `ATMOD` card, i.e. model 1, the U.S. standard atmosphere
  in Linsley's parameterization (top at 112.8 km).
- **Primary:** proton (`PRMPAR 14`), 1 PeV, θ = 60°, φ uniform in 0–360°.
- **Physics switches:** `ELMFLG F F` (no EGS4, no NKG: no electromagnetic
  cascade and hence no photo-production of hadrons/muons);
  `MAGNET 0.00001 0.00001` µT (field effectively off; B = 0 divides by zero in
  CORSIKA); `ECUTS 0.3 0.3 0.0316 0.0316` GeV (hadrons, muons, e, γ);
  `MUADDI T`, `MUMULT T` (Molière muon multiple scattering);
  `STEPFC 1.0`; `FIXCHI 0.` (start at the top of the atmosphere).
- **Thinning:** `THIN 1.E-6 1.0 0.` with `THINH 1.0 1.E2`: thinning level
  10⁻⁶ of the primary energy for EM particles and hadrons alike, **radial
  thinning off** (`RMAX 0`). With the default radial thinning, the leading
  hadrons within 1 m of the axis carry heavy weights and the uninteracted
  primary is removed; both bias the high-energy hadron spectra of a finite sample.
- **Statistics:** 5000 showers = 2500 shards × `NSHOW 2`. Shard *I*:
  `RUNNR 100000+I`, `SEED (I+1)·1000+1/2/3` (three sequences).

### Observation levels

`OBSLEV` altitudes are chosen so that the slant depth along the 60° axis
(= vertical depth / cos 60°) hits round numbers (altitudes rounded to 10 m;
slant depth reproduced to < 0.5 g/cm²):

| Level | Altitude [km] | Vertical depth [g/cm²] | Slant depth [g/cm²] |
|---|---|---|---|
| 1 | 20.83 | 50.0 | 100 |
| 2 | 14.96 | 124.9 | 250 |
| 3 | 11.95 | 200.1 | 400 |
| 4 | 8.81 | 324.9 | 650 |
| 5 | 7.36 | 400.3 | 800 |
| 6 | 5.75 | 499.9 | 1000 |
| 7 | 0.30 | 999.8 | 2000 |

## What was histogrammed

- **Species:** n + n̄, p + p̄, π⁺ + π⁻, K⁺ + K⁻, μ⁺ + μ⁻, ν_μ + ν̄_μ,
  ν_e + ν̄_e.
- **Variable:** kinetic energy E_kin = √(p² + m²) − m, on the MCEq v2 energy
  grid (10 bins per decade, edges 10^(k/10 − 1.1) GeV).
- **Plane:** CORSIKA scores on horizontal planes, MCEq on planes perpendicular to
  the shower axis. Each particle on a horizontal level was re-scored onto the
  perpendicular plane through the core: kept if its axial grammage offset from
  that plane is below 30 g/cm² (a strip across the core, at ±90° from the
  vertical plane containing the axis) and weighted by the inverse azimuthal
  acceptance of the strip on its ring about the axis (assumes azimuthal symmetry,
  which is why the geomagnetic field is off). All particle directions are kept.
- **Normalization:** sum of thinning weights per bin, divided by the number of
  showers (particles per energy bin per primary).
- **Errors:** shower-to-shower standard error of the mean, per bin and for the
  E_kin > 1 GeV integral.
- **Reduction:** the aggregation step reads the 2500 particle files once and
  writes `c7_spectra_by_level_perpstrip.npz` (several strip widths and angle
  cuts); `make_reference.py <that file> --sel w30` writes `c7_reference.npz`
  (float32 values/errors, float64 edges, `meta` JSON with provenance).

## Matching MCEq settings

Everything not listed is the MCEq default.

| MCEq setting | Value | Why |
|---|---|---|
| `config.mceq_db_fname` | `mceq_base_air_2d_legendre_v2.h5` | 2D (Legendre) cascade. Angle-integrated spectra still need 2D: the 1D straight-ahead approximation puts every particle on the axis and over-predicts sub-GeV hadrons at depth |
| `interaction_model` | `SIBYLL23E` | SIBYLL 2.3e as in CORSIKA; its 2D tables are fetched as `mceq_model_SIBYLL23E_air_2d_legendre_v2.h5` (≈ 970 MB, on top of the 456 MB base file) |
| `low_energy_model`, `he_le_transition` | `FLUKA20251`, 80 GeV | CORSIKA `HILOW` default: FLUKA below 80 GeV |
| `he_le_trwidth` | 0.3 (default) | MCEq blends the two models over 0.3 decades; CORSIKA switches sharply. Not a matching knob |
| `density_model` | `("CORSIKA", ("USStd", None))` | `ATMOD 1` (Linsley US standard). The MCEq default `BK_USStd` is a different parameterization |
| `config.r_E` | 10¹⁵ m | flat Earth, as CORSIKA without `CURVED`; slant depth = vertical depth / cos θ |
| `theta_deg` | 60 | `THETAP 60 60` |
| primary | `set_single_primary_particle(1e6, pdg_id=2212)`, `primary_model=None` | `PRMPAR 14`, `ERANGE 1e6 1e6` |
| `config.muon_scattering_finite_nuclei` | `False` | CORSIKA's Molière scattering (`MUMULT T`) uses point nuclei |
| `config.enable_em` | `False` (default) | `ELMFLG F F` |
| `config.e_min` | 0.1 GeV (default) | MCEq has no transport cut; the comparison starts at the `ECUTS` 0.3 GeV for hadrons and muons |
| depths | `solve(int_grid=X, grid_var="X")`, X = `h2X(OBSLEV)` of the 60° flat atmosphere | the seven `OBSLEV` planes |
| read-out | l = 0 row of the Legendre state (`get_solution(..., integrate=True)`) | angle-integrated count per bin, same normalization as the reference |

The harness solved the paper version with the atmosphere shifted up by 12 km
(exponential continuation below sea level) so that it could also read the 2D
solution on *horizontal* planes, which needs axial depths beyond the ground for
directions pointing away from the vertical. The on-axis l = 0 spectra used here
do not need that: the unshifted atmosphere reproduces them to < 3·10⁻⁴ in the E > 1 GeV integrals
and < 3·10⁻³ in single bins (the largest deviations sit in bins with tiny content).
