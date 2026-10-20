# CORSIKA 7 reference: proton 1 TeV, vertical, pure FLUKA, EM off, unthinned

Reference histograms for the MCEq-vs-CORSIKA 2D comparison in the MCEq v2 paper
(Figs. `fig:corsika_2d_spectra` and `fig:corsika_2d_angular`), used by
`examples/showers_2d/corsika_p1tev_vertical.py`.

## Physics case

A single 1 TeV proton enters the atmosphere vertically. Every hadronic
interaction, including the first, is simulated by FLUKA. The electromagnetic
cascade is switched off and there is no thinning. Hadrons, muons and neutrinos
are recorded at three observation levels:

| Level | Altitude | Vertical depth (US Standard, Linsley) |
|---|---|---|
| 1 | 15 km | 124.14 g/cm² |
| 2 | 5 km | 552.95 g/cm² |
| 3 | ground (0 m) | 1036.10 g/cm² |

The angular spread of sub-GeV to few-GeV secondaries comes from the transverse
momentum in hadronic interactions and decays and from muon multiple scattering.
A one-dimensional (straight-ahead) cascade equation cannot describe it, which is
why this case tests MCEq's 2D (energy × angle) solver.

## CORSIKA run

- **Code**: CORSIKA 7.8050, patched with a β-exact `PRANGE` fix and
  isospin-corrected K*± decays (the K* fix is inert here because SIBYLL never
  runs).
- **Binary**: `corsika78050Linux_SIBYLL_fluka_thin_fix2k`, configured with
  `CORHEMODEL=SIBYLL CORLEMODEL=FLUKA --enable-CHARM CORDETECTOR=HORIZONTAL
  CORTIMELIB=TIMEAUTO --enable-THIN --enable-NEUTRINO --enable-KEEPSOURCE`
  (flat atmosphere, no `CURVED`; `NEUTRINO` writes ν_e/ν_μ, IDs 66–69, to the
  particle file; `THIN` only because this build requires the `THIN` keyword).
- **Models**: FLUKA 2025.1 for all hadronic interactions (`HILOW 1500` puts the
  high/low-energy transition above the primary energy); SIBYLL 2.3e is linked but
  never called.
- **Steering card**: `steering.inp` is shard 0, verbatim. The shards differ only
  in `RUNNR = 610000 + I` and the four seeds `SEED (I + 1 + 50000)·1000 + {1,2,3,4}`
  for shard `I = 0 … 2499`. `DIRECT` (the output directory) is set to `./` here; otherwise the card is verbatim.
- **Statistics**: 2500 shards × 2000 showers = 5,000,000 showers.

Card keywords that matter for the comparison:

| Keyword | Value | Meaning |
|---|---|---|
| `PRMPAR` / `ERANGE` | 14 / 1000 1000 | proton, total energy 1 TeV (monoenergetic) |
| `THETAP` | 0 0 | vertical |
| `FIXCHI` | 0 | primary starts at the top of the atmosphere (first interaction point still sampled) |
| `ATMOD` | 1 | US Standard atmosphere, Linsley parametrisation |
| `MAGNET` | 1e-5 1e-5 µT | geomagnetic field effectively off (CORSIKA rejects exactly zero) |
| `HILOW` | 1500 | all hadronic interactions by FLUKA |
| `ECUTS` | 0.1 0.1 0.3 0.3 | kinetic-energy cuts [GeV]: hadrons, muons, e±, γ |
| `ELMFLG` | F F | EGS4 and NKG off: no EM cascade, no photo-hadron production |
| `MUMULT` | T | Molière muon multiple scattering |
| `MUADDI` | F | no extra muon information |
| `THIN` / `THINH` | 1e-20 1e30 0 / 1 1 | no energy or radial thinning; all weights are 1 |
| `PAROUT` | T F | write the particle file (`DATnnnnnn`) |

## What was histogrammed

Per shower, the particle file is read (thinning-aware reader, all weights 1),
and for each particle at each level:

- **kinetic energy** `E_kin = sqrt(p² + m²) − m` (0 for neutrinos' mass);
- **angle θ** between the particle momentum and the primary direction taken from
  the event header (vertical here, so θ is the zenith angle).

Two kinds of histograms are stored in `c7_reference.npz`, all as particles per
bin per primary shower (not divided by bin width), with Poisson errors
`sqrt(counts) / n_showers`:

- **Angular**, `ang_<sp>` (level, energy bin, θ bin): 2° bins from 0 to 90° plus a
  θ ≥ 90° column, in four kinetic-energy bins 0.1–0.1259, 0.3162–0.3981,
  0.7943–1.0 and 1.995–2.512 GeV (bins of the MCEq energy grid).
- **Energy spectra**, `spec_<sp>` (level, energy bin): the MCEq grid edges
  `logspace(-1.1, 4, 52)` GeV. Hadrons (p, n, π⁺+π⁻) include all angles; μ±, ν_μ+ν̄_μ
  and ν_e+ν̄_e are summed over θ < 90° (the upward-going fraction is negligible).

Species keys: `p`, `n`, `pi` (π⁺+π⁻), `mu` (μ⁺+μ⁻), `numu`, `nue` (particle +
antiparticle). The `meta` entry is a JSON string with the same description.

**Provenance**: one aggregation pass over the 2500 particle files produced four
histogram files (`c7_lowe_angles.npz`, `c7_nu_angles.npz`,
`c7_hadron_spectra.npz`, `c7_2d_espectra_nob_x100.npz`, read with
[`corsikaio`](https://github.com/cta-observatory/pycorsikaio)).
`make_reference.py <dir>` reduces those to `c7_reference.npz` (34 KiB) without
touching the raw files.

## Matching MCEq settings

Everything not listed is an MCEq v2 default.

| MCEq setting | Value | Why |
|---|---|---|
| `config.mceq_db_fname` | `mceq_base_air_2d_legendre_v2.h5` | 2D (Legendre-mode) database; needed for angular distributions |
| `interaction_model` | `"FLUKA20251"` | FLUKA 2025.1, as `HILOW 1500` |
| `low_energy_model` | `None` | no separate low-energy model: one model at all energies, as `HILOW 1500` |
| `primary_model` + `set_single_primary_particle(1e3, pdg_id=2212)` | `None`, 1 TeV p | single monoenergetic primary instead of a flux |
| `theta_deg` | 0 | `THETAP 0` |
| `density_model` | `("CORSIKA", ("USStd", None))` | `ATMOD 1` (the default is the Bianca–Keilhauer `BK_USStd`) |
| `config.muon_scattering_finite_nuclei` | `False` | CORSIKA's Molière scattering treats nuclei as point charges; MCEq's default finite-nucleus form factor gives fewer large-angle muons (≈ 3 % tilt at ground, 0.3–2 GeV) |
| `solve(int_grid=…)` | 124.14, 552.95, 1036.10 g/cm² | the three `OBSLEV`s, from `density_model.h2X` |
| `config.enable_em` | `False` (default) | `ELMFLG F F` |
| `config.e_min` | 0.1 GeV (default) | matches `ECUTS` 0.1 for hadrons and muons; the grid starts at the 0.0794–0.1 GeV bin |
| `config.muon_multiple_scattering` | `True` (default) | `MUMULT T` |
| `config.r_E`, `X_start` | defaults | flat atmosphere; the primary starts at X = 0 (`FIXCHI 0`) |
| — | — | MCEq has no geomagnetic field, matching `MAGNET` ≈ 0 |

MCEq angular readout: the Legendre modes at each depth are converted with
`convert_to_theta_space(..., theta_res=1200)`, the density is multiplied by θ and
integrated over each 2° bin, then multiplied by the energy-bin width, giving
particles per bin per primary like the CORSIKA histograms. Muons are summed over
the three helicity states.
