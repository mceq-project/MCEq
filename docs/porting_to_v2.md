(porting-to-v2)=

# Porting from MCEq 1.x to 2

A typical v1.4 script — construct `MCEqRun`, set the zenith, `solve()`,
`get_solution()` — runs unchanged on MCEq 2 and gives slightly different
numbers. This page lists what has to change in code and what changes in the
results. The numerical method is described in {ref}`solver-math`.

## Imports and installation

- `import mceq_config` still works with a `DeprecationWarning`; use
  `import MCEq.config as config`.
- Python 3.10 or newer. GPU support comes from the `cuda` (CUDA 12) or `cuda13`
  extra, `pip install "MCEq[cuda]"`. MKL is used when `libmkl_rt` can be loaded.
- `gtracr` is a dependency (geomagnetic cutoffs).
- The module layout is that of v1.4: `MCEq.core.MCEqRun`,
  `MCEq.particlemanager`, `MCEq.data`, `MCEq.geometry.density_profiles`,
  `MCEq.ddm`, `MCEq.ddm_utils`, `MCEq.misc`. New modules: `MCEq.batch`,
  `MCEq.solvers`, `MCEq.operators`, `MCEq.initial_state`, `MCEq.bin_average`,
  `MCEq.geometry.msis21`, `MCEq.geometry.cutoff`.

## Databases and interaction models

The default database is `mceq_base_air_1d_v2.h5` (FLUKA20251 and SIBYLL23E);
the other models are in `mceq_extra_models_air_1d_v2.h5`. Both are downloaded
on first use, see {ref}`data-installation`. Every model of the v1.4 database
(`DPMJETIII193`, `EPOSLHC`, `EPOSLHCR`, `QGSJETII04`, `QGSJETIII`, `SIBYLL21`,
`SIBYLL23D`, `SIBYLL23E` and the `SIBYLL23ESTAR*` variants) is available under
the same name; FLUKA20251, PYTHIA8CASCADE and PYTHIA8ANGANTYR are new.
`mceq_db_lext_dpm193_v142.h5` can still be selected with
`config.mceq_db_fname`.

FLUKA20251 is blended in below 80 GeV by default
(`config.low_energy_extension`): the high-energy models are tabulated from
80 GeV projectile energy and produce no secondaries below it. Choose another
model with `MCEqRun(..., low_energy_model=...)`, or switch the blending off
with `low_energy_model=None`. In v1.4 no model was blended in by default.

## Solver configuration

The forward-Euler integrator and the resonance approximation are replaced by
the ETD2RK exponential integrator, which propagates every species explicitly.

| v1.4 | v2 |
|---|---|
| `integrator`, `ode_params` | removed; ETD2RK is the only integrator |
| `leading_process`, `stability_margin`, `dXmax` | `config.etd2_path` (`eps`, `dX_max`, `dX_min`, `fd_span`), or the same keywords of `solve()` |
| `hybrid_crossover`, `adv_set["no_mixing"]`, `adv_set["exclude_from_mixing"]` | removed; `adv_set["force_resonance"]` (absolute PDG IDs) treats chosen particles as resonances |
| `adv_set["disable_resonance_decay"]`, `adv_set["disable_leading_mesons"]` | removed |
| `kernel_config = "numpy"/"MKL"/"CUDA"/"accelerate"` | unchanged; the full names are `numpy_etd2`, `mkl_etd2`, `cuda_etd2`, `accelerate_etd2`. `"auto"` tries Accelerate, MKL, numpy; CUDA is only used when requested |
| `base_url`, `release_tag`, `file_checksum` | `config.mceq_db_manifest` |

`solve()` takes `eps`, `dX_max`, `dX_min`, `fd_span`, `X_start` and `dtype`
(`np.float32` for single precision). With `eps` and `dX_max` left at `None`
the step limits are chosen from the assembled system: 0.03 and 5 g/cm² for 1D
hadron and lepton transport, 0.01 and 2 g/cm² for 2D databases or with EM
particles. Check convergence by repeating a solve with smaller values.

## Defaults that change results

| setting | v1.4 | v2 |
|---|---|---|
| `mceq_db_fname` | `mceq_db_lext_dpm193_v142.h5` | `mceq_base_air_1d_v2.h5` |
| `average_loss_operator` | `True` | `False` |
| `loss_stencil_method` (new) | | `"expfit_low_upwind2"` |
| `generic_losses_all_charged` | `False` | `True`: all charged hadrons lose energy continuously |
| `r_E` (Earth radius) | 6391 km | 6371.315 km (slant depths up to 0.15 % shorter near the horizon) |
| `adv_set["disabled_particles"]` | `[11]` | `[]`; γ, e± and π⁰ are dropped unless `config.enable_em` |

`adv_set["disable_interactions_of_unstable"]` now keeps only nucleons and
antinucleons as projectiles; in v1.4 it removed them instead.

## Removed and changed names

| v1.4 | v2 |
|---|---|
| `MCEqParticle.is_secondary(p)`, `is_child(p)` | `p in proj.hadr_yields`, `p in proj.decay_dists` |
| `MCEqParticle.track_decays`, `track_interactions`, `init_custom_particle_data`, `set_hadronic_channels`, `set_decay_channels` | removed; the `ParticleManager` wires the channels |
| `MCEqParticle.hadridx`, `residx`, `is_mixed`, `mix_idx`, `E_mix`, `int_idx` | removed; `is_resonance` is the only switch |
| `MCEqParticle.inverse_decay_length(cut=...)` | `inverse_decay_length()` (whole energy range) |
| `MCEqParticle.dN_dxf` | removed |
| `MCEqParticle.inel_cross_section` | kept; also available as `prod_cross_section` |
| `MCEqParticle` equality | by identity instead of by name |
| `ParticleManager.keys`, `nspec`, `set_cross_sections_db`, `current_cross_sections` | removed; iterate `pman.all_particles` |
| `MCEqRun.cuda_context`, `finalize_pmodel` | removed |
| `MCEqRun.set_theta_deg` | deprecated; `set_zenith_azimuth(zenith_deg, azimuth_deg=None)` |
| `MCEqRun.grid_sol` | a `(n_grid, dim_states)` array instead of a list |
| `MCEq.charm_models` | removed |
| `MCEq.misc.theta_deg`, `is_charm_pdgid`, `getAZN_corsika`, `corsikaid2pdg`, `pdg2corsikaid` | removed |
| `MCEq.solvers.solv_numpy`, `solv_MKL_sparse`, `solv_CUDA_sparse`, `solv_spacc_sparse` | `MCEq.solvers.solve_etd2(..., backend=...)` |
| `MSIS00Atmosphere(location, season)` | unchanged; `season=None` and no `doy` gives June |
| `AIRSAtmosphere(location, season)` | needs `table_path`, the directory of the AIRS tables |

Z-factors are computed as published, which changes their values:

- `z_factor(..., definition="thunman")` weights the yield table with the
  nucleon flux and the cross-section ratio (Thunman, Ingelman and Gondolo 1996).
- `decay_z_factor(parent, child, gamma=None)` takes a power-law parent
  spectrum `E**gamma`, by default with the nucleon index of the primary model.
- `definition="primary_e"` is unchanged.

## New in v2

- `solve_batch` and `solve_fullsky` ({mod}`MCEq.batch`) solve many directions,
  atmospheres or initial states in one call and return an `MCEqBatchResult`.
- NRLMSIS 2.1 atmospheres (`"MSIS21"`, needs the `nrlmsis` package) with the
  same interface as NRLMSISE-00, and detector-centred atmospheres
  (`MSIS00_IC`, `MSIS00_KM3NeT`, `MSIS21_IC`, `MSIS21_KM3NeT`) with a zenith
  and azimuth. `MCEq.core.DENSITY_MODELS` lists the keys of
  `set_density_model`.
- The geomagnetic rigidity cutoff in `solve_fullsky` with
  `geomagnetic_cutoff=True`.
- 2D databases with Legendre angular modes; angular densities come from
  `convert_to_theta_space`.
- Electromagnetic cascades with `config.enable_em = True` and the EM database
  `config.em_db_fname`.
- Single precision on every backend (`dtype=np.float32`).
- `MCEqRun.close()` releases backend handles and device memory.
