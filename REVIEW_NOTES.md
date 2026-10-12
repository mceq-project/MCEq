# REVIEW_NOTES (temporary: delete before merging read_everything)

Handoff for resuming the cleanup in another session. The `#anfe:` comments in `src/` are the
maintainer's review annotations. Remove each one once the change it asks for is made.
The report below is the result of a full read of `src/MCEq` against those annotations.
Line numbers refer to commit "WIP: maintainer review annotations" and will drift after the
rebase onto main (#189 and the Legendre branch).

## Maintainer decisions (2026-10-13)

1. **Config groups / RunConfig (S1):** never shipped. Remove `config/groups.py`, `config/run.py`
   and the `_cfg` plumbing, and go back to plain module globals. Keep `config/detect.py`.
2. **PDG→family fallback disagreement (10313/10323, leptons).**
   - Two fallbacks exist:
     - `InteractionCrossSections.get_cs` (`data/cross_sections.py`) maps 10313/10323 (K1(1270)) to the K± cross section and gives leptons zeros.
     - `equivalences.family_fallback`, used only by model blending, returns "no equivalent" for both.
   - Neither case is reachable with current data:
     - No cross-section table in any shipped DB lists these IDs or a lepton (checked base, extra-models and CI files).
     - They never enter a particle list (checked SIBYLL2.3d, DPMJETIII193, EPOS-LHC).
   - The special case was already in v1.4.2 (`data.py:1042`).
   - **Decision:** remove the 10313/10323 special case and the test that pins the disagreement
     (`tests/test_data_model_names.py::test_the_two_pdg_policies_are_not_unified`). Re-add it only when something exposes it.
3. **particletools → scikit-hep `particle` (S6):** keep the current particle table for now.
   This is a follow-up; do not migrate in this branch.
4. **ULP-exact pin tests:** convert to tolerances. The Fortran-derived codes already moved off
   last-digit compatibility.
5. **geometry vs environment (S10):** the rename is not worth it.
   - Move the code back under `MCEq/geometry/` with the v1.4.2 module names:
     - `environment/parameters.py` → `geometry/atmosphere_parameters.py` (locations, CORSIKA parameters, converters);
     - `environment/msis00_backend.py` → `geometry/nrlmsise00_mceq.py`;
     - `density_profiles.py` holds the atmosphere classes again.
   - The NRLMSISE-00 C extension stays, but it does not have to live in the same module as the Python wrapper.
   - Drop the `environment/` package and every geometry shim.
6. **Order of work**, committed in separate steps with tests run between them:
   1. Dead code and prose trims (no numerical change).
   2. Bug fixes B1–B18, one commit each (some change numbers, e.g. B1).
   3. Structural:
      1. S1 config
      2. S2 lazy-import workarounds
      3. S7 data/
      4. S8 DDM → `MCEq/ddm.py`
      5. S4/S5 ownership + particles
      6. S3 driver
      7. S9 solvers/operators
      8. S10 geometry
      9. S12 test pruning

## Test baseline note

On Windows, `pytest -n auto` / `-n 8` loses two xdist workers near 92–100% ("node down: Not
properly terminated"), and the master then dies with `OSError: [Errno 22]` when it restarts them.
This happened before any refactor edit. A serial run (`-p no:xdist`) reached 89% with no failures (735 passed) and was stopped while running `tests/test_solvers.py::test_etd2_mkl_matches_numpy_real`. Suspect the MKL tests. The crashing tests have not been confirmed yet; run
`pytest -n 8 --max-worker-restart=0 -rfE` or a serial run to name them, then record a clean baseline
before the first edit.

---

# MCEq `read_everything`: prose and overengineering review

Basis: 59 `anfe:` comments, plus a full read of all 88 files in `src/MCEq` (about 19k lines) by 6 reviewers.
Call counts come from grep over `src/ tests/ docs/`. API claims were checked against tag `v1.4.2`.
Line numbers refer to the current working tree, which includes the `#anfe` lines.

---------------------------------------------------------------------------------------------------

## 0. The style these findings are measured against

**Prose**
- **P1:** A module docstring is one line saying what the module contains. Add more only if it is genuinely needed. Never list exports, and never restate the function docstrings.
- **P2:** No dev-report text: history, "as before", "unchanged", "moved", "used to", refactor IDs (C1/C4/C5, Cure-B, ledger, census, ratchet, phase N), layering justifications, or explanations of what is *not* done.
- **P3:** Never cite test files or test pins as the reason for code.
- **P4:** A docstring should not be longer than the code. A private or 5-line function gets one line, or a few inline comments.
- **P5:** Say what is computed and what it is for. No numeric side stories or benchmark figures.
- **P6:** No AI-slop phrasing ("the one X", "honoring", "rather than", "drop-in", em-dash chains, jokes).

**Design**
- **O1:** No compatibility shims or re-exports for names that were not public in v1.4.2. Do not expose v2 internals through old paths.
- **O2:** Inline helpers used only once. Delete dead code and placeholders.
- **O3:** No abstraction without payoff: no Protocols, resolver injection, lazy-import tricks, or wrappers around a dict read.
- **O4:** No module or package that holds a single function or constant. Merge modules that belong together.
- **O5:** Copy-paste near-duplicates become one private function plus thin wrappers.
- **O6:** Ownership has to be right. Build things correctly from the start instead of filtering or patching afterwards. No dangerous dunders.
- **O7:** Remove retired features completely (ρ-slices, the resonance approximation).

---------------------------------------------------------------------------------------------------

## 1. Corrections to some `anfe` comments

| anfe comment | What was found |
|---|---|
| `core.py:13`: MatrixBuilder was never public before v2 | Not quite. `class MatrixBuilder` was in v1.4.2 `core.py:1371`. Only `MCEqBatchResult` is new on that path. |
| `misc.py:40`: these symbols were never in misc | `energy_grid`, `_xmat`/`gen_xmat`, `_eval_energy_cuts` and `normalize_hadronic_model_name` **were** in v1.4.2 `misc`. Moving them back into misc removes `data/energy_grid.py` and the whole `_MiscCompatModule` shim with no break. |
| `species/constants.py`: backward-compat stuff not needed | The neutrino name map (`numu`, `antinumu`, …) defines the **current public species names** used by `get_solution("total_numu")`, the examples and the tracking aliases. Keep it as MCEq's name table, without the "previous versions" wording. |
| `particlemanager.py`: probably only the BSM example used it | No BSM example exists, in the tree or in git history. No notebook or `.rst` imports the module. It can go, or (preferred) `species/` can collapse back into one flat `particlemanager.py`, as in v1. |
| `models/__init__.py`: can ddm live directly in MCEq? | Yes, and that restores the API. `from MCEq.ddm import DataDrivenModel` (the v1.4.2 path) currently raises `ModuleNotFoundError` (verified). |
| `em_tables.py:93` `helicity_duplicate`: a hadronic equivalent probably exists | No helicity duplicator exists elsewhere. The function is **unreachable** through MCEqRun anyway, because `driver/system.py:run_physics_view` forces helicity off whenever EM is on. Delete it. |

---------------------------------------------------------------------------------------------------

## 2. Bugs found along the way (separate from style; some change numbers)

| # | Location | Bug | Inherited from v1? |
|---|---|---|---|
| B1 | `driver/observables.py:95` | `if proj == 2112` is always False, because `__eq__` returns NotImplemented for int. Neutron Z-factors use the **proton flux** (verified). | yes |
| B2 | `driver/observables.py:80,117` | `z_factor()` with default args (`primary_e`, `use_cs_scaling=True`) hits its own assert. | yes |
| B3 | `species/manager.py:549-555` | `"numu" in pman` raises KeyError while `pman["numu"]` works. `__contains__` and `__getitem__` are inconsistent. | ? |
| B4 | `species/particle.py:575` | `__neq__` is not a dunder (Python uses `__ne__`), so it is dead. | yes |
| B5 | `species/manager.py:474-481` | Every resonance has `mceqidx=-1`, so `mceqidx2*[-1]` ends up holding the last resonance. | ? |
| B6 | `species/manager.py:349` | `not (p.is_em == exclude_em)` keeps only EM particles when `exclude_em=False`. Latent: all callers pass True. | ? |
| B7 | `species/manager.py:499` | The `info(...)` format string is never filled. | |
| B8 | `driver/mceq_run.py:1045` | `getattr(config, "em_step_dense_eig_max", 4000)` reads global config, not `self._cfg`. | new |
| B9 | `driver/system.py:208` | The "Should not happen in practice" branch **is** reachable. `particle_list` (the user parents) is compared with `_particle_list` (parents+children). Same mismatch at `mceq_run.py:366`. | v1 had a TODO there |
| B10 | `environment/msis00_backend.py:81-85` | A joke error for `"SanGrasso"` ("wherever that is"). That **was a valid v1.4.2 location key** (`atmosphere_parameters.py:244`), so old code breaks and gets mocked (verified). | new |
| B11 | `environment/geomagnetic/cutoff.py:364-432` | `build_phi0_with_cutoff` drops the neutron flux when neutrons are absent; `initial_state` adds it to protons. | new |
| B12 | `models/ddm/ddm.py:15-18` | The default DDM file no longer follows `config.data_dir` (v1.4.2 did). | new |
| B13 | `models/ddm/ddm.py:684-742` | The documented `e_min`/`e_max` are ignored by `ddm_matrices`, and the documented defaults (0/inf) do not match the code (-1). | yes |
| B14 | `models/ddm/ddm_utils.py:142,148` | A missing `f` prefix on a log line, and a level-0 log. | yes |
| B15 | `environment/msis00_backend.py:173-227` | `test()` is broken on py3 (`range(360/30)`). | yes |
| B16 | `environment/parameters.py:295-315` | The docstring is a parenthesised string, so `__doc__` is None. | yes |
| B17 | `driver/batch.py:312` | The `return_pixel_index` parameter is accepted but ignored, and its documented return value never happens. | new |
| B18 | `data/hdf5_backend.py:176` | `equivalences={}` is a mutable default that shadows the module-level table. | |
| B19 | `mkl_runtime.py:49` | The Windows linter complaint (your anfe note). It goes away with the resolver rewrite (S9). | |

---------------------------------------------------------------------------------------------------

## 3. Structural proposals (the big decisions)

Risk legend: **API**: a public-name change compared with v1.4.2. **NUM**: results change. **TEST**: test churn only.

### S1. Config: back to plain module globals (−~340 lines, −38 `_cfg` refs)
- `config/groups.py` (186 lines): `GroupView` reads straight through to the same globals, so it gives **no isolation**.
- `config/run.py` (150 lines): `RunConfig` snapshots have **0 non-test users** and were not in v1.4.2.
- `with_overrides` has 1 caller (the EM-forces-helicity-off rule), which a local bool can replace. `GROUP_OF` has 0 uses.
- Keep `detect.py`: the lazy CUDA/MKL probes are justified. Drop `reset_cache`, `PROBES`/`available` and the dead `KERNELS` auto flag.
- Turn `secant_mode` (tri-state, duplicated in run.py) into `secant_enabled(is_2d) -> bool`.
- Risk: removes `MCEqRun(config=...)` and `config.grid.e_min`-style access. Both are v2-only and unreleased. TEST: `test_group_injection.py`, `test_conftest_isolation.py`.
- **This is the root of much driver prose.** Half of the "injected, no fallback to live config" sentences exist because of this layer.

### S2. Remove the layering workarounds (`import_module("MCEq.config")` ×7, resolver injection)
- Copies: `misc._views`, `solvers/path._live_config`, `environment/base.py:18-30`, `environment/geometry.py:108`, `environment/target.py:46`, `geomagnetic/cutoff.py:96,386`.
- There is no import cycle. These exist only to satisfy a self-imposed import-linter contract. Use `from MCEq import config`, or relax the contract.
- This also removes `misc._views`, which answers your "why do we need this mechanism at all".

### S3. Driver: collapse the split back into `MCEqRun`
- Today `mceq_run.py` (1086 lines) plus 6 modules (1552 lines) are glued together by 10 module-function-as-method bindings (`solve_batch = batch.solve_batch`) and **13 forwarding properties** (`_mceq_db`, `_interactions`, `_decays`, `pman`, `int_m`, …). v1.4.2 did the same job in about 1350 lines.
- `observables`, `results.get_solution` and `paths` go back to real methods. `results.py` keeps only `MCEqBatchResult`. Fold `angular.py` (35 lines, 1 caller) into `_compiled_operator`.
- **Initial state:** drop the name-based replay (`InitialState.resize(restore_list, replay)` plus a lambda back into MCEqRun plus 5 delegators). Store the initial condition as data and build phi0 against the current pman at solve time. Share the nucleon-superposition fill with `build_phi0_with_cutoff`, which also fixes B11.
- Risk: none for NUM or API. TEST churn where tests monkeypatch module functions.

### S4. Ownership: `reload_for_model`, ParticleManager and the EM species (your system.py and strip_em_species notes)
1. `Decays` and `ContinuousLosses` load once, because they don't depend on the model. Only cross sections and interactions reload.
2. `ParticleManager(pdg_list, tables, energy_grid, physics)` builds the species list **without EM species unless `enable_em`**. That deletes `misc.strip_em_species` and `EM_PDG_IDS`, and the duplicate `is_em` definition in particle.py goes with it.
3. The pman **pulls** its channels: `pman.load(cs, int, dec, loss)` wires `hadr_yields`/`decay_dists` itself. That deletes `set_cs`, `set_hadronic_channels`, `set_decay_channels`, `_set_channels`, the particle→pman back-reference, and the three copy-pasted setter branches.
4. A changed particle list means a new pman, not an in-place update branch. B9's "Should not happen" disappears.
5. The ad-hoc caches on MCEqRun (`_em_step_scale_cache`, `_loss_step_cache`, `_operator_cache`, `_backend_cache`) are invalidated together on rebuild.
- Risk: tracking requests must be re-applied on a fresh pman. DDM and mod_pprod overrides live on `Interactions` and survive. Public `pman`, `matrix_builder`, `int_m` and `dec_m` are unchanged.

### S5. `species/` → one `particlemanager.py` (1203 → ~650 lines)
- **MCEqParticle** becomes `@dataclass(eq=False)`, which gives identity eq/hash and fixes B4 and the name-based `__eq__`.
  - `from_table(pdg, hel, physics, ...)` replaces `_init_defaults_from_pythia_database`, `init_custom_particle_data` and `_apply_force_resonance`.
  - Derived properties: `is_charged`, `can_interact`, `is_projectile`, `children = list(decay_dists)`, `hadr_secondaries = list(hadr_yields)`. That removes the parallel list+dict bookkeeping.
  - `is_secondary`/`is_child` (1 caller each) become `sec in proj.hadr_yields` / `sec in proj.decay_dists`.
  - `track_decays`/`track_interactions` become one `track(tp, table)`. The three `dN_*` become one `_dN(table, ..., xlab)` plus 3 one-liners.
  - Tracking copies use `dataclasses.replace(...)`, which removes the "shared dict until first re-wire" subtlety.
  - Delete the dead fields: `int_idx`, `current_cross_sections`, `current_hadronic_model`, `A/Z/N`, the duplicate `is_stable`/`decay_dists` init, and `E_crit` with its overridden sort.
- **ParticleManager:** delete `set_cross_sections_db` (0 src callers), `keys`, `nspec`, `_tracking_requested_by_default`, the `SibyllParticleTable` fallback, the 3 unused index maps and the duplicate `tracking_relations`. Fix B3 and B5.
- Also: `prod_cross_section` was named `inel_cross_section` in v1.4.2. It was renamed without an alias, which is an **API** break.
- Risk: identity equality matters for code that keeps particle references across a rebuild. Inside src, references are always re-fetched. Run the full test suite.

### S6. particletools → scikit-hep `particle` (your constants.py note)
- `_pdata` call sites: `particle.py:137-177` (9), `constants.py:38`, `manager.py:554,565`, `manager.py:379-381` (dead), `ddm.py:135,830`.
- **NUM risk:**
  - PDG-2024 masses and lifetimes differ from the PYTHIA table, which shifts decay lengths, βγ and the 2D baselines.
  - Names differ (`p+`, `n0`, `K0L`, `Lambda0`), and species names are public through `get_solution` and the tracking aliases.
  - `None`/inf lifetimes break the `ctau==0` `ZeroDivisionError` path.
- Lowest-risk option: **freeze a small generated table** (pdg → name, mass, ctau, charge, kind) and drop the runtime dependency entirely.

### S7. `data/`: 15 → 6 files, public `MCEq.data` unchanged
- Delete `protocols.py` (0 runtime uses) and `em_tables.py`:
  - ρ-slice code: delete everything.
  - `medium_for_em`: has **no effect**; its result is overwritten.
  - `helicity_duplicate`: unreachable.
  - `em_group_path`: an f-string.
  - Also remove `em_air_density` from config, groups and docs, and the `em=` keyword of `HDF5Backend`.
- Merge `model_names.py` and `equivalences.py` into `hadronic_models.py` (about 170 lines). This removes `HADRONIC_MODEL_FAMILIES` (it duplicates the dict keys) and the module/dict name clash, along with about 17 lines of prose explaining that clash. Inline `apply_equivalences`/`reverse_equivalences` (1 caller each), as in v1.4.2.
- Move `energy_grid.py` back into misc (see section 1).
- Put the 4 table classes (`interaction_tables`, `decay_tables`, `cross_sections`, `continuous_losses`: about 640 lines) back into one `tables.py`. One helper then rebuilds `relations`/`particles`, replacing 4 copies.
- Merge `db_manifest.py` into `download.py` and move `ModelStoreCache` into hdf5_backend. That drops the injected `fetcher`.
- Optionally fold `blending.py` (1 caller) into hdf5_backend.
- `data/__init__.py`: export only the v1.4.2 names. Today it also exports `isfile`, `join`, `info`, `blend_*`, `_eval_energy_cuts` and other internals.
- **Needs your decision:** the two PDG→family fallbacks (`cross_sections.get_cs` and `equivalences.family_fallback`) disagree on 10313/10323 and on leptons, and a test pins that disagreement. Unifying them is cleaner but **NUM**: it changes blended cross sections for those IDs.

### S8. DDM → `src/MCEq/ddm.py` and delete `models/` (866+520 → ~250+150 lines)
- The code is almost verbatim v1.4.2, so the overengineering is inherited. About half of ddm.py is numpydoc on one-line properties.
- Replace `_DDMChannel` + `DDMSplineDB` + the string keys with `{(proj, sec): sorted list}`.
- Delete `_eval_spline_and_correction` (0 uses), `_mk_channel`, `_load_from_file`, `_LIMIT_PROPAGATE` (always inf) and possibly `_gen_dndx`.
- Write one matrix-fill loop that computes dndx once per entry. That halves DDM init time. To stay bit-identical it must keep the index-based bracketing and the single-spline `[:-1]` quirk.
- Move the isospin-partner bookkeeping out of `MCEqRun.inject_ddm` into `ddm_matrices`.
- Keep `gen_matrix_variations`. It was public in v1.4.2 and has 0 tests, so add one.

### S9. Solvers and operators
- **Dissolve `solvers/backends/base.py`.**
  - It is not a base class. It holds a 20-line docstring stored as a never-read string constant (`_PRECISION_CONTRACT`), a 2-entry dtype map, and `ScipyApplyOff`.
  - Pass `np.dtype` throughout, instead of converting dtype → fp=32/64 → dtype.
- Merge `MklApplyOff` and `SpaccApplyOff` (about 70 lines each, near copies) into one ctypes strategy.
- Merge `CudaOperator` (1 caller) into `CudaBackend`.
- **One backend dispatch.** `etd2._BACKENDS` + `solve_etd2` (0 src callers, about 20 test callers) duplicate `MCEqRun._etd2_backend`, which also has its own alias list next to `config/detect.ALIASES`.
- `solvers/__init__.py` re-exports 21 names, and only 7 are used. Some of them expose operator internals through another path.
- **Step-size policy in one place.** Delete `operators/stiffness.py` (1 caller) and move `continuous_loss_rate` and `compiled.em_step_scale` into `solvers/path.py`. That removes the `driver/paths.em_cascade_dx_cap` → `solvers/path.em_cascade_dx_cap` double wrapper, and `MCEqRun._em_cascade_dx_cap` (0 src callers).
- **`mkl_runtime`:** move `has_mkl`/`mkl_library_path` into it. Delete `set_resolver`, `_has`, `_path`, the lambdas and `config._load_mkl`. `load()` becomes about 6 lines.
- **MatrixBuilder:**
  - Delete `max_lint`/`max_ldec`. They are a forward-Euler leftover that is computed and never read.
  - Replace the lazy-split/assembly-key guard machinery (about 110 lines) with an eager build in `construct_matrices`.
  - Make the two-level stencil cache one helper.
- **Dead Legendre API:** `moments`, `forward_weights` and `hemisphere_projection` have 0 src callers. Check the DB-generation tools before deleting.
- **Retired features still carried:**
  - `adv_set["force_resonance"]` ("escape hatch"; `particle.py:553`, `matrix_builder.py:589`).
  - Loss stencils `biased`/`centered`.
  - `path.py:68` links to "the resonance approximation" doc section.
  - `_EM_BLOWUP_CAVEAT`: a never-read string constant.

### S10. Environment: the MSIS class zoo (about 1510 lines in 4 modules)
- Today there are 8 public classes, a mixin with abstract hooks, and **2 copies** each of `update_parameters`, KM3NeT validation, IceCube presets, `_latitude`, the azimuth-averaged spline builder and the default-parameter setup.
- `*LocationCentered` subclasses `MSIS00Atmosphere` only to bypass its `__init__`.
- `msis21.set_theta` is byte-identical to the base class version.
- Proposal: one `MSISAtmosphere(EarthsAtmosphere)` composed with a backend object (msis00 ctypes or nrlmsis), one detector-centred subclass, and the IceCube/KM3NeT presets as kwargs. Keep `MSIS00Atmosphere` and `MSIS00IceCubeCentered` as thin aliases (they are v1.4.2 API).
- Merge `msis00_backend.py` into msis00. That removes the single-implementor base class, the dead `surface_vert_depth`, the test-only `height2depth` and the broken `test()`.
- Replace `registry.py` (1 caller, 65 lines) with a plain dict next to `set_density_model`. Its "lazy import" rationale is false.
- Move `build_phi0_with_cutoff` to the driver (S3). `geomagnetic/` then becomes one module, `environment/cutoff.py`, with no fake-tqdm stand-in. The current progress bar jumps 0→100% anyway.
- `geometry/density_profiles.py` shim:
  - Limit it to the v1.4.2 names. It currently adds 7 v2-only names and a `__getattr__` MSIS21 forwarder whose stated rationale is false.
  - Decide consistently on the `atmosphere_parameters`/`nrlmsise00_mceq` shims. They were removed although they were v1.4.2 public, while the lower-level ctypes shims were kept.
- `fit_column_splines`: compute the heights internally and drop the `h_intp`/keyword-only contract and about 50 lines of ULP prose. **NUM** at the 1–2 ULP level; the pin tests need a tolerance.
- Delete the `__main__` demos in `geometry.py`, `corsika.py` and `msis00.py`, and the dict-reading getters in `parameters.py` (0 src callers).

### S11. misc, version, top-level
- `misc.py` (442 → ~150 lines): delete or inline `theta_deg`, `_get_closest`, `pdg2corsikaid`, **`corsikaid2pdg`** (0 callers; not in your notes), `getAZN_corsika`, `is_charm_pdgid`, `reset_plt`, `put_ticks`, `strip_em_species`/`EM_PDG_IDS` and the `_MiscCompatModule` machinery.
- `version.py`:
  - pyproject has a static `version = "2.0.0"`, and nothing imports `MCEq.version`. `MCEq.__version__` doesn't exist. `docs/conf.py:41` has a third copy (`"2.0.0-dev"`, with a "D-6" comment).
  - Fix: use `__version__ = importlib.metadata.version("MCEq")` in `__init__.py` and in `docs/conf.py`. Alternatively, `dynamic=["version"]` with the scikit-build regex provider.
- `core.py`: keep `MCEqRun` and `MatrixBuilder` (both v1.4.2), and drop `MCEqBatchResult` from that path.
- `mceq_config/` is legitimate: it was in v1.4.2. Stale `mceq_config.` references remain in docstrings at `driver/observables.py:26,53,69`, `environment/target.py:27` and `misc.py:182`.

### S12. Tests are protecting the scaffolding
`test_phase4_surface.py`, `test_channel_protocol.py`, `test_data_em_tables.py`, `test_group_injection.py`, `test_conftest_isolation.py`, `test_registry.py` and parts of `test_data_bug_pins.py`, `test_data_model_names.py` and `test_environment_pins.py` pin module paths, object identities, ρ-machinery and ULP evaluation order. They need pruning together with the refactor, or they will block it. `tests/fixme_test_particlemanager.py` is not collected by pytest; delete it or revive it.

---------------------------------------------------------------------------------------------------

## 4. Per-file findings (prose and local overengineering)

Format: `line` | codes | sev | problem → fix. Items already covered by your `anfe:` notes or by S1–S12 are not repeated.

### driver/
**`__init__.py`**
- `1-6` | P1 | low | A tour of the modules → `"""MCEqRun and the modules it delegates to."""`

**`system.py`**
- `1-16` | P1,P2 | med | The split story → `"""Database handle, data tables, particle manager and matrix builder of a run."""`
- `26-49` | P2,P4 | med | `run_physics_view`: 15 lines of docstring for 8 of code ("C1", "util sits below config") → one line, and move the rule into config validation.
- `53-68` | P2 | low | The class docstring narrates construction → `"""Tables, particle manager and matrices for one interaction model."""`
- `81-84` | P6 | low | "this one instance is what every table… read" → delete.
- `97,147` + `mceq_run.py:80,362` | O2 | low | `normalize_hadronic_model_name` runs 4× on one path → normalize once.
- `134-146` | P2 | med | "branches from the normalize-and-load point… only MCEqRun can drive the order" → `"""Load interaction_model and rewire the particle manager."""`
- `211-227` | P2,O2 | low | `regenerate`: 1 line of code with a 4-line docstring and 1 caller → inline.

**`mceq_run.py`**
- `48-52` | P3,O3 | med | The class attribute `_cfg = config` exists "so unbound calls from test stubs keep working" → set it only in `__init__`.
- `55-57,69-72,88-90,102-103` | P2 | med | "lives in driver/system.py; the public properties below forward…" → delete.
- `348-356` | P2,P6 | med | "the system's branch ladder", "skip short-circuit" → `"""Select the interaction model and rebuild matrices."""` Document `particle_list` and `update_particle_list`.
- `405-447` | O3 | med | 5 delegators whose docstrings say only "delegates to InitialState", so `help()` loses the parameter docs → see S3.
- `433-447` + `initial_state.py:274-357` | O3 | med | The `initial_state(components)` dict mini-DSL (70 lines) next to `get_initial_state()` has a confusingly close name → drop it.
- `549-569` | P4,P5 | med | A 6-line story on why `theta_deg` is assigned late → `# store only after set_theta accepted the angle`
- `642-644` | O2 | low | `delay_init` is "accepted but currently unused" → drop it.
- `688,713-722` | P3,O3 | low | A `**kwargs` bag for `skip_integration_path` ("tests / experts") → remove; `solve_from_integration_path` covers it.
- `742-753,800-805` + `batch.py:200-206` | O5,P4 | med | The 2D `np.tile(phi0, n_k)` appears 3× with 3 essays → `_stack_modes(phi0)`.
- `818-841` | O6 | low | `_is_geomag_eligible_atmosphere` is an isinstance switch with 1 caller → make it a property of the atmosphere.
- `860-864` | P6,P3 | low | "One constant set serves every column…" and the `getattr(self, "_cfg", config)` stub fallback → `"""sec(theta) operators for the solve, or None if secant mode is off."""`
- `1026-1035` | P4 | low | A 10-line docstring on a 12-line memo → `"""Spectral radius of the EM block of int_m (cached per matrix)."""`
- `1059-1086` | O3 | low | A "thin delegator" to `paths.calculate_integration_path` → put the body back in the method.

**`initial_state.py`**
- `1-12` | P1,P2 | med | Method-name-vs-bound-method justification → `"""Initial condition (primary flux or user spectra) of the cascade."""`
- `68-71` | P2 | low | The same justification again → delete.
- `74-77` | O2 | low | `finalize_pmodel` is never read → delete.
- `81-103` | O2 | low | An unreachable "Protons not in equation system" branch → delete.
- `94-123` | O5 | low | `pman[(2212,0)]` is written 6× → bind it once.

**`results.py`**
- `1-15` | P1 | med | Explains the method binding and cites `docs §11.9` → `"""Spectrum extraction and the batch-solve result container."""`
- `81,88-91` | O3 | low | The `shared_path` property only wraps a value, and only tests read it → plain attribute.
- `198-208` vs `313-320` | O5 | low | The `grid_idx` selection is duplicated → one helper.
- `273-275` | O6 | low | The `return_as` defaults differ between the two `get_solution` versions → `None` in both.
- `46-47` | O2 | low | `pixel_index` is never read → drop it.

**`batch.py`**
- `1-11` | P1,P2 | med | Docstring prose about `stacklevel=2` → `"""Batched solves: many initial states and/or directions in one call."""`
- `22-41` | O2,P5,P6 | med | `_warn_low_energy_cuda_fp32` has 1 caller and its warning text reads like a test report → inline it with a one-line message.
- `87-95,127-134` | P2,P5 | low | Secant and fp32-cancellation essays → "dtype: precision of the state buffers."
- `157-198` vs `378-406` | O5 | med | phi0 resolution and validation are copy-pasted (~40 lines ×2) → `_resolve_phi0(run, phi0, K)`.
- `335-343` | P2,P6 | low | "sweet spot across the full-sky benchmarks for K ≤ 2664" → `carousel_K: pipeline width (default min(K, 128)).`
- `417` | O2 | low | Duplicate `import warnings`.
- `494-497` | O6 | low | The result is mutated after construction → pass the sky grid to the constructor.
- (with paths.py and mceq_run.py) | O5 | med | 5 path knobs are threaded through 7 signatures → one `path_opts`, or read them from config.

**`paths.py`**
- `1-15` | P1,P2 | med | "so subclass overrides … still reach this code" → `"""Integration-path construction, including per-condition batched paths."""`
- `38-54` | P5,O6 | med | The docstring says only "Memoize on…", and the function reaches into the builder's privates `_contloss_*` → state what is computed; move it to the builder or the path module (S9).
- `114-124` | O2 | low | The log message re-resolves defaults → log the resolved values.
- `137-147` | P2,P4 | low | 10 lines on fork, CoW and "bitwise equal" → `# Run instance shared with forked path workers.`
- `200-217` | O3 | low | Double validation, and the messages name a private function → keep only the key check.

**`observables.py`**
- `1-6` | P1 | low | Explains the binding mechanism → `"""Particle numbers and Z-factors from a solved cascade."""`
- `105-109` | O2 | med | A cupy `asnumpy` branch (new in v2) on numpy data → delete.
- `99/103,159/161` | O2 | low | `zfac` is assigned twice → keep one.
- `163-165` | O2 | low | Commented-out code → delete.

### data/
**`__init__.py`**
- `1-4,51-53` | O1 | high | Re-exports `isfile`, `join` and `info` → drop them.
- `10-55` | O1 | high | Exports v2 internals → keep the v1.4.2 names only.
- `6-9` | P2,P3 | med | Cites `test_phase4_surface.py` → delete.
- `35-37,58-59` | P4 | low | Comments explaining what `__all__` is → delete.

**`model_names.py`**
- `1-14` | P2,P3 | high | → `"""Hadronic model names, families and projectile equivalences."""` (after the merge)
- `49-63` | P4 | low | `family_of`: 15 lines of docstring for 4 lines of code → one line.

**`equivalences.py`**
- `14-26` | P3 | high | "Two object identities must be preserved" exists only to defend test pins → delete.
- `178-240` | O2,P2 | med | `reverse_/apply_equivalences` have 1 caller each, and the latter takes 8 arguments only because it was extracted → inline them.

**`blending.py`**
- `1-18` | P1,P2,P6 | med | → `"""Runtime blending of a high- and a low-energy hadronic model."""`
- `50-67` | P2,P4 | med | "matching the former compiled extension", "sole record" → 2 lines.

**`hdf5_backend.py`**
- `1-8` | P1,P2 | med | → `"""HDF5 database backend: reads, decodes and blends interaction, decay, cross-section and loss tables."""`
- `35-38,90-91` | P2,P6 | med | "injected by the driver -- there is no fallback" → delete.
- `290-310` vs `359-374` | O5,O6 | med | The air fallback is written twice → `_resolve_interaction_store` on both paths.
- `312-335` | O2 | med | `_blend_interaction_dbs` (1 caller, forwards arguments only) → inline.
- `540-577` | O5 | med | The `missing_model_error` block is copy-pasted → `_require_store`.
- `384-390,434-440` | O5 | low | The empty-index literal appears twice.
- `401,589` | low | The `cs_db` log message says "EM matrices".
- `606-620` + `continuous_losses.py:51-69` | O6 | med | `ContinuousLosses` reopens the HDF5 file through raw h5py to swap the generic curve → choose the curve in the backend.

**`hdf5_store.py`**
- `1-11` | P1,P2 | med | "the IO seam", "imports nothing from MCEq" → `"""Read-only access to one MCEq HDF5 file, returning numpy arrays."""`
- `22-29` | P6 | low | "no cast applied, the dtype cast is backend policy" → list the fields.

**`interaction_tables.py`**
- `1-7` | P2 | med | "no import-time edge" → `"""Interaction yield matrices with channel overrides and production modifications."""`
- `43-53` | O2,P4 | low | Dead `self.band` and a 7-line attribute comment.
- `56-79` | P2,P4 | med | `set_channel_override`: 24 lines of docstring for 4 lines of code, with history ("no longer… silently discards") → 3 lines.
- `23,80` | low | Redundant local imports.
- `247-250` | P2 | low | Bug-history comment → `# the other nucleon`

**`decay_tables.py`**
- `1-5` | P2 | low | → `"""Decay yield matrices."""`
- `22` | O2 | low | `parent_list` is never read.

**`cross_sections.py`**
- `89,91` | low | `parent in list(self.index_d)` → `parent in self.index_d`.

**`continuous_losses.py`**
- `1-13` | P1,P2 | med | → `"""Continuous energy-loss tables and the generic hadron dE/dX spline."""`
- `78-84` | low | A possible KeyError → use `index.get("generic")`.

**`db_manifest.py` / `download.py`**
- `db_manifest.py:96-103` | O3 | med | `download` is injected as `fetcher` → merge the modules.
- `download.py:127-143,191` | P2,P4 | med | "this module imports no configuration itself" → `"""Download config.mceq_db_fname into config.data_dir if missing or corrupt."""`
- `download.py:107,159` | O5 | low | The legacy checksum special case appears twice.
- `download.py:222,237,338` | low | db_manifest is imported locally 3×.

### operators/
**`__init__.py`**
- `1-30` | P1,P2 | high | A 30-line submodule tour plus import-cost notes → `"""Cascade operator assembly: interaction/decay matrices, loss stencils, Legendre modes, muon scattering, compiled ETD2 operator."""`

**`matrix_builder.py`**
- `58-63` | P2,O5 | low | `UPWIND_ROWS_FLOOR` duplicates a config default and cites a docs section.
- `67-77` | P2 | low | "no fallback to live config" → `"""Builds int_m and dec_m from a ParticleManager's yields."""`
- `294-332` | O5 | med | `_upwind_rows_for` runs twice per species, with a two-level cache → one helper returning `(op, n_rows)`.
- `298…674` | O2 | low | `getattr(self._losses, "...", default)` restates config defaults 2–3× → plain attribute access.
- `403-427` | P4,P2 | med | `dEdx_band`/`_csr_from_bands` essays → one line each.
- `498-526` | P4,P2 | med | `_csr_from_blocks`: 28 lines including "see secant for why it is not stitched" → 2 lines.
- `580-593` | P2 | low | `_follow_chains` uses v1 naming and "we" → one line.
- `646-683` | P2,P4,O5 | med | "Called from __init__ and nowhere else…", and the clamp is duplicated from loss_stencil.

**`loss_stencil.py`**
- `1-70` | P1,P2,P5 | med | Keep the d/du definition, eq. (19) and which families are exact on constants. Drop the comparison with the docs, the benchmark numbers and the row-layout narrative.
- `129-144` | P6,O2 | low | The `offsets=None` default is never used.
- `165-169` | O5 | low | The method names are hand-listed → build them from `STENCIL_METHODS`.

**`compiled.py`**
- `1-14` | P1 | low | → `"""Diagonal/off-diagonal split and secant layout of the operator for the ETD2RK step loop."""`
- `117-121,212-214` | P6,P2 | low | "nothing here enforces that", "MCEqRun caches…" → delete.
- `36-40,165-167` | O2 | low | `identity_layout` (1 use) and `split` (test-only).
- `249-305` | P3,P4,P5 | high | `em_step_scale`: 22 lines of docstring plus an ARPACK story, "Cure-B" and a test citation, plus a three-tier fallback → `"""Spectral radius of the off-diagonal e±/γ block of int_m; 0 if no EM species."""` Keep only the dense method plus the norm bound.

**`secant.py`**
- `31-42` | P2 | med | Argues why the coupling is not stitched → "The coupling is applied in the step loop, not stitched into the CSR."
- `49-55` | P5 | low | → `#: Max |imag|/real accepted for an eigenvalue of S_P.`
- `148` | O2 | low | `T_PP` is test-only.

**`legendre.py`**
- `16-18` | P2 | low | A Hankel remark → delete.

**`scattering.py`**
- `52-54` + `matrix_builder.py:481` | O2 | low | `MUON_MASS` "fallback only" plus a try/except → pass the mass from pman.
- `73-76` | P6 | low | → delete the sentence.

### solvers/
**`__init__.py`**
- `1-22` | P1,P6 | med | → `"""ETD2RK cascade solvers and their sparse/device backends."""`

**`etd2.py`**
- `13-15` | P2 | low | A layering justification → delete.
- `49-56` | P5 | low | "30-90 ms per call…" → "Limit OpenBLAS/BLIS to one thread while MKL runs the SpMM."
- `226` | low | Local `from time import time`.

**`numerics.py`**
- `25-29` | P3 | med | Cites `test_phi_factors_accuracy` → "Taylor-switch radii; worst rel. error 7.9e-13 / 5.4e-12."
- `109-115` | O2 | low | `phi_work` has 1 use → inline.

**`path.py`**
- `9-11` | P2 | low | → delete.
- `34-44` | O2 | med | `_EM_BLOWUP_CAVEAT` is a never-read constant → a 2-line comment at the `errstate`.
- `159-223` | P5,O6 | med | A 22-line "why hybrid sample" story, and `except Exception` silently falls back → 2 lines; catch the specific error.
- `242-262` | P2,P6 | med | "legacy schedule reproduced exactly", "(cure B)" → `"""EM step cap em.step_safety/r_EM, or inf when disabled."""`

**`schedule.py`**
- `9-13` | P2 | low | Stale "stage 7" and "belongs here" → keep the first paragraph only.

**`backends/__init__.py`**
- `1-12` | P2 | low | → one line.

**`backends/host.py`** (+ `mkl.py:362`, `accelerate.py:51`)
- `148-161` | O5 | low | 3 copies of the ctypes pointer memo → one helper.

**`backends/mkl.py`**
- `9-10,128-129` | P2,P5 | med | "BSR was a 1.5x win…" (something not done) → delete.
- `50-52` | O2 | low | Duplicate ctypes imports, and a global argtypes flag.
- `216-249` | P6 | low | "Mirrors SpaccMatrix…" is false → trim to the formula.
- `236,273` | O2 | low | The `layout=102` default is never used.
- `299-307` | P4 | low | → `"""Free the MKL handle (idempotent)."""`

**`backends/cuda.py`**
- `24-36,226-232,274-299` | P4,P5,P3 | med | CuPy 13-vs-14 story, "S = 1 within 3 %", a 25-line kernel-builder docstring → 1–2 lines each.
- `237,271-338` | O3 | low | Two hand-rolled memos → `functools.cache`.
- `370` | O2 | low | `fp_precision` is never read.

**`_kernels/etd2` and `_kernels/spacc`**
- (both) | O7,O5 | low | A duplicated Python 2.7 `SO` fallback, and inconsistent errors (NameError vs ImportError) → one `_load_ext` helper.
- `spacc:1` | P1 | low | No module docstring.
- `spacc:90` | low | `free_mstore()` runs at import.
- `spacc:113-169` | P5,P4 | low | Benchmark and close() stories.
- `spacc:196` | O2 | low | `gemv_npargs` is test-only.

### environment/ and geometry/
**`geometry/density_profiles.py`**
- `1-13` | P2,P3 | high | Counts "the exact 19 names…" and cites a test → `"""Atmosphere classes re-exported under their v1.4 import path."""`

**`environment/__init__.py`**
- `1-15` | P1,P2 | med | "census-absent module shims", plus a false layering claim → `"""Atmospheres, Earth geometry and geomagnetic cutoffs."""`

**`environment/registry.py`**
- `1-8,53-60` | P2,P6 | med | "costs nothing", "the caller owns the user-facing message" → (S10).

**`environment/base.py`**
- `81-83,155-164` | P5 | low | `max_den` is "the path minimum, despite the name" → rename it.
- `140-145` | P3,P5 | med | A ULP-preservation comment that cites a test → delete.
- `190-192` | P2 | low | "there is no spline cache" → delete.

**`environment/column.py`**
- `1-10` | P2,P6 | med | → `"""Slant-depth integration and the rho(X) splines shared by all atmospheres."""`
- `18-72` | P3,P4,P5 | high | A 55-line docstring on 15 lines of code → Args/Returns plus one sentence.
- `79-90` | P2,P5 | med | "a behaviour change, not a relocation", "numpy may dispatch…" → delete.

**`environment/geometry.py`**
- `93-96` | P4 | low | Drop "so a caller that passes all three never touches config".
- `118-141` | O5 | low | Overlapping validations, duplicated in `__init__` and `set_h_obs`.

**`environment/location_centered.py`**
- `1-22` | P2,P3,P5 | high | The `A**2` vs `A*A` libm story, "only the exception type is pinned by the tests" → `"""Detector-centred geometry shared by the MSIS00 and MSIS21 backends."""`
- `34-44` | P3 | med | Cites a test, plus "~1e-14 … not a defect" → keep only the hooks list.

**`environment/msis00.py`**
- `35-49` | O5 | low | A hard-coded location list duplicates `LOCATIONS`.
- `55-57` | P2 | med | Bug history → delete.
- `189-250` | P4 | med | 60-line class docstring, copied again in msis21.
- `300-306` | O2 | low | Empty section banners.
- `395-401` | P2 | med | "kept for backward compatibility" → `"""MSIS00LocationCentered preset for IceCube."""`

**`environment/msis00_backend.py`**
- `61-63` | P2 | low | "Kept so the field is never unset…" → delete.
- `107-118` | P3,P4 | med | Cites a test → `"""Set lst=12 h and the matching UT seconds."""` If the MSIS00/MSIS21 noon conventions differ, that is a physics issue to fix, not to document.
- `137-146` | P4 | low | 8 lines for one assignment.
- `24-26` | O3 | low | Aliases of module constants.

**`environment/msis21.py`**
- `1-21` | P1,P2 | high | A benchmark story, "Drop-in replacements" → `"""NRLMSIS 2.1 atmospheres (requires the nrlmsis package)."""`
- `45…488` | P6 | med | "Drop-in replacement for…" appears 5× → one-line class docstrings.
- `217-223,455-456` | P2,P5 | med | Benchmark docstrings.
- `361-367` … | O2 | low | Empty or decorative banners.

**`environment/tabulated.py`**
- `34-37,108-110` | P2 | med | Layering comments → delete.
- (also) | low | A hard-coded OneDrive path, and commented-out code.

**`environment/target.py`**
- `1-6` | P6 | med | "the protocol boundary between…" → `"""Piecewise-constant material target (non-atmospheric medium)."""`

**`environment/parameters.py`**
- `350-355` | P2 | low | → `# KM3NeT detector sites (lon, lat in deg; depth in m)`

**`environment/corsika.py` and `isothermal.py`**
- `location`/`season` are reset to None by the base `__init__`. This was already wrong in v1.4.2.

**`geomagnetic/cutoff.py`**
- `1-30` | P1,P2 | med | Names the wrong caller (`core.MCEqRun`) → `"""Geomagnetic rigidity cutoffs from gtracr (cached per site)."""`
- `41-48,133-146,253-258` | P2,P6 | med | "cannot drift away from the position it names" → one line each.

**`_ext/nrlmsise00`**
- `46-66` | P4,P5,O6 | med | A 20-line essay on the `_field_` typo → make `nrlmsise_input` a plain holder, or fix `_fields_`.
- `_ext/*` `__init__` comments | P2,P4 | low | Trim.

### config/
**`config/__init__.py`**
- `149-167` | O2 | low | Misplaced `#:` doc, and 2 helpers used once.
- `201-228` | P5 | med | The `em_adaptive_step`, `em_step_safety` and `em_step_dense_eig_max` comments are benchmark essays → one line each.
- `244-271` | P5,P2 | med | `generic_losses_all_charged` and `average_loss_operator` carry history ("pre-2026-06 numbers").
- `276-298` | P5,O7 | low | The stencil list → one line per method. Drop `biased`/`centered`?
- `324-336` | P2,P5 | med | `muon_helicity_dependence` cites DB versions → `#: Helicity-dependent muon decays. Keep on. Forced off with enable_em.`
- `348-358` | P6 | low | "every entry point, every kernel, fp32 and fp64".
- `412-417` | O7 | med | The `force_resonance` "escape hatch" → remove it (S9).
- `434-445` | P2,P6 | med | "one owner below config…" → `# Platform probes resolve on first read (see detect.py).`
- `453-454` | O2 | low | Lazy `pf`/`prefix` entries have 0 uses.
- `474-484` | P2,P3,O2 | med | `_load_mkl` is test-only and cites a test.
- `497-527` | P4 | low | `set_mkl_threads` mentions internals → one line.

**`config/detect.py`**
- `1-7,28,76-81` | P1,P4 | low | Trim the docstrings.

### models/ddm/
**`ddm.py`**
- `89-135` | P4 | low | Numpydoc blocks on one-line properties.
- `173-175,242` | O2 | low | Redundant local imports.
- `461-463` | O2,P2 | low | A class attribute plus a comment explaining why it is harmless.
- `569-588` | P4 | low | Documents a nonexistent `ebeam` parameter.

**`ddm_utils.py`**
- `366-438` | O5 | low | Two `quad` calls that differ only in their limits.

### species/, misc, top level
**`species/manager.py`**
- `1-5` | P1,P2 | med | → `"""ParticleManager: registry of particles and their state-vector indices."""`
- `36-38,47-55` | P2,P6 | low | Signature-default justification and stale paths.
- `62` vs `76` | O2 | med | `tracking_relations` is initialised as `{}` and then as `[]`.
- `83,483` | O2 | low | The particle tables are printed twice.
- `142` | O3 | low | A defensive `getattr` on `generic_table`.
- `181-186` | P2 | med | "replacing the older obs_particle mechanism… 71XX" → one line, and document `include_antiparticle`.
- `240-259` | O2,O3 | low | Commented-out print, and a 100-iteration unique-id loop.
- `247,308,555` | low | `x in list(dict)`.
- `363-382` | O2,P2 | med | Dead `SibyllParticleTable` fallback.
- `421-450` | O5 | low | `add_new_particle` logs the same block 3× (keep the method: v1.4.2 API).
- `461-472` | O2 | low | A list comprehension used for its side effects.

**`species/particle.py`**
- `29-41` | P6 | low | The class docstring omits real args.
- `62-197` | O2 | low | Fields that are dead or duplicated → S5.
- `194-197` | style | → `bool(np.any(self.cs > 0))`
- `321` | O3 | low | A defensive `getattr` on `rate_index`.
- `405-419` | P4,O6 | low | Uninitialised `_decay_rate_factor`, and the ZeroDivisionError path relies on a Python float.
- `431` | O2 | low | A local import just for `mbarn2cm2`.
- `471,519` | P3 | low | References to the private `_set_mod_pprod`, and `**kwargs` that are ignored.

**`species/constants.py`**
- `14` | O6 | low | `info()` runs at import.
- whole file | O4 | med | One singleton plus one constant → fold into the particle module.

**`misc.py`**
- `259,282` | O2 | low | Dead `Z, A = 1, 1` assignments.
- `340` | O2 | low | Redundant `import sys`.
- `158-183` | P6 | low | Mentions a `RunConfig` snapshot, and names `mceq_config.A_target`.
- `389,395` | style | `_views()` is called twice.

**`particlemanager.py`**
- `1-7,19-25` | P2,P3 | med | Cites tests and a docs page that no longer exists → S5.
