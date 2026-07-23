# AGENTS.md

Guidance for AI coding agents working in the MCEq source tree — whether helping a
user run calculations, answering physics/API questions, or navigating the code.

## What MCEq is

MCEq (Matrix Cascade Equations) numerically solves the one-dimensional cascade
equations that describe particle-density evolution through the atmosphere (or
other media). Typical use: compute inclusive atmospheric lepton fluxes (muons,
neutrinos) for a chosen hadronic-interaction model, primary cosmic-ray flux,
zenith angle, and atmosphere. Results are differential energy spectra on a
discrete energy grid, or total particle numbers.

## Quick start for a user's calculation

```python
import crflux.models as pm
from MCEq.core import MCEqRun

mceq = MCEqRun(
    interaction_model="SIBYLL2.3d",
    primary_model=(pm.HillasGaisser2012, "H3a"),
    theta_deg=0.0,
)
mceq.solve()
e_grid = mceq.e_grid                      # energy grid centers [GeV]
flux = mceq.get_solution("total_numu", mag=3)  # E^3-weighted flux
```

`get_solution` accepts prefixes: `total_` (default), `conv_` (conventional),
`pr_` (prompt), `pi_`/`k_` (parent meson). The required HDF5 database
(`config.mceq_db_fname`) downloads automatically on first import.

Working examples to point users at (Jupyter notebooks in `docs/examples/`):
`Basic_flux.ipynb` (start here), `Compare_primary_fluxes.ipynb`,
`Muon spectra.ipynb`, `Zenith_distribution_IceCube.ipynb`,
`Dependence_of_spectrum_on_atmosphere.ipynb`, `KPi_demonstration.ipynb`,
`Partial_hadron_contribution.ipynb`. Rendered docs: quickstart and tutorial
under `docs/` (Sphinx).

## Common user questions → where to look

- **Change interaction model** — `mceq.set_interaction_model("EPOS-LHC")`
  (tags are normalized, see `MCEq.misc.normalize_hadronic_model_name`);
  available models are groups in the HDF5 database (`src/MCEq/data.py`).
- **Change primary flux** — `mceq.set_primary_model(...)` with a `crflux`
  model class; single primaries via `set_single_primary_particle`; custom
  spectra via `set_initial_spectrum`.
- **Zenith angle / direction** — `set_theta_deg(...)`, or v2's
  `set_zenith_azimuth(zenith_deg, azimuth_deg=None)` for azimuth-aware
  (location-centered) atmospheres.
- **Atmosphere / medium** — `mceq.set_density_model(...)`; models in
  `src/MCEq/geometry/density_profiles.py`: `CorsikaAtmosphere`,
  `MSIS00Atmosphere`, `MSIS00LocationCentered` (v2: lon/lat-aware, azimuth
  averaging), `MSIS00IceCubeCentered`, `GeneralizedTarget` (non-atmosphere
  targets).
- **Intermediate depths** — pass `int_grid` to `solve()` and read out with
  `get_solution(..., grid_idx=...)`.
- **Tweak particle production** — `set_mod_pprod` for ad-hoc modifications;
  the Data-Driven Model (`src/MCEq/ddm.py`) for spline-based, data-driven
  corrections.
- **Config knobs** — `src/MCEq/config.py` holds module-level globals (energy
  range, solver kernel, debug level, database file). Set them *before*
  constructing `MCEqRun`, e.g. `import MCEq.config as config; config.e_min = 1.0`.
  `mceq_config/` is a deprecated shim.

## What changed in v2 (vs. the 1.x API users may know)

- **ETD2RK exponential integrator** replaces the forward-Euler solvers; kernels
  are `numpy_etd2`, `mkl_etd2`, `accelerate_etd2`, and CUDA
  (`src/MCEq/solvers.py`). `config.kernel_config = "auto"` picks the fastest.
- **Resonance approximation retired** — short-lived resonances are handled by
  the integrator directly; no `hybrid`/resonance-approx code paths remain.
- **Location-centered MSIS atmospheres** with azimuth dependence and
  `set_zenith_azimuth`.
- Migration notes: `docs/mceq_v1.x_v2_diff.md` (and `docs/v14v13_diff.rst` for
  older history).

## Code map

1. `src/MCEq/core.py` — `MCEqRun`, the user-facing class: builds the system
   from the HDF5 DB, owns interaction/decay matrices, orchestrates solving.
2. `src/MCEq/config.py` — module-level configuration + DB auto-download.
3. `src/MCEq/data.py` — HDF5 backend (`HDF5Backend`, `Interactions`, `Decays`,
   `InteractionCrossSections`, `ContinuousLosses`).
4. `src/MCEq/particlemanager.py` — `ParticleManager`/`MCEqParticle`: particle
   properties and matrix-index mapping (uses `particletools`).
5. `src/MCEq/solvers.py` — ETD2 integration kernels (numpy/MKL/Accelerate/CUDA).
6. `src/MCEq/geometry/` — `geometry.py` (spherical-Earth path lengths),
   `density_profiles.py` (atmospheres), C extensions in `nrlmsise00/` and
   `corsikaatm/` (built via CMake).
7. `src/MCEq/ddm.py`, `ddm_utils.py` — Data-Driven Model.

## Build & test

Built with **scikit-build-core** + CMake (C extensions for the atmosphere
models land next to the Python sources in `src/MCEq/`).

```bash
pip install -e ".[test]"   # editable install with test deps
pytest tests/               # all tests (uses a reduced DB for speed)
pytest tests/test_core.py::test_name -v
ruff check . && ruff format --check .
```

Tests: `tests/test_core.py` (integration), `tests/test_solvers.py` (toy-problem
solver correctness), `tests/test_ddm.py`, `tests/test_charm_models.py`,
`tests/geometry/`. Session-scoped fixtures in `tests/conftest.py`.

## Contributing

Changelog uses **towncrier**: add a fragment `changes/<issue>.<type>.md` with
type one of `feat`, `bugfix`, `api`, `chore`, `docs`. Don't commit planning
documents, scratch analyses, or dev TODOs to this repo — it is the public,
user-facing package.
