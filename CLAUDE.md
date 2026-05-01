# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCEq (Matrix Cascade Equations) is a numerical solver for cascade equations that model particle density evolution through gaseous/dense media. Primary application: simulating particle cascades in Earth's atmosphere. It outputs differential energy spectra and total particle counts.

## Build & Development

MCEq uses **scikit-build-core** with CMake for C extensions (NRLMSISE-00 atmosphere model and CORSIKA atmosphere parametrizations). The build produces Python extension modules (`.pyd`/`.so`) from C sources in `src/MCEq/geometry/`.

```bash
# Install in editable mode (builds C extensions into src/MCEq/)
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
# or using dependency groups:
pip install -e . --dependency-groups test

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_core.py

# Run a specific test
pytest tests/test_core.py::test_function_name -v

# Lint
ruff check .
ruff format --check .
```

## Architecture

### Core flow
1. **`MCEq.config`** (`src/MCEq/config.py`) — Module-level configuration (energy range, solver selection, atmosphere model, debug level, etc.). Auto-detects fastest available solver kernel: CUDA > MKL > Accelerate > numpy. Also handles database file download on first import.
2. **`MCEq.core.MCEqRun`** (`src/MCEq/core.py`) — Main user-facing class. Initializes the cascade equation system from an HDF5 database, manages interaction/decay matrices, and orchestrates solving. Key methods: `set_interaction_model()`, `set_primary_model()`, `set_theta_deg()`, `set_density_model()`, `solve()`, `get_solution()`.
3. **`MCEq.data`** (`src/MCEq/data.py`) — HDF5 backend for loading interaction matrices, decay matrices, cross sections, and continuous losses. Classes: `HDF5Backend`, `Interactions`, `Decays`, `InteractionCrossSections`, `ContinuousLosses`.
4. **`MCEq.particlemanager`** (`src/MCEq/particlemanager.py`) — `ParticleManager` and `MCEqParticle` classes that track particle properties (PDG IDs, lifetimes, interaction channels) and map them to matrix indices. Uses `particletools` for PDG data.
5. **`MCEq.solvers`** (`src/MCEq/solvers.py`) — Forward-Euler integrators: `solv_numpy` (pure numpy), `CUDASparseContext`/`solv_CUDA_sparse` (GPU via cupy), and MKL-based solvers.

### Geometry subsystem (`src/MCEq/geometry/`)
- **`geometry.py`** — `EarthGeometry` class: spherical Earth model, path length and zenith angle calculations.
- **`density_profiles.py`** — Atmosphere density models. Abstract base `EarthsAtmosphere` with implementations: `CorsikaAtmosphere`, `MSIS00Atmosphere`, `GeneralizedTarget`, etc.
- **`atmosphere_parameters.py`** — CORSIKA atmosphere parametrization data.
- **`nrlmsise00/`**, **`corsikaatm/`** — C extensions for atmosphere models, built via CMake.

### Data-Driven Model (DDM) (`src/MCEq/ddm.py`, `ddm_utils.py`)
Allows replacing default interaction model matrices with data-driven spline-based corrections. `DataDrivenModel` loads spline databases and generates modified particle production matrices.

### Configuration
- `mceq_config/` is a **deprecated** compatibility shim that redirects to `MCEq.config`.
- Config is module-level globals in `src/MCEq/config.py` — modify attributes directly (e.g., `config.e_min = 1.0`).

## Testing

Tests use **pytest** with session-scoped fixtures in `tests/conftest.py`. The test fixtures use a reduced database file (`mceq_db_v140reduced_compact.h5`) for speed. Test categories:
- `tests/test_core.py` — Integration tests for `MCEqRun`
- `tests/test_solvers.py` — Solver correctness tests with toy problems
- `tests/test_ddm.py`, `tests/test_charm_models.py` — DDM and charm model tests
- `tests/geometry/` — Atmosphere model and geometry tests

## Key Dependencies

- `h5py` — HDF5 database access
- `particletools` — PDG particle data
- `crflux` — Cosmic ray primary flux models
- `scipy` — Sparse matrices, ODE integrator, interpolation
- `numpy` — Core numerics
- Optional: `cupy` for CUDA GPU acceleration

## Changelog

Uses **towncrier** for changelog management. Add fragments to `changes/` directory with format `<issue>.<type>` where type is one of: `feat`, `bugfix`, `api`, `chore`, `docs`.
