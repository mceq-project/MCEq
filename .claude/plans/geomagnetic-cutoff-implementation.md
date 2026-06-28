# Plan: Geomagnetic Cutoff Integration via gtracr + crflux v2

## Context

MCEq now supports 3D location-coupled atmosphere models (`MSIS00LocationCentered`) with azimuth-aware density profiles. The next step is integrating geomagnetic cutoff rigidity, which modifies the primary cosmic ray flux depending on the shower direction. The cutoff is direction-dependent (varies with zenith and azimuth) and location-dependent.

**gtracr** v2.0.0 computes geomagnetic cutoff maps via Monte Carlo trajectory tracing through the IGRF-13 field model. **crflux** v2.0.0 already accepts a `geomagnetic_cutoff` parameter (rigidity in GV) that zeros flux below `E_cut = Z * R_cutoff` per nucleus.

The key challenge: currently azimuth averaging happens inside the density model (averages density over azimuth directions). With geomagnetic cutoff, the primary flux also varies per direction, so the cascade must be solved per-direction and the *solutions* averaged.

## Implementation

### 1. New file: `src/MCEq/geomagnetic.py`

`GeomagneticCutoffMap` class that wraps gtracr and handles caching.

```python
class GeomagneticCutoffMap:
    def __init__(self, location, particle_altitude=100, date=None,
                 bfield_type='igrf', min_rigidity=0.5, max_rigidity=80.0,
                 delta_rigidity=1.0, n_workers=None, solver='rk4',
                 iter_num=10000, nbins_azimuth=180, nbins_zenith=90):
        """Store parameters. Does not compute until compute() is called."""

    def compute(self, force=False):
        """Compute or load cached cutoff map. Uses gtracr.GMRC."""

    def get_cutoff(self, zenith_deg, azimuth_deg) -> float:
        """Interpolate cutoff rigidity for a direction (GV).
        Uses scipy.interpolate.RegularGridInterpolator on the fine grid."""
```

**Fine cutoff grid:** Default 180 azimuth x 90 zenith bins (2-degree resolution). The `get_cutoff()` method uses `scipy.interpolate.RegularGridInterpolator` to return accurate values for any arbitrary zenith/azimuth pair. The solve azimuth loop uses the density model's own (coarser) azimuth grid, looking up cutoffs via interpolation.

**Caching strategy:**
- Cache directory: `platformdirs.user_cache_dir("MCEq") / "cutoff_maps/"`
- Cache key: SHA256 hash of `(location, particle_altitude, date, bfield_type, rigidity_range, delta_rigidity, solver, iter_num, nbins, gtracr.__version__)`
- Format: `.npz` files containing `azimuth_centres`, `zenith_centres`, `rcutoff_grid`, plus a metadata dict
- If gtracr version changes, cache key changes and map is recomputed

**`location` parameter** accepts:
- A string name matching gtracr's `location_dict` (e.g., `"ORCA"`, `"IceCube"`, `"Kamioka"`)
- A `(longitude, latitude)` tuple for arbitrary locations -- creates a custom `gtracr.location.Location` object with the given coordinates

### 2. Modify `src/MCEq/core.py`

#### 2a. Store primary model class + tag in `set_primary_model()`

Currently only stores the instantiated `self.pmodel`. We also need to store the class and constructor args so we can re-instantiate with different `geomagnetic_cutoff` values per direction.

Add to `set_primary_model`:
```python
self._pmodel_class = model_class_or_object if isinstance(model_class_or_object, type) else None
self._pmodel_tag = tag
```

#### 2b. New method: `_compute_phi0(cutoff_rigidity=None)`

Extract the `_phi0` computation (lines 561-597) into a reusable method. When `cutoff_rigidity` is not None, temporarily set `self.pmodel.geomagnetic_cutoff = cutoff_rigidity` before computing fluxes, then restore it.

#### 2c. New method: `set_geomagnetic_cutoff(cutoff_map_or_rigidity)`

```python
def set_geomagnetic_cutoff(self, cutoff_map_or_rigidity=None):
    """Enable direction-dependent or uniform geomagnetic cutoff.

    Args:
        cutoff_map_or_rigidity:
            - GeomagneticCutoffMap: direction-dependent cutoff
            - float: uniform rigidity cutoff in GV
            - None: disable cutoff
    """
    self._geomagnetic_cutoff = cutoff_map_or_rigidity
```

#### 2d. Modify `set_zenith_azimuth()`

After setting angles, if a cutoff map is active and a **single azimuth** is specified, look up the cutoff for that direction and recompute `_phi0`:

```python
if self._geomagnetic_cutoff is not None and azimuth_deg is not None:
    if isinstance(self._geomagnetic_cutoff, GeomagneticCutoffMap):
        cutoff = self._geomagnetic_cutoff.get_cutoff(zenith_deg, azimuth_deg)
    else:
        cutoff = float(self._geomagnetic_cutoff)
    self._compute_phi0(cutoff)
```

#### 2e. Modify `solve()` for azimuth-averaged + cutoff case

When `_geomagnetic_cutoff` is a `GeomagneticCutoffMap` AND the density model is in azimuth-averaging mode, the solve loop becomes:

```python
def solve(self, int_grid=None, grid_var="X", **kwargs):
    if self._should_solve_per_azimuth():
        return self._solve_azimuth_averaged(int_grid, grid_var, **kwargs)
    # ... existing solve code (extracted into _solve_single) ...
```

New `_solve_azimuth_averaged()`:
1. For each azimuth in `density_model._azimuth_angles`:
   - Set density model to single-azimuth mode: `dm.set_theta(zenith, azimuth_deg=azi)`
   - Force recomputation of integration path: `self.integration_path = None`
   - Look up cutoff: `cutoff = self._geomagnetic_cutoff.get_cutoff(zenith, azi)`
   - Recompute primary flux: `self._compute_phi0(cutoff)`
   - Solve: `self._solve_single(int_grid, grid_var, **kwargs)`
   - Accumulate: `solution_sum += self._solution`
2. Average: `self._solution = solution_sum / n_azimuth`
3. Restore density model to averaging mode and restore `_phi0`

The existing solve logic (kernel dispatch on lines 1007-1063) gets extracted into `_solve_single()` to avoid code duplication.

### 3. Modify `src/MCEq/config.py`

Add default settings for geomagnetic cutoff computation:

```python
# Geomagnetic cutoff defaults (used by GeomagneticCutoffMap)
geomagnetic_defaults = dict(
    particle_altitude=100,   # km
    bfield_type="igrf",
    min_rigidity=0.5,        # GV
    max_rigidity=80.0,       # GV
    delta_rigidity=1.0,      # GV
    solver="rk4",
    iter_num=10000,
    nbins_azimuth=180,
    nbins_zenith=90,
)
```

### 4. Modify `pyproject.toml`

Add core dependencies:
```toml
"gtracr>=2.0",
"platformdirs",
```

Both are core dependencies -- gtracr is needed for geomagnetic cutoff computation, platformdirs for the cache directory.

### 5. Tests: `tests/test_geomagnetic.py`

- Test `GeomagneticCutoffMap` cache round-trip (compute, save, reload)
- Test `get_cutoff()` returns reasonable values (e.g., ~14 GV vertical at Kamioka)
- Test `set_geomagnetic_cutoff()` with uniform float value
- Test that single-azimuth solve with cutoff produces different flux than without
- Test azimuth-averaged solve with cutoff map runs without error

### 6. Changelog: `changes/NNN.feat.md`

## Critical Files

| File | Action |
|------|--------|
| `src/MCEq/geomagnetic.py` | **New** -- `GeomagneticCutoffMap` class |
| `src/MCEq/core.py` | **Modify** -- `set_primary_model`, `set_zenith_azimuth`, `solve`, new methods |
| `src/MCEq/config.py` | **Modify** -- add `geomagnetic_defaults` |
| `pyproject.toml` | **Modify** -- add core dependencies |
| `tests/test_geomagnetic.py` | **New** -- tests |

## User-Facing API

```python
from MCEq.core import MCEqRun
from MCEq.geomagnetic import GeomagneticCutoffMap
from MCEq.geometry.density_profiles import MSIS00KM3NeTCentered
import crflux.models as pm

# 1. Compute (or load cached) cutoff map
cutoff_map = GeomagneticCutoffMap(location="ORCA", date="2026-03-30")
cutoff_map.compute()  # expensive first time, cached thereafter

# 2. Set up MCEq with location-aware atmosphere
mceq = MCEqRun(
    interaction_model="SIBYLL2.3d",
    primary_model=(pm.GlobalSplineFitBeta, None),
)
mceq.set_density_model(MSIS00KM3NeTCentered("ORCA", season="January"))

# 3. Enable geomagnetic cutoff
mceq.set_geomagnetic_cutoff(cutoff_map)

# 4a. Azimuth-averaged solve (per-direction cutoff handled automatically)
mceq.set_zenith_azimuth(60.0)  # no azimuth -> averaging with per-direction cutoff
mceq.solve()

# 4b. Or single-direction solve (cutoff auto-applied for that direction)
mceq.set_zenith_azimuth(60.0, azimuth_deg=180.0)
mceq.solve()

# 5. Uniform cutoff (no map needed)
mceq.set_geomagnetic_cutoff(14.0)  # 14 GV uniform
```

## Performance Note

Azimuth-averaged solve with cutoff is ~N times slower than without (N = number of azimuth bins from the density model, default 36), since each direction requires a separate cascade solve. This is inherent -- each direction has different primary flux due to the geomagnetic field.

## Verification

1. Run `pytest tests/test_geomagnetic.py -v`
2. Verify cache files appear in `platformdirs.user_cache_dir("MCEq")/cutoff_maps/`
3. Verify that changing gtracr version string triggers recomputation
4. Compare vertical flux at equator (high cutoff ~14 GV) vs pole (low cutoff ~1 GV) -- equatorial should show suppressed low-energy flux
5. Run existing tests to verify no regressions: `pytest tests/ -v`
