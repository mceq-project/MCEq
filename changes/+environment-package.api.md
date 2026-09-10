The atmosphere and geometry code moved from `MCEq.geometry` to a new
`MCEq.environment` package: `base`, `column`, `corsika`, `isothermal`,
`msis00`, `msis21`, `msis00_backend`, `location_centered`, `tabulated`,
`target`, `geometry`, `parameters`, `registry`, `geomagnetic.cutoff`, and the
C extensions under `environment._ext`. Every old `MCEq.geometry.*` module
remains as a re-export, so existing imports — including
`from MCEq.geometry.density_profiles import *` — keep working unchanged.

**Rebuild after pulling this.** The compiled `_libnrlmsise00` and
`_libcorsikaatm` shared objects are not tracked by git, so a `git pull`
relocates the loaders while leaving the built libraries behind under
`src/MCEq/geometry/`. Run `pip install -e .` (or `uv sync --reinstall-package
mceq`), or move the `.so` by hand. A stale `build/<wheel_tag>/` directory
should be deleted too: it was configured against the old install paths, and
`cmake --install` on it without reconfiguring writes the libraries back to the
old location. Both loaders now raise a self-describing `ImportError` when they
cannot find their library, instead of the `NameError` and silently-missing
attribute they used to produce.
