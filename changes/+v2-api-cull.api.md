v2.0 API cull (release cut, ruling D-6; 2026-09-09 public-usage census).
Removed: the flat compat shims `MCEq.hankel`, `MCEq.ddm`, `MCEq.ddm_utils`,
`MCEq.download`, `MCEq.secant`, `MCEq.operator_assembly` (canonical homes
`MCEq.driver.results`, `MCEq.models.ddm.ddm`, `MCEq.models.ddm.ddm_utils`,
`MCEq.data.download`, `MCEq.operators.secant`, `MCEq.operators.compiled`);
the census-absent `MCEq.geometry` module shims `column`, `gtracr_cutoff`,
`location_centered`, `msis21_atmosphere`, `registry`, `nrlmsise00_mceq`
(canonical homes under `MCEq.environment.*`); and `MCEqRun.solve_multirhs`
(use `solve_batch` and read `res.sol` / `res.grid_sol`). `MCEq.ddm` was
deleted despite one live downstream import per the maintainer re-ruling of
2026-09-09 (patch PR filed upstream). Also removed at M14: the
`MCEq.geometry.atmosphere_parameters` re-export (canonical home
`MCEq.environment.parameters`). Kept: `MCEq.core` and `mceq_config`
(permanent façades), the census-kept sub-APIs (`MCEq.particlemanager`,
`MCEq.data`, `MCEq.geometry.density_profiles` / `.geometry`), the flat
config names (silent and canonical, ruling D-3). See `docs/migration_v2_layout.md`.
