Components below the driver take their settings as an argument object instead of
reaching into `MCEq.config`: `HDF5Backend(paths, grid, physics, em)`,
`ParticleManager(physics)`, `build_secant_kernel_ops(spec, paths)`,
`EarthGeometry(r_E, h_atm, h_obs)`, `etd2_nonuniform_path(step)` and the rest.
Every parameter defaults to `None` and falls back to the live view, so an
un-injected component reads exactly what it read before.

`MCEq.secant` and `MCEq.ddm` no longer reference config at all.
