`MCEqRun.z_factor` weights neutron projectiles with the neutron flux of the primary model; it used the proton flux for every projectile.
`MCEqRun.z_factor(proj, sec)` with the default `definition="primary_e"` no longer fails its own assertion: `use_cs_scaling` applies to the Thunman definition only.
