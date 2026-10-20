`MCEqRun` blends in FLUKA20251 below 80 GeV by default
(`config.low_energy_extension["model"] = "FLUKA20251"`). The high-energy
models are tabulated from 80 GeV projectile energy, so before this change the
default calculation had no secondary production below 80 GeV: ground-level
fluxes near 1 GeV were several times too low and a geomagnetic rigidity
cutoff had almost no effect. `low_energy_model=None` restores the old
behaviour. A medium without FLUKA tables (e.g. hydrogen) runs without
blending unless a low-energy model is requested explicitly.
