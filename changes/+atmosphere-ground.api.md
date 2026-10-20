Columns end at the ground of the atmosphere model by default.
``config.h_obs`` is now ``None``, which places the observation level at sea
level for most models and at the ice surface (2834 m) for the CORSIKA South
Pole parameter sets (``SouthPole``, ``PL_SouthPole``, ``SDR_SouthPole``) and
for AIRS: these are fits above the ice, and below it they extrapolate, worst
for ``SDR_SouthPole``/October with three times the density of air at sea
level. Vertical depths of these models drop from about 1020 g/cm² (1301 for
October) to 686–706 g/cm². An observation level below a model's ground,
through ``config.h_obs`` or ``set_h_obs``, raises ``ValueError``. Atmospheres
with a raised observation level accept zenith angles up to 90°, the range
their columns support; ``set_h_obs`` at a vertical zenith rebuilds the depth
splines, which it skipped before.
