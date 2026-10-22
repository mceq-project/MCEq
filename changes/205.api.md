Columns end at the ground of the atmosphere's site by default.
``config.h_obs`` is now ``None``, which places the observation level at the
site's altitude in ``LOCATIONS``: 2834 m for the South Pole models (CORSIKA
``SouthPole``, ``PL_SouthPole``, ``SDR_SouthPole``, MSIS ``SouthPole``,
AIRS), 110 m for Karlsruhe, a few to a few hundred metres for the other MSIS
sites, sea level for models without a site (``USStd``, ``BK_USStd``, KM3NeT)
and for the balloon sites ``PeaceRiver`` and ``FtSumner``, whose listed
heights are float altitudes. The South Pole fits hold only above the ice;
below it they extrapolate, worst for ``SDR_SouthPole``/October with three
times the density of air at sea level. Vertical depths of the South Pole
models drop from about 1020 g/cm² (1301 for October) to 686–706 g/cm².

An observation level below the ground, through ``config.h_obs`` or
``set_h_obs``, issues ``BelowGroundWarning``. Atmospheres with a raised
observation level accept zenith angles up to 90°, the range their columns
support; ``set_h_obs`` at a vertical zenith rebuilds the depth splines, which
it skipped before.
