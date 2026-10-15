`CorsikaAtmosphere` and `IsothermalAtmosphere` keep the `location` and `season`
they were built with; the base constructor used to reset both to `None`. As a
result the CORSIKA parameter sets of one site now get the geomagnetic cutoff in
`solve_fullsky` by default, as the MSIS atmospheres do: SouthPole,
PL_SouthPole and SDR_SouthPole at the South Pole, Karlsruhe, ANTARES/KM3NeT-ORCA
at ORCA, and KM3NeT-ARCA at ARCA. USStd, BK_USStd and the detector-averaged
KM3NeT set have no site and get no cutoff. Pass `geomagnetic_cutoff=False` to
keep the old results.
