The geomagnetic cutoff of the atmospheres of a named site (the CORSIKA sets of
one site and `MSISAtmosphere`) comes from maps shipped with MCEq: downgoing
protons on a 5° zenith and azimuth grid, IGRF at `MCEq.geometry.cutoff.CUTOFF_DATE`
(2025-01-01), made by `scripts/make_cutoff_maps.py`. These atmospheres reject
upgoing directions. Detector-centred MSIS atmospheres still trace the full sky
with gtracr, now at `CUTOFF_DATE` instead of today unless `date` is given.
Without an `azimuth_grid`, `solve_fullsky` averages the transmitted primary
flux over azimuth; it used to take azimuth 0. gtracr is now a dependency.
